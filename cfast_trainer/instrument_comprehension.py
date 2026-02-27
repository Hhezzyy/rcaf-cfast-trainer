from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

from .clock import Clock
from .cognitive_core import (
    AnswerScorer,
    Problem,
    SeededRng,
    TimedTextInputTest,
    clamp01,
    lerp_int,
)


@dataclass(frozen=True, slots=True)
class InstrumentComprehensionConfig:
    # Candidate guide indicates ~26 minutes total including instructions.
    # Keep scored block long by default; tests override with short durations.
    scored_duration_s: float = 22.0 * 60.0
    practice_questions: int = 3


class InstrumentComprehensionTrialKind(StrEnum):
    INSTRUMENTS_TO_DESCRIPTION = "instruments_to_description"
    DESCRIPTION_TO_INSTRUMENTS = "description_to_instruments"
    INSTRUMENTS_TO_AIRCRAFT = "instruments_to_aircraft"


@dataclass(frozen=True, slots=True)
class InstrumentState:
    speed_kts: int
    altitude_ft: int
    vertical_rate_fpm: int
    bank_deg: int
    pitch_deg: int
    heading_deg: int
    slip: int  # -1=slip left, 0=balanced, 1=slip right


@dataclass(frozen=True, slots=True)
class InstrumentOption:
    code: int
    state: InstrumentState
    description: str


@dataclass(frozen=True, slots=True)
class InstrumentComprehensionPayload:
    kind: InstrumentComprehensionTrialKind
    prompt_state: InstrumentState
    prompt_description: str
    options: tuple[InstrumentOption, ...]
    option_errors: tuple[int, ...]
    full_credit_error: int
    zero_credit_error: int


def _norm_heading(deg: int) -> int:
    return int(deg) % 360


def _heading_error(a: int, b: int) -> int:
    aa = _norm_heading(a)
    bb = _norm_heading(b)
    diff = abs(aa - bb)
    return min(diff, 360 - diff)


def _heading_cardinal_8(heading_deg: int) -> str:
    h = float(_norm_heading(heading_deg))
    idx = int(((h + 22.5) % 360) // 45)
    labels = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
    return labels[idx]


def _vertical_phrase(vertical_rate_fpm: int) -> str:
    vr = int(vertical_rate_fpm)
    if abs(vr) < 100:
        return "maintaining altitude"
    if vr > 0:
        return "rapidly ascending" if vr >= 800 else "ascending gradually"
    return "rapidly descending" if vr <= -800 else "descending gradually"


def _describe_state(state: InstrumentState) -> str:
    bank_short = "level"
    if abs(int(state.bank_deg)) > 2:
        turn_dir = "L" if int(state.bank_deg) < 0 else "R"
        bank_short = f"{turn_dir} bank {abs(int(state.bank_deg))}°"
    slip_short = "bal"
    if int(state.slip) < 0:
        slip_short = "slip L"
    elif int(state.slip) > 0:
        slip_short = "slip R"

    heading_cardinal = _heading_cardinal_8(state.heading_deg)
    return (
        f"{state.speed_kts}kt, {_vertical_phrase(state.vertical_rate_fpm)}, "
        f"ALT {state.altitude_ft}, "
        f"HDG {heading_cardinal}/{state.heading_deg:03d}, "
        f"{bank_short}, {slip_short}."
    )


def _state_error(true: InstrumentState, other: InstrumentState) -> int:
    speed_err = abs(int(true.speed_kts) - int(other.speed_kts)) // 5
    alt_err = abs(int(true.altitude_ft) - int(other.altitude_ft)) // 100
    vs_err = abs(int(true.vertical_rate_fpm) - int(other.vertical_rate_fpm)) // 100
    bank_err = abs(int(true.bank_deg) - int(other.bank_deg)) * 2
    pitch_err = abs(int(true.pitch_deg) - int(other.pitch_deg)) * 3
    heading_err = _heading_error(true.heading_deg, other.heading_deg)
    slip_penalty = 0 if int(true.slip) == int(other.slip) else 14
    return int(speed_err + alt_err + vs_err + bank_err + pitch_err + heading_err + slip_penalty)


class InstrumentComprehensionScorer(AnswerScorer):
    """Exact option gets full credit; near interpretation gets partial credit."""

    def score(self, *, problem: Problem, user_answer: int, raw: str) -> float:
        _ = raw

        payload = problem.payload
        if not isinstance(payload, InstrumentComprehensionPayload):
            return 1.0 if int(user_answer) == int(problem.answer) else 0.0

        by_code = {opt.code: idx for idx, opt in enumerate(payload.options)}
        idx = by_code.get(int(user_answer))
        if idx is None or idx >= len(payload.option_errors):
            return 0.0

        err = max(0, int(payload.option_errors[idx]))
        full = max(0, int(payload.full_credit_error))
        zero = max(full + 1, int(payload.zero_credit_error))

        if err <= full:
            return 1.0
        if err >= zero:
            return 0.0
        return clamp01((zero - err) / float(zero - full))


class InstrumentComprehensionGenerator:
    """Deterministic generator with 2 parts:
    1) Instrument analysis: instruments->description and reverse
    2) Instruments->aircraft orientation matching
    """

    def __init__(self, *, seed: int) -> None:
        self._rng = SeededRng(seed)
        self._kind_offset = int(self._rng.randint(0, 2))
        self._trial_index = 0

    def next_problem(self, *, difficulty: float) -> Problem:
        d = clamp01(difficulty)
        true_state = self._sample_state(difficulty=d)
        kind = self._pick_kind()

        distractors = self._build_distractors(state=true_state, difficulty=d, kind=kind)
        candidates = [true_state, *distractors]

        order = self._rng.sample((0, 1, 2, 3), k=4)
        options: list[InstrumentOption] = []
        option_errors: list[int] = []
        correct_code = 0

        for code, idx in enumerate(order, start=1):
            candidate_state = candidates[idx]
            option = InstrumentOption(
                code=code,
                state=candidate_state,
                description=_describe_state(candidate_state),
            )
            options.append(option)
            option_errors.append(_state_error(true_state, candidate_state))
            if idx == 0:
                correct_code = code

        prompt = self._prompt_for(kind=kind)
        payload = InstrumentComprehensionPayload(
            kind=kind,
            prompt_state=true_state,
            prompt_description=_describe_state(true_state),
            options=tuple(options),
            option_errors=tuple(option_errors),
            full_credit_error=0,
            zero_credit_error=lerp_int(130, 70, d),
        )
        return Problem(prompt=prompt, answer=int(correct_code), payload=payload)

    def _pick_kind(self) -> InstrumentComprehensionTrialKind:
        order = (
            InstrumentComprehensionTrialKind.INSTRUMENTS_TO_DESCRIPTION,
            InstrumentComprehensionTrialKind.DESCRIPTION_TO_INSTRUMENTS,
            InstrumentComprehensionTrialKind.INSTRUMENTS_TO_AIRCRAFT,
        )
        idx = (self._kind_offset + self._trial_index) % len(order)
        self._trial_index += 1
        return order[idx]

    def _prompt_for(self, *, kind: InstrumentComprehensionTrialKind) -> str:
        if kind is InstrumentComprehensionTrialKind.INSTRUMENTS_TO_DESCRIPTION:
            return "Read the instrument panel and choose the best matching description (1-4)."
        if kind is InstrumentComprehensionTrialKind.DESCRIPTION_TO_INSTRUMENTS:
            return "Read the description and choose the matching instrument panel (1-4)."
        return "Read the instruments and choose the aircraft orientation image that matches (1-4)."

    def _sample_state(self, *, difficulty: float) -> InstrumentState:
        speed_step = max(5, lerp_int(20, 5, difficulty))
        altitude_step = max(100, lerp_int(500, 100, difficulty))
        heading_step = max(5, lerp_int(45, 5, difficulty))

        speed = self._sample_quantized(120, 360, speed_step)
        altitude = self._sample_quantized(1000, 9500, altitude_step)
        heading = self._sample_quantized(0, 359, heading_step)

        bank_mag = self._rng.randint(lerp_int(8, 4, difficulty), lerp_int(35, 22, difficulty))
        bank_sign = -1 if self._rng.random() < 0.5 else 1
        bank = int(bank_sign * bank_mag)

        pitch_mag = self._rng.randint(lerp_int(2, 1, difficulty), lerp_int(12, 8, difficulty))
        pitch_sign = -1 if self._rng.random() < 0.55 else 1
        pitch = int(pitch_sign * pitch_mag)

        vertical_sign = (
            -1 if pitch < 0 else 1 if pitch > 0 else (-1 if self._rng.random() < 0.5 else 1)
        )
        vertical_mag = self._rng.randint(0, lerp_int(1200, 2200, difficulty))
        vertical_rate = int(vertical_sign * (vertical_mag // 100) * 100)

        slip_choices = (0, 0, -1, 1) if difficulty >= 0.45 else (0, 0, 0, -1, 1)
        slip = int(self._rng.choice(slip_choices))

        return InstrumentState(
            speed_kts=int(speed),
            altitude_ft=int(altitude),
            vertical_rate_fpm=int(vertical_rate),
            bank_deg=int(bank),
            pitch_deg=int(pitch),
            heading_deg=int(heading),
            slip=int(slip),
        )

    def _build_distractors(
        self,
        *,
        state: InstrumentState,
        difficulty: float,
        kind: InstrumentComprehensionTrialKind,
    ) -> tuple[InstrumentState, InstrumentState, InstrumentState]:
        near_heading = max(5, lerp_int(30, 10, difficulty))
        far_heading = lerp_int(95, 38, difficulty)
        speed_delta = lerp_int(40, 15, difficulty)
        altitude_delta = lerp_int(1600, 400, difficulty)
        vs_delta = lerp_int(900, 300, difficulty)

        if kind is InstrumentComprehensionTrialKind.INSTRUMENTS_TO_AIRCRAFT:
            s1 = replace(state, heading_deg=_norm_heading(state.heading_deg + near_heading))
            s2 = replace(state, bank_deg=-state.bank_deg)
            s3 = replace(
                state,
                bank_deg=-state.bank_deg,
                pitch_deg=-state.pitch_deg,
                heading_deg=_norm_heading(state.heading_deg + far_heading),
            )
            return s1, s2, s3

        s1 = replace(state, speed_kts=self._clamp(state.speed_kts + speed_delta, 120, 360))
        s2 = replace(state, altitude_ft=self._clamp(state.altitude_ft + altitude_delta, 1000, 9500))
        s3 = replace(
            state,
            vertical_rate_fpm=self._clamp(state.vertical_rate_fpm - vs_delta, -2500, 2500),
            heading_deg=_norm_heading(state.heading_deg + near_heading),
            bank_deg=-state.bank_deg,
            slip=-state.slip if state.slip != 0 else 1,
        )
        return s1, s2, s3

    def _sample_quantized(self, lo: int, hi: int, step: int) -> int:
        if step <= 1:
            return int(self._rng.randint(lo, hi))
        min_q = lo // step
        max_q = hi // step
        q = int(self._rng.randint(min_q, max_q))
        v = q * step
        if v < lo:
            v += step
        if v > hi:
            v -= step
        return int(max(lo, min(hi, v)))

    def _clamp(self, v: int, lo: int, hi: int) -> int:
        return int(lo if v < lo else hi if v > hi else v)


def build_instrument_comprehension_test(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    config: InstrumentComprehensionConfig | None = None,
) -> TimedTextInputTest:
    cfg = config or InstrumentComprehensionConfig()

    instructions = [
        "Instrument Comprehension",
        "",
        "Part 1:",
        "- Read multiple instrument readings and pick the best description",
        "- Reverse items also appear (description -> choose instrument panel)",
        "",
        "Part 2:",
        "- Read instruments and choose the matching aircraft orientation image",
        "",
        "Controls:",
        "- Type option number (1-4)",
        "- Press Enter to submit",
        "",
        "You will get a short practice, then a timed scored block.",
    ]

    return TimedTextInputTest(
        title="Instrument Comprehension",
        instructions=instructions,
        generator=InstrumentComprehensionGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        practice_questions=cfg.practice_questions,
        scored_duration_s=cfg.scored_duration_s,
        scorer=InstrumentComprehensionScorer(),
    )
