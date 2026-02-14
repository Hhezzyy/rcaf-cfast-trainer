from __future__ import annotations

from dataclasses import dataclass
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
    # Candidate guide indicates ~26 minutes including instructions.
    scored_duration_s: float = 22.0 * 60.0
    practice_questions: int = 3


class InstrumentComprehensionTrialKind(StrEnum):
    ATTITUDE_MATCH = "attitude_match"
    ATTITUDE_WITH_CALLOUT = "attitude_with_callout"


@dataclass(frozen=True, slots=True)
class InstrumentOption:
    code: int
    label: str
    bank_deg: int
    pitch_deg: int
    heading_deg: int


@dataclass(frozen=True, slots=True)
class InstrumentComprehensionPayload:
    kind: InstrumentComprehensionTrialKind
    heading_deg: int
    bank_deg: int
    pitch_deg: int
    verbal_cue: str
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


def _attitude_error(
    *,
    true_bank_deg: int,
    true_pitch_deg: int,
    true_heading_deg: int,
    option_bank_deg: int,
    option_pitch_deg: int,
    option_heading_deg: int,
) -> int:
    # Weighted Manhattan metric tuned for training feedback.
    bank_err = abs(int(true_bank_deg) - int(option_bank_deg))
    pitch_err = abs(int(true_pitch_deg) - int(option_pitch_deg))
    heading_err = _heading_error(int(true_heading_deg), int(option_heading_deg))
    return int(bank_err * 2 + pitch_err * 3 + heading_err)


class InstrumentComprehensionScorer(AnswerScorer):
    """Exact option gets full credit; near attitude matches receive partial credit."""

    def score(self, *, problem: Problem, user_answer: int, raw: str) -> float:
        _ = raw

        payload = problem.payload
        if not isinstance(payload, InstrumentComprehensionPayload):
            return 1.0 if int(user_answer) == int(problem.answer) else 0.0

        by_code = {opt.code: idx for idx, opt in enumerate(payload.options)}
        idx = by_code.get(int(user_answer))
        if idx is None:
            return 0.0
        if idx >= len(payload.option_errors):
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
    """Deterministic generator for instrument attitude interpretation trials."""

    def __init__(self, *, seed: int) -> None:
        self._rng = SeededRng(seed)

    def next_problem(self, *, difficulty: float) -> Problem:
        d = clamp01(difficulty)

        heading = self._sample_heading(d)
        bank = self._signed_magnitude(
            lo=lerp_int(22, 8, d),
            hi=lerp_int(38, 24, d),
        )
        pitch = self._signed_magnitude(
            lo=lerp_int(8, 3, d),
            hi=lerp_int(16, 10, d),
        )

        near_heading_delta = max(5, lerp_int(25, 8, d))
        far_heading_delta = lerp_int(95, 40, d)

        kind = (
            InstrumentComprehensionTrialKind.ATTITUDE_MATCH
            if self._rng.random() < 0.5
            else InstrumentComprehensionTrialKind.ATTITUDE_WITH_CALLOUT
        )

        verbal_cue = self._verbal_cue(kind=kind, bank_deg=bank, pitch_deg=pitch)

        candidates = [
            ("Match", bank, pitch, heading),
            ("Near heading", bank, pitch, heading + near_heading_delta),
            ("Opposite bank", -bank, pitch, heading),
            ("Opposite bank + pitch", -bank, -pitch, heading + far_heading_delta),
        ]

        order = self._rng.sample((0, 1, 2, 3), k=4)
        options: list[InstrumentOption] = []
        option_errors: list[int] = []
        correct_code = 0

        for code, idx in enumerate(order, start=1):
            _, opt_bank, opt_pitch, opt_heading_raw = candidates[idx]
            opt_heading = _norm_heading(opt_heading_raw)
            label = (
                f"{self._bank_text(opt_bank)}, {self._pitch_text(opt_pitch)}, "
                f"HDG {opt_heading:03d}"
            )
            options.append(
                InstrumentOption(
                    code=code,
                    label=label,
                    bank_deg=int(opt_bank),
                    pitch_deg=int(opt_pitch),
                    heading_deg=int(opt_heading),
                )
            )
            option_errors.append(
                _attitude_error(
                    true_bank_deg=bank,
                    true_pitch_deg=pitch,
                    true_heading_deg=heading,
                    option_bank_deg=opt_bank,
                    option_pitch_deg=opt_pitch,
                    option_heading_deg=opt_heading,
                )
            )
            if idx == 0:
                correct_code = code

        zero_credit_error = lerp_int(110, 70, d)
        payload = InstrumentComprehensionPayload(
            kind=kind,
            heading_deg=int(heading),
            bank_deg=int(bank),
            pitch_deg=int(pitch),
            verbal_cue=verbal_cue,
            options=tuple(options),
            option_errors=tuple(option_errors),
            full_credit_error=0,
            zero_credit_error=zero_credit_error,
        )

        prompt = (
            "Select the option number (1-4) that best matches the instruments."
            if kind is InstrumentComprehensionTrialKind.ATTITUDE_MATCH
            else "Select the option number (1-4) that matches instruments and callout."
        )

        return Problem(prompt=prompt, answer=int(correct_code), payload=payload)

    def _sample_heading(self, difficulty: float) -> int:
        step = max(1, lerp_int(45, 10, difficulty))
        if step <= 1:
            return int(self._rng.randint(0, 359))

        max_q = 359 // step
        q = int(self._rng.randint(0, max_q))
        return int((q * step) % 360)

    def _signed_magnitude(self, *, lo: int, hi: int) -> int:
        mag_lo = max(1, int(min(lo, hi)))
        mag_hi = max(mag_lo, int(max(lo, hi)))
        mag = int(self._rng.randint(mag_lo, mag_hi))
        sign = -1 if self._rng.random() < 0.5 else 1
        return int(sign * mag)

    def _bank_text(self, bank_deg: int) -> str:
        return f"bank {'left' if bank_deg < 0 else 'right'} {abs(int(bank_deg))}°"

    def _pitch_text(self, pitch_deg: int) -> str:
        return f"nose {'down' if pitch_deg < 0 else 'up'} {abs(int(pitch_deg))}°"

    def _verbal_cue(
        self,
        *,
        kind: InstrumentComprehensionTrialKind,
        bank_deg: int,
        pitch_deg: int,
    ) -> str:
        if kind is InstrumentComprehensionTrialKind.ATTITUDE_MATCH:
            return (
                f"Instrument cue: {self._bank_text(bank_deg)}; "
                f"{self._pitch_text(pitch_deg)}."
            )
        turn = "turning left" if bank_deg < 0 else "turning right"
        trend = "descending" if pitch_deg < 0 else "climbing"
        return f"Crew callout: {turn}, {trend}."


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
        "Inspect bank, pitch, and heading information to infer aircraft orientation.",
        "Use instrument, numerical, and verbal cues together.",
        "",
        "Controls:",
        "- Read the instrument panel and options",
        "- Type the option number (1-4)",
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
