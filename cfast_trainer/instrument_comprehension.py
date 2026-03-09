from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

from .clock import Clock
from .cognitive_core import (
    AnswerScorer,
    AttemptSummary,
    Phase,
    Problem,
    QuestionEvent,
    SeededRng,
    TestSnapshot,
    clamp01,
    lerp_int,
)


@dataclass(frozen=True, slots=True)
class InstrumentComprehensionConfig:
    # Candidate guide indicates ~26 minutes total including instructions.
    scored_duration_s: float = 22.0 * 60.0
    practice_questions: int = 2


class InstrumentComprehensionTrialKind(StrEnum):
    INSTRUMENTS_TO_DESCRIPTION = "instruments_to_description"
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


_PART_ORDER = (
    InstrumentComprehensionTrialKind.INSTRUMENTS_TO_AIRCRAFT,
    InstrumentComprehensionTrialKind.INSTRUMENTS_TO_DESCRIPTION,
)


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


def _turn_phrase(bank_deg: int) -> str:
    bank = int(bank_deg)
    if abs(bank) <= 4:
        return "maintaining direction"
    return "turning left" if bank < 0 else "turning right"


def _altitude_phrase(*, altitude_ft: int, vertical_rate_fpm: int) -> str:
    altitude = int(altitude_ft)
    vertical_rate = int(vertical_rate_fpm)
    if abs(vertical_rate) < 100:
        return f"maintaining height at {altitude} feet"
    if vertical_rate > 0:
        return f"climbing through {altitude} feet"
    return f"descending through {altitude} feet"


def _describe_state(state: InstrumentState) -> str:
    altitude_text = _altitude_phrase(
        altitude_ft=int(state.altitude_ft),
        vertical_rate_fpm=int(state.vertical_rate_fpm),
    )
    return (
        f"Flying at {int(state.speed_kts)} kt, "
        f"{_turn_phrase(int(state.bank_deg))}, "
        f"heading {_heading_cardinal_8(int(state.heading_deg))}, "
        f"{altitude_text}."
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


def altimeter_hand_turns(altitude_ft: int) -> tuple[float, float]:
    """Return clockwise turns from the 12 o'clock position.

    The short hand makes one full revolution per 10,000 ft and indicates
    thousands. The long hand makes one full revolution per 1,000 ft and
    indicates hundreds.
    """

    altitude = max(0, int(altitude_ft))
    thousands_turn = (altitude % 10000) / 10000.0
    hundreds_turn = (altitude % 1000) / 1000.0
    return thousands_turn, hundreds_turn


def airspeed_turn(speed_kts: int) -> float:
    """Return the clockwise turns for a 0-360 knot circular airspeed dial."""

    speed = max(0, min(360, int(speed_kts)))
    return speed / 360.0


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
    """Deterministic generator for the guide-style two-part instrument test."""

    def __init__(self, *, seed: int) -> None:
        self._rng = SeededRng(seed)
        self._kind_offset = int(self._rng.randint(0, len(_PART_ORDER) - 1))
        self._trial_index = 0

    def next_problem(self, *, difficulty: float) -> Problem:
        kind = _PART_ORDER[(self._kind_offset + self._trial_index) % len(_PART_ORDER)]
        self._trial_index += 1
        return self.next_problem_for_kind(kind=kind, difficulty=difficulty)

    def next_problem_for_kind(
        self,
        *,
        kind: InstrumentComprehensionTrialKind,
        difficulty: float,
    ) -> Problem:
        d = clamp01(difficulty)
        true_state = self._sample_state(difficulty=d)

        distractors = self._build_distractors(state=true_state, difficulty=d, kind=kind)
        candidates = [true_state, *distractors]

        order = self._rng.sample((0, 1, 2, 3, 4), k=5)
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

        payload = InstrumentComprehensionPayload(
            kind=kind,
            prompt_state=true_state,
            prompt_description=_describe_state(true_state),
            options=tuple(options),
            option_errors=tuple(option_errors),
            full_credit_error=0,
            zero_credit_error=lerp_int(130, 70, d),
        )
        return Problem(
            prompt=self._prompt_for(kind=kind),
            answer=int(correct_code),
            payload=payload,
        )

    def _prompt_for(self, *, kind: InstrumentComprehensionTrialKind) -> str:
        if kind is InstrumentComprehensionTrialKind.INSTRUMENTS_TO_AIRCRAFT:
            return "Read the instruments and choose the matching aircraft image (A/S/D/F/G)."
        return (
            "Read the instrument panel and choose the best matching "
            "flight description (A/S/D/F/G)."
        )

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
    ) -> tuple[InstrumentState, InstrumentState, InstrumentState, InstrumentState]:
        near_heading = max(5, lerp_int(32, 10, difficulty))
        far_heading = lerp_int(95, 42, difficulty)
        speed_delta = lerp_int(40, 15, difficulty)
        altitude_delta = lerp_int(1600, 400, difficulty)
        vs_delta = lerp_int(900, 300, difficulty)

        if kind is InstrumentComprehensionTrialKind.INSTRUMENTS_TO_AIRCRAFT:
            s1 = replace(state, heading_deg=_norm_heading(state.heading_deg + near_heading))
            s2 = replace(state, bank_deg=-state.bank_deg)
            s3 = replace(state, pitch_deg=-state.pitch_deg)
            s4 = replace(
                state,
                bank_deg=-state.bank_deg,
                pitch_deg=-state.pitch_deg,
                heading_deg=_norm_heading(state.heading_deg + far_heading),
            )
            return s1, s2, s3, s4

        s1 = replace(state, speed_kts=self._clamp(state.speed_kts + speed_delta, 120, 360))
        s2 = replace(state, altitude_ft=self._clamp(state.altitude_ft + altitude_delta, 1000, 9500))
        s3 = replace(
            state,
            heading_deg=_norm_heading(state.heading_deg + near_heading),
            bank_deg=-state.bank_deg,
        )
        s4 = replace(
            state,
            vertical_rate_fpm=self._clamp(-state.vertical_rate_fpm - vs_delta, -2500, 2500),
            pitch_deg=-state.pitch_deg,
            heading_deg=_norm_heading(state.heading_deg + far_heading),
        )
        return s1, s2, s3, s4

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

    @staticmethod
    def _clamp(v: int, lo: int, hi: int) -> int:
        return int(lo if v < lo else hi if v > hi else v)


class InstrumentComprehensionEngine:
    """Two-part guide-style instrument comprehension engine.

    Part 1: instruments -> aircraft orientation image
    Part 2: instruments -> flight description
    """

    def __init__(
        self,
        *,
        clock: Clock,
        seed: int,
        difficulty: float = 0.5,
        config: InstrumentComprehensionConfig | None = None,
    ) -> None:
        cfg = config or InstrumentComprehensionConfig()
        self._clock = clock
        self._seed = int(seed)
        self._difficulty = clamp01(difficulty)
        self._cfg = cfg
        self._generator = InstrumentComprehensionGenerator(seed=self._seed)
        self._scorer = InstrumentComprehensionScorer()
        self._instructions = [
            "Instrument Comprehension",
            "",
            "Part 1: read the attitude and heading instruments, "
            "then choose the matching aircraft image.",
            "Part 2: read the full instrument panel, "
            "then choose the matching flight description.",
            "",
            "Controls:",
            "- Press A, S, D, F, or G to choose an option",
            "- Press Enter to submit",
        ]

        self._phase = Phase.INSTRUCTIONS
        self._part_idx = 0
        self._current: Problem | None = None
        self._presented_at_s: float | None = None
        self._practice_answered = 0
        self._scored_started_at_s: float | None = None
        self._pending_done_action: str | None = None  # "start_scored" | "next_part_practice"

        self._events: list[QuestionEvent] = []
        self._scored_attempted = 0
        self._scored_correct = 0
        self._scored_total_score = 0.0
        self._scored_max_score = 0.0

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def difficulty(self) -> float:
        return float(self._difficulty)

    def events(self) -> list[QuestionEvent]:
        return list(self._events)

    def instructions(self) -> list[str]:
        return list(self._instructions)

    def can_exit(self) -> bool:
        return self._phase in (
            Phase.INSTRUCTIONS,
            Phase.PRACTICE,
            Phase.PRACTICE_DONE,
            Phase.RESULTS,
        )

    def start(self) -> None:
        self.start_practice()

    def start_practice(self) -> None:
        if self._phase is not Phase.INSTRUCTIONS:
            return
        if int(self._cfg.practice_questions) <= 0:
            self._phase = Phase.PRACTICE_DONE
            self._pending_done_action = "start_scored"
            self._current = None
            self._presented_at_s = None
            return
        self._part_idx = 0
        self._practice_answered = 0
        self._pending_done_action = None
        self._phase = Phase.PRACTICE
        self._deal_new_problem()

    def start_scored(self) -> None:
        if self._phase is Phase.RESULTS:
            return
        if self._phase is Phase.INSTRUCTIONS:
            self._part_idx = 0
            self._phase = Phase.SCORED
            self._pending_done_action = None
            self._scored_started_at_s = self._clock.now()
            self._deal_new_problem()
            return
        if self._phase is not Phase.PRACTICE_DONE:
            return

        if self._pending_done_action == "next_part_practice":
            if self._part_idx + 1 >= len(_PART_ORDER):
                self._phase = Phase.RESULTS
                self._current = None
                self._presented_at_s = None
                return
            self._part_idx += 1
            self._practice_answered = 0
            self._pending_done_action = None
            if int(self._cfg.practice_questions) <= 0:
                self._phase = Phase.SCORED
                self._scored_started_at_s = self._clock.now()
                self._deal_new_problem()
                return
            self._phase = Phase.PRACTICE
            self._deal_new_problem()
            return

        self._phase = Phase.SCORED
        self._pending_done_action = None
        self._scored_started_at_s = self._clock.now()
        self._deal_new_problem()

    def update(self) -> None:
        if self._phase is not Phase.SCORED:
            return
        remaining = self.time_remaining_s()
        if remaining is not None and remaining <= 0.0:
            self._finish_scored_part()

    def time_remaining_s(self) -> float | None:
        if self._phase is not Phase.SCORED or self._scored_started_at_s is None:
            return None
        per_part_duration = self._scored_duration_per_part_s()
        elapsed = self._clock.now() - self._scored_started_at_s
        return max(0.0, per_part_duration - elapsed)

    def current_prompt(self) -> str:
        part_num = self._part_idx + 1
        if self._phase is Phase.INSTRUCTIONS:
            return "Press Enter to begin Part 1 practice."
        if self._phase is Phase.PRACTICE_DONE:
            if self._pending_done_action == "next_part_practice":
                return (
                    f"Part {part_num} complete. Press Enter to continue "
                    f"to Part {part_num + 1} practice."
                )
            return f"Part {part_num} practice complete. Press Enter to start the scored block."
        if self._phase is Phase.RESULTS:
            s = self.scored_summary()
            acc_pct = int(round(s.accuracy * 100))
            rt = "n/a" if s.mean_response_time_s is None else f"{s.mean_response_time_s:.2f}s"
            return (
                f"Results\nAttempted: {s.attempted}\nCorrect: {s.correct}\n"
                f"Accuracy: {acc_pct}%\nMean RT: {rt}\nThroughput: {s.throughput_per_min:.1f}/min"
            )
        if self._current is None:
            return ""
        return self._current.prompt

    def submit_answer(self, raw: object) -> bool:
        raw_in = raw if isinstance(raw, str) else str(raw)
        raw = raw_in.strip()
        if self._handle_control_command(raw):
            return True

        if self._phase not in (Phase.PRACTICE, Phase.SCORED):
            return False
        expired = self._phase is Phase.SCORED and self.time_remaining_s() == 0
        if expired and raw == "":
            self._finish_scored_part()
            return False
        if raw == "":
            return False

        try:
            user_answer = int(raw)
        except ValueError:
            if expired:
                self._finish_scored_part()
            return False

        assert self._current is not None
        assert self._presented_at_s is not None

        answered_at_s = self._clock.now()
        response_time_s = max(0.0, answered_at_s - self._presented_at_s)
        score = float(
            self._scorer.score(
                problem=self._current,
                user_answer=user_answer,
                raw=raw_in,
            )
        )
        if score < 0.0:
            score = 0.0
        elif score > 1.0:
            score = 1.0
        is_full_correct = score >= 1.0 - 1e-9

        event = QuestionEvent(
            index=len(self._events),
            phase=self._phase,
            prompt=self._current.prompt,
            correct_answer=self._current.answer,
            user_answer=user_answer,
            is_correct=is_full_correct,
            presented_at_s=self._presented_at_s,
            answered_at_s=answered_at_s,
            response_time_s=response_time_s,
            raw=raw_in,
            score=score,
            max_score=1.0,
        )
        self._events.append(event)

        if self._phase is Phase.SCORED:
            self._scored_attempted += 1
            self._scored_total_score += score
            self._scored_max_score += 1.0
            if is_full_correct:
                self._scored_correct += 1
        else:
            self._practice_answered += 1

        if expired and self._phase is Phase.SCORED:
            self._finish_scored_part()
            return True

        if (
            self._phase is Phase.PRACTICE
            and self._practice_answered >= int(self._cfg.practice_questions)
        ):
            self._phase = Phase.PRACTICE_DONE
            self._pending_done_action = "start_scored"
            self._current = None
            self._presented_at_s = None
            return True

        self._deal_new_problem()
        return True

    def scored_summary(self) -> AttemptSummary:
        duration_s = float(self._cfg.scored_duration_s)
        attempted = self._scored_attempted
        correct = self._scored_correct
        accuracy = 0.0 if attempted == 0 else correct / attempted
        throughput = (attempted / duration_s) * 60.0 if duration_s > 0 else 0.0
        rts = [e.response_time_s for e in self._events if e.phase is Phase.SCORED]
        mean_rt = None if not rts else sum(rts) / len(rts)
        total_score = float(self._scored_total_score)
        max_score = float(self._scored_max_score)
        score_ratio = 0.0 if max_score == 0.0 else total_score / max_score
        return AttemptSummary(
            attempted=attempted,
            correct=correct,
            accuracy=accuracy,
            duration_s=duration_s,
            throughput_per_min=throughput,
            mean_response_time_s=mean_rt,
            total_score=total_score,
            max_score=max_score,
            score_ratio=score_ratio,
        )

    def snapshot(self) -> TestSnapshot:
        payload = None if self._current is None else self._current.payload
        return TestSnapshot(
            title="Instrument Comprehension",
            phase=self._phase,
            prompt=self.current_prompt(),
            input_hint="Use A, S, D, F, or G then Enter",
            time_remaining_s=self.time_remaining_s(),
            attempted_scored=self._scored_attempted,
            correct_scored=self._scored_correct,
            payload=payload,
        )

    def _deal_new_problem(self) -> None:
        kind = _PART_ORDER[self._part_idx]
        self._current = self._generator.next_problem_for_kind(
            kind=kind,
            difficulty=self._difficulty,
        )
        self._presented_at_s = self._clock.now()

    def _finish_scored_part(self) -> None:
        self._current = None
        self._presented_at_s = None
        self._scored_started_at_s = None
        if self._part_idx + 1 < len(_PART_ORDER):
            self._phase = Phase.PRACTICE_DONE
            self._pending_done_action = "next_part_practice"
            return
        self._to_results()

    def _scored_duration_per_part_s(self) -> float:
        return float(self._cfg.scored_duration_s) / float(max(1, len(_PART_ORDER)))

    def _to_results(self) -> None:
        self._phase = Phase.RESULTS
        self._pending_done_action = None
        self._current = None
        self._presented_at_s = None
        self._scored_started_at_s = None

    def _handle_control_command(self, raw: str) -> bool:
        token = str(raw).strip().lower()
        if token in {"__skip_all__", "skip_all"}:
            self._to_results()
            return True

        if token in {"__skip_practice__", "skip_practice"}:
            if self._phase is not Phase.PRACTICE:
                return False
            self._phase = Phase.PRACTICE_DONE
            self._pending_done_action = "start_scored"
            self._current = None
            self._presented_at_s = None
            return True

        if token in {"__skip_section__", "skip_section"}:
            if self._phase is Phase.SCORED:
                self._finish_scored_part()
                return True
            if self._phase is Phase.PRACTICE_DONE:
                if self._part_idx + 1 < len(_PART_ORDER):
                    self._part_idx += 1
                    self._practice_answered = 0
                    self._phase = Phase.PRACTICE
                    self._pending_done_action = None
                    self._deal_new_problem()
                    return True
                self._to_results()
                return True
            return False

        return False


def build_instrument_comprehension_test(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    config: InstrumentComprehensionConfig | None = None,
) -> InstrumentComprehensionEngine:
    return InstrumentComprehensionEngine(
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        config=config,
    )
