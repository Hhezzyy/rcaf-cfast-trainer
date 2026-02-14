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
class AnglesBearingsDegreesConfig:
    # Candidate guide indicates ~10 minutes including instructions.
    scored_duration_s: float = 8.0 * 60.0
    practice_questions: int = 3


class AnglesBearingsQuestionKind(StrEnum):
    ANGLE_BETWEEN_LINES = "angle_between_lines"
    BEARING_FROM_REFERENCE = "bearing_from_reference"


@dataclass(frozen=True, slots=True)
class AnglesBearingsDegreesPayload:
    kind: AnglesBearingsQuestionKind
    reference_bearing_deg: int
    target_bearing_deg: int
    object_label: str
    full_credit_error_deg: int
    zero_credit_error_deg: int


def _circular_error_deg(a: int, b: int) -> int:
    aa = int(a) % 360
    bb = int(b) % 360
    diff = abs(aa - bb)
    return min(diff, 360 - diff)


class AnglesBearingsScorer(AnswerScorer):
    """Linear estimation score with full-credit and zero-credit error bands."""

    def score(self, *, problem: Problem, user_answer: int, raw: str) -> float:
        _ = raw

        payload = problem.payload
        if not isinstance(payload, AnglesBearingsDegreesPayload):
            return 1.0 if int(user_answer) == int(problem.answer) else 0.0

        full = max(0, int(payload.full_credit_error_deg))
        zero = max(full + 1, int(payload.zero_credit_error_deg))

        if payload.kind is AnglesBearingsQuestionKind.BEARING_FROM_REFERENCE:
            err = _circular_error_deg(int(user_answer), int(problem.answer))
        else:
            err = abs(int(user_answer) - int(problem.answer))

        if err <= full:
            return 1.0
        if err >= zero:
            return 0.0

        return clamp01((zero - err) / float(zero - full))


class AnglesBearingsDegreesGenerator:
    """Deterministic generator for angle and bearing estimation trials."""

    def __init__(self, *, seed: int) -> None:
        self._rng = SeededRng(seed)
        self._labels = tuple("ABCDEF")

    def next_problem(self, *, difficulty: float) -> Problem:
        d = clamp01(difficulty)
        if self._rng.random() < 0.5:
            return self._make_angle_problem(d)
        return self._make_bearing_problem(d)

    def _make_angle_problem(self, difficulty: float) -> Problem:
        reference_bearing = self._rng.randint(0, 359)
        quant_step = lerp_int(15, 1, difficulty)
        min_angle = lerp_int(75, 12, difficulty)
        max_angle = lerp_int(130, 170, difficulty)

        angle = self._sample_quantized(min_angle, max_angle, quant_step)
        clockwise = self._rng.random() < 0.5
        if clockwise:
            target_bearing = (reference_bearing + angle) % 360
        else:
            target_bearing = (reference_bearing - angle) % 360

        full_credit = lerp_int(4, 2, difficulty)
        zero_credit = lerp_int(35, 15, difficulty)

        payload = AnglesBearingsDegreesPayload(
            kind=AnglesBearingsQuestionKind.ANGLE_BETWEEN_LINES,
            reference_bearing_deg=reference_bearing,
            target_bearing_deg=target_bearing,
            object_label="",
            full_credit_error_deg=full_credit,
            zero_credit_error_deg=zero_credit,
        )
        return Problem(
            prompt="Estimate the smaller angle between the two rays (degrees).",
            answer=int(angle),
            payload=payload,
        )

    def _make_bearing_problem(self, difficulty: float) -> Problem:
        quant_step = lerp_int(15, 1, difficulty)
        bearing = self._sample_quantized(0, 359, quant_step)
        label = str(self._rng.choice(self._labels))

        full_credit = lerp_int(5, 2, difficulty)
        zero_credit = lerp_int(40, 18, difficulty)

        payload = AnglesBearingsDegreesPayload(
            kind=AnglesBearingsQuestionKind.BEARING_FROM_REFERENCE,
            reference_bearing_deg=0,
            target_bearing_deg=bearing,
            object_label=label,
            full_credit_error_deg=full_credit,
            zero_credit_error_deg=zero_credit,
        )
        return Problem(
            prompt=f"Estimate the bearing of point {label} from the center (000-359).",
            answer=int(bearing),
            payload=payload,
        )

    def _sample_quantized(self, lo: int, hi: int, step: int) -> int:
        if step <= 1:
            return int(self._rng.randint(lo, hi))

        max_q = hi // step
        min_q = lo // step
        q = self._rng.randint(min_q, max_q)
        value = q * step
        if value < lo:
            value += step
        if value > hi:
            value -= step
        return int(max(lo, min(hi, value)))


def build_angles_bearings_degrees_test(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    config: AnglesBearingsDegreesConfig | None = None,
) -> TimedTextInputTest:
    cfg = config or AnglesBearingsDegreesConfig()

    instructions = [
        "Angles, Bearings and Degrees",
        "",
        "Estimate angles and bearings quickly and accurately.",
        "Part A: estimate the smaller angle between two lines.",
        "Part B: estimate the bearing of an object from a reference point.",
        "",
        "Controls:",
        "- Type your estimate as whole degrees",
        "- Press Enter to submit",
        "",
        "You will get a short practice, then a timed scored block.",
    ]

    return TimedTextInputTest(
        title="Angles, Bearings and Degrees",
        instructions=instructions,
        generator=AnglesBearingsDegreesGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        practice_questions=cfg.practice_questions,
        scored_duration_s=cfg.scored_duration_s,
        scorer=AnglesBearingsScorer(),
    )
