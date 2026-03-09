from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .clock import Clock
from .cognitive_core import (
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
class AnglesBearingsOption:
    code: int
    text: str
    value_deg: int


@dataclass(frozen=True, slots=True)
class AnglesBearingsDegreesPayload:
    kind: AnglesBearingsQuestionKind
    stem: str
    reference_bearing_deg: int
    target_bearing_deg: int
    angle_measure: str | None
    object_label: str
    options: tuple[AnglesBearingsOption, ...]
    correct_code: int
    correct_value_deg: int


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

        stem = "Estimate the smaller angle between the two rays."
        distractors = (
            360 - angle,  # Common mistake: report the reflex angle.
            (target_bearing - reference_bearing) % 360,  # Common mistake: one-way sweep.
            angle + quant_step,  # Common mistake: close arithmetic overshoot.
        )
        return self._multiple_choice_problem(
            kind=AnglesBearingsQuestionKind.ANGLE_BETWEEN_LINES,
            stem=stem,
            correct_value=angle,
            reference_bearing_deg=reference_bearing,
            target_bearing_deg=target_bearing,
            angle_measure="smaller",
            object_label="",
            distractors=distractors,
        )

    def _make_bearing_problem(self, difficulty: float) -> Problem:
        quant_step = lerp_int(15, 1, difficulty)
        bearing = self._sample_quantized(0, 359, quant_step)
        label = str(self._rng.choice(self._labels))
        stem = f"Estimate the bearing of point {label} from the center (000-359)."
        distractors = (
            (bearing + 180) % 360,  # Common mistake: reciprocal bearing.
            (90 - bearing) % 360,  # Common mistake: math-angle conversion error.
            (360 - bearing) % 360,  # Common mistake: counterclockwise read.
        )
        return self._multiple_choice_problem(
            kind=AnglesBearingsQuestionKind.BEARING_FROM_REFERENCE,
            stem=stem,
            correct_value=bearing,
            reference_bearing_deg=0,
            target_bearing_deg=bearing,
            angle_measure=None,
            object_label=label,
            distractors=distractors,
        )

    def _multiple_choice_problem(
        self,
        *,
        kind: AnglesBearingsQuestionKind,
        stem: str,
        correct_value: int,
        reference_bearing_deg: int,
        target_bearing_deg: int,
        angle_measure: str | None,
        object_label: str,
        distractors: tuple[int, int, int],
    ) -> Problem:
        values: list[int] = [self._normalize_value(kind=kind, value=correct_value)]
        for distractor in distractors:
            value = self._normalize_value(kind=kind, value=distractor)
            if value in values:
                continue
            values.append(value)
            if len(values) == 4:
                break

        while len(values) < 5:
            candidate = self._fallback_distractor(
                kind=kind,
                correct_value=int(correct_value),
            )
            if candidate in values:
                continue
            values.append(candidate)

        order = self._rng.sample([0, 1, 2, 3, 4], k=5)
        shuffled = [values[idx] for idx in order]

        options = tuple(
            AnglesBearingsOption(
                code=idx + 1,
                text=self._format_option(kind=kind, value=value),
                value_deg=int(value),
            )
            for idx, value in enumerate(shuffled)
        )
        correct_code = next(opt.code for opt in options if opt.value_deg == int(correct_value))

        prompt_lines = [stem, ""]
        for option in options:
            prompt_lines.append(f"{option.code}) {option.text}")
        prompt = "\n".join(prompt_lines)

        payload = AnglesBearingsDegreesPayload(
            kind=kind,
            stem=stem,
            reference_bearing_deg=int(reference_bearing_deg),
            target_bearing_deg=int(target_bearing_deg),
            angle_measure=angle_measure,
            object_label=str(object_label),
            options=options,
            correct_code=int(correct_code),
            correct_value_deg=int(correct_value),
        )
        return Problem(
            prompt=prompt,
            answer=int(correct_code),
            payload=payload,
        )

    def _normalize_value(self, *, kind: AnglesBearingsQuestionKind, value: int) -> int:
        if kind is AnglesBearingsQuestionKind.BEARING_FROM_REFERENCE:
            return int(value) % 360
        normalized = int(value) % 360
        if normalized == 0:
            return 360
        return normalized

    def _fallback_distractor(self, *, kind: AnglesBearingsQuestionKind, correct_value: int) -> int:
        if kind is AnglesBearingsQuestionKind.BEARING_FROM_REFERENCE:
            delta = self._rng.choice((10, 20, 30, 40, 50))
            direction = -1 if self._rng.random() < 0.5 else 1
            return (int(correct_value) + (direction * delta)) % 360

        delta = self._rng.choice((5, 10, 15, 20, 25, 30))
        direction = -1 if self._rng.random() < 0.5 else 1
        candidate = int(correct_value) + (direction * delta)
        if candidate < 1:
            return 1
        if candidate > 359:
            return 359
        return candidate

    def _format_option(self, *, kind: AnglesBearingsQuestionKind, value: int) -> str:
        if kind is AnglesBearingsQuestionKind.BEARING_FROM_REFERENCE:
            return f"{int(value) % 360:03d}"
        return str(int(value))

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
        "Angles, Bearings and Degrees (Multiple Choice)",
        "",
        "Estimate angles and bearings quickly and accurately.",
        "Part A: estimate the smaller angle between two lines.",
        "Part B: estimate the bearing of an object from a reference point.",
        "",
        "Controls:",
        "- Press A, S, D, F, or G to choose an option",
        "- Use Up/Down to move between options",
        "- Press Enter to submit",
        "",
        "You will get a short practice, then a timed scored block.",
        "Once the timed block starts, continue until completion.",
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
    )
