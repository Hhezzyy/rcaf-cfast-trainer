from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import cast

from .angles_bearings_degrees import (
    AnglesBearingsDegreesGenerator,
    AnglesBearingsQuestionKind,
    AnglesBearingsTrainingPayload,
    format_angles_bearings_value,
)
from .ant_drills import (
    ANT_DRILL_MODE_PROFILES,
    AntAdaptiveDifficultyConfig,
    AntDrillMode,
    TimedCapDrill,
    _clamp01,
)
from .clock import Clock
from .cognitive_core import AnswerScorer, Problem, SeededRng


def _difficulty_to_level(difficulty: float) -> int:
    return max(1, min(10, int(round(_clamp01(difficulty) * 9.0)) + 1))


def _level_to_difficulty(level: int) -> float:
    clamped = max(1, min(10, int(level)))
    return float(clamped - 1) / 9.0


def _normalize_mode(mode: AntDrillMode | str) -> AntDrillMode:
    return mode if isinstance(mode, AntDrillMode) else AntDrillMode(str(mode).strip().lower())


def _round_to_step(value: int, step: int) -> int:
    step = max(1, int(step))
    return int(round(int(value) / step) * step)


def _round_bearing_to_step(value: int, step: int) -> int:
    return _round_to_step(int(value) % 360, step) % 360


def _anchor_angle_values(level: int, *, intermediate: bool) -> tuple[int, ...]:
    values = [0, 45, 90, 135, 180]
    if intermediate:
        if level >= 3:
            values.extend([30, 60, 120, 150])
        if level >= 7:
            values.extend([15, 75, 105, 165])
    return tuple(dict.fromkeys(values))


def _anchor_bearing_values(level: int, *, intermediate: bool) -> tuple[int, ...]:
    values = [0, 45, 90, 135, 180, 225, 270, 315]
    if intermediate:
        if level >= 3:
            values.extend([30, 60, 120, 150, 210, 240, 300, 330])
        if level >= 7:
            values.extend([15, 75, 105, 165, 195, 255, 285, 345])
    return tuple(dict.fromkeys(values))


def _bearing_test_values(level: int) -> tuple[int, ...]:
    values = [0, 90, 180, 270]
    if level >= 5:
        values.extend([45, 135, 225, 315])
    return tuple(values)


def _angle_base_cap(level: int) -> float:
    return (16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0)[level - 1]


def _bearing_base_cap(level: int) -> float:
    return (16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0)[level - 1]


def _family_run_cap(level: int) -> float:
    return (15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0)[level - 1]


def _anchor_prompt(kind: AnglesBearingsQuestionKind, value: int) -> str:
    if kind is AnglesBearingsQuestionKind.BEARING_FROM_REFERENCE:
        if value % 360 == 0:
            return "Find the bearing of the point. North may be entered as 000 or 360."
        return "Find the bearing of the point and enter it exactly."
    return "Estimate the smaller angle between the rays and enter it exactly."


def _bearing_marker_radius_ratio(rng: SeededRng, level: int) -> float:
    if level <= 3:
        choices = (0.86, 0.80, 0.74)
    elif level <= 6:
        choices = (0.74, 0.66, 0.58)
    else:
        choices = (0.62, 0.54, 0.46, 0.38)
    return float(rng.choice(choices))


def _training_payload(
    *,
    kind: AnglesBearingsQuestionKind,
    stem: str,
    reference_bearing_deg: int,
    target_bearing_deg: int,
    rounded_value_deg: int,
    exact_value_deg: int,
    base_cap_s: float,
    allow_north_360_alias: bool = False,
    marker_radius_ratio: float = 0.84,
) -> AnglesBearingsTrainingPayload:
    display_answer = format_angles_bearings_value(
        kind=kind,
        value=exact_value_deg,
        north_alias=allow_north_360_alias,
    )
    return AnglesBearingsTrainingPayload(
        kind=kind,
        stem=stem,
        reference_bearing_deg=int(reference_bearing_deg) % 360,
        target_bearing_deg=int(target_bearing_deg) % 360,
        angle_measure="smaller" if kind is AnglesBearingsQuestionKind.ANGLE_BETWEEN_LINES else None,
        object_label="A" if kind is AnglesBearingsQuestionKind.BEARING_FROM_REFERENCE else "",
        rounded_value_deg=int(rounded_value_deg),
        exact_value_deg=int(exact_value_deg),
        display_answer_text=display_answer,
        input_digits=3,
        answer_digits=3 if kind is AnglesBearingsQuestionKind.BEARING_FROM_REFERENCE else 1,
        allow_north_360_alias=bool(allow_north_360_alias),
        base_cap_s=float(base_cap_s),
        marker_radius_ratio=float(marker_radius_ratio),
    )


class AbdTypedAnswerScorer(AnswerScorer):
    def score(self, *, problem: Problem, user_answer: int, raw: str) -> float:
        payload = cast(AnglesBearingsTrainingPayload | None, problem.payload)
        if payload is None:
            return 0.0
        token = str(raw).strip()
        if token == "":
            return 0.0
        if (
            payload.kind is AnglesBearingsQuestionKind.BEARING_FROM_REFERENCE
            and payload.allow_north_360_alias
            and int(problem.answer) == 0
            and token == "360"
        ):
            return 1.0
        return 1.0 if int(user_answer) == int(problem.answer) else 0.0


@dataclass(frozen=True, slots=True)
class AbdDrillConfig:
    practice_questions: int | None = None
    scored_duration_s: float | None = None
    adaptive: AntAdaptiveDifficultyConfig = field(
        default_factory=lambda: AntAdaptiveDifficultyConfig(enabled=False)
    )


@dataclass(frozen=True, slots=True)
class AbdFamilyRunConfig:
    family: AnglesBearingsQuestionKind
    practice_questions: int | None = None
    scored_duration_s: float | None = None
    adaptive: AntAdaptiveDifficultyConfig = field(
        default_factory=lambda: AntAdaptiveDifficultyConfig(enabled=False)
    )


class _AbdTypedGenerator:
    def cap_for_problem(self, *, problem: Problem, level: int) -> float | None:
        _ = level
        payload = cast(AnglesBearingsTrainingPayload | None, problem.payload)
        if payload is None or payload.base_cap_s is None:
            return None
        return float(payload.base_cap_s)


class AbdCardinalAnchorsGenerator(_AbdTypedGenerator):
    def __init__(self, *, seed: int) -> None:
        self._rng = SeededRng(seed)

    def next_problem(self, *, difficulty: float) -> Problem:
        level = _difficulty_to_level(difficulty)
        kind = AnglesBearingsQuestionKind(str(self._rng.choice(tuple(AnglesBearingsQuestionKind))))
        if kind is AnglesBearingsQuestionKind.BEARING_FROM_REFERENCE:
            value = int(self._rng.choice(_anchor_bearing_values(level, intermediate=False)))
            stem = _anchor_prompt(kind, value)
            payload = _training_payload(
                kind=kind,
                stem=stem,
                reference_bearing_deg=0,
                target_bearing_deg=value,
                rounded_value_deg=value % 360,
                exact_value_deg=value % 360,
                base_cap_s=_bearing_base_cap(level),
                allow_north_360_alias=value % 360 == 0,
                marker_radius_ratio=_bearing_marker_radius_ratio(self._rng, level),
            )
            return Problem(prompt=stem, answer=value % 360, payload=payload)

        angle = int(self._rng.choice(_anchor_angle_values(level, intermediate=False)))
        reference = 0
        clockwise = self._rng.random() < 0.5
        target = (reference + angle) % 360 if clockwise else (reference - angle) % 360
        stem = _anchor_prompt(kind, angle)
        payload = _training_payload(
            kind=kind,
            stem=stem,
            reference_bearing_deg=reference,
            target_bearing_deg=target,
            rounded_value_deg=angle,
            exact_value_deg=angle,
            base_cap_s=_angle_base_cap(level),
        )
        return Problem(prompt=stem, answer=angle, payload=payload)


class AbdIntermediateAnchorsGenerator(_AbdTypedGenerator):
    def __init__(self, *, seed: int) -> None:
        self._rng = SeededRng(seed)

    def next_problem(self, *, difficulty: float) -> Problem:
        level = _difficulty_to_level(difficulty)
        kind = AnglesBearingsQuestionKind(str(self._rng.choice(tuple(AnglesBearingsQuestionKind))))
        if kind is AnglesBearingsQuestionKind.BEARING_FROM_REFERENCE:
            value = int(self._rng.choice(_anchor_bearing_values(level, intermediate=True)))
            stem = "Find the bearing of the point and enter it exactly."
            payload = _training_payload(
                kind=kind,
                stem=stem,
                reference_bearing_deg=0,
                target_bearing_deg=value,
                rounded_value_deg=value % 360,
                exact_value_deg=value % 360,
                base_cap_s=max(6.0, _bearing_base_cap(level) - 1.0),
                marker_radius_ratio=_bearing_marker_radius_ratio(self._rng, level),
            )
            return Problem(prompt=stem, answer=value % 360, payload=payload)

        angle = int(self._rng.choice(_anchor_angle_values(level, intermediate=True)))
        reference = 0
        clockwise = self._rng.random() < 0.5
        target = (reference + angle) % 360 if clockwise else (reference - angle) % 360
        stem = "Estimate the smaller angle between the rays and enter it exactly."
        payload = _training_payload(
            kind=kind,
            stem=stem,
            reference_bearing_deg=reference,
            target_bearing_deg=target,
            rounded_value_deg=angle,
            exact_value_deg=angle,
            base_cap_s=max(6.0, _angle_base_cap(level) - 1.0),
        )
        return Problem(prompt=stem, answer=angle, payload=payload)


class AbdAngleCalibrationGenerator(_AbdTypedGenerator):
    def __init__(self, *, seed: int) -> None:
        self._rng = SeededRng(seed)

    def next_problem(self, *, difficulty: float) -> Problem:
        level = _difficulty_to_level(difficulty)
        min_angle = max(5, 95 - (level * 6))
        max_angle = min(179, 110 + (level * 7))
        exact = int(self._rng.randint(min_angle, max_angle))
        rounded = max(0, min(180, _round_to_step(exact, 5)))
        reference = 0 if level <= 6 else int(self._rng.randint(0, 359))
        clockwise = self._rng.random() < 0.5
        target = (reference + exact) % 360 if clockwise else (reference - exact) % 360
        stem = "Estimate the smaller angle between the rays. Enter the nearest 5 degrees."
        payload = _training_payload(
            kind=AnglesBearingsQuestionKind.ANGLE_BETWEEN_LINES,
            stem=stem,
            reference_bearing_deg=reference,
            target_bearing_deg=target,
            rounded_value_deg=rounded,
            exact_value_deg=exact,
            base_cap_s=_angle_base_cap(level),
        )
        return Problem(prompt=stem, answer=rounded, payload=payload)


class AbdBearingCalibrationGenerator(_AbdTypedGenerator):
    def __init__(self, *, seed: int) -> None:
        self._rng = SeededRng(seed)

    def next_problem(self, *, difficulty: float) -> Problem:
        level = _difficulty_to_level(difficulty)
        quant_step = 90 if level <= 4 else 45
        exact = int(self._rng.choice(_bearing_test_values(level)))
        rounded = exact % 360
        stem = "Estimate the bearing of point A and enter it exactly as 000-359."
        payload = _training_payload(
            kind=AnglesBearingsQuestionKind.BEARING_FROM_REFERENCE,
            stem=stem,
            reference_bearing_deg=0,
            target_bearing_deg=exact,
            rounded_value_deg=rounded,
            exact_value_deg=exact,
            base_cap_s=_bearing_base_cap(level),
            marker_radius_ratio=_bearing_marker_radius_ratio(self._rng, level),
        )
        return Problem(prompt=stem, answer=rounded, payload=payload)


class AbdMixedTempoGenerator(_AbdTypedGenerator):
    def __init__(self, *, seed: int) -> None:
        self._rng = SeededRng(seed)
        self._angle = AbdAngleCalibrationGenerator(seed=seed + 101)
        self._bearing = AbdBearingCalibrationGenerator(seed=seed + 202)
        self._index = 0
        self._last_kind: AnglesBearingsQuestionKind | None = None

    def next_problem(self, *, difficulty: float) -> Problem:
        self._index += 1
        level = _difficulty_to_level(difficulty)
        tier = "hard" if self._index % 3 == 0 else "easy"
        local_level = min(10, level + 1) if tier == "hard" else max(1, level - 2)
        local_difficulty = _level_to_difficulty(local_level)

        if self._last_kind is AnglesBearingsQuestionKind.ANGLE_BETWEEN_LINES:
            use_kind = (
                AnglesBearingsQuestionKind.BEARING_FROM_REFERENCE
                if self._rng.random() < 0.8
                else AnglesBearingsQuestionKind.ANGLE_BETWEEN_LINES
            )
        elif self._last_kind is AnglesBearingsQuestionKind.BEARING_FROM_REFERENCE:
            use_kind = (
                AnglesBearingsQuestionKind.ANGLE_BETWEEN_LINES
                if self._rng.random() < 0.8
                else AnglesBearingsQuestionKind.BEARING_FROM_REFERENCE
            )
        else:
            use_kind = AnglesBearingsQuestionKind(str(self._rng.choice(tuple(AnglesBearingsQuestionKind))))

        problem = (
            self._angle.next_problem(difficulty=local_difficulty)
            if use_kind is AnglesBearingsQuestionKind.ANGLE_BETWEEN_LINES
            else self._bearing.next_problem(difficulty=local_difficulty)
        )
        payload = cast(AnglesBearingsTrainingPayload | None, problem.payload)
        if payload is not None:
            scale = 0.85 if tier == "hard" else 1.08
            payload = replace(payload, base_cap_s=max(4.5, float(payload.base_cap_s or 8.0) * scale))
            problem = Problem(
                prompt=problem.prompt,
                answer=problem.answer,
                tolerance=problem.tolerance,
                payload=payload,
            )
        self._last_kind = use_kind
        return problem


class AbdTestStyleFamilyRunGenerator:
    def __init__(
        self,
        *,
        seed: int,
        family: AnglesBearingsQuestionKind,
    ) -> None:
        self._base = AnglesBearingsDegreesGenerator(seed=seed, allowed_kinds=(family,))

    def next_problem(self, *, difficulty: float) -> Problem:
        return self._base.next_problem(difficulty=difficulty)

    def cap_for_problem(self, *, problem: Problem, level: int) -> float | None:
        _ = problem
        return _family_run_cap(level)


def _build_abd_drill(
    *,
    title_base: str,
    instructions: Sequence[str],
    generator: object,
    clock: Clock,
    seed: int,
    difficulty: float,
    mode: AntDrillMode | str,
    config: AbdDrillConfig,
    base_caps_by_level: Sequence[float],
) -> TimedCapDrill:
    normalized_mode = _normalize_mode(mode)
    profile = ANT_DRILL_MODE_PROFILES[normalized_mode]
    practice_questions = (
        profile.practice_questions if config.practice_questions is None else int(config.practice_questions)
    )
    scored_duration_s = (
        profile.scored_duration_s if config.scored_duration_s is None else float(config.scored_duration_s)
    )
    return TimedCapDrill(
        title=f"{title_base} ({profile.label})",
        instructions=list(instructions),
        generator=generator,
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        practice_questions=practice_questions,
        scored_duration_s=scored_duration_s,
        mode=normalized_mode,
        base_caps_by_level=base_caps_by_level,
        adaptive_config=config.adaptive,
        scorer=AbdTypedAnswerScorer(),
        immediate_feedback_override=True,
    )


def build_abd_cardinal_anchors_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: AbdDrillConfig | None = None,
) -> TimedCapDrill:
    cfg = config or AbdDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_abd_drill(
        title_base="Angles, Bearings and Degrees: Cardinal Anchors",
        instructions=(
            "Angles, Bearings and Degrees: Cardinal Anchors",
            f"Mode: {profile.label}",
            "Build the clean anchor points first: cardinals plus diagonal 45-degree landmarks.",
            "North can be entered as 000 or 360 only when the prompt says so.",
            "Exact answers flash with the next item.",
            "Press Enter to begin practice.",
        ),
        generator=AbdCardinalAnchorsGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0),
    )


def build_abd_intermediate_anchors_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: AbdDrillConfig | None = None,
) -> TimedCapDrill:
    cfg = config or AbdDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_abd_drill(
        title_base="Angles, Bearings and Degrees: Intermediate Anchors",
        instructions=(
            "Angles, Bearings and Degrees: Intermediate Anchors",
            f"Mode: {profile.label}",
            "Add diagonals first, then cleaner in-between landmarks before arbitrary estimates.",
            "Enter exact answers and use the next-item flash to calibrate fast.",
            "Press Enter to begin practice.",
        ),
        generator=AbdIntermediateAnchorsGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0),
    )


def build_abd_angle_calibration_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: AbdDrillConfig | None = None,
) -> TimedCapDrill:
    cfg = config or AbdDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_abd_drill(
        title_base="Angles, Bearings and Degrees: Angle Calibration",
        instructions=(
            "Angles, Bearings and Degrees: Angle Calibration",
            f"Mode: {profile.label}",
            "Type the nearest 5 degrees for each angle-between-lines item.",
            "The exact angle flashes with the next prompt so you keep tempo while calibrating.",
            "Press Enter to begin practice.",
        ),
        generator=AbdAngleCalibrationGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0),
    )


def build_abd_bearing_calibration_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: AbdDrillConfig | None = None,
) -> TimedCapDrill:
    cfg = config or AbdDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_abd_drill(
        title_base="Angles, Bearings and Degrees: Bearing Calibration",
        instructions=(
            "Angles, Bearings and Degrees: Bearing Calibration",
            f"Mode: {profile.label}",
            "Type the nearest 5-degree bearing for each item.",
            "The exact bearing flashes with the next prompt so you keep tempo while calibrating.",
            "Press Enter to begin practice.",
        ),
        generator=AbdBearingCalibrationGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0),
    )


def build_abd_mixed_tempo_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.TEMPO,
    config: AbdDrillConfig | None = None,
) -> TimedCapDrill:
    cfg = config or AbdDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_abd_drill(
        title_base="Angles, Bearings and Degrees: Mixed Tempo",
        instructions=(
            "Angles, Bearings and Degrees: Mixed Tempo",
            f"Mode: {profile.label}",
            "Typed mixed stream with an easy/easy/hard cadence.",
            "Enter the nearest 5 degrees, accept misses fast, and recalibrate from the exact flash on the next item.",
            "Press Enter to begin practice.",
        ),
        generator=AbdMixedTempoGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0),
    )


def build_abd_test_style_family_run_drill(
    *,
    clock: Clock,
    seed: int,
    family: AnglesBearingsQuestionKind,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.TEMPO,
    config: AbdFamilyRunConfig | None = None,
) -> TimedCapDrill:
    normalized_mode = _normalize_mode(mode)
    profile = ANT_DRILL_MODE_PROFILES[normalized_mode]
    cfg = config or AbdFamilyRunConfig(family=family)
    practice_questions = (
        profile.practice_questions if cfg.practice_questions is None else int(cfg.practice_questions)
    )
    scored_duration_s = (
        profile.scored_duration_s if cfg.scored_duration_s is None else float(cfg.scored_duration_s)
    )
    family_label = (
        "Angle Run"
        if family is AnglesBearingsQuestionKind.ANGLE_BETWEEN_LINES
        else "Bearing Run"
    )
    family_note = (
        "All items are angle-between-lines multiple choice."
        if family is AnglesBearingsQuestionKind.ANGLE_BETWEEN_LINES
        else "All items are bearing-from-reference multiple choice."
    )
    return TimedCapDrill(
        title=f"Angles, Bearings and Degrees: Test-Style {family_label} ({profile.label})",
        instructions=[
            f"Angles, Bearings and Degrees: Test-Style {family_label}",
            f"Mode: {profile.label}",
            family_note,
            "This block uses the real ABD multiple-choice layout and answer flow.",
            "Feedback still flashes with the next prompt because this is workout content, not scored test mode.",
            "Press Enter to begin practice.",
        ],
        generator=AbdTestStyleFamilyRunGenerator(seed=seed, family=family),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        practice_questions=practice_questions,
        scored_duration_s=scored_duration_s,
        mode=normalized_mode,
        base_caps_by_level=(15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0),
        adaptive_config=cfg.adaptive,
        immediate_feedback_override=True,
    )
