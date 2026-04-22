from __future__ import annotations

from .angles_bearings_degrees import AnglesBearingsQuestionKind
from .ant_drills import AntDrillMode
from .ant_workouts import AntWorkoutBlockPlan, AntWorkoutPlan


def _block(
    block_id: str,
    label: str,
    description: str,
    focus_skills: tuple[str, ...],
    drill_code: str,
    mode: AntDrillMode,
    minutes: float,
) -> AntWorkoutBlockPlan:
    return AntWorkoutBlockPlan(
        block_id=block_id,
        label=label,
        description=description,
        focus_skills=focus_skills,
        drill_code=drill_code,
        mode=mode,
        duration_min=minutes,
    )


def build_abd_workout_plan(*, duration_scale: float = 1.0) -> AntWorkoutPlan:
    scale = max(0.05, float(duration_scale))
    blocks = (
        _block(
            "cardinal_warmup",
            "Cardinal Anchor Warm-Up",
            "Build the clean cardinals plus diagonal 45-degree landmarks first so later estimates have stable reference points.",
            ("Cardinal bearings", "Diagonal anchors", "Straight-angle anchors"),
            "abd_cardinal_anchors",
            AntDrillMode.BUILD,
            8 * scale,
        ),
        _block(
            "intermediate_warmup",
            "Intermediate Anchor Warm-Up",
            "Add the cleaner in-between landmarks like 30, 60, 120, and 150 before arbitrary estimates.",
            ("Intermediate landmarks",),
            "abd_intermediate_anchors",
            AntDrillMode.BUILD,
            7 * scale,
        ),
        _block(
            "angle_calibration",
            "Angle Calibration",
            "Type the nearest 5 degrees for line-angle items and use the exact flash to recalibrate without stopping.",
            ("Angle estimation", "Calibration"),
            "abd_angle_calibration",
            AntDrillMode.BUILD,
            15 * scale,
        ),
        _block(
            "bearing_calibration",
            "Bearing Calibration",
            "Type the nearest 5 degrees for bearing items and correct your eye using the exact flash on the next prompt.",
            ("Bearing estimation", "Calibration"),
            "abd_bearing_calibration",
            AntDrillMode.BUILD,
            15 * scale,
        ),
        _block(
            "mixed_tempo",
            "Mixed Tempo Pressure",
            "Mixed typed angle and bearing estimates with an easy/easy/hard cadence to train recovery after misses.",
            ("Task switching", "Tempo recovery", "Nearest-5 estimation"),
            "abd_mixed_tempo",
            AntDrillMode.TEMPO,
            20 * scale,
        ),
        _block(
            "angle_run",
            "Test-Style Angle Run",
            "Real ABD multiple-choice angle-between-lines items under workout pressure.",
            ("Full-question solving", "Angle family run"),
            "abd_family_run_angle",
            AntDrillMode.TEMPO,
            12 * scale,
        ),
        _block(
            "bearing_run",
            "Test-Style Bearing Run",
            "Real ABD multiple-choice bearing items under workout pressure.",
            ("Full-question solving", "Bearing family run"),
            "abd_family_run_bearing",
            AntDrillMode.TEMPO,
            13 * scale,
        ),
    )

    return AntWorkoutPlan(
        code="angles_bearings_degrees_workout",
        title="Angles, Bearings and Degrees Workout (90m)",
        description=(
            "Standard 90-minute Angles, Bearings and Degrees workout with "
            "anchor warm-ups, typed calibration, and test-style pressure blocks."
        ),
        notes=(
            "Block setup screens do not count toward the 90-minute drill clock.",
            "Warm-up and calibration blocks use typed answers; the final family runs switch to the real multiple-choice ABD screen.",
            "North is taught as 000/360 in the anchor block, but later bearing blocks return to normal 000-359 scoring.",
        ),
        blocks=blocks,
    )


def abd_workout_menu_entries() -> tuple[tuple[str, str], ...]:
    return (("angles_bearings_degrees_workout", "Angles, Bearings and Degrees Workout (90m)"),)


def family_for_workout_drill_code(code: str) -> AnglesBearingsQuestionKind | None:
    token = str(code).strip().lower()
    if token == "abd_family_run_angle":
        return AnglesBearingsQuestionKind.ANGLE_BETWEEN_LINES
    if token == "abd_family_run_bearing":
        return AnglesBearingsQuestionKind.BEARING_FROM_REFERENCE
    return None
