from __future__ import annotations

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


def build_ic_workout_plan(*, duration_scale: float = 1.0) -> AntWorkoutPlan:
    scale = max(0.05, float(duration_scale))
    blocks = (
        _block(
            "heading_anchor",
            "Heading Anchor Warm-Up",
            "Keep bank and pitch nearly level so the heading picture and compass-rose relation become automatic first.",
            ("Heading anchoring", "Part 1 warm-up"),
            "ic_heading_anchor",
            AntDrillMode.BUILD,
            8 * scale,
        ),
        _block(
            "attitude_frame",
            "Attitude Frame Warm-Up",
            "Hold heading coarse and sharpen bank/pitch aircraft-image discrimination before the full Part 1 run.",
            ("Attitude framing", "Bank/pitch discrimination"),
            "ic_attitude_frame",
            AntDrillMode.BUILD,
            8 * scale,
        ),
        _block(
            "part1_orientation",
            "Part 1 Orientation Run",
            "Use the real aircraft-card answer set and build speed on full instrument-to-aircraft matching.",
            ("Part 1 matching", "Orientation discrimination"),
            "ic_part1_orientation_run",
            AntDrillMode.TEMPO,
            12 * scale,
        ),
        _block(
            "reverse_panel_prime",
            "Reverse Panel Prime",
            "Read one aircraft image and clean up the easier full-panel reverse matches before the tempo run.",
            ("Part 2 warm-up", "Reverse panel matching"),
            "ic_reverse_panel_prime",
            AntDrillMode.BUILD,
            8 * scale,
        ),
        _block(
            "reverse_panel_run",
            "Reverse Panel Run",
            "Build speed on full aircraft-image-to-instrument-panel matching before the final mixed blocks.",
            ("Part 2 matching", "Panel interpretation"),
            "ic_reverse_panel_run",
            AntDrillMode.TEMPO,
            12 * scale,
        ),
        _block(
            "description_prime",
            "Description Prime",
            "Clean up the easier one-dimension description distractors before the full Part 3 interpretation run.",
            ("Part 3 warm-up", "Description filtering"),
            "ic_description_prime",
            AntDrillMode.BUILD,
            8 * scale,
        ),
        _block(
            "description_run",
            "Part 3 Description Run",
            "Run the full instrument-panel-to-description interpretation flow under tempo pressure.",
            ("Part 3 interpretation", "Description matching"),
            "ic_description_run",
            AntDrillMode.TEMPO,
            12 * scale,
        ),
        _block(
            "mixed_part_run",
            "Mixed Three-Part Run",
            "Rehearse the full family with a fixed 2x Part 1, 2x Part 2, 2x Part 3 rhythm before the final pressure block.",
            ("Part switching", "Balanced IC rhythm"),
            "ic_mixed_part_run",
            AntDrillMode.TEMPO,
            12 * scale,
        ),
        _block(
            "pressure_run",
            "Pressure Run",
            "Finish with Part 1, Part 2, and Part 3 alternating item-by-item under the hardest cap profile in this family.",
            ("Pressure tolerance", "Mixed-part recovery"),
            "ic_pressure_run",
            AntDrillMode.STRESS,
            10 * scale,
        ),
    )
    return AntWorkoutPlan(
        code="instrument_comprehension_workout",
        title="Instrument Comprehension Workout (90m)",
        description=(
            "Standard 90-minute Instrument Comprehension workout with typed reflection, Part 1/2/3 "
            "warm-ups, balanced mixed runs, and a final pressure block."
        ),
        notes=(
            "Typed reflections and block setup screens do not count toward the 90-minute drill clock.",
            "All timed blocks stay on the real Instrument Comprehension layouts and keep the A/S/D/F/G multiple-choice answer flow.",
            "The family stays balanced across Part 1 aircraft matching, Part 2 reverse panel matching, and Part 3 description interpretation.",
        ),
        blocks=blocks,
    )


def ic_workout_menu_entries() -> tuple[tuple[str, str], ...]:
    return (("instrument_comprehension_workout", "Instrument Comprehension Workout (90m)"),)
