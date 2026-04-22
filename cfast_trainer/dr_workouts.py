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


def build_dr_workout_plan(*, duration_scale: float = 1.0) -> AntWorkoutPlan:
    scale = max(0.05, float(duration_scale))
    blocks = (
        _block(
            "visible_copy",
            "Visible Copy Warm-Up",
            "Type full strings while they remain visible so the encoding rhythm turns on before hidden recall begins.",
            ("Encoding rhythm", "Chunking", "Typed reproduction"),
            "dr_visible_copy",
            AntDrillMode.BUILD,
            10 * scale,
        ),
        _block(
            "position_probe",
            "Position Probe Warm-Up",
            "Anchor serial positions and short slices while the digits are still visible.",
            ("Serial-position anchoring", "Left-to-right scan discipline"),
            "dr_position_probe",
            AntDrillMode.BUILD,
            10 * scale,
        ),
        _block(
            "family_primer",
            "Visible Family Primer",
            "Touch the count-target, different-digit, and difference-count families with visible support before hidden-memory pressure starts.",
            ("Family familiarity", "Visible-supported extraction"),
            "dr_visible_family_primer",
            AntDrillMode.BUILD,
            8 * scale,
        ),
        _block(
            "recall_build",
            "Recall Build",
            "Move into real hidden-memory full-string recall with shorter display and mask timing.",
            ("Full-string recall", "Hidden-memory encoding"),
            "dr_recall_run",
            AntDrillMode.BUILD,
            14 * scale,
        ),
        _block(
            "count_target",
            "Count Target Tempo",
            "Count one target digit after the display disappears without getting stuck on repeated digits.",
            ("Target counting", "Dense-repeat handling"),
            "dr_count_target",
            AntDrillMode.TEMPO,
            10 * scale,
        ),
        _block(
            "different_digit",
            "Different Digit Tempo",
            "Identify the one changed digit after the display disappears and keep moving when the two strings feel similar.",
            ("Different-digit detection", "Similarity discrimination"),
            "dr_different_digit",
            AntDrillMode.TEMPO,
            10 * scale,
        ),
        _block(
            "difference_count",
            "Difference Count Tempo",
            "Judge how many positions changed between the two strings without drifting into full-string re-read mode.",
            ("Difference-count discrimination", "Fast comparison tally"),
            "dr_difference_count",
            AntDrillMode.TEMPO,
            10 * scale,
        ),
        _block(
            "grouped_family_run",
            "Grouped Family Run",
            "Run hidden-memory family chunks in a fixed order before the final mixed block.",
            ("Grouped family solving", "Family switching"),
            "dr_grouped_family_run",
            AntDrillMode.TEMPO,
            10 * scale,
        ),
        _block(
            "mixed_pressure",
            "Mixed Pressure",
            "Finish with a mixed hidden-memory pressure block that rotates recall, count, different-digit, and difference-count items under the hardest timings.",
            ("Mixed-family switching", "Pressure tolerance"),
            "dr_mixed_pressure",
            AntDrillMode.STRESS,
            8 * scale,
        ),
    )
    return AntWorkoutPlan(
        code="digit_recognition_workout",
        title="Digit Recognition Workout (90m)",
        description=(
            "Standard 90-minute Digit Recognition workout with visible scaffold blocks, "
            "hidden-memory family work, and a final mixed pressure block."
        ),
        notes=(
            "Block setup screens do not count toward the 90-minute drill clock.",
            "Early blocks allow visible support; later blocks switch to the real hidden-memory show, mask, and question rhythm.",
            "All Digit Recognition blocks stay typed digits only, with the family ladder expanding from recall into four non-recall comparison/count prompts.",
        ),
        blocks=blocks,
    )


def dr_workout_menu_entries() -> tuple[tuple[str, str], ...]:
    return (("digit_recognition_workout", "Digit Recognition Workout (90m)"),)
