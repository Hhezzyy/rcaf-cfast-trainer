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


def build_vs_workout_plan(*, duration_scale: float = 1.0) -> AntWorkoutPlan:
    scale = max(0.05, float(duration_scale))
    blocks = (
        _block(
            "target_preview",
            "Target Preview Warm-Up",
            "Reacquire the target fast on the full board before you worry about max tempo.",
            ("Target reacquisition", "Full-board familiarity"),
            "vs_target_preview",
            AntDrillMode.BUILD,
            10 * scale,
        ),
        _block(
            "clean_scan",
            "Clean Scan Warm-Up",
            "Use the full board with easier distractors and lock in a disciplined scan path.",
            ("Scan rhythm", "Full-board familiarity"),
            "vs_clean_scan",
            AntDrillMode.BUILD,
            10 * scale,
        ),
        _block(
            "letter_family",
            "Letter Family Run",
            "Stay on alphanumeric boards only and build speed without family switching yet.",
            ("Letter discrimination", "Typed block entry"),
            "vs_family_run_letters",
            AntDrillMode.TEMPO,
            15 * scale,
        ),
        _block(
            "line_figure_family",
            "Line Figure Family Run",
            "Stay on symbol-code boards only so line-figure confusions get cleaner before mixing returns.",
            ("Line-figure discrimination", "Typed block entry"),
            "vs_family_run_symbols",
            AntDrillMode.TEMPO,
            15 * scale,
        ),
        _block(
            "mixed_tempo",
            "Mixed Tempo",
            "Rotate both live families with easy-easy-hard pacing so you recover fast after a bad miss.",
            ("Family switching", "Tempo recovery", "Distractor tolerance"),
            "vs_mixed_tempo",
            AntDrillMode.TEMPO,
            20 * scale,
        ),
        _block(
            "pressure_run",
            "Pressure Run",
            "Finish with the real mixed-board answer mode under the hardest caps and tightest distractor similarity.",
            ("Pressure tolerance", "Mixed-family scanning", "Distractor tolerance"),
            "vs_pressure_run",
            AntDrillMode.STRESS,
            20 * scale,
        ),
    )
    return AntWorkoutPlan(
        code="visual_search_workout",
        title="Visual Search Workout (90m)",
        description=(
            "Standard 90-minute Visual Search workout with typed reflection, full-board warm-ups, "
            "family runs, mixed tempo, and a final pressure block."
        ),
        notes=(
            "Typed reflections and block setup screens do not count toward the 90-minute drill clock.",
            "Every block stays on the real 3x4 Visual Search board with one target and typed two-digit block-number answers.",
            "Early blocks keep distractors easier; later blocks tighten similarity and family switching without changing the board layout.",
        ),
        blocks=blocks,
    )


def vs_workout_menu_entries() -> tuple[tuple[str, str], ...]:
    return (("visual_search_workout", "Visual Search Workout (90m)"),)
