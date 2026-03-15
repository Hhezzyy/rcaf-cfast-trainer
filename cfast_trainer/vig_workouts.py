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


def build_vig_workout_plan(*, duration_scale: float = 1.0) -> AntWorkoutPlan:
    scale = max(0.05, float(duration_scale))
    blocks = (
        _block(
            "entry_anchor",
            "Entry Anchor Warm-Up",
            "Start with the slowest full-board rhythm so row and column entry stays clean before density builds.",
            ("Coordinate entry stability", "Early scan rhythm"),
            "vig_entry_anchor",
            AntDrillMode.BUILD,
            10 * scale,
        ),
        _block(
            "clean_scan",
            "Clean Scan Warm-Up",
            "Keep the real board and symbol mix, but use a lighter stream to settle a disciplined scan path.",
            ("Full-board scan discipline", "Coordinate entry stability"),
            "vig_clean_scan",
            AntDrillMode.BUILD,
            10 * scale,
        ),
        _block(
            "steady_capture",
            "Steady Capture Run",
            "Move into sustained baseline work without changing the live Vigilance rules or score meaning.",
            ("Sustained capture rhythm", "Accuracy under pace"),
            "vig_steady_capture_run",
            AntDrillMode.TEMPO,
            15 * scale,
        ),
        _block(
            "density_ladder",
            "Density Ladder",
            "Increase overlap and cleanup pressure while keeping the same row/column response flow.",
            ("Overlap management", "Scan recovery"),
            "vig_density_ladder",
            AntDrillMode.TEMPO,
            15 * scale,
        ),
        _block(
            "tempo_sweep",
            "Tempo Sweep",
            "Hold the real task at a faster sustained pace without changing the board layout or point values.",
            ("Sustained tempo", "Accuracy under pace"),
            "vig_tempo_sweep",
            AntDrillMode.TEMPO,
            20 * scale,
        ),
        _block(
            "pressure_run",
            "Pressure Run",
            "Finish on the hardest full-board stream in the Vigilance family and recover immediately after misses.",
            ("Pressure tolerance", "Coordinate-entry resilience"),
            "vig_pressure_run",
            AntDrillMode.STRESS,
            20 * scale,
        ),
    )
    return AntWorkoutPlan(
        code="vigilance_workout",
        title="Vigilance Workout (90m)",
        description=(
            "Standard 90-minute Vigilance workout with row/column entry warm-ups, denser tempo blocks, "
            "and a final full-board pressure run."
        ),
        notes=(
            "Typed reflections and block setup screens do not count toward the 90-minute drill clock.",
            "Every block keeps the real 9x9 Vigilance board, the normal symbol rarity and point values, and the standard row/column entry flow.",
            "Early blocks bias scan discipline and clean entry; later blocks only change pace and overlap density.",
        ),
        blocks=blocks,
    )


def vig_workout_menu_entries() -> tuple[tuple[str, str], ...]:
    return (("vigilance_workout", "Vigilance Workout (90m)"),)
