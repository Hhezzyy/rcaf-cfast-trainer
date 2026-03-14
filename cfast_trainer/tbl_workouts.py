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


def build_tbl_workout_plan(*, duration_scale: float = 1.0) -> AntWorkoutPlan:
    scale = max(0.05, float(duration_scale))
    blocks = (
        _block(
            "part1_anchor",
            "Part 1 Anchor Warm-Up",
            "Start on clean single-card row and column lookups before denser scans and card switches appear.",
            ("Single-table lookup", "Keyboard answer flow"),
            "tbl_part1_anchor",
            AntDrillMode.BUILD,
            10 * scale,
        ),
        _block(
            "part1_scan",
            "Part 1 Scan Run",
            "Push wider scans and tighter distractors while still staying on the single-card workflow.",
            ("Scan speed", "Single-card discrimination"),
            "tbl_part1_scan_run",
            AntDrillMode.BUILD,
            10 * scale,
        ),
        _block(
            "part2_prime",
            "Part 2 Prime",
            "Warm up the easier two-card chain before the full correction workflow tightens up.",
            ("Two-card chaining", "Intermediate value control"),
            "tbl_part2_prime",
            AntDrillMode.BUILD,
            10 * scale,
        ),
        _block(
            "part2_correction",
            "Part 2 Correction Run",
            "Run the full two-card correction workflow at tempo with the real live table layout still active.",
            ("Two-card correction", "Cross-card carryover"),
            "tbl_part2_correction_run",
            AntDrillMode.TEMPO,
            10 * scale,
        ),
        _block(
            "part_switch",
            "Part Switch Run",
            "Alternate one-card and two-card items so the workflow change itself stops costing time.",
            ("Part switching", "Workflow reset"),
            "tbl_part_switch_run",
            AntDrillMode.TEMPO,
            10 * scale,
        ),
        _block(
            "card_family",
            "Card Family Run",
            "Rotate through the expanded card-pack and card-set library so one familiar sheet cannot carry the block.",
            ("Card-family adaptation", "Pattern refresh"),
            "tbl_card_family_run",
            AntDrillMode.TEMPO,
            10 * scale,
        ),
        _block(
            "mixed_tempo",
            "Mixed Tempo Run",
            "Hold a fixed 2x Part 1, 2x Part 2 rhythm and keep the same keyboard flow across both sections.",
            ("Mixed table tempo", "Section rhythm control"),
            "tbl_mixed_tempo",
            AntDrillMode.TEMPO,
            15 * scale,
        ),
        _block(
            "pressure_run",
            "Pressure Run",
            "Finish with alternating Part 1 and Part 2 items under the hardest cap profile while partial credit still applies.",
            ("Pressure tolerance", "Full Table Reading integration"),
            "tbl_pressure_run",
            AntDrillMode.STRESS,
            15 * scale,
        ),
    )
    return AntWorkoutPlan(
        code="table_reading_workout",
        title="Table Reading Workout (90m)",
        description=(
            "Standard 90-minute Table Reading workout with typed reflection, single-card and two-card "
            "warm-ups, switching blocks, and a final pressure run."
        ),
        notes=(
            "Typed reflections and block setup screens do not count toward the 90-minute drill clock.",
            "Every block reuses the live Table Reading UI with the real card tables and the normal answer strip.",
            "Controls stay keyboard-only throughout: Up/Down to move selection, A/S/D/F/G or 1-5 to choose, then Enter to submit.",
        ),
        blocks=blocks,
    )


def tbl_workout_menu_entries() -> tuple[tuple[str, str], ...]:
    return (("table_reading_workout", "Table Reading Workout (90m)"),)
