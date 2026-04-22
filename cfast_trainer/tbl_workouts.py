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
            "Start on clean one-table row and column lookups before denser scans and tab switches appear.",
            ("Single-table lookup", "Tabbed answer flow"),
            "tbl_part1_anchor",
            AntDrillMode.BUILD,
            10 * scale,
        ),
        _block(
            "part1_scan",
            "Part 1 Scan Run",
            "Push wider scans and tighter distractors while still staying on the one-table workflow.",
            ("Scan speed", "Single-table discrimination"),
            "tbl_part1_scan_run",
            AntDrillMode.BUILD,
            10 * scale,
        ),
        _block(
            "part2_prime",
            "Part 2 Prime",
            "Warm up the easier two-table chain before the full correction workflow tightens up.",
            ("Two-table chaining", "Intermediate value control"),
            "tbl_part2_prime",
            AntDrillMode.BUILD,
            10 * scale,
        ),
        _block(
            "part2_correction",
            "Part 2 Correction Run",
            "Run the full two-table correction workflow at tempo with the real live table layout still active.",
            ("Two-table correction", "Cross-table carryover"),
            "tbl_part2_correction_run",
            AntDrillMode.TEMPO,
            10 * scale,
        ),
        _block(
            "part_switch",
            "Part Switch Run",
            "Alternate one-table, two-table, and typed-answer items so the workflow change itself stops costing time.",
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
            "Hold a mixed rhythm across choices, numeric entry, letter search, and multi-table lookups.",
            ("Mixed table tempo", "Answer-mode switching"),
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
            "Standard 90-minute Table Reading workout with tabbed one-table and "
            "multi-table warm-ups, answer-mode switching blocks, and a final pressure run."
        ),
        notes=(
            "Block setup screens do not count toward the 90-minute drill clock.",
            "Every block reuses the live Table Reading UI with a question tab plus table-data tabs.",
            "Controls stay keyboard-first throughout: Tab switches views; choices use A/S/D/F/G or 1-5; typed numeric and letter answers use Enter.",
        ),
        blocks=blocks,
    )


def tbl_workout_menu_entries() -> tuple[tuple[str, str], ...]:
    return (("table_reading_workout", "Table Reading Workout (90m)"),)
