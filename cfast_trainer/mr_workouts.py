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


def build_mr_workout_plan(*, duration_scale: float = 1.0) -> AntWorkoutPlan:
    scale = max(0.05, float(duration_scale))
    blocks = (
        _block(
            "relevant_info",
            "Relevant Info Scan",
            "Read the whole stem, find the one value that matters, and ignore the filler numbers around it.",
            ("Relevant information extraction", "Filler resistance"),
            "mr_relevant_info_scan",
            AntDrillMode.BUILD,
            10 * scale,
        ),
        _block(
            "unit_prime",
            "Unit And Relation Prime",
            "Prime the clean conversions and one-step relations that hold the later word problems together.",
            ("Unit conversion", "Relation priming"),
            "mr_unit_relation_prime",
            AntDrillMode.BUILD,
            10 * scale,
        ),
        _block(
            "one_step",
            "One-Step Solve",
            "Type exact answers to the clean one-step versions of the live math reasoning domains.",
            ("One-step solving",),
            "mr_one_step_solve",
            AntDrillMode.BUILD,
            15 * scale,
        ),
        _block(
            "multi_step",
            "Multi-Step Solve",
            "Move into filler-heavy word problems without losing the ask, the setup, or the tempo.",
            ("Multi-step solving", "Failure recovery"),
            "mr_multi_step_solve",
            AntDrillMode.TEMPO,
            20 * scale,
        ),
        _block(
            "domain_run",
            "Domain Run",
            "Real multiple-choice grouped domain work with motion and fuel first, then the remaining families.",
            ("Grouped domain solving", "Multiple-choice rhythm"),
            "mr_domain_run",
            AntDrillMode.TEMPO,
            15 * scale,
        ),
        _block(
            "mixed_pressure",
            "Mixed Pressure Set",
            "Late-workout mixed-domain multiple-choice pressure block with the tightest caps.",
            ("Pressure tolerance", "Mixed-domain switching"),
            "mr_mixed_pressure_set",
            AntDrillMode.STRESS,
            20 * scale,
        ),
    )
    return AntWorkoutPlan(
        code="math_reasoning_workout",
        title="Mathematics Reasoning Workout (90m)",
        description=(
            "Standard 90-minute Mathematics Reasoning workout with typed reflection, extraction and setup warm-ups, "
            "typed solve blocks, and late multiple-choice pressure work."
        ),
        notes=(
            "Typed reflections and block setup screens do not count toward the 90-minute drill clock.",
            "Warm-up and core blocks stay typed; the late blocks switch to the real multiple-choice Mathematics Reasoning format.",
            "The extraction block is about finding the right values first, not solving too early.",
        ),
        blocks=blocks,
    )


def mr_workout_menu_entries() -> tuple[tuple[str, str], ...]:
    return (("math_reasoning_workout", "Mathematics Reasoning Workout (90m)"),)
