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


def build_no_workout_plan(*, duration_scale: float = 1.0) -> AntWorkoutPlan:
    scale = max(0.05, float(duration_scale))
    blocks = (
        _block(
            "fact_prime",
            "Fact Prime",
            "Prime memoized addition, subtraction, multiplication, and division families before the heavier arithmetic blocks.",
            ("Arithmetic fact retrieval", "Pattern priming"),
            "no_fact_prime",
            AntDrillMode.BUILD,
            10 * scale,
        ),
        _block(
            "operator_ladders",
            "Operator Ladders",
            "Reinforce one operator family at a time so the arithmetic pattern becomes automatic before the ladder moves on.",
            ("Operator isolation", "Pattern reinforcement"),
            "no_operator_ladders",
            AntDrillMode.BUILD,
            10 * scale,
        ),
        _block(
            "clean_compute",
            "Clean Compute",
            "Use curated arithmetic shapes that transfer well into harder mixed mental math without overloading you too early.",
            ("Clean arithmetic", "Transferable patterns"),
            "no_clean_compute",
            AntDrillMode.BUILD,
            20 * scale,
        ),
        _block(
            "mixed_tempo",
            "Mixed Tempo",
            "Mixed arithmetic under caps with easy/easy/hard cadence so you can miss one item without losing the next three.",
            ("Tempo control", "Failure recovery"),
            "no_mixed_tempo",
            AntDrillMode.TEMPO,
            20 * scale,
        ),
        _block(
            "crunch_sprint",
            "Crunch Sprint",
            "Same mixed arithmetic stream, but under a harsher tempo profile that forces quick resets after misses.",
            ("Time crunch", "Failure recovery"),
            "no_mixed_tempo",
            AntDrillMode.STRESS,
            15 * scale,
        ),
        _block(
            "pressure_run",
            "Pressure Run",
            "Late-workout full-style Numerical Operations pressure block with the tightest cap profile.",
            ("Full-test rhythm", "Pressure tolerance"),
            "no_pressure_run",
            AntDrillMode.STRESS,
            15 * scale,
        ),
    )

    return AntWorkoutPlan(
        code="numerical_operations_workout",
        title="Numerical Operations Workout (90m)",
        description=(
            "Standard 90-minute Numerical Operations workout with warm-up blocks, "
            "core arithmetic tempo work, and late pressure runs."
        ),
        notes=(
            "Block setup screens do not count toward the 90-minute drill clock.",
            "Every block stays typed, matching the real Numerical Operations answer mode.",
            "The late blocks tighten caps to train recovery after misses rather than perfect streaks.",
        ),
        blocks=blocks,
    )


def no_workout_menu_entries() -> tuple[tuple[str, str], ...]:
    return (("numerical_operations_workout", "Numerical Operations Workout (90m)"),)
