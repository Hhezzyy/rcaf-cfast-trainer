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


def build_cu_workout_plan(*, duration_scale: float = 1.0) -> AntWorkoutPlan:
    scale = max(0.05, float(duration_scale))
    blocks = (
        _block(
            "controls_anchor",
            "Controls Anchor Warm-Up",
            "Warm up pump and pressure correction before the mixed-code work begins.",
            ("Controls correction", "Pressure management"),
            "cu_controls_anchor",
            AntDrillMode.BUILD,
            10 * scale,
        ),
        _block(
            "navigation_anchor",
            "Navigation Anchor Warm-Up",
            "Bring speed back to the target knots cleanly before the family starts mixing domains.",
            ("Navigation correction", "Speed control"),
            "cu_navigation_anchor",
            AntDrillMode.BUILD,
            10 * scale,
        ),
        _block(
            "engine_balance",
            "Engine Balance Run",
            "Build active-tank switching and tank-spread control while the rest of the task stays neutral.",
            ("Engine balance", "Fuel management"),
            "cu_engine_balance_run",
            AntDrillMode.BUILD,
            10 * scale,
        ),
        _block(
            "sensors_prime",
            "Sensors Timing Prime",
            "Prime Alpha, Bravo, air, and ground timing windows before the mixed blocks.",
            ("Sensor timing", "Panel timing"),
            "cu_sensors_timing_prime",
            AntDrillMode.BUILD,
            10 * scale,
        ),
        _block(
            "objective_prime",
            "Objective Prime",
            "Warm up parcel entry, field switching, and dispenser timing while other domains stay neutral.",
            ("Objective execution", "Parcel timing"),
            "cu_objective_prime",
            AntDrillMode.BUILD,
            10 * scale,
        ),
        _block(
            "state_code_run",
            "State Code Run",
            "Blend controls, navigation, and sensors so the live 4-digit comms/state code becomes cleaner under tempo pressure.",
            ("State-code synthesis", "Focused mixed scan"),
            "cu_state_code_run",
            AntDrillMode.TEMPO,
            10 * scale,
        ),
        _block(
            "mixed_tempo",
            "Mixed Tempo Run",
            "Repeat the fixed Controls, Navigation, Engine, Sensors, Objectives, State Code, and Full Mixed cycle.",
            ("Domain switching", "Balanced mixed tempo"),
            "cu_mixed_tempo",
            AntDrillMode.TEMPO,
            15 * scale,
        ),
        _block(
            "pressure_run",
            "Pressure Run",
            "Finish on the full live task with all domains active and the hardest timing, reveal, and warning pressure in the family.",
            ("Pressure tolerance", "Full-task recovery"),
            "cu_pressure_run",
            AntDrillMode.STRESS,
            15 * scale,
        ),
    )
    return AntWorkoutPlan(
        code="cognitive_updating_workout",
        title="Cognitive Updating Workout (90m)",
        description=(
            "Standard 90-minute Cognitive Updating workout with focused domain anchors, state-code integration, "
            "a fixed mixed-tempo cycle, and a final pressure block."
        ),
        notes=(
            "Block setup screens do not count toward the 90-minute drill clock.",
            "All timed blocks stay on the live dual-MFD Cognitive Updating screen with the normal controls and page flow.",
            "Focused blocks dim inactive panels instead of removing them so the family still looks and feels like the full task.",
        ),
        blocks=blocks,
    )


def cu_workout_menu_entries() -> tuple[tuple[str, str], ...]:
    return (("cognitive_updating_workout", "Cognitive Updating Workout (90m)"),)
