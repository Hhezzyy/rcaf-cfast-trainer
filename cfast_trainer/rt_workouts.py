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


def build_rt_workout_plan(*, duration_scale: float = 1.0) -> AntWorkoutPlan:
    scale = max(0.05, float(duration_scale))
    blocks = (
        _block(
            "lock_anchor",
            "Lock Anchor Warm-Up",
            "Start with open soldier and truck tracks so stable centering, HUD lock, and clean capture timing come online first.",
            ("Lock quality", "Stable centering"),
            "rt_lock_anchor",
            AntDrillMode.BUILD,
            10 * scale,
        ),
        _block(
            "building_handoff",
            "Building Handoff Prime",
            "Train building holds and clean emergence reacquisition before heavier occlusion and speed pressure arrive.",
            ("Handoff reacquisition", "Building-hold discipline"),
            "rt_building_handoff_prime",
            AntDrillMode.BUILD,
            10 * scale,
        ),
        _block(
            "terrain_recovery",
            "Terrain Recovery Run",
            "Use ridge losses to sharpen prediction and fast reacquisition without leaving the live capture workflow.",
            ("Occlusion recovery", "Predictive tracking"),
            "rt_terrain_recovery_run",
            AntDrillMode.BUILD,
            10 * scale,
        ),
        _block(
            "capture_timing",
            "Capture Timing Prime",
            "Bias the block toward cleaner box captures while the target stream stays fully live.",
            ("Capture timing", "Shot discipline"),
            "rt_capture_timing_prime",
            AntDrillMode.BUILD,
            10 * scale,
        ),
        _block(
            "ground_tempo",
            "Ground Tempo Run",
            "Push soldier and truck tracking speed once the lock and capture anchors are warm.",
            ("Ground tempo", "Lock recovery"),
            "rt_ground_tempo_run",
            AntDrillMode.TEMPO,
            10 * scale,
        ),
        _block(
            "air_speed",
            "Air Speed Run",
            "Shift to helicopter and jet passes where preview timing and velocity changes cost more.",
            ("Air-speed tracking", "Fast-target handling"),
            "rt_air_speed_run",
            AntDrillMode.TEMPO,
            10 * scale,
        ),
        _block(
            "mixed_tempo",
            "Mixed Tempo Run",
            "Repeat the fixed six-segment cycle so lock, handoff, terrain, capture, ground tempo, and air tempo all rotate under one live scene.",
            ("Mixed tempo", "Context switching"),
            "rt_mixed_tempo",
            AntDrillMode.TEMPO,
            15 * scale,
        ),
        _block(
            "pressure",
            "Pressure Run",
            "Finish with all target kinds, all cover modes, all handoff modes, minimal assist, and the hardest capture pressure in the family.",
            ("Pressure tolerance", "Full Rapid Tracking integration"),
            "rt_pressure_run",
            AntDrillMode.STRESS,
            15 * scale,
        ),
    )
    return AntWorkoutPlan(
        code="rapid_tracking_workout",
        title="Rapid Tracking Workout (90m)",
        description=(
            "Standard 90-minute Rapid Tracking workout with focused lock and handoff anchors, "
            "terrain and capture work, mixed tempo rotation, and a final pressure run."
        ),
        notes=(
            "Block setup screens do not count toward the 90-minute drill clock.",
            "Every block reuses the live Rapid Tracking scene, capture box, and real HOTAS or keyboard capture controls.",
            "Tracking windows remain the primary accuracy metric, and block score combines tracking score with normalized capture points.",
        ),
        blocks=blocks,
    )


def rt_workout_menu_entries() -> tuple[tuple[str, str], ...]:
    return (("rapid_tracking_workout", "Rapid Tracking Workout (90m)"),)
