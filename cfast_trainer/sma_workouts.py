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


def build_sma_workout_plan(*, duration_scale: float = 1.0) -> AntWorkoutPlan:
    scale = max(0.05, float(duration_scale))
    blocks = (
        _block(
            "joystick_horizontal_anchor",
            "Joystick Horizontal Anchor Warm-Up",
            "Track only horizontal drift with joystick-only control before full two-axis work starts.",
            ("Joystick horizontal precision", "Settling control"),
            "sma_joystick_horizontal_anchor",
            AntDrillMode.BUILD,
            10 * scale,
        ),
        _block(
            "joystick_vertical_anchor",
            "Joystick Vertical Anchor Warm-Up",
            "Track only vertical drift with joystick-only control so the up-down response stays clean.",
            ("Joystick vertical precision", "Settling control"),
            "sma_joystick_vertical_anchor",
            AntDrillMode.BUILD,
            10 * scale,
        ),
        _block(
            "joystick_hold",
            "Joystick Hold Run",
            "Return to full two-axis joystick-only tracking under steadier drift before split work begins.",
            ("Joystick-only tracking", "Two-axis stabilization"),
            "sma_joystick_hold_run",
            AntDrillMode.BUILD,
            10 * scale,
        ),
        _block(
            "split_horizontal_prime",
            "Split Horizontal Prime",
            "Use rudder for horizontal tracking only before the full split-coordination block tightens up.",
            ("Rudder control", "Split-control priming"),
            "sma_split_horizontal_prime",
            AntDrillMode.BUILD,
            10 * scale,
        ),
        _block(
            "split_coordination",
            "Split Coordination Run",
            "Work full two-axis split control under balanced drift until the hand-foot coordination stabilizes.",
            ("Split-control coordination", "Two-axis stabilization"),
            "sma_split_coordination_run",
            AntDrillMode.TEMPO,
            10 * scale,
        ),
        _block(
            "mode_switch",
            "Mode Switch Run",
            "Alternate joystick-only and split segments so switching control schemes stops costing you recovery time.",
            ("Mode switching", "Context reset control"),
            "sma_mode_switch_run",
            AntDrillMode.TEMPO,
            10 * scale,
        ),
        _block(
            "disturbance_tempo",
            "Disturbance Tempo Run",
            "Cycle steady and pulse disturbance profiles across joystick-only and split segments without losing the tracking rhythm.",
            ("Disturbance recovery", "Tempo adaptation", "Mode switching"),
            "sma_disturbance_tempo",
            AntDrillMode.TEMPO,
            15 * scale,
        ),
        _block(
            "pressure_run",
            "Pressure Run",
            "Finish with the highest disturbance load while alternating joystick-only and split control on the live SMA screen.",
            ("Pressure tolerance", "Split-control coordination", "Joystick-only tracking"),
            "sma_pressure_run",
            AntDrillMode.STRESS,
            15 * scale,
        ),
    )
    return AntWorkoutPlan(
        code="sensory_motor_apparatus_workout",
        title="Sensory Motor Apparatus Workout (90m)",
        description=(
            "Standard 90-minute Sensory Motor Apparatus workout with focused axis warm-ups, "
            "mode-switch blocks, disturbance cycling, and a pressure finish."
        ),
        notes=(
            "Block setup screens do not count toward the 90-minute drill clock.",
            "Every block reuses the live Sensory Motor Apparatus tracking screen with the same dot and cross display.",
            "The family is joystick and HOTAS first, but keyboard fallback remains available in every block.",
        ),
        blocks=blocks,
    )


def sma_workout_menu_entries() -> tuple[tuple[str, str], ...]:
    return (("sensory_motor_apparatus_workout", "Sensory Motor Apparatus Workout (90m)"),)
