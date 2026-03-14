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


def build_tr_workout_plan(*, duration_scale: float = 1.0) -> AntWorkoutPlan:
    scale = max(0.05, float(duration_scale))
    blocks = (
        _block(
            "scene_anchor",
            "Scene Anchor Warm-Up",
            "Work the map panel by itself with clean shape and affiliation targets before the modifier tags return.",
            ("Map discrimination", "Shape-affiliation anchoring"),
            "tr_scene_anchor",
            AntDrillMode.BUILD,
            10 * scale,
        ),
        _block(
            "light_anchor",
            "Light Pattern Warm-Up",
            "Stay on the light panel and make the pattern timing feel automatic before multitask switching starts.",
            ("Light pattern matching", "Cadence anchoring"),
            "tr_light_anchor",
            AntDrillMode.BUILD,
            10 * scale,
        ),
        _block(
            "scan_anchor",
            "Scan Stream Warm-Up",
            "Stay on the scan panel and lock in the reveal rhythm before denser switching returns.",
            ("Scan stream matching", "Symbol discrimination"),
            "tr_scan_anchor",
            AntDrillMode.BUILD,
            10 * scale,
        ),
        _block(
            "system_anchor",
            "System Column Warm-Up",
            "Stay on the scrolling code columns until the target-code handoff is stable.",
            ("System-code matching", "Scrolling-column timing"),
            "tr_system_anchor",
            AntDrillMode.BUILD,
            10 * scale,
        ),
        _block(
            "scene_modifiers",
            "Scene Modifier Run",
            "Bring damaged and high-priority tags back into the map panel before panel switching starts.",
            ("Map discrimination", "Modifier-tag reading"),
            "tr_scene_modifier_run",
            AntDrillMode.TEMPO,
            10 * scale,
        ),
        _block(
            "panel_switch",
            "Panel Switch Run",
            "Cycle one panel at a time so the scan handoff stays deliberate instead of frantic.",
            ("Panel switching", "Context reset control"),
            "tr_panel_switch_run",
            AntDrillMode.TEMPO,
            10 * scale,
        ),
        _block(
            "mixed_tempo",
            "Mixed Tempo Run",
            "Use fixed multi-panel combinations to build balanced throughput before the full pressure block.",
            ("Panel switching", "Mixed-panel throughput", "Cadence control"),
            "tr_mixed_tempo",
            AntDrillMode.TEMPO,
            15 * scale,
        ),
        _block(
            "pressure_run",
            "Pressure Run",
            "Finish on the full four-panel live task with the fastest cadence profile in the family.",
            ("Full multitask integration", "Pressure tolerance", "Mixed-panel throughput"),
            "tr_pressure_run",
            AntDrillMode.STRESS,
            15 * scale,
        ),
    )
    return AntWorkoutPlan(
        code="target_recognition_workout",
        title="Target Recognition Workout (90m)",
        description=(
            "Standard 90-minute Target Recognition workout with typed reflection, focused panel warm-ups, "
            "switching blocks, and a full-pressure multitask finish."
        ),
        notes=(
            "Typed reflections and block setup screens do not count toward the 90-minute drill clock.",
            "Every block reuses the live four-panel Target Recognition layout and mouse-first interaction flow.",
            "Focused blocks keep inactive panels visible but OFF so the layout stays familiar while only the intended panels score.",
        ),
        blocks=blocks,
    )


def tr_workout_menu_entries() -> tuple[tuple[str, str], ...]:
    return (("target_recognition_workout", "Target Recognition Workout (90m)"),)
