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


def build_sa_workout_plan(*, duration_scale: float = 1.0) -> AntWorkoutPlan:
    scale = max(0.05, float(duration_scale))
    blocks = (
        _block(
            "picture_anchor",
            "Picture Anchor Warm-Up",
            "Warm up the live tactical picture first so future-position and simple contact matching settle before heavier recall starts.",
            ("Picture tracking", "Contact stabilization"),
            "sa_picture_anchor",
            AntDrillMode.BUILD,
            10 * scale,
        ),
        _block(
            "contact_identification",
            "Contact Identification Prime",
            "Match described coded state to the correct indexed track until the right-side panel handoff feels clean.",
            ("Contact identification", "Index-panel reading"),
            "sa_contact_identification_prime",
            AntDrillMode.BUILD,
            10 * scale,
        ),
        _block(
            "status_recall",
            "Status Recall Prime",
            "Train channel, code, altitude, and fuel/status recall from the visible and announced picture before the prediction block tightens up.",
            ("Status recall", "Coded-state tracking"),
            "sa_status_recall_prime",
            AntDrillMode.BUILD,
            10 * scale,
        ),
        _block(
            "future_projection",
            "Future Projection Run",
            "Project headings and route updates forward without losing the live traffic picture.",
            ("Future projection", "Route forecasting"),
            "sa_future_projection_run",
            AntDrillMode.BUILD,
            10 * scale,
        ),
        _block(
            "action_selection",
            "Action Selection Run",
            "Bring intervention and tactical priority decisions online once the core picture-building skills are warm.",
            ("Intervention selection", "Priority control"),
            "sa_action_selection_run",
            AntDrillMode.TEMPO,
            10 * scale,
        ),
        _block(
            "family_switch",
            "Family Switch Run",
            "Cycle merge, fuel, route-handoff, and channel-shift families so scenario switching stops costing you orientation time.",
            ("Scenario-family switching", "Context reset control"),
            "sa_family_switch_run",
            AntDrillMode.TEMPO,
            10 * scale,
        ),
        _block(
            "mixed_tempo",
            "Mixed Tempo Run",
            "Repeat the fixed four-query rhythm while the live tactical picture keeps evolving underneath it.",
            ("Mixed query tempo", "Full-picture continuity"),
            "sa_mixed_tempo",
            AntDrillMode.TEMPO,
            15 * scale,
        ),
        _block(
            "pressure_run",
            "Pressure Run",
            "Finish with all channels and all query kinds active under denser updates and shorter response windows.",
            ("Pressure tolerance", "Full Situational Awareness integration"),
            "sa_pressure_run",
            AntDrillMode.STRESS,
            15 * scale,
        ),
    )
    return AntWorkoutPlan(
        code="situational_awareness_workout",
        title="Situational Awareness Workout (90m)",
        description=(
            "Standard 90-minute Situational Awareness workout with typed reflection, focused live-picture blocks, "
            "family and query switching, and a final full-pressure run."
        ),
        notes=(
            "Typed reflections and block setup screens do not count toward the 90-minute drill clock.",
            "Every block reuses the live continuous Situational Awareness screen with direct-response query modes.",
            "Controls stay the live model throughout: click or type grid cells, choose track rows with 1-5, and answer action cards with click or 1-4.",
        ),
        blocks=blocks,
    )


def sa_workout_menu_entries() -> tuple[tuple[str, str], ...]:
    return (("situational_awareness_workout", "Situational Awareness Workout (90m)"),)
