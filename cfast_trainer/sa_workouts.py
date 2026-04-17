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
            "Warm up the sparse grid first so current location, side, and vehicle recognition settle before the heavier report blocks.",
            ("Picture tracking", "Identity stabilization"),
            "sa_picture_anchor",
            AntDrillMode.BUILD,
            10 * scale,
        ),
        _block(
            "contact_identification",
            "Contact Identification Prime",
            "Match callsigns to side and vehicle type until the right-side status panel and grid flashes feel automatic.",
            ("Contact identification", "Type and side reading"),
            "sa_contact_identification_prime",
            AntDrillMode.BUILD,
            10 * scale,
        ),
        _block(
            "status_recall",
            "Report Correlation Prime",
            "Train sighting-grid, variation, and rule-call recall from short status-panel flashes and radio chatter before the route block tightens up.",
            ("Report correlation", "Coded-state tracking"),
            "sa_status_recall_prime",
            AntDrillMode.BUILD,
            10 * scale,
        ),
        _block(
            "future_projection",
            "Route Tracking Run",
            "Separate ordered destination from actual movement after the last cue has already faded.",
            ("Route tracking", "Intent vs movement"),
            "sa_future_projection_run",
            AntDrillMode.BUILD,
            10 * scale,
        ),
        _block(
            "action_selection",
            "Action Selection Run",
            "Bring rule-based action judgments online once the core hidden-picture skills are warm.",
            ("Action selection", "Priority control"),
            "sa_action_selection_run",
            AntDrillMode.TEMPO,
            10 * scale,
        ),
        _block(
            "family_switch",
            "Family Switch Run",
            "Cycle conflict, status, handoff, and channel/waypoint families so scenario switching stops costing you orientation time.",
            ("Scenario-family switching", "Context reset control"),
            "sa_family_switch_run",
            AntDrillMode.TEMPO,
            10 * scale,
        ),
        _block(
            "mixed_tempo",
            "Mixed Tempo Run",
            "Repeat the full query rhythm while the world keeps evolving behind fading grid updates, the incoming bar, and the status panel.",
            ("Mixed query tempo", "Full-picture continuity"),
            "sa_mixed_tempo",
            AntDrillMode.TEMPO,
            15 * scale,
        ),
        _block(
            "pressure_run",
            "Pressure Run",
            "Finish with all channels and all query kinds active under the shortest cue windows and fastest radio cadence.",
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
            "Standard 90-minute Situational Awareness workout with typed reflection, sparse-picture anchors, "
            "report and route correlation, family and query switching, and a final full-pressure hidden-state run."
        ),
        notes=(
            "Typed reflections and block setup screens do not count toward the 90-minute drill clock.",
            "Every block reuses the live continuous Situational Awareness screen with the incoming bar, flashing status panel, radio chatter, and direct-response queries.",
            "Controls stay the live model throughout: click or type grid cells, then answer choice prompts with click or 1-5.",
        ),
        blocks=blocks,
    )


def sa_workout_menu_entries() -> tuple[tuple[str, str], ...]:
    return (("situational_awareness_workout", "Situational Awareness Workout (90m)"),)
