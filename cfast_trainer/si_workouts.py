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


def build_si_workout_plan(*, duration_scale: float = 1.0) -> AntWorkoutPlan:
    scale = max(0.05, float(duration_scale))
    blocks = (
        _block(
            "landmark_anchor",
            "Landmark Anchor Warm-Up",
            "Start on the static landscape section and lock object-to-terrain relations before full-scene reconstruction begins.",
            ("Landmark anchoring", "Static viewpoint linking"),
            "si_landmark_anchor",
            AntDrillMode.BUILD,
            10 * scale,
        ),
        _block(
            "reconstruction_run",
            "Scene Reconstruction Run",
            "Stay on the static section and practice turning three study views into one correct top-down map.",
            ("Scene reconstruction", "Static map synthesis"),
            "si_reconstruction_run",
            AntDrillMode.BUILD,
            10 * scale,
        ),
        _block(
            "static_mixed_run",
            "Static Mixed Run",
            "Keep the live static part intact with landmark-grid and reconstruction questions alternating under tempo pressure.",
            ("Static question switching", "Landscape integration"),
            "si_static_mixed_run",
            AntDrillMode.TEMPO,
            10 * scale,
        ),
        _block(
            "route_anchor",
            "Route Anchor Warm-Up",
            "Shift to the aircraft section and stabilize route-shape recognition before continuation and grid pressure arrive.",
            ("Route anchoring", "Aircraft path reading"),
            "si_route_anchor",
            AntDrillMode.BUILD,
            10 * scale,
        ),
        _block(
            "continuation_prime",
            "Continuation Prime",
            "Stay on the aircraft section and bias the block toward next-position continuation rather than full-route comparison.",
            ("Forward projection", "Continuation discrimination"),
            "si_continuation_prime",
            AntDrillMode.BUILD,
            10 * scale,
        ),
        _block(
            "aircraft_grid_run",
            "Aircraft Grid Run",
            "Keep the aircraft section live and answer only aircraft-location grid questions under tempo pressure.",
            ("Aircraft grid placement", "Route-to-cell translation"),
            "si_aircraft_grid_run",
            AntDrillMode.TEMPO,
            10 * scale,
        ),
        _block(
            "mixed_tempo",
            "Mixed Part Tempo",
            "Run the full static section first and the full aircraft section second so both SI parts stay balanced inside one block.",
            ("Part switching", "Balanced Spatial Integration rhythm"),
            "si_mixed_tempo",
            AntDrillMode.TEMPO,
            15 * scale,
        ),
        _block(
            "pressure_run",
            "Pressure Run",
            "Finish with both SI parts, full question mix, and the tightest study and answer caps in this family.",
            ("Pressure tolerance", "Full Spatial Integration chaining"),
            "si_pressure_run",
            AntDrillMode.STRESS,
            15 * scale,
        ),
    )
    return AntWorkoutPlan(
        code="spatial_integration_workout",
        title="Spatial Integration Workout (90m)",
        description=(
            "Standard 90-minute Spatial Integration workout with static-scene anchoring, aircraft-route work, "
            "balanced mixed-part runs, and a final pressure block."
        ),
        notes=(
            "Typed reflections and block setup screens do not count toward the 90-minute drill clock.",
            "Every block keeps the same three-view study loop, frozen-scene presentation, compass behavior, and grid-click or 1-4 answer flow as the live Spatial Integration runtime.",
            "The workout stays balanced: 30 minutes static-only, 30 minutes aircraft-only, then 30 minutes of mixed-part chaining.",
        ),
        blocks=blocks,
    )


def si_workout_menu_entries() -> tuple[tuple[str, str], ...]:
    return (("spatial_integration_workout", "Spatial Integration Workout (90m)"),)

