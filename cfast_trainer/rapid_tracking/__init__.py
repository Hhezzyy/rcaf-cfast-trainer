from __future__ import annotations

from .config import (
    RAPID_TRACKING_CHALLENGE_ORDER,
    RAPID_TRACKING_TARGET_KIND_ORDER,
    RapidTrackingConfig,
    RapidTrackingDifficultyProfile,
    RapidTrackingLayoutPolicy,
    RapidTrackingTrainingProfile,
    RapidTrackingTrainingSegment,
)
from .debug import RapidTrackingDebugState, rapid_tracking_debug_lines
from .entities import RapidTrackingCompoundLayout, RapidTrackingPayload, RapidTrackingSummary
from .legacy import (
    RAPID_TRACKING_CONTROL_SCHEMES,
    RapidTrackingDriftGenerator,
    _ellipse_contains_point,
    _repair_layout_for_visual_stability,
    _readable_layout_evaluation,
    _visual_stability_layout_summary,
    build_rapid_tracking_compound_layout,
    normalize_rapid_tracking_control_scheme,
    rapid_tracking_target_cue,
    rapid_tracking_target_description,
    rapid_tracking_target_label,
    score_window,
    select_scene_seed,
)
from .procgen import (
    RapidTrackingBackdropTerrain,
    RapidTrackingV1TrainingWorldBuilder,
    TrainingWorldBuilder,
    build_distant_terrain_ring,
)
from .renderer import RapidTrackingExerciseRenderer, RapidTrackingUiContext, render_rapid_tracking_screen
from .scene import RapidTrackingEngine, RapidTrackingExercise
from .simulation import RapidTrackingSimulation


def build_rapid_tracking_test(
    *,
    clock,
    seed: int,
    difficulty: float = 0.5,
    config=None,
    title: str = "Rapid Tracking",
    practice_segments=None,
    scored_segments=None,
    control_scheme: str = "joystick_only",
    layout_policy=RapidTrackingLayoutPolicy.DEFAULT,
) -> RapidTrackingEngine:
    return RapidTrackingEngine(
        clock=clock,
        seed=int(seed),
        difficulty=float(difficulty),
        config=config,
        title=str(title),
        practice_segments=practice_segments,
        scored_segments=scored_segments,
        control_scheme=control_scheme,
        layout_policy=layout_policy,
    )


__all__ = [
    "RAPID_TRACKING_CHALLENGE_ORDER",
    "RAPID_TRACKING_CONTROL_SCHEMES",
    "RAPID_TRACKING_TARGET_KIND_ORDER",
    "RapidTrackingConfig",
    "RapidTrackingDebugState",
    "RapidTrackingDifficultyProfile",
    "RapidTrackingDriftGenerator",
    "RapidTrackingEngine",
    "RapidTrackingExercise",
    "RapidTrackingExerciseRenderer",
    "RapidTrackingLayoutPolicy",
    "RapidTrackingPayload",
    "RapidTrackingSimulation",
    "RapidTrackingSummary",
    "RapidTrackingTrainingProfile",
    "RapidTrackingTrainingSegment",
    "RapidTrackingUiContext",
    "RapidTrackingV1TrainingWorldBuilder",
    "RapidTrackingCompoundLayout",
    "RapidTrackingBackdropTerrain",
    "TrainingWorldBuilder",
    "_ellipse_contains_point",
    "_repair_layout_for_visual_stability",
    "_readable_layout_evaluation",
    "_visual_stability_layout_summary",
    "build_rapid_tracking_compound_layout",
    "build_distant_terrain_ring",
    "build_rapid_tracking_test",
    "normalize_rapid_tracking_control_scheme",
    "rapid_tracking_debug_lines",
    "rapid_tracking_target_cue",
    "rapid_tracking_target_description",
    "rapid_tracking_target_label",
    "render_rapid_tracking_screen",
    "score_window",
    "select_scene_seed",
]
