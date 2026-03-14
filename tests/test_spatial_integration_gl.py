from __future__ import annotations

from cfast_trainer.spatial_integration import (
    SpatialIntegrationAnswerMode,
    SpatialIntegrationLandmark,
    SpatialIntegrationPart,
    SpatialIntegrationPayload,
    SpatialIntegrationPoint,
    SpatialIntegrationQuestionKind,
    SpatialIntegrationReferenceView,
    SpatialIntegrationSceneView,
    SpatialIntegrationTrialStage,
    SpatialIntegrationVector,
)
from cfast_trainer.spatial_integration_gl import build_scene_layout


def test_build_scene_layout_projects_topdown_landmarks_deterministically() -> None:
    payload = SpatialIntegrationPayload(
        part=SpatialIntegrationPart.STATIC,
        trial_stage=SpatialIntegrationTrialStage.STUDY,
        block_kind="practice",
        scene_id=1,
        scene_index_in_block=1,
        scenes_in_block=2,
        study_view_index=1,
        study_views_in_scene=3,
        question_index_in_scene=0,
        questions_in_scene=3,
        stage_time_remaining_s=4.5,
        part_time_remaining_s=None,
        kind=SpatialIntegrationQuestionKind.LANDMARK_GRID,
        answer_mode=None,
        stem="Where was TWR?",
        query_label="TWR",
        north_arrow_deg=0,
        scene_view=SpatialIntegrationSceneView.TOPDOWN,
        grid_cols=5,
        grid_rows=4,
        alt_levels=3,
        reference_views=(
            SpatialIntegrationReferenceView("Top", SpatialIntegrationSceneView.TOPDOWN, 0),
        ),
        active_reference_view=SpatialIntegrationReferenceView("Top", SpatialIntegrationSceneView.TOPDOWN, 0),
        hills=(),
        landmarks=(
            SpatialIntegrationLandmark(label="TWR", x=1, y=1),
            SpatialIntegrationLandmark(label="HGR", x=3, y=2),
        ),
        answer_map_landmarks=(),
        route_points=(),
        route_current_index=0,
        aircraft_prev=SpatialIntegrationPoint(x=1, y=0, z=1),
        aircraft_now=SpatialIntegrationPoint(x=2, y=1, z=1),
        velocity=SpatialIntegrationVector(dx=1, dy=0, dz=0),
        show_aircraft_motion=False,
        options=(),
        correct_code=0,
        correct_point=SpatialIntegrationPoint(x=1, y=1, z=0),
        correct_answer_token="B2",
    )

    layout = build_scene_layout(payload=payload, size=(800, 600))

    assert layout.scene_view is SpatialIntegrationSceneView.TOPDOWN
    assert layout.horizon_y == 96.0
    assert len(layout.landmarks) == 2
    assert layout.aircraft_future is None
    assert layout.landmarks[0].screen_x < layout.landmarks[1].screen_x


def test_build_scene_layout_includes_future_marker_for_moving_oblique_scene() -> None:
    payload = SpatialIntegrationPayload(
        part=SpatialIntegrationPart.AIRCRAFT,
        trial_stage=SpatialIntegrationTrialStage.STUDY,
        block_kind="scored",
        scene_id=2,
        scene_index_in_block=3,
        scenes_in_block=None,
        study_view_index=2,
        study_views_in_scene=3,
        question_index_in_scene=0,
        questions_in_scene=3,
        stage_time_remaining_s=4.0,
        part_time_remaining_s=12.0,
        kind=SpatialIntegrationQuestionKind.AIRCRAFT_ROUTE_SELECTION,
        answer_mode=None,
        stem="Which route is correct?",
        query_label="AIRCRAFT",
        north_arrow_deg=0,
        scene_view=SpatialIntegrationSceneView.OBLIQUE,
        grid_cols=6,
        grid_rows=6,
        alt_levels=4,
        reference_views=(
            SpatialIntegrationReferenceView("Oblique", SpatialIntegrationSceneView.OBLIQUE, 30),
        ),
        active_reference_view=SpatialIntegrationReferenceView("Oblique", SpatialIntegrationSceneView.OBLIQUE, 30),
        hills=(),
        landmarks=(SpatialIntegrationLandmark(label="RDG", x=4, y=2),),
        answer_map_landmarks=(),
        route_points=(),
        route_current_index=0,
        aircraft_prev=SpatialIntegrationPoint(x=1, y=1, z=1),
        aircraft_now=SpatialIntegrationPoint(x=2, y=2, z=2),
        velocity=SpatialIntegrationVector(dx=1, dy=1, dz=1),
        show_aircraft_motion=True,
        options=(),
        correct_code=0,
        correct_point=SpatialIntegrationPoint(x=3, y=3, z=3),
        correct_answer_token="3",
    )

    layout = build_scene_layout(payload=payload, size=(800, 600))

    assert layout.scene_view is SpatialIntegrationSceneView.OBLIQUE
    assert layout.horizon_y == 252.0
    assert layout.aircraft_future is not None
    assert layout.aircraft_future[0] > layout.aircraft_now[0]
    assert layout.aircraft_future[1] < layout.aircraft_now[1]
