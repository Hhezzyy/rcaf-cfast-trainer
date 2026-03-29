from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pygame
import pytest

from cfast_trainer.aircraft_art import (
    apply_fixed_wing_view_rotation,
    fixed_wing_heading_from_screen_heading,
    project_fixed_wing_point,
    rotate_fixed_wing_point,
)
from cfast_trainer.auditory_capacity import AuditoryCapacityGate, build_auditory_capacity_test
from cfast_trainer.gl_scenes import (
    AuditoryGlScene,
    RapidTrackingGlScene,
    SpatialIntegrationGlScene,
    TraceTest1GlScene,
    TraceTest2GlScene,
)
from cfast_trainer.modern_gl_renderer import (
    ModernSceneRenderer,
    _GeometryBatch,
    _SceneAssetLibrary,
    _build_auditory_scene_plan,
    _build_rapid_tracking_scene_plan,
    _build_spatial_integration_scene_plan,
    _build_trace_test_1_scene_plan,
    _build_trace_test_2_scene_plan,
    _project_aircraft_marker_polygons,
    _scene_local_top_left_to_screen,
    _world_hpr_from_tangent,
)
from cfast_trainer.render_assets import RenderAssetCatalog
from cfast_trainer.rapid_tracking import build_rapid_tracking_test
from cfast_trainer.spatial_integration import build_spatial_integration_test
from cfast_trainer.trace_test_1 import (
    TraceTest1Generator,
    TraceTest1Payload,
    TraceTest1PromptPlan,
    TraceTest1TrialStage,
    trace_test_1_scene_frames,
)
from cfast_trainer.trace_test_2 import TraceTest2Generator, trace_test_2_track_tangent


class _FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class _FakeScreen:
    def use(self) -> None:
        return None


class _FakeContext:
    def __init__(self) -> None:
        self.screen = _FakeScreen()
        self.viewport: tuple[int, int, int, int] | None = None
        self.clear_calls: list[tuple[float, float, float, float]] = []

    def clear(self, r: float, g: float, b: float, a: float) -> None:
        self.clear_calls.append((r, g, b, a))


def _auditory_payload():
    clock = _FakeClock()
    engine = build_auditory_capacity_test(clock=clock, seed=17, difficulty=0.58)
    engine.start_practice()
    clock.advance(0.2)
    engine.update()
    payload = engine.snapshot().payload
    assert payload is not None
    return payload


def _rapid_tracking_payload():
    clock = _FakeClock()
    engine = build_rapid_tracking_test(clock=clock, seed=551, difficulty=0.63)
    engine.start_scored()
    clock.advance(0.6)
    engine.update()
    payload = engine.snapshot().payload
    assert payload is not None
    return payload


def _spatial_integration_payload():
    clock = _FakeClock()
    engine = build_spatial_integration_test(clock=clock, seed=61, difficulty=0.58)
    engine.start_practice()
    for _ in range(8):
        payload = engine.snapshot().payload
        if payload is not None:
            return payload
        clock.advance(0.2)
        engine.update()
    raise AssertionError("expected a spatial integration payload")


def _trace_test_1_payload() -> TraceTest1Payload:
    prompt = TraceTest1Generator(seed=44).next_problem(difficulty=0.82).payload
    assert isinstance(prompt, TraceTest1PromptPlan)
    progress = 0.68
    return TraceTest1Payload(
        trial_stage=TraceTest1TrialStage.ANSWER_OPEN,
        stage_time_remaining_s=1.0,
        observe_progress=progress,
        prompt_index=prompt.prompt_index,
        active_command=prompt.red_plan.command,
        blue_commands=tuple(blue_plan.command for blue_plan in prompt.blue_plans),
        scene=trace_test_1_scene_frames(prompt=prompt, progress=progress),
        options=(),
        correct_code={
            "LEFT": 1,
            "RIGHT": 2,
            "PUSH": 3,
            "PULL": 4,
        }[prompt.red_plan.command.value],
        prompt_window_s=4.3,
        answer_open_progress=prompt.answer_open_progress,
        speed_multiplier=prompt.speed_multiplier,
        viewpoint_bearing_deg=180,
    )


def _trace_test_2_payload():
    payload = TraceTest2Generator(seed=71).next_problem(difficulty=0.58).payload
    assert payload is not None
    return payload


def _project_marker_reference_point(
    *,
    rect: pygame.Rect,
    window_height: int,
    center_top_left: tuple[float, float],
    point: tuple[float, float, float],
    heading_deg: float,
    pitch_deg: float,
    bank_deg: float,
    size: float,
    view_pitch_deg: float,
) -> tuple[float, float]:
    rotated = rotate_fixed_wing_point(
        point,
        heading_deg=fixed_wing_heading_from_screen_heading(heading_deg),
        pitch_deg=float(pitch_deg),
        bank_deg=float(bank_deg),
    )
    viewed = apply_fixed_wing_view_rotation(
        rotated,
        view_pitch_deg=float(view_pitch_deg),
    )
    local_x, local_y, _depth = project_fixed_wing_point(
        viewed,
        cx=int(round(center_top_left[0])),
        cy=int(round(center_top_left[1])),
        scale=max(8.0, float(size)),
    )
    return _scene_local_top_left_to_screen(
        rect=rect,
        window_height=window_height,
        x=float(local_x),
        y=float(local_y),
    )


def _marker_centroid(faces) -> tuple[float, float]:
    points = [point for face in faces for point in face.points]
    return (
        sum(point[0] for point in points) / float(len(points)),
        sum(point[1] for point in points) / float(len(points)),
    )


def _angle_delta_deg(a: float, b: float) -> float:
    return ((float(a) - float(b) + 180.0) % 360.0) - 180.0


def _auditory_scene_for_render_frame() -> AuditoryGlScene:
    return AuditoryGlScene(
        world=pygame.Rect(0, 0, 640, 360),
        payload=_auditory_payload(),
        time_remaining_s=19.5,
        time_fill_ratio=0.48,
    )


def _auditory_scene_with_visible_gates() -> AuditoryGlScene:
    payload = replace(
        _auditory_payload(),
        gates=(
            AuditoryCapacityGate(
                gate_id=701,
                x_norm=1.55,
                y_norm=0.0,
                color="GREEN",
                shape="SQUARE",
                aperture_norm=0.18,
            ),
            AuditoryCapacityGate(
                gate_id=702,
                x_norm=1.10,
                y_norm=0.0,
                color="RED",
                shape="TRIANGLE",
                aperture_norm=0.22,
            ),
            AuditoryCapacityGate(
                gate_id=703,
                x_norm=0.65,
                y_norm=0.0,
                color="BLUE",
                shape="CIRCLE",
                aperture_norm=0.20,
            ),
        ),
    )
    return AuditoryGlScene(
        world=pygame.Rect(0, 0, 640, 360),
        payload=payload,
        time_remaining_s=19.5,
        time_fill_ratio=0.48,
    )


def _auditory_scene_with_ball_offset(*, ball_x: float, ball_y: float) -> AuditoryGlScene:
    payload = replace(
        _auditory_payload(),
        ball_x=float(ball_x),
        ball_y=float(ball_y),
    )
    return AuditoryGlScene(
        world=pygame.Rect(0, 0, 640, 360),
        payload=payload,
        time_remaining_s=19.5,
        time_fill_ratio=0.48,
    )


def _build_pipeline_probe_renderer() -> ModernSceneRenderer:
    renderer = ModernSceneRenderer.__new__(ModernSceneRenderer)
    renderer._ctx = _FakeContext()
    renderer._win_w = 640
    renderer._win_h = 360
    renderer._batch = _GeometryBatch()
    renderer._last_scene_debug = {}
    renderer._vortex_texture = object()
    renderer._ui_texture = None
    renderer._ui_tex_size = (0, 0)
    return renderer


def test_auditory_scene_plan_contains_tunnel_geometry_and_ball() -> None:
    scene = _auditory_scene_for_render_frame()

    plan = _build_auditory_scene_plan(scene)

    counts = {asset_id: 0 for asset_id in plan.asset_ids}
    for instance in plan.asset_instances:
        counts[instance.asset_id] = counts.get(instance.asset_id, 0) + 1

    assert plan.camera is not None
    assert plan.asset_ids == (
        "auditory_ball",
        "auditory_tunnel_rib",
        "auditory_tunnel_segment",
    )
    assert counts["auditory_ball"] == 1
    assert counts["auditory_tunnel_segment"] >= 10
    assert counts["auditory_tunnel_rib"] >= 10
    assert counts["auditory_tunnel_segment"] <= 16
    assert counts["auditory_tunnel_rib"] <= 16
    assert plan.entity_count == len(plan.asset_instances)


def test_auditory_scene_plan_includes_volumetric_gate_assets_when_visible() -> None:
    scene = _auditory_scene_with_visible_gates()

    plan = _build_auditory_scene_plan(scene)
    gate_ids = {instance.asset_id for instance in plan.asset_instances if instance.asset_id.startswith("auditory_gate_")}

    assert {"auditory_gate_circle", "auditory_gate_triangle", "auditory_gate_square"} <= set(plan.asset_ids)
    assert gate_ids == {"auditory_gate_circle", "auditory_gate_triangle", "auditory_gate_square"}
    for instance in plan.asset_instances:
        if instance.asset_id.startswith("auditory_gate_"):
            assert instance.scale[0] > 0.0
            assert instance.scale[1] > 0.0
            assert instance.scale[2] > 0.0
            assert instance.color is not None
            assert instance.color[3] > 0.0


def test_auditory_scene_plan_keeps_centered_ball_camera_baseline() -> None:
    scene = _auditory_scene_with_ball_offset(ball_x=0.0, ball_y=0.0)

    plan = _build_auditory_scene_plan(scene)

    assert plan.camera is not None
    assert plan.camera.position == pytest.approx(
        (-0.04195182978272108, 2.3399754843693628, 0.08465317048332292)
    )
    assert plan.camera.heading_deg == pytest.approx(1.196667680590167)
    assert plan.camera.pitch_deg == pytest.approx(0.09763773088790131)


def test_auditory_scene_plan_camera_follows_ball_left_and_right() -> None:
    center_plan = _build_auditory_scene_plan(_auditory_scene_with_ball_offset(ball_x=0.0, ball_y=0.0))
    left_plan = _build_auditory_scene_plan(_auditory_scene_with_ball_offset(ball_x=-0.82, ball_y=0.0))
    right_plan = _build_auditory_scene_plan(_auditory_scene_with_ball_offset(ball_x=0.82, ball_y=0.0))

    assert center_plan.camera is not None
    assert left_plan.camera is not None
    assert right_plan.camera is not None
    assert left_plan.camera.position[0] < right_plan.camera.position[0]
    assert abs(right_plan.camera.position[0] - left_plan.camera.position[0]) >= 1.2
    assert _angle_delta_deg(left_plan.camera.heading_deg, center_plan.camera.heading_deg) < 0.0
    assert _angle_delta_deg(right_plan.camera.heading_deg, center_plan.camera.heading_deg) > 0.0
    assert abs(_angle_delta_deg(right_plan.camera.heading_deg, left_plan.camera.heading_deg)) >= 2.0


def test_auditory_scene_plan_camera_follows_ball_low_and_high() -> None:
    center_plan = _build_auditory_scene_plan(_auditory_scene_with_ball_offset(ball_x=0.0, ball_y=0.0))
    low_plan = _build_auditory_scene_plan(_auditory_scene_with_ball_offset(ball_x=0.0, ball_y=-0.60))
    high_plan = _build_auditory_scene_plan(_auditory_scene_with_ball_offset(ball_x=0.0, ball_y=0.60))

    assert center_plan.camera is not None
    assert low_plan.camera is not None
    assert high_plan.camera is not None
    assert low_plan.camera.position[2] < high_plan.camera.position[2]
    assert abs(high_plan.camera.position[2] - low_plan.camera.position[2]) >= 0.7
    assert low_plan.camera.pitch_deg < center_plan.camera.pitch_deg < high_plan.camera.pitch_deg
    assert abs(high_plan.camera.pitch_deg - low_plan.camera.pitch_deg) >= 1.5


def test_auditory_scene_plan_camera_is_repeatable_for_identical_payload() -> None:
    scene = _auditory_scene_with_ball_offset(ball_x=0.45, ball_y=-0.25)

    first_plan = _build_auditory_scene_plan(scene)
    second_plan = _build_auditory_scene_plan(scene)

    assert first_plan.camera is not None
    assert second_plan.camera is not None
    assert first_plan.camera == second_plan.camera


def test_render_frame_flushes_scene_textures_before_color_and_ui() -> None:
    renderer = _build_pipeline_probe_renderer()
    order: list[object] = []

    def fake_draw_scene(*, scene) -> None:
        assert isinstance(scene, AuditoryGlScene)
        order.append("draw_scene")
        renderer._batch.textured.append(object())
        renderer._batch.triangles.append(SimpleNamespace())

    def fake_flush_scene_textures() -> None:
        order.append(("scene_textures", len(renderer._batch.textured)))
        renderer._batch.textured.clear()

    def fake_flush_color_geometry() -> None:
        order.append(("color", len(renderer._batch.textured), len(renderer._batch.triangles)))

    def fake_draw_ui_surface(*, ui_surface: pygame.Surface) -> None:
        order.append(("ui", len(renderer._batch.textured), ui_surface.get_size()))

    renderer._draw_scene = fake_draw_scene
    renderer._flush_scene_textures = fake_flush_scene_textures
    renderer._flush_color_geometry = fake_flush_color_geometry
    renderer._draw_ui_surface = fake_draw_ui_surface

    renderer.render_frame(
        ui_surface=pygame.Surface((640, 360), pygame.SRCALPHA),
        scene=_auditory_scene_for_render_frame(),
    )

    assert order == [
        "draw_scene",
        ("scene_textures", 1),
        ("color", 0, 1),
        ("ui", 0, (640, 360)),
    ]


def test_auditory_render_frame_does_not_defer_vortex_flush_until_ui() -> None:
    renderer = _build_pipeline_probe_renderer()
    seen_textured_counts: list[int] = []

    def fake_draw_scene(*, scene) -> None:
        assert isinstance(scene, AuditoryGlScene)
        renderer._batch.textured.append(object())

    def fake_flush_scene_textures() -> None:
        seen_textured_counts.append(len(renderer._batch.textured))
        renderer._batch.textured.clear()

    def fake_draw_ui_surface(*, ui_surface: pygame.Surface) -> None:
        _ = ui_surface
        seen_textured_counts.append(len(renderer._batch.textured))

    renderer._draw_scene = fake_draw_scene
    renderer._flush_scene_textures = fake_flush_scene_textures
    renderer._flush_color_geometry = lambda: None
    renderer._draw_ui_surface = fake_draw_ui_surface

    renderer.render_frame(
        ui_surface=pygame.Surface((640, 360), pygame.SRCALPHA),
        scene=_auditory_scene_for_render_frame(),
    )

    assert seen_textured_counts == [1, 0]


def test_rapid_tracking_scene_plan_includes_target_and_scenery_assets() -> None:
    scene = RapidTrackingGlScene(
        world=pygame.Rect(0, 0, 640, 360),
        payload=_rapid_tracking_payload(),
        active_phase=True,
    )

    plan = _build_rapid_tracking_scene_plan(scene)
    asset_ids = set(plan.asset_ids)
    instance_ids = {instance.asset_id for instance in plan.asset_instances}

    assert plan.camera is not None
    assert {
        "building_hangar",
        "building_tower",
        "forest_canopy_patch",
        "helicopter_green",
        "plane_blue",
        "plane_green",
        "plane_yellow",
        "shrubs_low_cluster",
        "soldiers_patrol",
        "trees_field_cluster",
        "trees_pine_cluster",
        "truck_olive",
    } <= asset_ids
    assert {
        "building_hangar",
        "building_tower",
        "helicopter_green",
        "plane_blue",
        "plane_green",
        "plane_yellow",
        "shrubs_low_cluster",
        "soldiers_patrol",
        "trees_field_cluster",
        "trees_pine_cluster",
        "truck_olive",
    } <= instance_ids
    assert len(plan.asset_instances) >= 40
    assert plan.entity_count == len(plan.asset_instances)


def test_scene_asset_library_builds_hangar_builtin_mesh() -> None:
    library = _SceneAssetLibrary(RenderAssetCatalog())

    mesh = library.mesh("building_hangar")

    assert mesh.asset_id == "building_hangar"
    assert len(mesh.triangles) > 0


def test_spatial_integration_scene_plan_includes_aircraft_and_landmarks() -> None:
    scene = SpatialIntegrationGlScene(
        world=pygame.Rect(0, 0, 640, 360),
        payload=_spatial_integration_payload(),
    )

    plan = _build_spatial_integration_scene_plan(scene)
    instance_ids = [instance.asset_id for instance in plan.asset_instances]

    assert plan.camera is not None
    assert "plane_blue" in instance_ids
    assert any(asset_id == "building_tower" for asset_id in instance_ids)
    assert any(asset_id in {"trees_field_cluster", "forest_canopy_patch"} for asset_id in instance_ids)
    assert plan.entity_count == len(plan.asset_instances)


def test_trace_test_1_scene_plan_uses_plane_mesh_instances_and_frame_hpr() -> None:
    payload = _trace_test_1_payload()
    scene = TraceTest1GlScene(world=pygame.Rect(0, 0, 640, 360), payload=payload)

    plan = _build_trace_test_1_scene_plan(scene)

    assert plan.camera is not None
    assert plan.asset_ids == ("plane_blue", "plane_green", "plane_red", "plane_yellow")
    assert plan.overlay_primitives == ()
    assert plan.entity_count == 1 + len(payload.scene.blue_frames)
    assert plan.asset_instances[0].asset_id == "plane_red"
    assert plan.asset_instances[0].hpr_deg == pytest.approx(
        (
            payload.scene.red_frame.attitude.yaw_deg,
            payload.scene.red_frame.attitude.pitch_deg,
            payload.scene.red_frame.attitude.roll_deg,
        )
    )
    assert all(instance.asset_id.startswith("plane_") for instance in plan.asset_instances)


def test_trace_test_2_scene_plan_uses_plane_mesh_instances_and_tangent_hpr() -> None:
    payload = _trace_test_2_payload()
    scene = TraceTest2GlScene(world=pygame.Rect(0, 0, 640, 360), payload=payload)

    plan = _build_trace_test_2_scene_plan(scene)

    assert plan.camera is not None
    assert plan.asset_ids == ("plane_blue", "plane_green", "plane_red", "plane_yellow")
    assert plan.overlay_primitives == ()
    assert plan.entity_count == len(payload.aircraft)
    for instance, track in zip(plan.asset_instances, payload.aircraft, strict=True):
        tangent = trace_test_2_track_tangent(track=track, progress=payload.observe_progress)
        assert instance.asset_id.startswith("plane_")
        assert instance.hpr_deg == pytest.approx(_world_hpr_from_tangent(tangent=tangent, roll_deg=0.0))


def test_trace_marker_projection_preserves_heading_bank_and_stable_translation() -> None:
    rect = pygame.Rect(120, 80, 640, 360)
    window_height = 900
    size = 16.0
    center_a = (220.0, 120.0)
    center_b = (220.0, 180.0)

    faces_a = _project_aircraft_marker_polygons(
        rect=rect,
        window_height=window_height,
        center_top_left=center_a,
        heading_deg=18.0,
        size=size,
        color=(0.8, 0.3, 0.3, 1.0),
        outline=(1.0, 1.0, 1.0, 1.0),
        pitch_deg=6.0,
        bank_deg=22.0,
        view_pitch_deg=0.0,
    )
    faces_repeat = _project_aircraft_marker_polygons(
        rect=rect,
        window_height=window_height,
        center_top_left=center_a,
        heading_deg=18.0,
        size=size,
        color=(0.8, 0.3, 0.3, 1.0),
        outline=(1.0, 1.0, 1.0, 1.0),
        pitch_deg=6.0,
        bank_deg=22.0,
        view_pitch_deg=0.0,
    )
    faces_b = _project_aircraft_marker_polygons(
        rect=rect,
        window_height=window_height,
        center_top_left=center_b,
        heading_deg=18.0,
        size=size,
        color=(0.8, 0.3, 0.3, 1.0),
        outline=(1.0, 1.0, 1.0, 1.0),
        pitch_deg=6.0,
        bank_deg=22.0,
        view_pitch_deg=0.0,
    )

    assert faces_a == faces_repeat
    centroid_a = _marker_centroid(faces_a)
    centroid_b = _marker_centroid(faces_b)
    assert abs(centroid_b[0] - centroid_a[0]) < 0.2
    assert centroid_b[1] < centroid_a[1]

    nose = _project_marker_reference_point(
        rect=rect,
        window_height=window_height,
        center_top_left=center_a,
        point=(0.0, 3.42, 0.12),
        heading_deg=0.0,
        pitch_deg=0.0,
        bank_deg=0.0,
        size=size,
        view_pitch_deg=0.0,
    )
    tail = _project_marker_reference_point(
        rect=rect,
        window_height=window_height,
        center_top_left=center_a,
        point=(0.0, -2.48, 0.18),
        heading_deg=0.0,
        pitch_deg=0.0,
        bank_deg=0.0,
        size=size,
        view_pitch_deg=0.0,
    )
    assert nose[0] > tail[0]

    left_wing = _project_marker_reference_point(
        rect=rect,
        window_height=window_height,
        center_top_left=center_a,
        point=(-3.86, 0.56, 0.16),
        heading_deg=0.0,
        pitch_deg=0.0,
        bank_deg=20.0,
        size=size,
        view_pitch_deg=0.0,
    )
    right_wing = _project_marker_reference_point(
        rect=rect,
        window_height=window_height,
        center_top_left=center_a,
        point=(3.86, 0.56, 0.16),
        heading_deg=0.0,
        pitch_deg=0.0,
        bank_deg=20.0,
        size=size,
        view_pitch_deg=0.0,
    )
    assert right_wing[1] < left_wing[1]
