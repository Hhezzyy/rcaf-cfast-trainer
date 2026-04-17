from __future__ import annotations

import json
import math
import sys
from dataclasses import replace
from importlib.machinery import ModuleSpec
from types import SimpleNamespace
from types import ModuleType

import pygame
import pytest

if "moderngl" not in sys.modules:
    moderngl_stub = ModuleType("moderngl")
    moderngl_stub.__spec__ = ModuleSpec("moderngl", loader=None)
    sys.modules["moderngl"] = moderngl_stub
moderngl_for_tests = sys.modules["moderngl"]
for _name, _value in {
    "BLEND": 1,
    "CULL_FACE": 2,
    "DEPTH_TEST": 3,
    "SCISSOR_TEST": 4,
    "SRC_ALPHA": 5,
    "ONE_MINUS_SRC_ALPHA": 6,
    "ONE": 7,
    "LINEAR": 8,
    "TRIANGLES": 9,
}.items():
    if not hasattr(moderngl_for_tests, _name):
        setattr(moderngl_for_tests, _name, _value)

from cfast_trainer.aircraft_art import (
    apply_fixed_wing_view_rotation,
    fixed_wing_heading_from_screen_heading,
    project_fixed_wing_point,
    rotate_fixed_wing_point,
)
from cfast_trainer.auditory_capacity import AuditoryCapacityGate, build_auditory_capacity_test
from cfast_trainer.auditory_capacity_view import (
    BALL_FORWARD_IDLE_NORM,
    TUNNEL_GEOMETRY_END_DISTANCE,
    run_travel_distance,
)
from cfast_trainer.gl_scenes import (
    AuditoryGlScene,
    RapidTrackingGlScene,
    SpatialIntegrationGlScene,
    TraceTest1GlScene,
    TraceTest2GlScene,
)
from cfast_trainer.modern_gl_renderer import (
    ModernSceneRenderer,
    _AssetInstance,
    _build_auditory_scene_plan,
    _build_rapid_tracking_scene_plan,
    _build_spatial_integration_scene_plan,
    _build_trace_test_1_scene_plan,
    _build_trace_test_2_scene_plan,
    _GeometryBatch,
    _grounded_asset_world_z,
    _PreparedWorldTriangle,
    _project_aircraft_marker_polygons,
    _rapid_tracking_layout_world_xy,
    _rapid_tracking_render_polyline,
    _rapid_tracking_road_piece_instance,
    _rapid_tracking_static_bundle,
    _rapid_tracking_static_group_layer,
    _rapid_tracking_static_group_visible,
    _rapid_tracking_static_scene,
    _RapidTrackingStaticGroup,
    _RapidTrackingStaticScene,
    _scene_local_top_left_to_screen,
    _scene_screen_bounds,
    _SceneAssetLibrary,
    _SceneCamera,
    _ScenePlan,
)
from cfast_trainer.rapid_tracking import (
    build_rapid_tracking_compound_layout,
    build_rapid_tracking_test,
)
from cfast_trainer.render_assets import RenderAssetCatalog
from cfast_trainer.spatial_integration import (
    SpatialIntegrationLandmark,
    SpatialIntegrationPart,
    SpatialIntegrationSceneView,
    build_spatial_integration_test,
)
from cfast_trainer.spatial_integration_visuals import (
    spatial_integration_landmark_asset_id,
)
from cfast_trainer.trace_test_1 import (
    TraceTest1Generator,
    TraceTest1Payload,
    TraceTest1PromptPlan,
    TraceTest1TrialStage,
    trace_test_1_scene_frames,
)
from cfast_trainer.trace_test_2 import TraceTest2Generator


class _FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class _FakeScreen:
    def __init__(self, ctx: "_FakeContext") -> None:
        self._ctx = ctx

    def use(self) -> None:
        self._ctx.active_target = "screen"
        self._ctx.events.append(("use", "screen"))


class _FakeResource:
    def __init__(self, ctx: "_FakeContext", label: str) -> None:
        self._ctx = ctx
        self.label = label
        self.released = False

    def release(self) -> None:
        self.released = True
        self._ctx.events.append(("release", self.label))


class _FakeTexture(_FakeResource):
    def __init__(self, ctx: "_FakeContext", label: str, size: tuple[int, int]) -> None:
        super().__init__(ctx, label)
        self.size = size
        self.filter: tuple[int, int] | None = None
        self.repeat_x = False
        self.repeat_y = False
        self.writes: list[bytes] = []

    def use(self, *, location: int = 0) -> None:
        self._ctx.events.append(("texture_use", self.label, int(location)))

    def write(self, pixels: bytes) -> None:
        self.writes.append(pixels)
        self._ctx.events.append(("texture_write", self.label, len(pixels)))


class _FakeFramebuffer(_FakeResource):
    def __init__(self, ctx: "_FakeContext", label: str) -> None:
        super().__init__(ctx, label)

    def use(self) -> None:
        self._ctx.active_target = self.label
        self._ctx.events.append(("use", self.label))


class _FakeBuffer(_FakeResource):
    pass


class _FakeVertexArray(_FakeResource):
    def render(self, mode: int) -> None:
        self._ctx.events.append(("render", self.label, int(mode)))


class _FakeContext:
    def __init__(self) -> None:
        self.events: list[tuple[object, ...]] = []
        self.active_target = "screen"
        self.screen = _FakeScreen(self)
        self._viewport: tuple[int, int, int, int] | None = None
        self._scissor: tuple[int, int, int, int] | None = None
        self._blend_func: tuple[int, ...] | None = None
        self.clear_calls: list[
            tuple[str, tuple[float, float, float, float], dict[str, object]]
        ] = []
        self.created_textures: list[_FakeTexture] = []
        self.created_depth_renderbuffers: list[_FakeResource] = []
        self.created_framebuffers: list[_FakeFramebuffer] = []

    @property
    def viewport(self) -> tuple[int, int, int, int] | None:
        return self._viewport

    @viewport.setter
    def viewport(self, value: tuple[int, int, int, int] | None) -> None:
        self._viewport = value
        self.events.append(("viewport", value))

    @property
    def scissor(self) -> tuple[int, int, int, int] | None:
        return self._scissor

    @scissor.setter
    def scissor(self, value: tuple[int, int, int, int] | None) -> None:
        self._scissor = value
        self.events.append(("scissor", value))

    @property
    def blend_func(self) -> tuple[int, ...] | None:
        return self._blend_func

    @blend_func.setter
    def blend_func(self, value: tuple[int, ...]) -> None:
        self._blend_func = tuple(value)
        self.events.append(("blend_func", tuple(value)))

    def clear(
        self,
        r: float = 0.0,
        g: float = 0.0,
        b: float = 0.0,
        a: float = 0.0,
        **kwargs: object,
    ) -> None:
        rgba = (float(r), float(g), float(b), float(a))
        self.clear_calls.append((self.active_target, rgba, dict(kwargs)))
        self.events.append(("clear", self.active_target, rgba, dict(kwargs)))

    def enable(self, flag: int) -> None:
        self.events.append(("enable", int(flag)))

    def disable(self, flag: int) -> None:
        self.events.append(("disable", int(flag)))

    def texture(
        self,
        size: tuple[int, int],
        components: int,
        data: bytes | None = None,
    ) -> _FakeTexture:
        label = f"texture:{len(self.created_textures)}"
        texture = _FakeTexture(self, label, tuple(size))
        self.created_textures.append(texture)
        self.events.append(("texture", label, tuple(size), int(components), data is not None))
        return texture

    def depth_renderbuffer(self, size: tuple[int, int]) -> _FakeResource:
        label = f"depth:{len(self.created_depth_renderbuffers)}"
        depth = _FakeResource(self, label)
        self.created_depth_renderbuffers.append(depth)
        self.events.append(("depth_renderbuffer", label, tuple(size)))
        return depth

    def framebuffer(
        self,
        *,
        color_attachments: list[_FakeTexture],
        depth_attachment: _FakeResource,
    ) -> _FakeFramebuffer:
        label = f"framebuffer:{len(self.created_framebuffers)}"
        framebuffer = _FakeFramebuffer(self, label)
        self.created_framebuffers.append(framebuffer)
        self.events.append(
            (
                "framebuffer",
                label,
                tuple(texture.label for texture in color_attachments),
                depth_attachment.label,
            )
        )
        return framebuffer

    def buffer(self, data: bytes) -> _FakeBuffer:
        self.events.append(("buffer", len(data)))
        return _FakeBuffer(self, "buffer")

    def vertex_array(self, _program: object, _bindings: list[object]) -> _FakeVertexArray:
        self.events.append(("vertex_array", len(_bindings)))
        return _FakeVertexArray(self, "vao")


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


def _trace_test_1_payload(*, progress: float = 0.68) -> TraceTest1Payload:
    prompt = TraceTest1Generator(seed=44).next_problem(difficulty=0.82).payload
    assert isinstance(prompt, TraceTest1PromptPlan)
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


def _triangle_signature(batch: _GeometryBatch) -> tuple[tuple[float, float, float, float], ...]:
    return tuple(
        (
            round(vertex.x, 3),
            round(vertex.y, 3),
            round(vertex.r, 3),
            round(vertex.g, 3),
        )
        for vertex in batch.triangles
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
    renderer._last_rt_world_debug = {}
    renderer._scene_assets = _SceneAssetLibrary(RenderAssetCatalog())
    renderer._vortex_texture = object()
    renderer._ui_texture = None
    renderer._ui_tex_size = (0, 0)
    renderer._scene_color_texture = None
    renderer._scene_depth_renderbuffer = None
    renderer._scene_framebuffer = None
    renderer._scene_fb_size = (0, 0)
    renderer._color_program = object()
    renderer._scene_program = object()
    renderer._texture_program = object()
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
    gate_ids = {
        instance.asset_id
        for instance in plan.asset_instances
        if instance.asset_id.startswith("auditory_gate_")
    }

    assert {"auditory_gate_circle", "auditory_gate_triangle", "auditory_gate_square"} <= set(
        plan.asset_ids
    )
    assert gate_ids == {"auditory_gate_circle", "auditory_gate_triangle", "auditory_gate_square"}
    for instance in plan.asset_instances:
        if instance.asset_id.startswith("auditory_gate_"):
            assert instance.scale[0] > 0.0
            assert instance.scale[1] > 0.0
            assert instance.scale[2] > 0.0
            assert instance.color is not None
            assert instance.color[3] > 0.0


def test_auditory_scene_plan_camera_follows_presentation_travel_but_ignores_ball_offsets() -> None:
    baseline = _build_auditory_scene_plan(_auditory_scene_with_ball_offset(ball_x=0.0, ball_y=0.0))
    lateral = _build_auditory_scene_plan(_auditory_scene_with_ball_offset(ball_x=0.82, ball_y=-0.60))
    shifted_payload = replace(
        _auditory_payload(),
        phase_elapsed_s=9.4,
        presentation_travel_distance=run_travel_distance(
            session_seed=int(_auditory_payload().session_seed),
            phase_elapsed_s=9.4,
        ),
        ball_x=0.82,
        ball_y=-0.60,
        ball_forward_norm=0.74,
    )
    shifted = _build_auditory_scene_plan(
        AuditoryGlScene(
            world=pygame.Rect(0, 0, 640, 360),
            payload=shifted_payload,
            time_remaining_s=14.0,
            time_fill_ratio=0.42,
        )
    )

    assert baseline.camera is not None
    assert lateral.camera is not None
    assert shifted.camera is not None
    assert lateral.camera == baseline.camera
    assert shifted.camera.position[1] > baseline.camera.position[1]
    assert shifted.camera.position[0] > baseline.camera.position[0]


def test_auditory_scene_plan_gate_positions_advance_with_travel_but_keep_ball_spacing() -> None:
    base_scene = _auditory_scene_with_visible_gates()
    advanced_scene = AuditoryGlScene(
        world=pygame.Rect(0, 0, 640, 360),
        payload=replace(
            base_scene.payload,
            phase_elapsed_s=7.1,
            presentation_travel_distance=run_travel_distance(
                session_seed=int(base_scene.payload.session_seed),
                phase_elapsed_s=7.1,
            ),
            ball_forward_norm=0.72,
        )
        if base_scene.payload is not None
        else None,
        time_remaining_s=15.0,
        time_fill_ratio=0.33,
    )

    first_plan = _build_auditory_scene_plan(base_scene)
    second_plan = _build_auditory_scene_plan(advanced_scene)
    first_ball = next(
        instance for instance in first_plan.asset_instances if instance.asset_id == "auditory_ball"
    )
    second_ball = next(
        instance for instance in second_plan.asset_instances if instance.asset_id == "auditory_ball"
    )
    first_gates = sorted(
        instance.position[1] - first_ball.position[1]
        for instance in first_plan.asset_instances
        if instance.asset_id.startswith("auditory_gate_")
    )
    second_gates = sorted(
        instance.position[1] - second_ball.position[1]
        for instance in second_plan.asset_instances
        if instance.asset_id.startswith("auditory_gate_")
    )

    assert second_ball.position[1] > first_ball.position[1]
    assert second_gates == pytest.approx(first_gates)


def test_auditory_scene_plan_ball_advances_with_presentation_travel() -> None:
    scene_near = AuditoryGlScene(
        world=pygame.Rect(0, 0, 640, 360),
        payload=replace(_auditory_payload(), ball_forward_norm=float(BALL_FORWARD_IDLE_NORM)),
        time_remaining_s=18.0,
        time_fill_ratio=0.44,
    )
    scene_far = AuditoryGlScene(
        world=pygame.Rect(0, 0, 640, 360),
        payload=replace(
            _auditory_payload(),
            phase_elapsed_s=9.4,
            presentation_travel_distance=run_travel_distance(
                session_seed=int(_auditory_payload().session_seed),
                phase_elapsed_s=9.4,
            ),
            ball_forward_norm=0.74,
        ),
        time_remaining_s=18.0,
        time_fill_ratio=0.44,
    )

    near_plan = _build_auditory_scene_plan(scene_near)
    far_plan = _build_auditory_scene_plan(scene_far)
    near_ball = next(
        instance for instance in near_plan.asset_instances if instance.asset_id == "auditory_ball"
    )
    far_ball = next(
        instance for instance in far_plan.asset_instances if instance.asset_id == "auditory_ball"
    )

    assert far_ball.position[1] > near_ball.position[1]


def test_auditory_scene_plan_uses_core_collision_count_and_roll() -> None:
    payload = replace(
        _auditory_payload(),
        presentation_travel_distance=20.0,
        ball_contact_ratio=1.05,
        collisions=3,
    )

    plan = _build_auditory_scene_plan(
        AuditoryGlScene(
            world=pygame.Rect(0, 0, 640, 360),
            payload=payload,
            time_remaining_s=18.0,
            time_fill_ratio=0.44,
        )
    )
    ball = next(instance for instance in plan.asset_instances if instance.asset_id == "auditory_ball")

    assert plan.debug is not None
    assert ball.hpr_deg[2] == pytest.approx(plan.debug["auditory_ball_roll_deg"])
    assert abs(float(ball.hpr_deg[2])) > 1.0
    assert ball.color is not None
    assert ball.color[0] > ball.color[1]
    assert plan.debug["auditory_collision_active"] is True
    assert plan.debug["auditory_collision_penalties"] == 3
    assert plan.debug["auditory_core_collisions"] == 3


def test_auditory_draw_scene_exposes_follower_debug_fields() -> None:
    renderer = _build_pipeline_probe_renderer()
    payload = replace(
        _auditory_payload(),
        presentation_travel_distance=24.0,
        ball_contact_ratio=1.01,
        collisions=2,
    )
    scene = AuditoryGlScene(
        world=pygame.Rect(0, 0, 640, 360),
        payload=payload,
        time_remaining_s=18.0,
        time_fill_ratio=0.44,
    )

    renderer._draw_auditory_scene = lambda *, scene, scene_plan: scene_plan.entity_count
    renderer._draw_scene(scene=scene)
    debug = renderer.debug_last_scene()

    assert debug["kind"] == "auditory"
    assert debug["auditory_travel_distance"] == pytest.approx(24.0)
    assert debug["auditory_collision_active"] is True
    assert debug["auditory_core_collisions"] == 2


def test_auditory_scene_plan_gate_flash_overrides_base_gate_color() -> None:
    payload = replace(
        _auditory_payload(),
        gates=(
            AuditoryCapacityGate(
                gate_id=801,
                x_norm=1.05,
                y_norm=0.0,
                color="RED",
                shape="CIRCLE",
                aperture_norm=0.20,
            ),
            AuditoryCapacityGate(
                gate_id=802,
                x_norm=1.05,
                y_norm=0.0,
                color="RED",
                shape="SQUARE",
                aperture_norm=0.20,
                flash_color="WHITE",
                flash_strength=1.0,
            ),
            AuditoryCapacityGate(
                gate_id=803,
                x_norm=1.05,
                y_norm=0.0,
                color="RED",
                shape="TRIANGLE",
                aperture_norm=0.20,
                flash_color="ERROR_RED",
                flash_strength=1.0,
            ),
        ),
    )
    plan = _build_auditory_scene_plan(
        AuditoryGlScene(
            world=pygame.Rect(0, 0, 640, 360),
            payload=payload,
            time_remaining_s=12.0,
            time_fill_ratio=0.40,
        )
    )
    gate_colors = {
        instance.asset_id: instance.color
        for instance in plan.asset_instances
        if instance.asset_id.startswith("auditory_gate_")
    }

    base_red = gate_colors["auditory_gate_circle"]
    white_flash = gate_colors["auditory_gate_square"]
    error_red = gate_colors["auditory_gate_triangle"]

    assert white_flash[1] > base_red[1]
    assert white_flash[2] > base_red[2]
    assert error_red[1] > base_red[1]
    assert error_red[2] < white_flash[2]


def test_auditory_scene_plan_keeps_far_end_tunnel_ring_in_front_of_camera() -> None:
    scene = _auditory_scene_for_render_frame()
    plan = _build_auditory_scene_plan(scene)
    assert plan.camera is not None

    farthest_rib = max(
        (
            instance
            for instance in plan.asset_instances
            if instance.asset_id == "auditory_tunnel_rib"
        ),
        key=lambda instance: instance.position[1],
    )

    assert scene.payload is not None
    assert farthest_rib.position[1] == pytest.approx(
        float(scene.payload.presentation_travel_distance) + TUNNEL_GEOMETRY_END_DISTANCE,
        abs=0.01,
    )
    assert farthest_rib.position[1] > plan.camera.position[1]
    assert farthest_rib.position[1] < plan.camera.far_clip


def test_render_frame_flushes_scene_target_before_color_composite_and_ui() -> None:
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
    renderer._composite_scene_target = lambda: order.append("composite")
    renderer._draw_ui_surface = fake_draw_ui_surface

    renderer.render_frame(
        ui_surface=pygame.Surface((640, 360), pygame.SRCALPHA),
        scene=_auditory_scene_for_render_frame(),
    )

    assert order == [
        "draw_scene",
        ("scene_textures", 1),
        ("color", 0, 1),
        "composite",
        ("ui", 0, (640, 360)),
    ]


def test_render_frame_clears_default_and_scene_framebuffers_before_ui() -> None:
    renderer = _build_pipeline_probe_renderer()
    ctx = renderer._ctx
    assert isinstance(ctx, _FakeContext)
    order: list[str] = []

    def fake_draw_scene(*, scene) -> None:
        assert isinstance(scene, RapidTrackingGlScene)
        order.append("draw_scene")

    renderer._draw_scene = fake_draw_scene
    renderer._flush_scene_textures = lambda: order.append("scene_textures")
    renderer._flush_color_geometry = lambda: order.append("color")
    renderer._composite_scene_target = lambda: order.append("composite")
    renderer._draw_ui_surface = lambda *, ui_surface: order.append("ui")

    scene = RapidTrackingGlScene(
        world=pygame.Rect(10, 20, 300, 180),
        payload=None,
        active_phase=True,
    )
    renderer.render_frame(
        ui_surface=pygame.Surface((640, 360), pygame.SRCALPHA),
        scene=scene,
    )

    assert ctx.clear_calls[0] == ("screen", (0.01, 0.02, 0.06, 1.0), {"depth": 1.0})
    assert ctx.clear_calls[1] == ("framebuffer:0", (0.0, 0.0, 0.0, 0.0), {"depth": 1.0})
    assert order == ["draw_scene", "scene_textures", "color", "composite", "ui"]
    assert len(ctx.created_textures) == 1
    assert len(ctx.created_depth_renderbuffers) == 1
    assert len(ctx.created_framebuffers) == 1


def test_scene_target_scissors_rapid_tracking_and_restores_state() -> None:
    renderer = _build_pipeline_probe_renderer()
    ctx = renderer._ctx
    assert isinstance(ctx, _FakeContext)
    scene = RapidTrackingGlScene(
        world=pygame.Rect(12, 28, 320, 144),
        payload=None,
        active_phase=True,
    )
    expected_scissor = renderer._scene_scissor_rect(scene.world)

    def fake_draw_scene(*, scene: object) -> None:
        ctx.events.append(("draw_scene", scene.__class__.__name__))

    renderer._draw_scene = fake_draw_scene
    renderer._flush_scene_textures = lambda: ctx.events.append(("scene_textures",))
    renderer._flush_color_geometry = lambda: ctx.events.append(("color",))

    renderer._render_scene_target(scene=scene)

    assert ("scissor", expected_scissor) in ctx.events
    assert ctx.scissor is None
    scissor_on = ctx.events.index(("enable", moderngl_for_tests.SCISSOR_TEST))
    draw_idx = ctx.events.index(("draw_scene", "RapidTrackingGlScene"))
    scissor_off = ctx.events.index(("disable", moderngl_for_tests.SCISSOR_TEST))
    assert scissor_on < draw_idx < scissor_off
    assert ctx.blend_func == (
        moderngl_for_tests.SRC_ALPHA,
        moderngl_for_tests.ONE_MINUS_SRC_ALPHA,
        moderngl_for_tests.ONE,
        moderngl_for_tests.ONE_MINUS_SRC_ALPHA,
    )
    assert ("disable", moderngl_for_tests.DEPTH_TEST) in ctx.events


def test_scene_target_scissors_spatial_integration_before_ui_composite() -> None:
    renderer = _build_pipeline_probe_renderer()
    ctx = renderer._ctx
    assert isinstance(ctx, _FakeContext)
    order: list[str] = []
    scene = SpatialIntegrationGlScene(
        world=pygame.Rect(28, 36, 420, 220),
        payload=None,
    )
    expected_scissor = renderer._scene_scissor_rect(scene.world)

    renderer._draw_scene = lambda *, scene: order.append("draw_scene")
    renderer._flush_scene_textures = lambda: order.append("scene_textures")
    renderer._flush_color_geometry = lambda: order.append("color")
    renderer._composite_scene_target = lambda: order.append("composite")
    renderer._draw_ui_surface = lambda *, ui_surface: order.append("ui")

    renderer.render_frame(
        ui_surface=pygame.Surface((640, 360), pygame.SRCALPHA),
        scene=scene,
    )

    assert ("scissor", expected_scissor) in ctx.events
    assert order == ["draw_scene", "scene_textures", "color", "composite", "ui"]


def test_world_geometry_flush_does_not_clear_backdrop_color() -> None:
    renderer = _build_pipeline_probe_renderer()
    renderer._batch.scene_triangles.extend(
        [
            SimpleNamespace(x=0.0, y=0.0, depth=0.3, r=0.2, g=0.4, b=0.6, a=1.0),
            SimpleNamespace(x=1.0, y=0.0, depth=0.4, r=0.2, g=0.4, b=0.6, a=1.0),
            SimpleNamespace(x=0.0, y=1.0, depth=0.5, r=0.2, g=0.4, b=0.6, a=1.0),
        ]
    )

    renderer._flush_world_geometry()

    ctx = renderer._ctx
    assert isinstance(ctx, _FakeContext)
    assert ctx.clear_calls == []
    assert not renderer._batch.scene_triangles


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
        "shrubs_low_cluster",
        "soldiers_patrol",
        "terrain_hill_mound",
        "terrain_lake_patch",
        "terrain_rock_cluster",
        "trees_field_cluster",
        "trees_pine_cluster",
        "truck_olive",
        "vehicle_tracked",
        "road_dirt_segment",
        "road_paved_segment",
    } <= asset_ids
    assert {
        "building_hangar",
        "building_tower",
    } <= asset_ids
    assert plan.static_groups
    assert {group.layer for group in plan.static_groups} >= {"near", "far"}
    assert any(
        asset_id in {"truck_olive", "vehicle_tracked", "soldiers_patrol"}
        for asset_id in instance_ids
    )
    assert any(
        asset_id
        in {
            "forest_canopy_patch",
            "shrubs_low_cluster",
            "trees_field_cluster",
            "trees_pine_cluster",
        }
        for asset_id in instance_ids
    )
    assert len(plan.asset_instances) <= 95
    assert plan.entity_count >= len(plan.asset_instances)


def test_rapid_tracking_scene_plan_reuses_cached_static_world_budget() -> None:
    static_a = _rapid_tracking_static_scene(551)
    static_b = _rapid_tracking_static_scene(551)
    bundle_a = _rapid_tracking_static_bundle(551)
    bundle_b = _rapid_tracking_static_bundle(551)

    assert static_a is static_b
    assert bundle_a is bundle_b
    assert len(static_a.ambient_instances) <= 20
    assert bundle_a.instance_count == len(static_a.core_instances)
    assert bundle_a.triangle_count > 0
    assert {group.layer for group in bundle_a.groups} >= {"near", "far"}
    assert max(group.radius for group in bundle_a.groups if group.layer != "far") <= 80.0
    assert {
        "building_hangar",
        "building_tower",
        "road_dirt_segment",
        "road_paved_segment",
        "terrain_hill_mound",
        "terrain_lake_patch",
        "terrain_rock_cluster",
    } <= set(bundle_a.asset_ids)


def test_rapid_tracking_static_bundle_changes_with_scene_seed() -> None:
    bundle_a = _rapid_tracking_static_bundle(551)
    bundle_b = _rapid_tracking_static_bundle(552)

    signature_a = tuple(
        (group.group_id, round(group.center[0], 2), round(group.center[1], 2))
        for group in bundle_a.groups
    )
    signature_b = tuple(
        (group.group_id, round(group.center[0], 2), round(group.center[1], 2))
        for group in bundle_b.groups
    )

    assert signature_a != signature_b


def test_rapid_tracking_static_group_visibility_rejects_offscreen_clusters() -> None:
    camera = _SceneCamera(
        position=(0.0, 0.0, 12.0),
        heading_deg=0.0,
        pitch_deg=0.0,
        h_fov_deg=60.0,
        v_fov_deg=40.0,
    )
    triangles = (
        _PreparedWorldTriangle(
            points=((490.0, 10.0, 0.0), (500.0, 10.0, 0.0), (500.0, 12.0, 0.0)),
            normal=(0.0, 0.0, 1.0),
            base_rgb=(0.4, 0.5, 0.4),
        ),
    )
    hidden_group = _RapidTrackingStaticGroup(
        group_id="offscreen",
        layer="near",
        center=(500.0, 10.0, 0.0),
        radius=12.0,
        triangles=triangles,
        instance_count=1,
    )
    visible_group = _RapidTrackingStaticGroup(
        group_id="visible",
        layer="near",
        center=(0.0, 240.0, 0.0),
        radius=18.0,
        triangles=triangles,
        instance_count=1,
    )

    assert _rapid_tracking_static_group_visible(camera=camera, group=hidden_group) is False
    assert _rapid_tracking_static_group_visible(camera=camera, group=visible_group) is True


def test_rapid_tracking_scene_plan_uses_scene_seed_for_camera_and_scenery(monkeypatch) -> None:
    payload = replace(_rapid_tracking_payload(), scene_seed=987654)
    scene = RapidTrackingGlScene(
        world=pygame.Rect(0, 0, 640, 360),
        payload=payload,
        active_phase=True,
    )
    captured: dict[str, int] = {}
    original_static_scene = _rapid_tracking_static_scene

    def fake_camera_rig_state(**kwargs):
        captured["camera_seed"] = int(kwargs["seed"])
        return SimpleNamespace(
            cam_world_x=0.0,
            cam_world_y=0.0,
            cam_world_z=12.0,
            view_heading_deg=0.0,
            view_pitch_deg=0.0,
            fov_deg=60.0,
        )

    def fake_static_scene(seed: int):
        captured["static_seed"] = int(seed)
        return original_static_scene(int(seed))

    def fake_estimated_target_world_z(**kwargs):
        captured["target_world_z_seed"] = int(kwargs["seed"])
        return 0.0

    monkeypatch.setattr(
        "cfast_trainer.modern_gl_renderer.rapid_tracking_camera_rig_state", fake_camera_rig_state
    )
    monkeypatch.setattr(
        "cfast_trainer.modern_gl_renderer._rapid_tracking_static_scene", fake_static_scene
    )
    monkeypatch.setattr(
        "cfast_trainer.modern_gl_renderer.estimated_target_world_z", fake_estimated_target_world_z
    )

    _build_rapid_tracking_scene_plan(scene)

    assert captured == {
        "camera_seed": int(payload.scene_seed),
        "static_seed": int(payload.scene_seed),
        "target_world_z_seed": int(payload.scene_seed),
    }


def test_rapid_tracking_scene_plan_limits_readable_ambient_clutter(monkeypatch) -> None:
    payload = _rapid_tracking_payload()
    readable_payload = replace(payload, scene_seed=int(payload.session_seed) + 99)
    scene = RapidTrackingGlScene(
        world=pygame.Rect(0, 0, 640, 360),
        payload=readable_payload,
        active_phase=True,
    )
    layout = build_rapid_tracking_compound_layout(seed=int(readable_payload.scene_seed))
    ambient_instances = tuple(
        _AssetInstance(
            asset_id=asset_id,
            position=(float(idx), 0.0, 0.0),
        )
        for idx, asset_id in enumerate(
            (
                "forest_canopy_patch",
                "trees_field_cluster",
                "forest_canopy_patch",
                "shrubs_low_cluster",
                "trees_pine_cluster",
                "forest_canopy_patch",
                "shrubs_low_cluster",
                "trees_field_cluster",
                "trees_pine_cluster",
                "shrubs_low_cluster",
            )
        )
    )
    static_scene = _RapidTrackingStaticScene(
        layout=layout,
        core_instances=(),
        ambient_instances=ambient_instances,
        asset_ids=tuple(sorted({instance.asset_id for instance in ambient_instances})),
    )
    rank_by_position = {
        tuple(instance.position): idx for idx, instance in enumerate(ambient_instances)
    }

    monkeypatch.setattr(
        "cfast_trainer.modern_gl_renderer._rapid_tracking_static_scene", lambda _seed: static_scene
    )
    monkeypatch.setattr(
        "cfast_trainer.modern_gl_renderer._rapid_tracking_static_instance_visible",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        "cfast_trainer.modern_gl_renderer._rapid_tracking_visibility_rank",
        lambda instance, **kwargs: rank_by_position[tuple(instance.position)],
    )

    readable_plan = _build_rapid_tracking_scene_plan(scene)
    default_plan = _build_rapid_tracking_scene_plan(
        RapidTrackingGlScene(
            world=pygame.Rect(0, 0, 640, 360),
            payload=replace(readable_payload, scene_seed=int(readable_payload.session_seed)),
            active_phase=True,
        )
    )

    readable_ambient = readable_plan.asset_instances[:-1]
    default_ambient = default_plan.asset_instances[:-1]

    assert len(default_ambient) == 6
    assert sum(1 for instance in default_ambient if instance.asset_id == "forest_canopy_patch") == 1
    assert len(readable_ambient) == 4
    assert (
        sum(1 for instance in readable_ambient if instance.asset_id == "forest_canopy_patch") <= 1
    )


def test_rapid_tracking_local_hills_never_enter_backdrop_layer() -> None:
    static_scene = _rapid_tracking_static_scene(551)
    local_hills = [
        instance
        for instance in static_scene.core_instances
        if instance.asset_id == "terrain_hill_mound" and instance.bucket != "backdrop"
    ]

    assert local_hills
    assert all(
        _rapid_tracking_static_group_layer(instance, center_x=0.0, center_y=0.0) == "near"
        for instance in local_hills
    )


def test_rapid_tracking_world_projection_clips_horizontal_frustum_spill() -> None:
    renderer = _build_pipeline_probe_renderer()
    camera = _SceneCamera(
        position=(0.0, 0.0, 0.0),
        heading_deg=0.0,
        pitch_deg=0.0,
        h_fov_deg=60.0,
        v_fov_deg=40.0,
    )
    triangle = _PreparedWorldTriangle(
        points=((18.0, 40.0, -4.0), (40.0, 40.0, 4.0), (18.0, 80.0, 2.0)),
        normal=(0.0, 0.0, 1.0),
        base_rgb=(0.44, 0.56, 0.48),
    )
    scene_plan = _ScenePlan(
        kind="rapid_tracking",
        rect=pygame.Rect(0, 0, 640, 360),
        camera=camera,
        asset_instances=(),
        overlay_primitives=(),
        asset_ids=(),
        entity_count=1,
        static_groups=(
            _RapidTrackingStaticGroup(
                group_id="clip",
                layer="near",
                center=(25.0, 53.0, 0.0),
                radius=36.0,
                triangles=(triangle,),
                instance_count=1,
            ),
        ),
    )

    renderer._render_scene_plan(scene_plan=scene_plan)

    assert renderer._batch.scene_triangles
    assert all(
        math.isfinite(vertex.x) and math.isfinite(vertex.y)
        for vertex in renderer._batch.scene_triangles
    )
    assert all(0.0 <= vertex.x <= 640.0 for vertex in renderer._batch.scene_triangles)
    assert all(0.0 <= vertex.y <= 360.0 for vertex in renderer._batch.scene_triangles)
    debug = renderer._last_rt_world_debug
    assert debug["world_triangles_input"] == 1
    assert debug["world_triangles_clipped"] == 1
    assert debug["world_triangles_emitted"] >= 1


def test_rapid_tracking_world_projection_clips_near_plane_to_finite_triangles() -> None:
    renderer = _build_pipeline_probe_renderer()
    camera = _SceneCamera(
        position=(0.0, 0.0, 0.0),
        heading_deg=0.0,
        pitch_deg=0.0,
        h_fov_deg=60.0,
        v_fov_deg=40.0,
        near_clip=0.12,
    )
    triangle = _PreparedWorldTriangle(
        points=((0.0, 0.05, 0.0), (0.18, 1.20, 0.0), (-0.18, 1.20, 0.12)),
        normal=(0.0, 0.0, 1.0),
        base_rgb=(0.50, 0.54, 0.58),
    )
    scene_plan = _ScenePlan(
        kind="rapid_tracking",
        rect=pygame.Rect(0, 0, 640, 360),
        camera=camera,
        asset_instances=(),
        overlay_primitives=(),
        asset_ids=(),
        entity_count=1,
        static_groups=(
            _RapidTrackingStaticGroup(
                group_id="near-clip",
                layer="near",
                center=(0.0, 0.82, 0.04),
                radius=1.2,
                triangles=(triangle,),
                instance_count=1,
            ),
        ),
    )

    renderer._render_scene_plan(scene_plan=scene_plan)

    assert renderer._batch.scene_triangles
    assert all(
        math.isfinite(vertex.x) and math.isfinite(vertex.y)
        for vertex in renderer._batch.scene_triangles
    )
    assert all(0.0 <= vertex.x <= 640.0 for vertex in renderer._batch.scene_triangles)
    assert all(0.0 <= vertex.y <= 360.0 for vertex in renderer._batch.scene_triangles)
    assert renderer._last_rt_world_debug["world_triangles_clipped"] == 1
    assert renderer._last_rt_world_debug["max_projected_extent_px"] <= 640.0


def test_spatial_integration_world_projection_clips_horizontal_frustum_spill() -> None:
    renderer = _build_pipeline_probe_renderer()
    rect = pygame.Rect(40, 30, 240, 160)
    camera = _SceneCamera(
        position=(0.0, 0.0, 0.0),
        heading_deg=0.0,
        pitch_deg=0.0,
        h_fov_deg=60.0,
        v_fov_deg=40.0,
    )
    triangle = _PreparedWorldTriangle(
        points=((18.0, 40.0, -4.0), (40.0, 40.0, 4.0), (18.0, 80.0, 2.0)),
        normal=(0.0, 0.0, 1.0),
        base_rgb=(0.44, 0.56, 0.48),
    )
    scene_plan = _ScenePlan(
        kind="spatial_integration",
        rect=rect,
        camera=camera,
        asset_instances=(),
        overlay_primitives=(),
        asset_ids=(),
        entity_count=1,
        static_groups=(
            _RapidTrackingStaticGroup(
                group_id="spatial-clip",
                layer="near",
                center=(25.0, 53.0, 0.0),
                radius=36.0,
                triangles=(triangle,),
                instance_count=1,
            ),
        ),
    )

    renderer._render_scene_plan(scene_plan=scene_plan)

    bounds = _scene_screen_bounds(rect=rect, window_height=renderer._win_h)
    assert renderer._batch.triangles
    assert all(
        (bounds[0] - 1e-4) <= vertex.x <= (bounds[2] + 1e-4)
        and (bounds[1] - 1e-4) <= vertex.y <= (bounds[3] + 1e-4)
        for vertex in renderer._batch.triangles
    )


def test_rapid_tracking_playfield_groups_stay_within_radius_cap_for_seed_551() -> None:
    bundle = _rapid_tracking_static_bundle(551)
    playfield_groups = [group for group in bundle.groups if group.layer != "far"]

    assert playfield_groups
    assert max(group.radius for group in playfield_groups) <= 80.0


def test_grounded_asset_world_z_uses_mesh_base_height(monkeypatch) -> None:
    class _FakeLibrary:
        @staticmethod
        def mesh(_asset_id: str):
            return SimpleNamespace(base_z=1.5)

    monkeypatch.setattr(
        "cfast_trainer.modern_gl_renderer._rapid_tracking_static_asset_library",
        lambda: _FakeLibrary(),
    )

    assert _grounded_asset_world_z(
        asset_id="road_paved_segment", terrain_z=10.0, scale=(1.0, 1.0, 2.0)
    ) == pytest.approx(7.01)


def test_rapid_tracking_render_polyline_subdivides_long_world_segments() -> None:
    layout = build_rapid_tracking_compound_layout(seed=551)
    sampled = _rapid_tracking_render_polyline(
        layout,
        (
            (float(layout.compound_center_x) - 2.8, float(layout.compound_center_y) - 0.1),
            (float(layout.compound_center_x) + 2.8, float(layout.compound_center_y) - 0.1),
        ),
    )

    assert len(sampled) > 2
    for start_xy, end_xy in zip(sampled, sampled[1:], strict=False):
        start_wx, start_wy = _rapid_tracking_layout_world_xy(
            layout,
            track_x=float(start_xy[0]),
            track_y=float(start_xy[1]),
        )
        end_wx, end_wy = _rapid_tracking_layout_world_xy(
            layout,
            track_x=float(end_xy[0]),
            track_y=float(end_xy[1]),
        )
        assert math.hypot(float(end_wx - start_wx), float(end_wy - start_wy)) <= 24.0001


def test_rapid_tracking_seed_551_scene_debug_reports_clipped_world_geometry() -> None:
    renderer = _build_pipeline_probe_renderer()
    renderer._flush_color_geometry = lambda: None
    renderer._flush_world_geometry = lambda: None

    renderer._draw_scene(
        scene=RapidTrackingGlScene(
            world=pygame.Rect(0, 0, 640, 360),
            payload=_rapid_tracking_payload(),
            active_phase=True,
        )
    )

    debug = renderer.debug_last_scene()

    assert debug["kind"] == "rapid_tracking"
    assert debug["world_triangles_input"] > 0
    assert debug["world_triangles_emitted"] > 0
    assert debug["world_triangles_clipped"] > 0
    assert debug["world_triangles_rejected"] >= 0
    assert debug["max_projected_extent_px"] <= 640.0


def test_rapid_tracking_road_piece_uses_average_sampled_height(monkeypatch) -> None:
    layout = build_rapid_tracking_compound_layout(seed=551)
    start_xy = (-0.5, 0.2)
    end_xy = (0.7, 0.2)
    start_wx, start_wy = _rapid_tracking_layout_world_xy(
        layout, track_x=float(start_xy[0]), track_y=float(start_xy[1])
    )
    end_wx, end_wy = _rapid_tracking_layout_world_xy(
        layout, track_x=float(end_xy[0]), track_y=float(end_xy[1])
    )
    mid_wx = (start_wx + end_wx) * 0.5
    mid_wy = (start_wy + end_wy) * 0.5

    def fake_terrain_height(wx: float, wy: float) -> float:
        if abs(float(wx) - float(start_wx)) < 1e-6 and abs(float(wy) - float(start_wy)) < 1e-6:
            return 5.0
        if abs(float(wx) - float(mid_wx)) < 1e-6 and abs(float(wy) - float(mid_wy)) < 1e-6:
            return 11.0
        if abs(float(wx) - float(end_wx)) < 1e-6 and abs(float(wy) - float(end_wy)) < 1e-6:
            return 17.0
        raise AssertionError(f"unexpected sample: {(wx, wy)}")

    monkeypatch.setattr(
        "cfast_trainer.modern_gl_renderer.rapid_tracking_terrain_height", fake_terrain_height
    )

    instance = _rapid_tracking_road_piece_instance(
        layout,
        asset_id="road_paved_segment",
        start_xy=start_xy,
        end_xy=end_xy,
        width=8.0,
    )

    assert instance.position[2] == pytest.approx(11.01)


def test_rapid_tracking_static_scene_dedupes_duplicate_tree_cluster_positions() -> None:
    static_scene = _rapid_tracking_static_scene(551)
    tree_positions = [
        (round(float(instance.position[0]), 3), round(float(instance.position[1]), 3))
        for instance in static_scene.ambient_instances
        if instance.asset_id in {"trees_field_cluster", "trees_pine_cluster"}
    ]

    assert tree_positions
    assert len(tree_positions) == len(set(tree_positions))


def test_scene_asset_library_uses_file_backed_rt_meshes_when_available() -> None:
    library = _SceneAssetLibrary(RenderAssetCatalog())
    catalog = RenderAssetCatalog()

    mesh = library.mesh("building_hangar")

    assert catalog.resolve("building_hangar") is not None
    assert catalog.resolve("building_hangar").path is not None
    assert catalog.resolve("plane_red").path == catalog.resolve("plane_blue").path
    assert mesh.asset_id == "building_hangar"
    assert len(mesh.triangles) > 0


def test_scene_asset_library_falls_back_to_builtin_when_candidate_is_missing(tmp_path) -> None:
    manifest = {
        "assets": {
            "building_hangar": {
                "category": "scenery",
                "builtin_kind": "hangar",
                "scale": 1.0,
                "candidates": ["missing/building_hangar.obj"],
            }
        }
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    library = _SceneAssetLibrary(
        RenderAssetCatalog(
            asset_root=tmp_path,
            manifest_path=manifest_path,
        )
    )

    mesh = library.mesh("building_hangar")

    assert mesh.asset_id == "building_hangar"
    assert len(mesh.triangles) > 0


def test_scene_asset_library_builds_builtin_spatial_integration_landmark_meshes() -> None:
    library = _SceneAssetLibrary(RenderAssetCatalog())

    tent_mesh = library.mesh("spatial_tent_canvas")
    sheep_mesh = library.mesh("spatial_sheep_flock")

    assert tent_mesh.asset_id == "spatial_tent_canvas"
    assert sheep_mesh.asset_id == "spatial_sheep_flock"
    assert len(tent_mesh.triangles) > 0
    assert len(sheep_mesh.triangles) > 0
    assert tent_mesh.role_palette["body"] != (0.78, 0.80, 0.84)
    assert sheep_mesh.role_palette["body"] != (0.78, 0.80, 0.84)


def test_spatial_integration_scene_plan_includes_aircraft_and_landmarks() -> None:
    scene = SpatialIntegrationGlScene(
        world=pygame.Rect(0, 0, 640, 360),
        payload=_spatial_integration_payload(),
    )

    plan = _build_spatial_integration_scene_plan(scene)
    instance_ids = [instance.asset_id for instance in plan.asset_instances]

    assert plan.camera is not None
    assert "plane_blue" in instance_ids
    assert any(
        asset_id
        in {
            "building_hangar",
            "building_tower",
            "truck_olive",
            "soldiers_patrol",
            "spatial_tent_canvas",
            "spatial_sheep_flock",
            "trees_field_cluster",
            "forest_canopy_patch",
        }
        for asset_id in instance_ids
    )
    assert plan.entity_count == len(plan.asset_instances)


def test_spatial_integration_backdrops_are_nonblack_for_oblique_and_topdown() -> None:
    payload = _spatial_integration_payload()
    for scene_view in (
        SpatialIntegrationSceneView.OBLIQUE,
        SpatialIntegrationSceneView.TOPDOWN,
    ):
        renderer = _build_pipeline_probe_renderer()
        renderer._render_scene_plan = lambda *, scene_plan: None
        scene = SpatialIntegrationGlScene(
            world=pygame.Rect(0, 0, 640, 360),
            payload=replace(payload, scene_view=scene_view),
        )
        plan = _build_spatial_integration_scene_plan(scene)

        renderer._draw_spatial_integration_scene(scene=scene, scene_plan=plan)

        backdrop_colors = [
            (vertex.r, vertex.g, vertex.b)
            for vertex in renderer._batch.triangles[:12]
        ]
        assert backdrop_colors
        assert min(sum(color) for color in backdrop_colors) > 0.5


def test_spatial_integration_scene_plan_maps_landmark_kinds_to_asset_inventory() -> None:
    payload = replace(
        _spatial_integration_payload(),
        part=SpatialIntegrationPart.AIRCRAFT,
        show_aircraft_motion=True,
        query_label="BLD1",
        landmarks=(
            SpatialIntegrationLandmark(label="BLD1", x=1, y=1, kind="building"),
            SpatialIntegrationLandmark(label="TWR1", x=2, y=2, kind="tower"),
            SpatialIntegrationLandmark(label="TRK1", x=3, y=3, kind="truck"),
            SpatialIntegrationLandmark(label="SOL1", x=4, y=2, kind="foot_soldiers"),
            SpatialIntegrationLandmark(label="WOOD", x=5, y=4, kind="forest"),
            SpatialIntegrationLandmark(label="TENT", x=2, y=4, kind="tent"),
            SpatialIntegrationLandmark(label="SHP1", x=4, y=5, kind="sheep"),
        ),
    )
    plan = _build_spatial_integration_scene_plan(
        SpatialIntegrationGlScene(world=pygame.Rect(0, 0, 640, 360), payload=payload)
    )
    instance_ids = {instance.asset_id for instance in plan.asset_instances}
    expected_landmark_assets = {
        spatial_integration_landmark_asset_id(label=landmark.label, kind=landmark.kind)
        for landmark in payload.landmarks
    }

    assert {
        "plane_blue",
        "plane_green",
        "plane_yellow",
    } <= instance_ids
    assert expected_landmark_assets <= instance_ids
    assert set(plan.asset_ids) == instance_ids


def test_spatial_integration_scene_plan_offsets_same_cell_landmarks_deterministically() -> None:
    payload = replace(
        _spatial_integration_payload(),
        query_label="BLD1",
        landmarks=(
            SpatialIntegrationLandmark(label="BLD1", x=3, y=3, kind="building"),
            SpatialIntegrationLandmark(label="TENT", x=3, y=3, kind="tent"),
        ),
    )
    scene = SpatialIntegrationGlScene(world=pygame.Rect(0, 0, 640, 360), payload=payload)

    plan_a = _build_spatial_integration_scene_plan(scene)
    plan_b = _build_spatial_integration_scene_plan(scene)
    landmark_instances = [
        instance
        for instance in plan_a.asset_instances
        if instance.asset_id in {"building_hangar", "spatial_tent_canvas"}
    ]

    assert len(landmark_instances) == 2
    assert landmark_instances[0].position != landmark_instances[1].position
    assert plan_a.asset_instances == plan_b.asset_instances


def test_trace_test_1_scene_plan_builds_real_world_camera_and_aircraft_instances() -> None:
    payload = _trace_test_1_payload()
    scene = TraceTest1GlScene(world=pygame.Rect(0, 0, 640, 360), payload=payload)

    plan = _build_trace_test_1_scene_plan(scene)

    assert plan.camera is not None
    assert plan.overlay_primitives == ()
    assert plan.entity_count == 1 + len(payload.scene.blue_frames)
    assert plan.asset_instances == ()
    assert plan.asset_ids == ()


def test_trace_test_2_scene_plan_builds_real_world_camera_and_practice_ghosts() -> None:
    payload = _trace_test_2_payload()
    scene = TraceTest2GlScene(
        world=pygame.Rect(0, 0, 640, 360),
        payload=payload,
        practice_mode=True,
    )

    plan = _build_trace_test_2_scene_plan(scene)

    assert plan.camera is not None
    assert plan.overlay_primitives == ()
    assert len(plan.asset_instances) > 0
    assert len(plan.asset_instances) < plan.entity_count
    assert plan.entity_count == len(payload.aircraft) + len(plan.asset_instances)
    assert plan.entity_count > len(payload.aircraft)
    assert {"plane_blue", "plane_green", "plane_red", "plane_yellow"} >= set(plan.asset_ids)


def test_trace_test_1_draw_scene_renders_projected_marker_geometry(monkeypatch) -> None:
    renderer = _build_pipeline_probe_renderer()
    scene = TraceTest1GlScene(world=pygame.Rect(0, 0, 640, 360), payload=_trace_test_1_payload())
    scene_plan = _build_trace_test_1_scene_plan(scene)

    monkeypatch.setattr(
        renderer,
        "_render_scene_plan",
        lambda *, scene_plan: (_ for _ in ()).throw(AssertionError("unexpected world scene render")),
    )

    entity_count = renderer._draw_trace_test_1_scene(scene=scene, scene_plan=scene_plan)

    assert entity_count == scene_plan.entity_count
    assert renderer._batch.triangles


def test_trace_test_2_draw_scene_renders_projected_marker_geometry(monkeypatch) -> None:
    renderer = _build_pipeline_probe_renderer()
    scene = TraceTest2GlScene(
        world=pygame.Rect(0, 0, 640, 360),
        payload=_trace_test_2_payload(),
        practice_mode=False,
    )
    scene_plan = _build_trace_test_2_scene_plan(scene)

    monkeypatch.setattr(
        renderer,
        "_render_scene_plan",
        lambda *, scene_plan: (_ for _ in ()).throw(AssertionError("unexpected world scene render")),
    )

    entity_count = renderer._draw_trace_test_2_scene(scene=scene, scene_plan=scene_plan)

    assert entity_count == scene_plan.entity_count
    assert renderer._batch.triangles


def test_trace_test_1_draw_scene_marker_geometry_changes_with_progress() -> None:
    early_renderer = _build_pipeline_probe_renderer()
    early_scene = TraceTest1GlScene(
        world=pygame.Rect(0, 0, 640, 360),
        payload=_trace_test_1_payload(progress=0.18),
    )
    early_plan = _build_trace_test_1_scene_plan(early_scene)
    early_renderer._draw_trace_test_1_scene(scene=early_scene, scene_plan=early_plan)

    late_renderer = _build_pipeline_probe_renderer()
    late_scene = TraceTest1GlScene(
        world=pygame.Rect(0, 0, 640, 360),
        payload=_trace_test_1_payload(progress=0.82),
    )
    late_plan = _build_trace_test_1_scene_plan(late_scene)
    late_renderer._draw_trace_test_1_scene(scene=late_scene, scene_plan=late_plan)

    assert _triangle_signature(early_renderer._batch) != _triangle_signature(late_renderer._batch)


def test_trace_test_2_draw_scene_marker_geometry_changes_with_progress() -> None:
    payload = _trace_test_2_payload()

    early_renderer = _build_pipeline_probe_renderer()
    early_scene = TraceTest2GlScene(
        world=pygame.Rect(0, 0, 640, 360),
        payload=replace(payload, observe_progress=0.18),
        practice_mode=False,
    )
    early_plan = _build_trace_test_2_scene_plan(early_scene)
    early_renderer._draw_trace_test_2_scene(scene=early_scene, scene_plan=early_plan)

    late_renderer = _build_pipeline_probe_renderer()
    late_scene = TraceTest2GlScene(
        world=pygame.Rect(0, 0, 640, 360),
        payload=replace(payload, observe_progress=0.82),
        practice_mode=False,
    )
    late_plan = _build_trace_test_2_scene_plan(late_scene)
    late_renderer._draw_trace_test_2_scene(scene=late_scene, scene_plan=late_plan)

    assert _triangle_signature(early_renderer._batch) != _triangle_signature(late_renderer._batch)


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
