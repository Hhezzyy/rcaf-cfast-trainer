from __future__ import annotations

import sys
from dataclasses import replace
from importlib.machinery import ModuleSpec
from types import ModuleType, SimpleNamespace

import pygame
import pytest

if "moderngl" not in sys.modules:
    moderngl_stub = ModuleType("moderngl")
    moderngl_stub.__spec__ = ModuleSpec("moderngl", loader=None)
    sys.modules["moderngl"] = moderngl_stub

from cfast_trainer.app import (
    App,
    CognitiveTestScreen,
    MenuItem,
    MenuScreen,
    _AuditoryPandaRequirementState,
)
from cfast_trainer.auditory_capacity import (
    AUDITORY_GATE_SPAWN_X_NORM,
    AuditoryCapacityGate,
    AuditoryCapacityPayload,
    build_auditory_capacity_test,
)
from cfast_trainer.auditory_capacity_panda3d import (
    AuditoryCapacityPanda3DRenderer,
    _tube_center_at_distance,
    _tube_frame,
    panda3d_auditory_rendering_available,
)
from cfast_trainer.auditory_capacity_view import (
    BALL_FORWARD_IDLE_NORM,
    GATE_DEPTH_SLOTS_NORM,
    gate_distance_from_x_norm,
    run_travel_distance,
)
from cfast_trainer.cognitive_core import Phase
from cfast_trainer.gl_scenes import AuditoryGlScene
from cfast_trainer.modern_gl_renderer import (
    ModernSceneRenderer,
    _build_auditory_scene_plan,
    _rapid_tracking_static_asset_library,
)


class _FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class _ExplodingRenderer:
    def __init__(self, *, size: tuple[int, int]) -> None:
        self.size = tuple(size)

    def render(self, **_: object) -> pygame.Surface:
        raise RuntimeError("boom")

    def close(self) -> None:
        return None


@pytest.fixture
def pygame_headless(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    pygame.quit()
    pygame.init()
    yield
    pygame.quit()


def _build_screen() -> tuple[App, CognitiveTestScreen]:
    display = pygame.display.set_mode((960, 540))
    font = pygame.font.Font(None, 36)
    app = App(surface=display, font=font, opengl_enabled=True)
    app.set_surface(pygame.Surface(display.get_size(), pygame.SRCALPHA))
    app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))

    clock = _FakeClock()
    screen = CognitiveTestScreen(
        app,
        engine_factory=lambda: build_auditory_capacity_test(
            clock=clock,
            seed=17,
            difficulty=0.58,
        ),
        test_code="auditory_capacity",
    )
    app.push(screen)
    screen._engine.start_practice()
    clock.advance(0.2)
    screen._engine.update()
    return app, screen


def _mark_auditory_panda_ready(screen: CognitiveTestScreen) -> None:
    screen._auditory_panda_requirement = _AuditoryPandaRequirementState(
        checked=True,
        ready=True,
    )
    screen._auditory_panda_failed = False


def _payload_with_gate(
    *,
    x_norm: float,
    gate_id: int = 401,
    slot_index: int | None = None,
    ball_forward_norm: float = BALL_FORWARD_IDLE_NORM,
    phase_elapsed_s: float | None = None,
    presentation_travel_distance: float | None = None,
) -> AuditoryCapacityPayload:
    clock = _FakeClock()
    engine = build_auditory_capacity_test(clock=clock, seed=17, difficulty=0.58)
    engine.start_practice()
    clock.advance(0.2)
    engine.update()
    payload = engine.snapshot().payload
    assert payload is not None
    elapsed = float(payload.phase_elapsed_s if phase_elapsed_s is None else phase_elapsed_s)
    travel_distance = (
        run_travel_distance(
            session_seed=int(payload.session_seed),
            phase_elapsed_s=elapsed,
        )
        if presentation_travel_distance is None
        else float(presentation_travel_distance)
    )
    return replace(
        payload,
        ball_x=0.0,
        ball_y=0.0,
        phase_elapsed_s=elapsed,
        presentation_travel_distance=float(travel_distance),
        ball_forward_norm=float(ball_forward_norm),
        gates=(
            AuditoryCapacityGate(
                gate_id=gate_id,
                x_norm=float(x_norm),
                y_norm=0.0,
                color="RED",
                shape="CIRCLE",
                aperture_norm=0.18,
                visual_slot_index=slot_index,
            ),
        ),
    )


def _stub_modern_auditory_renderer(*, size: tuple[int, int]) -> ModernSceneRenderer:
    renderer = ModernSceneRenderer.__new__(ModernSceneRenderer)
    renderer._win_w = int(size[0])
    renderer._win_h = int(size[1])
    renderer._scene_assets = _rapid_tracking_static_asset_library()
    renderer._batch = SimpleNamespace(triangles=[], scene_triangles=[], textured=[])
    renderer._last_rt_world_debug = {}
    return renderer


def test_panda3d_auditory_rendering_disabled_for_dummy_video(monkeypatch) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.delenv("CFAST_AUDITORY_RENDERER", raising=False)

    assert panda3d_auditory_rendering_available() is False


def test_panda3d_auditory_rendering_ignores_non_panda_preference(monkeypatch) -> None:
    monkeypatch.setenv("CFAST_AUDITORY_RENDERER", "pygame")
    monkeypatch.delenv("SDL_VIDEODRIVER", raising=False)
    monkeypatch.setattr(
        "cfast_trainer.auditory_capacity_panda3d.importlib.util.find_spec",
        lambda _name: object(),
    )

    assert panda3d_auditory_rendering_available() is True


def test_panda3d_auditory_rendering_handles_missing_direct_package(monkeypatch) -> None:
    monkeypatch.delenv("SDL_VIDEODRIVER", raising=False)
    monkeypatch.setattr(
        "cfast_trainer.auditory_capacity_panda3d.importlib.util.find_spec",
        lambda _name: (_ for _ in ()).throw(ModuleNotFoundError("direct")),
    )

    assert panda3d_auditory_rendering_available() is False


def test_auditory_screen_queues_gl_runtime_when_opengl_enabled(
    pygame_headless,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, screen = _build_screen()
    _mark_auditory_panda_ready(screen)
    monkeypatch.setattr(screen, "_sync_auditory_audio", lambda **_: None)

    app.render()

    gl_scene = app.consume_gl_scene()
    assert gl_scene is not None
    assert type(gl_scene).__name__ == "AuditoryGlScene"
    assert gl_scene.payload is not None
    assert gl_scene.payload.session_seed == 17
    assert gl_scene.payload.segment_label == "Practice"
    assert gl_scene.world.x > 0
    assert gl_scene.world.y > 0
    probe = (gl_scene.world.centerx, gl_scene.world.bottom - 40)
    assert app.surface.get_at(probe).a == 0
    assert app.surface.get_at((gl_scene.world.x, gl_scene.world.y)).a > 0
    assert screen._auditory_panda_renderer is None


def test_auditory_screen_bypasses_legacy_panda_renderer_when_gl_standard(
    pygame_headless,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, screen = _build_screen()
    _mark_auditory_panda_ready(screen)
    monkeypatch.setattr(screen, "_sync_auditory_audio", lambda **_: None)
    monkeypatch.setattr(
        screen,
        "_get_auditory_panda_renderer",
        lambda **_: _ExplodingRenderer(size=(320, 200)),
    )

    app.render()

    gl_scene = app.consume_gl_scene()
    assert gl_scene is not None
    assert type(gl_scene).__name__ == "AuditoryGlScene"
    assert screen._auditory_panda_requirement.checked is True
    assert screen._auditory_panda_requirement.ready is True


def test_auditory_world_frame_keeps_gl_viewport_transparent_in_freeze_mode(
    pygame_headless,
) -> None:
    app, screen = _build_screen()
    _mark_auditory_panda_ready(screen)
    payload = screen._engine.snapshot().payload
    assert payload is not None

    live_frame = screen._render_auditory_capacity_world_frame(
        size=(320, 180),
        phase=Phase.PRACTICE,
        payload=payload,
        time_remaining_s=24.0,
        time_fill_ratio=0.5,
        freeze_mode=None,
        gl_scene_world=pygame.Rect(24, 36, 320, 180),
    )
    frozen_frame = screen._render_auditory_capacity_world_frame(
        size=(320, 180),
        phase=Phase.PRACTICE,
        payload=payload,
        time_remaining_s=24.0,
        time_fill_ratio=0.5,
        freeze_mode="pause",
        gl_scene_world=pygame.Rect(24, 36, 320, 180),
    )

    probe = (live_frame.get_width() // 2, live_frame.get_height() - 40)
    assert live_frame.get_at(probe).a == 0
    assert frozen_frame.get_at(probe).a == 0
    assert live_frame.get_at((0, 0)).a > 0
    assert frozen_frame.get_at((0, 0)).a > 0


def test_gate_position_comes_from_live_x_norm_not_visual_slot_index() -> None:
    if not panda3d_auditory_rendering_available():
        pytest.skip("Panda3D unavailable")

    renderer = AuditoryCapacityPanda3DRenderer(size=(640, 360))
    try:
        payload_a = _payload_with_gate(
            x_norm=0.85,
            gate_id=402,
            slot_index=0,
        )
        payload_b = replace(payload_a, slot_index=len(GATE_DEPTH_SLOTS_NORM) - 1)
        travel_distance = float(payload_a.presentation_travel_distance)

        renderer._update_ball(payload=payload_a)
        renderer._update_gates(payload=payload_a)
        gate_pos_a = renderer._gate_nodes[402].getPos()

        renderer._update_ball(payload=payload_b)
        renderer._update_gates(payload=payload_b)
        gate_pos_b = renderer._gate_nodes[402].getPos()

        expected_distance = gate_distance_from_x_norm(
            0.85,
            travel_distance=travel_distance,
            spawn_x_norm=AUDITORY_GATE_SPAWN_X_NORM,
            player_x_norm=renderer._GATE_PLAYER_X_NORM,
            retire_x_norm=renderer._GATE_RETIRE_X_NORM,
        )

        assert tuple(gate_pos_b) == pytest.approx(tuple(gate_pos_a))
        assert float(gate_pos_a[1]) == pytest.approx(expected_distance, abs=0.01)
    finally:
        renderer.close()


def test_gate_moves_closer_as_live_x_norm_advances() -> None:
    if not panda3d_auditory_rendering_available():
        pytest.skip("Panda3D unavailable")

    renderer = AuditoryCapacityPanda3DRenderer(size=(640, 360))
    try:
        payload_a = _payload_with_gate(
            x_norm=1.10,
            slot_index=len(GATE_DEPTH_SLOTS_NORM) - 2,
            ball_forward_norm=0.20,
        )
        payload_b = replace(payload_a, x_norm=0.25)

        renderer._update_ball(payload=payload_a)
        renderer._update_gates(payload=payload_a)
        gate_pos_a = renderer._gate_nodes[401].getPos()
        ball_pos_a = renderer._ball_root.getPos()

        renderer._update_ball(payload=payload_b)
        renderer._update_gates(payload=payload_b)
        gate_pos_b = renderer._gate_nodes[401].getPos()
        ball_pos_b = renderer._ball_root.getPos()

        assert float(gate_pos_b[1]) < float(gate_pos_a[1])
        assert tuple(ball_pos_b) == pytest.approx(tuple(ball_pos_a))
    finally:
        renderer.close()


def test_modern_gl_auditory_scene_keeps_front_edge_gate_geometry_when_near_ball() -> None:
    payload = _payload_with_gate(
        x_norm=0.01,
        slot_index=len(GATE_DEPTH_SLOTS_NORM) - 1,
        ball_forward_norm=0.20,
    )
    baseline_scene = AuditoryGlScene(
        world=pygame.Rect(0, 0, 960, 540),
        payload=replace(payload, gates=()),
        time_remaining_s=10.0,
        time_fill_ratio=0.5,
    )
    gated_scene = AuditoryGlScene(
        world=pygame.Rect(0, 0, 960, 540),
        payload=payload,
        time_remaining_s=10.0,
        time_fill_ratio=0.5,
    )

    baseline_renderer = _stub_modern_auditory_renderer(size=(960, 540))
    baseline_renderer._render_scene_plan(scene_plan=_build_auditory_scene_plan(baseline_scene))
    baseline_triangles = len(baseline_renderer._batch.triangles)

    gated_renderer = _stub_modern_auditory_renderer(size=(960, 540))
    gated_renderer._render_scene_plan(scene_plan=_build_auditory_scene_plan(gated_scene))

    assert len(gated_renderer._batch.triangles) > baseline_triangles


def test_renderer_prefers_repo_auditory_assets_when_present() -> None:
    if not panda3d_auditory_rendering_available():
        pytest.skip("Panda3D unavailable")

    renderer = AuditoryCapacityPanda3DRenderer(size=(640, 360))
    try:
        assert set(renderer.loaded_asset_ids) >= {
            "auditory_ball",
            "auditory_tunnel_rib",
            "auditory_tunnel_segment",
        }
        assert "auditory_ball" not in set(renderer.fallback_asset_ids)
        assert "auditory_tunnel_segment" not in set(renderer.fallback_asset_ids)
        assert "auditory_tunnel_rib" not in set(renderer.fallback_asset_ids)
        assert renderer._tube_root.getNumChildren() >= 40
        assert renderer._tube_root.findAllMatches("**/+GeomNode").getNumPaths() >= 40
    finally:
        renderer.close()


def test_camera_is_stable_across_ball_offsets() -> None:
    if not panda3d_auditory_rendering_available():
        pytest.skip("Panda3D unavailable")

    renderer = AuditoryCapacityPanda3DRenderer(size=(640, 360))
    try:
        centered = replace(_payload_with_gate(x_norm=0.85, slot_index=7), ball_x=0.0, ball_y=0.0)
        offset = replace(centered, ball_x=0.82, ball_y=-0.60)

        renderer._update_ball(payload=centered)
        cam_pos_a = renderer._base.cam.getPos()
        renderer._update_ball(payload=offset)
        cam_pos_b = renderer._base.cam.getPos()

        assert tuple(cam_pos_b) == pytest.approx(tuple(cam_pos_a))
    finally:
        renderer.close()


def test_camera_advances_with_presentation_travel_progress() -> None:
    if not panda3d_auditory_rendering_available():
        pytest.skip("Panda3D unavailable")

    renderer = AuditoryCapacityPanda3DRenderer(size=(640, 360))
    try:
        near_payload = replace(
            _payload_with_gate(x_norm=0.85, slot_index=7),
            ball_forward_norm=float(BALL_FORWARD_IDLE_NORM),
        )
        far_payload = replace(
            near_payload,
            phase_elapsed_s=9.4,
            presentation_travel_distance=run_travel_distance(
                session_seed=int(near_payload.session_seed),
                phase_elapsed_s=9.4,
            ),
            ball_forward_norm=0.74,
        )

        renderer._update_ball(payload=near_payload)
        cam_pos_a = renderer._base.cam.getPos()
        renderer._update_ball(payload=far_payload)
        cam_pos_b = renderer._base.cam.getPos()

        assert float(cam_pos_b[1]) > float(cam_pos_a[1])
        assert float(cam_pos_b[0]) > float(cam_pos_a[0])
    finally:
        renderer.close()


def test_travel_offset_no_longer_changes_ball_or_gate_positions() -> None:
    if not panda3d_auditory_rendering_available():
        pytest.skip("Panda3D unavailable")

    renderer = AuditoryCapacityPanda3DRenderer(size=(640, 360))
    try:
        payload = _payload_with_gate(
            x_norm=0.70,
            slot_index=len(GATE_DEPTH_SLOTS_NORM) - 1,
            ball_forward_norm=0.44,
        )

        renderer._travel_offset = 0.0
        renderer._update_ball(payload=payload)
        renderer._update_gates(payload=payload)
        gate_pos_a = renderer._gate_nodes[401].getPos()
        ball_pos_a = renderer._ball_root.getPos()

        renderer._travel_offset = 87.0
        renderer._update_ball(payload=payload)
        renderer._update_gates(payload=payload)
        gate_pos_b = renderer._gate_nodes[401].getPos()
        ball_pos_b = renderer._ball_root.getPos()

        assert tuple(gate_pos_b) == pytest.approx(tuple(gate_pos_a))
        assert tuple(ball_pos_b) == pytest.approx(tuple(ball_pos_a))
    finally:
        renderer.close()


def test_looped_tube_spline_is_continuous_across_wrap() -> None:
    if not panda3d_auditory_rendering_available():
        pytest.skip("Panda3D unavailable")

    renderer = AuditoryCapacityPanda3DRenderer(size=(640, 360))
    try:
        span = float(renderer._span)
        before_center = _tube_center_at_distance(span - 0.05, span=span)
        after_center = _tube_center_at_distance(0.0, span=span)
        before_frame = _tube_frame(span - 0.05, span=span)
        after_frame = _tube_frame(0.0, span=span)

        assert before_center[0] == pytest.approx(after_center[0], abs=0.20)
        assert before_center[1] == pytest.approx(after_center[1], abs=0.20)
        assert before_frame[1][0] == pytest.approx(after_frame[1][0], abs=0.08)
        assert before_frame[1][1] == pytest.approx(after_frame[1][1], abs=0.08)
        assert before_frame[1][2] == pytest.approx(after_frame[1][2], abs=0.08)
    finally:
        renderer.close()


def test_panda_ball_uses_visual_feedback_color_without_changing_logical_state() -> None:
    if not panda3d_auditory_rendering_available():
        pytest.skip("Panda3D unavailable")

    renderer = AuditoryCapacityPanda3DRenderer(size=(640, 360))
    try:
        payload = replace(
            _payload_with_gate(x_norm=0.8),
            ball_color="GREEN",
            ball_visual_color="YELLOW",
            ball_visual_strength=1.0,
            ball_contact_ratio=0.0,
        )
        renderer._update_ball(payload=payload)
        rgba = renderer._ball_root.getColor()

        assert rgba[0] > 0.85
        assert rgba[1] > 0.70
        assert rgba[2] < 0.55
    finally:
        renderer.close()
