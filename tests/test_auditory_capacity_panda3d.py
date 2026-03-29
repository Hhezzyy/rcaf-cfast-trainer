from __future__ import annotations

from dataclasses import replace

import pygame
import pytest

from cfast_trainer.app import (
    App,
    CognitiveTestScreen,
    MenuItem,
    MenuScreen,
    _AuditoryPandaRequirementState,
)
from cfast_trainer.auditory_capacity import (
    AUDITORY_GATE_RETIRE_X_NORM,
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
from cfast_trainer.cognitive_core import Phase


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


def _payload_with_gate(*, x_norm: float, gate_id: int = 401) -> AuditoryCapacityPayload:
    clock = _FakeClock()
    engine = build_auditory_capacity_test(clock=clock, seed=17, difficulty=0.58)
    engine.start_practice()
    clock.advance(0.2)
    engine.update()
    payload = engine.snapshot().payload
    assert payload is not None
    return replace(
        payload,
        ball_x=0.0,
        ball_y=0.0,
        gates=(
            AuditoryCapacityGate(
                gate_id=gate_id,
                x_norm=float(x_norm),
                y_norm=0.0,
                color="RED",
                shape="CIRCLE",
                aperture_norm=0.18,
            ),
        ),
    )


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


def test_gate_ahead_distance_mapping_matches_arrival_expectations() -> None:
    spawned = AuditoryCapacityPanda3DRenderer._gate_ahead_distance_for_x_norm(
        AUDITORY_GATE_SPAWN_X_NORM
    )
    arrival = AuditoryCapacityPanda3DRenderer._gate_ahead_distance_for_x_norm(0.0)
    slightly_past = AuditoryCapacityPanda3DRenderer._gate_ahead_distance_for_x_norm(-0.20)
    retired = AuditoryCapacityPanda3DRenderer._gate_ahead_distance_for_x_norm(
        AUDITORY_GATE_RETIRE_X_NORM
    )

    assert spawned == pytest.approx(38.0)
    assert arrival == pytest.approx(0.8)
    assert -0.8 <= slightly_past <= 0.3
    assert slightly_past < arrival
    assert retired == pytest.approx(-6.0)


def test_newly_spawned_gate_stays_far_down_tunnel_and_fades_in() -> None:
    if not panda3d_auditory_rendering_available():
        pytest.skip("Panda3D unavailable")

    renderer = AuditoryCapacityPanda3DRenderer(size=(640, 360))
    try:
        payload = _payload_with_gate(x_norm=AUDITORY_GATE_SPAWN_X_NORM, gate_id=402)
        renderer._travel_offset = 8.0
        renderer._update_ball(payload=payload)
        renderer._update_gates(payload=payload)

        assert 402 in renderer._gate_nodes
        gate_np = renderer._gate_nodes[402]
        gate_pos = gate_np.getPos()
        ball_pos = renderer._ball_root.getPos()
        alpha = float(gate_np.getColorScale()[3])

        assert float(gate_pos[1] - ball_pos[1]) >= 24.0
        assert alpha <= 0.35
    finally:
        renderer.close()


def test_gate_near_arrival_stays_visible_in_front_of_camera() -> None:
    if not panda3d_auditory_rendering_available():
        pytest.skip("Panda3D unavailable")

    renderer = AuditoryCapacityPanda3DRenderer(size=(640, 360))
    try:
        payload = _payload_with_gate(x_norm=0.0)
        renderer._travel_offset = 12.0
        renderer._update_ball(payload=payload)
        renderer._update_gates(payload=payload)

        assert 401 in renderer._gate_nodes
        gate_pos = renderer._gate_nodes[401].getPos()
        ball_pos = renderer._ball_root.getPos()
        cam_pos = renderer._base.cam.getPos()

        assert float(gate_pos[1]) >= float(cam_pos[1])
        assert -1.5 <= float(gate_pos[1] - ball_pos[1]) <= 2.0
    finally:
        renderer.close()


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


def test_tunnel_geometry_backfills_wrap_zone_and_extends_past_path_end() -> None:
    if not panda3d_auditory_rendering_available():
        pytest.skip("Panda3D unavailable")

    renderer = AuditoryCapacityPanda3DRenderer(size=(640, 360))
    try:
        assert renderer._tube_geometry_start_distance <= -30.0
        assert renderer._tube_geometry_end_distance >= (renderer._span + 20.0)
        assert renderer._travel_wrap_threshold < (renderer._span - renderer._ball_anchor_distance)
    finally:
        renderer.close()


def test_travel_wrap_preserves_modulo_position_before_visible_seam() -> None:
    if not panda3d_auditory_rendering_available():
        pytest.skip("Panda3D unavailable")

    renderer = AuditoryCapacityPanda3DRenderer(size=(640, 360))
    try:
        renderer._travel_offset = renderer._travel_wrap_threshold - 0.05
        expected_ball_distance = (
            renderer._ball_anchor_distance + renderer._travel_offset + (0.05 * 5.2)
        )

        renderer._advance_travel_offset(0.05)
        wrapped_ball_distance = renderer._ball_anchor_distance + renderer._travel_offset

        assert renderer._travel_offset < 0.0
        assert wrapped_ball_distance < 0.0
        assert (wrapped_ball_distance % renderer._span) == pytest.approx(
            expected_ball_distance % renderer._span
        )
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
