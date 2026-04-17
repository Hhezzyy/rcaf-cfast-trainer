from __future__ import annotations

from dataclasses import replace
import math

import pygame
import pytest

from cfast_trainer.app import App, CognitiveTestScreen, MenuItem, MenuScreen
from cfast_trainer.rapid_tracking import build_rapid_tracking_test
from cfast_trainer.rapid_tracking_panda3d import (
    _RapidTrackingDecoy,
    RapidTrackingPanda3DRenderer,
    panda3d_rapid_tracking_rendering_available,
)
from cfast_trainer.rapid_tracking_view import RapidTrackingCameraRigState, camera_pose_compat
from cfast_trainer.rapid_tracking_view import (
    RAPID_TRACKING_PITCH_MAX_DEG,
    RAPID_TRACKING_PITCH_MIN_DEG,
)


def _angle_delta_deg(a: float, b: float) -> float:
    return ((float(a) - float(b) + 180.0) % 360.0) - 180.0


class _FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _run_probe(*, seed: int = 551) -> dict[str, object]:
    pygame.init()
    screen: CognitiveTestScreen | None = None
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        app = App(surface=surface, font=font, opengl_enabled=True)
        app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))
        clock = _FakeClock()
        screen = CognitiveTestScreen(
            app,
            engine_factory=lambda: build_rapid_tracking_test(
                clock=clock,
                seed=int(seed),
                difficulty=0.63,
            ),
            test_code="rapid_tracking",
        )
        app.push(screen)
        screen._engine.start_scored()
        clock.advance(0.6)
        screen._engine.update()
        app.render()
        scene = app.consume_gl_scene()
        assert scene is not None
        payload = scene.payload
        return {
            "scene_type": type(scene).__name__,
            "session_seed": int(payload.session_seed),
            "target_kind": str(payload.target_kind),
            "target_variant": str(payload.target_variant),
            "target_world": (
                round(float(payload.target_world_x), 6),
                round(float(payload.target_world_y), 6),
            ),
            "target_cover_state": str(payload.target_cover_state),
        }
    finally:
        if screen is not None:
            close = getattr(screen, "close", None)
            if callable(close):
                close()
        pygame.quit()


def _camera_rig(
    *,
    cam_world_x: float = 0.0,
    cam_world_y: float = 0.0,
    cam_world_z: float = 12.0,
    view_heading_deg: float = 0.0,
    view_pitch_deg: float = 0.0,
) -> RapidTrackingCameraRigState:
    return RapidTrackingCameraRigState(
        cam_world_x=float(cam_world_x),
        cam_world_y=float(cam_world_y),
        cam_world_z=float(cam_world_z),
        carrier_heading_deg=float(view_heading_deg),
        heading_deg=float(view_heading_deg),
        pitch_deg=float(view_pitch_deg),
        view_heading_deg=float(view_heading_deg),
        view_pitch_deg=float(view_pitch_deg),
        roll_deg=0.0,
        fov_deg=60.0,
        orbit_weight=0.0,
        orbit_radius=0.0,
        altitude_agl=12.0,
        neutral_heading_deg=float(view_heading_deg),
        neutral_pitch_deg=float(view_pitch_deg),
    )


def _unit_renderer() -> RapidTrackingPanda3DRenderer:
    renderer = RapidTrackingPanda3DRenderer.__new__(RapidTrackingPanda3DRenderer)
    renderer._size = (960, 540)
    renderer._terrain_height = lambda _x, _y: 0.0
    return renderer


class _StubNode:
    def __init__(self) -> None:
        self.visible = False
        self.alpha = 0.0
        self.pos = (0.0, 0.0, 0.0)
        self.hpr = (0.0, 0.0, 0.0)

    def hide(self) -> None:
        self.visible = False

    def show(self) -> None:
        self.visible = True

    def setAlphaScale(self, value: float) -> None:
        self.alpha = float(value)

    def setPos(self, *args: float) -> None:
        if len(args) == 1:
            x, y, z = args[0]
        else:
            x, y, z = args
        self.pos = (float(x), float(y), float(z))

    def setHpr(self, *args: float) -> None:
        if len(args) == 1:
            heading, pitch, roll = args[0]
        else:
            heading, pitch, roll = args
        self.hpr = (float(heading), float(pitch), float(roll))


def test_panda3d_rapid_tracking_rendering_disabled_for_dummy_video(monkeypatch) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.delenv("CFAST_RAPID_TRACKING_RENDERER", raising=False)

    assert panda3d_rapid_tracking_rendering_available() is False


def test_panda3d_rapid_tracking_rendering_ignores_non_panda_preference(monkeypatch) -> None:
    monkeypatch.setenv("CFAST_RAPID_TRACKING_RENDERER", "pygame")
    monkeypatch.delenv("SDL_VIDEODRIVER", raising=False)
    monkeypatch.setattr(
        "cfast_trainer.rapid_tracking_panda3d.importlib.util.find_spec",
        lambda _name: object(),
    )

    assert panda3d_rapid_tracking_rendering_available() is True


def test_camera_rig_descends_from_high_orbit_into_low_run_path() -> None:
    start = RapidTrackingPanda3DRenderer._camera_rig_state(
        elapsed_s=0.0,
        progress=0.0,
        camera_yaw_deg=None,
        camera_pitch_deg=None,
        zoom=0.0,
        target_kind="soldier",
        turbulence_strength=0.6,
    )
    end = RapidTrackingPanda3DRenderer._camera_rig_state(
        elapsed_s=90.0,
        progress=1.0,
        camera_yaw_deg=None,
        camera_pitch_deg=None,
        zoom=0.0,
        target_kind="truck",
        turbulence_strength=0.6,
    )

    assert start.altitude_agl > end.altitude_agl
    assert start.altitude_agl >= 20.0
    assert end.altitude_agl <= 11.0
    assert start.orbit_weight > 0.8
    assert end.orbit_weight < 0.1


def test_camera_rig_uses_direct_camera_pose_without_recentering() -> None:
    rig = RapidTrackingPanda3DRenderer._camera_rig_state(
        elapsed_s=20.0,
        seed=551,
        progress=0.45,
        camera_yaw_deg=132.0,
        camera_pitch_deg=-18.5,
        zoom=0.0,
        target_kind="truck",
        target_world_x=42.0,
        target_world_y=96.0,
        focus_world_x=8.0,
        focus_world_y=78.0,
        turbulence_strength=0.6,
    )

    assert rig.heading_deg == pytest.approx(132.0)
    assert rig.pitch_deg == pytest.approx(-18.5)


def test_panda_camera_rig_clamps_requested_and_shaken_pitch_to_shared_bounds() -> None:
    steep_up = RapidTrackingPanda3DRenderer._camera_rig_state(
        elapsed_s=20.0,
        seed=551,
        progress=0.45,
        camera_yaw_deg=132.0,
        camera_pitch_deg=88.0,
        zoom=0.0,
        target_kind="truck",
        target_world_x=42.0,
        target_world_y=96.0,
        focus_world_x=8.0,
        focus_world_y=78.0,
        turbulence_strength=1.8,
    )
    steep_down = RapidTrackingPanda3DRenderer._camera_rig_state(
        elapsed_s=20.0,
        seed=551,
        progress=0.45,
        camera_yaw_deg=132.0,
        camera_pitch_deg=-88.0,
        zoom=0.0,
        target_kind="truck",
        target_world_x=42.0,
        target_world_y=96.0,
        focus_world_x=8.0,
        focus_world_y=78.0,
        turbulence_strength=1.8,
    )

    assert steep_up.pitch_deg == pytest.approx(RAPID_TRACKING_PITCH_MAX_DEG)
    assert steep_down.pitch_deg == pytest.approx(RAPID_TRACKING_PITCH_MIN_DEG)
    assert RAPID_TRACKING_PITCH_MIN_DEG <= steep_up.view_pitch_deg <= RAPID_TRACKING_PITCH_MAX_DEG
    assert RAPID_TRACKING_PITCH_MIN_DEG <= steep_down.view_pitch_deg <= RAPID_TRACKING_PITCH_MAX_DEG


def test_camera_rig_uses_wide_default_fov_and_narrow_hold_zoom() -> None:
    wide = RapidTrackingPanda3DRenderer._camera_rig_state(
        elapsed_s=20.0,
        seed=551,
        progress=0.45,
        camera_yaw_deg=132.0,
        camera_pitch_deg=-18.5,
        zoom=0.0,
        target_kind="truck",
        target_world_x=42.0,
        target_world_y=96.0,
        focus_world_x=8.0,
        focus_world_y=78.0,
        turbulence_strength=0.6,
    )
    zoomed = RapidTrackingPanda3DRenderer._camera_rig_state(
        elapsed_s=20.0,
        seed=551,
        progress=0.45,
        camera_yaw_deg=132.0,
        camera_pitch_deg=-18.5,
        zoom=1.0,
        target_kind="truck",
        target_world_x=42.0,
        target_world_y=96.0,
        focus_world_x=8.0,
        focus_world_y=78.0,
        turbulence_strength=0.6,
    )

    assert 124.0 <= wide.fov_deg <= 136.0
    assert 36.0 <= zoomed.fov_deg <= 42.0


def test_camera_rig_allows_large_yaw_offsets_past_old_sweep_limit() -> None:
    rig = RapidTrackingPanda3DRenderer._camera_rig_state(
        elapsed_s=20.0,
        seed=551,
        progress=0.45,
        camera_yaw_deg=220.0,
        camera_pitch_deg=-8.0,
        zoom=0.0,
        target_kind="truck",
        target_world_x=42.0,
        target_world_y=96.0,
        focus_world_x=8.0,
        focus_world_y=78.0,
        turbulence_strength=0.6,
    )
    compat_x, _compat_y = camera_pose_compat(
        heading_deg=rig.heading_deg,
        pitch_deg=rig.pitch_deg,
        neutral_heading_deg=rig.neutral_heading_deg,
        neutral_pitch_deg=rig.neutral_pitch_deg,
    )

    assert abs(_angle_delta_deg(rig.heading_deg, rig.neutral_heading_deg)) > 66.0
    assert abs(compat_x) > 5.2


def test_camera_space_projection_maps_center_and_offscreen_right() -> None:
    center = RapidTrackingPanda3DRenderer._camera_space_to_viewport(
        cam_x=0.0,
        cam_y=10.0,
        cam_z=0.0,
        size=(800, 600),
        h_fov_deg=60.0,
        v_fov_deg=45.0,
    )
    right = RapidTrackingPanda3DRenderer._camera_space_to_viewport(
        cam_x=20.0,
        cam_y=10.0,
        cam_z=0.0,
        size=(800, 600),
        h_fov_deg=60.0,
        v_fov_deg=45.0,
    )

    assert center[0] == 400.0
    assert center[1] == 300.0
    assert center[2] is True
    assert center[3] is True
    assert right[0] > 800.0
    assert right[2] is False
    assert right[3] is True


def test_ground_routes_stay_axis_aligned_and_tanks_spin_in_place() -> None:
    side_a = RapidTrackingPanda3DRenderer._ground_route_pose(
        elapsed_s=8.0,
        phase=0.2,
        speed=0.36,
        lateral_bias=12.0,
        depth_bias=26.0,
        route="ground_side",
    )
    side_b = RapidTrackingPanda3DRenderer._ground_route_pose(
        elapsed_s=18.0,
        phase=0.2,
        speed=0.36,
        lateral_bias=12.0,
        depth_bias=26.0,
        route="ground_side",
    )
    convoy_a = RapidTrackingPanda3DRenderer._ground_route_pose(
        elapsed_s=8.0,
        phase=0.4,
        speed=0.34,
        lateral_bias=-18.0,
        depth_bias=34.0,
        route="ground_convoy",
    )
    convoy_b = RapidTrackingPanda3DRenderer._ground_route_pose(
        elapsed_s=18.0,
        phase=0.4,
        speed=0.34,
        lateral_bias=-18.0,
        depth_bias=34.0,
        route="ground_convoy",
    )
    tank_a = RapidTrackingPanda3DRenderer._ground_route_pose(
        elapsed_s=8.0,
        phase=0.55,
        speed=0.38,
        lateral_bias=-11.0,
        depth_bias=18.0,
        route="tank_hold",
        tank_spin=True,
    )
    tank_b = RapidTrackingPanda3DRenderer._ground_route_pose(
        elapsed_s=18.0,
        phase=0.55,
        speed=0.38,
        lateral_bias=-11.0,
        depth_bias=18.0,
        route="tank_hold",
        tank_spin=True,
    )

    assert side_a[1] == side_b[1]
    assert side_a[0] != side_b[0]
    assert convoy_a[0] == convoy_b[0]
    assert convoy_a[1] != convoy_b[1]
    assert tank_a[:2] == tank_b[:2]
    assert tank_a[2] != tank_b[2]


def test_airborne_apparent_hpr_ignores_camera_only_motion_for_active_target() -> None:
    renderer = _unit_renderer()

    hpr = renderer._airborne_apparent_hpr(
        cache_key=101,
        current_pos=(0.0, 120.0, 20.0),
        next_pos=(0.0, 120.0, 20.0),
        current_rig=_camera_rig(cam_world_x=0.0),
        next_rig=_camera_rig(cam_world_x=18.0),
        default_heading=37.0,
        pitch_flair=0.0,
        roll_flair=0.0,
        pitch_limit=18.0,
        roll_limit=30.0,
    )

    assert hpr[0] == pytest.approx(37.0)


def test_airborne_apparent_hpr_uses_scripted_default_heading_when_motion_is_nearly_zero() -> None:
    renderer = _unit_renderer()

    hpr = renderer._airborne_apparent_hpr(
        cache_key=102,
        current_pos=(0.0, 120.0, 20.0),
        next_pos=(0.0, 120.0, 20.0),
        current_rig=_camera_rig(),
        next_rig=_camera_rig(),
        default_heading=123.0,
        pitch_flair=0.0,
        roll_flair=0.0,
        pitch_limit=18.0,
        roll_limit=30.0,
    )

    assert hpr[0] == pytest.approx(123.0)


def test_airborne_apparent_hpr_uses_world_motion_heading_even_when_camera_turns() -> None:
    renderer = _unit_renderer()

    hpr = renderer._airborne_apparent_hpr(
        cache_key=103,
        current_pos=(0.0, 120.0, 20.0),
        next_pos=(18.0, 120.0, 20.0),
        current_rig=_camera_rig(cam_world_x=0.0, view_heading_deg=45.0),
        next_rig=_camera_rig(cam_world_x=18.0, view_heading_deg=225.0),
        default_heading=0.0,
        pitch_flair=0.0,
        roll_flair=0.0,
        pitch_limit=18.0,
        roll_limit=30.0,
    )

    assert abs(_angle_delta_deg(hpr[0], 90.0)) <= 1.0


def test_decoy_air_pose_uses_world_motion_heading_contract() -> None:
    renderer = _unit_renderer()
    decoy = _RapidTrackingDecoy(
        node=None,
        kind="plane",
        phase=0.35,
        speed=0.62,
        lateral_bias=16.0,
        depth_bias=18.0,
        altitude=20.0,
        route="air_cross",
    )

    current_pos, default_heading, pitch_flair, roll_flair = renderer._decoy_air_pose(
        decoy=decoy,
        elapsed_s=4.0,
    )
    next_pos, _next_heading, _next_pitch, _next_roll = renderer._decoy_air_pose(
        decoy=decoy,
        elapsed_s=4.0 + renderer._AIR_MOTION_SAMPLE_DT_S,
    )
    hpr = renderer._airborne_apparent_hpr(
        cache_key=201,
        current_pos=current_pos,
        next_pos=next_pos,
        current_rig=_camera_rig(cam_world_x=0.0),
        next_rig=_camera_rig(cam_world_x=24.0),
        default_heading=default_heading,
        pitch_flair=pitch_flair,
        roll_flair=roll_flair,
        pitch_limit=16.0,
        roll_limit=24.0,
    )
    expected_heading = math.degrees(
        math.atan2(
            float(next_pos[0] - current_pos[0]),
            float(next_pos[1] - current_pos[1]),
        )
    ) % 360.0

    assert abs(_angle_delta_deg(hpr[0], expected_heading)) <= 5.0


def test_update_target_hides_obscured_dynamic_target_and_reacquires_at_world_position() -> None:
    class _FakeClock:
        def __init__(self) -> None:
            self.t = 0.0

        def now(self) -> float:
            return self.t

    clock = _FakeClock()
    engine = build_rapid_tracking_test(clock=clock, seed=551, difficulty=0.63)
    engine.start_scored()
    payload = engine.snapshot().payload
    assert payload is not None

    renderer = _unit_renderer()
    node = _StubNode()
    renderer._active_target_node = None
    renderer._active_target_kind = ""
    renderer._active_scenery_target_node = None
    renderer._dynamic_target_nodes = [node]
    renderer._truck_pool = [node]
    renderer._soldier_pool = []
    renderer._helicopter_pool = []
    renderer._jet_pool = []
    renderer._selected_target_node = None
    renderer._selected_target_kind = ""
    renderer._selected_target_variant = ""
    renderer._scenery_nodes_by_kind = {
        "hangar": [],
        "tower": [],
        "truck": [],
        "soldiers": [],
    }
    renderer._ground_target_world_pos = lambda world_x, world_y, clearance: (
        float(world_x),
        float(world_y),
        float(clearance),
    )
    renderer._select_target_node_from_candidates = lambda **_kwargs: node

    visible_payload = replace(
        payload,
        target_kind="truck",
        target_variant="olive",
        target_world_x=5.0,
        target_world_y=90.0,
        target_vx=0.7,
        target_vy=0.0,
        target_visible=True,
    )
    hidden_payload = replace(
        visible_payload,
        target_world_x=7.0,
        target_visible=False,
    )
    reacquired_payload = replace(
        visible_payload,
        target_world_x=9.0,
        target_visible=True,
    )

    renderer._update_target(payload=visible_payload)
    assert node.visible is True
    assert node.alpha == pytest.approx(1.0)
    assert node.pos == pytest.approx((5.0, 90.0, 0.34))
    assert abs(_angle_delta_deg(node.hpr[0], 90.0)) <= 1e-6

    renderer._update_target(payload=hidden_payload)
    assert node.visible is False
    assert node.alpha == pytest.approx(0.0)
    assert node.pos == pytest.approx((7.0, 90.0, 0.34))
    assert renderer._active_target_node is node

    renderer._update_target(payload=reacquired_payload)
    assert node.visible is True
    assert node.alpha == pytest.approx(1.0)
    assert node.pos == pytest.approx((9.0, 90.0, 0.34))


def test_world_span_is_expanded_for_large_scene() -> None:
    assert RapidTrackingPanda3DRenderer._WORLD_EXTENT_SCALE == 5.0
    assert RapidTrackingPanda3DRenderer._TERRAIN_HALF_SPAN == 1300.0


def test_seeded_road_segment_ids_cover_vehicle_network() -> None:
    assert RapidTrackingPanda3DRenderer._seeded_road_segment_ids() == ()


def test_seeded_road_segments_are_built_from_multiple_pieces() -> None:
    if not panda3d_rapid_tracking_rendering_available():
        pytest.skip("Panda3D unavailable")

    pygame.init()
    renderer = RapidTrackingPanda3DRenderer(size=(640, 360))
    try:
        renderer._ensure_seeded_layout(seed=551)
        assert renderer._road_segment_nodes
        assert len(renderer._road_segment_nodes) >= 8
        assert all(node.getNumChildren() >= 1 for node in renderer._road_segment_nodes)
        assert any(node.getNumChildren() >= 4 for node in renderer._road_segment_nodes)
    finally:
        renderer.close()
        pygame.quit()


def test_rapid_tracking_screen_queues_modern_gl_runtime() -> None:
    probe = _run_probe(seed=551)

    assert probe["scene_type"] == "RapidTrackingGlScene"
    assert probe["session_seed"] == 551
    assert probe["target_kind"] in {"soldier", "building", "truck", "helicopter", "jet"}
    assert probe["target_cover_state"] in {"open", "building", "terrain"}


def test_same_seed_reproduces_seeded_gl_payload() -> None:
    first = _run_probe(seed=808)
    second = _run_probe(seed=808)

    assert first == second


def test_different_seed_changes_seeded_gl_payload() -> None:
    first = _run_probe(seed=808)
    second = _run_probe(seed=909)

    assert first["scene_type"] == second["scene_type"] == "RapidTrackingGlScene"
    assert first["session_seed"] != second["session_seed"]
    assert first["target_world"] != second["target_world"]


def test_target_world_position_does_not_follow_camera_pan() -> None:
    if not panda3d_rapid_tracking_rendering_available():
        pytest.skip("Panda3D unavailable")

    class _FakeClock:
        def __init__(self) -> None:
            self.t = 0.0

        def now(self) -> float:
            return self.t

        def advance(self, dt: float) -> None:
            self.t += dt

    pygame.init()
    renderer = RapidTrackingPanda3DRenderer(size=(640, 360))
    try:
        clock = _FakeClock()
        engine = build_rapid_tracking_test(clock=clock, seed=551, difficulty=0.63)
        engine.start_scored()
        clock.advance(0.6)
        engine.update()
        payload = engine.snapshot().payload
        assert payload is not None
        assert payload.target_kind == "soldier"

        renderer.render(payload=payload)
        node = renderer._active_target_node
        assert node is not None
        first = node.getPos(renderer._base.render)

        panned_payload = replace(
            payload,
            camera_yaw_deg=float(payload.camera_yaw_deg) + 18.0,
            camera_pitch_deg=float(payload.camera_pitch_deg) - 6.0,
        )
        renderer.render(payload=panned_payload)
        second_node = renderer._active_target_node
        assert second_node is not None
        second = second_node.getPos(renderer._base.render)

        assert second_node == node
        assert abs(float(second.x) - float(first.x)) < 1e-4
        assert abs(float(second.y) - float(first.y)) < 1e-4
        assert abs(float(second.z) - float(first.z)) < 1e-4
    finally:
        renderer.close()
        pygame.quit()


def test_runtime_panda_camera_keeps_turning_under_held_horizontal_input() -> None:
    if not panda3d_rapid_tracking_rendering_available():
        pytest.skip("Panda3D unavailable")

    class _FakeClock:
        def __init__(self) -> None:
            self.t = 0.0

        def now(self) -> float:
            return self.t

        def advance(self, dt: float) -> None:
            self.t += dt

    pygame.init()
    renderer = RapidTrackingPanda3DRenderer(size=(640, 360))
    try:
        clock = _FakeClock()
        engine = build_rapid_tracking_test(clock=clock, seed=551, difficulty=0.63)
        engine.start_scored()

        headings: list[float] = []
        camera_yaw_values: list[float] = []
        for _ in range(7):
            engine.set_control(horizontal=1.0, vertical=0.0)
            clock.advance(0.35)
            engine.update()
            payload = engine.snapshot().payload
            assert payload is not None
            renderer.render(payload=payload)
            rig = renderer.last_camera_rig_state()
            assert rig is not None
            headings.append(float(rig.heading_deg))
            camera_yaw_values.append(float(payload.camera_yaw_deg))

        first_delta = abs(_angle_delta_deg(camera_yaw_values[2], camera_yaw_values[0]))
        late_delta = abs(_angle_delta_deg(camera_yaw_values[-1], camera_yaw_values[2]))
        assert first_delta > 8.0
        assert late_delta > 8.0
        assert abs(_angle_delta_deg(headings[-1], camera_yaw_values[-1])) < 1.0
    finally:
        renderer.close()
        pygame.quit()
