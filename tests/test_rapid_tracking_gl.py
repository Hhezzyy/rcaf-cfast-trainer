from __future__ import annotations

import sys
from dataclasses import replace
from importlib.machinery import ModuleSpec
from types import ModuleType, SimpleNamespace

if "moderngl" not in sys.modules:
    moderngl_stub = ModuleType("moderngl")
    moderngl_stub.__spec__ = ModuleSpec("moderngl", loader=None)
    sys.modules["moderngl"] = moderngl_stub

import pygame
import pytest

from cfast_trainer.gl_scenes import RapidTrackingGlScene
from cfast_trainer.modern_gl_renderer import (
    ModernSceneRenderer,
    _build_rapid_tracking_scene_plan,
    _rapid_tracking_backdrop_plan,
    _rapid_tracking_static_asset_library,
)
from cfast_trainer.rapid_tracking import build_rapid_tracking_test
from cfast_trainer.rapid_tracking_gl import (
    _TERRAIN_HALF_SPAN,
    _WORLD_EXTENT_SCALE,
    build_scene_target,
    camera_rig_state,
    camera_space_to_viewport,
    ground_route_pose,
    select_building_scenery,
)
from cfast_trainer.rapid_tracking_view import camera_pose_compat, target_projection


class _FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _stub_rt_renderer(*, size: tuple[int, int]) -> ModernSceneRenderer:
    renderer = ModernSceneRenderer.__new__(ModernSceneRenderer)
    renderer._win_w = int(size[0])
    renderer._win_h = int(size[1])
    renderer._scene_assets = _rapid_tracking_static_asset_library()
    renderer._batch = SimpleNamespace(triangles=[], scene_triangles=[], textured=[])
    renderer._last_rt_world_debug = {}
    return renderer


def _angle_delta_deg(a: float, b: float) -> float:
    return ((float(a) - float(b) + 180.0) % 360.0) - 180.0


def test_camera_rig_descends_from_high_orbit_into_low_run_path() -> None:
    start = camera_rig_state(
        elapsed_s=0.0,
        progress=0.0,
        camera_yaw_deg=None,
        camera_pitch_deg=None,
        zoom=0.0,
        target_kind="soldier",
        turbulence_strength=0.6,
    )
    end = camera_rig_state(
        elapsed_s=90.0,
        progress=1.0,
        camera_yaw_deg=None,
        camera_pitch_deg=None,
        zoom=0.0,
        target_kind="truck",
        turbulence_strength=0.6,
    )

    assert start.altitude_agl > end.altitude_agl
    assert start.altitude_agl >= 40.0
    assert end.altitude_agl <= 11.0
    assert start.orbit_weight > 0.8
    assert end.orbit_weight < 0.1


def test_camera_rig_uses_direct_camera_pose_without_recentering() -> None:
    rig = camera_rig_state(
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


def test_camera_rig_uses_wide_default_fov_and_narrow_hold_zoom() -> None:
    wide = camera_rig_state(
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
    zoomed = camera_rig_state(
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


def test_camera_rig_turbulence_variation_is_seeded_and_repeatable() -> None:
    common = dict(
        progress=0.44,
        camera_yaw_deg=128.0,
        camera_pitch_deg=-10.0,
        zoom=0.0,
        target_kind="truck",
        target_world_x=42.0,
        target_world_y=96.0,
        focus_world_x=8.0,
        focus_world_y=78.0,
    )
    exact_a = camera_rig_state(
        elapsed_s=18.0,
        seed=551,
        turbulence_strength=0.8,
        **common,
    )
    exact_b = camera_rig_state(
        elapsed_s=18.0,
        seed=551,
        turbulence_strength=0.8,
        **common,
    )
    later_rough = camera_rig_state(
        elapsed_s=18.12,
        seed=551,
        turbulence_strength=0.8,
        **common,
    )
    exact_calm = camera_rig_state(
        elapsed_s=18.0,
        seed=551,
        turbulence_strength=0.0,
        **common,
    )
    later_calm = camera_rig_state(
        elapsed_s=18.12,
        seed=551,
        turbulence_strength=0.0,
        **common,
    )
    other_seed = camera_rig_state(
        elapsed_s=18.0,
        seed=552,
        turbulence_strength=0.8,
        **common,
    )

    assert exact_a == exact_b
    rough_delta = abs(later_rough.view_heading_deg - exact_a.view_heading_deg) + abs(
        later_rough.view_pitch_deg - exact_a.view_pitch_deg
    ) + abs(later_rough.roll_deg - exact_a.roll_deg) + abs(
        later_rough.cam_world_z - exact_a.cam_world_z
    )
    calm_delta = abs(later_calm.view_heading_deg - exact_calm.view_heading_deg) + abs(
        later_calm.view_pitch_deg - exact_calm.view_pitch_deg
    ) + abs(later_calm.roll_deg - exact_calm.roll_deg) + abs(
        later_calm.cam_world_z - exact_calm.cam_world_z
    )
    assert rough_delta > calm_delta
    assert abs(exact_a.view_heading_deg - exact_a.heading_deg) > 0.3
    assert other_seed.cam_world_x != pytest.approx(exact_a.cam_world_x)


def test_camera_pose_can_move_past_old_sweep_limit() -> None:
    rig = camera_rig_state(
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
    compat_x, compat_y = camera_pose_compat(
        heading_deg=rig.heading_deg,
        pitch_deg=rig.pitch_deg,
        neutral_heading_deg=rig.neutral_heading_deg,
        neutral_pitch_deg=rig.neutral_pitch_deg,
    )

    assert abs(_angle_delta_deg(rig.heading_deg, rig.neutral_heading_deg)) > 66.0
    assert abs(compat_x) > 5.2
    assert compat_y != pytest.approx(0.0)


def test_target_projection_changes_with_camera_pose_only() -> None:
    neutral = camera_rig_state(
        elapsed_s=18.0,
        seed=551,
        progress=0.42,
        camera_yaw_deg=None,
        camera_pitch_deg=None,
        zoom=0.0,
        target_kind="truck",
        target_world_x=42.0,
        target_world_y=96.0,
        focus_world_x=8.0,
        focus_world_y=78.0,
        turbulence_strength=0.6,
    )
    rig_left = camera_rig_state(
        elapsed_s=18.0,
        seed=551,
        progress=0.42,
        camera_yaw_deg=float(neutral.neutral_heading_deg) - 8.0,
        camera_pitch_deg=float(neutral.neutral_pitch_deg),
        zoom=0.0,
        target_kind="truck",
        target_world_x=42.0,
        target_world_y=96.0,
        focus_world_x=8.0,
        focus_world_y=78.0,
        turbulence_strength=0.6,
    )
    rig_right = camera_rig_state(
        elapsed_s=18.0,
        seed=551,
        progress=0.42,
        camera_yaw_deg=float(neutral.neutral_heading_deg) + 8.0,
        camera_pitch_deg=float(neutral.neutral_pitch_deg),
        zoom=0.0,
        target_kind="truck",
        target_world_x=42.0,
        target_world_y=96.0,
        focus_world_x=8.0,
        focus_world_y=78.0,
        turbulence_strength=0.6,
    )

    left = target_projection(
        rig=rig_left,
        target_kind="truck",
        target_world_x=42.0,
        target_world_y=96.0,
        elapsed_s=18.0,
        scene_progress=0.42,
        seed=551,
        size=(800, 600),
    )
    right = target_projection(
        rig=rig_right,
        target_kind="truck",
        target_world_x=42.0,
        target_world_y=96.0,
        elapsed_s=18.0,
        scene_progress=0.42,
        seed=551,
        size=(800, 600),
    )

    assert left.in_front is True
    assert right.in_front is True
    assert left.target_rel_x != pytest.approx(right.target_rel_x)
    assert left.screen_x != pytest.approx(right.screen_x)


def test_target_projection_moves_monotonically_with_sampled_camera_yaw_offsets() -> None:
    neutral = camera_rig_state(
        elapsed_s=18.0,
        seed=551,
        progress=0.42,
        camera_yaw_deg=None,
        camera_pitch_deg=None,
        zoom=0.0,
        target_kind="truck",
        target_world_x=42.0,
        target_world_y=96.0,
        focus_world_x=8.0,
        focus_world_y=78.0,
        turbulence_strength=0.6,
    )

    sampled_rel_x: list[float] = []
    sampled_screen_x: list[float] = []
    for yaw_offset_deg in (-12.0, -6.0, 0.0, 6.0, 12.0):
        rig = camera_rig_state(
            elapsed_s=18.0,
            seed=551,
            progress=0.42,
            camera_yaw_deg=float(neutral.neutral_heading_deg) + yaw_offset_deg,
            camera_pitch_deg=float(neutral.neutral_pitch_deg),
            zoom=0.0,
            target_kind="truck",
            target_world_x=42.0,
            target_world_y=96.0,
            focus_world_x=8.0,
            focus_world_y=78.0,
            turbulence_strength=0.6,
        )
        projection = target_projection(
            rig=rig,
            target_kind="truck",
            target_world_x=42.0,
            target_world_y=96.0,
            elapsed_s=18.0,
            scene_progress=0.42,
            seed=551,
            size=(800, 600),
        )
        sampled_rel_x.append(float(projection.target_rel_x))
        sampled_screen_x.append(float(projection.screen_x))

    assert sampled_rel_x == sorted(sampled_rel_x, reverse=True)
    assert sampled_screen_x == sorted(sampled_screen_x, reverse=True)


def test_camera_space_projection_maps_center_and_offscreen_right() -> None:
    center = camera_space_to_viewport(
        cam_x=0.0,
        cam_y=10.0,
        cam_z=0.0,
        size=(800, 600),
        h_fov_deg=60.0,
        v_fov_deg=45.0,
    )
    right = camera_space_to_viewport(
        cam_x=20.0,
        cam_y=10.0,
        cam_z=0.0,
        size=(800, 600),
        h_fov_deg=60.0,
        v_fov_deg=45.0,
    )

    assert center == (400.0, 300.0, True, True)
    assert right[0] > 800.0
    assert right[2] is False
    assert right[3] is True


def test_ground_routes_stay_axis_aligned_and_tanks_spin_in_place() -> None:
    side_a = ground_route_pose(
        elapsed_s=8.0,
        phase=0.2,
        speed=0.36,
        lateral_bias=12.0,
        depth_bias=26.0,
        route="ground_side",
    )
    side_b = ground_route_pose(
        elapsed_s=18.0,
        phase=0.2,
        speed=0.36,
        lateral_bias=12.0,
        depth_bias=26.0,
        route="ground_side",
    )
    convoy_a = ground_route_pose(
        elapsed_s=8.0,
        phase=0.4,
        speed=0.34,
        lateral_bias=-18.0,
        depth_bias=34.0,
        route="ground_convoy",
    )
    convoy_b = ground_route_pose(
        elapsed_s=18.0,
        phase=0.4,
        speed=0.34,
        lateral_bias=-18.0,
        depth_bias=34.0,
        route="ground_convoy",
    )
    tank_a = ground_route_pose(
        elapsed_s=8.0,
        phase=0.55,
        speed=0.38,
        lateral_bias=-11.0,
        depth_bias=18.0,
        route="tank_hold",
        tank_spin=True,
    )
    tank_b = ground_route_pose(
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


def test_building_target_uses_preexisting_scenery_slots() -> None:
    hangar_a = select_building_scenery(
        variant="hangar",
        target_rel_x=-1.05,
        target_rel_y=0.52,
    )
    hangar_b = select_building_scenery(
        variant="hangar",
        target_rel_x=0.18,
        target_rel_y=-0.20,
    )
    tower = select_building_scenery(
        variant="tower",
        target_rel_x=0.78,
        target_rel_y=-0.44,
    )

    assert hangar_a.startswith("hangar-")
    assert hangar_b.startswith("hangar-")
    assert tower.startswith("tower-")
    assert hangar_a != hangar_b


def test_engine_driven_targets_map_to_gl_scene_targets_by_kind() -> None:
    clock = _FakeClock()
    engine = build_rapid_tracking_test(clock=clock, seed=551, difficulty=0.63)
    engine.start_scored()
    seen: dict[str, object] = {}

    for _ in range(800):
        payload = engine.snapshot().payload
        assert payload is not None
        target = build_scene_target(payload=payload, size=(960, 540))
        if target.overlay.on_screen:
            seen.setdefault(payload.target_kind, target)
        if {"soldier", "building", "truck", "helicopter", "jet"} <= set(seen):
            break
        clock.advance(0.1)
        engine.update()

    assert {"soldier", "building", "truck", "helicopter", "jet"} <= set(seen)
    assert seen["building"].source == "scenery"
    assert seen["building"].scenery_id is not None
    assert seen["soldier"].source == "dynamic"
    assert seen["truck"].source == "dynamic"
    assert seen["helicopter"].source == "dynamic"
    assert seen["jet"].source == "dynamic"


def test_world_span_is_expanded_for_large_scene() -> None:
    assert _WORLD_EXTENT_SCALE == 5.0
    assert _TERRAIN_HALF_SPAN == 1300.0


def test_rapid_tracking_backdrop_plan_lightens_ground_and_clamps_steep_up_horizon() -> None:
    clock = _FakeClock()
    engine = build_rapid_tracking_test(clock=clock, seed=551, difficulty=0.63)
    engine.start_scored()
    payload = engine.snapshot().payload
    assert payload is not None

    neutral = _rapid_tracking_backdrop_plan(viewport_size=(960, 540), payload=payload)
    steep_up = _rapid_tracking_backdrop_plan(
        viewport_size=(960, 540),
        payload=replace(payload, camera_pitch_deg=82.0),
    )

    assert 180.0 <= neutral.horizon_y <= 320.0
    assert steep_up.horizon_y <= 0.0
    assert steep_up.horizon_y < neutral.horizon_y
    assert min(neutral.ground_base_rgb) > 0.35
    assert min(neutral.ground_horizon_rgb) > 0.25


def test_rapid_tracking_scene_plan_emits_world_geometry_at_representative_camera_pose() -> None:
    clock = _FakeClock()
    engine = build_rapid_tracking_test(clock=clock, seed=551, difficulty=0.63)
    engine.start_scored()
    payload = engine.snapshot().payload
    assert payload is not None

    scene = RapidTrackingGlScene(
        world=pygame.Rect(0, 0, 960, 540),
        payload=payload,
        active_phase=True,
    )
    scene_plan = _build_rapid_tracking_scene_plan(scene)
    renderer = _stub_rt_renderer(size=(960, 540))

    renderer._render_scene_plan(scene_plan=scene_plan)

    assert renderer._last_rt_world_debug["world_triangles_emitted"] > 0
    assert renderer._last_rt_world_debug["world_triangles_rejected"] >= 0
    assert len(renderer._batch.scene_triangles) > 0
