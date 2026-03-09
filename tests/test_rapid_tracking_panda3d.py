from __future__ import annotations

from cfast_trainer.rapid_tracking_panda3d import (
    RapidTrackingPanda3DRenderer,
    panda3d_rapid_tracking_rendering_available,
)


def test_panda3d_rapid_tracking_rendering_disabled_for_dummy_video(monkeypatch) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.delenv("CFAST_RAPID_TRACKING_RENDERER", raising=False)

    assert panda3d_rapid_tracking_rendering_available() is False


def test_panda3d_rapid_tracking_rendering_disabled_when_forced_to_pygame(monkeypatch) -> None:
    monkeypatch.setenv("CFAST_RAPID_TRACKING_RENDERER", "pygame")
    monkeypatch.delenv("SDL_VIDEODRIVER", raising=False)

    assert panda3d_rapid_tracking_rendering_available() is False


def test_camera_rig_descends_from_high_orbit_into_low_run_path() -> None:
    start = RapidTrackingPanda3DRenderer._camera_rig_state(
        elapsed_s=0.0,
        progress=0.0,
        cam_x=0.0,
        cam_y=0.0,
        zoom=0.0,
        target_kind="soldier",
        target_rel_x=0.0,
        target_rel_y=0.4,
        assist_strength=0.0,
        turbulence_strength=0.6,
    )
    end = RapidTrackingPanda3DRenderer._camera_rig_state(
        elapsed_s=90.0,
        progress=1.0,
        cam_x=0.0,
        cam_y=0.0,
        zoom=0.0,
        target_kind="truck",
        target_rel_x=0.0,
        target_rel_y=0.2,
        assist_strength=0.0,
        turbulence_strength=0.6,
    )

    assert start.altitude_agl > end.altitude_agl
    assert start.altitude_agl >= 20.0
    assert end.altitude_agl <= 7.0
    assert start.orbit_weight > 0.8
    assert end.orbit_weight < 0.1


def test_camera_rig_late_air_targets_pull_view_up_and_into_turn() -> None:
    ground = RapidTrackingPanda3DRenderer._camera_rig_state(
        elapsed_s=85.0,
        progress=0.94,
        cam_x=0.0,
        cam_y=0.0,
        zoom=0.0,
        target_kind="truck",
        target_rel_x=0.42,
        target_rel_y=-0.65,
        assist_strength=0.0,
        turbulence_strength=0.6,
    )
    air = RapidTrackingPanda3DRenderer._camera_rig_state(
        elapsed_s=85.0,
        progress=0.94,
        cam_x=0.0,
        cam_y=0.0,
        zoom=0.0,
        target_kind="jet",
        target_rel_x=0.42,
        target_rel_y=-0.65,
        assist_strength=0.9,
        turbulence_strength=0.6,
    )

    assert air.pitch_deg > ground.pitch_deg
    assert abs(air.heading_deg - air.carrier_heading_deg) > abs(
        ground.heading_deg - ground.carrier_heading_deg
    )


def test_camera_rig_low_assist_and_high_turbulence_change_behavior() -> None:
    helped = RapidTrackingPanda3DRenderer._camera_rig_state(
        elapsed_s=42.0,
        progress=0.42,
        cam_x=0.0,
        cam_y=0.0,
        zoom=0.0,
        target_kind="soldier",
        target_rel_x=0.35,
        target_rel_y=0.1,
        assist_strength=0.9,
        turbulence_strength=0.0,
    )
    rough = RapidTrackingPanda3DRenderer._camera_rig_state(
        elapsed_s=42.0,
        progress=0.42,
        cam_x=0.0,
        cam_y=0.0,
        zoom=0.0,
        target_kind="soldier",
        target_rel_x=0.35,
        target_rel_y=0.1,
        assist_strength=0.0,
        turbulence_strength=1.3,
    )

    assert abs(helped.heading_deg - helped.carrier_heading_deg) > abs(
        rough.heading_deg - rough.carrier_heading_deg
    )
    assert abs(rough.cam_world_z - helped.cam_world_z) > 0.01


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


def test_world_span_is_expanded_for_large_scene() -> None:
    assert RapidTrackingPanda3DRenderer._WORLD_EXTENT_SCALE == 5.0
    assert RapidTrackingPanda3DRenderer._TERRAIN_HALF_SPAN == 1300.0
