from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from cfast_trainer.rapid_tracking import (
    RapidTrackingConfig,
    RapidTrackingDriftGenerator,
    RapidTrackingPayload,
    RapidTrackingTrainingProfile,
    RapidTrackingTrainingSegment,
    _ellipse_contains_point,
    build_rapid_tracking_compound_layout,
    build_rapid_tracking_test,
    score_window,
)
from cfast_trainer.rapid_tracking_view import track_to_world_xy as rapid_tracking_track_to_world_xy


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def test_drift_generator_determinism_same_seed_same_sequence() -> None:
    seed = 901
    g1 = RapidTrackingDriftGenerator(seed=seed)
    g2 = RapidTrackingDriftGenerator(seed=seed)

    seq1 = [g1.next_vector(difficulty=0.62) for _ in range(40)]
    seq2 = [g2.next_vector(difficulty=0.62) for _ in range(40)]

    assert seq1 == seq2


def test_score_window_exact_partial_and_zero() -> None:
    threshold = 0.19

    exact = score_window(mean_error=0.10, good_window_error=threshold)
    partial = score_window(mean_error=0.285, good_window_error=threshold)
    zero = score_window(mean_error=0.40, good_window_error=threshold)

    assert exact == 1.0
    assert partial == pytest.approx(0.5)
    assert zero == 0.0


def test_timer_boundary_transitions_to_results_and_rejects_submit() -> None:
    clock = FakeClock()
    engine = build_rapid_tracking_test(
        clock=clock,
        seed=17,
        difficulty=0.5,
        config=RapidTrackingConfig(
            practice_duration_s=0.0,
            scored_duration_s=2.0,
            tick_hz=120.0,
        ),
    )

    engine.start_scored()
    assert engine.phase.value == "scored"
    assert engine.can_exit() is False

    clock.advance(2.0)
    engine.update()

    assert engine.phase.value == "results"
    assert engine.can_exit() is True
    assert engine.submit_answer("0") is False

    summary = engine.scored_summary()
    assert summary.duration_s == pytest.approx(2.0)
    assert summary.attempted >= 1


def test_engine_determinism_same_seed_same_control_script() -> None:
    config = RapidTrackingConfig(
        practice_duration_s=0.0,
        scored_duration_s=2.0,
        tick_hz=120.0,
    )
    controls = [(0.4, -0.2), (0.1, 0.0), (-0.2, 0.3), (0.0, 0.0)]

    c1 = FakeClock()
    c2 = FakeClock()
    e1 = build_rapid_tracking_test(clock=c1, seed=222, difficulty=0.67, config=config)
    e2 = build_rapid_tracking_test(clock=c2, seed=222, difficulty=0.67, config=config)
    e1.start_scored()
    e2.start_scored()

    for i in range(20):
        cx, cy = controls[i % len(controls)]
        e1.set_control(horizontal=cx, vertical=cy)
        e2.set_control(horizontal=cx, vertical=cy)
        c1.advance(0.1)
        c2.advance(0.1)
        e1.update()
        e2.update()

    s1 = e1.scored_summary()
    s2 = e2.scored_summary()
    assert s1 == s2
    assert e1.snapshot().payload == e2.snapshot().payload


def test_same_seed_reproduces_target_motion_and_obscuration_samples() -> None:
    config = RapidTrackingConfig(
        practice_duration_s=0.0,
        scored_duration_s=4.0,
        tick_hz=120.0,
    )
    controls = [(0.35, -0.18), (0.0, 0.0), (-0.22, 0.16), (0.18, -0.08)]

    def sampled_states(seed: int) -> list[tuple[float, float, float, float, bool, str, str]]:
        clock = FakeClock()
        engine = build_rapid_tracking_test(
            clock=clock,
            seed=seed,
            difficulty=0.63,
            config=config,
        )
        engine.start_scored()
        samples: list[tuple[float, float, float, float, bool, str, str]] = []
        for step in range(24):
            cx, cy = controls[step % len(controls)]
            engine.set_control(horizontal=cx, vertical=cy)
            clock.advance(0.1)
            engine.update()
            payload = engine.snapshot().payload
            assert payload is not None
            samples.append(
                (
                    float(payload.target_world_x),
                    float(payload.target_world_y),
                    float(payload.target_vx),
                    float(payload.target_vy),
                    bool(payload.target_visible),
                    str(payload.target_cover_state),
                    str(payload.target_kind),
                )
            )
        return samples

    assert sampled_states(551) == sampled_states(551)


def test_camera_yaw_keeps_advancing_under_sustained_horizontal_input() -> None:
    clock = FakeClock()
    engine = build_rapid_tracking_test(
        clock=clock,
        seed=552,
        difficulty=0.63,
        config=RapidTrackingConfig(
            practice_duration_s=0.0,
            scored_duration_s=5.0,
            tick_hz=120.0,
        ),
    )
    engine.start_scored()

    yaw_values: list[float] = []
    compat_x_values: list[float] = []
    for _ in range(6):
        engine.set_control(horizontal=1.0, vertical=0.0)
        clock.advance(0.25)
        engine.update()
        payload = engine.snapshot().payload
        assert payload is not None
        yaw_values.append(float(payload.camera_yaw_deg))
        compat_x_values.append(float(payload.camera_x))

    first_delta = ((yaw_values[2] - yaw_values[0] + 180.0) % 360.0) - 180.0
    late_delta = ((yaw_values[-1] - yaw_values[2] + 180.0) % 360.0) - 180.0
    assert abs(first_delta) > 8.0
    assert abs(late_delta) > 8.0
    assert max(abs(value) for value in compat_x_values) > 1.5


def test_camera_pitch_keeps_advancing_under_sustained_vertical_input() -> None:
    clock = FakeClock()
    engine = build_rapid_tracking_test(
        clock=clock,
        seed=553,
        difficulty=0.63,
        config=RapidTrackingConfig(
            practice_duration_s=0.0,
            scored_duration_s=5.0,
            tick_hz=120.0,
        ),
    )
    engine.start_scored()

    pitch_values: list[float] = []
    for _ in range(5):
        engine.set_control(horizontal=0.0, vertical=-1.0)
        clock.advance(0.25)
        engine.update()
        payload = engine.snapshot().payload
        assert payload is not None
        pitch_values.append(float(payload.camera_pitch_deg))

    assert pitch_values[0] < pitch_values[2] < pitch_values[-1]


def test_trigger_capture_scores_when_target_inside_camera_box() -> None:
    clock = FakeClock()
    engine = build_rapid_tracking_test(
        clock=clock,
        seed=44,
        difficulty=0.5,
        config=RapidTrackingConfig(
            practice_duration_s=0.0,
            scored_duration_s=5.0,
            tick_hz=120.0,
        ),
    )

    engine.start_scored()
    engine._target_x = 0.0
    engine._target_y = 0.0
    engine._target_kind = "jet"
    engine._target_is_moving = True
    engine._target_terrain_occluded = False
    engine._reset_camera_pose_to_target()

    assert engine.submit_answer("CAPTURE") is True

    payload = engine.snapshot().payload
    assert payload is not None
    assert payload.target_in_capture_box is True
    assert payload.capture_zoom > 0.0
    assert payload.capture_hits == 1
    assert payload.capture_attempts == 1
    assert payload.capture_points == 2
    summary = engine.scored_summary()
    assert summary.capture_points == 2
    assert summary.capture_max_points == 2
    assert summary.capture_score_ratio == pytest.approx(1.0)


def test_trigger_capture_respects_cooldown() -> None:
    clock = FakeClock()
    engine = build_rapid_tracking_test(
        clock=clock,
        seed=77,
        difficulty=0.5,
        config=RapidTrackingConfig(
            practice_duration_s=0.0,
            scored_duration_s=5.0,
            tick_hz=120.0,
        ),
    )

    engine.start_scored()
    engine._target_x = 0.0
    engine._target_y = 0.0
    engine._target_kind = "truck"
    engine._target_is_moving = False
    engine._target_terrain_occluded = False
    engine._reset_camera_pose_to_target()

    assert engine.submit_answer("CAPTURE") is True
    assert engine.submit_answer("CAPTURE") is False

    summary = engine.scored_summary()
    assert summary.capture_attempts == 1
    assert summary.capture_hits == 1


def test_capture_hold_start_bonus_and_release_drive_zoom_state() -> None:
    clock = FakeClock()
    engine = build_rapid_tracking_test(
        clock=clock,
        seed=88,
        difficulty=0.5,
        config=RapidTrackingConfig(
            practice_duration_s=0.0,
            scored_duration_s=5.0,
            tick_hz=120.0,
            capture_hold_blend_response_hz=20.0,
        ),
    )

    engine.start_scored()
    engine._target_x = 0.0
    engine._target_y = 0.0
    engine._target_kind = "truck"
    engine._target_terrain_occluded = False
    engine._reset_camera_pose_to_target()
    engine._target_in_capture_box = lambda require_visible: True  # type: ignore[method-assign]

    assert engine.submit_answer("CAPTURE_HOLD_START") is True
    assert engine._capture_hold_active is True

    clock.advance(0.30)
    engine.update()
    active_payload = engine.snapshot().payload
    assert active_payload is not None
    assert active_payload.capture_zoom > 0.8
    assert active_payload.capture_points >= 2
    held_points = active_payload.capture_points

    assert engine.submit_answer("CAPTURE_HOLD_END") is True
    clock.advance(0.30)
    engine.update()
    released_payload = engine.snapshot().payload
    assert released_payload is not None
    assert released_payload.capture_points == held_points
    assert released_payload.capture_zoom < 0.2


def test_capture_hold_bonus_only_accrues_while_target_stays_in_box() -> None:
    clock = FakeClock()
    engine = build_rapid_tracking_test(
        clock=clock,
        seed=89,
        difficulty=0.5,
        config=RapidTrackingConfig(
            practice_duration_s=0.0,
            scored_duration_s=5.0,
            tick_hz=120.0,
            capture_hold_blend_response_hz=20.0,
        ),
    )

    engine.start_scored()
    engine._target_x = 0.0
    engine._target_y = 0.0
    engine._target_kind = "truck"
    engine._target_terrain_occluded = False
    engine._reset_camera_pose_to_target()
    engine._target_in_capture_box = lambda require_visible: True  # type: ignore[method-assign]

    assert engine.submit_answer("CAPTURE_HOLD_START") is True
    clock.advance(0.26)
    engine.update()
    points_in_box = engine.snapshot().payload.capture_points

    engine._target_in_capture_box = lambda require_visible: False  # type: ignore[method-assign]
    engine._target_x = 0.8
    engine._target_y = 0.8
    clock.advance(0.30)
    engine.update()
    payload = engine.snapshot().payload
    assert payload is not None
    assert payload.capture_points == points_in_box
    assert engine._capture_max_points > payload.capture_points


def test_training_segment_payload_metadata_and_filtered_kinds_are_exposed() -> None:
    clock = FakeClock()
    engine = build_rapid_tracking_test(
        clock=clock,
        seed=101,
        difficulty=0.5,
        config=RapidTrackingConfig(
            practice_duration_s=0.0,
            scored_duration_s=30.0,
            tick_hz=120.0,
        ),
        title="Rapid Tracking: Lock Anchor",
        scored_segments=(
            RapidTrackingTrainingSegment(
                label="Lock Anchor",
                duration_s=30.0,
                focus_label="Stable lock quality",
                active_target_kinds=("soldier", "truck"),
                active_challenges=("lock_quality",),
                profile=RapidTrackingTrainingProfile(
                    target_kinds=("soldier", "truck"),
                    cover_modes=("open",),
                    handoff_modes=("smooth",),
                ),
            ),
        ),
    )

    engine.start_scored()
    snap = engine.snapshot()
    payload = snap.payload
    assert isinstance(payload, RapidTrackingPayload)
    assert snap.title == "Rapid Tracking: Lock Anchor"
    assert payload.focus_label == "Stable lock quality"
    assert payload.segment_label == "Lock Anchor"
    assert payload.active_target_kinds == ("soldier", "truck")
    assert payload.active_challenges == ("lock_quality",)
    assert {segment.kind for segment in engine._active_training_script} <= {"soldier", "truck"}
    assert {segment.cover_mode for segment in engine._active_training_script} <= {"open"}
    assert {segment.handoff for segment in engine._active_training_script} <= {"smooth"}


def test_scene_script_covers_requested_target_types_and_handoffs() -> None:
    clock = FakeClock()
    engine = build_rapid_tracking_test(
        clock=clock,
        seed=311,
        difficulty=0.58,
        config=RapidTrackingConfig(
            practice_duration_s=0.0,
            scored_duration_s=10.0,
            tick_hz=120.0,
        ),
    )

    engine.start_scored()
    payload = engine.snapshot().payload
    assert payload is not None

    profile = engine._difficulty_profile()
    seen_kinds = {segment.kind for segment in profile.scene_script}
    seen_handoffs = {segment.handoff for segment in profile.scene_script}
    seen_cover_modes = {segment.cover_mode for segment in profile.scene_script}

    assert payload.session_seed == 311
    assert {"soldier", "building", "truck", "helicopter", "jet"} <= seen_kinds
    assert {"smooth", "jump"} <= seen_handoffs
    assert {"open", "building", "terrain"} <= seen_cover_modes


def test_target_speed_order_matches_brief() -> None:
    seed = 901
    difficulty = 0.52
    config = RapidTrackingConfig(
        practice_duration_s=0.0,
        scored_duration_s=10.0,
        tick_hz=120.0,
    )
    engine = build_rapid_tracking_test(
        clock=FakeClock(),
        seed=seed,
        difficulty=difficulty,
        config=config,
    )
    engine.start_scored()

    scene_script = engine._difficulty_profile().scene_script

    def sampled_speed(kind: str) -> float:
        speeds: list[float] = []
        for idx, segment in enumerate(scene_script):
            if segment.kind != kind:
                continue
            sample_engine = build_rapid_tracking_test(
                clock=FakeClock(),
                seed=seed,
                difficulty=difficulty,
                config=config,
            )
            sample_engine.start_scored()
            if idx == 0:
                sample_engine._start_scene_segment(initial=True)
            else:
                start_anchor = sample_engine._anchor_lookup[segment.start_anchor_id]
                sample_engine._target_x = float(start_anchor.x)
                sample_engine._target_y = float(start_anchor.y)
                sample_engine._script_index = idx - 1
                sample_engine._start_scene_segment(initial=False)
            sample_engine._sim_elapsed_s = (
                sample_engine._segment_started_s
                + (sample_engine._segment_duration_s * 0.5)
            )
            sample_engine._advance_target()
            speeds.append(math.hypot(sample_engine._target_vx, sample_engine._target_vy))
        if not speeds:
            raise AssertionError(f"missing segment for {kind}")
        return sum(speeds) / len(speeds)

    soldier_speed = sampled_speed("soldier")
    building_speed = sampled_speed("building")
    truck_speed = sampled_speed("truck")
    helicopter_speed = sampled_speed("helicopter")
    jet_speed = sampled_speed("jet")

    assert building_speed == pytest.approx(0.0)
    assert soldier_speed < truck_speed < helicopter_speed < jet_speed


def test_truck_segments_bind_to_seeded_road_anchors() -> None:
    clock = FakeClock()
    engine = build_rapid_tracking_test(
        clock=clock,
        seed=903,
        difficulty=0.52,
        config=RapidTrackingConfig(
            practice_duration_s=0.0,
            scored_duration_s=10.0,
            tick_hz=120.0,
        ),
    )

    road_anchor_ids = {anchor.anchor_id for anchor in engine._compound_layout.road_anchors}
    truck_segments = [
        segment
        for segment in engine._difficulty_profile().scene_script
        if segment.kind == "truck"
    ]

    assert truck_segments
    assert {segment.route_kind for segment in truck_segments} <= {
        "road_convoy",
        "dirt_transfer",
        "offroad_armor_leg",
    }
    assert any(segment.route_kind == "road_convoy" for segment in truck_segments)
    assert any(segment.route_kind in {"dirt_transfer", "offroad_armor_leg"} for segment in truck_segments)
    offroad_segments = [segment for segment in truck_segments if segment.route_kind == "offroad_armor_leg"]
    assert all(segment.variant == "tracked" for segment in offroad_segments)
    on_road_segments = [segment for segment in truck_segments if segment.route_kind != "offroad_armor_leg"]
    assert all(segment.start_anchor_id in road_anchor_ids for segment in on_road_segments)
    assert all(segment.end_anchor_id in road_anchor_ids for segment in on_road_segments)


def test_truck_motion_follows_seeded_leg_direction_without_sideways_jump() -> None:
    clock = FakeClock()
    engine = build_rapid_tracking_test(
        clock=clock,
        seed=902,
        difficulty=0.52,
        config=RapidTrackingConfig(
            practice_duration_s=0.0,
            scored_duration_s=10.0,
            tick_hz=120.0,
        ),
    )

    engine.start_scored()

    truck_segment = next(
        segment
        for segment in engine._difficulty_profile().scene_script
        if segment.kind == "truck" and segment.cover_mode == "open"
    )
    truck_index = next(
        idx
        for idx, segment in enumerate(engine._difficulty_profile().scene_script)
        if segment.kind == "truck" and segment.cover_mode == "open"
    )
    engine._script_index = truck_index - 1
    engine._start_scene_segment(initial=False)

    start_anchor = engine._anchor_lookup[truck_segment.start_anchor_id]
    end_anchor = engine._anchor_lookup[truck_segment.end_anchor_id]
    route_dx = end_anchor.x - start_anchor.x
    route_dy = end_anchor.y - start_anchor.y
    route_length = max(1e-6, math.hypot(route_dx, route_dy))
    route_dir = (route_dx / route_length, route_dy / route_length)

    sampled_positions: list[tuple[float, float]] = []
    sampled_velocities: list[tuple[float, float]] = []
    for ratio in (0.15, 0.35, 0.55, 0.75):
        engine._sim_elapsed_s = engine._segment_started_s + (engine._segment_duration_s * ratio)
        engine._advance_target()
        sampled_positions.append((engine._target_x, engine._target_y))
        sampled_velocities.append((engine._target_vx, engine._target_vy))

    progress_values = [
        ((x - start_anchor.x) * route_dir[0]) + ((y - start_anchor.y) * route_dir[1])
        for x, y in sampled_positions
    ]
    lateral_offsets = [
        abs(((x - start_anchor.x) * -route_dir[1]) + ((y - start_anchor.y) * route_dir[0]))
        for x, y in sampled_positions
    ]
    velocity_alignment = [
        ((vx * route_dir[0]) + (vy * route_dir[1])) / max(1e-6, math.hypot(vx, vy))
        for vx, vy in sampled_velocities
        if math.hypot(vx, vy) > 1e-6
    ]

    assert progress_values == sorted(progress_values)
    assert max(lateral_offsets) < 0.08
    assert velocity_alignment
    assert min(velocity_alignment) > 0.92


def test_ground_targets_are_not_marked_obscured_too_early() -> None:
    clock = FakeClock()
    engine = build_rapid_tracking_test(
        clock=clock,
        seed=731,
        difficulty=0.58,
        config=RapidTrackingConfig(
            practice_duration_s=0.0,
            scored_duration_s=5.0,
            tick_hz=120.0,
        ),
    )

    ridge = engine._mountain_ridge_for(0.0)

    assert engine._is_occluded_by_terrain(
        target_rel_x=0.0,
        target_rel_y=ridge + 0.05,
        target_kind="soldier",
    ) is False
    assert engine._is_occluded_by_terrain(
        target_rel_x=0.0,
        target_rel_y=ridge + 0.18,
        target_kind="soldier",
    ) is True


def test_same_seed_shares_layout_between_practice_and_scored_but_different_phase_scripts() -> None:
    clock = FakeClock()
    engine = build_rapid_tracking_test(
        clock=clock,
        seed=801,
        difficulty=0.58,
        config=RapidTrackingConfig(
            practice_duration_s=10.0,
            scored_duration_s=10.0,
            tick_hz=120.0,
        ),
    )

    profile = engine._difficulty_profile()
    assert engine._scenario.layout == engine._compound_layout
    assert profile.practice_script != profile.scene_script
    assert engine._scenario.layout.seed == 801


def test_different_seed_changes_compound_layout_and_schedule() -> None:
    clock = FakeClock()
    a = build_rapid_tracking_test(clock=clock, seed=901, difficulty=0.58)
    b = build_rapid_tracking_test(clock=clock, seed=902, difficulty=0.58)

    assert a._compound_layout != b._compound_layout
    assert a._difficulty_profile().scene_script != b._difficulty_profile().scene_script


def test_compound_layout_spreads_bases_and_pois_across_a_wider_world() -> None:
    layout = build_rapid_tracking_compound_layout(seed=551)
    zones = (*layout.bases, *layout.pois)

    xs = [float(zone.x) for zone in zones]
    ys = [float(zone.y) for zone in zones]
    assert max(xs) - min(xs) >= 10.5
    assert max(ys) - min(ys) >= 3.0

    world_points = [
        rapid_tracking_track_to_world_xy(
            track_x=float(zone.x),
            track_y=float(zone.y),
            path_lateral_bias=float(layout.path_lateral_bias),
        )
        for zone in zones
    ]
    world_xs = [float(point[0]) for point in world_points]
    world_ys = [float(point[1]) for point in world_points]
    assert max(world_xs) - min(world_xs) >= 400.0
    assert max(world_ys) - min(world_ys) >= 220.0


def test_compound_layout_keeps_obstacles_clear_of_pois_and_each_other() -> None:
    layout = build_rapid_tracking_compound_layout(seed=551)
    zones = (*layout.bases, *layout.pois)

    for zone in zones:
        for obstacle in layout.obstacles:
            assert not _ellipse_contains_point(
                x=float(zone.x),
                y=float(zone.y),
                center_x=float(obstacle.x),
                center_y=float(obstacle.y),
                radius_x=float(obstacle.radius_x),
                radius_y=float(obstacle.radius_y),
                rotation_deg=float(obstacle.rotation_deg),
                padding=float(zone.radius) + 0.08,
            )

    for idx, obstacle in enumerate(layout.obstacles):
        for other in layout.obstacles[idx + 1 :]:
            assert not _ellipse_contains_point(
                x=float(obstacle.x),
                y=float(obstacle.y),
                center_x=float(other.x),
                center_y=float(other.y),
                radius_x=float(other.radius_x),
                radius_y=float(other.radius_y),
                rotation_deg=float(other.rotation_deg),
                padding=0.06,
            )
            assert not _ellipse_contains_point(
                x=float(other.x),
                y=float(other.y),
                center_x=float(obstacle.x),
                center_y=float(obstacle.y),
                radius_x=float(obstacle.radius_x),
                radius_y=float(obstacle.radius_y),
                rotation_deg=float(obstacle.rotation_deg),
                padding=0.06,
            )


def test_compound_layout_keeps_base_loop_and_poi_spurs_connected() -> None:
    layout = build_rapid_tracking_compound_layout(seed=551)
    paved_segments = tuple(segment for segment in layout.road_segments if segment.surface == "paved")
    dirt_segments = tuple(segment for segment in layout.road_segments if segment.surface == "dirt")

    assert len(paved_segments) >= len(layout.bases)
    assert len(dirt_segments) == len(layout.pois)

    paved_segment_ids = " ".join(segment.segment_id for segment in paved_segments)
    dirt_segment_ids = " ".join(segment.segment_id for segment in dirt_segments)
    for base in layout.bases:
        assert base.poi_id in paved_segment_ids
    for poi in layout.pois:
        assert poi.poi_id in dirt_segment_ids


def test_compound_layout_includes_seeded_foliage_clusters() -> None:
    same_a = build_rapid_tracking_compound_layout(seed=431)
    same_b = build_rapid_tracking_compound_layout(seed=431)
    other = build_rapid_tracking_compound_layout(seed=432)

    assert same_a.compound_center_x == pytest.approx(same_b.compound_center_x)
    assert same_a.compound_center_y == pytest.approx(same_b.compound_center_y)
    assert same_a.shrub_clusters == same_b.shrub_clusters
    assert same_a.tree_clusters == same_b.tree_clusters
    assert same_a.forest_clusters == same_b.forest_clusters

    assert same_a.shrub_clusters
    assert same_a.tree_clusters
    assert same_a.forest_clusters
    assert {cluster.asset_id for cluster in same_a.shrub_clusters} == {"shrubs_low_cluster"}
    assert {cluster.asset_id for cluster in same_a.tree_clusters} == {"trees_field_cluster"}
    assert {cluster.asset_id for cluster in same_a.forest_clusters} == {"forest_canopy_patch"}

    assert (
        same_a.compound_center_x != other.compound_center_x
        or same_a.compound_center_y != other.compound_center_y
        or same_a.shrub_clusters != other.shrub_clusters
        or same_a.tree_clusters != other.tree_clusters
        or same_a.forest_clusters != other.forest_clusters
    )


def test_building_handoff_segments_bind_to_real_building_anchors() -> None:
    clock = FakeClock()
    engine = build_rapid_tracking_test(clock=clock, seed=777, difficulty=0.58)

    building_segments = [
        segment for segment in engine._difficulty_profile().scene_script if segment.cover_mode == "building"
    ]
    building_anchor_ids = {anchor.anchor_id for anchor in engine._compound_layout.building_anchors}

    assert building_segments
    for segment in building_segments:
        assert segment.kind == "building"
        assert segment.focus_anchor_id in building_anchor_ids
        assert segment.start_anchor_id in building_anchor_ids
        assert segment.end_anchor_id in building_anchor_ids


def test_terrain_cover_segments_hide_target_without_changing_kind_identity() -> None:
    clock = FakeClock()
    engine = build_rapid_tracking_test(clock=clock, seed=778, difficulty=0.58)
    terrain_index = next(
        idx
        for idx, segment in enumerate(engine._difficulty_profile().scene_script)
        if segment.cover_mode == "terrain"
    )

    engine.start_scored()
    engine._script_index = terrain_index - 1
    engine._start_scene_segment(initial=False)
    engine._sim_elapsed_s = engine._segment_started_s + (engine._segment_duration_s * 0.5)
    engine._advance_target()
    engine._step(0.0)

    payload = engine.snapshot().payload
    assert payload is not None
    assert payload.target_kind in {"soldier", "truck", "helicopter"}
    assert payload.target_cover_state == "terrain"
    assert payload.target_visible is False


def test_smooth_terrain_cover_keeps_truck_identity_while_hidden() -> None:
    clock = FakeClock()
    engine = build_rapid_tracking_test(clock=clock, seed=551, difficulty=0.63)
    engine.start_scored()

    script = engine._difficulty_profile().scene_script
    terrain_index = next(
        idx
        for idx, segment in enumerate(script)
        if segment.kind == "truck"
        and segment.cover_mode == "terrain"
        and idx > 0
        and script[idx - 1].kind == "truck"
        and script[idx - 1].handoff == "smooth"
    )
    previous_index = terrain_index - 1

    engine._script_index = previous_index - 1
    engine._start_scene_segment(initial=False)
    engine._sim_elapsed_s = engine._segment_started_s + (engine._segment_duration_s * 0.98)
    clock.t = engine._phase_started_at_s + engine._sim_elapsed_s
    engine._advance_target()
    engine._step(0.0)
    visible_before = engine.snapshot().payload
    assert visible_before is not None

    engine._script_index = terrain_index - 1
    engine._start_scene_segment(initial=False)
    engine._sim_elapsed_s = engine._segment_started_s
    clock.t = engine._phase_started_at_s + engine._sim_elapsed_s
    engine._advance_target()
    engine._step(0.0)
    hidden_start = engine.snapshot().payload
    assert hidden_start is not None

    engine._sim_elapsed_s = engine._segment_started_s + (engine._segment_duration_s * 0.5)
    clock.t = engine._phase_started_at_s + engine._sim_elapsed_s
    engine._advance_target()
    engine._step(0.0)
    hidden_mid = engine.snapshot().payload
    assert hidden_mid is not None

    assert visible_before.target_kind == "truck"
    assert visible_before.target_visible is True
    assert hidden_start.target_kind == "truck"
    assert hidden_start.target_cover_state == "terrain"
    assert hidden_start.target_visible is False
    assert hidden_mid.target_kind == "truck"
    assert hidden_mid.target_visible is False
    hidden_motion = math.hypot(
        float(hidden_mid.target_world_x) - float(hidden_start.target_world_x),
        float(hidden_mid.target_world_y) - float(hidden_start.target_world_y),
    )

    assert hidden_motion >= 4.0


def test_low_difficulty_limits_handoffs_to_ground_targets_and_enables_assist() -> None:
    clock = FakeClock()
    engine = build_rapid_tracking_test(
        clock=clock,
        seed=121,
        difficulty=0.0,
        config=RapidTrackingConfig(
            practice_duration_s=0.0,
            scored_duration_s=30.0,
            tick_hz=120.0,
        ),
    )

    profile = engine._difficulty_profile()
    assert profile.tier == "low"
    assert profile.camera_assist_strength > 0.0
    assert profile.turbulence_strength == pytest.approx(0.0)
    assert profile.loop_limit == 1
    assert {segment.kind for segment in profile.scene_script} <= {"soldier", "building", "truck"}


def test_mid_difficulty_runs_full_loop_once_without_camera_assist() -> None:
    clock = FakeClock()
    engine = build_rapid_tracking_test(
        clock=clock,
        seed=122,
        difficulty=0.5,
        config=RapidTrackingConfig(
            practice_duration_s=0.0,
            scored_duration_s=30.0,
            tick_hz=120.0,
        ),
    )

    profile = engine._difficulty_profile()
    assert profile.tier == "mid"
    assert profile.camera_assist_strength == pytest.approx(0.0)
    assert profile.loop_limit == 1
    seen_kinds = {segment.kind for segment in profile.scene_script}
    assert {"soldier", "helicopter", "jet"} <= seen_kinds


def test_high_difficulty_allows_three_loops_and_stronger_turbulence() -> None:
    clock = FakeClock()
    engine = build_rapid_tracking_test(
        clock=clock,
        seed=123,
        difficulty=1.0,
        config=RapidTrackingConfig(
            practice_duration_s=0.0,
            scored_duration_s=30.0,
            tick_hz=120.0,
        ),
    )

    high = engine._difficulty_profile()
    mid = build_rapid_tracking_test(
        clock=clock,
        seed=124,
        difficulty=0.5,
        config=RapidTrackingConfig(
            practice_duration_s=0.0,
            scored_duration_s=30.0,
            tick_hz=120.0,
        ),
    )._difficulty_profile()

    assert high.tier == "high"
    assert high.loop_limit == 3
    assert high.duration_scale < mid.duration_scale
    assert high.turbulence_strength > mid.turbulence_strength


def test_loop_limit_stops_after_requested_number_of_loops() -> None:
    clock = FakeClock()
    engine = build_rapid_tracking_test(
        clock=clock,
        seed=125,
        difficulty=1.0,
        config=RapidTrackingConfig(
            practice_duration_s=0.0,
            scored_duration_s=30.0,
            tick_hz=120.0,
        ),
    )

    engine.start_scored()
    profile = engine._difficulty_profile()

    for _ in range(len(profile.scene_script) * profile.loop_limit):
        engine._start_scene_segment(initial=False)

    assert math.isinf(engine._target_switch_at_s)
