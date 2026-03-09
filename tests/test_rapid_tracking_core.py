from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from cfast_trainer.rapid_tracking import (
    RapidTrackingConfig,
    RapidTrackingDriftGenerator,
    build_rapid_tracking_test,
    score_window,
)


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
    engine._camera_x = 0.0
    engine._camera_y = 0.0
    engine._target_kind = "jet"
    engine._target_is_moving = True
    engine._target_terrain_occluded = False

    assert engine.submit_answer("CAPTURE") is True

    payload = engine.snapshot().payload
    assert payload is not None
    assert payload.target_in_capture_box is True
    assert payload.capture_zoom > 0.0
    assert payload.capture_hits == 1
    assert payload.capture_attempts == 1
    assert payload.capture_points == 2
    assert engine.scored_summary().capture_points == 2


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
    engine._camera_x = 0.0
    engine._camera_y = 0.0
    engine._target_kind = "truck"
    engine._target_is_moving = False
    engine._target_terrain_occluded = False

    assert engine.submit_answer("CAPTURE") is True
    assert engine.submit_answer("CAPTURE") is False

    summary = engine.scored_summary()
    assert summary.capture_attempts == 1
    assert summary.capture_hits == 1


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

    seen_kinds = {payload.target_kind}
    seen_handoffs = {payload.target_handoff_mode}

    for _ in range(len(engine._SCENE_SCRIPT) - 1):
        engine._start_scene_segment(initial=False)
        payload = engine.snapshot().payload
        assert payload is not None
        seen_kinds.add(payload.target_kind)
        seen_handoffs.add(payload.target_handoff_mode)

    assert {"soldier", "building", "truck", "helicopter", "jet"} <= seen_kinds
    assert {"smooth", "jump"} <= seen_handoffs


def test_target_speed_order_matches_brief() -> None:
    clock = FakeClock()
    engine = build_rapid_tracking_test(
        clock=clock,
        seed=901,
        difficulty=0.52,
        config=RapidTrackingConfig(
            practice_duration_s=0.0,
            scored_duration_s=10.0,
            tick_hz=120.0,
        ),
    )

    engine.start_scored()

    def sampled_speed(kind: str) -> float:
        for idx, segment in enumerate(engine._SCENE_SCRIPT):
            if segment.kind != kind:
                continue
            if idx == 0:
                engine._start_scene_segment(initial=True)
            else:
                engine._script_index = idx - 1
                engine._start_scene_segment(initial=False)
            engine._sim_elapsed_s = engine._segment_started_s + (engine._segment_duration_s * 0.5)
            engine._advance_target()
            return math.hypot(engine._target_vx, engine._target_vy)
        raise AssertionError(f"missing segment for {kind}")

    soldier_speed = sampled_speed("soldier")
    building_speed = sampled_speed("building")
    truck_speed = sampled_speed("truck")
    helicopter_speed = sampled_speed("helicopter")
    jet_speed = sampled_speed("jet")

    assert building_speed == pytest.approx(0.0)
    assert soldier_speed < truck_speed < helicopter_speed < jet_speed


def test_truck_motion_stays_axis_aligned_without_diagonal_drift() -> None:
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

    truck_segment = next(segment for segment in engine._SCENE_SCRIPT if segment.kind == "truck")
    truck_index = next(
        idx for idx, segment in enumerate(engine._SCENE_SCRIPT) if segment.kind == "truck"
    )
    engine._script_index = truck_index - 1
    engine._start_scene_segment(initial=False)

    dominant_axis_x = abs(truck_segment.end_x - truck_segment.start_x) >= abs(
        truck_segment.end_y - truck_segment.start_y
    )

    sampled_positions: list[tuple[float, float]] = []
    sampled_velocities: list[tuple[float, float]] = []
    for ratio in (0.15, 0.35, 0.55, 0.75):
        engine._sim_elapsed_s = engine._segment_started_s + (engine._segment_duration_s * ratio)
        engine._advance_target()
        sampled_positions.append((engine._target_x, engine._target_y))
        sampled_velocities.append((engine._target_vx, engine._target_vy))

    if dominant_axis_x:
        anchor_y = sampled_positions[0][1]
        assert max(abs(y - anchor_y) for _, y in sampled_positions) < 1e-6
        assert max(abs(vy) for _, vy in sampled_velocities) < 1e-6
        assert max(abs(vx) for vx, _ in sampled_velocities) > 0.01
    else:
        anchor_x = sampled_positions[0][0]
        assert max(abs(x - anchor_x) for x, _ in sampled_positions) < 1e-6
        assert max(abs(vx) for vx, _ in sampled_velocities) < 1e-6
        assert max(abs(vy) for _, vy in sampled_velocities) > 0.01


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
