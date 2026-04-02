from __future__ import annotations

from dataclasses import dataclass

import pytest

from cfast_trainer.ant_drills import AntDrillMode
from cfast_trainer.cognitive_core import Phase
from cfast_trainer.rt_drills import (
    RtDrillConfig,
    build_rt_air_speed_run_drill,
    build_rt_building_handoff_prime_drill,
    build_rt_capture_timing_prime_drill,
    build_rt_ground_tempo_run_drill,
    build_rt_lock_anchor_drill,
    build_rt_mixed_tempo_drill,
    build_rt_obscured_target_prediction_drill,
    build_rt_pressure_run_drill,
    build_rt_terrain_recovery_run_drill,
)
from cfast_trainer.rapid_tracking import RapidTrackingPayload
from cfast_trainer.rapid_tracking import RapidTrackingLayoutPolicy


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


def _difficulty_for_level(level: int) -> float:
    return float(level - 1) / 9.0


def _run_drill(drill, clock: FakeClock, *, duration_s: float) -> tuple[list[tuple], object]:
    drill.start_practice()
    controls = ((0.22, -0.12), (0.08, 0.0), (-0.16, 0.18), (0.0, 0.0))
    frames: list[tuple] = []
    steps = max(1, int(round(duration_s / 0.5))) + 4
    for idx in range(steps):
        if drill.phase is not Phase.SCORED:
            break
        cx, cy = controls[idx % len(controls)]
        drill.set_control(horizontal=cx, vertical=cy)
        snap = drill.snapshot()
        payload = snap.payload
        assert isinstance(payload, RapidTrackingPayload)
        if payload.target_in_capture_box and idx % 3 == 0:
            assert drill.submit_answer("CAPTURE") is True
        frames.append(
            (
                snap.phase.value,
                payload.active_target_kinds,
                payload.active_challenges,
                payload.segment_label,
                payload.focus_label,
                payload.target_kind,
                payload.target_cover_state,
                payload.capture_points,
                payload.capture_attempts,
            )
        )
        clock.advance(0.5)
        drill.update()
    return frames, drill.scored_summary()


@pytest.mark.parametrize(
    "builder",
    (
        build_rt_lock_anchor_drill,
        build_rt_building_handoff_prime_drill,
        build_rt_terrain_recovery_run_drill,
        build_rt_capture_timing_prime_drill,
        build_rt_ground_tempo_run_drill,
        build_rt_air_speed_run_drill,
        build_rt_mixed_tempo_drill,
        build_rt_pressure_run_drill,
        build_rt_obscured_target_prediction_drill,
    ),
)
def test_rt_drills_are_deterministic_for_same_seed_and_controls(builder) -> None:
    c1 = FakeClock()
    c2 = FakeClock()
    d1 = builder(
        clock=c1,
        seed=515,
        difficulty=0.55,
        mode=AntDrillMode.BUILD,
        config=RtDrillConfig(scored_duration_s=18.0),
    )
    d2 = builder(
        clock=c2,
        seed=515,
        difficulty=0.55,
        mode=AntDrillMode.BUILD,
        config=RtDrillConfig(scored_duration_s=18.0),
    )

    frames1, summary1 = _run_drill(d1, c1, duration_s=18.0)
    frames2, summary2 = _run_drill(d2, c2, duration_s=18.0)

    assert frames1 == frames2
    assert summary1 == summary2


@pytest.mark.parametrize(
    ("builder", "expected_targets", "expected_challenges"),
    (
        (build_rt_lock_anchor_drill, ("soldier", "truck"), ("lock_quality",)),
        (build_rt_building_handoff_prime_drill, ("building", "soldier", "truck"), ("handoff_reacquisition",)),
        (
            build_rt_terrain_recovery_run_drill,
            ("soldier", "truck", "helicopter"),
            ("occlusion_recovery", "handoff_reacquisition"),
        ),
        (build_rt_capture_timing_prime_drill, ("soldier", "truck", "helicopter"), ("capture_timing",)),
        (build_rt_ground_tempo_run_drill, ("soldier", "truck"), ("ground_tempo", "lock_quality")),
        (build_rt_air_speed_run_drill, ("helicopter", "jet"), ("air_speed", "capture_timing")),
        (
            build_rt_obscured_target_prediction_drill,
            ("soldier", "truck", "helicopter"),
            ("occlusion_recovery",),
        ),
    ),
)
def test_rt_focused_drills_emit_expected_target_kinds_and_challenges(
    builder,
    expected_targets,
    expected_challenges,
) -> None:
    clock = FakeClock()
    drill = builder(
        clock=clock,
        seed=91,
        difficulty=0.5,
        mode=AntDrillMode.BUILD,
        config=RtDrillConfig(scored_duration_s=18.0),
    )
    drill.start_practice()
    payload = drill.snapshot().payload
    assert isinstance(payload, RapidTrackingPayload)
    assert payload.active_target_kinds == expected_targets
    assert payload.active_challenges == expected_challenges


def test_rt_mixed_tempo_repeats_fixed_six_segment_cycle() -> None:
    clock = FakeClock()
    drill = build_rt_mixed_tempo_drill(
        clock=clock,
        seed=17,
        difficulty=0.5,
        mode=AntDrillMode.TEMPO,
        config=RtDrillConfig(scored_duration_s=390.0),
    )
    drill.start_practice()

    observed: list[str] = []
    for _ in range(900):
        payload = drill.snapshot().payload
        assert isinstance(payload, RapidTrackingPayload)
        if not observed or observed[-1] != payload.segment_label:
            observed.append(payload.segment_label)
        if len(observed) >= 7:
            break
        clock.advance(0.5)
        drill.update()

    assert observed[:7] == (
        [
            "Lock Anchor",
            "Building Handoff",
            "Terrain Recovery",
            "Capture Timing",
            "Ground Tempo",
            "Air Speed",
            "Lock Anchor",
        ]
    )


def test_rt_pressure_run_keeps_all_targets_and_challenges_active() -> None:
    clock = FakeClock()
    drill = build_rt_pressure_run_drill(
        clock=clock,
        seed=37,
        difficulty=0.5,
        mode=AntDrillMode.STRESS,
        config=RtDrillConfig(scored_duration_s=12.0),
    )
    drill.start_practice()

    for _ in range(4):
        payload = drill.snapshot().payload
        assert isinstance(payload, RapidTrackingPayload)
        assert payload.active_target_kinds == ("soldier", "building", "truck", "helicopter", "jet")
        assert payload.active_challenges == (
            "lock_quality",
            "handoff_reacquisition",
            "occlusion_recovery",
            "capture_timing",
            "ground_tempo",
            "air_speed",
        )
        clock.advance(0.5)
        drill.update()


def test_rt_mixed_and_pressure_drills_use_readable_balanced_layout_policy() -> None:
    clock = FakeClock()
    mixed = build_rt_mixed_tempo_drill(
        clock=clock,
        seed=91,
        difficulty=0.5,
        mode=AntDrillMode.TEMPO,
        config=RtDrillConfig(scored_duration_s=18.0),
    )
    pressure = build_rt_pressure_run_drill(
        clock=clock,
        seed=91,
        difficulty=0.5,
        mode=AntDrillMode.STRESS,
        config=RtDrillConfig(scored_duration_s=18.0),
    )
    default_drill = build_rt_lock_anchor_drill(
        clock=clock,
        seed=91,
        difficulty=0.5,
        mode=AntDrillMode.BUILD,
        config=RtDrillConfig(scored_duration_s=18.0),
    )

    mixed.start_practice()
    pressure.start_practice()
    default_drill.start_practice()

    mixed_payload = mixed.snapshot().payload
    pressure_payload = pressure.snapshot().payload
    default_payload = default_drill.snapshot().payload

    assert mixed.layout_policy is RapidTrackingLayoutPolicy.READABLE_BALANCED
    assert pressure.layout_policy is RapidTrackingLayoutPolicy.READABLE_BALANCED
    assert default_drill.layout_policy is RapidTrackingLayoutPolicy.DEFAULT
    assert isinstance(mixed_payload, RapidTrackingPayload)
    assert isinstance(pressure_payload, RapidTrackingPayload)
    assert isinstance(default_payload, RapidTrackingPayload)
    assert mixed_payload.scene_seed != mixed_payload.session_seed
    assert pressure_payload.scene_seed != pressure_payload.session_seed
    assert default_payload.scene_seed == default_payload.session_seed


def test_rt_obscured_target_prediction_levels_l2_l5_l8_are_materially_different() -> None:
    def summarize(level: int) -> tuple[float, float, float]:
        clock = FakeClock()
        drill = build_rt_obscured_target_prediction_drill(
            clock=clock,
            seed=1001,
            difficulty=_difficulty_for_level(level),
            mode=AntDrillMode.BUILD,
            config=RtDrillConfig(scored_duration_s=18.0),
        )
        drill.start_practice()
        payload = drill.snapshot().payload
        assert isinstance(payload, RapidTrackingPayload)
        return (
            float(payload.turbulence_strength),
            float(payload.target_time_to_switch_s),
            float(payload.target_switch_preview_s),
        )

    low_turbulence, low_switch_time, low_preview = summarize(2)
    mid_turbulence, mid_switch_time, mid_preview = summarize(5)
    high_turbulence, high_switch_time, high_preview = summarize(8)

    assert low_turbulence < mid_turbulence < high_turbulence
    assert low_switch_time > mid_switch_time > high_switch_time
    assert low_preview == pytest.approx(mid_preview)
    assert mid_preview == pytest.approx(high_preview)
