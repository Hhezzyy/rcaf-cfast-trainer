from __future__ import annotations

from dataclasses import dataclass

import pytest

from cfast_trainer.ant_drills import AntDrillMode
from cfast_trainer.cognitive_core import Phase
from cfast_trainer.sma_drills import (
    SmaDrillConfig,
    build_sma_disturbance_tempo_drill,
    build_sma_joystick_hold_run_drill,
    build_sma_joystick_horizontal_anchor_drill,
    build_sma_joystick_vertical_anchor_drill,
    build_sma_mode_switch_run_drill,
    build_sma_overshoot_recovery_drill,
    build_sma_pressure_run_drill,
    build_sma_split_axis_control_drill,
    build_sma_split_coordination_run_drill,
    build_sma_split_horizontal_prime_drill,
)
from cfast_trainer.sensory_motor_apparatus import SensoryMotorApparatusPayload


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _difficulty_for_level(level: int) -> float:
    return float(level - 1) / 9.0


def _run_drill(drill, clock: FakeClock, *, duration_s: float = 3.0) -> tuple[list[tuple], object]:
    drill.start_practice()
    controls = ((0.35, -0.20), (-0.15, 0.25), (0.10, -0.05), (0.0, 0.0))
    payload_frames: list[tuple] = []
    idx = 0
    steps = max(1, int(round(duration_s / 0.25))) + 4
    for _ in range(steps):
        if drill.phase is not Phase.SCORED:
            break
        cx, cy = controls[idx % len(controls)]
        idx += 1
        drill.set_control(horizontal=cx, vertical=cy)
        clock.advance(0.25)
        drill.update()
        snap = drill.snapshot()
        payload = snap.payload
        assert isinstance(payload, SensoryMotorApparatusPayload)
        payload_frames.append(
            (
                snap.phase.value,
                payload.control_mode,
                payload.axis_focus,
                payload.segment_label,
                payload.segment_index,
                payload.dot_x,
                payload.dot_y,
                payload.disturbance_x,
                payload.disturbance_y,
                payload.mean_error,
            )
        )
    return payload_frames, drill.scored_summary()


@pytest.mark.parametrize(
    ("builder", "duration"),
    (
        (build_sma_joystick_horizontal_anchor_drill, 3.0),
        (build_sma_joystick_vertical_anchor_drill, 3.0),
        (build_sma_joystick_hold_run_drill, 3.0),
        (build_sma_split_horizontal_prime_drill, 3.0),
        (build_sma_split_coordination_run_drill, 3.0),
        (build_sma_mode_switch_run_drill, 3.0),
        (build_sma_disturbance_tempo_drill, 3.0),
        (build_sma_pressure_run_drill, 3.0),
        (build_sma_split_axis_control_drill, 3.0),
        (build_sma_overshoot_recovery_drill, 3.0),
    ),
)
def test_sma_drills_are_deterministic_for_same_seed_and_control_script(builder, duration) -> None:
    c1 = FakeClock()
    c2 = FakeClock()
    d1 = builder(
        clock=c1,
        seed=515,
        difficulty=0.55,
        mode=AntDrillMode.BUILD,
        config=SmaDrillConfig(scored_duration_s=duration),
    )
    d2 = builder(
        clock=c2,
        seed=515,
        difficulty=0.55,
        mode=AntDrillMode.BUILD,
        config=SmaDrillConfig(scored_duration_s=duration),
    )

    frames1, summary1 = _run_drill(d1, c1, duration_s=duration)
    frames2, summary2 = _run_drill(d2, c2, duration_s=duration)

    assert frames1 == frames2
    assert summary1 == summary2


@pytest.mark.parametrize(
    ("builder", "control_mode", "axis_focus"),
    (
        (build_sma_joystick_horizontal_anchor_drill, "joystick_only", "horizontal"),
        (build_sma_joystick_vertical_anchor_drill, "joystick_only", "vertical"),
        (build_sma_joystick_hold_run_drill, "joystick_only", "both"),
        (build_sma_split_horizontal_prime_drill, "split", "horizontal"),
        (build_sma_split_coordination_run_drill, "split", "both"),
    ),
)
def test_sma_focused_drills_emit_expected_control_mode_and_axis_focus(
    builder, control_mode: str, axis_focus: str
) -> None:
    clock = FakeClock()
    drill = builder(
        clock=clock,
        seed=91,
        difficulty=0.5,
        mode=AntDrillMode.BUILD,
        config=SmaDrillConfig(scored_duration_s=2.0),
    )
    drill.start_practice()
    snap = drill.snapshot()
    payload = snap.payload
    assert isinstance(payload, SensoryMotorApparatusPayload)
    assert payload.control_mode == control_mode
    assert payload.axis_focus == axis_focus


def test_sma_split_axis_control_repeats_horizontal_then_vertical_split_segments() -> None:
    clock = FakeClock()
    drill = build_sma_split_axis_control_drill(
        clock=clock,
        seed=41,
        difficulty=0.5,
        mode=AntDrillMode.TEMPO,
        config=SmaDrillConfig(scored_duration_s=65.0),
    )
    drill.start_practice()

    observed: list[tuple[str, str]] = []
    for _ in range(140):
        payload = drill.snapshot().payload
        assert isinstance(payload, SensoryMotorApparatusPayload)
        marker = (payload.segment_label, payload.axis_focus)
        if not observed or observed[-1] != marker:
            observed.append(marker)
        clock.advance(0.5)
        drill.update()
        if len(observed) >= 3:
            break

    assert observed[:3] == [
        ("Split Axis - Horizontal", "horizontal"),
        ("Split Axis - Vertical", "vertical"),
        ("Split Axis - Horizontal", "horizontal"),
    ]


def test_sma_mode_switch_run_repeats_joystick_then_split_segments() -> None:
    clock = FakeClock()
    drill = build_sma_mode_switch_run_drill(
        clock=clock,
        seed=17,
        difficulty=0.5,
        mode=AntDrillMode.TEMPO,
        config=SmaDrillConfig(scored_duration_s=125.0),
    )
    drill.start_practice()

    observed: list[tuple[str, str]] = []
    for _ in range(250):
        snap = drill.snapshot()
        payload = snap.payload
        assert isinstance(payload, SensoryMotorApparatusPayload)
        marker = (payload.segment_label, payload.control_mode)
        if not observed or observed[-1] != marker:
            observed.append(marker)
        clock.advance(0.5)
        drill.update()
        if len(observed) >= 3:
            break

    assert observed[:3] == [
        ("Mode Switch - Joystick", "joystick_only"),
        ("Mode Switch - Split", "split"),
        ("Mode Switch - Joystick", "joystick_only"),
    ]


def test_sma_disturbance_tempo_repeats_fixed_thirty_second_cycle() -> None:
    clock = FakeClock()
    drill = build_sma_disturbance_tempo_drill(
        clock=clock,
        seed=27,
        difficulty=0.5,
        mode=AntDrillMode.TEMPO,
        config=SmaDrillConfig(scored_duration_s=125.0),
    )
    drill.start_practice()

    observed: list[str] = []
    for _ in range(260):
        snap = drill.snapshot()
        payload = snap.payload
        assert isinstance(payload, SensoryMotorApparatusPayload)
        if not observed or observed[-1] != payload.segment_label:
            observed.append(payload.segment_label)
        clock.advance(0.5)
        drill.update()
        if len(observed) >= 5:
            break

    assert observed[:5] == [
        "Tempo - Joystick Steady",
        "Tempo - Joystick Pulse",
        "Tempo - Split Steady",
        "Tempo - Split Pulse",
        "Tempo - Joystick Steady",
    ]


def test_sma_pressure_run_alternates_control_modes_with_pressure_profile() -> None:
    clock = FakeClock()
    drill = build_sma_pressure_run_drill(
        clock=clock,
        seed=37,
        difficulty=0.5,
        mode=AntDrillMode.STRESS,
        config=SmaDrillConfig(scored_duration_s=65.0),
    )
    drill.start_practice()

    seen: list[tuple[str, str]] = []
    for _ in range(140):
        snap = drill.snapshot()
        payload = snap.payload
        assert isinstance(payload, SensoryMotorApparatusPayload)
        marker = (payload.segment_label, payload.control_mode)
        if not seen or seen[-1] != marker:
            seen.append(marker)
        clock.advance(0.5)
        drill.update()
        if len(seen) >= 3:
            break

    assert seen[:3] == [
        ("Pressure - Joystick", "joystick_only"),
        ("Pressure - Split", "split"),
        ("Pressure - Joystick", "joystick_only"),
    ]
    assert all("Pressure" in label for label, _mode in seen[:3])


@pytest.mark.parametrize(
    "builder",
    (
        build_sma_split_axis_control_drill,
        build_sma_overshoot_recovery_drill,
    ),
)
def test_wave2_psychomotor_drills_levels_l2_l5_l8_are_materially_different(builder) -> None:
    def summarize(level: int) -> tuple[float, float, float, float]:
        clock = FakeClock()
        drill = builder(
            clock=clock,
            seed=919,
            difficulty=_difficulty_for_level(level),
            mode=AntDrillMode.BUILD,
            config=SmaDrillConfig(scored_duration_s=4.0),
        )
        drill.start_practice()
        payload = drill.snapshot().payload
        assert isinstance(payload, SensoryMotorApparatusPayload)
        frames, _summary = _run_drill(drill, clock, duration_s=2.5)
        mean_abs_disturbance = sum(abs(frame[7]) + abs(frame[8]) for frame in frames) / len(frames)
        engine_cfg = drill._engine._config
        return (
            float(engine_cfg.control_gain),
            float(engine_cfg.on_target_radius),
            float(engine_cfg.guide_band_half_width),
            float(mean_abs_disturbance),
        )

    low_gain, low_radius, low_guide, low_disturbance = summarize(2)
    mid_gain, mid_radius, mid_guide, mid_disturbance = summarize(5)
    high_gain, high_radius, high_guide, high_disturbance = summarize(8)

    assert low_gain < mid_gain < high_gain
    assert low_radius > mid_radius > high_radius
    assert low_guide > mid_guide > high_guide
    assert low_disturbance < mid_disturbance < high_disturbance
