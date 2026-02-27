from __future__ import annotations

from dataclasses import dataclass

import pytest

from cfast_trainer.auditory_capacity import (
    AuditoryCapacityColorRule,
    AuditoryCapacityConfig,
    AuditoryCapacityScenarioGenerator,
    _CueState,
    _LiveGate,
    _SequenceState,
    build_auditory_capacity_test,
    score_sequence_answer,
)
from cfast_trainer.cognitive_core import Phase


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


def test_generator_determinism_same_seed_same_sequence() -> None:
    g1 = AuditoryCapacityScenarioGenerator(seed=909)
    g2 = AuditoryCapacityScenarioGenerator(seed=909)

    callsigns_1 = g1.assign_callsigns()
    callsigns_2 = g2.assign_callsigns()
    assert callsigns_1 == callsigns_2

    current_1 = "RED"
    current_2 = "RED"
    last_1: str | None = None
    last_2: str | None = None

    seq1 = []
    seq2 = []
    for _ in range(20):
        step_1 = (
            g1.next_callsign_cue(difficulty=0.62),
            g1.next_sequence(difficulty=0.62),
            g1.next_gate(difficulty=0.62),
            g1.next_disturbance(difficulty=0.62),
            g1.jittered_interval(base_s=1.8, difficulty=0.62),
            g1.build_color_rules(),
        )
        seq1.append(step_1)

        step_2 = (
            g2.next_callsign_cue(difficulty=0.62),
            g2.next_sequence(difficulty=0.62),
            g2.next_gate(difficulty=0.62),
            g2.next_disturbance(difficulty=0.62),
            g2.jittered_interval(base_s=1.8, difficulty=0.62),
            g2.build_color_rules(),
        )
        seq2.append(step_2)

        cmd_1 = g1.next_color_command(current_color=current_1, last_command=last_1)
        cmd_2 = g2.next_color_command(current_color=current_2, last_command=last_2)
        assert cmd_1 == cmd_2
        last_1, last_2 = cmd_1, cmd_2
        current_1, current_2 = cmd_1, cmd_2

    assert seq1 == seq2


def test_sequence_scoring_exact_partial_and_zero() -> None:
    assert score_sequence_answer("12345", "12345") == 1.0
    assert score_sequence_answer("12345", "12335") == pytest.approx(0.8)
    assert score_sequence_answer("12345", "99999") == 0.0


def test_timer_boundary_transitions_to_results_and_rejects_submit() -> None:
    clock = FakeClock()
    engine = build_auditory_capacity_test(
        clock=clock,
        seed=7,
        difficulty=0.5,
        config=AuditoryCapacityConfig(
            practice_duration_s=0.0,
            scored_duration_s=2.0,
            tick_hz=120.0,
        ),
    )

    engine.start_practice()
    assert engine.phase is Phase.PRACTICE_DONE

    engine.start_scored()
    assert engine.phase is Phase.SCORED
    assert engine.can_exit() is False

    clock.advance(2.0)
    engine.update()

    assert engine.phase is Phase.RESULTS
    assert engine.can_exit() is True
    assert engine.submit_answer("BEEP") is False


def test_false_alarm_counts_as_attempted_in_scored_phase() -> None:
    clock = FakeClock()
    engine = build_auditory_capacity_test(
        clock=clock,
        seed=23,
        difficulty=0.5,
        config=AuditoryCapacityConfig(
            practice_duration_s=0.0,
            scored_duration_s=10.0,
            tick_hz=120.0,
        ),
    )

    engine.start_practice()
    engine.start_scored()

    # No cue active at t=0. False response should be scored as a false alarm.
    assert engine.submit_answer("BEEP") is True

    summary = engine.scored_summary()
    assert summary.attempted == 1
    assert summary.correct == 0
    assert summary.total_score == 0.0


def test_background_noise_ramps_up_while_distortion_stays_disabled() -> None:
    clock = FakeClock()
    engine = build_auditory_capacity_test(
        clock=clock,
        seed=77,
        difficulty=0.6,
        config=AuditoryCapacityConfig(
            practice_duration_s=0.0,
            scored_duration_s=20.0,
            tick_hz=120.0,
        ),
    )

    engine.start_practice()
    engine.start_scored()
    snap_start = engine.snapshot()
    assert snap_start.payload is not None
    noise_start = snap_start.payload.background_noise_level
    dist_start = snap_start.payload.distortion_level

    clock.advance(10.0)
    engine.update()
    snap_mid = engine.snapshot()
    assert snap_mid.payload is not None
    noise_mid = snap_mid.payload.background_noise_level
    dist_mid = snap_mid.payload.distortion_level

    clock.advance(9.0)
    engine.update()
    snap_late = engine.snapshot()
    assert snap_late.payload is not None
    noise_late = snap_late.payload.background_noise_level
    dist_late = snap_late.payload.distortion_level

    assert noise_start < noise_mid < noise_late
    assert dist_start == pytest.approx(0.0)
    assert dist_mid == pytest.approx(0.0)
    assert dist_late == pytest.approx(0.0)


def test_gate_logic_uses_current_color_shape_rule_and_hold_override() -> None:
    clock = FakeClock()
    engine = build_auditory_capacity_test(
        clock=clock,
        seed=99,
        difficulty=0.5,
        config=AuditoryCapacityConfig(
            practice_duration_s=0.0,
            scored_duration_s=30.0,
            tick_hz=120.0,
        ),
    )
    engine.start_practice()
    engine.start_scored()
    assert engine.phase is Phase.SCORED

    # Stabilize state for deterministic gate resolution.
    engine._next_gate_at_s = 10_000.0
    engine._ball_x = 0.0
    engine._ball_y = 0.0
    engine._outside_tube = False
    engine._ball_color = "RED"
    engine._rules = (
        AuditoryCapacityColorRule(color="RED", required_shape="SQUARE"),
        AuditoryCapacityColorRule(color="GREEN", required_shape="CIRCLE"),
        AuditoryCapacityColorRule(color="BLUE", required_shape="TRIANGLE"),
        AuditoryCapacityColorRule(color="YELLOW", required_shape="SQUARE"),
    )

    # BLUE/SQUARE should be passed (shape matches RED rule) even with non-matching color.
    gate_pass = _LiveGate(
        gate_id=1,
        x_norm=0.0,
        y_norm=0.0,
        color="BLUE",
        shape="SQUARE",
        aperture_norm=0.2,
        scored=False,
    )
    # RED/TRIANGLE should be avoided (shape mismatch) even with matching color.
    gate_avoid = _LiveGate(
        gate_id=2,
        x_norm=0.0,
        y_norm=0.0,
        color="RED",
        shape="TRIANGLE",
        aperture_norm=0.2,
        scored=False,
    )
    engine._gates = [gate_pass, gate_avoid]
    engine._update_gates(0.0)

    scored_gate_events = [
        evt for evt in engine.events() if evt.kind.value == "gate" and evt.phase is Phase.SCORED
    ]
    assert len(scored_gate_events) >= 2
    last_two = scored_gate_events[-2:]
    assert [evt.is_correct for evt in last_two] == [True, False]
    assert engine._gate_hits == 1
    assert engine._gate_misses == 1

    # HOLD cue overrides default pass logic for current-color gates.
    engine._active_callsign = _CueState(
        target="EAGLE",
        issued_at_s=engine._sim_elapsed_s,
        expires_at_s=engine._sim_elapsed_s + 2.0,
        expects_response=True,
        hold_current_color_gate=True,
    )
    hold_gate = _LiveGate(
        gate_id=3,
        x_norm=0.0,
        y_norm=0.0,
        color="RED",
        shape="SQUARE",
        aperture_norm=0.2,
        scored=False,
    )
    engine._gates = [hold_gate]
    engine._update_gates(0.0)

    latest_gate = [
        evt for evt in engine.events() if evt.kind.value == "gate" and evt.phase is Phase.SCORED
    ][-1]
    assert latest_gate.is_correct is False
    assert "HOLD" in latest_gate.expected
    assert engine._gate_misses == 2


def test_submit_answer_accepts_direct_color_and_direct_digits() -> None:
    clock = FakeClock()
    engine = build_auditory_capacity_test(
        clock=clock,
        seed=111,
        difficulty=0.5,
        config=AuditoryCapacityConfig(
            practice_duration_s=0.0,
            scored_duration_s=20.0,
            tick_hz=120.0,
        ),
    )
    engine.start_practice()
    engine.start_scored()
    assert engine.phase is Phase.SCORED

    engine._ball_color = "RED"
    engine._active_color = _CueState(
        target="GREEN",
        issued_at_s=0.0,
        expires_at_s=5.0,
        expects_response=True,
    )
    assert engine.submit_answer("W") is True
    assert engine._ball_color == "GREEN"

    engine._active_sequence = _SequenceState(
        target="4831",
        show_until_s=0.0,
        expire_at_s=5.0,
    )
    assert engine.submit_answer("4831") is True
    latest = engine.events()[-1]
    assert latest.kind.value == "sequence"
    assert latest.is_correct is True
