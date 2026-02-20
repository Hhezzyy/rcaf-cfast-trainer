from __future__ import annotations

from dataclasses import dataclass

import pytest

from cfast_trainer.auditory_capacity import (
    AuditoryCapacityConfig,
    AuditoryCapacityScenarioGenerator,
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
