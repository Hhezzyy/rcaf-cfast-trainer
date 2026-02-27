from __future__ import annotations

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
