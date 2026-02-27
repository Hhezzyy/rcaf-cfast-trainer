from __future__ import annotations

from dataclasses import dataclass

import pytest

from cfast_trainer.cognitive_core import Phase
from cfast_trainer.rapid_tracking import RapidTrackingConfig, build_rapid_tracking_test


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def test_headless_scripted_run_produces_expected_summary() -> None:
    seed = 551
    difficulty = 0.63
    clock = FakeClock()

    engine = build_rapid_tracking_test(
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        config=RapidTrackingConfig(
            practice_duration_s=1.0,
            scored_duration_s=3.0,
            tick_hz=120.0,
        ),
    )

    engine.start_practice()
    assert engine.phase is Phase.PRACTICE

    practice_controls = [(0.30, -0.20), (-0.15, 0.24), (0.06, -0.08), (0.0, 0.0)]
    idx = 0
    while engine.phase is Phase.PRACTICE:
        cx, cy = practice_controls[idx % len(practice_controls)]
        idx += 1
        engine.set_control(horizontal=cx, vertical=cy)
        clock.advance(0.1)
        engine.update()

    assert engine.phase is Phase.PRACTICE_DONE

    engine.start_scored()
    assert engine.phase is Phase.SCORED

    scored_controls = [(0.44, -0.32), (0.18, -0.12), (0.0, 0.0), (-0.22, 0.31), (0.12, -0.16)]
    idx = 0
    for _ in range(30):
        cx, cy = scored_controls[idx % len(scored_controls)]
        idx += 1
        engine.set_control(horizontal=cx, vertical=cy)
        clock.advance(0.1)
        engine.update()

    assert engine.phase is Phase.RESULTS

    summary = engine.scored_summary()
    assert summary.attempted == 3
    assert 0 <= summary.correct <= 3
    assert summary.accuracy == pytest.approx(summary.correct / 3.0)
    assert summary.throughput_per_min == pytest.approx(60.0)
    assert summary.mean_error >= 0.0
    assert summary.rms_error >= summary.mean_error
    assert summary.on_target_s >= 0.0
    assert summary.on_target_ratio >= 0.0
    assert summary.obscured_time_s >= 0.0
    assert 0.0 <= summary.obscured_tracking_ratio <= 1.0
    assert 0.0 <= summary.moving_target_ratio <= 1.0
    assert summary.max_score == pytest.approx(3.0)
