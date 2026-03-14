from __future__ import annotations

from dataclasses import dataclass

import pytest

from cfast_trainer.cognitive_core import Phase
from cfast_trainer.sensory_motor_apparatus import (
    SensoryMotorApparatusConfig,
    build_sensory_motor_apparatus_test,
)


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def test_headless_scripted_run_produces_expected_summary() -> None:
    seed = 404
    difficulty = 0.6
    clock = FakeClock()

    engine = build_sensory_motor_apparatus_test(
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        config=SensoryMotorApparatusConfig(
            practice_duration_s=1.0,
            scored_duration_s=3.0,
            tick_hz=120.0,
        ),
    )

    engine.start_practice()
    assert engine.phase is Phase.PRACTICE

    practice_controls = [(0.35, -0.20), (-0.15, 0.30), (0.10, -0.05), (0.0, 0.0)]
    idx = 0
    while engine.phase is Phase.PRACTICE:
        cx, cy = practice_controls[idx % len(practice_controls)]
        idx += 1
        engine.set_control(horizontal=cx, vertical=cy)
        clock.advance(0.1)
        engine.update()

    assert engine.phase is Phase.PRACTICE_DONE
    payload = engine.snapshot().payload
    assert payload is not None
    assert payload.block_kind == "practice"
    assert payload.block_index == 2
    assert payload.control_mode == "split"

    engine.start_scored()
    assert engine.phase is Phase.PRACTICE

    while engine.phase is Phase.PRACTICE:
        cx, cy = practice_controls[idx % len(practice_controls)]
        idx += 1
        engine.set_control(horizontal=cx, vertical=cy)
        clock.advance(0.1)
        engine.update()

    assert engine.phase is Phase.PRACTICE_DONE
    payload = engine.snapshot().payload
    assert payload is not None
    assert payload.block_kind == "scored"
    assert payload.block_index == 1
    assert payload.control_mode == "joystick_only"

    engine.start_scored()
    assert engine.phase is Phase.SCORED

    scored_controls = [
        (0.40, -0.35),
        (0.20, -0.10),
        (0.0, 0.0),
        (-0.25, 0.30),
        (0.15, -0.15),
    ]
    idx = 0
    while engine.phase is Phase.SCORED:
        cx, cy = scored_controls[idx % len(scored_controls)]
        idx += 1
        engine.set_control(horizontal=cx, vertical=cy)
        clock.advance(0.1)
        engine.update()

    assert engine.phase is Phase.PRACTICE_DONE
    payload = engine.snapshot().payload
    assert payload is not None
    assert payload.block_kind == "scored"
    assert payload.block_index == 2
    assert payload.control_mode == "split"

    engine.start_scored()
    assert engine.phase is Phase.SCORED

    while engine.phase is Phase.SCORED:
        cx, cy = scored_controls[idx % len(scored_controls)]
        idx += 1
        engine.set_control(horizontal=cx, vertical=cy)
        clock.advance(0.1)
        engine.update()

    assert engine.phase is Phase.RESULTS

    summary = engine.scored_summary()
    assert summary.attempted == 3
    assert summary.correct >= 0
    assert 0.0 <= summary.accuracy <= 1.0
    assert summary.throughput_per_min == pytest.approx(60.0)
    assert summary.mean_error > 0.0
    assert summary.rms_error >= summary.mean_error
    assert summary.on_target_s >= 0.0
    assert 0.0 <= summary.on_target_ratio <= 1.0
    assert summary.total_score >= 0.0
    assert summary.max_score == pytest.approx(3.0)
    assert 0.0 <= summary.score_ratio <= 1.0
