from __future__ import annotations

from dataclasses import dataclass

import pytest

from cfast_trainer.cognitive_core import Phase
from cfast_trainer.trace_test_1 import (
    TraceTest1Config,
    TraceTest1Payload,
    TraceTest1TrialStage,
    build_trace_test_1_test,
)


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


def _advance_to_answer_open(engine, clock: FakeClock, *, dt: float = 0.16) -> TraceTest1Payload:
    clock.advance(dt)
    engine.update()
    payload = engine.snapshot().payload
    assert isinstance(payload, TraceTest1Payload)
    assert payload.trial_stage is TraceTest1TrialStage.ANSWER_OPEN
    return payload


def test_headless_scripted_run_produces_expected_summary() -> None:
    difficulty = 0.61
    clock = FakeClock()

    engine = build_trace_test_1_test(
        clock=clock,
        seed=5150,
        difficulty=difficulty,
        config=TraceTest1Config(
            scored_duration_s=6.0,
            practice_questions=1,
            practice_observe_s=0.4,
            scored_observe_s=0.4,
        ),
    )

    engine.start_practice()
    assert engine.phase is Phase.PRACTICE

    payload = _advance_to_answer_open(engine, clock)
    assert engine.submit_answer(str(payload.correct_code)) is True
    assert engine.phase is Phase.PRACTICE_DONE

    engine.start_scored()
    assert engine.phase is Phase.SCORED

    p1 = _advance_to_answer_open(engine, clock)
    assert engine.submit_answer(str(p1.correct_code)) is True

    p2 = _advance_to_answer_open(engine, clock)
    wrong = 1 if int(p2.correct_code) != 1 else 2
    assert engine.submit_answer(str(wrong)) is True

    p3 = _advance_to_answer_open(engine, clock)
    assert engine.submit_answer(str(p3.correct_code)) is True

    clock.advance(6.0)
    engine.update()
    assert engine.phase is Phase.RESULTS

    summary = engine.scored_summary()
    assert summary.attempted == 3
    assert summary.correct == 2
    assert summary.accuracy == pytest.approx(2.0 / 3.0)
    assert summary.mean_response_time_s == pytest.approx(0.016, abs=0.02)
    assert summary.total_score == pytest.approx(2.0)
    assert summary.max_score == pytest.approx(3.0)
    assert summary.score_ratio == pytest.approx(2.0 / 3.0)
