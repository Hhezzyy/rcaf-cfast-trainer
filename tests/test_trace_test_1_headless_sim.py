from __future__ import annotations

from dataclasses import dataclass

import pytest

from cfast_trainer.cognitive_core import Phase
from cfast_trainer.trace_test_1 import (
    TraceTest1Config,
    TraceTest1Generator,
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


def test_headless_scripted_run_produces_expected_summary() -> None:
    seed = 5150
    difficulty = 0.61
    clock = FakeClock()

    engine = build_trace_test_1_test(
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        config=TraceTest1Config(
            scored_duration_s=6.0,
            practice_questions=1,
            practice_observe_s=0.2,
            scored_observe_s=0.2,
        ),
    )
    mirror = TraceTest1Generator(seed=seed)

    engine.start_practice()
    assert engine.phase is Phase.PRACTICE

    practice_problem = mirror.next_problem(difficulty=difficulty)
    practice_payload = engine.snapshot().payload
    assert isinstance(practice_payload, TraceTest1Payload)
    assert practice_payload.trial_stage is TraceTest1TrialStage.QUESTION
    clock.advance(0.5)
    assert engine.submit_answer(str(practice_problem.answer)) is True
    assert engine.phase is Phase.PRACTICE_DONE

    engine.start_scored()
    assert engine.phase is Phase.SCORED

    p1 = mirror.next_problem(difficulty=difficulty)
    clock.advance(0.5)
    assert engine.submit_answer(str(p1.answer)) is True

    p2 = mirror.next_problem(difficulty=difficulty)
    wrong = 1 if int(p2.answer) != 1 else 2
    clock.advance(0.5)
    assert engine.submit_answer(str(wrong)) is True

    p3 = mirror.next_problem(difficulty=difficulty)
    clock.advance(0.5)
    assert engine.submit_answer(str(p3.answer)) is True

    clock.advance(6.0)
    engine.update()
    assert engine.phase is Phase.RESULTS

    summary = engine.scored_summary()
    assert summary.attempted == 3
    assert summary.correct == 2
    assert summary.accuracy == pytest.approx(2.0 / 3.0)
    assert summary.mean_response_time_s == pytest.approx(0.5)
    assert summary.total_score == pytest.approx(2.0)
    assert summary.max_score == pytest.approx(3.0)
    assert summary.score_ratio == pytest.approx(2.0 / 3.0)
