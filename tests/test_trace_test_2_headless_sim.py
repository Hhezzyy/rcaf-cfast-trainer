from __future__ import annotations

from dataclasses import dataclass

import pytest

from cfast_trainer.cognitive_core import Phase
from cfast_trainer.trace_test_2 import (
    TraceTest2Config,
    TraceTest2Generator,
    TraceTest2Payload,
    build_trace_test_2_test,
)


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


def test_headless_scripted_run_produces_expected_summary() -> None:
    seed = 440
    difficulty = 0.58
    clock = FakeClock()

    engine = build_trace_test_2_test(
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        config=TraceTest2Config(
            scored_duration_s=8.0,
            practice_questions=1,
            practice_observe_s=0.5,
            scored_observe_s=0.5,
        ),
    )
    mirror = TraceTest2Generator(seed=seed)

    engine.start_practice()
    assert engine.phase is Phase.PRACTICE
    practice_problem = mirror.next_problem(difficulty=difficulty)
    clock.advance(0.5)
    engine.update()
    practice_payload = engine.snapshot().payload
    assert isinstance(practice_payload, TraceTest2Payload)
    assert engine.submit_answer(str(practice_problem.answer)) is True
    assert engine.phase is Phase.PRACTICE_DONE

    engine.start_scored()
    assert engine.phase is Phase.SCORED

    p1 = mirror.next_problem(difficulty=difficulty)
    clock.advance(0.5)
    engine.update()
    assert engine.submit_answer(str(p1.answer)) is True

    p2 = mirror.next_problem(difficulty=difficulty)
    clock.advance(0.5)
    engine.update()
    wrong = 1 if int(p2.answer) != 1 else 2
    assert engine.submit_answer(str(wrong)) is True

    clock.advance(8.0)
    engine.update()
    assert engine.phase is Phase.RESULTS

    summary = engine.scored_summary()
    assert summary.attempted == 2
    assert summary.correct == 1
    assert summary.accuracy == pytest.approx(0.5)
    assert summary.mean_response_time_s == pytest.approx(0.5)
    assert summary.total_score == pytest.approx(1.0)
    assert summary.max_score == pytest.approx(2.0)
    assert summary.score_ratio == pytest.approx(0.5)
