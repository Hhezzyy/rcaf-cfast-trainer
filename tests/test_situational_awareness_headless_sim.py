from __future__ import annotations

from dataclasses import dataclass

import pytest

from cfast_trainer.cognitive_core import Phase
from cfast_trainer.situational_awareness import (
    SituationalAwarenessConfig,
    SituationalAwarenessGenerator,
    build_situational_awareness_test,
)


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


def test_headless_scripted_run_tracks_scored_attempts_and_partial_credit() -> None:
    seed = 5151
    difficulty = 0.65
    clock = FakeClock()

    engine = build_situational_awareness_test(
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        config=SituationalAwarenessConfig(scored_duration_s=6.0, practice_questions=1),
    )
    mirror = SituationalAwarenessGenerator(seed=seed)

    engine.start_practice()
    assert engine.phase is Phase.PRACTICE

    p_practice = mirror.next_problem(difficulty=difficulty)
    clock.advance(0.2)
    assert engine.submit_answer(str(p_practice.answer)) is True
    assert engine.phase is Phase.PRACTICE_DONE

    engine.start_scored()
    assert engine.phase is Phase.SCORED

    p1 = mirror.next_problem(difficulty=difficulty)
    clock.advance(0.5)
    assert engine.submit_answer(str(p1.answer)) is True

    p2 = mirror.next_problem(difficulty=difficulty)
    clock.advance(0.5)
    wrong_answer = (int(p2.answer) % 4) + 1
    assert wrong_answer != p2.answer
    assert engine.submit_answer(str(wrong_answer)) is True

    p3 = mirror.next_problem(difficulty=difficulty)
    clock.advance(0.5)
    assert engine.submit_answer(str(p3.answer)) is True

    clock.advance(6.0)
    engine.update()
    assert engine.phase is Phase.RESULTS

    summary = engine.scored_summary()
    assert summary.attempted == 3
    assert summary.correct == 2
    assert summary.accuracy == pytest.approx(2 / 3)
    assert summary.mean_response_time_s == pytest.approx(0.5)

    scored_events = [event for event in engine.events() if event.phase is Phase.SCORED]
    assert len(scored_events) == 3
    assert scored_events[0].score == pytest.approx(1.0)
    assert scored_events[2].score == pytest.approx(1.0)
    assert scored_events[1].score <= 0.5
