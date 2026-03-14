from __future__ import annotations

from dataclasses import dataclass

import pytest

from cfast_trainer.system_logic import SystemLogicConfig, SystemLogicGenerator, build_system_logic_test


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def test_headless_scripted_run_produces_expected_summary() -> None:
    seed = 222
    difficulty = 0.5
    clock = FakeClock()

    engine = build_system_logic_test(
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        config=SystemLogicConfig(scored_duration_s=6.0, practice_questions=1),
    )

    engine.start_practice()
    assert engine.phase.value == "practice"

    gen = SystemLogicGenerator(seed=seed)

    practice_problem = gen.next_problem(difficulty=difficulty)
    clock.advance(0.25)
    assert engine.submit_answer(str(practice_problem.answer)) is True
    assert engine.phase.value == "practice_done"

    engine.start_scored()
    assert engine.phase.value == "scored"

    p1 = gen.next_problem(difficulty=difficulty)
    clock.advance(0.5)
    assert engine.submit_answer(str(p1.answer)) is True

    p2 = gen.next_problem(difficulty=difficulty)
    wrong_answer = 1 if p2.answer != 1 else 2
    clock.advance(0.5)
    assert engine.submit_answer(str(wrong_answer)) is True

    p3 = gen.next_problem(difficulty=difficulty)
    clock.advance(0.5)
    assert engine.submit_answer(str(p3.answer)) is True

    clock.advance(6.0)
    engine.update()
    assert engine.phase.value == "results"

    summary = engine.scored_summary()
    assert summary.attempted == 3
    assert summary.correct == 2
    assert summary.accuracy == pytest.approx(2 / 3)
    assert summary.throughput_per_min == pytest.approx(30.0)
    assert summary.mean_response_time_s == pytest.approx(0.5)
    assert summary.total_score == pytest.approx(2.0)
    assert summary.max_score == pytest.approx(3.0)
    assert summary.score_ratio == pytest.approx(2 / 3)
