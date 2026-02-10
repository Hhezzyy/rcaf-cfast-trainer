from __future__ import annotations

from dataclasses import dataclass

from cfast_trainer.numerical_operations import NumericalOperationsConfig, NumericalOperationsGenerator, build_numerical_operations_test


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def test_headless_scripted_run_produces_expected_summary() -> None:
    seed = 555
    clock = FakeClock()

    engine = build_numerical_operations_test(
        clock=clock,
        seed=seed,
        difficulty=0.8,
        config=NumericalOperationsConfig(scored_duration_s=6.0, practice_questions=1),
    )

    # Instructions -> Practice
    engine.start_practice()
    assert engine.phase.value == "practice"

    gen = NumericalOperationsGenerator(seed=seed)

    # Practice answer (correct)
    p_practice = gen.next_problem(difficulty=0.8)
    clock.advance(0.2)
    assert engine.submit_answer(str(p_practice.answer)) is True
    assert engine.phase.value == "practice_done"

    # Start scored
    engine.start_scored()
    assert engine.phase.value == "scored"

    # Answer two correctly, one incorrectly.
    p1 = gen.next_problem(difficulty=0.8)
    clock.advance(0.5)
    assert engine.submit_answer(str(p1.answer)) is True

    p2 = gen.next_problem(difficulty=0.8)
    clock.advance(0.5)
    assert engine.submit_answer(str(p2.answer)) is True

    p3 = gen.next_problem(difficulty=0.8)
    clock.advance(0.5)
    assert engine.submit_answer(str(p3.answer + 1)) is True

    # Expire timer.
    clock.advance(6.0)
    engine.update()
    assert engine.phase.value == "results"

    s = engine.scored_summary()
    assert s.attempted == 3
    assert s.correct == 2
    assert s.accuracy == 2 / 3