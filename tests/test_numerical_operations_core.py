from __future__ import annotations

from dataclasses import dataclass

from cfast_trainer.numerical_operations import (
    NumericalOperationsConfig,
    NumericalOperationsGenerator,
    build_numerical_operations_test,
)


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def test_generator_determinism_same_seed_same_sequence() -> None:
    seed = 123
    gen1 = NumericalOperationsGenerator(seed=seed)
    gen2 = NumericalOperationsGenerator(seed=seed)

    seq1 = [gen1.next_problem(difficulty=0.7) for _ in range(50)]
    seq2 = [gen2.next_problem(difficulty=0.7) for _ in range(50)]

    assert [(p.prompt, p.answer) for p in seq1] == [(p.prompt, p.answer) for p in seq2]


def test_scoring_counts_only_scored_phase() -> None:
    # Zero practice questions: go straight into SCORED.
    seed = 42
    clock = FakeClock()

    engine = build_numerical_operations_test(
        clock=clock,
        seed=seed,
        difficulty=0.5,
        config=NumericalOperationsConfig(scored_duration_s=10.0, practice_questions=0),
    )

    engine.start_scored()

    # Mirror the generator stream to know correct answers.
    gen = NumericalOperationsGenerator(seed=seed)

    for _ in range(3):
        p = gen.next_problem(difficulty=0.5)
        clock.advance(0.25)
        assert engine.submit_answer(str(p.answer)) is True

    s = engine.scored_summary()
    assert s.attempted == 3
    assert s.correct == 3
    assert s.accuracy == 1.0


def test_timer_boundary_transitions_to_results_and_rejects_late_submit() -> None:
    seed = 7
    clock = FakeClock()

    engine = build_numerical_operations_test(
        clock=clock,
        seed=seed,
        difficulty=0.5,
        config=NumericalOperationsConfig(scored_duration_s=2.0, practice_questions=0),
    )
    engine.start_scored()

    clock.advance(2.0)
    engine.update()

    assert engine.phase.value == "results"
    assert engine.submit_answer("0") is False