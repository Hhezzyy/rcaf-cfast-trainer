from __future__ import annotations

from dataclasses import dataclass

from cfast_trainer.math_reasoning import (
    MathReasoningConfig,
    MathReasoningGenerator,
    build_math_reasoning_test,
)


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def test_headless_scripted_run_produces_expected_summary() -> None:
    seed = 99
    difficulty = 0.5
    clock = FakeClock()

    engine = build_math_reasoning_test(
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        config=MathReasoningConfig(scored_duration_s=6.0, practice_questions=1),
    )

    # Instructions -> Practice
    engine.start_practice()
    assert engine.phase.value == "practice"

    # Mirror generator stream to know exact answers.
    gen = MathReasoningGenerator(seed=seed)

    # Practice (correct)
    p_practice = gen.next_problem(difficulty=difficulty)
    clock.advance(0.25)
    assert engine.submit_answer(str(p_practice.answer)) is True
    assert engine.phase.value == "practice_done"

    # Start scored
    engine.start_scored()
    assert engine.phase.value == "scored"

    # Two correct, one incorrect.
    p1 = gen.next_problem(difficulty=difficulty)
    clock.advance(0.5)
    assert engine.submit_answer(str(p1.answer)) is True

    p2 = gen.next_problem(difficulty=difficulty)
    clock.advance(0.5)
    assert engine.submit_answer(str(p2.answer)) is True

    p3 = gen.next_problem(difficulty=difficulty)
    clock.advance(0.5)
    assert engine.submit_answer(str(p3.answer + 10)) is True

    # Expire timer.
    clock.advance(6.0)
    engine.update()
    assert engine.phase.value == "results"

    s = engine.scored_summary()
    assert s.attempted == 3
    assert s.correct == 2