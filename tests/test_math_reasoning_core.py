from __future__ import annotations

from dataclasses import dataclass

from cfast_trainer.math_reasoning import MathReasoningConfig, MathReasoningGenerator, build_math_reasoning_test


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def test_generator_determinism_same_seed_same_sequence() -> None:
    seed = 99
    gen1 = MathReasoningGenerator(seed=seed)
    gen2 = MathReasoningGenerator(seed=seed)

    seq1 = [gen1.next_problem(difficulty=0.3) for _ in range(25)]
    seq2 = [gen2.next_problem(difficulty=0.3) for _ in range(25)]

    assert [(p.prompt, p.answer) for p in seq1] == [(p.prompt, p.answer) for p in seq2]


def test_session_lifecycle_including_practice_gate() -> None:
    seed = 101
    clock = FakeClock()

    engine = build_math_reasoning_test(
        clock=clock,
        seed=seed,
        difficulty=0.5,
        config=MathReasoningConfig(scored_duration_s=5.0, practice_questions=2),
    )

    assert engine.phase.value == "instructions"
    engine.start_practice()
    assert engine.phase.value == "practice"

    gen = MathReasoningGenerator(seed=seed)

    # Practice Q1
    p1 = gen.next_problem(difficulty=0.5)
    clock.advance(0.5)
    assert engine.submit_answer(str(p1.answer)) is True
    assert engine.phase.value == "practice"

    # Practice Q2 completes practice and gates before timed start.
    p2 = gen.next_problem(difficulty=0.5)
    clock.advance(0.5)
    assert engine.submit_answer(str(p2.answer)) is True
    assert engine.phase.value == "practice_done"

    engine.start_scored()
    assert engine.phase.value == "scored"

    # Scored Q1
    p3 = gen.next_problem(difficulty=0.5)
    clock.advance(0.5)
    assert engine.submit_answer(str(p3.answer)) is True

    # Expire timer
    clock.advance(5.0)
    engine.update()
    assert engine.phase.value == "results"

    s = engine.scored_summary()
    assert s.attempted == 1
    assert s.correct == 1