from __future__ import annotations

from dataclasses import dataclass

from cfast_trainer.cognitive_core import Phase, SeededRng
from cfast_trainer.colours_letters_numbers import (
    ColoursLettersNumbersGenerator,
    build_colours_letters_numbers_test,
)


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


def test_generator_is_deterministic_for_same_seed() -> None:
    seed = 12345
    g1 = ColoursLettersNumbersGenerator(SeededRng(seed))
    g2 = ColoursLettersNumbersGenerator(SeededRng(seed))

    seq1 = [g1.next_trial(difficulty=0.5) for _ in range(9)]
    seq2 = [g2.next_trial(difficulty=0.5) for _ in range(9)]

    assert seq1 == seq2


def test_scoring_mixed_trial_kinds() -> None:
    seed = 42
    clock = FakeClock()

    mirror = ColoursLettersNumbersGenerator(SeededRng(seed))
    trials = [mirror.next_trial(difficulty=0.5) for _ in range(3)]

    engine = build_colours_letters_numbers_test(clock=clock, seed=seed, practice=False, scored_duration_s=15.0)
    engine.start_practice()
    engine.start_scored()

    for trial in trials:
        clock.advance(0.2)
        assert engine.submit_answer(trial.expected) is True

    s = engine.scored_summary()
    assert s.attempted == 3
    assert s.correct == 3
    assert s.accuracy == 1.0


def test_timer_boundary_transitions_to_results_and_rejects_late_submit() -> None:
    clock = FakeClock()
    engine = build_colours_letters_numbers_test(clock=clock, seed=7, practice=False, scored_duration_s=1.0)
    engine.start_practice()
    engine.start_scored()

    clock.advance(1.0)
    engine.update()

    assert engine.phase is Phase.RESULTS
    assert engine.submit_answer("0") is False
