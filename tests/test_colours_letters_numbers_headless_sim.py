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


def test_headless_sim_practice_then_scored_mixed_correctness() -> None:
    seed = 99
    clock = FakeClock()

    mirror = ColoursLettersNumbersGenerator(SeededRng(seed))
    practice = [mirror.next_trial(difficulty=0.5) for _ in range(4)]
    scored = [mirror.next_trial(difficulty=0.5) for _ in range(4)]

    engine = build_colours_letters_numbers_test(clock=clock, seed=seed, practice=True, scored_duration_s=8.0)
    assert engine.phase is Phase.INSTRUCTIONS

    engine.start_practice()
    assert engine.phase is Phase.PRACTICE

    for trial in practice:
        clock.advance(0.15)
        assert engine.submit_answer(trial.expected) is True

    assert engine.phase is Phase.PRACTICE_DONE

    engine.start_scored()
    assert engine.phase is Phase.SCORED

    for i, trial in enumerate(scored):
        clock.advance(0.2)
        answer = trial.expected
        if i == 3:
            answer = str(int(trial.expected) + 1)
        assert engine.submit_answer(answer) is True

    clock.advance(8.0)
    engine.update()

    assert engine.phase is Phase.RESULTS
    s = engine.scored_summary()
    assert s.attempted == 4
    assert s.correct == 3
