from __future__ import annotations

from dataclasses import dataclass

from cfast_trainer.cognitive_core import Phase, SeededRng
from cfast_trainer.digit_recognition import DigitRecognitionGenerator, DigitRecognitionQuestionKind, build_digit_recognition_test


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


def _advance_to_question(clock: FakeClock, engine) -> None:
    clock.advance(1.3)
    engine.update()
    clock.advance(0.3)
    engine.update()


def test_headless_sim_practice_then_scored_mixed_correctness() -> None:
    seed = 99
    clock = FakeClock()

    mirror = DigitRecognitionGenerator(SeededRng(seed))
    practice = [mirror.next_trial(difficulty=0.5) for _ in range(3)]
    scored = [mirror.next_trial(difficulty=0.5) for _ in range(4)]

    engine = build_digit_recognition_test(clock=clock, seed=seed, practice=True, scored_duration_s=8.0)
    assert engine.phase is Phase.INSTRUCTIONS

    engine.start_practice()
    assert engine.phase is Phase.PRACTICE

    for t in practice:
        _advance_to_question(clock, engine)
        assert engine.submit_answer(t.expected) is True

    assert engine.phase is Phase.PRACTICE_DONE

    engine.start_scored()
    assert engine.phase is Phase.SCORED

    for i, t in enumerate(scored):
        _advance_to_question(clock, engine)
        ans = t.expected
        if i == 3:
            if t.kind is DigitRecognitionQuestionKind.RECALL:
                ans = (t.expected[:-1] + ("0" if t.expected[-1] != "0" else "1")) if len(t.expected) > 0 else "0"
            else:
                ans = str(int(t.expected) + 1)
        assert engine.submit_answer(ans) is True

    clock.advance(8.0)
    engine.update()

    assert engine.phase is Phase.RESULTS
    s = engine.scored_summary()
    assert s.attempted == 4
    assert s.correct == 3