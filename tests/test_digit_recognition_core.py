from __future__ import annotations

from dataclasses import dataclass

from cfast_trainer.cognitive_core import Phase, SeededRng
from cfast_trainer.digit_recognition import DigitRecognitionGenerator, build_digit_recognition_test


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


def _advance_to_question(clock: FakeClock, engine) -> None:
    clock.advance(1.3)
    engine.update()  # SHOW -> MASK
    clock.advance(0.3)
    engine.update()  # MASK -> QUESTION


def test_generator_is_deterministic_for_same_seed() -> None:
    seed = 12345
    g1 = DigitRecognitionGenerator(SeededRng(seed))
    g2 = DigitRecognitionGenerator(SeededRng(seed))

    seq1 = [g1.next_trial(difficulty=0.5) for _ in range(10)]
    seq2 = [g2.next_trial(difficulty=0.5) for _ in range(10)]

    assert seq1 == seq2


def test_scoring_recall_and_count_target() -> None:
    seed = 42
    clock = FakeClock()

    mirror = DigitRecognitionGenerator(SeededRng(seed))
    t1 = mirror.next_trial(difficulty=0.5)
    t2 = mirror.next_trial(difficulty=0.5)

    engine = build_digit_recognition_test(clock=clock, seed=seed, practice=False, scored_duration_s=10.0)
    engine.start_practice()
    engine.start_scored()

    _advance_to_question(clock, engine)
    assert engine.submit_answer(t1.expected) is True

    _advance_to_question(clock, engine)
    assert engine.submit_answer(t2.expected) is True

    s = engine.scored_summary()
    assert s.attempted == 2
    assert s.correct == 2
    assert s.accuracy == 1.0


def test_timer_boundary_transitions_to_results_and_rejects_late_submit() -> None:
    clock = FakeClock()
    engine = build_digit_recognition_test(clock=clock, seed=7, practice=False, scored_duration_s=1.0)
    engine.start_practice()
    engine.start_scored()

    clock.advance(1.0)
    engine.update()

    assert engine.phase is Phase.RESULTS
    assert engine.submit_answer("0") is False