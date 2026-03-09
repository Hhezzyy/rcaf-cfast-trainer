from __future__ import annotations

from dataclasses import dataclass

from cfast_trainer.cognitive_core import Phase, SeededRng
from cfast_trainer.digit_recognition import (
    DigitRecognitionGenerator,
    DigitRecognitionQuestionKind,
    build_digit_recognition_test,
)


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


def test_scoring_mixed_question_bank() -> None:
    seed = 42
    clock = FakeClock()

    mirror = DigitRecognitionGenerator(SeededRng(seed))
    trials = [mirror.next_trial(difficulty=0.5) for _ in range(4)]

    engine = build_digit_recognition_test(
        clock=clock, seed=seed, practice=False, scored_duration_s=10.0
    )
    engine.start_practice()
    engine.start_scored()

    for trial in trials:
        _advance_to_question(clock, engine)
        assert engine.submit_answer(trial.expected) is True

    s = engine.scored_summary()
    assert s.attempted == 4
    assert s.correct == 4
    assert s.accuracy == 1.0


def test_generator_emits_recall_count_and_difference_trials() -> None:
    gen = DigitRecognitionGenerator(SeededRng(123))
    trials = [gen.next_trial(difficulty=0.5) for _ in range(8)]
    kinds = {trial.kind for trial in trials}

    assert DigitRecognitionQuestionKind.RECALL in kinds
    assert DigitRecognitionQuestionKind.COUNT_TARGET in kinds
    assert DigitRecognitionQuestionKind.DIFFERENT_DIGIT in kinds

    for trial in trials:
        if trial.kind is DigitRecognitionQuestionKind.COUNT_TARGET:
            target = trial.prompt.split("HOW MANY ", 1)[1][0]
            assert trial.expected == str(sum(1 for ch in trial.digits if ch == target))
        if trial.kind is DigitRecognitionQuestionKind.DIFFERENT_DIGIT:
            assert trial.comparison_digits is not None
            diffs = [
                idx
                for idx, (left, right) in enumerate(
                    zip(trial.digits, trial.comparison_digits, strict=True)
                )
                if left != right
            ]
            assert len(diffs) == 1
            assert trial.expected == trial.comparison_digits[diffs[0]]


def test_timer_boundary_transitions_to_results_and_rejects_late_submit() -> None:
    clock = FakeClock()
    engine = build_digit_recognition_test(
        clock=clock, seed=7, practice=False, scored_duration_s=1.0
    )
    engine.start_practice()
    engine.start_scored()

    clock.advance(1.0)
    engine.update()

    assert engine.phase is Phase.RESULTS
    assert engine.submit_answer("0") is False
