from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from cfast_trainer.ant_drills import AntDrillMode
from cfast_trainer.digit_recognition import (
    DigitRecognitionPayload,
    DigitRecognitionQuestionKind,
    DigitRecognitionTrainingSpec,
)
from cfast_trainer.dr_drills import (
    DrGroupedFamilyRunGenerator,
    DrMixedPressureGenerator,
    DrRecallRunGenerator,
    build_dr_position_probe_drill,
    build_dr_recall_run_drill,
    build_dr_visible_copy_drill,
    build_dr_visible_family_primer_drill,
)


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


def test_visible_copy_keeps_digits_visible_while_accepting_input() -> None:
    clock = FakeClock()
    engine = build_dr_visible_copy_drill(clock=clock, seed=11, difficulty=0.4)

    engine.start_practice()
    payload = cast(DigitRecognitionPayload, engine.snapshot().payload)

    assert payload.accepting_input is True
    assert payload.display_lines is not None
    assert payload.prompt_text == "Type the full string while it remains visible."


def test_position_probe_transitions_to_visible_prompted_question() -> None:
    clock = FakeClock()
    engine = build_dr_position_probe_drill(clock=clock, seed=23, difficulty=0.7)

    engine.start_practice()
    initial = cast(DigitRecognitionPayload, engine.snapshot().payload)
    assert initial.accepting_input is False
    assert initial.display_lines is not None

    spec = cast(DigitRecognitionTrainingSpec, engine._current.payload)
    clock.advance(spec.initial_display_s + 0.05)
    engine.update()

    payload = cast(DigitRecognitionPayload, engine.snapshot().payload)
    assert payload.accepting_input is True
    assert payload.display_lines is not None
    assert "Digits stay visible." in payload.prompt_text


def test_visible_family_primer_uses_visible_supported_non_recall_families() -> None:
    clock = FakeClock()
    engine = build_dr_visible_family_primer_drill(clock=clock, seed=31, difficulty=0.5)

    engine.start_practice()
    spec = cast(DigitRecognitionTrainingSpec, engine._current.payload)

    assert spec.kind in {
        DigitRecognitionQuestionKind.COUNT_TARGET,
        DigitRecognitionQuestionKind.DIFFERENT_DIGIT,
    }
    assert spec.keep_display_visible_during_question is True
    if spec.kind is DigitRecognitionQuestionKind.DIFFERENT_DIGIT:
        assert len(spec.display_lines) == 2


def test_recall_run_hides_digits_after_show_and_mask() -> None:
    clock = FakeClock()
    engine = build_dr_recall_run_drill(
        clock=clock,
        seed=41,
        difficulty=0.6,
        mode=AntDrillMode.BUILD,
    )

    engine.start_practice()
    spec = cast(DigitRecognitionTrainingSpec, engine._current.payload)
    assert spec.keep_display_visible_during_question is False

    clock.advance(spec.initial_display_s + 0.05)
    engine.update()
    clock.advance(spec.mask_s + 0.05)
    engine.update()

    payload = cast(DigitRecognitionPayload, engine.snapshot().payload)
    assert payload.accepting_input is True
    assert payload.display_lines is None


def test_grouped_family_run_preserves_fixed_family_order() -> None:
    generator = DrGroupedFamilyRunGenerator(seed=55)
    families = [
        cast(DigitRecognitionTrainingSpec, generator.next_problem(difficulty=0.5).payload).kind
        for _ in range(6)
    ]

    assert families == [
        DigitRecognitionQuestionKind.RECALL,
        DigitRecognitionQuestionKind.RECALL,
        DigitRecognitionQuestionKind.COUNT_TARGET,
        DigitRecognitionQuestionKind.COUNT_TARGET,
        DigitRecognitionQuestionKind.DIFFERENT_DIGIT,
        DigitRecognitionQuestionKind.DIFFERENT_DIGIT,
    ]


def test_mixed_pressure_emits_all_three_families() -> None:
    generator = DrMixedPressureGenerator(seed=61)
    families = {
        cast(DigitRecognitionTrainingSpec, generator.next_problem(difficulty=0.8).payload).kind
        for _ in range(48)
    }

    assert families == {
        DigitRecognitionQuestionKind.RECALL,
        DigitRecognitionQuestionKind.COUNT_TARGET,
        DigitRecognitionQuestionKind.DIFFERENT_DIGIT,
    }


def test_recall_generator_harder_difficulty_uses_longer_strings_and_shorter_timings() -> None:
    generator = DrRecallRunGenerator(seed=77)

    low = cast(DigitRecognitionTrainingSpec, generator.next_problem(difficulty=0.1).payload)
    high = cast(DigitRecognitionTrainingSpec, generator.next_problem(difficulty=0.9).payload)

    assert len(high.expected_digits) >= len(low.expected_digits)
    assert high.initial_display_s <= low.initial_display_s
    assert high.mask_s <= low.mask_s
