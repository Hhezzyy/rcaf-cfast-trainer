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
    DrDifferenceCountGenerator,
    DrGroupedFamilyRunGenerator,
    DrMixedPressureGenerator,
    DrRecallRunGenerator,
    build_dr_difference_count_drill,
    build_dr_position_probe_drill,
    build_dr_recall_after_interference_drill,
    build_dr_recall_run_drill,
    build_dr_visual_digit_query_drill,
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
        DigitRecognitionQuestionKind.DIFFERENCE_COUNT,
    }
    assert spec.keep_display_visible_during_question is True
    if spec.kind in {
        DigitRecognitionQuestionKind.DIFFERENT_DIGIT,
        DigitRecognitionQuestionKind.DIFFERENCE_COUNT,
    }:
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
        for _ in range(8)
    ]

    assert families == [
        DigitRecognitionQuestionKind.RECALL,
        DigitRecognitionQuestionKind.RECALL,
        DigitRecognitionQuestionKind.COUNT_TARGET,
        DigitRecognitionQuestionKind.COUNT_TARGET,
        DigitRecognitionQuestionKind.DIFFERENT_DIGIT,
        DigitRecognitionQuestionKind.DIFFERENT_DIGIT,
        DigitRecognitionQuestionKind.DIFFERENCE_COUNT,
        DigitRecognitionQuestionKind.DIFFERENCE_COUNT,
    ]


def test_mixed_pressure_emits_all_four_families() -> None:
    generator = DrMixedPressureGenerator(seed=61)
    families = {
        cast(DigitRecognitionTrainingSpec, generator.next_problem(difficulty=0.8).payload).kind
        for _ in range(48)
    }

    assert families == {
        DigitRecognitionQuestionKind.RECALL,
        DigitRecognitionQuestionKind.COUNT_TARGET,
        DigitRecognitionQuestionKind.DIFFERENT_DIGIT,
        DigitRecognitionQuestionKind.DIFFERENCE_COUNT,
    }


def test_difference_count_drill_uses_hidden_two_string_comparison() -> None:
    clock = FakeClock()
    engine = build_dr_difference_count_drill(clock=clock, seed=63, difficulty=0.7)

    engine.start_practice()
    spec = cast(DigitRecognitionTrainingSpec, engine._current.payload)

    assert spec.kind is DigitRecognitionQuestionKind.DIFFERENCE_COUNT
    assert len(spec.display_lines) == 2
    assert spec.keep_display_visible_during_question is False


def test_difference_count_generator_harder_difficulty_raises_count_or_pressure() -> None:
    generator = DrDifferenceCountGenerator(seed=75)

    low = cast(DigitRecognitionTrainingSpec, generator.next_problem(difficulty=0.1).payload)
    high = cast(DigitRecognitionTrainingSpec, generator.next_problem(difficulty=0.9).payload)

    assert int(high.expected_digits) >= int(low.expected_digits)
    assert high.initial_display_s <= low.initial_display_s


def test_recall_generator_harder_difficulty_uses_longer_strings_and_shorter_timings() -> None:
    generator = DrRecallRunGenerator(seed=77)

    low = cast(DigitRecognitionTrainingSpec, generator.next_problem(difficulty=0.1).payload)
    high = cast(DigitRecognitionTrainingSpec, generator.next_problem(difficulty=0.9).payload)

    assert len(high.expected_digits) >= len(low.expected_digits)
    assert high.initial_display_s <= low.initial_display_s
    assert high.mask_s <= low.mask_s


def _difficulty_for_level(level: int) -> float:
    return float(level - 1) / 9.0


def test_visual_digit_query_shows_interference_during_mask_then_accepts_query() -> None:
    clock = FakeClock()
    engine = build_dr_visual_digit_query_drill(clock=clock, seed=81, difficulty=0.6)

    engine.start_practice()
    spec = cast(DigitRecognitionTrainingSpec, engine._current.payload)
    assert spec.family_tag == "visual_digit_query"
    clock.advance(spec.initial_display_s + 0.05)
    engine.update()

    masked = cast(DigitRecognitionPayload, engine.snapshot().payload)
    assert masked.accepting_input is False
    assert masked.display_lines == spec.mask_display_lines
    assert "interference" in engine.current_prompt().lower()

    clock.advance(spec.mask_s + 0.05)
    engine.update()
    payload = cast(DigitRecognitionPayload, engine.snapshot().payload)
    assert payload.accepting_input is True
    assert payload.display_lines is None


def test_recall_after_interference_uses_hidden_recall_with_interference_stream() -> None:
    clock = FakeClock()
    engine = build_dr_recall_after_interference_drill(clock=clock, seed=91, difficulty=0.7)

    engine.start_practice()
    spec = cast(DigitRecognitionTrainingSpec, engine._current.payload)
    assert spec.family_tag == "recall_after_interference"
    assert spec.interference_rate >= 1
    clock.advance(spec.initial_display_s + 0.05)
    engine.update()
    masked = cast(DigitRecognitionPayload, engine.snapshot().payload)
    assert masked.display_lines == spec.mask_display_lines

    clock.advance(spec.mask_s + 0.05)
    engine.update()
    payload = cast(DigitRecognitionPayload, engine.snapshot().payload)
    assert payload.accepting_input is True
    assert payload.display_lines is None


def test_wave1_memory_drills_l2_l5_l8_scale_materially() -> None:
    builders = (
        build_dr_visual_digit_query_drill,
        build_dr_recall_after_interference_drill,
    )
    for builder in builders:
        signatures: list[tuple[int, float, int, int, float]] = []
        for level in (2, 5, 8):
            clock = FakeClock()
            engine = builder(clock=clock, seed=121, difficulty=_difficulty_for_level(level))
            engine.start_scored()
            spec = cast(DigitRecognitionTrainingSpec, engine._current.payload)
            signatures.append(
                (
                    spec.span_length,
                    round(float(spec.delay_s), 3),
                    spec.query_complexity,
                    spec.interference_rate,
                    round(float(engine._question_cap_s), 3),
                )
            )
        low, mid, high = signatures
        assert low[0] <= mid[0] <= high[0]
        assert low[1] <= mid[1] <= high[1]
        assert low[3] <= mid[3] <= high[3]
        assert low[4] > mid[4] > high[4]
        assert low != mid != high
