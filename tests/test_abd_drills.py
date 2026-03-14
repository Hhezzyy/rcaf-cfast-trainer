from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from cfast_trainer.abd_drills import (
    AbdAngleCalibrationGenerator,
    AbdBearingCalibrationGenerator,
    AbdCardinalAnchorsGenerator,
    AbdDrillConfig,
    AbdFamilyRunConfig,
    AbdIntermediateAnchorsGenerator,
    AbdMixedTempoGenerator,
    AbdTestStyleFamilyRunGenerator,
    AbdTypedAnswerScorer,
    build_abd_test_style_family_run_drill,
)
from cfast_trainer.angles_bearings_degrees import (
    AnglesBearingsDegreesPayload,
    AnglesBearingsQuestionKind,
    AnglesBearingsTrainingPayload,
)
from cfast_trainer.ant_drills import AntDrillMode


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t


def test_cardinal_anchors_low_level_emit_only_cardinals_and_straights() -> None:
    generator = AbdCardinalAnchorsGenerator(seed=19)

    for _ in range(60):
        problem = generator.next_problem(difficulty=0.0)
        payload = cast(AnglesBearingsTrainingPayload, problem.payload)
        if payload.kind is AnglesBearingsQuestionKind.BEARING_FROM_REFERENCE:
            assert payload.exact_value_deg in {0, 45, 90, 135, 180, 225, 270, 315}
            assert payload.allow_north_360_alias is (payload.exact_value_deg == 0)
        else:
            assert payload.exact_value_deg in {0, 45, 90, 135, 180}
            assert payload.reference_bearing_deg == 0
            assert payload.allow_north_360_alias is False


def test_intermediate_anchors_add_non_eighth_landmarks() -> None:
    generator = AbdIntermediateAnchorsGenerator(seed=29)
    seen_non_eighth = False

    for _ in range(80):
        problem = generator.next_problem(difficulty=0.8)
        payload = cast(AnglesBearingsTrainingPayload, problem.payload)
        if payload.kind is AnglesBearingsQuestionKind.BEARING_FROM_REFERENCE:
            if payload.exact_value_deg not in {0, 45, 90, 135, 180, 225, 270, 315}:
                seen_non_eighth = True
                break
        elif payload.exact_value_deg not in {0, 45, 90, 135, 180}:
            seen_non_eighth = True
            break

    assert seen_non_eighth is True


def test_bearing_calibration_progresses_from_cardinals_to_eighths() -> None:
    low = AbdBearingCalibrationGenerator(seed=61)
    high = AbdBearingCalibrationGenerator(seed=61)

    low_values = {
        cast(AnglesBearingsTrainingPayload, low.next_problem(difficulty=0.2).payload).exact_value_deg
        for _ in range(20)
    }
    high_values = {
        cast(AnglesBearingsTrainingPayload, high.next_problem(difficulty=0.9).payload).exact_value_deg
        for _ in range(30)
    }

    assert all(value % 90 == 0 for value in low_values)
    assert all(value % 45 == 0 for value in high_values)
    assert any(value % 90 != 0 for value in high_values)


def test_bearing_drills_can_place_dots_closer_to_center_at_higher_levels() -> None:
    generator = AbdBearingCalibrationGenerator(seed=67)
    ratios = [
        cast(AnglesBearingsTrainingPayload, generator.next_problem(difficulty=0.95).payload).marker_radius_ratio
        for _ in range(20)
    ]

    assert min(ratios) <= 0.46


def test_angle_calibration_uses_nearest_five_answer_but_flashes_exact() -> None:
    generator = AbdAngleCalibrationGenerator(seed=7)
    problem = generator.next_problem(difficulty=0.6)
    payload = cast(AnglesBearingsTrainingPayload, problem.payload)

    assert payload.kind is AnglesBearingsQuestionKind.ANGLE_BETWEEN_LINES
    assert problem.answer % 5 == 0
    assert abs(int(problem.answer) - payload.exact_value_deg) <= 2
    assert payload.display_answer_text == str(payload.exact_value_deg)
    assert payload.reference_bearing_deg == 0


def test_angle_calibration_rotates_only_at_higher_levels() -> None:
    low = AbdAngleCalibrationGenerator(seed=71)
    high = AbdAngleCalibrationGenerator(seed=71)

    low_refs = {
        cast(AnglesBearingsTrainingPayload, low.next_problem(difficulty=0.2).payload).reference_bearing_deg
        for _ in range(12)
    }
    high_refs = {
        cast(AnglesBearingsTrainingPayload, high.next_problem(difficulty=0.95).payload).reference_bearing_deg
        for _ in range(20)
    }

    assert low_refs == {0}
    assert high_refs != {0}


def test_north_alias_only_applies_when_prompt_explicitly_teaches_it() -> None:
    scorer = AbdTypedAnswerScorer()

    cardinal = AbdCardinalAnchorsGenerator(seed=3)
    for _ in range(120):
        problem = cardinal.next_problem(difficulty=0.0)
        payload = cast(AnglesBearingsTrainingPayload, problem.payload)
        if payload.kind is AnglesBearingsQuestionKind.BEARING_FROM_REFERENCE and problem.answer == 0:
            assert payload.allow_north_360_alias is True
            assert scorer.score(problem=problem, user_answer=360, raw="360") == 1.0
            break
    else:
        raise AssertionError("Expected a north-anchor prompt")

    calibration = AbdBearingCalibrationGenerator(seed=11)
    for _ in range(400):
        problem = calibration.next_problem(difficulty=0.7)
        payload = cast(AnglesBearingsTrainingPayload, problem.payload)
        if problem.answer == 0:
            assert payload.allow_north_360_alias is False
            assert scorer.score(problem=problem, user_answer=360, raw="360") == 0.0
            break
    else:
        raise AssertionError("Expected a rounded-north calibration prompt")


def test_mixed_tempo_is_deterministic_for_same_seed() -> None:
    g1 = AbdMixedTempoGenerator(seed=41)
    g2 = AbdMixedTempoGenerator(seed=41)

    seq1 = [g1.next_problem(difficulty=0.65) for _ in range(25)]
    seq2 = [g2.next_problem(difficulty=0.65) for _ in range(25)]

    def summarize(problem: object) -> tuple[object, ...]:
        payload = cast(AnglesBearingsTrainingPayload, problem.payload)
        return (
            problem.answer,
            payload.kind,
            payload.reference_bearing_deg,
            payload.target_bearing_deg,
            payload.rounded_value_deg,
            payload.exact_value_deg,
            payload.base_cap_s,
        )

    assert [summarize(problem) for problem in seq1] == [summarize(problem) for problem in seq2]


def test_test_style_family_run_emits_only_requested_family() -> None:
    angle_generator = AbdTestStyleFamilyRunGenerator(
        seed=17,
        family=AnglesBearingsQuestionKind.ANGLE_BETWEEN_LINES,
    )
    bearing_generator = AbdTestStyleFamilyRunGenerator(
        seed=18,
        family=AnglesBearingsQuestionKind.BEARING_FROM_REFERENCE,
    )

    for _ in range(20):
        angle_problem = angle_generator.next_problem(difficulty=0.6)
        angle_payload = cast(AnglesBearingsDegreesPayload, angle_problem.payload)
        assert angle_payload.kind is AnglesBearingsQuestionKind.ANGLE_BETWEEN_LINES

        bearing_problem = bearing_generator.next_problem(difficulty=0.6)
        bearing_payload = cast(AnglesBearingsDegreesPayload, bearing_problem.payload)
        assert bearing_payload.kind is AnglesBearingsQuestionKind.BEARING_FROM_REFERENCE


def test_test_style_family_run_builder_uses_real_mc_payloads() -> None:
    engine = build_abd_test_style_family_run_drill(
        clock=FakeClock(),
        seed=23,
        family=AnglesBearingsQuestionKind.ANGLE_BETWEEN_LINES,
        difficulty=0.5,
        mode=AntDrillMode.TEMPO,
        config=AbdFamilyRunConfig(
            family=AnglesBearingsQuestionKind.ANGLE_BETWEEN_LINES,
            practice_questions=0,
            scored_duration_s=30.0,
        ),
    )
    engine.start_scored()

    payload = engine.snapshot().payload
    assert isinstance(payload, AnglesBearingsDegreesPayload)
