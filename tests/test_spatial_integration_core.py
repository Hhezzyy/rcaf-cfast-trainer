from __future__ import annotations

from dataclasses import dataclass

import pytest

from cfast_trainer.cognitive_core import Phase, Problem
from cfast_trainer.spatial_integration import (
    SpatialIntegrationAnswerMode,
    SpatialIntegrationConfig,
    SpatialIntegrationGenerator,
    SpatialIntegrationPart,
    SpatialIntegrationPayload,
    SpatialIntegrationQuestionKind,
    SpatialIntegrationScorer,
    SpatialIntegrationTrialStage,
    build_spatial_integration_test,
)


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _wait_for_question(
    *,
    engine: object,
    clock: FakeClock,
    max_steps: int = 120,
    dt: float = 0.05,
) -> SpatialIntegrationPayload:
    for _ in range(max_steps):
        snap = engine.snapshot()
        payload = snap.payload if isinstance(snap.payload, SpatialIntegrationPayload) else None
        if payload is not None and payload.trial_stage is SpatialIntegrationTrialStage.QUESTION:
            return payload
        clock.advance(dt)
        engine.update()
    raise AssertionError("Timed out waiting for question stage.")


def test_scene_generator_is_deterministic_for_both_parts() -> None:
    seed = 515
    for part in (SpatialIntegrationPart.STATIC, SpatialIntegrationPart.AIRCRAFT):
        g1 = SpatialIntegrationGenerator(seed=seed)
        g2 = SpatialIntegrationGenerator(seed=seed)

        seq1 = [g1.next_scene_cluster(part=part, difficulty=0.64) for _ in range(8)]
        seq2 = [g2.next_scene_cluster(part=part, difficulty=0.64) for _ in range(8)]

        assert seq1 == seq2


def test_static_scene_cluster_has_expected_question_mix() -> None:
    gen = SpatialIntegrationGenerator(seed=18)
    scene = gen.next_scene_cluster(part=SpatialIntegrationPart.STATIC, difficulty=0.6)

    assert scene.part is SpatialIntegrationPart.STATIC
    assert len(scene.reference_views) == 3
    assert all(view.scene_view.value == "oblique" for view in scene.reference_views)
    assert scene.questions[0].kind is SpatialIntegrationQuestionKind.LANDMARK_GRID
    assert scene.questions[0].answer_mode is SpatialIntegrationAnswerMode.GRID_CLICK
    assert scene.questions[1].kind is SpatialIntegrationQuestionKind.LANDMARK_GRID
    assert scene.questions[2].kind is SpatialIntegrationQuestionKind.SCENE_RECONSTRUCTION
    assert scene.questions[2].answer_mode is SpatialIntegrationAnswerMode.OPTION_PICK
    assert len(scene.questions[2].options) == 4
    assert any(opt.answer_token == "correct" for opt in scene.questions[2].options)


def test_aircraft_scene_cluster_has_expected_question_mix() -> None:
    gen = SpatialIntegrationGenerator(seed=19)
    scene = gen.next_scene_cluster(part=SpatialIntegrationPart.AIRCRAFT, difficulty=0.6)

    assert scene.part is SpatialIntegrationPart.AIRCRAFT
    assert len(scene.reference_views) == 3
    assert all(view.scene_view.value == "oblique" for view in scene.reference_views)
    assert len(scene.route_points) >= 5
    assert scene.questions[0].kind is SpatialIntegrationQuestionKind.AIRCRAFT_ROUTE_SELECTION
    assert scene.questions[0].answer_mode is SpatialIntegrationAnswerMode.OPTION_PICK
    assert scene.questions[1].kind is SpatialIntegrationQuestionKind.AIRCRAFT_ROUTE_SELECTION
    assert scene.questions[2].kind is SpatialIntegrationQuestionKind.AIRCRAFT_LOCATION_GRID
    assert scene.questions[2].answer_mode is SpatialIntegrationAnswerMode.GRID_CLICK
    assert len(scene.questions[0].options) == 4
    assert len(scene.questions[1].options) == 4


def test_generator_supports_multiple_objects_in_one_grid_cell() -> None:
    gen = SpatialIntegrationGenerator(seed=77)
    scenes = [gen.next_scene_cluster(part=SpatialIntegrationPart.STATIC, difficulty=0.6) for _ in range(12)]

    assert any(len({(item.x, item.y) for item in scene.landmarks}) < len(scene.landmarks) for scene in scenes)


def test_scorer_is_exact_for_grid_and_option_answers() -> None:
    gen = SpatialIntegrationGenerator(seed=44)
    static_scene = gen.next_scene_cluster(part=SpatialIntegrationPart.STATIC, difficulty=0.5)
    static_question = static_scene.questions[0]
    static_problem = Problem(
        prompt=static_question.stem,
        answer=1,
        payload=SpatialIntegrationPayload(
            part=static_scene.part,
            trial_stage=SpatialIntegrationTrialStage.QUESTION,
            block_kind="practice",
            scene_id=static_scene.scene_id,
            scene_index_in_block=1,
            scenes_in_block=2,
            study_view_index=3,
            study_views_in_scene=3,
            question_index_in_scene=1,
            questions_in_scene=3,
            stage_time_remaining_s=8.0,
            part_time_remaining_s=None,
            kind=static_question.kind,
            answer_mode=static_question.answer_mode,
            stem=static_question.stem,
            query_label=static_question.query_label,
            north_arrow_deg=0,
            scene_view=static_scene.scene_view,
            grid_cols=static_scene.grid_cols,
            grid_rows=static_scene.grid_rows,
            alt_levels=static_scene.alt_levels,
            reference_views=static_scene.reference_views,
            active_reference_view=None,
            hills=static_scene.hills,
            landmarks=static_scene.landmarks,
            answer_map_landmarks=static_question.answer_map_landmarks,
            route_points=static_scene.route_points,
            route_current_index=static_scene.route_current_index,
            aircraft_prev=static_scene.aircraft_prev,
            aircraft_now=static_scene.aircraft_now,
            velocity=static_scene.velocity,
            show_aircraft_motion=static_scene.show_aircraft_motion,
            options=(),
            correct_code=0,
            correct_point=static_question.correct_point,
            correct_answer_token=static_question.correct_answer_token,
        ),
    )

    scorer = SpatialIntegrationScorer()
    assert scorer.score(problem=static_problem, user_answer=1, raw=static_question.correct_answer_token) == pytest.approx(1.0)
    expected_wrong = 1.0 if static_question.correct_answer_token == "A1" else 0.0
    assert scorer.score(problem=static_problem, user_answer=1, raw="A1") == pytest.approx(expected_wrong)

    aircraft_scene = gen.next_scene_cluster(part=SpatialIntegrationPart.AIRCRAFT, difficulty=0.5)
    aircraft_question = aircraft_scene.questions[0]
    correct_code = next(opt.code for opt in aircraft_question.options if opt.answer_token == "correct")
    wrong_code = next(opt.code for opt in aircraft_question.options if opt.answer_token != "correct")
    aircraft_problem = Problem(
        prompt=aircraft_question.stem,
        answer=int(correct_code),
        payload=SpatialIntegrationPayload(
            part=aircraft_scene.part,
            trial_stage=SpatialIntegrationTrialStage.QUESTION,
            block_kind="practice",
            scene_id=aircraft_scene.scene_id,
            scene_index_in_block=1,
            scenes_in_block=2,
            study_view_index=3,
            study_views_in_scene=3,
            question_index_in_scene=1,
            questions_in_scene=3,
            stage_time_remaining_s=8.0,
            part_time_remaining_s=None,
            kind=aircraft_question.kind,
            answer_mode=aircraft_question.answer_mode,
            stem=aircraft_question.stem,
            query_label=aircraft_question.query_label,
            north_arrow_deg=0,
            scene_view=aircraft_scene.scene_view,
            grid_cols=aircraft_scene.grid_cols,
            grid_rows=aircraft_scene.grid_rows,
            alt_levels=aircraft_scene.alt_levels,
            reference_views=aircraft_scene.reference_views,
            active_reference_view=None,
            hills=aircraft_scene.hills,
            landmarks=aircraft_scene.landmarks,
            answer_map_landmarks=aircraft_question.answer_map_landmarks,
            route_points=aircraft_scene.route_points,
            route_current_index=aircraft_scene.route_current_index,
            aircraft_prev=aircraft_scene.aircraft_prev,
            aircraft_now=aircraft_scene.aircraft_now,
            velocity=aircraft_scene.velocity,
            show_aircraft_motion=aircraft_scene.show_aircraft_motion,
            options=aircraft_question.options,
            correct_code=int(correct_code),
            correct_point=aircraft_question.correct_point,
            correct_answer_token=str(correct_code),
        ),
    )

    assert scorer.score(problem=aircraft_problem, user_answer=int(correct_code), raw=str(correct_code)) == pytest.approx(1.0)
    assert scorer.score(problem=aircraft_problem, user_answer=int(wrong_code), raw=str(wrong_code)) == pytest.approx(0.0)


def test_engine_advances_static_practice_scene_through_three_questions() -> None:
    clock = FakeClock()
    engine = build_spatial_integration_test(
        clock=clock,
        seed=7,
        difficulty=0.5,
        config=SpatialIntegrationConfig(
            practice_scenes_per_part=1,
            static_scored_duration_s=20.0,
            aircraft_scored_duration_s=20.0,
            static_study_s=0.2,
            aircraft_study_s=0.2,
            question_time_limit_s=0.3,
        ),
    )
    engine.start_practice()
    assert engine.phase is Phase.PRACTICE

    first_study = engine.snapshot().payload
    assert isinstance(first_study, SpatialIntegrationPayload)
    assert first_study.trial_stage is SpatialIntegrationTrialStage.STUDY
    assert first_study.study_view_index == 1

    clock.advance(0.08)
    engine.update()
    second_study = engine.snapshot().payload
    assert isinstance(second_study, SpatialIntegrationPayload)
    assert second_study.trial_stage is SpatialIntegrationTrialStage.STUDY
    assert second_study.study_view_index == 2

    clock.advance(0.08)
    engine.update()
    third_study = engine.snapshot().payload
    assert isinstance(third_study, SpatialIntegrationPayload)
    assert third_study.trial_stage is SpatialIntegrationTrialStage.STUDY
    assert third_study.study_view_index == 3

    payload_1 = _wait_for_question(engine=engine, clock=clock)
    assert payload_1.part is SpatialIntegrationPart.STATIC
    assert payload_1.question_index_in_scene == 1
    assert engine.submit_answer(payload_1.correct_answer_token) is True

    payload_2 = engine.snapshot().payload
    assert isinstance(payload_2, SpatialIntegrationPayload)
    assert payload_2.question_index_in_scene == 2
    assert engine.submit_answer(payload_2.correct_answer_token) is True

    payload_3 = engine.snapshot().payload
    assert isinstance(payload_3, SpatialIntegrationPayload)
    assert payload_3.question_index_in_scene == 3
    assert payload_3.answer_mode is SpatialIntegrationAnswerMode.OPTION_PICK
    assert engine.submit_answer(str(payload_3.correct_code)) is True
    assert engine.phase is Phase.PRACTICE_DONE


def test_engine_part_timer_finishes_current_question_then_rolls_forward() -> None:
    clock = FakeClock()
    engine = build_spatial_integration_test(
        clock=clock,
        seed=11,
        difficulty=0.55,
        config=SpatialIntegrationConfig(
            practice_scenes_per_part=1,
            static_scored_duration_s=0.01,
            aircraft_scored_duration_s=0.01,
            static_study_s=0.15,
            aircraft_study_s=0.15,
            question_time_limit_s=0.25,
        ),
    )
    engine.start_scored()
    assert engine.phase is Phase.SCORED

    payload = _wait_for_question(engine=engine, clock=clock)
    assert payload.part is SpatialIntegrationPart.STATIC
    assert engine.submit_answer(payload.correct_answer_token) is True
    assert engine.phase is Phase.PRACTICE_DONE

    summary = engine.scored_summary()
    assert summary.attempted == 1
    assert summary.correct == 1
