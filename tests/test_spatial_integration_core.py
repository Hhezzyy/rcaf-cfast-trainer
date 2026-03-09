from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest

from cfast_trainer.cognitive_core import Phase
from cfast_trainer.spatial_integration import (
    SpatialIntegrationConfig,
    SpatialIntegrationGenerator,
    SpatialIntegrationPayload,
    SpatialIntegrationQuestionKind,
    SpatialIntegrationSceneView,
    SpatialIntegrationScorer,
    SpatialIntegrationSection,
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
    max_steps: int = 80,
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


def test_generator_determinism_same_seed_same_sequence() -> None:
    seed = 515
    for section in (
        SpatialIntegrationSection.PART_A,
        SpatialIntegrationSection.PART_B,
        SpatialIntegrationSection.PART_C,
    ):
        g1 = SpatialIntegrationGenerator(seed=seed)
        g2 = SpatialIntegrationGenerator(seed=seed)

        seq1 = [g1.next_problem(section=section, difficulty=0.64) for _ in range(25)]
        seq2 = [g2.next_problem(section=section, difficulty=0.64) for _ in range(25)]

        assert [(p.prompt, p.answer, p.payload) for p in seq1] == [
            (p.prompt, p.answer, p.payload) for p in seq2
        ]


def test_generated_problem_has_five_unique_options_and_correct_code() -> None:
    gen = SpatialIntegrationGenerator(seed=18)
    for section in (
        SpatialIntegrationSection.PART_A,
        SpatialIntegrationSection.PART_B,
        SpatialIntegrationSection.PART_C,
    ):
        problem = gen.next_problem(section=section, difficulty=0.6)
        payload = cast(SpatialIntegrationPayload, problem.payload)

        assert isinstance(payload, SpatialIntegrationPayload)
        assert len(payload.options) == 5
        assert [opt.code for opt in payload.options] == [1, 2, 3, 4, 5]

        labels = [opt.label for opt in payload.options]
        assert len(set(labels)) == 5
        assert problem.answer == payload.correct_code
        assert any(
            opt.code == payload.correct_code and opt.point == payload.correct_point
            for opt in payload.options
        )

        if section is SpatialIntegrationSection.PART_A:
            assert payload.kind is SpatialIntegrationQuestionKind.LANDMARK_LOCATION
            assert payload.scene_view is SpatialIntegrationSceneView.TOPDOWN
            assert payload.show_aircraft_motion is False
        elif section is SpatialIntegrationSection.PART_B:
            assert payload.kind is SpatialIntegrationQuestionKind.LANDMARK_LOCATION
            assert payload.scene_view is SpatialIntegrationSceneView.OBLIQUE
            assert payload.show_aircraft_motion is False
        else:
            assert payload.kind is SpatialIntegrationQuestionKind.AIRCRAFT_EXTRAPOLATION
            assert payload.scene_view is SpatialIntegrationSceneView.OBLIQUE
            assert payload.show_aircraft_motion is True


def test_scorer_exact_and_partial_credit_behavior() -> None:
    gen = SpatialIntegrationGenerator(seed=90)
    problem = gen.next_problem(section=SpatialIntegrationSection.PART_C, difficulty=0.85)
    payload = cast(SpatialIntegrationPayload, problem.payload)
    scorer = SpatialIntegrationScorer()

    assert scorer.score(
        problem=problem, user_answer=payload.correct_code, raw=str(payload.correct_code)
    ) == pytest.approx(1.0)

    wrong = min(
        (opt for opt in payload.options if opt.code != payload.correct_code),
        key=lambda opt: int(opt.error),
    )
    full = int(payload.full_credit_error)
    zero = int(payload.zero_credit_error)
    if wrong.error <= full:
        expected = 1.0
    elif wrong.error >= zero:
        expected = 0.0
    else:
        expected = (zero - wrong.error) / float(zero - full)

    near = scorer.score(problem=problem, user_answer=wrong.code, raw=str(wrong.code))
    assert near == pytest.approx(expected, abs=1e-9)
    assert scorer.score(problem=problem, user_answer=9, raw="9") == pytest.approx(0.0)


def test_memorize_stage_transitions_to_question_and_accepts_answer() -> None:
    clock = FakeClock()
    engine = build_spatial_integration_test(
        clock=clock,
        seed=7,
        difficulty=0.5,
        config=SpatialIntegrationConfig(
            practice_questions=1,
            scored_questions_per_section=1,
            practice_memorize_s=0.2,
            scored_memorize_s=0.2,
        ),
    )
    engine.start_practice()
    assert engine.phase is Phase.PRACTICE

    snap = engine.snapshot()
    payload = cast(SpatialIntegrationPayload, snap.payload)
    assert payload.trial_stage is SpatialIntegrationTrialStage.MEMORIZE
    assert engine.submit_answer(str(payload.correct_code)) is False
    assert engine.time_remaining_s() == pytest.approx(0.2, abs=0.05)

    payload_q = _wait_for_question(engine=engine, clock=clock)
    assert payload_q.trial_stage is SpatialIntegrationTrialStage.QUESTION
    assert engine.time_remaining_s() is None
    assert engine.submit_answer(str(payload_q.correct_code)) is True
    assert engine.phase is Phase.PRACTICE_DONE


def test_skip_controls_allow_fast_navigation() -> None:
    clock = FakeClock()
    engine = build_spatial_integration_test(
        clock=clock,
        seed=11,
        difficulty=0.55,
        config=SpatialIntegrationConfig(
            practice_questions=2,
            scored_questions_per_section=2,
            practice_memorize_s=0.2,
            scored_memorize_s=0.2,
        ),
    )
    engine.start_practice()
    assert engine.phase is Phase.PRACTICE

    assert engine.submit_answer("__skip_practice__") is True
    assert engine.phase is Phase.PRACTICE_DONE

    engine.start_scored()
    assert engine.phase is Phase.SCORED

    assert engine.submit_answer("__skip_section__") is True
    assert engine.phase is Phase.PRACTICE_DONE

    assert engine.submit_answer("__skip_all__") is True
    assert engine.phase is Phase.RESULTS
