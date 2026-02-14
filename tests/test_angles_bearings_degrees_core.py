from __future__ import annotations

from dataclasses import dataclass

import pytest

from cfast_trainer.angles_bearings_degrees import (
    AnglesBearingsDegreesConfig,
    AnglesBearingsDegreesGenerator,
    AnglesBearingsDegreesPayload,
    AnglesBearingsQuestionKind,
    AnglesBearingsScorer,
    build_angles_bearings_degrees_test,
)
from cfast_trainer.cognitive_core import Problem


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def test_generator_determinism_same_seed_same_sequence() -> None:
    seed = 123
    g1 = AnglesBearingsDegreesGenerator(seed=seed)
    g2 = AnglesBearingsDegreesGenerator(seed=seed)

    seq1 = [g1.next_problem(difficulty=0.65) for _ in range(50)]
    seq2 = [g2.next_problem(difficulty=0.65) for _ in range(50)]

    assert [(p.prompt, p.answer, p.payload) for p in seq1] == [
        (p.prompt, p.answer, p.payload) for p in seq2
    ]


def test_scoring_exact_and_estimation_behavior() -> None:
    scorer = AnglesBearingsScorer()

    angle_payload = AnglesBearingsDegreesPayload(
        kind=AnglesBearingsQuestionKind.ANGLE_BETWEEN_LINES,
        reference_bearing_deg=10,
        target_bearing_deg=94,
        object_label="",
        full_credit_error_deg=2,
        zero_credit_error_deg=20,
    )
    angle_problem = Problem(
        prompt="Estimate angle",
        answer=84,
        payload=angle_payload,
    )

    assert scorer.score(problem=angle_problem, user_answer=84, raw="84") == 1.0
    assert scorer.score(problem=angle_problem, user_answer=86, raw="86") == 1.0
    assert scorer.score(problem=angle_problem, user_answer=130, raw="130") == 0.0

    angle_mid = scorer.score(problem=angle_problem, user_answer=94, raw="94")
    assert angle_mid == pytest.approx((20 - 10) / (20 - 2), abs=1e-9)

    bearing_payload = AnglesBearingsDegreesPayload(
        kind=AnglesBearingsQuestionKind.BEARING_FROM_REFERENCE,
        reference_bearing_deg=0,
        target_bearing_deg=355,
        object_label="A",
        full_credit_error_deg=2,
        zero_credit_error_deg=20,
    )
    bearing_problem = Problem(
        prompt="Estimate bearing",
        answer=355,
        payload=bearing_payload,
    )

    # Wrap-around check: 005 is 10 degrees away from 355.
    bearing_mid = scorer.score(problem=bearing_problem, user_answer=5, raw="5")
    assert bearing_mid == pytest.approx((20 - 10) / (20 - 2), abs=1e-9)


def test_timer_boundary_transitions_to_results_and_rejects_late_submit() -> None:
    clock = FakeClock()
    engine = build_angles_bearings_degrees_test(
        clock=clock,
        seed=7,
        difficulty=0.5,
        config=AnglesBearingsDegreesConfig(scored_duration_s=2.0, practice_questions=0),
    )
    engine.start_scored()

    assert engine.time_remaining_s() == pytest.approx(2.0)
    clock.advance(1.25)
    assert engine.time_remaining_s() == pytest.approx(0.75)

    clock.advance(0.75)
    engine.update()
    assert engine.phase.value == "results"
    assert engine.submit_answer("0") is False
