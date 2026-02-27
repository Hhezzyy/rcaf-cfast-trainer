from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest

from cfast_trainer.angles_bearings_degrees import (
    AnglesBearingsDegreesConfig,
    AnglesBearingsDegreesGenerator,
    AnglesBearingsDegreesPayload,
    AnglesBearingsQuestionKind,
    build_angles_bearings_degrees_test,
)


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


def test_generated_problem_has_four_unique_options_and_correct_code() -> None:
    gen = AnglesBearingsDegreesGenerator(seed=31)
    problem = gen.next_problem(difficulty=0.6)
    payload = cast(AnglesBearingsDegreesPayload, problem.payload)

    assert isinstance(payload, AnglesBearingsDegreesPayload)
    assert len(payload.options) == 4
    assert [opt.code for opt in payload.options] == [1, 2, 3, 4]

    values = [opt.value_deg for opt in payload.options]
    assert len(set(values)) == 4
    assert problem.answer == payload.correct_code
    assert any(
        opt.code == payload.correct_code and opt.value_deg == payload.correct_value_deg
        for opt in payload.options
    )


def test_generator_emits_both_angle_and_bearing_trials() -> None:
    gen = AnglesBearingsDegreesGenerator(seed=77)
    kinds: set[AnglesBearingsQuestionKind] = set()
    for _ in range(40):
        payload = cast(AnglesBearingsDegreesPayload, gen.next_problem(difficulty=0.7).payload)
        kinds.add(payload.kind)
    assert kinds == {
        AnglesBearingsQuestionKind.ANGLE_BETWEEN_LINES,
        AnglesBearingsQuestionKind.BEARING_FROM_REFERENCE,
    }


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
