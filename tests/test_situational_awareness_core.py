from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest

from cfast_trainer.situational_awareness import (
    SituationalAwarenessConfig,
    SituationalAwarenessGenerator,
    SituationalAwarenessPayload,
    SituationalAwarenessQuestionKind,
    build_situational_awareness_test,
    project_contact_cell,
)


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


def _first_problem_of_kind(
    gen: SituationalAwarenessGenerator,
    *,
    difficulty: float,
    kind: SituationalAwarenessQuestionKind,
) -> tuple[SituationalAwarenessPayload, object]:
    for _ in range(160):
        problem = gen.next_problem(difficulty=difficulty)
        payload = cast(SituationalAwarenessPayload, problem.payload)
        if payload.kind is kind:
            return payload, problem
    raise AssertionError(f"Failed to generate kind={kind} within bounded attempts.")


def test_generator_determinism_same_seed_same_sequence() -> None:
    seed = 308
    g1 = SituationalAwarenessGenerator(seed=seed)
    g2 = SituationalAwarenessGenerator(seed=seed)

    seq1 = [g1.next_problem(difficulty=0.62) for _ in range(30)]
    seq2 = [g2.next_problem(difficulty=0.62) for _ in range(30)]

    assert [(p.prompt, p.answer, p.payload) for p in seq1] == [
        (p.prompt, p.answer, p.payload) for p in seq2
    ]


def test_projection_problem_answer_matches_projected_contact_cell() -> None:
    gen = SituationalAwarenessGenerator(seed=91)
    payload, problem = _first_problem_of_kind(
        gen,
        difficulty=0.5,
        kind=SituationalAwarenessQuestionKind.POSITION_PROJECTION,
    )

    assert payload.query_callsign is not None
    query_contact = next(c for c in payload.contacts if c.callsign == payload.query_callsign)
    expected = project_contact_cell(query_contact, payload.horizon_min)

    assert payload.correct_cell == expected
    assert payload.correct_code == problem.answer


def test_conflict_problem_contains_pair_that_projects_to_same_cell() -> None:
    gen = SituationalAwarenessGenerator(seed=111)
    payload, problem = _first_problem_of_kind(
        gen,
        difficulty=0.8,
        kind=SituationalAwarenessQuestionKind.CONFLICT_PREDICTION,
    )

    assert payload.conflict_pair is not None
    left = next(c for c in payload.contacts if c.callsign == payload.conflict_pair[0])
    right = next(c for c in payload.contacts if c.callsign == payload.conflict_pair[1])

    left_final = project_contact_cell(left, payload.horizon_min)
    right_final = project_contact_cell(right, payload.horizon_min)
    assert left_final == right_final

    correct_option = next(opt for opt in payload.options if opt.code == payload.correct_code)
    assert payload.conflict_pair[0] in correct_option.text
    assert payload.conflict_pair[1] in correct_option.text
    assert payload.correct_code == problem.answer


def test_timer_boundary_transitions_to_results_and_rejects_late_submit() -> None:
    clock = FakeClock()
    engine = build_situational_awareness_test(
        clock=clock,
        seed=17,
        difficulty=0.5,
        config=SituationalAwarenessConfig(scored_duration_s=2.0, practice_questions=0),
    )
    engine.start_scored()

    clock.advance(2.0)
    engine.update()

    assert engine.phase.value == "results"
    assert engine.submit_answer("1") is False


def test_default_config_uses_shorter_training_duration() -> None:
    cfg = SituationalAwarenessConfig()
    assert cfg.scored_duration_s == pytest.approx(12.0 * 60.0)
