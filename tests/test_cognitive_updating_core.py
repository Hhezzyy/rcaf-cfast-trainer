from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest

from cfast_trainer.cognitive_updating import (
    CognitiveUpdatingConfig,
    CognitiveUpdatingGenerator,
    CognitiveUpdatingPayload,
    CognitiveUpdatingScorer,
    build_cognitive_updating_test,
)


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _signature(payload: CognitiveUpdatingPayload) -> tuple[str, tuple[tuple[str, int], ...]]:
    panels = tuple((panel.name, len(panel.documents)) for panel in payload.panels)
    return payload.question, panels


def test_generator_determinism_same_seed_same_sequence() -> None:
    seed = 447
    gen_a = CognitiveUpdatingGenerator(seed=seed)
    gen_b = CognitiveUpdatingGenerator(seed=seed)

    seq_a = [gen_a.next_problem(difficulty=0.6) for _ in range(20)]
    seq_b = [gen_b.next_problem(difficulty=0.6) for _ in range(20)]

    view_a = [
        (problem.prompt, problem.answer, _signature(cast(CognitiveUpdatingPayload, problem.payload)))
        for problem in seq_a
    ]
    view_b = [
        (problem.prompt, problem.answer, _signature(cast(CognitiveUpdatingPayload, problem.payload)))
        for problem in seq_b
    ]
    assert view_a == view_b


def test_generated_payload_has_multiple_components_and_submenus() -> None:
    payload = cast(CognitiveUpdatingPayload, CognitiveUpdatingGenerator(seed=91).next_problem(difficulty=0.5).payload)

    assert len(payload.panels) == 5
    assert [panel.name for panel in payload.panels] == [
        "Messages",
        "Objectives",
        "Controls",
        "Sensors",
        "Engine",
    ]
    assert all(len(panel.documents) >= 2 for panel in payload.panels)


def test_scorer_exact_and_estimation_behaviour() -> None:
    problem = CognitiveUpdatingGenerator(seed=203).next_problem(difficulty=0.75)
    payload = cast(CognitiveUpdatingPayload, problem.payload)
    scorer = CognitiveUpdatingScorer()

    exact = scorer.score(problem=problem, user_answer=problem.answer, raw=str(problem.answer))
    tolerance = max(1, payload.estimate_tolerance)
    near = scorer.score(
        problem=problem,
        user_answer=int(problem.answer) + tolerance,
        raw=str(int(problem.answer) + tolerance),
    )
    far = scorer.score(
        problem=problem,
        user_answer=int(problem.answer) + (tolerance * 3),
        raw=str(int(problem.answer) + (tolerance * 3)),
    )

    assert exact == 1.0
    assert near == pytest.approx(0.5)
    assert far == 0.0


def test_timer_boundary_transitions_to_results_and_rejects_late_submit() -> None:
    clock = FakeClock()
    engine = build_cognitive_updating_test(
        clock=clock,
        seed=123,
        difficulty=0.5,
        config=CognitiveUpdatingConfig(scored_duration_s=2.0, practice_questions=0),
    )
    engine.start_scored()

    clock.advance(2.0)
    engine.update()

    assert engine.phase.value == "results"
    assert engine.submit_answer("222") is False

    summary = engine.scored_summary()
    assert summary.attempted == 0
    assert summary.correct == 0
