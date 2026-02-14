from __future__ import annotations

from dataclasses import dataclass

import pytest

from cfast_trainer.cognitive_core import Problem
from cfast_trainer.visual_search import (
    VisualSearchConfig,
    VisualSearchGenerator,
    VisualSearchPayload,
    VisualSearchScorer,
    VisualSearchTaskKind,
    build_visual_search_test,
)


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


def test_generator_determinism_same_seed_same_sequence() -> None:
    seed = 2468
    g1 = VisualSearchGenerator(seed=seed)
    g2 = VisualSearchGenerator(seed=seed)

    seq1 = [g1.next_problem(difficulty=0.55) for _ in range(25)]
    seq2 = [g2.next_problem(difficulty=0.55) for _ in range(25)]

    assert [(p.prompt, p.answer, p.payload) for p in seq1] == [
        (p.prompt, p.answer, p.payload) for p in seq2
    ]


def test_scoring_exact_and_estimation_behavior() -> None:
    scorer = VisualSearchScorer()
    payload = VisualSearchPayload(
        kind=VisualSearchTaskKind.ALPHANUMERIC,
        rows=2,
        cols=3,
        target="A7",
        cells=("A7", "B1", "A7", "D1", "E1", "B7"),
        full_credit_error=0,
        zero_credit_error=3,
    )
    problem = Problem(prompt="Count A7", answer=2, payload=payload)

    assert scorer.score(problem=problem, user_answer=2, raw="2") == 1.0
    assert scorer.score(problem=problem, user_answer=3, raw="3") == pytest.approx(2 / 3, abs=1e-9)
    assert scorer.score(problem=problem, user_answer=0, raw="0") == pytest.approx(1 / 3, abs=1e-9)
    assert scorer.score(problem=problem, user_answer=5, raw="5") == 0.0


def test_timer_boundary_transitions_to_results_and_rejects_late_submit() -> None:
    clock = FakeClock()
    engine = build_visual_search_test(
        clock=clock,
        seed=7,
        difficulty=0.5,
        config=VisualSearchConfig(scored_duration_s=2.0, practice_questions=0),
    )
    engine.start_scored()

    assert engine.time_remaining_s() == pytest.approx(2.0)
    clock.advance(2.0)
    engine.update()

    assert engine.phase.value == "results"
    assert engine.submit_answer("0") is False
