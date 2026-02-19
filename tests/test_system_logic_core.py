from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest

from cfast_trainer.system_logic import (
    SystemLogicConfig,
    SystemLogicGenerator,
    SystemLogicPayload,
    SystemLogicScorer,
    build_system_logic_test,
)


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _signature(payload: SystemLogicPayload) -> tuple[str, tuple[tuple[str, int], ...]]:
    folders = tuple((folder.name, len(folder.documents)) for folder in payload.folders)
    return payload.question, folders


def test_generator_determinism_same_seed_same_sequence() -> None:
    seed = 913
    gen_a = SystemLogicGenerator(seed=seed)
    gen_b = SystemLogicGenerator(seed=seed)

    seq_a = [gen_a.next_problem(difficulty=0.6) for _ in range(20)]
    seq_b = [gen_b.next_problem(difficulty=0.6) for _ in range(20)]

    view_a = [
        (problem.prompt, problem.answer, _signature(cast(SystemLogicPayload, problem.payload)))
        for problem in seq_a
    ]
    view_b = [
        (problem.prompt, problem.answer, _signature(cast(SystemLogicPayload, problem.payload)))
        for problem in seq_b
    ]
    assert view_a == view_b


def test_generated_payload_has_multiple_folders_and_submenus() -> None:
    gen = SystemLogicGenerator(seed=77)
    payload = cast(SystemLogicPayload, gen.next_problem(difficulty=0.5).payload)

    assert len(payload.folders) == 4
    assert all(len(folder.documents) >= 2 for folder in payload.folders)


def test_scorer_exact_and_estimation_behaviour() -> None:
    gen = SystemLogicGenerator(seed=41)
    problem = gen.next_problem(difficulty=0.75)
    payload = cast(SystemLogicPayload, problem.payload)
    scorer = SystemLogicScorer()

    exact = scorer.score(problem=problem, user_answer=problem.answer, raw=str(problem.answer))
    tolerance = max(1, payload.estimate_tolerance)
    near = scorer.score(
        problem=problem,
        user_answer=int(problem.answer) + tolerance,
        raw=str(int(problem.answer) + tolerance),
    )
    far = scorer.score(
        problem=problem,
        user_answer=int(problem.answer) + tolerance * 3,
        raw=str(int(problem.answer) + tolerance * 3),
    )

    assert exact == 1.0
    assert near == pytest.approx(0.5)
    assert far == 0.0


def test_timer_boundary_transitions_to_results_and_rejects_late_submit() -> None:
    clock = FakeClock()
    engine = build_system_logic_test(
        clock=clock,
        seed=123,
        difficulty=0.5,
        config=SystemLogicConfig(scored_duration_s=2.0, practice_questions=0),
    )
    engine.start_scored()

    clock.advance(2.0)
    engine.update()

    assert engine.phase.value == "results"
    assert engine.submit_answer("100") is False

    summary = engine.scored_summary()
    assert summary.attempted == 0
    assert summary.correct == 0
