from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest

from cfast_trainer.table_reading import (
    TableReadingConfig,
    TableReadingGenerator,
    TableReadingPart,
    TableReadingPayload,
    TableReadingScorer,
    build_table_reading_test,
)


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _signature(payload: TableReadingPayload) -> tuple[str, str, str, str, str, int, int]:
    return (
        payload.part.value,
        payload.primary_row_label,
        payload.primary_column_label,
        payload.secondary_row_label or "",
        payload.secondary_column_label or "",
        payload.correct_code,
        payload.correct_value,
    )


def test_generator_determinism_same_seed_same_sequence() -> None:
    seed = 321
    difficulty = 0.65

    gen_a = TableReadingGenerator(seed=seed)
    gen_b = TableReadingGenerator(seed=seed)

    seq_a = [gen_a.next_problem(difficulty=difficulty) for _ in range(30)]
    seq_b = [gen_b.next_problem(difficulty=difficulty) for _ in range(30)]

    view_a = [
        (
            problem.prompt,
            problem.answer,
            _signature(cast(TableReadingPayload, problem.payload)),
        )
        for problem in seq_a
    ]
    view_b = [
        (
            problem.prompt,
            problem.answer,
            _signature(cast(TableReadingPayload, problem.payload)),
        )
        for problem in seq_b
    ]

    assert view_a == view_b


def test_generator_emits_both_parts() -> None:
    gen = TableReadingGenerator(seed=77)
    parts: set[TableReadingPart] = set()
    for _ in range(60):
        payload = cast(TableReadingPayload, gen.next_problem(difficulty=0.7).payload)
        parts.add(payload.part)

    assert parts == {TableReadingPart.PART_ONE, TableReadingPart.PART_TWO}


def test_scorer_exact_and_estimation_behaviour() -> None:
    gen = TableReadingGenerator(seed=19)
    problem = gen.next_problem(difficulty=0.5)
    payload = cast(TableReadingPayload, problem.payload)
    scorer = TableReadingScorer()

    exact = scorer.score(
        problem=problem,
        user_answer=payload.correct_value,
        raw=str(payload.correct_value),
    )

    near_code = sorted(
        (
            abs(option.value - payload.correct_value),
            option.code,
        )
        for option in payload.options
        if option.value != payload.correct_value
    )[0][1]
    near = scorer.score(
        problem=problem,
        user_answer=int(near_code),
        raw=str(near_code),
    )

    far_value = payload.correct_value + max(10, payload.estimate_tolerance * 3)
    far = scorer.score(
        problem=problem,
        user_answer=far_value,
        raw=str(far_value),
    )

    via_code = scorer.score(
        problem=problem,
        user_answer=payload.correct_code,
        raw=str(payload.correct_code),
    )

    assert exact == 1.0
    assert near == pytest.approx(0.5)
    assert far == 0.0
    assert via_code == 1.0


def test_timer_boundary_transitions_to_results_and_rejects_late_submit() -> None:
    clock = FakeClock()
    engine = build_table_reading_test(
        clock=clock,
        seed=123,
        difficulty=0.5,
        config=TableReadingConfig(scored_duration_s=2.0, practice_questions=0),
    )
    engine.start_scored()

    clock.advance(2.0)
    engine.update()

    assert engine.phase.value == "results"
    assert engine.submit_answer("1") is False

    summary = engine.scored_summary()
    assert summary.attempted == 0
    assert summary.correct == 0
