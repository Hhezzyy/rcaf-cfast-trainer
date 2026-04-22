from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest

from cfast_trainer.adaptive_difficulty import difficulty_ratio_for_level
from cfast_trainer.table_reading import (
    TableReadingAnswerMode,
    TableReadingConfig,
    TableReadingGenerator,
    TableReadingItemKind,
    TableReadingPart,
    TableReadingPayload,
    TableReadingScorer,
    build_table_reading_test,
    table_reading_table_size_for_difficulty,
    table_reading_family_for_payload,
)


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _signature(payload: TableReadingPayload) -> tuple[object, ...]:
    return (
        payload.part.value,
        payload.item_kind.value,
        payload.answer_mode.value,
        payload.primary_table.title,
        payload.primary_row_label,
        payload.primary_column_label,
        payload.secondary_row_label or "",
        payload.secondary_column_label or "",
        table_reading_family_for_payload(payload),
        payload.correct_code,
        payload.correct_value,
        payload.correct_answer_text,
        tuple((tab.title, tuple(table.title for table in tab.tables)) for tab in payload.data_tabs),
    )


def test_difficulty_maps_linearly_to_table_size() -> None:
    assert table_reading_table_size_for_difficulty(0.0) == 5
    assert table_reading_table_size_for_difficulty(0.5) == 28
    assert table_reading_table_size_for_difficulty(1.0) == 50


def test_table_reading_user_levels_map_linearly_to_table_size() -> None:
    sizes = tuple(
        table_reading_table_size_for_difficulty(
            difficulty_ratio_for_level("table_reading", level)
        )
        for level in range(1, 11)
    )

    assert sizes[0] == 5
    assert sizes[5] == 30
    assert sizes[-1] == 50
    assert sizes == tuple(range(5, 51, 5))


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


def test_generator_emits_all_item_kinds_and_both_parts() -> None:
    gen = TableReadingGenerator(seed=77)
    parts: set[TableReadingPart] = set()
    kinds: set[TableReadingItemKind] = set()
    for _ in range(60):
        payload = cast(TableReadingPayload, gen.next_problem(difficulty=0.7).payload)
        parts.add(payload.part)
        kinds.add(payload.item_kind)

    assert parts == {TableReadingPart.PART_ONE, TableReadingPart.PART_TWO}
    assert kinds == set(TableReadingItemKind)


def test_generator_exposes_multiple_card_families_for_both_parts() -> None:
    assert TableReadingGenerator.supported_part_one_families() == (
        "lookup",
        "station",
        "sector",
        "dispatch",
        "range",
    )
    assert TableReadingGenerator.supported_part_two_families() == (
        "wind",
        "crosswind",
        "offset",
        "descent",
        "timing",
    )


def test_generator_selection_hook_is_deterministic_for_same_family_and_part() -> None:
    gen_a = TableReadingGenerator(seed=404)
    gen_b = TableReadingGenerator(seed=404)

    problems_a = tuple(
        _signature(
            cast(
                TableReadingPayload,
                gen_a.next_problem_for_selection(
                    difficulty=0.6,
                    part=TableReadingPart.PART_TWO,
                    family="crosswind",
                    profile="prime",
                ).payload,
            )
        )
        for _ in range(4)
    )
    problems_b = tuple(
        _signature(
            cast(
                TableReadingPayload,
                gen_b.next_problem_for_selection(
                    difficulty=0.6,
                    part=TableReadingPart.PART_TWO,
                    family="crosswind",
                    profile="prime",
                ).payload,
            )
        )
        for _ in range(4)
    )

    assert problems_a == problems_b


def test_default_generator_rotates_across_multiple_part_one_and_part_two_card_families() -> None:
    gen = TableReadingGenerator(seed=909)
    part_one_families: set[str] = set()
    part_two_families: set[str] = set()

    for _ in range(80):
        payload = cast(TableReadingPayload, gen.next_problem(difficulty=0.7).payload)
        family = table_reading_family_for_payload(payload)
        if payload.part is TableReadingPart.PART_ONE:
            part_one_families.add(family)
        else:
            part_two_families.add(family)

    assert part_one_families == {"lookup", "station", "sector", "dispatch", "range", "letters"}
    assert part_two_families == {"wind", "crosswind", "offset", "descent", "timing"}


def test_scorer_supports_choice_numeric_reverse_and_letter_answers() -> None:
    gen = TableReadingGenerator(seed=19)
    scorer = TableReadingScorer()

    problem = gen.next_problem_for_selection(
        difficulty=0.5,
        part=TableReadingPart.PART_ONE,
        item_kind=TableReadingItemKind.SINGLE_TABLE_LOOKUP,
        answer_mode=TableReadingAnswerMode.MULTIPLE_CHOICE,
    )
    payload = cast(TableReadingPayload, problem.payload)
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

    numeric_problem = gen.next_problem_for_selection(
        difficulty=0.5,
        part=TableReadingPart.PART_ONE,
        item_kind=TableReadingItemKind.SINGLE_TABLE_LOOKUP,
        answer_mode=TableReadingAnswerMode.NUMERIC,
    )
    numeric_payload = cast(TableReadingPayload, numeric_problem.payload)
    assert scorer.score(
        problem=numeric_problem,
        user_answer=numeric_payload.correct_value,
        raw=numeric_payload.correct_answer_text,
    ) == 1.0

    reverse_problem = gen.next_problem_for_selection(
        difficulty=0.5,
        part=TableReadingPart.PART_ONE,
        item_kind=TableReadingItemKind.REVERSE_LOOKUP,
    )
    reverse_payload = cast(TableReadingPayload, reverse_problem.payload)
    assert scorer.score(problem=reverse_problem, user_answer=0, raw=reverse_payload.correct_answer_text) == 1.0
    assert scorer.score(problem=reverse_problem, user_answer=0, raw="ZZ") == 0.0

    letter_problem = gen.next_problem_for_selection(
        difficulty=0.5,
        item_kind=TableReadingItemKind.LETTER_SEARCH,
        answer_mode=TableReadingAnswerMode.LETTER_STRING,
    )
    letter_payload = cast(TableReadingPayload, letter_problem.payload)
    assert letter_payload.answer_mode is TableReadingAnswerMode.LETTER_STRING
    assert scorer.score(problem=letter_problem, user_answer=0, raw=letter_payload.correct_answer_text) == 1.0


def test_payload_tabs_match_item_table_depth() -> None:
    gen = TableReadingGenerator(seed=64)
    one = cast(
        TableReadingPayload,
        gen.next_problem_for_selection(
            difficulty=0.4,
            item_kind=TableReadingItemKind.SINGLE_TABLE_LOOKUP,
        ).payload,
    )
    two = cast(
        TableReadingPayload,
        gen.next_problem_for_selection(
            difficulty=0.4,
            item_kind=TableReadingItemKind.TWO_TABLE_LOOKUP,
        ).payload,
    )
    three = cast(
        TableReadingPayload,
        gen.next_problem_for_selection(
            difficulty=0.4,
            item_kind=TableReadingItemKind.THREE_TABLE_LOOKUP,
        ).payload,
    )

    assert tuple(tab.title for tab in one.data_tabs) == ("Table",)
    assert tuple(tab.title for tab in two.data_tabs) == ("Table 1", "Table 2")
    assert tuple(tab.title for tab in three.data_tabs) == ("Table 1", "Tables 2-3")
    assert len(three.data_tabs[1].tables) == 2


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
