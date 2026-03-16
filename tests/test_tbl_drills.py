from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest

from cfast_trainer.table_reading import (
    TableReadingGenerator,
    TableReadingPart,
    TableReadingPayload,
    TableReadingScorer,
    table_reading_family_for_payload,
)
from cfast_trainer.tbl_drills import (
    build_tbl_card_family_run_drill,
    build_tbl_distractor_grid_drill,
    build_tbl_lookup_compute_drill,
    build_tbl_mixed_tempo_drill,
    build_tbl_part1_anchor_drill,
    build_tbl_part1_scan_run_drill,
    build_tbl_part2_correction_run_drill,
    build_tbl_part2_prime_drill,
    build_tbl_part_switch_run_drill,
    build_tbl_pressure_run_drill,
    build_tbl_shrinking_cap_run_drill,
    build_tbl_single_lookup_anchor_drill,
    build_tbl_two_table_xref_drill,
)


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _problem_signature(engine) -> tuple[object, ...]:
    current = engine._current
    assert current is not None
    payload = current.payload
    assert isinstance(payload, TableReadingPayload)
    return (
        current.prompt,
        current.answer,
        payload.part.value,
        table_reading_family_for_payload(payload),
        payload.primary_table.title,
        payload.secondary_table.title if payload.secondary_table is not None else "",
        payload.primary_row_label,
        payload.primary_column_label,
        payload.secondary_row_label or "",
        payload.secondary_column_label or "",
        tuple((option.code, option.value) for option in payload.options),
    )


def _capture_signatures(builder, *, seed: int, count: int = 6) -> tuple[tuple[object, ...], ...]:
    clock = FakeClock()
    engine = builder(clock=clock, seed=seed, difficulty=0.5)
    engine.start_scored()
    captured: list[tuple[object, ...]] = []
    for _ in range(count):
        captured.append(_problem_signature(engine))
        answer = engine._current.answer
        assert engine.submit_answer(str(answer)) is True
    return tuple(captured)


def test_tbl_drill_families_are_deterministic_for_same_seed() -> None:
    builders = (
        build_tbl_part1_anchor_drill,
        build_tbl_part1_scan_run_drill,
        build_tbl_part2_prime_drill,
        build_tbl_part2_correction_run_drill,
        build_tbl_part_switch_run_drill,
        build_tbl_card_family_run_drill,
        build_tbl_mixed_tempo_drill,
        build_tbl_pressure_run_drill,
        build_tbl_single_lookup_anchor_drill,
        build_tbl_two_table_xref_drill,
        build_tbl_distractor_grid_drill,
        build_tbl_lookup_compute_drill,
        build_tbl_shrinking_cap_run_drill,
    )

    for builder in builders:
        assert _capture_signatures(builder, seed=1717) == _capture_signatures(builder, seed=1717)


@pytest.mark.parametrize(
    ("builder", "expected_part"),
    (
        (build_tbl_part1_anchor_drill, TableReadingPart.PART_ONE),
        (build_tbl_part1_scan_run_drill, TableReadingPart.PART_ONE),
        (build_tbl_part2_prime_drill, TableReadingPart.PART_TWO),
        (build_tbl_part2_correction_run_drill, TableReadingPart.PART_TWO),
        (build_tbl_single_lookup_anchor_drill, TableReadingPart.PART_ONE),
        (build_tbl_two_table_xref_drill, TableReadingPart.PART_TWO),
        (build_tbl_distractor_grid_drill, TableReadingPart.PART_ONE),
    ),
)
def test_tbl_focused_drills_emit_only_the_intended_part(builder, expected_part: TableReadingPart) -> None:
    clock = FakeClock()
    engine = builder(clock=clock, seed=81, difficulty=0.5)
    engine.start_scored()
    for _ in range(6):
        payload = engine._current.payload
        assert isinstance(payload, TableReadingPayload)
        assert payload.part is expected_part
        assert engine.submit_answer(str(engine._current.answer)) is True


def test_tbl_part_switch_run_alternates_parts_item_by_item() -> None:
    clock = FakeClock()
    engine = build_tbl_part_switch_run_drill(clock=clock, seed=404, difficulty=0.5)
    engine.start_scored()

    seen: list[TableReadingPart] = []
    for _ in range(6):
        payload = cast(TableReadingPayload, engine._current.payload)
        seen.append(payload.part)
        assert engine.submit_answer(str(engine._current.answer)) is True

    assert seen == [
        TableReadingPart.PART_ONE,
        TableReadingPart.PART_TWO,
        TableReadingPart.PART_ONE,
        TableReadingPart.PART_TWO,
        TableReadingPart.PART_ONE,
        TableReadingPart.PART_TWO,
    ]


def test_tbl_mixed_tempo_repeats_two_part1_then_two_part2() -> None:
    clock = FakeClock()
    engine = build_tbl_mixed_tempo_drill(clock=clock, seed=505, difficulty=0.5)
    engine.start_scored()

    seen: list[TableReadingPart] = []
    for _ in range(8):
        payload = cast(TableReadingPayload, engine._current.payload)
        seen.append(payload.part)
        assert engine.submit_answer(str(engine._current.answer)) is True

    assert seen == [
        TableReadingPart.PART_ONE,
        TableReadingPart.PART_ONE,
        TableReadingPart.PART_TWO,
        TableReadingPart.PART_TWO,
        TableReadingPart.PART_ONE,
        TableReadingPart.PART_ONE,
        TableReadingPart.PART_TWO,
        TableReadingPart.PART_TWO,
    ]


def test_tbl_card_family_run_cycles_the_expanded_library() -> None:
    clock = FakeClock()
    engine = build_tbl_card_family_run_drill(clock=clock, seed=606, difficulty=0.5)
    engine.start_scored()

    seen = []
    for _ in range(6):
        payload = cast(TableReadingPayload, engine._current.payload)
        seen.append((payload.part, table_reading_family_for_payload(payload)))
        assert engine.submit_answer(str(engine._current.answer)) is True

    expected_sequence = [
        (TableReadingPart.PART_ONE, family)
        for family in TableReadingGenerator.supported_part_one_families()
    ] + [
        (TableReadingPart.PART_TWO, family)
        for family in TableReadingGenerator.supported_part_two_families()
    ]
    assert seen == expected_sequence[:6]


def test_tbl_pressure_run_alternates_parts_and_keeps_partial_credit_scoring() -> None:
    clock = FakeClock()
    engine = build_tbl_pressure_run_drill(clock=clock, seed=707, difficulty=0.5)
    scorer = TableReadingScorer()
    engine.start_scored()

    seen: list[TableReadingPart] = []
    partial_scores: list[float] = []
    for _ in range(4):
        current = engine._current
        assert current is not None
        payload = cast(TableReadingPayload, current.payload)
        seen.append(payload.part)
        nearest_wrong = sorted(
            (
                abs(option.value - payload.correct_value),
                option.code,
            )
            for option in payload.options
            if option.value != payload.correct_value
        )[0][1]
        partial = scorer.score(problem=current, user_answer=int(nearest_wrong), raw=str(nearest_wrong))
        partial_scores.append(partial)
        assert engine.submit_answer(str(current.answer)) is True

    assert seen == [
        TableReadingPart.PART_ONE,
        TableReadingPart.PART_TWO,
        TableReadingPart.PART_ONE,
        TableReadingPart.PART_TWO,
    ]
    assert all(score == pytest.approx(0.5) for score in partial_scores)


@pytest.mark.parametrize(
    ("builder", "expects_secondary"),
    (
        (build_tbl_single_lookup_anchor_drill, False),
        (build_tbl_two_table_xref_drill, True),
    ),
)
def test_tbl_lookup_profiles_keep_expected_table_structure(builder, expects_secondary: bool) -> None:
    clock = FakeClock()
    engine = builder(clock=clock, seed=909, difficulty=0.5)
    engine.start_scored()

    for _ in range(6):
        payload = cast(TableReadingPayload, engine._current.payload)
        assert (payload.secondary_table is not None) is expects_secondary
        assert engine.submit_answer(str(engine._current.answer)) is True


def test_tbl_lookup_compute_requires_a_lookup_then_simple_transform() -> None:
    clock = FakeClock()
    engine = build_tbl_lookup_compute_drill(clock=clock, seed=303, difficulty=0.5)
    engine.start_scored()

    for _ in range(6):
        current = engine._current
        assert current is not None
        payload = cast(TableReadingPayload, current.payload)
        assert "Then apply" in current.prompt
        assert payload.correct_value == current.answer
        assert engine.submit_answer(str(current.answer)) is True


def test_tbl_shrinking_cap_run_tightens_the_per_item_cap() -> None:
    clock = FakeClock()
    engine = build_tbl_shrinking_cap_run_drill(clock=clock, seed=404, difficulty=0.5)
    engine.start_scored()

    caps: list[float] = []
    for _ in range(5):
        assert engine._current_cap_s is not None
        caps.append(float(engine._current_cap_s))
        assert engine.submit_answer(str(engine._current.answer)) is True

    assert caps == sorted(caps, reverse=True)
    assert len(set(round(cap, 4) for cap in caps)) > 1
