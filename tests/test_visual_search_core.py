from __future__ import annotations

from dataclasses import dataclass

import pytest

from cfast_trainer.cognitive_core import Problem
from cfast_trainer.visual_search import (
    VisualSearchConfig,
    VisualSearchGenerator,
    VisualSearchPayload,
    VisualSearchProfile,
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


def _assert_valid_block_codes(payload: VisualSearchPayload) -> None:
    cell_count = payload.rows * payload.cols
    assert len(payload.cell_codes) == cell_count
    assert len(set(payload.cell_codes)) == cell_count
    assert all(10 <= code <= 99 for code in payload.cell_codes)
    assert all(len(str(code)) == 2 for code in payload.cell_codes)


def test_generated_answer_matches_target_block_code() -> None:
    gen = VisualSearchGenerator(seed=123)
    p = gen.next_problem(difficulty=0.5)
    payload = p.payload
    assert isinstance(payload, VisualSearchPayload)
    assert payload.rows == 4
    assert payload.cols == 5
    _assert_valid_block_codes(payload)

    target_indices = [i for i, tok in enumerate(payload.cells) if tok == payload.target]
    assert len(target_indices) == 1
    idx = target_indices[0]
    assert p.answer == payload.cell_codes[idx]


def test_block_numbers_move_between_questions() -> None:
    gen = VisualSearchGenerator(seed=456)

    first = gen.next_problem(difficulty=0.5)
    second = gen.next_problem(difficulty=0.5)
    first_payload = first.payload
    second_payload = second.payload

    assert isinstance(first_payload, VisualSearchPayload)
    assert isinstance(second_payload, VisualSearchPayload)
    _assert_valid_block_codes(first_payload)
    _assert_valid_block_codes(second_payload)
    assert first_payload.cell_codes != second_payload.cell_codes


def test_official_grid_scales_from_3x4_to_7x6() -> None:
    gen = VisualSearchGenerator(seed=321)

    low_payload = gen.next_problem(difficulty=0.0).payload
    high_payload = gen.next_problem(difficulty=1.0).payload

    assert isinstance(low_payload, VisualSearchPayload)
    assert isinstance(high_payload, VisualSearchPayload)
    assert (low_payload.rows, low_payload.cols) == (3, 4)
    assert (high_payload.rows, high_payload.cols) == (7, 6)
    _assert_valid_block_codes(high_payload)


def _difficulty_for_level(level: int) -> float:
    clamped = max(1, min(10, int(level)))
    return float(clamped - 1) / 9.0


def test_level_8_still_uses_mixed_base_high_band_before_same_base_overload() -> None:
    gen = VisualSearchGenerator(seed=654)
    payload = gen.next_problem(difficulty=_difficulty_for_level(8)).payload

    assert isinstance(payload, VisualSearchPayload)
    assert len({VisualSearchGenerator.token_base(token) for token in payload.cells}) > 1


def test_level_9_symbol_family_switches_to_same_base_overload() -> None:
    gen = VisualSearchGenerator(
        seed=654,
        profile=VisualSearchProfile(
            allowed_kinds=(VisualSearchTaskKind.SYMBOL_CODE,),
            high_band_symbol_only=False,
        ),
    )
    payload = gen.next_problem(difficulty=_difficulty_for_level(9)).payload

    assert isinstance(payload, VisualSearchPayload)
    target_base = VisualSearchGenerator.token_base(payload.target)
    assert all(VisualSearchGenerator.token_base(token) == target_base for token in payload.cells)
    assert any("+" in token for token in payload.cells)


def test_level_10_symbol_family_uses_unique_same_base_boards() -> None:
    gen = VisualSearchGenerator(
        seed=654,
        profile=VisualSearchProfile(
            allowed_kinds=(VisualSearchTaskKind.SYMBOL_CODE,),
            high_band_symbol_only=False,
        ),
    )
    payload = gen.next_problem(difficulty=1.0).payload

    assert isinstance(payload, VisualSearchPayload)
    assert payload.kind is VisualSearchTaskKind.SYMBOL_CODE
    assert len(set(payload.cells)) == len(payload.cells)
    assert "@" in payload.target
    assert all("@" in token for token in payload.cells)
    target_base = VisualSearchGenerator.token_base(payload.target)
    assert all(VisualSearchGenerator.token_base(token) == target_base for token in payload.cells)
    assert any("+" in token for token in payload.cells)
    _assert_valid_block_codes(payload)


def test_level_9_alphanumeric_family_uses_three_character_string_boards() -> None:
    gen = VisualSearchGenerator(
        seed=765,
        profile=VisualSearchProfile(
            allowed_kinds=(VisualSearchTaskKind.ALPHANUMERIC,),
            high_band_symbol_only=False,
        ),
    )
    payload = gen.next_problem(difficulty=_difficulty_for_level(9)).payload

    assert isinstance(payload, VisualSearchPayload)
    assert payload.kind is VisualSearchTaskKind.ALPHANUMERIC
    assert len(payload.target) == 3
    assert all("@" not in token for token in payload.cells)
    assert all(len(token) == 3 for token in payload.cells)
    assert len(set(payload.cells)) == len(payload.cells)


def test_level_10_alphanumeric_family_uses_four_character_string_boards() -> None:
    gen = VisualSearchGenerator(
        seed=766,
        profile=VisualSearchProfile(
            allowed_kinds=(VisualSearchTaskKind.ALPHANUMERIC,),
            high_band_symbol_only=False,
        ),
    )
    payload = gen.next_problem(difficulty=1.0).payload

    assert isinstance(payload, VisualSearchPayload)
    assert payload.kind is VisualSearchTaskKind.ALPHANUMERIC
    assert len(payload.target) == 4
    assert all("@" not in token for token in payload.cells)
    assert all(len(token) == 4 for token in payload.cells)
    assert len(set(payload.cells)) == len(payload.cells)


def test_profile_restricts_generator_to_selected_family() -> None:
    gen = VisualSearchGenerator(
        seed=987,
        profile=VisualSearchProfile(allowed_kinds=(VisualSearchTaskKind.ALPHANUMERIC,)),
    )

    payloads = [gen.next_problem(difficulty=0.7).payload for _ in range(12)]

    assert all(isinstance(payload, VisualSearchPayload) for payload in payloads)
    assert {
        payload.kind
        for payload in payloads
        if isinstance(payload, VisualSearchPayload)
    } == {VisualSearchTaskKind.ALPHANUMERIC}


def test_top_band_same_base_rule_is_stricter_than_mid_band_similarity() -> None:
    profile = VisualSearchProfile(
        allowed_kinds=(VisualSearchTaskKind.SYMBOL_CODE,),
        high_band_symbol_only=False,
    )
    low_gen = VisualSearchGenerator(seed=4321, profile=profile)
    high_gen = VisualSearchGenerator(seed=4321, profile=profile)

    low_payload = low_gen.next_problem(difficulty=_difficulty_for_level(8)).payload
    high_payload = high_gen.next_problem(difficulty=_difficulty_for_level(9)).payload

    assert isinstance(low_payload, VisualSearchPayload)
    assert isinstance(high_payload, VisualSearchPayload)
    assert len({VisualSearchGenerator.token_base(token) for token in low_payload.cells}) > 1
    assert len({VisualSearchGenerator.token_base(token) for token in high_payload.cells}) == 1


def test_token_marks_parses_single_and_multi_mark_tokens() -> None:
    assert VisualSearchGenerator.token_marks("BOX@TR") == ("TR",)
    assert VisualSearchGenerator.token_marks("BOX@T+TR") == ("T", "TR")
    assert VisualSearchGenerator.token_mark("BOX@T+TR") == "T+TR"


def test_scoring_exact_and_estimation_behavior() -> None:
    scorer = VisualSearchScorer()
    payload = VisualSearchPayload(
        kind=VisualSearchTaskKind.ALPHANUMERIC,
        rows=2,
        cols=3,
        target="A7",
        cells=("A7", "B1", "A7", "D1", "E1", "B7"),
        cell_codes=(12, 57, 44, 3, 98, 6),
        full_credit_error=0,
        zero_credit_error=1,
    )
    problem = Problem(prompt="Find A7", answer=57, payload=payload)

    assert scorer.score(problem=problem, user_answer=57, raw="57") == 1.0
    assert scorer.score(problem=problem, user_answer=58, raw="58") == 0.0
    assert scorer.score(problem=problem, user_answer=62, raw="62") == 0.0
    assert scorer.score(problem=problem, user_answer=70, raw="70") == 0.0


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
