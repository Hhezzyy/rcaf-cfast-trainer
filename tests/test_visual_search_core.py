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


def test_generated_answer_matches_target_block_code() -> None:
    gen = VisualSearchGenerator(seed=123)
    p = gen.next_problem(difficulty=0.5)
    payload = p.payload
    assert isinstance(payload, VisualSearchPayload)
    assert payload.rows == 3
    assert payload.cols == 4
    assert sorted(payload.cell_codes) == list(range(10, 22))

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
    assert sorted(first_payload.cell_codes) == list(range(10, 22))
    assert sorted(second_payload.cell_codes) == list(range(10, 22))
    assert first_payload.cell_codes != second_payload.cell_codes


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


def test_harder_difficulty_uses_more_confusable_distractors() -> None:
    letter_clusters = (
        ("E", "F", "H", "K", "L"),
        ("A", "M", "R"),
        ("B", "G", "P", "R", "S"),
    )
    symbol_clusters = (
        ("X_MARK", "DOUBLE_CROSS", "STAR", "BOLT"),
        ("L_HOOK", "PIN", "FORK"),
        ("BOX", "TRIANGLE", "RING_SPOKE"),
        ("S_BEND", "LOLLIPOP"),
    )

    def confusable_share(problem_payload: VisualSearchPayload) -> float:
        if problem_payload.kind is VisualSearchTaskKind.ALPHANUMERIC:
            clusters = letter_clusters
        else:
            clusters = symbol_clusters
        confusable: set[str] = set()
        for cluster in clusters:
            if problem_payload.target in cluster:
                confusable.update(token for token in cluster if token != problem_payload.target)
        distractors = [token for token in problem_payload.cells if token != problem_payload.target]
        if not distractors:
            return 0.0
        return sum(1 for token in distractors if token in confusable) / float(len(distractors))

    low_gen = VisualSearchGenerator(seed=4321)
    high_gen = VisualSearchGenerator(seed=4321)

    low_share = sum(
        confusable_share(low_gen.next_problem(difficulty=0.1).payload) for _ in range(40)
    ) / 40.0
    high_share = sum(
        confusable_share(high_gen.next_problem(difficulty=0.9).payload) for _ in range(40)
    ) / 40.0

    assert high_share > low_share


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
