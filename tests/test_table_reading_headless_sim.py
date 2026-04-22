from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest

from cfast_trainer.cognitive_core import Phase
from cfast_trainer.table_reading import (
    TableReadingAnswerMode,
    TableReadingConfig,
    TableReadingGenerator,
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


def _nearest_wrong_code(payload: TableReadingPayload) -> int:
    ordered = sorted(
        (
            abs(option.value - payload.correct_value),
            option.code,
        )
        for option in payload.options
        if option.value != payload.correct_value
    )
    return int(ordered[0][1])


def _correct_submission(payload: TableReadingPayload) -> str:
    if payload.answer_mode is TableReadingAnswerMode.MULTIPLE_CHOICE:
        return str(payload.correct_code)
    return str(payload.correct_answer_text)


def _wrong_submission(payload: TableReadingPayload) -> str:
    if payload.answer_mode is TableReadingAnswerMode.MULTIPLE_CHOICE:
        return str(_nearest_wrong_code(payload))
    if payload.answer_mode is TableReadingAnswerMode.NUMERIC:
        return str(payload.correct_value + max(10, payload.estimate_tolerance * 3))
    return "ZZ"


def test_headless_scripted_run_produces_expected_summary_and_scores() -> None:
    seed = 909
    difficulty = 0.6
    clock = FakeClock()

    engine = build_table_reading_test(
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        config=TableReadingConfig(scored_duration_s=6.0, practice_questions=1),
    )

    mirror = TableReadingGenerator(seed=seed)
    scorer = TableReadingScorer()

    engine.start_practice()
    assert engine.phase is Phase.PRACTICE

    p_practice = mirror.next_problem(difficulty=difficulty)
    practice_payload = cast(TableReadingPayload, p_practice.payload)
    clock.advance(0.2)
    assert engine.submit_answer(_correct_submission(practice_payload)) is True
    assert engine.phase is Phase.PRACTICE_DONE

    engine.start_scored()
    assert engine.phase is Phase.SCORED

    expected_total_score = 0.0
    expected_full_correct = 0

    p1 = mirror.next_problem(difficulty=difficulty)
    p1_payload = cast(TableReadingPayload, p1.payload)
    a1 = _correct_submission(p1_payload)
    s1 = scorer.score(problem=p1, user_answer=0, raw=a1)
    expected_total_score += s1
    if s1 >= 1.0 - 1e-9:
        expected_full_correct += 1
    clock.advance(0.5)
    assert engine.submit_answer(a1) is True

    p2 = mirror.next_problem(difficulty=difficulty)
    p2_payload = cast(TableReadingPayload, p2.payload)
    a2 = _wrong_submission(p2_payload)
    s2 = scorer.score(problem=p2, user_answer=0, raw=a2)
    expected_total_score += s2
    if s2 >= 1.0 - 1e-9:
        expected_full_correct += 1
    clock.advance(0.5)
    assert engine.submit_answer(a2) is True

    p3 = mirror.next_problem(difficulty=difficulty)
    p3_payload = cast(TableReadingPayload, p3.payload)
    a3 = _wrong_submission(p3_payload)
    s3 = scorer.score(problem=p3, user_answer=0, raw=a3)
    expected_total_score += s3
    if s3 >= 1.0 - 1e-9:
        expected_full_correct += 1
    clock.advance(0.5)
    assert engine.submit_answer(a3) is True

    clock.advance(6.0)
    engine.update()
    assert engine.phase is Phase.RESULTS

    summary = engine.scored_summary()
    assert summary.attempted == 3
    assert summary.correct == expected_full_correct
    assert summary.accuracy == pytest.approx(expected_full_correct / 3)
    assert summary.throughput_per_min == pytest.approx(30.0)
    assert summary.mean_response_time_s == pytest.approx(0.5)
    assert summary.total_score == pytest.approx(expected_total_score)
    assert summary.max_score == pytest.approx(3.0)
    assert summary.score_ratio == pytest.approx(expected_total_score / 3.0)
