from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import pytest

from cfast_trainer.numerical_operations import (
    NumericalOperationsConfig,
    NumericalOperationsGenerator,
    build_numerical_operations_test,
)
from cfast_trainer.persistence import ResultsStore
from cfast_trainer.rapid_tracking import RapidTrackingConfig, build_rapid_tracking_test
from cfast_trainer.results import attempt_result_from_engine


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _run_numerical_attempt(*, seed: int) -> object:
    clock = FakeClock()
    engine = build_numerical_operations_test(
        clock=clock,
        seed=seed,
        difficulty=0.8,
        config=NumericalOperationsConfig(scored_duration_s=6.0, practice_questions=1),
    )

    engine.start_practice()
    gen = NumericalOperationsGenerator(seed=seed)

    practice_problem = gen.next_problem(difficulty=0.8)
    clock.advance(0.2)
    assert engine.submit_answer(str(practice_problem.answer)) is True
    assert engine.phase.value == "practice_done"

    engine.start_scored()
    first_scored = gen.next_problem(difficulty=0.8)
    second_scored = gen.next_problem(difficulty=0.8)

    clock.advance(0.5)
    assert engine.submit_answer(str(first_scored.answer)) is True
    clock.advance(0.5)
    assert engine.submit_answer(str(second_scored.answer + 1)) is True

    clock.advance(6.0)
    engine.update()
    assert engine.phase.value == "results"
    return engine


def _run_rapid_tracking_attempt(*, seed: int) -> object:
    clock = FakeClock()
    engine = build_rapid_tracking_test(
        clock=clock,
        seed=seed,
        difficulty=0.5,
        config=RapidTrackingConfig(
            practice_duration_s=0.0,
            scored_duration_s=2.0,
            tick_hz=120.0,
        ),
    )

    engine.start_scored()
    clock.advance(2.0)
    engine.update()
    assert engine.phase.value == "results"
    return engine


def test_attempt_result_from_engine_handles_custom_summary_metrics_without_question_events() -> None:
    engine = _run_rapid_tracking_attempt(seed=17)

    result = attempt_result_from_engine(engine, test_code="rapid_tracking")
    summary = engine.scored_summary()

    assert result.attempted == summary.attempted
    assert result.duration_s == pytest.approx(summary.duration_s)
    assert result.events == []
    assert result.mean_rt_ms is None
    assert result.metrics["mean_error"] == f"{summary.mean_error:.6f}"
    assert result.metrics["capture_points"] == str(summary.capture_points)


def test_results_store_reuses_session_and_reads_session_summaries(tmp_path) -> None:
    store = ResultsStore(tmp_path / "results.sqlite3")
    numerical = attempt_result_from_engine(
        _run_numerical_attempt(seed=555),
        test_code="numerical_operations",
    )
    rapid = attempt_result_from_engine(
        _run_rapid_tracking_attempt(seed=44),
        test_code="rapid_tracking",
    )

    saved_one = store.record_attempt(
        result=numerical,
        app_version="test",
        input_profile_id="default",
    )
    saved_two = store.record_attempt(
        result=rapid,
        app_version="test",
        input_profile_id="default",
    )

    assert saved_one.session_id == saved_two.session_id

    session = store.session_summary()
    assert session is not None
    assert session.attempt_count == 2
    assert session.unique_tests == 2

    numerical_summary = store.test_session_summary("numerical_operations")
    assert numerical_summary is not None
    assert numerical_summary.attempt_count == 1
    assert numerical_summary.latest_accuracy == pytest.approx(numerical.accuracy)

    with sqlite3.connect(store.path) as conn:
        attempt_row = conn.execute(
            "SELECT COUNT(*), COUNT(input_profile_id) FROM attempt"
        ).fetchone()
        event_count = conn.execute("SELECT COUNT(*) FROM cognitive_event").fetchone()

    assert attempt_row == (2, 2)
    assert event_count == (numerical.attempted,)
