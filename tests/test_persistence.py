from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import pytest

from cfast_trainer.adaptive_difficulty import build_resolved_difficulty_context, family_id_for_code
from cfast_trainer.ant_drills import AntDifficultyChange
from cfast_trainer.cognitive_core import AttemptSummary, Phase, QuestionEvent
from cfast_trainer.numerical_operations import (
    NumericalOperationsConfig,
    NumericalOperationsGenerator,
    build_numerical_operations_test,
)
from cfast_trainer.persistence import SCHEMA_VERSION, ResultsStore, load_session_summary, record_attempt
from cfast_trainer.rapid_tracking import RapidTrackingConfig, build_rapid_tracking_test
from cfast_trainer.results import attempt_result_from_engine

_COMMON_METRIC_KEYS = (
    "attempted",
    "correct",
    "score_ratio",
    "duration_s",
    "completed",
    "aborted",
    "mean_rt_ms",
    "median_rt_ms",
    "rt_variance_ms2",
    "timeout_count",
    "timeout_rate",
    "longest_lapse_streak",
    "first_half_accuracy",
    "second_half_accuracy",
    "first_half_mean_rt_ms",
    "second_half_mean_rt_ms",
    "first_3m_accuracy",
    "last_3m_accuracy",
    "first_3m_timeout_rate",
    "last_3m_timeout_rate",
    "post_error_next_item_rt_inflation_ms",
    "post_error_next_item_accuracy_drop",
    "difficulty_level_start",
    "difficulty_level_end",
    "difficulty_change_count",
)


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


@dataclass
class _StaticTelemetryEngine:
    summary: AttemptSummary
    question_events: list[QuestionEvent]
    phase: Phase = Phase.RESULTS
    seed: int = 321
    difficulty: float = 4.0 / 9.0
    practice_questions: int = 1
    scored_duration_s: float = 120.0
    scored_started_at_s: float = 0.0
    difficulty_changes_seq: tuple[AntDifficultyChange, ...] = ()
    _result_metrics_overrides: dict[str, str] | None = None
    _resolved_difficulty_context: object | None = None

    @property
    def _difficulty(self) -> float:
        return float(self.difficulty)

    @property
    def _scored_started_at_s(self) -> float:
        return float(self.scored_started_at_s)

    def events(self) -> list[QuestionEvent]:
        return list(self.question_events)

    def scored_summary(self) -> AttemptSummary:
        return self.summary

    def difficulty_changes(self) -> list[AntDifficultyChange]:
        return list(self.difficulty_changes_seq)


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


def _build_static_telemetry_engine(*, phase: Phase = Phase.RESULTS) -> _StaticTelemetryEngine:
    return _StaticTelemetryEngine(
        phase=phase,
        summary=AttemptSummary(
            attempted=4,
            correct=2,
            accuracy=0.5,
            duration_s=120.0,
            throughput_per_min=2.0,
            mean_response_time_s=1.2,
            total_score=2.0,
            max_score=4.0,
            score_ratio=0.5,
        ),
        question_events=[
            QuestionEvent(
                index=1,
                phase=Phase.SCORED,
                prompt="Q1",
                correct_answer=1,
                user_answer=1,
                is_correct=True,
                presented_at_s=9.6,
                answered_at_s=10.0,
                response_time_s=0.4,
                raw="1",
                score=1.0,
            ),
            QuestionEvent(
                index=2,
                phase=Phase.SCORED,
                prompt="Q2",
                correct_answer=2,
                user_answer=0,
                is_correct=False,
                presented_at_s=29.4,
                answered_at_s=30.0,
                response_time_s=0.6,
                raw="__timeout__",
                score=0.0,
                is_timeout=True,
            ),
            QuestionEvent(
                index=3,
                phase=Phase.SCORED,
                prompt="Q3",
                correct_answer=3,
                user_answer=4,
                is_correct=False,
                presented_at_s=48.2,
                answered_at_s=50.0,
                response_time_s=1.8,
                raw="4",
                score=0.0,
            ),
            QuestionEvent(
                index=4,
                phase=Phase.SCORED,
                prompt="Q4",
                correct_answer=4,
                user_answer=4,
                is_correct=True,
                presented_at_s=108.0,
                answered_at_s=110.0,
                response_time_s=2.0,
                raw="4",
                score=1.0,
            ),
        ],
        difficulty_changes_seq=(
            AntDifficultyChange(
                after_attempt=3,
                old_level=5,
                new_level=7,
                reason="accuracy_high",
            ),
        ),
    )


def _create_v1_results_db(path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE session (
                id INTEGER PRIMARY KEY,
                created_at_utc TEXT NOT NULL
            );
            CREATE TABLE attempt (
                id INTEGER PRIMARY KEY,
                session_id INTEGER NOT NULL REFERENCES session(id) ON DELETE CASCADE,
                test_code TEXT NOT NULL,
                test_version INTEGER NOT NULL,
                app_version TEXT NOT NULL,
                rng_seed INTEGER NOT NULL,
                difficulty REAL NOT NULL,
                input_profile_id TEXT,
                practice_questions INTEGER NOT NULL,
                scored_duration_s REAL NOT NULL,
                started_at_utc TEXT NOT NULL,
                completed_at_utc TEXT NOT NULL
            );
            CREATE TABLE metric (
                attempt_id INTEGER NOT NULL REFERENCES attempt(id) ON DELETE CASCADE,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                PRIMARY KEY (attempt_id, key)
            );
            CREATE TABLE cognitive_event (
                id INTEGER PRIMARY KEY,
                attempt_id INTEGER NOT NULL REFERENCES attempt(id) ON DELETE CASCADE,
                seq INTEGER NOT NULL,
                phase TEXT NOT NULL,
                prompt TEXT NOT NULL,
                expected TEXT NOT NULL,
                response TEXT NOT NULL,
                is_correct INTEGER NOT NULL,
                presented_at_ms INTEGER NOT NULL,
                answered_at_ms INTEGER NOT NULL,
                rt_ms INTEGER NOT NULL
            );
            """
        )
        conn.execute(
            "INSERT INTO session(id, created_at_utc) VALUES (?, ?)",
            (1, "2026-03-15T00:00:00Z"),
        )
        conn.execute(
            """
            INSERT INTO attempt(
                id, session_id, test_code, test_version, app_version, rng_seed, difficulty,
                input_profile_id, practice_questions, scored_duration_s, started_at_utc,
                completed_at_utc
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                1,
                1,
                "legacy_numerical",
                1,
                "v1",
                99,
                0.5,
                "legacy",
                1,
                60.0,
                "2026-03-15T00:00:05Z",
                "2026-03-15T00:01:05Z",
            ),
        )
        conn.execute(
            "INSERT INTO metric(attempt_id, key, value) VALUES (?, ?, ?)",
            (1, "accuracy", "1.000000"),
        )
        conn.execute(
            """
            INSERT INTO cognitive_event(
                attempt_id, seq, phase, prompt, expected, response, is_correct,
                presented_at_ms, answered_at_ms, rt_ms
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (1, 0, "scored", "Legacy Q", "2", "2", 1, 1000, 1500, 500),
        )
        conn.execute("PRAGMA user_version=1")


def test_attempt_result_from_engine_handles_custom_summary_metrics_without_question_events() -> None:
    engine = _run_rapid_tracking_attempt(seed=17)

    result = attempt_result_from_engine(engine, test_code="rapid_tracking")
    summary = engine.scored_summary()

    assert result.attempted == summary.attempted
    assert result.duration_s == pytest.approx(summary.duration_s)
    assert len(result.events) == summary.attempted
    assert all(event.family == "question" for event in result.events)
    assert result.mean_rt_ms is not None
    assert result.metrics["mean_error"] == f"{summary.mean_error:.6f}"
    assert result.metrics["capture_points"] == str(summary.capture_points)
    assert result.metrics["rms_tracking_error"] == f"{summary.rms_error:.6f}"
    assert result.metrics["overshoot_count"] != ""
    assert result.metrics["reversal_count"] != ""


def test_attempt_result_from_engine_emits_required_common_metrics_for_completed_and_aborted() -> None:
    completed = attempt_result_from_engine(
        _build_static_telemetry_engine(),
        test_code="telemetry_static",
    )
    aborted = attempt_result_from_engine(
        _build_static_telemetry_engine(phase=Phase.SCORED),
        test_code="telemetry_static",
    )

    for result in (completed, aborted):
        for key in _COMMON_METRIC_KEYS:
            assert key in result.metrics

    assert completed.metrics["completed"] == "1"
    assert completed.metrics["aborted"] == "0"
    assert aborted.metrics["completed"] == "0"
    assert aborted.metrics["aborted"] == "1"


def test_attempt_result_from_engine_computes_short_run_telemetry_analytics() -> None:
    engine = _build_static_telemetry_engine()

    result = attempt_result_from_engine(engine, test_code="telemetry_static")

    assert result.attempted == 4
    assert result.correct == 2
    assert result.mean_rt_ms == pytest.approx(1200.0)
    assert result.median_rt_ms == pytest.approx(1200.0)
    assert result.difficulty_level_start == 5
    assert result.difficulty_level_end == 7
    assert len(result.events) == 5
    assert result.metrics["rt_variance_ms2"] == "500000.000000"
    assert result.metrics["timeout_count"] == "1"
    assert result.metrics["timeout_rate"] == "0.250000"
    assert result.metrics["longest_lapse_streak"] == "2"
    assert result.metrics["first_3m_attempted"] == "4"
    assert result.metrics["last_3m_attempted"] == "4"
    assert result.metrics["first_3m_accuracy"] == "0.500000"
    assert result.metrics["last_3m_accuracy"] == "0.500000"
    assert result.metrics["post_correct_next_item_mean_rt_ms"] == "600.000000"
    assert result.metrics["post_error_next_item_mean_rt_ms"] == "1900.000000"
    assert result.metrics["post_error_next_item_rt_inflation_ms"] == "1300.000000"
    assert result.metrics["post_error_next_item_accuracy_drop"] == "-0.500000"
    assert result.metrics["difficulty_change_count"] == "1"
    assert result.metrics["first_half_attempted"] == "3"
    assert result.metrics["first_half_accuracy"] == "0.333333"
    assert result.metrics["first_half_mean_rt_ms"] == "933.333333"
    assert result.metrics["first_half_timeout_rate"] == "0.333333"
    assert result.metrics["second_half_attempted"] == "1"
    assert result.metrics["second_half_accuracy"] == "1.000000"
    assert result.metrics["second_half_mean_rt_ms"] == "2000.000000"
    assert result.metrics["second_half_timeout_rate"] == "0.000000"
    assert result.metrics["half_accuracy_drop"] == "-0.666667"
    assert result.metrics["half_mean_rt_inflation_ms"] == "1066.666667"


def test_attempt_result_from_engine_emits_realized_difficulty_profile_metrics() -> None:
    engine = _build_static_telemetry_engine()
    engine._resolved_difficulty_context = build_resolved_difficulty_context(
        "vs_target_preview",
        mode="adaptive",
        launch_level=5,
        fixed_level=5,
        adaptive_enabled=True,
    )

    result = attempt_result_from_engine(engine, test_code="vs_target_preview")

    assert result.metrics["difficulty_family_id"] == family_id_for_code("vs_target_preview")
    assert result.metrics["difficulty_profile_level_start"] == "5"
    assert result.metrics["difficulty_profile_level_end"] == "7"
    assert result.metrics["difficulty_profile_level"] == result.metrics["difficulty_profile_level_end"]
    assert result.metrics["difficulty_profile_mode_start"] == "build"
    assert result.metrics["difficulty_profile_mode_end"] == "build"
    assert result.metrics["difficulty_profile_mode"] == result.metrics["difficulty_profile_mode_end"]

    changed_axes = []
    for axis in (
        "content_complexity",
        "time_pressure",
        "distractor_density",
        "multitask_concurrency",
        "memory_span_delay",
        "switch_frequency",
        "control_sensitivity",
        "spatial_ambiguity",
        "source_integration_depth",
    ):
        start_key = f"difficulty_axis_{axis}_start"
        end_key = f"difficulty_axis_{axis}_end"
        assert start_key in result.metrics
        assert end_key in result.metrics
        assert result.metrics[f"difficulty_axis_{axis}"] == result.metrics[end_key]
        if result.metrics[start_key] != result.metrics[end_key]:
            changed_axes.append(axis)
    assert changed_axes


def test_results_store_persists_activity_sessions_telemetry_and_session_rollups(tmp_path) -> None:
    store = ResultsStore(tmp_path / "results.sqlite3")
    result = attempt_result_from_engine(
        _build_static_telemetry_engine(),
        test_code="telemetry_static",
    )

    saved = store.record_attempt(
        result=result,
        app_version="test",
        input_profile_id="default",
    )

    session = store.session_summary()
    assert session is not None
    assert session.attempt_count == 1
    assert session.activity_count == 1
    assert session.completed_activity_count == 1
    assert session.aborted_activity_count == 0
    assert session.unique_tests == 1
    assert session.mean_accuracy == pytest.approx(0.5)
    assert session.mean_score_ratio == pytest.approx(0.5)

    with sqlite3.connect(store.path) as conn:
        attempt_row = conn.execute(
            """
            SELECT activity_session_id, difficulty_level_start, difficulty_level_end
            FROM attempt
            WHERE id=?
            """,
            (saved.attempt_id,),
        ).fetchone()
        metric_rows = dict(
            conn.execute(
                "SELECT key, value FROM attempt_metric WHERE attempt_id=?",
                (saved.attempt_id,),
            ).fetchall()
        )
        activity_row = conn.execute(
            "SELECT completion_reason FROM activity_session WHERE id=?",
            (saved.activity_session_id,),
        ).fetchone()
        activity_metrics = dict(
            conn.execute(
                "SELECT key, value FROM activity_metric WHERE activity_session_id=?",
                (saved.activity_session_id,),
            ).fetchall()
        )
        telemetry_rows = conn.execute(
            """
            SELECT kind, is_timeout, difficulty_level
            FROM telemetry_event
            WHERE activity_session_id=?
            ORDER BY seq
            """,
            (saved.activity_session_id,),
        ).fetchall()
        session_metric_rows = dict(
            conn.execute(
                "SELECT key, value FROM session_metric WHERE session_id=?",
                (saved.session_id,),
            ).fetchall()
        )

    assert attempt_row == (saved.activity_session_id, 5, 7)
    assert activity_row == ("completed",)
    assert metric_rows["timeout_count"] == "1"
    assert metric_rows["longest_lapse_streak"] == "2"
    assert metric_rows["post_error_next_item_rt_inflation_ms"] == "1300.000000"
    assert activity_metrics["difficulty_level_end"] == "7"
    assert [row[0] for row in telemetry_rows] == [
        "activity_started",
        "question",
        "question",
        "question",
        "question",
        "difficulty_change",
        "activity_completed",
    ]
    assert telemetry_rows[2][1] == 1
    assert telemetry_rows[5][2] == 7
    assert session_metric_rows["mean_rt_ms"] == "1200.000000"
    assert session_metric_rows["timeout_rate"] == "0.250000"
    assert session_metric_rows["longest_lapse_streak"] == "2.000000"


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
    assert saved_one.activity_session_id != saved_two.activity_session_id

    session = store.session_summary()
    assert session is not None
    assert session.activity_count == 2
    assert session.completed_activity_count == 2
    assert session.aborted_activity_count == 0
    assert session.attempt_count == 2
    assert session.unique_tests == 2

    numerical_summary = store.test_session_summary("numerical_operations")
    assert numerical_summary is not None
    assert numerical_summary.attempt_count == 1
    assert numerical_summary.latest_accuracy == pytest.approx(numerical.accuracy)

    with sqlite3.connect(store.path) as conn:
        attempt_row = conn.execute(
            "SELECT COUNT(*), COUNT(input_profile_id), COUNT(activity_session_id) FROM attempt"
        ).fetchone()
        activity_row = conn.execute(
            "SELECT COUNT(*), COUNT(CASE WHEN completion_reason = 'completed' THEN 1 END) FROM activity_session"
        ).fetchone()
        telemetry_row = conn.execute(
            "SELECT COUNT(*) FROM telemetry_event"
        ).fetchone()

    assert attempt_row == (2, 2, 2)
    assert activity_row == (2, 2)
    assert telemetry_row == (8,)


def test_difficulty_state_updates_and_resets_code_family_and_primitive_scopes(tmp_path) -> None:
    store = ResultsStore(tmp_path / "results.sqlite3")
    result = attempt_result_from_engine(
        _run_numerical_attempt(seed=444),
        test_code="numerical_operations",
    )

    store.record_attempt(
        result=result,
        app_version="test",
        input_profile_id="default",
    )

    code_state = store.difficulty_state(scope_kind="code", scope_key="numerical_operations")
    family_state = store.difficulty_state(
        scope_kind="family",
        scope_key=family_id_for_code("numerical_operations"),
    )
    primitive_state = store.difficulty_state(
        scope_kind="primitive",
        scope_key="mental_arithmetic_automaticity",
    )

    assert code_state is not None
    assert family_state is not None
    assert primitive_state is not None
    assert code_state.mastery is None
    assert family_state.mastery is None
    assert primitive_state.mastery is not None
    assert primitive_state.speed is not None
    assert primitive_state.confidence is not None
    assert primitive_state.level_confidence is not None
    assert primitive_state.leverage is not None

    store.reset_difficulty_state(test_code="numerical_operations")

    assert store.difficulty_state(scope_kind="code", scope_key="numerical_operations") is None
    assert (
        store.difficulty_state(
            scope_kind="family",
            scope_key=family_id_for_code("numerical_operations"),
        )
        is None
    )
    assert (
        store.difficulty_state(
            scope_kind="primitive",
            scope_key="mental_arithmetic_automaticity",
        )
        is None
    )


def test_ranked_primitive_scope_keys_use_canonical_ids_for_persisted_state(tmp_path) -> None:
    store = ResultsStore(tmp_path / "results.sqlite3")
    result = attempt_result_from_engine(
        _build_static_telemetry_engine(),
        test_code="ant_time_flip",
    )

    store.record_attempt(
        result=result,
        app_version="test",
        input_profile_id="default",
    )

    canonical = store.difficulty_state(
        scope_kind="primitive",
        scope_key="mental_arithmetic_automaticity",
    )
    legacy = store.difficulty_state(
        scope_kind="primitive",
        scope_key="airborne_numerical_applied_math",
    )

    assert canonical is not None
    assert canonical.mastery is not None
    assert legacy is None


def test_replacement_alias_attempts_merge_under_canonical_code_scope(tmp_path) -> None:
    store = ResultsStore(tmp_path / "results.sqlite3")

    legacy = attempt_result_from_engine(
        _build_static_telemetry_engine(),
        test_code="ic_attitude_frame",
    )
    canonical = attempt_result_from_engine(
        _build_static_telemetry_engine(),
        test_code="ic_instrument_attitude_matching",
    )

    store.record_attempt(result=legacy, app_version="test", input_profile_id="default")
    store.record_attempt(result=canonical, app_version="test", input_profile_id="default")

    merged = store.difficulty_state(
        scope_kind="code",
        scope_key="ic_instrument_attitude_matching",
    )
    legacy_state = store.difficulty_state(
        scope_kind="code",
        scope_key="ic_attitude_frame",
    )

    assert merged is not None
    assert merged.sample_count == 2
    assert legacy_state is None


def test_adaptive_session_variants_merge_under_shared_adaptive_code_scope(tmp_path) -> None:
    store = ResultsStore(tmp_path / "results.sqlite3")

    short = attempt_result_from_engine(
        _build_static_telemetry_engine(),
        test_code="adaptive_session_short",
    )
    micro = attempt_result_from_engine(
        _build_static_telemetry_engine(),
        test_code="adaptive_session_micro",
    )

    store.record_attempt(result=short, app_version="test", input_profile_id="default")
    store.record_attempt(result=micro, app_version="test", input_profile_id="default")

    merged = store.difficulty_state(scope_kind="code", scope_key="adaptive_session")
    short_state = store.difficulty_state(scope_kind="code", scope_key="adaptive_session_short")
    micro_state = store.difficulty_state(scope_kind="code", scope_key="adaptive_session_micro")

    assert merged is not None
    assert merged.sample_count == 2
    assert short_state is None
    assert micro_state is None


def test_recent_attempt_history_preserves_block_metrics_for_adaptive_variants(tmp_path) -> None:
    store = ResultsStore(tmp_path / "results.sqlite3")
    engine = _build_static_telemetry_engine()
    engine._result_metrics_overrides = {
        "block.01.primitive_id": "mental_arithmetic_automaticity",
        "block.01.drill_code": "ma_percentage_snap",
        "block.01.target_area": "quantitative_core",
        "block.01.form_factor": "micro",
    }
    result = attempt_result_from_engine(
        engine,
        test_code="adaptive_session_short",
    )

    store.record_attempt(result=result, app_version="test", input_profile_id="default")
    history = store.recent_attempt_history(since_days=28)

    assert len(history) == 1
    assert history[0].test_code == "adaptive_session_short"
    assert history[0].metrics["block.01.primitive_id"] == "mental_arithmetic_automaticity"
    assert history[0].metrics["block.01.drill_code"] == "ma_percentage_snap"
    assert history[0].metrics["block.01.target_area"] == "quantitative_core"
    assert history[0].metrics["block.01.form_factor"] == "micro"


def test_recent_attempt_history_prefers_child_items_and_preserves_parent_origin_metadata(
    tmp_path,
) -> None:
    store = ResultsStore(tmp_path / "results.sqlite3")
    session_id = store.start_app_session(app_version="test")
    parent_engine = _build_static_telemetry_engine()
    parent_engine._result_metrics_overrides = {
        "block.01.primitive_id": "mental_arithmetic_automaticity",
        "block.01.drill_code": "ma_percentage_snap",
        "block.01.target_area": "quantitative_core",
        "block.01.form_factor": "micro",
    }
    parent_activity_session_id = store.start_activity_session(
        activity_code="adaptive_session",
        activity_kind="adaptive_session",
        app_version="test",
        test_version=1,
        engine=parent_engine,
        input_profile_id="default",
    )

    child_saved = store.record_attempt(
        result=attempt_result_from_engine(
            _build_static_telemetry_engine(),
            test_code="ma_percentage_snap",
        ),
        app_version="test",
        input_profile_id="default",
        activity_code="ma_percentage_snap",
        activity_kind="adaptive_item",
        parent_activity_session_id=parent_activity_session_id,
        origin_activity_code="adaptive_session",
        origin_activity_kind="adaptive_session",
        origin_item_index=1,
    )
    store.complete_activity_session(
        activity_session_id=parent_activity_session_id,
        result=attempt_result_from_engine(parent_engine, test_code="adaptive_session"),
        app_version="test",
        completion_reason="completed",
        input_profile_id="default",
    )

    history_all = store.recent_attempt_history(since_days=28)
    history_child_first = store.recent_attempt_history(
        since_days=28,
        child_item_preferred=True,
    )

    with sqlite3.connect(store.path) as conn:
        child_row = conn.execute(
            """
            SELECT session_id, parent_activity_session_id, origin_activity_code, origin_activity_kind, origin_item_index
            FROM activity_session
            WHERE id=?
            """,
            (child_saved.activity_session_id,),
        ).fetchone()

    assert child_row == (
        session_id,
        parent_activity_session_id,
        "adaptive_session",
        "adaptive_session",
        1,
    )
    assert {entry.test_code for entry in history_all} == {"adaptive_session", "ma_percentage_snap"}
    assert len(history_child_first) == 1
    assert history_child_first[0].test_code == "ma_percentage_snap"
    assert history_child_first[0].activity_kind == "adaptive_item"
    assert history_child_first[0].parent_activity_session_id == parent_activity_session_id
    assert history_child_first[0].origin_activity_code == "adaptive_session"
    assert history_child_first[0].origin_activity_kind == "adaptive_session"
    assert history_child_first[0].origin_item_index == 1


def test_results_store_aborts_activity_sessions_without_attempt_rows(tmp_path) -> None:
    store = ResultsStore(tmp_path / "results.sqlite3")
    store.start_app_session(app_version="test")
    result = attempt_result_from_engine(
        _build_static_telemetry_engine(phase=Phase.SCORED),
        test_code="telemetry_static",
    )

    activity_session_id = store.start_activity_session(
        activity_code="telemetry_static",
        activity_kind="cognitive_test",
        app_version="test",
        test_version=1,
        engine=_build_static_telemetry_engine(phase=Phase.SCORED),
        input_profile_id="default",
    )
    store.abort_activity_session(
        activity_session_id=activity_session_id,
        app_version="test",
        completion_reason="back_abort",
        result=result,
        input_profile_id="default",
    )

    session = store.session_summary()
    assert session is not None
    assert session.activity_count == 1
    assert session.completed_activity_count == 0
    assert session.aborted_activity_count == 1
    assert session.attempt_count == 0

    with sqlite3.connect(store.path) as conn:
        attempt_count = conn.execute("SELECT COUNT(*) FROM attempt").fetchone()
        activity_row = conn.execute(
            "SELECT completion_reason FROM activity_session WHERE id=?",
            (activity_session_id,),
        ).fetchone()
        activity_metrics = dict(
            conn.execute(
                "SELECT key, value FROM activity_metric WHERE activity_session_id=?",
                (activity_session_id,),
            ).fetchall()
        )
        telemetry_kinds = [
            row[0]
            for row in conn.execute(
                "SELECT kind FROM telemetry_event WHERE activity_session_id=? ORDER BY seq",
                (activity_session_id,),
            ).fetchall()
        ]

    assert attempt_count == (0,)
    assert activity_row == ("back_abort",)
    assert activity_metrics["attempted"] == "4"
    assert telemetry_kinds[-1] == "activity_aborted"


def test_close_app_session_finalizes_open_activity_sessions_and_materializes_summary(tmp_path) -> None:
    store = ResultsStore(tmp_path / "results.sqlite3")
    session_id = store.start_app_session(app_version="test")
    activity_session_id = store.start_activity_session(
        activity_code="telemetry_static",
        activity_kind="cognitive_test",
        app_version="test",
        test_version=1,
        engine=_build_static_telemetry_engine(phase=Phase.PRACTICE),
        input_profile_id="default",
    )

    store.close_app_session(exit_reason="app_quit")

    summary = load_session_summary(db_path=store.path, session_id=session_id)
    assert summary is not None
    assert summary.exit_reason == "app_quit"
    assert summary.activity_count == 1
    assert summary.aborted_activity_count == 1
    assert summary.attempt_count == 0

    with sqlite3.connect(store.path) as conn:
        session_row = conn.execute(
            "SELECT ended_at_utc, exit_reason FROM session WHERE id=?",
            (session_id,),
        ).fetchone()
        activity_row = conn.execute(
            "SELECT ended_at_utc, completion_reason FROM activity_session WHERE id=?",
            (activity_session_id,),
        ).fetchone()
        session_kinds = [
            row[0]
            for row in conn.execute(
                """
                SELECT kind
                FROM telemetry_event
                WHERE session_id=? AND activity_session_id IS NULL
                ORDER BY seq
                """,
                (session_id,),
            ).fetchall()
        ]

    assert session_row is not None
    assert session_row[0] is not None
    assert session_row[1] == "app_quit"
    assert activity_row is not None
    assert activity_row[0] is not None
    assert activity_row[1] == "app_quit"
    assert session_kinds == ["app_quit"]


def test_results_store_migrates_v1_database_and_continues_writing(tmp_path) -> None:
    path = tmp_path / "legacy-results.sqlite3"
    _create_v1_results_db(path)

    saved = record_attempt(
        db_path=path,
        session_id=1,
        result=attempt_result_from_engine(
            _run_numerical_attempt(seed=101),
            test_code="numerical_operations",
        ),
        app_version="test",
        input_profile_id="default",
    )

    with sqlite3.connect(path) as conn:
        user_version = conn.execute("PRAGMA user_version").fetchone()
        attempt_count = conn.execute("SELECT COUNT(*) FROM attempt").fetchone()
        activity_count = conn.execute("SELECT COUNT(*) FROM activity_session").fetchone()
        legacy_link = conn.execute(
            "SELECT activity_session_id, difficulty_level_start, difficulty_level_end FROM attempt WHERE id=1"
        ).fetchone()
        copied_metric = conn.execute(
            "SELECT value FROM attempt_metric WHERE attempt_id=1 AND key='accuracy'"
        ).fetchone()

    summary = load_session_summary(db_path=path, session_id=1)

    assert user_version == (SCHEMA_VERSION,)
    assert attempt_count == (2,)
    assert activity_count == (2,)
    assert legacy_link is not None
    assert legacy_link[0] is not None
    assert legacy_link[1:] == (6, 6)
    assert copied_metric == ("1.000000",)
    assert summary is not None
    assert summary.attempt_count == 2
    assert summary.activity_count == 2
    assert summary.completed_activity_count == 2
    assert saved.session_id == 1
