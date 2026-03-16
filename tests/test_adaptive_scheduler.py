from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
import pytest

from cfast_trainer.ant_drills import AntDrillMode
from cfast_trainer.adaptive_scheduler import (
    _training_mode_for_role,
    AdaptiveSession,
    AdaptiveSessionBlock,
    AdaptiveSessionPlan,
    AdaptiveStage,
    build_adaptive_session_plan,
    collect_adaptive_evidence,
    rank_adaptive_primitives,
)
from cfast_trainer.app import App, AdaptiveSessionScreen, BenchmarkScreen, MenuItem, MenuScreen
from cfast_trainer.benchmark import BenchmarkSession, build_benchmark_plan
from cfast_trainer.cognitive_core import Phase
from cfast_trainer.cognitive_core import TestSnapshot as SnapshotModel
from cfast_trainer.persistence import AttemptHistoryEntry, ResultsStore
from cfast_trainer.results import attempt_result_from_engine
from cfast_trainer.telemetry import TelemetryEvent


def _iso(hours_ago: float) -> str:
    dt = datetime.now(UTC) - timedelta(hours=float(hours_ago))
    return dt.replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fmt(value: float | int | str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.6f}"


def _base_metrics(
    *,
    score_ratio: float | None = None,
    accuracy: float | None = None,
    first_accuracy: float | None = None,
    last_accuracy: float | None = None,
    first_timeout_rate: float | None = None,
    last_timeout_rate: float | None = None,
    first_half_accuracy: float | None = None,
    second_half_accuracy: float | None = None,
    first_half_timeout_rate: float | None = None,
    second_half_timeout_rate: float | None = None,
    post_error_inflation_ms: float | None = None,
) -> dict[str, str]:
    metrics: dict[str, str] = {}
    if score_ratio is not None:
        metrics["score_ratio"] = _fmt(score_ratio)
    if accuracy is not None:
        metrics["accuracy"] = _fmt(accuracy)
    if first_accuracy is not None:
        metrics["first_3m_accuracy"] = _fmt(first_accuracy)
    if last_accuracy is not None:
        metrics["last_3m_accuracy"] = _fmt(last_accuracy)
    if first_timeout_rate is not None:
        metrics["first_3m_timeout_rate"] = _fmt(first_timeout_rate)
    if last_timeout_rate is not None:
        metrics["last_3m_timeout_rate"] = _fmt(last_timeout_rate)
    if first_half_accuracy is not None:
        metrics["first_half_accuracy"] = _fmt(first_half_accuracy)
    if second_half_accuracy is not None:
        metrics["second_half_accuracy"] = _fmt(second_half_accuracy)
    if first_half_timeout_rate is not None:
        metrics["first_half_timeout_rate"] = _fmt(first_half_timeout_rate)
    if second_half_timeout_rate is not None:
        metrics["second_half_timeout_rate"] = _fmt(second_half_timeout_rate)
    if post_error_inflation_ms is not None:
        metrics["post_error_next_item_rt_inflation_ms"] = _fmt(post_error_inflation_ms)
    return metrics


def _history_entry(
    *,
    test_code: str,
    hours_ago: float,
    metrics: dict[str, str],
    activity_kind: str = "test",
    attempt_id: int = 1,
) -> AttemptHistoryEntry:
    stamp = _iso(hours_ago)
    return AttemptHistoryEntry(
        attempt_id=attempt_id,
        session_id=1,
        activity_session_id=attempt_id,
        activity_code=test_code,
        activity_kind=activity_kind,
        test_code=test_code,
        test_version=1,
        rng_seed=100 + attempt_id,
        difficulty=0.5,
        started_at_utc=stamp,
        completed_at_utc=stamp,
        difficulty_level_start=5,
        difficulty_level_end=5,
        metrics=dict(metrics),
    )


def _rank_map(entries: list[AttemptHistoryEntry]):
    ranked = rank_adaptive_primitives(entries)
    return {item.primitive_id: item for item in ranked}


def test_collect_adaptive_evidence_maps_benchmark_drills_and_coarse_workouts() -> None:
    benchmark = _history_entry(
        test_code="benchmark_battery",
        hours_ago=6,
        metrics={
            "probe.numerical_operations.completed": "1",
            "probe.numerical_operations.score_ratio": "0.400000",
            "probe.numerical_operations.first_3m_accuracy": "0.700000",
            "probe.numerical_operations.last_3m_accuracy": "0.500000",
            "probe.numerical_operations.post_error_next_item_rt_inflation_ms": "900.000000",
        },
        attempt_id=1,
    )
    direct = _history_entry(
        test_code="ma_one_step_fluency",
        hours_ago=4,
        metrics=_base_metrics(score_ratio=0.55),
        attempt_id=2,
    )
    workout = _history_entry(
        test_code="visual_search_workout",
        hours_ago=2,
        metrics=_base_metrics(score_ratio=0.62),
        activity_kind="workout",
        attempt_id=3,
    )

    evidence = collect_adaptive_evidence([benchmark, direct, workout])

    assert {(item.primitive_id, item.source_kind) for item in evidence} == {
        ("mental_arithmetic_automaticity", "benchmark_probe"),
        ("mental_arithmetic_automaticity", "direct"),
        ("visual_scan_discipline", "coarse_workout"),
    }
    assert any(item.source_code == "numerical_operations" for item in evidence)
    assert any(item.coarse for item in evidence if item.primitive_id == "visual_scan_discipline")


def test_collect_adaptive_evidence_maps_new_symbolic_and_dual_task_drills() -> None:
    entries = [
        _history_entry(
            test_code="sl_one_rule_identify",
            hours_ago=5,
            metrics=_base_metrics(score_ratio=0.61),
            attempt_id=10,
        ),
        _history_entry(
            test_code="dtb_tracking_command_filter",
            hours_ago=4,
            metrics=_base_metrics(score_ratio=0.58, post_error_inflation_ms=900.0),
            attempt_id=11,
        ),
    ]

    evidence = collect_adaptive_evidence(entries)

    assert {(item.primitive_id, item.source_code) for item in evidence} == {
        ("symbolic_rule_extraction", "sl_one_rule_identify"),
        ("dual_task_stability_fatigue", "dtb_tracking_command_filter"),
    }


def test_rank_adaptive_primitives_prioritizes_weaker_performance() -> None:
    entries = [
        _history_entry(
            test_code="numerical_operations",
            hours_ago=4,
            metrics=_base_metrics(score_ratio=0.35),
            attempt_id=1,
        ),
        _history_entry(
            test_code="table_reading",
            hours_ago=4,
            metrics=_base_metrics(score_ratio=0.80),
            attempt_id=2,
        ),
    ]

    ranked = rank_adaptive_primitives(entries)

    assert ranked[0].primitive_id == "mental_arithmetic_automaticity"
    assert ranked[0].weakness > ranked[1].weakness


def test_rank_adaptive_primitives_raises_fatigue_sensitive_domain() -> None:
    entries = [
        _history_entry(
            test_code="table_reading",
            hours_ago=3,
            metrics=_base_metrics(
                score_ratio=0.75,
                first_accuracy=0.92,
                last_accuracy=0.48,
                first_timeout_rate=0.00,
                last_timeout_rate=0.25,
            ),
            attempt_id=1,
        ),
        _history_entry(
            test_code="numerical_operations",
            hours_ago=3,
            metrics=_base_metrics(score_ratio=0.74, first_accuracy=0.78, last_accuracy=0.74),
            attempt_id=2,
        ),
    ]

    ranked = _rank_map(entries)

    assert ranked["table_cross_reference_speed"].fatigue > ranked["mental_arithmetic_automaticity"].fatigue
    assert ranked["table_cross_reference_speed"].priority > ranked["mental_arithmetic_automaticity"].priority


def test_rank_adaptive_primitives_prefers_split_half_fatigue_metrics_when_present() -> None:
    entries = [
        _history_entry(
            test_code="table_reading",
            hours_ago=3,
            metrics=_base_metrics(
                score_ratio=0.75,
                first_accuracy=0.70,
                last_accuracy=0.70,
                first_half_accuracy=0.95,
                second_half_accuracy=0.35,
                first_half_timeout_rate=0.00,
                second_half_timeout_rate=0.25,
            ),
            attempt_id=1,
        ),
        _history_entry(
            test_code="numerical_operations",
            hours_ago=3,
            metrics=_base_metrics(
                score_ratio=0.74,
                first_accuracy=0.86,
                last_accuracy=0.70,
            ),
            attempt_id=2,
        ),
    ]

    ranked = _rank_map(entries)

    assert ranked["table_cross_reference_speed"].fatigue > ranked["mental_arithmetic_automaticity"].fatigue
    assert ranked["table_cross_reference_speed"].priority > ranked["mental_arithmetic_automaticity"].priority


def test_rank_adaptive_primitives_raises_post_error_slowdown() -> None:
    entries = [
        _history_entry(
            test_code="cln_full_steady",
            hours_ago=5,
            metrics=_base_metrics(score_ratio=0.72, post_error_inflation_ms=1300.0),
            attempt_id=1,
        ),
        _history_entry(
            test_code="rt_ground_tempo_run",
            hours_ago=5,
            metrics=_base_metrics(score_ratio=0.72, post_error_inflation_ms=150.0),
            attempt_id=2,
        ),
    ]

    ranked = _rank_map(entries)

    assert ranked["dual_task_stability_fatigue"].post_error > ranked["tracking_stability_low_load"].post_error
    assert ranked["dual_task_stability_fatigue"].priority > ranked["tracking_stability_low_load"].priority


def test_rank_adaptive_primitives_scores_retention_decay_at_48_to_72_hours() -> None:
    entries = [
        _history_entry(
            test_code="numerical_operations",
            hours_ago=72,
            metrics=_base_metrics(score_ratio=0.92),
            attempt_id=1,
        ),
        _history_entry(
            test_code="numerical_operations",
            hours_ago=12,
            metrics=_base_metrics(score_ratio=0.48),
            attempt_id=2,
        ),
        _history_entry(
            test_code="table_reading",
            hours_ago=12,
            metrics=_base_metrics(score_ratio=0.78),
            attempt_id=3,
        ),
    ]

    ranked = _rank_map(entries)

    assert ranked["mental_arithmetic_automaticity"].retention > 0.40
    assert ranked["mental_arithmetic_automaticity"].retention > ranked["table_cross_reference_speed"].retention


def test_rank_adaptive_primitives_uses_bottleneck_and_profile_weight_to_break_ties() -> None:
    entries = [
        _history_entry(
            test_code="numerical_operations",
            hours_ago=3,
            metrics=_base_metrics(score_ratio=0.65),
            attempt_id=1,
        ),
        _history_entry(
            test_code="rt_lock_anchor",
            hours_ago=3,
            metrics=_base_metrics(score_ratio=0.65),
            attempt_id=2,
        ),
    ]

    ranked = _rank_map(entries)

    assert ranked["mental_arithmetic_automaticity"].priority > ranked["tracking_stability_low_load"].priority


def test_build_adaptive_session_plan_defaults_to_six_blocks_and_collapses_to_four_plus_two() -> None:
    entries = [
        _history_entry(
            test_code="numerical_operations",
            hours_ago=80,
            metrics=_base_metrics(
                score_ratio=0.38,
                first_accuracy=0.85,
                last_accuracy=0.45,
                post_error_inflation_ms=1200.0,
            ),
            attempt_id=1,
        ),
        _history_entry(
            test_code="table_reading",
            hours_ago=16,
            metrics=_base_metrics(score_ratio=0.52, post_error_inflation_ms=900.0),
            attempt_id=2,
        ),
    ]

    plan = build_adaptive_session_plan(history=entries, seed=999)

    assert plan is not None
    assert len(plan.blocks) == 6
    assert sum(1 for block in plan.blocks if block.primitive_id == "mental_arithmetic_automaticity") == 4
    assert sum(1 for block in plan.blocks if block.primitive_id == "table_cross_reference_speed") == 2
    assert len({block.primitive_id for block in plan.blocks}) == 2
    assert plan.blocks[0].mode == "anchor"
    assert plan.blocks[0].drill_code == "ma_one_step_fluency"
    assert all(
        block.drill_code.startswith("ma_")
        for block in plan.blocks
        if block.primitive_id == "mental_arithmetic_automaticity"
    )
    assert all(
        block.drill_code.startswith("tbl_")
        for block in plan.blocks
        if block.primitive_id == "table_cross_reference_speed"
    )
    assert any(block.mode == "reset" for block in plan.blocks if block.primitive_id == "mental_arithmetic_automaticity")
    assert any(
        block.mode == "fatigue_pressure"
        for block in plan.blocks
        if block.primitive_id == "mental_arithmetic_automaticity"
    )


def test_build_adaptive_session_plan_uses_three_two_one_when_third_priority_is_real() -> None:
    entries = [
        _history_entry(test_code="numerical_operations", hours_ago=70, metrics=_base_metrics(score_ratio=0.40), attempt_id=1),
        _history_entry(
            test_code="table_reading",
            hours_ago=60,
            metrics=_base_metrics(
                score_ratio=0.47,
                first_accuracy=0.90,
                last_accuracy=0.60,
                post_error_inflation_ms=700.0,
            ),
            attempt_id=2,
        ),
        _history_entry(
            test_code="visual_search",
            hours_ago=60,
            metrics=_base_metrics(score_ratio=0.42, first_accuracy=0.88, last_accuracy=0.52),
            attempt_id=3,
        ),
    ]

    plan = build_adaptive_session_plan(history=entries, seed=321)

    assert plan is not None
    counts = {primitive_id: 0 for primitive_id in {block.primitive_id for block in plan.blocks}}
    for block in plan.blocks:
        counts[block.primitive_id] += 1
    assert sorted(counts.values(), reverse=True) == [3, 2, 1]
    assert any(
        block.drill_code.startswith("ma_")
        for block in plan.blocks
        if block.primitive_id == "mental_arithmetic_automaticity"
    )
    assert any(
        block.drill_code.startswith("tbl_")
        for block in plan.blocks
        if block.primitive_id == "table_cross_reference_speed"
    )


def test_build_adaptive_session_plan_returns_none_on_cold_start() -> None:
    assert build_adaptive_session_plan(history=[], seed=123) is None


def test_training_mode_for_role_prefers_fresh_recovery_and_fatigue_probe() -> None:
    assert _training_mode_for_role("ma_one_step_fluency", role="anchor") is AntDrillMode.FRESH
    assert _training_mode_for_role("vs_mixed_tempo", role="anchor") is AntDrillMode.BUILD
    assert _training_mode_for_role("ma_one_step_fluency", role="tempo") is AntDrillMode.PRESSURE
    assert _training_mode_for_role("ma_one_step_fluency", role="reset") is AntDrillMode.RECOVERY
    assert _training_mode_for_role("ma_one_step_fluency", role="fatigue_pressure") is AntDrillMode.FATIGUE_PROBE


@dataclass
class _FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return float(self.t)

    def advance(self, dt: float) -> None:
        self.t += float(dt)


@dataclass(frozen=True, slots=True)
class _FakeSummary:
    attempted: int
    correct: int
    accuracy: float
    duration_s: float
    throughput_per_min: float
    mean_response_time_s: float | None
    total_score: float
    max_score: float
    score_ratio: float
    difficulty_level: int
    difficulty_level_start: int
    difficulty_level_end: int
    difficulty_change_count: int = 0


class _FakeBlockEngine:
    def __init__(
        self,
        *,
        clock: _FakeClock,
        title: str,
        seed: int,
        difficulty_level: int,
        scored_duration_s: float,
        attempted: int,
        correct: int,
    ) -> None:
        self._clock = clock
        self._title = title
        self.seed = int(seed)
        self.difficulty = float(difficulty_level - 1) / 9.0
        self.practice_questions = 0
        self.scored_duration_s = float(scored_duration_s)
        self.phase = Phase.INSTRUCTIONS
        self._started_at_s: float | None = None
        self._attempted = int(attempted)
        self._correct = int(correct)
        self._difficulty_level = int(difficulty_level)
        self._events = self._build_events()

    def _build_events(self) -> list[TelemetryEvent]:
        events: list[TelemetryEvent] = []
        for index in range(self._attempted):
            is_correct = index < self._correct
            events.append(
                TelemetryEvent(
                    family="question",
                    kind="question",
                    phase=Phase.SCORED.value,
                    seq=index,
                    item_index=index + 1,
                    is_scored=True,
                    is_correct=is_correct,
                    is_timeout=False,
                    response_time_ms=600 + (index * 100),
                    score=1.0 if is_correct else 0.0,
                    max_score=1.0,
                    difficulty_level=self._difficulty_level,
                    occurred_at_ms=(index + 1) * 1000,
                    prompt=f"Q{index + 1}",
                    expected=str(index + 1),
                    response=str(index + 1 if is_correct else 0),
                )
            )
        return events

    def start_scored(self) -> None:
        self.phase = Phase.SCORED
        self._started_at_s = self._clock.now()

    def update(self) -> None:
        if self.phase is not Phase.SCORED or self._started_at_s is None:
            return
        if (self._clock.now() - self._started_at_s) >= self.scored_duration_s:
            self.phase = Phase.RESULTS

    def submit_answer(self, raw: str) -> bool:
        _ = raw
        return False

    def snapshot(self) -> SnapshotModel:
        remaining = None
        if self.phase is Phase.SCORED and self._started_at_s is not None:
            remaining = max(0.0, self.scored_duration_s - (self._clock.now() - self._started_at_s))
        attempted = self._attempted if self.phase is Phase.RESULTS else 0
        correct = self._correct if self.phase is Phase.RESULTS else 0
        return SnapshotModel(
            title=self._title,
            phase=self.phase,
            prompt=self._title,
            input_hint="",
            time_remaining_s=remaining,
            attempted_scored=attempted,
            correct_scored=correct,
            payload=None,
        )

    def scored_summary(self) -> _FakeSummary:
        accuracy = 0.0 if self._attempted <= 0 else float(self._correct) / float(self._attempted)
        throughput = 0.0 if self.scored_duration_s <= 0.0 else (self._attempted / self.scored_duration_s) * 60.0
        return _FakeSummary(
            attempted=self._attempted,
            correct=self._correct,
            accuracy=accuracy,
            duration_s=self.scored_duration_s,
            throughput_per_min=throughput,
            mean_response_time_s=0.7,
            total_score=float(self._correct),
            max_score=float(self._attempted),
            score_ratio=accuracy,
            difficulty_level=self._difficulty_level,
            difficulty_level_start=self._difficulty_level,
            difficulty_level_end=self._difficulty_level,
        )

    def events(self) -> list[TelemetryEvent]:
        return list(self._events)


def _small_adaptive_plan(clock: _FakeClock) -> AdaptiveSessionPlan:
    def _builder(
        *,
        title: str,
        seed: int,
        difficulty_level: int,
        attempted: int,
        correct: int,
    ):
        return lambda: _FakeBlockEngine(
            clock=clock,
            title=title,
            seed=seed,
            difficulty_level=difficulty_level,
            scored_duration_s=1.0,
            attempted=attempted,
            correct=correct,
        )

    return AdaptiveSessionPlan(
        code="adaptive_session",
        title="Adaptive Test Plan",
        version=1,
        generated_at_utc=_iso(0),
        description="Small adaptive test plan.",
        notes=("Test note",),
        ranked_primitives=rank_adaptive_primitives(
            [
                _history_entry(test_code="numerical_operations", hours_ago=72, metrics=_base_metrics(score_ratio=0.45), attempt_id=1),
                _history_entry(test_code="table_reading", hours_ago=24, metrics=_base_metrics(score_ratio=0.60), attempt_id=2),
            ]
        ),
        blocks=(
            AdaptiveSessionBlock(
                block_index=0,
                primitive_id="mental_arithmetic_automaticity",
                primitive_label="Mental Arithmetic Automaticity",
                drill_code="no_fact_prime",
                mode="anchor",
                duration_s=1.0,
                difficulty_level=4,
                seed=111,
                reason_tags=("weak", "retention"),
                priority=0.8,
                drill_mode=AntDrillMode.BUILD,
                builder=_builder(
                    title="Arithmetic Anchor",
                    seed=111,
                    difficulty_level=4,
                    attempted=2,
                    correct=1,
                ),
            ),
            AdaptiveSessionBlock(
                block_index=1,
                primitive_id="table_cross_reference_speed",
                primitive_label="Table Cross-Reference Speed",
                drill_code="tbl_part2_prime",
                mode="tempo",
                duration_s=1.0,
                difficulty_level=5,
                seed=222,
                reason_tags=("weak",),
                priority=0.6,
                drill_mode=AntDrillMode.TEMPO,
                builder=_builder(
                    title="Table Tempo",
                    seed=222,
                    difficulty_level=5,
                    attempted=2,
                    correct=2,
                ),
            ),
        ),
    )


def test_adaptive_session_persists_one_attempt_with_block_metrics_and_telemetry(tmp_path) -> None:
    store = ResultsStore(tmp_path / "results.sqlite3")
    store.start_app_session(app_version="test")
    clock = _FakeClock()
    session = AdaptiveSession(clock=clock, seed=444, plan=_small_adaptive_plan(clock))
    activity_session_id = store.start_activity_session(
        activity_code="adaptive_session",
        activity_kind="adaptive_session",
        app_version="test",
        test_version=1,
        engine=session,
    )

    session.activate()
    while session.stage is AdaptiveStage.BLOCK:
        clock.advance(1.2)
        session.update()

    saved = store.complete_activity_session(
        activity_session_id=activity_session_id,
        result=attempt_result_from_engine(session, test_code="adaptive_session", test_version=1),
        app_version="test",
        completion_reason="completed",
    )

    assert saved is not None
    with sqlite3.connect(store.path) as conn:
        attempt_count = conn.execute("SELECT COUNT(*) FROM attempt").fetchone()
        activity_count = conn.execute("SELECT COUNT(*) FROM activity_session").fetchone()
        metric_rows = dict(
            conn.execute(
                "SELECT key, value FROM attempt_metric WHERE attempt_id=?",
                (saved.attempt_id,),
            ).fetchall()
        )
        telemetry_kinds = [
            row[0]
            for row in conn.execute(
                "SELECT kind FROM telemetry_event WHERE activity_session_id=? ORDER BY seq",
                (saved.activity_session_id,),
            ).fetchall()
        ]

    assert attempt_count == (1,)
    assert activity_count == (1,)
    assert metric_rows["scheduler.version"] == "1"
    assert metric_rows["block.01.primitive_id"] == "mental_arithmetic_automaticity"
    assert metric_rows["block.02.drill_code"] == "tbl_part2_prime"
    assert metric_rows["block.01.attempted"] == "2"
    assert metric_rows["block.02.score_ratio"] == "1.000000"
    assert telemetry_kinds == [
        "activity_started",
        "block_started",
        "question",
        "question",
        "block_completed",
        "block_started",
        "question",
        "question",
        "block_completed",
        "activity_completed",
    ]
    assert metric_rows["block.01.training_mode"] == AntDrillMode.BUILD.value
    assert metric_rows["block.02.training_mode"] == AntDrillMode.TEMPO.value


def test_adaptive_session_results_snapshot_includes_split_half_lines() -> None:
    clock = _FakeClock()
    session = AdaptiveSession(clock=clock, seed=444, plan=_small_adaptive_plan(clock))

    session.activate()
    while session.stage is AdaptiveStage.BLOCK:
        clock.advance(1.2)
        session.update()

    snapshot = session.snapshot()

    assert snapshot.stage is AdaptiveStage.RESULTS
    assert any("Block splits:" == line for line in snapshot.note_lines)
    assert any("1H " in line and "2H " in line for line in snapshot.note_lines)


def test_results_store_recent_attempt_history_returns_metrics_and_activity_kind(tmp_path) -> None:
    store = ResultsStore(tmp_path / "history.sqlite3")
    store.start_app_session(app_version="test")
    clock = _FakeClock()
    session = AdaptiveSession(clock=clock, seed=555, plan=_small_adaptive_plan(clock))
    activity_session_id = store.start_activity_session(
        activity_code="adaptive_session",
        activity_kind="adaptive_session",
        app_version="test",
        test_version=1,
        engine=session,
    )
    session.activate()
    while session.stage is AdaptiveStage.BLOCK:
        clock.advance(1.2)
        session.update()
    saved = store.complete_activity_session(
        activity_session_id=activity_session_id,
        result=attempt_result_from_engine(session, test_code="adaptive_session", test_version=1),
        app_version="test",
        completion_reason="completed",
    )

    history = store.recent_attempt_history(since_days=28)

    assert saved is not None
    assert len(history) == 1
    assert history[0].activity_kind == "adaptive_session"
    assert history[0].test_code == "adaptive_session"
    assert history[0].metrics["block.01.primitive_id"] == "mental_arithmetic_automaticity"


def test_adaptive_session_bootstrap_enter_routes_to_benchmark(tmp_path) -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        store = ResultsStore(tmp_path / "bootstrap.sqlite3")
        app = App(surface=surface, font=font, results_store=store, app_version="test")
        app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))

        screen = AdaptiveSessionScreen(
            app,
            session=None,
            benchmark_screen_factory=lambda: BenchmarkScreen(
                app,
                session=BenchmarkSession(plan=build_benchmark_plan(clock=_FakeClock())),
            ),
        )
        app.push(screen)

        screen.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))

        assert isinstance(app._screens[-1], BenchmarkScreen)
    finally:
        pygame.quit()


def test_ui_smoke_open_adaptive_session_from_main_menu_bootstraps_to_benchmark(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    monkeypatch.setenv("CFAST_RESULTS_DB_PATH", str(tmp_path / "smoke.sqlite3"))

    from cfast_trainer.app import run

    def inject(frame: int) -> None:
        if frame == 1:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        elif frame == 2:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        elif frame == 3:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        elif frame == 4:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))
        elif frame == 8:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))

    assert run(max_frames=30, event_injector=inject) == 0
