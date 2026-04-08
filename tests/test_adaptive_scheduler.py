from __future__ import annotations

import os
import sqlite3
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from importlib.machinery import ModuleSpec
from types import ModuleType

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
import pytest

if "moderngl" not in sys.modules:
    moderngl_stub = ModuleType("moderngl")
    moderngl_stub.__spec__ = ModuleSpec("moderngl", loader=None)
    sys.modules["moderngl"] = moderngl_stub

import cfast_trainer.adaptive_scheduler as adaptive_scheduler
from cfast_trainer.adaptive_scheduler import (
    AdaptiveSession,
    AdaptiveSessionBlock,
    AdaptiveSessionPlan,
    AdaptiveStage,
    _candidate,
    _training_mode_for_role,
    build_adaptive_session_plan,
    collect_adaptive_evidence,
    rank_adaptive_primitives,
)
from cfast_trainer.ant_drills import AntDrillMode
from cfast_trainer.app import (
    INTRO_LOADING_MIN_FRAMES,
    AdaptiveSessionScreen,
    App,
    BenchmarkScreen,
    MenuItem,
    MenuScreen,
)
from cfast_trainer.benchmark import BenchmarkSession, build_benchmark_plan
from cfast_trainer.cognitive_core import Phase
from cfast_trainer.cognitive_core import TestSnapshot as SnapshotModel
from cfast_trainer.persistence import AttemptHistoryEntry, ResultsStore
from cfast_trainer.primitive_ranking import rank_primitives
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
    duration_s: float | None = None,
    training_mode: str | None = None,
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
    if duration_s is not None:
        metrics["duration_s"] = _fmt(duration_s)
    if training_mode is not None:
        metrics["training_mode"] = str(training_mode)
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
    difficulty_level_start: int = 5,
    difficulty_level_end: int = 5,
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
        difficulty_level_start=difficulty_level_start,
        difficulty_level_end=difficulty_level_end,
        metrics=dict(metrics),
    )


def _benchmark_history_entry(
    *,
    hours_ago: float,
    probe_metrics: dict[str, dict[str, str | float | int]],
    attempt_id: int,
) -> AttemptHistoryEntry:
    metrics: dict[str, str] = {}
    for probe_code, values in probe_metrics.items():
        prefix = f"probe.{probe_code}."
        merged: dict[str, str | float | int] = {"completed": "1"}
        merged.update(values)
        for key, value in merged.items():
            metrics[f"{prefix}{key}"] = _fmt(value)
    return _history_entry(
        test_code="benchmark_battery",
        hours_ago=hours_ago,
        metrics=metrics,
        attempt_id=attempt_id,
    )


def _adaptive_history_entry(
    *,
    hours_ago: float,
    blocks: tuple[tuple[str, str, str], ...],
    attempt_id: int,
    test_code: str = "adaptive_session",
) -> AttemptHistoryEntry:
    metrics: dict[str, str] = {}
    for index, (primitive_id, drill_code, target_area) in enumerate(blocks, start=1):
        prefix = f"block.{index:02d}."
        metrics[f"{prefix}primitive_id"] = primitive_id
        metrics[f"{prefix}drill_code"] = drill_code
        metrics[f"{prefix}target_area"] = target_area
    return _history_entry(
        test_code=test_code,
        hours_ago=hours_ago,
        metrics=metrics,
        activity_kind="adaptive_session",
        attempt_id=attempt_id,
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


def test_scheduler_candidate_prefers_registry_backed_catalog_metadata() -> None:
    candidate = _candidate(
        "vs_multi_target_class_search",
        "wrong_primitive",
        "wrong_target_area",
        "short",
        ("target_anchor",),
    )

    assert candidate.primitive_id == "visual_scan_discipline"
    assert candidate.target_area == "class_search"
    assert candidate.form_factor == "micro"


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


def test_collect_adaptive_evidence_maps_supported_official_benchmark_probes() -> None:
    benchmark = _benchmark_history_entry(
        hours_ago=3,
        attempt_id=12,
        probe_metrics={
            "system_logic": _base_metrics(score_ratio=0.41),
            "rapid_tracking": {**_base_metrics(score_ratio=0.39), "overshoot_count": 5},
            "colours_letters_numbers": {
                **_base_metrics(score_ratio=0.37, post_error_inflation_ms=850.0),
                "intrusion_count": 3,
            },
            "instrument_comprehension": _base_metrics(score_ratio=0.22),
        },
    )

    evidence = collect_adaptive_evidence([benchmark])
    observed = {(item.primitive_id, item.source_code, item.source_kind) for item in evidence}

    assert ("symbolic_rule_extraction", "system_logic", "benchmark_probe") in observed
    assert ("tracking_stability_low_load", "rapid_tracking", "benchmark_probe") in observed
    assert ("dual_task_stability_fatigue", "colours_letters_numbers", "benchmark_probe") in observed
    assert all(item.source_code != "instrument_comprehension" for item in evidence)


def test_collect_adaptive_evidence_uses_five_most_recent_results_per_primitive() -> None:
    entries = [
        _history_entry(
            test_code="ma_one_step_fluency",
            hours_ago=hours_ago,
            metrics=_base_metrics(score_ratio=0.40 + (index * 0.05)),
            activity_kind="drill",
            attempt_id=index + 1,
        )
        for index, hours_ago in enumerate((1, 2, 3, 4, 5, 6))
    ]

    evidence = collect_adaptive_evidence(entries)

    assert len([item for item in evidence if item.primitive_id == "mental_arithmetic_automaticity"]) == 5


def test_collect_adaptive_evidence_weights_tests_difficulty_and_harder_modes_more_heavily() -> None:
    entries = [
        _history_entry(
            test_code="vs_multi_target_class_search",
            hours_ago=3,
            metrics=_base_metrics(score_ratio=0.72, duration_s=120.0, training_mode="build"),
            activity_kind="drill",
            attempt_id=21,
            difficulty_level_start=3,
            difficulty_level_end=3,
        ),
        _history_entry(
            test_code="vs_multi_target_class_search",
            hours_ago=2,
            metrics=_base_metrics(score_ratio=0.72, duration_s=120.0, training_mode="pressure"),
            activity_kind="drill",
            attempt_id=22,
            difficulty_level_start=8,
            difficulty_level_end=8,
        ),
        _history_entry(
            test_code="visual_search",
            hours_ago=1,
            metrics=_base_metrics(score_ratio=0.72, duration_s=480.0),
            activity_kind="test",
            attempt_id=23,
            difficulty_level_start=8,
            difficulty_level_end=8,
        ),
    ]

    evidence = collect_adaptive_evidence(entries)
    weights = {
        (item.source_kind, item.training_mode, item.difficulty_level): item.evidence_weight
        for item in evidence
    }

    assert weights[("integrated_test", "", 8)] > weights[("direct", "build", 3)]
    assert weights[("direct", "pressure", 8)] > weights[("direct", "build", 3)]


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


def test_rank_adaptive_primitives_keeps_confidence_lower_with_sparse_history() -> None:
    sparse = [
        _history_entry(
            test_code="ma_one_step_fluency",
            hours_ago=4,
            metrics=_base_metrics(score_ratio=0.62, duration_s=120.0),
            activity_kind="drill",
            attempt_id=31,
        ),
    ]
    dense = [
        _history_entry(
            test_code="ma_one_step_fluency",
            hours_ago=4 + index,
            metrics=_base_metrics(score_ratio=0.62 + (index * 0.02), duration_s=120.0),
            activity_kind="drill",
            attempt_id=40 + index,
        )
        for index in range(5)
    ]

    sparse_ranked = _rank_map(sparse)
    dense_ranked = _rank_map(dense)

    assert (
        sparse_ranked["mental_arithmetic_automaticity"].confidence
        < dense_ranked["mental_arithmetic_automaticity"].confidence
    )


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


def test_build_adaptive_session_plan_is_deterministic_for_same_history_seed_and_variant() -> None:
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

    first = build_adaptive_session_plan(history=entries, seed=999, variant="full")
    second = build_adaptive_session_plan(history=entries, seed=999, variant="full")

    assert first is not None
    assert second is not None
    assert [
        (
            block.mode,
            block.primitive_id,
            block.drill_code,
            block.difficulty_level,
            block.drill_mode.value,
            block.form_factor,
            block.target_area,
            block.linked_primitive_id,
            block.comparable_level,
        )
        for block in first.blocks
    ] == [
        (
            block.mode,
            block.primitive_id,
            block.drill_code,
            block.difficulty_level,
            block.drill_mode.value,
            block.form_factor,
            block.target_area,
            block.linked_primitive_id,
            block.comparable_level,
        )
        for block in second.blocks
    ]

def test_build_adaptive_session_plan_returns_single_live_block_and_top5_pool() -> None:
    entries = [
        _history_entry(
            test_code="numerical_operations",
            hours_ago=60,
            metrics=_base_metrics(
                score_ratio=0.40,
                first_half_accuracy=0.95,
                second_half_accuracy=0.42,
                first_half_timeout_rate=0.00,
                second_half_timeout_rate=0.18,
            ),
            attempt_id=1,
        ),
    ]

    plan = build_adaptive_session_plan(history=entries, seed=321)

    assert plan is not None
    assert plan.code == "adaptive_session"
    assert len(plan.blocks) == 1
    assert plan.blocks[0].mode == "adaptive_live"
    assert plan.blocks[0].duration_s == pytest.approx(150.0)
    assert plan.last_selected_primitive_id == "mental_arithmetic_automaticity"
    eligible = [item for item in plan.ranked_primitives if item.eligible]
    assert len(eligible) == 5
    assert sum(item.selected_weight for item in eligible) == pytest.approx(1.0)
    assert plan.blocks[0].primitive_id != plan.last_selected_primitive_id
    assert plan.blocks[0].primitive_id in {item.primitive_id for item in eligible}


def test_build_adaptive_session_plan_uses_unmeasured_pool_on_cold_start() -> None:
    plan = build_adaptive_session_plan(history=[], seed=123)

    assert plan is not None
    assert plan.code == "adaptive_session"
    assert len(plan.blocks) == 1
    assert plan.blocks[0].duration_s == pytest.approx(150.0)
    assert plan.blocks[0].form_factor in {"micro", "short"}
    selected = next(
        item for item in plan.ranked_primitives if item.primitive_id == plan.blocks[0].primitive_id
    )
    assert selected.unmeasured is True
    assert selected.eligible is True
    assert "latest 5 completed attempts per drill" in plan.notes[1]


def test_build_adaptive_session_plan_can_target_tracking_from_official_benchmark_history() -> None:
    benchmark = _benchmark_history_entry(
        hours_ago=6,
        attempt_id=60,
        probe_metrics={
            "numerical_operations": _base_metrics(score_ratio=0.88),
            "visual_search": _base_metrics(score_ratio=0.84),
            "table_reading": _base_metrics(score_ratio=0.82),
            "system_logic": _base_metrics(score_ratio=0.80),
            "colours_letters_numbers": _base_metrics(score_ratio=0.79),
            "rapid_tracking": {
                **_base_metrics(score_ratio=0.18, accuracy=0.24),
                "overshoot_count": 7,
                "rt_variance_ms2": "810000.000000",
            },
        },
    )

    plan = build_adaptive_session_plan(history=[benchmark], seed=808, variant="full")

    assert plan is not None
    assert plan.blocks[0].primitive_id == "tracking_stability_low_load"
    assert any(block.drill_code.startswith(("rt_", "sma_")) for block in plan.blocks)


def test_build_adaptive_session_plan_does_not_repeat_exact_same_drill_when_alternative_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mental_only_catalog = tuple(
        candidate
        for candidate in adaptive_scheduler.ADAPTIVE_DRILL_CATALOG
        if candidate.primitive_id == "mental_arithmetic_automaticity"
    )
    monkeypatch.setattr(adaptive_scheduler, "ADAPTIVE_DRILL_CATALOG", mental_only_catalog)

    entries = [
        _history_entry(
            test_code="numerical_operations",
            hours_ago=12,
            metrics=_base_metrics(score_ratio=0.42),
            attempt_id=11,
        ),
        _adaptive_history_entry(
            hours_ago=2,
            blocks=(("mental_arithmetic_automaticity", "ma_percentage_snap", "quantitative_core"),),
            attempt_id=12,
        ),
    ]

    plan = build_adaptive_session_plan(history=entries, seed=444, variant="full")

    assert plan is not None
    assert len(plan.blocks) == 1
    assert plan.blocks[0].primitive_id == "mental_arithmetic_automaticity"
    assert plan.blocks[0].drill_code != "ma_percentage_snap"


def test_build_adaptive_session_plan_marks_only_top5_primitives_as_eligible() -> None:
    entries = [
        _history_entry(
            test_code="visual_search",
            hours_ago=60,
            metrics={
                **_base_metrics(
                    score_ratio=0.46,
                    first_half_accuracy=0.96,
                    second_half_accuracy=0.40,
                    first_half_timeout_rate=0.00,
                    second_half_timeout_rate=0.22,
                ),
                "longest_lapse_streak": "4",
            },
            attempt_id=12,
        ),
    ]

    plan = build_adaptive_session_plan(history=entries, seed=446, variant="full")

    assert plan is not None
    eligible = [item for item in plan.ranked_primitives if item.eligible]
    ineligible = [item for item in plan.ranked_primitives if not item.eligible]
    assert len(eligible) == 5
    assert all(item.selected_weight > 0.0 for item in eligible)
    assert all(item.selected_weight == 0.0 for item in ineligible)
    assert eligible[0].priority >= eligible[-1].priority
    assert plan.blocks[0].primitive_id in {item.primitive_id for item in eligible}


def test_rank_adaptive_primitives_increases_priority_for_high_fatigue_history() -> None:
    low_fatigue_entries = [
        _history_entry(
            test_code="numerical_operations",
            hours_ago=5 + index,
            metrics=_base_metrics(score_ratio=0.30, duration_s=120.0),
            activity_kind="test",
            attempt_id=80 + index,
            difficulty_level_start=6,
            difficulty_level_end=6,
        )
        for index in range(5)
    ]
    high_fatigue_entries = [
        _history_entry(
                test_code="numerical_operations",
                hours_ago=5 + index,
                metrics=_base_metrics(
                    score_ratio=0.30,
                duration_s=900.0,
                first_half_accuracy=0.90,
                second_half_accuracy=0.52,
                first_half_timeout_rate=0.00,
                second_half_timeout_rate=0.20,
                post_error_inflation_ms=1100.0,
            ),
            activity_kind="test",
            attempt_id=90 + index,
            difficulty_level_start=6,
            difficulty_level_end=6,
        )
        for index in range(5)
    ]

    low_ranked = _rank_map(low_fatigue_entries)
    high_ranked = _rank_map(high_fatigue_entries)

    assert (
        high_ranked["mental_arithmetic_automaticity"].fatigue
        > low_ranked["mental_arithmetic_automaticity"].fatigue
    )
    assert (
        high_ranked["mental_arithmetic_automaticity"].priority
        > low_ranked["mental_arithmetic_automaticity"].priority
    )


def test_recent_same_domain_history_prevents_one_domain_spam() -> None:
    entries = [
        _history_entry(
            test_code="numerical_operations",
            hours_ago=6,
            metrics=_base_metrics(score_ratio=0.47, post_error_inflation_ms=800.0),
            attempt_id=20,
        ),
        _history_entry(
            test_code="table_reading",
            hours_ago=8,
            metrics=_base_metrics(score_ratio=0.50, first_half_accuracy=0.88, second_half_accuracy=0.56),
            attempt_id=21,
        ),
        _adaptive_history_entry(
            hours_ago=4,
            blocks=(("mental_arithmetic_automaticity", "ma_percentage_snap", "quantitative_core"),),
            attempt_id=30,
        ),
        _adaptive_history_entry(
            hours_ago=24,
            blocks=(("mental_arithmetic_automaticity", "ma_rate_time_distance", "applied_rate_fuel"),),
            attempt_id=31,
            test_code="adaptive_session_short",
        ),
        _adaptive_history_entry(
            hours_ago=50,
            blocks=(("table_cross_reference_speed", "tbl_single_lookup_anchor", "single_lookup"),),
            attempt_id=32,
            test_code="adaptive_session_micro",
        ),
    ]

    plan = build_adaptive_session_plan(history=entries, seed=555, variant="full")

    assert plan is not None
    assert plan.blocks[0].primitive_id == "table_cross_reference_speed"


def test_scheduler_changes_priority_from_history() -> None:
    low_confidence_entries = [
        _history_entry(
            test_code="numerical_operations",
            hours_ago=8,
            metrics={
                **_base_metrics(
                    score_ratio=0.55,
                    first_half_accuracy=0.94,
                    second_half_accuracy=0.52,
                    post_error_inflation_ms=900.0,
                ),
                "longest_lapse_streak": "4",
            },
            attempt_id=40,
        ),
    ]
    stable_entries = [
        _history_entry(
            test_code="numerical_operations",
            hours_ago=12,
            metrics=_base_metrics(score_ratio=0.72, first_half_accuracy=0.78, second_half_accuracy=0.76),
            attempt_id=41,
        ),
        _history_entry(
            test_code="ma_percentage_snap",
            hours_ago=6,
            metrics=_base_metrics(score_ratio=0.76, accuracy=0.76),
            attempt_id=42,
        ),
    ]

    low_confidence = build_adaptive_session_plan(history=low_confidence_entries, seed=777, variant="full")
    stable = build_adaptive_session_plan(history=stable_entries, seed=777, variant="full")

    assert low_confidence is not None
    assert stable is not None
    low_ranked = {item.primitive_id: item for item in low_confidence.ranked_primitives}
    stable_ranked = {item.primitive_id: item for item in stable.ranked_primitives}
    assert (
        low_ranked["mental_arithmetic_automaticity"].priority
        > stable_ranked["mental_arithmetic_automaticity"].priority
    )
    assert (
        low_ranked["mental_arithmetic_automaticity"].recommended_level
        != stable_ranked["mental_arithmetic_automaticity"].recommended_level
    )


def test_variant_selection_uses_single_adaptive_session_shape() -> None:
    entries = [
        _history_entry(
            test_code="numerical_operations",
            hours_ago=72,
            metrics=_base_metrics(
                score_ratio=0.40,
                first_half_accuracy=0.95,
                second_half_accuracy=0.44,
                post_error_inflation_ms=1100.0,
            ),
            attempt_id=50,
        ),
    ]

    micro = build_adaptive_session_plan(history=entries, seed=111, variant="micro")
    short = build_adaptive_session_plan(history=entries, seed=111, variant="short")
    full = build_adaptive_session_plan(history=entries, seed=111, variant="full")

    assert micro is not None
    assert short is not None
    assert full is not None
    assert micro.code == "adaptive_session"
    assert short.code == "adaptive_session"
    assert full.code == "adaptive_session"
    assert {block.duration_s for block in micro.blocks} == {150.0}
    assert {block.duration_s for block in short.blocks} == {150.0}
    assert {block.duration_s for block in full.blocks} == {150.0}
    assert {block.form_factor for block in micro.blocks} <= {"micro"}
    assert {block.form_factor for block in short.blocks} <= {"micro", "short"}
    assert {block.form_factor for block in full.blocks} <= {"micro", "short"}
    assert all(block.mode == "adaptive_live" for block in full.blocks)


def test_build_adaptive_session_plan_filters_to_available_catalog_primitives(monkeypatch) -> None:
    visual_only_catalog = tuple(
        candidate
        for candidate in adaptive_scheduler.ADAPTIVE_DRILL_CATALOG
        if candidate.primitive_id == "visual_scan_discipline"
    )
    monkeypatch.setattr(adaptive_scheduler, "ADAPTIVE_DRILL_CATALOG", visual_only_catalog)

    entries = [
        _history_entry(
            test_code="numerical_operations",
            hours_ago=10,
            metrics=_base_metrics(score_ratio=0.22),
            attempt_id=70,
        ),
        _history_entry(
            test_code="visual_search",
            hours_ago=10,
            metrics=_base_metrics(score_ratio=0.44),
            attempt_id=71,
        ),
    ]

    plan = build_adaptive_session_plan(history=entries, seed=909, variant="full")

    assert plan is not None
    assert plan.blocks
    assert {block.primitive_id for block in plan.blocks} == {"visual_scan_discipline"}
    assert all(block.drill_code.startswith("vs_") for block in plan.blocks)


def test_training_mode_for_role_uses_new_role_rules() -> None:
    assert _training_mode_for_role("ma_one_step_fluency", role="target_anchor") is AntDrillMode.FRESH
    assert _training_mode_for_role("vs_mixed_tempo", role="target_anchor") is AntDrillMode.BUILD
    assert _training_mode_for_role("ma_one_step_fluency", role="target_tempo") is AntDrillMode.TEMPO
    assert _training_mode_for_role(
        "ma_one_step_fluency",
        role="adjacent_cross_train",
        target_confidence=0.40,
    ) is AntDrillMode.BUILD
    assert _training_mode_for_role("ma_one_step_fluency", role="reassessment_probe") is AntDrillMode.BUILD
    assert _training_mode_for_role(
        "ma_one_step_fluency",
        role="target_pressure_fatigue",
        fatigue_dominant=True,
    ) is AntDrillMode.FATIGUE_PROBE
    assert _training_mode_for_role(
        "ma_one_step_fluency",
        role="late_repeat_transfer",
        retention_repeat=True,
    ) is AntDrillMode.BUILD


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
        token = str(raw).strip().lower()
        if token in {"__skip_section__", "skip_section", "__skip_all__", "skip_all"}:
            self.phase = Phase.RESULTS
            return True
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

    ranking = rank_primitives(
        [
            _history_entry(test_code="numerical_operations", hours_ago=72, metrics=_base_metrics(score_ratio=0.45), attempt_id=1),
            _history_entry(test_code="table_reading", hours_ago=24, metrics=_base_metrics(score_ratio=0.60), attempt_id=2),
        ]
    )

    return AdaptiveSessionPlan(
        code="adaptive_session",
        title="Adaptive Test Plan",
        version=1,
        generated_at_utc=_iso(0),
        description="Small adaptive test plan.",
        notes=("Test note",),
        ranked_primitives=ranking.weakest_primitives,
        variant="full",
        blocks=(
            AdaptiveSessionBlock(
                block_index=0,
                primitive_id="mental_arithmetic_automaticity",
                primitive_label="Mental Arithmetic Automaticity",
                drill_code="ma_percentage_snap",
                mode="target_anchor",
                duration_s=1.0,
                difficulty_level=4,
                seed=111,
                reason_tags=("weak", "retention"),
                priority=0.8,
                drill_mode=AntDrillMode.BUILD,
                form_factor="micro",
                target_area="quantitative_core",
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
                drill_code="tbl_two_table_xref",
                mode="adjacent_cross_train",
                duration_s=1.0,
                difficulty_level=5,
                seed=222,
                reason_tags=("weak",),
                priority=0.6,
                drill_mode=AntDrillMode.TEMPO,
                form_factor="short",
                target_area="two_source_xref",
                linked_primitive_id="mental_arithmetic_automaticity",
                builder=_builder(
                    title="Table Tempo",
                    seed=222,
                    difficulty_level=5,
                    attempted=2,
                    correct=2,
                ),
            ),
        ),
        domain_summaries=ranking.domain_summaries,
    )


def _complete_adaptive_session(session: AdaptiveSession, clock: _FakeClock) -> None:
    session.activate()
    if session.stage is not AdaptiveStage.BLOCK:
        raise AssertionError(f"unexpected stage {session.stage}")
    engine = session.current_engine()
    assert engine is not None
    assert getattr(engine, "phase", None) is Phase.INSTRUCTIONS
    starter = getattr(engine, "start_practice", None)
    assert callable(starter)
    starter()
    assert getattr(engine, "phase", None) is Phase.SCORED
    clock.advance(1.2)
    session.update()
    assert session.stage is AdaptiveStage.BLOCK_RESULTS
    session.finish_session()
    assert session.stage is AdaptiveStage.RESULTS


def _start_adaptive_block(screen: AdaptiveSessionScreen, surface: pygame.Surface) -> None:
    screen.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))
    screen.render(surface)
    for _ in range(INTRO_LOADING_MIN_FRAMES):
        screen.render(surface)
    screen.handle_event(
        pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": 0, "unicode": ""})
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

    _complete_adaptive_session(session, clock)

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
    assert metric_rows["block.01.attempted"] == "2"
    assert metric_rows["block.01.form_factor"] == "micro"
    assert metric_rows["block.01.target_area"] == "quantitative_core"
    assert "scheduler.recommended_level.mental_arithmetic_automaticity" in metric_rows
    assert "scheduler.level_confidence.mental_arithmetic_automaticity" in metric_rows
    assert "scheduler.domain.quantitative.weakest_primitive_id" in metric_rows
    assert telemetry_kinds == [
        "activity_started",
        "block_started",
        "question",
        "question",
        "block_completed",
        "activity_completed",
    ]
    assert metric_rows["block.01.training_mode"] == AntDrillMode.BUILD.value


def test_adaptive_session_results_snapshot_includes_split_half_lines() -> None:
    clock = _FakeClock()
    session = AdaptiveSession(clock=clock, seed=444, plan=_small_adaptive_plan(clock))

    _complete_adaptive_session(session, clock)

    snapshot = session.snapshot()

    assert snapshot.stage is AdaptiveStage.RESULTS
    assert any("Drill splits:" == line for line in snapshot.note_lines)
    assert any("1H " in line and "2H " in line for line in snapshot.note_lines)


def test_adaptive_session_replans_remaining_blocks_after_completed_result() -> None:
    entries = [
        _history_entry(
            test_code="ma_one_step_fluency",
            hours_ago=4,
            metrics=_base_metrics(score_ratio=0.52, duration_s=120.0, training_mode="build"),
            activity_kind="drill",
            attempt_id=201,
        ),
        _history_entry(
            test_code="tbl_single_lookup_anchor",
            hours_ago=3,
            metrics=_base_metrics(score_ratio=0.53, duration_s=120.0, training_mode="build"),
            activity_kind="drill",
            attempt_id=202,
        ),
    ]
    base_plan = build_adaptive_session_plan(history=entries, seed=602, variant="full")

    assert base_plan is not None
    assert base_plan.replan_enabled
    assert len(base_plan.blocks) == 1

    clock = _FakeClock()
    first_block = base_plan.blocks[0]
    first_block_with_builder = AdaptiveSessionBlock(
        block_index=first_block.block_index,
        primitive_id=first_block.primitive_id,
        primitive_label=first_block.primitive_label,
        drill_code=first_block.drill_code,
        mode=first_block.mode,
        duration_s=first_block.duration_s,
        difficulty_level=first_block.difficulty_level,
        seed=first_block.seed,
        reason_tags=first_block.reason_tags,
        priority=first_block.priority,
        drill_mode=first_block.drill_mode,
        builder=lambda: _FakeBlockEngine(
            clock=clock,
            title="Perfect First Block",
            seed=first_block.seed,
            difficulty_level=first_block.difficulty_level,
            scored_duration_s=1.0,
            attempted=4,
            correct=4,
        ),
        form_factor=first_block.form_factor,
        target_area=first_block.target_area,
        linked_primitive_id=first_block.linked_primitive_id,
        comparable_level=first_block.comparable_level,
    )
    plan = AdaptiveSessionPlan(
        code=base_plan.code,
        title=base_plan.title,
        version=base_plan.version,
        generated_at_utc=base_plan.generated_at_utc,
        description=base_plan.description,
        notes=base_plan.notes,
        ranked_primitives=base_plan.ranked_primitives,
        blocks=(first_block_with_builder, *base_plan.blocks[1:]),
        variant=base_plan.variant,
        domain_summaries=base_plan.domain_summaries,
        replan_enabled=True,
    )

    session = AdaptiveSession(clock=clock, seed=602, plan=plan, history=entries)
    session.activate()
    engine = session.current_engine()
    assert engine is not None
    assert getattr(engine, "phase", None) is Phase.INSTRUCTIONS
    starter = getattr(engine, "start_practice", None)
    assert callable(starter)
    starter()
    clock.advance(1.2)
    session.update()
    assert session.stage is AdaptiveStage.BLOCK_RESULTS
    session.continue_after_block_results()

    current = session.current_block_plan()

    assert session.stage is AdaptiveStage.BLOCK
    assert session._replan_count == 1
    assert current is not None
    assert current.block_index == 1
    assert len(session._plan.blocks) == 2
    assert current.primitive_id != first_block.primitive_id
    assert current.drill_code != first_block.drill_code


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
    _complete_adaptive_session(session, clock)
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


def test_adaptive_session_bootstrap_enter_routes_to_retry_factory(tmp_path) -> None:
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
            test_code="adaptive_session",
            screen_factory=lambda: BenchmarkScreen(
                app,
                session=BenchmarkSession(plan=build_benchmark_plan(clock=_FakeClock())),
            ),
        )
        app.push(screen)

        screen.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))

        assert isinstance(app._screens[-1], BenchmarkScreen)
    finally:
        pygame.quit()


def test_adaptive_session_bootstrap_keypad_enter_routes_to_retry_factory(tmp_path) -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        store = ResultsStore(tmp_path / "bootstrap-kp.sqlite3")
        app = App(surface=surface, font=font, results_store=store, app_version="test")
        app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))

        screen = AdaptiveSessionScreen(
            app,
            session=None,
            test_code="adaptive_session",
            screen_factory=lambda: BenchmarkScreen(
                app,
                session=BenchmarkSession(plan=build_benchmark_plan(clock=_FakeClock())),
            ),
        )
        app.push(screen)

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_KP_ENTER, "unicode": ""})
        )

        assert isinstance(app._screens[-1], BenchmarkScreen)
    finally:
        pygame.quit()


def test_adaptive_pause_menu_shows_unified_actions_and_settings(tmp_path) -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        store = ResultsStore(tmp_path / "adaptive-settings.sqlite3")
        app = App(surface=surface, font=font, results_store=store, app_version="test")
        app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))
        clock = _FakeClock()
        screen = AdaptiveSessionScreen(
            app,
            session=AdaptiveSession(clock=clock, seed=555, plan=_small_adaptive_plan(clock)),
            test_code="adaptive_session",
        )
        app.push(screen)
        _start_adaptive_block(screen, surface)

        screen.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "unicode": ""}))

        assert screen._pause_menu_options() == (
            "Resume",
            "Skip Current Segment",
            "Restart Current",
            "Settings",
            "End Session",
            "Main Menu",
        )

        settings_index = screen._pause_menu_options().index("Settings")
        for _ in range(settings_index):
            screen.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        screen.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))

        assert [key for key, _label, _value in screen._pause_settings_rows()] == [
            "seed_mode",
            "seed_value",
            "review_mode",
            "joystick_bindings",
            "apply_restart",
            "back",
        ]
    finally:
        pygame.quit()


def test_adaptive_pause_menu_skip_current_segment_advances_to_next_block() -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        app = App(surface=surface, font=font, app_version="test")
        app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))
        clock = _FakeClock()
        session = AdaptiveSession(clock=clock, seed=555, plan=_small_adaptive_plan(clock))
        screen = AdaptiveSessionScreen(app, session=session, test_code="adaptive_session")
        app.push(screen)
        _start_adaptive_block(screen, surface)

        assert session.snapshot().block_index == 1
        screen.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "unicode": ""}))
        skip_index = screen._pause_menu_options().index("Skip Current Segment")
        for _ in range(skip_index):
            screen.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        screen.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))

        assert session.stage is AdaptiveStage.BLOCK_RESULTS
        assert session.snapshot().block_index == 1
    finally:
        pygame.quit()


def test_adaptive_keypad_enter_continues_after_block_results() -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        app = App(surface=surface, font=font, app_version="test")
        app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))
        clock = _FakeClock()
        session = AdaptiveSession(clock=clock, seed=555, plan=_small_adaptive_plan(clock))
        screen = AdaptiveSessionScreen(app, session=session, test_code="adaptive_session")
        app.push(screen)
        _start_adaptive_block(screen, surface)

        session.debug_skip_current_block()
        assert session.stage is AdaptiveStage.BLOCK_RESULTS

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_KP_ENTER, "unicode": ""})
        )

        assert session.stage is AdaptiveStage.BLOCK
        assert session.snapshot().block_index == 2
        assert getattr(session.current_engine(), "phase", None) is Phase.INSTRUCTIONS
    finally:
        pygame.quit()


def test_adaptive_escape_opens_pause_on_block_results_and_final_results() -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        app = App(surface=surface, font=font, app_version="test")
        app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))
        clock = _FakeClock()
        session = AdaptiveSession(clock=clock, seed=555, plan=_small_adaptive_plan(clock))
        screen = AdaptiveSessionScreen(app, session=session, test_code="adaptive_session")
        app.push(screen)
        _start_adaptive_block(screen, surface)

        session.debug_skip_current_block()
        assert session.stage is AdaptiveStage.BLOCK_RESULTS

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "unicode": ""})
        )

        assert screen._pause_menu_active is True
        screen._pause_menu_hitboxes = {}
        screen.render(surface)
        assert screen._pause_menu_hitboxes

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "unicode": ""})
        )
        assert screen._pause_menu_active is False

        session.finish_session()
        screen.render(surface)
        assert session.stage is AdaptiveStage.RESULTS

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "unicode": ""})
        )

        assert screen._pause_menu_active is True
        screen._pause_menu_hitboxes = {}
        screen.render(surface)
        assert screen._pause_menu_hitboxes
    finally:
        pygame.quit()


def test_adaptive_pause_menu_skip_does_not_persist_attempt(tmp_path) -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        store = ResultsStore(tmp_path / "adaptive-skip.sqlite3")
        app = App(surface=surface, font=font, results_store=store, app_version="test")
        app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))
        clock = _FakeClock()
        session = AdaptiveSession(clock=clock, seed=555, plan=_small_adaptive_plan(clock))
        screen = AdaptiveSessionScreen(app, session=session, test_code="adaptive_session")
        app.push(screen)
        _start_adaptive_block(screen, surface)

        screen.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "unicode": ""}))
        skip_index = screen._pause_menu_options().index("Skip Current Segment")
        for _ in range(skip_index):
            screen.handle_event(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""})
            )
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
        )

        screen.render(surface)

        with sqlite3.connect(store.path) as conn:
            attempt_count = conn.execute("SELECT COUNT(*) FROM attempt").fetchone()
            activity_count = conn.execute("SELECT COUNT(*) FROM activity_session").fetchone()

        session_summary = store.session_summary()
        assert session_summary is not None
        assert session_summary.activity_count == 1
        assert session_summary.completed_activity_count == 0
        assert session_summary.attempt_count == 0
        assert attempt_count == (0,)
        assert activity_count == (1,)
        assert screen._results_persistence_lines == ["Local save skipped in dev mode."]
    finally:
        pygame.quit()


def test_ui_smoke_open_adaptive_session_from_main_menu_bootstraps_directly(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    monkeypatch.setenv("CFAST_RESULTS_DB_PATH", str(tmp_path / "smoke.sqlite3"))

    from cfast_trainer.app import run

    def inject(frame: int) -> None:
        if frame == 1:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))
        elif frame == 2:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))

    assert run(max_frames=25, event_injector=inject) == 0
