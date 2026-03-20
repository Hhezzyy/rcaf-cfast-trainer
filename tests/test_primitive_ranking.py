from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from cfast_trainer.persistence import AttemptHistoryEntry
from cfast_trainer.primitive_ranking import canonical_ranked_primitive_id_for_code, rank_primitives


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
    timeout_rate: float | None = None,
    mean_rt_ms: float | None = None,
    median_rt_ms: float | None = None,
    rt_variance_ms2: float | None = None,
    first_accuracy: float | None = None,
    last_accuracy: float | None = None,
    first_timeout_rate: float | None = None,
    last_timeout_rate: float | None = None,
    first_half_accuracy: float | None = None,
    second_half_accuracy: float | None = None,
    first_half_timeout_rate: float | None = None,
    second_half_timeout_rate: float | None = None,
    post_error_inflation_ms: float | None = None,
    post_error_accuracy_drop: float | None = None,
    longest_lapse_streak: int | None = None,
    distractor_capture_count: int | None = None,
    switch_cost_ms: float | None = None,
    overshoot_count: int | None = None,
    intrusion_count: int | None = None,
    omission_count: int | None = None,
    order_error_count: int | None = None,
) -> dict[str, str]:
    metrics: dict[str, str] = {}
    if score_ratio is not None:
        metrics["score_ratio"] = _fmt(score_ratio)
    if accuracy is not None:
        metrics["accuracy"] = _fmt(accuracy)
    if timeout_rate is not None:
        metrics["timeout_rate"] = _fmt(timeout_rate)
    if mean_rt_ms is not None:
        metrics["mean_rt_ms"] = _fmt(mean_rt_ms)
    if median_rt_ms is not None:
        metrics["median_rt_ms"] = _fmt(median_rt_ms)
    if rt_variance_ms2 is not None:
        metrics["rt_variance_ms2"] = _fmt(rt_variance_ms2)
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
    if post_error_accuracy_drop is not None:
        metrics["post_error_next_item_accuracy_drop"] = _fmt(post_error_accuracy_drop)
    if longest_lapse_streak is not None:
        metrics["longest_lapse_streak"] = _fmt(longest_lapse_streak)
    if distractor_capture_count is not None:
        metrics["distractor_capture_count"] = _fmt(distractor_capture_count)
    if switch_cost_ms is not None:
        metrics["switch_cost_ms"] = _fmt(switch_cost_ms)
    if overshoot_count is not None:
        metrics["overshoot_count"] = _fmt(overshoot_count)
    if intrusion_count is not None:
        metrics["intrusion_count"] = _fmt(intrusion_count)
    if omission_count is not None:
        metrics["omission_count"] = _fmt(omission_count)
    if order_error_count is not None:
        metrics["order_error_count"] = _fmt(order_error_count)
    return metrics


def _history_entry(
    *,
    test_code: str,
    hours_ago: float,
    metrics: dict[str, str],
    attempt_id: int,
    activity_kind: str = "test",
    level: int = 5,
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
        difficulty_level_start=level,
        difficulty_level_end=level,
        metrics=dict(metrics),
    )


def _rank_map(entries: list[AttemptHistoryEntry], *, seed: int = 0):
    result = rank_primitives(entries, seed=seed)
    return {item.primitive_id: item for item in result.weakest_primitives}


def test_rank_primitives_is_deterministic_for_same_history_and_seed() -> None:
    entries = [
        _history_entry(
            test_code="numerical_operations",
            hours_ago=10,
            metrics=_base_metrics(score_ratio=0.42, timeout_rate=0.08, mean_rt_ms=1700.0),
            attempt_id=1,
        ),
        _history_entry(
            test_code="table_reading",
            hours_ago=8,
            metrics=_base_metrics(
                score_ratio=0.42,
                timeout_rate=0.08,
                mean_rt_ms=1700.0,
                first_half_accuracy=0.90,
                second_half_accuracy=0.58,
            ),
            attempt_id=2,
        ),
    ]

    first = rank_primitives(entries, seed=77)
    second = rank_primitives(entries, seed=77)

    assert [
        (item.primitive_id, round(item.priority, 6), item.recommended_level)
        for item in first.weakest_primitives
    ] == [
        (item.primitive_id, round(item.priority, 6), item.recommended_level)
        for item in second.weakest_primitives
    ]


def test_canonical_ranked_primitive_id_prefers_catalog_backed_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "cfast_trainer.primitive_ranking.guide_ranking_primitive_id_for_code",
        lambda code: "visual_scan_discipline" if code == "mystery_catalog_code" else None,
    )

    assert canonical_ranked_primitive_id_for_code("mystery_catalog_code") == "visual_scan_discipline"


def test_low_confidence_primitive_keeps_exploration_bonus() -> None:
    entries = [
        _history_entry(
            test_code="table_reading",
            hours_ago=6,
            metrics=_base_metrics(score_ratio=0.70, timeout_rate=0.02, mean_rt_ms=1200.0),
            attempt_id=1,
        ),
        _history_entry(
            test_code="numerical_operations",
            hours_ago=9,
            metrics=_base_metrics(score_ratio=0.70, timeout_rate=0.02, mean_rt_ms=1200.0),
            attempt_id=2,
        ),
        _history_entry(
            test_code="ma_one_step_fluency",
            hours_ago=5,
            metrics=_base_metrics(score_ratio=0.74, timeout_rate=0.01, mean_rt_ms=950.0),
            attempt_id=3,
        ),
        _history_entry(
            test_code="ma_rate_time_distance",
            hours_ago=3,
            metrics=_base_metrics(score_ratio=0.76, timeout_rate=0.01, mean_rt_ms=910.0),
            attempt_id=4,
        ),
    ]

    ranked = _rank_map(entries)

    assert ranked["table_cross_reference_speed"].confidence < ranked["mental_arithmetic_automaticity"].confidence
    assert (
        ranked["table_cross_reference_speed"].exploration_bonus
        > ranked["mental_arithmetic_automaticity"].exploration_bonus
    )


def test_high_leverage_weak_primitive_outranks_lower_leverage_peer() -> None:
    entries = [
        _history_entry(
            test_code="numerical_operations",
            hours_ago=4,
            metrics=_base_metrics(score_ratio=0.55, timeout_rate=0.04, mean_rt_ms=1500.0),
            attempt_id=1,
        ),
        _history_entry(
            test_code="rapid_tracking",
            hours_ago=4,
            metrics=_base_metrics(score_ratio=0.55, timeout_rate=0.04, mean_rt_ms=1500.0),
            attempt_id=2,
        ),
    ]

    ranked = _rank_map(entries)

    assert ranked["mental_arithmetic_automaticity"].priority > ranked["tracking_stability_low_load"].priority


def test_late_session_collapse_materially_raises_priority() -> None:
    entries = [
        _history_entry(
            test_code="table_reading",
            hours_ago=3,
            metrics=_base_metrics(
                score_ratio=0.74,
                timeout_rate=0.05,
                mean_rt_ms=1600.0,
                first_half_accuracy=0.94,
                second_half_accuracy=0.30,
                first_half_timeout_rate=0.00,
                second_half_timeout_rate=0.24,
            ),
            attempt_id=1,
        ),
        _history_entry(
            test_code="numerical_operations",
            hours_ago=3,
            metrics=_base_metrics(
                score_ratio=0.74,
                timeout_rate=0.05,
                mean_rt_ms=1600.0,
                first_half_accuracy=0.82,
                second_half_accuracy=0.78,
            ),
            attempt_id=2,
        ),
    ]

    ranked = _rank_map(entries)

    assert ranked["table_cross_reference_speed"].fatigue_penalty > 0.50
    assert ranked["table_cross_reference_speed"].priority > ranked["mental_arithmetic_automaticity"].priority


def test_lapse_streak_penalty_materially_raises_priority() -> None:
    entries = [
        _history_entry(
            test_code="visual_search",
            hours_ago=4,
            metrics=_base_metrics(
                score_ratio=0.70,
                timeout_rate=0.18,
                longest_lapse_streak=4,
                mean_rt_ms=1450.0,
            ),
            attempt_id=10,
        ),
        _history_entry(
            test_code="numerical_operations",
            hours_ago=4,
            metrics=_base_metrics(
                score_ratio=0.70,
                timeout_rate=0.04,
                longest_lapse_streak=1,
                mean_rt_ms=1450.0,
            ),
            attempt_id=11,
        ),
    ]

    ranked = _rank_map(entries)

    assert ranked["visual_scan_discipline"].lapse_penalty > ranked["mental_arithmetic_automaticity"].lapse_penalty
    assert ranked["visual_scan_discipline"].priority > ranked["mental_arithmetic_automaticity"].priority


def test_distractor_and_switch_penalties_raise_visual_search_priority() -> None:
    entries = [
        _history_entry(
            test_code="vs_priority_switch_search",
            hours_ago=5,
            metrics=_base_metrics(
                score_ratio=0.72,
                mean_rt_ms=1300.0,
                distractor_capture_count=4,
                switch_cost_ms=850.0,
            ),
            attempt_id=20,
        ),
        _history_entry(
            test_code="table_reading",
            hours_ago=5,
            metrics=_base_metrics(
                score_ratio=0.72,
                mean_rt_ms=1300.0,
            ),
            attempt_id=21,
        ),
    ]

    ranked = _rank_map(entries)

    assert ranked["visual_scan_discipline"].distractor_penalty > 0.25
    assert ranked["visual_scan_discipline"].switch_penalty > 0.50
    assert ranked["visual_scan_discipline"].priority > ranked["table_cross_reference_speed"].priority


def test_control_and_interference_penalties_raise_tracking_and_dual_task_priority() -> None:
    entries = [
        _history_entry(
            test_code="rapid_tracking",
            hours_ago=6,
            metrics=_base_metrics(
                score_ratio=0.74,
                mean_rt_ms=1400.0,
                rt_variance_ms2=1800000.0,
                overshoot_count=6,
            ),
            attempt_id=30,
        ),
        _history_entry(
            test_code="cln_sequence_math_recall",
            hours_ago=6,
            metrics=_base_metrics(
                score_ratio=0.74,
                mean_rt_ms=1400.0,
                intrusion_count=3,
                omission_count=2,
                order_error_count=1,
            ),
            attempt_id=31,
        ),
        _history_entry(
            test_code="table_reading",
            hours_ago=6,
            metrics=_base_metrics(
                score_ratio=0.74,
                mean_rt_ms=1400.0,
            ),
            attempt_id=32,
        ),
    ]

    ranked = _rank_map(entries)

    assert ranked["tracking_stability_low_load"].control_penalty > 0.30
    assert ranked["dual_task_stability_fatigue"].interference_penalty > 0.40
    assert ranked["tracking_stability_low_load"].priority > ranked["table_cross_reference_speed"].priority
    assert ranked["dual_task_stability_fatigue"].priority > ranked["table_cross_reference_speed"].priority


def test_recommended_level_promotes_with_clean_score_ratio_signals() -> None:
    entries = [
        _history_entry(
            test_code="ma_one_step_fluency",
            hours_ago=4,
            metrics=_base_metrics(
                score_ratio=0.92,
                timeout_rate=0.00,
                mean_rt_ms=880.0,
                rt_variance_ms2=40000.0,
                first_half_accuracy=0.93,
                second_half_accuracy=0.92,
                post_error_inflation_ms=80.0,
                post_error_accuracy_drop=0.02,
            ),
            attempt_id=1,
            level=5,
        ),
        _history_entry(
            test_code="ma_rate_time_distance",
            hours_ago=2,
            metrics=_base_metrics(
                score_ratio=0.91,
                timeout_rate=0.00,
                mean_rt_ms=900.0,
                rt_variance_ms2=36000.0,
                first_half_accuracy=0.92,
                second_half_accuracy=0.91,
                post_error_inflation_ms=60.0,
                post_error_accuracy_drop=0.01,
            ),
            attempt_id=2,
            level=5,
        ),
    ]

    ranked = _rank_map(entries)
    item = ranked["mental_arithmetic_automaticity"]

    assert item.recommended_level == 6
    assert item.level_confidence > 0.60


def test_recommended_level_holds_when_only_raw_accuracy_is_high() -> None:
    entries = [
        _history_entry(
            test_code="ma_one_step_fluency",
            hours_ago=2,
            metrics=_base_metrics(
                accuracy=0.98,
                timeout_rate=0.00,
                mean_rt_ms=900.0,
                rt_variance_ms2=40000.0,
                first_half_accuracy=0.98,
                second_half_accuracy=0.97,
            ),
            attempt_id=1,
            level=5,
        ),
    ]

    ranked = _rank_map(entries)

    assert ranked["mental_arithmetic_automaticity"].recommended_level == 5


def test_recommended_level_demotes_on_meltdown_signals() -> None:
    entries = [
        _history_entry(
            test_code="rt_pressure_run",
            hours_ago=1,
            metrics=_base_metrics(
                score_ratio=0.55,
                timeout_rate=0.24,
                mean_rt_ms=2400.0,
                rt_variance_ms2=2500000.0,
                first_half_accuracy=0.82,
                second_half_accuracy=0.34,
                post_error_inflation_ms=1100.0,
                post_error_accuracy_drop=0.20,
            ),
            attempt_id=1,
            level=5,
        ),
    ]

    ranked = _rank_map(entries)
    item = ranked["tracking_stability_low_load"]

    assert item.recommended_level == 4
    assert item.last_meltdown_level == 5
    assert item.instability_penalty > 0.60
