from __future__ import annotations

import math
from dataclasses import dataclass

from .adaptive_difficulty import DifficultyFamilyId, family_id_for_code
from .results import AttemptResult


@dataclass(frozen=True, slots=True)
class SkillEvidenceScore:
    mastery_score: float
    weakness: float
    speed_score: float
    stability_score: float
    fatigue_drop: float
    timeout_penalty: float
    false_alarm_penalty: float
    confidence: float
    recommended_training_target: str


_MEMORY_FAMILIES: frozenset[DifficultyFamilyId] = frozenset(
    {"auditory_multitask", "cln_multitask", "visual_memory_updating"}
)
_SEARCH_FAMILIES: frozenset[DifficultyFamilyId] = frozenset({"search_vigilance"})
_PSYCHOMOTOR_FAMILIES: frozenset[DifficultyFamilyId] = frozenset({"psychomotor_tracking"})
_SPATIAL_FAMILIES: frozenset[DifficultyFamilyId] = frozenset(
    {"angle_bearing", "instrument_orientation", "spatial_integration_trace"}
)


def skill_evidence_from_attempt(result: AttemptResult) -> SkillEvidenceScore:
    return skill_evidence_from_metrics(
        result.metrics,
        test_code=result.test_code,
        family_id=None,
    )


def skill_evidence_from_metrics(
    metrics: dict[str, str],
    *,
    test_code: str | None,
    family_id: DifficultyFamilyId | None = None,
) -> SkillEvidenceScore:
    resolved_family = family_id if family_id is not None else family_id_for_code(test_code)
    attempted = _attempted_count(metrics)
    timeout_penalty = _timeout_penalty(metrics, attempted=attempted)
    false_alarm_penalty = _false_alarm_penalty(metrics, attempted=attempted)
    fatigue_drop = _fatigue_drop(metrics)
    stability_score = _stability_score(
        metrics,
        fatigue_drop=fatigue_drop,
        timeout_penalty=timeout_penalty,
    )
    speed_score = _speed_score(metrics)
    mastery_score = _family_mastery_score(
        metrics,
        family_id=resolved_family,
        speed_score=speed_score,
        stability_score=stability_score,
        timeout_penalty=timeout_penalty,
        false_alarm_penalty=false_alarm_penalty,
        fatigue_drop=fatigue_drop,
        attempted=attempted,
    )
    confidence = _confidence(metrics, attempted=attempted, family_id=resolved_family)
    target = _recommended_target(
        family_id=resolved_family,
        mastery_score=mastery_score,
        speed_score=speed_score,
        stability_score=stability_score,
        fatigue_drop=fatigue_drop,
        timeout_penalty=timeout_penalty,
        false_alarm_penalty=false_alarm_penalty,
    )
    return SkillEvidenceScore(
        mastery_score=_clamp(mastery_score),
        weakness=_clamp(1.0 - mastery_score),
        speed_score=_clamp(speed_score),
        stability_score=_clamp(stability_score),
        fatigue_drop=_clamp(fatigue_drop),
        timeout_penalty=_clamp(timeout_penalty),
        false_alarm_penalty=_clamp(false_alarm_penalty),
        confidence=_clamp(confidence),
        recommended_training_target=target,
    )


def _metric_float(metrics: dict[str, str], *keys: str) -> float | None:
    for key in keys:
        raw = metrics.get(key)
        if raw is None:
            continue
        token = str(raw).strip()
        if token == "":
            continue
        try:
            value = float(token)
        except Exception:
            continue
        if math.isfinite(value):
            return float(value)
    return None


def _metric_int(metrics: dict[str, str], *keys: str) -> int | None:
    value = _metric_float(metrics, *keys)
    if value is None:
        return None
    return int(round(value))


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _mean_available(*values: float | None) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return None
    return sum(clean) / float(len(clean))


def _weighted_average(pairs: tuple[tuple[float | None, float], ...]) -> float | None:
    total = 0.0
    weight_total = 0.0
    for value, weight in pairs:
        if value is None:
            continue
        clean_weight = max(0.0, float(weight))
        total += _clamp(float(value)) * clean_weight
        weight_total += clean_weight
    if weight_total <= 0.0:
        return None
    return total / weight_total


def _attempted_count(metrics: dict[str, str]) -> float:
    attempted = _metric_float(metrics, "attempted")
    if attempted is not None and attempted > 0.0:
        return max(1.0, float(attempted))
    correct = _metric_float(metrics, "correct")
    accuracy = _metric_float(metrics, "accuracy")
    if correct is not None and accuracy is not None and accuracy > 0.0:
        return max(1.0, float(correct) / max(float(accuracy), 1.0e-6))
    return 0.0


def _base_performance(metrics: dict[str, str]) -> float | None:
    return _first_ratio(
        metrics,
        "score_ratio",
        "accuracy",
        "memory_recall_accuracy",
        "orientation_correct",
        "secondary_task_accuracy",
        "primary_task_accuracy",
    )


def _first_ratio(metrics: dict[str, str], *keys: str) -> float | None:
    value = _metric_float(metrics, *keys)
    if value is None:
        return None
    return _clamp(value)


def _speed_score(metrics: dict[str, str]) -> float:
    rt_ms = _metric_float(metrics, "mean_rt_ms", "median_rt_ms")
    rt_score = None
    if rt_ms is not None:
        if rt_ms <= 750.0:
            rt_score = 1.0
        elif rt_ms >= 5000.0:
            rt_score = 0.0
        else:
            rt_score = 1.0 - ((float(rt_ms) - 750.0) / (5000.0 - 750.0))

    first_throughput = _metric_float(metrics, "first_half_throughput", "first_3m_throughput")
    second_throughput = _metric_float(metrics, "second_half_throughput", "last_3m_throughput")
    throughput = _metric_float(metrics, "throughput_per_min")
    throughput_score = None
    if throughput is not None:
        reference = max(8.0, float(first_throughput or throughput))
        throughput_score = _clamp(float(throughput) / reference)
    if first_throughput is not None and second_throughput is not None and first_throughput > 0.0:
        trend = _clamp(float(second_throughput) / max(float(first_throughput), 1.0e-6))
        throughput_score = trend if throughput_score is None else max(throughput_score, trend)

    return _mean_available(rt_score, throughput_score) or 0.65


def _fatigue_drop(metrics: dict[str, str]) -> float:
    first_accuracy = _metric_float(metrics, "first_half_accuracy", "first_3m_accuracy")
    second_accuracy = _metric_float(metrics, "second_half_accuracy", "last_3m_accuracy")
    accuracy_drop = 0.0
    if first_accuracy is not None and second_accuracy is not None:
        accuracy_drop = max(0.0, float(first_accuracy) - float(second_accuracy))

    first_throughput = _metric_float(metrics, "first_half_throughput", "first_3m_throughput")
    second_throughput = _metric_float(metrics, "second_half_throughput", "last_3m_throughput")
    throughput_drop = 0.0
    if first_throughput is not None and second_throughput is not None and first_throughput > 0.0:
        throughput_drop = max(0.0, (float(first_throughput) - float(second_throughput)) / float(first_throughput))

    first_timeout = _metric_float(metrics, "first_half_timeout_rate", "first_3m_timeout_rate")
    second_timeout = _metric_float(metrics, "second_half_timeout_rate", "last_3m_timeout_rate")
    timeout_rise = 0.0
    if first_timeout is not None and second_timeout is not None:
        timeout_rise = max(0.0, float(second_timeout) - float(first_timeout))

    return _clamp(max(accuracy_drop, (0.65 * accuracy_drop) + (0.25 * throughput_drop) + (0.10 * timeout_rise)))


def _timeout_penalty(metrics: dict[str, str], *, attempted: float) -> float:
    rate = _metric_float(metrics, "timeout_rate")
    if rate is not None:
        return _clamp(rate)
    count = _metric_float(metrics, "timeout_count")
    if count is not None and attempted > 0.0:
        return _clamp(float(count) / float(attempted))
    return 0.0


def _false_alarm_penalty(metrics: dict[str, str], *, attempted: float) -> float:
    rate = _metric_float(metrics, "false_alarm_rate", "false_command_rate")
    if rate is not None:
        return _clamp(rate)
    count = _metric_float(
        metrics,
        "false_alarm_count",
        "false_alarms",
        "bridge.false_alarms",
    )
    if count is not None and attempted > 0.0:
        return _clamp(float(count) / float(attempted))
    return 0.0


def _stability_score(
    metrics: dict[str, str],
    *,
    fatigue_drop: float,
    timeout_penalty: float,
) -> float:
    variance = _metric_float(metrics, "rt_variance_ms2")
    rt_instability = 0.0
    if variance is not None and variance >= 0.0:
        rt_instability = _clamp(math.sqrt(float(variance)) / 1600.0)
    post_error = _metric_float(metrics, "post_error_next_item_rt_inflation_ms")
    post_error_penalty = 0.0 if post_error is None else _clamp(max(0.0, post_error) / 1800.0)
    instability = (0.35 * rt_instability) + (0.30 * fatigue_drop) + (0.20 * timeout_penalty) + (0.15 * post_error_penalty)
    return _clamp(1.0 - instability)


def _family_mastery_score(
    metrics: dict[str, str],
    *,
    family_id: DifficultyFamilyId,
    speed_score: float,
    stability_score: float,
    timeout_penalty: float,
    false_alarm_penalty: float,
    fatigue_drop: float,
    attempted: float,
) -> float:
    if family_id in _PSYCHOMOTOR_FAMILIES:
        return _psychomotor_mastery(metrics, stability_score=stability_score)
    if family_id in _SPATIAL_FAMILIES:
        return _spatial_mastery(metrics, stability_score=stability_score)
    if family_id in _MEMORY_FAMILIES:
        return _memory_multitask_mastery(
            metrics,
            stability_score=stability_score,
            timeout_penalty=timeout_penalty,
            false_alarm_penalty=false_alarm_penalty,
            attempted=attempted,
        )
    if family_id in _SEARCH_FAMILIES:
        base = _base_performance(metrics)
        return _weighted_average(
            (
                (base, 0.62),
                (speed_score, 0.15),
                (stability_score, 0.15),
                (1.0 - false_alarm_penalty, 0.05),
                (1.0 - timeout_penalty, 0.03),
            )
        ) or 0.65
    base = _base_performance(metrics)
    return _weighted_average(
        (
            (base, 0.70),
            (speed_score, 0.10),
            (stability_score, 0.12),
            (1.0 - timeout_penalty, 0.05),
            (1.0 - fatigue_drop, 0.03),
        )
    ) or 0.65


def _psychomotor_mastery(metrics: dict[str, str], *, stability_score: float) -> float:
    time_on_target = _first_ratio(metrics, "time_on_target_ratio", "on_target_ratio")
    tracking_error = _metric_float(
        metrics,
        "tracking_error_mean",
        "mean_tracking_error",
        "mean_error",
        "rms_tracking_error",
        "rms_error",
    )
    tracking_quality = None
    if tracking_error is not None:
        tracking_quality = 1.0 - _clamp(float(tracking_error) / 1.0)
    prediction_error = _metric_float(
        metrics,
        "prediction_error_mean",
        "obscured_mean_error",
        "visible_mean_error",
    )
    prediction_quality = None
    if prediction_error is not None:
        prediction_quality = 1.0 - _clamp(float(prediction_error) / 1.0)
    prediction_score = _first_ratio(metrics, "prediction_score_ratio", "obscured_tracking_ratio")
    base = _base_performance(metrics)
    return _weighted_average(
        (
            (time_on_target, 0.34),
            (tracking_quality, 0.24),
            (prediction_quality, 0.12),
            (prediction_score, 0.10),
            (base, 0.12),
            (stability_score, 0.08),
        )
    ) or 0.65


def _spatial_mastery(metrics: dict[str, str], *, stability_score: float) -> float:
    angular_error = _metric_float(metrics, "angular_error_deg", "angle_error_deg")
    angular_quality = None if angular_error is None else 1.0 - _clamp(float(angular_error) / 90.0)
    orientation = _first_ratio(metrics, "orientation_correct")
    position_error = _metric_float(metrics, "position_error", "recall_distance_error")
    position_quality = None if position_error is None else 1.0 - _clamp(float(position_error) / 10.0)
    view_error = _metric_float(metrics, "view_integration_error")
    view_quality = None if view_error is None else 1.0 - _clamp(float(view_error))
    base = _base_performance(metrics)
    return _weighted_average(
        (
            (base, 0.36),
            (orientation, 0.20),
            (angular_quality, 0.16),
            (position_quality, 0.12),
            (view_quality, 0.10),
            (stability_score, 0.06),
        )
    ) or 0.65


def _memory_multitask_mastery(
    metrics: dict[str, str],
    *,
    stability_score: float,
    timeout_penalty: float,
    false_alarm_penalty: float,
    attempted: float,
) -> float:
    recall = _first_ratio(metrics, "memory_recall_accuracy", "recall_accuracy")
    primary = _first_ratio(metrics, "primary_task_accuracy")
    secondary = _first_ratio(metrics, "secondary_task_accuracy")
    dual_drop = _metric_float(metrics, "dual_task_drop")
    dual_quality = None if dual_drop is None else 1.0 - _clamp(float(dual_drop))
    omission_rate = _metric_float(metrics, "omission_rate")
    if omission_rate is None and attempted > 0.0:
        omissions = _metric_float(metrics, "omission_count", "missed_audio_commands")
        if omissions is not None:
            omission_rate = float(omissions) / float(attempted)
    base = _base_performance(metrics)
    return _weighted_average(
        (
            (recall, 0.25),
            (primary, 0.15),
            (secondary, 0.15),
            (base, 0.22),
            (dual_quality, 0.08),
            (stability_score, 0.07),
            (1.0 - (omission_rate or timeout_penalty), 0.05),
            (1.0 - false_alarm_penalty, 0.03),
        )
    ) or 0.65


def _confidence(
    metrics: dict[str, str],
    *,
    attempted: float,
    family_id: DifficultyFamilyId,
) -> float:
    core = sum(
        1
        for key in ("score_ratio", "accuracy", "mean_rt_ms", "timeout_rate")
        if _metric_float(metrics, key) is not None
    )
    family_bonus = 0
    if family_id in _PSYCHOMOTOR_FAMILIES:
        family_bonus = sum(
            1
            for key in ("tracking_error_mean", "time_on_target_ratio", "overshoot_count", "control_reversal_count")
            if _metric_float(metrics, key) is not None
        )
    elif family_id in _SPATIAL_FAMILIES:
        family_bonus = sum(
            1
            for key in ("angular_error_deg", "orientation_correct", "position_error", "view_integration_error")
            if _metric_float(metrics, key) is not None
        )
    elif family_id in _MEMORY_FAMILIES:
        family_bonus = sum(
            1
            for key in ("memory_recall_accuracy", "primary_task_accuracy", "secondary_task_accuracy", "dual_task_drop")
            if _metric_float(metrics, key) is not None
        )
    volume = _clamp(float(attempted) / 20.0) if attempted > 0.0 else 0.0
    coverage = _clamp((float(core) + float(family_bonus)) / 7.0)
    return _clamp((0.58 * volume) + (0.42 * coverage))


def _recommended_target(
    *,
    family_id: DifficultyFamilyId,
    mastery_score: float,
    speed_score: float,
    stability_score: float,
    fatigue_drop: float,
    timeout_penalty: float,
    false_alarm_penalty: float,
) -> str:
    if timeout_penalty >= 0.20:
        return "timeout_control"
    if false_alarm_penalty >= 0.15:
        return "false_alarm_control"
    if family_id in _PSYCHOMOTOR_FAMILIES and mastery_score < 0.70:
        return "control_quality"
    if family_id in _SPATIAL_FAMILIES and mastery_score < 0.70:
        return "spatial_accuracy"
    if fatigue_drop >= 0.18:
        return "fatigue_stability"
    if stability_score < 0.68:
        return "response_stability"
    if speed_score < 0.55:
        return "speed_and_throughput"
    if mastery_score < 0.70:
        return "accuracy"
    return "learning_zone"
