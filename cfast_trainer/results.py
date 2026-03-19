from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, fields, is_dataclass

from .adaptive_difficulty import difficulty_profile_for_code, family_id_for_code
from .cognitive_core import Phase, TimedTextInputTest
from .telemetry import (
    TelemetryEvent,
    difficulty_level_summary_from_engine,
    telemetry_analytics_from_events,
    telemetry_events_from_engine,
)


@dataclass(frozen=True, slots=True)
class AttemptResult:
    """Persistable summary + metric bundle for a completed or partial activity."""

    test_code: str
    test_version: int
    seed: int
    difficulty: float
    practice_questions: int
    scored_duration_s: float
    duration_s: float

    attempted: int
    correct: int
    accuracy: float
    throughput_per_min: float
    mean_rt_ms: float | None
    median_rt_ms: float | None
    total_score: float | None
    max_score: float | None
    score_ratio: float | None
    difficulty_level_start: int | None
    difficulty_level_end: int | None

    metrics: dict[str, str]
    events: list[TelemetryEvent]


_DIFFICULTY_AXIS_NAMES = (
    "content_complexity",
    "time_pressure",
    "distractor_density",
    "multitask_concurrency",
    "memory_span_delay",
    "switch_frequency",
    "control_sensitivity",
    "spatial_ambiguity",
    "source_integration_depth",
)

_OPTIONAL_METRIC_KEYS = (
    "arithmetic_error_type",
    "distractor_capture_count",
    "revision_count",
    "intrusion_count",
    "omission_count",
    "order_error_count",
    "rms_tracking_error",
    "overshoot_count",
    "reversal_count",
    "switch_cost_ms",
    "false_command_rate",
)

_COMPOSITE_DIFFICULTY_OVERRIDE_CODES = {
    "benchmark_battery",
    "adaptive_session",
    "adaptive_session_short",
    "adaptive_session_micro",
}
_TRUTHY_COMPLETED_TOKENS = {"1", "true", "True"}


def _lookup_attr_path(obj: object, path: str) -> object | None:
    cur = obj
    for part in path.split("."):
        if cur is None or not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def _coerce_int(value: object, *, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _coerce_float(value: object, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _extract_engine_int(engine: object, *paths: str, default: int = 0) -> int:
    for path in paths:
        value = _lookup_attr_path(engine, path)
        if value is not None:
            return _coerce_int(value, default=default)
    return int(default)


def _extract_engine_float(engine: object, *paths: str, default: float = 0.0) -> float:
    for path in paths:
        value = _lookup_attr_path(engine, path)
        if value is not None:
            return _coerce_float(value, default=default)
    return float(default)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return float(out)


def _format_metric_value(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return f"{float(value):.6f}"
    if isinstance(value, str):
        token = value.strip()
        return token if token != "" else None
    return None


def _extra_summary_metrics(summary: object) -> dict[str, str]:
    if not is_dataclass(summary):
        return {}

    reserved = {
        "attempted",
        "correct",
        "accuracy",
        "duration_s",
        "throughput_per_min",
        "mean_response_time_s",
        "total_score",
        "max_score",
        "score_ratio",
        "difficulty_level",
        "difficulty_level_start",
        "difficulty_level_end",
    }

    metrics: dict[str, str] = {}
    for field in fields(summary):
        if field.name in reserved:
            continue
        value = _format_metric_value(getattr(summary, field.name, None))
        if value is not None:
            metrics[field.name] = value
    return metrics


def _extra_engine_metrics(engine: object) -> dict[str, str]:
    metrics: dict[str, str] = {}
    getter = getattr(engine, "result_metrics", None)
    if not callable(getter):
        raw = {}
    else:
        try:
            raw = getter()
        except Exception:
            raw = {}
    if isinstance(raw, dict):
        for key, raw_value in raw.items():
            if isinstance(raw_value, str):
                metrics[str(key)] = raw_value
                continue
            value = _format_metric_value(raw_value)
            if value is not None:
                metrics[str(key)] = value

    raw_overrides = getattr(engine, "_result_metrics_overrides", None)
    if isinstance(raw_overrides, dict):
        for key, raw_value in raw_overrides.items():
            if isinstance(raw_value, str):
                metrics[str(key)] = raw_value
                continue
            value = _format_metric_value(raw_value)
            if value is not None:
                metrics[str(key)] = value
    return metrics


def _training_mode_metric(engine: object, summary: object) -> str | None:
    for value in (
        _lookup_attr_path(engine, "_mode"),
        _lookup_attr_path(engine, "mode"),
        getattr(summary, "mode", None),
    ):
        raw = getattr(value, "value", value)
        token = str(raw).strip().lower() if raw is not None else ""
        if token != "":
            return token
    return None


def _engine_phase(engine: object) -> Phase | None:
    phase = _lookup_attr_path(engine, "phase")
    if isinstance(phase, Phase):
        return phase
    snapshot = getattr(engine, "snapshot", None)
    if not callable(snapshot):
        return None
    try:
        snap = snapshot()
    except Exception:
        return None
    snap_phase = getattr(snap, "phase", None)
    return snap_phase if isinstance(snap_phase, Phase) else None


def _fmt_float_metric(value: object) -> str:
    out = _optional_float(value)
    return "" if out is None else f"{out:.6f}"


def _fmt_int_metric(value: object) -> str:
    if value is None:
        return ""
    return str(_coerce_int(value))


def _normalized_metric_token(value: object) -> str:
    raw = getattr(value, "value", value)
    return str(raw).strip()


def _normalize_profile_mode(value: object | None) -> str:
    token = _normalized_metric_token(value).lower()
    if token in {"anchor", "fresh"}:
        return "anchor"
    if token in {"build", "fixed", "adaptive", "recovery", "workout", ""}:
        return "build"
    if token == "tempo":
        return "tempo"
    if token in {"pressure", "stress"}:
        return "pressure"
    if token == "fatigue_probe":
        return "fatigue_probe"
    return "build"


def _metric_float(metrics: dict[str, str], key: str) -> float | None:
    raw = metrics.get(key)
    if raw is None:
        return None
    token = str(raw).strip()
    if token == "":
        return None
    try:
        value = float(token)
    except Exception:
        return None
    return value if math.isfinite(value) else None


def _metric_int(metrics: dict[str, str], key: str) -> int | None:
    raw = metrics.get(key)
    if raw is None:
        return None
    token = str(raw).strip()
    if token == "":
        return None
    try:
        return int(token)
    except Exception:
        return None


def _sequence_token(value: object | None) -> str:
    if value is None:
        return ""
    return "".join(ch for ch in str(value).upper() if ch.isalnum())


def _numeric_token(value: object | None) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text == "":
        return ""
    if text.startswith("-"):
        digits = "".join(ch for ch in text[1:] if ch.isdigit())
        return f"-{digits}" if digits != "" else ""
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits


def _int_from_token(value: object | None) -> int | None:
    token = _numeric_token(value)
    if token in {"", "-"}:
        return None
    try:
        return int(token)
    except Exception:
        return None


def _lcs_length(left: str, right: str) -> int:
    if left == "" or right == "":
        return 0
    prev = [0] * (len(right) + 1)
    for left_char in left:
        current = [0]
        for index, right_char in enumerate(right, start=1):
            if left_char == right_char:
                current.append(prev[index - 1] + 1)
            else:
                current.append(max(prev[index], current[-1]))
        prev = current
    return prev[-1]


def _profile_mode_for_activity(
    *,
    engine: object,
    summary: object,
    raw_engine_metrics: dict[str, str],
) -> str:
    candidates = [
        _lookup_attr_path(engine, "_resolved_difficulty_context.mode"),
        raw_engine_metrics.get("training_mode"),
        _training_mode_metric(engine, summary),
    ]
    for candidate in candidates:
        token = _normalized_metric_token(candidate)
        if token != "":
            return _normalize_profile_mode(token)
    return "build"


def _difficulty_metric_template() -> dict[str, str]:
    metrics = {
        "difficulty_family_id": "",
        "difficulty_profile_level": "",
        "difficulty_profile_mode": "",
        "difficulty_profile_level_start": "",
        "difficulty_profile_level_end": "",
        "difficulty_profile_mode_start": "",
        "difficulty_profile_mode_end": "",
    }
    for axis in _DIFFICULTY_AXIS_NAMES:
        metrics[f"difficulty_axis_{axis}"] = ""
        metrics[f"difficulty_axis_{axis}_start"] = ""
        metrics[f"difficulty_axis_{axis}_end"] = ""
    return metrics


def _completed_child_groups(
    *,
    test_code: str,
    raw_engine_metrics: dict[str, str],
) -> list[dict[str, str]]:
    prefix = "probe." if test_code == "benchmark_battery" else "block."
    groups: dict[str, dict[str, str]] = {}
    for key, value in raw_engine_metrics.items():
        if not key.startswith(prefix):
            continue
        remainder = key[len(prefix) :]
        group_key, sep, subkey = remainder.partition(".")
        if not sep or group_key == "":
            continue
        groups.setdefault(group_key, {})[subkey] = str(value)

    ordered: list[tuple[int, str, dict[str, str]]] = []
    for group_key, metrics in groups.items():
        if metrics.get("completed", "") not in _TRUTHY_COMPLETED_TOKENS:
            continue
        if prefix == "probe.":
            order = _metric_int(metrics, "index")
            ordered.append((10_000 if order is None else int(order), group_key, metrics))
            continue
        ordered.append((_coerce_int(group_key, default=10_000), group_key, metrics))

    ordered.sort(key=lambda item: (item[0], item[1]))
    return [metrics for _order, _group_key, metrics in ordered]


def _copy_profile_fields(
    *,
    source: dict[str, str],
    target: dict[str, str],
    suffix: str,
) -> None:
    level_key = f"difficulty_profile_level_{suffix}"
    mode_key = f"difficulty_profile_mode_{suffix}"
    target[level_key] = (
        source.get(level_key, "")
        or source.get("difficulty_profile_level", "")
    )
    target[mode_key] = (
        source.get(mode_key, "")
        or source.get("difficulty_profile_mode", "")
    )
    for axis in _DIFFICULTY_AXIS_NAMES:
        axis_key = f"difficulty_axis_{axis}_{suffix}"
        target[axis_key] = (
            source.get(axis_key, "")
            or source.get(f"difficulty_axis_{axis}", "")
        )


def _difficulty_profile_metrics(
    *,
    test_code: str,
    engine: object,
    summary: object,
    raw_engine_metrics: dict[str, str],
    difficulty_level_start: int | None,
    difficulty_level_end: int | None,
) -> dict[str, str]:
    metrics = _difficulty_metric_template()
    if test_code in _COMPOSITE_DIFFICULTY_OVERRIDE_CODES:
        children = _completed_child_groups(test_code=test_code, raw_engine_metrics=raw_engine_metrics)
        if not children:
            return metrics
        metrics["difficulty_family_id"] = str(test_code)
        _copy_profile_fields(source=children[0], target=metrics, suffix="start")
        _copy_profile_fields(source=children[-1], target=metrics, suffix="end")
    else:
        start_level = difficulty_level_start
        end_level = difficulty_level_end if difficulty_level_end is not None else difficulty_level_start
        mode = _profile_mode_for_activity(engine=engine, summary=summary, raw_engine_metrics=raw_engine_metrics)
        metrics["difficulty_family_id"] = str(family_id_for_code(test_code))
        if start_level is not None:
            profile = difficulty_profile_for_code(test_code, start_level, mode)
            metrics["difficulty_profile_level_start"] = str(int(profile.level))
            metrics["difficulty_profile_mode_start"] = str(profile.intended_use)
            for axis in _DIFFICULTY_AXIS_NAMES:
                metrics[f"difficulty_axis_{axis}_start"] = _fmt_float_metric(
                    getattr(profile.axes, axis)
                )
        if end_level is not None:
            profile = difficulty_profile_for_code(test_code, end_level, mode)
            metrics["difficulty_profile_level_end"] = str(int(profile.level))
            metrics["difficulty_profile_mode_end"] = str(profile.intended_use)
            for axis in _DIFFICULTY_AXIS_NAMES:
                metrics[f"difficulty_axis_{axis}_end"] = _fmt_float_metric(
                    getattr(profile.axes, axis)
                )

    metrics["difficulty_profile_level"] = metrics["difficulty_profile_level_end"]
    metrics["difficulty_profile_mode"] = metrics["difficulty_profile_mode_end"]
    for axis in _DIFFICULTY_AXIS_NAMES:
        metrics[f"difficulty_axis_{axis}"] = metrics[f"difficulty_axis_{axis}_end"]
    return metrics


def _task_key_for_event(event: TelemetryEvent) -> str | None:
    extra = event.extra or {}
    if event.family == "dual_task_bridge":
        cue_kind = str(extra.get("cue_kind", "")).strip().lower()
        return None if cue_kind == "" else f"dual_task_bridge:{cue_kind}"
    if event.family == "auditory":
        command_type = str(extra.get("command_type", "")).strip().lower()
        return f"auditory:{command_type or event.kind}"
    for key in ("task_kind", "question_kind", "channel", "kind"):
        token = str(extra.get(key, "")).strip().lower()
        if token != "":
            return f"{event.family}:{token}"
    return None


def _classify_arithmetic_error(events: list[TelemetryEvent], *, test_code: str) -> str:
    if test_code in _COMPOSITE_DIFFICULTY_OVERRIDE_CODES:
        return ""
    if family_id_for_code(test_code) != "quantitative":
        return ""

    labels: list[str] = []
    for event in events:
        if not event.is_scored or event.is_correct is None or bool(event.is_correct):
            continue
        if event.is_timeout:
            labels.append("timeout")
            continue
        expected = _int_from_token(event.expected)
        response = _int_from_token(event.response)
        if expected is None or response is None:
            labels.append("other")
            continue
        expected_token = _numeric_token(event.expected).lstrip("-")
        response_token = _numeric_token(event.response).lstrip("-")
        if (
            expected_token
            and response_token
            and len(expected_token) == len(response_token)
            and expected_token != response_token
            and sorted(expected_token) == sorted(response_token)
        ):
            labels.append("digit_transposition")
            continue
        if expected != 0 and response == (-expected):
            labels.append("sign_error")
            continue
        if expected != 0 and (
            response == (expected * 10)
            or (response * 10) == expected
        ):
            labels.append("place_value_shift")
            continue
        tolerance = max(2, int(round(abs(expected) * 0.05)))
        if abs(response - expected) <= tolerance:
            labels.append("near_miss")
            continue
        labels.append("other")

    if not labels:
        return ""
    counts = Counter(labels)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _sequence_error_metrics(events: list[TelemetryEvent], *, test_code: str) -> tuple[bool, int, int, int]:
    if family_id_for_code(test_code) not in {"visual_memory_updating", "cln_multitask"}:
        return False, 0, 0, 0

    applicable = False
    intrusion_count = 0
    omission_count = 0
    order_error_count = 0

    for event in events:
        if not event.is_scored or event.is_correct is None or bool(event.is_correct):
            continue
        expected = _sequence_token(event.expected)
        response = _sequence_token(event.response)
        if expected == "" or response == "" or len(expected) <= 1 or len(response) <= 1:
            continue
        if not (
            expected.isdigit() == response.isdigit()
            or expected.isalpha() == response.isalpha()
        ):
            continue
        applicable = True
        lcs = _lcs_length(expected, response)
        omission_count += max(0, len(expected) - lcs)
        intrusion_count += max(0, len(response) - lcs)
        if len(expected) == len(response) and sorted(expected) == sorted(response) and expected != response:
            order_error_count += max(1, len(expected) - lcs)

    return applicable, intrusion_count, omission_count, order_error_count


def _standardized_optional_metrics(
    *,
    engine: object,
    summary: object,
    raw_engine_metrics: dict[str, str],
    events: list[TelemetryEvent],
    test_code: str,
) -> dict[str, str]:
    metrics = {key: "" for key in _OPTIONAL_METRIC_KEYS}

    arithmetic_error_type = _classify_arithmetic_error(events, test_code=test_code)
    if arithmetic_error_type != "":
        metrics["arithmetic_error_type"] = arithmetic_error_type

    distractor_count = None
    for key in (
        "auditory.false_responses_to_distractors",
        "false_responses_to_distractors",
    ):
        distractor_count = _metric_int(raw_engine_metrics, key)
        if distractor_count is not None:
            break
    if distractor_count is not None:
        metrics["distractor_capture_count"] = str(int(distractor_count))

    revision_applicable = any(event.family == "cognitive_updating" for event in events)
    if revision_applicable:
        revision_count = 0
        for event in events:
            extra = event.extra or {}
            action = str(extra.get("action", "")).strip().lower()
            if "clear" in action:
                revision_count += 1
        metrics["revision_count"] = str(int(revision_count))

    sequence_applicable, intrusion_count, omission_count, order_error_count = _sequence_error_metrics(
        events,
        test_code=test_code,
    )
    if sequence_applicable:
        metrics["intrusion_count"] = str(int(intrusion_count))
        metrics["omission_count"] = str(int(omission_count))
        metrics["order_error_count"] = str(int(order_error_count))

    rms_error = getattr(summary, "rms_error", None)
    if rms_error is None:
        rms_error = _metric_float(raw_engine_metrics, "rms_error")
    if rms_error is not None:
        metrics["rms_tracking_error"] = _fmt_float_metric(rms_error)
        overshoot_count = getattr(summary, "overshoot_count", None)
        if overshoot_count is None:
            overshoot_count = _metric_int(raw_engine_metrics, "overshoot_count")
        reversal_count = getattr(summary, "reversal_count", None)
        if reversal_count is None:
            reversal_count = _metric_int(raw_engine_metrics, "reversal_count")
        metrics["overshoot_count"] = str(int(overshoot_count or 0))
        metrics["reversal_count"] = str(int(reversal_count or 0))

    command_attempted = _metric_int(raw_engine_metrics, "bridge.command_attempted")
    false_commands = _metric_int(raw_engine_metrics, "bridge.false_alarms")
    if command_attempted is not None:
        rate = 0.0 if command_attempted <= 0 else float(false_commands or 0) / float(command_attempted)
        metrics["false_command_rate"] = f"{rate:.6f}"

    switch_rts: list[int] = []
    repeat_rts: list[int] = []
    previous_key: str | None = None
    for event in sorted(
        (item for item in events if item.is_scored and item.response_time_ms is not None),
        key=lambda item: (item.item_index if item.item_index is not None else 10_000_000, item.seq),
    ):
        task_key = _task_key_for_event(event)
        if task_key is None:
            continue
        if previous_key is not None:
            if task_key == previous_key:
                repeat_rts.append(int(event.response_time_ms))
            else:
                switch_rts.append(int(event.response_time_ms))
        previous_key = task_key
    if switch_rts or repeat_rts:
        if switch_rts and repeat_rts:
            switch_mean = sum(switch_rts) / float(len(switch_rts))
            repeat_mean = sum(repeat_rts) / float(len(repeat_rts))
            metrics["switch_cost_ms"] = f"{(switch_mean - repeat_mean):.6f}"
        else:
            metrics["switch_cost_ms"] = ""

    return metrics


def _filtered_nonreserved_metrics(
    raw_metrics: dict[str, str],
    *,
    reserved_keys: set[str],
) -> dict[str, str]:
    return {
        key: value
        for key, value in raw_metrics.items()
        if key not in reserved_keys
    }


def attempt_result_from_engine(
    engine: object,
    *,
    test_code: str,
    test_version: int = 1,
) -> AttemptResult:
    """Build an AttemptResult from any engine exposing scored_summary()."""

    scorer = getattr(engine, "scored_summary", None)
    if not callable(scorer):
        raise TypeError("engine does not expose scored_summary()")

    summary = scorer()
    events = telemetry_events_from_engine(engine)
    raw_engine_metrics = _extra_engine_metrics(engine)
    summary_metrics = _extra_summary_metrics(summary)

    attempted = _coerce_int(getattr(summary, "attempted", 0))
    correct = _coerce_int(getattr(summary, "correct", 0))
    accuracy = _coerce_float(getattr(summary, "accuracy", 0.0))
    throughput_per_min = _coerce_float(getattr(summary, "throughput_per_min", 0.0))
    scored_duration_s = _extract_engine_float(
        engine,
        "scored_duration_s",
        "_scored_duration_s",
        "_cfg.scored_duration_s",
        "_config.scored_duration_s",
        default=_coerce_float(getattr(summary, "duration_s", 0.0)),
    )
    duration_s = _coerce_float(getattr(summary, "duration_s", scored_duration_s))
    total_score = _optional_float(getattr(summary, "total_score", None))
    max_score = _optional_float(getattr(summary, "max_score", None))
    score_ratio = _optional_float(getattr(summary, "score_ratio", None))
    if score_ratio is None and total_score is not None and max_score not in (None, 0.0):
        score_ratio = total_score / float(max_score)

    difficulty_level_start, difficulty_level_end, difficulty_change_count = (
        difficulty_level_summary_from_engine(engine, summary=summary)
    )
    is_complete = _engine_phase(engine) is Phase.RESULTS
    analytics = telemetry_analytics_from_events(
        events,
        duration_s=scored_duration_s,
        is_complete=is_complete,
        difficulty_level_start=difficulty_level_start,
        difficulty_level_end=difficulty_level_end,
        difficulty_change_count=difficulty_change_count,
    )

    metrics = {
        "attempted": str(attempted),
        "correct": str(correct),
        "accuracy": f"{accuracy:.6f}",
        "duration_s": f"{duration_s:.6f}",
        "throughput_per_min": f"{throughput_per_min:.6f}",
        "completed": "1" if is_complete else "0",
        "aborted": "0" if is_complete else "1",
        "total_score": "" if total_score is None else f"{total_score:.6f}",
        "max_score": "" if max_score is None else f"{max_score:.6f}",
        "score_ratio": "" if score_ratio is None else f"{score_ratio:.6f}",
    }
    metrics.update(analytics.as_metric_strings())
    metrics.update(
        _difficulty_profile_metrics(
            test_code=str(test_code),
            engine=engine,
            summary=summary,
            raw_engine_metrics=raw_engine_metrics,
            difficulty_level_start=difficulty_level_start,
            difficulty_level_end=difficulty_level_end,
        )
    )
    metrics.update(
        _standardized_optional_metrics(
            engine=engine,
            summary=summary,
            raw_engine_metrics=raw_engine_metrics,
            events=events,
            test_code=str(test_code),
        )
    )
    reserved_keys = set(metrics)
    metrics.update(_filtered_nonreserved_metrics(summary_metrics, reserved_keys=reserved_keys))
    metrics.update(_filtered_nonreserved_metrics(raw_engine_metrics, reserved_keys=reserved_keys))
    training_mode = _training_mode_metric(engine, summary)
    if training_mode and "training_mode" not in metrics:
        metrics["training_mode"] = training_mode

    return AttemptResult(
        test_code=str(test_code),
        test_version=int(test_version),
        seed=_extract_engine_int(engine, "seed", "_seed"),
        difficulty=_extract_engine_float(engine, "difficulty", "_difficulty"),
        practice_questions=_extract_engine_int(
            engine,
            "practice_questions",
            "_practice_questions",
            "_cfg.practice_questions",
            "_config.practice_questions",
        ),
        scored_duration_s=float(scored_duration_s),
        duration_s=float(duration_s),
        attempted=attempted,
        correct=correct,
        accuracy=accuracy,
        throughput_per_min=throughput_per_min,
        mean_rt_ms=analytics.mean_rt_ms,
        median_rt_ms=analytics.median_rt_ms,
        total_score=total_score,
        max_score=max_score,
        score_ratio=score_ratio,
        difficulty_level_start=difficulty_level_start,
        difficulty_level_end=difficulty_level_end,
        metrics=metrics,
        events=events,
    )


def attempt_result_from_timed_test(
    test: TimedTextInputTest,
    *,
    test_code: str,
    test_version: int = 1,
) -> AttemptResult:
    """Compatibility wrapper for TimedTextInputTest callers."""

    return attempt_result_from_engine(
        test,
        test_code=test_code,
        test_version=test_version,
    )
