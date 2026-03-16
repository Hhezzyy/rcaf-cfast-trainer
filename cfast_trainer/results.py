from __future__ import annotations

import math
from dataclasses import dataclass, fields, is_dataclass

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
    getter = getattr(engine, "result_metrics", None)
    if not callable(getter):
        return {}
    try:
        raw = getter()
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}

    metrics: dict[str, str] = {}
    for key, raw_value in raw.items():
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
    analytics = telemetry_analytics_from_events(
        events,
        duration_s=scored_duration_s,
        is_complete=_engine_phase(engine) is Phase.RESULTS,
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
        "mean_rt_ms": "" if analytics.mean_rt_ms is None else f"{analytics.mean_rt_ms:.3f}",
        "median_rt_ms": "" if analytics.median_rt_ms is None else f"{analytics.median_rt_ms:.3f}",
        "total_score": "" if total_score is None else f"{total_score:.6f}",
        "max_score": "" if max_score is None else f"{max_score:.6f}",
        "score_ratio": "" if score_ratio is None else f"{score_ratio:.6f}",
    }
    metrics.update(_extra_summary_metrics(summary))
    metrics.update(analytics.as_metric_strings())
    metrics.update(_extra_engine_metrics(engine))
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
