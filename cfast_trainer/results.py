from __future__ import annotations

import math
from dataclasses import dataclass, fields, is_dataclass

from .cognitive_core import Phase, QuestionEvent, TimedTextInputTest


@dataclass(frozen=True, slots=True)
class AttemptResult:
    """Persistable summary + metric bundle for a completed scored attempt."""

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

    metrics: dict[str, str]
    events: list[QuestionEvent]


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


def _question_events_from_engine(engine: object) -> list[QuestionEvent]:
    getter = getattr(engine, "events", None)
    if not callable(getter):
        return []
    try:
        raw_events = getter()
    except Exception:
        return []
    if not isinstance(raw_events, (list, tuple)):
        return []
    return [
        event
        for event in raw_events
        if isinstance(event, QuestionEvent) and event.phase is Phase.SCORED
    ]


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
    }

    metrics: dict[str, str] = {}
    for field in fields(summary):
        if field.name in reserved:
            continue
        value = _format_metric_value(getattr(summary, field.name, None))
        if value is not None:
            metrics[field.name] = value
    return metrics


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
    scored_events = _question_events_from_engine(engine)
    rts_ms = sorted(int(round(e.response_time_s * 1000.0)) for e in scored_events)

    mean_ms: float | None
    median_ms: float | None
    if not rts_ms:
        mean_rt_s = _optional_float(getattr(summary, "mean_response_time_s", None))
        mean_ms = None if mean_rt_s is None else mean_rt_s * 1000.0
        median_ms = None
    else:
        mean_ms = float(sum(rts_ms)) / float(len(rts_ms))
        mid = len(rts_ms) // 2
        if len(rts_ms) % 2 == 1:
            median_ms = float(rts_ms[mid])
        else:
            median_ms = float(rts_ms[mid - 1] + rts_ms[mid]) / 2.0

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

    metrics = {
        "attempted": str(attempted),
        "correct": str(correct),
        "accuracy": f"{accuracy:.6f}",
        "duration_s": f"{duration_s:.6f}",
        "throughput_per_min": f"{throughput_per_min:.6f}",
        "mean_rt_ms": "" if mean_ms is None else f"{mean_ms:.3f}",
        "median_rt_ms": "" if median_ms is None else f"{median_ms:.3f}",
        "total_score": "" if total_score is None else f"{total_score:.6f}",
        "max_score": "" if max_score is None else f"{max_score:.6f}",
        "score_ratio": "" if score_ratio is None else f"{score_ratio:.6f}",
    }
    metrics.update(_extra_summary_metrics(summary))

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
        mean_rt_ms=mean_ms,
        median_rt_ms=median_ms,
        total_score=total_score,
        max_score=max_score,
        score_ratio=score_ratio,
        metrics=metrics,
        events=scored_events,
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
