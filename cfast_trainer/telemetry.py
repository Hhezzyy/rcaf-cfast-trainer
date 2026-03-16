from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median
from typing import TYPE_CHECKING

from .cognitive_core import Phase, QuestionEvent

if TYPE_CHECKING:
    from .ant_drills import AntDifficultyChange
    from .auditory_capacity import AuditoryCapacityEvent
    from .cognitive_updating import CognitiveUpdatingActionEvent


_TIMEOUT_RAW_TOKENS = {"__timeout__", "timeout", "timed_out", "time_out"}


@dataclass(frozen=True, slots=True)
class TelemetryEvent:
    family: str
    kind: str
    phase: str
    seq: int
    item_index: int | None
    is_scored: bool
    is_correct: bool | None
    is_timeout: bool
    response_time_ms: int | None
    score: float | None
    max_score: float | None
    difficulty_level: int | None
    occurred_at_ms: int | None
    prompt: str | None = None
    expected: str | None = None
    response: str | None = None
    extra: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class TelemetryAnalytics:
    mean_rt_ms: float | None
    median_rt_ms: float | None
    rt_variance_ms2: float | None
    timeout_count: int
    timeout_rate: float
    longest_lapse_streak: int
    first_half_attempted: int
    first_half_accuracy: float | None
    first_half_mean_rt_ms: float | None
    first_half_timeout_rate: float | None
    second_half_attempted: int
    second_half_accuracy: float | None
    second_half_mean_rt_ms: float | None
    second_half_timeout_rate: float | None
    half_accuracy_drop: float | None
    half_mean_rt_inflation_ms: float | None
    first_3m_attempted: int
    first_3m_accuracy: float | None
    first_3m_mean_rt_ms: float | None
    first_3m_timeout_rate: float | None
    last_3m_attempted: int
    last_3m_accuracy: float | None
    last_3m_mean_rt_ms: float | None
    last_3m_timeout_rate: float | None
    post_error_next_item_mean_rt_ms: float | None
    post_correct_next_item_mean_rt_ms: float | None
    post_error_next_item_rt_inflation_ms: float | None
    difficulty_level_start: int | None
    difficulty_level_end: int | None
    difficulty_change_count: int

    def as_metric_strings(self) -> dict[str, str]:
        def _fmt_float(value: float | None) -> str:
            return "" if value is None or not math.isfinite(value) else f"{value:.6f}"

        def _fmt_int(value: int | None) -> str:
            return "" if value is None else str(int(value))

        return {
            "mean_rt_ms": _fmt_float(self.mean_rt_ms),
            "median_rt_ms": _fmt_float(self.median_rt_ms),
            "rt_variance_ms2": _fmt_float(self.rt_variance_ms2),
            "timeout_count": str(self.timeout_count),
            "timeout_rate": _fmt_float(self.timeout_rate),
            "longest_lapse_streak": str(self.longest_lapse_streak),
            "first_half_attempted": str(self.first_half_attempted),
            "first_half_accuracy": _fmt_float(self.first_half_accuracy),
            "first_half_mean_rt_ms": _fmt_float(self.first_half_mean_rt_ms),
            "first_half_timeout_rate": _fmt_float(self.first_half_timeout_rate),
            "second_half_attempted": str(self.second_half_attempted),
            "second_half_accuracy": _fmt_float(self.second_half_accuracy),
            "second_half_mean_rt_ms": _fmt_float(self.second_half_mean_rt_ms),
            "second_half_timeout_rate": _fmt_float(self.second_half_timeout_rate),
            "half_accuracy_drop": _fmt_float(self.half_accuracy_drop),
            "half_mean_rt_inflation_ms": _fmt_float(self.half_mean_rt_inflation_ms),
            "first_3m_attempted": str(self.first_3m_attempted),
            "first_3m_accuracy": _fmt_float(self.first_3m_accuracy),
            "first_3m_mean_rt_ms": _fmt_float(self.first_3m_mean_rt_ms),
            "first_3m_timeout_rate": _fmt_float(self.first_3m_timeout_rate),
            "last_3m_attempted": str(self.last_3m_attempted),
            "last_3m_accuracy": _fmt_float(self.last_3m_accuracy),
            "last_3m_mean_rt_ms": _fmt_float(self.last_3m_mean_rt_ms),
            "last_3m_timeout_rate": _fmt_float(self.last_3m_timeout_rate),
            "post_error_next_item_mean_rt_ms": _fmt_float(self.post_error_next_item_mean_rt_ms),
            "post_correct_next_item_mean_rt_ms": _fmt_float(
                self.post_correct_next_item_mean_rt_ms
            ),
            "post_error_next_item_rt_inflation_ms": _fmt_float(
                self.post_error_next_item_rt_inflation_ms
            ),
            "difficulty_level_start": _fmt_int(self.difficulty_level_start),
            "difficulty_level_end": _fmt_int(self.difficulty_level_end),
            "difficulty_change_count": str(self.difficulty_change_count),
        }


def difficulty_level_from_ratio(value: object, *, default: int | None = None) -> int | None:
    try:
        ratio = float(value)
    except Exception:
        return default
    if not math.isfinite(ratio):
        return default
    clamped = 0.0 if ratio <= 0.0 else 1.0 if ratio >= 1.0 else float(ratio)
    return max(1, min(10, int(round(clamped * 9.0)) + 1))


def telemetry_events_from_engine(engine: object) -> list[TelemetryEvent]:
    sequences: list[list[TelemetryEvent]] = []
    seq_offset = 0
    for raw_events in _event_sources_from_engine(engine):
        normalized = _normalize_event_source(
            engine=engine,
            raw_events=raw_events,
            seq_offset=seq_offset,
        )
        if normalized:
            sequences.append(normalized)
            seq_offset += len(normalized)

    difficulty_changes = _difficulty_changes_from_engine(engine)
    if difficulty_changes:
        sequences.append(
            _normalize_difficulty_changes(
                changes=difficulty_changes,
                seq_offset=seq_offset,
            )
        )

    merged: list[TelemetryEvent] = []
    for chunk in sequences:
        merged.extend(chunk)
    return merged


def telemetry_analytics_from_events(
    events: list[TelemetryEvent],
    *,
    duration_s: float | None,
    is_complete: bool,
    difficulty_level_start: int | None,
    difficulty_level_end: int | None,
    difficulty_change_count: int,
) -> TelemetryAnalytics:
    scored_items = sorted(
        (
            event
            for event in events
            if event.is_scored and event.item_index is not None and event.is_correct is not None
        ),
        key=lambda event: (event.item_index or 0, event.seq),
    )
    rt_values = [int(event.response_time_ms) for event in scored_items if event.response_time_ms is not None]

    mean_rt_ms = _mean(rt_values)
    median_rt_ms = None if not rt_values else float(median(rt_values))
    rt_variance_ms2 = _population_variance(rt_values)
    timeout_count = sum(1 for event in scored_items if event.is_timeout)
    timeout_rate = 0.0 if not scored_items else float(timeout_count) / float(len(scored_items))
    longest_lapse_streak = _longest_lapse_streak(scored_items)

    observed_duration_ms = max(
        (int(event.occurred_at_ms) for event in scored_items if event.occurred_at_ms is not None),
        default=0,
    )
    duration_ms = observed_duration_ms
    if is_complete and duration_s is not None and duration_s > 0.0:
        duration_ms = max(duration_ms, int(round(float(duration_s) * 1000.0)))

    midpoint_ms = max(0, duration_ms // 2)
    first_half = _half_window_events(scored_items, start_ms=0, end_ms=midpoint_ms, include_end=False)
    second_half = _half_window_events(
        scored_items,
        start_ms=midpoint_ms,
        end_ms=duration_ms,
        include_end=True,
    )
    first_half_stats = _window_stats(first_half)
    second_half_stats = _window_stats(second_half)

    first_window = _window_events(scored_items, start_ms=0, end_ms=min(180_000, duration_ms))
    last_window_start_ms = max(0, duration_ms - 180_000)
    last_window = _window_events(scored_items, start_ms=last_window_start_ms, end_ms=duration_ms)

    first_stats = _window_stats(first_window)
    last_stats = _window_stats(last_window)

    post_error_rts: list[int] = []
    post_correct_rts: list[int] = []
    previous: TelemetryEvent | None = None
    for event in scored_items:
        if previous is None or event.response_time_ms is None:
            previous = event
            continue
        if bool(previous.is_correct):
            post_correct_rts.append(int(event.response_time_ms))
        else:
            post_error_rts.append(int(event.response_time_ms))
        previous = event

    post_error_mean = _mean(post_error_rts)
    post_correct_mean = _mean(post_correct_rts)
    post_error_inflation = None
    if post_error_mean is not None and post_correct_mean is not None:
        post_error_inflation = post_error_mean - post_correct_mean

    half_accuracy_drop = None
    if first_half_stats["accuracy"] is not None and second_half_stats["accuracy"] is not None:
        half_accuracy_drop = float(first_half_stats["accuracy"]) - float(second_half_stats["accuracy"])
    half_mean_rt_inflation = None
    if first_half_stats["mean_rt_ms"] is not None and second_half_stats["mean_rt_ms"] is not None:
        half_mean_rt_inflation = float(second_half_stats["mean_rt_ms"]) - float(
            first_half_stats["mean_rt_ms"]
        )

    return TelemetryAnalytics(
        mean_rt_ms=mean_rt_ms,
        median_rt_ms=median_rt_ms,
        rt_variance_ms2=rt_variance_ms2,
        timeout_count=timeout_count,
        timeout_rate=timeout_rate,
        longest_lapse_streak=longest_lapse_streak,
        first_half_attempted=first_half_stats["attempted"],
        first_half_accuracy=first_half_stats["accuracy"],
        first_half_mean_rt_ms=first_half_stats["mean_rt_ms"],
        first_half_timeout_rate=first_half_stats["timeout_rate"],
        second_half_attempted=second_half_stats["attempted"],
        second_half_accuracy=second_half_stats["accuracy"],
        second_half_mean_rt_ms=second_half_stats["mean_rt_ms"],
        second_half_timeout_rate=second_half_stats["timeout_rate"],
        half_accuracy_drop=half_accuracy_drop,
        half_mean_rt_inflation_ms=half_mean_rt_inflation,
        first_3m_attempted=first_stats["attempted"],
        first_3m_accuracy=first_stats["accuracy"],
        first_3m_mean_rt_ms=first_stats["mean_rt_ms"],
        first_3m_timeout_rate=first_stats["timeout_rate"],
        last_3m_attempted=last_stats["attempted"],
        last_3m_accuracy=last_stats["accuracy"],
        last_3m_mean_rt_ms=last_stats["mean_rt_ms"],
        last_3m_timeout_rate=last_stats["timeout_rate"],
        post_error_next_item_mean_rt_ms=post_error_mean,
        post_correct_next_item_mean_rt_ms=post_correct_mean,
        post_error_next_item_rt_inflation_ms=post_error_inflation,
        difficulty_level_start=difficulty_level_start,
        difficulty_level_end=difficulty_level_end,
        difficulty_change_count=difficulty_change_count,
    )


def difficulty_level_summary_from_engine(engine: object, *, summary: object | None) -> tuple[int | None, int | None, int]:
    start = _lookup_optional_int(summary, "difficulty_level_start")
    end = _lookup_optional_int(summary, "difficulty_level_end")
    current = _lookup_optional_int(summary, "difficulty_level")
    summary_change_count = _lookup_optional_int(summary, "difficulty_change_count")
    if start is None:
        start = current
    if end is None:
        end = current

    changes = _difficulty_changes_from_engine(engine)
    if changes:
        if start is None:
            start = int(changes[0].old_level)
        if end is None:
            end = int(changes[-1].new_level)
        return start, end, len(changes)

    if summary_change_count is not None:
        if start is None:
            start = difficulty_level_from_ratio(
                _lookup_attr_path(engine, "_difficulty"),
                default=difficulty_level_from_ratio(_lookup_attr_path(engine, "difficulty")),
            )
        if end is None:
            end = start
        return start, end, max(0, int(summary_change_count))

    if start is None:
        start = difficulty_level_from_ratio(
            _lookup_attr_path(engine, "_difficulty"),
            default=difficulty_level_from_ratio(_lookup_attr_path(engine, "difficulty")),
        )
    if end is None:
        end = start
    return start, end, 0


def lifecycle_event(
    *,
    family: str,
    kind: str,
    seq: int,
    phase: str = "",
    occurred_at_ms: int | None = None,
    extra: dict[str, object] | None = None,
) -> TelemetryEvent:
    return TelemetryEvent(
        family=family,
        kind=kind,
        phase=str(phase),
        seq=int(seq),
        item_index=None,
        is_scored=False,
        is_correct=None,
        is_timeout=False,
        response_time_ms=None,
        score=None,
        max_score=None,
        difficulty_level=None,
        occurred_at_ms=occurred_at_ms,
        extra=extra,
    )


def _event_sources_from_engine(engine: object) -> list[object]:
    sources: list[object] = []
    getter = getattr(engine, "events", None)
    if callable(getter):
        try:
            value = getter()
        except Exception:
            value = None
        if value:
            sources.append(value)

    extra = getattr(engine, "_telemetry_runtime_events", None)
    if callable(extra):
        try:
            extra = extra()
        except Exception:
            extra = None
    if extra:
        sources.append(extra)
    return sources


def _normalize_event_source(
    *,
    engine: object,
    raw_events: object,
    seq_offset: int,
) -> list[TelemetryEvent]:
    if not isinstance(raw_events, (list, tuple)):
        return []
    if not raw_events:
        return []

    first = raw_events[0]
    if isinstance(first, QuestionEvent):
        return _normalize_question_events(engine=engine, events=raw_events, seq_offset=seq_offset)

    event_type_name = type(first).__name__
    if event_type_name == "AuditoryCapacityEvent":
        return _normalize_auditory_events(events=raw_events, seq_offset=seq_offset)
    if event_type_name == "CognitiveUpdatingActionEvent":
        return _normalize_cognitive_updating_events(events=raw_events, seq_offset=seq_offset)
    if isinstance(first, TelemetryEvent):
        return [
            TelemetryEvent(
                family=event.family,
                kind=event.kind,
                phase=event.phase,
                seq=seq_offset + offset,
                item_index=event.item_index,
                is_scored=event.is_scored,
                is_correct=event.is_correct,
                is_timeout=event.is_timeout,
                response_time_ms=event.response_time_ms,
                score=event.score,
                max_score=event.max_score,
                difficulty_level=event.difficulty_level,
                occurred_at_ms=event.occurred_at_ms,
                prompt=event.prompt,
                expected=event.expected,
                response=event.response,
                extra=event.extra,
            )
            for offset, event in enumerate(raw_events)
        ]
    return []


def _normalize_question_events(
    *,
    engine: object,
    events: list[QuestionEvent] | tuple[QuestionEvent, ...],
    seq_offset: int,
) -> list[TelemetryEvent]:
    difficulty_changes = _difficulty_changes_from_engine(engine)
    start_level, _, _ = difficulty_level_summary_from_engine(engine, summary=None)
    phase_start_s = _phase_started_at_s(engine)
    normalized: list[TelemetryEvent] = []
    for offset, event in enumerate(events):
        presented_at_s = float(event.presented_at_s)
        answered_at_s = float(event.answered_at_s)
        if (
            event.phase is Phase.SCORED
            and phase_start_s is not None
            and presented_at_s >= (phase_start_s - 1e-9)
            and answered_at_s >= (phase_start_s - 1e-9)
        ):
            presented_at_s -= phase_start_s
            answered_at_s -= phase_start_s

        response_text = str(event.raw).strip() or str(event.user_answer)
        is_timeout = bool(
            getattr(event, "is_timeout", False)
            or str(event.raw).strip().lower() in _TIMEOUT_RAW_TOKENS
        )
        extra = None
        metadata = getattr(event, "content_metadata", None)
        if isinstance(metadata, dict) and metadata:
            extra = dict(metadata)
        normalized.append(
            TelemetryEvent(
                family="question",
                kind="question",
                phase=str(event.phase.value),
                seq=seq_offset + offset,
                item_index=int(event.index),
                is_scored=event.phase is Phase.SCORED,
                is_correct=bool(event.is_correct),
                is_timeout=is_timeout,
                response_time_ms=int(round(float(event.response_time_s) * 1000.0)),
                score=float(event.score),
                max_score=float(event.max_score),
                difficulty_level=_difficulty_level_for_item(
                    item_index=int(event.index),
                    start_level=start_level,
                    changes=difficulty_changes,
                ),
                occurred_at_ms=int(round(answered_at_s * 1000.0)),
                prompt=str(event.prompt),
                expected=str(event.correct_answer),
                response=response_text,
                extra=extra,
            )
        )
    return normalized


def _normalize_auditory_events(
    *,
    events: list[AuditoryCapacityEvent] | tuple[AuditoryCapacityEvent, ...],
    seq_offset: int,
) -> list[TelemetryEvent]:
    normalized: list[TelemetryEvent] = []
    item_index = 0
    for offset, event in enumerate(events):
        is_scored = str(event.phase) == str(Phase.SCORED)
        current_item_index = item_index
        item_index += 1
        occurred_at_ms = None
        if getattr(event, "occurred_at_s", None) is not None:
            occurred_at_ms = int(round(float(event.occurred_at_s) * 1000.0))
        rt = getattr(event, "response_time_s", None)
        extra: dict[str, object] = {}
        if getattr(event, "command_type", None) is not None:
            extra["command_type"] = str(event.command_type)
        if getattr(event, "addressed_call_sign", None) is not None:
            extra["addressed_call_sign"] = str(event.addressed_call_sign)
        normalized.append(
            TelemetryEvent(
                family="auditory",
                kind=str(event.kind.value),
                phase=str(event.phase.value),
                seq=seq_offset + offset,
                item_index=current_item_index,
                is_scored=is_scored,
                is_correct=bool(event.is_correct),
                is_timeout=False,
                response_time_ms=None if rt is None else int(round(float(rt) * 1000.0)),
                score=float(event.score),
                max_score=1.0,
                difficulty_level=None,
                occurred_at_ms=occurred_at_ms,
                expected=str(event.expected),
                response=str(event.response),
                extra=extra or None,
            )
        )
    return normalized


def _normalize_cognitive_updating_events(
    *,
    events: list[CognitiveUpdatingActionEvent] | tuple[CognitiveUpdatingActionEvent, ...],
    seq_offset: int,
) -> list[TelemetryEvent]:
    normalized: list[TelemetryEvent] = []
    for offset, event in enumerate(events):
        normalized.append(
            TelemetryEvent(
                family="cognitive_updating",
                kind="action",
                phase=Phase.SCORED.value,
                seq=seq_offset + offset,
                item_index=None,
                is_scored=False,
                is_correct=None,
                is_timeout=False,
                response_time_ms=None,
                score=None,
                max_score=None,
                difficulty_level=None,
                occurred_at_ms=int(round(float(event.at_s) * 1000.0)),
                response=str(event.value),
                extra={"action": str(event.action)},
            )
        )
    return normalized


def _normalize_difficulty_changes(
    *,
    changes: list[AntDifficultyChange],
    seq_offset: int,
) -> list[TelemetryEvent]:
    return [
        TelemetryEvent(
            family="adaptive",
            kind="difficulty_change",
            phase=Phase.SCORED.value,
            seq=seq_offset + offset,
            item_index=None,
            is_scored=False,
            is_correct=None,
            is_timeout=False,
            response_time_ms=None,
            score=None,
            max_score=None,
            difficulty_level=int(change.new_level),
            occurred_at_ms=None,
            extra={
                "after_attempt": int(change.after_attempt),
                "old_level": int(change.old_level),
                "new_level": int(change.new_level),
                "reason": str(change.reason),
            },
        )
        for offset, change in enumerate(changes)
    ]


def _difficulty_changes_from_engine(engine: object) -> list[AntDifficultyChange]:
    getter = getattr(engine, "difficulty_changes", None)
    if not callable(getter):
        return []
    try:
        raw = getter()
    except Exception:
        return []
    if not isinstance(raw, (list, tuple)):
        return []
    return [change for change in raw if type(change).__name__ == "AntDifficultyChange"]


def _difficulty_level_for_item(
    *,
    item_index: int,
    start_level: int | None,
    changes: list[AntDifficultyChange],
) -> int | None:
    if start_level is None and not changes:
        return None
    level = start_level
    for change in changes:
        if (item_index + 1) <= int(change.after_attempt):
            break
        level = int(change.new_level)
    return level


def _phase_started_at_s(engine: object) -> float | None:
    for path in ("_scored_started_at_s", "_phase_started_at_s"):
        value = _lookup_attr_path(engine, path)
        try:
            if value is None:
                continue
            out = float(value)
        except Exception:
            continue
        if math.isfinite(out):
            return out
    return None


def _window_events(
    events: list[TelemetryEvent],
    *,
    start_ms: int,
    end_ms: int,
) -> list[TelemetryEvent]:
    if end_ms <= start_ms:
        return []
    return [
        event
        for event in events
        if event.occurred_at_ms is not None and start_ms <= int(event.occurred_at_ms) <= end_ms
    ]


def _window_stats(events: list[TelemetryEvent]) -> dict[str, float | int | None]:
    attempted = len(events)
    if attempted <= 0:
        return {
            "attempted": 0,
            "accuracy": None,
            "mean_rt_ms": None,
            "timeout_rate": None,
        }

    correct = sum(1 for event in events if bool(event.is_correct))
    timeout_count = sum(1 for event in events if event.is_timeout)
    rts = [int(event.response_time_ms) for event in events if event.response_time_ms is not None]
    return {
        "attempted": attempted,
        "accuracy": float(correct) / float(attempted),
        "mean_rt_ms": _mean(rts),
        "timeout_rate": float(timeout_count) / float(attempted),
    }


def _half_window_events(
    events: list[TelemetryEvent],
    *,
    start_ms: int,
    end_ms: int,
    include_end: bool,
) -> list[TelemetryEvent]:
    if end_ms <= start_ms:
        return []
    if include_end:
        return [
            event
            for event in events
            if event.occurred_at_ms is not None and start_ms <= int(event.occurred_at_ms) <= end_ms
        ]
    return [
        event
        for event in events
        if event.occurred_at_ms is not None and start_ms <= int(event.occurred_at_ms) < end_ms
    ]


def _longest_lapse_streak(events: list[TelemetryEvent]) -> int:
    best = 0
    current = 0
    for event in events:
        if bool(event.is_correct):
            current = 0
            continue
        current += 1
        if current > best:
            best = current
    return best


def _mean(values: list[int]) -> float | None:
    if not values:
        return None
    return float(sum(values)) / float(len(values))


def _population_variance(values: list[int]) -> float | None:
    if not values:
        return None
    mean_value = float(sum(values)) / float(len(values))
    return float(sum((float(value) - mean_value) ** 2 for value in values)) / float(len(values))


def _lookup_optional_int(obj: object, path: str) -> int | None:
    value = _lookup_attr_path(obj, path)
    try:
        return None if value is None else int(value)
    except Exception:
        return None


def _lookup_attr_path(obj: object, path: str) -> object | None:
    cur = obj
    for part in path.split("."):
        if cur is None or not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur
