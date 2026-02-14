from __future__ import annotations

from dataclasses import dataclass

from .cognitive_core import Phase, QuestionEvent, TimedTextInputTest


@dataclass(frozen=True, slots=True)
class AttemptResult:
    """Persistable summary + event log for a completed attempt.

    This is intentionally generic so it can be reused across tests that use
    TimedTextInputTest.
    """

    test_code: str
    test_version: int
    seed: int
    difficulty: float
    practice_questions: int
    scored_duration_s: float

    attempted: int
    correct: int
    accuracy: float
    throughput_per_min: float
    mean_rt_ms: float | None
    median_rt_ms: float | None

    events: list[QuestionEvent]


def attempt_result_from_timed_test(
    test: TimedTextInputTest,
    *,
    test_code: str,
    test_version: int = 1,
) -> AttemptResult:
    """Build an AttemptResult from a finished TimedTextInputTest."""

    summary = test.scored_summary()
    scored_events = [e for e in test.events() if e.phase is Phase.SCORED]
    rts_ms = sorted(int(round(e.response_time_s * 1000.0)) for e in scored_events)

    mean_ms: float | None
    median_ms: float | None
    if not rts_ms:
        mean_ms = None
        median_ms = None
    else:
        mean_ms = float(sum(rts_ms)) / float(len(rts_ms))
        mid = len(rts_ms) // 2
        if len(rts_ms) % 2 == 1:
            median_ms = float(rts_ms[mid])
        else:
            median_ms = float(rts_ms[mid - 1] + rts_ms[mid]) / 2.0

    return AttemptResult(
        test_code=str(test_code),
        test_version=int(test_version),
        seed=int(test.seed),
        difficulty=float(test.difficulty),
        practice_questions=int(test.practice_questions),
        scored_duration_s=float(test.scored_duration_s),
        attempted=int(summary.attempted),
        correct=int(summary.correct),
        accuracy=float(summary.accuracy),
        throughput_per_min=float(summary.throughput_per_min),
        mean_rt_ms=mean_ms,
        median_rt_ms=median_ms,
        events=scored_events,
    )
