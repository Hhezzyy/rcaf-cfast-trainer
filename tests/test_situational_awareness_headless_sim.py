from __future__ import annotations

from dataclasses import dataclass

import pytest

from cfast_trainer.cognitive_core import Phase
from cfast_trainer.situational_awareness import (
    SituationalAwarenessConfig,
    SituationalAwarenessPayload,
    build_situational_awareness_test,
)


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


def _advance_until_payload(engine, clock: FakeClock, *, with_query: bool, max_steps: int = 240):
    for _ in range(max_steps):
        payload = engine.snapshot().payload
        if isinstance(payload, SituationalAwarenessPayload):
            if with_query and payload.active_query is not None:
                return payload
            if not with_query:
                return payload
        clock.advance(1.0)
        engine.update()
    raise AssertionError("Failed to reach a bounded Situational Awareness payload state.")


def test_headless_scripted_run_tracks_attempts_correct_and_timeouts() -> None:
    clock = FakeClock()
    engine = build_situational_awareness_test(
        clock=clock,
        seed=5151,
        difficulty=0.65,
        config=SituationalAwarenessConfig(
            practice_scenarios=1,
            practice_scenario_duration_s=24.0,
            scored_duration_s=60.0,
            scored_scenario_duration_s=60.0,
            query_interval_min_s=12,
            query_interval_max_s=12,
        ),
    )

    engine.start_practice()
    assert engine.phase is Phase.PRACTICE

    practice_payload = _advance_until_payload(engine, clock, with_query=True)
    assert practice_payload.active_query is not None
    assert engine.submit_answer(practice_payload.active_query.correct_answer_token) is True

    clock.advance(24.0)
    engine.update()
    assert engine.phase is Phase.PRACTICE_DONE

    engine.start_scored()
    assert engine.phase is Phase.SCORED

    first = _advance_until_payload(engine, clock, with_query=True)
    assert first.active_query is not None
    assert engine.submit_answer(first.active_query.correct_answer_token) is True

    second = _advance_until_payload(engine, clock, with_query=True)
    assert second.active_query is not None
    wrong = "1"
    if wrong == second.active_query.correct_answer_token:
        wrong = "2"
    assert engine.submit_answer(wrong) is True

    third = _advance_until_payload(engine, clock, with_query=True)
    assert third.active_query is not None
    timeout_steps = max(1, int(round(third.active_query.expires_in_s)))
    for _ in range(timeout_steps):
        clock.advance(1.0)
        engine.update()

    clock.advance(60.0)
    engine.update()
    assert engine.phase is Phase.RESULTS

    summary = engine.scored_summary()
    assert summary.attempted >= 3
    assert summary.correct >= 1
    assert summary.total_score <= summary.max_score

    scored_events = [event for event in engine.events() if event.phase is Phase.SCORED]
    assert scored_events[0].score == pytest.approx(1.0)
    assert scored_events[1].score == pytest.approx(0.0)
    assert scored_events[2].score == pytest.approx(0.0)


def test_grid_cell_and_numeric_direct_response_modes_are_accepted() -> None:
    clock = FakeClock()
    engine = build_situational_awareness_test(
        clock=clock,
        seed=808,
        difficulty=0.55,
        config=SituationalAwarenessConfig(
            practice_scenarios=0,
            scored_duration_s=60.0,
            scored_scenario_duration_s=60.0,
            query_interval_min_s=12,
            query_interval_max_s=12,
        ),
    )
    engine.start_scored()

    answered_modes: set[str] = set()
    for _ in range(120):
        payload = engine.snapshot().payload
        if isinstance(payload, SituationalAwarenessPayload) and payload.active_query is not None:
            mode = str(payload.active_query.answer_mode)
            if mode not in answered_modes:
                assert engine.submit_answer(payload.active_query.correct_answer_token) is True
                answered_modes.add(mode)
                if answered_modes >= {"grid_cell", "track_index"}:
                    break
        clock.advance(1.0)
        engine.update()

    assert {"grid_cell", "track_index"} <= answered_modes
