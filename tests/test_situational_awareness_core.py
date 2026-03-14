from __future__ import annotations

from dataclasses import dataclass

import pytest

from cfast_trainer.cognitive_core import Phase
from cfast_trainer.situational_awareness import (
    SituationalAwarenessAnswerMode,
    SituationalAwarenessConfig,
    SituationalAwarenessPayload,
    SituationalAwarenessQueryKind,
    SituationalAwarenessTrainingProfile,
    SituationalAwarenessTrainingSegment,
    build_situational_awareness_test,
)


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


def _payload_signature(payload: SituationalAwarenessPayload | None) -> object:
    if payload is None:
        return None
    query = payload.active_query
    return (
        str(payload.scenario_family),
        payload.scenario_index,
        round(payload.scenario_elapsed_s, 1),
        tuple(
            (
                track.index,
                track.callsign,
                track.cell_label,
                track.heading,
                track.channel,
                track.altitude_fl,
                track.fuel_state,
                track.waypoint,
            )
            for track in payload.tracks
        ),
        None
        if query is None
        else (
            str(query.kind),
            str(query.answer_mode),
            query.prompt,
            query.correct_answer_token,
            query.subject_callsign,
            query.future_offset_s,
            tuple((choice.code, choice.text) for choice in query.answer_choices),
        ),
        payload.recent_feed_lines,
    )


def _advance_until_query(
    engine,
    clock: FakeClock,
    *,
    max_steps: int = 180,
    required_kind: SituationalAwarenessQueryKind | None = None,
) -> SituationalAwarenessPayload:
    for _ in range(max_steps):
        payload = engine.snapshot().payload
        if isinstance(payload, SituationalAwarenessPayload) and payload.active_query is not None:
            if required_kind is None or payload.active_query.kind is required_kind:
                return payload
        clock.advance(1.0)
        engine.update()
    raise AssertionError(f"Failed to find query kind={required_kind!r} within bounded steps.")


def test_situational_awareness_determinism_same_seed_same_answers() -> None:
    cfg = SituationalAwarenessConfig(
        scored_duration_s=45.0,
        practice_scenarios=0,
        scored_scenario_duration_s=45.0,
        query_interval_min_s=12,
        query_interval_max_s=12,
    )
    clock1 = FakeClock()
    clock2 = FakeClock()
    engine1 = build_situational_awareness_test(clock=clock1, seed=308, difficulty=0.7, config=cfg)
    engine2 = build_situational_awareness_test(clock=clock2, seed=308, difficulty=0.7, config=cfg)
    engine1.start_scored()
    engine2.start_scored()

    signatures1: list[object] = []
    signatures2: list[object] = []
    answered1: set[int] = set()
    answered2: set[int] = set()
    for _ in range(40):
        snap1 = engine1.snapshot()
        snap2 = engine2.snapshot()
        payload1 = snap1.payload if isinstance(snap1.payload, SituationalAwarenessPayload) else None
        payload2 = snap2.payload if isinstance(snap2.payload, SituationalAwarenessPayload) else None
        signatures1.append(_payload_signature(payload1))
        signatures2.append(_payload_signature(payload2))

        if payload1 is not None and payload1.active_query is not None:
            q1 = payload1.active_query.query_id
            if q1 not in answered1:
                assert engine1.submit_answer(payload1.active_query.correct_answer_token) is True
                answered1.add(q1)
        if payload2 is not None and payload2.active_query is not None:
            q2 = payload2.active_query.query_id
            if q2 not in answered2:
                assert engine2.submit_answer(payload2.active_query.correct_answer_token) is True
                answered2.add(q2)

        clock1.advance(1.0)
        clock2.advance(1.0)
        engine1.update()
        engine2.update()

    assert signatures1 == signatures2
    assert engine1.scored_summary() == engine2.scored_summary()


def test_situational_awareness_different_seed_changes_family_or_query_stream() -> None:
    cfg = SituationalAwarenessConfig(
        scored_duration_s=45.0,
        practice_scenarios=0,
        scored_scenario_duration_s=45.0,
        query_interval_min_s=12,
        query_interval_max_s=12,
    )
    clock1 = FakeClock()
    clock2 = FakeClock()
    engine1 = build_situational_awareness_test(clock=clock1, seed=111, difficulty=0.65, config=cfg)
    engine2 = build_situational_awareness_test(clock=clock2, seed=222, difficulty=0.65, config=cfg)
    engine1.start_scored()
    engine2.start_scored()

    seen1: list[object] = []
    seen2: list[object] = []
    for _ in range(20):
        payload1 = engine1.snapshot().payload
        payload2 = engine2.snapshot().payload
        if isinstance(payload1, SituationalAwarenessPayload):
            seen1.append((str(payload1.scenario_family), payload1.active_query.prompt if payload1.active_query else None))
        if isinstance(payload2, SituationalAwarenessPayload):
            seen2.append((str(payload2.scenario_family), payload2.active_query.prompt if payload2.active_query else None))
        clock1.advance(1.0)
        clock2.advance(1.0)
        engine1.update()
        engine2.update()

    assert seen1 != seen2


def test_future_position_query_matches_future_live_track_cell() -> None:
    cfg = SituationalAwarenessConfig(
        scored_duration_s=90.0,
        practice_scenarios=0,
        scored_scenario_duration_s=90.0,
        query_interval_min_s=12,
        query_interval_max_s=12,
    )
    clock_query = FakeClock()
    clock_projection = FakeClock()
    engine_query = build_situational_awareness_test(
        clock=clock_query, seed=91, difficulty=0.5, config=cfg
    )
    engine_projection = build_situational_awareness_test(
        clock=clock_projection, seed=91, difficulty=0.5, config=cfg
    )
    engine_query.start_scored()
    engine_projection.start_scored()

    payload = _advance_until_query(
        engine_query,
        clock_query,
        required_kind=SituationalAwarenessQueryKind.FUTURE_POSITION,
    )
    query = payload.active_query
    assert query is not None
    assert query.answer_mode is SituationalAwarenessAnswerMode.GRID_CELL
    assert query.subject_callsign is not None
    assert query.future_offset_s is not None

    while clock_projection.now() < clock_query.now():
        clock_projection.advance(1.0)
        engine_projection.update()

    for _ in range(query.future_offset_s):
        clock_projection.advance(1.0)
        engine_projection.update()

    projection_payload = engine_projection.snapshot().payload
    assert isinstance(projection_payload, SituationalAwarenessPayload)
    subject = next(
        track for track in projection_payload.tracks if track.callsign == query.subject_callsign
    )
    assert subject.cell_label == query.correct_answer_token


def test_code_status_query_answer_is_visible_in_status_panel() -> None:
    cfg = SituationalAwarenessConfig(
        scored_duration_s=75.0,
        practice_scenarios=0,
        scored_scenario_duration_s=75.0,
        query_interval_min_s=12,
        query_interval_max_s=12,
    )
    clock = FakeClock()
    engine = build_situational_awareness_test(clock=clock, seed=222, difficulty=0.7, config=cfg)
    engine.start_scored()

    payload = _advance_until_query(
        engine,
        clock,
        required_kind=SituationalAwarenessQueryKind.CODE_OR_STATUS_RECALL,
    )
    query = payload.active_query
    assert query is not None
    assert query.answer_mode is SituationalAwarenessAnswerMode.TRACK_INDEX
    assert query.correct_answer_token in {str(entry.track_index) for entry in payload.status_entries}


def test_action_query_scoring_is_binary_and_timeout_counts_as_miss() -> None:
    cfg = SituationalAwarenessConfig(
        scored_duration_s=75.0,
        practice_scenarios=0,
        scored_scenario_duration_s=75.0,
        query_interval_min_s=12,
        query_interval_max_s=12,
    )
    clock = FakeClock()
    engine = build_situational_awareness_test(clock=clock, seed=515, difficulty=0.8, config=cfg)
    engine.start_scored()

    payload = _advance_until_query(
        engine,
        clock,
        required_kind=SituationalAwarenessQueryKind.ACTION_SELECTION,
    )
    query = payload.active_query
    assert query is not None
    wrong = "2" if query.correct_answer_token != "2" else "3"
    assert engine.submit_answer(wrong) is True

    payload = _advance_until_query(engine, clock)
    query = payload.active_query
    assert query is not None
    deadline = int(round(query.expires_in_s))
    for _ in range(max(1, deadline)):
        clock.advance(1.0)
        engine.update()

    scored_events = [event for event in engine.events() if event.phase is Phase.SCORED]
    assert len(scored_events) >= 2
    assert scored_events[0].score == pytest.approx(0.0)
    assert scored_events[1].score == pytest.approx(0.0)


def test_default_config_matches_guide_focused_runtime() -> None:
    cfg = SituationalAwarenessConfig()
    assert cfg.scored_duration_s == pytest.approx(25.0 * 60.0)
    assert cfg.practice_scenarios == 3
    assert cfg.practice_scenario_duration_s == pytest.approx(45.0)


def test_custom_segment_payload_exposes_focus_metadata_and_restricts_query_kinds() -> None:
    clock = FakeClock()
    engine = build_situational_awareness_test(
        clock=clock,
        seed=808,
        difficulty=0.6,
        config=SituationalAwarenessConfig(
            practice_scenarios=0,
            scored_duration_s=45.0,
            scored_scenario_duration_s=45.0,
        ),
        practice_segments=(),
        scored_segments=(
            SituationalAwarenessTrainingSegment(
                label="Projection Focus",
                duration_s=45.0,
                active_channels=("pictorial", "numerical"),
                active_query_kinds=("future_position",),
                focus_label="Future projection",
                profile=SituationalAwarenessTrainingProfile(
                    min_track_count=3,
                    max_track_count=4,
                    query_interval_min_s=10,
                    query_interval_max_s=10,
                    response_window_s=9,
                ),
            ),
        ),
    )
    engine.start_scored()

    payload = _advance_until_query(engine, clock)
    assert payload.active_channels == ("pictorial", "numerical")
    assert payload.active_query_kinds == ("future_position",)
    assert payload.focus_label == "Future projection"
    assert payload.segment_label == "Projection Focus"
    assert payload.active_query is not None
    assert payload.active_query.kind is SituationalAwarenessQueryKind.FUTURE_POSITION
