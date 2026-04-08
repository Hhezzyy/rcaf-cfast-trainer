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
    cue_card = payload.cue_card
    return (
        str(payload.scenario_family),
        payload.scenario_index,
        round(payload.scenario_elapsed_s, 1),
        tuple(
            (
                contact.callsign,
                contact.allegiance,
                contact.asset_type,
                contact.cell_label,
                contact.destination_cell,
                round(contact.fade, 2),
            )
            for contact in payload.visible_contacts
        ),
        None
        if cue_card is None
        else (
            cue_card.callsign,
            cue_card.allegiance,
            cue_card.asset_type,
            cue_card.instructed_destination,
            cue_card.actual_destination,
            cue_card.last_report_text,
            cue_card.task_text,
            cue_card.channel_text,
            round(cue_card.fade, 2),
        ),
        payload.top_strip_text,
        round(payload.top_strip_fade, 2),
        tuple(payload.radio_log),
        tuple(payload.speech_prefetch_lines),
        None
        if query is None
        else (
            str(query.kind),
            str(query.answer_mode),
            query.prompt,
            query.correct_answer_token,
            query.subject_callsign,
            tuple((choice.code, choice.text) for choice in query.answer_choices),
        ),
        tuple(payload.announcement_lines),
        payload.display_clock_text,
    )


def _advance_until_query(
    engine,
    clock: FakeClock,
    *,
    max_steps: int = 240,
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
        query_interval_min_s=10,
        query_interval_max_s=10,
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
            qid = payload1.active_query.query_id
            if qid not in answered1:
                assert engine1.submit_answer(payload1.active_query.correct_answer_token) is True
                answered1.add(qid)
        if payload2 is not None and payload2.active_query is not None:
            qid = payload2.active_query.query_id
            if qid not in answered2:
                assert engine2.submit_answer(payload2.active_query.correct_answer_token) is True
                answered2.add(qid)

        clock1.advance(1.0)
        clock2.advance(1.0)
        engine1.update()
        engine2.update()

    assert signatures1 == signatures2
    assert engine1.scored_summary() == engine2.scored_summary()


def test_situational_awareness_different_seed_changes_callsign_or_query_stream() -> None:
    cfg = SituationalAwarenessConfig(
        scored_duration_s=45.0,
        practice_scenarios=0,
        scored_scenario_duration_s=45.0,
        query_interval_min_s=10,
        query_interval_max_s=10,
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
            seen1.append(
                (
                    str(payload1.scenario_family),
                    tuple(contact.callsign for contact in payload1.visible_contacts),
                    payload1.active_query.prompt if payload1.active_query else None,
                )
            )
        if isinstance(payload2, SituationalAwarenessPayload):
            seen2.append(
                (
                    str(payload2.scenario_family),
                    tuple(contact.callsign for contact in payload2.visible_contacts),
                    payload2.active_query.prompt if payload2.active_query else None,
                )
            )
        clock1.advance(1.0)
        clock2.advance(1.0)
        engine1.update()
        engine2.update()

    assert seen1 != seen2


def test_route_queries_match_ordered_and_actual_destinations() -> None:
    clock = FakeClock()
    engine = build_situational_awareness_test(
        clock=clock,
        seed=91,
        difficulty=0.65,
        config=SituationalAwarenessConfig(
            practice_scenarios=0,
            scored_duration_s=90.0,
            scored_scenario_duration_s=90.0,
            query_interval_min_s=10,
            query_interval_max_s=10,
        ),
        practice_segments=(),
        scored_segments=(
            SituationalAwarenessTrainingSegment(
                label="Route",
                duration_s=90.0,
                active_channels=("pictorial", "numerical", "aural"),
                active_query_kinds=(
                    SituationalAwarenessQueryKind.INSTRUCTED_DESTINATION.value,
                    SituationalAwarenessQueryKind.ACTUAL_DESTINATION.value,
                ),
                focus_label="Ordered versus actual route",
                profile=SituationalAwarenessTrainingProfile(
                    query_interval_min_s=10,
                    query_interval_max_s=10,
                    response_window_s=10,
                ),
            ),
        ),
    )
    engine.start_scored()

    instructed_payload = _advance_until_query(
        engine,
        clock,
        required_kind=SituationalAwarenessQueryKind.INSTRUCTED_DESTINATION,
    )
    instructed_query = instructed_payload.active_query
    assert instructed_query is not None
    lead = engine._live_assets[instructed_query.subject_callsign]
    assert instructed_query.correct_answer_token == lead.instructed_destination

    engine.submit_answer(instructed_query.correct_answer_token)
    actual_payload = _advance_until_query(
        engine,
        clock,
        required_kind=SituationalAwarenessQueryKind.ACTUAL_DESTINATION,
    )
    actual_query = actual_payload.active_query
    assert actual_query is not None
    lead = engine._live_assets[actual_query.subject_callsign]
    assert actual_query.correct_answer_token == lead.actual_destination
    assert actual_query.correct_answer_token != lead.instructed_destination


def test_report_variation_query_uses_five_choice_mode_and_unique_options() -> None:
    clock = FakeClock()
    engine = build_situational_awareness_test(
        clock=clock,
        seed=222,
        difficulty=0.7,
        config=SituationalAwarenessConfig(
            practice_scenarios=0,
            scored_duration_s=45.0,
            scored_scenario_duration_s=45.0,
        ),
        practice_segments=(),
        scored_segments=(
            SituationalAwarenessTrainingSegment(
                label="Report Correlation",
                duration_s=45.0,
                active_channels=("coded", "numerical", "aural"),
                active_query_kinds=(SituationalAwarenessQueryKind.REPORT_VARIATION.value,),
                focus_label="Variation recall",
                profile=SituationalAwarenessTrainingProfile(
                    query_interval_min_s=10,
                    query_interval_max_s=10,
                    response_window_s=9,
                ),
            ),
        ),
    )
    engine.start_scored()

    payload = _advance_until_query(
        engine,
        clock,
        required_kind=SituationalAwarenessQueryKind.REPORT_VARIATION,
    )
    query = payload.active_query
    assert query is not None
    assert query.answer_mode is SituationalAwarenessAnswerMode.CHOICE
    assert query.correct_answer_token in {"1", "2", "3", "4", "5"}
    assert len(query.answer_choices) == 5
    assert len({choice.text for choice in query.answer_choices}) == 5


def test_grid_location_queries_target_hidden_subject_after_cues_fade() -> None:
    cfg = SituationalAwarenessConfig(
        scored_duration_s=90.0,
        practice_scenarios=0,
        scored_scenario_duration_s=90.0,
        query_interval_min_s=10,
        query_interval_max_s=10,
    )
    clock = FakeClock()
    engine = build_situational_awareness_test(clock=clock, seed=444, difficulty=0.6, config=cfg)
    engine.start_scored()

    for _ in range(180):
        payload = engine.snapshot().payload
        if isinstance(payload, SituationalAwarenessPayload) and payload.active_query is not None:
            query = payload.active_query
            if query.kind in (
                SituationalAwarenessQueryKind.CURRENT_LOCATION,
                SituationalAwarenessQueryKind.INSTRUCTED_DESTINATION,
                SituationalAwarenessQueryKind.ACTUAL_DESTINATION,
            ):
                visible_callsigns = {contact.callsign for contact in payload.visible_contacts}
                assert query.subject_callsign not in visible_callsigns
                return
        clock.advance(1.0)
        engine.update()

    raise AssertionError("Expected a hidden-subject grid query in the scored stream.")


def test_radio_announcements_only_emit_on_actual_updates() -> None:
    clock = FakeClock()
    engine = build_situational_awareness_test(
        clock=clock,
        seed=515,
        difficulty=0.65,
        config=SituationalAwarenessConfig(
            practice_scenarios=0,
            scored_duration_s=60.0,
            scored_scenario_duration_s=60.0,
            query_interval_min_s=10,
            query_interval_max_s=10,
        ),
    )
    engine.start_scored()

    first_payload = engine.snapshot().payload
    assert isinstance(first_payload, SituationalAwarenessPayload)
    assert first_payload.announcement_lines == ()
    assert first_payload.speech_prefetch_lines

    seen_radio = False
    seen_query_after_radio = False
    for _ in range(40):
        payload = engine.snapshot().payload
        assert isinstance(payload, SituationalAwarenessPayload)
        if payload.announcement_lines:
            seen_radio = True
        if seen_radio and payload.active_query is not None:
            assert payload.announcement_lines == ()
            seen_query_after_radio = True
            break
        clock.advance(1.0)
        engine.update()

    assert seen_radio is True
    assert seen_query_after_radio is True


def test_high_difficulty_can_emit_multiword_callsigns() -> None:
    clock = FakeClock()
    engine = build_situational_awareness_test(
        clock=clock,
        seed=808,
        difficulty=0.95,
        config=SituationalAwarenessConfig(
            practice_scenarios=0,
            scored_duration_s=60.0,
            scored_scenario_duration_s=60.0,
            min_track_count=6,
            max_track_count=6,
        ),
    )
    engine.start_scored()
    payload = engine.snapshot().payload
    assert isinstance(payload, SituationalAwarenessPayload)
    callsigns = {asset.callsign for asset in engine._live_assets.values()}
    assert any(" " in callsign for callsign in callsigns)


def test_rule_action_scoring_is_binary_and_timeout_counts_as_miss() -> None:
    cfg = SituationalAwarenessConfig(
        scored_duration_s=75.0,
        practice_scenarios=0,
        scored_scenario_duration_s=75.0,
        query_interval_min_s=10,
        query_interval_max_s=10,
    )
    clock = FakeClock()
    engine = build_situational_awareness_test(clock=clock, seed=515, difficulty=0.8, config=cfg)
    engine.start_scored()

    payload = _advance_until_query(
        engine,
        clock,
        required_kind=SituationalAwarenessQueryKind.RULE_ACTION,
    )
    query = payload.active_query
    assert query is not None
    wrong = "2" if query.correct_answer_token != "2" else "3"
    assert engine.submit_answer(wrong) is True

    baseline_events = len(engine.events())
    for _ in range(20):
        clock.advance(1.0)
        engine.update()
        if len(engine.events()) > baseline_events:
            break

    scored_events = [event for event in engine.events() if event.phase is Phase.SCORED]
    assert len(scored_events) >= 2
    assert any(event.score == pytest.approx(0.0) and event.raw == wrong for event in scored_events)
    assert any(
        event.score == pytest.approx(0.0) and event.user_answer == 0 and event.raw == ""
        for event in scored_events
    )


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
                label="Route Focus",
                duration_s=45.0,
                active_channels=("pictorial", "numerical"),
                active_query_kinds=(SituationalAwarenessQueryKind.ACTUAL_DESTINATION.value,),
                focus_label="Ordered versus actual route",
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
    assert payload.active_query_kinds == ("actual_destination",)
    assert payload.focus_label == "Ordered versus actual route"
    assert payload.segment_label == "Route Focus"
    assert payload.active_query is not None
    assert payload.active_query.kind is SituationalAwarenessQueryKind.ACTUAL_DESTINATION
