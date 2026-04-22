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
            cue_card.next_waypoint,
            cue_card.next_waypoint_at_text,
            cue_card.altitude_text,
            cue_card.communications_text,
            round(cue_card.fade, 2),
        ),
        payload.top_strip_text,
        round(payload.top_strip_fade, 2),
        tuple(payload.radio_log),
        tuple(payload.speech_prefetch_lines),
        payload.round_index,
        payload.round_total,
        payload.north_heading_deg,
        None
        if query is None
        else (
            str(query.kind),
            str(query.answer_mode),
            query.prompt,
            query.correct_answer_token,
            query.subject_callsign,
            tuple((choice.code, choice.text) for choice in query.answer_choices),
            tuple(query.accepted_tokens),
            query.entry_label,
            query.entry_placeholder,
            query.entry_max_chars,
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


def _build_targeted_query_engine(
    *,
    query_kind: SituationalAwarenessQueryKind,
    seed: int = 515,
    difficulty: float = 0.7,
    profile: SituationalAwarenessTrainingProfile | None = None,
):
    return build_situational_awareness_test(
        clock=FakeClock(),
        seed=seed,
        difficulty=difficulty,
        config=SituationalAwarenessConfig(
            practice_scenarios=0,
            scored_duration_s=45.0,
            scored_scenario_duration_s=45.0,
            query_interval_min_s=10,
            query_interval_max_s=10,
        ),
        practice_segments=(),
        scored_segments=(
            SituationalAwarenessTrainingSegment(
                label="Targeted",
                duration_s=45.0,
                active_channels=("pictorial", "coded", "numerical", "aural"),
                active_query_kinds=(query_kind.value,),
                focus_label=query_kind.value,
                profile=profile
                or SituationalAwarenessTrainingProfile(
                    query_interval_min_s=10,
                    query_interval_max_s=10,
                    response_window_s=10,
                ),
            ),
        ),
    )


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


def test_grid_location_queries_freeze_and_display_subject_while_active() -> None:
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
                visible_contacts = {contact.callsign: contact for contact in payload.visible_contacts}
                frozen = engine._active_query_frozen_contact
                assert frozen is not None
                assert query.subject_callsign == frozen.callsign
                assert query.subject_callsign in visible_contacts
                contact = visible_contacts[query.subject_callsign]
                assert contact.x == pytest.approx(frozen.x)
                assert contact.y == pytest.approx(frozen.y)
                return
        clock.advance(1.0)
        engine.update()

    raise AssertionError("Expected a frozen-subject grid query in the scored stream.")


def test_queried_callsign_is_retired_for_rest_of_scenario() -> None:
    cfg = SituationalAwarenessConfig(
        scored_duration_s=70.0,
        practice_scenarios=0,
        scored_scenario_duration_s=70.0,
        query_interval_min_s=8,
        query_interval_max_s=8,
        min_track_count=6,
        max_track_count=6,
    )
    clock = FakeClock()
    engine = build_situational_awareness_test(clock=clock, seed=446, difficulty=0.75, config=cfg)
    engine.start_scored()

    first_payload = _advance_until_query(engine, clock)
    first_query = first_payload.active_query
    assert first_query is not None
    retired = first_query.subject_callsign
    assert retired is not None

    assert engine.submit_answer(first_query.correct_answer_token) is True
    assert retired in engine._retired_callsigns

    seen_later_query_subjects: list[str] = []
    for _ in range(38):
        payload = engine.snapshot().payload
        assert isinstance(payload, SituationalAwarenessPayload)
        assert retired not in {contact.callsign for contact in payload.visible_contacts}
        assert payload.cue_card is None or payload.cue_card.callsign != retired
        assert retired not in payload.top_strip_text
        assert all(retired not in line for line in payload.radio_log)
        if payload.active_query is not None:
            assert payload.active_query.subject_callsign != retired
            if payload.active_query.subject_callsign is not None:
                seen_later_query_subjects.append(payload.active_query.subject_callsign)
            assert engine.submit_answer(payload.active_query.correct_answer_token) is True
        clock.advance(1.0)
        engine.update()

    assert seen_later_query_subjects


def test_active_query_freezes_live_asset_until_answered_then_motion_resumes() -> None:
    cfg = SituationalAwarenessConfig(
        scored_duration_s=45.0,
        practice_scenarios=0,
        scored_scenario_duration_s=45.0,
        query_interval_min_s=10,
        query_interval_max_s=10,
        min_track_count=5,
        max_track_count=5,
    )
    clock = FakeClock()
    engine = build_situational_awareness_test(
        clock=clock,
        seed=448,
        difficulty=0.8,
        config=cfg,
        practice_segments=(),
        scored_segments=(
            SituationalAwarenessTrainingSegment(
                label="Current Location",
                duration_s=45.0,
                active_channels=("pictorial", "coded", "numerical", "aural"),
                active_query_kinds=(SituationalAwarenessQueryKind.CURRENT_LOCATION.value,),
                focus_label="Current location",
                profile=SituationalAwarenessTrainingProfile(
                    query_interval_min_s=10,
                    query_interval_max_s=10,
                    response_window_s=10,
                    min_track_count=5,
                    max_track_count=5,
                ),
            ),
        ),
    )
    engine.start_scored()

    payload = _advance_until_query(
        engine,
        clock,
        required_kind=SituationalAwarenessQueryKind.CURRENT_LOCATION,
    )
    query = payload.active_query
    assert query is not None
    subject = query.subject_callsign
    assert subject is not None
    asset = engine._live_assets[subject]
    asset.movement_mode = "linear"
    asset.heading = "W" if asset.x >= 8.0 else "E"
    asset.speed_cells_per_min = 60
    frozen_x = asset.x
    frozen_y = asset.y

    clock.advance(3.0)
    engine.update()
    assert asset.x == pytest.approx(frozen_x)
    assert asset.y == pytest.approx(frozen_y)

    active_payload = engine.snapshot().payload
    assert isinstance(active_payload, SituationalAwarenessPayload)
    active_contact = next(
        contact for contact in active_payload.visible_contacts if contact.callsign == subject
    )
    assert active_contact.x == pytest.approx(frozen_x)
    assert active_contact.y == pytest.approx(frozen_y)

    assert engine.submit_answer(query.correct_answer_token) is True
    clock.advance(1.0)
    engine.update()
    assert asset.x != pytest.approx(frozen_x)
    assert asset.y == pytest.approx(frozen_y)


def test_situational_awareness_seeded_motion_modes_include_stationary_and_direct_to_target() -> None:
    cfg = SituationalAwarenessConfig(
        scored_duration_s=45.0,
        practice_scenarios=0,
        scored_scenario_duration_s=45.0,
        query_interval_min_s=40,
        query_interval_max_s=40,
        min_track_count=5,
        max_track_count=5,
    )
    clock_a = FakeClock()
    clock_b = FakeClock()
    engine_a = build_situational_awareness_test(clock=clock_a, seed=449, difficulty=0.8, config=cfg)
    engine_b = build_situational_awareness_test(clock=clock_b, seed=449, difficulty=0.8, config=cfg)
    engine_a.start_scored()
    engine_b.start_scored()

    modes_a = {
        callsign: asset.movement_mode
        for callsign, asset in sorted(engine_a._live_assets.items())
    }
    modes_b = {
        callsign: asset.movement_mode
        for callsign, asset in sorted(engine_b._live_assets.items())
    }
    assert modes_a == modes_b
    assert list(modes_a.values()).count("stationary") == 1
    assert list(modes_a.values()).count("direct_to_target") >= 2

    def distance_to_destination(asset) -> float:
        row = "ABCDEFGHIJ".index(asset.actual_destination[0])
        col = int(asset.actual_destination[1:])
        dx = float(col) - float(asset.x)
        dy = float(row) - float(asset.y)
        return (dx * dx + dy * dy) ** 0.5

    stationary = next(
        asset for asset in engine_a._live_assets.values() if asset.movement_mode == "stationary"
    )
    direct_assets = [
        asset for asset in engine_a._live_assets.values() if asset.movement_mode == "direct_to_target"
    ]
    stationary_start = (stationary.x, stationary.y)
    direct_start_distances = {
        asset.callsign: distance_to_destination(asset)
        for asset in direct_assets
    }

    clock_a.advance(5.0)
    engine_a.update()

    assert (stationary.x, stationary.y) == pytest.approx(stationary_start)
    for asset in direct_assets:
        assert distance_to_destination(asset) < direct_start_distances[asset.callsign]


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
    engine = build_situational_awareness_test(
        clock=clock,
        seed=515,
        difficulty=0.8,
        config=cfg,
        practice_segments=(),
        scored_segments=(
            SituationalAwarenessTrainingSegment(
                label="Rule Action",
                duration_s=75.0,
                active_channels=("pictorial", "coded", "numerical", "aural"),
                active_query_kinds=(SituationalAwarenessQueryKind.RULE_ACTION.value,),
                focus_label="Rule action only",
                profile=SituationalAwarenessTrainingProfile(
                    query_interval_min_s=10,
                    query_interval_max_s=10,
                    response_window_s=10,
                ),
            ),
        ),
    )
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


def test_default_scored_run_uses_three_rounds_and_fixed_cardinal_north_per_round() -> None:
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

    seen_rounds: list[int] = []
    seen_headings: list[int] = []
    active_heading: int | None = None
    for _ in range(80):
        payload = engine.snapshot().payload
        if isinstance(payload, SituationalAwarenessPayload):
            assert payload.round_total == 3
            assert payload.north_heading_deg in (0, 90, 180, 270)
            if payload.round_index not in seen_rounds:
                seen_rounds.append(payload.round_index)
                seen_headings.append(payload.north_heading_deg)
                active_heading = payload.north_heading_deg
            else:
                assert payload.north_heading_deg == active_heading
        if engine.phase is Phase.RESULTS:
            break
        clock.advance(1.0)
        engine.update()

    assert seen_rounds == [1, 2, 3]
    assert len(set(seen_headings)) == 3


def test_scored_round_seeds_are_fresh_and_repeat_for_same_seed() -> None:
    def _round_trace(seed: int) -> list[tuple[int, int, int]]:
        clock = FakeClock()
        engine = build_situational_awareness_test(
            clock=clock,
            seed=seed,
            difficulty=0.65,
            config=SituationalAwarenessConfig(
                practice_scenarios=0,
                scored_duration_s=60.0,
                scored_scenario_duration_s=60.0,
                query_interval_min_s=30,
                query_interval_max_s=30,
            ),
        )
        engine.start_scored()
        seen: list[tuple[int, int, int]] = []
        for _ in range(80):
            payload = engine.snapshot().payload
            if isinstance(payload, SituationalAwarenessPayload):
                current = (
                    payload.round_index,
                    int(engine._scenario_plan.context["round_seed"]),
                    payload.north_heading_deg,
                )
                if not seen or seen[-1][0] != current[0]:
                    seen.append(current)
            if engine.phase is Phase.RESULTS:
                break
            clock.advance(1.0)
            engine.update()
        return seen

    first = _round_trace(515)
    second = _round_trace(515)
    assert first == second
    assert [round_index for round_index, _round_seed, _heading in first] == [1, 2, 3]
    assert len({round_seed for _round_index, round_seed, _heading in first}) == 3


def test_info_panel_focus_switches_on_updates_and_top_strip_expires_cleanly() -> None:
    clock = FakeClock()
    engine = build_situational_awareness_test(
        clock=clock,
        seed=111,
        difficulty=0.7,
        config=SituationalAwarenessConfig(
            practice_scenarios=0,
            scored_duration_s=20.0,
            scored_scenario_duration_s=20.0,
            query_interval_min_s=30,
            query_interval_max_s=30,
        ),
        practice_segments=(),
        scored_segments=(
            SituationalAwarenessTrainingSegment(
                label="Panel",
                duration_s=20.0,
                active_channels=("pictorial", "coded", "numerical", "aural"),
                active_query_kinds=(SituationalAwarenessQueryKind.CURRENT_LOCATION.value,),
                focus_label="Panel",
                profile=SituationalAwarenessTrainingProfile(
                    query_interval_min_s=30,
                    query_interval_max_s=30,
                    cue_card_ttl_s=2,
                    top_strip_ttl_s=2,
                ),
            ),
        ),
    )
    engine.start_scored()

    initial_payload = engine.snapshot().payload
    assert isinstance(initial_payload, SituationalAwarenessPayload)
    assert initial_payload.cue_card is not None
    initial_callsign = initial_payload.cue_card.callsign

    for _ in range(9):
        clock.advance(1.0)
        engine.update()

    switched_payload = engine.snapshot().payload
    assert isinstance(switched_payload, SituationalAwarenessPayload)
    assert switched_payload.cue_card is not None
    assert switched_payload.cue_card.callsign != initial_callsign
    assert switched_payload.top_strip_text != ""
    focused_asset = engine._live_assets[switched_payload.cue_card.callsign]
    assert switched_payload.cue_card.altitude_text == f"FL{focused_asset.altitude_fl}"
    assert switched_payload.cue_card.communications_text == engine._asset_comms_text(focused_asset)

    for _ in range(3):
        clock.advance(1.0)
        engine.update()

    expired_payload = engine.snapshot().payload
    assert isinstance(expired_payload, SituationalAwarenessPayload)
    assert expired_payload.top_strip_text == ""
    assert expired_payload.cue_card is not None
    assert expired_payload.cue_card.callsign == switched_payload.cue_card.callsign


def test_altitude_query_uses_numeric_input_and_rejects_invalid_tokens() -> None:
    engine = _build_targeted_query_engine(query_kind=SituationalAwarenessQueryKind.ALTITUDE)
    clock = engine._clock
    engine.start_scored()

    payload = _advance_until_query(
        engine,
        clock,
        required_kind=SituationalAwarenessQueryKind.ALTITUDE,
    )
    query = payload.active_query
    assert query is not None
    assert query.answer_mode is SituationalAwarenessAnswerMode.NUMERIC
    assert query.entry_label == "Altitude"
    assert query.entry_placeholder == "180"
    assert engine.submit_answer("FL180") is False
    assert engine.submit_answer(query.correct_answer_token) is True


def test_communications_query_uses_numeric_input_and_matches_active_channel() -> None:
    engine = _build_targeted_query_engine(query_kind=SituationalAwarenessQueryKind.COMMUNICATION_CHANNEL)
    clock = engine._clock
    engine.start_scored()

    payload = _advance_until_query(
        engine,
        clock,
        required_kind=SituationalAwarenessQueryKind.COMMUNICATION_CHANNEL,
    )
    query = payload.active_query
    assert query is not None
    assert query.answer_mode is SituationalAwarenessAnswerMode.NUMERIC
    subject = engine._live_assets[query.subject_callsign]
    assert query.correct_answer_token == str(subject.channel)
    assert engine.submit_answer("CHANNEL 3") is False
    assert engine.submit_answer(query.correct_answer_token) is True


def test_current_allegiance_query_uses_token_input_and_normalizes_colour_synonyms() -> None:
    engine = _build_targeted_query_engine(query_kind=SituationalAwarenessQueryKind.CURRENT_ALLEGIANCE)
    clock = engine._clock
    engine.start_scored()

    payload = _advance_until_query(
        engine,
        clock,
        required_kind=SituationalAwarenessQueryKind.CURRENT_ALLEGIANCE,
    )
    query = payload.active_query
    assert query is not None
    assert query.answer_mode is SituationalAwarenessAnswerMode.TOKEN
    assert query.accepted_tokens == ("FRIENDLY", "HOSTILE", "UNKNOWN")
    assert engine.submit_answer("BLUE") is False
    synonym = {
        "FRIENDLY": "YELLOW",
        "HOSTILE": "RED",
        "UNKNOWN": "WHITE",
    }[query.correct_answer_token]
    assert engine.submit_answer(synonym) is True
