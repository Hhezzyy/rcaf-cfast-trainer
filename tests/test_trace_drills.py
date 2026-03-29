from __future__ import annotations

from dataclasses import dataclass

import pytest

from cfast_trainer.ant_drills import AntDrillMode
from cfast_trainer.cognitive_core import Phase
from cfast_trainer.trace_drills import (
    TraceDrillConfig,
    build_trace_mixed_tempo_drill,
    build_trace_pressure_run_drill,
    build_tt1_command_switch_run_drill,
    build_tt1_lateral_anchor_drill,
    build_tt1_vertical_anchor_drill,
    build_tt2_position_recall_run_drill,
    build_tt2_steady_anchor_drill,
    build_tt2_turn_trace_run_drill,
)
from cfast_trainer.trace_test_1 import TraceTest1Payload, TraceTest1TrialStage
from cfast_trainer.trace_test_2 import (
    TraceTest2Payload,
    TraceTest2QuestionKind,
    TraceTest2TrialStage,
)


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


def _difficulty_for_level(level: int) -> float:
    return float(level - 1) / 9.0


def _wait_for_answer_ready(
    *,
    drill: object,
    clock: FakeClock,
    max_steps: int = 400,
    dt: float = 0.05,
) -> TraceTest1Payload | TraceTest2Payload | None:
    for _ in range(max_steps):
        snap = drill.snapshot()
        if snap.phase in (Phase.RESULTS, Phase.PRACTICE_DONE):
            return None
        payload = snap.payload
        if isinstance(payload, TraceTest1Payload):
            if payload.trial_stage is TraceTest1TrialStage.ANSWER_OPEN:
                return payload
        elif isinstance(payload, TraceTest2Payload):
            if payload.trial_stage is TraceTest2TrialStage.QUESTION:
                return payload
        clock.advance(dt)
        drill.update()
    raise AssertionError("Timed out waiting for trace drill answer window.")


def _answer_payload(payload: TraceTest1Payload | TraceTest2Payload) -> str:
    return str(payload.correct_code)


@pytest.mark.parametrize(
    ("builder", "expected_codes", "expected_kinds"),
    (
        (build_tt1_lateral_anchor_drill, {1, 2}, None),
        (build_tt1_vertical_anchor_drill, {3, 4}, None),
        (build_tt1_command_switch_run_drill, {1, 2, 3, 4}, None),
        (
            build_tt2_steady_anchor_drill,
            None,
            {TraceTest2QuestionKind.NO_DIRECTION_CHANGE},
        ),
        (
            build_tt2_turn_trace_run_drill,
            None,
            {
                TraceTest2QuestionKind.TURNED_LEFT,
                TraceTest2QuestionKind.TURNED_RIGHT,
            },
        ),
        (
            build_tt2_position_recall_run_drill,
            None,
            {
                TraceTest2QuestionKind.ENDED_LEFTMOST,
                TraceTest2QuestionKind.ENDED_HIGHEST,
            },
        ),
    ),
)
def test_trace_single_drill_builders_apply_expected_filters(
    builder,
    expected_codes,
    expected_kinds,
) -> None:
    clock = FakeClock()
    drill = builder(
        clock=clock,
        seed=701,
        difficulty=0.56,
        mode=AntDrillMode.BUILD,
        config=TraceDrillConfig(scored_duration_s=24.0),
    )

    drill.start_practice()
    payload = _wait_for_answer_ready(drill=drill, clock=clock)
    assert payload is not None

    assert drill.practice_questions == 2
    assert drill.scored_duration_s == pytest.approx(24.0)
    if expected_codes is not None:
        assert isinstance(payload, TraceTest1Payload)
        assert payload.correct_code in expected_codes
    else:
        assert isinstance(payload, TraceTest2Payload)
        assert payload.question_kind in expected_kinds


def test_trace_mode_defaults_match_ant_profile_for_single_and_mixed_drills() -> None:
    build_clock = FakeClock()
    build_drill = build_tt1_lateral_anchor_drill(
        clock=build_clock,
        seed=17,
        difficulty=0.5,
        mode=AntDrillMode.BUILD,
    )
    tempo_clock = FakeClock()
    tempo_drill = build_trace_mixed_tempo_drill(
        clock=tempo_clock,
        seed=17,
        difficulty=0.5,
        mode=AntDrillMode.TEMPO,
    )
    stress_clock = FakeClock()
    stress_drill = build_trace_pressure_run_drill(
        clock=stress_clock,
        seed=17,
        difficulty=0.5,
        mode=AntDrillMode.STRESS,
    )

    assert build_drill.practice_questions == 2
    assert build_drill.scored_duration_s == pytest.approx(180.0)
    assert tempo_drill.practice_questions == 0
    assert tempo_drill.scored_duration_s == pytest.approx(150.0)
    assert stress_drill.practice_questions == 0
    assert stress_drill.scored_duration_s == pytest.approx(180.0)


def test_trace_mixed_drill_sequences_tt1_before_tt2_in_practice() -> None:
    clock = FakeClock()
    drill = build_trace_mixed_tempo_drill(
        clock=clock,
        seed=44,
        difficulty=0.5,
        mode=AntDrillMode.BUILD,
        config=TraceDrillConfig(scored_duration_s=40.0),
    )

    drill.start_practice()
    seen_types: list[type] = []
    for _ in range(6):
        payload = _wait_for_answer_ready(drill=drill, clock=clock)
        if payload is None:
            break
        if not seen_types or seen_types[-1] is not type(payload):
            seen_types.append(type(payload))
        assert drill.submit_answer(_answer_payload(payload)) is True
        drill.update()

    assert seen_types[:2] == [TraceTest1Payload, TraceTest2Payload]


def test_trace_mixed_drill_splits_scored_time_evenly_across_segments() -> None:
    clock = FakeClock()
    drill = build_trace_mixed_tempo_drill(
        clock=clock,
        seed=55,
        difficulty=0.5,
        mode=AntDrillMode.TEMPO,
        config=TraceDrillConfig(scored_duration_s=40.0),
    )

    assert drill.scored_duration_s == pytest.approx(40.0)
    assert len(drill._segments) == 2
    assert drill._segments[0].engine._cfg.scored_duration_s == pytest.approx(20.0)
    assert drill._segments[1].engine._cfg.scored_duration_s == pytest.approx(20.0)


def test_trace_mixed_drill_aggregates_events_and_summary_across_segments() -> None:
    clock = FakeClock()
    drill = build_trace_mixed_tempo_drill(
        clock=clock,
        seed=66,
        difficulty=0.5,
        mode=AntDrillMode.TEMPO,
        config=TraceDrillConfig(scored_duration_s=12.0),
    )

    drill.start_practice()
    if drill.phase is Phase.PRACTICE_DONE:
        drill.start_scored()
    while drill.phase is Phase.PRACTICE:
        payload = _wait_for_answer_ready(drill=drill, clock=clock)
        assert payload is not None
        assert drill.submit_answer(_answer_payload(payload)) is True
        drill.update()
    if drill.phase is Phase.PRACTICE_DONE:
        drill.start_scored()

    tt1_payload = _wait_for_answer_ready(drill=drill, clock=clock)
    assert isinstance(tt1_payload, TraceTest1Payload)
    assert drill.submit_answer(_answer_payload(tt1_payload)) is True
    drill.update()
    assert drill.submit_answer("__skip_section__") is True
    drill.update()

    tt2_payload = _wait_for_answer_ready(drill=drill, clock=clock)
    assert isinstance(tt2_payload, TraceTest2Payload)
    assert drill.submit_answer(_answer_payload(tt2_payload)) is True
    drill.update()
    assert drill.submit_answer("__skip_all__") is True

    summary = drill.scored_summary()
    events = drill.events()

    assert drill.phase is Phase.RESULTS
    assert summary.attempted >= 2
    assert summary.correct >= 2
    assert summary.duration_s == pytest.approx(12.0)
    assert len(events) >= summary.attempted


def test_trace_orientation_decode_levels_l2_l5_l8_are_materially_different() -> None:
    def summarize(level: int) -> tuple[int, float, float]:
        clock = FakeClock()
        drill = build_tt1_command_switch_run_drill(
            clock=clock,
            seed=811,
            difficulty=_difficulty_for_level(level),
            mode=AntDrillMode.BUILD,
            config=TraceDrillConfig(scored_duration_s=24.0),
        )
        drill.start_practice()
        payload = _wait_for_answer_ready(drill=drill, clock=clock)
        assert isinstance(payload, TraceTest1Payload)
        return (
            len(payload.scene.blue_frames),
            float(payload.speed_multiplier),
            float(payload.answer_open_progress),
        )

    low_blue, low_speed, low_open = summarize(2)
    mid_blue, mid_speed, mid_open = summarize(5)
    high_blue, high_speed, high_open = summarize(8)

    assert low_blue < mid_blue < high_blue
    assert low_speed < mid_speed < high_speed
    assert low_open > mid_open > high_open


def test_trace_movement_recall_levels_l2_l5_l8_are_materially_different() -> None:
    def summarize(level: int) -> tuple[float, float]:
        clock = FakeClock()
        drill = build_tt2_position_recall_run_drill(
            clock=clock,
            seed=823,
            difficulty=_difficulty_for_level(level),
            mode=AntDrillMode.BUILD,
            config=TraceDrillConfig(scored_duration_s=24.0),
        )
        drill.start_practice()
        payload = _wait_for_answer_ready(drill=drill, clock=clock)
        assert isinstance(payload, TraceTest2Payload)
        ended_xs = [track.ended_screen_x for track in payload.aircraft]
        ended_altitudes = [track.ended_altitude_z for track in payload.aircraft]
        return (max(ended_xs) - min(ended_xs), max(ended_altitudes) - min(ended_altitudes))

    low_x_span, low_alt_span = summarize(2)
    mid_x_span, mid_alt_span = summarize(5)
    high_x_span, high_alt_span = summarize(8)

    assert low_x_span < mid_x_span < high_x_span
    assert high_alt_span < low_alt_span < mid_alt_span
    assert (max(low_alt_span, mid_alt_span, high_alt_span) - min(low_alt_span, mid_alt_span, high_alt_span)) >= 6.0
