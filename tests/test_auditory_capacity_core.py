from __future__ import annotations

from dataclasses import dataclass

import pytest

from cfast_trainer.auditory_capacity import (
    AuditoryCapacityCommandType,
    AuditoryCapacityConfig,
    AuditoryCapacityGateDirective,
    AuditoryCapacityInstructionEvent,
    AuditoryCapacityScenarioGenerator,
    AuditoryCapacityTrainingProfile,
    AuditoryCapacityTrainingSegment,
    _LiveGate,
    _PendingStateCommand,
    build_auditory_capacity_test,
    project_inside_tube,
    score_sequence_answer,
    tube_contact_ratio,
)
from cfast_trainer.cognitive_core import Phase


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


def _difficulty_ratio(level: int) -> float:
    return max(0.0, min(1.0, float(level - 1) / 9.0))


def _event(
    *,
    event_id: int,
    at_s: float,
    call_sign: str,
    command_type: AuditoryCapacityCommandType,
    payload: str | int | None,
    expires_at_s: float,
    is_distractor: bool = False,
    speaker_id: str = "lead",
) -> AuditoryCapacityInstructionEvent:
    return AuditoryCapacityInstructionEvent(
        event_id=event_id,
        timestamp_s=at_s,
        addressed_call_sign=call_sign,
        speaker_id=speaker_id,
        command_type=command_type,
        payload=payload,
        expires_at_s=expires_at_s,
        is_distractor=is_distractor,
    )


def _build_engine(*, scored_duration_s: float = 10.0):
    clock = FakeClock()
    engine = build_auditory_capacity_test(
        clock=clock,
        seed=17,
        difficulty=0.5,
        config=AuditoryCapacityConfig(
            practice_enabled=False,
            practice_duration_s=0.0,
            scored_duration_s=scored_duration_s,
            run_duration_seconds=scored_duration_s,
            tick_hz=120.0,
            gate_spawn_rate=0.20,
            command_rate=0.40,
            recall_prompt_rate=0.20,
            response_window_seconds=1.25,
        ),
    )
    engine.start_practice()
    engine.start_scored()
    engine._assigned_callsigns = ("EAGLE", "RAVEN", "VIPER")
    engine._active_callsign = "ALL ASSIGNED"
    engine._dist_x = 0.0
    engine._dist_y = 0.0
    engine._dist_until_s = 9999.0
    engine._next_gate_at_s = 9999.0
    return clock, engine


def test_generator_determinism_same_seed_same_sequence() -> None:
    cfg = AuditoryCapacityConfig(
        callsign_count=3,
        command_rate=0.35,
        recall_prompt_rate=0.08,
        response_window_seconds=2.0,
    )
    g1 = AuditoryCapacityScenarioGenerator(seed=909)
    g2 = AuditoryCapacityScenarioGenerator(seed=909)

    callsigns_1 = g1.assign_callsigns(count=cfg.callsign_count)
    callsigns_2 = g2.assign_callsigns(count=cfg.callsign_count)
    assert callsigns_1 == callsigns_2

    script_1 = g1.build_instruction_script(
        duration_s=18.0,
        difficulty=0.62,
        config=cfg,
        callsigns=callsigns_1,
        starting_callsign=callsigns_1[0],
    )
    script_2 = g2.build_instruction_script(
        duration_s=18.0,
        difficulty=0.62,
        config=cfg,
        callsigns=callsigns_2,
        starting_callsign=callsigns_2[0],
    )

    assert script_1 == script_2
    assert [evt.command_type for evt in script_1]


def test_payload_exposes_session_seed_and_phase_elapsed() -> None:
    clock, engine = _build_engine()

    clock.advance(0.25)
    engine.update()

    payload = engine.snapshot().payload
    assert payload is not None
    assert payload.session_seed == 17
    assert payload.phase_elapsed_s == pytest.approx(0.25, abs=0.02)


def test_sequence_scoring_exact_partial_and_zero() -> None:
    assert score_sequence_answer("12345", "12345") == 1.0
    assert score_sequence_answer("12345", "12335") == pytest.approx(0.8)
    assert score_sequence_answer("12345", "99999") == 0.0


def test_curved_tube_helpers_project_diagonal_points_back_to_ellipse() -> None:
    ratio = tube_contact_ratio(
        x=0.80,
        y=0.33,
        tube_half_width=0.84,
        tube_half_height=0.50,
    )
    assert ratio > 1.0

    px, py, raw_ratio = project_inside_tube(
        x=0.80,
        y=0.33,
        tube_half_width=0.84,
        tube_half_height=0.50,
    )

    assert raw_ratio == pytest.approx(ratio)
    assert tube_contact_ratio(
        x=px,
        y=py,
        tube_half_width=0.84,
        tube_half_height=0.50,
    ) == pytest.approx(0.995, rel=1e-3)


def test_circle_gate_reduces_ball_control_speed_vs_square_gate() -> None:
    clock_circle, circle_engine = _build_engine()
    clock_square, square_engine = _build_engine()

    circle_engine._gates = [
        _LiveGate(
            gate_id=11,
            x_norm=0.28,
            y_norm=0.0,
            color="RED",
            shape="CIRCLE",
            aperture_norm=0.18,
        )
    ]
    square_engine._gates = [
        _LiveGate(
            gate_id=12,
            x_norm=0.28,
            y_norm=0.0,
            color="RED",
            shape="SQUARE",
            aperture_norm=0.18,
        )
    ]
    circle_engine.set_control(horizontal=1.0, vertical=0.0)
    square_engine.set_control(horizontal=1.0, vertical=0.0)

    clock_circle.advance(0.25)
    clock_square.advance(0.25)
    circle_engine.update()
    square_engine.update()

    circle_payload = circle_engine.snapshot().payload
    square_payload = square_engine.snapshot().payload
    assert circle_payload is not None
    assert square_payload is not None
    assert abs(circle_payload.ball_x) < abs(square_payload.ball_x)


def test_briefing_starts_quiet_before_distractors_ramp_up() -> None:
    clock = FakeClock()
    engine = build_auditory_capacity_test(
        clock=clock,
        seed=31,
        difficulty=0.5,
        config=AuditoryCapacityConfig(
            practice_enabled=True,
            practice_duration_s=20.0,
            scored_duration_s=20.0,
            run_duration_seconds=20.0,
        ),
    )
    engine.start_practice()

    clock.advance(0.5)
    engine.update()
    early = engine.snapshot().payload
    assert early is not None
    assert early.briefing_active is True
    assert early.background_noise_level == pytest.approx(0.0)
    assert early.beep_active is False

    for _ in range(10):
        clock.advance(0.5)
        engine.update()
    later = engine.snapshot().payload
    assert later is not None
    assert later.briefing_active is False
    assert later.background_noise_level > 0.0


def test_custom_scored_segments_expose_active_channel_focus_and_metadata() -> None:
    clock = FakeClock()
    engine = build_auditory_capacity_test(
        clock=clock,
        seed=57,
        difficulty=0.5,
        config=AuditoryCapacityConfig(
            practice_enabled=False,
            practice_duration_s=0.0,
            scored_duration_s=1.2,
            run_duration_seconds=1.2,
        ),
        scored_segments=(
            AuditoryCapacityTrainingSegment(
                label="Gate Flight",
                duration_s=1.2,
                active_channels=("gates",),
                profile=AuditoryCapacityTrainingProfile(
                    enable_state_commands=False,
                    enable_gate_directives=False,
                    enable_digit_sequences=False,
                    enable_trigger_cues=False,
                    enable_distractors=False,
                    noise_level_scale=0.0,
                    distortion_level_scale=0.0,
                ),
            ),
        ),
    )
    engine.start_practice()
    engine.start_scored()

    for _ in range(4):
        clock.advance(0.25)
        engine.update()

    payload = engine.snapshot().payload
    assert payload is not None
    assert payload.active_channels == ("gates",)
    assert payload.segment_label == "Gate Flight"
    assert payload.segment_index == 1
    assert payload.segment_total == 1
    assert payload.color_command is None
    assert payload.number_command is None
    assert payload.sequence_display is None
    assert payload.sequence_response_open is False
    assert payload.beep_active is False
    assert payload.next_color_in_s is None
    assert payload.next_sequence_in_s is None
    assert payload.next_beep_in_s is None
    assert payload.background_noise_level == pytest.approx(0.0)


def test_custom_scored_segments_advance_metadata_as_segments_change() -> None:
    clock = FakeClock()
    engine = build_auditory_capacity_test(
        clock=clock,
        seed=58,
        difficulty=0.5,
        config=AuditoryCapacityConfig(
            practice_enabled=False,
            practice_duration_s=0.0,
            scored_duration_s=1.0,
            run_duration_seconds=1.0,
        ),
        scored_segments=(
            AuditoryCapacityTrainingSegment(
                label="Gate Flight",
                duration_s=0.4,
                active_channels=("gates",),
            ),
            AuditoryCapacityTrainingSegment(
                label="Digit Recall",
                duration_s=0.6,
                active_channels=("gates", "digit_recall"),
            ),
        ),
    )
    engine.start_practice()
    engine.start_scored()

    first = engine.snapshot().payload
    assert first is not None
    assert first.segment_label == "Gate Flight"
    assert first.segment_index == 1
    assert first.segment_total == 2

    clock.advance(0.55)
    engine.update()
    second = engine.snapshot().payload
    assert second is not None
    assert second.segment_label == "Digit Recall"
    assert second.segment_index == 2
    assert second.segment_total == 2
    assert second.active_channels == ("gates", "digit_recall")


@pytest.mark.parametrize(
    ("level", "expected_lengths"),
    (
        (1, {4, 5}),
        (4, {5, 6}),
        (6, {6, 7}),
        (8, {7, 8}),
        (9, {8, 9}),
        (10, {10}),
    ),
)
def test_runtime_digit_sequences_follow_default_difficulty_buckets(
    level: int,
    expected_lengths: set[int],
) -> None:
    clock = FakeClock()
    engine = build_auditory_capacity_test(
        clock=clock,
        seed=41 + level,
        difficulty=_difficulty_ratio(level),
    )
    engine.start_practice()
    engine.start_scored()

    lengths = {
        len("".join(ch for ch in str(engine._build_runtime_digit_sequence_event(timestamp_s=1.0 + idx).payload) if ch.isdigit()))
        for idx in range(24)
    }
    assert lengths <= expected_lengths
    assert min(lengths) >= min(expected_lengths)
    assert max(lengths) <= max(expected_lengths)


def test_default_profile_adds_fourth_callsign_at_level_eight_and_above() -> None:
    low_clock = FakeClock()
    high_clock = FakeClock()
    low_engine = build_auditory_capacity_test(
        clock=low_clock,
        seed=71,
        difficulty=_difficulty_ratio(7),
    )
    high_engine = build_auditory_capacity_test(
        clock=high_clock,
        seed=72,
        difficulty=_difficulty_ratio(8),
    )

    assert len(low_engine._assigned_callsigns) == 3
    assert len(high_engine._assigned_callsigns) == 4


def test_default_profile_gate_interval_shrinks_to_top_end_pressure() -> None:
    low_engine = build_auditory_capacity_test(
        clock=FakeClock(),
        seed=81,
        difficulty=_difficulty_ratio(1),
    )
    mid_engine = build_auditory_capacity_test(
        clock=FakeClock(),
        seed=82,
        difficulty=_difficulty_ratio(6),
    )
    high_engine = build_auditory_capacity_test(
        clock=FakeClock(),
        seed=83,
        difficulty=_difficulty_ratio(10),
    )

    assert low_engine._effective_gate_interval_s() == pytest.approx(4.2, abs=0.05)
    assert high_engine._effective_gate_interval_s() == pytest.approx(1.6, abs=0.05)
    assert low_engine._effective_gate_interval_s() > mid_engine._effective_gate_interval_s()
    assert mid_engine._effective_gate_interval_s() > high_engine._effective_gate_interval_s()


def test_default_profile_payload_exposes_split_background_and_instructor_mix() -> None:
    engine = build_auditory_capacity_test(
        clock=FakeClock(),
        seed=91,
        difficulty=_difficulty_ratio(10),
    )
    engine.start_practice()
    engine.start_scored()
    engine._phase_duration_s = 40.0
    engine._briefing_duration_s = 4.0
    engine._sim_elapsed_s = 20.0

    payload = engine.snapshot().payload
    assert payload is not None
    assert len(payload.assigned_callsigns) == 4
    assert payload.background_noise_level > 0.0
    assert payload.background_distortion_level > 0.0
    assert payload.distortion_level == pytest.approx(payload.background_distortion_level)
    assert payload.instructor_noise_level > payload.background_noise_level
    assert payload.instructor_distortion_level > payload.background_distortion_level
    assert payload.instructor_rate_wpm == 273
    assert payload.ambient_layer_target == 3


def test_next_gate_directive_binds_replaces_and_clears_after_resolution() -> None:
    _, engine = _build_engine()
    engine._gates = [
        _LiveGate(
            gate_id=11,
            x_norm=0.48,
            y_norm=0.0,
            color="GREEN",
            shape="SQUARE",
            aperture_norm=0.18,
        ),
        _LiveGate(
            gate_id=12,
            x_norm=0.24,
            y_norm=0.0,
            color="RED",
            shape="TRIANGLE",
            aperture_norm=0.18,
        ),
    ]

    avoid_triangle = AuditoryCapacityInstructionEvent(
        event_id=10,
        timestamp_s=0.1,
        addressed_call_sign="RAVEN",
        speaker_id="lead",
        command_type=AuditoryCapacityCommandType.GATE_DIRECTIVE,
        payload=AuditoryCapacityGateDirective(
            action="AVOID",
            match_kind="SHAPE",
            match_value="TRIANGLE",
        ),
        expires_at_s=1.0,
        is_distractor=False,
    )
    engine._activate_instruction(avoid_triangle)
    assert engine._active_gate_directive is not None
    assert engine._active_gate_directive.target_gate_id == 12

    pass_green = AuditoryCapacityInstructionEvent(
        event_id=11,
        timestamp_s=0.2,
        addressed_call_sign="EAGLE",
        speaker_id="lead",
        command_type=AuditoryCapacityCommandType.GATE_DIRECTIVE,
        payload=AuditoryCapacityGateDirective(
            action="PASS",
            match_kind="COLOR",
            match_value="GREEN",
        ),
        expires_at_s=1.2,
        is_distractor=False,
    )
    engine._activate_instruction(pass_green)
    assert engine._active_gate_directive is not None
    assert engine._active_gate_directive.target_gate_id == 11

    engine._ball_y = 0.0
    engine._outside_tube = False
    engine._gates = [
        _LiveGate(
            gate_id=11,
            x_norm=0.0,
            y_norm=0.0,
            color="GREEN",
            shape="SQUARE",
            aperture_norm=0.18,
            scored=False,
        )
    ]
    engine._update_gates(0.0)

    assert engine._active_gate_directive is None


def test_beep_task_accepts_one_press_and_rejects_early_late_and_extra() -> None:
    _, engine = _build_engine()

    assert engine.submit_answer("SPACE") is True
    assert engine.events()[-1].kind.value == "false_response"

    beep = AuditoryCapacityInstructionEvent(
        event_id=99,
        timestamp_s=0.4,
        addressed_call_sign="ALL ASSIGNED",
        speaker_id="lead",
        command_type=AuditoryCapacityCommandType.PRESS_TRIGGER,
        payload="beep",
        expires_at_s=1.25,
        is_distractor=False,
    )
    engine._activate_instruction(beep)
    engine._sim_elapsed_s = 0.52
    assert engine.submit_answer("SPACE") is True
    assert engine.events()[-1].kind.value == "trigger"
    assert engine.events()[-1].is_correct is True

    assert engine.submit_answer("SPACE") is True
    assert engine.events()[-1].kind.value == "false_response"

    engine._activate_instruction(beep)
    engine._sim_elapsed_s = 1.40
    engine._update_instruction_channel()
    assert engine.events()[-1].kind.value == "trigger"
    assert engine.events()[-1].is_correct is False


def test_briefing_script_uses_concise_instruction_copy() -> None:
    _, engine = _build_engine()

    lines = [str(event.payload) for event in engine._build_briefing_script(duration_s=4.0)]

    assert lines == [
        "Your call signs are EAGLE, RAVEN, VIPER. Respond only to those call signs.",
        (
            "Use Q W E R for colour, keypad numbers for the ball, and type "
            f"{engine._instruction_digit_range_label()} sequences with Enter."
        ),
        "Press trigger or Space on the beep. Gate instructions apply to the next matching gate.",
    ]


def test_gate_directive_instruction_text_drops_filler_language() -> None:
    _, engine = _build_engine()
    event = AuditoryCapacityInstructionEvent(
        event_id=10,
        timestamp_s=0.1,
        addressed_call_sign="RAVEN",
        speaker_id="lead",
        command_type=AuditoryCapacityCommandType.GATE_DIRECTIVE,
        payload=AuditoryCapacityGateDirective(
            action="AVOID",
            match_kind="SHAPE",
            match_value="TRIANGLE",
        ),
        expires_at_s=1.0,
        is_distractor=False,
    )

    text = engine._instruction_text(event)

    assert "Do not go through it" not in text
    assert text in {
        "RAVEN. Avoid the next triangle gate.",
        "RAVEN. Skip the next triangle gate.",
    }


def test_distractor_envelope_falls_faster_than_it_rises() -> None:
    _, engine = _build_engine(scored_duration_s=12.0)
    engine._phase_duration_s = 12.0
    engine._briefing_duration_s = 3.0

    engine._sim_elapsed_s = 4.5
    rising = engine._distractor_envelope()
    engine._sim_elapsed_s = 8.4
    peak = engine._distractor_envelope()
    engine._sim_elapsed_s = 10.8
    falling = engine._distractor_envelope()

    assert 0.0 < rising < peak
    assert 0.0 < falling < peak
    assert (peak - falling) > (rising - 0.0)


def test_timer_boundary_transitions_to_results_and_rejects_submit() -> None:
    clock = FakeClock()
    engine = build_auditory_capacity_test(
        clock=clock,
        seed=7,
        difficulty=0.5,
        config=AuditoryCapacityConfig(
            practice_enabled=False,
            practice_duration_s=0.0,
            scored_duration_s=2.0,
            run_duration_seconds=2.0,
            tick_hz=120.0,
            gate_spawn_rate=0.20,
            command_rate=0.30,
            recall_prompt_rate=0.10,
            response_window_seconds=1.0,
        ),
    )

    engine.start_practice()
    assert engine.phase is Phase.PRACTICE_DONE

    engine.start_scored()
    assert engine.phase is Phase.SCORED
    assert engine.can_exit() is False

    clock.advance(2.0)
    engine.update()

    assert engine.phase is Phase.RESULTS
    assert engine.can_exit() is True
    assert engine.submit_answer("COL:GREEN") is False


def test_assigned_callsign_commands_update_state_and_external_distractor_is_scored() -> None:
    clock, engine = _build_engine()
    engine._instruction_script = (
        _event(
            event_id=1,
            at_s=0.10,
            call_sign="RAVEN",
            command_type=AuditoryCapacityCommandType.CHANGE_COLOUR,
            payload="GREEN",
            expires_at_s=1.00,
        ),
        _event(
            event_id=2,
            at_s=0.30,
            call_sign="COBRA",
            command_type=AuditoryCapacityCommandType.CHANGE_NUMBER,
            payload=7,
            expires_at_s=1.10,
            is_distractor=True,
        ),
        _event(
            event_id=3,
            at_s=0.52,
            call_sign="VIPER",
            command_type=AuditoryCapacityCommandType.CHANGE_NUMBER,
            payload=4,
            expires_at_s=1.30,
        ),
    )
    engine._next_instruction_index = 0

    clock.advance(0.11)
    engine.update()
    assert engine.snapshot().payload is not None
    assert engine.snapshot().payload.color_command == "GREEN"
    assert engine.set_colour("GREEN") is True
    assert engine.snapshot().payload.ball_color == "GREEN"

    clock.advance(0.22)
    engine.update()
    assert engine.set_number(7) is True
    assert engine.snapshot().payload.ball_number == 1

    clock.advance(0.25)
    engine.update()
    assert engine.snapshot().payload.number_command == 4
    assert engine.set_number(4) is True

    payload = engine.snapshot().payload
    assert payload is not None
    assert payload.ball_color == "GREEN"
    assert payload.ball_number == 4
    assert payload.assigned_callsigns == ("EAGLE", "RAVEN", "VIPER")
    assert payload.correct_command_executions == 2
    assert payload.false_responses_to_distractors == 1


def test_logical_colour_persists_while_visual_colour_flash_returns_to_white() -> None:
    _, engine = _build_engine()
    engine._pending_state_command = _PendingStateCommand(
        event=_event(
            event_id=41,
            at_s=0.1,
            call_sign="RAVEN",
            command_type=AuditoryCapacityCommandType.CHANGE_COLOUR,
            payload="GREEN",
            expires_at_s=1.0,
        ),
        expected_color="GREEN",
        expected_number=None,
    )

    assert engine.set_colour("GREEN") is True
    payload = engine.snapshot().payload
    assert payload is not None
    assert payload.ball_color == "GREEN"
    assert payload.ball_visual_color == "GREEN"
    assert payload.ball_visual_strength > 0.9

    engine._sim_elapsed_s += 0.35
    payload = engine.snapshot().payload
    assert payload is not None
    assert payload.ball_color == "GREEN"
    assert payload.ball_visual_color == "WHITE"
    assert payload.ball_visual_strength == pytest.approx(0.0)


def test_ball_visual_feedback_flashes_green_on_points_and_red_on_errors() -> None:
    _, engine = _build_engine()
    engine._pending_state_command = _PendingStateCommand(
        event=_event(
            event_id=51,
            at_s=0.1,
            call_sign="VIPER",
            command_type=AuditoryCapacityCommandType.CHANGE_NUMBER,
            payload=4,
            expires_at_s=1.0,
        ),
        expected_color=None,
        expected_number=4,
    )

    assert engine.set_number(4) is True
    payload = engine.snapshot().payload
    assert payload is not None
    assert payload.ball_visual_color == "GREEN"
    assert payload.ball_visual_strength > 0.9

    assert engine.submit_answer("BLUE") is True
    payload = engine.snapshot().payload
    assert payload is not None
    assert payload.ball_visual_color == "RED"
    assert payload.ball_visual_strength > 0.9


def test_digit_accumulation_and_delayed_recall_scores_exact() -> None:
    clock, engine = _build_engine()
    engine._instruction_script = (
        _event(
            event_id=1,
            at_s=0.10,
            call_sign="VIPER",
            command_type=AuditoryCapacityCommandType.DIGIT_APPEND,
            payload="4",
            expires_at_s=0.60,
        ),
        _event(
            event_id=2,
            at_s=0.32,
            call_sign="RAVEN",
            command_type=AuditoryCapacityCommandType.DIGIT_APPEND,
            payload="7",
            expires_at_s=0.82,
        ),
        _event(
            event_id=3,
            at_s=0.74,
            call_sign="EAGLE",
            command_type=AuditoryCapacityCommandType.RECALL_DIGITS,
            payload=None,
            expires_at_s=2.10,
        ),
    )
    engine._next_instruction_index = 0

    clock.advance(0.11)
    engine.update()
    payload = engine.snapshot().payload
    assert payload is not None
    assert payload.sequence_display == "4"

    clock.advance(0.25)
    engine.update()
    payload = engine.snapshot().payload
    assert payload is not None
    assert payload.sequence_display == "7"

    clock.advance(0.45)
    engine.update()
    clock.advance(0.50)
    engine.update()
    clock.advance(0.35)
    engine.update()
    payload = engine.snapshot().payload
    assert payload is not None
    assert payload.sequence_response_open is True
    assert engine.submit_digit_recall("47") is True

    payload = engine.snapshot().payload
    assert payload is not None
    assert payload.digit_recall_attempts == 1
    assert payload.digit_recall_accuracy == pytest.approx(1.0)


def test_instructions_show_assigned_callsigns_before_runtime_starts() -> None:
    clock = FakeClock()
    engine = build_auditory_capacity_test(
        clock=clock,
        seed=23,
        difficulty=0.5,
        config=AuditoryCapacityConfig(
            practice_enabled=True,
            practice_duration_s=5.0,
            scored_duration_s=5.0,
            run_duration_seconds=5.0,
            gate_spawn_rate=0.20,
            command_rate=0.30,
            recall_prompt_rate=0.10,
            response_window_seconds=1.0,
        ),
    )

    prompt = engine.current_prompt()
    for callsign in engine._assigned_callsigns:
        assert callsign in prompt


def test_forbidden_gate_rule_changes_gate_scoring() -> None:
    _, engine = _build_engine()
    engine._forbidden_gate_shape = "TRIANGLE"
    engine._ball_x = 0.0
    engine._ball_y = 0.0
    engine._outside_tube = False
    engine._gates = [
        _LiveGate(
            gate_id=1,
            x_norm=0.0,
            y_norm=0.0,
            color="GREEN",
            shape="CIRCLE",
            aperture_norm=0.20,
            scored=False,
        ),
        _LiveGate(
            gate_id=2,
            x_norm=0.0,
            y_norm=0.0,
            color="RED",
            shape="TRIANGLE",
            aperture_norm=0.20,
            scored=False,
        ),
    ]

    engine._update_gates(0.0)

    payload = engine.snapshot().payload
    assert payload is not None
    assert payload.gate_hits == 1
    assert payload.gate_misses == 1
    assert payload.forbidden_gate_hits == 1
    assert payload.gates == ()


def test_gate_scoring_plane_does_not_move_with_ball_horizontal_offset() -> None:
    _, engine = _build_engine()
    engine._ball_x = 0.72
    engine._ball_y = 0.0
    engine._outside_tube = False
    engine._gates = [
        _LiveGate(
            gate_id=9,
            x_norm=0.12,
            y_norm=0.0,
            color="GREEN",
            shape="SQUARE",
            aperture_norm=0.20,
            scored=False,
        )
    ]

    engine._update_gates(0.0)

    payload = engine.snapshot().payload
    assert payload is not None
    assert payload.gate_hits == 0
    assert payload.gate_misses == 0
    assert len(payload.gates) == 1
    assert payload.gates[0].gate_id == 9

    engine._gates[0].x_norm = 0.0
    engine._update_gates(0.0)

    payload = engine.snapshot().payload
    assert payload is not None
    assert payload.gate_hits == 1
    assert payload.gate_misses == 0
    assert payload.gates == ()
