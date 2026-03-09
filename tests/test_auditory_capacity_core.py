from __future__ import annotations

from dataclasses import dataclass

import pytest

from cfast_trainer.auditory_capacity import (
    AuditoryCapacityCommandType,
    AuditoryCapacityConfig,
    AuditoryCapacityInstructionEvent,
    AuditoryCapacityScenarioGenerator,
    _LiveGate,
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
