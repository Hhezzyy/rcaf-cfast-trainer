from __future__ import annotations

from dataclasses import dataclass

import pytest

from cfast_trainer.auditory_capacity import (
    AuditoryCapacityCommandType,
    AuditoryCapacityConfig,
    AuditoryCapacityGateDirective,
    AuditoryCapacityInstructionEvent,
    AuditoryCapacityPayload,
    _LiveGate,
    build_auditory_capacity_test,
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
    payload: str | int | AuditoryCapacityGateDirective | None,
    expires_at_s: float,
    is_distractor: bool = False,
) -> AuditoryCapacityInstructionEvent:
    return AuditoryCapacityInstructionEvent(
        event_id=event_id,
        timestamp_s=at_s,
        addressed_call_sign=call_sign,
        speaker_id="lead",
        command_type=command_type,
        payload=payload,
        expires_at_s=expires_at_s,
        is_distractor=is_distractor,
    )


def test_headless_scripted_run_covers_next_gate_directive_digit_group_and_beep() -> None:
    clock = FakeClock()
    engine = build_auditory_capacity_test(
        clock=clock,
        seed=441,
        difficulty=0.58,
        config=AuditoryCapacityConfig(
            practice_enabled=False,
            practice_duration_s=0.0,
            scored_duration_s=4.2,
            run_duration_seconds=4.2,
            tick_hz=120.0,
            gate_spawn_rate=0.20,
            command_rate=0.50,
            recall_prompt_rate=0.18,
            response_window_seconds=1.15,
            sequence_display_s=0.35,
            sequence_response_s=1.10,
        ),
    )

    engine.start_practice()
    engine.start_scored()
    engine._assigned_callsigns = ("EAGLE", "RAVEN", "VIPER")
    engine._active_callsign = "ALL ASSIGNED"
    engine._briefing_duration_s = 0.0
    engine._dist_x = 0.0
    engine._dist_y = 0.0
    engine._dist_until_s = 9999.0
    engine._next_gate_at_s = 9999.0
    engine._next_state_command_at_s = 9999.0
    engine._next_gate_directive_at_s = 9999.0
    engine._next_digit_sequence_at_s = 9999.0
    engine._next_beep_at_s = 9999.0
    engine._instruction_script = (
        _event(
            event_id=1,
            at_s=0.10,
            call_sign="RAVEN",
            command_type=AuditoryCapacityCommandType.CHANGE_COLOUR,
            payload="GREEN",
            expires_at_s=0.95,
        ),
        _event(
            event_id=2,
            at_s=0.32,
            call_sign="VIPER",
            command_type=AuditoryCapacityCommandType.CHANGE_NUMBER,
            payload=4,
            expires_at_s=1.08,
        ),
        _event(
            event_id=3,
            at_s=0.56,
            call_sign="EAGLE",
            command_type=AuditoryCapacityCommandType.DIGIT_SEQUENCE,
            payload="47218",
            expires_at_s=2.05,
        ),
        _event(
            event_id=4,
            at_s=0.94,
            call_sign="RAVEN",
            command_type=AuditoryCapacityCommandType.GATE_DIRECTIVE,
            payload=AuditoryCapacityGateDirective(
                action="AVOID",
                match_kind="SHAPE",
                match_value="TRIANGLE",
            ),
            expires_at_s=1.80,
        ),
        _event(
            event_id=5,
            at_s=1.28,
            call_sign="ALL ASSIGNED",
            command_type=AuditoryCapacityCommandType.PRESS_TRIGGER,
            payload="beep",
            expires_at_s=2.13,
        ),
    )
    engine._next_instruction_index = 0

    remembered_digits = ""
    handled_instruction_ids: set[int] = set()
    recalled_for_uid: set[int] = set()
    beep_answered = False
    gate_inserted = False

    for _ in range(540):
        clock.advance(1.0 / 60.0)
        engine.update()
        snap = engine.snapshot()

        if snap.phase is Phase.RESULTS:
            break

        payload = snap.payload
        if not isinstance(payload, AuditoryCapacityPayload):
            continue

        if payload.instruction_uid is not None and payload.instruction_uid not in handled_instruction_ids:
            if payload.callsign_cue in payload.assigned_callsigns and payload.color_command is not None:
                engine.set_colour(payload.color_command)
                handled_instruction_ids.add(payload.instruction_uid)
            elif payload.callsign_cue in payload.assigned_callsigns and payload.number_command is not None:
                engine.set_number(payload.number_command)
                handled_instruction_ids.add(payload.instruction_uid)
            elif payload.callsign_cue in payload.assigned_callsigns and payload.sequence_display is not None:
                remembered_digits = payload.sequence_display
                handled_instruction_ids.add(payload.instruction_uid)

        if payload.sequence_response_open and payload.instruction_uid is not None:
            if payload.instruction_uid not in recalled_for_uid:
                engine.submit_digit_recall(remembered_digits)
                recalled_for_uid.add(payload.instruction_uid)

        if payload.beep_active and not beep_answered:
            engine.submit_answer("SPACE")
            beep_answered = True

        if not gate_inserted and payload.target_gate_action == "AVOID":
            engine._gates.append(
                _LiveGate(
                    gate_id=77,
                    x_norm=0.22,
                    y_norm=0.0,
                    color="RED",
                    shape="TRIANGLE",
                    aperture_norm=0.18,
                    scored=False,
                )
            )
            engine._bind_gate_directive_to_next_match()
            gate_inserted = True

        lead_gate = next((gate for gate in payload.gates if gate.gate_id == 77), None)
        target_y = 0.34 if lead_gate is not None else 0.0
        horizontal = max(-1.0, min(1.0, -payload.ball_x * 2.3))
        vertical = max(-1.0, min(1.0, (target_y - payload.ball_y) * 5.0))
        engine.set_control(horizontal=horizontal, vertical=vertical)

    assert engine.phase is Phase.RESULTS

    scored_events = [evt for evt in engine.events() if evt.phase is Phase.SCORED]
    recall_events = [evt for evt in scored_events if evt.kind.value == "digit_recall"]
    gate_events = [evt for evt in scored_events if evt.kind.value == "gate"]
    trigger_events = [evt for evt in scored_events if evt.kind.value == "trigger"]

    assert recall_events and recall_events[-1].is_correct is True
    assert gate_events and gate_events[-1].is_correct is True
    assert trigger_events and trigger_events[-1].is_correct is True

    summary = engine.scored_summary()
    assert summary.attempted >= 5
    assert summary.correct >= 5

    final_payload = engine._build_payload()
    assert final_payload.ball_color == "GREEN"
    assert final_payload.ball_number == 4
    assert final_payload.digit_recall_attempts == 1
    assert final_payload.digit_recall_accuracy == pytest.approx(1.0)
    assert final_payload.forbidden_gate_hits == 0
