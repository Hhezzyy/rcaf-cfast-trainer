from __future__ import annotations

from dataclasses import dataclass

import pytest

from cfast_trainer.ac_drills import (
    AC_CHANNEL_ORDER,
    AcDrillConfig,
    build_ac_callsign_filter_run_drill,
    build_ac_digit_sequence_prime_drill,
    build_ac_gate_anchor_drill,
    build_ac_gate_directive_run_drill,
    build_ac_mixed_tempo_drill,
    build_ac_pressure_run_drill,
    build_ac_state_command_prime_drill,
    build_ac_trigger_cue_anchor_drill,
)
from cfast_trainer.ant_drills import AntDrillMode
from cfast_trainer.auditory_capacity import AuditoryCapacityPayload
from cfast_trainer.cognitive_core import Phase


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _run_drill(drill, clock: FakeClock, *, duration_s: float = 3.0) -> tuple[list[tuple], object]:
    drill.start_practice()
    remembered: dict[int, str] = {}
    handled_commands: set[int] = set()
    recalled: set[int] = set()
    beep_answered: set[int] = set()
    frames: list[tuple] = []

    steps = max(1, int(round(duration_s / 0.25))) + 4
    for _ in range(steps):
        if drill.phase is not Phase.SCORED:
            break
        snap = drill.snapshot()
        payload = snap.payload
        assert isinstance(payload, AuditoryCapacityPayload)

        instruction_uid = payload.instruction_uid
        if instruction_uid is not None and instruction_uid not in handled_commands:
            if payload.color_command is not None:
                drill.set_colour(payload.color_command)
                handled_commands.add(instruction_uid)
            elif payload.number_command is not None:
                drill.set_number(payload.number_command)
                handled_commands.add(instruction_uid)

        if instruction_uid is not None and payload.sequence_display is not None:
            remembered[instruction_uid] = payload.sequence_display
        if (
            instruction_uid is not None
            and payload.sequence_response_open
            and instruction_uid not in recalled
            and instruction_uid in remembered
        ):
            drill.submit_answer(remembered[instruction_uid])
            recalled.add(instruction_uid)

        if payload.beep_active and instruction_uid is not None and instruction_uid not in beep_answered:
            drill.submit_answer("SPACE")
            beep_answered.add(instruction_uid)

        target_y = payload.gates[0].y_norm if payload.gates else 0.0
        drill.set_control(
            horizontal=max(-1.0, min(1.0, -payload.ball_x * 2.0)),
            vertical=max(-1.0, min(1.0, (target_y - payload.ball_y) * 5.0)),
        )
        frames.append(
            (
                snap.phase.value,
                payload.active_channels,
                payload.segment_label,
                payload.segment_index,
                payload.segment_total,
                payload.instruction_command_type,
                payload.color_command,
                payload.number_command,
                payload.sequence_display,
                payload.beep_active,
                len(payload.gates),
                round(payload.ball_x, 4),
                round(payload.ball_y, 4),
            )
        )
        clock.advance(0.25)
        drill.update()
    return frames, drill.scored_summary()


@pytest.mark.parametrize(
    ("builder", "duration"),
    (
        (build_ac_gate_anchor_drill, 3.0),
        (build_ac_state_command_prime_drill, 3.0),
        (build_ac_gate_directive_run_drill, 3.0),
        (build_ac_digit_sequence_prime_drill, 3.0),
        (build_ac_trigger_cue_anchor_drill, 3.0),
        (build_ac_callsign_filter_run_drill, 3.0),
        (build_ac_mixed_tempo_drill, 3.0),
        (build_ac_pressure_run_drill, 3.0),
    ),
)
def test_ac_drills_are_deterministic_for_same_seed_and_control_script(builder, duration) -> None:
    c1 = FakeClock()
    c2 = FakeClock()
    d1 = builder(
        clock=c1,
        seed=515,
        difficulty=0.55,
        mode=AntDrillMode.BUILD,
        config=AcDrillConfig(scored_duration_s=duration),
    )
    d2 = builder(
        clock=c2,
        seed=515,
        difficulty=0.55,
        mode=AntDrillMode.BUILD,
        config=AcDrillConfig(scored_duration_s=duration),
    )

    frames1, summary1 = _run_drill(d1, c1, duration_s=duration)
    frames2, summary2 = _run_drill(d2, c2, duration_s=duration)

    assert frames1 == frames2
    assert summary1 == summary2


@pytest.mark.parametrize(
    ("builder", "expected_channels"),
    (
        (build_ac_gate_anchor_drill, ("gates",)),
        (build_ac_state_command_prime_drill, ("gates", "state_commands")),
        (build_ac_gate_directive_run_drill, ("gates", "gate_directives", "distractors")),
        (build_ac_digit_sequence_prime_drill, ("gates", "digit_recall")),
        (build_ac_trigger_cue_anchor_drill, ("gates", "trigger")),
        (
            build_ac_callsign_filter_run_drill,
            ("gates", "state_commands", "gate_directives", "distractors"),
        ),
    ),
)
def test_ac_focused_drills_emit_expected_active_channels(builder, expected_channels) -> None:
    clock = FakeClock()
    drill = builder(
        clock=clock,
        seed=91,
        difficulty=0.5,
        mode=AntDrillMode.BUILD,
        config=AcDrillConfig(scored_duration_s=2.0),
    )
    drill.start_practice()
    snap = drill.snapshot()
    payload = snap.payload
    assert isinstance(payload, AuditoryCapacityPayload)
    assert payload.active_channels == expected_channels


def test_ac_mixed_tempo_repeats_fixed_six_segment_cycle() -> None:
    clock = FakeClock()
    drill = build_ac_mixed_tempo_drill(
        clock=clock,
        seed=17,
        difficulty=0.5,
        mode=AntDrillMode.TEMPO,
        config=AcDrillConfig(scored_duration_s=550.0),
    )
    drill.start_practice()

    observed: list[str] = []
    for _ in range(1200):
        snap = drill.snapshot()
        payload = snap.payload
        assert isinstance(payload, AuditoryCapacityPayload)
        if not observed or observed[-1] != payload.segment_label:
            observed.append(payload.segment_label)
        if len(observed) >= 7:
            break
        clock.advance(0.5)
        drill.update()

    assert observed[:7] == [
        "Gate Flight",
        "State Commands",
        "Gate Directives",
        "Digit Recall",
        "Trigger + Callsign Filter",
        "Full Mixed",
        "Gate Flight",
    ]


def test_ac_pressure_run_keeps_all_channels_active() -> None:
    clock = FakeClock()
    drill = build_ac_pressure_run_drill(
        clock=clock,
        seed=37,
        difficulty=0.5,
        mode=AntDrillMode.STRESS,
        config=AcDrillConfig(scored_duration_s=5.0),
    )
    drill.start_practice()

    for _ in range(4):
        payload = drill.snapshot().payload
        assert isinstance(payload, AuditoryCapacityPayload)
        assert payload.active_channels == AC_CHANNEL_ORDER
        clock.advance(0.5)
        drill.update()


@pytest.mark.parametrize(
    ("builder", "expected_snippets"),
    (
        (
            build_ac_gate_directive_run_drill,
            (
                "Respond only to your assigned call signs while you follow gate directives.",
                "Gate directives apply to the next matching gate only.",
            ),
        ),
        (
            build_ac_callsign_filter_run_drill,
            (
                "Respond only to your assigned call signs while you fly gates, state commands, and gate directives.",
                "Gate directives apply to the next matching gate.",
            ),
        ),
        (
            build_ac_pressure_run_drill,
            (
                "Respond only to your assigned call signs, treat gate directives as next-match only, and recover immediately after misses.",
            ),
        ),
    ),
)
def test_ac_drill_intro_copy_matches_live_rule_set(builder, expected_snippets) -> None:
    clock = FakeClock()
    drill = builder(
        clock=clock,
        seed=77,
        difficulty=0.5,
        mode=AntDrillMode.BUILD,
        config=AcDrillConfig(scored_duration_s=2.0),
    )

    prompt = drill.snapshot().prompt

    for snippet in expected_snippets:
        assert snippet in prompt


@pytest.mark.parametrize(
    ("builder", "required", "forbidden"),
    (
        (
            build_ac_gate_anchor_drill,
            ("WASD/arrows or HOTAS fly",),
            ("Q/W/E/R colour", "digits+Enter recall", "Space/trigger beep"),
        ),
        (
            build_ac_state_command_prime_drill,
            ("WASD/arrows or HOTAS fly", "Q/W/E/R colour", "keypad 0-9 number"),
            ("digits+Enter recall", "Space/trigger beep", "assigned call signs only"),
        ),
        (
            build_ac_gate_directive_run_drill,
            ("WASD/arrows or HOTAS fly", "assigned call signs only", "next matching gate only"),
            ("Q/W/E/R colour", "digits+Enter recall", "Space/trigger beep"),
        ),
        (
            build_ac_digit_sequence_prime_drill,
            ("WASD/arrows or HOTAS fly", "digits+Enter recall"),
            ("Q/W/E/R colour", "Space/trigger beep", "assigned call signs only"),
        ),
        (
            build_ac_trigger_cue_anchor_drill,
            ("WASD/arrows or HOTAS fly", "Space/trigger beep"),
            ("Q/W/E/R colour", "digits+Enter recall", "assigned call signs only"),
        ),
        (
            build_ac_callsign_filter_run_drill,
            (
                "WASD/arrows or HOTAS fly",
                "Q/W/E/R colour",
                "keypad 0-9 number",
                "assigned call signs only",
                "next matching gate only",
            ),
            ("digits+Enter recall", "Space/trigger beep"),
        ),
    ),
)
def test_ac_focused_drill_input_hint_only_advertises_live_controls(
    builder,
    required,
    forbidden,
) -> None:
    clock = FakeClock()
    drill = builder(
        clock=clock,
        seed=41,
        difficulty=0.5,
        mode=AntDrillMode.BUILD,
        config=AcDrillConfig(scored_duration_s=2.0),
    )
    drill.start_practice()

    hint = drill.snapshot().input_hint

    for token in required:
        assert token in hint
    for token in forbidden:
        assert token not in hint
