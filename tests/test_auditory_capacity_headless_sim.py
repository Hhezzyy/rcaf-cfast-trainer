from __future__ import annotations

from dataclasses import dataclass

import pytest

from cfast_trainer.auditory_capacity import (
    AuditoryCapacityConfig,
    AuditoryCapacityPayload,
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


def _drive_to_phase_end(
    *,
    clock: FakeClock,
    engine: object,
    until_phase: Phase,
    max_steps: int = 6000,
) -> None:
    remembered_sequence: str | None = None

    for _ in range(max_steps):
        clock.advance(1.0 / 60.0)
        engine.update()
        snap = engine.snapshot()

        if snap.phase is until_phase:
            return

        payload = snap.payload
        if not isinstance(payload, AuditoryCapacityPayload):
            continue

        # Keep the ball centered while softly tracking the next gate line.
        target_y = 0.0
        if payload.gates:
            lead = min(payload.gates, key=lambda g: g.x_norm)
            target_y = lead.y_norm

        horizontal = max(-1.0, min(1.0, -payload.ball_x * 2.1))
        vertical = max(-1.0, min(1.0, (target_y - payload.ball_y) * 3.8))
        engine.set_control(horizontal=horizontal, vertical=vertical)

        if payload.callsign_cue is not None and payload.callsign_cue in payload.assigned_callsigns:
            engine.submit_answer("CALL")

        if payload.beep_active:
            engine.submit_answer("BEEP")

        if payload.color_command is not None:
            engine.submit_answer(f"COL:{payload.color_command}")

        if payload.sequence_display is not None:
            remembered_sequence = payload.sequence_display

        if payload.sequence_response_open and remembered_sequence is not None:
            engine.submit_answer(f"SEQ:{remembered_sequence}")
            remembered_sequence = None

    raise AssertionError(f"Phase {until_phase.value} was not reached in {max_steps} steps")


def _run_scripted_attempt(
    seed: int,
) -> tuple[tuple[float | int | None, ...], list[tuple[object, ...]]]:
    clock = FakeClock()
    engine = build_auditory_capacity_test(
        clock=clock,
        seed=seed,
        difficulty=0.58,
        config=AuditoryCapacityConfig(
            practice_duration_s=1.0,
            scored_duration_s=6.0,
            tick_hz=120.0,
            gate_interval_s=1.2,
            callsign_interval_s=1.1,
            beep_interval_s=0.95,
            color_command_interval_s=1.8,
            sequence_interval_s=2.2,
            cue_window_s=1.0,
            sequence_display_s=0.55,
            sequence_response_s=1.2,
        ),
    )

    assert engine.phase is Phase.INSTRUCTIONS

    engine.start_practice()
    assert engine.phase is Phase.PRACTICE
    _drive_to_phase_end(clock=clock, engine=engine, until_phase=Phase.PRACTICE_DONE)
    assert engine.phase is Phase.PRACTICE_DONE

    engine.start_scored()
    assert engine.phase is Phase.SCORED
    _drive_to_phase_end(clock=clock, engine=engine, until_phase=Phase.RESULTS)
    assert engine.phase is Phase.RESULTS

    summary = engine.scored_summary()
    compact_summary = (
        summary.attempted,
        summary.correct,
        round(summary.accuracy, 9),
        round(summary.throughput_per_min, 9),
        None if summary.mean_response_time_s is None else round(summary.mean_response_time_s, 9),
        round(summary.total_score, 9),
        round(summary.max_score, 9),
        round(summary.score_ratio, 9),
    )

    scored_events = [
        (
            evt.kind,
            evt.expected,
            evt.response,
            evt.is_correct,
            round(evt.score, 9),
            None if evt.response_time_s is None else round(evt.response_time_s, 9),
        )
        for evt in engine.events()
        if evt.phase is Phase.SCORED
    ]

    return compact_summary, scored_events


def test_headless_scripted_run_is_exactly_deterministic_and_produces_scored_output() -> None:
    summary_1, events_1 = _run_scripted_attempt(seed=441)
    summary_2, events_2 = _run_scripted_attempt(seed=441)

    assert summary_1 == summary_2
    assert events_1 == events_2

    attempted, correct, accuracy, _, mean_rt, total_score, max_score, score_ratio = summary_1
    assert attempted >= 10
    assert correct >= 3
    assert accuracy == pytest.approx(correct / attempted)
    assert mean_rt is not None
    assert max_score == pytest.approx(float(attempted))
    assert total_score <= max_score
    assert score_ratio == pytest.approx(total_score / max_score)
