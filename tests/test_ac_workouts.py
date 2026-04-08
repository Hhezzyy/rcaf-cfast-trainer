from __future__ import annotations

from dataclasses import dataclass

from cfast_trainer.ant_drills import AntDrillMode
from cfast_trainer.ant_workouts import (
    AntWorkoutBlockPlan,
    AntWorkoutPlan,
    AntWorkoutSession,
    AntWorkoutStage,
)
from cfast_trainer.auditory_capacity import AuditoryCapacityPayload
from cfast_trainer.results import attempt_result_from_engine
from cfast_trainer.ac_workouts import build_ac_workout_plan


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _build_small_ac_workout_plan() -> AntWorkoutPlan:
    return AntWorkoutPlan(
        code="auditory_capacity_workout",
        title="Auditory Workout Smoke",
        description="Short deterministic workout for tests.",
        notes=("Reflections are untimed.",),
        blocks=(
            AntWorkoutBlockPlan(
                block_id="gate-anchor",
                label="Gate Anchor",
                description="Warm-up gate flight block.",
                focus_skills=("Psychomotor gate flight",),
                drill_code="ac_gate_anchor",
                mode=AntDrillMode.BUILD,
                duration_min=0.25,
            ),
            AntWorkoutBlockPlan(
                block_id="mixed",
                label="Mixed Tempo",
                description="Mixed auditory block.",
                focus_skills=("Mixed auditory tempo",),
                drill_code="ac_mixed_tempo",
                mode=AntDrillMode.TEMPO,
                duration_min=0.25,
            ),
        ),
    )


def _run_current_block(session: AntWorkoutSession, clock: FakeClock) -> None:
    remembered: dict[int, str] = {}
    handled_commands: set[int] = set()
    recalled: set[int] = set()
    beep_answered: set[int] = set()

    while session.stage is AntWorkoutStage.BLOCK:
        engine = session.current_engine()
        assert engine is not None
        snap = engine.snapshot()
        payload = snap.payload
        if isinstance(payload, AuditoryCapacityPayload):
            instruction_uid = payload.instruction_uid
            if instruction_uid is not None and instruction_uid not in handled_commands:
                if payload.color_command is not None:
                    engine.set_colour(payload.color_command)
                    handled_commands.add(instruction_uid)
                elif payload.number_command is not None:
                    engine.set_number(payload.number_command)
                    handled_commands.add(instruction_uid)

            if instruction_uid is not None and payload.sequence_display is not None:
                remembered[instruction_uid] = payload.sequence_display
            if (
                instruction_uid is not None
                and payload.sequence_response_open
                and instruction_uid not in recalled
                and instruction_uid in remembered
            ):
                session.submit_answer(remembered[instruction_uid])
                recalled.add(instruction_uid)

            if payload.beep_active and instruction_uid is not None and instruction_uid not in beep_answered:
                session.submit_answer("SPACE")
                beep_answered.add(instruction_uid)

            target_y = payload.gates[0].y_norm if payload.gates else 0.0
            engine.set_control(
                horizontal=max(-1.0, min(1.0, -payload.ball_x * 2.0)),
                vertical=max(-1.0, min(1.0, (target_y - payload.ball_y) * 5.0)),
            )
        clock.advance(0.25)
        session.update()


def _complete_small_ac_workout(clock: FakeClock) -> AntWorkoutSession:
    session = AntWorkoutSession(
        clock=clock,
        seed=911,
        plan=_build_small_ac_workout_plan(),
        starting_level=5,
    )
    session.activate()
    session.append_text("stay smooth on the controls")
    session.activate()
    session.append_text("keep call sign filtering clean")
    session.activate()
    session.activate()
    _run_current_block(session, clock)
    assert session.stage is AntWorkoutStage.BLOCK_RESULTS
    session.activate()
    assert session.stage is AntWorkoutStage.BLOCK_SETUP
    session.activate()
    _run_current_block(session, clock)
    assert session.stage is AntWorkoutStage.BLOCK_RESULTS
    session.activate()
    assert session.stage is AntWorkoutStage.POST_REFLECTION
    session.append_text("gates stayed calmer than expected")
    session.activate()
    session.append_text("mixed channel rhythm held together")
    session.activate()
    assert session.stage is AntWorkoutStage.RESULTS
    return session


def test_ac_workout_session_runs_to_results() -> None:
    clock = FakeClock()
    session = _complete_small_ac_workout(clock)
    summary = session.scored_summary()
    result = attempt_result_from_engine(session, test_code="auditory_capacity_workout")

    assert summary.workout_code == "auditory_capacity_workout"
    assert summary.completed_blocks == 2
    assert summary.attempted > 0
    assert result.metrics["workout_code"] == "auditory_capacity_workout"
    assert "pre_focus_one" not in result.metrics
    assert "post_next_rule" not in result.metrics


def test_real_ac_workout_matches_standard_90_minute_structure() -> None:
    plan = build_ac_workout_plan()

    assert plan.scored_duration_s == 90.0 * 60.0
    assert tuple(block.drill_code for block in plan.blocks) == (
        "ac_gate_anchor",
        "ac_state_command_prime",
        "ac_gate_directive_run",
        "ac_digit_sequence_prime",
        "ac_trigger_cue_anchor",
        "ac_callsign_filter_run",
        "ac_mixed_tempo",
        "ac_pressure_run",
    )
    assert {
        "Psychomotor gate flight",
        "State-command filtering",
        "Next-gate directives",
        "Digit recall",
        "Trigger response",
        "Pressure tolerance",
    }.issubset(set(plan.focus_skills))
