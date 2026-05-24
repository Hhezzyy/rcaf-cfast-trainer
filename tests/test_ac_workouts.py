from __future__ import annotations

from dataclasses import dataclass

from cfast_trainer.ant_drills import AntDrillMode
from cfast_trainer.ant_workouts import (
    AntWorkoutBlockPlan,
    AntWorkoutPlan,
    AntWorkoutSession,
    AntWorkoutStage,
)
from cfast_trainer.godot_owned import GodotOwnedPayload
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
        notes=("Block setup is untimed.",),
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


def _complete_current_godot_block(session: AntWorkoutSession, clock: FakeClock) -> None:
    engine = session.current_engine()
    assert engine is not None
    payload = engine.snapshot().payload
    assert isinstance(payload, GodotOwnedPayload)
    assert payload.spec.kind == "auditory_capacity"
    assert payload.spec.config["workout"] is True
    assert payload.spec.config["drill"] is True
    engine.apply_godot_authoritative_message(
        {
            "command": "complete",
            "summary": {
                "attempted": 5,
                "correct": 4,
                "accuracy": 0.8,
                "duration_s": 15.0,
                "throughput_per_min": 20.0,
                "total_score": 4.0,
                "max_score": 5.0,
                "score_ratio": 0.8,
                "difficulty_level_start": 5,
                "difficulty_level_end": 5,
            },
        }
    )
    clock.advance(0.5)
    session.update()


def _complete_small_ac_workout(clock: FakeClock) -> AntWorkoutSession:
    session = AntWorkoutSession(
        clock=clock,
        seed=911,
        plan=_build_small_ac_workout_plan(),
        starting_level=5,
    )
    session.activate()
    session.activate()
    session.activate()
    session.activate()
    _complete_current_godot_block(session, clock)
    assert session.stage is AntWorkoutStage.BLOCK_RESULTS
    session.activate()
    assert session.stage is AntWorkoutStage.BLOCK_SETUP
    session.activate()
    _complete_current_godot_block(session, clock)
    assert session.stage is AntWorkoutStage.BLOCK_RESULTS
    session.activate()
    assert session.stage is AntWorkoutStage.RESULTS
    session.activate()
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
