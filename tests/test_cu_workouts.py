from __future__ import annotations

from dataclasses import dataclass

from cfast_trainer.ant_drills import AntDrillMode
from cfast_trainer.ant_workouts import (
    AntWorkoutBlockPlan,
    AntWorkoutPlan,
    AntWorkoutSession,
    AntWorkoutStage,
)
from cfast_trainer.cu_workouts import build_cu_workout_plan
from cfast_trainer.results import attempt_result_from_engine


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _build_small_cu_workout_plan() -> AntWorkoutPlan:
    return AntWorkoutPlan(
        code="cognitive_updating_workout",
        title="CU Workout Smoke",
        description="Short deterministic workout for tests.",
        notes=("Block setup is untimed.",),
        blocks=(
            AntWorkoutBlockPlan(
                block_id="controls",
                label="Controls Anchor",
                description="Warm-up controls block.",
                focus_skills=("Controls correction",),
                drill_code="cu_controls_anchor",
                mode=AntDrillMode.BUILD,
                duration_min=0.25,
            ),
            AntWorkoutBlockPlan(
                block_id="mixed",
                label="Mixed Tempo",
                description="Mixed CU block.",
                focus_skills=("Mixed tempo",),
                drill_code="cu_mixed_tempo",
                mode=AntDrillMode.TEMPO,
                duration_min=0.25,
            ),
        ),
    )


def _finish_current_block_with_one_correct_answer(session: AntWorkoutSession, clock: FakeClock) -> None:
    engine = session.current_engine()
    assert engine is not None
    answer = engine._current.answer
    assert session.submit_answer(str(answer)) is True
    remaining = engine.time_remaining_s()
    assert remaining is not None
    clock.advance(remaining + 0.1)
    session.update()


def _complete_small_cu_workout(clock: FakeClock) -> AntWorkoutSession:
    session = AntWorkoutSession(
        clock=clock,
        seed=911,
        plan=_build_small_cu_workout_plan(),
        starting_level=5,
    )
    session.activate()
    session.activate()
    session.activate()
    session.activate()
    _finish_current_block_with_one_correct_answer(session, clock)
    assert session.stage is AntWorkoutStage.BLOCK_RESULTS
    session.activate()
    assert session.stage is AntWorkoutStage.BLOCK_SETUP
    session.activate()
    _finish_current_block_with_one_correct_answer(session, clock)
    assert session.stage is AntWorkoutStage.BLOCK_RESULTS
    session.activate()
    assert session.stage is AntWorkoutStage.RESULTS
    session.activate()
    session.activate()
    assert session.stage is AntWorkoutStage.RESULTS
    return session


def test_cu_workout_session_runs_to_results() -> None:
    clock = FakeClock()
    session = _complete_small_cu_workout(clock)
    summary = session.scored_summary()
    result = attempt_result_from_engine(session, test_code="cognitive_updating_workout")

    assert summary.workout_code == "cognitive_updating_workout"
    assert summary.completed_blocks == 2
    assert result.metrics["workout_code"] == "cognitive_updating_workout"
    assert "pre_focus_one" not in result.metrics
    assert "post_next_rule" not in result.metrics


def test_real_cu_workout_matches_standard_90_minute_structure() -> None:
    plan = build_cu_workout_plan()

    assert plan.scored_duration_s == 90.0 * 60.0
    assert tuple(block.drill_code for block in plan.blocks) == (
        "cu_controls_anchor",
        "cu_navigation_anchor",
        "cu_engine_balance_run",
        "cu_sensors_timing_prime",
        "cu_objective_prime",
        "cu_state_code_run",
        "cu_mixed_tempo",
        "cu_pressure_run",
    )
    assert {
        "Controls correction",
        "Navigation correction",
        "Engine balance",
        "Sensor timing",
        "Objective execution",
        "State-code synthesis",
        "Pressure tolerance",
    }.issubset(set(plan.focus_skills))
