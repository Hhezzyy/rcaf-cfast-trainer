from __future__ import annotations

from dataclasses import dataclass

from cfast_trainer.ant_drills import AntDrillMode
from cfast_trainer.ant_workouts import (
    AntWorkoutBlockPlan,
    AntWorkoutPlan,
    AntWorkoutSession,
    AntWorkoutStage,
)
from cfast_trainer.no_workouts import build_no_workout_plan
from cfast_trainer.results import attempt_result_from_engine


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _build_small_no_workout_plan() -> AntWorkoutPlan:
    return AntWorkoutPlan(
        code="numerical_operations_workout",
        title="NO Workout Smoke",
        description="Short deterministic workout for tests.",
        notes=("Reflections are untimed.",),
        blocks=(
            AntWorkoutBlockPlan(
                block_id="prime",
                label="Fact Prime",
                description="Warm-up.",
                focus_skills=("Arithmetic fact retrieval",),
                drill_code="no_fact_prime",
                mode=AntDrillMode.BUILD,
                duration_min=0.25,
            ),
            AntWorkoutBlockPlan(
                block_id="pressure",
                label="Pressure Run",
                description="Late pressure.",
                focus_skills=("Pressure tolerance",),
                drill_code="no_pressure_run",
                mode=AntDrillMode.STRESS,
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


def _complete_small_no_workout(clock: FakeClock) -> AntWorkoutSession:
    session = AntWorkoutSession(
        clock=clock,
        seed=222,
        plan=_build_small_no_workout_plan(),
        starting_level=5,
    )
    session.activate()
    session.append_text("Prime patterns")
    session.activate()
    session.append_text("Keep tempo after misses")
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
    assert session.stage is AntWorkoutStage.POST_REFLECTION
    session.append_text("Division was slower")
    session.activate()
    session.append_text("Reset on the next item")
    session.activate()
    assert session.stage is AntWorkoutStage.RESULTS
    return session


def test_no_workout_session_runs_and_omits_reflection_metrics() -> None:
    clock = FakeClock()
    session = _complete_small_no_workout(clock)
    summary = session.scored_summary()
    result = attempt_result_from_engine(session, test_code="numerical_operations_workout")

    assert summary.workout_code == "numerical_operations_workout"
    assert summary.completed_blocks == 2
    assert result.metrics["workout_code"] == "numerical_operations_workout"
    assert "pre_focus_one" not in result.metrics
    assert "post_next_rule" not in result.metrics


def test_real_no_workout_matches_standard_90_minute_structure() -> None:
    plan = build_no_workout_plan()

    assert plan.scored_duration_s == 90.0 * 60.0
    assert tuple(block.drill_code for block in plan.blocks) == (
        "no_fact_prime",
        "no_operator_ladders",
        "no_clean_compute",
        "no_mixed_tempo",
        "no_mixed_tempo",
        "no_pressure_run",
    )
    assert {
        "Arithmetic fact retrieval",
        "Operator isolation",
        "Clean arithmetic",
        "Tempo control",
        "Time crunch",
        "Full-test rhythm",
    }.issubset(set(plan.focus_skills))
