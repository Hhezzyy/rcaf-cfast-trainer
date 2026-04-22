from __future__ import annotations

from dataclasses import dataclass

from cfast_trainer.ant_drills import AntDrillMode
from cfast_trainer.ant_workouts import (
    AntWorkoutBlockPlan,
    AntWorkoutPlan,
    AntWorkoutSession,
    AntWorkoutStage,
)
from cfast_trainer.mr_workouts import build_mr_workout_plan
from cfast_trainer.results import attempt_result_from_engine


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _build_small_mr_workout_plan() -> AntWorkoutPlan:
    return AntWorkoutPlan(
        code="math_reasoning_workout",
        title="MR Workout Smoke",
        description="Short deterministic workout for tests.",
        notes=("Block setup is untimed.",),
        blocks=(
            AntWorkoutBlockPlan(
                block_id="extract",
                label="Relevant Info",
                description="Typed extraction.",
                focus_skills=("Relevant information extraction",),
                drill_code="mr_relevant_info_scan",
                mode=AntDrillMode.BUILD,
                duration_min=0.25,
            ),
            AntWorkoutBlockPlan(
                block_id="domain",
                label="Domain Run",
                description="Real multiple-choice domain run.",
                focus_skills=("Grouped domain solving",),
                drill_code="mr_domain_run",
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


def _complete_small_mr_workout(clock: FakeClock) -> AntWorkoutSession:
    session = AntWorkoutSession(
        clock=clock,
        seed=999,
        plan=_build_small_mr_workout_plan(),
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


def test_mr_workout_session_runs_typed_and_mc_blocks() -> None:
    clock = FakeClock()
    session = _complete_small_mr_workout(clock)
    summary = session.scored_summary()
    result = attempt_result_from_engine(session, test_code="math_reasoning_workout")

    assert summary.workout_code == "math_reasoning_workout"
    assert summary.completed_blocks == 2
    assert result.metrics["workout_code"] == "math_reasoning_workout"
    assert "pre_focus_one" not in result.metrics
    assert "post_next_rule" not in result.metrics


def test_real_mr_workout_matches_standard_90_minute_structure() -> None:
    plan = build_mr_workout_plan()

    assert plan.scored_duration_s == 90.0 * 60.0
    assert tuple(block.drill_code for block in plan.blocks) == (
        "mr_relevant_info_scan",
        "mr_unit_relation_prime",
        "mr_one_step_solve",
        "mr_multi_step_solve",
        "mr_domain_run",
        "mr_mixed_pressure_set",
    )
    assert {
        "Relevant information extraction",
        "Unit conversion",
        "One-step solving",
        "Multi-step solving",
        "Grouped domain solving",
        "Pressure tolerance",
    }.issubset(set(plan.focus_skills))
