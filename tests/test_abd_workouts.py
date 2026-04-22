from __future__ import annotations

from dataclasses import dataclass

from cfast_trainer.abd_workouts import build_abd_workout_plan
from cfast_trainer.ant_drills import AntDrillMode
from cfast_trainer.ant_workouts import (
    AntWorkoutBlockPlan,
    AntWorkoutPlan,
    AntWorkoutSession,
    AntWorkoutStage,
)
from cfast_trainer.results import attempt_result_from_engine


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _build_small_abd_workout_plan() -> AntWorkoutPlan:
    return AntWorkoutPlan(
        code="angles_bearings_degrees_workout",
        title="ABD Workout Smoke",
        description="Short deterministic workout for tests.",
        notes=("Block setup is untimed.",),
        blocks=(
            AntWorkoutBlockPlan(
                block_id="anchors",
                label="Anchor Block",
                description="Anchor warm-up.",
                focus_skills=("Cardinal bearings",),
                drill_code="abd_cardinal_anchors",
                mode=AntDrillMode.BUILD,
                duration_min=0.25,
            ),
            AntWorkoutBlockPlan(
                block_id="angle_run",
                label="Angle Run",
                description="Real multiple-choice angle run.",
                focus_skills=("Full-question solving",),
                drill_code="abd_family_run_angle",
                mode=AntDrillMode.TEMPO,
                duration_min=0.25,
            ),
        ),
    )


def _finish_current_block_with_one_correct_answer(
    session: AntWorkoutSession,
    clock: FakeClock,
) -> None:
    engine = session.current_engine()
    assert engine is not None
    answer = engine._current.answer
    assert session.submit_answer(str(answer)) is True
    remaining = engine.time_remaining_s()
    assert remaining is not None
    clock.advance(remaining + 0.1)
    session.update()


def _complete_small_abd_workout(clock: FakeClock) -> AntWorkoutSession:
    session = AntWorkoutSession(
        clock=clock,
        seed=321,
        plan=_build_small_abd_workout_plan(),
        starting_level=5,
    )

    assert session.stage is AntWorkoutStage.INTRO
    session.activate()

    assert session.stage is AntWorkoutStage.BLOCK_SETUP
    session.activate()
    assert session.stage is AntWorkoutStage.BLOCK
    _finish_current_block_with_one_correct_answer(session, clock)

    assert session.stage is AntWorkoutStage.BLOCK_RESULTS
    session.activate()
    assert session.stage is AntWorkoutStage.BLOCK_SETUP
    session.activate()
    assert session.stage is AntWorkoutStage.BLOCK
    _finish_current_block_with_one_correct_answer(session, clock)

    assert session.stage is AntWorkoutStage.BLOCK_RESULTS
    session.activate()
    assert session.stage is AntWorkoutStage.RESULTS
    session.activate()
    session.activate()

    assert session.stage is AntWorkoutStage.RESULTS
    return session


def test_abd_workout_session_runs_typed_and_mc_blocks() -> None:
    clock = FakeClock()
    session = _complete_small_abd_workout(clock)
    summary = session.scored_summary()

    assert summary.workout_code == "angles_bearings_degrees_workout"
    assert summary.block_count == 2
    assert summary.completed_blocks == 2
    assert summary.attempted == 2
    assert summary.correct == 2
    assert len(session.events()) == 2


def test_abd_workout_attempt_result_omits_setup_metrics() -> None:
    clock = FakeClock()
    session = _complete_small_abd_workout(clock)
    result = attempt_result_from_engine(session, test_code="angles_bearings_degrees_workout")

    assert result.metrics["workout_code"] == "angles_bearings_degrees_workout"
    assert "pre_focus_one" not in result.metrics
    assert "post_next_rule" not in result.metrics


def test_real_abd_workout_matches_standard_90_minute_structure() -> None:
    plan = build_abd_workout_plan()

    assert plan.scored_duration_s == 90.0 * 60.0
    assert tuple(block.drill_code for block in plan.blocks) == (
        "abd_cardinal_anchors",
        "abd_intermediate_anchors",
        "abd_angle_calibration",
        "abd_bearing_calibration",
        "abd_mixed_tempo",
        "abd_family_run_angle",
        "abd_family_run_bearing",
    )
    assert {
        "Cardinal bearings",
        "Straight-angle anchors",
        "Diagonal anchors",
        "Intermediate landmarks",
        "Angle estimation",
        "Bearing estimation",
        "Task switching",
        "Full-question solving",
    }.issubset(set(plan.focus_skills))
