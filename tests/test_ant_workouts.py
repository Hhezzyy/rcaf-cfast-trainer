from __future__ import annotations

from dataclasses import dataclass

import pytest

from cfast_trainer.ant_drills import AntDrillMode
from cfast_trainer.ant_workouts import (
    AntWorkoutBlockPlan,
    AntWorkoutPlan,
    AntWorkoutSession,
    AntWorkoutStage,
    build_ant_workout_plan,
)
from cfast_trainer.persistence import ResultsStore
from cfast_trainer.results import attempt_result_from_engine


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _build_small_workout_plan() -> AntWorkoutPlan:
    return AntWorkoutPlan(
        code="airborne_numerical_workout",
        title="Airborne Numerical Workout Smoke",
        description="Short deterministic workout for tests.",
        notes=("Reflections are untimed.",),
        blocks=(
            AntWorkoutBlockPlan(
                block_id="snap",
                label="Snap Facts Block",
                description="Arithmetic under hard caps.",
                focus_skills=("Arithmetic retrieval",),
                drill_code="ant_snap_facts_sprint",
                mode=AntDrillMode.BUILD,
                duration_min=0.25,
            ),
            AntWorkoutBlockPlan(
                block_id="scenario",
                label="Scenario Pressure Block",
                description="Grouped scenario work under pressure.",
                focus_skills=("Full-question solving",),
                drill_code="airborne_scenario_pressure",
                mode=AntDrillMode.BUILD,
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


def _complete_small_workout(clock: FakeClock) -> AntWorkoutSession:
    session = AntWorkoutSession(
        clock=clock,
        seed=123,
        plan=_build_small_workout_plan(),
        starting_level=5,
    )

    assert session.stage is AntWorkoutStage.INTRO
    assert session.can_exit() is True
    session.adjust_starting_level(1)
    assert session.difficulty == pytest.approx((6 - 1) / 9.0)

    session.activate()
    assert session.stage is AntWorkoutStage.PRE_REFLECTION
    session.append_text("Need better tempo")
    session.activate()
    session.append_text("Move on after misses")
    session.activate()

    assert session.stage is AntWorkoutStage.BLOCK_SETUP
    session.adjust_block_level(-1)
    first_setup = session.snapshot()
    assert first_setup.block_default_level == 6
    assert first_setup.block_override_level == 5
    session.activate()

    assert session.stage is AntWorkoutStage.BLOCK
    _finish_current_block_with_one_correct_answer(session, clock)

    assert session.stage is AntWorkoutStage.BLOCK_SETUP
    second_setup = session.snapshot()
    assert second_setup.block_default_level == 6
    assert second_setup.block_override_level == 6
    session.activate()

    assert session.stage is AntWorkoutStage.BLOCK
    _finish_current_block_with_one_correct_answer(session, clock)

    assert session.stage is AntWorkoutStage.POST_REFLECTION
    session.append_text("Got slow on the scenario transition")
    session.activate()
    session.append_text("Read the ask first")
    session.activate()

    assert session.stage is AntWorkoutStage.RESULTS
    assert session.can_exit() is True
    return session


def test_airborne_numerical_workout_session_runs_reflections_setups_blocks_and_results() -> None:
    clock = FakeClock()
    session = _complete_small_workout(clock)

    summary = session.scored_summary()

    assert summary.block_count == 2
    assert summary.completed_blocks == 2
    assert summary.attempted == 2
    assert summary.correct == 2
    assert summary.workout_code == "airborne_numerical_workout"
    assert summary.difficulty_level_start == 6
    assert summary.difficulty_level_end == 6
    assert len(session.events()) == 2


def test_airborne_numerical_workout_attempt_result_omits_reflection_metrics(tmp_path) -> None:
    clock = FakeClock()
    session = _complete_small_workout(clock)

    result = attempt_result_from_engine(session, test_code="airborne_numerical_workout")
    store = ResultsStore(tmp_path / "results.sqlite3")
    store.record_attempt(result=result, app_version="test", input_profile_id="default")

    assert result.attempted == 2
    assert result.correct == 2
    assert result.metrics["workout_code"] == "airborne_numerical_workout"
    assert "pre_focus_one" not in result.metrics
    assert "post_next_rule" not in result.metrics

    session_summary = store.session_summary()
    workout_summary = store.test_session_summary("airborne_numerical_workout")

    assert session_summary is not None
    assert session_summary.attempt_count == 1
    assert workout_summary is not None
    assert workout_summary.attempt_count == 1
    assert workout_summary.latest_accuracy == 1.0


def test_real_airborne_numerical_workout_matches_standard_90_minute_structure() -> None:
    plan = build_ant_workout_plan("airborne_numerical_workout")
    drill_codes = tuple(block.drill_code for block in plan.blocks)

    assert plan.scored_duration_s == pytest.approx(90.0 * 60.0)
    assert drill_codes == (
        "ant_info_grabber",
        "ant_snap_facts_sprint",
        "ant_time_flip",
        "ant_distance_scan",
        "ant_route_time_solve",
        "ant_endurance_solve",
        "ant_fuel_burn_solve",
        "airborne_scenario_steady",
        "airborne_scenario_pressure",
    )
    assert {
        "Information search",
        "Retention",
        "Arithmetic retrieval",
        "Time conversion",
        "Distance scanning",
        "Route-time solving",
        "Endurance solving",
        "Fuel-burn solving",
        "Full-question solving",
    }.issubset(set(plan.focus_skills))
