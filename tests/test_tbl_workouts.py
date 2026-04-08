from __future__ import annotations

from dataclasses import dataclass

from cfast_trainer.ant_drills import AntDrillMode
from cfast_trainer.ant_workouts import (
    AntWorkoutBlockPlan,
    AntWorkoutPlan,
    AntWorkoutSession,
    AntWorkoutStage,
)
from cfast_trainer.results import attempt_result_from_engine
from cfast_trainer.tbl_workouts import build_tbl_workout_plan


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _build_small_tbl_workout_plan() -> AntWorkoutPlan:
    return AntWorkoutPlan(
        code="table_reading_workout",
        title="TRBL Workout Smoke",
        description="Short deterministic workout for tests.",
        notes=("Reflections are untimed.",),
        blocks=(
            AntWorkoutBlockPlan(
                block_id="part1",
                label="Part 1 Anchor",
                description="Warm-up single-card block.",
                focus_skills=("Single-table lookup",),
                drill_code="tbl_part1_anchor",
                mode=AntDrillMode.BUILD,
                duration_min=0.25,
            ),
            AntWorkoutBlockPlan(
                block_id="mixed",
                label="Mixed Tempo",
                description="Mixed table block.",
                focus_skills=("Mixed table tempo",),
                drill_code="tbl_mixed_tempo",
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


def _complete_small_tbl_workout(clock: FakeClock) -> AntWorkoutSession:
    session = AntWorkoutSession(
        clock=clock,
        seed=911,
        plan=_build_small_tbl_workout_plan(),
        starting_level=5,
    )
    session.activate()
    session.append_text("stay orderly")
    session.activate()
    session.append_text("carry values cleanly")
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
    session.append_text("scan rhythm held")
    session.activate()
    session.append_text("switching stayed clean")
    session.activate()
    assert session.stage is AntWorkoutStage.RESULTS
    return session


def test_tbl_workout_session_runs_to_results() -> None:
    clock = FakeClock()
    session = _complete_small_tbl_workout(clock)
    summary = session.scored_summary()
    result = attempt_result_from_engine(session, test_code="table_reading_workout")

    assert summary.workout_code == "table_reading_workout"
    assert summary.completed_blocks == 2
    assert result.metrics["workout_code"] == "table_reading_workout"
    assert "pre_focus_one" not in result.metrics
    assert "post_next_rule" not in result.metrics


def test_real_tbl_workout_matches_standard_90_minute_structure() -> None:
    plan = build_tbl_workout_plan()

    assert plan.scored_duration_s == 90.0 * 60.0
    assert tuple(block.drill_code for block in plan.blocks) == (
        "tbl_part1_anchor",
        "tbl_part1_scan_run",
        "tbl_part2_prime",
        "tbl_part2_correction_run",
        "tbl_part_switch_run",
        "tbl_card_family_run",
        "tbl_mixed_tempo",
        "tbl_pressure_run",
    )
    assert {
        "Single-table lookup",
        "Scan speed",
        "Two-card chaining",
        "Two-card correction",
        "Part switching",
        "Card-family adaptation",
        "Pressure tolerance",
    }.issubset(set(plan.focus_skills))
