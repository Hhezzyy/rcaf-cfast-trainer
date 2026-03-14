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
from cfast_trainer.vs_workouts import build_vs_workout_plan


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _build_small_vs_workout_plan() -> AntWorkoutPlan:
    return AntWorkoutPlan(
        code="visual_search_workout",
        title="VS Workout Smoke",
        description="Short deterministic workout for tests.",
        notes=("Reflections are untimed.",),
        blocks=(
            AntWorkoutBlockPlan(
                block_id="preview",
                label="Target Preview",
                description="Preview warm-up.",
                focus_skills=("Target reacquisition",),
                drill_code="vs_target_preview",
                mode=AntDrillMode.BUILD,
                duration_min=0.25,
            ),
            AntWorkoutBlockPlan(
                block_id="pressure",
                label="Pressure Run",
                description="Late pressure block.",
                focus_skills=("Pressure tolerance",),
                drill_code="vs_pressure_run",
                mode=AntDrillMode.STRESS,
                duration_min=0.25,
            ),
        ),
    )


def _finish_current_block_with_one_correct_answer(session: AntWorkoutSession, clock: FakeClock) -> None:
    engine = session.current_engine()
    assert engine is not None
    assert session.submit_answer(str(engine._current.answer)) is True
    remaining = engine.time_remaining_s()
    assert remaining is not None
    clock.advance(remaining + 0.1)
    session.update()


def _complete_small_vs_workout(clock: FakeClock) -> AntWorkoutSession:
    session = AntWorkoutSession(
        clock=clock,
        seed=812,
        plan=_build_small_vs_workout_plan(),
        starting_level=5,
    )

    session.activate()
    session.append_text("Train disciplined scanning")
    session.activate()
    session.append_text("Reset after misses")
    session.activate()

    assert session.stage is AntWorkoutStage.BLOCK_SETUP
    session.activate()
    assert session.stage is AntWorkoutStage.BLOCK
    _finish_current_block_with_one_correct_answer(session, clock)

    assert session.stage is AntWorkoutStage.BLOCK_SETUP
    session.activate()
    assert session.stage is AntWorkoutStage.BLOCK
    _finish_current_block_with_one_correct_answer(session, clock)

    assert session.stage is AntWorkoutStage.POST_REFLECTION
    session.append_text("Similar symbols slowed the scan")
    session.activate()
    session.append_text("Commit to row order next time")
    session.activate()

    assert session.stage is AntWorkoutStage.RESULTS
    return session


def test_vs_workout_session_runs_preview_and_pressure_blocks() -> None:
    clock = FakeClock()
    session = _complete_small_vs_workout(clock)
    summary = session.scored_summary()

    assert summary.workout_code == "visual_search_workout"
    assert summary.block_count == 2
    assert summary.completed_blocks == 2
    assert summary.attempted == 2
    assert summary.correct == 2


def test_vs_workout_attempt_result_omits_reflection_metrics() -> None:
    clock = FakeClock()
    session = _complete_small_vs_workout(clock)
    result = attempt_result_from_engine(session, test_code="visual_search_workout")

    assert result.metrics["workout_code"] == "visual_search_workout"
    assert "pre_focus_one" not in result.metrics
    assert "post_next_rule" not in result.metrics


def test_real_vs_workout_matches_standard_90_minute_structure() -> None:
    plan = build_vs_workout_plan()

    assert plan.scored_duration_s == 90.0 * 60.0
    assert tuple(block.drill_code for block in plan.blocks) == (
        "vs_target_preview",
        "vs_clean_scan",
        "vs_family_run_letters",
        "vs_family_run_symbols",
        "vs_mixed_tempo",
        "vs_pressure_run",
    )
    assert {
        "Target reacquisition",
        "Full-board familiarity",
        "Scan rhythm",
        "Letter discrimination",
        "Line-figure discrimination",
        "Family switching",
        "Pressure tolerance",
    }.issubset(set(plan.focus_skills))
