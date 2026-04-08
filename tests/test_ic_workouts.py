from __future__ import annotations

import sys
from dataclasses import dataclass
from importlib.machinery import ModuleSpec
from types import ModuleType

if "moderngl" not in sys.modules:
    moderngl_stub = ModuleType("moderngl")
    moderngl_stub.__spec__ = ModuleSpec("moderngl", loader=None)
    sys.modules["moderngl"] = moderngl_stub

from cfast_trainer.ant_drills import AntDrillMode
from cfast_trainer.ant_workouts import (
    AntWorkoutBlockPlan,
    AntWorkoutPlan,
    AntWorkoutSession,
    AntWorkoutStage,
)
from cfast_trainer.ic_workouts import build_ic_workout_plan
from cfast_trainer.results import attempt_result_from_engine


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _build_small_ic_workout_plan() -> AntWorkoutPlan:
    return AntWorkoutPlan(
        code="instrument_comprehension_workout",
        title="IC Workout Smoke",
        description="Short deterministic workout for tests.",
        notes=("Reflections are untimed.",),
        blocks=(
            AntWorkoutBlockPlan(
                block_id="heading",
                label="Heading Anchor",
                description="Warm-up heading drill.",
                focus_skills=("Heading anchoring",),
                drill_code="ic_heading_anchor",
                mode=AntDrillMode.BUILD,
                duration_min=0.25,
            ),
            AntWorkoutBlockPlan(
                block_id="description",
                label="Reverse Panel Run",
                description="Full part 2 run.",
                focus_skills=("Part 2 matching",),
                drill_code="ic_reverse_panel_run",
                mode=AntDrillMode.TEMPO,
                duration_min=0.25,
            ),
            AntWorkoutBlockPlan(
                block_id="description",
                label="Description Run",
                description="Full part 3 run.",
                focus_skills=("Part 3 interpretation",),
                drill_code="ic_description_run",
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


def _complete_small_ic_workout(clock: FakeClock) -> AntWorkoutSession:
    session = AntWorkoutSession(
        clock=clock,
        seed=818,
        plan=_build_small_ic_workout_plan(),
        starting_level=5,
    )
    session.activate()
    session.append_text("Read the instruments cleanly")
    session.activate()
    session.append_text("Reset on the next item")
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
    assert session.stage is AntWorkoutStage.BLOCK_SETUP
    session.activate()
    _finish_current_block_with_one_correct_answer(session, clock)
    assert session.stage is AntWorkoutStage.BLOCK_RESULTS
    session.activate()
    assert session.stage is AntWorkoutStage.POST_REFLECTION
    session.append_text("Descriptions slowed me down")
    session.activate()
    session.append_text("Reverse mapping felt better")
    session.activate()
    session.append_text("Keep the read order stable")
    session.activate()
    assert session.stage is AntWorkoutStage.RESULTS
    return session


def test_ic_workout_session_runs_to_results() -> None:
    clock = FakeClock()
    session = _complete_small_ic_workout(clock)
    summary = session.scored_summary()
    result = attempt_result_from_engine(session, test_code="instrument_comprehension_workout")

    assert summary.workout_code == "instrument_comprehension_workout"
    assert summary.completed_blocks == 3
    assert result.metrics["workout_code"] == "instrument_comprehension_workout"
    assert "pre_focus_one" not in result.metrics
    assert "post_next_rule" not in result.metrics


def test_real_ic_workout_matches_standard_90_minute_structure() -> None:
    plan = build_ic_workout_plan()

    assert plan.scored_duration_s == 90.0 * 60.0
    assert tuple(block.drill_code for block in plan.blocks) == (
        "ic_heading_anchor",
        "ic_attitude_frame",
        "ic_part1_orientation_run",
        "ic_reverse_panel_prime",
        "ic_reverse_panel_run",
        "ic_description_prime",
        "ic_description_run",
        "ic_mixed_part_run",
        "ic_pressure_run",
    )
    assert {
        "Heading anchoring",
        "Bank/pitch discrimination",
        "Part 1 matching",
        "Part 2 matching",
        "Part 3 interpretation",
        "Part switching",
        "Pressure tolerance",
    }.issubset(set(plan.focus_skills))
