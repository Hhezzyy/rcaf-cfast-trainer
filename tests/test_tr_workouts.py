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
from cfast_trainer.tr_workouts import build_tr_workout_plan


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _build_small_tr_workout_plan() -> AntWorkoutPlan:
    return AntWorkoutPlan(
        code="target_recognition_workout",
        title="TR Workout Smoke",
        description="Short deterministic workout for tests.",
        notes=("Reflections are untimed.",),
        blocks=(
            AntWorkoutBlockPlan(
                block_id="scene",
                label="Scene Anchor",
                description="Warm-up map drill.",
                focus_skills=("Map discrimination",),
                drill_code="tr_scene_anchor",
                mode=AntDrillMode.BUILD,
                duration_min=0.25,
            ),
            AntWorkoutBlockPlan(
                block_id="mixed",
                label="Mixed Tempo",
                description="Mixed panel drill.",
                focus_skills=("Panel switching",),
                drill_code="tr_mixed_tempo",
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


def _complete_small_tr_workout(clock: FakeClock) -> AntWorkoutSession:
    session = AntWorkoutSession(
        clock=clock,
        seed=939,
        plan=_build_small_tr_workout_plan(),
        starting_level=5,
    )
    session.activate()
    session.append_text("Read the active panel first")
    session.activate()
    session.append_text("Reset on each panel switch")
    session.activate()
    session.activate()
    _finish_current_block_with_one_correct_answer(session, clock)
    session.activate()
    _finish_current_block_with_one_correct_answer(session, clock)
    session.append_text("Switching was the slow point")
    session.activate()
    session.append_text("Keep the panel order cleaner")
    session.activate()
    assert session.stage is AntWorkoutStage.RESULTS
    return session


def test_tr_workout_session_runs_to_results() -> None:
    clock = FakeClock()
    session = _complete_small_tr_workout(clock)
    summary = session.scored_summary()
    result = attempt_result_from_engine(session, test_code="target_recognition_workout")

    assert summary.workout_code == "target_recognition_workout"
    assert summary.completed_blocks == 2
    assert result.metrics["workout_code"] == "target_recognition_workout"
    assert "pre_focus_one" not in result.metrics
    assert "post_next_rule" not in result.metrics


def test_real_tr_workout_matches_standard_90_minute_structure() -> None:
    plan = build_tr_workout_plan()

    assert plan.scored_duration_s == 90.0 * 60.0
    assert tuple(block.drill_code for block in plan.blocks) == (
        "tr_scene_anchor",
        "tr_light_anchor",
        "tr_scan_anchor",
        "tr_system_anchor",
        "tr_scene_modifier_run",
        "tr_panel_switch_run",
        "tr_mixed_tempo",
        "tr_pressure_run",
    )
    assert {
        "Map discrimination",
        "Light pattern matching",
        "Scan stream matching",
        "System-code matching",
        "Panel switching",
        "Pressure tolerance",
    }.issubset(set(plan.focus_skills))
