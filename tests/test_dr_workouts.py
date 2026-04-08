from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from cfast_trainer.ant_drills import AntDrillMode
from cfast_trainer.ant_workouts import (
    AntWorkoutBlockPlan,
    AntWorkoutPlan,
    AntWorkoutSession,
    AntWorkoutStage,
)
from cfast_trainer.digit_recognition import DigitRecognitionTrainingSpec
from cfast_trainer.dr_workouts import build_dr_workout_plan
from cfast_trainer.results import attempt_result_from_engine


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _build_small_dr_workout_plan() -> AntWorkoutPlan:
    return AntWorkoutPlan(
        code="digit_recognition_workout",
        title="DR Workout Smoke",
        description="Short deterministic workout for tests.",
        notes=("Reflections are untimed.",),
        blocks=(
            AntWorkoutBlockPlan(
                block_id="visible",
                label="Visible Copy",
                description="Visible warm-up.",
                focus_skills=("Encoding rhythm",),
                drill_code="dr_visible_copy",
                mode=AntDrillMode.BUILD,
                duration_min=0.25,
            ),
            AntWorkoutBlockPlan(
                block_id="mixed",
                label="Mixed Pressure",
                description="Hidden-memory mixed block.",
                focus_skills=("Mixed-family switching",),
                drill_code="dr_mixed_pressure",
                mode=AntDrillMode.STRESS,
                duration_min=0.25,
            ),
        ),
    )


def _advance_dr_engine_to_question(session: AntWorkoutSession, clock: FakeClock) -> DigitRecognitionTrainingSpec:
    if session.stage is AntWorkoutStage.BLOCK_SETUP:
        session.activate()
    engine = session.current_engine()
    assert engine is not None
    while True:
        payload = engine.snapshot().payload
        if getattr(payload, "accepting_input", False):
            return cast(DigitRecognitionTrainingSpec, engine._current.payload)
        spec = cast(DigitRecognitionTrainingSpec, engine._current.payload)
        step = 0.05
        if getattr(spec, "initial_display_s", 0.0) > 0.0:
            step = max(step, float(spec.initial_display_s) + 0.05)
        if getattr(spec, "mask_s", 0.0) > 0.0 and getattr(payload, "display_lines", None) is None:
            step = max(step, float(spec.mask_s) + 0.05)
        clock.advance(step)
        session.update()


def _finish_current_block_with_one_correct_answer(session: AntWorkoutSession, clock: FakeClock) -> None:
    spec = _advance_dr_engine_to_question(session, clock)
    assert session.submit_answer(spec.expected_digits) is True
    engine = session.current_engine()
    assert engine is not None
    remaining = engine.time_remaining_s()
    assert remaining is not None
    clock.advance(remaining + 0.1)
    session.update()


def _complete_small_dr_workout(clock: FakeClock) -> AntWorkoutSession:
    session = AntWorkoutSession(
        clock=clock,
        seed=444,
        plan=_build_small_dr_workout_plan(),
        starting_level=5,
    )
    session.activate()
    session.append_text("Train clean encoding")
    session.activate()
    session.append_text("Reset fast after misses")
    session.activate()
    session.activate()
    _finish_current_block_with_one_correct_answer(session, clock)
    session.activate()
    _finish_current_block_with_one_correct_answer(session, clock)
    session.activate()
    session.append_text("Different-digit items were slower")
    session.activate()
    session.append_text("Re-encode the next display immediately")
    session.activate()
    assert session.stage is AntWorkoutStage.RESULTS
    return session


def test_dr_workout_session_runs_visible_and_hidden_memory_blocks() -> None:
    clock = FakeClock()
    session = _complete_small_dr_workout(clock)
    summary = session.scored_summary()
    result = attempt_result_from_engine(session, test_code="digit_recognition_workout")

    assert summary.workout_code == "digit_recognition_workout"
    assert summary.completed_blocks == 2
    assert result.metrics["workout_code"] == "digit_recognition_workout"
    assert "pre_focus_one" not in result.metrics
    assert "post_next_rule" not in result.metrics


def test_real_dr_workout_matches_standard_90_minute_structure() -> None:
    plan = build_dr_workout_plan()

    assert plan.scored_duration_s == 90.0 * 60.0
    assert tuple(block.drill_code for block in plan.blocks) == (
        "dr_visible_copy",
        "dr_position_probe",
        "dr_visible_family_primer",
        "dr_recall_run",
        "dr_count_target",
        "dr_different_digit",
        "dr_difference_count",
        "dr_grouped_family_run",
        "dr_mixed_pressure",
    )
    assert {
        "Encoding rhythm",
        "Serial-position anchoring",
        "Family familiarity",
        "Full-string recall",
        "Target counting",
        "Different-digit detection",
        "Difference-count discrimination",
        "Grouped family solving",
        "Mixed-family switching",
    }.issubset(set(plan.focus_skills))
