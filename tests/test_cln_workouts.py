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
from cfast_trainer.cln_workouts import build_cln_workout_plan
from cfast_trainer.colours_letters_numbers import ColoursLettersNumbersTrainingPayload
from cfast_trainer.results import attempt_result_from_engine


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


def _build_small_cln_workout_plan() -> AntWorkoutPlan:
    return AntWorkoutPlan(
        code="colours_letters_numbers_workout",
        title="CLN Workout Smoke",
        description="Short deterministic workout for tests.",
        notes=("Block setup is untimed.",),
        blocks=(
            AntWorkoutBlockPlan(
                block_id="sequence_copy",
                label="Sequence Copy",
                description="Visible sequence warm-up.",
                focus_skills=("Encoding rhythm",),
                drill_code="cln_sequence_copy",
                mode=AntDrillMode.BUILD,
                duration_min=0.25,
            ),
            AntWorkoutBlockPlan(
                block_id="full_steady",
                label="Full Steady",
                description="Three-channel CLN block.",
                focus_skills=("Full multitask integration",),
                drill_code="cln_full_steady",
                mode=AntDrillMode.TEMPO,
                duration_min=0.25,
            ),
        ),
    )


def _complete_small_cln_workout(clock: FakeClock) -> AntWorkoutSession:
    session = AntWorkoutSession(
        clock=clock,
        seed=505,
        plan=_build_small_cln_workout_plan(),
        starting_level=5,
    )

    session.activate()

    assert session.stage is AntWorkoutStage.BLOCK_SETUP
    session.activate()
    assert session.stage is AntWorkoutStage.BLOCK
    first_engine = session.current_engine()
    assert first_engine is not None
    first_payload = cast(ColoursLettersNumbersTrainingPayload, first_engine.snapshot().payload)
    assert first_payload.target_sequence is not None
    assert session.submit_answer(f"MEMSEQ:{first_payload.target_sequence}") is True
    remaining = first_engine.time_remaining_s()
    assert remaining is not None
    clock.advance(remaining + 0.1)
    session.update()

    assert session.stage is AntWorkoutStage.BLOCK_RESULTS
    session.activate()
    assert session.stage is AntWorkoutStage.BLOCK_SETUP
    session.activate()
    assert session.stage is AntWorkoutStage.BLOCK
    second_engine = session.current_engine()
    assert second_engine is not None
    second_payload = cast(ColoursLettersNumbersTrainingPayload, second_engine.snapshot().payload)
    assert second_payload.memory_active is True
    assert second_payload.math_active is True
    assert second_payload.colour_active is True
    session.debug_skip_block()

    assert session.stage is AntWorkoutStage.BLOCK_RESULTS
    session.activate()
    assert session.stage is AntWorkoutStage.RESULTS
    session.activate()
    session.activate()

    assert session.stage is AntWorkoutStage.RESULTS
    return session


def test_cln_workout_session_runs_scaffold_and_full_blocks() -> None:
    clock = FakeClock()
    session = _complete_small_cln_workout(clock)
    summary = session.scored_summary()

    assert summary.workout_code == "colours_letters_numbers_workout"
    assert summary.block_count == 2
    assert summary.completed_blocks == 2
    assert summary.attempted >= 1


def test_cln_workout_attempt_result_omits_setup_metrics() -> None:
    clock = FakeClock()
    session = _complete_small_cln_workout(clock)
    result = attempt_result_from_engine(session, test_code="colours_letters_numbers_workout")

    assert result.metrics["workout_code"] == "colours_letters_numbers_workout"
    assert "pre_focus_one" not in result.metrics
    assert "post_next_rule" not in result.metrics


def test_real_cln_workout_matches_standard_90_minute_structure() -> None:
    plan = build_cln_workout_plan()

    assert plan.scored_duration_s == 90.0 * 60.0
    assert tuple(block.drill_code for block in plan.blocks) == (
        "cln_sequence_copy",
        "cln_sequence_match",
        "cln_math_prime",
        "cln_colour_lane",
        "cln_memory_math",
        "cln_memory_colour",
        "cln_full_steady",
        "cln_overdrive_blue_return",
        "cln_overdrive_six_choice_memory",
        "cln_overdrive_dual_math",
    )
    assert {
        "Encoding rhythm",
        "Delayed recall",
        "Arithmetic priming",
        "Lane mapping",
        "Memory under interference",
        "Colour-lane multitask",
        "Full multitask integration",
        "Pressure tolerance",
        "Blue-lane recovery",
        "Memory discrimination",
        "Dual-math switching",
    }.issubset(set(plan.focus_skills))
