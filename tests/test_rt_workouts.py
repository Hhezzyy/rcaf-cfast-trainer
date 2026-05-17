from __future__ import annotations

from dataclasses import dataclass

from cfast_trainer.ant_drills import AntDrillMode
from cfast_trainer.ant_workouts import (
    AntWorkoutBlockPlan,
    AntWorkoutPlan,
    AntWorkoutSession,
    AntWorkoutStage,
)
from cfast_trainer.godot_owned import GodotOwnedPayload
from cfast_trainer.results import attempt_result_from_engine
from cfast_trainer.rt_workouts import build_rt_workout_plan


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


def _build_small_rt_workout_plan() -> AntWorkoutPlan:
    return AntWorkoutPlan(
        code="rapid_tracking_workout",
        title="Rapid Tracking Workout Smoke",
        description="Short deterministic workout for tests.",
        notes=("Block setup is untimed.",),
        blocks=(
            AntWorkoutBlockPlan(
                block_id="lock-anchor",
                label="Lock Anchor",
                description="Warm-up lock block.",
                focus_skills=("Lock quality",),
                drill_code="rt_lock_anchor",
                mode=AntDrillMode.BUILD,
                duration_min=0.25,
            ),
            AntWorkoutBlockPlan(
                block_id="mixed-tempo",
                label="Mixed Tempo",
                description="Mixed rapid tracking block.",
                focus_skills=("Mixed tempo",),
                drill_code="rt_mixed_tempo",
                mode=AntDrillMode.TEMPO,
                duration_min=0.25,
            ),
        ),
    )


def _complete_current_godot_block(session: AntWorkoutSession, clock: FakeClock) -> None:
    engine = session.current_engine()
    assert engine is not None
    payload = engine.snapshot().payload
    assert isinstance(payload, GodotOwnedPayload)
    engine.apply_godot_authoritative_message(
        {
            "command": "complete",
            "summary": {
                "attempted": 4,
                "correct": 3,
                "accuracy": 0.75,
                "duration_s": 15.0,
                "throughput_per_min": 16.0,
                "total_score": 3.0,
                "max_score": 4.0,
                "score_ratio": 0.75,
                "difficulty_level_start": 5,
                "difficulty_level_end": 5,
            },
        }
    )
    clock.advance(0.5)
    session.update()


def _complete_small_rt_workout(clock: FakeClock) -> AntWorkoutSession:
    session = AntWorkoutSession(
        clock=clock,
        seed=911,
        plan=_build_small_rt_workout_plan(),
        starting_level=5,
    )
    session.activate()
    session.activate()
    session.activate()
    session.activate()
    _complete_current_godot_block(session, clock)
    assert session.stage is AntWorkoutStage.BLOCK_RESULTS
    session.activate()
    assert session.stage is AntWorkoutStage.BLOCK_SETUP
    session.activate()
    _complete_current_godot_block(session, clock)
    assert session.stage is AntWorkoutStage.BLOCK_RESULTS
    session.activate()
    assert session.stage is AntWorkoutStage.RESULTS
    session.activate()
    session.activate()
    assert session.stage is AntWorkoutStage.RESULTS
    return session


def test_rt_workout_session_runs_to_results() -> None:
    clock = FakeClock()
    session = _complete_small_rt_workout(clock)
    summary = session.scored_summary()
    result = attempt_result_from_engine(session, test_code="rapid_tracking_workout")

    assert summary.workout_code == "rapid_tracking_workout"
    assert summary.completed_blocks == 2
    assert summary.attempted > 0
    assert result.metrics["workout_code"] == "rapid_tracking_workout"
    assert "pre_focus_one" not in result.metrics
    assert "post_next_rule" not in result.metrics


def test_real_rt_workout_matches_standard_90_minute_structure() -> None:
    plan = build_rt_workout_plan()

    assert plan.scored_duration_s == 90.0 * 60.0
    assert tuple(block.drill_code for block in plan.blocks) == (
        "rt_lock_anchor",
        "rt_building_handoff_prime",
        "rt_obscured_target_prediction",
        "rt_capture_timing_prime",
        "rt_ground_tempo_run",
        "rt_air_speed_run",
        "rt_mixed_tempo",
        "rt_pressure_run",
    )
    assert {
        "Lock quality",
        "Handoff reacquisition",
        "Occlusion recovery",
        "Capture timing",
        "Ground tempo",
        "Air-speed tracking",
        "Pressure tolerance",
    }.issubset(set(plan.focus_skills))
