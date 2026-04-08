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
from cfast_trainer.sma_workouts import build_sma_workout_plan


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _build_small_sma_workout_plan() -> AntWorkoutPlan:
    return AntWorkoutPlan(
        code="sensory_motor_apparatus_workout",
        title="SMA Workout Smoke",
        description="Short deterministic workout for tests.",
        notes=("Reflections are untimed.",),
        blocks=(
            AntWorkoutBlockPlan(
                block_id="joy-horizontal",
                label="Joystick Horizontal Anchor",
                description="Warm-up horizontal precision block.",
                focus_skills=("Joystick horizontal precision",),
                drill_code="sma_joystick_horizontal_anchor",
                mode=AntDrillMode.BUILD,
                duration_min=0.25,
            ),
            AntWorkoutBlockPlan(
                block_id="mode-switch",
                label="Mode Switch Run",
                description="Mixed control-mode block.",
                focus_skills=("Mode switching",),
                drill_code="sma_mode_switch_run",
                mode=AntDrillMode.TEMPO,
                duration_min=0.25,
            ),
        ),
    )


def _run_current_block(session: AntWorkoutSession, clock: FakeClock) -> None:
    while session.stage is AntWorkoutStage.BLOCK:
        engine = session.current_engine()
        assert engine is not None
        set_control = getattr(engine, "set_control", None)
        if callable(set_control):
            set_control(horizontal=0.18, vertical=-0.12)
        clock.advance(0.25)
        session.update()


def _complete_small_sma_workout(clock: FakeClock) -> AntWorkoutSession:
    session = AntWorkoutSession(
        clock=clock,
        seed=911,
        plan=_build_small_sma_workout_plan(),
        starting_level=5,
    )
    session.activate()
    session.append_text("stay relaxed on the entry")
    session.activate()
    session.append_text("keep hand-foot timing clean")
    session.activate()
    session.activate()
    _run_current_block(session, clock)
    assert session.stage is AntWorkoutStage.BLOCK_RESULTS
    session.activate()
    assert session.stage is AntWorkoutStage.BLOCK_SETUP
    session.activate()
    _run_current_block(session, clock)
    assert session.stage is AntWorkoutStage.BLOCK_RESULTS
    session.activate()
    assert session.stage is AntWorkoutStage.POST_REFLECTION
    session.append_text("horizontal settling improved")
    session.activate()
    session.append_text("switches stayed smooth")
    session.activate()
    assert session.stage is AntWorkoutStage.RESULTS
    return session


def test_sma_workout_session_runs_to_results() -> None:
    clock = FakeClock()
    session = _complete_small_sma_workout(clock)
    summary = session.scored_summary()
    result = attempt_result_from_engine(session, test_code="sensory_motor_apparatus_workout")

    assert summary.workout_code == "sensory_motor_apparatus_workout"
    assert summary.completed_blocks == 2
    assert summary.attempted > 0
    assert result.metrics["workout_code"] == "sensory_motor_apparatus_workout"
    assert "pre_focus_one" not in result.metrics
    assert "post_next_rule" not in result.metrics


def test_real_sma_workout_matches_standard_90_minute_structure() -> None:
    plan = build_sma_workout_plan()

    assert plan.scored_duration_s == 90.0 * 60.0
    assert tuple(block.drill_code for block in plan.blocks) == (
        "sma_joystick_horizontal_anchor",
        "sma_joystick_vertical_anchor",
        "sma_joystick_hold_run",
        "sma_split_horizontal_prime",
        "sma_split_coordination_run",
        "sma_mode_switch_run",
        "sma_disturbance_tempo",
        "sma_pressure_run",
    )
    assert {
        "Joystick horizontal precision",
        "Joystick vertical precision",
        "Split-control coordination",
        "Mode switching",
        "Disturbance recovery",
        "Pressure tolerance",
    }.issubset(set(plan.focus_skills))
