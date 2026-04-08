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
from cfast_trainer.si_workouts import build_si_workout_plan
from cfast_trainer.spatial_integration import (
    SpatialIntegrationAnswerMode,
    SpatialIntegrationPayload,
    SpatialIntegrationTrialStage,
)


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


def _build_small_si_workout_plan() -> AntWorkoutPlan:
    return AntWorkoutPlan(
        code="spatial_integration_workout",
        title="Spatial Integration Workout Smoke",
        description="Short deterministic SI workout for tests.",
        notes=("Reflections are untimed.",),
        blocks=(
            AntWorkoutBlockPlan(
                block_id="landmark-anchor",
                label="Landmark Anchor",
                description="Static landmark block.",
                focus_skills=("Landmark anchoring",),
                drill_code="si_landmark_anchor",
                mode=AntDrillMode.BUILD,
                duration_min=0.30,
            ),
            AntWorkoutBlockPlan(
                block_id="mixed-tempo",
                label="Mixed Tempo",
                description="Balanced static then aircraft block.",
                focus_skills=("Part switching",),
                drill_code="si_mixed_tempo",
                mode=AntDrillMode.TEMPO,
                duration_min=0.30,
            ),
        ),
    )


def _answer_current(payload: SpatialIntegrationPayload) -> str:
    if payload.answer_mode is SpatialIntegrationAnswerMode.GRID_CLICK:
        return payload.correct_answer_token
    return str(payload.correct_code)


def _run_current_block(session: AntWorkoutSession, clock: FakeClock) -> None:
    while session.stage is AntWorkoutStage.BLOCK:
        engine = session.current_engine()
        assert engine is not None
        snap = engine.snapshot()
        payload = snap.payload
        if isinstance(payload, SpatialIntegrationPayload):
            if payload.trial_stage is SpatialIntegrationTrialStage.QUESTION:
                assert session.submit_answer(_answer_current(payload)) is True
                session.update()
                continue
        clock.advance(0.2)
        session.update()


def _complete_small_si_workout(clock: FakeClock) -> AntWorkoutSession:
    session = AntWorkoutSession(
        clock=clock,
        seed=808,
        plan=_build_small_si_workout_plan(),
        starting_level=5,
    )
    session.activate()
    session.append_text("build one scene at a time")
    session.activate()
    session.append_text("reset after a bad viewpoint shift")
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
    session.append_text("aircraft transition was the hardest reset")
    session.activate()
    session.append_text("anchor the hills before the object")
    session.activate()
    assert session.stage is AntWorkoutStage.RESULTS
    return session


def test_si_workout_session_runs_to_results() -> None:
    clock = FakeClock()
    session = _complete_small_si_workout(clock)
    summary = session.scored_summary()
    result = attempt_result_from_engine(session, test_code="spatial_integration_workout")

    assert summary.workout_code == "spatial_integration_workout"
    assert summary.completed_blocks == 2
    assert summary.attempted > 0
    assert result.metrics["workout_code"] == "spatial_integration_workout"
    assert "pre_focus_one" not in result.metrics
    assert "post_next_rule" not in result.metrics


def test_real_si_workout_matches_standard_90_minute_structure() -> None:
    plan = build_si_workout_plan()

    assert plan.scored_duration_s == 90.0 * 60.0
    assert tuple(block.drill_code for block in plan.blocks) == (
        "si_landmark_anchor",
        "si_reconstruction_run",
        "si_static_mixed_run",
        "si_route_anchor",
        "si_continuation_prime",
        "si_aircraft_grid_run",
        "si_mixed_tempo",
        "si_pressure_run",
    )
    assert {
        "Landmark anchoring",
        "Scene reconstruction",
        "Route anchoring",
        "Continuation discrimination",
        "Part switching",
        "Pressure tolerance",
    }.issubset(set(plan.focus_skills))
