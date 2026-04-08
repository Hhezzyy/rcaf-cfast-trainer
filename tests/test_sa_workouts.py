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
from cfast_trainer.sa_workouts import build_sa_workout_plan
from cfast_trainer.situational_awareness import SituationalAwarenessPayload


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _build_small_sa_workout_plan() -> AntWorkoutPlan:
    return AntWorkoutPlan(
        code="situational_awareness_workout",
        title="Situational Awareness Workout Smoke",
        description="Short deterministic workout for tests.",
        notes=("Reflections are untimed.",),
        blocks=(
            AntWorkoutBlockPlan(
                block_id="picture-anchor",
                label="Picture Anchor",
                description="Warm-up picture block.",
                focus_skills=("Picture tracking",),
                drill_code="sa_picture_anchor",
                mode=AntDrillMode.BUILD,
                duration_min=0.25,
            ),
            AntWorkoutBlockPlan(
                block_id="mixed-tempo",
                label="Mixed Tempo",
                description="Mixed Situational Awareness block.",
                focus_skills=("Mixed query tempo",),
                drill_code="sa_mixed_tempo",
                mode=AntDrillMode.TEMPO,
                duration_min=0.25,
            ),
        ),
    )


def _run_current_block(session: AntWorkoutSession, clock: FakeClock) -> None:
    answered: set[int] = set()
    while session.stage is AntWorkoutStage.BLOCK:
        engine = session.current_engine()
        assert engine is not None
        snap = engine.snapshot()
        payload = snap.payload
        if isinstance(payload, SituationalAwarenessPayload) and payload.active_query is not None:
            query_id = payload.active_query.query_id
            if query_id not in answered:
                session.submit_answer(payload.active_query.correct_answer_token)
                answered.add(query_id)
        clock.advance(1.0)
        session.update()


def _complete_small_sa_workout(clock: FakeClock) -> AntWorkoutSession:
    session = AntWorkoutSession(
        clock=clock,
        seed=911,
        plan=_build_small_sa_workout_plan(),
        starting_level=5,
    )
    reflection_text = {
        AntWorkoutStage.PRE_REFLECTION: (
            "stay ahead of the picture",
            "reset on every new query",
        ),
        AntWorkoutStage.POST_REFLECTION: (
            "status recall stayed clean",
            "mixed prompts needed faster resets",
        ),
    }
    reflection_index = {
        AntWorkoutStage.PRE_REFLECTION: 0,
        AntWorkoutStage.POST_REFLECTION: 0,
    }

    while session.stage is not AntWorkoutStage.RESULTS:
        if session.stage in (AntWorkoutStage.INTRO, AntWorkoutStage.BLOCK_SETUP, AntWorkoutStage.BLOCK_RESULTS):
            session.activate()
            continue
        if session.stage is AntWorkoutStage.BLOCK:
            _run_current_block(session, clock)
            continue
        if session.stage in (AntWorkoutStage.PRE_REFLECTION, AntWorkoutStage.POST_REFLECTION):
            idx = reflection_index[session.stage]
            session.append_text(reflection_text[session.stage][idx])
            reflection_index[session.stage] = idx + 1
            session.activate()
            continue

    assert session.stage is AntWorkoutStage.RESULTS
    return session


def test_sa_workout_session_runs_to_results() -> None:
    clock = FakeClock()
    session = _complete_small_sa_workout(clock)
    summary = session.scored_summary()
    result = attempt_result_from_engine(session, test_code="situational_awareness_workout")

    assert summary.workout_code == "situational_awareness_workout"
    assert summary.completed_blocks == 2
    assert summary.attempted > 0
    assert result.metrics["workout_code"] == "situational_awareness_workout"
    assert "pre_focus_one" not in result.metrics
    assert "post_next_rule" not in result.metrics


def test_real_sa_workout_matches_standard_90_minute_structure() -> None:
    plan = build_sa_workout_plan()

    assert plan.scored_duration_s == 90.0 * 60.0
    assert tuple(block.drill_code for block in plan.blocks) == (
        "sa_picture_anchor",
        "sa_contact_identification_prime",
        "sa_status_recall_prime",
        "sa_future_projection_run",
        "sa_action_selection_run",
        "sa_family_switch_run",
        "sa_mixed_tempo",
        "sa_pressure_run",
    )
    assert {
        "Picture tracking",
        "Contact identification",
        "Report correlation",
        "Route tracking",
        "Action selection",
        "Pressure tolerance",
    }.issubset(set(plan.focus_skills))
