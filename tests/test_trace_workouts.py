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
from cfast_trainer.trace_test_1 import TraceTest1Payload, TraceTest1TrialStage
from cfast_trainer.trace_test_2 import TraceTest2Payload, TraceTest2TrialStage
from cfast_trainer.trace_workouts import (
    build_trace_test_1_workout_plan,
    build_trace_test_2_workout_plan,
)


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


def _build_small_tt1_workout_plan() -> AntWorkoutPlan:
    return AntWorkoutPlan(
        code="trace_test_1_workout",
        title="Trace Test 1 Workout Smoke",
        description="Short deterministic TT1 workout for tests.",
        notes=("Reflections are untimed.",),
        blocks=(
            AntWorkoutBlockPlan(
                block_id="tt1-lateral",
                label="TT1 Lateral Anchor",
                description="Short TT1 lateral block.",
                focus_skills=("TT1 lateral discrimination",),
                drill_code="tt1_lateral_anchor",
                mode=AntDrillMode.BUILD,
                duration_min=0.20,
            ),
            AntWorkoutBlockPlan(
                block_id="tt1-command",
                label="TT1 Command Switch",
                description="Short TT1 command switch block.",
                focus_skills=("TT1 command switching",),
                drill_code="tt1_command_switch_run",
                mode=AntDrillMode.TEMPO,
                duration_min=0.20,
            ),
        ),
    )


def _build_small_tt2_workout_plan() -> AntWorkoutPlan:
    return AntWorkoutPlan(
        code="trace_test_2_workout",
        title="Trace Test 2 Workout Smoke",
        description="Short deterministic TT2 workout for tests.",
        notes=("Reflections are untimed.",),
        blocks=(
            AntWorkoutBlockPlan(
                block_id="tt2-steady",
                label="TT2 Steady Anchor",
                description="Short TT2 steady block.",
                focus_skills=("TT2 steady-track recall",),
                drill_code="tt2_steady_anchor",
                mode=AntDrillMode.BUILD,
                duration_min=0.20,
            ),
            AntWorkoutBlockPlan(
                block_id="tt2-position",
                label="TT2 Position Recall",
                description="Short TT2 position block.",
                focus_skills=("TT2 end-state recall",),
                drill_code="tt2_position_recall_run",
                mode=AntDrillMode.TEMPO,
                duration_min=0.20,
            ),
        ),
    )


def _run_current_block(session: AntWorkoutSession, clock: FakeClock) -> None:
    last_tt1_prompt_index: int | None = None
    while session.stage is AntWorkoutStage.BLOCK:
        engine = session.current_engine()
        assert engine is not None
        snap = engine.snapshot()
        payload = snap.payload
        if isinstance(payload, TraceTest1Payload):
            if payload.trial_stage is TraceTest1TrialStage.ANSWER_OPEN:
                if payload.prompt_index == last_tt1_prompt_index:
                    clock.advance(0.1)
                    session.update()
                    continue
                assert session.submit_answer(str(payload.correct_code)) is True
                last_tt1_prompt_index = payload.prompt_index
                session.update()
                continue
        elif isinstance(payload, TraceTest2Payload):
            if payload.trial_stage is TraceTest2TrialStage.QUESTION:
                assert session.submit_answer(str(payload.correct_code)) is True
                last_tt1_prompt_index = None
                session.update()
                continue
        else:
            last_tt1_prompt_index = None
        clock.advance(0.1)
        session.update()


def _complete_small_trace_workout(clock: FakeClock, plan: AntWorkoutPlan) -> AntWorkoutSession:
    session = AntWorkoutSession(
        clock=clock,
        seed=909,
        plan=plan,
        starting_level=5,
    )
    session.activate()
    session.append_text("keep the first cue clean")
    session.activate()
    session.append_text("reset immediately after misses")
    session.activate()
    session.activate()
    _run_current_block(session, clock)
    session.activate()
    _run_current_block(session, clock)
    session.append_text("second block felt faster")
    session.activate()
    session.append_text("anchor the first family earlier")
    session.activate()
    assert session.stage is AntWorkoutStage.RESULTS
    return session


def test_trace_test_1_workout_session_runs_to_results() -> None:
    clock = FakeClock()
    session = _complete_small_trace_workout(clock, _build_small_tt1_workout_plan())
    summary = session.scored_summary()
    result = attempt_result_from_engine(session, test_code="trace_test_1_workout")

    assert summary.workout_code == "trace_test_1_workout"
    assert summary.completed_blocks == 2
    assert summary.attempted > 0
    assert result.metrics["workout_code"] == "trace_test_1_workout"
    assert "pre_focus_one" not in result.metrics
    assert "post_next_rule" not in result.metrics


def test_trace_test_2_workout_session_runs_to_results() -> None:
    clock = FakeClock()
    session = _complete_small_trace_workout(clock, _build_small_tt2_workout_plan())
    summary = session.scored_summary()
    result = attempt_result_from_engine(session, test_code="trace_test_2_workout")

    assert summary.workout_code == "trace_test_2_workout"
    assert summary.completed_blocks == 2
    assert summary.attempted > 0
    assert result.metrics["workout_code"] == "trace_test_2_workout"
    assert "pre_focus_one" not in result.metrics
    assert "post_next_rule" not in result.metrics


def test_real_trace_test_1_workout_matches_standard_90_minute_structure() -> None:
    plan = build_trace_test_1_workout_plan()

    assert plan.scored_duration_s == 90.0 * 60.0
    assert tuple(block.drill_code for block in plan.blocks) == (
        "tt1_lateral_anchor",
        "tt1_vertical_anchor",
        "tt1_command_switch_run",
        "tt1_lateral_anchor",
        "tt1_vertical_anchor",
        "tt1_command_switch_run",
        "tt1_command_switch_run",
        "tt1_command_switch_run",
    )


def test_real_trace_test_2_workout_matches_standard_90_minute_structure() -> None:
    plan = build_trace_test_2_workout_plan()

    assert plan.scored_duration_s == 90.0 * 60.0
    assert tuple(block.drill_code for block in plan.blocks) == (
        "tt2_steady_anchor",
        "tt2_turn_trace_run",
        "tt2_position_recall_run",
        "tt2_steady_anchor",
        "tt2_turn_trace_run",
        "tt2_position_recall_run",
        "tt2_turn_trace_run",
        "tt2_position_recall_run",
    )
