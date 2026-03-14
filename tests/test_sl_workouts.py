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
from cfast_trainer.sl_workouts import build_sl_workout_plan


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _build_small_sl_workout_plan() -> AntWorkoutPlan:
    return AntWorkoutPlan(
        code="system_logic_workout",
        title="SL Workout Smoke",
        description="Short deterministic workout for tests.",
        notes=("Reflections are untimed.",),
        blocks=(
            AntWorkoutBlockPlan(
                block_id="quant",
                label="Quantitative Anchor",
                description="Warm-up quantitative block.",
                focus_skills=("Quantitative reasoning",),
                drill_code="sl_quantitative_anchor",
                mode=AntDrillMode.BUILD,
                duration_min=0.25,
            ),
            AntWorkoutBlockPlan(
                block_id="mixed",
                label="Mixed Tempo",
                description="Mixed System Logic block.",
                focus_skills=("Mixed reasoning tempo",),
                drill_code="sl_mixed_tempo",
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


def _complete_small_sl_workout(clock: FakeClock) -> AntWorkoutSession:
    session = AntWorkoutSession(
        clock=clock,
        seed=911,
        plan=_build_small_sl_workout_plan(),
        starting_level=5,
    )
    session.activate()
    session.append_text("Use the index deliberately")
    session.activate()
    session.append_text("Keep the pane scan ordered")
    session.activate()
    session.activate()
    _finish_current_block_with_one_correct_answer(session, clock)
    session.activate()
    _finish_current_block_with_one_correct_answer(session, clock)
    session.append_text("Switches stayed manageable")
    session.activate()
    session.append_text("Keep graph and rule tied together")
    session.activate()
    assert session.stage is AntWorkoutStage.RESULTS
    return session


def test_sl_workout_session_runs_to_results() -> None:
    clock = FakeClock()
    session = _complete_small_sl_workout(clock)
    summary = session.scored_summary()
    result = attempt_result_from_engine(session, test_code="system_logic_workout")

    assert summary.workout_code == "system_logic_workout"
    assert summary.completed_blocks == 2
    assert result.metrics["workout_code"] == "system_logic_workout"
    assert "pre_focus_one" not in result.metrics
    assert "post_next_rule" not in result.metrics


def test_real_sl_workout_matches_standard_90_minute_structure() -> None:
    plan = build_sl_workout_plan()

    assert plan.scored_duration_s == 90.0 * 60.0
    assert tuple(block.drill_code for block in plan.blocks) == (
        "sl_quantitative_anchor",
        "sl_flow_trace_anchor",
        "sl_graph_rule_anchor",
        "sl_fault_diagnosis_prime",
        "sl_index_switch_run",
        "sl_family_run",
        "sl_mixed_tempo",
        "sl_pressure_run",
    )
    assert {
        "Quantitative reasoning",
        "Flow tracing",
        "Graph interpretation",
        "Fault diagnosis",
        "Index switching",
        "System-family switching",
        "Pressure tolerance",
    }.issubset(set(plan.focus_skills))
