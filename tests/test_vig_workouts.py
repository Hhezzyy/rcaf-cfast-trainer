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
from cfast_trainer.vig_workouts import build_vig_workout_plan
from cfast_trainer.vigilance import VigilancePayload


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


def _build_small_vig_workout_plan() -> AntWorkoutPlan:
    return AntWorkoutPlan(
        code="vigilance_workout",
        title="Vigilance Workout Smoke",
        description="Short deterministic Vigilance workout for tests.",
        notes=("Reflections are untimed.",),
        blocks=(
            AntWorkoutBlockPlan(
                block_id="entry-anchor",
                label="Entry Anchor",
                description="Short entry warm-up block.",
                focus_skills=("Coordinate entry stability",),
                drill_code="vig_entry_anchor",
                mode=AntDrillMode.BUILD,
                duration_min=0.20,
            ),
            AntWorkoutBlockPlan(
                block_id="pressure-run",
                label="Pressure Run",
                description="Short pressure block.",
                focus_skills=("Pressure tolerance",),
                drill_code="vig_pressure_run",
                mode=AntDrillMode.STRESS,
                duration_min=0.20,
            ),
        ),
    )


def _run_current_block(session: AntWorkoutSession, clock: FakeClock) -> None:
    captured_ids: set[int] = set()
    while session.stage is AntWorkoutStage.BLOCK:
        engine = session.current_engine()
        assert engine is not None
        payload = engine.snapshot().payload
        if isinstance(payload, VigilancePayload):
            for symbol in payload.symbols:
                if symbol.symbol_id in captured_ids:
                    continue
                captured_ids.add(symbol.symbol_id)
                if symbol.symbol_id % 2 == 1:
                    assert session.submit_answer(f"{symbol.row},{symbol.col}") is True
                    break
        clock.advance(0.1)
        session.update()


def _complete_small_vig_workout(clock: FakeClock) -> AntWorkoutSession:
    session = AntWorkoutSession(
        clock=clock,
        seed=919,
        plan=_build_small_vig_workout_plan(),
        starting_level=5,
    )
    session.activate()
    session.append_text("scan in rows")
    session.activate()
    session.append_text("reset after misses")
    session.activate()

    assert session.stage is AntWorkoutStage.BLOCK_SETUP
    session.activate()
    _run_current_block(session, clock)

    assert session.stage is AntWorkoutStage.BLOCK_SETUP
    session.activate()
    _run_current_block(session, clock)

    assert session.stage is AntWorkoutStage.POST_REFLECTION
    session.append_text("late block was denser")
    session.activate()
    session.append_text("keep row-first discipline")
    session.activate()
    assert session.stage is AntWorkoutStage.RESULTS
    return session


def test_vigilance_workout_session_runs_to_results() -> None:
    clock = FakeClock()
    session = _complete_small_vig_workout(clock)
    summary = session.scored_summary()
    result = attempt_result_from_engine(session, test_code="vigilance_workout")

    assert summary.workout_code == "vigilance_workout"
    assert summary.completed_blocks == 2
    assert summary.attempted > 0
    assert result.metrics["workout_code"] == "vigilance_workout"
    assert "pre_focus_one" not in result.metrics
    assert "post_next_rule" not in result.metrics


def test_real_vigilance_workout_matches_standard_90_minute_structure() -> None:
    plan = build_vig_workout_plan()

    assert plan.scored_duration_s == 90.0 * 60.0
    assert tuple(block.drill_code for block in plan.blocks) == (
        "vig_entry_anchor",
        "vig_clean_scan",
        "vig_steady_capture_run",
        "vig_density_ladder",
        "vig_tempo_sweep",
        "vig_pressure_run",
    )
