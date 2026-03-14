from __future__ import annotations

from dataclasses import dataclass

import pytest

from cfast_trainer.cognitive_core import Phase
from cfast_trainer.spatial_integration import (
    SpatialIntegrationAnswerMode,
    SpatialIntegrationConfig,
    SpatialIntegrationPart,
    SpatialIntegrationPayload,
    SpatialIntegrationTrialStage,
    build_spatial_integration_test,
)


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _wait_for_question(
    *,
    engine: object,
    clock: FakeClock,
    max_steps: int = 120,
    dt: float = 0.05,
) -> SpatialIntegrationPayload:
    for _ in range(max_steps):
        snap = engine.snapshot()
        payload = snap.payload if isinstance(snap.payload, SpatialIntegrationPayload) else None
        if payload is not None and payload.trial_stage is SpatialIntegrationTrialStage.QUESTION:
            return payload
        clock.advance(dt)
        engine.update()
    raise AssertionError("Timed out waiting for question stage.")


def _answer_current(payload: SpatialIntegrationPayload) -> str:
    if payload.answer_mode is SpatialIntegrationAnswerMode.GRID_CLICK:
        return payload.correct_answer_token
    return str(payload.correct_code)


def test_headless_run_covers_both_parts_and_binary_summary() -> None:
    clock = FakeClock()
    engine = build_spatial_integration_test(
        clock=clock,
        seed=321,
        difficulty=0.65,
        config=SpatialIntegrationConfig(
            practice_scenes_per_part=1,
            static_scored_duration_s=0.01,
            aircraft_scored_duration_s=0.01,
            static_study_s=0.15,
            aircraft_study_s=0.15,
            question_time_limit_s=0.3,
        ),
    )

    # Part 1 practice
    engine.start_practice()
    assert engine.phase is Phase.PRACTICE
    for _ in range(3):
        p = _wait_for_question(engine=engine, clock=clock)
        assert p.part is SpatialIntegrationPart.STATIC
        assert engine.submit_answer(_answer_current(p)) is True
    assert engine.phase is Phase.PRACTICE_DONE

    # Part 1 scored: one question only because the part timer is already spent during study.
    engine.start_scored()
    assert engine.phase is Phase.SCORED
    p = _wait_for_question(engine=engine, clock=clock)
    assert p.part is SpatialIntegrationPart.STATIC
    assert engine.submit_answer(_answer_current(p)) is True
    assert engine.phase is Phase.PRACTICE_DONE

    # Part 2 practice
    engine.start_scored()
    assert engine.phase is Phase.PRACTICE
    for _ in range(3):
        p = _wait_for_question(engine=engine, clock=clock)
        assert p.part is SpatialIntegrationPart.AIRCRAFT
        assert engine.submit_answer(_answer_current(p)) is True
    assert engine.phase is Phase.PRACTICE_DONE

    # Part 2 scored: answer incorrectly.
    engine.start_scored()
    assert engine.phase is Phase.SCORED
    p = _wait_for_question(engine=engine, clock=clock)
    assert p.part is SpatialIntegrationPart.AIRCRAFT
    if p.answer_mode is SpatialIntegrationAnswerMode.OPTION_PICK:
        wrong = next(opt.code for opt in p.options if opt.code != p.correct_code)
        assert engine.submit_answer(str(wrong)) is True
    else:
        wrong = "A1" if p.correct_answer_token != "A1" else "B1"
        assert engine.submit_answer(wrong) is True

    assert engine.phase is Phase.RESULTS

    summary = engine.scored_summary()
    assert summary.attempted == 2
    assert summary.correct == 1
    assert summary.accuracy == pytest.approx(0.5)
    assert summary.score_ratio == pytest.approx(0.5)
    assert summary.mean_response_time_s is not None

    scored_events = [e for e in engine.events() if e.phase is Phase.SCORED]
    assert len(scored_events) == 2
    assert scored_events[0].score == pytest.approx(1.0)
    assert scored_events[1].score == pytest.approx(0.0)
