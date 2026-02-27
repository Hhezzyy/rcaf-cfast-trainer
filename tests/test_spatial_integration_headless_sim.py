from __future__ import annotations

from dataclasses import dataclass

import pytest

from cfast_trainer.cognitive_core import Phase
from cfast_trainer.spatial_integration import (
    SpatialIntegrationConfig,
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


def _answer_code(
    *,
    engine: object,
    clock: FakeClock,
    code: int | str,
    response_time_s: float,
) -> bool:
    clock.advance(response_time_s)
    engine.update()
    return bool(engine.submit_answer(str(code)))


def test_headless_sectioned_run_produces_expected_scored_summary() -> None:
    clock = FakeClock()
    engine = build_spatial_integration_test(
        clock=clock,
        seed=321,
        difficulty=0.65,
        config=SpatialIntegrationConfig(
            practice_questions=1,
            scored_questions_per_section=1,
            practice_memorize_s=0.15,
            scored_memorize_s=0.15,
        ),
    )

    # Section A practice
    engine.start_practice()
    assert engine.phase is Phase.PRACTICE
    p = _wait_for_question(engine=engine, clock=clock)
    assert (
        _answer_code(engine=engine, clock=clock, code=p.correct_code, response_time_s=0.5) is True
    )
    assert engine.phase is Phase.PRACTICE_DONE

    # Section A scored: exact correct.
    engine.start_scored()
    assert engine.phase is Phase.SCORED
    p = _wait_for_question(engine=engine, clock=clock)
    assert (
        _answer_code(engine=engine, clock=clock, code=p.correct_code, response_time_s=0.5) is True
    )
    assert engine.phase is Phase.PRACTICE_DONE

    # Section B practice
    engine.start_scored()
    assert engine.phase is Phase.PRACTICE
    p = _wait_for_question(engine=engine, clock=clock)
    assert (
        _answer_code(engine=engine, clock=clock, code=p.correct_code, response_time_s=0.5) is True
    )
    assert engine.phase is Phase.PRACTICE_DONE

    # Section B scored: nearest wrong for partial credit.
    engine.start_scored()
    assert engine.phase is Phase.SCORED
    p = _wait_for_question(engine=engine, clock=clock)
    near = min(
        (opt for opt in p.options if opt.code != p.correct_code),
        key=lambda opt: int(opt.error),
    )
    assert _answer_code(engine=engine, clock=clock, code=near.code, response_time_s=0.5) is True
    assert engine.phase is Phase.PRACTICE_DONE

    # Section C practice
    engine.start_scored()
    assert engine.phase is Phase.PRACTICE
    p = _wait_for_question(engine=engine, clock=clock)
    assert (
        _answer_code(engine=engine, clock=clock, code=p.correct_code, response_time_s=0.5) is True
    )
    assert engine.phase is Phase.PRACTICE_DONE

    # Section C scored: invalid answer for zero credit.
    engine.start_scored()
    assert engine.phase is Phase.SCORED
    _ = _wait_for_question(engine=engine, clock=clock)
    assert _answer_code(engine=engine, clock=clock, code="9", response_time_s=0.5) is True
    assert engine.phase is Phase.RESULTS

    summary = engine.scored_summary()
    assert summary.attempted == 3
    assert summary.correct == 1
    assert summary.accuracy == pytest.approx(1 / 3)
    assert summary.mean_response_time_s == pytest.approx(0.5, abs=0.03)
    assert summary.score_ratio > (1 / 3)
    assert summary.score_ratio < 1.0

    scored_events = [e for e in engine.events() if e.phase is Phase.SCORED]
    assert len(scored_events) == 3
    assert scored_events[0].score == pytest.approx(1.0)
    assert 0.0 < scored_events[1].score < 1.0
    assert scored_events[2].score == pytest.approx(0.0)
