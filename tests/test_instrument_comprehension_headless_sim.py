from __future__ import annotations

from dataclasses import dataclass

import pytest

from cfast_trainer.cognitive_core import Phase
from cfast_trainer.instrument_comprehension import (
    InstrumentComprehensionConfig,
    InstrumentComprehensionGenerator,
    InstrumentComprehensionPayload,
    build_instrument_comprehension_test,
)


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


def test_headless_scripted_run_produces_expected_summary_and_partial_scores() -> None:
    seed = 4242
    difficulty = 0.6
    clock = FakeClock()

    engine = build_instrument_comprehension_test(
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        config=InstrumentComprehensionConfig(scored_duration_s=6.0, practice_questions=1),
    )
    mirror = InstrumentComprehensionGenerator(seed=seed)

    engine.start_practice()
    assert engine.phase is Phase.PRACTICE

    p_practice = mirror.next_problem(difficulty=difficulty)
    clock.advance(0.2)
    assert engine.submit_answer(str(p_practice.answer)) is True
    assert engine.phase is Phase.PRACTICE_DONE

    engine.start_scored()
    assert engine.phase is Phase.SCORED

    p1 = mirror.next_problem(difficulty=difficulty)
    clock.advance(0.5)
    assert engine.submit_answer(str(p1.answer)) is True

    p2 = mirror.next_problem(difficulty=difficulty)
    assert isinstance(p2.payload, InstrumentComprehensionPayload)
    near_option = min(
        (
            (opt.code, err)
            for opt, err in zip(p2.payload.options, p2.payload.option_errors, strict=False)
            if opt.code != p2.answer
        ),
        key=lambda pair: pair[1],
    )[0]
    clock.advance(0.5)
    assert engine.submit_answer(str(near_option)) is True

    p3 = mirror.next_problem(difficulty=difficulty)
    assert isinstance(p3.payload, InstrumentComprehensionPayload)
    worst_option = max(
        ((opt.code, err) for opt, err in zip(p3.payload.options, p3.payload.option_errors, strict=False)),
        key=lambda pair: pair[1],
    )[0]
    clock.advance(0.5)
    assert engine.submit_answer(str(worst_option)) is True

    clock.advance(6.0)
    engine.update()
    assert engine.phase is Phase.RESULTS

    summary = engine.scored_summary()
    assert summary.attempted == 3
    assert summary.correct == 1
    assert summary.accuracy == pytest.approx(1 / 3)
    assert summary.mean_response_time_s == pytest.approx(0.5)

    scored_events = [e for e in engine.events() if e.phase is Phase.SCORED]
    assert len(scored_events) == 3
    assert scored_events[0].score == pytest.approx(1.0)
    assert 0.0 < scored_events[1].score < 1.0
    assert scored_events[2].score == pytest.approx(0.0)
