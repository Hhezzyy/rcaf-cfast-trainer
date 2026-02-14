from __future__ import annotations

from dataclasses import dataclass

import pytest

from cfast_trainer.cognitive_core import Problem
from cfast_trainer.instrument_comprehension import (
    InstrumentComprehensionConfig,
    InstrumentComprehensionGenerator,
    InstrumentComprehensionPayload,
    InstrumentComprehensionScorer,
    InstrumentComprehensionTrialKind,
    InstrumentOption,
    build_instrument_comprehension_test,
)


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


def test_generator_determinism_same_seed_same_sequence() -> None:
    seed = 901
    g1 = InstrumentComprehensionGenerator(seed=seed)
    g2 = InstrumentComprehensionGenerator(seed=seed)

    seq1 = [g1.next_problem(difficulty=0.6) for _ in range(30)]
    seq2 = [g2.next_problem(difficulty=0.6) for _ in range(30)]

    assert [(p.prompt, p.answer, p.payload) for p in seq1] == [
        (p.prompt, p.answer, p.payload) for p in seq2
    ]


def test_scoring_exact_and_estimation_behavior() -> None:
    scorer = InstrumentComprehensionScorer()

    payload = InstrumentComprehensionPayload(
        kind=InstrumentComprehensionTrialKind.ATTITUDE_MATCH,
        heading_deg=90,
        bank_deg=20,
        pitch_deg=8,
        verbal_cue="Instrument cue",
        options=(
            InstrumentOption(
                code=1,
                label="bank right 20°, nose up 8°, HDG 090",
                bank_deg=20,
                pitch_deg=8,
                heading_deg=90,
            ),
            InstrumentOption(
                code=2,
                label="bank right 20°, nose up 8°, HDG 110",
                bank_deg=20,
                pitch_deg=8,
                heading_deg=110,
            ),
            InstrumentOption(
                code=3,
                label="bank left 20°, nose up 8°, HDG 090",
                bank_deg=-20,
                pitch_deg=8,
                heading_deg=90,
            ),
            InstrumentOption(
                code=4,
                label="bank left 20°, nose down 8°, HDG 170",
                bank_deg=-20,
                pitch_deg=-8,
                heading_deg=170,
            ),
        ),
        option_errors=(0, 20, 55, 100),
        full_credit_error=0,
        zero_credit_error=80,
    )
    problem = Problem(prompt="Pick the best option", answer=1, payload=payload)

    assert scorer.score(problem=problem, user_answer=1, raw="1") == pytest.approx(1.0)
    assert scorer.score(problem=problem, user_answer=2, raw="2") == pytest.approx((80 - 20) / 80, abs=1e-9)
    assert scorer.score(problem=problem, user_answer=3, raw="3") == pytest.approx((80 - 55) / 80, abs=1e-9)
    assert scorer.score(problem=problem, user_answer=4, raw="4") == pytest.approx(0.0)
    assert scorer.score(problem=problem, user_answer=9, raw="9") == pytest.approx(0.0)


def test_timer_boundary_transitions_to_results_and_rejects_late_submit() -> None:
    clock = FakeClock()
    engine = build_instrument_comprehension_test(
        clock=clock,
        seed=7,
        difficulty=0.5,
        config=InstrumentComprehensionConfig(scored_duration_s=2.0, practice_questions=0),
    )
    engine.start_scored()

    assert engine.time_remaining_s() == pytest.approx(2.0)
    clock.advance(2.0)
    engine.update()

    assert engine.phase.value == "results"
    assert engine.submit_answer("1") is False
