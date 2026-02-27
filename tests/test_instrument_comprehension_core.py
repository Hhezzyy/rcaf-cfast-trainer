from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from cfast_trainer.cognitive_core import Problem
from cfast_trainer.instrument_comprehension import (
    InstrumentComprehensionConfig,
    InstrumentComprehensionGenerator,
    InstrumentComprehensionPayload,
    InstrumentComprehensionScorer,
    InstrumentComprehensionTrialKind,
    InstrumentOption,
    InstrumentState,
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


def test_generator_emits_all_required_trial_kinds() -> None:
    gen = InstrumentComprehensionGenerator(seed=1234)
    seen: set[InstrumentComprehensionTrialKind] = set()
    for _ in range(120):
        p = gen.next_problem(difficulty=0.6)
        payload = p.payload
        assert isinstance(payload, InstrumentComprehensionPayload)
        seen.add(payload.kind)

    assert seen == {
        InstrumentComprehensionTrialKind.INSTRUMENTS_TO_DESCRIPTION,
        InstrumentComprehensionTrialKind.DESCRIPTION_TO_INSTRUMENTS,
        InstrumentComprehensionTrialKind.INSTRUMENTS_TO_AIRCRAFT,
    }


def test_scoring_exact_and_estimation_behavior() -> None:
    scorer = InstrumentComprehensionScorer()

    base = InstrumentState(
        speed_kts=240,
        altitude_ft=1500,
        vertical_rate_fpm=-400,
        bank_deg=-10,
        pitch_deg=-4,
        heading_deg=315,
        slip=0,
    )
    payload = InstrumentComprehensionPayload(
        kind=InstrumentComprehensionTrialKind.INSTRUMENTS_TO_DESCRIPTION,
        prompt_state=base,
        prompt_description="Speed 240 knots ...",
        options=(
            InstrumentOption(code=1, state=base, description="A"),
            InstrumentOption(code=2, state=replace(base, heading_deg=330), description="B"),
            InstrumentOption(code=3, state=replace(base, altitude_ft=2200), description="C"),
            InstrumentOption(
                code=4, state=replace(base, bank_deg=25, pitch_deg=8), description="D"
            ),
        ),
        option_errors=(0, 15, 42, 98),
        full_credit_error=0,
        zero_credit_error=90,
    )
    problem = Problem(prompt="Pick best", answer=1, payload=payload)

    assert scorer.score(problem=problem, user_answer=1, raw="1") == pytest.approx(1.0)
    assert scorer.score(problem=problem, user_answer=2, raw="2") == pytest.approx(
        (90 - 15) / 90, abs=1e-9
    )
    assert scorer.score(problem=problem, user_answer=3, raw="3") == pytest.approx(
        (90 - 42) / 90, abs=1e-9
    )
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
