from __future__ import annotations

from dataclasses import dataclass

import pytest

from cfast_trainer.cognitive_core import Phase
from cfast_trainer.vigilance import VigilanceConfig, VigilancePayload, build_vigilance_test


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


def test_headless_scripted_run_produces_expected_summary() -> None:
    seed = 7301
    difficulty = 0.58
    clock = FakeClock()

    engine = build_vigilance_test(
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        config=VigilanceConfig(
            practice_duration_s=3.0,
            scored_duration_s=6.0,
            spawn_interval_s=0.9,
            max_active_symbols=4,
        ),
    )

    engine.start_practice()
    practice_captures = 0
    practice_seen: set[int] = set()

    while engine.phase is Phase.PRACTICE:
        clock.advance(0.1)
        engine.update()
        payload = engine.snapshot().payload
        assert isinstance(payload, VigilancePayload)

        for symbol in payload.symbols:
            if symbol.symbol_id in practice_seen:
                continue
            practice_seen.add(symbol.symbol_id)
            if practice_captures < 2:
                assert engine.submit_answer(f"{symbol.row},{symbol.col}") is True
                practice_captures += 1
                break

    assert engine.phase is Phase.PRACTICE_DONE
    practice_payload = engine.snapshot().payload
    assert isinstance(practice_payload, VigilancePayload)
    assert practice_payload.points_total == 2

    engine.start_scored()
    scored_seen: set[int] = set()
    seen_count = 0

    while engine.phase is Phase.SCORED:
        clock.advance(0.1)
        engine.update()
        payload = engine.snapshot().payload
        assert isinstance(payload, VigilancePayload)

        for symbol in payload.symbols:
            if symbol.symbol_id in scored_seen:
                continue
            scored_seen.add(symbol.symbol_id)
            seen_count += 1
            if seen_count in (1, 3):
                assert engine.submit_answer(f"{symbol.row}{symbol.col}") is True
                break

    assert engine.phase is Phase.RESULTS

    summary = engine.scored_summary()
    assert summary.attempted == 2
    assert summary.correct == 2
    assert summary.accuracy == pytest.approx(1.0)
    assert summary.points == 4
    assert summary.missed == 1
    assert summary.mean_capture_time_s is not None
    assert summary.mean_capture_time_s == pytest.approx(0.029623139224990158)
