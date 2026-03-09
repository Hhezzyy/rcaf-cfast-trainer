from __future__ import annotations

from dataclasses import dataclass

import pytest

from cfast_trainer.vigilance import (
    VigilanceConfig,
    VigilancePayload,
    VigilanceSymbolKind,
    build_vigilance_test,
)


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


def _advance_until_symbol(
    *,
    engine,
    clock: FakeClock,
    step_s: float = 0.05,
    timeout_s: float = 5.0,
) -> VigilancePayload:
    elapsed = 0.0
    while elapsed < timeout_s:
        clock.advance(step_s)
        engine.update()
        payload = engine.snapshot().payload
        assert isinstance(payload, VigilancePayload)
        if payload.symbols:
            return payload
        elapsed += step_s
    raise AssertionError("expected a visible vigilance symbol before timeout")


def test_symbol_stream_is_deterministic_for_same_seed_and_script() -> None:
    config = VigilanceConfig(
        practice_duration_s=0.0,
        scored_duration_s=4.0,
        spawn_interval_s=1.0,
        max_active_symbols=5,
    )
    c1 = FakeClock()
    c2 = FakeClock()
    e1 = build_vigilance_test(clock=c1, seed=808, difficulty=0.62, config=config)
    e2 = build_vigilance_test(clock=c2, seed=808, difficulty=0.62, config=config)

    e1.start_scored()
    e2.start_scored()

    for step in range(40):
        c1.advance(0.1)
        c2.advance(0.1)
        e1.update()
        e2.update()

        snap1 = e1.snapshot()
        snap2 = e2.snapshot()
        assert snap1 == snap2

        payload = snap1.payload
        assert isinstance(payload, VigilancePayload)
        if payload.symbols and step % 4 == 0:
            symbol = payload.symbols[0]
            answer = f"{symbol.row},{symbol.col}"
            assert e1.submit_answer(answer) is True
            assert e2.submit_answer(answer) is True

    assert e1.scored_summary() == e2.scored_summary()


def test_payload_legend_and_visible_symbol_shape_match_expected_grid() -> None:
    clock = FakeClock()
    engine = build_vigilance_test(
        clock=clock,
        seed=101,
        difficulty=0.62,
        config=VigilanceConfig(
            practice_duration_s=3.0,
            scored_duration_s=6.0,
            spawn_interval_s=10.0,
            max_active_symbols=7,
        ),
    )

    engine.start_practice()
    payload = _advance_until_symbol(engine=engine, clock=clock)

    assert payload.rows == 9
    assert payload.cols == 9
    assert payload.legend == (
        (VigilanceSymbolKind.STAR, 1),
        (VigilanceSymbolKind.DIAMOND, 2),
        (VigilanceSymbolKind.TRIANGLE, 3),
        (VigilanceSymbolKind.HEXAGON, 4),
    )

    symbol = payload.symbols[0]
    assert 1 <= symbol.row <= 9
    assert 1 <= symbol.col <= 9
    assert symbol.points == {
        VigilanceSymbolKind.STAR: 1,
        VigilanceSymbolKind.DIAMOND: 2,
        VigilanceSymbolKind.TRIANGLE: 3,
        VigilanceSymbolKind.HEXAGON: 4,
    }[symbol.kind]
    assert symbol.time_left_s > 0.0


def test_capture_visible_symbol_updates_scored_summary_and_payload() -> None:
    clock = FakeClock()
    engine = build_vigilance_test(
        clock=clock,
        seed=22,
        difficulty=0.5,
        config=VigilanceConfig(
            practice_duration_s=0.0,
            scored_duration_s=5.0,
            spawn_interval_s=10.0,
            max_active_symbols=4,
        ),
    )

    engine.start_scored()
    assert engine.can_exit() is False

    payload = _advance_until_symbol(engine=engine, clock=clock)
    symbol = payload.symbols[0]

    assert engine.submit_answer(f"{symbol.row},{symbol.col}") is True

    updated = engine.snapshot().payload
    assert isinstance(updated, VigilancePayload)
    assert updated.captured_total == 1
    assert updated.missed_total == 0
    assert updated.points_total == symbol.points
    assert all(active.symbol_id != symbol.symbol_id for active in updated.symbols)

    summary = engine.scored_summary()
    assert summary.attempted == 1
    assert summary.correct == 1
    assert summary.points == symbol.points
    assert summary.mean_capture_time_s is not None


def test_expired_symbol_counts_as_missed_in_scored_phase() -> None:
    clock = FakeClock()
    engine = build_vigilance_test(
        clock=clock,
        seed=31,
        difficulty=0.5,
        config=VigilanceConfig(
            practice_duration_s=0.0,
            scored_duration_s=8.0,
            spawn_interval_s=10.0,
            max_active_symbols=4,
        ),
    )

    engine.start_scored()
    payload = _advance_until_symbol(engine=engine, clock=clock)
    symbol = payload.symbols[0]

    clock.advance(symbol.time_left_s + 0.05)
    engine.update()

    updated = engine.snapshot().payload
    assert isinstance(updated, VigilancePayload)
    assert updated.captured_total == 0
    assert updated.missed_total == 1
    assert all(active.symbol_id != symbol.symbol_id for active in updated.symbols)

    summary = engine.scored_summary()
    assert summary.attempted == 0
    assert summary.correct == 0
    assert summary.points == 0
    assert summary.missed == 1
    assert summary.mean_capture_time_s is None


def test_timer_boundary_transitions_to_results_and_rejects_late_submit() -> None:
    clock = FakeClock()
    engine = build_vigilance_test(
        clock=clock,
        seed=7,
        difficulty=0.5,
        config=VigilanceConfig(
            practice_duration_s=0.0,
            scored_duration_s=2.0,
            spawn_interval_s=1.0,
            max_active_symbols=4,
        ),
    )

    engine.start_scored()

    assert engine.time_remaining_s() == pytest.approx(2.0)
    clock.advance(2.0)
    engine.update()

    assert engine.phase.value == "results"
    assert engine.can_exit() is True
    assert engine.submit_answer("1,1") is False
