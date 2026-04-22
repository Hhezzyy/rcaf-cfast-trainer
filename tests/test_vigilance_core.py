from __future__ import annotations

from dataclasses import dataclass

import pytest

from cfast_trainer.vigilance import (
    VigilanceConfig,
    VigilancePayload,
    VigilanceSymbolKind,
    build_vigilance_test,
    _vigilance_difficulty_params,
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


def test_difficulty_params_make_easy_stream_less_pressured_than_hard_stream() -> None:
    config = VigilanceConfig(
        practice_duration_s=0.0,
        scored_duration_s=8.0,
        spawn_interval_s=0.85,
        max_active_symbols=7,
    )

    easy = _vigilance_difficulty_params(0.0, config)
    hard = _vigilance_difficulty_params(1.0, config)

    assert easy.max_active_symbols < hard.max_active_symbols
    assert easy.spawn_interval_s > hard.spawn_interval_s
    assert easy.lifetime_scale > hard.lifetime_scale
    assert easy.lifetime_floor_s > hard.lifetime_floor_s
    assert easy.symbol_kind_thresholds[0] > hard.symbol_kind_thresholds[0]
    assert easy.symbol_kind_thresholds[1] > hard.symbol_kind_thresholds[1]
    assert easy.symbol_kind_thresholds[2] > hard.symbol_kind_thresholds[2]

    easy_engine = build_vigilance_test(clock=FakeClock(), seed=901, difficulty=0.0, config=config)
    hard_engine = build_vigilance_test(clock=FakeClock(), seed=901, difficulty=1.0, config=config)

    easy_star_lifetime = easy_engine._symbol_lifetime_s(kind=VigilanceSymbolKind.STAR)
    hard_star_lifetime = hard_engine._symbol_lifetime_s(kind=VigilanceSymbolKind.STAR)

    assert easy_star_lifetime > hard_star_lifetime


def test_high_difficulty_produces_more_spawn_and_overlap_pressure_than_low_difficulty() -> None:
    def observe_pressure(difficulty: float) -> tuple[int, int, int]:
        clock = FakeClock()
        engine = build_vigilance_test(
            clock=clock,
            seed=902,
            difficulty=difficulty,
            config=VigilanceConfig(
                practice_duration_s=0.0,
                scored_duration_s=8.0,
                spawn_interval_s=0.85,
                max_active_symbols=7,
            ),
        )
        engine.start_scored()

        seen_ids: set[int] = set()
        max_visible = 0
        while engine.phase.value == "scored":
            clock.advance(0.1)
            engine.update()
            payload = engine.snapshot().payload
            assert isinstance(payload, VigilancePayload)
            seen_ids.update(symbol.symbol_id for symbol in payload.symbols)
            max_visible = max(max_visible, len(payload.symbols))

        return len(seen_ids), max_visible, engine.scored_summary().missed

    easy_seen, easy_max_visible, easy_missed = observe_pressure(0.0)
    hard_seen, hard_max_visible, hard_missed = observe_pressure(1.0)

    assert easy_seen < hard_seen
    assert easy_max_visible < hard_max_visible
    assert easy_missed < hard_missed


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
    assert e1.events() == e2.events()


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

    events = engine.events()
    assert len(events) == 1
    event = events[0]
    assert event.correct_answer == (symbol.row * 10) + symbol.col
    assert event.user_answer == (symbol.row * 10) + symbol.col
    assert event.raw == f"{symbol.row},{symbol.col}"
    assert event.is_correct is True
    assert event.score == symbol.points
    assert event.max_score == symbol.points

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

    events = engine.events()
    assert len(events) == 1
    event = events[0]
    assert event.correct_answer == (symbol.row * 10) + symbol.col
    assert event.user_answer == 0
    assert event.raw == ""
    assert event.is_correct is False
    assert event.score == 0
    assert event.max_score == symbol.points

    summary = engine.scored_summary()
    assert summary.attempted == 0
    assert summary.correct == 0
    assert summary.points == 0
    assert summary.missed == 1
    assert summary.mean_capture_time_s is None


def test_events_are_deterministic_for_same_seed_and_script() -> None:
    config = VigilanceConfig(
        practice_duration_s=0.0,
        scored_duration_s=8.0,
        spawn_interval_s=0.8,
        max_active_symbols=4,
    )
    c1 = FakeClock()
    c2 = FakeClock()
    e1 = build_vigilance_test(clock=c1, seed=515, difficulty=0.58, config=config)
    e2 = build_vigilance_test(clock=c2, seed=515, difficulty=0.58, config=config)

    e1.start_scored()
    e2.start_scored()

    seen_ids: set[int] = set()
    while e1.phase.value != "results" or e2.phase.value != "results":
        c1.advance(0.1)
        c2.advance(0.1)
        e1.update()
        e2.update()

        payload = e1.snapshot().payload
        assert payload == e2.snapshot().payload
        assert isinstance(payload, VigilancePayload)
        for symbol in payload.symbols:
            if symbol.symbol_id in seen_ids:
                continue
            seen_ids.add(symbol.symbol_id)
            if symbol.symbol_id % 3 == 0:
                assert e1.submit_answer(f"{symbol.row},{symbol.col}") is True
                assert e2.submit_answer(f"{symbol.row},{symbol.col}") is True
            break
        if e1.phase.value == "results" and e2.phase.value == "results":
            break

    assert e1.events() == e2.events()


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
