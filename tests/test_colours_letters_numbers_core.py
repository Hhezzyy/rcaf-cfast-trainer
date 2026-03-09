from __future__ import annotations

from dataclasses import dataclass

from cfast_trainer.cognitive_core import Phase, SeededRng
from cfast_trainer.colours_letters_numbers import (
    ColoursLettersNumbersConfig,
    ColoursLettersNumbersGenerator,
    _LiveDiamond,
    build_colours_letters_numbers_test,
)


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


def test_generator_is_deterministic_for_same_seed() -> None:
    seed = 12345
    g1 = ColoursLettersNumbersGenerator(SeededRng(seed))
    g2 = ColoursLettersNumbersGenerator(SeededRng(seed))

    seq1 = [g1.next_trial(difficulty=0.5) for _ in range(10)]
    seq2 = [g2.next_trial(difficulty=0.5) for _ in range(10)]

    assert seq1 == seq2


def test_sequence_is_shown_first_then_corner_options_activate() -> None:
    clock = FakeClock()
    cfg = ColoursLettersNumbersConfig(
        scored_duration_s=20.0,
        practice_rounds=1,
        round_duration_s=8.0,
        sequence_show_s=1.0,
        memory_recall_delay_s=0.75,
        memory_recall_delay_max_s=0.75,
        diamond_spawn_interval_s=99.0,
        diamond_spawn_interval_max_s=99.0,
        diamond_speed_norm_per_s=0.2,
        diamond_speed_max_norm_per_s=0.2,
        max_live_diamonds=2,
    )
    engine = build_colours_letters_numbers_test(clock=clock, seed=77, difficulty=0.5, config=cfg)

    engine.start_practice()
    p1 = engine.snapshot().payload
    assert p1 is not None
    assert p1.target_sequence is not None
    assert p1.options_active is False

    clock.advance(1.01)
    engine.update()
    p2 = engine.snapshot().payload
    assert p2 is not None
    assert p2.target_sequence is None
    assert p2.options_active is False

    clock.advance(0.75)
    engine.update()
    p3 = engine.snapshot().payload
    assert p3 is not None
    assert p3.target_sequence is None
    assert p3.options_active is True


def test_color_lane_keys_follow_qwer_left_to_right() -> None:
    seed = 42
    clock = FakeClock()
    cfg = ColoursLettersNumbersConfig(
        scored_duration_s=30.0,
        practice_rounds=0,
        round_duration_s=8.0,
        sequence_show_s=0.5,
        memory_recall_delay_s=0.0,
        memory_recall_delay_max_s=0.0,
        diamond_spawn_interval_s=99.0,
        diamond_spawn_interval_max_s=99.0,
        diamond_speed_norm_per_s=0.2,
        diamond_speed_max_norm_per_s=0.2,
        max_live_diamonds=2,
    )

    engine = build_colours_letters_numbers_test(clock=clock, seed=seed, difficulty=0.5, config=cfg)
    engine.start_practice()
    engine.start_scored()

    engine._diamonds = [
        _LiveDiamond(id=1, color="RED", row=0, x_norm=0.60, speed_norm_per_s=0.2)
    ]  # type: ignore[attr-defined]
    assert engine.submit_answer("CLR:Q") is True

    engine._diamonds = [
        _LiveDiamond(id=2, color="YELLOW", row=0, x_norm=0.72, speed_norm_per_s=0.2)
    ]  # type: ignore[attr-defined]
    assert engine.submit_answer("CLR:W") is True

    engine._diamonds = [
        _LiveDiamond(id=3, color="GREEN", row=0, x_norm=0.83, speed_norm_per_s=0.2)
    ]  # type: ignore[attr-defined]
    assert engine.submit_answer("CLR:E") is True

    engine._diamonds = [
        _LiveDiamond(id=4, color="BLUE", row=0, x_norm=0.94, speed_norm_per_s=0.2)
    ]  # type: ignore[attr-defined]
    assert engine.submit_answer("CLR:R") is True


def test_scoring_memory_and_math_are_distinct_channels() -> None:
    seed = 42
    clock = FakeClock()
    cfg = ColoursLettersNumbersConfig(
        scored_duration_s=30.0,
        practice_rounds=0,
        round_duration_s=8.0,
        sequence_show_s=0.5,
        memory_recall_delay_s=0.0,
        memory_recall_delay_max_s=0.0,
        diamond_spawn_interval_s=99.0,
        diamond_spawn_interval_max_s=99.0,
        diamond_speed_norm_per_s=0.2,
        diamond_speed_max_norm_per_s=0.2,
        max_live_diamonds=2,
    )

    mirror = ColoursLettersNumbersGenerator(SeededRng(seed))
    memory_trial = mirror.next_trial(difficulty=0.5)
    math_trial = mirror.next_trial(difficulty=0.5)

    engine = build_colours_letters_numbers_test(clock=clock, seed=seed, difficulty=0.5, config=cfg)
    engine.start_practice()
    engine.start_scored()

    clock.advance(0.6)
    engine.update()
    assert engine.submit_answer(f"MEM:{memory_trial.expected_option_code}") is True
    assert engine.submit_answer(str(math_trial.math_answer)) is True

    s = engine.scored_summary()
    assert s.attempted == 2
    assert s.correct == 2
    assert s.accuracy == 1.0


def test_memory_cycle_rollover_does_not_reset_math_or_diamonds() -> None:
    clock = FakeClock()
    cfg = ColoursLettersNumbersConfig(
        scored_duration_s=30.0,
        practice_rounds=0,
        round_duration_s=0.7,
        sequence_show_s=0.2,
        memory_recall_delay_s=0.0,
        memory_recall_delay_max_s=0.0,
        diamond_spawn_interval_s=0.1,
        diamond_spawn_interval_max_s=0.1,
        diamond_speed_norm_per_s=0.2,
        diamond_speed_max_norm_per_s=0.2,
        max_live_diamonds=5,
    )
    engine = build_colours_letters_numbers_test(clock=clock, seed=123, difficulty=0.5, config=cfg)
    engine.start_practice()
    engine.start_scored()

    clock.advance(0.35)
    engine.update()
    p1 = engine.snapshot().payload
    assert p1 is not None
    assert p1.diamonds
    tracked_id = p1.diamonds[0].id
    math_prompt = p1.math_prompt

    # End current memory cycle and force a rollover to the next one.
    clock.advance(0.6)
    engine.update()
    p2 = engine.snapshot().payload
    assert p2 is not None

    # Memory rollover should not reset math or the colour-diamond stream.
    assert p2.math_prompt == math_prompt
    assert tracked_id in {d.id for d in p2.diamonds}


def test_timer_boundary_transitions_to_results_and_rejects_late_submit() -> None:
    clock = FakeClock()
    cfg = ColoursLettersNumbersConfig(
        scored_duration_s=1.0,
        practice_rounds=0,
        round_duration_s=8.0,
        sequence_show_s=0.5,
        memory_recall_delay_s=0.0,
        memory_recall_delay_max_s=0.0,
        diamond_spawn_interval_s=99.0,
        diamond_spawn_interval_max_s=99.0,
        diamond_speed_norm_per_s=0.2,
        diamond_speed_max_norm_per_s=0.2,
        max_live_diamonds=2,
    )
    engine = build_colours_letters_numbers_test(clock=clock, seed=7, difficulty=0.5, config=cfg)
    engine.start_practice()
    engine.start_scored()

    clock.advance(1.0)
    engine.update()

    assert engine.phase is Phase.RESULTS
    assert engine.submit_answer("0") is False


def test_large_update_delta_is_capped_to_prevent_diamond_state_jumps() -> None:
    clock = FakeClock()
    cfg = ColoursLettersNumbersConfig(
        scored_duration_s=120.0,
        practice_rounds=1,
        round_duration_s=120.0,
        sequence_show_s=0.5,
        memory_recall_delay_s=0.0,
        memory_recall_delay_max_s=0.0,
        diamond_spawn_interval_s=0.05,
        diamond_spawn_interval_max_s=0.05,
        diamond_speed_norm_per_s=0.5,
        diamond_speed_max_norm_per_s=0.5,
        max_live_diamonds=4,
    )
    engine = build_colours_letters_numbers_test(clock=clock, seed=17, difficulty=0.5, config=cfg)
    engine.start_practice()

    clock.advance(0.06)
    engine.update()
    p1 = engine.snapshot().payload
    assert p1 is not None
    assert p1.diamonds
    tracked_id = p1.diamonds[0].id

    clock.advance(6.0)
    engine.update()
    p2 = engine.snapshot().payload
    assert p2 is not None
    ids = {d.id for d in p2.diamonds}

    # Large frame gaps should not instantly advance all live diamonds out of view.
    assert tracked_id in ids


def test_memory_recall_delay_varies_per_round_within_configured_bounds() -> None:
    clock = FakeClock()
    cfg = ColoursLettersNumbersConfig(
        scored_duration_s=20.0,
        practice_rounds=2,
        round_duration_s=0.3,
        sequence_show_s=0.1,
        memory_recall_delay_s=0.05,
        memory_recall_delay_max_s=0.2,
        diamond_spawn_interval_s=99.0,
        diamond_spawn_interval_max_s=99.0,
        diamond_speed_norm_per_s=0.2,
        diamond_speed_max_norm_per_s=0.2,
        max_live_diamonds=2,
    )
    engine = build_colours_letters_numbers_test(clock=clock, seed=77, difficulty=0.5, config=cfg)

    engine.start_practice()
    first_delay = engine._memory_recall_delay_s_current  # type: ignore[attr-defined]
    assert 0.05 <= first_delay <= 0.2

    clock.advance(cfg.sequence_show_s + first_delay + cfg.round_duration_s + 0.01)
    engine.update()
    second_delay = engine._memory_recall_delay_s_current  # type: ignore[attr-defined]
    assert 0.05 <= second_delay <= 0.2
    assert second_delay != first_delay


def test_diamond_spawn_and_speed_vary_within_configured_ranges() -> None:
    clock = FakeClock()
    cfg = ColoursLettersNumbersConfig(
        scored_duration_s=20.0,
        practice_rounds=1,
        round_duration_s=20.0,
        sequence_show_s=0.1,
        memory_recall_delay_s=0.0,
        memory_recall_delay_max_s=0.0,
        diamond_spawn_interval_s=0.1,
        diamond_spawn_interval_max_s=0.25,
        diamond_speed_norm_per_s=0.2,
        diamond_speed_max_norm_per_s=0.35,
        max_live_diamonds=6,
    )
    engine = build_colours_letters_numbers_test(clock=clock, seed=321, difficulty=0.5, config=cfg)
    engine.start_practice()

    intervals = []
    speeds = []
    for _ in range(4):
        current_interval = engine._spawn_cooldown_s  # type: ignore[attr-defined]
        intervals.append(current_interval)
        clock.advance(current_interval + 0.01)
        engine.update()
        speeds.extend(d.speed_norm_per_s for d in engine._diamonds)  # type: ignore[attr-defined]

    assert all(0.1 <= interval <= 0.25 for interval in intervals)
    assert max(intervals) > min(intervals)
    assert speeds
    assert all(0.2 <= speed <= 0.35 for speed in speeds)
    assert max(speeds) > min(speeds)


def test_default_diamond_ranges_allow_faster_later_hits() -> None:
    cfg = ColoursLettersNumbersConfig()

    distance_to_zone = 0.54 - 0.02
    slow_arrival_s = distance_to_zone / cfg.diamond_speed_norm_per_s
    fast_later_arrival_s = (
        cfg.diamond_spawn_interval_max_s
        + (distance_to_zone / cfg.diamond_speed_max_norm_per_s)
    )

    assert fast_later_arrival_s < slow_arrival_s
