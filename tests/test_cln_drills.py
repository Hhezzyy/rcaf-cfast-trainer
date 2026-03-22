from __future__ import annotations

from typing import cast

from cfast_trainer.cln_drills import (
    _LiveDiamond,
    build_cln_colour_lane_drill,
    build_cln_full_pressure_drill,
    build_cln_memory_colour_drill,
    build_cln_memory_math_drill,
    build_cln_overdrive_blue_return_drill,
    build_cln_overdrive_dual_math_drill,
    build_cln_overdrive_six_choice_memory_drill,
    build_cln_sequence_copy_drill,
    build_cln_sequence_math_recall_drill,
    build_cln_sequence_match_drill,
)
from cfast_trainer.colours_letters_numbers import ColoursLettersNumbersTrainingPayload


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


def _solve_math(prompt: str) -> int:
    text = str(prompt).replace("=", "").replace("x", "*").replace("×", "*")
    parts = text.strip().split()
    assert len(parts) == 3
    a = int(parts[0])
    op = parts[1]
    b = int(parts[2])
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    raise AssertionError(f"Unsupported operator: {op}")


def test_sequence_copy_keeps_sequence_visible_and_accepts_typed_memory_input() -> None:
    clock = FakeClock()
    engine = build_cln_sequence_copy_drill(clock=clock, seed=11, difficulty=0.4)

    engine.start_practice()
    payload = cast(ColoursLettersNumbersTrainingPayload, engine.snapshot().payload)

    assert payload.target_sequence is not None
    assert payload.memory_input_active is True
    assert payload.math_active is False
    assert payload.colour_active is False
    assert payload.show_text_entry is True


def test_sequence_match_activates_choice_grid_after_delay() -> None:
    clock = FakeClock()
    engine = build_cln_sequence_match_drill(clock=clock, seed=23, difficulty=0.6)

    engine.start_practice()
    initial = cast(ColoursLettersNumbersTrainingPayload, engine.snapshot().payload)
    assert initial.target_sequence is not None
    assert initial.options_active is False

    clock.advance(engine._sequence_show_s() + engine._memory_recall_delay_s_current + 0.05)
    engine.update()

    payload = cast(ColoursLettersNumbersTrainingPayload, engine.snapshot().payload)
    assert payload.target_sequence is None
    assert payload.options_active is True
    assert payload.memory_input_active is False


def test_sequence_math_recall_requires_one_math_answer_before_memory_opens() -> None:
    clock = FakeClock()
    engine = build_cln_sequence_math_recall_drill(clock=clock, seed=29, difficulty=0.3)

    engine.start_practice()
    initial = cast(ColoursLettersNumbersTrainingPayload, engine.snapshot().payload)
    assert initial.target_sequence is not None

    clock.advance(engine._sequence_show_s() + engine._memory_recall_delay_s_current + 0.05)
    engine.update()
    gated = cast(ColoursLettersNumbersTrainingPayload, engine.snapshot().payload)
    assert gated.memory_input_active is False
    assert gated.options_active is False
    assert gated.math_active is True

    assert engine.submit_answer(str(_solve_math(gated.math_prompt))) is True
    opened = cast(ColoursLettersNumbersTrainingPayload, engine.snapshot().payload)
    assert opened.memory_input_active is True
    assert opened.options_active is False


def test_sequence_math_recall_switches_to_choice_mode_at_higher_difficulty() -> None:
    clock = FakeClock()
    engine = build_cln_sequence_math_recall_drill(clock=clock, seed=37, difficulty=0.9)

    engine.start_practice()
    clock.advance(engine._sequence_show_s() + engine._memory_recall_delay_s_current + 0.05)
    engine.update()
    gated = cast(ColoursLettersNumbersTrainingPayload, engine.snapshot().payload)

    assert engine.submit_answer(str(_solve_math(gated.math_prompt))) is True
    opened = cast(ColoursLettersNumbersTrainingPayload, engine.snapshot().payload)
    assert opened.memory_input_active is False
    assert opened.options_active is True


def test_colour_lane_drill_is_colour_only() -> None:
    clock = FakeClock()
    engine = build_cln_colour_lane_drill(clock=clock, seed=31, difficulty=0.5)

    engine.start_practice()
    payload = cast(ColoursLettersNumbersTrainingPayload, engine.snapshot().payload)

    assert payload.memory_active is False
    assert payload.math_active is False
    assert payload.colour_active is True
    assert payload.show_text_entry is False
    assert payload.lane_colors == ("RED", "YELLOW", "GREEN")


def test_memory_math_and_memory_colour_limit_active_channels() -> None:
    clock = FakeClock()

    memory_math = build_cln_memory_math_drill(clock=clock, seed=41, difficulty=0.5)
    memory_math.start_practice()
    mm_payload = cast(ColoursLettersNumbersTrainingPayload, memory_math.snapshot().payload)
    assert mm_payload.memory_active is True
    assert mm_payload.math_active is True
    assert mm_payload.colour_active is False

    memory_colour = build_cln_memory_colour_drill(clock=clock, seed=42, difficulty=0.5)
    memory_colour.start_practice()
    mc_payload = cast(ColoursLettersNumbersTrainingPayload, memory_colour.snapshot().payload)
    assert mc_payload.memory_active is True
    assert mc_payload.math_active is False
    assert mc_payload.colour_active is True


def test_full_pressure_is_deterministic_for_same_seed_and_difficulty() -> None:
    clock_a = FakeClock()
    clock_b = FakeClock()
    engine_a = build_cln_full_pressure_drill(clock=clock_a, seed=77, difficulty=0.8)
    engine_b = build_cln_full_pressure_drill(clock=clock_b, seed=77, difficulty=0.8)

    engine_a.start_practice()
    engine_b.start_practice()
    clock_a.advance(1.2)
    clock_b.advance(1.2)
    engine_a.update()
    engine_b.update()

    payload_a = cast(ColoursLettersNumbersTrainingPayload, engine_a.snapshot().payload)
    payload_b = cast(ColoursLettersNumbersTrainingPayload, engine_b.snapshot().payload)

    assert payload_a.target_sequence == payload_b.target_sequence
    assert payload_a.math_prompt == payload_b.math_prompt
    assert payload_a.diamonds == payload_b.diamonds


def test_overdrive_blue_return_restores_blue_lane_mapping() -> None:
    clock = FakeClock()
    engine = build_cln_overdrive_blue_return_drill(clock=clock, seed=101, difficulty=0.6)

    engine.start_practice()
    payload = cast(ColoursLettersNumbersTrainingPayload, engine.snapshot().payload)

    assert payload.lane_colors == ("RED", "YELLOW", "GREEN", "BLUE")
    engine._diamonds = [
        _LiveDiamond(id=77, color="BLUE", row=0, x_norm=0.92, speed_norm_per_s=0.2)
    ]  # type: ignore[attr-defined]
    assert engine.submit_answer("CLR:R") is True


def test_overdrive_six_choice_memory_generates_six_options_deterministically() -> None:
    clock_a = FakeClock()
    clock_b = FakeClock()
    engine_a = build_cln_overdrive_six_choice_memory_drill(clock=clock_a, seed=131, difficulty=0.7)
    engine_b = build_cln_overdrive_six_choice_memory_drill(clock=clock_b, seed=131, difficulty=0.7)

    engine_a.start_practice()
    engine_b.start_practice()
    payload_a = cast(ColoursLettersNumbersTrainingPayload, engine_a.snapshot().payload)
    payload_b = cast(ColoursLettersNumbersTrainingPayload, engine_b.snapshot().payload)

    assert len(payload_a.options) == 0
    clock_a.advance(engine_a._sequence_show_s() + engine_a._memory_recall_delay_s_current + 0.05)
    clock_b.advance(engine_b._sequence_show_s() + engine_b._memory_recall_delay_s_current + 0.05)
    engine_a.update()
    engine_b.update()
    opened_a = cast(ColoursLettersNumbersTrainingPayload, engine_a.snapshot().payload)
    opened_b = cast(ColoursLettersNumbersTrainingPayload, engine_b.snapshot().payload)
    assert opened_a.options_active is True
    assert len(opened_a.options) == 6
    assert opened_a.options == opened_b.options
    assert opened_a.memory_choice_keys == ("A", "S", "D", "F", "G", "H")


def test_overdrive_dual_math_emits_bonus_multiple_choice_panel() -> None:
    clock = FakeClock()
    engine = build_cln_overdrive_dual_math_drill(clock=clock, seed=151, difficulty=0.8)

    engine.start_practice()
    payload = cast(ColoursLettersNumbersTrainingPayload, engine.snapshot().payload)

    assert payload.math_active is True
    assert payload.secondary_math_choice_active is True
    assert len(payload.secondary_math_options) == 5
    assert payload.secondary_math_prompt.strip() != ""
