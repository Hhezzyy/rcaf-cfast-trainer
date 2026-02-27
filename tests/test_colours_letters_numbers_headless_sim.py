from __future__ import annotations

from dataclasses import dataclass

from cfast_trainer.cognitive_core import Phase
from cfast_trainer.colours_letters_numbers import (
    ColoursLettersNumbersConfig,
    build_colours_letters_numbers_test,
)


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


def _solve_math(prompt: str) -> int:
    text = str(prompt).replace("=", "").replace("x", "*")
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


def test_headless_sim_multitask_sequence_math_and_color_hits() -> None:
    seed = 99
    difficulty = 0.5
    clock = FakeClock()

    cfg = ColoursLettersNumbersConfig(
        scored_duration_s=3.0,
        practice_rounds=1,
        round_duration_s=0.6,
        sequence_show_s=0.25,
        diamond_spawn_interval_s=0.2,
        diamond_speed_norm_per_s=0.85,
        max_live_diamonds=5,
    )

    engine = build_colours_letters_numbers_test(
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        config=cfg,
    )
    assert engine.phase is Phase.INSTRUCTIONS

    engine.start_practice()
    assert engine.phase is Phase.PRACTICE

    practice_payload = engine.snapshot().payload
    assert practice_payload is not None
    assert practice_payload.target_sequence is not None
    practice_target = practice_payload.target_sequence

    clock.advance(0.3)
    engine.update()
    practice_payload = engine.snapshot().payload
    assert practice_payload is not None
    practice_code = next(o.code for o in practice_payload.options if o.label == practice_target)
    assert engine.submit_answer(f"MEM:{practice_code}") is True
    assert engine.submit_answer(str(_solve_math(practice_payload.math_prompt))) is True

    # Practice completion follows the memory-cycle timer, not math submission.
    clock.advance(0.35)
    engine.update()
    assert engine.phase is Phase.PRACTICE_DONE

    engine.start_scored()
    assert engine.phase is Phase.SCORED

    scored_payload = engine.snapshot().payload
    assert scored_payload is not None
    assert scored_payload.target_sequence is not None
    scored_target = scored_payload.target_sequence

    clock.advance(0.3)
    engine.update()
    scored_payload = engine.snapshot().payload
    assert scored_payload is not None
    scored_code = next(o.code for o in scored_payload.options if o.label == scored_target)
    assert engine.submit_answer(f"MEM:{scored_code}") is True
    assert engine.submit_answer(str(_solve_math(scored_payload.math_prompt))) is True

    hit = False
    key_for_color = {
        "RED": "R",
        "YELLOW": "E",
        "GREEN": "W",
        "BLUE": "Q",
    }
    for _ in range(200):
        clock.advance(0.05)
        engine.update()
        snap = engine.snapshot()
        payload = snap.payload
        if payload is not None:
            lane_count = max(1, len(payload.lane_colors))
            lane_w = (payload.lane_end_norm - payload.lane_start_norm) / float(lane_count)
            for d in payload.diamonds:
                if d.color not in payload.lane_colors:
                    continue
                idx = payload.lane_colors.index(d.color)
                lane_start = payload.lane_start_norm + (idx * lane_w)
                lane_end = lane_start + lane_w
                if lane_start <= d.x_norm <= lane_end:
                    key = key_for_color[d.color]
                    if engine.submit_answer(f"CLR:{key}"):
                        hit = True
                        break
        if engine.phase is Phase.RESULTS:
            break

    assert hit is True
    assert engine.phase is Phase.RESULTS

    summary = engine.scored_summary()
    assert summary.attempted >= 2
    assert summary.correct >= 2
    assert summary.total_score > 0.0
