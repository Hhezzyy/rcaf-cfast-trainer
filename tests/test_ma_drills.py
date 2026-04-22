from __future__ import annotations

from dataclasses import dataclass

import pytest

from cfast_trainer.ant_drills import ANT_DRILL_MODE_PROFILES, AntDrillMode
from cfast_trainer.ma_drills import (
    MaDrillConfig,
    MaProblemPayload,
    build_ma_fuel_endurance_drill,
    build_ma_mixed_conversion_caps_drill,
    build_ma_one_step_fluency_drill,
    build_ma_percentage_snap_drill,
    build_ma_rate_time_distance_drill,
    build_ma_written_numerical_extraction_drill,
)


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _signature(engine) -> tuple[object, ...]:
    current = engine._current
    assert current is not None
    payload = current.payload
    assert isinstance(payload, MaProblemPayload)
    return (
        current.prompt,
        current.answer,
        payload.family,
        payload.variant,
        None if engine._current_cap_s is None else round(float(engine._current_cap_s), 4),
    )


def _capture_signatures(builder, *, seed: int, count: int = 6) -> tuple[tuple[object, ...], ...]:
    clock = FakeClock()
    engine = builder(
        clock=clock,
        seed=seed,
        difficulty=0.5,
        config=MaDrillConfig(practice_questions=0, scored_duration_s=18.0),
    )
    engine.start_scored()
    signatures: list[tuple[object, ...]] = []
    for _ in range(count):
        signatures.append(_signature(engine))
        assert engine.submit_answer(str(engine._current.answer)) is True
    return tuple(signatures)


@pytest.mark.parametrize(
    "builder",
    (
        build_ma_one_step_fluency_drill,
        build_ma_percentage_snap_drill,
        build_ma_rate_time_distance_drill,
        build_ma_fuel_endurance_drill,
        build_ma_written_numerical_extraction_drill,
        build_ma_mixed_conversion_caps_drill,
    ),
)
def test_ma_drills_are_deterministic_for_same_seed(builder) -> None:
    assert _capture_signatures(builder, seed=1717) == _capture_signatures(builder, seed=1717)


@pytest.mark.parametrize(
    ("builder", "expected_family"),
    (
        (build_ma_one_step_fluency_drill, "one_step_fluency"),
        (build_ma_percentage_snap_drill, "percentage_snap"),
        (build_ma_rate_time_distance_drill, "rate_time_distance"),
        (build_ma_fuel_endurance_drill, "fuel_endurance"),
        (build_ma_written_numerical_extraction_drill, "written_numerical_extraction"),
        (build_ma_mixed_conversion_caps_drill, "mixed_conversion_caps"),
    ),
)
def test_ma_drills_stay_on_the_intended_problem_family(builder, expected_family: str) -> None:
    clock = FakeClock()
    engine = builder(
        clock=clock,
        seed=81,
        difficulty=0.5,
        config=MaDrillConfig(practice_questions=0, scored_duration_s=18.0),
    )
    engine.start_scored()
    for _ in range(6):
        current = engine._current
        assert current is not None
        payload = current.payload
        assert isinstance(payload, MaProblemPayload)
        assert payload.family == expected_family
        assert engine.submit_answer(str(current.answer)) is True


def test_ma_one_step_fluency_keeps_single_operator_prompts() -> None:
    clock = FakeClock()
    engine = build_ma_one_step_fluency_drill(
        clock=clock,
        seed=14,
        difficulty=0.5,
        config=MaDrillConfig(practice_questions=0, scored_duration_s=18.0),
    )
    engine.start_scored()

    seen_variants: set[str] = set()
    for _ in range(8):
        current = engine._current
        assert current is not None
        payload = current.payload
        assert isinstance(payload, MaProblemPayload)
        seen_variants.add(payload.variant)
        assert engine.submit_answer(str(current.answer)) is True

    assert seen_variants <= {"addition", "subtraction", "multiplication", "division"}


def test_ma_mixed_conversion_caps_use_payload_caps_and_keep_typed_hint() -> None:
    clock = FakeClock()
    engine = build_ma_mixed_conversion_caps_drill(
        clock=clock,
        seed=29,
        difficulty=0.5,
        mode=AntDrillMode.TEMPO,
        config=MaDrillConfig(practice_questions=0, scored_duration_s=18.0),
    )
    engine.start_scored()

    expected_scale = ANT_DRILL_MODE_PROFILES[AntDrillMode.TEMPO].cap_scale
    caps: list[float] = []
    for _ in range(4):
        current = engine._current
        assert current is not None
        payload = current.payload
        assert isinstance(payload, MaProblemPayload)
        assert payload.base_cap_s is not None
        input_hint = engine.snapshot().input_hint
        assert "Type answer then Enter" in input_hint
        assert "Cap " not in input_hint
        assert engine._current_cap_s == pytest.approx(payload.base_cap_s * expected_scale)
        caps.append(float(engine._current_cap_s))
        assert engine.submit_answer(str(current.answer)) is True

    assert all(cap > 0.0 for cap in caps)


def _difficulty_for_level(level: int) -> float:
    return float(level - 1) / 9.0


def test_written_numerical_extraction_l2_l5_l8_scale_materially() -> None:
    signatures: list[tuple[int, int, int, int, float | None]] = []
    for level in (2, 5, 8):
        clock = FakeClock()
        engine = build_ma_written_numerical_extraction_drill(
            clock=clock,
            seed=313,
            difficulty=_difficulty_for_level(level),
            config=MaDrillConfig(practice_questions=0, scored_duration_s=18.0),
        )
        engine.start_scored()
        payload = engine._current.payload
        assert isinstance(payload, MaProblemPayload)
        signatures.append(
            (
                payload.operand_size,
                payload.step_count,
                payload.unit_burden,
                payload.answer_closeness,
                None if engine._current_cap_s is None else round(float(engine._current_cap_s), 3),
            )
        )

    low, mid, high = signatures
    assert low[1] <= mid[1] <= high[1]
    assert low[2] <= mid[2] <= high[2]
    assert low[4] is not None and mid[4] is not None and high[4] is not None
    assert low[4] > mid[4] > high[4]
    assert low != mid != high
