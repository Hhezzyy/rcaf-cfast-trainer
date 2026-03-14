from __future__ import annotations

from dataclasses import dataclass

import pytest

from cfast_trainer.ant_drills import AntDrillMode
from cfast_trainer.cognitive_updating import CognitiveUpdatingPayload
from cfast_trainer.cu_drills import (
    CuDrillConfig,
    build_cu_controls_anchor_drill,
    build_cu_engine_balance_run_drill,
    build_cu_mixed_tempo_drill,
    build_cu_navigation_anchor_drill,
    build_cu_objective_prime_drill,
    build_cu_pressure_run_drill,
    build_cu_sensors_timing_prime_drill,
    build_cu_state_code_run_drill,
)


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _problem_signature(engine) -> tuple[object, ...]:
    current = engine._current
    assert current is not None
    payload = current.payload
    assert isinstance(payload, CognitiveUpdatingPayload)
    return (
        current.prompt,
        current.answer,
        payload.active_domains,
        payload.scenario_family,
        payload.focus_label,
        payload.starting_upper_tab_index,
        payload.starting_lower_tab_index,
        payload.required_knots,
        payload.pressure_low,
        payload.pressure_high,
    )


def _capture_signatures(builder, *, seed: int, count: int = 6) -> tuple[tuple[object, ...], ...]:
    clock = FakeClock()
    engine = builder(
        clock=clock,
        seed=seed,
        difficulty=0.5,
        mode=AntDrillMode.BUILD,
        config=CuDrillConfig(scored_duration_s=12.0),
    )
    engine.start_scored()
    captured: list[tuple[object, ...]] = []
    for _ in range(count):
        captured.append(_problem_signature(engine))
        assert engine.submit_answer(str(engine._current.answer)) is True
    return tuple(captured)


@pytest.mark.parametrize(
    "builder",
    (
        build_cu_controls_anchor_drill,
        build_cu_navigation_anchor_drill,
        build_cu_engine_balance_run_drill,
        build_cu_sensors_timing_prime_drill,
        build_cu_objective_prime_drill,
        build_cu_state_code_run_drill,
        build_cu_mixed_tempo_drill,
        build_cu_pressure_run_drill,
    ),
)
def test_cu_drills_are_deterministic_for_same_seed(builder) -> None:
    assert _capture_signatures(builder, seed=1717) == _capture_signatures(builder, seed=1717)


@pytest.mark.parametrize(
    ("builder", "expected_domains"),
    (
        (build_cu_controls_anchor_drill, ("controls", "state_code")),
        (build_cu_navigation_anchor_drill, ("navigation", "state_code")),
        (build_cu_engine_balance_run_drill, ("engine", "state_code")),
        (build_cu_sensors_timing_prime_drill, ("sensors", "state_code")),
        (build_cu_objective_prime_drill, ("objectives", "state_code")),
        (build_cu_state_code_run_drill, ("controls", "navigation", "sensors", "state_code")),
    ),
)
def test_cu_focused_drills_emit_expected_active_domains(builder, expected_domains) -> None:
    clock = FakeClock()
    engine = builder(
        clock=clock,
        seed=91,
        difficulty=0.5,
        mode=AntDrillMode.BUILD,
        config=CuDrillConfig(scored_duration_s=6.0),
    )
    engine.start_scored()
    payload = engine._current.payload
    assert isinstance(payload, CognitiveUpdatingPayload)
    assert payload.active_domains == expected_domains


def test_cu_mixed_tempo_repeats_fixed_item_cycle() -> None:
    clock = FakeClock()
    engine = build_cu_mixed_tempo_drill(
        clock=clock,
        seed=505,
        difficulty=0.5,
        mode=AntDrillMode.TEMPO,
        config=CuDrillConfig(scored_duration_s=20.0),
    )
    engine.start_scored()

    observed: list[str] = []
    for _ in range(8):
        payload = engine._current.payload
        assert isinstance(payload, CognitiveUpdatingPayload)
        observed.append(payload.focus_label)
        assert engine.submit_answer(str(engine._current.answer)) is True

    assert observed == [
        "Controls",
        "Navigation",
        "Engine",
        "Sensors",
        "Objectives",
        "State Code",
        "Full Mixed",
        "Controls",
    ]


def test_cu_pressure_run_keeps_all_domains_active() -> None:
    clock = FakeClock()
    engine = build_cu_pressure_run_drill(
        clock=clock,
        seed=606,
        difficulty=0.5,
        mode=AntDrillMode.STRESS,
        config=CuDrillConfig(scored_duration_s=8.0),
    )
    engine.start_scored()

    for _ in range(4):
        payload = engine._current.payload
        assert isinstance(payload, CognitiveUpdatingPayload)
        assert payload.active_domains == (
            "controls",
            "navigation",
            "engine",
            "sensors",
            "objectives",
            "state_code",
        )
        assert engine.submit_answer(str(engine._current.answer)) is True
