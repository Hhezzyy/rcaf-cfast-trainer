from __future__ import annotations

from dataclasses import dataclass

from cfast_trainer.math_reasoning import MathReasoningPayload, MathReasoningTrainingPayload
from cfast_trainer.mr_drills import (
    MrDomainRunGenerator,
    build_mr_mixed_pressure_set_drill,
    build_mr_relevant_info_scan_drill,
)


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t


def test_mr_relevant_info_scan_uses_typed_payload_without_options() -> None:
    clock = FakeClock()
    engine = build_mr_relevant_info_scan_drill(clock=clock, seed=313, difficulty=0.5)

    engine.start_practice()
    payload = engine._current.payload

    assert isinstance(payload, MathReasoningTrainingPayload)
    assert "Find the stated" in engine._current.prompt
    assert payload.input_digits >= 4


def test_mr_domain_run_starts_with_motion_and_fuel_domains() -> None:
    generator = MrDomainRunGenerator(seed=404)
    domains = [
        generator.next_problem(difficulty=0.55).payload.domain_key
        for _ in range(8)
    ]

    assert domains[:2] == ["distance_from_speed_time", "distance_from_speed_time"]
    assert domains[2:4] == ["time_from_distance_speed", "time_from_distance_speed"]
    assert domains[4:6] == ["speed_from_distance_time", "speed_from_distance_time"]
    assert domains[6:8] == ["fuel_remaining", "fuel_remaining"]


def test_mr_mixed_pressure_uses_real_multiple_choice_payloads() -> None:
    clock = FakeClock()
    engine = build_mr_mixed_pressure_set_drill(clock=clock, seed=515, difficulty=0.7)

    engine.start_practice()
    payload = engine._current.payload

    assert isinstance(payload, MathReasoningPayload)
    assert len(payload.options) == 5
