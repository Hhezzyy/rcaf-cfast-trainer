from __future__ import annotations

from dataclasses import dataclass

from cfast_trainer.airborne_numerical import AirborneNumericalGenerator, TEMPLATES, UNIT_PROFILES
from cfast_trainer.cognitive_core import Phase, SeededRng
from cfast_trainer.cognitive_updating import supported_cognitive_updating_scenario_families
from cfast_trainer.math_reasoning import (
    MR_ALL_DOMAIN_KEYS,
    MR_FUEL_ENDURANCE,
    MR_RESERVE_MARGIN,
    MathReasoningGenerator,
)
from cfast_trainer.system_logic import SystemLogicGenerator
from cfast_trainer.table_reading import TableReadingGenerator
from cfast_trainer.telemetry import telemetry_events_from_engine
from cfast_trainer.trace_test_2 import TraceTest2Generator, TraceTest2QuestionKind


@dataclass
class FakeClock:
    now_s: float = 0.0

    def now(self) -> float:
        return float(self.now_s)

    def advance(self, seconds: float) -> None:
        self.now_s += float(seconds)


def test_airborne_numerical_expansion_exposes_new_metadata() -> None:
    generator = AirborneNumericalGenerator(SeededRng(123), scripted_diverse_problems=8)
    scenarios = [generator.generate().payload for _ in range(8)]
    assert len(TEMPLATES) >= 10
    assert len(UNIT_PROFILES) >= 6
    assert {scenario.question_kind for scenario in scenarios} >= {
        "fuel_endurance",
        "parcel_effect",
    }
    assert all(scenario.content_family for scenario in scenarios)
    assert all(scenario.variant_id for scenario in scenarios)
    assert all(scenario.content_pack in {"table_table", "chart_table", "table_chart", "chart_chart"} for scenario in scenarios)


def test_table_reading_expansion_adds_new_families() -> None:
    generator = TableReadingGenerator(seed=44)
    assert {"dispatch", "range"} <= set(generator.supported_part_one_families())
    assert {"descent", "timing"} <= set(generator.supported_part_two_families())
    problem = generator.next_problem_for_selection(difficulty=0.6)
    payload = problem.payload
    assert payload is not None
    assert payload.content_family
    assert payload.variant_id
    assert payload.content_pack in {"part_one", "part_two"}


def test_system_logic_supports_pressurization_family() -> None:
    generator = SystemLogicGenerator(seed=9)
    problem = generator.next_problem_for_selection(difficulty=0.5, family="pressurization")
    payload = problem.payload
    assert payload is not None
    assert payload.system_family == "pressurization"
    assert payload.content_family == "pressurization"
    assert payload.variant_id


def test_math_reasoning_has_new_domains() -> None:
    assert MR_FUEL_ENDURANCE in MR_ALL_DOMAIN_KEYS
    assert MR_RESERVE_MARGIN in MR_ALL_DOMAIN_KEYS
    generator = MathReasoningGenerator(seed=7)
    endurance = generator.next_scenario_spec(difficulty=0.5, domain_key=MR_FUEL_ENDURANCE)
    reserve = generator.next_scenario_spec(difficulty=0.5, domain_key=MR_RESERVE_MARGIN)
    assert endurance.domain_key == MR_FUEL_ENDURANCE
    assert reserve.domain_key == MR_RESERVE_MARGIN


def test_cognitive_updating_supports_new_scenario_families() -> None:
    supported = supported_cognitive_updating_scenario_families()
    assert "crosscheck" in supported
    assert "recovery_window" in supported


def test_trace_test_2_has_new_question_kinds() -> None:
    problem = TraceTest2Generator(
        seed=3,
        allowed_question_kinds=(
            TraceTest2QuestionKind.ENDED_RIGHTMOST,
            TraceTest2QuestionKind.ENDED_LOWEST,
        ),
    ).next_problem(difficulty=0.6)
    payload = problem.payload
    assert payload is not None
    assert payload.question_kind in {
        TraceTest2QuestionKind.ENDED_RIGHTMOST,
        TraceTest2QuestionKind.ENDED_LOWEST,
    }
    assert payload.content_family == "motion_memory"
    assert payload.variant_id


def test_timed_test_question_metadata_flows_into_telemetry() -> None:
    from cfast_trainer.table_reading import build_table_reading_test

    clock = FakeClock()
    engine = build_table_reading_test(clock=clock, seed=91, difficulty=0.5)
    engine.start_practice()
    payload = engine.snapshot().payload
    assert payload is not None
    clock.advance(0.4)
    assert engine.submit_answer(str(payload.correct_code))
    assert engine.phase in {Phase.PRACTICE, Phase.PRACTICE_DONE}
    events = telemetry_events_from_engine(engine)
    assert events
    scored_like = events[0]
    assert scored_like.extra is not None
    assert scored_like.extra["content_family"] == payload.content_family
    assert scored_like.extra["variant_id"] == payload.variant_id
