from __future__ import annotations

from dataclasses import dataclass

from cfast_trainer.sl_drills import (
    build_sl_family_run_drill,
    build_sl_fast_reject_drill,
    build_sl_fault_diagnosis_prime_drill,
    build_sl_flow_trace_anchor_drill,
    build_sl_graph_rule_anchor_drill,
    build_sl_index_switch_run_drill,
    build_sl_missing_step_complete_drill,
    build_sl_mixed_tempo_drill,
    build_sl_one_rule_identify_drill,
    build_sl_pressure_run_drill,
    build_sl_quantitative_anchor_drill,
    build_sl_rule_match_drill,
    build_sl_two_source_reconcile_drill,
    canonical_reasoning_family_for_payload,
)
from cfast_trainer.system_logic import SystemLogicPayload


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _normalize_family(name: str) -> str:
    token = str(name).strip().lower()
    if token.startswith("thermal"):
        return "thermal"
    return token


def _problem_signature(engine) -> tuple[object, ...]:
    current = engine._current
    assert current is not None
    payload = current.payload
    assert isinstance(payload, SystemLogicPayload)
    return (
        current.prompt,
        current.answer,
        payload.system_family,
        payload.reasoning_mode,
        tuple((entry.code, entry.label, entry.top_document.kind, entry.bottom_document.kind) for entry in payload.index_entries),
        tuple((choice.code, choice.text) for choice in payload.answer_choices),
        payload.correct_choice_code,
    )


def _capture_signatures(builder, *, seed: int, count: int = 6) -> tuple[tuple[object, ...], ...]:
    clock = FakeClock()
    engine = builder(clock=clock, seed=seed, difficulty=0.5)
    engine.start_scored()
    captured: list[tuple[object, ...]] = []
    for _ in range(count):
        captured.append(_problem_signature(engine))
        answer = engine._current.answer
        assert engine.submit_answer(str(answer)) is True
    return tuple(captured)


def test_sl_drill_families_are_deterministic_for_same_seed() -> None:
    builders = (
        build_sl_quantitative_anchor_drill,
        build_sl_flow_trace_anchor_drill,
        build_sl_graph_rule_anchor_drill,
        build_sl_fault_diagnosis_prime_drill,
        build_sl_index_switch_run_drill,
        build_sl_family_run_drill,
        build_sl_mixed_tempo_drill,
        build_sl_pressure_run_drill,
        build_sl_one_rule_identify_drill,
        build_sl_missing_step_complete_drill,
        build_sl_two_source_reconcile_drill,
        build_sl_rule_match_drill,
        build_sl_fast_reject_drill,
    )

    for builder in builders:
        assert _capture_signatures(builder, seed=1717) == _capture_signatures(builder, seed=1717)


def test_sl_focused_drills_emit_only_the_intended_reasoning_family() -> None:
    focused = (
        (build_sl_quantitative_anchor_drill, "quantitative"),
        (build_sl_flow_trace_anchor_drill, "trace"),
        (build_sl_graph_rule_anchor_drill, "graph_rule"),
        (build_sl_fault_diagnosis_prime_drill, "diagnosis"),
        (build_sl_one_rule_identify_drill, "graph_rule"),
        (build_sl_missing_step_complete_drill, "trace"),
    )

    for builder, expected_reasoning in focused:
        clock = FakeClock()
        engine = builder(clock=clock, seed=81, difficulty=0.5)
        engine.start_scored()
        for _ in range(6):
            payload = engine._current.payload
            assert isinstance(payload, SystemLogicPayload)
            assert canonical_reasoning_family_for_payload(payload) == expected_reasoning
            assert engine.submit_answer(str(engine._current.answer)) is True


def test_sl_family_run_repeats_fixed_system_family_cycle() -> None:
    clock = FakeClock()
    engine = build_sl_family_run_drill(clock=clock, seed=404, difficulty=0.5)
    engine.start_scored()

    seen: list[str] = []
    for _ in range(7):
        payload = engine._current.payload
        assert isinstance(payload, SystemLogicPayload)
        seen.append(_normalize_family(payload.system_family))
        assert engine.submit_answer(str(engine._current.answer)) is True

    assert seen[:5] == ["oil", "fuel", "electrical", "hydraulic", "thermal"]
    assert seen[5:] == ["oil", "fuel"]


def test_sl_mixed_tempo_repeats_fixed_reasoning_cycle() -> None:
    clock = FakeClock()
    engine = build_sl_mixed_tempo_drill(clock=clock, seed=505, difficulty=0.5)
    engine.start_scored()

    reasonings: list[str] = []
    for _ in range(8):
        payload = engine._current.payload
        assert isinstance(payload, SystemLogicPayload)
        reasonings.append(canonical_reasoning_family_for_payload(payload))
        assert engine.submit_answer(str(engine._current.answer)) is True

    assert reasonings == [
        "quantitative",
        "trace",
        "graph_rule",
        "diagnosis",
        "quantitative",
        "trace",
        "graph_rule",
        "diagnosis",
    ]


def test_sl_pressure_run_covers_all_reasoning_modes_and_all_families() -> None:
    clock = FakeClock()
    engine = build_sl_pressure_run_drill(clock=clock, seed=606, difficulty=0.5)
    engine.start_scored()

    families: set[str] = set()
    reasonings: set[str] = set()
    for _ in range(10):
        payload = engine._current.payload
        assert isinstance(payload, SystemLogicPayload)
        families.add(_normalize_family(payload.system_family))
        reasonings.add(canonical_reasoning_family_for_payload(payload))
        assert engine.submit_answer(str(engine._current.answer)) is True

    assert families == {"oil", "fuel", "electrical", "hydraulic", "thermal"}
    assert reasonings == {"quantitative", "trace", "graph_rule", "diagnosis"}


def test_sl_two_source_reconcile_repeats_its_fixed_reasoning_cycle() -> None:
    clock = FakeClock()
    engine = build_sl_two_source_reconcile_drill(clock=clock, seed=707, difficulty=0.5)
    engine.start_scored()

    reasonings: list[str] = []
    for _ in range(8):
        payload = engine._current.payload
        assert isinstance(payload, SystemLogicPayload)
        reasonings.append(canonical_reasoning_family_for_payload(payload))
        assert engine.submit_answer(str(engine._current.answer)) is True

    assert reasonings == [
        "quantitative",
        "diagnosis",
        "graph_rule",
        "diagnosis",
        "quantitative",
        "diagnosis",
        "graph_rule",
        "diagnosis",
    ]


def test_sl_rule_match_and_fast_reject_keep_their_focused_reasoning_profiles() -> None:
    clock = FakeClock()
    rule_match = build_sl_rule_match_drill(clock=clock, seed=808, difficulty=0.5)
    rule_match.start_scored()

    rule_reasonings: list[str] = []
    for _ in range(4):
        payload = rule_match._current.payload
        assert isinstance(payload, SystemLogicPayload)
        rule_reasonings.append(canonical_reasoning_family_for_payload(payload))
        assert rule_match.submit_answer(str(rule_match._current.answer)) is True

    assert rule_reasonings == ["graph_rule", "quantitative", "graph_rule", "trace"]

    reject_clock = FakeClock()
    fast_reject = build_sl_fast_reject_drill(clock=reject_clock, seed=808, difficulty=0.5)
    fast_reject.start_scored()

    reject_reasonings: list[str] = []
    for _ in range(4):
        payload = fast_reject._current.payload
        assert isinstance(payload, SystemLogicPayload)
        reject_reasonings.append(canonical_reasoning_family_for_payload(payload))
        assert len(payload.answer_choices) == 5
        assert fast_reject.submit_answer(str(fast_reject._current.answer)) is True

    assert reject_reasonings == ["diagnosis", "graph_rule", "diagnosis", "quantitative"]
