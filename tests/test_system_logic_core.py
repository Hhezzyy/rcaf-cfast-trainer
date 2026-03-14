from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from cfast_trainer.system_logic import (
    SystemLogicConfig,
    SystemLogicGenerator,
    SystemLogicPayload,
    SystemLogicScorer,
    build_system_logic_test,
)


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _signature(payload: SystemLogicPayload) -> tuple[object, ...]:
    return (
        payload.system_family,
        payload.reasoning_mode,
        tuple(
            (
                entry.code,
                entry.label,
                entry.top_document.kind,
                entry.bottom_document.kind,
            )
            for entry in payload.index_entries
        ),
        tuple(choice.text for choice in payload.answer_choices),
        payload.correct_choice_code,
    )


def test_generator_determinism_same_seed_same_sequence() -> None:
    seed = 913
    gen_a = SystemLogicGenerator(seed=seed)
    gen_b = SystemLogicGenerator(seed=seed)

    seq_a = [gen_a.next_problem(difficulty=0.6) for _ in range(12)]
    seq_b = [gen_b.next_problem(difficulty=0.6) for _ in range(12)]

    view_a = [
        (problem.prompt, problem.answer, _signature(cast(SystemLogicPayload, problem.payload)))
        for problem in seq_a
    ]
    view_b = [
        (problem.prompt, problem.answer, _signature(cast(SystemLogicPayload, problem.payload)))
        for problem in seq_b
    ]
    assert view_a == view_b


def test_different_seed_changes_configuration() -> None:
    payload_a = cast(SystemLogicPayload, SystemLogicGenerator(seed=77).next_problem(difficulty=0.5).payload)
    payload_b = cast(SystemLogicPayload, SystemLogicGenerator(seed=78).next_problem(difficulty=0.5).payload)

    assert _signature(payload_a) != _signature(payload_b)


def test_generated_payload_has_guide_shape_and_multi_source_reasoning() -> None:
    gen = SystemLogicGenerator(seed=77)
    payload = cast(SystemLogicPayload, gen.next_problem(difficulty=0.5).payload)

    assert len(payload.index_entries) == 4
    assert tuple(entry.code for entry in payload.index_entries) == (0, 1, 2, 3)
    assert len(payload.answer_choices) == 5
    assert tuple(choice.code for choice in payload.answer_choices) == (1, 2, 3, 4, 5)
    assert len(set(payload.required_index_codes)) >= 2
    assert len(set(payload.required_document_kinds)) >= 2
    assert all(entry.top_document.kind for entry in payload.index_entries)
    assert all(entry.bottom_document.kind for entry in payload.index_entries)


def test_generator_supports_family_and_reasoning_selection_for_drills() -> None:
    gen = SystemLogicGenerator(seed=88)

    oil_quant = cast(
        SystemLogicPayload,
        gen.next_problem_for_selection(
            difficulty=0.5,
            family="oil",
            reasoning_family="quantitative",
        ).payload,
    )
    fuel_trace = cast(
        SystemLogicPayload,
        gen.next_problem_for_selection(
            difficulty=0.5,
            family="fuel",
            reasoning_family="trace",
        ).payload,
    )

    assert oil_quant.system_family == "oil"
    assert oil_quant.reasoning_mode == "quantitative_duration"
    assert fuel_trace.system_family == "fuel"
    assert fuel_trace.reasoning_mode == "dependency_trace"


def test_each_family_has_more_than_one_reasoning_template() -> None:
    families = ("oil", "fuel", "electrical", "hydraulic", "thermal")
    for offset, family in enumerate(families):
        gen = SystemLogicGenerator(seed=200 + offset)
        seen: set[str] = set()
        for _ in range(10):
            payload = cast(
                SystemLogicPayload,
                gen.next_problem_for_selection(difficulty=0.5, family=family).payload,
            )
            seen.add(payload.reasoning_mode)
        assert len(seen) >= 2


def test_scorer_is_binary_multiple_choice() -> None:
    gen = SystemLogicGenerator(seed=41)
    problem = gen.next_problem(difficulty=0.75)
    scorer = SystemLogicScorer()

    exact = scorer.score(problem=problem, user_answer=problem.answer, raw=str(problem.answer))
    wrong = scorer.score(problem=problem, user_answer=((problem.answer % 5) + 1), raw="9")

    assert exact == 1.0
    assert wrong == 0.0


def test_default_config_matches_guide_timing() -> None:
    cfg = SystemLogicConfig()
    assert cfg.practice_questions == 3
    assert cfg.scored_duration_s == 34.0 * 60.0


def test_timer_boundary_transitions_to_results_and_rejects_late_submit() -> None:
    clock = FakeClock()
    engine = build_system_logic_test(
        clock=clock,
        seed=123,
        difficulty=0.5,
        config=SystemLogicConfig(scored_duration_s=2.0, practice_questions=0),
    )
    engine.start_scored()

    clock.advance(2.0)
    engine.update()

    assert engine.phase.value == "results"
    assert engine.submit_answer("1") is False

    summary = engine.scored_summary()
    assert summary.attempted == 0
    assert summary.correct == 0
