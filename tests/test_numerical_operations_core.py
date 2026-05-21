from __future__ import annotations

from dataclasses import dataclass
import re

import pytest

from cfast_trainer.numerical_operations import (
    NumericalOperationsConfig,
    NumericalOperationsGenerator,
    NumericalOperationsProblemProfile,
    build_numerical_operations_test,
)


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def test_generator_determinism_same_seed_same_sequence() -> None:
    seed = 123
    gen1 = NumericalOperationsGenerator(seed=seed)
    gen2 = NumericalOperationsGenerator(seed=seed)

    seq1 = [gen1.next_problem(difficulty=0.7) for _ in range(50)]
    seq2 = [gen2.next_problem(difficulty=0.7) for _ in range(50)]

    assert [(p.prompt, p.answer) for p in seq1] == [(p.prompt, p.answer) for p in seq2]


def _prompt_numbers(prompt: str) -> list[int]:
    return [int(match) for match in re.findall(r"\d+", prompt)]


def _is_easy_high_level_operand(value: int) -> bool:
    value = abs(int(value))
    return value in {0, 1, 2, 5, 10, 100} or value % 5 == 0 or value % 10 == 0 or value % 100 == 0


@pytest.mark.parametrize("operator_family", ("+", "-"))
def test_level_10_addition_and_subtraction_use_hard_five_digit_operands(
    operator_family: str,
) -> None:
    gen = NumericalOperationsGenerator(
        seed=20260518,
        profile=NumericalOperationsProblemProfile(operator_family=operator_family),
    )

    for _ in range(20):
        problem = gen.next_problem(difficulty=1.0)
        left, right = _prompt_numbers(problem.prompt)[:2]

        assert 10_000 <= left <= 99_999
        assert 10_000 <= right <= 99_999
        assert not _is_easy_high_level_operand(left)
        assert not _is_easy_high_level_operand(right)
        if operator_family == "-":
            assert left > right
            assert problem.answer == left - right
        else:
            assert problem.answer == left + right


def test_level_10_multiplication_uses_hard_three_by_two_digit_operands() -> None:
    gen = NumericalOperationsGenerator(
        seed=20260519,
        profile=NumericalOperationsProblemProfile(operator_family="*"),
    )

    for _ in range(20):
        problem = gen.next_problem(difficulty=1.0)
        left, right = _prompt_numbers(problem.prompt)[:2]

        assert 100 <= left <= 999
        assert 11 <= right <= 99
        assert not _is_easy_high_level_operand(left)
        assert not _is_easy_high_level_operand(right)
        assert right % 2 == 1
        assert problem.answer == left * right


def test_level_10_division_uses_hard_inverse_three_by_two_digit_terms() -> None:
    gen = NumericalOperationsGenerator(
        seed=20260520,
        profile=NumericalOperationsProblemProfile(operator_family="/"),
    )

    for _ in range(20):
        problem = gen.next_problem(difficulty=1.0)
        dividend, divisor = _prompt_numbers(problem.prompt)[:2]
        quotient = int(problem.answer)

        assert 11 <= divisor <= 99
        assert 100 <= quotient <= 999
        assert not _is_easy_high_level_operand(divisor)
        assert divisor % 2 == 1
        assert not _is_easy_high_level_operand(quotient)
        assert dividend == divisor * quotient


@pytest.mark.parametrize("operator_family", ("+", "-", "*", "/"))
def test_cln_light_profile_stays_capped_at_max_difficulty(operator_family: str) -> None:
    gen = NumericalOperationsGenerator(
        seed=3030,
        profile=NumericalOperationsProblemProfile(
            operator_family=operator_family,
            operand_profile="cln_light",
        ),
    )

    for _ in range(40):
        problem = gen.next_problem(difficulty=1.0)
        left, right = _prompt_numbers(problem.prompt)[:2]

        if operator_family in {"+", "-"}:
            assert 1 <= left <= 60
            assert 1 <= right <= 60
            if operator_family == "-":
                assert left > right
        elif operator_family == "*":
            assert 2 <= left <= 30
            assert 2 <= right <= 30
            assert problem.answer == left * right
        else:
            quotient = int(problem.answer)
            assert 2 <= right <= 30
            assert 2 <= quotient <= 30
            assert left == right * quotient


def test_scoring_counts_only_scored_phase() -> None:
    # Zero practice questions: go straight into SCORED.
    seed = 42
    clock = FakeClock()

    engine = build_numerical_operations_test(
        clock=clock,
        seed=seed,
        difficulty=0.5,
        config=NumericalOperationsConfig(scored_duration_s=10.0, practice_questions=0),
    )

    engine.start_scored()

    # Mirror the generator stream to know correct answers.
    gen = NumericalOperationsGenerator(seed=seed)

    for _ in range(3):
        p = gen.next_problem(difficulty=0.5)
        clock.advance(0.25)
        assert engine.submit_answer(str(p.answer)) is True

    s = engine.scored_summary()
    assert s.attempted == 3
    assert s.correct == 3
    assert s.accuracy == 1.0


def test_timer_boundary_transitions_to_results_and_rejects_late_submit() -> None:
    seed = 7
    clock = FakeClock()

    engine = build_numerical_operations_test(
        clock=clock,
        seed=seed,
        difficulty=0.5,
        config=NumericalOperationsConfig(scored_duration_s=2.0, practice_questions=0),
    )
    engine.start_scored()

    clock.advance(2.0)
    engine.update()

    assert engine.phase.value == "results"
    assert engine.submit_answer("0") is False
