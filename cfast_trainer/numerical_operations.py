from __future__ import annotations

from dataclasses import dataclass

from .clock import Clock
from .cognitive_core import Problem, SeededRng, TimedTextInputTest, lerp_int


@dataclass(frozen=True, slots=True)
class NumericalOperationsConfig:
    scored_duration_s: float = 120.0
    practice_questions: int = 5


@dataclass(frozen=True, slots=True)
class NumericalOperationsProblemProfile:
    operator_family: str | None = None
    operand_profile: str = "default"


class NumericalOperationsGenerator:
    """Generates mental arithmetic problems (+ - × ÷) with integer results."""

    def __init__(
        self,
        *,
        seed: int,
        profile: NumericalOperationsProblemProfile | None = None,
    ) -> None:
        self._rng = SeededRng(seed)
        self._profile = profile or NumericalOperationsProblemProfile()

    def next_problem(self, *, difficulty: float) -> Problem:
        # difficulty 0..1 controls operand ranges and operator mix.
        difficulty = max(0.0, min(1.0, difficulty))

        op = self._pick_operator(difficulty)

        if op == "+":
            a, b = self._operands(difficulty)
            return Problem(prompt=f"{a} + {b} =", answer=a + b)
        if op == "-":
            a, b = self._operands(difficulty)
            if b > a:
                a, b = b, a
            return Problem(prompt=f"{a} - {b} =", answer=a - b)
        if op == "*":
            a, b = self._operands_mult(difficulty)
            return Problem(prompt=f"{a} × {b} =", answer=a * b)

        # Division: force clean integer quotient.
        dividend, divisor, quotient = self.division_terms(difficulty)
        return Problem(prompt=f"{dividend} ÷ {divisor} =", answer=quotient)

    def _pick_operator(self, difficulty: float) -> str:
        forced = (self._profile.operator_family or "").strip()
        if forced in {"+", "-", "*", "/"}:
            return forced
        # Easy: more + and -; hard: more * and ÷.
        r = self._rng.uniform(0.0, 1.0)
        if r < (0.45 - 0.20 * difficulty):
            return "+"
        if r < (0.80 - 0.20 * difficulty):
            return "-"
        if r < (0.92 + 0.04 * difficulty):
            return "*"
        return "/"

    def _operands(self, difficulty: float) -> tuple[int, int]:
        profile = self._profile.operand_profile
        if profile == "fact_prime":
            return self._fact_prime_operands(difficulty)
        if profile == "clean_compute":
            return self._clean_compute_operands(difficulty)
        lo = 1
        hi = lerp_int(9, 99, difficulty)
        return self._rng.randint(lo, hi), self._rng.randint(lo, hi)

    def _operands_mult(self, difficulty: float) -> tuple[int, int]:
        profile = self._profile.operand_profile
        if profile == "fact_prime":
            return self._fact_prime_mult_operands(difficulty)
        if profile == "clean_compute":
            return self._clean_compute_mult_operands(difficulty)
        # Keep multiplication within reasonable mental range.
        a_hi = lerp_int(9, 25, difficulty)
        b_hi = lerp_int(9, 15, difficulty)
        return self._rng.randint(2, a_hi), self._rng.randint(2, b_hi)

    def _fact_prime_operands(self, difficulty: float) -> tuple[int, int]:
        family = self._rng.choice(
            (
                "doubles",
                "near_doubles",
                "complements_10",
                "complements_100",
                "clean_tens",
            )
        )
        if family == "doubles":
            hi = lerp_int(9, 35, difficulty)
            value = self._rng.randint(2, hi)
            return value, value
        if family == "near_doubles":
            hi = lerp_int(9, 40, difficulty)
            value = self._rng.randint(4, hi)
            return value, value + self._rng.choice((1, 2))
        if family == "complements_10":
            a = self._rng.randint(1, 9)
            return a, 10 - a
        if family == "complements_100":
            anchor = self._rng.choice((10, 20, 30, 40, 50, 60, 70, 80, 90))
            return anchor, 100 - anchor
        a = self._rng.choice((10, 20, 25, 30, 40, 50))
        b = self._rng.choice((5, 10, 15, 20, 25, 30))
        return a, b

    def _fact_prime_mult_operands(self, difficulty: float) -> tuple[int, int]:
        family = self._rng.choice(("table", "clean_tens", "halves"))
        if family == "table":
            hi = lerp_int(9, 12, difficulty)
            return self._rng.randint(2, hi), self._rng.randint(2, hi)
        if family == "halves":
            even = self._rng.choice((4, 6, 8, 10, 12, 14, 16, 18))
            return even, 5
        return self._rng.choice((10, 12, 15, 20, 25)), self._rng.randint(2, 9)

    def _clean_compute_operands(self, difficulty: float) -> tuple[int, int]:
        anchor = self._rng.choice((10, 20, 25, 30, 40, 50, 60, 75, 80, 90))
        spread = lerp_int(8, 45, difficulty)
        other = self._rng.randint(max(2, anchor - spread), min(99, anchor + spread))
        return anchor, other

    def _clean_compute_mult_operands(self, difficulty: float) -> tuple[int, int]:
        a = self._rng.choice((5, 6, 8, 9, 10, 12, 15, 20, 25))
        hi = lerp_int(9, 18, difficulty)
        return a, self._rng.randint(2, hi)

    def division_terms(self, difficulty: float) -> tuple[int, int, int]:
        profile = self._profile.operand_profile
        if profile == "fact_prime":
            divisor = self._rng.randint(2, 12)
            quotient = self._rng.randint(2, lerp_int(9, 14, difficulty))
        elif profile == "clean_compute":
            divisor = self._rng.choice((2, 4, 5, 8, 10, 12))
            quotient = self._rng.randint(2, lerp_int(12, 24, difficulty))
        else:
            divisor = self._rng.randint(2, lerp_int(9, 25, difficulty))
            quotient = self._rng.randint(2, lerp_int(9, 25, difficulty))
        dividend = divisor * quotient
        return dividend, divisor, quotient


def build_numerical_operations_test(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    config: NumericalOperationsConfig | None = None,
) -> TimedTextInputTest:
    """Factory for the Numerical Operations test session."""

    cfg = config or NumericalOperationsConfig()

    instructions = [
        "Numerical Operations (Mental Arithmetic)",
        "",
        "Answer as many arithmetic problems as you can.",
        "Use mental math: no calculator, no paper.",
        "",
        "Controls:",
        "- Type your answer",
        "- Press Enter to submit",
        "",
        "You will get a short practice, then a timed 2-minute scored block.",
    ]

    generator = NumericalOperationsGenerator(seed=seed)

    return TimedTextInputTest(
        title="Numerical Operations",
        instructions=instructions,
        generator=generator,
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        practice_questions=cfg.practice_questions,
        scored_duration_s=cfg.scored_duration_s,
    )
