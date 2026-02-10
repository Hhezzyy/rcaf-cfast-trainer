from __future__ import annotations

from dataclasses import dataclass

from .clock import Clock
from .cognitive_core import Problem, SeededRng, TimedTextInputTest, lerp_int


@dataclass(frozen=True, slots=True)
class NumericalOperationsConfig:
    scored_duration_s: float = 120.0
    practice_questions: int = 5


class NumericalOperationsGenerator:
    """Generates mental arithmetic problems (+ - × ÷) with integer results."""

    def __init__(self, *, seed: int) -> None:
        self._rng = SeededRng(seed)

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
        divisor = self._rng.randint(2, lerp_int(9, 25, difficulty))
        quotient = self._rng.randint(2, lerp_int(9, 25, difficulty))
        dividend = divisor * quotient
        return Problem(prompt=f"{dividend} ÷ {divisor} =", answer=quotient)

    def _pick_operator(self, difficulty: float) -> str:
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
        lo = 1
        hi = lerp_int(9, 99, difficulty)
        return self._rng.randint(lo, hi), self._rng.randint(lo, hi)

    def _operands_mult(self, difficulty: float) -> tuple[int, int]:
        # Keep multiplication within reasonable mental range.
        a_hi = lerp_int(9, 25, difficulty)
        b_hi = lerp_int(9, 15, difficulty)
        return self._rng.randint(2, a_hi), self._rng.randint(2, b_hi)


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