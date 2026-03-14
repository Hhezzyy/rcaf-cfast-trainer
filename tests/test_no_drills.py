from __future__ import annotations

from dataclasses import dataclass

from cfast_trainer.ant_drills import AntDrillMode
from cfast_trainer.no_drills import (
    NoFactPrimeGenerator,
    NoOperatorLaddersGenerator,
    build_no_fact_prime_drill,
    build_no_pressure_run_drill,
)


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t


def test_no_fact_prime_is_deterministic_for_same_seed() -> None:
    gen_a = NoFactPrimeGenerator(seed=111)
    gen_b = NoFactPrimeGenerator(seed=111)

    seq_a = [gen_a.next_problem(difficulty=0.35) for _ in range(8)]
    seq_b = [gen_b.next_problem(difficulty=0.35) for _ in range(8)]

    assert [(item.prompt, item.answer) for item in seq_a] == [
        (item.prompt, item.answer) for item in seq_b
    ]


def test_no_operator_ladders_groups_operator_families() -> None:
    generator = NoOperatorLaddersGenerator(seed=77)
    prompts = [generator.next_problem(difficulty=0.4).prompt for _ in range(12)]
    operators = [prompt.split()[1] for prompt in prompts]

    assert operators[:3] == ["+", "+", "+"]
    assert operators[3:6] == ["-", "-", "-"]
    assert operators[6:9] == ["×", "×", "×"]
    assert operators[9:12] == ["÷", "÷", "÷"]


def test_no_pressure_run_uses_numerical_operations_title_shell() -> None:
    clock = FakeClock()
    engine = build_no_pressure_run_drill(
        clock=clock,
        seed=29,
        difficulty=0.6,
        mode=AntDrillMode.STRESS,
    )

    engine.start_practice()
    snap = engine.snapshot()

    assert snap.title.startswith("Numerical Operations: Pressure Run")
    assert any(symbol in snap.prompt for symbol in ("+", "-", "×", "÷"))
