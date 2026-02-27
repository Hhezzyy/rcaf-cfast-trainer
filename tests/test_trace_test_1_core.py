from __future__ import annotations

from dataclasses import dataclass

import pytest

from cfast_trainer.trace_test_1 import (
    TraceTest1Command,
    TraceTest1Config,
    TraceTest1Generator,
    TraceTest1Payload,
    build_trace_test_1_test,
)


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


def test_generator_determinism_same_seed_same_sequence() -> None:
    seed = 901
    g1 = TraceTest1Generator(seed=seed)
    g2 = TraceTest1Generator(seed=seed)

    seq1 = [g1.next_problem(difficulty=0.58) for _ in range(30)]
    seq2 = [g2.next_problem(difficulty=0.58) for _ in range(30)]

    assert [(p.prompt, p.answer, p.payload) for p in seq1] == [
        (p.prompt, p.answer, p.payload) for p in seq2
    ]


def test_generated_answer_maps_to_payload_correct_code_and_options() -> None:
    gen = TraceTest1Generator(seed=51)
    problem = gen.next_problem(difficulty=0.5)

    payload = problem.payload
    assert isinstance(payload, TraceTest1Payload)
    assert len(payload.options) == 4
    assert tuple(option.label for option in payload.options) == ("Left", "Right", "Push", "Pull")
    assert tuple(option.code for option in payload.options) == (1, 2, 3, 4)
    assert payload.correct_code == problem.answer
    assert 1 <= payload.correct_code <= 4


def test_generator_emits_all_command_types() -> None:
    gen = TraceTest1Generator(seed=1234)
    seen: set[TraceTest1Command] = set()

    for _ in range(140):
        problem = gen.next_problem(difficulty=0.6)
        payload = problem.payload
        assert isinstance(payload, TraceTest1Payload)
        option_map = {int(option.code): option.command for option in payload.options}
        seen.add(option_map[int(payload.correct_code)])

    assert seen == {
        TraceTest1Command.LEFT,
        TraceTest1Command.RIGHT,
        TraceTest1Command.PUSH,
        TraceTest1Command.PULL,
    }


def test_timer_boundary_transitions_to_results_and_rejects_late_submit() -> None:
    clock = FakeClock()
    engine = build_trace_test_1_test(
        clock=clock,
        seed=7,
        difficulty=0.5,
        config=TraceTest1Config(scored_duration_s=2.0, practice_questions=0),
    )
    engine.start_scored()

    assert engine.time_remaining_s() == pytest.approx(2.0)
    clock.advance(2.0)
    engine.update()

    assert engine.phase.value == "results"
    assert engine.submit_answer("1") is False
