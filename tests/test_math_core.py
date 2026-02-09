"""Tests for the deterministic mathematics test core.

These unit tests exercise the problem generator, scoring logic, timer
handling and metric aggregation provided by ``math_core``.  A ``FakeClock``
is used where appropriate to simulate the passage of time without
introducing real delays.  The tests avoid any dependency on pygame and
should run in any environment.
"""

from __future__ import annotations

import math

import pytest

from cfast_trainer.math_core import (
    FakeClock,
    MathProblemGenerator,
    MathTestEngine,
)


def test_generator_determinism() -> None:
    """Generators with the same seed should produce identical sequences."""
    seed = 12345
    g1 = MathProblemGenerator(seed=seed, difficulty=2)
    g2 = MathProblemGenerator(seed=seed, difficulty=2)
    seq1 = [g1.next_problem() for _ in range(10)]
    seq2 = [g2.next_problem() for _ in range(10)]
    # Compare operator, operands and answers
    for p1, p2 in zip(seq1, seq2):
        assert (p1.operand1, p1.operator, p1.operand2, p1.answer) == (
            p2.operand1,
            p2.operator,
            p2.operand2,
            p2.answer,
        )


def test_scoring_and_metrics() -> None:
    """Submitting answers updates scoring metrics correctly."""
    clock = FakeClock()
    engine = MathTestEngine(duration_seconds=10.0, seed=42, difficulty=1, clock=clock)
    engine.start()
    # First problem: answer correctly
    current = engine.current_problem
    assert current is not None
    clock.advance(1.0)
    engine.submit_answer(current.answer)
    # Second problem: answer correctly
    current = engine.current_problem
    assert current is not None
    clock.advance(1.0)
    engine.submit_answer(current.answer)
    # Third problem: answer incorrectly
    current = engine.current_problem
    assert current is not None
    clock.advance(1.0)
    engine.submit_answer(current.answer + 1)  # wrong answer
    # Ensure the engine isn't finished yet
    assert not engine.finished
    # Summary should reflect 3 attempts, 2 correct
    summary = engine.summary()
    assert summary['attempted'] == 3
    assert summary['correct'] == 2
    # Accuracy 2/3 ~= 0.6667
    assert math.isclose(summary['accuracy'], 2 / 3, rel_tol=1e-6)
    # Throughput: 3 problems over 10 seconds => 18 per minute
    assert math.isclose(summary['throughput'], (3 / 10) * 60, rel_tol=1e-6)
    # Average response time should be exactly 1 second
    assert math.isclose(summary['avg_response_time'], 1.0, rel_tol=1e-6)


def test_timer_boundary_behavior() -> None:
    """Problems answered after time expires should not be recorded."""
    clock = FakeClock()
    engine = MathTestEngine(duration_seconds=3.0, seed=7, difficulty=1, clock=clock)
    engine.start()
    # First answer near end of test
    first = engine.current_problem
    assert first is not None
    clock.advance(2.9)
    engine.submit_answer(first.answer)
    # Now advance past the time limit
    clock.advance(2.1)  # total now 5 seconds; duration was 3
    # Attempt to submit another answer; should be ignored
    engine.submit_answer(None)
    summary = engine.summary()
    # Only one attempt should be recorded
    assert summary['attempted'] == 1
    assert summary['correct'] == 1
    # Engine should report finished
    assert engine.finished


def test_headless_simulation() -> None:
    """Simulate a short run with scripted answers using a fake clock."""
    clock = FakeClock()
    engine = MathTestEngine(duration_seconds=5.0, seed=1, difficulty=1, clock=clock)
    engine.start()
    answers_given = 0
    while not engine.finished:
        problem = engine.current_problem
        assert problem is not None
        # Always answer correctly
        clock.advance(0.5)
        if engine.finished:
            break
        engine.submit_answer(problem.answer)
        answers_given += 1
    summary = engine.summary()
    # All answers should be correct
    assert summary['attempted'] == answers_given
    assert summary['correct'] == answers_given
    assert summary['accuracy'] == 1.0
    # Throughput should equal answers_given / duration * 60
    assert math.isclose(summary['throughput'], (answers_given / 5.0) * 60.0, rel_tol=1e-6)