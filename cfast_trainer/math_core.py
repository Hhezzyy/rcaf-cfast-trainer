"""Deterministic core logic for the Numerical Operations (Mathematics Reasoning) test.

This module contains a set of classes that implement the timed mental arithmetic
test described in the Canadian Forces Aircrew Selection Centre candidate guide.
It intentionally avoids any dependency on pygame or other input/output libraries
so that it can be exercised headlessly and unit tested deterministically.  A
``Clock`` abstraction provides monotonic time and can be replaced with a
``FakeClock`` for tests.  A seeded random number generator supplies an
independent stream of problems, ensuring reproducibility across runs when the
same seed is provided.

The core concepts include:

* ``Clock`` interface with concrete implementations for real time and faked time
  progression.
* ``MathProblem`` dataclass representing a single arithmetic problem and its
  correct answer.
* ``MathProblemGenerator`` producing a sequence of math problems based on a
  difficulty scalar.
* ``MathTestEngine`` managing the lifecycle of a single test attempt: start
  time, remaining time, problem presentation, answer submission, scoring and
  final summary metrics.

The engine deliberately does not enforce any particular number of problems;
instead it continues to serve new problems until the time limit expires.
This mirrors the real test, where candidates work through as many items as
possible in the allotted time.
"""

from __future__ import annotations

import math
import operator
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Optional, Tuple


class Clock:
    """Abstract base class for timekeeping.

    A ``Clock`` exposes a single method, :meth:`time`, which must return
    monotonically increasing time in seconds.  Real-time and faked versions
    derive from this base class.  Tests should provide a ``FakeClock`` so
    that deterministic time advancement is under their control.
    """

    def time(self) -> float:
        """Return the current monotonic time in seconds."""
        raise NotImplementedError


class RealClock(Clock):
    """Clock implementation based on ``time.monotonic``."""

    def time(self) -> float:  # pragma: no cover - trivial wrapper
        return time.monotonic()


class FakeClock(Clock):
    """Controllable clock for deterministic testing.

    The clock starts at zero by default and advances only when
    :meth:`advance` is called.  Tests use this to simulate the passage of
    time without waiting in real time.
    """

    def __init__(self, start: float = 0.0) -> None:
        self._now = float(start)

    def time(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("Cannot advance clock backwards")
        self._now += seconds


@dataclass(frozen=True)
class MathProblem:
    """Representation of a single arithmetic problem.

    Attributes:
        operand1: The first operand.
        operand2: The second operand.
        operator: A string identifying the operation ('+', '-', '*', '/').
        answer: The correct answer to the problem (an integer).
        display: A formatted string representation of the problem (e.g., "3 + 4 = ").
    """

    operand1: int
    operand2: int
    operator: str
    answer: int = field(init=False)
    display: str = field(init=False)

    def __post_init__(self) -> None:
        op_map: Dict[str, Callable[[int, int], int]] = {
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul,
            '/': lambda a, b: a // b,
        }
        func = op_map[self.operator]
        # Compute the answer using integer division for '/'
        object.__setattr__(self, 'answer', func(self.operand1, self.operand2))
        object.__setattr__(self, 'display', f"{self.operand1} {self.operator} {self.operand2} = ")


class MathProblemGenerator:
    """Generates a reproducible sequence of arithmetic problems.

    The generator uses a ``random.Random`` instance to produce operands and
    choose operations.  A difficulty scalar can be used to scale the range of
    operands; at difficulty 1 the operands are in the range [1, 9].  At
    difficulty 2 the range expands to [1, 19], and so on.  Division problems
    are constrained so that the dividend is divisible by the divisor and the
    result is an integer.
    """

    def __init__(self, seed: Optional[int] = None, difficulty: int = 1) -> None:
        self.rng = random.Random(seed)
        self.difficulty = max(1, difficulty)
        # Mapping of operators to weights; addition and subtraction are twice as likely
        self.operators: List[str] = ['+', '-', '*', '/']
        self.weights: List[int] = [4, 4, 3, 2]

    def _rand_operand(self) -> int:
        # Scale operand range with difficulty
        # For difficulty=1 returns [1,9], difficulty=2 returns [1,19], etc.
        return self.rng.randint(1, self.difficulty * 10 - 1)

    def next_problem(self) -> MathProblem:
        op = self.rng.choices(self.operators, weights=self.weights, k=1)[0]
        if op == '/':
            # Generate operands such that dividend is divisible by divisor and no division by zero
            b = self._rand_operand()
            # Ensure b is not zero (randrange above prevents zero) but safe guard anyway
            if b == 0:
                b = 1
            # Choose quotient in the same operand range
            quotient = self._rand_operand()
            a = b * quotient
            return MathProblem(a, b, op)
        else:
            a = self._rand_operand()
            b = self._rand_operand()
            return MathProblem(a, b, op)


@dataclass
class MathAttempt:
    """Record of a single problem attempt during a test."""

    problem: MathProblem
    response: Optional[int]
    correct: bool
    response_time: float


class MathTestEngine:
    """Controls the flow and scoring of a timed mathematics reasoning test.

    The engine is initialised with a total duration and optional random seed.
    Once :meth:`start` is called, it generates problems one at a time until
    the time limit expires.  Client code must call :meth:`submit_answer`
    whenever the user supplies an answer; this records the response and
    immediately advances to the next problem if time remains.  When the time
    is up, :meth:`finished` becomes ``True`` and no further problems are
    generated.
    """

    def __init__(self, duration_seconds: float = 120.0, *, seed: Optional[int] = None,
                 clock: Optional[Clock] = None, difficulty: int = 1) -> None:
        self.duration = float(duration_seconds)
        self.clock: Clock = clock if clock is not None else RealClock()
        self.generator = MathProblemGenerator(seed, difficulty)
        self.current_problem: Optional[MathProblem] = None
        self.attempts: List[MathAttempt] = []
        self._start_time: Optional[float] = None
        self._problem_start_time: Optional[float] = None
        self._finished = False

    def start(self) -> None:
        """Begin the test timer and generate the first problem."""
        if self._start_time is not None:
            raise RuntimeError("Test already started")
        self._start_time = self.clock.time()
        self._problem_start_time = self._start_time
        self.current_problem = self.generator.next_problem()

    @property
    def time_remaining(self) -> float:
        """Amount of time left in the test in seconds (never negative)."""
        if self._start_time is None:
            return self.duration
        remaining = self.duration - (self.clock.time() - self._start_time)
        return max(0.0, remaining)

    @property
    def finished(self) -> bool:
        """Return True if the time limit has been reached."""
        if self._finished:
            return True
        if self._start_time is None:
            return False
        # Recompute finished each call to avoid missing end-of-test
        if self.clock.time() - self._start_time >= self.duration:
            self._finished = True
        return self._finished

    def _record_attempt(self, response: Optional[int], correct: bool, response_time: float) -> None:
        assert self.current_problem is not None
        self.attempts.append(
            MathAttempt(
                problem=self.current_problem,
                response=response,
                correct=correct,
                response_time=response_time,
            )
        )

    def submit_answer(self, answer: Optional[int]) -> None:
        """Record the user's answer for the current problem and advance.

        If ``answer`` is None, it is treated as an unanswered problem (counts
        as incorrect).  If the test time has already expired, calling this
        method does nothing.
        """
        if self.finished:
            return
        if self.current_problem is None:
            raise RuntimeError("Test has not started")
        # Compute response time for this problem
        assert self._problem_start_time is not None
        now = self.clock.time()
        response_time = now - self._problem_start_time
        # Determine correctness
        correct = answer is not None and answer == self.current_problem.answer
        self._record_attempt(answer, correct, response_time)
        # Advance to next problem if time permits
        if not self.finished:
            self.current_problem = self.generator.next_problem()
            self._problem_start_time = self.clock.time()
        else:
            self.current_problem = None

    def summary(self) -> Dict[str, float | int]:
        """Return a dictionary with aggregate results for this test attempt.

        Metrics include:
        * ``attempted`` – number of problems attempted (answered or unanswered)
        * ``correct`` – number of correct answers
        * ``accuracy`` – proportion of correct answers (0 if none attempted)
        * ``throughput`` – problems attempted per minute
        * ``avg_response_time`` – average time per attempt in seconds
        """
        attempted = len(self.attempts)
        correct = sum(1 for a in self.attempts if a.correct)
        accuracy = (correct / attempted) if attempted > 0 else 0.0
        total_time = sum(a.response_time for a in self.attempts)
        avg_response_time = (total_time / attempted) if attempted > 0 else 0.0
        # Throughput: attempted per minute over the duration used
        elapsed = 0.0
        if self._start_time is not None:
            end_time = self._start_time + self.duration
            # If finished early, we still consider the full allotted time for throughput
            elapsed = self.duration
        throughput = (attempted / elapsed * 60.0) if elapsed > 0 else 0.0
        return {
            'attempted': attempted,
            'correct': correct,
            'accuracy': accuracy,
            'throughput': throughput,
            'avg_response_time': avg_response_time,
        }