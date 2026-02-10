from __future__ import annotations

from dataclasses import dataclass
import random
import time
from typing import Protocol


class Clock(Protocol):
    def now(self) -> float:
        """Return monotonic seconds."""
        ...

class RealClock:
    """Adapter clock (real monotonic time)."""

    def now(self) -> float:
        return time.monotonic()


class FakeClock:
    """Deterministic test clock."""

    def __init__(self, *, start: float = 0.0) -> None:
        self._t = float(start)

    def now(self) -> float:
        return self._t

    def advance(self, dt: float) -> None:
        if dt < 0:
            raise ValueError("dt must be non-negative")
        self._t += float(dt)

    def set(self, t: float) -> None:
        self._t = float(t)


@dataclass(frozen=True)
class ArithmeticProblem:
    text: str
    answer: int
    a: int
    b: int
    op: str


@dataclass(frozen=True)
class Trial:
    problem: ArithmeticProblem
    presented_at: float
    answered_at: float
    response_text: str
    response_value: int | None
    correct: bool


@dataclass(frozen=True)
class NumericalOperationsResults:
    seed: int
    difficulty: float
    attempted: int
    correct: int
    accuracy: float
    throughput_per_min: float
    mean_response_time_s: float
    response_times_s: tuple[float, ...]


@dataclass(frozen=True)
class NumericalOperationsSessionConfig:
    seed: int
    practice_questions: int = 5
    scored_duration_s: float = 120.0
    difficulty: float = 0.35


class ProblemGenerator:
    def __init__(self, rng: random.Random, *, difficulty: float) -> None:
        self._rng = rng
        self._difficulty = _clamp01(difficulty)

    def next_problem(self) -> ArithmeticProblem:
        op = self._rng.choice(["+", "-", "×", "÷"])

        if op == "+":
            max_v = _scale_int(9, 99, self._difficulty)
            a = self._rng.randint(0, max_v)
            b = self._rng.randint(0, max_v)
            ans = a + b
        elif op == "-":
            max_v = _scale_int(9, 99, self._difficulty)
            a = self._rng.randint(0, max_v)
            b = self._rng.randint(0, max_v)
            if b > a:
                a, b = b, a
            ans = a - b
        elif op == "×":
            max_v = _scale_int(9, 20, self._difficulty)
            a = self._rng.randint(0, max_v)
            b = self._rng.randint(0, max_v)
            ans = a * b
        else:  # "÷"
            max_v = _scale_int(9, 20, self._difficulty)
            divisor = self._rng.randint(1, max_v)
            quotient = self._rng.randint(0, max_v)
            dividend = divisor * quotient
            a, b = dividend, divisor
            ans = quotient

        return ArithmeticProblem(text=f"{a} {op} {b} =", answer=ans, a=a, b=b, op=op)


def _scale_int(lo: int, hi: int, t: float) -> int:
    return int(round(lo + (hi - lo) * _clamp01(t)))


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def _try_parse_int(text: str) -> int | None:
    s = text.strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None


class NumericalOperationsSession:
    """
    Deterministic session state machine:

      INSTRUCTIONS -> PRACTICE -> READY -> SCORED -> RESULTS

    Core logic is deterministic given (seed, scripted inputs, fake clock).
    """

    def __init__(self, config: NumericalOperationsSessionConfig, *, clock: Clock) -> None:
        self._config = config
        self._clock = clock
        self._rng = random.Random(config.seed)
        self._gen = ProblemGenerator(self._rng, difficulty=config.difficulty)

        self._phase: str = "INSTRUCTIONS"

        self._practice_done = 0
        self._practice_trials: list[Trial] = []

        self._scored_trials: list[Trial] = []
        self._scored_ends_at: float | None = None

        self._current: ArithmeticProblem | None = None
        self._presented_at: float | None = None

        self._results: NumericalOperationsResults | None = None

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def current_problem(self) -> ArithmeticProblem | None:
        return self._current

    @property
    def time_remaining_s(self) -> float:
        if self._phase != "SCORED":
            return 0.0
        assert self._scored_ends_at is not None
        return max(0.0, self._scored_ends_at - self._clock.now())

    @property
    def results(self) -> NumericalOperationsResults | None:
        return self._results

    def start_practice(self) -> None:
        if self._phase != "INSTRUCTIONS":
            return
        if self._config.practice_questions <= 0:
            self._phase = "READY"
            self._current = None
            self._presented_at = None
            return
        self._phase = "PRACTICE"
        self._next_problem()

    def start_scored(self) -> None:
        if self._phase != "READY":
            return
        now = self._clock.now()
        self._scored_ends_at = now + float(self._config.scored_duration_s)
        self._phase = "SCORED"
        self._next_problem()

    def tick(self) -> None:
        if self._phase == "SCORED":
            self._end_if_expired()

    def submit_answer(self, answer_text: str) -> str | None:
        if self._phase not in ("PRACTICE", "SCORED"):
            return None

        now = self._clock.now()

        if self._phase == "SCORED":
            assert self._scored_ends_at is not None
            if now >= self._scored_ends_at:
                self._end_scored()
                return None

        if self._current is None or self._presented_at is None:
            return None

        value = _try_parse_int(answer_text)
        correct = value is not None and value == self._current.answer

        trial = Trial(
            problem=self._current,
            presented_at=self._presented_at,
            answered_at=now,
            response_text=answer_text,
            response_value=value,
            correct=correct,
        )

        if self._phase == "PRACTICE":
            self._practice_trials.append(trial)
            self._practice_done += 1
            msg = "Correct." if correct else f"Incorrect. Answer: {trial.problem.answer}"
            if self._practice_done >= self._config.practice_questions:
                self._phase = "READY"
                self._current = None
                self._presented_at = None
                return "Practice complete."
            self._next_problem()
            return msg

        # SCORED
        self._scored_trials.append(trial)
        self._end_if_expired()
        if self._phase == "SCORED":
            self._next_problem()
        return None

    def _next_problem(self) -> None:
        self._current = self._gen.next_problem()
        self._presented_at = self._clock.now()

    def _end_if_expired(self) -> None:
        assert self._scored_ends_at is not None
        if self._clock.now() >= self._scored_ends_at:
            self._end_scored()

    def _end_scored(self) -> None:
        self._phase = "RESULTS"
        self._current = None
        self._presented_at = None
        self._results = _compute_results(self._config, self._scored_trials)


def _compute_results(config: NumericalOperationsSessionConfig, trials: list[Trial]) -> NumericalOperationsResults:
    attempted = len(trials)
    correct = sum(1 for t in trials if t.correct)
    accuracy = (correct / attempted) if attempted else 0.0

    duration_min = float(config.scored_duration_s) / 60.0
    throughput = (attempted / duration_min) if duration_min > 0 else 0.0

    rts = [max(0.0, t.answered_at - t.presented_at) for t in trials]
    mean_rt = (sum(rts) / len(rts)) if rts else 0.0

    return NumericalOperationsResults(
        seed=config.seed,
        difficulty=config.difficulty,
        attempted=attempted,
        correct=correct,
        accuracy=float(accuracy),
        throughput_per_min=float(throughput),
        mean_response_time_s=float(mean_rt),
        response_times_s=tuple(rts),
    )