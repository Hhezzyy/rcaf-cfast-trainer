from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from .clock import Clock


class ProblemGenerator(Protocol):
    """Deterministic generator of problems."""

    def next_problem(self, *, difficulty: float) -> "Problem":
        ...


class AnswerScorer(Protocol):
    def score(self, *, problem: "Problem", user_answer: int, raw: str) -> float:
        """Return score in [0.0, 1.0]."""
        ...


class Phase(str, Enum):
    INSTRUCTIONS = "instructions"
    PRACTICE = "practice"
    PRACTICE_DONE = "practice_done"
    SCORED = "scored"
    RESULTS = "results"


@dataclass(frozen=True, slots=True)
class Problem:
    prompt: str
    answer: int
    tolerance: int = 0  # abs(user_answer - answer) <= tolerance
    payload: object | None = None  # optional structured data for UI


@dataclass(frozen=True, slots=True)
class QuestionEvent:
    index: int
    phase: Phase
    prompt: str
    correct_answer: int
    user_answer: int
    is_correct: bool
    presented_at_s: float
    answered_at_s: float
    response_time_s: float
    raw: str = ""
    score: float = 0.0
    max_score: float = 1.0


@dataclass(frozen=True, slots=True)
class AttemptSummary:
    attempted: int
    correct: int
    accuracy: float
    duration_s: float
    throughput_per_min: float
    mean_response_time_s: float | None
    total_score: float = 0.0
    max_score: float = 0.0
    score_ratio: float = 0.0


@dataclass(frozen=True, slots=True)
class TestSnapshot:
    """View model for the UI (pure data)."""

    title: str
    phase: Phase
    prompt: str
    input_hint: str
    time_remaining_s: float | None
    attempted_scored: int
    correct_scored: int
    payload: object | None = None
    practice_feedback: str | None = None


class TimedTextInputTest:
    """Reusable test harness: instructions -> practice -> timed scored -> results.

    - Deterministic: problem stream comes from RNG seeded at construction.
    - Time is entirely via injected Clock.
    """

    def __init__(
        self,
        *,
        title: str,
        instructions: list[str],
        generator: ProblemGenerator,
        clock: Clock,
        seed: int,
        difficulty: float = 0.5,
        practice_questions: int = 5,
        scored_duration_s: float,
        scorer: AnswerScorer | None = None,
    ) -> None:
        if not (0.0 <= difficulty <= 1.0):
            raise ValueError("difficulty must be in [0.0, 1.0]")
        if practice_questions < 0:
            raise ValueError("practice_questions must be >= 0")
        if scored_duration_s <= 0:
            raise ValueError("scored_duration_s must be > 0")

        self._title = title
        self._instructions = instructions
        self._generator = generator
        self._clock = clock
        self._difficulty = difficulty
        self._practice_questions = practice_questions
        self._scored_duration_s = float(scored_duration_s)

        # RNG is owned by generator in most designs; still store seed for persistence.
        self._seed = int(seed)

        self._phase: Phase = Phase.INSTRUCTIONS
        self._current: Problem | None = None
        self._presented_at_s: float | None = None
        self._events: list[QuestionEvent] = []

        self._practice_answered = 0
        self._scored_started_at_s: float | None = None

        self._scored_attempted = 0
        self._scored_correct = 0

        self._scorer = scorer
        self._scored_total_score = 0.0
        self._scored_max_score = 0.0

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def phase(self) -> Phase:
        return self._phase

    def instructions(self) -> list[str]:
        return list(self._instructions)

    def can_exit(self) -> bool:
        # Enforce "must continue once started" during SCORED.
        return self._phase in (Phase.INSTRUCTIONS, Phase.PRACTICE, Phase.PRACTICE_DONE, Phase.RESULTS)

    def start_practice(self) -> None:
        if self._phase is not Phase.INSTRUCTIONS:
            return
        self._phase = Phase.PRACTICE
        self._deal_new_problem()

    def start_scored(self) -> None:
        if self._phase in (Phase.SCORED, Phase.RESULTS):
            return
        self._phase = Phase.SCORED
        self._scored_started_at_s = self._clock.now()
        self._deal_new_problem()

    def update(self) -> None:
        if self._phase is not Phase.SCORED:
            return
        remaining = self.time_remaining_s()
        if remaining is not None and remaining <= 0.0:
            self._phase = Phase.RESULTS

    def time_remaining_s(self) -> float | None:
        if self._phase is not Phase.SCORED:
            return None
        assert self._scored_started_at_s is not None
        remaining = self._scored_duration_s - (self._clock.now() - self._scored_started_at_s)
        return max(0.0, remaining)

    def current_prompt(self) -> str:
        if self._phase is Phase.INSTRUCTIONS:
            return "Press Enter to begin practice."
        if self._phase is Phase.PRACTICE_DONE:
            return "Practice complete. Press Enter to start the timed test."
        if self._phase is Phase.RESULTS:
            s = self.scored_summary()
            acc_pct = int(round(s.accuracy * 100))
            rt = "n/a" if s.mean_response_time_s is None else f"{s.mean_response_time_s:.2f}s"
            return (
                f"Results\nAttempted: {s.attempted}\nCorrect: {s.correct}\n"
                f"Accuracy: {acc_pct}%\nMean RT: {rt}\nThroughput: {s.throughput_per_min:.1f}/min"
            )
        if self._current is None:
            return ""
        return self._current.prompt

    def submit_answer(self, raw: str) -> bool:
        """Submit a typed answer. Returns True if accepted."""

        if self._phase not in (Phase.PRACTICE, Phase.SCORED):
            return False
        expired = self._phase is Phase.SCORED and self.time_remaining_s() == 0

        raw_in = raw
        raw = raw.strip()
        if expired and raw == "":
            # Timeout with no partial entry.
            self._phase = Phase.RESULTS
            return False
        if raw == "":
            return False

        try:
            user_answer = int(raw)
        except ValueError:
            if expired:
                self._phase = Phase.RESULTS
            return False

        assert self._current is not None
        assert self._presented_at_s is not None

        answered_at_s = self._clock.now()
        response_time_s = max(0.0, answered_at_s - self._presented_at_s)

        if self._scorer is None:
            tol = 0 if self._current.tolerance < 0 else self._current.tolerance
            score = 1.0 if abs(user_answer - self._current.answer) <= tol else 0.0
        else:
            score = float(self._scorer.score(problem=self._current, user_answer=user_answer, raw=raw_in))
            if score < 0.0:
                score = 0.0
            elif score > 1.0:
                score = 1.0

        is_full_correct = score >= 1.0 - 1e-9

        event = QuestionEvent(
            index=len(self._events),
            phase=self._phase,
            prompt=self._current.prompt,
            correct_answer=self._current.answer,
            user_answer=user_answer,
            is_correct=is_full_correct,
            presented_at_s=self._presented_at_s,
            answered_at_s=answered_at_s,
            response_time_s=response_time_s,
            raw=raw_in,
            score=score,
            max_score=1.0,
        )
        self._events.append(event)

        if self._phase is Phase.SCORED:
            self._scored_attempted += 1
            self._scored_max_score += 1.0
            self._scored_total_score += score
            if is_full_correct:
                self._scored_correct += 1
        else:
            self._practice_answered += 1

        # If time expired, accept the submission but end the test without dealing a new problem.
        if expired and self._phase is Phase.SCORED:
            self._phase = Phase.RESULTS
            self._current = None
            self._presented_at_s = None
            return True

        # Advance / transition.
        if self._phase is Phase.PRACTICE and self._practice_answered >= self._practice_questions:
            self._phase = Phase.PRACTICE_DONE
            self._current = None
            self._presented_at_s = None
            return True

        self._deal_new_problem()
        return True

    def scored_summary(self) -> AttemptSummary:
        duration_s = float(self._scored_duration_s)
        attempted = self._scored_attempted
        correct = self._scored_correct
        accuracy = 0.0 if attempted == 0 else correct / attempted
        throughput = (attempted / duration_s) * 60.0
        rts = [e.response_time_s for e in self._events if e.phase is Phase.SCORED]
        mean_rt = None if not rts else sum(rts) / len(rts)

        total_score = float(self._scored_total_score)
        max_score = float(self._scored_max_score)
        score_ratio = 0.0 if max_score == 0.0 else total_score / max_score

        return AttemptSummary(
            attempted=attempted,
            correct=correct,
            accuracy=accuracy,
            duration_s=duration_s,
            throughput_per_min=throughput,
            mean_response_time_s=mean_rt,
            total_score=total_score,
            max_score=max_score,
            score_ratio=score_ratio,
        )

    def snapshot(self) -> TestSnapshot:
        payload = None if self._current is None else self._current.payload
        return TestSnapshot(
            title=self._title,
            phase=self._phase,
            prompt=self.current_prompt(),
            input_hint="Type answer then Enter",
            time_remaining_s=self.time_remaining_s(),
            attempted_scored=self._scored_attempted,
            correct_scored=self._scored_correct,
            payload=payload,
        )

    def _deal_new_problem(self) -> None:
        self._current = self._generator.next_problem(difficulty=self._difficulty)
        self._presented_at_s = self._clock.now()


class SeededRng:
    """Simple seeded RNG wrapper to keep deterministic streams explicit."""

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(int(seed))

    def randint(self, a: int, b: int) -> int:
        return self._rng.randint(a, b)

    def choice(self, seq: list[str]) -> str:
        return self._rng.choice(seq)

    def uniform(self, a: float, b: float) -> float:
        return self._rng.uniform(a, b)


def lerp_int(a: int, b: int, t: float) -> int:
    """Linear interpolation in integer space (inclusive bounds)."""

    if t <= 0:
        return a
    if t >= 1:
        return b
    return int(round(a + (b - a) * t))


def clamp01(x: float) -> float:
    return 0.0 if x <= 0.0 else 1.0 if x >= 1.0 else float(x)


def round_half_up(x: float) -> int:
    # For consistent educational-style rounding when needed.
    return int(math.floor(x + 0.5))
