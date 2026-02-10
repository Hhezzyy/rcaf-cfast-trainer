from __future__ import annotations

from dataclasses import dataclass

from .clock import Clock
from .cognitive_core import Problem, SeededRng, TimedTextInputTest, lerp_int


@dataclass(frozen=True, slots=True)
class MathReasoningConfig:
    scored_duration_s: float = 720.0  # training-friendly default
    practice_questions: int = 3


class MathReasoningGenerator:
    """Generates time/speed/distance word problems with integer answers.

    The candidate guide describes the task as interpreting written descriptions to solve
    numerical problems using time/speed/distance calculations.
    """

    def __init__(self, *, seed: int) -> None:
        self._rng = SeededRng(seed)

    def next_problem(self, *, difficulty: float) -> Problem:
        difficulty = max(0.0, min(1.0, difficulty))

        kind = self._rng.randint(0, 2)
        if kind == 0:
            return self._distance_from_speed_time(difficulty)
        if kind == 1:
            return self._time_from_distance_speed(difficulty)
        return self._speed_from_distance_time(difficulty)

    def _distance_from_speed_time(self, difficulty: float) -> Problem:
        speed, minutes, distance = self._pick_consistent_trip(difficulty)
        prompt = (
            f"An aircraft travels at {speed} km/h for {minutes} minutes. "
            "How many kilometres does it travel?"
        )
        return Problem(prompt=prompt, answer=distance)

    def _time_from_distance_speed(self, difficulty: float) -> Problem:
        speed, minutes, distance = self._pick_consistent_trip(difficulty)
        prompt = (
            f"An aircraft travels {distance} km at {speed} km/h. "
            "How many minutes does it take?"
        )
        return Problem(prompt=prompt, answer=minutes)

    def _speed_from_distance_time(self, difficulty: float) -> Problem:
        speed, minutes, distance = self._pick_consistent_trip(difficulty)
        prompt = (
            f"An aircraft travels {distance} km in {minutes} minutes. "
            "What is its speed in km/h?"
        )
        return Problem(prompt=prompt, answer=speed)

    def _pick_consistent_trip(self, difficulty: float) -> tuple[int, int, int]:
        """Pick (speed, minutes, distance) with an exact integer distance."""

        minutes_choices_easy = [15, 20, 30, 40]
        minutes_choices_hard = [45, 50, 55, 60]
        minutes_choices = minutes_choices_easy if difficulty < 0.5 else minutes_choices_hard

        # Re-roll until distance is an integer.
        for _ in range(100):
            speed = self._pick_speed(difficulty)
            minutes = minutes_choices[self._rng.randint(0, len(minutes_choices) - 1)]
            if (speed * minutes) % 60 == 0:
                distance = (speed * minutes) // 60
                return speed, minutes, distance

        # Fallback: guarantee divisibility by using 30 minutes.
        speed = self._pick_speed(difficulty)
        minutes = 30
        distance = (speed * minutes) // 60
        return speed, minutes, distance

    def _pick_speed(self, difficulty: float) -> int:
        # Keep speed multiples of 10 for mental math.
        lo = 120
        hi = lerp_int(240, 600, difficulty)
        base = self._rng.randint(lo // 10, hi // 10)
        return base * 10


def build_math_reasoning_test(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    config: MathReasoningConfig | None = None,
) -> TimedTextInputTest:
    """Factory for the Mathematics Reasoning test session."""

    cfg = config or MathReasoningConfig()

    instructions = [
        "Mathematics Reasoning (Time/Speed/Distance)",
        "",
        "Read the word problem and enter the numeric answer.",
        "These are time/speed/distance problems.",
        "",
        "Controls:",
        "- Type your answer",
        "- Press Enter to submit",
        "",
        "You will get a short practice, then a timed scored block.",
    ]

    generator = MathReasoningGenerator(seed=seed)

    return TimedTextInputTest(
        title="Mathematics Reasoning",
        instructions=instructions,
        generator=generator,
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        practice_questions=cfg.practice_questions,
        scored_duration_s=cfg.scored_duration_s,
    )