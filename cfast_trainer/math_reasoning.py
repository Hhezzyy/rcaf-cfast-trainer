from __future__ import annotations

from dataclasses import dataclass

from .clock import Clock
from .cognitive_core import Problem, SeededRng, TimedTextInputTest, clamp01, lerp_int


@dataclass(frozen=True, slots=True)
class MathReasoningOption:
    code: int
    text: str
    value: int


@dataclass(frozen=True, slots=True)
class MathReasoningPayload:
    domain: str
    stem: str
    options: tuple[MathReasoningOption, ...]
    correct_code: int
    correct_value: int


@dataclass(frozen=True, slots=True)
class MathReasoningConfig:
    scored_duration_s: float = 720.0  # training-friendly default
    practice_questions: int = 3


class MathReasoningGenerator:
    """Deterministic generator for multiple-choice mathematics reasoning items."""

    def __init__(self, *, seed: int) -> None:
        self._rng = SeededRng(seed)

    def next_problem(self, *, difficulty: float) -> Problem:
        difficulty = clamp01(difficulty)
        kind = self._rng.randint(0, 6)
        if kind == 0:
            return self._distance_from_speed_time(difficulty)
        if kind == 1:
            return self._time_from_distance_speed(difficulty)
        if kind == 2:
            return self._speed_from_distance_time(difficulty)
        if kind == 3:
            return self._fuel_remaining(difficulty)
        if kind == 4:
            return self._percentage_change(difficulty)
        if kind == 5:
            return self._average_speed_two_legs(difficulty)
        return self._rate_scaling(difficulty)

    def _distance_from_speed_time(self, difficulty: float) -> Problem:
        speed, minutes, distance = self._pick_consistent_trip(difficulty)
        stem = (
            f"An aircraft travels at {speed} km/h for {minutes} minutes. "
            "How many kilometres does it travel?"
        )
        distractors = (
            speed * minutes,  # forgot to convert minutes to hours
            speed + minutes,  # used addition instead of rate x time
            max(1, distance - (speed // 6)),  # arithmetic slip
        )
        return self._multiple_choice_problem(
            domain="Time-Speed-Distance",
            stem=stem,
            correct_value=distance,
            distractors=distractors,
            unit="km",
        )

    def _time_from_distance_speed(self, difficulty: float) -> Problem:
        speed, minutes, distance = self._pick_consistent_trip(difficulty)
        stem = f"An aircraft travels {distance} km at {speed} km/h. How many minutes does it take?"
        hours = max(1, minutes // 60)
        distractors = (
            hours,  # forgot to convert hours into minutes
            minutes + 30,  # one-step arithmetic overshoot
            max(1, (distance * 10) // max(speed, 1)),  # treated hourly rate like /10
        )
        return self._multiple_choice_problem(
            domain="Time-Speed-Distance",
            stem=stem,
            correct_value=minutes,
            distractors=distractors,
            unit="minutes",
        )

    def _speed_from_distance_time(self, difficulty: float) -> Problem:
        speed, minutes, distance = self._pick_consistent_trip(difficulty)
        stem = f"An aircraft travels {distance} km in {minutes} minutes. What is its speed in km/h?"
        distractors = (
            max(1, distance // max(minutes, 1)),  # forgot to convert minutes to hours
            max(10, speed + 60),  # overestimate
            max(10, speed - 60),  # underestimate
        )
        return self._multiple_choice_problem(
            domain="Time-Speed-Distance",
            stem=stem,
            correct_value=speed,
            distractors=distractors,
            unit="km/h",
        )

    def _fuel_remaining(self, difficulty: float) -> Problem:
        burn_lo = lerp_int(8, 12, difficulty)
        burn_hi = lerp_int(20, 34, difficulty)
        burn_lpm = self._rng.randint(burn_lo, burn_hi)

        minutes_lo = lerp_int(20, 35, difficulty)
        minutes_hi = lerp_int(70, 130, difficulty)
        duration_min = self._rng.randint(minutes_lo, minutes_hi)

        consumed = burn_lpm * duration_min
        reserve = self._rng.randint(120, 380)
        start_fuel = consumed + reserve
        remaining = start_fuel - consumed

        stem = (
            f"An aircraft starts with {start_fuel} L of fuel and burns {burn_lpm} L/min "
            f"for {duration_min} minutes. How much fuel remains?"
        )
        distractors = (
            start_fuel + consumed,  # added instead of subtracted
            consumed,  # gave consumed fuel instead of remaining fuel
            max(0, start_fuel - burn_lpm),  # applied burn rate for one minute only
        )
        return self._multiple_choice_problem(
            domain="Fuel Planning",
            stem=stem,
            correct_value=remaining,
            distractors=distractors,
            unit="L",
        )

    def _percentage_change(self, difficulty: float) -> Problem:
        increase = self._rng.randint(0, 1) == 1
        base = self._rng.choice([60, 80, 100, 120, 140, 160, 180, 200])
        pct_pool_easy = [10, 20, 25, 50]
        pct_pool_hard = [10, 15, 20, 25, 30, 40, 50]
        pct_pool = pct_pool_easy if difficulty < 0.45 else pct_pool_hard

        pct = self._rng.choice(pct_pool)
        for _ in range(20):
            if (base * pct) % 100 == 0:
                break
            pct = self._rng.choice(pct_pool)

        delta = (base * pct) // 100
        if increase:
            correct = base + delta
            stem = (
                f"A flight segment is planned for {base} minutes. Due to headwind, "
                f"time increases by {pct}%. What is the new time?"
            )
            distractors = (
                delta,  # percent amount only
                base + pct,  # added percentage points directly
                max(1, base - delta),  # wrong direction (decrease)
            )
        else:
            correct = base - delta
            stem = (
                f"After optimization, a task that took {base} minutes is reduced by "
                f"{pct}%. What is the new time?"
            )
            distractors = (
                delta,  # percent amount only
                max(1, base - pct),  # subtracted percentage points directly
                base + delta,  # wrong direction (increase)
            )

        return self._multiple_choice_problem(
            domain="Percentages",
            stem=stem,
            correct_value=correct,
            distractors=distractors,
            unit="minutes",
        )

    def _average_speed_two_legs(self, difficulty: float) -> Problem:
        speed_lo = lerp_int(140, 180, difficulty)
        speed_hi = lerp_int(320, 520, difficulty)
        speed1 = self._rng.randint(speed_lo // 20, speed_hi // 20) * 20
        speed2 = self._rng.randint(speed_lo // 20, speed_hi // 20) * 20
        leg_minutes = self._rng.choice([30, 45, 60])

        dist1 = (speed1 * leg_minutes) // 60
        dist2 = (speed2 * leg_minutes) // 60
        total_distance = dist1 + dist2
        total_minutes = leg_minutes * 2
        avg_speed = (total_distance * 60) // max(1, total_minutes)

        stem = (
            f"Leg 1: {dist1} km in {leg_minutes} min. "
            f"Leg 2: {dist2} km in {leg_minutes} min. "
            "What is the average speed for both legs in km/h?"
        )
        distractors = (
            speed1 + speed2,  # forgot to average
            abs(speed1 - speed2),  # confusion with difference
            total_distance,  # reported distance instead of speed
        )
        return self._multiple_choice_problem(
            domain="Averages",
            stem=stem,
            correct_value=avg_speed,
            distractors=distractors,
            unit="km/h",
        )

    def _rate_scaling(self, difficulty: float) -> Problem:
        rate_lo = lerp_int(4, 6, difficulty)
        rate_hi = lerp_int(12, 18, difficulty)
        rate_per_min = self._rng.randint(rate_lo, rate_hi)

        base_minutes = self._rng.randint(3, 8)
        target_minutes = base_minutes + self._rng.randint(2, 7)
        base_items = rate_per_min * base_minutes
        correct = rate_per_min * target_minutes

        stem = (
            f"A crew checks {base_items} items in {base_minutes} minutes at a constant pace. "
            f"How many items can it check in {target_minutes} minutes?"
        )
        distractors = (
            base_items * target_minutes,  # scaled without dividing by baseline time
            base_items + target_minutes,  # added instead of using proportional rate
            max(1, base_items // max(1, target_minutes)),  # inverted the ratio
        )
        return self._multiple_choice_problem(
            domain="Rates",
            stem=stem,
            correct_value=correct,
            distractors=distractors,
            unit="items",
        )

    def _pick_consistent_trip(self, difficulty: float) -> tuple[int, int, int]:
        """Pick (speed, minutes, distance) with an exact integer distance."""

        minutes_choices_easy = [15, 20, 30, 40, 45, 60]
        minutes_choices_hard = [25, 35, 45, 50, 55, 60, 75, 90]
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
        # Keep speed multiples of 10 for mental math while scaling range with difficulty.
        lo = lerp_int(120, 160, difficulty)
        hi = lerp_int(240, 600, difficulty)
        base = self._rng.randint(lo // 10, hi // 10)
        return base * 10

    def _multiple_choice_problem(
        self,
        *,
        domain: str,
        stem: str,
        correct_value: int,
        distractors: tuple[int, int, int],
        unit: str,
    ) -> Problem:
        values: list[int] = [int(correct_value)]
        for value in distractors:
            item = int(value)
            if item < 0 or item in values:
                continue
            values.append(item)
            if len(values) == 4:
                break

        jitter = max(2, abs(int(correct_value)) // 6)
        while len(values) < 4:
            direction = -1 if self._rng.randint(0, 1) == 0 else 1
            scale = self._rng.randint(1, 3)
            candidate = int(correct_value) + direction * jitter * scale
            if candidate < 0 or candidate in values:
                continue
            values.append(candidate)

        order = self._rng.sample([0, 1, 2, 3], k=4)
        shuffled = [values[idx] for idx in order]

        options = tuple(
            MathReasoningOption(
                code=index + 1,
                text=self._format_option(value=value, unit=unit),
                value=value,
            )
            for index, value in enumerate(shuffled)
        )
        correct_code = next(opt.code for opt in options if opt.value == int(correct_value))

        prompt_lines = [stem, ""]
        for option in options:
            prompt_lines.append(f"{option.code}) {option.text}")
        prompt = "\n".join(prompt_lines)

        payload = MathReasoningPayload(
            domain=domain,
            stem=stem,
            options=options,
            correct_code=correct_code,
            correct_value=int(correct_value),
        )
        return Problem(prompt=prompt, answer=correct_code, payload=payload)

    def _format_option(self, *, value: int, unit: str) -> str:
        if unit == "":
            return str(value)
        return f"{value} {unit}"


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
        "Mathematics Reasoning (Multiple Choice)",
        "",
        "Read each word problem and select the best answer.",
        "Questions mix aviation-style rates, percentages, averages, and planning math.",
        "",
        "Controls:",
        "- Press 1, 2, 3, or 4 to choose an option",
        "- Press Enter to submit",
        "",
        "You will get a short practice, then a timed scored block.",
        "Once the timed block starts, continue until completion.",
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
