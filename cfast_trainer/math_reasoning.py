from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .clock import Clock
from .cognitive_core import Problem, SeededRng, TimedTextInputTest, clamp01, lerp_int


MR_DISTANCE_FROM_SPEED_TIME = "distance_from_speed_time"
MR_TIME_FROM_DISTANCE_SPEED = "time_from_distance_speed"
MR_SPEED_FROM_DISTANCE_TIME = "speed_from_distance_time"
MR_FUEL_REMAINING = "fuel_remaining"
MR_FUEL_ENDURANCE = "fuel_endurance"
MR_RESERVE_MARGIN = "reserve_margin"
MR_PERCENTAGE_CHANGE = "percentage_change"
MR_AVERAGE_SPEED_TWO_LEGS = "average_speed_two_legs"
MR_RATE_SCALING = "rate_scaling"

MR_ALL_DOMAIN_KEYS: tuple[str, ...] = (
    MR_DISTANCE_FROM_SPEED_TIME,
    MR_TIME_FROM_DISTANCE_SPEED,
    MR_SPEED_FROM_DISTANCE_TIME,
    MR_FUEL_REMAINING,
    MR_FUEL_ENDURANCE,
    MR_RESERVE_MARGIN,
    MR_PERCENTAGE_CHANGE,
    MR_AVERAGE_SPEED_TWO_LEGS,
    MR_RATE_SCALING,
)

MR_MOTION_FUEL_DOMAIN_KEYS: tuple[str, ...] = (
    MR_DISTANCE_FROM_SPEED_TIME,
    MR_TIME_FROM_DISTANCE_SPEED,
    MR_SPEED_FROM_DISTANCE_TIME,
    MR_FUEL_REMAINING,
    MR_FUEL_ENDURANCE,
    MR_RESERVE_MARGIN,
)

MR_REMAINING_DOMAIN_KEYS: tuple[str, ...] = (
    MR_RESERVE_MARGIN,
    MR_PERCENTAGE_CHANGE,
    MR_AVERAGE_SPEED_TWO_LEGS,
    MR_RATE_SCALING,
)


@dataclass(frozen=True, slots=True)
class MathReasoningOption:
    code: int
    text: str
    value: int


@dataclass(frozen=True, slots=True)
class MathReasoningFact:
    label: str
    value: int
    unit: str = ""

    @property
    def display_value(self) -> str:
        return f"{self.value} {self.unit}".strip()


@dataclass(frozen=True, slots=True)
class MathReasoningScenarioSpec:
    domain_key: str
    domain: str
    stem: str
    correct_value: int
    unit: str
    distractors: tuple[int, ...]
    facts: tuple[MathReasoningFact, ...]
    solution_label: str
    display_answer_text: str


@dataclass(frozen=True, slots=True)
class MathReasoningPayload:
    domain: str
    stem: str
    options: tuple[MathReasoningOption, ...]
    correct_code: int
    correct_value: int
    domain_key: str = ""
    display_answer_text: str = ""


@dataclass(frozen=True, slots=True)
class MathReasoningTrainingPayload:
    domain: str
    stem: str
    response_label: str
    answer_unit_label: str
    input_digits: int = 8
    display_answer_text: str = ""
    domain_key: str = ""
    base_cap_s: float | None = None


@dataclass(frozen=True, slots=True)
class MathReasoningConfig:
    scored_duration_s: float = 720.0  # training-friendly default
    practice_questions: int = 3


class MathReasoningGenerator:
    """Deterministic generator for mathematics reasoning scenarios and MC items."""

    def __init__(
        self,
        *,
        seed: int,
        allowed_domain_keys: Sequence[str] | None = None,
    ) -> None:
        self._rng = SeededRng(seed)
        if allowed_domain_keys is None:
            self._allowed_domain_keys = MR_ALL_DOMAIN_KEYS
        else:
            ordered = tuple(
                key for key in dict.fromkeys(str(item).strip().lower() for item in allowed_domain_keys) if key
            )
            if not ordered:
                raise ValueError("allowed_domain_keys must not be empty")
            invalid = tuple(key for key in ordered if key not in MR_ALL_DOMAIN_KEYS)
            if invalid:
                raise ValueError(f"Unsupported Math Reasoning domain keys: {invalid}")
            self._allowed_domain_keys = ordered

    def next_problem(self, *, difficulty: float) -> Problem:
        spec = self.next_scenario_spec(difficulty=difficulty)
        return self.problem_from_spec(spec)

    def next_scenario_spec(
        self,
        *,
        difficulty: float,
        domain_key: str | None = None,
        include_filler: bool = False,
    ) -> MathReasoningScenarioSpec:
        difficulty = clamp01(difficulty)
        token = self._pick_domain_key(domain_key=domain_key)
        if token == MR_DISTANCE_FROM_SPEED_TIME:
            return self._distance_from_speed_time_spec(difficulty, include_filler=include_filler)
        if token == MR_TIME_FROM_DISTANCE_SPEED:
            return self._time_from_distance_speed_spec(difficulty, include_filler=include_filler)
        if token == MR_SPEED_FROM_DISTANCE_TIME:
            return self._speed_from_distance_time_spec(difficulty, include_filler=include_filler)
        if token == MR_FUEL_REMAINING:
            return self._fuel_remaining_spec(difficulty, include_filler=include_filler)
        if token == MR_FUEL_ENDURANCE:
            return self._fuel_endurance_spec(difficulty, include_filler=include_filler)
        if token == MR_RESERVE_MARGIN:
            return self._reserve_margin_spec(difficulty, include_filler=include_filler)
        if token == MR_PERCENTAGE_CHANGE:
            return self._percentage_change_spec(difficulty, include_filler=include_filler)
        if token == MR_AVERAGE_SPEED_TWO_LEGS:
            return self._average_speed_two_legs_spec(difficulty, include_filler=include_filler)
        return self._rate_scaling_spec(difficulty, include_filler=include_filler)

    def problem_from_spec(self, spec: MathReasoningScenarioSpec) -> Problem:
        values: list[int] = [int(spec.correct_value)]
        for value in spec.distractors:
            item = int(value)
            if item < 0 or item in values:
                continue
            values.append(item)
            if len(values) == 5:
                break

        jitter = max(2, abs(int(spec.correct_value)) // 6)
        while len(values) < 5:
            direction = -1 if self._rng.randint(0, 1) == 0 else 1
            scale = self._rng.randint(1, 3)
            candidate = int(spec.correct_value) + direction * jitter * scale
            if candidate < 0 or candidate in values:
                continue
            values.append(candidate)

        order = self._rng.sample([0, 1, 2, 3, 4], k=5)
        shuffled = [values[idx] for idx in order]
        options = tuple(
            MathReasoningOption(
                code=index + 1,
                text=self._format_option(value=value, unit=spec.unit),
                value=value,
            )
            for index, value in enumerate(shuffled)
        )
        correct_code = next(opt.code for opt in options if opt.value == int(spec.correct_value))

        prompt_lines = [spec.stem, ""]
        for option in options:
            prompt_lines.append(f"{option.code}) {option.text}")
        prompt = "\n".join(prompt_lines)

        payload = MathReasoningPayload(
            domain=spec.domain,
            stem=spec.stem,
            options=options,
            correct_code=correct_code,
            correct_value=int(spec.correct_value),
            domain_key=spec.domain_key,
            display_answer_text=spec.display_answer_text,
        )
        return Problem(prompt=prompt, answer=correct_code, payload=payload)

    def _pick_domain_key(self, *, domain_key: str | None) -> str:
        if domain_key is not None:
            token = str(domain_key).strip().lower()
            if token not in self._allowed_domain_keys:
                raise ValueError(f"Unsupported Math Reasoning domain key: {domain_key}")
            return token
        if len(self._allowed_domain_keys) == 1:
            return self._allowed_domain_keys[0]
        return str(self._rng.choice(self._allowed_domain_keys))

    def _distance_from_speed_time_spec(
        self,
        difficulty: float,
        *,
        include_filler: bool,
    ) -> MathReasoningScenarioSpec:
        speed, minutes, distance = self._pick_consistent_trip(difficulty)
        stem = (
            f"An aircraft travels at {speed} km/h for {minutes} minutes. "
            "How many kilometres does it travel?"
        )
        distractors = (
            speed * minutes,
            speed + minutes,
            max(1, distance - (speed // 6)),
        )
        facts = (
            MathReasoningFact("Speed", speed, "km/h"),
            MathReasoningFact("Time", minutes, "minutes"),
        )
        stem = self._append_filler(
            stem,
            include_filler=include_filler,
            difficulty=difficulty,
            context="route",
        )
        return self._scenario_spec(
            domain_key=MR_DISTANCE_FROM_SPEED_TIME,
            domain="Time-Speed-Distance",
            stem=stem,
            correct_value=distance,
            unit="km",
            distractors=distractors,
            facts=facts,
            solution_label="Distance travelled",
        )

    def _time_from_distance_speed_spec(
        self,
        difficulty: float,
        *,
        include_filler: bool,
    ) -> MathReasoningScenarioSpec:
        speed, minutes, distance = self._pick_consistent_trip(difficulty)
        stem = f"An aircraft travels {distance} km at {speed} km/h. How many minutes does it take?"
        hours = max(1, minutes // 60)
        distractors = (
            hours,
            minutes + 30,
            max(1, (distance * 10) // max(speed, 1)),
        )
        facts = (
            MathReasoningFact("Distance", distance, "km"),
            MathReasoningFact("Speed", speed, "km/h"),
        )
        stem = self._append_filler(
            stem,
            include_filler=include_filler,
            difficulty=difficulty,
            context="route",
        )
        return self._scenario_spec(
            domain_key=MR_TIME_FROM_DISTANCE_SPEED,
            domain="Time-Speed-Distance",
            stem=stem,
            correct_value=minutes,
            unit="minutes",
            distractors=distractors,
            facts=facts,
            solution_label="Travel time",
        )

    def _speed_from_distance_time_spec(
        self,
        difficulty: float,
        *,
        include_filler: bool,
    ) -> MathReasoningScenarioSpec:
        speed, minutes, distance = self._pick_consistent_trip(difficulty)
        stem = f"An aircraft travels {distance} km in {minutes} minutes. What is its speed in km/h?"
        distractors = (
            max(1, distance // max(minutes, 1)),
            max(10, speed + 60),
            max(10, speed - 60),
        )
        facts = (
            MathReasoningFact("Distance", distance, "km"),
            MathReasoningFact("Time", minutes, "minutes"),
        )
        stem = self._append_filler(
            stem,
            include_filler=include_filler,
            difficulty=difficulty,
            context="route",
        )
        return self._scenario_spec(
            domain_key=MR_SPEED_FROM_DISTANCE_TIME,
            domain="Time-Speed-Distance",
            stem=stem,
            correct_value=speed,
            unit="km/h",
            distractors=distractors,
            facts=facts,
            solution_label="Speed",
        )

    def _fuel_remaining_spec(
        self,
        difficulty: float,
        *,
        include_filler: bool,
    ) -> MathReasoningScenarioSpec:
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
            start_fuel + consumed,
            consumed,
            max(0, start_fuel - burn_lpm),
        )
        facts = (
            MathReasoningFact("Start fuel", start_fuel, "L"),
            MathReasoningFact("Burn rate", burn_lpm, "L/min"),
            MathReasoningFact("Duration", duration_min, "minutes"),
        )
        stem = self._append_filler(
            stem,
            include_filler=include_filler,
            difficulty=difficulty,
            context="fuel",
        )
        return self._scenario_spec(
            domain_key=MR_FUEL_REMAINING,
            domain="Fuel Planning",
            stem=stem,
            correct_value=remaining,
            unit="L",
            distractors=distractors,
            facts=facts,
            solution_label="Fuel remaining",
        )

    def _percentage_change_spec(
        self,
        difficulty: float,
        *,
        include_filler: bool,
    ) -> MathReasoningScenarioSpec:
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
                delta,
                base + pct,
                max(1, base - delta),
            )
        else:
            correct = base - delta
            stem = (
                f"After optimization, a task that took {base} minutes is reduced by "
                f"{pct}%. What is the new time?"
            )
            distractors = (
                delta,
                max(1, base - pct),
                base + delta,
            )
        facts = (
            MathReasoningFact("Original time", base, "minutes"),
            MathReasoningFact("Percent change", pct, "%"),
        )
        stem = self._append_filler(
            stem,
            include_filler=include_filler,
            difficulty=difficulty,
            context="timing",
        )
        return self._scenario_spec(
            domain_key=MR_PERCENTAGE_CHANGE,
            domain="Percentages",
            stem=stem,
            correct_value=correct,
            unit="minutes",
            distractors=distractors,
            facts=facts,
            solution_label="New time",
        )

    def _fuel_endurance_spec(
        self,
        difficulty: float,
        *,
        include_filler: bool,
    ) -> MathReasoningScenarioSpec:
        burn_lo = lerp_int(10, 18, difficulty)
        burn_hi = lerp_int(24, 42, difficulty)
        burn_per_min = self._rng.randint(burn_lo, burn_hi)
        endurance_minutes = self._rng.choice((30, 40, 45, 50, 60, 75, 90, 105))
        start_fuel = burn_per_min * endurance_minutes
        stem = (
            f"An aircraft carries {start_fuel} L of usable fuel and burns {burn_per_min} L/min. "
            "How many minutes of endurance does it have?"
        )
        distractors = (
            max(1, burn_per_min + endurance_minutes),
            max(1, start_fuel - burn_per_min),
            max(1, endurance_minutes + 15),
        )
        facts = (
            MathReasoningFact("Usable fuel", start_fuel, "L"),
            MathReasoningFact("Burn rate", burn_per_min, "L/min"),
        )
        stem = self._append_filler(
            stem,
            include_filler=include_filler,
            difficulty=difficulty,
            context="fuel",
        )
        return self._scenario_spec(
            domain_key=MR_FUEL_ENDURANCE,
            domain="Fuel Planning",
            stem=stem,
            correct_value=endurance_minutes,
            unit="minutes",
            distractors=distractors,
            facts=facts,
            solution_label="Fuel endurance",
        )

    def _reserve_margin_spec(
        self,
        difficulty: float,
        *,
        include_filler: bool,
    ) -> MathReasoningScenarioSpec:
        trip_need = self._rng.choice((420, 480, 540, 600, 660, 720, 780))
        reserve = self._rng.choice((60, 90, 120, 150, 180))
        uplift = self._rng.choice((0, 40, 60, 80, 100, 120, 160))
        start_fuel = trip_need + reserve + uplift
        correct_margin = start_fuel - trip_need - reserve
        stem = (
            f"A sortie needs {trip_need} L for the trip and must keep {reserve} L in reserve. "
            f"It launches with {start_fuel} L. How much margin remains after protecting reserve?"
        )
        distractors = (
            max(0, start_fuel - trip_need),
            max(0, reserve - uplift),
            max(0, trip_need - reserve),
        )
        facts = (
            MathReasoningFact("Trip fuel", trip_need, "L"),
            MathReasoningFact("Reserve", reserve, "L"),
            MathReasoningFact("Start fuel", start_fuel, "L"),
        )
        stem = self._append_filler(
            stem,
            include_filler=include_filler,
            difficulty=difficulty,
            context="reserve",
        )
        return self._scenario_spec(
            domain_key=MR_RESERVE_MARGIN,
            domain="Fuel Planning",
            stem=stem,
            correct_value=correct_margin,
            unit="L",
            distractors=distractors,
            facts=facts,
            solution_label="Reserve margin",
        )

    def _average_speed_two_legs_spec(
        self,
        difficulty: float,
        *,
        include_filler: bool,
    ) -> MathReasoningScenarioSpec:
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
            speed1 + speed2,
            abs(speed1 - speed2),
            total_distance,
        )
        facts = (
            MathReasoningFact("Leg 1 distance", dist1, "km"),
            MathReasoningFact("Leg 2 distance", dist2, "km"),
            MathReasoningFact("Leg time", leg_minutes, "minutes"),
        )
        stem = self._append_filler(
            stem,
            include_filler=include_filler,
            difficulty=difficulty,
            context="legs",
        )
        return self._scenario_spec(
            domain_key=MR_AVERAGE_SPEED_TWO_LEGS,
            domain="Averages",
            stem=stem,
            correct_value=avg_speed,
            unit="km/h",
            distractors=distractors,
            facts=facts,
            solution_label="Average speed",
        )

    def _rate_scaling_spec(
        self,
        difficulty: float,
        *,
        include_filler: bool,
    ) -> MathReasoningScenarioSpec:
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
            base_items * target_minutes,
            base_items + target_minutes,
            max(1, base_items // max(1, target_minutes)),
        )
        facts = (
            MathReasoningFact("Base items", base_items, "items"),
            MathReasoningFact("Base time", base_minutes, "minutes"),
            MathReasoningFact("Target time", target_minutes, "minutes"),
        )
        stem = self._append_filler(
            stem,
            include_filler=include_filler,
            difficulty=difficulty,
            context="rate",
        )
        return self._scenario_spec(
            domain_key=MR_RATE_SCALING,
            domain="Rates",
            stem=stem,
            correct_value=correct,
            unit="items",
            distractors=distractors,
            facts=facts,
            solution_label="Scaled output",
        )

    def _scenario_spec(
        self,
        *,
        domain_key: str,
        domain: str,
        stem: str,
        correct_value: int,
        unit: str,
        distractors: tuple[int, ...],
        facts: tuple[MathReasoningFact, ...],
        solution_label: str,
    ) -> MathReasoningScenarioSpec:
        display = f"{int(correct_value)} {unit}".strip()
        return MathReasoningScenarioSpec(
            domain_key=domain_key,
            domain=domain,
            stem=stem,
            correct_value=int(correct_value),
            unit=unit,
            distractors=tuple(int(item) for item in distractors),
            facts=facts,
            solution_label=solution_label,
            display_answer_text=display,
        )

    def _pick_consistent_trip(self, difficulty: float) -> tuple[int, int, int]:
        minutes_choices_easy = [15, 20, 30, 40, 45, 60]
        minutes_choices_hard = [25, 35, 45, 50, 55, 60, 75, 90]
        minutes_choices = minutes_choices_easy if difficulty < 0.5 else minutes_choices_hard

        for _ in range(100):
            speed = self._pick_speed(difficulty)
            minutes = minutes_choices[self._rng.randint(0, len(minutes_choices) - 1)]
            if (speed * minutes) % 60 == 0:
                distance = (speed * minutes) // 60
                return speed, minutes, distance

        speed = self._pick_speed(difficulty)
        minutes = 30
        distance = (speed * minutes) // 60
        return speed, minutes, distance

    def _pick_speed(self, difficulty: float) -> int:
        lo = lerp_int(120, 160, difficulty)
        hi = lerp_int(240, 600, difficulty)
        base = self._rng.randint(lo // 10, hi // 10)
        return base * 10

    def _append_filler(
        self,
        stem: str,
        *,
        include_filler: bool,
        difficulty: float,
        context: str,
    ) -> str:
        if not include_filler:
            return stem
        filler_count = 1 if difficulty < 0.45 else 2 if difficulty < 0.8 else 3
        sentences: list[str] = []
        for _ in range(filler_count):
            token = self._rng.choice(("window", "stand", "callsign", "reserve", "checkpoint"))
            if token == "window":
                minutes = self._rng.randint(6, 28)
                sentences.append(f"Ignore the {minutes}-minute admin window before the task starts.")
            elif token == "stand":
                stand = self._rng.randint(2, 18)
                bay = self._rng.randint(1, 9)
                sentences.append(f"The aircraft is parked at stand {stand} bay {bay}, which does not affect the answer.")
            elif token == "callsign":
                callsign = f"{context[:1].upper()}{self._rng.randint(10, 98)}"
                sentences.append(f"Call sign {callsign} is just a reference tag.")
            elif token == "reserve":
                reserve = self._rng.randint(10, 70)
                sentences.append(f"Do not use the {reserve}-unit reserve note unless the question asks for it.")
            else:
                checkpoints = self._rng.randint(2, 6)
                sentences.append(f"The route briefing mentions {checkpoints} checkpoints, but that count is filler.")
        return f"{stem} {' '.join(sentences)}"

    def _format_option(self, *, value: int, unit: str) -> str:
        return f"{value} {unit}".strip()


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
        "- Press A, S, D, F, or G to choose an option",
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
