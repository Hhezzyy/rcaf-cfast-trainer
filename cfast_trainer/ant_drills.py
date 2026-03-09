from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Protocol, cast

from .airborne_numerical import (
    AirborneNumericalGenerator,
    AirborneScenario,
    AirborneScorer,
    build_ant_airborne_difficulty_profile,
)
from .clock import Clock
from .cognitive_core import AnswerScorer, Phase, Problem, QuestionEvent, SeededRng, TestSnapshot
from .lookup_retain import LookupRetainPromptSpec, LookupRetainScorer, expected_digits_for_problem


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _difficulty_to_level(difficulty: float) -> int:
    return max(1, min(10, int(round(_clamp01(difficulty) * 9.0)) + 1))


def _level_to_difficulty(level: int) -> float:
    clamped = max(1, min(10, int(level)))
    return float(clamped - 1) / 9.0


def _normalize_skin(skin: str) -> str:
    normalized = str(skin).strip().lower() or "mixed"
    if normalized not in {"mixed", "abstract", "aviation"}:
        return "mixed"
    return normalized


def _hhmm_to_minutes(hhmm: int) -> int:
    value = max(0, int(hhmm))
    return ((value // 100) * 60) + (value % 100)


def _minutes_to_hhmm(minutes: int) -> int:
    total = int(minutes) % (24 * 60)
    return ((total // 60) * 100) + (total % 60)


class AntDrillMode(str, Enum):
    BUILD = "build"
    TEMPO = "tempo"
    STRESS = "stress"


class AntProblemGenerator(Protocol):
    def next_problem(self, *, difficulty: float) -> Problem: ...


@dataclass(frozen=True, slots=True)
class AntDrillModeProfile:
    label: str
    practice_questions: int
    scored_duration_s: float
    cap_scale: float
    immediate_feedback: bool
    note: str


@dataclass(frozen=True, slots=True)
class AntAdaptiveModeProfile:
    window_size: int
    accuracy_low: float
    accuracy_high: float
    fixation_ceiling: float
    raise_streak: int


@dataclass(frozen=True, slots=True)
class AntAdaptiveDifficultyConfig:
    enabled: bool = True
    window_size: int | None = None


@dataclass(frozen=True, slots=True)
class AntSnapFactsSprintConfig:
    skin: str = "mixed"
    practice_questions: int | None = None
    scored_duration_s: float | None = None
    adaptive: AntAdaptiveDifficultyConfig = field(default_factory=AntAdaptiveDifficultyConfig)


@dataclass(frozen=True, slots=True)
class AntTimeFlipConfig:
    skin: str = "mixed"
    practice_questions: int | None = None
    scored_duration_s: float | None = None
    adaptive: AntAdaptiveDifficultyConfig = field(default_factory=AntAdaptiveDifficultyConfig)


@dataclass(frozen=True, slots=True)
class AntMixedTempoSetConfig:
    skin: str = "mixed"
    practice_questions: int | None = None
    scored_duration_s: float | None = None
    adaptive: AntAdaptiveDifficultyConfig = field(default_factory=AntAdaptiveDifficultyConfig)


@dataclass(frozen=True, slots=True)
class AntRouteTimeSolveConfig:
    practice_questions: int | None = None
    scored_duration_s: float | None = None
    adaptive: AntAdaptiveDifficultyConfig = field(default_factory=AntAdaptiveDifficultyConfig)


@dataclass(frozen=True, slots=True)
class AntFuelBurnSolveConfig:
    practice_questions: int | None = None
    scored_duration_s: float | None = None
    adaptive: AntAdaptiveDifficultyConfig = field(default_factory=AntAdaptiveDifficultyConfig)


@dataclass(frozen=True, slots=True)
class AntEnduranceSolveConfig:
    practice_questions: int | None = None
    scored_duration_s: float | None = None
    adaptive: AntAdaptiveDifficultyConfig = field(default_factory=AntAdaptiveDifficultyConfig)


@dataclass(frozen=True, slots=True)
class AntDistanceScanConfig:
    practice_questions: int | None = None
    scored_duration_s: float | None = None
    adaptive: AntAdaptiveDifficultyConfig = field(default_factory=AntAdaptiveDifficultyConfig)


@dataclass(frozen=True, slots=True)
class AntPayloadReferenceConfig:
    practice_questions: int | None = None
    scored_duration_s: float | None = None
    adaptive: AntAdaptiveDifficultyConfig = field(default_factory=AntAdaptiveDifficultyConfig)


@dataclass(frozen=True, slots=True)
class AntInfoGrabberConfig:
    practice_questions: int | None = None
    scored_duration_s: float | None = None
    adaptive: AntAdaptiveDifficultyConfig = field(default_factory=AntAdaptiveDifficultyConfig)


@dataclass(frozen=True, slots=True)
class AntProblemRuntimeMeta:
    family: str
    base_cap_s: float | None = None


@dataclass(frozen=True, slots=True)
class AntDifficultyChange:
    after_attempt: int
    old_level: int
    new_level: int
    reason: str


@dataclass(frozen=True, slots=True)
class AntDrillAttemptSummary:
    attempted: int
    correct: int
    accuracy: float
    duration_s: float
    throughput_per_min: float
    mean_response_time_s: float | None
    total_score: float = 0.0
    max_score: float = 0.0
    score_ratio: float = 0.0
    correct_per_min: float = 0.0
    timeouts: int = 0
    fixation_rate: float = 0.0
    max_timeout_streak: int = 0
    mode: str = ""
    difficulty_level: int = 1
    difficulty_level_start: int = 1
    difficulty_level_end: int = 1
    difficulty_change_count: int = 0
    adaptive_enabled: bool = False
    adaptive_window_size: int = 0


ANT_DRILL_MODE_PROFILES: dict[AntDrillMode, AntDrillModeProfile] = {
    AntDrillMode.BUILD: AntDrillModeProfile(
        label="Build",
        practice_questions=6,
        scored_duration_s=180.0,
        cap_scale=1.25,
        immediate_feedback=True,
        note="Longer caps and immediate correction. Stay clean and keep moving.",
    ),
    AntDrillMode.TEMPO: AntDrillModeProfile(
        label="Tempo",
        practice_questions=4,
        scored_duration_s=150.0,
        cap_scale=1.0,
        immediate_feedback=False,
        note="Caps tighten. Throughput matters more than polishing every item.",
    ),
    AntDrillMode.STRESS: AntDrillModeProfile(
        label="Stress",
        practice_questions=3,
        scored_duration_s=180.0,
        cap_scale=0.85,
        immediate_feedback=False,
        note="Monotony and pressure. Short dips are fine if you recover fast.",
    ),
}


ANT_ADAPTIVE_MODE_PROFILES: dict[AntDrillMode, AntAdaptiveModeProfile] = {
    AntDrillMode.BUILD: AntAdaptiveModeProfile(
        window_size=20,
        accuracy_low=0.90,
        accuracy_high=0.95,
        fixation_ceiling=0.05,
        raise_streak=12,
    ),
    AntDrillMode.TEMPO: AntAdaptiveModeProfile(
        window_size=15,
        accuracy_low=0.80,
        accuracy_high=0.90,
        fixation_ceiling=0.10,
        raise_streak=0,
    ),
    AntDrillMode.STRESS: AntAdaptiveModeProfile(
        window_size=12,
        accuracy_low=0.75,
        accuracy_high=0.85,
        fixation_ceiling=0.15,
        raise_streak=0,
    ),
}


class AntSnapFactsSprintGenerator:
    def __init__(self, *, seed: int, skin: str = "mixed") -> None:
        self._skin = _normalize_skin(skin)
        self._rng = SeededRng(seed)

    def next_problem(self, *, difficulty: float) -> Problem:
        level = _difficulty_to_level(difficulty)
        family = self._pick_family(level)
        if family == "add_easy":
            a = self._rng.randint(2, 9)
            b = self._rng.randint(2, 9)
            return Problem(prompt=self._format_prompt(a=a, op="+", b=b), answer=a + b)
        if family == "sub_easy":
            a = self._rng.randint(6, 18)
            b = self._rng.randint(1, min(9, a - 1))
            return Problem(prompt=self._format_prompt(a=a, op="-", b=b), answer=a - b)
        if family == "add_mid":
            a = self._rng.randint(8, 49)
            b = self._rng.randint(6, 39)
            return Problem(prompt=self._format_prompt(a=a, op="+", b=b), answer=a + b)
        if family == "sub_mid":
            a = self._rng.randint(25, 99)
            b = self._rng.randint(8, min(59, a - 1))
            return Problem(prompt=self._format_prompt(a=a, op="-", b=b), answer=a - b)
        if family == "mul_table":
            a = self._rng.randint(2, 12)
            b = self._rng.randint(2, 12)
            return Problem(prompt=self._format_prompt(a=a, op="x", b=b), answer=a * b)
        if family == "mul_wide":
            a = self._rng.randint(6, 18)
            b = self._rng.randint(4, 12)
            return Problem(prompt=self._format_prompt(a=a, op="x", b=b), answer=a * b)
        if family == "div_table":
            divisor = self._rng.randint(2, 12)
            quotient = self._rng.randint(2, 12)
            dividend = divisor * quotient
            return Problem(
                prompt=self._format_prompt(a=dividend, op="/", b=divisor),
                answer=quotient,
            )

        divisor = self._rng.randint(3, 18)
        quotient = self._rng.randint(4, 18)
        dividend = divisor * quotient
        return Problem(
            prompt=self._format_prompt(a=dividend, op="/", b=divisor),
            answer=quotient,
        )

    def _pick_family(self, level: int) -> str:
        if level <= 2:
            families = ("add_easy", "add_easy", "sub_easy", "sub_easy", "add_mid")
        elif level <= 4:
            families = ("add_mid", "sub_mid", "mul_table", "div_table", "mul_table")
        elif level <= 6:
            families = ("add_mid", "sub_mid", "mul_table", "div_table", "mul_wide")
        elif level <= 8:
            families = ("sub_mid", "mul_wide", "div_table", "div_wide", "mul_wide")
        else:
            families = ("add_mid", "sub_mid", "mul_wide", "div_wide", "div_wide")
        return self._rng.choice(families)

    def _pick_skin(self) -> str:
        if self._skin == "mixed":
            return self._rng.choice(("abstract", "aviation"))
        return self._skin

    def _format_prompt(self, *, a: int, op: str, b: int) -> str:
        expression = f"{a} {op} {b} ="
        skin = self._pick_skin()
        if skin == "abstract":
            return expression
        prefix = {
            "+": "Leg total",
            "-": "Fuel left",
            "x": "Formation count",
            "/": "Split check",
        }.get(op, "Quick check")
        return f"{prefix}: {expression}"


class AntTimeFlipGenerator:
    def __init__(self, *, seed: int, skin: str = "mixed") -> None:
        self._skin = _normalize_skin(skin)
        self._rng = SeededRng(seed)

    def next_problem(self, *, difficulty: float) -> Problem:
        level = _difficulty_to_level(difficulty)
        family = self._pick_family(level)
        if family == "hm_to_minutes":
            return self._hours_minutes_to_minutes(level)
        if family == "hhmm_add":
            return self._hhmm_shift(level=level, direction=1)
        if family == "hhmm_sub":
            return self._hhmm_shift(level=level, direction=-1)
        if family == "per_hour_to_min":
            return self._rate_hour_to_min(level)
        return self._rate_min_to_hour(level)

    def _pick_family(self, level: int) -> str:
        if level <= 2:
            families = ("hm_to_minutes", "hm_to_minutes", "hhmm_add")
        elif level <= 4:
            families = ("hm_to_minutes", "hhmm_add", "hhmm_sub", "hhmm_add")
        elif level <= 6:
            families = ("hhmm_add", "hhmm_sub", "per_hour_to_min", "hm_to_minutes")
        elif level <= 8:
            families = ("hhmm_add", "hhmm_sub", "per_hour_to_min", "per_min_to_hour")
        else:
            families = ("hhmm_add", "hhmm_sub", "per_hour_to_min", "per_min_to_hour", "hhmm_add")
        return self._rng.choice(families)

    def _pick_skin(self) -> str:
        if self._skin == "mixed":
            return self._rng.choice(("abstract", "aviation"))
        return self._skin

    def _hours_minutes_to_minutes(self, level: int) -> Problem:
        max_hours = 2 if level <= 3 else 4 if level <= 6 else 7
        hours = self._rng.randint(1, max_hours)
        step = 15 if level <= 2 else 10 if level <= 4 else 5
        minute_choices = tuple(range(step, 60, step))
        minutes = self._rng.choice(minute_choices)
        answer = (hours * 60) + minutes
        skin = self._pick_skin()
        if skin == "abstract":
            prompt = f"{hours} h {minutes:02d} min = ? min"
        else:
            prompt = f"Leg time convert: {hours} h {minutes:02d} min = ? min"
        return Problem(prompt=prompt, answer=answer)

    def _hhmm_shift(self, *, level: int, direction: int) -> Problem:
        step = 15 if level <= 2 else 10 if level <= 4 else 5
        start_hour = self._rng.randint(0, 22 if level >= 8 else 21)
        start_minute = self._rng.choice(tuple(range(0, 60, step)))
        start_total = (start_hour * 60) + start_minute
        deltas = (15, 20, 30, 45) if level <= 4 else (15, 20, 25, 30, 35, 40, 45, 50)
        delta = self._rng.choice(deltas)
        if direction < 0 and level < 8:
            start_total = max(start_total, delta + 15)
        answer = _minutes_to_hhmm(start_total + (direction * delta))
        start_hhmm = _minutes_to_hhmm(start_total)
        op = "+" if direction > 0 else "-"
        skin = self._pick_skin()
        if skin == "abstract":
            prompt = f"{start_hhmm:04d} {op} {delta} min = ? HHMM"
        elif direction > 0:
            prompt = f"ETA update: {start_hhmm:04d} + {delta} min = ? HHMM"
        else:
            prompt = f"Clock update: {start_hhmm:04d} - {delta} min = ? HHMM"
        return Problem(prompt=prompt, answer=answer)

    def _rate_hour_to_min(self, level: int) -> Problem:
        per_min_max = 6 if level <= 4 else 10 if level <= 7 else 14
        per_min = self._rng.randint(2, per_min_max)
        per_hour = per_min * 60
        unit = "km" if self._pick_skin() == "abstract" else self._rng.choice(("NM", "km"))
        prompt = f"{per_hour} {unit}/hr = ? {unit}/min"
        return Problem(prompt=prompt, answer=per_min)

    def _rate_min_to_hour(self, level: int) -> Problem:
        per_min_max = 7 if level <= 4 else 12 if level <= 7 else 16
        per_min = self._rng.randint(2, per_min_max)
        per_hour = per_min * 60
        unit = "km" if self._pick_skin() == "abstract" else self._rng.choice(("NM", "km"))
        prompt = f"{per_min} {unit}/min = ? {unit}/hr"
        return Problem(prompt=prompt, answer=per_hour)


def _airborne_family_for_scenario(scenario: AirborneScenario) -> str:
    if scenario.question_kind in {"arrival_time", "takeoff_time"}:
        return "route_time"
    if scenario.question_kind in {"empty_time", "fuel_endurance"}:
        return "endurance"
    if scenario.question_kind == "fuel_burned":
        return "fuel_burn"
    if scenario.question_kind == "distance_travelled":
        return "distance"
    if scenario.question_kind in {"parcel_weight", "parcel_effect"}:
        return "payload"
    raise ValueError(f"Unsupported airborne ANT family: {scenario.question_kind}")


def _airborne_has_chart(scenario: AirborneScenario) -> bool:
    return (
        scenario.fuel_reference_format == "chart"
        or scenario.parcel_reference_format == "chart"
    )


class _AntAirborneFamilyGenerator:
    family = "full"

    def __init__(self, *, seed: int) -> None:
        self._rng = SeededRng(seed)
        self._base = AirborneNumericalGenerator(self._rng)

    def next_problem(self, *, difficulty: float) -> Problem:
        level = _difficulty_to_level(difficulty)
        profile = build_ant_airborne_difficulty_profile(level, family=self.family)
        return self._base.generate(profile=profile)


class AntRouteTimeSolveGenerator(_AntAirborneFamilyGenerator):
    family = "route_time"
    _table_caps_by_level = (30.0, 28.0, 26.0, 24.0, 22.0, 20.0, 18.0, 16.0, 14.0, 12.0)
    _chart_caps_by_level = (36.0, 34.0, 32.0, 30.0, 28.0, 25.0, 22.0, 20.0, 18.0, 16.0)

    def cap_for_problem(self, *, problem: Problem, level: int) -> float | None:
        scenario = cast(AirborneScenario | None, problem.payload)
        if scenario is None:
            return None
        caps = (
            self._chart_caps_by_level
            if scenario.parcel_reference_format == "chart"
            else self._table_caps_by_level
        )
        return float(caps[max(1, min(10, int(level))) - 1])


class AntEnduranceSolveGenerator(_AntAirborneFamilyGenerator):
    family = "endurance"
    _table_caps_by_level = (28.0, 26.0, 24.0, 22.0, 20.0, 18.0, 16.0, 15.0, 13.0, 12.0)
    _chart_caps_by_level = (34.0, 32.0, 30.0, 28.0, 26.0, 24.0, 21.0, 19.0, 17.0, 15.0)

    def cap_for_problem(self, *, problem: Problem, level: int) -> float | None:
        scenario = cast(AirborneScenario | None, problem.payload)
        if scenario is None:
            return None
        advanced = scenario.fuel_reference_format == "chart" or scenario.fuel_minutes != 60
        caps = self._chart_caps_by_level if advanced else self._table_caps_by_level
        return float(caps[max(1, min(10, int(level))) - 1])


class AntFuelBurnSolveGenerator(_AntAirborneFamilyGenerator):
    family = "fuel_burn"
    _table_caps_by_level = (35.0, 33.0, 31.0, 29.0, 27.0, 24.0, 21.0, 18.0, 16.0, 14.0)
    _chart_caps_by_level = (42.0, 39.0, 36.0, 33.0, 30.0, 27.0, 24.0, 21.0, 18.0, 16.0)

    def cap_for_problem(self, *, problem: Problem, level: int) -> float | None:
        scenario = cast(AirborneScenario | None, problem.payload)
        if scenario is None:
            return None
        advanced = _airborne_has_chart(scenario) or scenario.speed_minutes != 60 or scenario.fuel_minutes != 60
        caps = self._chart_caps_by_level if advanced else self._table_caps_by_level
        return float(caps[max(1, min(10, int(level))) - 1])


class AntDistanceScanGenerator(_AntAirborneFamilyGenerator):
    family = "distance"
    _caps_by_level = (24.0, 22.0, 20.0, 18.0, 17.0, 16.0, 14.0, 13.0, 11.0, 10.0)

    def cap_for_problem(self, *, problem: Problem, level: int) -> float | None:
        _ = problem
        return float(self._caps_by_level[max(1, min(10, int(level))) - 1])


class AntPayloadReferenceGenerator(_AntAirborneFamilyGenerator):
    family = "payload"
    _table_caps_by_level = (24.0, 22.0, 20.0, 18.0, 17.0, 15.0, 13.0, 12.0, 10.0, 9.0)
    _chart_caps_by_level = (30.0, 28.0, 26.0, 24.0, 22.0, 19.0, 17.0, 15.0, 13.0, 11.0)

    def cap_for_problem(self, *, problem: Problem, level: int) -> float | None:
        scenario = cast(AirborneScenario | None, problem.payload)
        if scenario is None:
            return None
        caps = (
            self._chart_caps_by_level
            if scenario.parcel_reference_format == "chart"
            else self._table_caps_by_level
        )
        return float(caps[max(1, min(10, int(level))) - 1])


class AntInfoGrabberGenerator:
    _caps_by_level = (24.0, 22.0, 20.0, 19.0, 18.0, 17.0, 15.0, 14.0, 12.0, 11.0)

    def __init__(self, *, seed: int) -> None:
        self._rng = SeededRng(seed)
        self._base = AirborneNumericalGenerator(self._rng)

    def next_problem(self, *, difficulty: float) -> Problem:
        level = _difficulty_to_level(difficulty)
        scenario_problem = self._base.generate(
            profile=build_ant_airborne_difficulty_profile(level, family="full")
        )
        scenario = cast(AirborneScenario | None, scenario_problem.payload)
        if scenario is None:
            raise RuntimeError("info grabber expected airborne scenario payload")
        return self._build_info_problem(level=level, scenario=scenario)

    def cap_for_problem(self, *, problem: Problem, level: int) -> float | None:
        _ = problem
        return float(self._caps_by_level[max(1, min(10, int(level))) - 1])

    def _build_info_problem(self, *, level: int, scenario: AirborneScenario) -> Problem:
        builders = self._challenge_builders(level=level, scenario=scenario)
        builder = self._rng.choice(builders)
        return builder(level=level, scenario=scenario)

    def _challenge_builders(
        self,
        *,
        level: int,
        scenario: AirborneScenario,
    ) -> tuple[object, ...]:
        del scenario
        if level <= 2:
            return (
                self._route_distance_problem,
                self._shown_arrival_problem,
                self._shown_empty_problem,
            )
        if level <= 4:
            return (
                self._route_distance_problem,
                self._shown_arrival_problem,
                self._shown_empty_problem,
                self._current_speed_problem,
            )
        if level <= 6:
            return (
                self._route_distance_problem,
                self._shown_arrival_problem,
                self._shown_empty_problem,
                self._current_speed_problem,
                self._current_fuel_burn_problem,
            )
        return (
            self._route_distance_problem,
            self._shown_arrival_problem,
            self._shown_empty_problem,
            self._current_speed_problem,
            self._current_fuel_burn_problem,
        )

    def _route_distance_problem(self, *, level: int, scenario: AirborneScenario) -> Problem:
        spec = LookupRetainPromptSpec(
            target_label="Route total distance",
            target_digits=f"{scenario.route_distance_total:04d}",
            steps=(
                "Hold A and trace the active route once.",
                "Add the leg distances once, then hold that total.",
                *self._interference_steps(level=level, scenario=scenario, target_key="route_distance"),
            ),
        )
        updated = replace(
            scenario,
            question_kind="distance_travelled",
            answer_format="number",
            answer_label="Route Distance",
            answer_unit_label=scenario.distance_unit,
            answer_digits=4,
        )
        return Problem(
            prompt=spec.render_prompt(),
            answer=int(spec.target_digits),
            payload=updated,
        )

    def _shown_arrival_problem(self, *, level: int, scenario: AirborneScenario) -> Problem:
        spec = LookupRetainPromptSpec(
            target_label="Shown arrival time",
            target_digits=str(scenario.arrival_time_hhmm),
            steps=(
                "Read the shown arrival time on the journey table.",
                *self._interference_steps(level=level, scenario=scenario, target_key="arrival_time"),
            ),
        )
        updated = replace(
            scenario,
            question_kind="takeoff_time",
            given_time_label="Shown Arrival",
            given_time_hhmm=scenario.arrival_time_hhmm,
            answer_format="hhmm",
            answer_label="Recall",
            answer_unit_label="HHMM",
            answer_digits=4,
        )
        return Problem(prompt=spec.render_prompt(), answer=int(spec.target_digits), payload=updated)

    def _shown_empty_problem(self, *, level: int, scenario: AirborneScenario) -> Problem:
        spec = LookupRetainPromptSpec(
            target_label="Shown empty time",
            target_digits=str(scenario.empty_time_hhmm),
            steps=(
                "Read the shown empty time on the journey table.",
                *self._interference_steps(level=level, scenario=scenario, target_key="empty_time"),
            ),
        )
        updated = replace(
            scenario,
            question_kind="empty_time",
            given_time_label="Shown Empty",
            given_time_hhmm=scenario.empty_time_hhmm,
            answer_format="hhmm",
            answer_label="Recall",
            answer_unit_label="HHMM",
            answer_digits=4,
        )
        return Problem(prompt=spec.render_prompt(), answer=int(spec.target_digits), payload=updated)

    def _current_speed_problem(self, *, level: int, scenario: AirborneScenario) -> Problem:
        prompt_steps = [
            "Use the shown parcel weight as your anchor.",
            "Hold F and read the matching speed for that payload.",
        ]
        prompt_steps.extend(
            self._interference_steps(level=level, scenario=scenario, target_key="current_speed")
        )
        spec = LookupRetainPromptSpec(
            target_label="Current speed at the shown payload",
            target_digits=f"{scenario.speed_value:04d}",
            steps=tuple(prompt_steps),
        )
        updated = replace(
            scenario,
            question_kind="parcel_effect",
            answer_format="number",
            answer_label="Recall Speed",
            answer_unit_label=scenario.speed_unit,
            answer_digits=4,
        )
        return Problem(prompt=spec.render_prompt(), answer=int(spec.target_digits), payload=updated)

    def _current_fuel_burn_problem(self, *, level: int, scenario: AirborneScenario) -> Problem:
        prompt_steps = [
            "Use the shown parcel weight to recover the current speed.",
            "Hold F for speed, then D for the matching fuel burn.",
        ]
        prompt_steps.extend(
            self._interference_steps(level=level, scenario=scenario, target_key="current_fuel_burn")
        )
        spec = LookupRetainPromptSpec(
            target_label="Current fuel burn at the shown speed",
            target_digits=f"{scenario.fuel_burn_per_hr:04d}",
            steps=tuple(prompt_steps),
        )
        updated = replace(
            scenario,
            question_kind="fuel_burned",
            answer_format="number",
            answer_label="Recall Burn",
            answer_unit_label=scenario.fuel_unit.split("/")[0],
            answer_digits=4,
        )
        return Problem(prompt=spec.render_prompt(), answer=int(spec.target_digits), payload=updated)

    def _interference_steps(
        self,
        *,
        level: int,
        scenario: AirborneScenario,
        target_key: str,
    ) -> tuple[str, ...]:
        if level <= 2:
            return ("Enter it immediately after you find it.",)
        if level <= 4:
            return ("Release the reference, hold the digits for one beat, then answer.",)

        steps: list[str] = []
        if level <= 6:
            steps.append(self._secondary_lookup_step(scenario=scenario, target_key=target_key))
            steps.append("Do not enter the second value. Enter the original target.")
            return tuple(steps)

        steps.append(self._secondary_lookup_step(scenario=scenario, target_key=target_key))
        steps.append(self._micro_calc_step())
        if level >= 9:
            steps.append("Commit without reopening the original reference if you can help it.")
        steps.append("Ignore the interference work and enter the original target.")
        return tuple(steps)

    def _secondary_lookup_step(self, *, scenario: AirborneScenario, target_key: str) -> str:
        options = [
            f"Then confirm the destination {scenario.target_label} on the journey table.",
            "Then glance at the route via line once before you answer.",
            "Then reopen the other reference page for one quick scan.",
        ]
        if target_key != "current_fuel_burn":
            options.append("Then hold D and confirm the burn row for the recovered speed.")
        if target_key != "current_speed":
            options.append("Then hold F and confirm the speed row for the shown payload.")
        return str(self._rng.choice(tuple(options)))

    def _micro_calc_step(self) -> str:
        family = self._rng.choice(("add", "sub", "hm"))
        if family == "add":
            a = self._rng.randint(12, 49)
            b = self._rng.randint(11, 38)
            return f"Then mentally solve {a} + {b}, but do not enter it."
        if family == "sub":
            a = self._rng.randint(45, 96)
            b = self._rng.randint(9, min(34, a - 1))
            return f"Then mentally solve {a} - {b}, but do not enter it."
        hours = self._rng.randint(1, 3)
        minutes = self._rng.choice((10, 15, 20, 25, 30, 35, 40, 45))
        return f"Then convert {hours} h {minutes:02d} min to minutes, but do not enter it."


class AntMixedTempoSetGenerator:
    _arith_caps_by_level = (12.0, 10.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.5, 2.0)
    _unit_caps_by_level = (20.0, 18.0, 16.0, 14.0, 12.0, 10.0, 9.0, 8.0, 6.0, 5.0)

    def __init__(self, *, seed: int, skin: str = "mixed") -> None:
        self._rng = SeededRng(seed)
        self._snap = AntSnapFactsSprintGenerator(seed=seed + 101, skin=skin)
        self._time = AntTimeFlipGenerator(seed=seed + 202, skin=skin)
        self._route = AntRouteTimeSolveGenerator(seed=seed + 303)
        self._endurance = AntEnduranceSolveGenerator(seed=seed + 404)
        self._fuel = AntFuelBurnSolveGenerator(seed=seed + 505)
        self._distance = AntDistanceScanGenerator(seed=seed + 606)
        self._payload = AntPayloadReferenceGenerator(seed=seed + 707)
        self._last_family: str | None = None

    def next_problem(self, *, difficulty: float) -> Problem:
        level = _difficulty_to_level(difficulty)
        family = self._pick_family(level)
        if family == "snap":
            problem = self._snap.next_problem(difficulty=difficulty)
            base_cap_s = self._arith_caps_by_level[level - 1]
            self._last_family = family
            return Problem(
                prompt=problem.prompt,
                answer=problem.answer,
                tolerance=problem.tolerance,
                payload=AntProblemRuntimeMeta(family=family, base_cap_s=base_cap_s),
            )
        if family == "time":
            problem = self._time.next_problem(difficulty=difficulty)
            base_cap_s = self._unit_caps_by_level[level - 1]
            self._last_family = family
            return Problem(
                prompt=problem.prompt,
                answer=problem.answer,
                tolerance=problem.tolerance,
                payload=AntProblemRuntimeMeta(family=family, base_cap_s=base_cap_s),
            )

        airborne_generators = {
            "route_time": self._route,
            "endurance": self._endurance,
            "fuel_burn": self._fuel,
            "distance": self._distance,
            "payload": self._payload,
        }
        problem = airborne_generators[family].next_problem(difficulty=difficulty)
        self._last_family = family
        return problem

    def cap_for_problem(self, *, problem: Problem, level: int) -> float | None:
        payload = problem.payload
        if isinstance(payload, AntProblemRuntimeMeta) and payload.base_cap_s is not None:
            return float(payload.base_cap_s)

        scenario = cast(AirborneScenario | None, payload)
        if scenario is None:
            return None
        family = _airborne_family_for_scenario(scenario)
        airborne_generators = {
            "route_time": self._route,
            "endurance": self._endurance,
            "fuel_burn": self._fuel,
            "distance": self._distance,
            "payload": self._payload,
        }
        return airborne_generators[family].cap_for_problem(problem=problem, level=level)

    def _pick_family(self, level: int) -> str:
        if level <= 2:
            families = ("snap", "snap", "time", "time")
        elif level <= 4:
            families = ("snap", "time", "snap", "time", "route_time")
        elif level <= 6:
            families = (
                "snap",
                "time",
                "route_time",
                "endurance",
                "fuel_burn",
                "time",
                "snap",
            )
        elif level <= 8:
            families = (
                "snap",
                "time",
                "route_time",
                "endurance",
                "fuel_burn",
                "distance",
                "payload",
                "time",
            )
        else:
            families = (
                "route_time",
                "endurance",
                "fuel_burn",
                "distance",
                "payload",
                "snap",
                "time",
                "payload",
                "distance",
                "fuel_burn",
            )
        chosen = str(self._rng.choice(families))
        if self._last_family == chosen and self._rng.random() < 0.45:
            alternates = tuple(family for family in families if family != chosen)
            return str(self._rng.choice(alternates))
        return chosen


class TimedCapDrill:
    def __init__(
        self,
        *,
        title: str,
        instructions: Sequence[str],
        generator: AntProblemGenerator,
        clock: Clock,
        seed: int,
        difficulty: float,
        practice_questions: int,
        scored_duration_s: float,
        mode: AntDrillMode,
        base_caps_by_level: Sequence[float],
        adaptive_config: AntAdaptiveDifficultyConfig | None = None,
        scorer: AnswerScorer | None = None,
    ) -> None:
        self._title = str(title)
        self._instructions = list(instructions)
        self._generator = generator
        self._clock = clock
        self._seed = int(seed)
        self._difficulty = _clamp01(difficulty)
        self._practice_questions = max(0, int(practice_questions))
        self._scored_duration_s = max(1.0, float(scored_duration_s))
        self._mode = mode
        self._mode_profile = ANT_DRILL_MODE_PROFILES[mode]
        self._adaptive_mode_profile = ANT_ADAPTIVE_MODE_PROFILES[mode]
        self._adaptive_config = adaptive_config or AntAdaptiveDifficultyConfig()
        self._base_caps_by_level = tuple(float(v) for v in base_caps_by_level)
        self._scorer = scorer

        self._phase = Phase.INSTRUCTIONS
        self._current: Problem | None = None
        self._presented_at_s: float | None = None
        self._current_cap_s: float | None = None
        self._events: list[QuestionEvent] = []

        self._practice_answered = 0
        self._scored_started_at_s: float | None = None
        self._scored_attempted = 0
        self._scored_correct = 0
        self._scored_timeouts = 0
        self._scored_total_score = 0.0
        self._scored_max_score = 0.0
        self._timeout_streak = 0
        self._correct_streak = 0
        self._max_timeout_streak = 0
        self._recent_outcomes: list[tuple[bool, bool]] = []
        self._difficulty_changes: list[AntDifficultyChange] = []
        self._scored_start_level: int | None = None
        self._last_feedback = ""

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def difficulty(self) -> float:
        return float(self._difficulty)

    @property
    def practice_questions(self) -> int:
        return int(self._practice_questions)

    @property
    def scored_duration_s(self) -> float:
        return float(self._scored_duration_s)

    def events(self) -> list[QuestionEvent]:
        return list(self._events)

    def difficulty_changes(self) -> list[AntDifficultyChange]:
        return list(self._difficulty_changes)

    def can_exit(self) -> bool:
        return self._phase in (
            Phase.INSTRUCTIONS,
            Phase.PRACTICE,
            Phase.PRACTICE_DONE,
            Phase.RESULTS,
        )

    def start_practice(self) -> None:
        if self._phase is not Phase.INSTRUCTIONS:
            return
        if self._practice_questions <= 0:
            self._phase = Phase.PRACTICE_DONE
            return
        self._phase = Phase.PRACTICE
        self._last_feedback = ""
        self._deal_new_problem()

    def start_scored(self) -> None:
        if self._phase in (Phase.SCORED, Phase.RESULTS):
            return
        self._phase = Phase.SCORED
        self._last_feedback = ""
        self._scored_started_at_s = self._clock.now()
        self._scored_start_level = self._current_level()
        self._recent_outcomes.clear()
        self._difficulty_changes.clear()
        self._correct_streak = 0
        self._timeout_streak = 0
        self._scored_total_score = 0.0
        self._scored_max_score = 0.0
        self._deal_new_problem()

    def update(self) -> None:
        if self._phase not in (Phase.PRACTICE, Phase.SCORED):
            return
        if self._phase is Phase.SCORED and self.time_remaining_s() == 0.0:
            self._finish_results()
            return
        if self._item_remaining_s() <= 0.0:
            self._record_timeout()

    def submit_answer(self, raw: str) -> bool:
        if self._phase not in (Phase.PRACTICE, Phase.SCORED):
            return False

        token = str(raw).strip().lower()
        if token in {"__skip_practice__", "skip_practice"} and self._phase is Phase.PRACTICE:
            self._phase = Phase.PRACTICE_DONE
            self._current = None
            self._presented_at_s = None
            self._current_cap_s = None
            return True
        if token in {"__skip_section__", "skip_section", "__skip_all__", "skip_all"} and (
            self._phase is Phase.SCORED
        ):
            self._finish_results()
            return True

        if self._current is None or self._presented_at_s is None:
            return False
        if self._item_remaining_s() <= 0.0:
            self._record_timeout()
            return False

        raw_in = str(raw)
        stripped = raw_in.strip()
        if stripped == "":
            return False
        try:
            user_answer = int(stripped)
        except ValueError:
            return False

        answered_at_s = self._clock.now()
        response_time_s = max(0.0, answered_at_s - self._presented_at_s)
        score_value = self._score_answer(problem=self._current, user_answer=user_answer, raw=raw_in)
        is_correct = score_value >= 0.999999
        self._events.append(
            QuestionEvent(
                index=len(self._events),
                phase=self._phase,
                prompt=self._current.prompt,
                correct_answer=self._current.answer,
                user_answer=user_answer,
                is_correct=is_correct,
                presented_at_s=self._presented_at_s,
                answered_at_s=answered_at_s,
                response_time_s=response_time_s,
                raw=raw_in,
                score=score_value,
                max_score=1.0,
            )
        )

        if self._phase is Phase.SCORED:
            adaptive_note = self._record_scored_outcome(is_correct=is_correct, is_timeout=False)
        else:
            self._practice_answered += 1
            adaptive_note = None

        self._last_feedback = self._compose_feedback(
            is_timeout=False,
            is_correct=is_correct,
            correct_answer=self._display_answer(self._current),
            score_value=score_value,
            adaptive_note=adaptive_note,
        )

        if self._phase is Phase.SCORED and self.time_remaining_s() == 0.0:
            self._finish_results()
            return True
        if self._phase is Phase.PRACTICE and self._practice_answered >= self._practice_questions:
            self._phase = Phase.PRACTICE_DONE
            self._current = None
            self._presented_at_s = None
            self._current_cap_s = None
            return True

        self._deal_new_problem()
        return True

    def snapshot(self) -> TestSnapshot:
        payload = (
            None
            if self._phase not in (Phase.PRACTICE, Phase.SCORED) or self._current is None
            else self._current.payload
        )
        return TestSnapshot(
            title=self._title,
            phase=self._phase,
            prompt=self.current_prompt(),
            input_hint=self._input_hint(),
            time_remaining_s=self.time_remaining_s(),
            attempted_scored=self._scored_attempted,
            correct_scored=self._scored_correct,
            payload=payload,
            practice_feedback=self._last_feedback if self._last_feedback else None,
        )

    def current_prompt(self) -> str:
        if self._phase is Phase.INSTRUCTIONS:
            return "\n".join(self._instructions)
        if self._phase is Phase.PRACTICE_DONE:
            return (
                f"Practice complete.\nMode: {self._mode_profile.label}\n"
                f"Adaptive difficulty: {'On' if self._adaptive_config.enabled else 'Off'}\n"
                "Press Enter to start the timed block."
            )
        if self._phase is Phase.RESULTS:
            summary = self.scored_summary()
            return (
                "Results\n"
                f"Mode: {self._mode_profile.label}\n"
                f"Attempted: {summary.attempted}\n"
                f"Correct: {summary.correct}\n"
                f"Exact accuracy: {int(round(summary.accuracy * 100.0))}%\n"
                f"Score ratio: {summary.score_ratio * 100.0:.1f}%\n"
                f"Correct/min: {summary.correct_per_min:.1f}\n"
                f"Timeouts: {summary.timeouts} ({summary.fixation_rate * 100.0:.1f}%)\n"
                f"Level: {summary.difficulty_level_start} -> {summary.difficulty_level_end}\n"
                f"Difficulty changes: {summary.difficulty_change_count}\n"
                f"Max timeout streak: {summary.max_timeout_streak}"
            )
        if self._current is None:
            return ""
        return self._current.prompt

    def time_remaining_s(self) -> float | None:
        if self._phase is not Phase.SCORED:
            return None
        assert self._scored_started_at_s is not None
        remaining = self._scored_duration_s - (self._clock.now() - self._scored_started_at_s)
        return max(0.0, remaining)

    def scored_summary(self) -> AntDrillAttemptSummary:
        duration_s = float(self._scored_duration_s)
        attempted = self._scored_attempted
        correct = self._scored_correct
        accuracy = 0.0 if attempted == 0 else correct / attempted
        throughput = (attempted / duration_s) * 60.0 if duration_s > 0.0 else 0.0
        correct_per_min = (correct / duration_s) * 60.0 if duration_s > 0.0 else 0.0
        scored_events = [event for event in self._events if event.phase is Phase.SCORED]
        rts = [event.response_time_s for event in scored_events]
        mean_rt = None if not rts else sum(rts) / len(rts)
        fixation_rate = 0.0 if attempted == 0 else self._scored_timeouts / attempted
        start_level = self._scored_start_level if self._scored_start_level is not None else self._current_level()
        end_level = self._current_level()
        return AntDrillAttemptSummary(
            attempted=attempted,
            correct=correct,
            accuracy=accuracy,
            duration_s=duration_s,
            throughput_per_min=throughput,
            mean_response_time_s=mean_rt,
            total_score=float(self._scored_total_score),
            max_score=float(self._scored_max_score),
            score_ratio=(
                0.0
                if self._scored_max_score <= 0.0
                else float(self._scored_total_score) / float(self._scored_max_score)
            ),
            correct_per_min=correct_per_min,
            timeouts=self._scored_timeouts,
            fixation_rate=fixation_rate,
            max_timeout_streak=self._max_timeout_streak,
            mode=self._mode.value,
            difficulty_level=end_level,
            difficulty_level_start=start_level,
            difficulty_level_end=end_level,
            difficulty_change_count=len(self._difficulty_changes),
            adaptive_enabled=bool(self._adaptive_config.enabled),
            adaptive_window_size=self._adaptive_window_size(),
        )

    def _input_hint(self) -> str:
        if self._phase not in (Phase.PRACTICE, Phase.SCORED):
            return "Press Enter to continue"
        return f"L{self._current_level()} | Cap {self._item_remaining_s():0.1f}s | Type answer then Enter"

    def _current_level(self) -> int:
        return _difficulty_to_level(self._difficulty)

    def _item_remaining_s(self) -> float:
        if self._current_cap_s is None or self._presented_at_s is None:
            return 0.0
        elapsed = self._clock.now() - self._presented_at_s
        return max(0.0, self._current_cap_s - elapsed)

    def _deal_new_problem(self) -> None:
        self._current = self._generator.next_problem(difficulty=self._difficulty)
        self._presented_at_s = self._clock.now()
        cap_resolver = getattr(self._generator, "cap_for_problem", None)
        if callable(cap_resolver):
            resolved = cap_resolver(problem=self._current, level=self._current_level())
            if resolved is not None:
                scaled = float(resolved) * self._mode_profile.cap_scale
                self._current_cap_s = max(2.0, min(60.0, scaled))
                return
        meta = self._current.payload
        if isinstance(meta, AntProblemRuntimeMeta) and meta.base_cap_s is not None:
            scaled = float(meta.base_cap_s) * self._mode_profile.cap_scale
            self._current_cap_s = max(2.0, min(60.0, scaled))
            return
        self._current_cap_s = self._cap_for_level(self._current_level())

    def _score_answer(self, *, problem: Problem, user_answer: int, raw: str) -> float:
        if self._scorer is not None:
            try:
                return float(self._scorer.score(problem=problem, user_answer=user_answer, raw=raw))
            except Exception:
                return 0.0
        return 1.0 if user_answer == problem.answer else 0.0

    def _cap_for_level(self, level: int) -> float:
        capped_level = max(1, min(len(self._base_caps_by_level), int(level)))
        base = self._base_caps_by_level[capped_level - 1]
        scaled = float(base) * self._mode_profile.cap_scale
        return max(2.0, min(60.0, scaled))

    def _compose_feedback(
        self,
        *,
        is_timeout: bool,
        is_correct: bool,
        correct_answer: str,
        score_value: float,
        adaptive_note: str | None,
    ) -> str:
        if not self._mode_profile.immediate_feedback:
            return ""
        if is_timeout:
            text = f"Timeout. Correct answer: {correct_answer}"
        elif is_correct:
            text = "Correct. Commit and move."
        elif score_value > 0.0:
            text = f"Partial. Exact answer: {correct_answer}"
        else:
            text = f"Incorrect. Correct answer: {correct_answer}"
        if adaptive_note:
            text = f"{text} {adaptive_note}"
        return text

    def _record_scored_outcome(self, *, is_correct: bool, is_timeout: bool) -> str | None:
        self._scored_attempted += 1
        if is_timeout:
            self._scored_max_score += 1.0
        elif self._events:
            latest = self._events[-1]
            self._scored_total_score += float(latest.score)
            self._scored_max_score += float(latest.max_score)
        if is_correct:
            self._scored_correct += 1
            self._correct_streak += 1
            self._timeout_streak = 0
        else:
            self._correct_streak = 0
            if is_timeout:
                self._scored_timeouts += 1
                self._timeout_streak += 1
                self._max_timeout_streak = max(self._max_timeout_streak, self._timeout_streak)
            else:
                self._timeout_streak = 0
        self._recent_outcomes.append((is_correct, is_timeout))
        max_window = max(1, self._adaptive_window_size())
        if len(self._recent_outcomes) > max_window:
            del self._recent_outcomes[: len(self._recent_outcomes) - max_window]
        return self._maybe_adjust_difficulty()

    def _adaptive_window_size(self) -> int:
        override = self._adaptive_config.window_size
        if override is not None:
            return max(1, int(override))
        return int(self._adaptive_mode_profile.window_size)

    def _maybe_adjust_difficulty(self) -> str | None:
        if not self._adaptive_config.enabled:
            return None

        level = self._current_level()
        profile = self._adaptive_mode_profile
        window_size = self._adaptive_window_size()
        outcomes = list(self._recent_outcomes)

        if self._mode is AntDrillMode.BUILD:
            if len(outcomes) >= window_size:
                accuracy, fixation = self._window_metrics(outcomes)
                if fixation > profile.fixation_ceiling or accuracy < profile.accuracy_low:
                    return self._change_difficulty(
                        new_level=level - 1,
                        reason=f"window acc {accuracy:.2f}, fixation {fixation:.2f}",
                    )
            if self._correct_streak >= profile.raise_streak and self._timeout_streak == 0:
                return self._change_difficulty(
                    new_level=level + 1,
                    reason=f"clean streak {self._correct_streak}",
                )
            return None

        if len(outcomes) < window_size:
            return None
        accuracy, fixation = self._window_metrics(outcomes)
        if fixation > profile.fixation_ceiling or accuracy < profile.accuracy_low:
            return self._change_difficulty(
                new_level=level - 1,
                reason=f"window acc {accuracy:.2f}, fixation {fixation:.2f}",
            )
        if accuracy > profile.accuracy_high and fixation <= profile.fixation_ceiling:
            return self._change_difficulty(
                new_level=level + 1,
                reason=f"window acc {accuracy:.2f}, fixation {fixation:.2f}",
            )
        return None

    @staticmethod
    def _window_metrics(outcomes: Sequence[tuple[bool, bool]]) -> tuple[float, float]:
        if not outcomes:
            return 0.0, 0.0
        attempted = len(outcomes)
        correct = sum(1 for is_correct, _is_timeout in outcomes if is_correct)
        timeouts = sum(1 for _is_correct, is_timeout in outcomes if is_timeout)
        return (correct / attempted, timeouts / attempted)

    def _change_difficulty(self, *, new_level: int, reason: str) -> str | None:
        old_level = self._current_level()
        clamped = max(1, min(10, int(new_level)))
        if clamped == old_level:
            return None
        self._difficulty = _level_to_difficulty(clamped)
        self._difficulty_changes.append(
            AntDifficultyChange(
                after_attempt=self._scored_attempted,
                old_level=old_level,
                new_level=clamped,
                reason=reason,
            )
        )
        self._recent_outcomes.clear()
        self._correct_streak = 0
        self._timeout_streak = 0
        direction = "up" if clamped > old_level else "down"
        return f"Difficulty {direction} to L{clamped}."

    def _record_timeout(self) -> None:
        if self._current is None or self._presented_at_s is None:
            return
        answered_at_s = self._clock.now()
        response_time_s = max(0.0, answered_at_s - self._presented_at_s)
        self._events.append(
            QuestionEvent(
                index=len(self._events),
                phase=self._phase,
                prompt=self._current.prompt,
                correct_answer=self._current.answer,
                user_answer=0,
                is_correct=False,
                presented_at_s=self._presented_at_s,
                answered_at_s=answered_at_s,
                response_time_s=response_time_s,
                raw="__timeout__",
                score=0.0,
                max_score=1.0,
            )
        )

        if self._phase is Phase.SCORED:
            adaptive_note = self._record_scored_outcome(is_correct=False, is_timeout=True)
        else:
            self._practice_answered += 1
            adaptive_note = None

        self._last_feedback = self._compose_feedback(
            is_timeout=True,
            is_correct=False,
            correct_answer=self._display_answer(self._current),
            score_value=0.0,
            adaptive_note=adaptive_note,
        )

        if self._phase is Phase.SCORED and self.time_remaining_s() == 0.0:
            self._finish_results()
            return
        if self._phase is Phase.PRACTICE and self._practice_answered >= self._practice_questions:
            self._phase = Phase.PRACTICE_DONE
            self._current = None
            self._presented_at_s = None
            self._current_cap_s = None
            return

        self._deal_new_problem()

    def _finish_results(self) -> None:
        self._phase = Phase.RESULTS
        self._current = None
        self._presented_at_s = None
        self._current_cap_s = None

    @staticmethod
    def _display_answer(problem: Problem) -> str:
        return expected_digits_for_problem(problem)


def build_ant_snap_facts_sprint_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: AntSnapFactsSprintConfig | None = None,
) -> TimedCapDrill:
    normalized_mode = mode if isinstance(mode, AntDrillMode) else AntDrillMode(str(mode).strip().lower())
    profile = ANT_DRILL_MODE_PROFILES[normalized_mode]
    cfg = config or AntSnapFactsSprintConfig()
    generator = AntSnapFactsSprintGenerator(seed=seed, skin=cfg.skin)
    practice_questions = (
        profile.practice_questions if cfg.practice_questions is None else int(cfg.practice_questions)
    )
    scored_duration_s = (
        profile.scored_duration_s if cfg.scored_duration_s is None else float(cfg.scored_duration_s)
    )
    instructions = [
        "Airborne Numerical: Snap Facts Sprint",
        f"Mode: {profile.label}",
        profile.note,
        "Every item has a hard cap. On expiry it auto-advances and logs a timeout.",
        f"Adaptive difficulty: {'On' if cfg.adaptive.enabled else 'Off'}",
        "Press Enter to begin practice.",
    ]
    return TimedCapDrill(
        title=f"Airborne Numerical: Snap Facts Sprint ({profile.label})",
        instructions=instructions,
        generator=generator,
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        practice_questions=practice_questions,
        scored_duration_s=scored_duration_s,
        mode=normalized_mode,
        base_caps_by_level=(12.0, 10.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.5, 2.0),
        adaptive_config=cfg.adaptive,
    )


def build_ant_time_flip_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: AntTimeFlipConfig | None = None,
) -> TimedCapDrill:
    normalized_mode = mode if isinstance(mode, AntDrillMode) else AntDrillMode(str(mode).strip().lower())
    profile = ANT_DRILL_MODE_PROFILES[normalized_mode]
    cfg = config or AntTimeFlipConfig()
    generator = AntTimeFlipGenerator(seed=seed, skin=cfg.skin)
    practice_questions = (
        profile.practice_questions if cfg.practice_questions is None else int(cfg.practice_questions)
    )
    scored_duration_s = (
        profile.scored_duration_s if cfg.scored_duration_s is None else float(cfg.scored_duration_s)
    )
    instructions = [
        "Airborne Numerical: Time Flip",
        f"Mode: {profile.label}",
        profile.note,
        "Convert between minutes, HHMM, and clean per-hour/per-minute rates under caps.",
        f"Adaptive difficulty: {'On' if cfg.adaptive.enabled else 'Off'}",
        "Press Enter to begin practice.",
    ]
    return TimedCapDrill(
        title=f"Airborne Numerical: Time Flip ({profile.label})",
        instructions=instructions,
        generator=generator,
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        practice_questions=practice_questions,
        scored_duration_s=scored_duration_s,
        mode=normalized_mode,
        base_caps_by_level=(20.0, 18.0, 16.0, 14.0, 12.0, 10.0, 9.0, 8.0, 6.0, 5.0),
        adaptive_config=cfg.adaptive,
    )


def build_ant_mixed_tempo_set_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.TEMPO,
    config: AntMixedTempoSetConfig | None = None,
) -> TimedCapDrill:
    normalized_mode = mode if isinstance(mode, AntDrillMode) else AntDrillMode(str(mode).strip().lower())
    profile = ANT_DRILL_MODE_PROFILES[normalized_mode]
    cfg = config or AntMixedTempoSetConfig()
    generator = AntMixedTempoSetGenerator(seed=seed, skin=cfg.skin)
    practice_questions = (
        profile.practice_questions if cfg.practice_questions is None else int(cfg.practice_questions)
    )
    scored_duration_s = (
        profile.scored_duration_s if cfg.scored_duration_s is None else float(cfg.scored_duration_s)
    )
    instructions = [
        "Airborne Numerical: Mixed Tempo Set",
        f"Mode: {profile.label}",
        profile.note,
        "Mixed stream of the live Airborne Numerical families: retrieval, time/rate, route-time, endurance, fuel-burn, distance, and payload.",
        "Airborne items reuse the Airborne Numerical scenario overlays and live tolerance rules.",
        f"Adaptive difficulty: {'On' if cfg.adaptive.enabled else 'Off'}",
        "Press Enter to begin practice.",
    ]
    return TimedCapDrill(
        title=f"Airborne Numerical: Mixed Tempo Set ({profile.label})",
        instructions=instructions,
        generator=generator,
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        practice_questions=practice_questions,
        scored_duration_s=scored_duration_s,
        mode=normalized_mode,
        base_caps_by_level=(12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0),
        adaptive_config=cfg.adaptive,
        scorer=AirborneScorer(),
    )


def build_ant_route_time_solve_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: AntRouteTimeSolveConfig | None = None,
) -> TimedCapDrill:
    normalized_mode = mode if isinstance(mode, AntDrillMode) else AntDrillMode(str(mode).strip().lower())
    profile = ANT_DRILL_MODE_PROFILES[normalized_mode]
    cfg = config or AntRouteTimeSolveConfig()
    generator = AntRouteTimeSolveGenerator(seed=seed)
    practice_questions = (
        profile.practice_questions if cfg.practice_questions is None else int(cfg.practice_questions)
    )
    scored_duration_s = (
        profile.scored_duration_s if cfg.scored_duration_s is None else float(cfg.scored_duration_s)
    )
    instructions = [
        "Airborne Numerical: Route Time Solve",
        f"Mode: {profile.label}",
        profile.note,
        "Route-time items reuse the existing airborne scenario screen and 4-digit Airborne Numerical answer style.",
        "Hold A for distances. Hold F for speed and parcel reference when needed.",
        "Table reads expect exact answers. Chart reads allow the same Airborne Numerical tolerance as the live test.",
        f"Adaptive difficulty: {'On' if cfg.adaptive.enabled else 'Off'}",
        "Press Enter to begin practice.",
    ]
    return TimedCapDrill(
        title=f"Airborne Numerical: Route Time Solve ({profile.label})",
        instructions=instructions,
        generator=generator,
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        practice_questions=practice_questions,
        scored_duration_s=scored_duration_s,
        mode=normalized_mode,
        base_caps_by_level=AntRouteTimeSolveGenerator._table_caps_by_level,
        adaptive_config=cfg.adaptive,
        scorer=AirborneScorer(),
    )


def build_ant_fuel_burn_solve_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: AntFuelBurnSolveConfig | None = None,
) -> TimedCapDrill:
    normalized_mode = mode if isinstance(mode, AntDrillMode) else AntDrillMode(str(mode).strip().lower())
    profile = ANT_DRILL_MODE_PROFILES[normalized_mode]
    cfg = config or AntFuelBurnSolveConfig()
    generator = AntFuelBurnSolveGenerator(seed=seed)
    practice_questions = (
        profile.practice_questions if cfg.practice_questions is None else int(cfg.practice_questions)
    )
    scored_duration_s = (
        profile.scored_duration_s if cfg.scored_duration_s is None else float(cfg.scored_duration_s)
    )
    instructions = [
        "Airborne Numerical: Fuel Burn Solve",
        f"Mode: {profile.label}",
        profile.note,
        "Fuel-burn items reuse the existing airborne scenario screen, tables/charts, and the 4-digit Airborne Numerical answer style.",
        "Hold A for route distances, D for speed and fuel reference, and F for parcel speed reference.",
        "Chart-based reads keep the live Airborne Numerical estimation tolerance instead of forcing exact-school-math answers.",
        f"Adaptive difficulty: {'On' if cfg.adaptive.enabled else 'Off'}",
        "Press Enter to begin practice.",
    ]
    return TimedCapDrill(
        title=f"Airborne Numerical: Fuel Burn Solve ({profile.label})",
        instructions=instructions,
        generator=generator,
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        practice_questions=practice_questions,
        scored_duration_s=scored_duration_s,
        mode=normalized_mode,
        base_caps_by_level=AntFuelBurnSolveGenerator._table_caps_by_level,
        adaptive_config=cfg.adaptive,
        scorer=AirborneScorer(),
    )


def build_ant_endurance_solve_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: AntEnduranceSolveConfig | None = None,
) -> TimedCapDrill:
    normalized_mode = mode if isinstance(mode, AntDrillMode) else AntDrillMode(str(mode).strip().lower())
    profile = ANT_DRILL_MODE_PROFILES[normalized_mode]
    cfg = config or AntEnduranceSolveConfig()
    generator = AntEnduranceSolveGenerator(seed=seed)
    practice_questions = (
        profile.practice_questions if cfg.practice_questions is None else int(cfg.practice_questions)
    )
    scored_duration_s = (
        profile.scored_duration_s if cfg.scored_duration_s is None else float(cfg.scored_duration_s)
    )
    instructions = [
        "Airborne Numerical: Endurance Solve",
        f"Mode: {profile.label}",
        profile.note,
        "Endurance items reuse the airborne scenario UI for empty-time and fuel-endurance work.",
        "Hold D for speed and fuel reference. Use the live Airborne Numerical HHMM and 4-digit number formats.",
        "Chart-based fuel reads keep the same Airborne Numerical tolerance as the full test.",
        f"Adaptive difficulty: {'On' if cfg.adaptive.enabled else 'Off'}",
        "Press Enter to begin practice.",
    ]
    return TimedCapDrill(
        title=f"Airborne Numerical: Endurance Solve ({profile.label})",
        instructions=instructions,
        generator=generator,
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        practice_questions=practice_questions,
        scored_duration_s=scored_duration_s,
        mode=normalized_mode,
        base_caps_by_level=AntEnduranceSolveGenerator._table_caps_by_level,
        adaptive_config=cfg.adaptive,
        scorer=AirborneScorer(),
    )


def build_ant_distance_scan_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: AntDistanceScanConfig | None = None,
) -> TimedCapDrill:
    normalized_mode = mode if isinstance(mode, AntDrillMode) else AntDrillMode(str(mode).strip().lower())
    profile = ANT_DRILL_MODE_PROFILES[normalized_mode]
    cfg = config or AntDistanceScanConfig()
    generator = AntDistanceScanGenerator(seed=seed)
    practice_questions = (
        profile.practice_questions if cfg.practice_questions is None else int(cfg.practice_questions)
    )
    scored_duration_s = (
        profile.scored_duration_s if cfg.scored_duration_s is None else float(cfg.scored_duration_s)
    )
    instructions = [
        "Airborne Numerical: Distance Scan",
        f"Mode: {profile.label}",
        profile.note,
        "Distance items reuse the airborne route screen and train fast leg scanning plus one-pass route summing.",
        "Hold A for route distances. Read the ask first, then sweep the route once.",
        f"Adaptive difficulty: {'On' if cfg.adaptive.enabled else 'Off'}",
        "Press Enter to begin practice.",
    ]
    return TimedCapDrill(
        title=f"Airborne Numerical: Distance Scan ({profile.label})",
        instructions=instructions,
        generator=generator,
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        practice_questions=practice_questions,
        scored_duration_s=scored_duration_s,
        mode=normalized_mode,
        base_caps_by_level=AntDistanceScanGenerator._caps_by_level,
        adaptive_config=cfg.adaptive,
        scorer=AirborneScorer(),
    )


def build_ant_payload_reference_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: AntPayloadReferenceConfig | None = None,
) -> TimedCapDrill:
    normalized_mode = mode if isinstance(mode, AntDrillMode) else AntDrillMode(str(mode).strip().lower())
    profile = ANT_DRILL_MODE_PROFILES[normalized_mode]
    cfg = config or AntPayloadReferenceConfig()
    generator = AntPayloadReferenceGenerator(seed=seed)
    practice_questions = (
        profile.practice_questions if cfg.practice_questions is None else int(cfg.practice_questions)
    )
    scored_duration_s = (
        profile.scored_duration_s if cfg.scored_duration_s is None else float(cfg.scored_duration_s)
    )
    instructions = [
        "Airborne Numerical: Payload Reference",
        f"Mode: {profile.label}",
        profile.note,
        "Payload items reuse the parcel-speed reference and cover parcel weight plus parcel effect on speed.",
        "Hold F for the speed and parcel reference. Table items are exact; chart items preserve Airborne Numerical tolerance.",
        f"Adaptive difficulty: {'On' if cfg.adaptive.enabled else 'Off'}",
        "Press Enter to begin practice.",
    ]
    return TimedCapDrill(
        title=f"Airborne Numerical: Payload Reference ({profile.label})",
        instructions=instructions,
        generator=generator,
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        practice_questions=practice_questions,
        scored_duration_s=scored_duration_s,
        mode=normalized_mode,
        base_caps_by_level=AntPayloadReferenceGenerator._table_caps_by_level,
        adaptive_config=cfg.adaptive,
        scorer=AirborneScorer(),
    )


def build_ant_info_grabber_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: AntInfoGrabberConfig | None = None,
) -> TimedCapDrill:
    normalized_mode = mode if isinstance(mode, AntDrillMode) else AntDrillMode(str(mode).strip().lower())
    profile = ANT_DRILL_MODE_PROFILES[normalized_mode]
    cfg = config or AntInfoGrabberConfig()
    generator = AntInfoGrabberGenerator(seed=seed)
    practice_questions = (
        profile.practice_questions if cfg.practice_questions is None else int(cfg.practice_questions)
    )
    scored_duration_s = (
        profile.scored_duration_s if cfg.scored_duration_s is None else float(cfg.scored_duration_s)
    )
    instructions = [
        "Airborne Numerical: Info Grabber",
        f"Mode: {profile.label}",
        profile.note,
        "Use the live airborne UI to grab one exact target, then hold it through a delay or interference prompt.",
        "Answers are typed exactly. Partial digit-position credit still counts toward the score ratio.",
        "This is the reusable Airborne Numerical lookup-and-retain pattern; v1 stays typed-response only.",
        f"Adaptive difficulty: {'On' if cfg.adaptive.enabled else 'Off'}",
        "Press Enter to begin practice.",
    ]
    return TimedCapDrill(
        title=f"Airborne Numerical: Info Grabber ({profile.label})",
        instructions=instructions,
        generator=generator,
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        practice_questions=practice_questions,
        scored_duration_s=scored_duration_s,
        mode=normalized_mode,
        base_caps_by_level=AntInfoGrabberGenerator._caps_by_level,
        adaptive_config=cfg.adaptive,
        scorer=LookupRetainScorer(),
    )
