from __future__ import annotations

from dataclasses import dataclass, field

from .ant_drills import (
    ANT_DRILL_MODE_PROFILES,
    AntAdaptiveDifficultyConfig,
    AntDrillMode,
    TimedCapDrill,
)
from .clock import Clock
from .cognitive_core import Problem, SeededRng


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _difficulty_to_level(difficulty: float) -> int:
    return max(1, min(10, int(round(_clamp01(difficulty) * 9.0)) + 1))


def _normalize_mode(mode: AntDrillMode | str) -> AntDrillMode:
    return mode if isinstance(mode, AntDrillMode) else AntDrillMode(str(mode).strip().lower())


@dataclass(frozen=True, slots=True)
class MaDrillConfig:
    practice_questions: int | None = None
    scored_duration_s: float | None = None
    adaptive: AntAdaptiveDifficultyConfig = field(
        default_factory=lambda: AntAdaptiveDifficultyConfig(enabled=False)
    )


@dataclass(frozen=True, slots=True)
class MaProblemPayload:
    family: str
    variant: str
    base_cap_s: float | None = None


class _BaseMaGenerator:
    def __init__(self, *, seed: int) -> None:
        self._rng = SeededRng(seed)

    def cap_for_problem(self, *, problem: Problem, level: int) -> float | None:
        payload = problem.payload
        if isinstance(payload, MaProblemPayload) and payload.base_cap_s is not None:
            return float(payload.base_cap_s)
        return None

    @staticmethod
    def _problem(
        *,
        prompt: str,
        answer: int,
        family: str,
        variant: str,
        base_cap_s: float | None = None,
    ) -> Problem:
        return Problem(
            prompt=prompt,
            answer=int(answer),
            payload=MaProblemPayload(
                family=str(family),
                variant=str(variant),
                base_cap_s=base_cap_s,
            ),
        )


class MaOneStepFluencyGenerator(_BaseMaGenerator):
    _OPERATORS = ("+", "-", "*", "/")

    def next_problem(self, *, difficulty: float) -> Problem:
        level = _difficulty_to_level(difficulty)
        op = self._OPERATORS[(level + self._rng.randint(0, 3)) % len(self._OPERATORS)]
        if op == "+":
            hi = 14 if level <= 3 else 35 if level <= 6 else 85
            a = self._rng.randint(4, hi)
            b = self._rng.randint(3, hi)
            return self._problem(
                prompt=f"{a} + {b} =",
                answer=a + b,
                family="one_step_fluency",
                variant="addition",
            )
        if op == "-":
            hi = 18 if level <= 3 else 45 if level <= 6 else 95
            a = self._rng.randint(8, hi)
            b = self._rng.randint(2, min(a - 1, hi // 2 + 4))
            return self._problem(
                prompt=f"{a} - {b} =",
                answer=a - b,
                family="one_step_fluency",
                variant="subtraction",
            )
        if op == "*":
            a_hi = 10 if level <= 3 else 14 if level <= 6 else 18
            b_hi = 10 if level <= 3 else 12 if level <= 6 else 15
            a = self._rng.randint(2, a_hi)
            b = self._rng.randint(2, b_hi)
            return self._problem(
                prompt=f"{a} × {b} =",
                answer=a * b,
                family="one_step_fluency",
                variant="multiplication",
            )
        divisor = self._rng.randint(2, 10 if level <= 4 else 14 if level <= 7 else 18)
        quotient = self._rng.randint(2, 12 if level <= 4 else 18 if level <= 7 else 24)
        dividend = divisor * quotient
        return self._problem(
            prompt=f"{dividend} ÷ {divisor} =",
            answer=quotient,
            family="one_step_fluency",
            variant="division",
        )


class MaPercentageSnapGenerator(_BaseMaGenerator):
    _LOW = (10, 20, 25, 50)
    _MID = (5, 10, 15, 20, 25, 50, 75)
    _HIGH = (5, 10, 12, 15, 20, 25, 40, 50, 75)

    def next_problem(self, *, difficulty: float) -> Problem:
        level = _difficulty_to_level(difficulty)
        choices = self._LOW if level <= 3 else self._MID if level <= 7 else self._HIGH
        pct = int(self._rng.choice(choices))
        if pct in {25, 50, 75}:
            base = self._rng.choice(tuple(4 * value for value in range(20, 161)))
        elif pct in {5, 15, 20, 40}:
            base = self._rng.choice(tuple(5 * value for value in range(20, 161)))
        elif pct == 12:
            base = self._rng.choice(tuple(25 * value for value in range(8, 61)))
        else:
            base = self._rng.choice(tuple(10 * value for value in range(12, 181)))
        answer = (base * pct) // 100
        return self._problem(
            prompt=f"{pct}% of {base} =",
            answer=answer,
            family="percentage_snap",
            variant=f"{pct}_percent",
        )


class MaRateTimeDistanceGenerator(_BaseMaGenerator):
    def next_problem(self, *, difficulty: float) -> Problem:
        level = _difficulty_to_level(difficulty)
        variant = self._rng.choice(("distance", "time_minutes", "speed"))
        if variant == "distance":
            speed = self._rng.choice(tuple(range(60, 301, 15)))
            hours = self._rng.choice((1, 2, 3, 4, 5) if level <= 5 else (2, 3, 4, 5, 6))
            return self._problem(
                prompt=f"{speed} kt for {hours} hr = ? nm",
                answer=speed * hours,
                family="rate_time_distance",
                variant=variant,
            )
        if variant == "time_minutes":
            speed = self._rng.choice(tuple(range(60, 301, 15)))
            hours = self._rng.choice((1, 2, 3, 4, 5))
            distance = speed * hours
            return self._problem(
                prompt=f"{distance} nm at {speed} kt = ? min",
                answer=hours * 60,
                family="rate_time_distance",
                variant=variant,
            )
        hours = self._rng.choice((1, 2, 3, 4, 5))
        distance = self._rng.choice(tuple(range(90, 901, 15)))
        distance = max(60, (distance // hours) * hours)
        return self._problem(
            prompt=f"{distance} nm in {hours} hr = ? kt",
            answer=distance // hours,
            family="rate_time_distance",
            variant=variant,
        )


class MaFuelEnduranceGenerator(_BaseMaGenerator):
    def next_problem(self, *, difficulty: float) -> Problem:
        level = _difficulty_to_level(difficulty)
        variant = self._rng.choice(("fuel_used", "endurance_minutes", "burn_rate"))
        if variant == "fuel_used":
            burn_lph = self._rng.choice(tuple(range(24, 121, 6)))
            minutes = self._rng.choice((30, 45, 60, 75, 90, 120))
            answer = (burn_lph * minutes) // 60
            return self._problem(
                prompt=f"Burn {burn_lph} L/hr for {minutes} min = ? L",
                answer=answer,
                family="fuel_endurance",
                variant=variant,
            )
        if variant == "endurance_minutes":
            burn_lph = self._rng.choice(tuple(range(24, 121, 6)))
            hours = self._rng.choice((2, 3, 4, 5) if level <= 5 else (3, 4, 5, 6))
            fuel = burn_lph * hours
            return self._problem(
                prompt=f"{fuel} L usable at {burn_lph} L/hr = ? min",
                answer=hours * 60,
                family="fuel_endurance",
                variant=variant,
            )
        burn_lph = self._rng.choice(tuple(range(24, 121, 6)))
        hours = self._rng.choice((1, 2, 3, 4, 5))
        fuel = burn_lph * hours
        return self._problem(
            prompt=f"{fuel} L over {hours} hr = ? L/hr",
            answer=burn_lph,
            family="fuel_endurance",
            variant=variant,
        )


class MaMixedConversionCapsGenerator(_BaseMaGenerator):
    def next_problem(self, *, difficulty: float) -> Problem:
        level = _difficulty_to_level(difficulty)
        variant = self._rng.choice(
            (
                "hours_to_minutes",
                "minutes_to_seconds",
                "kg_to_g",
                "l_to_ml",
                "km_to_m",
                "hhmm_to_minutes",
            )
        )
        if variant == "hours_to_minutes":
            hours = self._rng.choice((1, 2, 3, 4, 5, 6))
            quarter = self._rng.choice((0, 15, 30, 45))
            answer = (hours * 60) + quarter
            prompt = f"{hours} hr {quarter:02d} min = ? min"
        elif variant == "minutes_to_seconds":
            minutes = self._rng.choice(tuple(range(2, 31)))
            answer = minutes * 60
            prompt = f"{minutes} min = ? s"
        elif variant == "kg_to_g":
            kg = self._rng.choice(tuple(range(2, 26)))
            answer = kg * 1000
            prompt = f"{kg} kg = ? g"
        elif variant == "l_to_ml":
            litres = self._rng.choice(tuple(range(2, 26)))
            answer = litres * 1000
            prompt = f"{litres} L = ? mL"
        elif variant == "km_to_m":
            km = self._rng.choice(tuple(range(2, 41)))
            answer = km * 1000
            prompt = f"{km} km = ? m"
        else:
            hours = self._rng.choice((1, 2, 3, 4, 5, 6))
            minutes = self._rng.choice((0, 10, 15, 20, 30, 40, 45, 50))
            hhmm = (hours * 100) + minutes
            answer = (hours * 60) + minutes
            prompt = f"{hhmm:04d} = ? min"
        cap = max(3.0, 8.5 - (level * 0.35))
        return self._problem(
            prompt=prompt,
            answer=answer,
            family="mixed_conversion_caps",
            variant=variant,
            base_cap_s=cap,
        )


def _build_ma_drill(
    *,
    title_base: str,
    instructions: tuple[str, ...],
    generator: object,
    clock: Clock,
    seed: int,
    difficulty: float,
    mode: AntDrillMode | str,
    config: MaDrillConfig,
    base_caps_by_level: tuple[float, ...],
) -> TimedCapDrill:
    normalized_mode = _normalize_mode(mode)
    profile = ANT_DRILL_MODE_PROFILES[normalized_mode]
    practice_questions = (
        profile.practice_questions if config.practice_questions is None else int(config.practice_questions)
    )
    scored_duration_s = (
        profile.scored_duration_s if config.scored_duration_s is None else float(config.scored_duration_s)
    )
    return TimedCapDrill(
        title=f"{title_base} ({profile.label})",
        instructions=list(instructions),
        generator=generator,
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        practice_questions=practice_questions,
        scored_duration_s=scored_duration_s,
        mode=normalized_mode,
        base_caps_by_level=base_caps_by_level,
        adaptive_config=config.adaptive,
        immediate_feedback_override=True,
    )


def build_ma_one_step_fluency_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: MaDrillConfig | None = None,
) -> TimedCapDrill:
    cfg = config or MaDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_ma_drill(
        title_base="Mental Arithmetic: One-Step Fluency",
        instructions=(
            "Mental Arithmetic: One-Step Fluency",
            f"Mode: {profile.label}",
            "Work direct one-step arithmetic until the operator changes stop costing setup time.",
            "Keep the typed rhythm clean and let the hard cap prevent over-fixating on one miss.",
            "Press Enter to begin practice.",
        ),
        generator=MaOneStepFluencyGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(10.0, 9.0, 8.0, 7.0, 6.5, 6.0, 5.5, 5.0, 4.5, 4.0),
    )


def build_ma_percentage_snap_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: MaDrillConfig | None = None,
) -> TimedCapDrill:
    cfg = config or MaDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_ma_drill(
        title_base="Mental Arithmetic: Percentage Snap",
        instructions=(
            "Mental Arithmetic: Percentage Snap",
            f"Mode: {profile.label}",
            "Snap to 5, 10, 20, 25, 50, and 75 percent transforms without rebuilding the whole problem.",
            "Use the cleaner percentage families to support several aviation-style and symbolic tasks at once.",
            "Press Enter to begin practice.",
        ),
        generator=MaPercentageSnapGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0),
    )


def build_ma_rate_time_distance_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.TEMPO,
    config: MaDrillConfig | None = None,
) -> TimedCapDrill:
    cfg = config or MaDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_ma_drill(
        title_base="Mental Arithmetic: Rate Time Distance",
        instructions=(
            "Mental Arithmetic: Rate Time Distance",
            f"Mode: {profile.label}",
            "Solve one-step speed, distance, and time transforms without reading them as full word problems.",
            "Keep the units stable and answer in the exact requested unit immediately.",
            "Press Enter to begin practice.",
        ),
        generator=MaRateTimeDistanceGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0),
    )


def build_ma_fuel_endurance_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.TEMPO,
    config: MaDrillConfig | None = None,
) -> TimedCapDrill:
    cfg = config or MaDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_ma_drill(
        title_base="Mental Arithmetic: Fuel Burn And Endurance",
        instructions=(
            "Mental Arithmetic: Fuel Burn And Endurance",
            f"Mode: {profile.label}",
            "Keep fuel-used, endurance, and burn-rate transforms automatic before the full scenario layer comes back.",
            "Use exact typed answers and recover immediately after a slow setup.",
            "Press Enter to begin practice.",
        ),
        generator=MaFuelEnduranceGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0),
    )


def build_ma_mixed_conversion_caps_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.STRESS,
    config: MaDrillConfig | None = None,
) -> TimedCapDrill:
    cfg = config or MaDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_ma_drill(
        title_base="Mental Arithmetic: Mixed Conversion Caps",
        instructions=(
            "Mental Arithmetic: Mixed Conversion Caps",
            f"Mode: {profile.label}",
            "Flip between time, mass, volume, and distance conversions under tighter per-item caps.",
            "The family changes quickly on purpose; do not spend extra time re-orienting.",
            "Press Enter to begin practice.",
        ),
        generator=MaMixedConversionCapsGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(9.0, 8.5, 8.0, 7.5, 7.0, 6.5, 6.0, 5.5, 5.0, 4.5),
    )
