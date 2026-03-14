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
from .numerical_operations import (
    NumericalOperationsGenerator,
    NumericalOperationsProblemProfile,
)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _difficulty_to_level(difficulty: float) -> int:
    return max(1, min(10, int(round(_clamp01(difficulty) * 9.0)) + 1))


def _level_to_difficulty(level: int) -> float:
    clamped = max(1, min(10, int(level)))
    return float(clamped - 1) / 9.0


def _normalize_mode(mode: AntDrillMode | str) -> AntDrillMode:
    return mode if isinstance(mode, AntDrillMode) else AntDrillMode(str(mode).strip().lower())


@dataclass(frozen=True, slots=True)
class NoDrillConfig:
    practice_questions: int | None = None
    scored_duration_s: float | None = None
    adaptive: AntAdaptiveDifficultyConfig = field(
        default_factory=lambda: AntAdaptiveDifficultyConfig(enabled=False)
    )


class NoFactPrimeGenerator:
    def __init__(self, *, seed: int) -> None:
        self._base = NumericalOperationsGenerator(
            seed=seed,
            profile=NumericalOperationsProblemProfile(operand_profile="fact_prime"),
        )

    def next_problem(self, *, difficulty: float) -> Problem:
        return self._base.next_problem(difficulty=min(0.8, float(difficulty)))


class NoOperatorLaddersGenerator:
    _operator_order = ("+", "-", "*", "/")

    def __init__(self, *, seed: int) -> None:
        self._index = 0
        self._generators = {
            op: NumericalOperationsGenerator(
                seed=seed + (idx + 1) * 101,
                profile=NumericalOperationsProblemProfile(
                    operator_family=op,
                    operand_profile="fact_prime" if op in {"*", "/"} else "clean_compute",
                ),
            )
            for idx, op in enumerate(self._operator_order)
        }

    def next_problem(self, *, difficulty: float) -> Problem:
        op = self._operator_order[(self._index // 3) % len(self._operator_order)]
        self._index += 1
        local_level = max(1, min(10, _difficulty_to_level(difficulty)))
        return self._generators[op].next_problem(difficulty=_level_to_difficulty(local_level))


class NoCleanComputeGenerator:
    def __init__(self, *, seed: int) -> None:
        self._base = NumericalOperationsGenerator(
            seed=seed,
            profile=NumericalOperationsProblemProfile(operand_profile="clean_compute"),
        )

    def next_problem(self, *, difficulty: float) -> Problem:
        return self._base.next_problem(difficulty=difficulty)


class NoMixedTempoGenerator:
    def __init__(self, *, seed: int) -> None:
        self._index = 0
        self._easy = NumericalOperationsGenerator(
            seed=seed + 101,
            profile=NumericalOperationsProblemProfile(operand_profile="clean_compute"),
        )
        self._hard = NumericalOperationsGenerator(
            seed=seed + 202,
            profile=NumericalOperationsProblemProfile(operand_profile="default"),
        )

    def next_problem(self, *, difficulty: float) -> Problem:
        self._index += 1
        level = _difficulty_to_level(difficulty)
        if self._index % 3 == 0:
            local_level = min(10, level + 2)
            return self._hard.next_problem(difficulty=_level_to_difficulty(local_level))
        local_level = max(1, level - 2)
        return self._easy.next_problem(difficulty=_level_to_difficulty(local_level))


class NoPressureRunGenerator:
    def __init__(self, *, seed: int) -> None:
        self._base = NumericalOperationsGenerator(
            seed=seed,
            profile=NumericalOperationsProblemProfile(operand_profile="default"),
        )

    def next_problem(self, *, difficulty: float) -> Problem:
        level = _difficulty_to_level(difficulty)
        local_level = min(10, level + 1)
        return self._base.next_problem(difficulty=_level_to_difficulty(local_level))


def _build_no_drill(
    *,
    title_base: str,
    instructions: tuple[str, ...],
    generator: object,
    clock: Clock,
    seed: int,
    difficulty: float,
    mode: AntDrillMode | str,
    config: NoDrillConfig,
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


def build_no_fact_prime_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: NoDrillConfig | None = None,
) -> TimedCapDrill:
    cfg = config or NoDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_no_drill(
        title_base="Numerical Operations: Fact Prime",
        instructions=(
            "Numerical Operations: Fact Prime",
            f"Mode: {profile.label}",
            "Prime doubles, near-doubles, complements, halves, and clean fact families before harder arithmetic.",
            "Keep the rhythm clean and let the next-item flash correct small misses without stopping your pace.",
            "Press Enter to begin practice.",
        ),
        generator=NoFactPrimeGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.5, 4.0, 3.5, 3.0),
    )


def build_no_operator_ladders_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: NoDrillConfig | None = None,
) -> TimedCapDrill:
    cfg = config or NoDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_no_drill(
        title_base="Numerical Operations: Operator Ladders",
        instructions=(
            "Numerical Operations: Operator Ladders",
            f"Mode: {profile.label}",
            "Stay on one operator family long enough to reinforce the pattern before the ladder advances.",
            "The order rotates +, -, x, and division so the warm-up still covers the full test family.",
            "Press Enter to begin practice.",
        ),
        generator=NoOperatorLaddersGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.5, 4.0),
    )


def build_no_clean_compute_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: NoDrillConfig | None = None,
) -> TimedCapDrill:
    cfg = config or NoDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_no_drill(
        title_base="Numerical Operations: Clean Compute",
        instructions=(
            "Numerical Operations: Clean Compute",
            f"Mode: {profile.label}",
            "Use curated operand shapes that reward memoized arithmetic patterns and clean transforms.",
            "This block stays typed end-to-end, just like the full Numerical Operations test.",
            "Press Enter to begin practice.",
        ),
        generator=NoCleanComputeGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.5, 4.0),
    )


def build_no_mixed_tempo_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.TEMPO,
    config: NoDrillConfig | None = None,
) -> TimedCapDrill:
    cfg = config or NoDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_no_drill(
        title_base="Numerical Operations: Mixed Tempo",
        instructions=(
            "Numerical Operations: Mixed Tempo",
            f"Mode: {profile.label}",
            "Mixed arithmetic stream with an easy/easy/hard cadence to train failure recovery without losing tempo.",
            "Misses should not change your rhythm. Let the next-item flash recalibrate you and keep moving.",
            "Press Enter to begin practice.",
        ),
        generator=NoMixedTempoGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(9.0, 8.5, 8.0, 7.5, 7.0, 6.0, 5.0, 4.5, 4.0, 3.5),
    )


def build_no_pressure_run_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.STRESS,
    config: NoDrillConfig | None = None,
) -> TimedCapDrill:
    cfg = config or NoDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_no_drill(
        title_base="Numerical Operations: Pressure Run",
        instructions=(
            "Numerical Operations: Pressure Run",
            f"Mode: {profile.label}",
            "This block keeps the full Numerical Operations visual style but tightens the caps and widens the arithmetic range.",
            "Use it as the late-workout crunch block, not as a perfection block.",
            "Press Enter to begin practice.",
        ),
        generator=NoPressureRunGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(8.0, 7.5, 7.0, 6.5, 6.0, 5.5, 5.0, 4.5, 4.0, 3.5),
    )
