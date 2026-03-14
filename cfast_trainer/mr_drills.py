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
from .math_reasoning import (
    MR_ALL_DOMAIN_KEYS,
    MR_MOTION_FUEL_DOMAIN_KEYS,
    MR_REMAINING_DOMAIN_KEYS,
    MathReasoningFact,
    MathReasoningGenerator,
    MathReasoningScenarioSpec,
    MathReasoningTrainingPayload,
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


def _format_answer(value: int, unit: str = "") -> str:
    return f"{int(value)} {unit}".strip()


def _training_problem(
    *,
    spec: MathReasoningScenarioSpec,
    prompt_stem: str,
    answer_value: int,
    response_label: str,
    answer_unit_label: str,
    base_cap_s: float,
) -> Problem:
    payload = MathReasoningTrainingPayload(
        domain=spec.domain,
        stem=prompt_stem,
        response_label=response_label,
        answer_unit_label=answer_unit_label,
        input_digits=max(4, len(str(abs(int(answer_value)))) + 2),
        display_answer_text=_format_answer(answer_value, answer_unit_label),
        domain_key=spec.domain_key,
        base_cap_s=base_cap_s,
    )
    return Problem(prompt=prompt_stem, answer=int(answer_value), payload=payload)


@dataclass(frozen=True, slots=True)
class MrDrillConfig:
    practice_questions: int | None = None
    scored_duration_s: float | None = None
    adaptive: AntAdaptiveDifficultyConfig = field(
        default_factory=lambda: AntAdaptiveDifficultyConfig(enabled=False)
    )


class _MrTypedGenerator:
    def cap_for_problem(self, *, problem: Problem, level: int) -> float | None:
        _ = level
        payload = problem.payload
        if isinstance(payload, MathReasoningTrainingPayload) and payload.base_cap_s is not None:
            return float(payload.base_cap_s)
        return None


class MrRelevantInfoScanGenerator(_MrTypedGenerator):
    def __init__(self, *, seed: int) -> None:
        self._rng = SeededRng(seed)
        self._base = MathReasoningGenerator(seed=seed + 101)

    def next_problem(self, *, difficulty: float) -> Problem:
        level = _difficulty_to_level(difficulty)
        spec = self._base.next_scenario_spec(
            difficulty=min(0.85, float(difficulty)),
            include_filler=True,
        )
        fact = self._pick_fact(spec.facts, level=level)
        stem = (
            f"{spec.stem}\n\nFind the stated {fact.label.lower()} and enter only the number."
        )
        return _training_problem(
            spec=spec,
            prompt_stem=stem,
            answer_value=fact.value,
            response_label=fact.label,
            answer_unit_label=fact.unit,
            base_cap_s=(26.0, 24.0, 22.0, 20.0, 19.0, 18.0, 17.0, 16.0, 14.0, 13.0)[level - 1],
        )

    def _pick_fact(self, facts: tuple[MathReasoningFact, ...], *, level: int) -> MathReasoningFact:
        if level <= 3:
            return facts[0]
        if level <= 6:
            return facts[min(1, len(facts) - 1)]
        return self._rng.choice(facts)


class MrUnitRelationPrimeGenerator(_MrTypedGenerator):
    def __init__(self, *, seed: int) -> None:
        self._rng = SeededRng(seed)

    def next_problem(self, *, difficulty: float) -> Problem:
        level = _difficulty_to_level(difficulty)
        family = self._rng.choice(("minutes", "per_min", "fuel", "percent"))
        if family == "minutes":
            hours = self._rng.randint(1, 4 if level <= 5 else 6)
            minutes = self._rng.choice((10, 15, 20, 30, 40, 45, 50))
            total = (hours * 60) + minutes
            stem = f"Convert {hours} h {minutes:02d} min into total minutes."
            return Problem(
                prompt=stem,
                answer=total,
                payload=MathReasoningTrainingPayload(
                    domain="Unit And Relation Prime",
                    stem=stem,
                    response_label="Minutes",
                    answer_unit_label="minutes",
                    input_digits=6,
                    display_answer_text=_format_answer(total, "minutes"),
                    domain_key="unit_prime",
                    base_cap_s=(14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.5, 6.0)[level - 1],
                ),
            )
        if family == "per_min":
            per_min = self._rng.randint(2, 10 if level <= 5 else 16)
            per_hour = per_min * 60
            stem = f"Convert {per_hour} items/hour into items per minute."
            return Problem(
                prompt=stem,
                answer=per_min,
                payload=MathReasoningTrainingPayload(
                    domain="Unit And Relation Prime",
                    stem=stem,
                    response_label="Items per minute",
                    answer_unit_label="items/min",
                    input_digits=6,
                    display_answer_text=_format_answer(per_min, "items/min"),
                    domain_key="unit_prime",
                    base_cap_s=(14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.5, 6.0)[level - 1],
                ),
            )
        if family == "fuel":
            burn = self._rng.randint(8, 18 if level <= 5 else 30)
            minutes = self._rng.choice((10, 15, 20, 25, 30))
            total = burn * minutes
            stem = f"At {burn} L/min for {minutes} minutes, how much fuel is used?"
            return Problem(
                prompt=stem,
                answer=total,
                payload=MathReasoningTrainingPayload(
                    domain="Unit And Relation Prime",
                    stem=stem,
                    response_label="Fuel used",
                    answer_unit_label="L",
                    input_digits=6,
                    display_answer_text=_format_answer(total, "L"),
                    domain_key="unit_prime",
                    base_cap_s=(14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.5, 6.0)[level - 1],
                ),
            )
        base = self._rng.choice((40, 60, 80, 100, 120, 160))
        pct = self._rng.choice((10, 20, 25, 50))
        delta = (base * pct) // 100
        stem = f"What is {pct}% of {base}?"
        return Problem(
            prompt=stem,
            answer=delta,
            payload=MathReasoningTrainingPayload(
                domain="Unit And Relation Prime",
                stem=stem,
                response_label="Percent amount",
                answer_unit_label="",
                input_digits=6,
                display_answer_text=str(delta),
                domain_key="unit_prime",
                base_cap_s=(14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.5, 6.0)[level - 1],
            ),
        )


class MrOneStepSolveGenerator(_MrTypedGenerator):
    def __init__(self, *, seed: int) -> None:
        self._base = MathReasoningGenerator(
            seed=seed,
            allowed_domain_keys=(
                MR_MOTION_FUEL_DOMAIN_KEYS
                + ("percentage_change", "rate_scaling")
            ),
        )

    def next_problem(self, *, difficulty: float) -> Problem:
        level = _difficulty_to_level(difficulty)
        local_level = max(1, level - 1)
        spec = self._base.next_scenario_spec(
            difficulty=_level_to_difficulty(local_level),
            include_filler=False,
        )
        stem = f"{spec.stem}\n\nEnter the exact answer."
        return _training_problem(
            spec=spec,
            prompt_stem=stem,
            answer_value=spec.correct_value,
            response_label=spec.solution_label,
            answer_unit_label=spec.unit,
            base_cap_s=(24.0, 22.0, 20.0, 18.0, 17.0, 16.0, 14.0, 13.0, 12.0, 11.0)[level - 1],
        )


class MrMultiStepSolveGenerator(_MrTypedGenerator):
    def __init__(self, *, seed: int) -> None:
        self._base = MathReasoningGenerator(seed=seed)

    def next_problem(self, *, difficulty: float) -> Problem:
        level = _difficulty_to_level(difficulty)
        local_level = min(10, level + 1)
        spec = self._base.next_scenario_spec(
            difficulty=_level_to_difficulty(local_level),
            include_filler=True,
        )
        stem = f"{spec.stem}\n\nIgnore filler, solve the question, and enter the exact answer."
        return _training_problem(
            spec=spec,
            prompt_stem=stem,
            answer_value=spec.correct_value,
            response_label=spec.solution_label,
            answer_unit_label=spec.unit,
            base_cap_s=(30.0, 28.0, 26.0, 24.0, 22.0, 20.0, 18.0, 17.0, 16.0, 15.0)[level - 1],
        )


class MrDomainRunGenerator:
    _domain_order = MR_MOTION_FUEL_DOMAIN_KEYS + MR_REMAINING_DOMAIN_KEYS

    def __init__(self, *, seed: int) -> None:
        self._base = MathReasoningGenerator(seed=seed, allowed_domain_keys=self._domain_order)
        self._domain_index = 0
        self._items_in_domain = 0

    def next_problem(self, *, difficulty: float) -> Problem:
        token = self._domain_order[self._domain_index]
        spec = self._base.next_scenario_spec(
            difficulty=difficulty,
            domain_key=token,
            include_filler=False,
        )
        self._items_in_domain += 1
        if self._items_in_domain >= 2:
            self._items_in_domain = 0
            self._domain_index = (self._domain_index + 1) % len(self._domain_order)
        return self._base.problem_from_spec(spec)

    def cap_for_problem(self, *, problem: Problem, level: int) -> float | None:
        _ = problem
        return float((28.0, 26.0, 24.0, 22.0, 20.0, 18.0, 16.0, 15.0, 14.0, 13.0)[level - 1])


class MrMixedPressureSetGenerator:
    def __init__(self, *, seed: int) -> None:
        self._base = MathReasoningGenerator(seed=seed, allowed_domain_keys=MR_ALL_DOMAIN_KEYS)

    def next_problem(self, *, difficulty: float) -> Problem:
        level = _difficulty_to_level(difficulty)
        local_level = min(10, level + 1)
        spec = self._base.next_scenario_spec(
            difficulty=_level_to_difficulty(local_level),
            include_filler=False,
        )
        return self._base.problem_from_spec(spec)

    def cap_for_problem(self, *, problem: Problem, level: int) -> float | None:
        _ = problem
        return float((22.0, 21.0, 20.0, 19.0, 18.0, 16.0, 15.0, 14.0, 13.0, 12.0)[level - 1])


def _build_mr_drill(
    *,
    title_base: str,
    instructions: tuple[str, ...],
    generator: object,
    clock: Clock,
    seed: int,
    difficulty: float,
    mode: AntDrillMode | str,
    config: MrDrillConfig,
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


def build_mr_relevant_info_scan_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: MrDrillConfig | None = None,
) -> TimedCapDrill:
    cfg = config or MrDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_mr_drill(
        title_base="Mathematics Reasoning: Relevant Info Scan",
        instructions=(
            "Mathematics Reasoning: Relevant Info Scan",
            f"Mode: {profile.label}",
            "Read the full stem, find the one value the prompt asks for, and ignore the filler values around it.",
            "This is the extraction block, not the solve block.",
            "Press Enter to begin practice.",
        ),
        generator=MrRelevantInfoScanGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(24.0, 22.0, 20.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0),
    )


def build_mr_unit_relation_prime_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: MrDrillConfig | None = None,
) -> TimedCapDrill:
    cfg = config or MrDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_mr_drill(
        title_base="Mathematics Reasoning: Unit And Relation Prime",
        instructions=(
            "Mathematics Reasoning: Unit And Relation Prime",
            f"Mode: {profile.label}",
            "Prime the clean conversions and one-step relations that support the bigger word problems.",
            "Keep these answers fast and clean so the later blocks can use the bandwidth.",
            "Press Enter to begin practice.",
        ),
        generator=MrUnitRelationPrimeGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.5, 6.0),
    )


def build_mr_one_step_solve_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: MrDrillConfig | None = None,
) -> TimedCapDrill:
    cfg = config or MrDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_mr_drill(
        title_base="Mathematics Reasoning: One-Step Solve",
        instructions=(
            "Mathematics Reasoning: One-Step Solve",
            f"Mode: {profile.label}",
            "Solve the clean one-step versions of the live Mathematics Reasoning domains before filler and longer setup arrive.",
            "Type the exact answer and use the next-item flash to correct fast.",
            "Press Enter to begin practice.",
        ),
        generator=MrOneStepSolveGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(24.0, 22.0, 20.0, 18.0, 17.0, 16.0, 14.0, 13.0, 12.0, 11.0),
    )


def build_mr_multi_step_solve_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.TEMPO,
    config: MrDrillConfig | None = None,
) -> TimedCapDrill:
    cfg = config or MrDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_mr_drill(
        title_base="Mathematics Reasoning: Multi-Step Solve",
        instructions=(
            "Mathematics Reasoning: Multi-Step Solve",
            f"Mode: {profile.label}",
            "These are the fuller word problems: more values, more filler, and more chances to fixate.",
            "Stay on tempo and do not let one slow setup break the block.",
            "Press Enter to begin practice.",
        ),
        generator=MrMultiStepSolveGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(30.0, 28.0, 26.0, 24.0, 22.0, 20.0, 18.0, 17.0, 16.0, 15.0),
    )


def build_mr_domain_run_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.TEMPO,
    config: MrDrillConfig | None = None,
) -> TimedCapDrill:
    cfg = config or MrDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    practice_questions = (
        profile.practice_questions if cfg.practice_questions is None else int(cfg.practice_questions)
    )
    scored_duration_s = (
        profile.scored_duration_s if cfg.scored_duration_s is None else float(cfg.scored_duration_s)
    )
    return TimedCapDrill(
        title=f"Mathematics Reasoning: Domain Run ({profile.label})",
        instructions=[
            "Mathematics Reasoning: Domain Run",
            f"Mode: {profile.label}",
            "Real multiple-choice Mathematics Reasoning items in grouped domain runs.",
            "The block stays on motion and fuel domains first, then rotates into percentages, averages, and rates.",
            "Press Enter to begin practice.",
        ],
        generator=MrDomainRunGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        practice_questions=practice_questions,
        scored_duration_s=scored_duration_s,
        mode=_normalize_mode(mode),
        base_caps_by_level=(28.0, 26.0, 24.0, 22.0, 20.0, 18.0, 16.0, 15.0, 14.0, 13.0),
        adaptive_config=cfg.adaptive,
        immediate_feedback_override=True,
    )


def build_mr_mixed_pressure_set_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.STRESS,
    config: MrDrillConfig | None = None,
) -> TimedCapDrill:
    cfg = config or MrDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    practice_questions = (
        profile.practice_questions if cfg.practice_questions is None else int(cfg.practice_questions)
    )
    scored_duration_s = (
        profile.scored_duration_s if cfg.scored_duration_s is None else float(cfg.scored_duration_s)
    )
    return TimedCapDrill(
        title=f"Mathematics Reasoning: Mixed Pressure Set ({profile.label})",
        instructions=[
            "Mathematics Reasoning: Mixed Pressure Set",
            f"Mode: {profile.label}",
            "Real mixed-domain multiple-choice work under the tightest cap profile in this family.",
            "Use fast setup triage, take the best answer, and recover immediately on the next problem.",
            "Press Enter to begin practice.",
        ],
        generator=MrMixedPressureSetGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        practice_questions=practice_questions,
        scored_duration_s=scored_duration_s,
        mode=_normalize_mode(mode),
        base_caps_by_level=(22.0, 21.0, 20.0, 19.0, 18.0, 16.0, 15.0, 14.0, 13.0, 12.0),
        adaptive_config=cfg.adaptive,
        immediate_feedback_override=True,
    )
