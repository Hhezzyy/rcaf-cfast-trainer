from __future__ import annotations

from dataclasses import dataclass, field, replace

from .ant_drills import (
    ANT_DRILL_MODE_PROFILES,
    AntAdaptiveDifficultyConfig,
    AntDrillMode,
    TimedCapDrill,
)
from .clock import Clock
from .cognitive_core import Phase, Problem
from .system_logic import (
    SystemLogicGenerator,
    SystemLogicPayload,
    SystemLogicScorer,
    canonical_system_logic_reasoning_family,
)


SL_FAMILY_CYCLE = ("oil", "fuel", "electrical", "hydraulic", "thermal")
SL_REASONING_CYCLE = ("quantitative", "trace", "graph_rule", "diagnosis")


def _normalize_mode(mode: AntDrillMode | str) -> AntDrillMode:
    if isinstance(mode, AntDrillMode):
        return mode
    return AntDrillMode(str(mode).strip().lower())


@dataclass(frozen=True, slots=True)
class SlDrillConfig:
    practice_questions: int | None = None
    scored_duration_s: float | None = None
    adaptive: AntAdaptiveDifficultyConfig = field(
        default_factory=lambda: AntAdaptiveDifficultyConfig(enabled=False)
    )


class SystemLogicTimedDrill(TimedCapDrill):
    def _input_hint(self) -> str:
        if self.phase not in (Phase.PRACTICE, Phase.SCORED):
            return "Press Enter to continue"
        return (
            f"L{self._current_level()} | Cap {self._item_remaining_s():0.1f}s | "
            "Up/Down + A-E or 1-5 then Enter"
        )


class _SystemLogicSingleReasoningGenerator(SystemLogicGenerator):
    _reasoning_family = "quantitative"

    def __init__(self, *, seed: int) -> None:
        super().__init__(seed=seed)
        self._family_index = 0
        self._families = SystemLogicGenerator.supported_families_for_reasoning_family(
            self._reasoning_family
        )

    def next_problem(self, *, difficulty: float) -> Problem:
        family = self._families[self._family_index % len(self._families)]
        self._family_index += 1
        return self.next_problem_for_selection(
            difficulty=difficulty,
            family=family,
            reasoning_family=self._reasoning_family,
        )


class SlQuantitativeAnchorGenerator(_SystemLogicSingleReasoningGenerator):
    _reasoning_family = "quantitative"


class SlFlowTraceAnchorGenerator(_SystemLogicSingleReasoningGenerator):
    _reasoning_family = "trace"


class SlGraphRuleAnchorGenerator(_SystemLogicSingleReasoningGenerator):
    _reasoning_family = "graph_rule"


class SlFaultDiagnosisPrimeGenerator(_SystemLogicSingleReasoningGenerator):
    _reasoning_family = "diagnosis"


class SlIndexSwitchRunGenerator(SystemLogicGenerator):
    def __init__(self, *, seed: int) -> None:
        super().__init__(seed=seed)
        self._family_index = 0

    def next_problem(self, *, difficulty: float) -> Problem:
        family = SL_FAMILY_CYCLE[self._family_index % len(SL_FAMILY_CYCLE)]
        self._family_index += 1
        return self.next_problem_for_selection(difficulty=difficulty, family=family)


class SlFamilyRunGenerator(SystemLogicGenerator):
    def __init__(self, *, seed: int) -> None:
        super().__init__(seed=seed)
        self._family_index = 0

    def next_problem(self, *, difficulty: float) -> Problem:
        family = SL_FAMILY_CYCLE[self._family_index % len(SL_FAMILY_CYCLE)]
        self._family_index += 1
        return self.next_problem_for_selection(difficulty=difficulty, family=family)


class SlMixedTempoGenerator(SystemLogicGenerator):
    def __init__(self, *, seed: int) -> None:
        super().__init__(seed=seed)
        self._reasoning_index = 0
        self._family_cursor = 0

    def next_problem(self, *, difficulty: float) -> Problem:
        reasoning = SL_REASONING_CYCLE[self._reasoning_index % len(SL_REASONING_CYCLE)]
        self._reasoning_index += 1
        family = self._next_family_for_reasoning(reasoning)
        return self.next_problem_for_selection(
            difficulty=difficulty,
            family=family,
            reasoning_family=reasoning,
        )

    def _next_family_for_reasoning(self, reasoning: str) -> str:
        supported = set(SystemLogicGenerator.supported_families_for_reasoning_family(reasoning))
        for _ in range(len(SL_FAMILY_CYCLE)):
            family = SL_FAMILY_CYCLE[self._family_cursor % len(SL_FAMILY_CYCLE)]
            self._family_cursor += 1
            if family in supported:
                return family
        return next(iter(supported))


class SlPressureRunGenerator(SystemLogicGenerator):
    _SEQUENCE = (
        ("fuel", "quantitative"),
        ("electrical", "trace"),
        ("hydraulic", "graph_rule"),
        ("oil", "diagnosis"),
        ("thermal", "graph_rule"),
        ("electrical", "quantitative"),
        ("fuel", "trace"),
        ("hydraulic", "diagnosis"),
        ("oil", "quantitative"),
        ("thermal", "diagnosis"),
    )

    def __init__(self, *, seed: int) -> None:
        super().__init__(seed=seed)
        self._sequence_index = 0

    def next_problem(self, *, difficulty: float) -> Problem:
        family, reasoning = self._SEQUENCE[self._sequence_index % len(self._SEQUENCE)]
        self._sequence_index += 1
        return self.next_problem_for_selection(
            difficulty=difficulty,
            family=family,
            reasoning_family=reasoning,
        )


class SlOneRuleIdentifyGenerator(_SystemLogicSingleReasoningGenerator):
    _reasoning_family = "graph_rule"


class SlMissingStepCompleteGenerator(_SystemLogicSingleReasoningGenerator):
    _reasoning_family = "trace"


class SlTwoSourceReconcileGenerator(SystemLogicGenerator):
    _SEQUENCE = ("quantitative", "diagnosis", "graph_rule", "diagnosis")

    def __init__(self, *, seed: int) -> None:
        super().__init__(seed=seed)
        self._reasoning_index = 0
        self._family_cursor = 0

    def next_problem(self, *, difficulty: float) -> Problem:
        reasoning = self._SEQUENCE[self._reasoning_index % len(self._SEQUENCE)]
        self._reasoning_index += 1
        family = self._next_family(reasoning)
        return self.next_problem_for_selection(
            difficulty=difficulty,
            family=family,
            reasoning_family=reasoning,
        )

    def _next_family(self, reasoning: str) -> str:
        supported = SystemLogicGenerator.supported_families_for_reasoning_family(reasoning)
        family = supported[self._family_cursor % len(supported)]
        self._family_cursor += 1
        return family


class SlRuleMatchGenerator(SystemLogicGenerator):
    _SEQUENCE = ("graph_rule", "quantitative", "graph_rule", "trace")

    def __init__(self, *, seed: int) -> None:
        super().__init__(seed=seed)
        self._reasoning_index = 0
        self._family_cursor = 0

    def next_problem(self, *, difficulty: float) -> Problem:
        reasoning = self._SEQUENCE[self._reasoning_index % len(self._SEQUENCE)]
        self._reasoning_index += 1
        supported = SystemLogicGenerator.supported_families_for_reasoning_family(reasoning)
        family = supported[self._family_cursor % len(supported)]
        self._family_cursor += 1
        return self.next_problem_for_selection(
            difficulty=difficulty,
            family=family,
            reasoning_family=reasoning,
        )


class SlFastRejectGenerator(SystemLogicGenerator):
    _SEQUENCE = ("diagnosis", "graph_rule", "diagnosis", "quantitative")

    def __init__(self, *, seed: int) -> None:
        super().__init__(seed=seed)
        self._reasoning_index = 0
        self._family_cursor = 0

    def next_problem(self, *, difficulty: float) -> Problem:
        reasoning = self._SEQUENCE[self._reasoning_index % len(self._SEQUENCE)]
        self._reasoning_index += 1
        supported = SystemLogicGenerator.supported_families_for_reasoning_family(reasoning)
        family = supported[self._family_cursor % len(supported)]
        self._family_cursor += 1
        problem = self.next_problem_for_selection(
            difficulty=difficulty,
            family=family,
            reasoning_family=reasoning,
        )
        return _fast_reject_variant(problem)


def _shared_word_score(*, correct_text: str, candidate_text: str) -> tuple[int, int, str]:
    correct_words = {
        token for token in str(correct_text).lower().replace("/", " ").replace("-", " ").split()
    }
    candidate_words = {
        token for token in str(candidate_text).lower().replace("/", " ").replace("-", " ").split()
    }
    shared = len(correct_words & candidate_words)
    return (shared, len(candidate_words), str(candidate_text))


def _fast_reject_variant(problem: Problem) -> Problem:
    payload = problem.payload
    if not isinstance(payload, SystemLogicPayload):
        return problem
    correct_text = next(
        (
            choice.text
            for choice in payload.answer_choices
            if int(choice.code) == int(payload.correct_choice_code)
        ),
        "",
    )
    ordered = tuple(
        sorted(
            payload.answer_choices,
            key=lambda choice: (
                int(choice.code) == int(payload.correct_choice_code),
                _shared_word_score(correct_text=correct_text, candidate_text=choice.text),
            ),
            reverse=True,
        )
    )
    prompt = (
        f"{problem.prompt}\n"
        "Reject the tempting distractor and choose the single answer fully supported by the guide."
    )
    return Problem(
        prompt=prompt,
        answer=problem.answer,
        payload=replace(payload, answer_choices=ordered),
    )


def _build_sl_drill(
    *,
    title_base: str,
    instructions: tuple[str, ...],
    generator: SystemLogicGenerator,
    clock: Clock,
    seed: int,
    difficulty: float,
    mode: AntDrillMode | str,
    config: SlDrillConfig,
    base_caps_by_level: tuple[float, ...],
) -> SystemLogicTimedDrill:
    normalized_mode = _normalize_mode(mode)
    profile = ANT_DRILL_MODE_PROFILES[normalized_mode]
    practice_questions = (
        profile.practice_questions if config.practice_questions is None else int(config.practice_questions)
    )
    scored_duration_s = (
        profile.scored_duration_s if config.scored_duration_s is None else float(config.scored_duration_s)
    )
    return SystemLogicTimedDrill(
        title=f"{title_base} ({profile.label})",
        instructions=instructions,
        generator=generator,
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        practice_questions=practice_questions,
        scored_duration_s=scored_duration_s,
        mode=normalized_mode,
        base_caps_by_level=base_caps_by_level,
        adaptive_config=config.adaptive,
        scorer=SystemLogicScorer(),
        immediate_feedback_override=True,
    )


def build_sl_quantitative_anchor_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: SlDrillConfig | None = None,
) -> SystemLogicTimedDrill:
    cfg = config or SlDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_sl_drill(
        title_base="System Logic: Quantitative Anchor",
        instructions=(
            "System Logic: Quantitative Anchor",
            f"Mode: {profile.label}",
            "Focus on duration, capacity, and resource items built from tables, facts, and equations.",
            "Keep the guide layout active and use the index deliberately before you answer.",
            "Press Enter to begin practice.",
        ),
        generator=SlQuantitativeAnchorGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(52.0, 48.0, 45.0, 42.0, 39.0, 36.0, 33.0, 30.0, 27.0, 24.0),
    )


def build_sl_flow_trace_anchor_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: SlDrillConfig | None = None,
) -> SystemLogicTimedDrill:
    cfg = config or SlDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_sl_drill(
        title_base="System Logic: Flow Trace Anchor",
        instructions=(
            "System Logic: Flow Trace Anchor",
            f"Mode: {profile.label}",
            "Follow feed paths, dependencies, and transfer routes through diagrams and facts first.",
            "Use the index to verify every step instead of guessing from one pane.",
            "Press Enter to begin practice.",
        ),
        generator=SlFlowTraceAnchorGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(50.0, 46.0, 43.0, 40.0, 37.0, 34.0, 31.0, 28.0, 25.0, 22.0),
    )


def build_sl_graph_rule_anchor_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: SlDrillConfig | None = None,
) -> SystemLogicTimedDrill:
    cfg = config or SlDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_sl_drill(
        title_base="System Logic: Graph + Rule Anchor",
        instructions=(
            "System Logic: Graph + Rule Anchor",
            f"Mode: {profile.label}",
            "Train graph reading and rule application before mixed diagnosis and family switching return.",
            "Use the graph banding and equation pane together; neither one is enough alone.",
            "Press Enter to begin practice.",
        ),
        generator=SlGraphRuleAnchorGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(52.0, 48.0, 45.0, 42.0, 39.0, 36.0, 33.0, 30.0, 27.0, 24.0),
    )


def build_sl_fault_diagnosis_prime_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: SlDrillConfig | None = None,
) -> SystemLogicTimedDrill:
    cfg = config or SlDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_sl_drill(
        title_base="System Logic: Fault Diagnosis Prime",
        instructions=(
            "System Logic: Fault Diagnosis Prime",
            f"Mode: {profile.label}",
            "Use mixed-source advisories and fault states without the faster mixed-workout tempo yet.",
            "Choose the single best diagnosis, not every symptom that happens to be visible.",
            "Press Enter to begin practice.",
        ),
        generator=SlFaultDiagnosisPrimeGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(54.0, 50.0, 47.0, 44.0, 41.0, 38.0, 35.0, 32.0, 29.0, 26.0),
    )


def build_sl_index_switch_run_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.TEMPO,
    config: SlDrillConfig | None = None,
) -> SystemLogicTimedDrill:
    cfg = config or SlDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_sl_drill(
        title_base="System Logic: Index Switch Run",
        instructions=(
            "System Logic: Index Switch Run",
            f"Mode: {profile.label}",
            "Every item is built so you must move the index and collate more than one subsystem view.",
            "Stay with the live two-pane System Logic layout and keep the index moving deliberately.",
            "Press Enter to begin practice.",
        ),
        generator=SlIndexSwitchRunGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(46.0, 43.0, 40.0, 37.0, 34.0, 31.0, 29.0, 27.0, 24.0, 22.0),
    )


def build_sl_family_run_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.TEMPO,
    config: SlDrillConfig | None = None,
) -> SystemLogicTimedDrill:
    cfg = config or SlDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_sl_drill(
        title_base="System Logic: Family Run",
        instructions=(
            "System Logic: Family Run",
            f"Mode: {profile.label}",
            "Rotate through oil, fuel, electrical, hydraulic, and thermal scenarios in a fixed order.",
            "Use it to stop family switches from costing extra orientation time.",
            "Press Enter to begin practice.",
        ),
        generator=SlFamilyRunGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(45.0, 42.0, 39.0, 36.0, 33.0, 30.0, 28.0, 26.0, 23.0, 21.0),
    )


def build_sl_mixed_tempo_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.TEMPO,
    config: SlDrillConfig | None = None,
) -> SystemLogicTimedDrill:
    cfg = config or SlDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_sl_drill(
        title_base="System Logic: Mixed Tempo",
        instructions=(
            "System Logic: Mixed Tempo",
            f"Mode: {profile.label}",
            "Reasoning rhythm is fixed: quantitative, trace, graph/rule, then diagnosis, repeating.",
            "Families rotate under that rhythm so the reasoning switch costs less time each block.",
            "Press Enter to begin practice.",
        ),
        generator=SlMixedTempoGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(42.0, 39.0, 36.0, 33.0, 31.0, 29.0, 27.0, 25.0, 23.0, 21.0),
    )


def build_sl_pressure_run_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.STRESS,
    config: SlDrillConfig | None = None,
) -> SystemLogicTimedDrill:
    cfg = config or SlDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_sl_drill(
        title_base="System Logic: Pressure Run",
        instructions=(
            "System Logic: Pressure Run",
            f"Mode: {profile.label}",
            "Hardest mixed run: all five system families and all reasoning modes stay active.",
            "Keep the guide-style two-pane flow stable even when the family and reasoning mode change every item.",
            "Press Enter to begin practice.",
        ),
        generator=SlPressureRunGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(38.0, 36.0, 34.0, 32.0, 30.0, 28.0, 26.0, 24.0, 22.0, 20.0),
    )


def build_sl_one_rule_identify_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: SlDrillConfig | None = None,
) -> SystemLogicTimedDrill:
    cfg = config or SlDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_sl_drill(
        title_base="System Logic: One-Rule Identify",
        instructions=(
            "System Logic: One-Rule Identify",
            f"Mode: {profile.label}",
            "Stay on the single rule or graph relation that unlocks the answer before mixed reasoning returns.",
            "Use the existing two-pane guide layout, but identify the decisive rule quickly and cleanly.",
            "Press Enter to begin practice.",
        ),
        generator=SlOneRuleIdentifyGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(50.0, 47.0, 44.0, 41.0, 38.0, 35.0, 32.0, 29.0, 26.0, 23.0),
    )


def build_sl_missing_step_complete_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: SlDrillConfig | None = None,
) -> SystemLogicTimedDrill:
    cfg = config or SlDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_sl_drill(
        title_base="System Logic: Missing-Step Complete",
        instructions=(
            "System Logic: Missing-Step Complete",
            f"Mode: {profile.label}",
            "Focus on the one missing transfer or dependency step instead of the entire mixed scenario.",
            "Use the index deliberately and verify the step sequence before you answer.",
            "Press Enter to begin practice.",
        ),
        generator=SlMissingStepCompleteGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(48.0, 45.0, 42.0, 39.0, 36.0, 33.0, 30.0, 27.0, 24.0, 22.0),
    )


def build_sl_two_source_reconcile_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.TEMPO,
    config: SlDrillConfig | None = None,
) -> SystemLogicTimedDrill:
    cfg = config or SlDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_sl_drill(
        title_base="System Logic: Two-Source Reconcile",
        instructions=(
            "System Logic: Two-Source Reconcile",
            f"Mode: {profile.label}",
            "Reconcile two guide sources quickly instead of over-reading the whole index.",
            "This block exists to make cross-checking table, rule, and fault evidence feel automatic.",
            "Press Enter to begin practice.",
        ),
        generator=SlTwoSourceReconcileGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(46.0, 43.0, 40.0, 37.0, 34.0, 31.0, 29.0, 27.0, 25.0, 23.0),
    )


def build_sl_rule_match_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.TEMPO,
    config: SlDrillConfig | None = None,
) -> SystemLogicTimedDrill:
    cfg = config or SlDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_sl_drill(
        title_base="System Logic: Rule Match",
        instructions=(
            "System Logic: Rule Match",
            f"Mode: {profile.label}",
            "Match the diagram, table, and rule relationship fast enough that the right pane stops feeling like extra overhead.",
            "Keep the same A-E answer flow; only the reasoning profile gets narrower.",
            "Press Enter to begin practice.",
        ),
        generator=SlRuleMatchGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(46.0, 43.0, 40.0, 37.0, 34.0, 31.0, 29.0, 27.0, 24.0, 22.0),
    )


def build_sl_fast_reject_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.STRESS,
    config: SlDrillConfig | None = None,
) -> SystemLogicTimedDrill:
    cfg = config or SlDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_sl_drill(
        title_base="System Logic: Fast Reject",
        instructions=(
            "System Logic: Fast Reject",
            f"Mode: {profile.label}",
            "Bias toward tempting distractors and force yourself to reject them without rereading the whole guide.",
            "Choose the single supported answer fast and move on.",
            "Press Enter to begin practice.",
        ),
        generator=SlFastRejectGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(42.0, 39.0, 36.0, 33.0, 31.0, 29.0, 27.0, 25.0, 23.0, 21.0),
    )


def canonical_reasoning_family_for_payload(payload) -> str:
    return canonical_system_logic_reasoning_family(getattr(payload, "reasoning_mode", ""))
