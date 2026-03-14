from __future__ import annotations

from dataclasses import dataclass, field

from .ant_drills import (
    ANT_DRILL_MODE_PROFILES,
    AntAdaptiveDifficultyConfig,
    AntDrillMode,
    TimedCapDrill,
)
from .clock import Clock
from .cognitive_core import Phase, Problem
from .table_reading import (
    TableReadingGenerator,
    TableReadingPart,
    TableReadingScorer,
)


PART_ONE_FAMILIES = TableReadingGenerator.supported_part_one_families()
PART_TWO_FAMILIES = TableReadingGenerator.supported_part_two_families()


def _normalize_mode(mode: AntDrillMode | str) -> AntDrillMode:
    if isinstance(mode, AntDrillMode):
        return mode
    return AntDrillMode(str(mode).strip().lower())


@dataclass(frozen=True, slots=True)
class TblDrillConfig:
    practice_questions: int | None = None
    scored_duration_s: float | None = None
    adaptive: AntAdaptiveDifficultyConfig = field(
        default_factory=lambda: AntAdaptiveDifficultyConfig(enabled=False)
    )


class TableReadingTimedDrill(TimedCapDrill):
    def _input_hint(self) -> str:
        if self.phase not in (Phase.PRACTICE, Phase.SCORED):
            return "Press Enter to continue"
        return (
            f"L{self._current_level()} | Cap {self._item_remaining_s():0.1f}s | "
            "Up/Down + A/S/D/F/G or 1-5 then Enter"
        )


class _BaseTableReadingSelectionGenerator:
    def __init__(self, *, seed: int) -> None:
        self._base = TableReadingGenerator(seed=seed)


class TblPart1AnchorGenerator(_BaseTableReadingSelectionGenerator):
    def __init__(self, *, seed: int) -> None:
        super().__init__(seed=seed)
        self._family_index = 0

    def next_problem(self, *, difficulty: float) -> Problem:
        family = PART_ONE_FAMILIES[self._family_index % len(PART_ONE_FAMILIES)]
        self._family_index += 1
        return self._base.next_problem_for_selection(
            difficulty=difficulty,
            part=TableReadingPart.PART_ONE,
            family=family,
            profile="anchor",
        )


class TblPart1ScanRunGenerator(_BaseTableReadingSelectionGenerator):
    def __init__(self, *, seed: int) -> None:
        super().__init__(seed=seed)
        self._family_index = 0

    def next_problem(self, *, difficulty: float) -> Problem:
        family = PART_ONE_FAMILIES[self._family_index % len(PART_ONE_FAMILIES)]
        self._family_index += 1
        return self._base.next_problem_for_selection(
            difficulty=difficulty,
            part=TableReadingPart.PART_ONE,
            family=family,
            profile="scan",
        )


class TblPart2PrimeGenerator(_BaseTableReadingSelectionGenerator):
    def __init__(self, *, seed: int) -> None:
        super().__init__(seed=seed)
        self._family_index = 0

    def next_problem(self, *, difficulty: float) -> Problem:
        family = PART_TWO_FAMILIES[self._family_index % len(PART_TWO_FAMILIES)]
        self._family_index += 1
        return self._base.next_problem_for_selection(
            difficulty=difficulty,
            part=TableReadingPart.PART_TWO,
            family=family,
            profile="prime",
        )


class TblPart2CorrectionRunGenerator(_BaseTableReadingSelectionGenerator):
    def __init__(self, *, seed: int) -> None:
        super().__init__(seed=seed)
        self._family_index = 0

    def next_problem(self, *, difficulty: float) -> Problem:
        family = PART_TWO_FAMILIES[self._family_index % len(PART_TWO_FAMILIES)]
        self._family_index += 1
        return self._base.next_problem_for_selection(
            difficulty=difficulty,
            part=TableReadingPart.PART_TWO,
            family=family,
            profile="run",
        )


class TblPartSwitchRunGenerator(_BaseTableReadingSelectionGenerator):
    def __init__(self, *, seed: int) -> None:
        super().__init__(seed=seed)
        self._problem_index = 0
        self._part_one_family_index = 0
        self._part_two_family_index = 0

    def next_problem(self, *, difficulty: float) -> Problem:
        part = TableReadingPart.PART_ONE if self._problem_index % 2 == 0 else TableReadingPart.PART_TWO
        self._problem_index += 1
        if part is TableReadingPart.PART_ONE:
            family = PART_ONE_FAMILIES[self._part_one_family_index % len(PART_ONE_FAMILIES)]
            self._part_one_family_index += 1
            return self._base.next_problem_for_selection(
                difficulty=difficulty,
                part=part,
                family=family,
                profile="scan",
            )
        family = PART_TWO_FAMILIES[self._part_two_family_index % len(PART_TWO_FAMILIES)]
        self._part_two_family_index += 1
        return self._base.next_problem_for_selection(
            difficulty=difficulty,
            part=part,
            family=family,
            profile="run",
        )


class TblCardFamilyRunGenerator(_BaseTableReadingSelectionGenerator):
    _SEQUENCE = tuple(
        (TableReadingPart.PART_ONE, family) for family in PART_ONE_FAMILIES
    ) + tuple(
        (TableReadingPart.PART_TWO, family) for family in PART_TWO_FAMILIES
    )

    def __init__(self, *, seed: int) -> None:
        super().__init__(seed=seed)
        self._sequence_index = 0

    def next_problem(self, *, difficulty: float) -> Problem:
        part, family = self._SEQUENCE[self._sequence_index % len(self._SEQUENCE)]
        self._sequence_index += 1
        profile = "scan" if part is TableReadingPart.PART_ONE else "run"
        return self._base.next_problem_for_selection(
            difficulty=difficulty,
            part=part,
            family=family,
            profile=profile,
        )


class TblMixedTempoGenerator(_BaseTableReadingSelectionGenerator):
    def __init__(self, *, seed: int) -> None:
        super().__init__(seed=seed)
        self._sequence_index = 0
        self._part_one_family_index = 0
        self._part_two_family_index = 0

    def next_problem(self, *, difficulty: float) -> Problem:
        in_part_one_window = (self._sequence_index % 4) < 2
        self._sequence_index += 1
        if in_part_one_window:
            family = PART_ONE_FAMILIES[self._part_one_family_index % len(PART_ONE_FAMILIES)]
            self._part_one_family_index += 1
            return self._base.next_problem_for_selection(
                difficulty=difficulty,
                part=TableReadingPart.PART_ONE,
                family=family,
                profile="scan",
            )
        family = PART_TWO_FAMILIES[self._part_two_family_index % len(PART_TWO_FAMILIES)]
        self._part_two_family_index += 1
        return self._base.next_problem_for_selection(
            difficulty=difficulty,
            part=TableReadingPart.PART_TWO,
            family=family,
            profile="run",
        )


class TblPressureRunGenerator(_BaseTableReadingSelectionGenerator):
    def __init__(self, *, seed: int) -> None:
        super().__init__(seed=seed)
        self._sequence_index = 0
        self._part_one_family_index = 0
        self._part_two_family_index = 0

    def next_problem(self, *, difficulty: float) -> Problem:
        part = TableReadingPart.PART_ONE if self._sequence_index % 2 == 0 else TableReadingPart.PART_TWO
        self._sequence_index += 1
        if part is TableReadingPart.PART_ONE:
            family = PART_ONE_FAMILIES[self._part_one_family_index % len(PART_ONE_FAMILIES)]
            self._part_one_family_index += 1
            return self._base.next_problem_for_selection(
                difficulty=difficulty,
                part=part,
                family=family,
                profile="pressure",
            )
        family = PART_TWO_FAMILIES[self._part_two_family_index % len(PART_TWO_FAMILIES)]
        self._part_two_family_index += 1
        return self._base.next_problem_for_selection(
            difficulty=difficulty,
            part=part,
            family=family,
            profile="pressure",
        )


def _build_tbl_drill(
    *,
    title_base: str,
    instructions: tuple[str, ...],
    generator: object,
    clock: Clock,
    seed: int,
    difficulty: float,
    mode: AntDrillMode | str,
    config: TblDrillConfig,
    base_caps_by_level: tuple[float, ...],
) -> TableReadingTimedDrill:
    normalized_mode = _normalize_mode(mode)
    profile = ANT_DRILL_MODE_PROFILES[normalized_mode]
    practice_questions = (
        profile.practice_questions if config.practice_questions is None else int(config.practice_questions)
    )
    scored_duration_s = (
        profile.scored_duration_s if config.scored_duration_s is None else float(config.scored_duration_s)
    )
    return TableReadingTimedDrill(
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
        scorer=TableReadingScorer(),
        immediate_feedback_override=True,
    )


def build_tbl_part1_anchor_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: TblDrillConfig | None = None,
) -> TableReadingTimedDrill:
    cfg = config or TblDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_tbl_drill(
        title_base="Table Reading: Part 1 Anchor",
        instructions=(
            "Table Reading: Part 1 Anchor",
            f"Mode: {profile.label}",
            "Stay on single-card row and column lookup with wider option spacing before denser scan work starts.",
            "Keep the live table renderer up, scan cleanly, and answer with A/S/D/F/G or 1-5 then Enter.",
            "Press Enter to begin practice.",
        ),
        generator=TblPart1AnchorGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(34.0, 31.0, 29.0, 27.0, 25.0, 23.0, 21.0, 19.0, 17.0, 15.0),
    )


def build_tbl_part1_scan_run_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: TblDrillConfig | None = None,
) -> TableReadingTimedDrill:
    cfg = config or TblDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_tbl_drill(
        title_base="Table Reading: Part 1 Scan Run",
        instructions=(
            "Table Reading: Part 1 Scan Run",
            f"Mode: {profile.label}",
            "Stay on Part 1 only, but push wider scans and tighter answer spacing across the full card family rotation.",
            "Keep the keyboard flow exact: Up/Down, A/S/D/F/G or 1-5, then Enter.",
            "Press Enter to begin practice.",
        ),
        generator=TblPart1ScanRunGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(28.0, 26.0, 24.0, 22.0, 20.0, 18.0, 16.0, 15.0, 14.0, 13.0),
    )


def build_tbl_part2_prime_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: TblDrillConfig | None = None,
) -> TableReadingTimedDrill:
    cfg = config or TblDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_tbl_drill(
        title_base="Table Reading: Part 2 Prime",
        instructions=(
            "Table Reading: Part 2 Prime",
            f"Mode: {profile.label}",
            "Warm up the two-card chain on easier drift and correction combinations before the full correction run starts.",
            "Use the same live screen and keyboard flow you use in the standalone test.",
            "Press Enter to begin practice.",
        ),
        generator=TblPart2PrimeGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(42.0, 39.0, 36.0, 33.0, 30.0, 27.0, 24.0, 22.0, 20.0, 18.0),
    )


def build_tbl_part2_correction_run_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: TblDrillConfig | None = None,
) -> TableReadingTimedDrill:
    cfg = config or TblDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_tbl_drill(
        title_base="Table Reading: Part 2 Correction Run",
        instructions=(
            "Table Reading: Part 2 Correction Run",
            f"Mode: {profile.label}",
            "Run the full two-card correction workflow with the full table-set rotation active.",
            "Keep the table scan ordered so you do not lose time between the first and second card.",
            "Press Enter to begin practice.",
        ),
        generator=TblPart2CorrectionRunGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(36.0, 33.0, 30.0, 27.0, 24.0, 22.0, 20.0, 18.0, 16.0, 15.0),
    )


def build_tbl_part_switch_run_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: TblDrillConfig | None = None,
) -> TableReadingTimedDrill:
    cfg = config or TblDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_tbl_drill(
        title_base="Table Reading: Part Switch Run",
        instructions=(
            "Table Reading: Part Switch Run",
            f"Mode: {profile.label}",
            "Alternate single-card and two-card items one by one so the workflow shift stops costing time.",
            "Treat the screen the same each time; only the lookup depth changes.",
            "Press Enter to begin practice.",
        ),
        generator=TblPartSwitchRunGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(33.0, 30.0, 28.0, 26.0, 24.0, 22.0, 20.0, 18.0, 16.0, 14.0),
    )


def build_tbl_card_family_run_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: TblDrillConfig | None = None,
) -> TableReadingTimedDrill:
    cfg = config or TblDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_tbl_drill(
        title_base="Table Reading: Card Family Run",
        instructions=(
            "Table Reading: Card Family Run",
            f"Mode: {profile.label}",
            "Cycle through the full card-pack and card-set library so one familiar sheet does not carry the whole block.",
            "Keep the same live renderer and answer flow while the packs rotate under you.",
            "Press Enter to begin practice.",
        ),
        generator=TblCardFamilyRunGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(31.0, 29.0, 27.0, 25.0, 23.0, 21.0, 19.0, 17.0, 15.0, 14.0),
    )


def build_tbl_mixed_tempo_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: TblDrillConfig | None = None,
) -> TableReadingTimedDrill:
    cfg = config or TblDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_tbl_drill(
        title_base="Table Reading: Mixed Tempo",
        instructions=(
            "Table Reading: Mixed Tempo",
            f"Mode: {profile.label}",
            "Run a fixed 2x Part 1, 2x Part 2 rhythm and keep the response flow identical across both sections.",
            "This is still the live Table Reading screen, not a simplified drill layout.",
            "Press Enter to begin practice.",
        ),
        generator=TblMixedTempoGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(29.0, 27.0, 25.0, 23.0, 21.0, 19.0, 17.0, 16.0, 15.0, 14.0),
    )


def build_tbl_pressure_run_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: TblDrillConfig | None = None,
) -> TableReadingTimedDrill:
    cfg = config or TblDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_tbl_drill(
        title_base="Table Reading: Pressure Run",
        instructions=(
            "Table Reading: Pressure Run",
            f"Mode: {profile.label}",
            "Alternate Part 1 and Part 2 under the hardest cap profile while keeping the same partial-credit scoring model.",
            "Do not over-fixate on misses; recover on the next card immediately.",
            "Press Enter to begin practice.",
        ),
        generator=TblPressureRunGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(25.0, 23.0, 21.0, 19.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0),
    )
