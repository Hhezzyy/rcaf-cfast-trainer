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
from .visual_search import (
    VisualSearchGenerator,
    VisualSearchProfile,
    VisualSearchScorer,
    VisualSearchPayload,
    VisualSearchTaskKind,
)


_STANDARD_DRILL_GRID_BY_LEVEL: tuple[tuple[int, int], ...] = (
    (3, 4),
    (3, 4),
    (4, 4),
    (4, 4),
    (4, 5),
    (4, 5),
    (5, 5),
    (5, 6),
    (6, 6),
    (7, 6),
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


def _kind_label(kind: VisualSearchTaskKind) -> str:
    if kind is VisualSearchTaskKind.ALPHANUMERIC:
        return "letters"
    if kind is VisualSearchTaskKind.SYMBOL_CODE:
        return "line figures"
    if kind is VisualSearchTaskKind.WARNING_SIGN:
        return "warning signs"
    return "color codes"


@dataclass(frozen=True, slots=True)
class VsDrillConfig:
    practice_questions: int | None = None
    scored_duration_s: float | None = None
    adaptive: AntAdaptiveDifficultyConfig = field(
        default_factory=lambda: AntAdaptiveDifficultyConfig(enabled=False)
    )


class VsTargetPreviewGenerator:
    def __init__(self, *, seed: int) -> None:
        self._base = VisualSearchGenerator(
            seed=seed,
            profile=VisualSearchProfile(
                similarity_floor=0.0,
                similarity_ceiling=0.52,
                family_switch_floor=0.05,
                family_switch_ceiling=0.30,
                preview_emphasis=True,
                grid_by_level=_STANDARD_DRILL_GRID_BY_LEVEL,
                high_band_symbol_only=False,
            ),
        )

    def next_problem(self, *, difficulty: float):
        return self._base.next_problem(difficulty=difficulty)


class VsCleanScanGenerator:
    def __init__(self, *, seed: int) -> None:
        self._base = VisualSearchGenerator(
            seed=seed,
            profile=VisualSearchProfile(
                similarity_floor=0.05,
                similarity_ceiling=0.66,
                family_switch_floor=0.10,
                family_switch_ceiling=0.40,
                grid_by_level=_STANDARD_DRILL_GRID_BY_LEVEL,
                high_band_symbol_only=False,
            ),
        )

    def next_problem(self, *, difficulty: float):
        return self._base.next_problem(difficulty=difficulty)


class VsFamilyRunGenerator:
    def __init__(self, *, seed: int, kind: VisualSearchTaskKind) -> None:
        self._base = VisualSearchGenerator(
            seed=seed,
            profile=VisualSearchProfile(
                allowed_kinds=(kind,),
                similarity_floor=0.15,
                similarity_ceiling=0.96,
                family_switch_floor=0.0,
                family_switch_ceiling=0.0,
                grid_by_level=_STANDARD_DRILL_GRID_BY_LEVEL,
                high_band_symbol_only=False,
            ),
        )

    def next_problem(self, *, difficulty: float):
        return self._base.next_problem(difficulty=difficulty)


class VsMixedTempoGenerator:
    def __init__(self, *, seed: int) -> None:
        self._index = 0
        self._easy = VisualSearchGenerator(
            seed=seed + 101,
            profile=VisualSearchProfile(
                similarity_floor=0.10,
                similarity_ceiling=0.70,
                family_switch_floor=0.15,
                family_switch_ceiling=0.55,
                grid_by_level=_STANDARD_DRILL_GRID_BY_LEVEL,
                high_band_symbol_only=False,
            ),
        )
        self._hard = VisualSearchGenerator(
            seed=seed + 202,
            profile=VisualSearchProfile(
                similarity_floor=0.55,
                similarity_ceiling=1.00,
                family_switch_floor=0.35,
                family_switch_ceiling=0.90,
                grid_by_level=_STANDARD_DRILL_GRID_BY_LEVEL,
                high_band_symbol_only=False,
            ),
        )

    def next_problem(self, *, difficulty: float):
        self._index += 1
        level = _difficulty_to_level(difficulty)
        if self._index % 3 == 0:
            local_level = min(10, level + 2)
            return self._hard.next_problem(difficulty=_level_to_difficulty(local_level))
        local_level = max(1, level - 2)
        return self._easy.next_problem(difficulty=_level_to_difficulty(local_level))


class VsPressureRunGenerator:
    def __init__(self, *, seed: int) -> None:
        self._base = VisualSearchGenerator(
            seed=seed,
            profile=VisualSearchProfile(
                similarity_floor=0.65,
                similarity_ceiling=1.00,
                family_switch_floor=0.45,
                family_switch_ceiling=0.95,
                grid_by_level=_STANDARD_DRILL_GRID_BY_LEVEL,
                high_band_symbol_only=False,
            ),
        )

    def next_problem(self, *, difficulty: float):
        level = _difficulty_to_level(difficulty)
        local_level = min(10, level + 1)
        return self._base.next_problem(difficulty=_level_to_difficulty(local_level))


class _Wave1SearchGenerator:
    _CLASS_POOL = (
        VisualSearchTaskKind.ALPHANUMERIC,
        VisualSearchTaskKind.SYMBOL_CODE,
        VisualSearchTaskKind.WARNING_SIGN,
        VisualSearchTaskKind.COLOR_PATTERN,
    )

    def __init__(self, *, seed: int) -> None:
        self._rng = SeededRng(seed)
        self._helper = VisualSearchGenerator(seed=seed + 701)
        self._item_index = 0
        self._last_kind: VisualSearchTaskKind | None = None

    def _class_count(self, *, level: int) -> int:
        if level <= 3:
            return 2
        if level <= 7:
            return 3
        return 4

    def _grid_shape(self, *, level: int) -> tuple[int, int]:
        if level <= 3:
            return (3, 4)
        if level <= 5:
            return (4, 5)
        if level <= 7:
            return (5, 5)
        return (5, 6)

    def _salience_level(self, *, level: int) -> float:
        if level <= 3:
            return 0.30
        if level <= 6:
            return 0.65
        return 0.92

    def _target_classes(self, *, level: int) -> tuple[VisualSearchTaskKind, ...]:
        class_count = self._class_count(level=level)
        offset = self._item_index % len(self._CLASS_POOL)
        ordered = self._CLASS_POOL[offset:] + self._CLASS_POOL[:offset]
        return ordered[:class_count]

    def _pick_target_kind(
        self,
        *,
        classes: tuple[VisualSearchTaskKind, ...],
        switch_every: int,
    ) -> VisualSearchTaskKind:
        if self._last_kind is None or self._item_index % max(1, switch_every) == 0:
            if self._last_kind in classes and len(classes) > 1:
                options = tuple(kind for kind in classes if kind is not self._last_kind)
                chosen = options[self._item_index % len(options)]
            else:
                chosen = classes[self._item_index % len(classes)]
            self._last_kind = chosen
        assert self._last_kind is not None
        return self._last_kind

    def _token_bank(self, kind: VisualSearchTaskKind) -> tuple[str, ...]:
        return self._helper._token_bank(kind)

    def _confusable(self, *, kind: VisualSearchTaskKind, target: str) -> tuple[str, ...]:
        return self._helper._confusable_members(kind=kind, target=target)

    def _build_problem(
        self,
        *,
        level: int,
        active_classes: tuple[VisualSearchTaskKind, ...],
        target_kind: VisualSearchTaskKind,
        prompt_prefix: str,
        switch_mode: str,
        priority_label: str,
    ) -> Problem:
        rows, cols = self._grid_shape(level=level)
        cell_count = rows * cols
        target_bank = self._token_bank(target_kind)
        target = str(target_bank[self._item_index % len(target_bank)])
        salience = self._salience_level(level=level)
        confusable = tuple(token for token in self._confusable(kind=target_kind, target=target) if token != target)
        if level >= 9:
            target_variant, overload_cells = self._helper._build_same_base_overload_board(
                kind=target_kind,
                target_base=target,
                count=cell_count,
                level=level,
            )
            target_idx = overload_cells.index(target_variant)
            distractors = [str(token) for token in overload_cells]
            target = target_variant
        elif level >= 8:
            marks = self._helper._variant_marks()
            ordered_bases: list[str] = [target]
            for token in confusable:
                if token not in ordered_bases:
                    ordered_bases.append(str(token))
            for kind in active_classes:
                if kind is target_kind:
                    continue
                bank = self._token_bank(kind)
                if bank and str(bank[0]) not in ordered_bases:
                    ordered_bases.append(str(bank[0]))
            for token in target_bank:
                token = str(token)
                if token in ordered_bases:
                    continue
                ordered_bases.append(token)
                if len(ordered_bases) * len(marks) >= cell_count:
                    break
            for kind in active_classes:
                if kind is target_kind:
                    continue
                for token in self._token_bank(kind):
                    token = str(token)
                    if token in ordered_bases:
                        continue
                    ordered_bases.append(token)
                    if len(ordered_bases) * len(marks) >= cell_count:
                        break
                if len(ordered_bases) * len(marks) >= cell_count:
                    break
            target_variant = self._helper._compose_variant_token(
                target,
                str(marks[self._item_index % len(marks)]),
            )
            variant_pool = [
                self._helper._compose_variant_token(base, mark)
                for base in ordered_bases
                for mark in marks
                if self._helper._compose_variant_token(base, mark) != target_variant
            ]
            distractors = [
                str(token)
                for token in self._rng.sample(tuple(variant_pool), k=max(0, cell_count - 1))
            ]
            target_idx = int(self._rng.randint(0, cell_count - 1))
            distractors.insert(target_idx, target_variant)
            target = target_variant
        else:
            cross_class: list[str] = []
            for kind in active_classes:
                if kind is target_kind:
                    continue
                cross_class.extend(self._token_bank(kind))
            same_class = tuple(token for token in target_bank if token != target)
            fallback_same = confusable or same_class
            distractors = []
            cross_ratio = 0.28 if level <= 3 else 0.44 if level <= 6 else 0.60
            for _ in range(cell_count):
                if cross_class and self._rng.random() < cross_ratio:
                    distractors.append(str(self._rng.choice(cross_class)))
                    continue
                if fallback_same and self._rng.random() < salience:
                    distractors.append(str(self._rng.choice(fallback_same)))
                    continue
                distractors.append(str(self._rng.choice(same_class if same_class else target_bank)))
            target_idx = int(self._rng.randint(0, cell_count - 1))
            distractors[target_idx] = target
        cell_codes = self._helper._build_cell_codes(count=cell_count, rng=self._rng)
        correct_code = int(cell_codes[target_idx])
        self._item_index += 1
        display_target = self._helper.token_base(target)
        return Problem(
            prompt=f"{prompt_prefix} Find {display_target} in the {_kind_label(target_kind)} class and enter its block number.",
            answer=correct_code,
            payload=VisualSearchPayload(
                kind=target_kind,
                rows=rows,
                cols=cols,
                target=target,
                cells=tuple(distractors),
                cell_codes=cell_codes,
                full_credit_error=0,
                zero_credit_error=1,
                class_count=len(active_classes),
                active_classes=tuple(kind.value for kind in active_classes),
                salience_level=salience,
                switch_mode=switch_mode,
                priority_label=priority_label,
            ),
        )


class VsMultiTargetClassSearchGenerator(_Wave1SearchGenerator):
    def next_problem(self, *, difficulty: float):
        level = _difficulty_to_level(difficulty)
        active_classes = self._target_classes(level=level)
        switch_every = 3 if level <= 3 else 2 if level <= 6 else 1
        target_kind = self._pick_target_kind(classes=active_classes, switch_every=switch_every)
        return self._build_problem(
            level=level,
            active_classes=active_classes,
            target_kind=target_kind,
            prompt_prefix=f"Multi-class scan across {len(active_classes)} classes.",
            switch_mode="class_cycle",
            priority_label="routine",
        )


class VsPrioritySwitchSearchGenerator(_Wave1SearchGenerator):
    def next_problem(self, *, difficulty: float):
        level = _difficulty_to_level(difficulty)
        active_classes = self._target_classes(level=level)
        switch_every = 4 if level <= 3 else 2 if level <= 6 else 1
        target_kind = self._pick_target_kind(classes=active_classes, switch_every=switch_every)
        mode = "priority_hold" if self._item_index % max(1, switch_every) else "priority_switch"
        return self._build_problem(
            level=level,
            active_classes=active_classes,
            target_kind=target_kind,
            prompt_prefix=f"Priority search: {_kind_label(target_kind).title()} are live; ignore routine classes.",
            switch_mode=mode,
            priority_label=_kind_label(target_kind),
        )


class VsMatrixRoutinePrioritySwitchGenerator(_Wave1SearchGenerator):
    def next_problem(self, *, difficulty: float):
        level = _difficulty_to_level(difficulty)
        active_classes = self._target_classes(level=level)
        switch_every = 3 if level <= 4 else 2 if level <= 7 else 1
        target_kind = self._pick_target_kind(classes=active_classes, switch_every=switch_every)
        is_priority = (self._item_index % max(1, switch_every)) == 0
        prefix = (
            f"Priority interrupt on the scan matrix. {_kind_label(target_kind).title()} now take priority."
            if is_priority
            else f"Routine matrix sweep. Stay on the {_kind_label(target_kind)} class."
        )
        return self._build_problem(
            level=level,
            active_classes=active_classes,
            target_kind=target_kind,
            prompt_prefix=prefix,
            switch_mode="priority_interrupt" if is_priority else "routine_sweep",
            priority_label="priority" if is_priority else "routine",
        )


def _build_vs_drill(
    *,
    title_base: str,
    instructions: tuple[str, ...],
    generator: object,
    clock: Clock,
    seed: int,
    difficulty: float,
    mode: AntDrillMode | str,
    config: VsDrillConfig,
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
        scorer=VisualSearchScorer(),
        immediate_feedback_override=True,
    )


def build_vs_target_preview_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: VsDrillConfig | None = None,
) -> TimedCapDrill:
    cfg = config or VsDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_vs_drill(
        title_base="Visual Search: Target Preview",
        instructions=(
            "Visual Search: Target Preview",
            f"Mode: {profile.label}",
            "Use the same typed answer flow as the test, but give yourself a clean beat to reacquire the target before scanning.",
            "Preview the target, then search row by row and type the matching block number without rushing the first eye movement.",
            "Press Enter to begin practice.",
        ),
        generator=VsTargetPreviewGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(18.0, 16.5, 15.0, 13.5, 12.0, 11.0, 10.0, 9.0, 8.5, 8.0),
    )


def build_vs_clean_scan_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: VsDrillConfig | None = None,
) -> TimedCapDrill:
    cfg = config or VsDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_vs_drill(
        title_base="Visual Search: Clean Scan",
        instructions=(
            "Visual Search: Clean Scan",
            f"Mode: {profile.label}",
            "Run the full board with easier distractors and focus on a disciplined scan pattern instead of guessing.",
            "This block is for building a stable left-to-right or row-by-row search rhythm before the family runs tighten up.",
            "Press Enter to begin practice.",
        ),
        generator=VsCleanScanGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(16.0, 14.5, 13.5, 12.5, 11.5, 10.0, 9.0, 8.0, 7.5, 7.0),
    )


def build_vs_family_run_drill(
    *,
    clock: Clock,
    seed: int,
    kind: VisualSearchTaskKind,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.TEMPO,
    config: VsDrillConfig | None = None,
) -> TimedCapDrill:
    cfg = config or VsDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    if kind is VisualSearchTaskKind.ALPHANUMERIC:
        label = "Letter Family Run"
        target_label = "letters"
    else:
        label = "Line Figure Family Run"
        target_label = "line figures"
    return _build_vs_drill(
        title_base=f"Visual Search: {label}",
        instructions=(
            f"Visual Search: {label}",
            f"Mode: {profile.label}",
            f"Stay on {target_label} only so the full-board scan gets faster before you mix families again.",
            "Type the numbered block cleanly and let the next-item flash correct any miss without stopping the run.",
            "Press Enter to begin practice.",
        ),
        generator=VsFamilyRunGenerator(seed=seed, kind=kind),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.5),
    )


def build_vs_mixed_tempo_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.TEMPO,
    config: VsDrillConfig | None = None,
) -> TimedCapDrill:
    cfg = config or VsDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_vs_drill(
        title_base="Visual Search: Mixed Tempo",
        instructions=(
            "Visual Search: Mixed Tempo",
            f"Mode: {profile.label}",
            "Mix letters and line figures with an easy-easy-hard cadence so one bad miss does not break the next search.",
            "Stay honest about your scan rhythm and recover immediately when the distractors get tighter.",
            "Press Enter to begin practice.",
        ),
        generator=VsMixedTempoGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.5, 5.0, 4.5),
    )


def build_vs_pressure_run_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.STRESS,
    config: VsDrillConfig | None = None,
) -> TimedCapDrill:
    cfg = config or VsDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_vs_drill(
        title_base="Visual Search: Pressure Run",
        instructions=(
            "Visual Search: Pressure Run",
            f"Mode: {profile.label}",
            "Keep the real mixed-board answer mode, but tighten the caps and pull distractors closer to the target cluster.",
            "Use this as the late-workout pressure block: search, commit, and recover immediately from misses.",
            "Press Enter to begin practice.",
        ),
        generator=VsPressureRunGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(9.0, 8.5, 8.0, 7.5, 7.0, 6.0, 5.5, 4.8, 4.2, 3.8),
    )


def build_vs_multi_target_class_search_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: VsDrillConfig | None = None,
) -> TimedCapDrill:
    cfg = config or VsDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_vs_drill(
        title_base="Visual Search: Multi-Target Class Search",
        instructions=(
            "Visual Search: Multi-Target Class Search",
            f"Mode: {profile.label}",
            "Mixed classes share the same board, but only one class is currently live.",
            "As level rises, more classes appear, same-class distractors get closer, and the cap tightens.",
            "Press Enter to begin practice.",
        ),
        generator=VsMultiTargetClassSearchGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0),
    )


def build_vs_priority_switch_search_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.TEMPO,
    config: VsDrillConfig | None = None,
) -> TimedCapDrill:
    cfg = config or VsDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_vs_drill(
        title_base="Visual Search: Priority Switch Search",
        instructions=(
            "Visual Search: Priority Switch Search",
            f"Mode: {profile.label}",
            "Routine classes remain on the board while the live priority class changes underneath you.",
            "Higher levels raise switch frequency, distractor density, and cap pressure without changing the typed answer flow.",
            "Press Enter to begin practice.",
        ),
        generator=VsPrioritySwitchSearchGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.5, 6.0),
    )


def build_vs_matrix_routine_priority_switch_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.STRESS,
    config: VsDrillConfig | None = None,
) -> TimedCapDrill:
    cfg = config or VsDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_vs_drill(
        title_base="Visual Search: Matrix Routine/Priority Switch",
        instructions=(
            "Visual Search: Matrix Routine/Priority Switch",
            f"Mode: {profile.label}",
            "Sweep the matrix in routine mode until a priority interrupt changes the live class.",
            "Later levels widen the matrix, reduce salience, and force faster routine/priority switching.",
            "Press Enter to begin practice.",
        ),
        generator=VsMatrixRoutinePrioritySwitchGenerator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.5, 6.0, 5.5),
    )
