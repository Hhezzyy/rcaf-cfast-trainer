from __future__ import annotations

from dataclasses import dataclass, field

from .ant_drills import (
    ANT_DRILL_MODE_PROFILES,
    AntAdaptiveDifficultyConfig,
    AntDrillMode,
    TimedCapDrill,
)
from .clock import Clock
from .cognitive_core import Phase, Problem, clamp01, lerp_int
from .target_recognition import (
    TargetRecognitionGenerator,
    TargetRecognitionPayload,
    TargetRecognitionSceneCriteria,
    TargetRecognitionSceneEntity,
    TargetRecognitionScorer,
)


TR_PANEL_ORDER = ("scene", "light", "scan", "system")


def _normalize_mode(mode: AntDrillMode | str) -> AntDrillMode:
    if isinstance(mode, AntDrillMode):
        return mode
    return AntDrillMode(str(mode).strip().lower())


def _difficulty_to_level(difficulty: float) -> int:
    return max(1, min(10, int(round(clamp01(difficulty) * 9.0)) + 1))


@dataclass(frozen=True, slots=True)
class TrDrillConfig:
    practice_questions: int | None = None
    scored_duration_s: float | None = None
    adaptive: AntAdaptiveDifficultyConfig = field(
        default_factory=lambda: AntAdaptiveDifficultyConfig(enabled=False)
    )


class TargetRecognitionTimedDrill(TimedCapDrill):
    def _input_hint(self) -> str:
        if self.phase not in (Phase.PRACTICE, Phase.SCORED):
            return "Press Enter to continue"
        return (
            f"L{self._current_level()} | Cap {self._item_remaining_s():0.1f}s | "
            "Mouse only: click active panel matches"
        )


class _TargetRecognitionTrainingGenerator(TargetRecognitionGenerator):
    _SIMPLE_SCAN_TOKENS = ("<>", "[]", "/\\", "()", "==", "{}")

    def __init__(
        self,
        *,
        seed: int,
        mode: AntDrillMode | str,
        active_panel_sequence: tuple[tuple[str, ...], ...],
        scene_mode: str = "full",
        scan_pool_mode: str = "full",
        cadence_style: str = "steady",
    ) -> None:
        super().__init__(seed=seed)
        self._mode = _normalize_mode(mode)
        self._active_panel_sequence = tuple(
            tuple(panel for panel in TR_PANEL_ORDER if panel in seq)
            for seq in active_panel_sequence
        )
        self._scene_mode = str(scene_mode)
        self._scan_pool_mode = str(scan_pool_mode)
        self._cadence_style = str(cadence_style)
        self._problem_index = 0

    def next_problem(self, *, difficulty: float) -> Problem:
        d = clamp01(difficulty)
        active_panels = self._active_panel_sequence[self._problem_index % len(self._active_panel_sequence)]
        self._problem_index += 1
        presence_prob = self._presence_probability(d)

        scene_rows = lerp_int(5, 7, d)
        scene_cols = lerp_int(8, 11, d)
        scene_has_target = self._rng.random() < presence_prob
        if self._scene_mode == "basic":
            (
                scene_entities,
                scene_target_criteria,
                scene_target_options,
                scene_cells,
            ) = self._build_scene_entities_basic(
                rows=scene_rows,
                cols=scene_cols,
                has_target=scene_has_target,
                difficulty=d,
            )
        else:
            (
                scene_entities,
                scene_target_criteria,
                scene_target_options,
                scene_cells,
            ) = self._build_scene_entities(
                rows=scene_rows,
                cols=scene_cols,
                has_target=scene_has_target,
                difficulty=d,
            )
        scene_target = self._scene_criteria_label(scene_target_criteria)

        light_target_pattern = self._build_light_pattern()
        light_has_target = self._rng.random() < presence_prob
        light_pattern = (
            light_target_pattern
            if light_has_target
            else self._build_different_light_pattern(light_target_pattern)
        )

        scan_pool = self._scan_token_pool_for_level(_difficulty_to_level(d))
        scan_target = str(self._rng.choice(scan_pool))
        scan_has_target = self._rng.random() < presence_prob
        scan_tokens = self._build_scan_tokens_from_pool(
            count=lerp_int(10, 16, d),
            target=scan_target,
            has_target=scan_has_target,
            pool=scan_pool,
        )

        system_row_count = lerp_int(16, 22, d)
        system_target = self._random_alnum_code(length=4)
        system_has_target = True
        system_cycles = self._build_system_cycles(
            row_count=system_row_count,
            first_target=system_target,
            cycle_count=lerp_int(3, 5, d),
        )
        system_rows = system_cycles[0].columns[0] if system_cycles else ()

        payload = TargetRecognitionPayload(
            scene_rows=scene_rows,
            scene_cols=scene_cols,
            scene_cells=scene_cells,
            scene_entities=scene_entities,
            scene_target=scene_target,
            scene_has_target=scene_has_target,
            scene_target_options=scene_target_options,
            light_pattern=light_pattern,
            light_target_pattern=light_target_pattern,
            light_has_target=light_has_target,
            scan_tokens=scan_tokens,
            scan_target=scan_target,
            scan_has_target=scan_has_target,
            system_rows=system_rows,
            system_target=system_target,
            system_has_target=system_has_target,
            system_cycles=system_cycles,
            system_step_interval_s=self._system_step_interval_s_for(d),
            full_credit_error=0,
            zero_credit_error=3,
            active_panels=active_panels,
            light_interval_range_s=self._light_interval_range_s_for(d),
            scan_interval_range_s=self._scan_interval_range_s_for(d),
            scan_repeat_range=self._scan_repeat_range_for(d),
        )

        expected_matches = sum(
            (
                int("scene" in active_panels and scene_has_target),
                int("light" in active_panels and light_has_target),
                int("scan" in active_panels and scan_has_target),
                int("system" in active_panels and system_has_target),
            )
        )

        return Problem(
            prompt=self._prompt_for_active_panels(active_panels),
            answer=expected_matches,
            payload=payload,
        )

    def _build_scene_entities_basic(
        self,
        *,
        rows: int,
        cols: int,
        has_target: bool,
        difficulty: float,
    ) -> tuple[
        tuple[TargetRecognitionSceneEntity, ...],
        TargetRecognitionSceneCriteria,
        tuple[str, ...],
        tuple[str, ...],
    ]:
        count = max(1, int(rows * cols))
        entities = tuple(
            TargetRecognitionSceneEntity(
                shape=str(self._rng.choice(self._SCENE_SHAPES)),
                affiliation=str(self._rng.choice(self._SCENE_AFFILIATIONS)),
                damaged=False,
                high_priority=False,
            )
            for _ in range(count)
        )
        present = self._scene_present_candidates_basic(entities=entities)
        if has_target:
            criteria = present[int(self._rng.randint(0, len(present) - 1))]
            pool = present
        else:
            absent = self._scene_absent_candidates_basic(entities=entities)
            criteria = absent[int(self._rng.randint(0, len(absent) - 1))]
            pool = absent

        labels = [self._scene_criteria_label(criteria)]
        seen = set(labels)
        target_count = max(1, lerp_int(4, 5, difficulty))
        for candidate in pool:
            label = self._scene_criteria_label(candidate)
            if label in seen:
                continue
            labels.append(label)
            seen.add(label)
            if len(labels) >= target_count:
                break
        cells = tuple(self._scene_entity_code(entity) for entity in entities)
        return entities, criteria, tuple(labels), cells

    def _scene_present_candidates_basic(
        self, *, entities: tuple[TargetRecognitionSceneEntity, ...]
    ) -> tuple[TargetRecognitionSceneCriteria, ...]:
        dedup: dict[tuple[str, str], TargetRecognitionSceneCriteria] = {}
        for entity in entities:
            candidate = TargetRecognitionSceneCriteria(
                shape=entity.shape,
                affiliation=entity.affiliation,
                require_damaged=None,
                require_high_priority=None,
            )
            dedup[(candidate.shape, candidate.affiliation)] = candidate
        candidates = list(dedup.values())
        for idx in range(len(candidates) - 1, 0, -1):
            swap = int(self._rng.randint(0, idx))
            candidates[idx], candidates[swap] = candidates[swap], candidates[idx]
        return tuple(candidates)

    def _scene_absent_candidates_basic(
        self, *, entities: tuple[TargetRecognitionSceneEntity, ...]
    ) -> tuple[TargetRecognitionSceneCriteria, ...]:
        present = {(entity.shape, entity.affiliation) for entity in entities}
        candidates = [
            TargetRecognitionSceneCriteria(
                shape=shape,
                affiliation=affiliation,
                require_damaged=None,
                require_high_priority=None,
            )
            for shape in self._SCENE_SHAPES
            for affiliation in self._SCENE_AFFILIATIONS
            if (shape, affiliation) not in present
        ]
        for idx in range(len(candidates) - 1, 0, -1):
            swap = int(self._rng.randint(0, idx))
            candidates[idx], candidates[swap] = candidates[swap], candidates[idx]
        if candidates:
            return tuple(candidates)
        return (
            TargetRecognitionSceneCriteria(
                shape="truck",
                affiliation="hostile",
                require_damaged=None,
                require_high_priority=None,
            ),
        )

    def _scan_token_pool_for_level(self, level: int) -> tuple[str, ...]:
        if self._scan_pool_mode != "simple":
            return tuple(self._SCAN_TOKENS)
        if level <= 3:
            return self._SIMPLE_SCAN_TOKENS[:4]
        if level <= 6:
            return self._SIMPLE_SCAN_TOKENS
        return tuple(dict.fromkeys((*self._SIMPLE_SCAN_TOKENS, *self._SCAN_TOKENS)))

    def _build_scan_tokens_from_pool(
        self,
        *,
        count: int,
        target: str,
        has_target: bool,
        pool: tuple[str, ...],
    ) -> tuple[str, ...]:
        count = max(1, int(count))
        distractors = tuple(token for token in pool if token != target)
        tokens = [str(self._rng.choice(distractors)) for _ in range(count)]
        if has_target:
            idx = int(self._rng.randint(0, count - 1))
            tokens[idx] = target
        return tuple(tokens)

    def _family_pace_delta(self) -> float:
        return {
            "steady": 0.35,
            "switch": 0.0,
            "mixed": -0.15,
            "pressure": -0.35,
        }.get(self._cadence_style, 0.0)

    def _light_interval_range_s_for(self, difficulty: float) -> tuple[float, float]:
        base = {
            AntDrillMode.BUILD: (6.4, 10.8),
            AntDrillMode.TEMPO: (5.2, 8.8),
            AntDrillMode.STRESS: (4.3, 7.0),
        }[self._mode]
        delta = self._family_pace_delta()
        low = max(3.0, base[0] + delta - (difficulty * 0.8))
        high = max(low + 0.4, base[1] + delta - (difficulty * 1.0))
        return (round(low, 2), round(high, 2))

    def _scan_interval_range_s_for(self, difficulty: float) -> tuple[float, float]:
        base = {
            AntDrillMode.BUILD: (6.0, 10.4),
            AntDrillMode.TEMPO: (4.9, 8.4),
            AntDrillMode.STRESS: (4.0, 6.7),
        }[self._mode]
        delta = self._family_pace_delta()
        low = max(3.0, base[0] + delta - (difficulty * 0.85))
        high = max(low + 0.4, base[1] + delta - (difficulty * 1.0))
        return (round(low, 2), round(high, 2))

    def _scan_repeat_range_for(self, difficulty: float) -> tuple[int, int]:
        base = {
            AntDrillMode.BUILD: (3, 4),
            AntDrillMode.TEMPO: (2, 4),
            AntDrillMode.STRESS: (2, 3),
        }[self._mode]
        if self._cadence_style == "pressure":
            return (2, 3)
        if self._cadence_style == "steady" and difficulty < 0.5:
            return (3, 4)
        return base

    def _system_step_interval_s_for(self, difficulty: float) -> float:
        base = {
            AntDrillMode.BUILD: 1.95,
            AntDrillMode.TEMPO: 1.55,
            AntDrillMode.STRESS: 1.30,
        }[self._mode]
        value = base + (self._family_pace_delta() * 0.45) - (difficulty * 0.18)
        return round(max(0.95, min(2.4, value)), 3)

    def _prompt_for_active_panels(self, active_panels: tuple[str, ...]) -> str:
        labels = {
            "scene": "Map",
            "light": "Light",
            "scan": "Scan",
            "system": "System",
        }
        active_text = ", ".join(labels[panel] for panel in active_panels)
        return f"Active panels: {active_text}. Register matches only in the active panels."


class TrSceneAnchorGenerator(_TargetRecognitionTrainingGenerator):
    def __init__(self, *, seed: int, mode: AntDrillMode | str) -> None:
        super().__init__(
            seed=seed,
            mode=mode,
            active_panel_sequence=(("scene",),),
            scene_mode="basic",
            cadence_style="steady",
        )


class TrSceneModifierRunGenerator(_TargetRecognitionTrainingGenerator):
    def __init__(self, *, seed: int, mode: AntDrillMode | str) -> None:
        super().__init__(
            seed=seed,
            mode=mode,
            active_panel_sequence=(("scene",),),
            scene_mode="full",
            cadence_style="steady",
        )


class TrLightAnchorGenerator(_TargetRecognitionTrainingGenerator):
    def __init__(self, *, seed: int, mode: AntDrillMode | str) -> None:
        super().__init__(
            seed=seed,
            mode=mode,
            active_panel_sequence=(("light",),),
            cadence_style="steady",
        )


class TrScanAnchorGenerator(_TargetRecognitionTrainingGenerator):
    def __init__(self, *, seed: int, mode: AntDrillMode | str) -> None:
        super().__init__(
            seed=seed,
            mode=mode,
            active_panel_sequence=(("scan",),),
            scan_pool_mode="simple",
            cadence_style="steady",
        )


class TrSystemAnchorGenerator(_TargetRecognitionTrainingGenerator):
    def __init__(self, *, seed: int, mode: AntDrillMode | str) -> None:
        super().__init__(
            seed=seed,
            mode=mode,
            active_panel_sequence=(("system",),),
            cadence_style="steady",
        )


class TrPanelSwitchRunGenerator(_TargetRecognitionTrainingGenerator):
    def __init__(self, *, seed: int, mode: AntDrillMode | str) -> None:
        super().__init__(
            seed=seed,
            mode=mode,
            active_panel_sequence=(("scene",), ("light",), ("scan",), ("system",)),
            cadence_style="switch",
        )


class TrMixedTempoGenerator(_TargetRecognitionTrainingGenerator):
    def __init__(self, *, seed: int, mode: AntDrillMode | str) -> None:
        super().__init__(
            seed=seed,
            mode=mode,
            active_panel_sequence=(
                ("scene", "light"),
                ("scan", "system"),
                ("scene", "scan"),
                ("light", "system"),
                ("scene", "light", "scan"),
                ("scene", "light", "scan", "system"),
            ),
            cadence_style="mixed",
        )


class TrPressureRunGenerator(_TargetRecognitionTrainingGenerator):
    def __init__(self, *, seed: int, mode: AntDrillMode | str) -> None:
        super().__init__(
            seed=seed,
            mode=mode,
            active_panel_sequence=(("scene", "light", "scan", "system"),),
            cadence_style="pressure",
        )


def _build_tr_drill(
    *,
    title_base: str,
    instructions: tuple[str, ...],
    generator: _TargetRecognitionTrainingGenerator,
    clock: Clock,
    seed: int,
    difficulty: float,
    mode: AntDrillMode | str,
    config: TrDrillConfig,
    base_caps_by_level: tuple[float, ...],
) -> TargetRecognitionTimedDrill:
    normalized_mode = _normalize_mode(mode)
    profile = ANT_DRILL_MODE_PROFILES[normalized_mode]
    practice_questions = (
        profile.practice_questions if config.practice_questions is None else int(config.practice_questions)
    )
    scored_duration_s = (
        profile.scored_duration_s if config.scored_duration_s is None else float(config.scored_duration_s)
    )
    return TargetRecognitionTimedDrill(
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
        scorer=TargetRecognitionScorer(),
        immediate_feedback_override=True,
    )


def build_tr_scene_anchor_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: TrDrillConfig | None = None,
) -> TargetRecognitionTimedDrill:
    cfg = config or TrDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_tr_drill(
        title_base="Target Recognition: Scene Anchor",
        instructions=(
            "Target Recognition: Scene Anchor",
            f"Mode: {profile.label}",
            "Stay on the map panel only and learn the clean shape plus affiliation combinations first.",
            "Damaged and high-priority tags are removed in this block. Inactive panels remain visible but OFF.",
            "Mouse only: use the live Target Recognition panel interactions.",
            "Press Enter to begin practice.",
        ),
        generator=TrSceneAnchorGenerator(seed=seed, mode=mode),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0),
    )


def build_tr_scene_modifier_run_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: TrDrillConfig | None = None,
) -> TargetRecognitionTimedDrill:
    cfg = config or TrDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_tr_drill(
        title_base="Target Recognition: Scene Modifier Run",
        instructions=(
            "Target Recognition: Scene Modifier Run",
            f"Mode: {profile.label}",
            "Stay on the map panel, but now include damaged and high-priority tags in the target list.",
            "Keep the full live map behavior while the other panels stay visible and OFF.",
            "Mouse only: use the live Target Recognition panel interactions.",
            "Press Enter to begin practice.",
        ),
        generator=TrSceneModifierRunGenerator(seed=seed, mode=mode),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0),
    )


def build_tr_light_anchor_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: TrDrillConfig | None = None,
) -> TargetRecognitionTimedDrill:
    cfg = config or TrDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_tr_drill(
        title_base="Target Recognition: Light Anchor",
        instructions=(
            "Target Recognition: Light Anchor",
            f"Mode: {profile.label}",
            "Train the light panel by itself first so the colour-pattern timing becomes automatic.",
            "Inactive panels stay visible but OFF, and light cadence tightens as the mode climbs.",
            "Mouse only: use the live Target Recognition panel interactions.",
            "Press Enter to begin practice.",
        ),
        generator=TrLightAnchorGenerator(seed=seed, mode=mode),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0),
    )


def build_tr_scan_anchor_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: TrDrillConfig | None = None,
) -> TargetRecognitionTimedDrill:
    cfg = config or TrDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_tr_drill(
        title_base="Target Recognition: Scan Anchor",
        instructions=(
            "Target Recognition: Scan Anchor",
            f"Mode: {profile.label}",
            "Train the scan stream by itself first, using an easier symbol pool in the lower end of the family.",
            "Inactive panels stay visible but OFF, and scan cadence tightens as the mode climbs.",
            "Mouse only: use the live Target Recognition panel interactions.",
            "Press Enter to begin practice.",
        ),
        generator=TrScanAnchorGenerator(seed=seed, mode=mode),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0),
    )


def build_tr_system_anchor_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: TrDrillConfig | None = None,
) -> TargetRecognitionTimedDrill:
    cfg = config or TrDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_tr_drill(
        title_base="Target Recognition: System Anchor",
        instructions=(
            "Target Recognition: System Anchor",
            f"Mode: {profile.label}",
            "Train the scrolling system columns by themselves so the target-code handoff becomes stable.",
            "Inactive panels stay visible but OFF, and system cadence tightens as the mode climbs.",
            "Mouse only: use the live Target Recognition panel interactions.",
            "Press Enter to begin practice.",
        ),
        generator=TrSystemAnchorGenerator(seed=seed, mode=mode),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0),
    )


def build_tr_panel_switch_run_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: TrDrillConfig | None = None,
) -> TargetRecognitionTimedDrill:
    cfg = config or TrDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_tr_drill(
        title_base="Target Recognition: Panel Switch Run",
        instructions=(
            "Target Recognition: Panel Switch Run",
            f"Mode: {profile.label}",
            "One panel is active at a time, and the block cycles Scene -> Light -> Scan -> System.",
            "Use the switching rhythm to reset your eyes cleanly instead of carrying the previous panel into the next one.",
            "Mouse only: use the live Target Recognition panel interactions.",
            "Press Enter to begin practice.",
        ),
        generator=TrPanelSwitchRunGenerator(seed=seed, mode=mode),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0),
    )


def build_tr_mixed_tempo_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: TrDrillConfig | None = None,
) -> TargetRecognitionTimedDrill:
    cfg = config or TrDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_tr_drill(
        title_base="Target Recognition: Mixed Tempo",
        instructions=(
            "Target Recognition: Mixed Tempo",
            f"Mode: {profile.label}",
            "Cycle through fixed multi-panel combinations so panel switching becomes deliberate before the full test-style run.",
            "The active-panel rhythm is fixed and deterministic for the session seed.",
            "Mouse only: use the live Target Recognition panel interactions.",
            "Press Enter to begin practice.",
        ),
        generator=TrMixedTempoGenerator(seed=seed, mode=mode),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0),
    )


def build_tr_pressure_run_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: TrDrillConfig | None = None,
) -> TargetRecognitionTimedDrill:
    cfg = config or TrDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return _build_tr_drill(
        title_base="Target Recognition: Pressure Run",
        instructions=(
            "Target Recognition: Pressure Run",
            f"Mode: {profile.label}",
            "All four panels stay live on every item, with the fastest cadence profile in this family.",
            "Treat this as the full multitask finish, not a place to chase perfect streaks.",
            "Mouse only: use the live Target Recognition panel interactions.",
            "Press Enter to begin practice.",
        ),
        generator=TrPressureRunGenerator(seed=seed, mode=mode),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
        base_caps_by_level=(15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0),
    )
