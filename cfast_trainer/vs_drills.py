from __future__ import annotations

from dataclasses import dataclass, field

from .ant_drills import (
    ANT_DRILL_MODE_PROFILES,
    AntAdaptiveDifficultyConfig,
    AntDrillMode,
    TimedCapDrill,
)
from .clock import Clock
from .visual_search import (
    VisualSearchGenerator,
    VisualSearchProfile,
    VisualSearchScorer,
    VisualSearchTaskKind,
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
                similarity_ceiling=0.45,
                family_switch_floor=0.05,
                family_switch_ceiling=0.30,
                preview_emphasis=True,
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
                similarity_ceiling=0.55,
                family_switch_floor=0.10,
                family_switch_ceiling=0.40,
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
                similarity_ceiling=0.90,
                family_switch_floor=0.0,
                family_switch_ceiling=0.0,
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
                similarity_ceiling=0.60,
                family_switch_floor=0.15,
                family_switch_ceiling=0.55,
            ),
        )
        self._hard = VisualSearchGenerator(
            seed=seed + 202,
            profile=VisualSearchProfile(
                similarity_floor=0.55,
                similarity_ceiling=0.95,
                family_switch_floor=0.35,
                family_switch_ceiling=0.90,
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
                similarity_ceiling=0.98,
                family_switch_floor=0.45,
                family_switch_ceiling=0.95,
            ),
        )

    def next_problem(self, *, difficulty: float):
        level = _difficulty_to_level(difficulty)
        local_level = min(10, level + 1)
        return self._base.next_problem(difficulty=_level_to_difficulty(local_level))


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
            "Use the same full 3x4 board as the test, but give yourself a clean beat to reacquire the target before scanning.",
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
