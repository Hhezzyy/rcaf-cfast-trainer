from __future__ import annotations

from dataclasses import dataclass

from .ant_drills import ANT_DRILL_MODE_PROFILES, AntDrillAttemptSummary, AntDrillMode
from .clock import Clock
from .cognitive_core import Phase, QuestionEvent, TestSnapshot, clamp01
from .vigilance import VigilanceConfig, build_vigilance_test


def _normalize_mode(mode: AntDrillMode | str) -> AntDrillMode:
    if isinstance(mode, AntDrillMode):
        return mode
    return AntDrillMode(str(mode).strip().lower())


def _difficulty_to_level(difficulty: float) -> int:
    return max(1, min(10, int(round(clamp01(difficulty) * 9.0)) + 1))


def _max_timeout_streak(events: tuple[QuestionEvent, ...]) -> int:
    max_streak = 0
    current = 0
    for event in events:
        if str(event.raw).strip() == "":
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak


_PRACTICE_DURATION_BY_MODE: dict[AntDrillMode, float] = {
    AntDrillMode.BUILD: 45.0,
    AntDrillMode.TEMPO: 30.0,
    AntDrillMode.STRESS: 20.0,
}


@dataclass(frozen=True, slots=True)
class VigilanceDrillConfig:
    practice_duration_s: float | None = None
    scored_duration_s: float | None = None
    spawn_interval_s: float | None = None
    max_active_symbols: int | None = None


class VigilanceDrill:
    def __init__(
        self,
        *,
        title: str,
        instructions: tuple[str, ...],
        engine: object,
        seed: int,
        difficulty: float,
        mode: AntDrillMode,
        practice_duration_s: float,
        scored_duration_s: float,
        spawn_interval_s: float,
        max_active_symbols: int,
    ) -> None:
        self._title = str(title)
        self._instructions = tuple(str(line) for line in instructions)
        self._engine = engine
        self._seed = int(seed)
        self._difficulty = float(difficulty)
        self._mode = mode
        self._practice_duration_s = float(practice_duration_s)
        self._scored_duration_s = float(scored_duration_s)
        self._spawn_interval_s = float(spawn_interval_s)
        self._max_active_symbols = int(max_active_symbols)

    def __getattr__(self, name: str):
        return getattr(self._engine, name)

    @property
    def phase(self) -> Phase:
        return self._engine.phase

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def difficulty(self) -> float:
        return self._difficulty

    @property
    def practice_duration_s(self) -> float:
        return self._practice_duration_s

    @property
    def scored_duration_s(self) -> float:
        return self._scored_duration_s

    @property
    def spawn_interval_s(self) -> float:
        return self._spawn_interval_s

    @property
    def max_active_symbols(self) -> int:
        return self._max_active_symbols

    def can_exit(self) -> bool:
        return self._engine.can_exit()

    def start_practice(self) -> None:
        self._engine.start_practice()

    def start_scored(self) -> None:
        self._engine.start_scored()

    def submit_answer(self, raw: str) -> bool:
        return bool(self._engine.submit_answer(raw))

    def update(self) -> None:
        self._engine.update()

    def snapshot(self) -> TestSnapshot:
        snap = self._engine.snapshot()
        prompt = str(snap.prompt)
        input_hint = str(snap.input_hint)
        if snap.phase is Phase.INSTRUCTIONS:
            prompt = "\n".join(self._instructions)
            input_hint = "Press Enter to begin."
        elif snap.phase is Phase.PRACTICE_DONE:
            input_hint = "Press Enter to continue."
        return TestSnapshot(
            title=self._title,
            phase=snap.phase,
            prompt=prompt,
            input_hint=input_hint,
            time_remaining_s=snap.time_remaining_s,
            attempted_scored=snap.attempted_scored,
            correct_scored=snap.correct_scored,
            payload=snap.payload,
            practice_feedback=snap.practice_feedback,
        )

    def events(self) -> list[QuestionEvent]:
        return self._engine.events()

    def scored_summary(self) -> AntDrillAttemptSummary:
        scored_events = tuple(event for event in self._engine.events() if event.phase is Phase.SCORED)
        attempted = len(scored_events)
        correct = sum(1 for event in scored_events if event.is_correct)
        duration_s = float(self._scored_duration_s)
        total_score = float(sum(float(event.score) for event in scored_events))
        max_score = float(sum(float(event.max_score) for event in scored_events))
        timeouts = sum(1 for event in scored_events if str(event.raw).strip() == "")
        accuracy = 0.0 if attempted == 0 else correct / attempted
        throughput = (attempted / duration_s) * 60.0 if duration_s > 0.0 else 0.0
        correct_per_min = (correct / duration_s) * 60.0 if duration_s > 0.0 else 0.0
        fixation_rate = 0.0 if attempted == 0 else timeouts / attempted
        capture_rts = [event.response_time_s for event in scored_events if event.is_correct]
        mean_rt = None if not capture_rts else sum(capture_rts) / len(capture_rts)
        difficulty_level = _difficulty_to_level(self._difficulty)
        return AntDrillAttemptSummary(
            attempted=attempted,
            correct=correct,
            accuracy=float(accuracy),
            duration_s=duration_s,
            throughput_per_min=float(throughput),
            mean_response_time_s=None if mean_rt is None else float(mean_rt),
            total_score=total_score,
            max_score=max_score,
            score_ratio=0.0 if max_score <= 0.0 else total_score / max_score,
            correct_per_min=float(correct_per_min),
            timeouts=int(timeouts),
            fixation_rate=float(fixation_rate),
            max_timeout_streak=_max_timeout_streak(scored_events),
            mode=self._mode.value,
            difficulty_level=difficulty_level,
            difficulty_level_start=difficulty_level,
            difficulty_level_end=difficulty_level,
            difficulty_change_count=0,
            adaptive_enabled=False,
            adaptive_window_size=0,
        )


def _build_vig_drill(
    *,
    title_base: str,
    instructions: tuple[str, ...],
    clock: Clock,
    seed: int,
    difficulty: float,
    mode: AntDrillMode | str,
    config: VigilanceDrillConfig | None,
    spawn_interval_s: float,
    max_active_symbols: int,
) -> VigilanceDrill:
    normalized_mode = _normalize_mode(mode)
    mode_profile = ANT_DRILL_MODE_PROFILES[normalized_mode]
    cfg = config or VigilanceDrillConfig()
    practice_duration_s = (
        float(cfg.practice_duration_s)
        if cfg.practice_duration_s is not None
        else _PRACTICE_DURATION_BY_MODE[normalized_mode]
    )
    scored_duration_s = (
        float(cfg.scored_duration_s)
        if cfg.scored_duration_s is not None
        else float(mode_profile.scored_duration_s)
    )
    resolved_spawn_interval_s = (
        float(cfg.spawn_interval_s)
        if cfg.spawn_interval_s is not None
        else float(spawn_interval_s)
    )
    resolved_max_active_symbols = (
        int(cfg.max_active_symbols)
        if cfg.max_active_symbols is not None
        else int(max_active_symbols)
    )
    engine = build_vigilance_test(
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        config=VigilanceConfig(
            practice_duration_s=practice_duration_s,
            scored_duration_s=scored_duration_s,
            spawn_interval_s=resolved_spawn_interval_s,
            max_active_symbols=resolved_max_active_symbols,
        ),
    )
    return VigilanceDrill(
        title=f"{title_base} ({mode_profile.label})",
        instructions=instructions,
        engine=engine,
        seed=seed,
        difficulty=difficulty,
        mode=normalized_mode,
        practice_duration_s=practice_duration_s,
        scored_duration_s=scored_duration_s,
        spawn_interval_s=resolved_spawn_interval_s,
        max_active_symbols=resolved_max_active_symbols,
    )


def build_vig_entry_anchor_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: VigilanceDrillConfig | None = None,
) -> VigilanceDrill:
    normalized_mode = _normalize_mode(mode)
    mode_profile = ANT_DRILL_MODE_PROFILES[normalized_mode]
    return _build_vig_drill(
        title_base="Vigilance: Entry Anchor",
        instructions=(
            "Vigilance: Entry Anchor",
            f"Mode: {mode_profile.label}",
            "Keep the real 9x9 Vigilance board and row/column entry flow, but start on the slowest symbol rhythm in this family.",
            "Use the block to settle clean coordinate entry and avoid typing ahead of what you actually saw.",
            "Press Enter to begin.",
        ),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=normalized_mode,
        config=config,
        spawn_interval_s=1.15,
        max_active_symbols=4,
    )


def build_vig_clean_scan_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: VigilanceDrillConfig | None = None,
) -> VigilanceDrill:
    normalized_mode = _normalize_mode(mode)
    mode_profile = ANT_DRILL_MODE_PROFILES[normalized_mode]
    return _build_vig_drill(
        title_base="Vigilance: Clean Scan",
        instructions=(
            "Vigilance: Clean Scan",
            f"Mode: {mode_profile.label}",
            "Stay on the full Vigilance board and build a disciplined scan path before the denser overlap blocks return.",
            "Keep the row and column entry exact; treat every miss like a scan breakdown, not a speed problem.",
            "Press Enter to begin.",
        ),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=normalized_mode,
        config=config,
        spawn_interval_s=0.95,
        max_active_symbols=5,
    )


def build_vig_steady_capture_run_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.TEMPO,
    config: VigilanceDrillConfig | None = None,
) -> VigilanceDrill:
    normalized_mode = _normalize_mode(mode)
    mode_profile = ANT_DRILL_MODE_PROFILES[normalized_mode]
    return _build_vig_drill(
        title_base="Vigilance: Steady Capture Run",
        instructions=(
            "Vigilance: Steady Capture Run",
            f"Mode: {mode_profile.label}",
            "Run the full Vigilance task at a sustained baseline pace without changing the board or entry method.",
            "Prioritize clean capture rhythm first, then let speed rise without breaking row and column accuracy.",
            "Press Enter to begin.",
        ),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=normalized_mode,
        config=config,
        spawn_interval_s=0.85,
        max_active_symbols=6,
    )


def build_vig_density_ladder_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.TEMPO,
    config: VigilanceDrillConfig | None = None,
) -> VigilanceDrill:
    normalized_mode = _normalize_mode(mode)
    mode_profile = ANT_DRILL_MODE_PROFILES[normalized_mode]
    return _build_vig_drill(
        title_base="Vigilance: Density Ladder",
        instructions=(
            "Vigilance: Density Ladder",
            f"Mode: {mode_profile.label}",
            "Keep the real board and symbol mix, but push into heavier overlap so cleanup and prioritization stay controlled.",
            "Do not change your entry style under density; keep the same deliberate row-then-column rhythm.",
            "Press Enter to begin.",
        ),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=normalized_mode,
        config=config,
        spawn_interval_s=0.78,
        max_active_symbols=7,
    )


def build_vig_tempo_sweep_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.TEMPO,
    config: VigilanceDrillConfig | None = None,
) -> VigilanceDrill:
    normalized_mode = _normalize_mode(mode)
    mode_profile = ANT_DRILL_MODE_PROFILES[normalized_mode]
    return _build_vig_drill(
        title_base="Vigilance: Tempo Sweep",
        instructions=(
            "Vigilance: Tempo Sweep",
            f"Mode: {mode_profile.label}",
            "Run the full Vigilance board at a faster steady pace while keeping the same point values and row/column entry flow.",
            "Stay ahead of the stream without guessing coordinates that were not actually locked in.",
            "Press Enter to begin.",
        ),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=normalized_mode,
        config=config,
        spawn_interval_s=0.68,
        max_active_symbols=7,
    )


def build_vig_pressure_run_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.STRESS,
    config: VigilanceDrillConfig | None = None,
) -> VigilanceDrill:
    normalized_mode = _normalize_mode(mode)
    mode_profile = ANT_DRILL_MODE_PROFILES[normalized_mode]
    return _build_vig_drill(
        title_base="Vigilance: Pressure Run",
        instructions=(
            "Vigilance: Pressure Run",
            f"Mode: {mode_profile.label}",
            "Finish on the hardest full-board Vigilance stream in this family without changing the board or score meaning.",
            "Accept that some symbols will expire; recover immediately and keep the next coordinate clean.",
            "Press Enter to begin.",
        ),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=normalized_mode,
        config=config,
        spawn_interval_s=0.60,
        max_active_symbols=8,
    )
