from __future__ import annotations

from dataclasses import dataclass

from .ant_drills import ANT_DRILL_MODE_PROFILES, AntDrillAttemptSummary, AntDrillMode
from .clock import Clock
from .cognitive_core import Phase, TestSnapshot, clamp01
from .sensory_motor_apparatus import (
    SensoryMotorApparatusConfig,
    SensoryMotorApparatusEngine,
    SensoryMotorSegment,
    build_sensory_motor_apparatus_test,
)


def _normalize_mode(mode: AntDrillMode | str) -> AntDrillMode:
    if isinstance(mode, AntDrillMode):
        return mode
    return AntDrillMode(str(mode).strip().lower())


def _difficulty_to_level(difficulty: float) -> int:
    return max(1, min(10, int(round(clamp01(difficulty) * 9.0)) + 1))


@dataclass(frozen=True, slots=True)
class SmaDrillConfig:
    scored_duration_s: float | None = None


class SensoryMotorContinuousDrill:
    def __init__(
        self,
        *,
        title: str,
        instructions: tuple[str, ...],
        engine: SensoryMotorApparatusEngine,
        seed: int,
        difficulty: float,
        mode: AntDrillMode,
        scored_duration_s: float,
    ) -> None:
        self._title = str(title)
        self._instructions = tuple(str(line) for line in instructions)
        self._engine = engine
        self._seed = int(seed)
        self._difficulty = float(difficulty)
        self._mode = mode
        self._scored_duration_s = float(scored_duration_s)

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
    def practice_questions(self) -> int:
        return 0

    @property
    def scored_duration_s(self) -> float:
        return self._scored_duration_s

    def can_exit(self) -> bool:
        return self._engine.can_exit()

    def start_practice(self) -> None:
        if self._engine.phase is Phase.INSTRUCTIONS:
            self._engine.start_scored()

    def start_scored(self) -> None:
        self._engine.start_scored()

    def submit_answer(self, raw: str) -> bool:
        return self._engine.submit_answer(raw)

    def set_control(self, *, horizontal: float, vertical: float) -> None:
        self._engine.set_control(horizontal=horizontal, vertical=vertical)

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

    def events(self):
        return self._engine.events()

    def scored_summary(self) -> AntDrillAttemptSummary:
        base = self._engine.scored_summary()
        events = self._engine.events()
        mean_rt = None
        if events:
            mean_rt = sum(event.response_time_s for event in events) / len(events)
        difficulty_level = _difficulty_to_level(self._difficulty)
        return AntDrillAttemptSummary(
            attempted=base.attempted,
            correct=base.correct,
            accuracy=base.accuracy,
            duration_s=base.duration_s,
            throughput_per_min=base.throughput_per_min,
            mean_response_time_s=mean_rt,
            total_score=base.total_score,
            max_score=base.max_score,
            score_ratio=base.score_ratio,
            correct_per_min=0.0 if base.duration_s <= 0.0 else (base.correct / base.duration_s) * 60.0,
            timeouts=0,
            fixation_rate=0.0,
            max_timeout_streak=0,
            mode=self._mode.value,
            difficulty_level=difficulty_level,
            difficulty_level_start=difficulty_level,
            difficulty_level_end=difficulty_level,
            difficulty_change_count=0,
            adaptive_enabled=False,
            adaptive_window_size=0,
        )


def _repeat_segments(
    *,
    total_duration_s: float,
    templates: tuple[SensoryMotorSegment, ...],
) -> tuple[SensoryMotorSegment, ...]:
    remaining = max(0.0, float(total_duration_s))
    built: list[SensoryMotorSegment] = []
    idx = 0
    while remaining > 1e-9:
        template = templates[idx % len(templates)]
        seg_duration = min(float(template.duration_s), remaining)
        if seg_duration <= 0.0:
            break
        built.append(replace_segment_duration(template, seg_duration))
        remaining -= seg_duration
        idx += 1
    return tuple(built)


def replace_segment_duration(segment: SensoryMotorSegment, duration_s: float) -> SensoryMotorSegment:
    return SensoryMotorSegment(
        control_mode=segment.control_mode,
        axis_focus=segment.axis_focus,
        disturbance_profile=segment.disturbance_profile,
        duration_s=float(duration_s),
        label=segment.label,
        pause_after=False,
    )


def _build_sma_drill(
    *,
    title_base: str,
    instructions: tuple[str, ...],
    scored_segments: tuple[SensoryMotorSegment, ...],
    clock: Clock,
    seed: int,
    difficulty: float,
    mode: AntDrillMode | str,
    config: SmaDrillConfig,
) -> SensoryMotorContinuousDrill:
    normalized_mode = _normalize_mode(mode)
    profile = ANT_DRILL_MODE_PROFILES[normalized_mode]
    scored_duration_s = (
        float(profile.scored_duration_s)
        if config.scored_duration_s is None
        else float(config.scored_duration_s)
    )
    engine = build_sensory_motor_apparatus_test(
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        title=title_base,
        config=SensoryMotorApparatusConfig(
            practice_duration_s=0.0,
            scored_duration_s=scored_duration_s,
        ),
        practice_segments=(),
        scored_segments=scored_segments,
    )
    return SensoryMotorContinuousDrill(
        title=f"{title_base} ({profile.label})",
        instructions=instructions,
        engine=engine,
        seed=seed,
        difficulty=difficulty,
        mode=normalized_mode,
        scored_duration_s=scored_duration_s,
    )


def build_sma_joystick_horizontal_anchor_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: SmaDrillConfig | None = None,
) -> SensoryMotorContinuousDrill:
    cfg = config or SmaDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    scored_duration_s = profile.scored_duration_s if cfg.scored_duration_s is None else cfg.scored_duration_s
    return _build_sma_drill(
        title_base="Sensory Motor Apparatus: Joystick Horizontal Anchor",
        instructions=(
            "Sensory Motor Apparatus: Joystick Horizontal Anchor",
            f"Mode: {profile.label}",
            "Track only horizontal drift with joystick X. Vertical input is ignored in this block.",
            "Use the guide band to settle quickly and hold clean left-right control.",
            "Press Enter to begin.",
        ),
        scored_segments=(
            SensoryMotorSegment(
                control_mode="joystick_only",
                axis_focus="horizontal",
                disturbance_profile="axis_bias_horizontal",
                duration_s=float(scored_duration_s),
                label="Joystick Horizontal Anchor",
                pause_after=False,
            ),
        ),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
    )


def build_sma_joystick_vertical_anchor_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: SmaDrillConfig | None = None,
) -> SensoryMotorContinuousDrill:
    cfg = config or SmaDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    scored_duration_s = profile.scored_duration_s if cfg.scored_duration_s is None else cfg.scored_duration_s
    return _build_sma_drill(
        title_base="Sensory Motor Apparatus: Joystick Vertical Anchor",
        instructions=(
            "Sensory Motor Apparatus: Joystick Vertical Anchor",
            f"Mode: {profile.label}",
            "Track only vertical drift with joystick axis 1. Horizontal input is ignored in this block.",
            "Use the guide band to settle quickly and hold clean up-down control.",
            "Press Enter to begin.",
        ),
        scored_segments=(
            SensoryMotorSegment(
                control_mode="joystick_only",
                axis_focus="vertical",
                disturbance_profile="axis_bias_vertical",
                duration_s=float(scored_duration_s),
                label="Joystick Vertical Anchor",
                pause_after=False,
            ),
        ),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
    )


def build_sma_joystick_hold_run_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: SmaDrillConfig | None = None,
) -> SensoryMotorContinuousDrill:
    cfg = config or SmaDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    scored_duration_s = profile.scored_duration_s if cfg.scored_duration_s is None else cfg.scored_duration_s
    return _build_sma_drill(
        title_base="Sensory Motor Apparatus: Joystick Hold Run",
        instructions=(
            "Sensory Motor Apparatus: Joystick Hold Run",
            f"Mode: {profile.label}",
            "Stay in joystick-only control and hold two-axis precision against slower steady drift.",
            "Use this block to clean up settling and recovery before split coordination starts.",
            "Press Enter to begin.",
        ),
        scored_segments=(
            SensoryMotorSegment(
                control_mode="joystick_only",
                axis_focus="both",
                disturbance_profile="steady",
                duration_s=float(scored_duration_s),
                label="Joystick Hold Run",
                pause_after=False,
            ),
        ),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
    )


def build_sma_split_horizontal_prime_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: SmaDrillConfig | None = None,
) -> SensoryMotorContinuousDrill:
    cfg = config or SmaDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    scored_duration_s = profile.scored_duration_s if cfg.scored_duration_s is None else cfg.scored_duration_s
    return _build_sma_drill(
        title_base="Sensory Motor Apparatus: Split Horizontal Prime",
        instructions=(
            "Sensory Motor Apparatus: Split Horizontal Prime",
            f"Mode: {profile.label}",
            "Use split controls with rudder-only horizontal tracking while vertical input is ignored.",
            "Prime the pedal handoff before returning to full split coordination.",
            "Press Enter to begin.",
        ),
        scored_segments=(
            SensoryMotorSegment(
                control_mode="split",
                axis_focus="horizontal",
                disturbance_profile="axis_bias_horizontal",
                duration_s=float(scored_duration_s),
                label="Split Horizontal Prime",
                pause_after=False,
            ),
        ),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
    )


def build_sma_split_coordination_run_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: SmaDrillConfig | None = None,
) -> SensoryMotorContinuousDrill:
    cfg = config or SmaDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    scored_duration_s = profile.scored_duration_s if cfg.scored_duration_s is None else cfg.scored_duration_s
    return _build_sma_drill(
        title_base="Sensory Motor Apparatus: Split Coordination Run",
        instructions=(
            "Sensory Motor Apparatus: Split Coordination Run",
            f"Mode: {profile.label}",
            "Use rudder plus vertical stick control together under balanced two-axis drift.",
            "This is the core split-control coordination block in the family.",
            "Press Enter to begin.",
        ),
        scored_segments=(
            SensoryMotorSegment(
                control_mode="split",
                axis_focus="both",
                disturbance_profile="balanced",
                duration_s=float(scored_duration_s),
                label="Split Coordination Run",
                pause_after=False,
            ),
        ),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
    )


def build_sma_mode_switch_run_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: SmaDrillConfig | None = None,
) -> SensoryMotorContinuousDrill:
    cfg = config or SmaDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    scored_duration_s = float(profile.scored_duration_s if cfg.scored_duration_s is None else cfg.scored_duration_s)
    segments = _repeat_segments(
        total_duration_s=scored_duration_s,
        templates=(
            SensoryMotorSegment(
                control_mode="joystick_only",
                axis_focus="both",
                disturbance_profile="balanced",
                duration_s=60.0,
                label="Mode Switch - Joystick",
                pause_after=False,
            ),
            SensoryMotorSegment(
                control_mode="split",
                axis_focus="both",
                disturbance_profile="balanced",
                duration_s=60.0,
                label="Mode Switch - Split",
                pause_after=False,
            ),
        ),
    )
    return _build_sma_drill(
        title_base="Sensory Motor Apparatus: Mode Switch Run",
        instructions=(
            "Sensory Motor Apparatus: Mode Switch Run",
            f"Mode: {profile.label}",
            "Switch between joystick-only and split control every 60 seconds without losing the tracking rhythm.",
            "Use this block to make the control-mode reset deliberate instead of frantic.",
            "Press Enter to begin.",
        ),
        scored_segments=segments,
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
    )


def build_sma_disturbance_tempo_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: SmaDrillConfig | None = None,
) -> SensoryMotorContinuousDrill:
    cfg = config or SmaDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    scored_duration_s = float(profile.scored_duration_s if cfg.scored_duration_s is None else cfg.scored_duration_s)
    segments = _repeat_segments(
        total_duration_s=scored_duration_s,
        templates=(
            SensoryMotorSegment(
                control_mode="joystick_only",
                axis_focus="both",
                disturbance_profile="steady",
                duration_s=30.0,
                label="Tempo - Joystick Steady",
                pause_after=False,
            ),
            SensoryMotorSegment(
                control_mode="joystick_only",
                axis_focus="both",
                disturbance_profile="pulse",
                duration_s=30.0,
                label="Tempo - Joystick Pulse",
                pause_after=False,
            ),
            SensoryMotorSegment(
                control_mode="split",
                axis_focus="both",
                disturbance_profile="steady",
                duration_s=30.0,
                label="Tempo - Split Steady",
                pause_after=False,
            ),
            SensoryMotorSegment(
                control_mode="split",
                axis_focus="both",
                disturbance_profile="pulse",
                duration_s=30.0,
                label="Tempo - Split Pulse",
                pause_after=False,
            ),
        ),
    )
    return _build_sma_drill(
        title_base="Sensory Motor Apparatus: Disturbance Tempo",
        instructions=(
            "Sensory Motor Apparatus: Disturbance Tempo",
            f"Mode: {profile.label}",
            "Cycle between steady and pulse disturbance profiles while alternating joystick-only and split segments.",
            "Use the segment header to stay ahead of each upcoming disturbance change.",
            "Press Enter to begin.",
        ),
        scored_segments=segments,
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
    )


def build_sma_pressure_run_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: SmaDrillConfig | None = None,
) -> SensoryMotorContinuousDrill:
    cfg = config or SmaDrillConfig()
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    scored_duration_s = float(profile.scored_duration_s if cfg.scored_duration_s is None else cfg.scored_duration_s)
    segments = _repeat_segments(
        total_duration_s=scored_duration_s,
        templates=(
            SensoryMotorSegment(
                control_mode="joystick_only",
                axis_focus="both",
                disturbance_profile="pressure",
                duration_s=30.0,
                label="Pressure - Joystick",
                pause_after=False,
            ),
            SensoryMotorSegment(
                control_mode="split",
                axis_focus="both",
                disturbance_profile="pressure",
                duration_s=30.0,
                label="Pressure - Split",
                pause_after=False,
            ),
        ),
    )
    return _build_sma_drill(
        title_base="Sensory Motor Apparatus: Pressure Run",
        instructions=(
            "Sensory Motor Apparatus: Pressure Run",
            f"Mode: {profile.label}",
            "Finish with the strongest and fastest disturbance profile while alternating joystick-only and split control.",
            "Keep moving through misses instead of chasing perfect recovery.",
            "Press Enter to begin.",
        ),
        scored_segments=segments,
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
    )
