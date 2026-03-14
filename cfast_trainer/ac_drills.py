from __future__ import annotations

from dataclasses import dataclass

from .ant_drills import ANT_DRILL_MODE_PROFILES, AntDrillAttemptSummary, AntDrillMode
from .auditory_capacity import (
    AuditoryCapacityConfig,
    AuditoryCapacityEngine,
    AuditoryCapacityPayload,
    AuditoryCapacityTrainingProfile,
    AuditoryCapacityTrainingSegment,
    build_auditory_capacity_test,
)
from .clock import Clock
from .cognitive_core import Phase, QuestionEvent, TestSnapshot, clamp01

AC_CHANNEL_ORDER = (
    "gates",
    "state_commands",
    "gate_directives",
    "digit_recall",
    "trigger",
    "distractors",
)
AC_CHANNEL_LABELS = {
    "gates": "Gate flight",
    "state_commands": "State commands",
    "gate_directives": "Gate directives",
    "digit_recall": "Digit recall",
    "trigger": "Trigger cues",
    "distractors": "Callsign filter",
}


def _normalize_mode(mode: AntDrillMode | str) -> AntDrillMode:
    if isinstance(mode, AntDrillMode):
        return mode
    return AntDrillMode(str(mode).strip().lower())


def _difficulty_to_level(difficulty: float) -> int:
    return max(1, min(10, int(round(clamp01(difficulty) * 9.0)) + 1))


@dataclass(frozen=True, slots=True)
class AcDrillConfig:
    scored_duration_s: float | None = None


class AuditoryCapacityContinuousDrill:
    def __init__(
        self,
        *,
        title: str,
        instructions: tuple[str, ...],
        engine: AuditoryCapacityEngine,
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
    def practice_questions(self) -> int:
        return 0

    @property
    def scored_duration_s(self) -> float:
        return self._scored_duration_s

    def can_exit(self) -> bool:
        return self._engine.can_exit()

    def start_practice(self) -> None:
        if self._engine.phase is Phase.INSTRUCTIONS:
            self._engine.start_practice()
        if self._engine.phase is Phase.PRACTICE_DONE:
            self._engine.start_scored()

    def start_scored(self) -> None:
        if self._engine.phase is Phase.INSTRUCTIONS:
            self._engine.start_practice()
        if self._engine.phase is Phase.PRACTICE_DONE:
            self._engine.start_scored()

    def submit_answer(self, raw: str) -> bool:
        token = str(raw).strip().lower()
        if token in {"__skip_section__", "skip_section", "__skip_all__", "skip_all"}:
            self._force_phase(Phase.RESULTS)
            return True
        return self._engine.submit_answer(raw)

    def set_control(self, *, horizontal: float, vertical: float) -> None:
        self._engine.set_control(horizontal=horizontal, vertical=vertical)

    def set_audio_overrides(
        self,
        *,
        noise_level: float | None = None,
        distortion_level: float | None = None,
        noise_source: str | None = None,
    ) -> None:
        self._engine.set_audio_overrides(
            noise_level=noise_level,
            distortion_level=distortion_level,
            noise_source=noise_source,
        )

    def update(self) -> None:
        self._engine.update()

    def snapshot(self) -> TestSnapshot:
        snap = self._engine.snapshot()
        prompt = str(snap.prompt)
        input_hint = str(snap.input_hint)
        payload = snap.payload
        if snap.phase is Phase.INSTRUCTIONS:
            prompt = "\n".join(self._instructions)
            input_hint = "Press Enter to begin."
        elif snap.phase is Phase.PRACTICE_DONE:
            input_hint = "Press Enter to continue."
        elif isinstance(payload, AuditoryCapacityPayload):
            focus = ", ".join(
                AC_CHANNEL_LABELS[channel]
                for channel in payload.active_channels
                if channel in AC_CHANNEL_LABELS
            )
            if focus == "":
                focus = "Full mixed channel set"
            input_hint = (
                f"{payload.segment_label} | Focus: {focus} | "
                "Q/W/E/R colour | keypad 0-9 number | digits+Enter recall | Space/trigger beep"
            )
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
        converted: list[QuestionEvent] = []
        for index, event in enumerate(self._engine.events()):
            rt = 0.0 if event.response_time_s is None else float(event.response_time_s)
            prompt = f"{event.kind.value}: {event.expected}"
            converted.append(
                QuestionEvent(
                    index=index,
                    phase=event.phase,
                    prompt=prompt,
                    correct_answer=1,
                    user_answer=1 if event.is_correct else 0,
                    is_correct=bool(event.is_correct),
                    presented_at_s=float(index),
                    answered_at_s=float(index) + rt,
                    response_time_s=rt,
                    raw=str(event.response),
                    score=float(event.score),
                    max_score=1.0,
                )
            )
        return converted

    def scored_summary(self) -> AntDrillAttemptSummary:
        base = self._engine.scored_summary()
        scored_events = [event for event in self._engine.events() if event.phase is Phase.SCORED]
        rts = [float(event.response_time_s) for event in scored_events if event.response_time_s is not None]
        mean_rt = None if not rts else sum(rts) / len(rts)
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

    def _force_phase(self, phase: Phase) -> None:
        if not hasattr(self._engine, "_phase"):
            return
        self._engine._phase = phase


def _mode_scaled_profile(
    *,
    profile: AuditoryCapacityTrainingProfile,
    mode: AntDrillMode,
) -> AuditoryCapacityTrainingProfile:
    if mode is AntDrillMode.BUILD:
        rate_scale = 0.86
        disturbance_scale = 0.76
        tube_scale = 1.10
        response_scale = 1.12
        noise_scale = 0.75
    elif mode is AntDrillMode.STRESS:
        rate_scale = 1.18
        disturbance_scale = 1.18
        tube_scale = 0.95
        response_scale = 0.88
        noise_scale = 1.20
    else:
        rate_scale = 1.0
        disturbance_scale = 1.0
        tube_scale = 1.0
        response_scale = 1.0
        noise_scale = 1.0
    return AuditoryCapacityTrainingProfile(
        enable_gates=profile.enable_gates,
        enable_state_commands=profile.enable_state_commands,
        enable_gate_directives=profile.enable_gate_directives,
        enable_digit_sequences=profile.enable_digit_sequences,
        enable_trigger_cues=profile.enable_trigger_cues,
        enable_distractors=profile.enable_distractors,
        gate_rate_scale=float(profile.gate_rate_scale) * rate_scale,
        command_rate_scale=float(profile.command_rate_scale) * rate_scale,
        directive_rate_scale=float(profile.directive_rate_scale) * rate_scale,
        sequence_rate_scale=float(profile.sequence_rate_scale) * rate_scale,
        beep_rate_scale=float(profile.beep_rate_scale) * rate_scale,
        response_window_scale=float(profile.response_window_scale) * response_scale,
        disturbance_scale=float(profile.disturbance_scale) * disturbance_scale,
        tube_width_scale=float(profile.tube_width_scale) * tube_scale,
        tube_height_scale=float(profile.tube_height_scale) * tube_scale,
        noise_level_scale=float(profile.noise_level_scale) * noise_scale,
        distortion_level_scale=float(profile.distortion_level_scale) * noise_scale,
        digit_sequence_min_len=int(profile.digit_sequence_min_len),
        digit_sequence_max_len=int(profile.digit_sequence_max_len),
    )


def _segment(
    *,
    label: str,
    duration_s: float,
    active_channels: tuple[str, ...],
    profile: AuditoryCapacityTrainingProfile,
) -> AuditoryCapacityTrainingSegment:
    return AuditoryCapacityTrainingSegment(
        label=label,
        duration_s=float(duration_s),
        active_channels=active_channels,
        profile=profile,
    )


def _repeat_segments(
    *,
    total_duration_s: float,
    templates: tuple[AuditoryCapacityTrainingSegment, ...],
) -> tuple[AuditoryCapacityTrainingSegment, ...]:
    remaining = max(0.0, float(total_duration_s))
    if remaining <= 1e-9:
        return ()
    repeated: list[AuditoryCapacityTrainingSegment] = []
    index = 0
    while remaining > 1e-9:
        template = templates[index % len(templates)]
        seg_duration = min(float(template.duration_s), remaining)
        if seg_duration <= 1e-9:
            break
        repeated.append(
            AuditoryCapacityTrainingSegment(
                label=template.label,
                duration_s=seg_duration,
                active_channels=template.active_channels,
                profile=template.profile,
            )
        )
        remaining -= seg_duration
        index += 1
    return tuple(repeated)


def _build_ac_drill(
    *,
    title_base: str,
    instructions: tuple[str, ...],
    scored_segments: tuple[AuditoryCapacityTrainingSegment, ...],
    clock: Clock,
    seed: int,
    difficulty: float,
    mode: AntDrillMode | str,
    config: AcDrillConfig,
) -> AuditoryCapacityContinuousDrill:
    normalized_mode = _normalize_mode(mode)
    profile = ANT_DRILL_MODE_PROFILES[normalized_mode]
    scored_duration_s = (
        float(profile.scored_duration_s)
        if config.scored_duration_s is None
        else float(config.scored_duration_s)
    )
    engine = build_auditory_capacity_test(
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        config=AuditoryCapacityConfig(
            practice_enabled=False,
            practice_duration_s=0.0,
            scored_duration_s=scored_duration_s,
            run_duration_seconds=scored_duration_s,
        ),
        practice_segments=(),
        scored_segments=scored_segments,
    )
    return AuditoryCapacityContinuousDrill(
        title=f"{title_base} ({profile.label})",
        instructions=instructions,
        engine=engine,
        seed=seed,
        difficulty=difficulty,
        mode=normalized_mode,
        scored_duration_s=scored_duration_s,
    )


def build_ac_gate_anchor_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: AcDrillConfig | None = None,
) -> AuditoryCapacityContinuousDrill:
    cfg = config or AcDrillConfig()
    normalized_mode = _normalize_mode(mode)
    profile = _mode_scaled_profile(
        profile=AuditoryCapacityTrainingProfile(
            enable_state_commands=False,
            enable_gate_directives=False,
            enable_digit_sequences=False,
            enable_trigger_cues=False,
            enable_distractors=False,
            gate_rate_scale=0.70,
            disturbance_scale=0.70,
            tube_width_scale=1.12,
            tube_height_scale=1.10,
            noise_level_scale=0.0,
        ),
        mode=normalized_mode,
    )
    scored_duration_s = (
        ANT_DRILL_MODE_PROFILES[normalized_mode].scored_duration_s
        if cfg.scored_duration_s is None
        else float(cfg.scored_duration_s)
    )
    return _build_ac_drill(
        title_base="Auditory Capacity: Gate Anchor",
        instructions=(
            "Auditory Capacity: Gate Anchor",
            f"Mode: {ANT_DRILL_MODE_PROFILES[normalized_mode].label}",
            "Fly the ball through the tunnel with gates only.",
            "No state commands, directives, digit recall, trigger cues, or distractors are active.",
            "Press Enter to begin.",
        ),
        scored_segments=(
            _segment(
                label="Gate Flight",
                duration_s=scored_duration_s,
                active_channels=("gates",),
                profile=profile,
            ),
        ),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=normalized_mode,
        config=cfg,
    )


def build_ac_state_command_prime_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: AcDrillConfig | None = None,
) -> AuditoryCapacityContinuousDrill:
    cfg = config or AcDrillConfig()
    normalized_mode = _normalize_mode(mode)
    profile = _mode_scaled_profile(
        profile=AuditoryCapacityTrainingProfile(
            enable_gate_directives=False,
            enable_digit_sequences=False,
            enable_trigger_cues=False,
            enable_distractors=False,
            gate_rate_scale=0.60,
            command_rate_scale=0.85,
            disturbance_scale=0.78,
            tube_width_scale=1.08,
            tube_height_scale=1.06,
            noise_level_scale=0.0,
        ),
        mode=normalized_mode,
    )
    scored_duration_s = (
        ANT_DRILL_MODE_PROFILES[normalized_mode].scored_duration_s
        if cfg.scored_duration_s is None
        else float(cfg.scored_duration_s)
    )
    return _build_ac_drill(
        title_base="Auditory Capacity: State Command Prime",
        instructions=(
            "Auditory Capacity: State Command Prime",
            f"Mode: {ANT_DRILL_MODE_PROFILES[normalized_mode].label}",
            "Keep low gate pressure while you react to colour and number commands only.",
            "Digit recall, trigger cues, directives, and distractors are disabled in this block.",
            "Press Enter to begin.",
        ),
        scored_segments=(
            _segment(
                label="State Commands",
                duration_s=scored_duration_s,
                active_channels=("gates", "state_commands"),
                profile=profile,
            ),
        ),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=normalized_mode,
        config=cfg,
    )


def build_ac_gate_directive_run_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: AcDrillConfig | None = None,
) -> AuditoryCapacityContinuousDrill:
    cfg = config or AcDrillConfig()
    normalized_mode = _normalize_mode(mode)
    profile = _mode_scaled_profile(
        profile=AuditoryCapacityTrainingProfile(
            enable_state_commands=False,
            enable_digit_sequences=False,
            enable_trigger_cues=False,
            enable_distractors=True,
            gate_rate_scale=0.75,
            directive_rate_scale=0.90,
            disturbance_scale=0.86,
            tube_width_scale=1.04,
            tube_height_scale=1.04,
            noise_level_scale=0.30,
        ),
        mode=normalized_mode,
    )
    scored_duration_s = (
        ANT_DRILL_MODE_PROFILES[normalized_mode].scored_duration_s
        if cfg.scored_duration_s is None
        else float(cfg.scored_duration_s)
    )
    return _build_ac_drill(
        title_base="Auditory Capacity: Gate Directive Run",
        instructions=(
            "Auditory Capacity: Gate Directive Run",
            f"Mode: {ANT_DRILL_MODE_PROFILES[normalized_mode].label}",
            "Follow next-matching-gate directives while still flying the gate stream.",
            "State commands, digit recall, and trigger cues are off; low distractor pressure remains active.",
            "Press Enter to begin.",
        ),
        scored_segments=(
            _segment(
                label="Gate Directives",
                duration_s=scored_duration_s,
                active_channels=("gates", "gate_directives", "distractors"),
                profile=profile,
            ),
        ),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=normalized_mode,
        config=cfg,
    )


def build_ac_digit_sequence_prime_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: AcDrillConfig | None = None,
) -> AuditoryCapacityContinuousDrill:
    cfg = config or AcDrillConfig()
    normalized_mode = _normalize_mode(mode)
    digit_max = 5 if normalized_mode is AntDrillMode.BUILD else 6
    profile = _mode_scaled_profile(
        profile=AuditoryCapacityTrainingProfile(
            enable_state_commands=False,
            enable_gate_directives=False,
            enable_trigger_cues=False,
            enable_distractors=False,
            gate_rate_scale=0.58,
            sequence_rate_scale=0.92,
            disturbance_scale=0.76,
            tube_width_scale=1.08,
            tube_height_scale=1.08,
            noise_level_scale=0.0,
            digit_sequence_min_len=5,
            digit_sequence_max_len=digit_max,
        ),
        mode=normalized_mode,
    )
    scored_duration_s = (
        ANT_DRILL_MODE_PROFILES[normalized_mode].scored_duration_s
        if cfg.scored_duration_s is None
        else float(cfg.scored_duration_s)
    )
    return _build_ac_drill(
        title_base="Auditory Capacity: Digit Sequence Prime",
        instructions=(
            "Auditory Capacity: Digit Sequence Prime",
            f"Mode: {ANT_DRILL_MODE_PROFILES[normalized_mode].label}",
            "Stay on low gate pressure while you memorize and recall the digit groups.",
            "BUILD stays on 5-digit groups; TEMPO and STRESS allow the normal 5-6 digit range.",
            "Press Enter to begin.",
        ),
        scored_segments=(
            _segment(
                label="Digit Recall",
                duration_s=scored_duration_s,
                active_channels=("gates", "digit_recall"),
                profile=profile,
            ),
        ),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=normalized_mode,
        config=cfg,
    )


def build_ac_trigger_cue_anchor_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: AcDrillConfig | None = None,
) -> AuditoryCapacityContinuousDrill:
    cfg = config or AcDrillConfig()
    normalized_mode = _normalize_mode(mode)
    profile = _mode_scaled_profile(
        profile=AuditoryCapacityTrainingProfile(
            enable_state_commands=False,
            enable_gate_directives=False,
            enable_digit_sequences=False,
            enable_distractors=False,
            gate_rate_scale=0.62,
            beep_rate_scale=0.92,
            disturbance_scale=0.74,
            tube_width_scale=1.08,
            tube_height_scale=1.08,
            noise_level_scale=0.0,
        ),
        mode=normalized_mode,
    )
    scored_duration_s = (
        ANT_DRILL_MODE_PROFILES[normalized_mode].scored_duration_s
        if cfg.scored_duration_s is None
        else float(cfg.scored_duration_s)
    )
    return _build_ac_drill(
        title_base="Auditory Capacity: Trigger Cue Anchor",
        instructions=(
            "Auditory Capacity: Trigger Cue Anchor",
            f"Mode: {ANT_DRILL_MODE_PROFILES[normalized_mode].label}",
            "Fly under low gate pressure and answer the trigger/beep cue cleanly.",
            "State commands, directives, digit recall, and distractors are disabled in this block.",
            "Press Enter to begin.",
        ),
        scored_segments=(
            _segment(
                label="Trigger Cues",
                duration_s=scored_duration_s,
                active_channels=("gates", "trigger"),
                profile=profile,
            ),
        ),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=normalized_mode,
        config=cfg,
    )


def build_ac_callsign_filter_run_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.TEMPO,
    config: AcDrillConfig | None = None,
) -> AuditoryCapacityContinuousDrill:
    cfg = config or AcDrillConfig()
    normalized_mode = _normalize_mode(mode)
    profile = _mode_scaled_profile(
        profile=AuditoryCapacityTrainingProfile(
            enable_digit_sequences=False,
            enable_trigger_cues=False,
            enable_distractors=True,
            gate_rate_scale=0.90,
            command_rate_scale=0.92,
            directive_rate_scale=0.95,
            disturbance_scale=0.95,
            tube_width_scale=1.00,
            tube_height_scale=1.00,
            noise_level_scale=0.72,
        ),
        mode=normalized_mode,
    )
    scored_duration_s = (
        ANT_DRILL_MODE_PROFILES[normalized_mode].scored_duration_s
        if cfg.scored_duration_s is None
        else float(cfg.scored_duration_s)
    )
    return _build_ac_drill(
        title_base="Auditory Capacity: Callsign Filter Run",
        instructions=(
            "Auditory Capacity: Callsign Filter Run",
            f"Mode: {ANT_DRILL_MODE_PROFILES[normalized_mode].label}",
            "Fly gates while filtering state commands and next-gate directives through the correct call sign.",
            "Distractors are active; digit recall and trigger cues stay off for this block.",
            "Press Enter to begin.",
        ),
        scored_segments=(
            _segment(
                label="Callsign Filter",
                duration_s=scored_duration_s,
                active_channels=("gates", "state_commands", "gate_directives", "distractors"),
                profile=profile,
            ),
        ),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=normalized_mode,
        config=cfg,
    )


def build_ac_mixed_tempo_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.TEMPO,
    config: AcDrillConfig | None = None,
) -> AuditoryCapacityContinuousDrill:
    cfg = config or AcDrillConfig()
    normalized_mode = _normalize_mode(mode)
    scored_duration_s = (
        ANT_DRILL_MODE_PROFILES[normalized_mode].scored_duration_s
        if cfg.scored_duration_s is None
        else float(cfg.scored_duration_s)
    )
    templates = (
        _segment(
            label="Gate Flight",
            duration_s=90.0,
            active_channels=("gates",),
            profile=_mode_scaled_profile(
                profile=AuditoryCapacityTrainingProfile(
                    enable_state_commands=False,
                    enable_gate_directives=False,
                    enable_digit_sequences=False,
                    enable_trigger_cues=False,
                    enable_distractors=False,
                    gate_rate_scale=0.80,
                    disturbance_scale=0.82,
                    tube_width_scale=1.08,
                    tube_height_scale=1.06,
                    noise_level_scale=0.0,
                ),
                mode=normalized_mode,
            ),
        ),
        _segment(
            label="State Commands",
            duration_s=90.0,
            active_channels=("gates", "state_commands"),
            profile=_mode_scaled_profile(
                profile=AuditoryCapacityTrainingProfile(
                    enable_gate_directives=False,
                    enable_digit_sequences=False,
                    enable_trigger_cues=False,
                    enable_distractors=False,
                    gate_rate_scale=0.70,
                    command_rate_scale=0.90,
                    disturbance_scale=0.84,
                    tube_width_scale=1.06,
                    tube_height_scale=1.04,
                    noise_level_scale=0.0,
                ),
                mode=normalized_mode,
            ),
        ),
        _segment(
            label="Gate Directives",
            duration_s=90.0,
            active_channels=("gates", "gate_directives", "distractors"),
            profile=_mode_scaled_profile(
                profile=AuditoryCapacityTrainingProfile(
                    enable_state_commands=False,
                    enable_digit_sequences=False,
                    enable_trigger_cues=False,
                    enable_distractors=True,
                    gate_rate_scale=0.82,
                    directive_rate_scale=0.95,
                    disturbance_scale=0.92,
                    noise_level_scale=0.30,
                ),
                mode=normalized_mode,
            ),
        ),
        _segment(
            label="Digit Recall",
            duration_s=90.0,
            active_channels=("gates", "digit_recall"),
            profile=_mode_scaled_profile(
                profile=AuditoryCapacityTrainingProfile(
                    enable_state_commands=False,
                    enable_gate_directives=False,
                    enable_trigger_cues=False,
                    enable_distractors=False,
                    gate_rate_scale=0.65,
                    sequence_rate_scale=0.95,
                    disturbance_scale=0.82,
                    tube_width_scale=1.04,
                    tube_height_scale=1.04,
                    noise_level_scale=0.0,
                    digit_sequence_min_len=5,
                    digit_sequence_max_len=6,
                ),
                mode=normalized_mode,
            ),
        ),
        _segment(
            label="Trigger + Callsign Filter",
            duration_s=90.0,
            active_channels=("gates", "state_commands", "gate_directives", "trigger", "distractors"),
            profile=_mode_scaled_profile(
                profile=AuditoryCapacityTrainingProfile(
                    enable_digit_sequences=False,
                    enable_distractors=True,
                    gate_rate_scale=0.92,
                    command_rate_scale=0.94,
                    directive_rate_scale=0.96,
                    beep_rate_scale=0.96,
                    disturbance_scale=0.96,
                    noise_level_scale=0.72,
                ),
                mode=normalized_mode,
            ),
        ),
        _segment(
            label="Full Mixed",
            duration_s=90.0,
            active_channels=AC_CHANNEL_ORDER,
            profile=_mode_scaled_profile(
                profile=AuditoryCapacityTrainingProfile(),
                mode=normalized_mode,
            ),
        ),
    )
    return _build_ac_drill(
        title_base="Auditory Capacity: Mixed Tempo",
        instructions=(
            "Auditory Capacity: Mixed Tempo",
            f"Mode: {ANT_DRILL_MODE_PROFILES[normalized_mode].label}",
            "The block repeats six 90-second segments: Gate Flight, State Commands, Gate Directives, Digit Recall, Trigger + Callsign Filter, then Full Mixed.",
            "Use the segment label in the live screen to anticipate the current channel focus.",
            "Press Enter to begin.",
        ),
        scored_segments=_repeat_segments(total_duration_s=scored_duration_s, templates=templates),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=normalized_mode,
        config=cfg,
    )


def build_ac_pressure_run_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.STRESS,
    config: AcDrillConfig | None = None,
) -> AuditoryCapacityContinuousDrill:
    cfg = config or AcDrillConfig()
    normalized_mode = _normalize_mode(mode)
    scored_duration_s = (
        ANT_DRILL_MODE_PROFILES[normalized_mode].scored_duration_s
        if cfg.scored_duration_s is None
        else float(cfg.scored_duration_s)
    )
    profile = _mode_scaled_profile(
        profile=AuditoryCapacityTrainingProfile(
            enable_gates=True,
            enable_state_commands=True,
            enable_gate_directives=True,
            enable_digit_sequences=True,
            enable_trigger_cues=True,
            enable_distractors=True,
            gate_rate_scale=1.18,
            command_rate_scale=1.16,
            directive_rate_scale=1.14,
            sequence_rate_scale=1.05,
            beep_rate_scale=1.12,
            response_window_scale=0.86,
            disturbance_scale=1.18,
            tube_width_scale=0.96,
            tube_height_scale=0.96,
            noise_level_scale=1.18,
            distortion_level_scale=1.12,
            digit_sequence_min_len=5,
            digit_sequence_max_len=6,
        ),
        mode=normalized_mode,
    )
    return _build_ac_drill(
        title_base="Auditory Capacity: Pressure Run",
        instructions=(
            "Auditory Capacity: Pressure Run",
            f"Mode: {ANT_DRILL_MODE_PROFILES[normalized_mode].label}",
            "All auditory channels stay active for the entire block under the hardest cadence and disturbance profile in this family.",
            "Keep moving through misses and recover on the next gate, command, recall, or trigger cue instead of freezing.",
            "Press Enter to begin.",
        ),
        scored_segments=(
            _segment(
                label="Pressure Run",
                duration_s=scored_duration_s,
                active_channels=AC_CHANNEL_ORDER,
                profile=profile,
            ),
        ),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=normalized_mode,
        config=cfg,
    )
