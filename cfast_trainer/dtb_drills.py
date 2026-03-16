from __future__ import annotations

from dataclasses import dataclass, replace

from .ant_drills import ANT_DRILL_MODE_PROFILES, AntDrillAttemptSummary, AntDrillMode
from .clock import Clock
from .cognitive_core import Phase, SeededRng, TestSnapshot, clamp01
from .rapid_tracking import (
    RapidTrackingConfig,
    RapidTrackingEngine,
    RapidTrackingPayload,
    RapidTrackingTrainingProfile,
    RapidTrackingTrainingSegment,
    build_rapid_tracking_test,
)
from .telemetry import TelemetryEvent, telemetry_events_from_engine


def _normalize_mode(mode: AntDrillMode | str) -> AntDrillMode:
    if isinstance(mode, AntDrillMode):
        return mode
    return AntDrillMode(str(mode).strip().lower())


def _difficulty_to_level(difficulty: float) -> int:
    return max(1, min(10, int(round(clamp01(difficulty) * 9.0)) + 1))


@dataclass(frozen=True, slots=True)
class DualTaskBridgeDrillConfig:
    scored_duration_s: float | None = None


@dataclass(frozen=True, slots=True)
class DualTaskBridgeStatus:
    channel: str
    filter_label: str
    prompt_text: str
    recall_input_active: bool
    visible_digits: str
    expected_response: str | None
    interference_active: bool
    recovery_active: bool
    latest_feedback: str


@dataclass(frozen=True, slots=True)
class _DualTaskCue:
    cue_index: int
    cue_kind: str
    start_s: float
    response_start_s: float
    deadline_s: float
    prompt_text: str
    filter_label: str
    expected_response: str | None
    visible_digits: str = ""
    interference_active: bool = False
    recovery_active: bool = False


@dataclass(frozen=True, slots=True)
class _ResolvedCue:
    cue: _DualTaskCue
    phase: Phase
    answered_at_s: float
    response_time_ms: int | None
    raw_response: str
    is_correct: bool
    is_timeout: bool


def _drill_config(
    mode: AntDrillMode,
    cfg: DualTaskBridgeDrillConfig | None,
) -> tuple[DualTaskBridgeDrillConfig, float]:
    profile = ANT_DRILL_MODE_PROFILES[mode]
    resolved = cfg or DualTaskBridgeDrillConfig()
    scored_duration_s = (
        profile.scored_duration_s
        if resolved.scored_duration_s is None
        else float(resolved.scored_duration_s)
    )
    return resolved, float(scored_duration_s)


def _profile_for_mode(
    mode: AntDrillMode,
    *,
    target_kinds: tuple[str, ...],
    active_challenges: tuple[str, ...],
    turbulence_scale: float,
    camera_assist: float,
    preview_scale: float,
    capture_box_scale: float,
) -> RapidTrackingTrainingProfile:
    if mode is AntDrillMode.FRESH:
        turbulence_scale *= 0.82
        camera_assist = max(camera_assist, 0.68)
        preview_scale *= 1.16
        capture_box_scale *= 1.16
    elif mode is AntDrillMode.BUILD:
        turbulence_scale *= 0.86
        camera_assist = max(camera_assist, 0.62)
        preview_scale *= 1.12
        capture_box_scale *= 1.12
    elif mode is AntDrillMode.PRESSURE:
        turbulence_scale *= 1.08
        camera_assist = min(camera_assist, 0.26)
        preview_scale *= 0.94
        capture_box_scale *= 0.94
    elif mode is AntDrillMode.RECOVERY:
        turbulence_scale *= 0.92
        camera_assist = max(camera_assist, 0.54)
        preview_scale *= 1.06
        capture_box_scale *= 1.08
    elif mode is AntDrillMode.STRESS:
        turbulence_scale *= 1.18
        camera_assist = min(camera_assist, 0.18)
        preview_scale *= 0.88
        capture_box_scale *= 0.88
    return RapidTrackingTrainingProfile(
        target_kinds=target_kinds,
        cover_modes=("open", "building", "terrain"),
        handoff_modes=("smooth", "jump"),
        turbulence_scale=max(0.0, float(turbulence_scale)),
        camera_assist_override=clamp01(float(camera_assist)),
        preview_duration_scale=max(0.3, float(preview_scale)),
        capture_box_scale=max(0.35, float(capture_box_scale)),
        capture_cooldown_scale=1.0,
        segment_duration_scale=1.0,
    )


class DualTaskBridgeDrill:
    _COLOR_KEYS = {
        "Q": "BLUE",
        "W": "GREEN",
        "E": "YELLOW",
        "R": "RED",
    }

    def __init__(
        self,
        *,
        title: str,
        instructions: tuple[str, ...],
        engine: RapidTrackingEngine,
        seed: int,
        difficulty: float,
        mode: AntDrillMode,
        scored_duration_s: float,
        bridge_label: str,
        bridge_focus: str,
        bridge_challenges: tuple[str, ...],
        cue_schedule: tuple[_DualTaskCue, ...],
    ) -> None:
        self._title = str(title)
        self._instructions = tuple(str(line) for line in instructions)
        self._engine = engine
        self._seed = int(seed)
        self._difficulty = float(difficulty)
        self._mode = mode
        self._scored_duration_s = float(scored_duration_s)
        self._bridge_label = str(bridge_label)
        self._bridge_focus = str(bridge_focus)
        self._bridge_challenges = tuple(str(item) for item in bridge_challenges)
        self._cue_schedule = tuple(cue_schedule)
        self._cue_cursor = 0
        self._active_cue: _DualTaskCue | None = None
        self._resolved_cues: list[_ResolvedCue] = []
        self._latest_feedback = ""
        self._feedback_until_s = 0.0

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

    def set_control(self, *, horizontal: float, vertical: float) -> None:
        self._engine.set_control(horizontal=horizontal, vertical=vertical)

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
        token = str(raw).strip()
        lowered = token.lower()
        if lowered in {"__skip_section__", "skip_section", "__skip_all__", "skip_all"}:
            if hasattr(self._engine, "_phase"):
                self._engine._phase = Phase.RESULTS
            return True
        if lowered == "capture":
            return bool(self._engine.submit_answer("CAPTURE"))

        cue = self._active_cue
        if cue is None or self._engine.phase not in (Phase.PRACTICE, Phase.SCORED):
            return False
        elapsed_s = self._phase_elapsed_s()

        if cue.cue_kind == "recall":
            if elapsed_s < cue.response_start_s:
                return False
            if lowered.startswith("digits:"):
                token = token.split(":", 1)[1]
            if not token.isdigit():
                return False
            return self._resolve_active_cue(
                raw_response=token,
                answered_at_s=elapsed_s,
                is_correct=token == cue.expected_response,
                is_timeout=False,
            )

        if cue.cue_kind == "command":
            response = token
            if lowered.startswith("cmd:"):
                response = token.split(":", 1)[1].strip().upper()
            else:
                response = self._COLOR_KEYS.get(token.upper(), token.upper())
            if response not in self._COLOR_KEYS.values():
                return False
            return self._resolve_active_cue(
                raw_response=response,
                answered_at_s=elapsed_s,
                is_correct=(cue.expected_response == response),
                is_timeout=False,
            )

        return False

    def update(self) -> None:
        self._engine.update()
        self._tick_bridge_state()

    def bridge_status(self) -> DualTaskBridgeStatus:
        cue = self._active_cue
        elapsed_s = self._phase_elapsed_s()
        if cue is None:
            next_cue = (
                None
                if self._cue_cursor >= len(self._cue_schedule)
                else self._cue_schedule[self._cue_cursor]
            )
            if next_cue is None:
                prompt = "Hold stable tracking. The bridge channels are clear for the remainder of this block."
            else:
                seconds = max(0, int(round(next_cue.start_s - elapsed_s)))
                prompt = (
                    f"Hold tracking stability. Next {next_cue.cue_kind.replace('_', ' ')} cue in {seconds}s."
                )
            return DualTaskBridgeStatus(
                channel="tracking_only",
                filter_label="",
                prompt_text=prompt,
                recall_input_active=False,
                visible_digits="",
                expected_response=None,
                interference_active=False,
                recovery_active=False,
                latest_feedback=self._current_feedback(elapsed_s),
            )
        return DualTaskBridgeStatus(
            channel=cue.cue_kind,
            filter_label=cue.filter_label,
            prompt_text=cue.prompt_text,
            recall_input_active=(cue.cue_kind == "recall" and elapsed_s >= cue.response_start_s),
            visible_digits=(cue.visible_digits if elapsed_s < cue.response_start_s else ""),
            expected_response=cue.expected_response,
            interference_active=cue.interference_active,
            recovery_active=cue.recovery_active,
            latest_feedback=self._current_feedback(elapsed_s),
        )

    def snapshot(self) -> TestSnapshot:
        snap = self._engine.snapshot()
        payload = snap.payload
        prompt = str(snap.prompt)
        input_hint = str(snap.input_hint)
        if snap.phase is Phase.INSTRUCTIONS:
            prompt = "\n".join(self._instructions)
            input_hint = "Press Enter to begin."
        elif snap.phase is Phase.PRACTICE_DONE:
            input_hint = "Press Enter to continue."
        elif isinstance(payload, RapidTrackingPayload):
            status = self.bridge_status()
            bridge_tags = tuple((*payload.active_challenges, *self._bridge_challenges))
            payload = replace(
                payload,
                focus_label=f"{self._bridge_focus} | {status.channel.replace('_', ' ')}",
                active_challenges=tuple(dict.fromkeys(bridge_tags)),
                segment_label=f"{self._bridge_label} | {payload.segment_label}",
            )
            prompt = status.prompt_text
            input_hint = (
                "Space capture | Q/W/E/R command filter | digits + Enter delayed report"
            )
        return TestSnapshot(
            title=self._title,
            phase=snap.phase,
            prompt=prompt,
            input_hint=input_hint,
            time_remaining_s=snap.time_remaining_s,
            attempted_scored=snap.attempted_scored,
            correct_scored=snap.correct_scored,
            payload=payload,
            practice_feedback=snap.practice_feedback,
        )

    def events(self) -> list[TelemetryEvent]:
        events = telemetry_events_from_engine(self._engine)
        seq = len(events)
        item_offset = max(
            (int(event.item_index) for event in events if event.item_index is not None),
            default=0,
        )
        difficulty_level = _difficulty_to_level(self._difficulty)
        for resolved in self._resolved_cues:
            cue = resolved.cue
            start_ms = int(round(cue.start_s * 1000.0))
            resolved_ms = int(round(resolved.answered_at_s * 1000.0))
            cue_extra = {
                "cue_kind": cue.cue_kind,
                "filter_label": cue.filter_label,
                "interference_active": cue.interference_active,
                "recovery_active": cue.recovery_active,
            }
            events.append(
                TelemetryEvent(
                    family="dual_task_bridge",
                    kind="cue_started",
                    phase=resolved.phase.value,
                    seq=seq,
                    item_index=None,
                    is_scored=False,
                    is_correct=None,
                    is_timeout=False,
                    response_time_ms=None,
                    score=None,
                    max_score=None,
                    difficulty_level=difficulty_level,
                    occurred_at_ms=start_ms,
                    prompt=cue.prompt_text,
                    expected=cue.expected_response,
                    extra=cue_extra,
                )
            )
            seq += 1
            events.append(
                TelemetryEvent(
                    family="dual_task_bridge",
                    kind=f"{cue.cue_kind}_resolved",
                    phase=resolved.phase.value,
                    seq=seq,
                    item_index=item_offset + cue.cue_index + 1,
                    is_scored=True,
                    is_correct=resolved.is_correct,
                    is_timeout=resolved.is_timeout,
                    response_time_ms=resolved.response_time_ms,
                    score=1.0 if resolved.is_correct else 0.0,
                    max_score=1.0,
                    difficulty_level=difficulty_level,
                    occurred_at_ms=resolved_ms,
                    prompt=cue.prompt_text,
                    expected=cue.expected_response,
                    response=resolved.raw_response,
                    extra=cue_extra,
                )
            )
            seq += 1
        return events

    def scored_summary(self) -> AntDrillAttemptSummary:
        base = self._engine.scored_summary()
        tracking_total = float(base.total_score) + float(base.capture_points)
        tracking_max = float(base.max_score) + float(base.capture_max_points)
        cue_attempted = len(self._resolved_cues)
        cue_correct = sum(1 for cue in self._resolved_cues if cue.is_correct)
        total_attempted = int(base.attempted) + cue_attempted
        total_correct = int(base.correct) + cue_correct
        total_score = tracking_total + float(cue_correct)
        max_score = tracking_max + float(cue_attempted)
        score_ratio = 0.0 if max_score <= 0.0 else total_score / max_score
        difficulty_level = _difficulty_to_level(self._difficulty)
        accuracy = 0.0 if total_attempted <= 0 else total_correct / float(total_attempted)
        throughput_per_min = (
            0.0 if base.duration_s <= 0.0 else (float(total_attempted) / float(base.duration_s)) * 60.0
        )
        correct_per_min = (
            0.0 if base.duration_s <= 0.0 else (float(total_correct) / float(base.duration_s)) * 60.0
        )
        return AntDrillAttemptSummary(
            attempted=total_attempted,
            correct=total_correct,
            accuracy=float(accuracy),
            duration_s=float(base.duration_s),
            throughput_per_min=float(throughput_per_min),
            mean_response_time_s=None,
            total_score=float(total_score),
            max_score=float(max_score),
            score_ratio=float(score_ratio),
            correct_per_min=float(correct_per_min),
            timeouts=sum(1 for cue in self._resolved_cues if cue.is_timeout),
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

    def result_metrics(self) -> dict[str, str]:
        command_attempted = sum(1 for cue in self._resolved_cues if cue.cue.cue_kind == "command")
        command_correct = sum(
            1 for cue in self._resolved_cues if cue.cue.cue_kind == "command" and cue.is_correct
        )
        recall_attempted = sum(1 for cue in self._resolved_cues if cue.cue.cue_kind == "recall")
        recall_correct = sum(
            1 for cue in self._resolved_cues if cue.cue.cue_kind == "recall" and cue.is_correct
        )
        false_alarms = sum(
            1
            for cue in self._resolved_cues
            if cue.cue.cue_kind == "command"
            and cue.cue.expected_response is None
            and not cue.is_correct
        )
        recovery_correct = sum(
            1 for cue in self._resolved_cues if cue.cue.recovery_active and cue.is_correct
        )
        return {
            "bridge.command_attempted": str(int(command_attempted)),
            "bridge.command_correct": str(int(command_correct)),
            "bridge.recall_attempted": str(int(recall_attempted)),
            "bridge.recall_correct": str(int(recall_correct)),
            "bridge.false_alarms": str(int(false_alarms)),
            "bridge.recovery_correct": str(int(recovery_correct)),
            "bridge.channel_count": str(int(len(self._resolved_cues))),
        }

    def _tick_bridge_state(self) -> None:
        if self._engine.phase not in (Phase.PRACTICE, Phase.SCORED):
            return
        elapsed_s = self._phase_elapsed_s()
        if self._active_cue is not None and elapsed_s >= self._active_cue.deadline_s:
            cue = self._active_cue
            is_correct = cue.expected_response is None
            self._resolve_active_cue(
                raw_response="",
                answered_at_s=cue.deadline_s,
                is_correct=is_correct,
                is_timeout=(cue.expected_response is not None),
            )
        if self._active_cue is None and self._cue_cursor < len(self._cue_schedule):
            next_cue = self._cue_schedule[self._cue_cursor]
            if elapsed_s >= next_cue.start_s:
                self._active_cue = next_cue
                self._cue_cursor += 1

    def _resolve_active_cue(
        self,
        *,
        raw_response: str,
        answered_at_s: float,
        is_correct: bool,
        is_timeout: bool,
    ) -> bool:
        cue = self._active_cue
        if cue is None:
            return False
        response_start_s = cue.response_start_s if cue.cue_kind == "recall" else cue.start_s
        response_time_ms = None
        if answered_at_s >= response_start_s:
            response_time_ms = int(round((answered_at_s - response_start_s) * 1000.0))
        self._resolved_cues.append(
            _ResolvedCue(
                cue=cue,
                phase=Phase.SCORED if self._engine.phase is Phase.SCORED else Phase.PRACTICE,
                answered_at_s=float(answered_at_s),
                response_time_ms=response_time_ms,
                raw_response=str(raw_response),
                is_correct=bool(is_correct),
                is_timeout=bool(is_timeout),
            )
        )
        self._active_cue = None
        label = "TIMEOUT" if is_timeout else "PASS" if is_correct else "MISS"
        self._latest_feedback = f"{cue.cue_kind.replace('_', ' ').title()}: {label}"
        self._feedback_until_s = float(answered_at_s) + 2.2
        return True

    def _current_feedback(self, elapsed_s: float) -> str:
        if elapsed_s <= self._feedback_until_s:
            return self._latest_feedback
        return ""

    def _phase_elapsed_s(self) -> float:
        snap = self._engine.snapshot()
        payload = snap.payload
        if isinstance(payload, RapidTrackingPayload):
            return float(payload.phase_elapsed_s)
        return 0.0


def _build_segment(
    *,
    label: str,
    focus_label: str,
    duration_s: float,
    active_target_kinds: tuple[str, ...],
    active_challenges: tuple[str, ...],
    profile: RapidTrackingTrainingProfile,
) -> tuple[RapidTrackingTrainingSegment, ...]:
    return (
        RapidTrackingTrainingSegment(
            label=label,
            duration_s=duration_s,
            focus_label=focus_label,
            active_target_kinds=active_target_kinds,
            active_challenges=active_challenges,
            profile=profile,
        ),
    )


def _digits_for(rng: SeededRng, *, length: int) -> str:
    return "".join(str(rng.randint(0, 9)) for _ in range(max(1, int(length))))


def _build_tracking_recall_schedule(
    *,
    rng: SeededRng,
    scored_duration_s: float,
    difficulty: float,
) -> tuple[_DualTaskCue, ...]:
    cues: list[_DualTaskCue] = []
    spacing = 24.0
    start_s = 9.0
    reveal_s = 2.3
    delay_s = 3.5
    response_window_s = 6.0
    digit_length = 3 if difficulty <= 0.35 else 4 if difficulty <= 0.7 else 5
    cue_index = 0
    while start_s + reveal_s + delay_s + response_window_s < scored_duration_s:
        digits = _digits_for(rng, length=digit_length)
        cues.append(
            _DualTaskCue(
                cue_index=cue_index,
                cue_kind="recall",
                start_s=start_s,
                response_start_s=start_s + reveal_s + delay_s,
                deadline_s=start_s + reveal_s + delay_s + response_window_s,
                prompt_text=f"Remember {digits}, keep tracking, then report the digits when the recall prompt opens.",
                filter_label="",
                expected_response=digits,
                visible_digits=digits,
            )
        )
        cue_index += 1
        start_s += spacing
    return tuple(cues)


def _build_command_schedule(
    *,
    rng: SeededRng,
    scored_duration_s: float,
    difficulty: float,
    include_recall: bool,
    include_interference: bool,
) -> tuple[_DualTaskCue, ...]:
    filters = ("BLUE/GREEN", "YELLOW/RED")
    filter_targets = {
        "BLUE/GREEN": ("BLUE", "GREEN"),
        "YELLOW/RED": ("YELLOW", "RED"),
    }
    colors = ("BLUE", "GREEN", "YELLOW", "RED")
    cues: list[_DualTaskCue] = []
    start_s = 7.0
    cue_index = 0
    filter_label = str(rng.choice(filters))
    recall_digit_length = 4 if difficulty <= 0.65 else 5
    while start_s + 4.0 < scored_duration_s:
        if include_recall and (cue_index % 4 == 3):
            digits = _digits_for(rng, length=recall_digit_length)
            cues.append(
                _DualTaskCue(
                    cue_index=cue_index,
                    cue_kind="recall",
                    start_s=start_s,
                    response_start_s=start_s + 2.0 + 3.0,
                    deadline_s=start_s + 2.0 + 3.0 + 5.5,
                    prompt_text=f"Remember {digits}. Keep tracking through the delay, then report the digits.",
                    filter_label=filter_label,
                    expected_response=digits,
                    visible_digits=digits,
                )
            )
            start_s += 11.5
            cue_index += 1
            continue

        interference_active = include_interference and (cue_index % 5 in {1, 2})
        recovery_active = include_interference and (cue_index % 5 == 3)
        if cue_index % 6 == 0:
            filter_label = str(rng.choice(filters))
        color = str(rng.choice(colors))
        targets = filter_targets[filter_label]
        expected = color if color in targets else None
        if interference_active:
            expected = None
        prompt = (
            f"Filter {filter_label}. {'Recover and answer the next target.' if recovery_active else ''} "
            f"Cue: {color}. "
        )
        if expected is None:
            prompt += "Hold response and keep the track stable."
        else:
            prompt += "Press the matching command key now."
        cues.append(
            _DualTaskCue(
                cue_index=cue_index,
                cue_kind="command",
                start_s=start_s,
                response_start_s=start_s,
                deadline_s=start_s + (2.0 if interference_active else 3.0),
                prompt_text=prompt,
                filter_label=filter_label,
                expected_response=expected,
                interference_active=interference_active,
                recovery_active=recovery_active,
            )
        )
        start_s += 4.2 if interference_active else 6.0
        cue_index += 1
    return tuple(cues)


def _build_dual_task_bridge(
    *,
    clock: Clock,
    seed: int,
    difficulty: float,
    mode: AntDrillMode | str,
    config: DualTaskBridgeDrillConfig | None,
    title: str,
    instructions: tuple[str, ...],
    bridge_label: str,
    bridge_focus: str,
    bridge_challenges: tuple[str, ...],
    active_target_kinds: tuple[str, ...],
    active_challenges: tuple[str, ...],
    turbulence_scale: float,
    camera_assist: float,
    preview_scale: float,
    capture_box_scale: float,
    cue_schedule_builder,
) -> DualTaskBridgeDrill:
    normalized_mode = _normalize_mode(mode)
    _resolved_cfg, scored_duration_s = _drill_config(normalized_mode, config)
    profile = _profile_for_mode(
        normalized_mode,
        target_kinds=active_target_kinds,
        active_challenges=active_challenges,
        turbulence_scale=turbulence_scale,
        camera_assist=camera_assist,
        preview_scale=preview_scale,
        capture_box_scale=capture_box_scale,
    )
    engine = build_rapid_tracking_test(
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        title=title,
        config=RapidTrackingConfig(practice_duration_s=0.0, scored_duration_s=scored_duration_s),
        scored_segments=_build_segment(
            label=bridge_label,
            focus_label=bridge_focus,
            duration_s=scored_duration_s,
            active_target_kinds=active_target_kinds,
            active_challenges=active_challenges,
            profile=profile,
        ),
    )
    rng = SeededRng(seed + 701)
    schedule = cue_schedule_builder(rng=rng, scored_duration_s=scored_duration_s, difficulty=difficulty)
    return DualTaskBridgeDrill(
        title=title,
        instructions=instructions,
        engine=engine,
        seed=seed,
        difficulty=difficulty,
        mode=normalized_mode,
        scored_duration_s=scored_duration_s,
        bridge_label=bridge_label,
        bridge_focus=bridge_focus,
        bridge_challenges=bridge_challenges,
        cue_schedule=schedule,
    )


def build_dtb_tracking_recall_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: DualTaskBridgeDrillConfig | None = None,
) -> DualTaskBridgeDrill:
    normalized_mode = _normalize_mode(mode)
    profile = ANT_DRILL_MODE_PROFILES[normalized_mode]
    return _build_dual_task_bridge(
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=normalized_mode,
        config=config,
        title="Dual-Task Bridge: Tracking + Recall",
        instructions=(
            "Dual-Task Bridge: Tracking + Recall",
            "",
            f"Mode: {profile.label}",
            "Stay on the Rapid Tracking scene while short digit holds arrive in the background.",
            "Track first, then type the delayed digit report only when the recall prompt opens.",
            "Space still captures. Digits plus Enter handle the report.",
        ),
        bridge_label="Tracking + Recall",
        bridge_focus="Tracking stability with delayed digit hold",
        bridge_challenges=("bridge_recall",),
        active_target_kinds=("soldier", "truck"),
        active_challenges=("lock_quality", "ground_tempo"),
        turbulence_scale=0.56,
        camera_assist=0.64,
        preview_scale=1.10,
        capture_box_scale=1.05,
        cue_schedule_builder=_build_tracking_recall_schedule,
    )


def build_dtb_tracking_command_filter_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: DualTaskBridgeDrillConfig | None = None,
) -> DualTaskBridgeDrill:
    normalized_mode = _normalize_mode(mode)
    profile = ANT_DRILL_MODE_PROFILES[normalized_mode]
    return _build_dual_task_bridge(
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=normalized_mode,
        config=config,
        title="Dual-Task Bridge: Tracking + Command Filter",
        instructions=(
            "Dual-Task Bridge: Tracking + Command Filter",
            "",
            f"Mode: {profile.label}",
            "Hold the track while command-color cues appear on top of the scene.",
            "Respond only to the active filter set and ignore the rest without losing tracking discipline.",
            "Space captures. Q/W/E/R answer command cues.",
        ),
        bridge_label="Tracking + Command Filter",
        bridge_focus="Tracking with clean go/no-go command filtering",
        bridge_challenges=("bridge_command_filter",),
        active_target_kinds=("soldier", "truck", "building"),
        active_challenges=("lock_quality", "handoff_reacquisition"),
        turbulence_scale=0.60,
        camera_assist=0.58,
        preview_scale=1.04,
        capture_box_scale=1.00,
        cue_schedule_builder=lambda **kwargs: _build_command_schedule(
            include_recall=False,
            include_interference=False,
            **kwargs,
        ),
    )


def build_dtb_tracking_filter_digit_report_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float,
    mode: AntDrillMode | str = AntDrillMode.TEMPO,
    config: DualTaskBridgeDrillConfig | None = None,
) -> DualTaskBridgeDrill:
    normalized_mode = _normalize_mode(mode)
    profile = ANT_DRILL_MODE_PROFILES[normalized_mode]
    return _build_dual_task_bridge(
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=normalized_mode,
        config=config,
        title="Dual-Task Bridge: Tracking + Filter + Digit Report",
        instructions=(
            "Dual-Task Bridge: Tracking + Filter + Digit Report",
            "",
            f"Mode: {profile.label}",
            "Run the command filter while delayed digit reports keep appearing underneath the tracking task.",
            "The bridge is deliberate: do not jump to full overload, just keep the lanes clean.",
            "Space captures. Q/W/E/R filter commands. Digits plus Enter report the delayed digits.",
        ),
        bridge_label="Tracking + Filter + Digit Report",
        bridge_focus="Tracking with mild command-filter and delayed digit load",
        bridge_challenges=("bridge_command_filter", "bridge_recall"),
        active_target_kinds=("soldier", "truck", "building"),
        active_challenges=("lock_quality", "ground_tempo", "handoff_reacquisition"),
        turbulence_scale=0.64,
        camera_assist=0.52,
        preview_scale=0.98,
        capture_box_scale=0.96,
        cue_schedule_builder=lambda **kwargs: _build_command_schedule(
            include_recall=True,
            include_interference=False,
            **kwargs,
        ),
    )


def build_dtb_tracking_interference_recovery_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float,
    mode: AntDrillMode | str = AntDrillMode.STRESS,
    config: DualTaskBridgeDrillConfig | None = None,
) -> DualTaskBridgeDrill:
    normalized_mode = _normalize_mode(mode)
    profile = ANT_DRILL_MODE_PROFILES[normalized_mode]
    return _build_dual_task_bridge(
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=normalized_mode,
        config=config,
        title="Dual-Task Bridge: Tracking + Interference + Recovery",
        instructions=(
            "Dual-Task Bridge: Tracking + Interference + Recovery",
            "",
            f"Mode: {profile.label}",
            "Interference bursts arrive in tight clusters, then the recovery cue asks you to respond cleanly.",
            "Do not chase every cue. Ignore the clutter, recover on the next valid target, and keep the track stable.",
            "Space captures. Q/W/E/R answer valid recovery cues.",
        ),
        bridge_label="Tracking + Interference + Recovery",
        bridge_focus="Tracking recovery after filtered interference bursts",
        bridge_challenges=("bridge_command_filter", "bridge_interference_recovery"),
        active_target_kinds=("soldier", "truck", "helicopter"),
        active_challenges=("lock_quality", "ground_tempo", "air_speed"),
        turbulence_scale=0.72,
        camera_assist=0.34,
        preview_scale=0.92,
        capture_box_scale=0.92,
        cue_schedule_builder=lambda **kwargs: _build_command_schedule(
            include_recall=False,
            include_interference=True,
            **kwargs,
        ),
    )
