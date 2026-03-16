from __future__ import annotations

from dataclasses import dataclass

from .ant_drills import ANT_DRILL_MODE_PROFILES, AntDrillAttemptSummary, AntDrillMode
from .clock import Clock
from .cognitive_core import Phase, TestSnapshot, clamp01
from .rapid_tracking import (
    RAPID_TRACKING_CHALLENGE_ORDER,
    RAPID_TRACKING_TARGET_KIND_ORDER,
    RapidTrackingConfig,
    RapidTrackingEngine,
    RapidTrackingPayload,
    RapidTrackingTrainingProfile,
    RapidTrackingTrainingSegment,
    build_rapid_tracking_test,
)


def _normalize_mode(mode: AntDrillMode | str) -> AntDrillMode:
    if isinstance(mode, AntDrillMode):
        return mode
    return AntDrillMode(str(mode).strip().lower())


def _difficulty_to_level(difficulty: float) -> int:
    return max(1, min(10, int(round(clamp01(difficulty) * 9.0)) + 1))


def _labelize(token: str) -> str:
    return str(token).replace("_", " ").title()


@dataclass(frozen=True, slots=True)
class RtDrillConfig:
    scored_duration_s: float | None = None


class RapidTrackingContinuousDrill:
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
            if hasattr(self._engine, "_phase"):
                self._engine._phase = Phase.RESULTS
            return True
        return self._engine.submit_answer(raw)

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
        elif isinstance(payload, RapidTrackingPayload):
            targets = ", ".join(_labelize(kind) for kind in payload.active_target_kinds)
            challenges = ", ".join(_labelize(challenge) for challenge in payload.active_challenges)
            input_hint = (
                f"{payload.segment_label} | Focus: {payload.focus_label} | "
                f"Targets: {targets} | Challenges: {challenges} | {snap.input_hint}"
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

    def events(self):
        return self._engine.events()

    def scored_summary(self) -> AntDrillAttemptSummary:
        base = self._engine.scored_summary()
        combined_total = float(base.total_score) + float(base.capture_points)
        combined_max = float(base.max_score) + float(base.capture_max_points)
        combined_ratio = 0.0 if combined_max <= 0.0 else combined_total / combined_max
        difficulty_level = _difficulty_to_level(self._difficulty)
        correct_per_min = 0.0 if base.duration_s <= 0.0 else (base.correct / base.duration_s) * 60.0
        return AntDrillAttemptSummary(
            attempted=base.attempted,
            correct=base.correct,
            accuracy=base.accuracy,
            duration_s=base.duration_s,
            throughput_per_min=base.throughput_per_min,
            mean_response_time_s=None,
            total_score=combined_total,
            max_score=combined_max,
            score_ratio=combined_ratio,
            correct_per_min=correct_per_min,
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
    templates: tuple[RapidTrackingTrainingSegment, ...],
) -> tuple[RapidTrackingTrainingSegment, ...]:
    remaining = max(0.0, float(total_duration_s))
    built: list[RapidTrackingTrainingSegment] = []
    idx = 0
    while remaining > 1e-9:
        template = templates[idx % len(templates)]
        seg_duration = min(float(template.duration_s), remaining)
        if seg_duration <= 0.0:
            break
        built.append(
            RapidTrackingTrainingSegment(
                label=template.label,
                duration_s=seg_duration,
                focus_label=template.focus_label,
                active_target_kinds=template.active_target_kinds,
                active_challenges=template.active_challenges,
                profile=template.profile,
            )
        )
        remaining -= seg_duration
        idx += 1
    return tuple(built)


def _profile_for_mode(
    mode: AntDrillMode,
    *,
    target_kinds: tuple[str, ...],
    cover_modes: tuple[str, ...],
    handoff_modes: tuple[str, ...],
    turbulence_scale: float,
    camera_assist: float | None,
    preview_scale: float,
    capture_box_scale: float,
    capture_cooldown_scale: float,
    segment_duration_scale: float,
) -> RapidTrackingTrainingProfile:
    if mode is AntDrillMode.FRESH:
        turbulence = turbulence_scale * 0.78
        assist = camera_assist if camera_assist is None else max(camera_assist, 0.62)
        preview = preview_scale * 1.20
        box = capture_box_scale * 1.18
        cooldown = capture_cooldown_scale * 0.90
        duration = segment_duration_scale * 1.16
    elif mode is AntDrillMode.BUILD:
        turbulence = turbulence_scale * 0.82
        assist = camera_assist if camera_assist is None else max(camera_assist, 0.55)
        preview = preview_scale * 1.16
        box = capture_box_scale * 1.14
        cooldown = capture_cooldown_scale * 0.92
        duration = segment_duration_scale * 1.12
    elif mode is AntDrillMode.PRESSURE:
        turbulence = turbulence_scale * 1.08
        assist = camera_assist if camera_assist is None else min(camera_assist, 0.28)
        preview = preview_scale * 0.92
        box = capture_box_scale * 0.92
        cooldown = capture_cooldown_scale * 0.92
        duration = segment_duration_scale * 0.92
    elif mode is AntDrillMode.RECOVERY:
        turbulence = turbulence_scale * 0.90
        assist = camera_assist if camera_assist is None else max(camera_assist, 0.50)
        preview = preview_scale * 1.08
        box = capture_box_scale * 1.10
        cooldown = capture_cooldown_scale * 0.96
        duration = segment_duration_scale * 1.04
    elif mode is AntDrillMode.STRESS:
        turbulence = turbulence_scale * 1.20
        assist = 0.0 if camera_assist is None else min(camera_assist, 0.10)
        preview = preview_scale * 0.82
        box = capture_box_scale * 0.84
        cooldown = capture_cooldown_scale * 0.84
        duration = segment_duration_scale * 0.84
    else:
        turbulence = turbulence_scale
        assist = camera_assist
        preview = preview_scale
        box = capture_box_scale
        cooldown = capture_cooldown_scale
        duration = segment_duration_scale
    return RapidTrackingTrainingProfile(
        target_kinds=target_kinds,
        cover_modes=cover_modes,
        handoff_modes=handoff_modes,
        turbulence_scale=max(0.0, float(turbulence)),
        camera_assist_override=None if assist is None else clamp01(float(assist)),
        preview_duration_scale=max(0.2, float(preview)),
        capture_box_scale=max(0.35, float(box)),
        capture_cooldown_scale=max(0.2, float(cooldown)),
        segment_duration_scale=max(0.3, float(duration)),
    )


def _drill_config(mode: AntDrillMode, cfg: RtDrillConfig | None) -> tuple[RtDrillConfig, float]:
    profile = ANT_DRILL_MODE_PROFILES[mode]
    resolved = cfg or RtDrillConfig()
    scored_duration_s = (
        profile.scored_duration_s
        if resolved.scored_duration_s is None
        else float(resolved.scored_duration_s)
    )
    return resolved, float(scored_duration_s)


def _build_rt_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float,
    mode: AntDrillMode | str,
    config: RtDrillConfig | None,
    test_code: str,
    title: str,
    instructions: tuple[str, ...],
    segments: tuple[RapidTrackingTrainingSegment, ...],
) -> RapidTrackingContinuousDrill:
    normalized_mode = _normalize_mode(mode)
    _resolved_cfg, scored_duration_s = _drill_config(normalized_mode, config)
    engine = build_rapid_tracking_test(
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        title=title,
        config=RapidTrackingConfig(
            practice_duration_s=0.0,
            scored_duration_s=scored_duration_s,
        ),
        scored_segments=segments,
    )
    return RapidTrackingContinuousDrill(
        title=title,
        instructions=instructions,
        engine=engine,
        seed=seed,
        difficulty=difficulty,
        mode=normalized_mode,
        scored_duration_s=scored_duration_s,
    )


def _single_segment(
    *,
    label: str,
    focus_label: str,
    active_target_kinds: tuple[str, ...],
    active_challenges: tuple[str, ...],
    profile: RapidTrackingTrainingProfile,
    duration_s: float,
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


def build_rt_lock_anchor_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: RtDrillConfig | None = None,
) -> RapidTrackingContinuousDrill:
    normalized_mode = _normalize_mode(mode)
    _, scored_duration_s = _drill_config(normalized_mode, config)
    return _build_rt_drill(
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=normalized_mode,
        config=config,
        test_code="rt_lock_anchor",
        title="Rapid Tracking: Lock Anchor",
        instructions=(
            "Rapid Tracking: Lock Anchor",
            "",
            "Work the open soldier and truck tracks first.",
            "Keep the target centered, let the HUD lock settle, and capture only when the center box is clean.",
            "Capture controls stay live: configured trigger, Space, or left-click.",
        ),
        segments=_single_segment(
            label="Lock Anchor",
            focus_label="Stable lock quality",
            active_target_kinds=("soldier", "truck"),
            active_challenges=("lock_quality",),
            profile=_profile_for_mode(
                normalized_mode,
                target_kinds=("soldier", "truck"),
                cover_modes=("open",),
                handoff_modes=("smooth",),
                turbulence_scale=0.52,
                camera_assist=0.72,
                preview_scale=1.18,
                capture_box_scale=1.12,
                capture_cooldown_scale=0.92,
                segment_duration_scale=1.16,
            ),
            duration_s=scored_duration_s,
        ),
    )


def build_rt_building_handoff_prime_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: RtDrillConfig | None = None,
) -> RapidTrackingContinuousDrill:
    normalized_mode = _normalize_mode(mode)
    _, scored_duration_s = _drill_config(normalized_mode, config)
    return _build_rt_drill(
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=normalized_mode,
        config=config,
        test_code="rt_building_handoff_prime",
        title="Rapid Tracking: Building Handoff Prime",
        instructions=(
            "Rapid Tracking: Building Handoff Prime",
            "",
            "Track the building hold, then reacquire the next emergence cleanly.",
            "This block biases preview and handoff stability over raw speed.",
            "Capture stays live during the handoff.",
        ),
        segments=_single_segment(
            label="Building Handoff",
            focus_label="Handoff reacquisition",
            active_target_kinds=("building", "soldier", "truck"),
            active_challenges=("handoff_reacquisition",),
            profile=_profile_for_mode(
                normalized_mode,
                target_kinds=("building", "soldier", "truck"),
                cover_modes=("open", "building"),
                handoff_modes=("smooth", "jump"),
                turbulence_scale=0.60,
                camera_assist=0.55,
                preview_scale=1.28,
                capture_box_scale=1.04,
                capture_cooldown_scale=0.95,
                segment_duration_scale=1.06,
            ),
            duration_s=scored_duration_s,
        ),
    )


def build_rt_terrain_recovery_run_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: RtDrillConfig | None = None,
) -> RapidTrackingContinuousDrill:
    normalized_mode = _normalize_mode(mode)
    _, scored_duration_s = _drill_config(normalized_mode, config)
    return _build_rt_drill(
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=normalized_mode,
        config=config,
        test_code="rt_terrain_recovery_run",
        title="Rapid Tracking: Terrain Recovery Run",
        instructions=(
            "Rapid Tracking: Terrain Recovery Run",
            "",
            "Use terrain losses to train prediction and quick reacquisition.",
            "Expect soldier, truck, and helicopter targets to disappear behind ridges and come back fast.",
            "Capture remains active during recovery windows.",
        ),
        segments=_single_segment(
            label="Terrain Recovery",
            focus_label="Occlusion recovery",
            active_target_kinds=("soldier", "truck", "helicopter"),
            active_challenges=("occlusion_recovery", "handoff_reacquisition"),
            profile=_profile_for_mode(
                normalized_mode,
                target_kinds=("soldier", "truck", "helicopter"),
                cover_modes=("terrain",),
                handoff_modes=("smooth",),
                turbulence_scale=0.78,
                camera_assist=0.34,
                preview_scale=0.96,
                capture_box_scale=1.0,
                capture_cooldown_scale=0.94,
                segment_duration_scale=0.96,
            ),
            duration_s=scored_duration_s,
        ),
    )


def build_rt_capture_timing_prime_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: RtDrillConfig | None = None,
) -> RapidTrackingContinuousDrill:
    normalized_mode = _normalize_mode(mode)
    _, scored_duration_s = _drill_config(normalized_mode, config)
    return _build_rt_drill(
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=normalized_mode,
        config=config,
        test_code="rt_capture_timing_prime",
        title="Rapid Tracking: Capture Timing Prime",
        instructions=(
            "Rapid Tracking: Capture Timing Prime",
            "",
            "Keep tracking live, but bias the block toward cleaner camera-box captures.",
            "Targets run a little slower and the capture box is more forgiving than the pressure sets.",
        ),
        segments=_single_segment(
            label="Capture Timing",
            focus_label="Capture timing",
            active_target_kinds=("soldier", "truck", "helicopter"),
            active_challenges=("capture_timing",),
            profile=_profile_for_mode(
                normalized_mode,
                target_kinds=("soldier", "truck", "helicopter"),
                cover_modes=("open", "terrain"),
                handoff_modes=("smooth",),
                turbulence_scale=0.48,
                camera_assist=0.44,
                preview_scale=1.12,
                capture_box_scale=1.26,
                capture_cooldown_scale=0.80,
                segment_duration_scale=1.14,
            ),
            duration_s=scored_duration_s,
        ),
    )


def build_rt_ground_tempo_run_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: RtDrillConfig | None = None,
) -> RapidTrackingContinuousDrill:
    normalized_mode = _normalize_mode(mode)
    _, scored_duration_s = _drill_config(normalized_mode, config)
    return _build_rt_drill(
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=normalized_mode,
        config=config,
        test_code="rt_ground_tempo_run",
        title="Rapid Tracking: Ground Tempo Run",
        instructions=(
            "Rapid Tracking: Ground Tempo Run",
            "",
            "Ground-only tempo block with quicker soldier and truck changes and less assist than the anchors.",
            "Keep the lock clean while the script speeds up.",
        ),
        segments=_single_segment(
            label="Ground Tempo",
            focus_label="Ground tempo",
            active_target_kinds=("soldier", "truck"),
            active_challenges=("ground_tempo", "lock_quality"),
            profile=_profile_for_mode(
                normalized_mode,
                target_kinds=("soldier", "truck"),
                cover_modes=("open", "terrain"),
                handoff_modes=("smooth",),
                turbulence_scale=0.86,
                camera_assist=0.20,
                preview_scale=0.90,
                capture_box_scale=0.98,
                capture_cooldown_scale=0.90,
                segment_duration_scale=0.82,
            ),
            duration_s=scored_duration_s,
        ),
    )


def build_rt_air_speed_run_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: RtDrillConfig | None = None,
) -> RapidTrackingContinuousDrill:
    normalized_mode = _normalize_mode(mode)
    _, scored_duration_s = _drill_config(normalized_mode, config)
    return _build_rt_drill(
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=normalized_mode,
        config=config,
        test_code="rt_air_speed_run",
        title="Rapid Tracking: Air Speed Run",
        instructions=(
            "Rapid Tracking: Air Speed Run",
            "",
            "Bias the script to helicopter and jet passes with tighter preview timing and faster speed changes.",
            "Capture stays active, but the main cost is staying ahead of the air target.",
        ),
        segments=_single_segment(
            label="Air Speed",
            focus_label="Air-speed tracking",
            active_target_kinds=("helicopter", "jet"),
            active_challenges=("air_speed", "capture_timing"),
            profile=_profile_for_mode(
                normalized_mode,
                target_kinds=("helicopter", "jet"),
                cover_modes=("open", "terrain"),
                handoff_modes=("smooth", "jump"),
                turbulence_scale=1.00,
                camera_assist=0.10,
                preview_scale=0.78,
                capture_box_scale=0.90,
                capture_cooldown_scale=0.86,
                segment_duration_scale=0.70,
            ),
            duration_s=scored_duration_s,
        ),
    )


def build_rt_mixed_tempo_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float,
    mode: AntDrillMode | str = AntDrillMode.TEMPO,
    config: RtDrillConfig | None = None,
) -> RapidTrackingContinuousDrill:
    normalized_mode = _normalize_mode(mode)
    _, scored_duration_s = _drill_config(normalized_mode, config)
    templates = (
        RapidTrackingTrainingSegment(
            label="Lock Anchor",
            duration_s=60.0,
            focus_label="Stable lock quality",
            active_target_kinds=("soldier", "truck"),
            active_challenges=("lock_quality",),
            profile=_profile_for_mode(
                normalized_mode,
                target_kinds=("soldier", "truck"),
                cover_modes=("open",),
                handoff_modes=("smooth",),
                turbulence_scale=0.52,
                camera_assist=0.72,
                preview_scale=1.18,
                capture_box_scale=1.12,
                capture_cooldown_scale=0.92,
                segment_duration_scale=1.16,
            ),
        ),
        RapidTrackingTrainingSegment(
            label="Building Handoff",
            duration_s=60.0,
            focus_label="Handoff reacquisition",
            active_target_kinds=("building", "soldier", "truck"),
            active_challenges=("handoff_reacquisition",),
            profile=_profile_for_mode(
                normalized_mode,
                target_kinds=("building", "soldier", "truck"),
                cover_modes=("open", "building"),
                handoff_modes=("smooth", "jump"),
                turbulence_scale=0.60,
                camera_assist=0.55,
                preview_scale=1.28,
                capture_box_scale=1.04,
                capture_cooldown_scale=0.95,
                segment_duration_scale=1.06,
            ),
        ),
        RapidTrackingTrainingSegment(
            label="Terrain Recovery",
            duration_s=60.0,
            focus_label="Occlusion recovery",
            active_target_kinds=("soldier", "truck", "helicopter"),
            active_challenges=("occlusion_recovery", "handoff_reacquisition"),
            profile=_profile_for_mode(
                normalized_mode,
                target_kinds=("soldier", "truck", "helicopter"),
                cover_modes=("terrain",),
                handoff_modes=("smooth",),
                turbulence_scale=0.78,
                camera_assist=0.34,
                preview_scale=0.96,
                capture_box_scale=1.0,
                capture_cooldown_scale=0.94,
                segment_duration_scale=0.96,
            ),
        ),
        RapidTrackingTrainingSegment(
            label="Capture Timing",
            duration_s=60.0,
            focus_label="Capture timing",
            active_target_kinds=("soldier", "truck", "helicopter"),
            active_challenges=("capture_timing",),
            profile=_profile_for_mode(
                normalized_mode,
                target_kinds=("soldier", "truck", "helicopter"),
                cover_modes=("open", "terrain"),
                handoff_modes=("smooth",),
                turbulence_scale=0.48,
                camera_assist=0.44,
                preview_scale=1.12,
                capture_box_scale=1.26,
                capture_cooldown_scale=0.80,
                segment_duration_scale=1.14,
            ),
        ),
        RapidTrackingTrainingSegment(
            label="Ground Tempo",
            duration_s=60.0,
            focus_label="Ground tempo",
            active_target_kinds=("soldier", "truck"),
            active_challenges=("ground_tempo", "lock_quality"),
            profile=_profile_for_mode(
                normalized_mode,
                target_kinds=("soldier", "truck"),
                cover_modes=("open", "terrain"),
                handoff_modes=("smooth",),
                turbulence_scale=0.86,
                camera_assist=0.20,
                preview_scale=0.90,
                capture_box_scale=0.98,
                capture_cooldown_scale=0.90,
                segment_duration_scale=0.82,
            ),
        ),
        RapidTrackingTrainingSegment(
            label="Air Speed",
            duration_s=60.0,
            focus_label="Air-speed tracking",
            active_target_kinds=("helicopter", "jet"),
            active_challenges=("air_speed", "capture_timing"),
            profile=_profile_for_mode(
                normalized_mode,
                target_kinds=("helicopter", "jet"),
                cover_modes=("open", "terrain"),
                handoff_modes=("smooth", "jump"),
                turbulence_scale=1.00,
                camera_assist=0.10,
                preview_scale=0.78,
                capture_box_scale=0.90,
                capture_cooldown_scale=0.86,
                segment_duration_scale=0.70,
            ),
        ),
    )
    return _build_rt_drill(
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=normalized_mode,
        config=config,
        test_code="rt_mixed_tempo",
        title="Rapid Tracking: Mixed Tempo",
        instructions=(
            "Rapid Tracking: Mixed Tempo",
            "",
            "Work the fixed six-segment cycle without losing the live capture rhythm.",
            "Lock anchor, building handoff, terrain recovery, capture timing, ground tempo, and air speed repeat for the whole block.",
        ),
        segments=_repeat_segments(total_duration_s=scored_duration_s, templates=templates),
    )


def build_rt_pressure_run_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float,
    mode: AntDrillMode | str = AntDrillMode.STRESS,
    config: RtDrillConfig | None = None,
) -> RapidTrackingContinuousDrill:
    normalized_mode = _normalize_mode(mode)
    _, scored_duration_s = _drill_config(normalized_mode, config)
    return _build_rt_drill(
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=normalized_mode,
        config=config,
        test_code="rt_pressure_run",
        title="Rapid Tracking: Pressure Run",
        instructions=(
            "Rapid Tracking: Pressure Run",
            "",
            "All target kinds, all cover modes, and all handoff modes stay live for the full block.",
            "Expect minimal assist, stronger turbulence, tighter previews, and the smallest capture box in this family.",
        ),
        segments=_single_segment(
            label="Pressure Run",
            focus_label="Full pressure tracking",
            active_target_kinds=RAPID_TRACKING_TARGET_KIND_ORDER,
            active_challenges=RAPID_TRACKING_CHALLENGE_ORDER,
            profile=_profile_for_mode(
                normalized_mode,
                target_kinds=RAPID_TRACKING_TARGET_KIND_ORDER,
                cover_modes=("open", "building", "terrain"),
                handoff_modes=("smooth", "jump"),
                turbulence_scale=1.24,
                camera_assist=0.0,
                preview_scale=0.70,
                capture_box_scale=0.78,
                capture_cooldown_scale=0.82,
                segment_duration_scale=0.68,
            ),
            duration_s=scored_duration_s,
        ),
    )
