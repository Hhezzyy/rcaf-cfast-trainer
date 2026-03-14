from __future__ import annotations

from dataclasses import dataclass

from .ant_drills import ANT_DRILL_MODE_PROFILES, AntDrillAttemptSummary, AntDrillMode
from .clock import Clock
from .cognitive_core import Phase, TestSnapshot, clamp01
from .situational_awareness import (
    SA_CHANNEL_ORDER,
    SA_QUERY_KIND_ORDER,
    SituationalAwarenessConfig,
    SituationalAwarenessPayload,
    SituationalAwarenessQueryKind,
    SituationalAwarenessScenarioFamily,
    SituationalAwarenessTest,
    SituationalAwarenessTrainingProfile,
    SituationalAwarenessTrainingSegment,
    build_situational_awareness_test,
)

SA_FAMILY_CYCLE = (
    SituationalAwarenessScenarioFamily.MERGE_CONFLICT,
    SituationalAwarenessScenarioFamily.FUEL_PRIORITY,
    SituationalAwarenessScenarioFamily.ROUTE_HANDOFF,
    SituationalAwarenessScenarioFamily.CHANNEL_WAYPOINT_CHANGE,
)


def _normalize_mode(mode: AntDrillMode | str) -> AntDrillMode:
    if isinstance(mode, AntDrillMode):
        return mode
    return AntDrillMode(str(mode).strip().lower())


def _difficulty_to_level(difficulty: float) -> int:
    return max(1, min(10, int(round(clamp01(difficulty) * 9.0)) + 1))


def _query_label(kind_name: str) -> str:
    return str(kind_name).replace("_", " ").title()


@dataclass(frozen=True, slots=True)
class SaDrillConfig:
    scored_duration_s: float | None = None


class SituationalAwarenessContinuousDrill:
    def __init__(
        self,
        *,
        title: str,
        instructions: tuple[str, ...],
        engine: SituationalAwarenessTest,
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
        elif isinstance(payload, SituationalAwarenessPayload):
            channel_text = ", ".join(channel.title() for channel in payload.active_channels)
            query_text = ", ".join(_query_label(kind) for kind in payload.active_query_kinds)
            input_hint = (
                f"{payload.segment_label} | Focus: {payload.focus_label} | "
                f"Channels: {channel_text} | Queries: {query_text} | {snap.input_hint}"
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
        scored_events = [event for event in self._engine.events() if event.phase is Phase.SCORED]
        timeouts = 0
        max_timeout_streak = 0
        current_timeout_streak = 0
        for event in scored_events:
            is_timeout = (not event.is_correct) and str(event.raw).strip() == ""
            if is_timeout:
                timeouts += 1
                current_timeout_streak += 1
                max_timeout_streak = max(max_timeout_streak, current_timeout_streak)
            else:
                current_timeout_streak = 0
        difficulty_level = _difficulty_to_level(self._difficulty)
        correct_per_min = 0.0 if base.duration_s <= 0.0 else (base.correct / base.duration_s) * 60.0
        fixation_rate = 0.0 if base.attempted <= 0 else timeouts / base.attempted
        return AntDrillAttemptSummary(
            attempted=base.attempted,
            correct=base.correct,
            accuracy=base.accuracy,
            duration_s=base.duration_s,
            throughput_per_min=base.throughput_per_min,
            mean_response_time_s=base.mean_response_time_s,
            total_score=base.total_score,
            max_score=base.max_score,
            score_ratio=base.score_ratio,
            correct_per_min=correct_per_min,
            timeouts=timeouts,
            fixation_rate=fixation_rate,
            max_timeout_streak=max_timeout_streak,
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
    templates: tuple[SituationalAwarenessTrainingSegment, ...],
) -> tuple[SituationalAwarenessTrainingSegment, ...]:
    remaining = max(0.0, float(total_duration_s))
    built: list[SituationalAwarenessTrainingSegment] = []
    idx = 0
    while remaining > 1e-9:
        template = templates[idx % len(templates)]
        seg_duration = min(float(template.duration_s), remaining)
        if seg_duration <= 0.0:
            break
        built.append(
            SituationalAwarenessTrainingSegment(
                label=template.label,
                duration_s=seg_duration,
                active_channels=template.active_channels,
                active_query_kinds=template.active_query_kinds,
                scenario_families=template.scenario_families,
                focus_label=template.focus_label,
                profile=template.profile,
            )
        )
        remaining -= seg_duration
        idx += 1
    return tuple(built)


def _mode_profile(
    mode: AntDrillMode,
    *,
    track_min: int,
    track_max: int,
    query_interval_min_s: int,
    query_interval_max_s: int,
    response_window_s: int,
    update_time_scale: float = 1.0,
    pressure_scale: float = 1.0,
) -> SituationalAwarenessTrainingProfile:
    if mode is AntDrillMode.BUILD:
        track_hi = max(track_min, track_max - 1)
        query_min = int(round(query_interval_min_s * 1.18))
        query_max = int(round(query_interval_max_s * 1.18))
        response = int(round(response_window_s * 1.20))
        update_scale = update_time_scale * 1.10
        pressure = pressure_scale * 0.88
    elif mode is AntDrillMode.STRESS:
        track_hi = max(track_min, track_max)
        query_min = int(round(query_interval_min_s * 0.82))
        query_max = int(round(query_interval_max_s * 0.82))
        response = int(round(response_window_s * 0.86))
        update_scale = update_time_scale * 0.86
        pressure = pressure_scale * 1.18
    else:
        track_hi = max(track_min, track_max)
        query_min = query_interval_min_s
        query_max = query_interval_max_s
        response = response_window_s
        update_scale = update_time_scale
        pressure = pressure_scale
    return SituationalAwarenessTrainingProfile(
        min_track_count=max(3, int(track_min)),
        max_track_count=max(track_min, int(track_hi)),
        query_interval_min_s=max(6, int(query_min)),
        query_interval_max_s=max(max(6, int(query_min)), int(query_max)),
        response_window_s=max(5, int(response)),
        update_time_scale=max(0.45, float(update_scale)),
        pressure_scale=max(0.5, float(pressure)),
    )


def _build_sa_drill(
    *,
    title_base: str,
    instructions: tuple[str, ...],
    scored_segments: tuple[SituationalAwarenessTrainingSegment, ...],
    clock: Clock,
    seed: int,
    difficulty: float,
    mode: AntDrillMode | str,
    config: SaDrillConfig,
) -> SituationalAwarenessContinuousDrill:
    normalized_mode = _normalize_mode(mode)
    mode_profile = ANT_DRILL_MODE_PROFILES[normalized_mode]
    scored_duration_s = (
        float(mode_profile.scored_duration_s)
        if config.scored_duration_s is None
        else float(config.scored_duration_s)
    )
    segments = _repeat_segments(total_duration_s=scored_duration_s, templates=scored_segments)
    engine = build_situational_awareness_test(
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        title=title_base,
        config=SituationalAwarenessConfig(
            scored_duration_s=scored_duration_s,
            practice_scenarios=0,
            practice_scenario_duration_s=0.0,
            scored_scenario_duration_s=scored_duration_s,
        ),
        practice_segments=(),
        scored_segments=segments,
    )
    return SituationalAwarenessContinuousDrill(
        title=f"{title_base} ({mode_profile.label})",
        instructions=instructions,
        engine=engine,
        seed=seed,
        difficulty=difficulty,
        mode=normalized_mode,
        scored_duration_s=scored_duration_s,
    )


def build_sa_picture_anchor_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: SaDrillConfig | None = None,
) -> SituationalAwarenessContinuousDrill:
    cfg = config or SaDrillConfig()
    normalized_mode = _normalize_mode(mode)
    profile = ANT_DRILL_MODE_PROFILES[normalized_mode]
    return _build_sa_drill(
        title_base="Situational Awareness: Picture Anchor",
        instructions=(
            "Situational Awareness: Picture Anchor",
            f"Mode: {profile.label}",
            "Keep the full tactical display live, but stay on picture tracking and simple contact identification.",
            "Watch the moving grid, hold the route picture, and answer quickly without over-reading the feed.",
            "Press Enter to begin.",
        ),
        scored_segments=(
            SituationalAwarenessTrainingSegment(
                label="Picture Anchor",
                duration_s=float(cfg.scored_duration_s or ANT_DRILL_MODE_PROFILES[normalized_mode].scored_duration_s),
                active_channels=("pictorial", "coded", "numerical"),
                active_query_kinds=(
                    SituationalAwarenessQueryKind.FUTURE_POSITION.value,
                    SituationalAwarenessQueryKind.CONTACT_IDENTIFICATION.value,
                ),
                scenario_families=(
                    SituationalAwarenessScenarioFamily.ROUTE_HANDOFF,
                    SituationalAwarenessScenarioFamily.MERGE_CONFLICT,
                ),
                focus_label="Picture tracking + contact identification",
                profile=_mode_profile(
                    normalized_mode,
                    track_min=3,
                    track_max=5,
                    query_interval_min_s=12,
                    query_interval_max_s=16,
                    response_window_s=16,
                    update_time_scale=1.05,
                    pressure_scale=0.9,
                ),
            ),
        ),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=normalized_mode,
        config=cfg,
    )


def build_sa_contact_identification_prime_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: SaDrillConfig | None = None,
) -> SituationalAwarenessContinuousDrill:
    cfg = config or SaDrillConfig()
    normalized_mode = _normalize_mode(mode)
    profile = ANT_DRILL_MODE_PROFILES[normalized_mode]
    return _build_sa_drill(
        title_base="Situational Awareness: Contact Identification Prime",
        instructions=(
            "Situational Awareness: Contact Identification Prime",
            f"Mode: {profile.label}",
            "Match the live coded and pictorial state to the correct indexed track.",
            "Use the right-side panel deliberately, but keep the moving picture in view before you answer.",
            "Press Enter to begin.",
        ),
        scored_segments=(
            SituationalAwarenessTrainingSegment(
                label="Contact Identification",
                duration_s=float(cfg.scored_duration_s or ANT_DRILL_MODE_PROFILES[normalized_mode].scored_duration_s),
                active_channels=("pictorial", "coded"),
                active_query_kinds=(SituationalAwarenessQueryKind.CONTACT_IDENTIFICATION.value,),
                focus_label="Contact identification",
                profile=_mode_profile(
                    normalized_mode,
                    track_min=4,
                    track_max=5,
                    query_interval_min_s=11,
                    query_interval_max_s=15,
                    response_window_s=14,
                    update_time_scale=1.0,
                    pressure_scale=0.95,
                ),
            ),
        ),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=normalized_mode,
        config=cfg,
    )


def build_sa_status_recall_prime_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: SaDrillConfig | None = None,
) -> SituationalAwarenessContinuousDrill:
    cfg = config or SaDrillConfig()
    normalized_mode = _normalize_mode(mode)
    profile = ANT_DRILL_MODE_PROFILES[normalized_mode]
    return _build_sa_drill(
        title_base="Situational Awareness: Status Recall Prime",
        instructions=(
            "Situational Awareness: Status Recall Prime",
            f"Mode: {profile.label}",
            "Focus on channel, code, altitude, and fuel/status recall from the live coded picture and announcements.",
            "Let the track rows and recent feed do the work instead of scanning every moving marker.",
            "Press Enter to begin.",
        ),
        scored_segments=(
            SituationalAwarenessTrainingSegment(
                label="Status Recall",
                duration_s=float(cfg.scored_duration_s or ANT_DRILL_MODE_PROFILES[normalized_mode].scored_duration_s),
                active_channels=("coded", "numerical", "aural"),
                active_query_kinds=(SituationalAwarenessQueryKind.CODE_OR_STATUS_RECALL.value,),
                scenario_families=(
                    SituationalAwarenessScenarioFamily.CHANNEL_WAYPOINT_CHANGE,
                    SituationalAwarenessScenarioFamily.FUEL_PRIORITY,
                    SituationalAwarenessScenarioFamily.ROUTE_HANDOFF,
                ),
                focus_label="Status and code recall",
                profile=_mode_profile(
                    normalized_mode,
                    track_min=4,
                    track_max=5,
                    query_interval_min_s=10,
                    query_interval_max_s=14,
                    response_window_s=13,
                    update_time_scale=0.94,
                    pressure_scale=1.0,
                ),
            ),
        ),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=normalized_mode,
        config=cfg,
    )


def build_sa_future_projection_run_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: SaDrillConfig | None = None,
) -> SituationalAwarenessContinuousDrill:
    cfg = config or SaDrillConfig()
    normalized_mode = _normalize_mode(mode)
    profile = ANT_DRILL_MODE_PROFILES[normalized_mode]
    return _build_sa_drill(
        title_base="Situational Awareness: Future Projection Run",
        instructions=(
            "Situational Awareness: Future Projection Run",
            f"Mode: {profile.label}",
            "Project headings, waypoints, and handoff changes forward and answer with the predicted grid cell.",
            "Keep the live route picture stable enough that you can answer before the next update steals your attention.",
            "Press Enter to begin.",
        ),
        scored_segments=(
            SituationalAwarenessTrainingSegment(
                label="Future Projection",
                duration_s=float(cfg.scored_duration_s or ANT_DRILL_MODE_PROFILES[normalized_mode].scored_duration_s),
                active_channels=("pictorial", "numerical", "aural"),
                active_query_kinds=(SituationalAwarenessQueryKind.FUTURE_POSITION.value,),
                scenario_families=(
                    SituationalAwarenessScenarioFamily.ROUTE_HANDOFF,
                    SituationalAwarenessScenarioFamily.MERGE_CONFLICT,
                    SituationalAwarenessScenarioFamily.CHANNEL_WAYPOINT_CHANGE,
                ),
                focus_label="Future projection",
                profile=_mode_profile(
                    normalized_mode,
                    track_min=4,
                    track_max=5,
                    query_interval_min_s=11,
                    query_interval_max_s=15,
                    response_window_s=14,
                    update_time_scale=0.92,
                    pressure_scale=1.04,
                ),
            ),
        ),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=normalized_mode,
        config=cfg,
    )


def build_sa_action_selection_run_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: SaDrillConfig | None = None,
) -> SituationalAwarenessContinuousDrill:
    cfg = config or SaDrillConfig()
    normalized_mode = _normalize_mode(mode)
    profile = ANT_DRILL_MODE_PROFILES[normalized_mode]
    return _build_sa_drill(
        title_base="Situational Awareness: Action Selection Run",
        instructions=(
            "Situational Awareness: Action Selection Run",
            f"Mode: {profile.label}",
            "Stay on intervention and priority selection while merge and fuel conflicts keep moving underneath you.",
            "Answer with the best immediate action instead of delaying for perfect certainty.",
            "Press Enter to begin.",
        ),
        scored_segments=(
            SituationalAwarenessTrainingSegment(
                label="Action Selection",
                duration_s=float(cfg.scored_duration_s or ANT_DRILL_MODE_PROFILES[normalized_mode].scored_duration_s),
                active_channels=SA_CHANNEL_ORDER,
                active_query_kinds=(SituationalAwarenessQueryKind.ACTION_SELECTION.value,),
                scenario_families=(
                    SituationalAwarenessScenarioFamily.MERGE_CONFLICT,
                    SituationalAwarenessScenarioFamily.FUEL_PRIORITY,
                ),
                focus_label="Intervention and tactical priority",
                profile=_mode_profile(
                    normalized_mode,
                    track_min=4,
                    track_max=5,
                    query_interval_min_s=10,
                    query_interval_max_s=14,
                    response_window_s=12,
                    update_time_scale=0.90,
                    pressure_scale=1.15,
                ),
            ),
        ),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=normalized_mode,
        config=cfg,
    )


def build_sa_family_switch_run_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: SaDrillConfig | None = None,
) -> SituationalAwarenessContinuousDrill:
    cfg = config or SaDrillConfig()
    normalized_mode = _normalize_mode(mode)
    profile = ANT_DRILL_MODE_PROFILES[normalized_mode]
    mode_profile = _mode_profile(
        normalized_mode,
        track_min=4,
        track_max=5,
        query_interval_min_s=10,
        query_interval_max_s=14,
        response_window_s=13,
        update_time_scale=0.96,
        pressure_scale=1.0,
    )
    templates = tuple(
        SituationalAwarenessTrainingSegment(
            label=family.value.replace("_", " ").title(),
            duration_s=60.0,
            active_channels=SA_CHANNEL_ORDER,
            active_query_kinds=SA_QUERY_KIND_ORDER,
            scenario_families=(family,),
            focus_label="Scenario-family switching",
            profile=mode_profile,
        )
        for family in SA_FAMILY_CYCLE
    )
    return _build_sa_drill(
        title_base="Situational Awareness: Family Switch Run",
        instructions=(
            "Situational Awareness: Family Switch Run",
            f"Mode: {profile.label}",
            "Cycle through merge, fuel, route-handoff, and channel-shift scenarios in a fixed order.",
            "The picture format stays the same; the family switch is what should stop costing you time.",
            "Press Enter to begin.",
        ),
        scored_segments=templates,
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=normalized_mode,
        config=cfg,
    )


def build_sa_mixed_tempo_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: SaDrillConfig | None = None,
) -> SituationalAwarenessContinuousDrill:
    cfg = config or SaDrillConfig()
    normalized_mode = _normalize_mode(mode)
    profile = ANT_DRILL_MODE_PROFILES[normalized_mode]
    mode_profile = _mode_profile(
        normalized_mode,
        track_min=4,
        track_max=5,
        query_interval_min_s=10,
        query_interval_max_s=13,
        response_window_s=12,
        update_time_scale=0.92,
        pressure_scale=1.05,
    )
    templates = tuple(
        SituationalAwarenessTrainingSegment(
            label=_query_label(kind_name),
            duration_s=90.0,
            active_channels=SA_CHANNEL_ORDER,
            active_query_kinds=(kind_name,),
            scenario_families=SA_FAMILY_CYCLE,
            focus_label=f"{_query_label(kind_name)} tempo",
            profile=mode_profile,
        )
        for kind_name in SA_QUERY_KIND_ORDER
    )
    return _build_sa_drill(
        title_base="Situational Awareness: Mixed Tempo",
        instructions=(
            "Situational Awareness: Mixed Tempo",
            f"Mode: {profile.label}",
            "Run the fixed four-query rhythm while scenario families keep rotating underneath it.",
            "Reset cleanly when the query type changes; the screen stays live and continuous the whole time.",
            "Press Enter to begin.",
        ),
        scored_segments=templates,
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=normalized_mode,
        config=cfg,
    )


def build_sa_pressure_run_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: SaDrillConfig | None = None,
) -> SituationalAwarenessContinuousDrill:
    cfg = config or SaDrillConfig()
    normalized_mode = _normalize_mode(mode)
    profile = ANT_DRILL_MODE_PROFILES[normalized_mode]
    return _build_sa_drill(
        title_base="Situational Awareness: Pressure Run",
        instructions=(
            "Situational Awareness: Pressure Run",
            f"Mode: {profile.label}",
            "Run the full live tactical picture with every query type active, denser updates, and shorter response windows.",
            "Recover immediately after misses and keep building the next answer from the current picture.",
            "Press Enter to begin.",
        ),
        scored_segments=(
            SituationalAwarenessTrainingSegment(
                label="Pressure Run",
                duration_s=float(cfg.scored_duration_s or ANT_DRILL_MODE_PROFILES[normalized_mode].scored_duration_s),
                active_channels=SA_CHANNEL_ORDER,
                active_query_kinds=SA_QUERY_KIND_ORDER,
                scenario_families=SA_FAMILY_CYCLE,
                focus_label="Full mixed pressure",
                profile=_mode_profile(
                    normalized_mode,
                    track_min=5,
                    track_max=5,
                    query_interval_min_s=8,
                    query_interval_max_s=12,
                    response_window_s=10,
                    update_time_scale=0.82,
                    pressure_scale=1.28,
                ),
            ),
        ),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=normalized_mode,
        config=cfg,
    )
