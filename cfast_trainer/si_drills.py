from __future__ import annotations

from dataclasses import dataclass

from .ant_drills import ANT_DRILL_MODE_PROFILES, AntDrillAttemptSummary, AntDrillMode
from .clock import Clock
from .cognitive_core import Phase, TestSnapshot, clamp01
from .spatial_integration import (
    SpatialIntegrationConfig,
    SpatialIntegrationPart,
    SpatialIntegrationPayload,
    SpatialIntegrationQuestionKind,
    build_spatial_integration_test,
)

_DEFAULT_SI_CONFIG = SpatialIntegrationConfig()
_STATIC_KINDS = (
    SpatialIntegrationQuestionKind.LANDMARK_GRID,
    SpatialIntegrationQuestionKind.SCENE_RECONSTRUCTION,
)
_AIRCRAFT_KINDS = (
    SpatialIntegrationQuestionKind.AIRCRAFT_ROUTE_SELECTION,
    SpatialIntegrationQuestionKind.AIRCRAFT_CONTINUATION_SELECTION,
    SpatialIntegrationQuestionKind.AIRCRAFT_LOCATION_GRID,
)
_ALL_KINDS = _STATIC_KINDS + _AIRCRAFT_KINDS


def _normalize_mode(mode: AntDrillMode | str) -> AntDrillMode:
    if isinstance(mode, AntDrillMode):
        return mode
    return AntDrillMode(str(mode).strip().lower())


def _difficulty_to_level(difficulty: float) -> int:
    return max(1, min(10, int(round(clamp01(difficulty) * 9.0)) + 1))


@dataclass(frozen=True, slots=True)
class SiDrillConfig:
    practice_scenes_per_part: int | None = None
    scored_duration_s: float | None = None


class SpatialIntegrationDrill:
    def __init__(
        self,
        *,
        title: str,
        instructions: tuple[str, ...],
        engine: object,
        seed: int,
        difficulty: float,
        mode: AntDrillMode,
        scored_duration_s: float,
        practice_scenes_per_part: int,
        parts: tuple[SpatialIntegrationPart, ...],
    ) -> None:
        self._title = str(title)
        self._instructions = tuple(str(line) for line in instructions)
        self._engine = engine
        self._seed = int(seed)
        self._difficulty = float(difficulty)
        self._mode = mode
        self._scored_duration_s = float(scored_duration_s)
        self._practice_scenes_per_part = max(0, int(practice_scenes_per_part))
        self._parts = tuple(parts)

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
        return self._practice_scenes_per_part * len(self._parts)

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
        accepted = bool(self._engine.submit_answer(raw))
        self._auto_advance_between_parts()
        return accepted

    def update(self) -> None:
        self._engine.update()
        self._auto_advance_between_parts()

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

    def _auto_advance_between_parts(self) -> None:
        while self._engine.phase is Phase.PRACTICE_DONE:
            pending = getattr(self._engine, "_pending_done_action", None)
            if pending == "start_next_practice":
                self._engine.start_scored()
                continue
            if pending == "start_scored" and self._practice_scenes_per_part <= 0:
                self._engine.start_scored()
                continue
            return

    def scored_summary(self) -> AntDrillAttemptSummary:
        base = self._engine.scored_summary()
        scored_events = [event for event in self._engine.events() if event.phase is Phase.SCORED]
        timeouts = 0
        max_timeout_streak = 0
        current_timeout_streak = 0
        for event in scored_events:
            raw = str(event.raw).strip().upper()
            is_timeout = raw in {"", "TIMEOUT"}
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


def _build_si_drill(
    *,
    title_base: str,
    instructions: tuple[str, ...],
    clock: Clock,
    seed: int,
    difficulty: float,
    mode: AntDrillMode | str,
    config: SiDrillConfig | None,
    parts: tuple[SpatialIntegrationPart, ...],
    allowed_question_kinds: tuple[SpatialIntegrationQuestionKind, ...],
) -> SpatialIntegrationDrill:
    normalized_mode = _normalize_mode(mode)
    mode_profile = ANT_DRILL_MODE_PROFILES[normalized_mode]
    cfg = config or SiDrillConfig()
    practice_scenes_per_part = (
        int(cfg.practice_scenes_per_part)
        if cfg.practice_scenes_per_part is not None
        else (1 if normalized_mode in (AntDrillMode.FRESH, AntDrillMode.BUILD) else 0)
    )
    scored_duration_s = (
        float(cfg.scored_duration_s)
        if cfg.scored_duration_s is not None
        else float(mode_profile.scored_duration_s)
    )
    per_part_duration = scored_duration_s / max(1, len(parts))
    scaled_study = float(mode_profile.cap_scale)
    engine = build_spatial_integration_test(
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        config=SpatialIntegrationConfig(
            practice_scenes_per_part=practice_scenes_per_part,
            static_scored_duration_s=per_part_duration,
            aircraft_scored_duration_s=per_part_duration,
            static_study_s=float(_DEFAULT_SI_CONFIG.static_study_s * scaled_study),
            aircraft_study_s=float(_DEFAULT_SI_CONFIG.aircraft_study_s * scaled_study),
            question_time_limit_s=float(_DEFAULT_SI_CONFIG.question_time_limit_s * scaled_study),
            start_part=parts[0].value,
            parts=tuple(part.value for part in parts),
            allowed_question_kinds=tuple(kind.value for kind in allowed_question_kinds),
        ),
    )
    return SpatialIntegrationDrill(
        title=f"{title_base} ({mode_profile.label})",
        instructions=instructions,
        engine=engine,
        seed=seed,
        difficulty=difficulty,
        mode=normalized_mode,
        scored_duration_s=scored_duration_s,
        practice_scenes_per_part=practice_scenes_per_part,
        parts=parts,
    )


def build_si_landmark_anchor_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: SiDrillConfig | None = None,
) -> SpatialIntegrationDrill:
    normalized_mode = _normalize_mode(mode)
    mode_profile = ANT_DRILL_MODE_PROFILES[normalized_mode]
    return _build_si_drill(
        title_base="Spatial Integration: Landmark Anchor",
        instructions=(
            "Spatial Integration: Landmark Anchor",
            f"Mode: {mode_profile.label}",
            "Static part only. Study the same landscape viewpoints, then answer only landmark grid-cell questions.",
            "Lock the object-to-terrain relationship first and type or click the matching cell without reconstructing the full map.",
            "Press Enter to begin.",
        ),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=normalized_mode,
        config=config,
        parts=(SpatialIntegrationPart.STATIC,),
        allowed_question_kinds=(SpatialIntegrationQuestionKind.LANDMARK_GRID,),
    )


def build_si_reconstruction_run_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: SiDrillConfig | None = None,
) -> SpatialIntegrationDrill:
    normalized_mode = _normalize_mode(mode)
    mode_profile = ANT_DRILL_MODE_PROFILES[normalized_mode]
    return _build_si_drill(
        title_base="Spatial Integration: Reconstruction Run",
        instructions=(
            "Spatial Integration: Reconstruction Run",
            f"Mode: {mode_profile.label}",
            "Static part only. Study the three reference views, then answer only full-scene reconstruction questions.",
            "Treat the landscape like one frozen scene seen from different viewpoints and pick the matching top-down map quickly.",
            "Press Enter to begin.",
        ),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=normalized_mode,
        config=config,
        parts=(SpatialIntegrationPart.STATIC,),
        allowed_question_kinds=(SpatialIntegrationQuestionKind.SCENE_RECONSTRUCTION,),
    )


def build_si_static_mixed_run_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.TEMPO,
    config: SiDrillConfig | None = None,
) -> SpatialIntegrationDrill:
    normalized_mode = _normalize_mode(mode)
    mode_profile = ANT_DRILL_MODE_PROFILES[normalized_mode]
    return _build_si_drill(
        title_base="Spatial Integration: Static Mixed Run",
        instructions=(
            "Spatial Integration: Static Mixed Run",
            f"Mode: {mode_profile.label}",
            "Static part only. Landmark grid questions and full-scene reconstruction stay mixed exactly as they do in the live SI landscape section.",
            "Keep the three-view study loop intact and recover fast when the question type switches.",
            "Press Enter to begin.",
        ),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=normalized_mode,
        config=config,
        parts=(SpatialIntegrationPart.STATIC,),
        allowed_question_kinds=_STATIC_KINDS,
    )


def build_si_route_anchor_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: SiDrillConfig | None = None,
) -> SpatialIntegrationDrill:
    normalized_mode = _normalize_mode(mode)
    mode_profile = ANT_DRILL_MODE_PROFILES[normalized_mode]
    return _build_si_drill(
        title_base="Spatial Integration: Route Anchor",
        instructions=(
            "Spatial Integration: Route Anchor",
            f"Mode: {mode_profile.label}",
            "Aircraft part only. Study the route from multiple viewpoints, then answer only route-map match questions.",
            "Focus on the aircraft path shape and terrain relationship before worrying about the next-step continuation.",
            "Press Enter to begin.",
        ),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=normalized_mode,
        config=config,
        parts=(SpatialIntegrationPart.AIRCRAFT,),
        allowed_question_kinds=(SpatialIntegrationQuestionKind.AIRCRAFT_ROUTE_SELECTION,),
    )


def build_si_continuation_prime_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: SiDrillConfig | None = None,
) -> SpatialIntegrationDrill:
    normalized_mode = _normalize_mode(mode)
    mode_profile = ANT_DRILL_MODE_PROFILES[normalized_mode]
    return _build_si_drill(
        title_base="Spatial Integration: Continuation Prime",
        instructions=(
            "Spatial Integration: Continuation Prime",
            f"Mode: {mode_profile.label}",
            "Aircraft part only. Study the route views, then answer only continuation questions about the next aircraft position.",
            "Bias this block toward forward projection instead of whole-route reconstruction.",
            "Press Enter to begin.",
        ),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=normalized_mode,
        config=config,
        parts=(SpatialIntegrationPart.AIRCRAFT,),
        allowed_question_kinds=(SpatialIntegrationQuestionKind.AIRCRAFT_CONTINUATION_SELECTION,),
    )


def build_si_aircraft_grid_run_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.TEMPO,
    config: SiDrillConfig | None = None,
) -> SpatialIntegrationDrill:
    normalized_mode = _normalize_mode(mode)
    mode_profile = ANT_DRILL_MODE_PROFILES[normalized_mode]
    return _build_si_drill(
        title_base="Spatial Integration: Aircraft Grid Run",
        instructions=(
            "Spatial Integration: Aircraft Grid Run",
            f"Mode: {mode_profile.label}",
            "Aircraft part only. Study the same route views, then answer only aircraft-location grid questions.",
            "Use the route and terrain cues to pin the aircraft to one cell without slowing down.",
            "Press Enter to begin.",
        ),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=normalized_mode,
        config=config,
        parts=(SpatialIntegrationPart.AIRCRAFT,),
        allowed_question_kinds=(SpatialIntegrationQuestionKind.AIRCRAFT_LOCATION_GRID,),
    )


def build_si_mixed_tempo_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.TEMPO,
    config: SiDrillConfig | None = None,
) -> SpatialIntegrationDrill:
    normalized_mode = _normalize_mode(mode)
    mode_profile = ANT_DRILL_MODE_PROFILES[normalized_mode]
    return _build_si_drill(
        title_base="Spatial Integration: Mixed Tempo",
        instructions=(
            "Spatial Integration: Mixed Tempo",
            f"Mode: {mode_profile.label}",
            "Run the static section first, then the aircraft section, with the full live question mix in both parts.",
            "Treat this as the closest single-drill approximation to the full SI cadence without adding extra scene motion.",
            "Press Enter to begin.",
        ),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=normalized_mode,
        config=config,
        parts=(SpatialIntegrationPart.STATIC, SpatialIntegrationPart.AIRCRAFT),
        allowed_question_kinds=_ALL_KINDS,
    )


def build_si_pressure_run_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.STRESS,
    config: SiDrillConfig | None = None,
) -> SpatialIntegrationDrill:
    normalized_mode = _normalize_mode(mode)
    mode_profile = ANT_DRILL_MODE_PROFILES[normalized_mode]
    return _build_si_drill(
        title_base="Spatial Integration: Pressure Run",
        instructions=(
            "Spatial Integration: Pressure Run",
            f"Mode: {mode_profile.label}",
            "Run the full static section first and the full aircraft section second under the tightest study and answer caps in this family.",
            "Stay on the same frozen-scene presentation and accept misses quickly when the viewpoint changes cost you time.",
            "Press Enter to begin.",
        ),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=normalized_mode,
        config=config,
        parts=(SpatialIntegrationPart.STATIC, SpatialIntegrationPart.AIRCRAFT),
        allowed_question_kinds=_ALL_KINDS,
    )
