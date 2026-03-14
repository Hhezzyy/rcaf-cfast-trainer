from __future__ import annotations

from dataclasses import dataclass, field, replace

from .ant_drills import (
    ANT_DRILL_MODE_PROFILES,
    AntAdaptiveDifficultyConfig,
    AntDrillAttemptSummary,
    AntDrillMode,
)
from .clock import Clock
from .cognitive_core import AttemptSummary, Phase, Problem, TestSnapshot, TimedTextInputTest
from .cognitive_updating import (
    CognitiveUpdatingGenerator,
    CognitiveUpdatingPayload,
    CognitiveUpdatingScorer,
    CognitiveUpdatingTrainingProfile,
    supported_cognitive_updating_scenario_families,
)


CU_SCENARIO_FAMILY_SEQUENCE = supported_cognitive_updating_scenario_families()


def _normalize_mode(mode: AntDrillMode | str) -> AntDrillMode:
    if isinstance(mode, AntDrillMode):
        return mode
    return AntDrillMode(str(mode).strip().lower())


@dataclass(frozen=True, slots=True)
class CuDrillConfig:
    practice_questions: int | None = None
    scored_duration_s: float | None = None
    adaptive: AntAdaptiveDifficultyConfig = field(
        default_factory=lambda: AntAdaptiveDifficultyConfig(enabled=False)
    )


class CognitiveUpdatingTimedDrill(TimedTextInputTest):
    def snapshot(self) -> TestSnapshot:
        snap = super().snapshot()
        if self.phase in (Phase.PRACTICE, Phase.SCORED):
            hint = "Use live tabs and controls, enter the 4-digit code, then press Enter"
        else:
            hint = "Press Enter to continue"
        return replace(snap, input_hint=hint)

    def scored_summary(self) -> AntDrillAttemptSummary:
        base: AttemptSummary = super().scored_summary()
        duration_s = float(base.duration_s)
        correct_per_min = (float(base.correct) / duration_s) * 60.0 if duration_s > 0.0 else 0.0
        return AntDrillAttemptSummary(
            attempted=base.attempted,
            correct=base.correct,
            accuracy=base.accuracy,
            duration_s=duration_s,
            throughput_per_min=base.throughput_per_min,
            mean_response_time_s=base.mean_response_time_s,
            total_score=base.total_score,
            max_score=base.max_score,
            score_ratio=base.score_ratio,
            correct_per_min=correct_per_min,
            timeouts=0,
            fixation_rate=0.0,
            max_timeout_streak=0,
            mode="",
            difficulty_level=1,
            difficulty_level_start=1,
            difficulty_level_end=1,
            difficulty_change_count=0,
            adaptive_enabled=False,
            adaptive_window_size=0,
        )


def _mode_training_profile(mode: AntDrillMode | str) -> CognitiveUpdatingTrainingProfile:
    normalized = _normalize_mode(mode)
    if normalized is AntDrillMode.BUILD:
        return CognitiveUpdatingTrainingProfile(
            camera_due_scale=1.18,
            sensor_due_scale=1.16,
            objective_deadline_scale=1.22,
            comms_time_limit_scale=1.16,
            message_reveal_scale=0.82,
            pressure_drift_scale=0.84,
            speed_drift_scale=0.84,
            tank_drain_scale=0.90,
        )
    if normalized is AntDrillMode.STRESS:
        return CognitiveUpdatingTrainingProfile(
            camera_due_scale=0.86,
            sensor_due_scale=0.84,
            objective_deadline_scale=0.82,
            comms_time_limit_scale=0.84,
            message_reveal_scale=1.12,
            pressure_drift_scale=1.22,
            speed_drift_scale=1.22,
            tank_drain_scale=1.16,
        )
    return CognitiveUpdatingTrainingProfile()


def _merge_training_profiles(
    base: CognitiveUpdatingTrainingProfile, override: CognitiveUpdatingTrainingProfile
) -> CognitiveUpdatingTrainingProfile:
    return CognitiveUpdatingTrainingProfile(
        active_domains=override.active_domains,
        scenario_family=override.scenario_family if override.scenario_family is not None else base.scenario_family,
        focus_label=override.focus_label,
        camera_due_scale=base.camera_due_scale * override.camera_due_scale,
        sensor_due_scale=base.sensor_due_scale * override.sensor_due_scale,
        objective_deadline_scale=base.objective_deadline_scale * override.objective_deadline_scale,
        comms_time_limit_scale=base.comms_time_limit_scale * override.comms_time_limit_scale,
        message_reveal_scale=base.message_reveal_scale * override.message_reveal_scale,
        pressure_drift_scale=base.pressure_drift_scale * override.pressure_drift_scale,
        speed_drift_scale=base.speed_drift_scale * override.speed_drift_scale,
        tank_drain_scale=base.tank_drain_scale * override.tank_drain_scale,
        warning_penalty_scale=base.warning_penalty_scale * override.warning_penalty_scale,
        starting_upper_tab_index=(
            override.starting_upper_tab_index
            if override.starting_upper_tab_index is not None
            else base.starting_upper_tab_index
        ),
        starting_lower_tab_index=(
            override.starting_lower_tab_index
            if override.starting_lower_tab_index is not None
            else base.starting_lower_tab_index
        ),
    )


class _BaseCognitiveUpdatingSelectionGenerator:
    def __init__(self, *, seed: int) -> None:
        self._base = CognitiveUpdatingGenerator(seed=seed)
        self._family_index = 0

    def _next_family(self) -> str:
        family = CU_SCENARIO_FAMILY_SEQUENCE[self._family_index % len(CU_SCENARIO_FAMILY_SEQUENCE)]
        self._family_index += 1
        return family

    def _problem_for_profile(
        self, *, difficulty: float, profile: CognitiveUpdatingTrainingProfile
    ) -> Problem:
        family = profile.scenario_family or self._next_family()
        return self._base.next_problem_for_selection(
            difficulty=difficulty,
            training_profile=profile,
            scenario_family=family,
        )


class _SingleProfileGenerator(_BaseCognitiveUpdatingSelectionGenerator):
    _profile = CognitiveUpdatingTrainingProfile()

    def __init__(self, *, seed: int, mode: AntDrillMode | str) -> None:
        super().__init__(seed=seed)
        self._profile_for_mode = _merge_training_profiles(_mode_training_profile(mode), self._profile)

    def next_problem(self, *, difficulty: float) -> Problem:
        return self._problem_for_profile(difficulty=difficulty, profile=self._profile_for_mode)


class CuControlsAnchorGenerator(_SingleProfileGenerator):
    _profile = CognitiveUpdatingTrainingProfile(
        active_domains=("controls", "state_code"),
        focus_label="Controls",
        pressure_drift_scale=0.82,
        starting_upper_tab_index=2,
        starting_lower_tab_index=1,
    )


class CuNavigationAnchorGenerator(_SingleProfileGenerator):
    _profile = CognitiveUpdatingTrainingProfile(
        active_domains=("navigation", "state_code"),
        focus_label="Navigation",
        speed_drift_scale=0.82,
        starting_upper_tab_index=3,
        starting_lower_tab_index=1,
    )


class CuEngineBalanceRunGenerator(_SingleProfileGenerator):
    _profile = CognitiveUpdatingTrainingProfile(
        active_domains=("engine", "state_code"),
        focus_label="Engine",
        tank_drain_scale=0.94,
        starting_upper_tab_index=5,
        starting_lower_tab_index=1,
    )


class CuSensorsTimingPrimeGenerator(_SingleProfileGenerator):
    _profile = CognitiveUpdatingTrainingProfile(
        active_domains=("sensors", "state_code"),
        focus_label="Sensors",
        camera_due_scale=0.92,
        sensor_due_scale=0.90,
        starting_upper_tab_index=4,
        starting_lower_tab_index=1,
    )


class CuObjectivePrimeGenerator(_SingleProfileGenerator):
    _profile = CognitiveUpdatingTrainingProfile(
        active_domains=("objectives", "state_code"),
        focus_label="Objectives",
        objective_deadline_scale=1.10,
        starting_upper_tab_index=1,
        starting_lower_tab_index=1,
    )


class CuStateCodeRunGenerator(_SingleProfileGenerator):
    _profile = CognitiveUpdatingTrainingProfile(
        active_domains=("controls", "navigation", "sensors", "state_code"),
        focus_label="State Code",
        starting_upper_tab_index=2,
        starting_lower_tab_index=4,
    )


class CuMixedTempoGenerator(_BaseCognitiveUpdatingSelectionGenerator):
    _SEQUENCE = (
        CognitiveUpdatingTrainingProfile(
            active_domains=("controls", "state_code"),
            focus_label="Controls",
            pressure_drift_scale=0.82,
            starting_upper_tab_index=2,
            starting_lower_tab_index=1,
        ),
        CognitiveUpdatingTrainingProfile(
            active_domains=("navigation", "state_code"),
            focus_label="Navigation",
            speed_drift_scale=0.82,
            starting_upper_tab_index=3,
            starting_lower_tab_index=1,
        ),
        CognitiveUpdatingTrainingProfile(
            active_domains=("engine", "state_code"),
            focus_label="Engine",
            tank_drain_scale=0.94,
            starting_upper_tab_index=5,
            starting_lower_tab_index=1,
        ),
        CognitiveUpdatingTrainingProfile(
            active_domains=("sensors", "state_code"),
            focus_label="Sensors",
            camera_due_scale=0.92,
            sensor_due_scale=0.90,
            starting_upper_tab_index=4,
            starting_lower_tab_index=1,
        ),
        CognitiveUpdatingTrainingProfile(
            active_domains=("objectives", "state_code"),
            focus_label="Objectives",
            objective_deadline_scale=1.10,
            starting_upper_tab_index=1,
            starting_lower_tab_index=1,
        ),
        CognitiveUpdatingTrainingProfile(
            active_domains=("controls", "navigation", "sensors", "state_code"),
            focus_label="State Code",
            starting_upper_tab_index=2,
            starting_lower_tab_index=4,
        ),
        CognitiveUpdatingTrainingProfile(
            active_domains=("controls", "navigation", "engine", "sensors", "objectives", "state_code"),
            focus_label="Full Mixed",
            starting_upper_tab_index=2,
            starting_lower_tab_index=1,
        ),
    )

    def __init__(self, *, seed: int, mode: AntDrillMode | str) -> None:
        super().__init__(seed=seed)
        self._base_profile = _mode_training_profile(mode)
        self._sequence_index = 0

    def next_problem(self, *, difficulty: float) -> Problem:
        profile = self._SEQUENCE[self._sequence_index % len(self._SEQUENCE)]
        self._sequence_index += 1
        merged = _merge_training_profiles(self._base_profile, profile)
        return self._problem_for_profile(difficulty=difficulty, profile=merged)


class CuPressureRunGenerator(_SingleProfileGenerator):
    _profile = CognitiveUpdatingTrainingProfile(
        active_domains=("controls", "navigation", "engine", "sensors", "objectives", "state_code"),
        focus_label="Full Mixed",
        camera_due_scale=0.84,
        sensor_due_scale=0.82,
        objective_deadline_scale=0.82,
        comms_time_limit_scale=0.84,
        message_reveal_scale=1.10,
        pressure_drift_scale=1.18,
        speed_drift_scale=1.18,
        tank_drain_scale=1.16,
        starting_upper_tab_index=4,
        starting_lower_tab_index=5,
    )


def _build_cu_drill(
    *,
    title_base: str,
    instructions: tuple[str, ...],
    generator: object,
    clock: Clock,
    seed: int,
    difficulty: float,
    mode: AntDrillMode | str,
    config: CuDrillConfig,
) -> CognitiveUpdatingTimedDrill:
    normalized_mode = _normalize_mode(mode)
    profile = ANT_DRILL_MODE_PROFILES[normalized_mode]
    practice_questions = (
        profile.practice_questions if config.practice_questions is None else int(config.practice_questions)
    )
    scored_duration_s = (
        profile.scored_duration_s if config.scored_duration_s is None else float(config.scored_duration_s)
    )
    return CognitiveUpdatingTimedDrill(
        title=f"{title_base} ({profile.label})",
        instructions=list(instructions),
        generator=generator,
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        practice_questions=practice_questions,
        scored_duration_s=scored_duration_s,
        scorer=CognitiveUpdatingScorer(),
    )


def _default_instructions(
    title: str,
    mode: AntDrillMode | str,
    line_one: str,
    line_two: str,
) -> tuple[str, ...]:
    profile = ANT_DRILL_MODE_PROFILES[_normalize_mode(mode)]
    return (
        title,
        f"Mode: {profile.label}",
        line_one,
        line_two,
        "Keep the live dual-MFD flow: switch tabs, adjust controls, enter the 4-digit code, then press Enter.",
        "Press Enter to begin practice.",
    )


def build_cu_controls_anchor_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: CuDrillConfig | None = None,
) -> CognitiveUpdatingTimedDrill:
    cfg = config or CuDrillConfig()
    return _build_cu_drill(
        title_base="Cognitive Updating: Controls Anchor",
        instructions=_default_instructions(
            "Cognitive Updating: Controls Anchor",
            mode,
            "Focus on pump and pressure correction while the other domains stay neutralized.",
            "Use the live comms entry path to submit the final state code without needing the inactive systems.",
        ),
        generator=CuControlsAnchorGenerator(seed=seed, mode=mode),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
    )


def build_cu_navigation_anchor_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: CuDrillConfig | None = None,
) -> CognitiveUpdatingTimedDrill:
    cfg = config or CuDrillConfig()
    return _build_cu_drill(
        title_base="Cognitive Updating: Navigation Anchor",
        instructions=_default_instructions(
            "Cognitive Updating: Navigation Anchor",
            mode,
            "Focus on bringing airspeed back to the required knots while the other domains stay neutral.",
            "Practice reading the live state and entering the final code without unrelated panel penalties.",
        ),
        generator=CuNavigationAnchorGenerator(seed=seed, mode=mode),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
    )


def build_cu_engine_balance_run_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: CuDrillConfig | None = None,
) -> CognitiveUpdatingTimedDrill:
    cfg = config or CuDrillConfig()
    return _build_cu_drill(
        title_base="Cognitive Updating: Engine Balance Run",
        instructions=_default_instructions(
            "Cognitive Updating: Engine Balance Run",
            mode,
            "Focus on active-tank switching and keeping the tank spread inside tolerance.",
            "The rest of the task stays visible but neutral while you build clean tank-balance habits.",
        ),
        generator=CuEngineBalanceRunGenerator(seed=seed, mode=mode),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
    )


def build_cu_sensors_timing_prime_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: CuDrillConfig | None = None,
) -> CognitiveUpdatingTimedDrill:
    cfg = config or CuDrillConfig()
    return _build_cu_drill(
        title_base="Cognitive Updating: Sensors Timing Prime",
        instructions=_default_instructions(
            "Cognitive Updating: Sensors Timing Prime",
            mode,
            "Focus on Alpha, Bravo, air, and ground timing windows while unrelated domains stay neutral.",
            "Keep the sensor page flow tidy and preserve time for the final code entry.",
        ),
        generator=CuSensorsTimingPrimeGenerator(seed=seed, mode=mode),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
    )


def build_cu_objective_prime_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: CuDrillConfig | None = None,
) -> CognitiveUpdatingTimedDrill:
    cfg = config or CuDrillConfig()
    return _build_cu_drill(
        title_base="Cognitive Updating: Objective Prime",
        instructions=_default_instructions(
            "Cognitive Updating: Objective Prime",
            mode,
            "Focus on parcel entry, field switching, and dispenser timing while the rest of the task stays neutral.",
            "Use the training focus label and inactive-panel dimming to keep your scan disciplined.",
        ),
        generator=CuObjectivePrimeGenerator(seed=seed, mode=mode),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
    )


def build_cu_state_code_run_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.TEMPO,
    config: CuDrillConfig | None = None,
) -> CognitiveUpdatingTimedDrill:
    cfg = config or CuDrillConfig()
    return _build_cu_drill(
        title_base="Cognitive Updating: State Code Run",
        instructions=_default_instructions(
            "Cognitive Updating: State Code Run",
            mode,
            "Controls, navigation, and sensors stay active so you can practice reading the live state code cleanly.",
            "Engine and objective pages stay visible but neutralized throughout this block.",
        ),
        generator=CuStateCodeRunGenerator(seed=seed, mode=mode),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
    )


def build_cu_mixed_tempo_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.TEMPO,
    config: CuDrillConfig | None = None,
) -> CognitiveUpdatingTimedDrill:
    cfg = config or CuDrillConfig()
    return _build_cu_drill(
        title_base="Cognitive Updating: Mixed Tempo",
        instructions=_default_instructions(
            "Cognitive Updating: Mixed Tempo",
            mode,
            "Move through the fixed Controls, Navigation, Engine, Sensors, Objectives, State Code, and Full Mixed rhythm.",
            "Let the focus label and inactive-panel dimming tell you what matters on each item.",
        ),
        generator=CuMixedTempoGenerator(seed=seed, mode=mode),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
    )


def build_cu_pressure_run_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.STRESS,
    config: CuDrillConfig | None = None,
) -> CognitiveUpdatingTimedDrill:
    cfg = config or CuDrillConfig()
    return _build_cu_drill(
        title_base="Cognitive Updating: Pressure Run",
        instructions=_default_instructions(
            "Cognitive Updating: Pressure Run",
            mode,
            "Run the full live task with all domains active, tighter timing, and the hardest reveal and drift pressure in this family.",
            "Recover quickly from warnings instead of camping on one panel too long.",
        ),
        generator=CuPressureRunGenerator(seed=seed, mode=mode),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=cfg,
    )
