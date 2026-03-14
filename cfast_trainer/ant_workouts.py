from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import cast

from .abd_drills import (
    AbdDrillConfig,
    AbdFamilyRunConfig,
    build_abd_angle_calibration_drill,
    build_abd_bearing_calibration_drill,
    build_abd_cardinal_anchors_drill,
    build_abd_intermediate_anchors_drill,
    build_abd_mixed_tempo_drill,
    build_abd_test_style_family_run_drill,
)
from .dr_drills import (
    DigitRecognitionDrillConfig,
    build_dr_count_target_drill,
    build_dr_different_digit_drill,
    build_dr_grouped_family_run_drill,
    build_dr_mixed_pressure_drill,
    build_dr_position_probe_drill,
    build_dr_recall_run_drill,
    build_dr_visible_copy_drill,
    build_dr_visible_family_primer_drill,
)
from .ic_drills import (
    IcDrillConfig,
    build_ic_attitude_frame_drill,
    build_ic_description_prime_drill,
    build_ic_description_run_drill,
    build_ic_heading_anchor_drill,
    build_ic_mixed_part_run_drill,
    build_ic_part1_orientation_run_drill,
    build_ic_pressure_run_drill,
    build_ic_reverse_panel_prime_drill,
    build_ic_reverse_panel_run_drill,
)
from .ac_drills import (
    AcDrillConfig,
    build_ac_callsign_filter_run_drill,
    build_ac_digit_sequence_prime_drill,
    build_ac_gate_anchor_drill,
    build_ac_gate_directive_run_drill,
    build_ac_mixed_tempo_drill,
    build_ac_pressure_run_drill,
    build_ac_state_command_prime_drill,
    build_ac_trigger_cue_anchor_drill,
)
from .cu_drills import (
    CuDrillConfig,
    build_cu_controls_anchor_drill,
    build_cu_engine_balance_run_drill,
    build_cu_mixed_tempo_drill,
    build_cu_navigation_anchor_drill,
    build_cu_objective_prime_drill,
    build_cu_pressure_run_drill,
    build_cu_sensors_timing_prime_drill,
    build_cu_state_code_run_drill,
)
from .sa_drills import (
    SaDrillConfig,
    build_sa_action_selection_run_drill,
    build_sa_contact_identification_prime_drill,
    build_sa_family_switch_run_drill,
    build_sa_future_projection_run_drill,
    build_sa_mixed_tempo_drill,
    build_sa_picture_anchor_drill,
    build_sa_pressure_run_drill,
    build_sa_status_recall_prime_drill,
)
from .rt_drills import (
    RtDrillConfig,
    build_rt_air_speed_run_drill,
    build_rt_building_handoff_prime_drill,
    build_rt_capture_timing_prime_drill,
    build_rt_ground_tempo_run_drill,
    build_rt_lock_anchor_drill,
    build_rt_mixed_tempo_drill,
    build_rt_pressure_run_drill,
    build_rt_terrain_recovery_run_drill,
)
from .tr_drills import (
    TrDrillConfig,
    build_tr_light_anchor_drill,
    build_tr_mixed_tempo_drill,
    build_tr_panel_switch_run_drill,
    build_tr_pressure_run_drill,
    build_tr_scan_anchor_drill,
    build_tr_scene_anchor_drill,
    build_tr_scene_modifier_run_drill,
    build_tr_system_anchor_drill,
)
from .sl_drills import (
    SlDrillConfig,
    build_sl_fault_diagnosis_prime_drill,
    build_sl_family_run_drill,
    build_sl_flow_trace_anchor_drill,
    build_sl_graph_rule_anchor_drill,
    build_sl_index_switch_run_drill,
    build_sl_mixed_tempo_drill,
    build_sl_pressure_run_drill,
    build_sl_quantitative_anchor_drill,
)
from .tbl_drills import (
    TblDrillConfig,
    build_tbl_card_family_run_drill,
    build_tbl_mixed_tempo_drill,
    build_tbl_part1_anchor_drill,
    build_tbl_part1_scan_run_drill,
    build_tbl_part2_correction_run_drill,
    build_tbl_part2_prime_drill,
    build_tbl_part_switch_run_drill,
    build_tbl_pressure_run_drill,
)
from .sma_drills import (
    SmaDrillConfig,
    build_sma_disturbance_tempo_drill,
    build_sma_joystick_hold_run_drill,
    build_sma_joystick_horizontal_anchor_drill,
    build_sma_joystick_vertical_anchor_drill,
    build_sma_mode_switch_run_drill,
    build_sma_pressure_run_drill,
    build_sma_split_coordination_run_drill,
    build_sma_split_horizontal_prime_drill,
)
from .cln_drills import (
    ClnDrillConfig,
    build_cln_colour_lane_drill,
    build_cln_full_pressure_drill,
    build_cln_full_steady_drill,
    build_cln_math_prime_drill,
    build_cln_memory_colour_drill,
    build_cln_memory_math_drill,
    build_cln_sequence_copy_drill,
    build_cln_sequence_match_drill,
)
from .airborne_numerical import (
    AirborneNumericalGenerator,
    AirborneScorer,
    AirborneScenario,
    build_ant_airborne_difficulty_profile,
)
from .angles_bearings_degrees import AnglesBearingsQuestionKind
from .mr_drills import (
    MrDrillConfig,
    build_mr_domain_run_drill,
    build_mr_mixed_pressure_set_drill,
    build_mr_multi_step_solve_drill,
    build_mr_one_step_solve_drill,
    build_mr_relevant_info_scan_drill,
    build_mr_unit_relation_prime_drill,
)
from .no_drills import (
    NoDrillConfig,
    build_no_clean_compute_drill,
    build_no_fact_prime_drill,
    build_no_mixed_tempo_drill,
    build_no_operator_ladders_drill,
    build_no_pressure_run_drill,
)
from .vs_drills import (
    VsDrillConfig,
    build_vs_clean_scan_drill,
    build_vs_family_run_drill,
    build_vs_mixed_tempo_drill,
    build_vs_pressure_run_drill,
    build_vs_target_preview_drill,
)
from .ant_drills import (
    AntAdaptiveDifficultyConfig,
    AntDistanceScanConfig,
    AntDistanceScanGenerator,
    AntDrillAttemptSummary,
    AntDrillMode,
    AntEnduranceSolveConfig,
    AntEnduranceSolveGenerator,
    AntFuelBurnSolveConfig,
    AntFuelBurnSolveGenerator,
    AntInfoGrabberConfig,
    AntPayloadReferenceGenerator,
    AntRouteTimeSolveConfig,
    AntRouteTimeSolveGenerator,
    AntSnapFactsSprintConfig,
    AntTimeFlipConfig,
    TimedCapDrill,
    build_ant_distance_scan_drill,
    build_ant_endurance_solve_drill,
    build_ant_fuel_burn_solve_drill,
    build_ant_info_grabber_drill,
    build_ant_route_time_solve_drill,
    build_ant_snap_facts_sprint_drill,
    build_ant_time_flip_drill,
)
from .clock import Clock
from .cognitive_core import Phase, Problem, QuestionEvent, SeededRng
from .visual_search import VisualSearchTaskKind


def _level_to_difficulty(level: int) -> float:
    clamped = max(1, min(10, int(level)))
    return float(clamped - 1) / 9.0


def _difficulty_to_level(difficulty: float) -> int:
    return max(1, min(10, int(round(max(0.0, min(1.0, float(difficulty))) * 9.0)) + 1))


class AntWorkoutStage(str, Enum):
    INTRO = "intro"
    PRE_REFLECTION = "pre_reflection"
    BLOCK_SETUP = "block_setup"
    BLOCK = "block"
    POST_REFLECTION = "post_reflection"
    RESULTS = "results"


@dataclass(frozen=True, slots=True)
class AntWorkoutBlockPlan:
    block_id: str
    label: str
    description: str
    focus_skills: tuple[str, ...]
    drill_code: str
    mode: AntDrillMode
    duration_min: float
    skin: str = "mixed"
    adaptive_enabled: bool = False

    @property
    def duration_s(self) -> float:
        return max(1.0, float(self.duration_min) * 60.0)


@dataclass(frozen=True, slots=True)
class AntWorkoutPlan:
    code: str
    title: str
    description: str
    notes: tuple[str, ...]
    blocks: tuple[AntWorkoutBlockPlan, ...]

    @property
    def scored_duration_s(self) -> float:
        return float(sum(block.duration_s for block in self.blocks))

    @property
    def focus_skills(self) -> tuple[str, ...]:
        seen: set[str] = set()
        ordered: list[str] = []
        for block in self.blocks:
            for skill in block.focus_skills:
                if skill not in seen:
                    seen.add(skill)
                    ordered.append(skill)
        return tuple(ordered)


@dataclass(frozen=True, slots=True)
class AntWorkoutSnapshot:
    stage: AntWorkoutStage
    title: str
    subtitle: str
    prompt: str
    options: tuple[str, ...]
    selected_index: int
    current_block_label: str
    block_index: int
    block_total: int
    block_time_remaining_s: float | None
    workout_time_remaining_s: float | None
    attempted_total: int
    correct_total: int
    fixation_rate: float
    note_lines: tuple[str, ...]
    text_value: str = ""
    text_max_length: int = 0
    difficulty_level: int = 5
    block_default_level: int | None = None
    block_override_level: int | None = None


@dataclass(frozen=True, slots=True)
class AntWorkoutBlockResult:
    block_id: str
    label: str
    drill_code: str
    mode: str
    duration_s: float
    attempted: int
    correct: int
    total_score: float
    max_score: float
    score_ratio: float
    timeouts: int
    fixation_rate: float
    difficulty_level_start: int
    difficulty_level_end: int
    difficulty_change_count: int


@dataclass(frozen=True, slots=True)
class AntWorkoutSummary:
    attempted: int
    correct: int
    accuracy: float
    duration_s: float
    throughput_per_min: float
    mean_response_time_s: float | None
    total_score: float = 0.0
    max_score: float = 0.0
    score_ratio: float = 0.0
    correct_per_min: float = 0.0
    timeouts: int = 0
    fixation_rate: float = 0.0
    max_timeout_streak: int = 0
    mode: str = "workout"
    difficulty_level: int = 1
    difficulty_level_start: int = 1
    difficulty_level_end: int = 1
    difficulty_change_count: int = 0
    adaptive_enabled: bool = False
    adaptive_window_size: int = 0
    block_count: int = 0
    completed_blocks: int = 0
    workout_code: str = ""


@dataclass(frozen=True, slots=True)
class _ReflectionPrompt:
    prompt: str


_TEXT_LIMIT = 140
_OPENING_REFLECTIONS: tuple[_ReflectionPrompt, ...] = (
    _ReflectionPrompt("Why are you training today?"),
    _ReflectionPrompt("When you feel behind, what rule will you follow instead of spiraling?"),
)
_CLOSING_REFLECTIONS: tuple[_ReflectionPrompt, ...] = (
    _ReflectionPrompt("What started to break down first?"),
    _ReflectionPrompt("What is one rule for the next session?"),
)


def _block(
    block_id: str,
    label: str,
    description: str,
    focus_skills: tuple[str, ...],
    drill_code: str,
    mode: AntDrillMode,
    minutes: float,
    *,
    skin: str = "mixed",
) -> AntWorkoutBlockPlan:
    return AntWorkoutBlockPlan(
        block_id=block_id,
        label=label,
        description=description,
        focus_skills=focus_skills,
        drill_code=drill_code,
        mode=mode,
        duration_min=minutes,
        skin=skin,
    )


def build_ant_workout_plan(code: str, *, duration_scale: float = 1.0) -> AntWorkoutPlan:
    scale = max(0.05, float(duration_scale))
    token = str(code).strip().lower()
    if token != "airborne_numerical_workout":
        raise ValueError(f"Unknown Airborne Numerical workout code: {code}")

    blocks = (
        _block(
            "search_and_retention",
            "Search And Retention",
            "Low-pressure value lookup and short retention before the math-heavy blocks.",
            ("Information search", "Retention"),
            "ant_info_grabber",
            AntDrillMode.BUILD,
            10 * scale,
        ),
        _block(
            "warmup_snap",
            "Warm-Up: Snap Facts",
            "Basic arithmetic retrieval under low pressure to set clean output tempo.",
            ("Arithmetic retrieval", "Tempo control"),
            "ant_snap_facts_sprint",
            AntDrillMode.BUILD,
            5 * scale,
        ),
        _block(
            "warmup_time",
            "Warm-Up: Time Flip",
            "Unit and time conversion under low pressure before scenario work.",
            ("Time conversion", "Unit conversion"),
            "ant_time_flip",
            AntDrillMode.BUILD,
            5 * scale,
        ),
        _block(
            "warmup_distance",
            "Warm-Up: Distance Scan",
            "Fast route scanning and one-pass distance totals before tempo pressure starts.",
            ("Distance scanning", "Fast parsing"),
            "ant_distance_scan",
            AntDrillMode.BUILD,
            5 * scale,
        ),
        _block(
            "core_route",
            "Core Tempo: Route Time Solve",
            "Practice keeping tempo on route-time setup, HHMM output, and recovery after misses.",
            ("Route-time solving", "Failure recovery"),
            "ant_route_time_solve",
            AntDrillMode.TEMPO,
            10 * scale,
        ),
        _block(
            "core_endurance",
            "Core Tempo: Endurance Solve",
            "Practice fuel-time conversion under cutoffs without freezing on exactness.",
            ("Endurance solving", "Failure recovery"),
            "ant_endurance_solve",
            AntDrillMode.TEMPO,
            10 * scale,
        ),
        _block(
            "core_fuel",
            "Core Tempo: Fuel Burn Solve",
            "Practice tempo on fuel-burn calculations with route setup and guess-and-go discipline.",
            ("Fuel-burn solving", "Failure recovery"),
            "ant_fuel_burn_solve",
            AntDrillMode.TEMPO,
            10 * scale,
        ),
        _block(
            "scenario_steady",
            "Steady Scenario Set",
            "Full Airborne Numerical questions in grouped family runs with moderate cutoffs.",
            ("Full-question solving", "Grouped scenario work"),
            "airborne_scenario_steady",
            AntDrillMode.BUILD,
            15 * scale,
        ),
        _block(
            "scenario_pressure",
            "Pressure Scenario Set",
            "Full Airborne Numerical questions in grouped family runs with tighter caps and stricter scoring.",
            ("Full-question solving", "Pressure tolerance"),
            "airborne_scenario_pressure",
            AntDrillMode.BUILD,
            20 * scale,
        ),
    )

    return AntWorkoutPlan(
        code="airborne_numerical_workout",
        title="Airborne Numerical Workout (90m)",
        description=(
            "Standard 90-minute Airborne Numerical workout with reflection, warm-up, "
            "tempo calculation work, and full-question scenario sets."
        ),
        notes=(
            "Typed reflections and block setup screens do not count toward the 90-minute drill clock.",
            "Each drill block can override the workout default difficulty before it starts.",
            "The workout keeps the full drill library available outside the workout menu.",
        ),
        blocks=blocks,
    )


def ant_workout_menu_entries() -> tuple[tuple[str, str], ...]:
    return (("airborne_numerical_workout", "Airborne Numerical Workout (90m)"),)


class _PressureAirborneScorer(AirborneScorer):
    def score(self, *, problem: Problem, user_answer: int, raw: str) -> float:
        _ = raw
        scenario = cast(AirborneScenario | None, problem.payload)
        tolerance = max(0, int(problem.tolerance))
        if tolerance > 1:
            tolerance = max(0, tolerance // 2)
        if scenario is not None and scenario.answer_format == "hhmm":
            correct = int(problem.answer)
            current = int(user_answer)
            diff = abs((correct // 100) * 60 + (correct % 100) - ((current // 100) * 60 + (current % 100)))
            return 1.0 if diff <= tolerance else 0.0
        diff = abs(int(user_answer) - int(problem.answer))
        return 1.0 if diff <= tolerance else 0.0


class _GroupedAirborneWorkoutGenerator:
    _family_order = ("route_time", "endurance", "fuel_burn", "distance", "payload")

    def __init__(self, *, seed: int, group_size: int, pressure: bool) -> None:
        self._base = AirborneNumericalGenerator(SeededRng(seed))
        self._group_size = max(1, int(group_size))
        self._pressure = bool(pressure)
        self._family_index = 0
        self._items_in_group = 0
        self._cap_generators = {
            "route_time": AntRouteTimeSolveGenerator(seed=seed + 101),
            "endurance": AntEnduranceSolveGenerator(seed=seed + 202),
            "fuel_burn": AntFuelBurnSolveGenerator(seed=seed + 303),
            "distance": AntDistanceScanGenerator(seed=seed + 404),
            "payload": AntPayloadReferenceGenerator(seed=seed + 505),
        }

    def next_problem(self, *, difficulty: float) -> Problem:
        level = _difficulty_to_level(difficulty)
        family = self._family_order[self._family_index]
        profile = build_ant_airborne_difficulty_profile(level, family=family)
        problem = self._base.generate(profile=profile)
        self._items_in_group += 1
        if self._items_in_group >= self._group_size:
            self._items_in_group = 0
            self._family_index = (self._family_index + 1) % len(self._family_order)
        return problem

    def cap_for_problem(self, *, problem: Problem, level: int) -> float | None:
        scenario = cast(AirborneScenario | None, problem.payload)
        if scenario is None:
            return None
        family = _airborne_family_for_scenario(scenario)
        resolver = self._cap_generators[family]
        base = resolver.cap_for_problem(problem=problem, level=level)
        if base is None:
            return None
        scale = 0.80 if self._pressure else 1.10
        floor = 7.0 if self._pressure else 9.0
        return max(floor, float(base) * scale)


def _airborne_family_for_scenario(scenario: AirborneScenario) -> str:
    if scenario.question_kind in {"arrival_time", "takeoff_time"}:
        return "route_time"
    if scenario.question_kind in {"empty_time", "fuel_endurance"}:
        return "endurance"
    if scenario.question_kind == "fuel_burned":
        return "fuel_burn"
    if scenario.question_kind == "distance_travelled":
        return "distance"
    return "payload"


def build_airborne_numerical_steady_scenario_set(
    clock: Clock,
    seed: int,
    *,
    difficulty: float = 0.5,
    scored_duration_s: float = 15.0 * 60.0,
) -> TimedCapDrill:
    return TimedCapDrill(
        title="Airborne Numerical: Steady Scenario Set",
        instructions=[
            "Full Airborne Numerical questions arrive in grouped family runs.",
            "The order stays fixed: route-time, endurance, fuel, distance, payload.",
            "Feedback flashes with the next prompt. Keep moving instead of reworking misses.",
        ],
        generator=_GroupedAirborneWorkoutGenerator(seed=seed, group_size=3, pressure=False),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        practice_questions=0,
        scored_duration_s=scored_duration_s,
        mode=AntDrillMode.BUILD,
        adaptive_config=AntAdaptiveDifficultyConfig(enabled=False),
        base_caps_by_level=(40.0,) * 10,
        scorer=AirborneScorer(),
    )


def build_airborne_numerical_pressure_scenario_set(
    clock: Clock,
    seed: int,
    *,
    difficulty: float = 0.5,
    scored_duration_s: float = 20.0 * 60.0,
) -> TimedCapDrill:
    return TimedCapDrill(
        title="Airborne Numerical: Pressure Scenario Set",
        instructions=[
            "Full Airborne Numerical questions arrive in grouped family runs under tighter caps.",
            "The order stays fixed: route-time, endurance, fuel, distance, payload.",
            "Use the fastest valid method, accept misses, and reset immediately on the next item.",
        ],
        generator=_GroupedAirborneWorkoutGenerator(seed=seed, group_size=2, pressure=True),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        practice_questions=0,
        scored_duration_s=scored_duration_s,
        mode=AntDrillMode.BUILD,
        adaptive_config=AntAdaptiveDifficultyConfig(enabled=False),
        base_caps_by_level=(30.0,) * 10,
        scorer=_PressureAirborneScorer(),
    )


class AntWorkoutSession:
    def __init__(
        self,
        *,
        clock: Clock,
        seed: int,
        plan: AntWorkoutPlan,
        starting_level: int = 5,
    ) -> None:
        self._clock = clock
        self._seed = int(seed)
        self._plan = plan
        self._starting_level = max(1, min(10, int(starting_level)))
        self._stage = AntWorkoutStage.INTRO
        self._reflection_input = ""
        self._reflection_index = 0
        self._opening_reflections: list[str] = []
        self._closing_reflections: list[str] = []
        self._current_block_index = -1
        self._current_block_plan: AntWorkoutBlockPlan | None = None
        self._current_engine: object | None = None
        self._current_block_level = self._starting_level
        self._pending_block_level = self._starting_level
        self._block_results: list[AntWorkoutBlockResult] = []
        self._events: list[QuestionEvent] = []
        self._last_finished_block_level = self._starting_level

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def difficulty(self) -> float:
        return _level_to_difficulty(self._starting_level)

    @property
    def practice_questions(self) -> int:
        return 0

    @property
    def scored_duration_s(self) -> float:
        return self._plan.scored_duration_s

    @property
    def stage(self) -> AntWorkoutStage:
        return self._stage

    @property
    def selected_index(self) -> int:
        return 0

    def current_engine(self) -> object | None:
        return self._current_engine

    def current_block_plan(self) -> AntWorkoutBlockPlan | None:
        return self._current_block_plan

    def can_exit(self) -> bool:
        return self._stage in (AntWorkoutStage.INTRO, AntWorkoutStage.RESULTS)

    def adjust_starting_level(self, delta: int) -> None:
        if self._stage is not AntWorkoutStage.INTRO:
            return
        self._starting_level = max(1, min(10, self._starting_level + int(delta)))

    def adjust_block_level(self, delta: int) -> None:
        if self._stage is not AntWorkoutStage.BLOCK_SETUP:
            return
        self._pending_block_level = max(1, min(10, self._pending_block_level + int(delta)))

    def append_text(self, text: str) -> None:
        if self._stage not in (AntWorkoutStage.PRE_REFLECTION, AntWorkoutStage.POST_REFLECTION):
            return
        cleaned = "".join(ch for ch in str(text) if 32 <= ord(ch) <= 126)
        if not cleaned:
            return
        room = max(0, _TEXT_LIMIT - len(self._reflection_input))
        if room <= 0:
            return
        self._reflection_input += cleaned[:room]

    def backspace_text(self) -> None:
        if self._stage in (AntWorkoutStage.PRE_REFLECTION, AntWorkoutStage.POST_REFLECTION):
            self._reflection_input = self._reflection_input[:-1]

    def activate(self) -> None:
        if self._stage is AntWorkoutStage.INTRO:
            self._stage = AntWorkoutStage.PRE_REFLECTION
            self._reflection_index = 0
            self._reflection_input = ""
            return

        if self._stage is AntWorkoutStage.PRE_REFLECTION:
            response = self._reflection_input.strip()
            if response == "":
                return
            self._opening_reflections.append(response[:_TEXT_LIMIT])
            self._reflection_input = ""
            self._reflection_index += 1
            if self._reflection_index >= len(_OPENING_REFLECTIONS):
                self._start_first_block()
            return

        if self._stage is AntWorkoutStage.BLOCK_SETUP:
            self._start_current_block()
            return

        if self._stage is AntWorkoutStage.POST_REFLECTION:
            response = self._reflection_input.strip()
            if response == "":
                return
            self._closing_reflections.append(response[:_TEXT_LIMIT])
            self._reflection_input = ""
            self._reflection_index += 1
            if self._reflection_index >= len(_CLOSING_REFLECTIONS):
                self._stage = AntWorkoutStage.RESULTS
            return

    def debug_skip_stage(self) -> None:
        if self._stage is AntWorkoutStage.RESULTS:
            return
        if self._stage in (AntWorkoutStage.PRE_REFLECTION, AntWorkoutStage.POST_REFLECTION):
            if self._reflection_input.strip() == "":
                self._reflection_input = "skip"
        self.activate()

    def debug_skip_block(self) -> None:
        if self._stage is not AntWorkoutStage.BLOCK:
            self.debug_skip_stage()
            return
        if not self._submit_current_engine_skip(
            ("__skip_section__", "skip_section", "__skip_all__", "skip_all")
        ):
            self._force_current_engine_phase(Phase.RESULTS)
        self.sync_runtime()

    def debug_finish(self) -> None:
        if self._stage is AntWorkoutStage.BLOCK:
            if not self._submit_current_engine_skip(
                ("__skip_all__", "skip_all", "__skip_section__", "skip_section")
            ):
                self._force_current_engine_phase(Phase.RESULTS)
            self._record_current_block_result()
        self._current_engine = None
        self._current_block_plan = None
        self._reflection_input = ""
        self._reflection_index = 0
        self._stage = AntWorkoutStage.RESULTS

    def submit_answer(self, raw: str) -> bool:
        if self._stage is not AntWorkoutStage.BLOCK or self._current_engine is None:
            return False
        submit = getattr(self._current_engine, "submit_answer", None)
        if not callable(submit):
            return False
        return bool(submit(raw))

    def update(self) -> None:
        if self._stage is not AntWorkoutStage.BLOCK or self._current_engine is None:
            return
        update = getattr(self._current_engine, "update", None)
        if callable(update):
            update()
        self.sync_runtime()

    def sync_runtime(self) -> None:
        if self._stage is not AntWorkoutStage.BLOCK or self._current_engine is None:
            return
        phase = getattr(self._current_engine, "phase", None)
        if phase is Phase.RESULTS:
            self._complete_current_block()

    def snapshot(self) -> AntWorkoutSnapshot:
        attempted_total, correct_total, fixation_rate = self._running_totals()
        if self._stage is AntWorkoutStage.INTRO:
            return AntWorkoutSnapshot(
                stage=self._stage,
                title=self._plan.title,
                subtitle="Workout Intro",
                prompt=self._plan.description,
                options=(),
                selected_index=0,
                current_block_label="",
                block_index=0,
                block_total=len(self._plan.blocks),
                block_time_remaining_s=None,
                workout_time_remaining_s=self._plan.scored_duration_s,
                attempted_total=attempted_total,
                correct_total=correct_total,
                fixation_rate=fixation_rate,
                note_lines=(
                    f"Workout default difficulty: {self._starting_level}/10",
                    f"Timed drill total: {int(round(self._plan.scored_duration_s / 60.0))} minutes",
                    f"Focus: {', '.join(self._plan.focus_skills)}",
                    *self._plan.notes,
                ),
                difficulty_level=self._starting_level,
            )

        if self._stage is AntWorkoutStage.PRE_REFLECTION:
            prompt = _OPENING_REFLECTIONS[self._reflection_index].prompt
            return AntWorkoutSnapshot(
                stage=self._stage,
                title=self._plan.title,
                subtitle=f"Opening Reflection {self._reflection_index + 1}/{len(_OPENING_REFLECTIONS)}",
                prompt=prompt,
                options=(),
                selected_index=0,
                current_block_label="",
                block_index=0,
                block_total=len(self._plan.blocks),
                block_time_remaining_s=None,
                workout_time_remaining_s=self._plan.scored_duration_s,
                attempted_total=attempted_total,
                correct_total=correct_total,
                fixation_rate=fixation_rate,
                note_lines=(
                    "Type a short answer and press Enter.",
                    "This reflection is not saved and does not count toward workout time.",
                ),
                text_value=self._reflection_input,
                text_max_length=_TEXT_LIMIT,
                difficulty_level=self._starting_level,
            )

        if self._stage is AntWorkoutStage.BLOCK_SETUP:
            block = self._current_block_plan
            assert block is not None
            return AntWorkoutSnapshot(
                stage=self._stage,
                title=self._plan.title,
                subtitle=f"Block Setup {self._current_block_index + 1}/{len(self._plan.blocks)}",
                prompt=block.description,
                options=(),
                selected_index=0,
                current_block_label=block.label,
                block_index=self._current_block_index + 1,
                block_total=len(self._plan.blocks),
                block_time_remaining_s=block.duration_s,
                workout_time_remaining_s=self._workout_time_remaining(block.duration_s),
                attempted_total=attempted_total,
                correct_total=correct_total,
                fixation_rate=fixation_rate,
                note_lines=(
                    f"Focus: {', '.join(block.focus_skills)}",
                    f"Mode: {block.mode.value}",
                    f"Duration: {int(round(block.duration_min))} minutes",
                    "Use Left/Right to adjust this block only. The next block resets to the workout default.",
                ),
                difficulty_level=self._starting_level,
                block_default_level=self._starting_level,
                block_override_level=self._pending_block_level,
            )

        if self._stage is AntWorkoutStage.POST_REFLECTION:
            prompt = _CLOSING_REFLECTIONS[self._reflection_index].prompt
            return AntWorkoutSnapshot(
                stage=self._stage,
                title=self._plan.title,
                subtitle=f"Closing Reflection {self._reflection_index + 1}/{len(_CLOSING_REFLECTIONS)}",
                prompt=prompt,
                options=(),
                selected_index=0,
                current_block_label="",
                block_index=len(self._plan.blocks),
                block_total=len(self._plan.blocks),
                block_time_remaining_s=None,
                workout_time_remaining_s=0.0,
                attempted_total=attempted_total,
                correct_total=correct_total,
                fixation_rate=fixation_rate,
                note_lines=(
                    "Type a short answer and press Enter.",
                    "This reflection is not saved and does not count toward workout time.",
                ),
                text_value=self._reflection_input,
                text_max_length=_TEXT_LIMIT,
                difficulty_level=self._starting_level,
            )

        if self._stage is AntWorkoutStage.RESULTS:
            summary = self.scored_summary()
            return AntWorkoutSnapshot(
                stage=self._stage,
                title=self._plan.title,
                subtitle="Workout Results",
                prompt=(
                    f"Correct/min {summary.correct_per_min:.1f} | "
                    f"Accuracy {summary.accuracy * 100.0:.1f}% | "
                    f"Fixation {summary.fixation_rate * 100.0:.1f}%"
                ),
                options=(),
                selected_index=0,
                current_block_label="",
                block_index=len(self._plan.blocks),
                block_total=len(self._plan.blocks),
                block_time_remaining_s=None,
                workout_time_remaining_s=0.0,
                attempted_total=attempted_total,
                correct_total=correct_total,
                fixation_rate=fixation_rate,
                note_lines=self._results_note_lines(summary),
                difficulty_level=self._starting_level,
            )

        assert self._current_engine is not None
        block_plan = self._current_block_plan
        engine_snapshot = self._current_engine.snapshot()
        engine_summary: AntDrillAttemptSummary = self._current_engine.scored_summary()
        current_attempted = attempted_total + max(0, int(engine_summary.attempted))
        current_correct = correct_total + max(0, int(engine_summary.correct))
        current_timeouts = sum(result.timeouts for result in self._block_results) + max(
            0, int(engine_summary.timeouts)
        )
        current_fixation_rate = 0.0 if current_attempted == 0 else current_timeouts / current_attempted
        return AntWorkoutSnapshot(
            stage=self._stage,
            title=self._plan.title,
            subtitle=f"Block {self._current_block_index + 1}/{len(self._plan.blocks)}",
            prompt=str(engine_snapshot.prompt),
            options=(),
            selected_index=0,
            current_block_label="" if block_plan is None else block_plan.label,
            block_index=self._current_block_index + 1,
            block_total=len(self._plan.blocks),
            block_time_remaining_s=engine_snapshot.time_remaining_s,
            workout_time_remaining_s=self._workout_time_remaining(engine_snapshot.time_remaining_s),
            attempted_total=current_attempted,
            correct_total=current_correct,
            fixation_rate=current_fixation_rate,
            note_lines=self._block_note_lines(block_plan),
            difficulty_level=self._current_block_level,
        )

    def events(self) -> list[QuestionEvent]:
        return list(self._events)

    def scored_summary(self) -> AntWorkoutSummary:
        attempted = sum(result.attempted for result in self._block_results)
        correct = sum(result.correct for result in self._block_results)
        total_score = sum(result.total_score for result in self._block_results)
        max_score = sum(result.max_score for result in self._block_results)
        timeouts = sum(result.timeouts for result in self._block_results)
        accuracy = 0.0 if attempted == 0 else correct / attempted
        duration_s = float(sum(result.duration_s for result in self._block_results))
        throughput = (attempted / duration_s) * 60.0 if duration_s > 0.0 else 0.0
        correct_per_min = (correct / duration_s) * 60.0 if duration_s > 0.0 else 0.0
        fixation_rate = 0.0 if attempted == 0 else timeouts / attempted
        rts = [event.response_time_s for event in self._events]
        mean_rt = None if not rts else sum(rts) / len(rts)
        difficulty_change_count = sum(result.difficulty_change_count for result in self._block_results)
        return AntWorkoutSummary(
            attempted=attempted,
            correct=correct,
            accuracy=accuracy,
            duration_s=duration_s,
            throughput_per_min=throughput,
            mean_response_time_s=mean_rt,
            total_score=float(total_score),
            max_score=float(max_score),
            score_ratio=0.0 if max_score <= 0.0 else float(total_score) / float(max_score),
            correct_per_min=correct_per_min,
            timeouts=timeouts,
            fixation_rate=fixation_rate,
            max_timeout_streak=self._max_timeout_streak(),
            mode="workout",
            difficulty_level=self._starting_level,
            difficulty_level_start=self._starting_level,
            difficulty_level_end=self._last_finished_block_level,
            difficulty_change_count=difficulty_change_count,
            adaptive_enabled=False,
            adaptive_window_size=0,
            block_count=len(self._plan.blocks),
            completed_blocks=len(self._block_results),
            workout_code=self._plan.code,
        )

    def _results_note_lines(self, summary: AntWorkoutSummary) -> tuple[str, ...]:
        return (
            f"Blocks completed: {summary.completed_blocks}/{summary.block_count}",
            f"Default level: {summary.difficulty_level_start}/10",
            f"Exact accuracy: {summary.accuracy * 100.0:.1f}%",
            f"Score ratio: {summary.score_ratio * 100.0:.1f}%",
            f"Focus coverage: {', '.join(self._plan.focus_skills)}",
            "Reflections were for focus only and were not saved.",
        )

    def _running_totals(self) -> tuple[int, int, float]:
        attempted = sum(result.attempted for result in self._block_results)
        correct = sum(result.correct for result in self._block_results)
        timeouts = sum(result.timeouts for result in self._block_results)
        fixation = 0.0 if attempted == 0 else timeouts / attempted
        return attempted, correct, fixation

    def _workout_time_remaining(self, current_block_remaining_s: float | None) -> float:
        remaining = 0.0 if current_block_remaining_s is None else float(current_block_remaining_s)
        for block in self._plan.blocks[self._current_block_index + 1 :]:
            remaining += block.duration_s
        return remaining

    def _block_note_lines(self, block_plan: AntWorkoutBlockPlan | None) -> tuple[str, ...]:
        if block_plan is None:
            return ()
        return (
            block_plan.description,
            f"Focus: {', '.join(block_plan.focus_skills)}",
            f"Mode: {block_plan.mode.value}",
            f"Block level: {self._current_block_level}/10",
        )

    def _start_first_block(self) -> None:
        self._current_block_index = -1
        self._block_results.clear()
        self._events.clear()
        self._prepare_next_block()

    def _prepare_next_block(self) -> None:
        self._current_block_index += 1
        if self._current_block_index >= len(self._plan.blocks):
            self._stage = AntWorkoutStage.POST_REFLECTION
            self._reflection_index = 0
            self._reflection_input = ""
            self._current_engine = None
            self._current_block_plan = None
            return
        self._current_block_plan = self._plan.blocks[self._current_block_index]
        self._pending_block_level = self._starting_level
        self._current_engine = None
        self._stage = AntWorkoutStage.BLOCK_SETUP

    def _start_current_block(self) -> None:
        block = self._current_block_plan
        assert block is not None
        self._stage = AntWorkoutStage.BLOCK
        self._current_block_level = self._pending_block_level
        difficulty = _level_to_difficulty(self._current_block_level)
        block_seed = self._seed + ((self._current_block_index + 1) * 101)
        if block.drill_code == "ant_snap_facts_sprint":
            engine = build_ant_snap_facts_sprint_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=AntSnapFactsSprintConfig(
                    skin=block.skin,
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                    adaptive=AntAdaptiveDifficultyConfig(enabled=False),
                ),
            )
        elif block.drill_code == "ant_time_flip":
            engine = build_ant_time_flip_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=AntTimeFlipConfig(
                    skin=block.skin,
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                    adaptive=AntAdaptiveDifficultyConfig(enabled=False),
                ),
            )
        elif block.drill_code == "ant_distance_scan":
            engine = build_ant_distance_scan_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=AntDistanceScanConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                    adaptive=AntAdaptiveDifficultyConfig(enabled=False),
                ),
            )
        elif block.drill_code == "ant_route_time_solve":
            engine = build_ant_route_time_solve_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=AntRouteTimeSolveConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                    adaptive=AntAdaptiveDifficultyConfig(enabled=False),
                ),
            )
        elif block.drill_code == "ant_endurance_solve":
            engine = build_ant_endurance_solve_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=AntEnduranceSolveConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                    adaptive=AntAdaptiveDifficultyConfig(enabled=False),
                ),
            )
        elif block.drill_code == "ant_fuel_burn_solve":
            engine = build_ant_fuel_burn_solve_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=AntFuelBurnSolveConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                    adaptive=AntAdaptiveDifficultyConfig(enabled=False),
                ),
            )
        elif block.drill_code == "ant_info_grabber":
            engine = build_ant_info_grabber_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=AntInfoGrabberConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                    adaptive=AntAdaptiveDifficultyConfig(enabled=False),
                ),
            )
        elif block.drill_code == "airborne_scenario_steady":
            engine = build_airborne_numerical_steady_scenario_set(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                scored_duration_s=block.duration_s,
            )
        elif block.drill_code == "airborne_scenario_pressure":
            engine = build_airborne_numerical_pressure_scenario_set(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                scored_duration_s=block.duration_s,
            )
        elif block.drill_code == "abd_cardinal_anchors":
            engine = build_abd_cardinal_anchors_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=AbdDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                ),
            )
        elif block.drill_code == "abd_intermediate_anchors":
            engine = build_abd_intermediate_anchors_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=AbdDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                ),
            )
        elif block.drill_code == "abd_angle_calibration":
            engine = build_abd_angle_calibration_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=AbdDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                ),
            )
        elif block.drill_code == "abd_bearing_calibration":
            engine = build_abd_bearing_calibration_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=AbdDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                ),
            )
        elif block.drill_code == "abd_mixed_tempo":
            engine = build_abd_mixed_tempo_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=AbdDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                ),
            )
        elif block.drill_code == "abd_family_run_angle":
            engine = build_abd_test_style_family_run_drill(
                clock=self._clock,
                seed=block_seed,
                family=AnglesBearingsQuestionKind.ANGLE_BETWEEN_LINES,
                difficulty=difficulty,
                mode=block.mode,
                config=AbdFamilyRunConfig(
                    family=AnglesBearingsQuestionKind.ANGLE_BETWEEN_LINES,
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                ),
            )
        elif block.drill_code == "abd_family_run_bearing":
            engine = build_abd_test_style_family_run_drill(
                clock=self._clock,
                seed=block_seed,
                family=AnglesBearingsQuestionKind.BEARING_FROM_REFERENCE,
                difficulty=difficulty,
                mode=block.mode,
                config=AbdFamilyRunConfig(
                    family=AnglesBearingsQuestionKind.BEARING_FROM_REFERENCE,
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                ),
            )
        elif block.drill_code == "no_fact_prime":
            engine = build_no_fact_prime_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=NoDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                    adaptive=AntAdaptiveDifficultyConfig(enabled=False),
                ),
            )
        elif block.drill_code == "no_operator_ladders":
            engine = build_no_operator_ladders_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=NoDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                    adaptive=AntAdaptiveDifficultyConfig(enabled=False),
                ),
            )
        elif block.drill_code == "no_clean_compute":
            engine = build_no_clean_compute_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=NoDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                    adaptive=AntAdaptiveDifficultyConfig(enabled=False),
                ),
            )
        elif block.drill_code == "no_mixed_tempo":
            engine = build_no_mixed_tempo_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=NoDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                    adaptive=AntAdaptiveDifficultyConfig(enabled=False),
                ),
            )
        elif block.drill_code == "no_pressure_run":
            engine = build_no_pressure_run_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=NoDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                    adaptive=AntAdaptiveDifficultyConfig(enabled=False),
                ),
            )
        elif block.drill_code == "mr_relevant_info_scan":
            engine = build_mr_relevant_info_scan_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=MrDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                    adaptive=AntAdaptiveDifficultyConfig(enabled=False),
                ),
            )
        elif block.drill_code == "mr_unit_relation_prime":
            engine = build_mr_unit_relation_prime_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=MrDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                    adaptive=AntAdaptiveDifficultyConfig(enabled=False),
                ),
            )
        elif block.drill_code == "mr_one_step_solve":
            engine = build_mr_one_step_solve_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=MrDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                    adaptive=AntAdaptiveDifficultyConfig(enabled=False),
                ),
            )
        elif block.drill_code == "mr_multi_step_solve":
            engine = build_mr_multi_step_solve_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=MrDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                    adaptive=AntAdaptiveDifficultyConfig(enabled=False),
                ),
            )
        elif block.drill_code == "mr_domain_run":
            engine = build_mr_domain_run_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=MrDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                    adaptive=AntAdaptiveDifficultyConfig(enabled=False),
                ),
            )
        elif block.drill_code == "mr_mixed_pressure_set":
            engine = build_mr_mixed_pressure_set_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=MrDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                    adaptive=AntAdaptiveDifficultyConfig(enabled=False),
                ),
            )
        elif block.drill_code == "dr_visible_copy":
            engine = build_dr_visible_copy_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=DigitRecognitionDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                ),
            )
        elif block.drill_code == "dr_position_probe":
            engine = build_dr_position_probe_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=DigitRecognitionDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                ),
            )
        elif block.drill_code == "dr_visible_family_primer":
            engine = build_dr_visible_family_primer_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=DigitRecognitionDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                ),
            )
        elif block.drill_code == "dr_recall_run":
            engine = build_dr_recall_run_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=DigitRecognitionDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                ),
            )
        elif block.drill_code == "dr_count_target":
            engine = build_dr_count_target_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=DigitRecognitionDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                ),
            )
        elif block.drill_code == "dr_different_digit":
            engine = build_dr_different_digit_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=DigitRecognitionDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                ),
            )
        elif block.drill_code == "dr_grouped_family_run":
            engine = build_dr_grouped_family_run_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=DigitRecognitionDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                ),
            )
        elif block.drill_code == "dr_mixed_pressure":
            engine = build_dr_mixed_pressure_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=DigitRecognitionDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                ),
            )
        elif block.drill_code == "ic_heading_anchor":
            engine = build_ic_heading_anchor_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=IcDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                    adaptive=AntAdaptiveDifficultyConfig(enabled=False),
                ),
            )
        elif block.drill_code == "ic_attitude_frame":
            engine = build_ic_attitude_frame_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=IcDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                    adaptive=AntAdaptiveDifficultyConfig(enabled=False),
                ),
            )
        elif block.drill_code == "ic_part1_orientation_run":
            engine = build_ic_part1_orientation_run_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=IcDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                    adaptive=AntAdaptiveDifficultyConfig(enabled=False),
                ),
            )
        elif block.drill_code == "ic_description_prime":
            engine = build_ic_description_prime_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=IcDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                    adaptive=AntAdaptiveDifficultyConfig(enabled=False),
                ),
            )
        elif block.drill_code == "ic_reverse_panel_prime":
            engine = build_ic_reverse_panel_prime_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=IcDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                    adaptive=AntAdaptiveDifficultyConfig(enabled=False),
                ),
            )
        elif block.drill_code == "ic_reverse_panel_run":
            engine = build_ic_reverse_panel_run_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=IcDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                    adaptive=AntAdaptiveDifficultyConfig(enabled=False),
                ),
            )
        elif block.drill_code == "ic_description_run":
            engine = build_ic_description_run_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=IcDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                    adaptive=AntAdaptiveDifficultyConfig(enabled=False),
                ),
            )
        elif block.drill_code == "ic_mixed_part_run":
            engine = build_ic_mixed_part_run_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=IcDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                    adaptive=AntAdaptiveDifficultyConfig(enabled=False),
                ),
            )
        elif block.drill_code == "ic_pressure_run":
            engine = build_ic_pressure_run_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=IcDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                    adaptive=AntAdaptiveDifficultyConfig(enabled=False),
                ),
            )
        elif block.drill_code == "tr_scene_anchor":
            engine = build_tr_scene_anchor_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=TrDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                    adaptive=AntAdaptiveDifficultyConfig(enabled=False),
                ),
            )
        elif block.drill_code == "tr_scene_modifier_run":
            engine = build_tr_scene_modifier_run_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=TrDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                    adaptive=AntAdaptiveDifficultyConfig(enabled=False),
                ),
            )
        elif block.drill_code == "tr_light_anchor":
            engine = build_tr_light_anchor_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=TrDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                    adaptive=AntAdaptiveDifficultyConfig(enabled=False),
                ),
            )
        elif block.drill_code == "tr_scan_anchor":
            engine = build_tr_scan_anchor_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=TrDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                    adaptive=AntAdaptiveDifficultyConfig(enabled=False),
                ),
            )
        elif block.drill_code == "tr_system_anchor":
            engine = build_tr_system_anchor_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=TrDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                    adaptive=AntAdaptiveDifficultyConfig(enabled=False),
                ),
            )
        elif block.drill_code == "tr_panel_switch_run":
            engine = build_tr_panel_switch_run_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=TrDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                    adaptive=AntAdaptiveDifficultyConfig(enabled=False),
                ),
            )
        elif block.drill_code == "tr_mixed_tempo":
            engine = build_tr_mixed_tempo_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=TrDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                    adaptive=AntAdaptiveDifficultyConfig(enabled=False),
                ),
            )
        elif block.drill_code == "tr_pressure_run":
            engine = build_tr_pressure_run_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=TrDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                    adaptive=AntAdaptiveDifficultyConfig(enabled=False),
                ),
            )
        elif block.drill_code == "sl_quantitative_anchor":
            engine = build_sl_quantitative_anchor_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=SlDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                    adaptive=AntAdaptiveDifficultyConfig(enabled=False),
                ),
            )
        elif block.drill_code == "sl_flow_trace_anchor":
            engine = build_sl_flow_trace_anchor_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=SlDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                    adaptive=AntAdaptiveDifficultyConfig(enabled=False),
                ),
            )
        elif block.drill_code == "sl_graph_rule_anchor":
            engine = build_sl_graph_rule_anchor_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=SlDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                    adaptive=AntAdaptiveDifficultyConfig(enabled=False),
                ),
            )
        elif block.drill_code == "sl_fault_diagnosis_prime":
            engine = build_sl_fault_diagnosis_prime_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=SlDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                    adaptive=AntAdaptiveDifficultyConfig(enabled=False),
                ),
            )
        elif block.drill_code == "sl_index_switch_run":
            engine = build_sl_index_switch_run_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=SlDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                    adaptive=AntAdaptiveDifficultyConfig(enabled=False),
                ),
            )
        elif block.drill_code == "sl_family_run":
            engine = build_sl_family_run_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=SlDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                    adaptive=AntAdaptiveDifficultyConfig(enabled=False),
                ),
            )
        elif block.drill_code == "sl_mixed_tempo":
            engine = build_sl_mixed_tempo_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=SlDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                    adaptive=AntAdaptiveDifficultyConfig(enabled=False),
                ),
            )
        elif block.drill_code == "sl_pressure_run":
            engine = build_sl_pressure_run_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=SlDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                    adaptive=AntAdaptiveDifficultyConfig(enabled=False),
                ),
            )
        elif block.drill_code == "tbl_part1_anchor":
            engine = build_tbl_part1_anchor_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=TblDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                    adaptive=AntAdaptiveDifficultyConfig(enabled=False),
                ),
            )
        elif block.drill_code == "tbl_part1_scan_run":
            engine = build_tbl_part1_scan_run_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=TblDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                    adaptive=AntAdaptiveDifficultyConfig(enabled=False),
                ),
            )
        elif block.drill_code == "tbl_part2_prime":
            engine = build_tbl_part2_prime_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=TblDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                    adaptive=AntAdaptiveDifficultyConfig(enabled=False),
                ),
            )
        elif block.drill_code == "tbl_part2_correction_run":
            engine = build_tbl_part2_correction_run_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=TblDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                    adaptive=AntAdaptiveDifficultyConfig(enabled=False),
                ),
            )
        elif block.drill_code == "tbl_part_switch_run":
            engine = build_tbl_part_switch_run_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=TblDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                    adaptive=AntAdaptiveDifficultyConfig(enabled=False),
                ),
            )
        elif block.drill_code == "tbl_card_family_run":
            engine = build_tbl_card_family_run_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=TblDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                    adaptive=AntAdaptiveDifficultyConfig(enabled=False),
                ),
            )
        elif block.drill_code == "tbl_mixed_tempo":
            engine = build_tbl_mixed_tempo_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=TblDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                    adaptive=AntAdaptiveDifficultyConfig(enabled=False),
                ),
            )
        elif block.drill_code == "tbl_pressure_run":
            engine = build_tbl_pressure_run_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=TblDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                    adaptive=AntAdaptiveDifficultyConfig(enabled=False),
                ),
            )
        elif block.drill_code == "sma_joystick_horizontal_anchor":
            engine = build_sma_joystick_horizontal_anchor_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=SmaDrillConfig(
                    scored_duration_s=block.duration_s,
                ),
            )
        elif block.drill_code == "sma_joystick_vertical_anchor":
            engine = build_sma_joystick_vertical_anchor_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=SmaDrillConfig(
                    scored_duration_s=block.duration_s,
                ),
            )
        elif block.drill_code == "sma_joystick_hold_run":
            engine = build_sma_joystick_hold_run_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=SmaDrillConfig(
                    scored_duration_s=block.duration_s,
                ),
            )
        elif block.drill_code == "sma_split_horizontal_prime":
            engine = build_sma_split_horizontal_prime_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=SmaDrillConfig(
                    scored_duration_s=block.duration_s,
                ),
            )
        elif block.drill_code == "sma_split_coordination_run":
            engine = build_sma_split_coordination_run_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=SmaDrillConfig(
                    scored_duration_s=block.duration_s,
                ),
            )
        elif block.drill_code == "sma_mode_switch_run":
            engine = build_sma_mode_switch_run_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=SmaDrillConfig(
                    scored_duration_s=block.duration_s,
                ),
            )
        elif block.drill_code == "sma_disturbance_tempo":
            engine = build_sma_disturbance_tempo_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=SmaDrillConfig(
                    scored_duration_s=block.duration_s,
                ),
            )
        elif block.drill_code == "sma_pressure_run":
            engine = build_sma_pressure_run_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=SmaDrillConfig(
                    scored_duration_s=block.duration_s,
                ),
            )
        elif block.drill_code == "ac_gate_anchor":
            engine = build_ac_gate_anchor_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=AcDrillConfig(
                    scored_duration_s=block.duration_s,
                ),
            )
        elif block.drill_code == "ac_state_command_prime":
            engine = build_ac_state_command_prime_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=AcDrillConfig(
                    scored_duration_s=block.duration_s,
                ),
            )
        elif block.drill_code == "ac_gate_directive_run":
            engine = build_ac_gate_directive_run_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=AcDrillConfig(
                    scored_duration_s=block.duration_s,
                ),
            )
        elif block.drill_code == "ac_digit_sequence_prime":
            engine = build_ac_digit_sequence_prime_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=AcDrillConfig(
                    scored_duration_s=block.duration_s,
                ),
            )
        elif block.drill_code == "ac_trigger_cue_anchor":
            engine = build_ac_trigger_cue_anchor_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=AcDrillConfig(
                    scored_duration_s=block.duration_s,
                ),
            )
        elif block.drill_code == "ac_callsign_filter_run":
            engine = build_ac_callsign_filter_run_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=AcDrillConfig(
                    scored_duration_s=block.duration_s,
                ),
            )
        elif block.drill_code == "ac_mixed_tempo":
            engine = build_ac_mixed_tempo_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=AcDrillConfig(
                    scored_duration_s=block.duration_s,
                ),
            )
        elif block.drill_code == "ac_pressure_run":
            engine = build_ac_pressure_run_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=AcDrillConfig(
                    scored_duration_s=block.duration_s,
                ),
            )
        elif block.drill_code == "cu_controls_anchor":
            engine = build_cu_controls_anchor_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=CuDrillConfig(scored_duration_s=block.duration_s),
            )
        elif block.drill_code == "cu_navigation_anchor":
            engine = build_cu_navigation_anchor_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=CuDrillConfig(scored_duration_s=block.duration_s),
            )
        elif block.drill_code == "cu_engine_balance_run":
            engine = build_cu_engine_balance_run_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=CuDrillConfig(scored_duration_s=block.duration_s),
            )
        elif block.drill_code == "cu_sensors_timing_prime":
            engine = build_cu_sensors_timing_prime_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=CuDrillConfig(scored_duration_s=block.duration_s),
            )
        elif block.drill_code == "cu_objective_prime":
            engine = build_cu_objective_prime_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=CuDrillConfig(scored_duration_s=block.duration_s),
            )
        elif block.drill_code == "cu_state_code_run":
            engine = build_cu_state_code_run_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=CuDrillConfig(scored_duration_s=block.duration_s),
            )
        elif block.drill_code == "cu_mixed_tempo":
            engine = build_cu_mixed_tempo_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=CuDrillConfig(scored_duration_s=block.duration_s),
            )
        elif block.drill_code == "cu_pressure_run":
            engine = build_cu_pressure_run_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=CuDrillConfig(scored_duration_s=block.duration_s),
            )
        elif block.drill_code == "sa_picture_anchor":
            engine = build_sa_picture_anchor_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=SaDrillConfig(scored_duration_s=block.duration_s),
            )
        elif block.drill_code == "sa_contact_identification_prime":
            engine = build_sa_contact_identification_prime_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=SaDrillConfig(scored_duration_s=block.duration_s),
            )
        elif block.drill_code == "sa_status_recall_prime":
            engine = build_sa_status_recall_prime_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=SaDrillConfig(scored_duration_s=block.duration_s),
            )
        elif block.drill_code == "sa_future_projection_run":
            engine = build_sa_future_projection_run_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=SaDrillConfig(scored_duration_s=block.duration_s),
            )
        elif block.drill_code == "sa_action_selection_run":
            engine = build_sa_action_selection_run_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=SaDrillConfig(scored_duration_s=block.duration_s),
            )
        elif block.drill_code == "sa_family_switch_run":
            engine = build_sa_family_switch_run_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=SaDrillConfig(scored_duration_s=block.duration_s),
            )
        elif block.drill_code == "sa_mixed_tempo":
            engine = build_sa_mixed_tempo_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=SaDrillConfig(scored_duration_s=block.duration_s),
            )
        elif block.drill_code == "sa_pressure_run":
            engine = build_sa_pressure_run_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=SaDrillConfig(scored_duration_s=block.duration_s),
            )
        elif block.drill_code == "rt_lock_anchor":
            engine = build_rt_lock_anchor_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=RtDrillConfig(scored_duration_s=block.duration_s),
            )
        elif block.drill_code == "rt_building_handoff_prime":
            engine = build_rt_building_handoff_prime_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=RtDrillConfig(scored_duration_s=block.duration_s),
            )
        elif block.drill_code == "rt_terrain_recovery_run":
            engine = build_rt_terrain_recovery_run_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=RtDrillConfig(scored_duration_s=block.duration_s),
            )
        elif block.drill_code == "rt_capture_timing_prime":
            engine = build_rt_capture_timing_prime_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=RtDrillConfig(scored_duration_s=block.duration_s),
            )
        elif block.drill_code == "rt_ground_tempo_run":
            engine = build_rt_ground_tempo_run_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=RtDrillConfig(scored_duration_s=block.duration_s),
            )
        elif block.drill_code == "rt_air_speed_run":
            engine = build_rt_air_speed_run_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=RtDrillConfig(scored_duration_s=block.duration_s),
            )
        elif block.drill_code == "rt_mixed_tempo":
            engine = build_rt_mixed_tempo_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=RtDrillConfig(scored_duration_s=block.duration_s),
            )
        elif block.drill_code == "rt_pressure_run":
            engine = build_rt_pressure_run_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=RtDrillConfig(scored_duration_s=block.duration_s),
            )
        elif block.drill_code == "cln_sequence_copy":
            engine = build_cln_sequence_copy_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=ClnDrillConfig(
                    practice_rounds=0,
                    scored_duration_s=block.duration_s,
                ),
            )
        elif block.drill_code == "cln_sequence_match":
            engine = build_cln_sequence_match_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=ClnDrillConfig(
                    practice_rounds=0,
                    scored_duration_s=block.duration_s,
                ),
            )
        elif block.drill_code == "cln_math_prime":
            engine = build_cln_math_prime_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=ClnDrillConfig(
                    practice_rounds=0,
                    scored_duration_s=block.duration_s,
                ),
            )
        elif block.drill_code == "cln_colour_lane":
            engine = build_cln_colour_lane_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=ClnDrillConfig(
                    practice_rounds=0,
                    scored_duration_s=block.duration_s,
                ),
            )
        elif block.drill_code == "cln_memory_math":
            engine = build_cln_memory_math_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=ClnDrillConfig(
                    practice_rounds=0,
                    scored_duration_s=block.duration_s,
                ),
            )
        elif block.drill_code == "cln_memory_colour":
            engine = build_cln_memory_colour_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=ClnDrillConfig(
                    practice_rounds=0,
                    scored_duration_s=block.duration_s,
                ),
            )
        elif block.drill_code == "cln_full_steady":
            engine = build_cln_full_steady_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=ClnDrillConfig(
                    practice_rounds=0,
                    scored_duration_s=block.duration_s,
                ),
            )
        elif block.drill_code == "cln_full_pressure":
            engine = build_cln_full_pressure_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=ClnDrillConfig(
                    practice_rounds=0,
                    scored_duration_s=block.duration_s,
                ),
            )
        elif block.drill_code == "vs_target_preview":
            engine = build_vs_target_preview_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=VsDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                ),
            )
        elif block.drill_code == "vs_clean_scan":
            engine = build_vs_clean_scan_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=VsDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                ),
            )
        elif block.drill_code == "vs_family_run_letters":
            engine = build_vs_family_run_drill(
                clock=self._clock,
                seed=block_seed,
                kind=VisualSearchTaskKind.ALPHANUMERIC,
                difficulty=difficulty,
                mode=block.mode,
                config=VsDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                ),
            )
        elif block.drill_code == "vs_family_run_symbols":
            engine = build_vs_family_run_drill(
                clock=self._clock,
                seed=block_seed,
                kind=VisualSearchTaskKind.SYMBOL_CODE,
                difficulty=difficulty,
                mode=block.mode,
                config=VsDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                ),
            )
        elif block.drill_code == "vs_mixed_tempo":
            engine = build_vs_mixed_tempo_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=VsDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                ),
            )
        elif block.drill_code == "vs_pressure_run":
            engine = build_vs_pressure_run_drill(
                clock=self._clock,
                seed=block_seed,
                difficulty=difficulty,
                mode=block.mode,
                config=VsDrillConfig(
                    practice_questions=0,
                    scored_duration_s=block.duration_s,
                ),
            )
        else:
            raise ValueError(f"Unknown drill code: {block.drill_code}")

        engine.start_scored()
        self._current_engine = engine

    def _complete_current_block(self) -> None:
        if self._current_engine is None or self._current_block_plan is None:
            return
        self._record_current_block_result()
        self._current_engine = None
        self._current_block_plan = None
        self._prepare_next_block()

    def _record_current_block_result(self) -> None:
        if self._current_engine is None or self._current_block_plan is None:
            return
        block_plan = self._current_block_plan
        summary: AntDrillAttemptSummary = self._current_engine.scored_summary()
        base_index = len(self._events)
        for offset, event in enumerate(self._current_engine.events()):
            self._events.append(
                QuestionEvent(
                    index=base_index + offset,
                    phase=event.phase,
                    prompt=event.prompt,
                    correct_answer=event.correct_answer,
                    user_answer=event.user_answer,
                    is_correct=event.is_correct,
                    presented_at_s=event.presented_at_s,
                    answered_at_s=event.answered_at_s,
                    response_time_s=event.response_time_s,
                    raw=event.raw,
                    score=event.score,
                    max_score=event.max_score,
                )
            )

        self._block_results.append(
            AntWorkoutBlockResult(
                block_id=block_plan.block_id,
                label=block_plan.label,
                drill_code=block_plan.drill_code,
                mode=block_plan.mode.value,
                duration_s=summary.duration_s,
                attempted=summary.attempted,
                correct=summary.correct,
                total_score=summary.total_score,
                max_score=summary.max_score,
                score_ratio=summary.score_ratio,
                timeouts=summary.timeouts,
                fixation_rate=summary.fixation_rate,
                difficulty_level_start=summary.difficulty_level_start,
                difficulty_level_end=summary.difficulty_level_end,
                difficulty_change_count=summary.difficulty_change_count,
            )
        )
        self._last_finished_block_level = self._current_block_level

    def _max_timeout_streak(self) -> int:
        max_streak = 0
        cur = 0
        for event in self._events:
            if event.raw == "__timeout__":
                cur += 1
                max_streak = max(max_streak, cur)
            else:
                cur = 0
        return max_streak

    def _submit_current_engine_skip(self, tokens: tuple[str, ...]) -> bool:
        if self._current_engine is None:
            return False
        submit = getattr(self._current_engine, "submit_answer", None)
        if not callable(submit):
            return False
        for token in tokens:
            if submit(token):
                return True
        return False

    def _force_current_engine_phase(self, phase: Phase) -> None:
        engine = self._current_engine
        if engine is None or not hasattr(engine, "_phase"):
            return
        engine._phase = phase
        for attr in ("_current", "_presented_at_s"):
            if hasattr(engine, attr):
                setattr(engine, attr, None)
