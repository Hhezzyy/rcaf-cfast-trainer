from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum

from .ant_drills import AntDrillMode
from .ant_workouts import AntWorkoutBlockPlan, build_workout_block_engine
from .clock import Clock
from .cognitive_core import Phase
from .persistence import AttemptHistoryEntry
from .results import AttemptResult, attempt_result_from_engine
from .telemetry import TelemetryEvent, telemetry_analytics_from_events
from .training_modes import split_half_note_fragment, supports_training_mode

_SCHEDULER_VERSION = 1
_SESSION_DURATION_S = 60.0 * 60.0
_BLOCK_DURATION_S = 10.0 * 60.0
_MAX_EVIDENCE_TOTAL = 24
_MAX_EVIDENCE_PER_PRIMITIVE = 6
_EVIDENCE_LOOKBACK_DAYS = 28
_PROFILE_MULTIPLIERS = {
    "mental_arithmetic_automaticity": 1.35,
    "table_cross_reference_speed": 1.25,
    "visual_scan_discipline": 1.20,
    "symbolic_rule_extraction": 1.15,
    "tracking_stability_low_load": 1.00,
    "dual_task_stability_fatigue": 1.30,
}
_BLOCK_ROLE_LEVEL = {
    "anchor": 4,
    "reset": 4,
    "tempo": 5,
    "fatigue_pressure": 6,
}
def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_utc_iso(value: str) -> datetime:
    return datetime.strptime(str(value), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


def _epoch_s(value: str) -> float:
    return _parse_utc_iso(value).timestamp()


def _metric_float(metrics: dict[str, str], key: str) -> float | None:
    raw = metrics.get(key)
    if raw is None:
        return None
    token = str(raw).strip()
    if token == "":
        return None
    try:
        value = float(token)
    except Exception:
        return None
    if not math.isfinite(value):
        return None
    return float(value)


def _metric_bool(metrics: dict[str, str], key: str) -> bool:
    raw = str(metrics.get(key, "")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _ewma(values: Iterable[float], *, alpha: float = 0.55) -> float | None:
    current: float | None = None
    for value in values:
        if current is None:
            current = float(value)
            continue
        current = (float(alpha) * float(value)) + ((1.0 - float(alpha)) * current)
    return current


@dataclass(frozen=True, slots=True)
class AdaptivePrimitive:
    primitive_id: str
    label: str
    benchmark_probe_codes: tuple[str, ...]
    direct_test_codes: tuple[str, ...]
    coarse_workout_codes: tuple[str, ...]
    bottleneck_test_codes: tuple[str, ...]
    profile_multiplier: float
    anchor_templates: tuple[str, ...]
    tempo_templates: tuple[str, ...]
    reset_templates: tuple[str, ...]
    fatigue_templates: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AdaptiveEvidence:
    primitive_id: str
    source_code: str
    source_kind: str
    completed_at_utc: str
    completed_at_epoch_s: float
    performance_score: float | None
    fatigue_score: float | None
    post_error_score: float | None
    coarse: bool


@dataclass(frozen=True, slots=True)
class AdaptivePriorityBreakdown:
    primitive_id: str
    label: str
    profile_multiplier: float
    weakness: float
    fatigue: float
    post_error: float
    retention: float
    bottleneck: float
    priority: float
    evidence_count: int
    coarse_evidence_count: int
    reason_tags: tuple[str, ...]
    last_trained_at_utc: str | None
    last_source_code: str | None


@dataclass(frozen=True, slots=True)
class AdaptiveSessionBlock:
    block_index: int
    primitive_id: str
    primitive_label: str
    drill_code: str
    mode: str
    duration_s: float
    difficulty_level: int
    seed: int
    reason_tags: tuple[str, ...]
    priority: float
    drill_mode: AntDrillMode
    builder: Callable[[], object] | None = None


@dataclass(frozen=True, slots=True)
class AdaptiveSessionPlan:
    code: str
    title: str
    version: int
    generated_at_utc: str
    description: str
    notes: tuple[str, ...]
    ranked_primitives: tuple[AdaptivePriorityBreakdown, ...]
    blocks: tuple[AdaptiveSessionBlock, ...]

    @property
    def scored_duration_s(self) -> float:
        return float(sum(block.duration_s for block in self.blocks))


class AdaptiveStage(str, Enum):
    INTRO = "intro"
    BLOCK = "block"
    RESULTS = "results"


@dataclass(frozen=True, slots=True)
class AdaptiveBlockResult:
    block_index: int
    primitive_id: str
    primitive_label: str
    drill_code: str
    mode: str
    duration_s: float
    difficulty_level: int
    attempted: int
    correct: int
    accuracy: float
    throughput_per_min: float
    mean_rt_ms: float | None
    total_score: float | None
    max_score: float | None
    score_ratio: float | None
    reason_tags: tuple[str, ...]
    completed: bool


@dataclass(frozen=True, slots=True)
class AdaptiveSnapshot:
    stage: AdaptiveStage
    title: str
    subtitle: str
    prompt: str
    note_lines: tuple[str, ...]
    block_index: int
    block_total: int
    current_block_label: str | None
    current_primitive_label: str | None
    block_time_remaining_s: float | None
    session_time_remaining_s: float | None
    attempted_total: int
    correct_total: int
    completed_block_results: tuple[AdaptiveBlockResult, ...]


@dataclass(frozen=True, slots=True)
class AdaptiveSummary:
    attempted: int
    correct: int
    accuracy: float
    duration_s: float
    throughput_per_min: float
    mean_response_time_s: float | None
    total_score: float = 0.0
    max_score: float = 0.0
    score_ratio: float = 0.0
    difficulty_level: int = 5
    difficulty_level_start: int | None = None
    difficulty_level_end: int | None = None
    difficulty_change_count: int = 0
    block_count: int = 0
    completed_blocks: int = 0
    scheduler_version: int = _SCHEDULER_VERSION


_PRIMITIVES: tuple[AdaptivePrimitive, ...] = (
    AdaptivePrimitive(
        primitive_id="mental_arithmetic_automaticity",
        label="Mental Arithmetic Automaticity",
        benchmark_probe_codes=("numerical_operations",),
        direct_test_codes=(
            "numerical_operations",
            "ma_one_step_fluency",
            "ma_percentage_snap",
            "ma_rate_time_distance",
            "ma_fuel_endurance",
            "ma_mixed_conversion_caps",
            "no_fact_prime",
            "no_operator_ladders",
            "no_clean_compute",
            "no_mixed_tempo",
            "no_pressure_run",
            "mr_unit_relation_prime",
            "mr_one_step_solve",
            "ant_snap_facts_sprint",
            "ant_time_flip",
            "ant_route_time_solve",
            "ant_endurance_solve",
            "ant_fuel_burn_solve",
        ),
        coarse_workout_codes=("numerical_operations_workout",),
        bottleneck_test_codes=(
            "numerical_operations",
            "airborne_numerical",
            "math_reasoning",
            "colours_letters_numbers",
        ),
        profile_multiplier=_PROFILE_MULTIPLIERS["mental_arithmetic_automaticity"],
        anchor_templates=(
            "ma_one_step_fluency",
            "ma_percentage_snap",
            "ma_rate_time_distance",
            "ma_fuel_endurance",
            "no_fact_prime",
        ),
        tempo_templates=(
            "ma_percentage_snap",
            "ma_rate_time_distance",
            "ma_fuel_endurance",
            "ma_mixed_conversion_caps",
            "no_mixed_tempo",
        ),
        reset_templates=(
            "ma_one_step_fluency",
            "ma_mixed_conversion_caps",
            "ma_rate_time_distance",
            "ma_percentage_snap",
            "no_clean_compute",
        ),
        fatigue_templates=(
            "ma_fuel_endurance",
            "ma_mixed_conversion_caps",
            "ma_rate_time_distance",
            "ma_percentage_snap",
            "no_pressure_run",
        ),
    ),
    AdaptivePrimitive(
        primitive_id="table_cross_reference_speed",
        label="Table Cross-Reference Speed",
        benchmark_probe_codes=("table_reading",),
        direct_test_codes=(
            "table_reading",
            "tbl_single_lookup_anchor",
            "tbl_two_table_xref",
            "tbl_distractor_grid",
            "tbl_lookup_compute",
            "tbl_shrinking_cap_run",
            "tbl_part1_anchor",
            "tbl_part1_scan_run",
            "tbl_part2_prime",
            "tbl_part2_correction_run",
            "tbl_part_switch_run",
            "tbl_card_family_run",
            "tbl_mixed_tempo",
            "tbl_pressure_run",
        ),
        coarse_workout_codes=("table_reading_workout",),
        bottleneck_test_codes=("table_reading", "airborne_numerical", "system_logic"),
        profile_multiplier=_PROFILE_MULTIPLIERS["table_cross_reference_speed"],
        anchor_templates=(
            "tbl_single_lookup_anchor",
            "tbl_two_table_xref",
            "tbl_distractor_grid",
            "tbl_lookup_compute",
            "tbl_part1_anchor",
        ),
        tempo_templates=(
            "tbl_two_table_xref",
            "tbl_lookup_compute",
            "tbl_distractor_grid",
            "tbl_shrinking_cap_run",
            "tbl_mixed_tempo",
        ),
        reset_templates=(
            "tbl_lookup_compute",
            "tbl_distractor_grid",
            "tbl_two_table_xref",
            "tbl_shrinking_cap_run",
            "tbl_part2_correction_run",
        ),
        fatigue_templates=(
            "tbl_shrinking_cap_run",
            "tbl_lookup_compute",
            "tbl_distractor_grid",
            "tbl_two_table_xref",
            "tbl_pressure_run",
        ),
    ),
    AdaptivePrimitive(
        primitive_id="visual_scan_discipline",
        label="Visual Scan Discipline",
        benchmark_probe_codes=("visual_search",),
        direct_test_codes=(
            "visual_search",
            "vs_target_preview",
            "vs_clean_scan",
            "vs_family_run_letters",
            "vs_family_run_symbols",
            "vs_mixed_tempo",
            "vs_pressure_run",
        ),
        coarse_workout_codes=("visual_search_workout",),
        bottleneck_test_codes=("visual_search", "vigilance", "target_recognition"),
        profile_multiplier=_PROFILE_MULTIPLIERS["visual_scan_discipline"],
        anchor_templates=("vs_target_preview", "vs_clean_scan"),
        tempo_templates=("__vs_family_run__", "vs_mixed_tempo"),
        reset_templates=("vs_clean_scan", "__vs_family_run__"),
        fatigue_templates=("vs_pressure_run",),
    ),
    AdaptivePrimitive(
        primitive_id="symbolic_rule_extraction",
        label="Symbolic Rule Extraction",
        benchmark_probe_codes=("sl_graph_rule_anchor",),
        direct_test_codes=(
            "system_logic",
            "sl_one_rule_identify",
            "sl_missing_step_complete",
            "sl_two_source_reconcile",
            "sl_rule_match",
            "sl_fast_reject",
            "sl_quantitative_anchor",
            "sl_flow_trace_anchor",
            "sl_graph_rule_anchor",
            "sl_fault_diagnosis_prime",
            "sl_index_switch_run",
            "sl_family_run",
            "sl_mixed_tempo",
            "sl_pressure_run",
        ),
        coarse_workout_codes=("system_logic_workout",),
        bottleneck_test_codes=("system_logic", "table_reading"),
        profile_multiplier=_PROFILE_MULTIPLIERS["symbolic_rule_extraction"],
        anchor_templates=(
            "sl_one_rule_identify",
            "sl_rule_match",
            "sl_two_source_reconcile",
            "sl_missing_step_complete",
            "sl_graph_rule_anchor",
        ),
        tempo_templates=(
            "sl_two_source_reconcile",
            "sl_rule_match",
            "sl_missing_step_complete",
            "sl_fast_reject",
            "sl_mixed_tempo",
        ),
        reset_templates=(
            "sl_missing_step_complete",
            "sl_fast_reject",
            "sl_rule_match",
            "sl_two_source_reconcile",
            "sl_index_switch_run",
        ),
        fatigue_templates=(
            "sl_fast_reject",
            "sl_rule_match",
            "sl_two_source_reconcile",
            "sl_missing_step_complete",
            "sl_pressure_run",
        ),
    ),
    AdaptivePrimitive(
        primitive_id="tracking_stability_low_load",
        label="Tracking Stability Under Low Load",
        benchmark_probe_codes=("rt_lock_anchor",),
        direct_test_codes=(
            "rapid_tracking",
            "rt_lock_anchor",
            "rt_building_handoff_prime",
            "rt_terrain_recovery_run",
            "rt_capture_timing_prime",
            "rt_ground_tempo_run",
            "rt_air_speed_run",
            "rt_mixed_tempo",
            "rt_pressure_run",
        ),
        coarse_workout_codes=("rapid_tracking_workout",),
        bottleneck_test_codes=("rapid_tracking",),
        profile_multiplier=_PROFILE_MULTIPLIERS["tracking_stability_low_load"],
        anchor_templates=("rt_lock_anchor",),
        tempo_templates=("rt_ground_tempo_run", "rt_mixed_tempo"),
        reset_templates=("rt_ground_tempo_run",),
        fatigue_templates=("rt_pressure_run",),
    ),
    AdaptivePrimitive(
        primitive_id="dual_task_stability_fatigue",
        label="Dual-Task Stability Under Fatigue",
        benchmark_probe_codes=("cln_sequence_math_recall",),
        direct_test_codes=(
            "colours_letters_numbers",
            "dtb_tracking_recall",
            "dtb_tracking_command_filter",
            "dtb_tracking_filter_digit_report",
            "dtb_tracking_interference_recovery",
            "cln_sequence_copy",
            "cln_sequence_match",
            "cln_math_prime",
            "cln_colour_lane",
            "cln_memory_math",
            "cln_memory_colour",
            "cln_full_steady",
            "cln_full_pressure",
            "cln_sequence_math_recall",
        ),
        coarse_workout_codes=("colours_letters_numbers_workout",),
        bottleneck_test_codes=(
            "colours_letters_numbers",
            "auditory_capacity",
            "cognitive_updating",
            "situational_awareness",
        ),
        profile_multiplier=_PROFILE_MULTIPLIERS["dual_task_stability_fatigue"],
        anchor_templates=(
            "dtb_tracking_recall",
            "dtb_tracking_command_filter",
            "dtb_tracking_filter_digit_report",
            "dtb_tracking_interference_recovery",
            "cln_memory_math",
        ),
        tempo_templates=(
            "dtb_tracking_command_filter",
            "dtb_tracking_filter_digit_report",
            "dtb_tracking_recall",
            "dtb_tracking_interference_recovery",
            "cln_full_steady",
        ),
        reset_templates=(
            "dtb_tracking_recall",
            "dtb_tracking_command_filter",
            "dtb_tracking_filter_digit_report",
            "dtb_tracking_interference_recovery",
            "cln_memory_math",
        ),
        fatigue_templates=(
            "dtb_tracking_interference_recovery",
            "dtb_tracking_filter_digit_report",
            "dtb_tracking_command_filter",
            "dtb_tracking_recall",
            "cln_full_pressure",
        ),
    ),
)
_PRIMITIVE_BY_ID = {primitive.primitive_id: primitive for primitive in _PRIMITIVES}
_DIRECT_CODE_TO_PRIMITIVE = {
    code: primitive.primitive_id
    for primitive in _PRIMITIVES
    for code in primitive.direct_test_codes
}
_WORKOUT_CODE_TO_PRIMITIVE = {
    code: primitive.primitive_id
    for primitive in _PRIMITIVES
    for code in primitive.coarse_workout_codes
}
_BENCHMARK_PROBE_TO_PRIMITIVE = {
    code: primitive.primitive_id
    for primitive in _PRIMITIVES
    for code in primitive.benchmark_probe_codes
}
_MAX_BOTTLENECK_COUNT = max(len(primitive.bottleneck_test_codes) for primitive in _PRIMITIVES)


def collect_adaptive_evidence(
    history: list[AttemptHistoryEntry],
    *,
    now_utc: str | None = None,
) -> list[AdaptiveEvidence]:
    _ = now_utc
    mapped: list[AdaptiveEvidence] = []
    for entry in history:
        mapped.extend(_evidence_from_attempt(entry))

    mapped.sort(key=lambda evidence: (evidence.completed_at_epoch_s, evidence.source_code), reverse=True)
    per_primitive_counts: dict[str, int] = defaultdict(int)
    limited: list[AdaptiveEvidence] = []
    for evidence in mapped:
        if len(limited) >= _MAX_EVIDENCE_TOTAL:
            break
        if per_primitive_counts[evidence.primitive_id] >= _MAX_EVIDENCE_PER_PRIMITIVE:
            continue
        limited.append(evidence)
        per_primitive_counts[evidence.primitive_id] += 1
    return limited


def rank_adaptive_primitives(
    history: list[AttemptHistoryEntry],
    *,
    now_utc: str | None = None,
) -> tuple[AdaptivePriorityBreakdown, ...]:
    now_token = now_utc or _utc_now_iso()
    now_epoch = _epoch_s(now_token)
    evidence = collect_adaptive_evidence(history, now_utc=now_token)
    grouped: dict[str, list[AdaptiveEvidence]] = {primitive.primitive_id: [] for primitive in _PRIMITIVES}
    for item in evidence:
        grouped[item.primitive_id].append(item)

    ranked: list[AdaptivePriorityBreakdown] = []
    for primitive in _PRIMITIVES:
        items = sorted(grouped.get(primitive.primitive_id, ()), key=lambda item: item.completed_at_epoch_s)
        performance_values = [
            float(item.performance_score)
            for item in items
            if item.performance_score is not None
        ]
        weakness = 0.35
        perf_ewma = _ewma(performance_values)
        if perf_ewma is not None:
            weakness = _clamp(1.0 - perf_ewma)

        fatigue_values = [float(item.fatigue_score) for item in items if item.fatigue_score is not None]
        fatigue = 0.0 if not fatigue_values else _clamp(_ewma(fatigue_values) or 0.0)

        post_error_values = [
            float(item.post_error_score)
            for item in items
            if item.post_error_score is not None
        ]
        post_error = 0.0 if not post_error_values else _clamp(_ewma(post_error_values) or 0.0)

        retention = _retention_score(items, now_epoch=now_epoch)
        bottleneck = float(len(primitive.bottleneck_test_codes)) / float(_MAX_BOTTLENECK_COUNT)
        priority = (
            (0.35 * weakness)
            + (0.20 * fatigue)
            + (0.15 * post_error)
            + (0.20 * retention)
            + (0.10 * bottleneck)
        ) * float(primitive.profile_multiplier)
        last_item = items[-1] if items else None
        coarse_count = sum(1 for item in items if item.coarse)

        contributions = (
            ("weak", 0.35 * weakness),
            ("fatigue", 0.20 * fatigue),
            ("post-error", 0.15 * post_error),
            ("retention", 0.20 * retention),
            ("bottleneck", 0.10 * bottleneck),
        )
        ordered_reasons = sorted(contributions, key=lambda pair: (pair[1], pair[0]), reverse=True)
        reason_tags = tuple(
            tag
            for tag, weight in ordered_reasons
            if weight >= 0.05
        )[:3]
        if not reason_tags:
            reason_tags = tuple(tag for tag, _weight in ordered_reasons[:2])

        ranked.append(
            AdaptivePriorityBreakdown(
                primitive_id=primitive.primitive_id,
                label=primitive.label,
                profile_multiplier=float(primitive.profile_multiplier),
                weakness=float(weakness),
                fatigue=float(fatigue),
                post_error=float(post_error),
                retention=float(retention),
                bottleneck=float(bottleneck),
                priority=float(priority),
                evidence_count=len(items),
                coarse_evidence_count=coarse_count,
                reason_tags=reason_tags,
                last_trained_at_utc=None if last_item is None else last_item.completed_at_utc,
                last_source_code=None if last_item is None else last_item.source_code,
            )
        )

    ranked.sort(key=lambda item: (item.priority, item.evidence_count, item.primitive_id), reverse=True)
    return tuple(ranked)


def build_adaptive_session_plan(
    *,
    history: list[AttemptHistoryEntry],
    seed: int,
    now_utc: str | None = None,
) -> AdaptiveSessionPlan | None:
    now_token = now_utc or _utc_now_iso()
    ranked = rank_adaptive_primitives(history, now_utc=now_token)
    if not any(item.evidence_count > 0 for item in ranked):
        return None

    selected = list(ranked[:3])
    if len(selected) == 1:
        allocations = [6]
    elif len(selected) == 2:
        allocations = [4, 2]
    elif selected[2].priority < 0.35:
        selected = selected[:2]
        allocations = [4, 2]
    else:
        allocations = [3, 2, 1]

    blocks: list[AdaptiveSessionBlock] = []
    for primitive_index, (breakdown, block_count) in enumerate(zip(selected, allocations, strict=True)):
        primitive = _PRIMITIVE_BY_ID[breakdown.primitive_id]
        for local_index in range(block_count):
            role = _select_block_role(
                breakdown=breakdown,
                local_index=local_index,
                block_count=block_count,
            )
            drill_code = _select_drill_code(
                primitive=primitive,
                role=role,
                local_index=local_index,
                breakdown=breakdown,
            )
            block_index = len(blocks)
            blocks.append(
                AdaptiveSessionBlock(
                    block_index=block_index,
                    primitive_id=primitive.primitive_id,
                    primitive_label=primitive.label,
                    drill_code=drill_code,
                    mode=role,
                    duration_s=_BLOCK_DURATION_S,
                    difficulty_level=_BLOCK_ROLE_LEVEL[role],
                    seed=int(seed) + ((block_index + 1) * 131),
                    reason_tags=tuple(breakdown.reason_tags),
                    priority=float(breakdown.priority),
                    drill_mode=_training_mode_for_role(drill_code, role=role),
                )
            )

    title = "Adaptive Session (60m)"
    description = (
        "Built from recent benchmark and training history. "
        "This session concentrates on the top weak primitives instead of spreading work evenly."
    )
    notes = tuple(
        f"{item.label}: priority {item.priority:.2f} [{', '.join(item.reason_tags)}]"
        for item in selected
    )
    return AdaptiveSessionPlan(
        code="adaptive_session",
        title=title,
        version=_SCHEDULER_VERSION,
        generated_at_utc=now_token,
        description=description,
        notes=notes,
        ranked_primitives=tuple(ranked),
        blocks=tuple(blocks),
    )


class AdaptiveSession:
    _NOMINAL_DIFFICULTY = 0.5

    def __init__(self, *, clock: Clock, seed: int, plan: AdaptiveSessionPlan) -> None:
        self._clock = clock
        self._seed = int(seed)
        self._plan = plan
        self._stage = AdaptiveStage.INTRO
        self._current_block_index = 0
        self._current_engine: object | None = None
        self._completed_attempts: list[AttemptResult] = []

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def difficulty(self) -> float:
        return float(self._NOMINAL_DIFFICULTY)

    @property
    def practice_questions(self) -> int:
        return 0

    @property
    def scored_duration_s(self) -> float:
        return self._plan.scored_duration_s

    @property
    def phase(self) -> Phase:
        if self._stage is AdaptiveStage.RESULTS:
            return Phase.RESULTS
        if self._stage is AdaptiveStage.BLOCK:
            return Phase.SCORED
        return Phase.INSTRUCTIONS

    @property
    def stage(self) -> AdaptiveStage:
        return self._stage

    def can_exit(self) -> bool:
        return self._stage in (AdaptiveStage.INTRO, AdaptiveStage.RESULTS)

    def current_engine(self) -> object | None:
        return self._current_engine

    def current_block_plan(self) -> AdaptiveSessionBlock | None:
        if self._stage is not AdaptiveStage.BLOCK:
            return None
        if not (0 <= self._current_block_index < len(self._plan.blocks)):
            return None
        return self._plan.blocks[self._current_block_index]

    def activate(self) -> None:
        if self._stage is not AdaptiveStage.INTRO:
            return
        if not self._plan.blocks:
            self._stage = AdaptiveStage.RESULTS
            return
        self._start_block(0)

    def start_practice(self) -> None:
        self.activate()

    def start_scored(self) -> None:
        self.activate()

    def submit_answer(self, raw: str) -> bool:
        engine = self._current_engine
        if self._stage is not AdaptiveStage.BLOCK or engine is None:
            return False
        submit = getattr(engine, "submit_answer", None)
        if not callable(submit):
            return False
        return bool(submit(raw))

    def update(self) -> None:
        if self._stage is not AdaptiveStage.BLOCK or self._current_engine is None:
            return
        updater = getattr(self._current_engine, "update", None)
        if callable(updater):
            updater()
        self.sync_runtime()

    def sync_runtime(self) -> None:
        if self._stage is not AdaptiveStage.BLOCK or self._current_engine is None:
            return
        if getattr(self._current_engine, "phase", None) is not Phase.RESULTS:
            return
        self._complete_current_block()

    def snapshot(self) -> AdaptiveSnapshot:
        completed_block_results = tuple(self._completed_block_results())
        summary = self.scored_summary()
        if self._stage is AdaptiveStage.INTRO:
            return AdaptiveSnapshot(
                stage=self._stage,
                title=self._plan.title,
                subtitle="Adaptive Primitive Scheduler",
                prompt=self._plan.description,
                note_lines=self._intro_note_lines(),
                block_index=0,
                block_total=len(self._plan.blocks),
                current_block_label=None,
                current_primitive_label=None,
                block_time_remaining_s=None,
                session_time_remaining_s=self._plan.scored_duration_s,
                attempted_total=summary.attempted,
                correct_total=summary.correct,
                completed_block_results=completed_block_results,
            )

        if self._stage is AdaptiveStage.RESULTS:
            return AdaptiveSnapshot(
                stage=self._stage,
                title=self._plan.title,
                subtitle="Adaptive Session Results",
                prompt=(
                    "Adaptive session complete.\n"
                    f"Attempted: {summary.attempted}\n"
                    f"Correct: {summary.correct}\n"
                    f"Accuracy: {summary.accuracy * 100.0:.1f}%\n"
                    f"Score: {summary.total_score:.1f}/{summary.max_score:.1f}"
                ),
                note_lines=self._results_note_lines(summary),
                block_index=len(self._plan.blocks),
                block_total=len(self._plan.blocks),
                current_block_label=None,
                current_primitive_label=None,
                block_time_remaining_s=0.0,
                session_time_remaining_s=0.0,
                attempted_total=summary.attempted,
                correct_total=summary.correct,
                completed_block_results=completed_block_results,
            )

        block = self.current_block_plan()
        return AdaptiveSnapshot(
            stage=self._stage,
            title=self._plan.title,
            subtitle=(
                f"Block {self._current_block_index + 1}/{len(self._plan.blocks)}"
                f": {'' if block is None else block.primitive_label}"
            ),
            prompt="",
            note_lines=(
                "Session-wide pause and restart only.",
                "This block was chosen from recent weakness, fatigue, retention, and post-error data.",
            ),
            block_index=self._current_block_index + 1,
            block_total=len(self._plan.blocks),
            current_block_label=None if block is None else block.drill_code,
            current_primitive_label=None if block is None else block.primitive_label,
            block_time_remaining_s=self._current_block_time_remaining_s(),
            session_time_remaining_s=self._session_time_remaining_s(),
            attempted_total=summary.attempted,
            correct_total=summary.correct,
            completed_block_results=completed_block_results,
        )

    def scored_summary(self) -> AdaptiveSummary:
        block_attempts = self._block_attempts(include_partial=True)
        attempted = sum(result.attempted for _block, result, _completed in block_attempts)
        correct = sum(result.correct for _block, result, _completed in block_attempts)
        accuracy = 0.0 if attempted <= 0 else float(correct) / float(attempted)
        duration_s = sum(result.duration_s for _block, result, _completed in block_attempts)
        throughput = 0.0 if duration_s <= 0.0 else (float(attempted) / float(duration_s)) * 60.0
        total_score = sum(float(result.total_score or 0.0) for _block, result, _completed in block_attempts)
        max_score = sum(float(result.max_score or 0.0) for _block, result, _completed in block_attempts)
        score_ratio = 0.0 if max_score <= 0.0 else total_score / max_score

        analytics = self._aggregate_analytics()
        available_blocks = [block for block, _result, _completed in block_attempts]
        difficulty_level_start = None if not available_blocks else available_blocks[0].difficulty_level
        difficulty_level_end = None if not available_blocks else available_blocks[-1].difficulty_level
        difficulty_change_count = 0
        previous_end = None
        for _block, result, _completed in block_attempts:
            try:
                difficulty_change_count += int(result.metrics.get("difficulty_change_count", "0") or "0")
            except Exception:
                pass
            if previous_end is not None and result.difficulty_level_start is not None:
                if previous_end != result.difficulty_level_start:
                    difficulty_change_count += 1
            previous_end = result.difficulty_level_end

        return AdaptiveSummary(
            attempted=attempted,
            correct=correct,
            accuracy=float(accuracy),
            duration_s=float(duration_s),
            throughput_per_min=float(throughput),
            mean_response_time_s=(
                None if analytics.mean_rt_ms is None else float(analytics.mean_rt_ms) / 1000.0
            ),
            total_score=float(total_score),
            max_score=float(max_score),
            score_ratio=float(score_ratio),
            difficulty_level=5,
            difficulty_level_start=difficulty_level_start,
            difficulty_level_end=difficulty_level_end,
            difficulty_change_count=difficulty_change_count,
            block_count=len(self._plan.blocks),
            completed_blocks=len(self._completed_attempts),
            scheduler_version=self._plan.version,
        )

    def result_metrics(self) -> dict[str, str]:
        metrics = {
            "scheduler.version": str(int(self._plan.version)),
            "scheduler.generated_at_utc": self._plan.generated_at_utc,
            "scheduler.block_count": str(len(self._plan.blocks)),
            "scheduler.total_duration_s": f"{self._plan.scored_duration_s:.6f}",
        }
        for breakdown in self._plan.ranked_primitives:
            metrics[f"scheduler.priority.{breakdown.primitive_id}"] = f"{breakdown.priority:.6f}"
            metrics[f"scheduler.reason.{breakdown.primitive_id}"] = ",".join(breakdown.reason_tags)

        attempt_lookup = {
            block.block_index: (block, result, completed)
            for block, result, completed in self._block_attempts(include_partial=True)
        }
        for block in self._plan.blocks:
            prefix = f"block.{block.block_index + 1:02d}."
            metrics[f"{prefix}primitive_id"] = block.primitive_id
            metrics[f"{prefix}drill_code"] = block.drill_code
            metrics[f"{prefix}mode"] = block.mode
            metrics[f"{prefix}training_mode"] = block.drill_mode.value
            metrics[f"{prefix}duration_s"] = f"{block.duration_s:.6f}"
            metrics[f"{prefix}difficulty_level"] = str(int(block.difficulty_level))
            metrics[f"{prefix}reason_tags"] = ",".join(block.reason_tags)
            metrics[f"{prefix}priority"] = f"{block.priority:.6f}"
            saved = attempt_lookup.get(block.block_index)
            metrics[f"{prefix}completed"] = "1" if saved and saved[2] else "0"
            if not saved:
                continue
            _saved_block, result, _completed = saved
            for key, value in result.metrics.items():
                metrics[f"{prefix}{key}"] = str(value)
        return metrics

    def events(self) -> list[TelemetryEvent]:
        events: list[TelemetryEvent] = []
        item_offset = 0
        for block, result, completed in self._block_attempts(include_partial=True):
            offset_ms = self._block_offset_ms(block.block_index)
            block_extra = {
                "block_index": int(block.block_index + 1),
                "primitive_id": block.primitive_id,
                "drill_code": block.drill_code,
                "mode": block.mode,
                "difficulty_level": int(block.difficulty_level),
            }
            events.append(
                TelemetryEvent(
                    family="adaptive_session",
                    kind="block_started",
                    phase=Phase.SCORED.value,
                    seq=len(events),
                    item_index=None,
                    is_scored=False,
                    is_correct=None,
                    is_timeout=False,
                    response_time_ms=None,
                    score=None,
                    max_score=None,
                    difficulty_level=int(block.difficulty_level),
                    occurred_at_ms=offset_ms,
                    prompt=block.primitive_label,
                    extra=block_extra,
                )
            )
            local_item_map = _local_item_index_map(result=result, item_offset=item_offset)
            item_offset += len(local_item_map)
            last_occurred_ms = offset_ms
            for inner_event in result.events:
                local_extra = dict(inner_event.extra or {})
                local_extra.update(block_extra)
                if inner_event.item_index is not None:
                    local_extra["inner_item_index"] = int(inner_event.item_index)
                occurred_at_ms = None
                if inner_event.occurred_at_ms is not None:
                    occurred_at_ms = offset_ms + int(inner_event.occurred_at_ms)
                    last_occurred_ms = max(last_occurred_ms, occurred_at_ms)
                mapped_item_index = None
                if inner_event.item_index is not None:
                    mapped_item_index = local_item_map.get(int(inner_event.item_index))
                events.append(
                    TelemetryEvent(
                        family=inner_event.family,
                        kind=inner_event.kind,
                        phase=inner_event.phase,
                        seq=len(events),
                        item_index=mapped_item_index,
                        is_scored=inner_event.is_scored,
                        is_correct=inner_event.is_correct,
                        is_timeout=inner_event.is_timeout,
                        response_time_ms=inner_event.response_time_ms,
                        score=inner_event.score,
                        max_score=inner_event.max_score,
                        difficulty_level=(
                            inner_event.difficulty_level
                            if inner_event.difficulty_level is not None
                            else int(block.difficulty_level)
                        ),
                        occurred_at_ms=occurred_at_ms,
                        prompt=inner_event.prompt,
                        expected=inner_event.expected,
                        response=inner_event.response,
                        extra=local_extra or None,
                    )
                )

            if completed:
                completed_at_ms = max(
                    last_occurred_ms,
                    offset_ms + int(round(float(block.duration_s) * 1000.0)),
                )
                extra = dict(block_extra)
                extra["attempted"] = int(result.attempted)
                extra["accuracy"] = float(result.accuracy)
                events.append(
                    TelemetryEvent(
                        family="adaptive_session",
                        kind="block_completed",
                        phase=Phase.SCORED.value,
                        seq=len(events),
                        item_index=None,
                        is_scored=False,
                        is_correct=None,
                        is_timeout=False,
                        response_time_ms=None,
                        score=result.total_score,
                        max_score=result.max_score,
                        difficulty_level=int(block.difficulty_level),
                        occurred_at_ms=completed_at_ms,
                        prompt=block.primitive_label,
                        extra=extra,
                    )
                )
        return events

    def _start_block(self, index: int) -> None:
        self._current_block_index = int(index)
        block = self._plan.blocks[self._current_block_index]
        self._current_engine = self._build_block_engine(block)
        self._stage = AdaptiveStage.BLOCK

    def _build_block_engine(self, block: AdaptiveSessionBlock) -> object:
        if block.builder is not None:
            engine = block.builder()
            starter = getattr(engine, "start_scored", None)
            if callable(starter):
                starter()
            return engine
        workout_block = AntWorkoutBlockPlan(
            block_id=f"adaptive_{block.block_index + 1:02d}",
            label=block.primitive_label,
            description=f"{block.primitive_label} [{', '.join(block.reason_tags)}]",
            focus_skills=(block.primitive_label,),
            drill_code=block.drill_code,
            mode=block.drill_mode,
            duration_min=float(block.duration_s) / 60.0,
        )
        return build_workout_block_engine(
            clock=self._clock,
            block_seed=block.seed,
            difficulty_level=block.difficulty_level,
            block=workout_block,
            block_index=0,
        )

    def _complete_current_block(self) -> None:
        block = self.current_block_plan()
        engine = self._current_engine
        if block is None or engine is None:
            return
        self._completed_attempts.append(
            attempt_result_from_engine(engine, test_code=block.drill_code)
        )
        next_index = self._current_block_index + 1
        self._current_engine = None
        if next_index >= len(self._plan.blocks):
            self._stage = AdaptiveStage.RESULTS
            self._current_block_index = len(self._plan.blocks)
            return
        self._start_block(next_index)

    def _block_attempts(
        self,
        *,
        include_partial: bool,
    ) -> list[tuple[AdaptiveSessionBlock, AttemptResult, bool]]:
        attempts = [
            (self._plan.blocks[index], result, True)
            for index, result in enumerate(self._completed_attempts)
        ]
        if include_partial and self._stage is AdaptiveStage.BLOCK and self._current_engine is not None:
            attempts.append(
                (
                    self._plan.blocks[self._current_block_index],
                    attempt_result_from_engine(
                        self._current_engine,
                        test_code=self._plan.blocks[self._current_block_index].drill_code,
                    ),
                    False,
                )
            )
        return attempts

    def _completed_block_results(self) -> list[AdaptiveBlockResult]:
        out: list[AdaptiveBlockResult] = []
        for block, result, completed in self._block_attempts(include_partial=True):
            out.append(
                AdaptiveBlockResult(
                    block_index=block.block_index,
                    primitive_id=block.primitive_id,
                    primitive_label=block.primitive_label,
                    drill_code=block.drill_code,
                    mode=block.mode,
                    duration_s=result.duration_s,
                    difficulty_level=block.difficulty_level,
                    attempted=result.attempted,
                    correct=result.correct,
                    accuracy=result.accuracy,
                    throughput_per_min=result.throughput_per_min,
                    mean_rt_ms=result.mean_rt_ms,
                    total_score=result.total_score,
                    max_score=result.max_score,
                    score_ratio=result.score_ratio,
                    reason_tags=block.reason_tags,
                    completed=completed,
                )
            )
        return out

    def _current_block_time_remaining_s(self) -> float | None:
        if self._stage is not AdaptiveStage.BLOCK:
            return 0.0 if self._stage is AdaptiveStage.RESULTS else None
        snap = _engine_snapshot(self._current_engine)
        value = getattr(snap, "time_remaining_s", None)
        try:
            if value is not None:
                return float(value)
        except Exception:
            pass
        block = self.current_block_plan()
        return None if block is None else float(block.duration_s)

    def _session_time_remaining_s(self) -> float | None:
        if self._stage is AdaptiveStage.INTRO:
            return self._plan.scored_duration_s
        if self._stage is AdaptiveStage.RESULTS:
            return 0.0
        current_remaining = self._current_block_time_remaining_s() or 0.0
        future = sum(block.duration_s for block in self._plan.blocks[self._current_block_index + 1 :])
        return float(current_remaining + future)

    def _aggregate_analytics(self):
        summary = self.scored_summary_base()
        return telemetry_analytics_from_events(
            self.events(),
            duration_s=summary.duration_s,
            is_complete=self._stage is AdaptiveStage.RESULTS,
            difficulty_level_start=summary.difficulty_level_start,
            difficulty_level_end=summary.difficulty_level_end,
            difficulty_change_count=summary.difficulty_change_count,
        )

    def scored_summary_base(self) -> AdaptiveSummary:
        block_attempts = self._block_attempts(include_partial=True)
        attempted = sum(result.attempted for _block, result, _completed in block_attempts)
        correct = sum(result.correct for _block, result, _completed in block_attempts)
        duration_s = sum(result.duration_s for _block, result, _completed in block_attempts)
        total_score = sum(float(result.total_score or 0.0) for _block, result, _completed in block_attempts)
        max_score = sum(float(result.max_score or 0.0) for _block, result, _completed in block_attempts)
        difficulty_level_start = None if not block_attempts else block_attempts[0][0].difficulty_level
        difficulty_level_end = None if not block_attempts else block_attempts[-1][0].difficulty_level
        difficulty_change_count = sum(
            int(result.metrics.get("difficulty_change_count", "0") or "0")
            for _block, result, _completed in block_attempts
        )
        return AdaptiveSummary(
            attempted=attempted,
            correct=correct,
            accuracy=0.0 if attempted <= 0 else float(correct) / float(attempted),
            duration_s=float(duration_s),
            throughput_per_min=0.0 if duration_s <= 0.0 else (float(attempted) / float(duration_s)) * 60.0,
            mean_response_time_s=None,
            total_score=float(total_score),
            max_score=float(max_score),
            score_ratio=0.0 if max_score <= 0.0 else float(total_score) / float(max_score),
            difficulty_level=5,
            difficulty_level_start=difficulty_level_start,
            difficulty_level_end=difficulty_level_end,
            difficulty_change_count=difficulty_change_count,
            block_count=len(self._plan.blocks),
            completed_blocks=len(self._completed_attempts),
            scheduler_version=self._plan.version,
        )

    def _block_offset_ms(self, block_index: int) -> int:
        total = 0.0
        for block in self._plan.blocks[:block_index]:
            total += float(block.duration_s)
        return int(round(total * 1000.0))

    def _intro_note_lines(self) -> tuple[str, ...]:
        lines = [
            "The scheduler recomputes on open from the last 28 days of mapped evidence.",
            "Allocation: top primitive 3 blocks, second 2, third 1 unless the third is too weakly supported.",
            "",
            "Ranked primitives:",
        ]
        for breakdown in self._plan.ranked_primitives[:3]:
            lines.append(
                f"{breakdown.label}: {breakdown.priority:.2f} [{', '.join(breakdown.reason_tags)}]"
            )
        lines.append("")
        lines.append("Planned blocks:")
        for block in self._plan.blocks:
            lines.append(
                f"{block.block_index + 1}. {block.primitive_label} -> {block.drill_code} ({block.mode})"
            )
        return tuple(lines)

    def _results_note_lines(self, summary: AdaptiveSummary) -> tuple[str, ...]:
        lines = [
            f"Blocks completed: {summary.completed_blocks}/{summary.block_count}",
            f"Score ratio: {summary.score_ratio * 100.0:.1f}%",
            f"Difficulty range: {summary.difficulty_level_start or '-'} to {summary.difficulty_level_end or '-'}",
            "Block splits:",
        ]
        for block, result, completed in self._block_attempts(include_partial=False):
            if not completed:
                continue
            split_fragment = split_half_note_fragment(result.metrics)
            if split_fragment is None:
                continue
            lines.append(f"{block.block_index + 1}. {block.primitive_label}: {split_fragment}")
        lines.append("Priority drivers:")
        for breakdown in self._plan.ranked_primitives[:3]:
            lines.append(
                f"{breakdown.label}: weak {breakdown.weakness:.2f}, fatigue {breakdown.fatigue:.2f}, "
                f"post-error {breakdown.post_error:.2f}, retention {breakdown.retention:.2f}"
            )
        return tuple(lines)


def _training_mode_for_role(drill_code: str, *, role: str) -> AntDrillMode:
    if role == "anchor":
        if supports_training_mode(drill_code, AntDrillMode.FRESH):
            return AntDrillMode.FRESH
        return AntDrillMode.BUILD
    if role == "reset":
        return AntDrillMode.RECOVERY
    if role == "fatigue_pressure":
        if supports_training_mode(drill_code, AntDrillMode.FATIGUE_PROBE):
            return AntDrillMode.FATIGUE_PROBE
        return AntDrillMode.PRESSURE
    return AntDrillMode.PRESSURE


def _select_block_role(
    *,
    breakdown: AdaptivePriorityBreakdown,
    local_index: int,
    block_count: int,
) -> str:
    last_trained_age_h = _hours_since(breakdown.last_trained_at_utc)
    high_retention = breakdown.retention >= 0.45 or last_trained_age_h >= 48.0
    high_post_error = breakdown.post_error >= 0.45
    high_fatigue = breakdown.fatigue >= 0.35
    if local_index == 0 and high_retention:
        return "anchor"
    if local_index == (block_count - 1) and high_fatigue:
        return "fatigue_pressure"
    if local_index > 0 and high_post_error:
        return "reset"
    return "tempo"


def _select_drill_code(
    *,
    primitive: AdaptivePrimitive,
    role: str,
    local_index: int,
    breakdown: AdaptivePriorityBreakdown,
) -> str:
    if role == "anchor":
        templates = primitive.anchor_templates
    elif role == "reset":
        templates = primitive.reset_templates
    elif role == "fatigue_pressure":
        templates = primitive.fatigue_templates
    else:
        templates = primitive.tempo_templates
    if not templates:
        raise RuntimeError(f"primitive {primitive.primitive_id} has no templates for role {role}")
    selected = templates[min(local_index, len(templates) - 1)]
    if selected == "__vs_family_run__":
        last = breakdown.last_source_code or ""
        if last == "vs_family_run_letters":
            return "vs_family_run_symbols"
        return "vs_family_run_letters"
    return selected


def _evidence_from_attempt(entry: AttemptHistoryEntry) -> list[AdaptiveEvidence]:
    if entry.test_code == "benchmark_battery":
        return _benchmark_evidence_from_attempt(entry)
    if entry.test_code == "adaptive_session":
        return _adaptive_block_evidence_from_attempt(entry)

    direct_primitive = _DIRECT_CODE_TO_PRIMITIVE.get(entry.test_code)
    if direct_primitive is not None:
        evidence = _build_evidence(
            primitive_id=direct_primitive,
            source_code=entry.test_code,
            source_kind="direct",
            completed_at_utc=entry.completed_at_utc,
            metrics=entry.metrics,
            coarse=False,
        )
        return [] if evidence is None else [evidence]

    coarse_primitive = _WORKOUT_CODE_TO_PRIMITIVE.get(entry.test_code)
    if coarse_primitive is not None:
        evidence = _build_evidence(
            primitive_id=coarse_primitive,
            source_code=entry.test_code,
            source_kind="coarse_workout",
            completed_at_utc=entry.completed_at_utc,
            metrics=entry.metrics,
            coarse=True,
        )
        return [] if evidence is None else [evidence]
    return []


def _benchmark_evidence_from_attempt(entry: AttemptHistoryEntry) -> list[AdaptiveEvidence]:
    out: list[AdaptiveEvidence] = []
    for probe_code, primitive_id in _BENCHMARK_PROBE_TO_PRIMITIVE.items():
        prefix = f"probe.{probe_code}."
        if not _metric_bool(entry.metrics, f"{prefix}completed"):
            continue
        metrics = {
            key[len(prefix) :]: value
            for key, value in entry.metrics.items()
            if key.startswith(prefix)
        }
        evidence = _build_evidence(
            primitive_id=primitive_id,
            source_code=probe_code,
            source_kind="benchmark_probe",
            completed_at_utc=entry.completed_at_utc,
            metrics=metrics,
            coarse=False,
        )
        if evidence is not None:
            out.append(evidence)
    return out


def _adaptive_block_evidence_from_attempt(entry: AttemptHistoryEntry) -> list[AdaptiveEvidence]:
    block_numbers = sorted(
        {
            key.split(".")[1]
            for key in entry.metrics
            if key.startswith("block.") and len(key.split(".")) >= 3
        }
    )
    out: list[AdaptiveEvidence] = []
    for number in block_numbers:
        prefix = f"block.{number}."
        primitive_id = entry.metrics.get(f"{prefix}primitive_id", "").strip()
        if primitive_id not in _PRIMITIVE_BY_ID:
            continue
        metrics = {
            key[len(prefix) :]: value
            for key, value in entry.metrics.items()
            if key.startswith(prefix)
        }
        source_code = metrics.get("drill_code", f"adaptive_block_{number}")
        evidence = _build_evidence(
            primitive_id=primitive_id,
            source_code=source_code,
            source_kind="adaptive_block",
            completed_at_utc=entry.completed_at_utc,
            metrics=metrics,
            coarse=False,
        )
        if evidence is not None:
            out.append(evidence)
    return out


def _build_evidence(
    *,
    primitive_id: str,
    source_code: str,
    source_kind: str,
    completed_at_utc: str,
    metrics: dict[str, str],
    coarse: bool,
) -> AdaptiveEvidence | None:
    score_ratio = _metric_float(metrics, "score_ratio")
    accuracy = _metric_float(metrics, "accuracy")
    performance_score = score_ratio if score_ratio is not None else accuracy
    fatigue_score = _fatigue_score_from_metrics(metrics)
    post_error_score = _post_error_score_from_metrics(metrics)
    if performance_score is None and fatigue_score is None and post_error_score is None:
        return None
    return AdaptiveEvidence(
        primitive_id=primitive_id,
        source_code=str(source_code),
        source_kind=str(source_kind),
        completed_at_utc=str(completed_at_utc),
        completed_at_epoch_s=_epoch_s(completed_at_utc),
        performance_score=None if performance_score is None else _clamp(performance_score),
        fatigue_score=fatigue_score,
        post_error_score=post_error_score,
        coarse=bool(coarse),
    )


def _fatigue_score_from_metrics(metrics: dict[str, str]) -> float | None:
    first_accuracy = _metric_float(metrics, "first_half_accuracy")
    last_accuracy = _metric_float(metrics, "second_half_accuracy")
    first_timeout = _metric_float(metrics, "first_half_timeout_rate")
    last_timeout = _metric_float(metrics, "second_half_timeout_rate")
    if first_accuracy is None and last_accuracy is None and first_timeout is None and last_timeout is None:
        first_accuracy = _metric_float(metrics, "first_3m_accuracy")
        last_accuracy = _metric_float(metrics, "last_3m_accuracy")
        first_timeout = _metric_float(metrics, "first_3m_timeout_rate")
        last_timeout = _metric_float(metrics, "last_3m_timeout_rate")
    if first_accuracy is None and last_accuracy is None and first_timeout is None and last_timeout is None:
        return None
    score = max(0.0, (first_accuracy or 0.0) - (last_accuracy or 0.0))
    score += 0.5 * max(0.0, (last_timeout or 0.0) - (first_timeout or 0.0))
    return _clamp(score)


def _post_error_score_from_metrics(metrics: dict[str, str]) -> float | None:
    inflation = _metric_float(metrics, "post_error_next_item_rt_inflation_ms")
    if inflation is None:
        return None
    return _clamp(float(inflation) / 1500.0)


def _retention_score(items: list[AdaptiveEvidence], *, now_epoch: float) -> float:
    performance_items = [item for item in items if item.performance_score is not None]
    for newer_index in range(len(performance_items) - 1, 0, -1):
        newer = performance_items[newer_index]
        for older_index in range(newer_index - 1, -1, -1):
            older = performance_items[older_index]
            delta_h = (newer.completed_at_epoch_s - older.completed_at_epoch_s) / 3600.0
            if delta_h < 48.0:
                continue
            if delta_h > 72.0:
                break
            return _clamp((older.performance_score or 0.0) - (newer.performance_score or 0.0))
    if not items:
        return 0.0
    age_h = max(0.0, (now_epoch - items[-1].completed_at_epoch_s) / 3600.0)
    if age_h < 48.0:
        return 0.0
    if age_h <= 72.0:
        return 0.5 * ((age_h - 48.0) / 24.0)
    if age_h <= 120.0:
        return 0.5 + (0.5 * ((age_h - 72.0) / 48.0))
    return 1.0


def _hours_since(last_trained_at_utc: str | None) -> float:
    if not last_trained_at_utc:
        return float("inf")
    try:
        return max(0.0, (datetime.now(UTC) - _parse_utc_iso(last_trained_at_utc)).total_seconds() / 3600.0)
    except Exception:
        return float("inf")


def _local_item_index_map(*, result: AttemptResult, item_offset: int) -> dict[int, int]:
    mapping: dict[int, int] = {}
    next_item = int(item_offset)
    for event in result.events:
        if event.item_index is None or int(event.item_index) in mapping:
            continue
        next_item += 1
        mapping[int(event.item_index)] = next_item
    return mapping


def _engine_snapshot(engine: object | None) -> object | None:
    if engine is None:
        return None
    getter = getattr(engine, "snapshot", None)
    if not callable(getter):
        return None
    try:
        return getter()
    except Exception:
        return None
