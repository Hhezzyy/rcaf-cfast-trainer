from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from .adaptive_difficulty import DifficultyFamilyId, difficulty_profile_for_family
from .canonical_drill_registry import resolved_canonical_drill_code

if TYPE_CHECKING:
    from .persistence import AttemptHistoryEntry

_MAX_EVIDENCE_TOTAL = 24
_MAX_EVIDENCE_PER_PRIMITIVE = 6
_DEFAULT_MASTERY = 0.65
_DEFAULT_SPEED = 0.65
_TIE_EPSILON = 1.0e-9


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_utc_iso(value: str) -> datetime:
    return datetime.strptime(str(value), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


def _epoch_s(value: str) -> float:
    return _parse_utc_iso(value).timestamp()


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


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


def _metric_int(metrics: dict[str, str], key: str) -> int | None:
    value = _metric_float(metrics, key)
    if value is None:
        return None
    try:
        return int(round(value))
    except Exception:
        return None


def _metric_bool(metrics: dict[str, str], key: str) -> bool:
    return str(metrics.get(key, "")).strip().lower() in {"1", "true", "yes", "on"}


def _metric_level(
    metrics: dict[str, str],
    *keys: str,
) -> int | None:
    for key in keys:
        value = _metric_float(metrics, key)
        if value is None:
            continue
        return max(1, min(10, int(round(value))))
    return None


def _ewma(prev: float | None, current: float | None, *, alpha: float) -> float | None:
    if current is None:
        return prev
    if prev is None:
        return float(current)
    return (float(alpha) * float(current)) + ((1.0 - float(alpha)) * float(prev))


def _alpha_for_source_kind(source_kind: str) -> float:
    if source_kind == "benchmark_probe":
        return 0.45
    if source_kind == "direct":
        return 0.35
    return 0.20


def _speed_score(rt_ms: float | None) -> float | None:
    if rt_ms is None:
        return None
    if rt_ms <= 750.0:
        return 1.0
    if rt_ms >= 5000.0:
        return 0.0
    span = 5000.0 - 750.0
    return _clamp(1.0 - ((float(rt_ms) - 750.0) / span))


def _performance_swing(metrics: dict[str, str]) -> float:
    first = _metric_float(metrics, "first_half_accuracy")
    last = _metric_float(metrics, "second_half_accuracy")
    if first is None and last is None:
        first = _metric_float(metrics, "first_3m_accuracy")
        last = _metric_float(metrics, "last_3m_accuracy")
    if first is None or last is None:
        return 0.0
    return _clamp(max(0.0, float(first) - float(last)))


def _fatigue_penalty(metrics: dict[str, str]) -> float | None:
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
    accuracy_drop = max(0.0, (first_accuracy or 0.0) - (last_accuracy or 0.0))
    timeout_rise = max(0.0, (last_timeout or 0.0) - (first_timeout or 0.0))
    return _clamp(accuracy_drop + (0.5 * timeout_rise))


def _post_error_penalty(metrics: dict[str, str]) -> float | None:
    rt_inflation = _metric_float(metrics, "post_error_next_item_rt_inflation_ms")
    accuracy_drop = _metric_float(metrics, "post_error_next_item_accuracy_drop")
    rt_penalty = None if rt_inflation is None else _clamp(float(rt_inflation) / 1500.0)
    if rt_penalty is not None and accuracy_drop is not None:
        return _clamp((rt_penalty + _clamp(accuracy_drop)) / 2.0)
    if rt_penalty is not None:
        return rt_penalty
    if accuracy_drop is not None:
        return _clamp(accuracy_drop)
    return None


def _instability_penalty(metrics: dict[str, str]) -> float | None:
    variance = _metric_float(metrics, "rt_variance_ms2")
    accuracy_swing = _performance_swing(metrics)
    rt_component = None
    if variance is not None and variance >= 0.0:
        rt_component = _clamp(math.sqrt(float(variance)) / 1200.0)
    if rt_component is None and accuracy_swing <= 0.0:
        return None
    if rt_component is None:
        rt_component = 0.0
    return _clamp((0.6 * rt_component) + (0.4 * accuracy_swing))


def _attempted_count(metrics: dict[str, str]) -> float:
    attempted = _metric_float(metrics, "attempted")
    if attempted is not None and attempted > 0.0:
        return max(1.0, float(attempted))
    correct = _metric_float(metrics, "correct")
    accuracy = _metric_float(metrics, "accuracy")
    timeout_count = _metric_float(metrics, "timeout_count")
    if correct is not None and accuracy is not None and accuracy > 0.0:
        estimated = float(correct) / max(accuracy, 1.0e-6)
        if timeout_count is not None:
            estimated = max(estimated, float(correct) + float(timeout_count))
        return max(1.0, estimated)
    return 12.0


def _lapse_penalty(metrics: dict[str, str]) -> float | None:
    timeout_rate = _metric_float(metrics, "timeout_rate")
    longest_streak = _metric_int(metrics, "longest_lapse_streak")
    if timeout_rate is None and longest_streak is None:
        return None
    timeout_component = 0.0 if timeout_rate is None else _clamp(timeout_rate)
    attempted = _attempted_count(metrics)
    streak_component = 0.0
    if longest_streak is not None and longest_streak > 0:
        streak_component = _clamp(float(longest_streak) / max(2.0, attempted * 0.25))
    return _clamp((0.65 * timeout_component) + (0.35 * streak_component))


def _distractor_penalty(metrics: dict[str, str]) -> float | None:
    captures = _metric_int(metrics, "distractor_capture_count")
    if captures is None:
        return None
    return _clamp(float(captures) / _attempted_count(metrics))


def _switch_penalty(metrics: dict[str, str]) -> float | None:
    switch_cost_ms = _metric_float(metrics, "switch_cost_ms")
    if switch_cost_ms is None:
        return None
    return _clamp(max(0.0, float(switch_cost_ms)) / 1200.0)


def _control_penalty(metrics: dict[str, str], *, instability_penalty: float | None) -> float | None:
    overshoot_count = _metric_int(metrics, "overshoot_count")
    overshoot_component = None
    if overshoot_count is not None:
        overshoot_component = _clamp(float(overshoot_count) / _attempted_count(metrics))
    if instability_penalty is None and overshoot_component is None:
        return None
    instability_component = 0.0 if instability_penalty is None else _clamp(instability_penalty)
    overshoot_component = 0.0 if overshoot_component is None else float(overshoot_component)
    return _clamp((0.60 * instability_component) + (0.40 * overshoot_component))


def _interference_penalty(metrics: dict[str, str]) -> float | None:
    intrusion_count = _metric_int(metrics, "intrusion_count")
    omission_count = _metric_int(metrics, "omission_count")
    order_error_count = _metric_int(metrics, "order_error_count")
    if intrusion_count is None and omission_count is None and order_error_count is None:
        return None
    total_errors = float((intrusion_count or 0) + (omission_count or 0) + (order_error_count or 0))
    return _clamp(total_errors / _attempted_count(metrics))


def _age_retention_need(last_trained_at_utc: str | None, *, now_epoch: float) -> float:
    if last_trained_at_utc is None:
        return 0.0
    age_h = max(0.0, (now_epoch - _epoch_s(last_trained_at_utc)) / 3600.0)
    if age_h < 48.0:
        return 0.0
    if age_h <= 72.0:
        return 0.5 * ((age_h - 48.0) / 24.0)
    if age_h <= 120.0:
        return 0.5 + (0.5 * ((age_h - 72.0) / 48.0))
    return 1.0


def _difficulty_family_label(family_id: DifficultyFamilyId) -> str:
    return difficulty_profile_for_family(family_id, 5, "build").label


@dataclass(frozen=True, slots=True)
class PrimitiveDefinition:
    primitive_id: str
    label: str
    domain_id: DifficultyFamilyId
    leverage: float
    benchmark_probe_codes: tuple[str, ...]
    integrated_test_codes: tuple[str, ...]
    direct_drill_codes: tuple[str, ...]
    coarse_workout_codes: tuple[str, ...]
    anchor_templates: tuple[str, ...]
    tempo_templates: tuple[str, ...]
    reset_templates: tuple[str, ...]
    fatigue_templates: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PrimitiveEvidence:
    primitive_id: str
    domain_id: DifficultyFamilyId
    source_code: str
    source_kind: str
    completed_at_utc: str
    completed_at_epoch_s: float
    alpha: float
    accuracy: float | None
    score_ratio: float | None
    performance_score: float | None
    timeout_rate: float | None
    mean_rt_ms: float | None
    median_rt_ms: float | None
    speed_score: float | None
    fatigue_penalty: float | None
    post_error_penalty: float | None
    instability_penalty: float | None
    lapse_penalty: float | None
    distractor_penalty: float | None
    switch_penalty: float | None
    control_penalty: float | None
    interference_penalty: float | None
    difficulty_level_start: int | None
    difficulty_level_end: int | None
    difficulty_level: int | None
    coarse: bool


@dataclass(frozen=True, slots=True)
class PrimitiveLevelRecommendation:
    recommended_level: int
    level_confidence: float
    last_successful_level: int | None
    last_meltdown_level: int | None


@dataclass(frozen=True, slots=True)
class PrimitiveState:
    primitive_id: str
    label: str
    domain_id: DifficultyFamilyId
    mastery: float
    speed: float
    fatigue_penalty: float
    post_error_penalty: float
    instability_penalty: float
    lapse_penalty: float
    distractor_penalty: float
    switch_penalty: float
    control_penalty: float
    interference_penalty: float
    retention_need: float
    confidence: float
    leverage: float
    recommended_level: int
    level_confidence: float
    last_successful_level: int | None
    last_meltdown_level: int | None
    last_start_level: int | None
    last_end_level: int | None
    ewma_accuracy: float | None
    ewma_score_ratio: float | None
    ewma_timeout_rate: float | None
    ewma_mean_rt_ms: float | None
    composite_score: float
    weakness: float
    evidence_count: int
    benchmark_evidence_count: int
    direct_evidence_count: int
    mixed_evidence_count: int
    last_trained_at_utc: str | None
    last_source_code: str | None

    @property
    def fatigue(self) -> float:
        return self.fatigue_penalty

    @property
    def post_error(self) -> float:
        return self.post_error_penalty

    @property
    def retention(self) -> float:
        return self.retention_need

    @property
    def lapse(self) -> float:
        return self.lapse_penalty

    @property
    def distractor(self) -> float:
        return self.distractor_penalty

    @property
    def switch(self) -> float:
        return self.switch_penalty

    @property
    def control(self) -> float:
        return self.control_penalty

    @property
    def interference(self) -> float:
        return self.interference_penalty

    @property
    def profile_multiplier(self) -> float:
        return self.leverage

    @property
    def bottleneck(self) -> float:
        return 0.0


@dataclass(frozen=True, slots=True)
class PrimitiveRankingItem:
    primitive_id: str
    label: str
    domain_id: DifficultyFamilyId
    mastery: float
    speed: float
    fatigue_penalty: float
    post_error_penalty: float
    instability_penalty: float
    lapse_penalty: float
    distractor_penalty: float
    switch_penalty: float
    control_penalty: float
    interference_penalty: float
    retention_need: float
    confidence: float
    leverage: float
    recommended_level: int
    level_confidence: float
    last_successful_level: int | None
    last_meltdown_level: int | None
    last_start_level: int | None
    last_end_level: int | None
    ewma_accuracy: float | None
    ewma_score_ratio: float | None
    ewma_timeout_rate: float | None
    ewma_mean_rt_ms: float | None
    composite_score: float
    weakness: float
    exploration_bonus: float
    priority: float
    evidence_count: int
    benchmark_evidence_count: int
    direct_evidence_count: int
    mixed_evidence_count: int
    reason_tags: tuple[str, ...]
    last_trained_at_utc: str | None
    last_source_code: str | None

    @property
    def fatigue(self) -> float:
        return self.fatigue_penalty

    @property
    def post_error(self) -> float:
        return self.post_error_penalty

    @property
    def retention(self) -> float:
        return self.retention_need

    @property
    def lapse(self) -> float:
        return self.lapse_penalty

    @property
    def distractor(self) -> float:
        return self.distractor_penalty

    @property
    def switch(self) -> float:
        return self.switch_penalty

    @property
    def control(self) -> float:
        return self.control_penalty

    @property
    def interference(self) -> float:
        return self.interference_penalty

    @property
    def profile_multiplier(self) -> float:
        return self.leverage

    @property
    def bottleneck(self) -> float:
        return 0.0

    @property
    def coarse_evidence_count(self) -> int:
        return self.mixed_evidence_count


@dataclass(frozen=True, slots=True)
class PrimitiveDomainSummary:
    domain_id: DifficultyFamilyId
    label: str
    weakest_primitive_id: str
    weakest_primitive_label: str
    priority_sum: float
    mean_composite_score: float
    mean_confidence: float


@dataclass(frozen=True, slots=True)
class PrimitiveRankingResult:
    generated_at_utc: str
    weakest_primitives: tuple[PrimitiveRankingItem, ...]
    domain_summaries: tuple[PrimitiveDomainSummary, ...]
    primitive_states: tuple[PrimitiveState, ...]


_PRIMITIVES: tuple[PrimitiveDefinition, ...] = (
    PrimitiveDefinition(
        primitive_id="mental_arithmetic_automaticity",
        label="Mental Arithmetic Automaticity",
        domain_id="quantitative",
        leverage=1.35,
        benchmark_probe_codes=("numerical_operations",),
        integrated_test_codes=("numerical_operations", "math_reasoning"),
        direct_drill_codes=(
            "ma_one_step_fluency",
            "ma_percentage_snap",
            "ma_written_numerical_extraction",
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
        anchor_templates=(
            "ma_one_step_fluency",
            "ma_percentage_snap",
            "ma_written_numerical_extraction",
            "ma_rate_time_distance",
            "ma_fuel_endurance",
        ),
        tempo_templates=(
            "ma_percentage_snap",
            "ma_rate_time_distance",
            "ma_fuel_endurance",
            "ma_written_numerical_extraction",
            "ma_mixed_conversion_caps",
        ),
        reset_templates=(
            "ma_one_step_fluency",
            "ma_mixed_conversion_caps",
            "ma_rate_time_distance",
            "ma_percentage_snap",
            "ma_written_numerical_extraction",
        ),
        fatigue_templates=(
            "ma_fuel_endurance",
            "ma_written_numerical_extraction",
            "ma_rate_time_distance",
            "ma_percentage_snap",
            "ma_mixed_conversion_caps",
        ),
    ),
    PrimitiveDefinition(
        primitive_id="table_cross_reference_speed",
        label="Table Cross-Reference Speed",
        domain_id="table_cross_reference",
        leverage=1.25,
        benchmark_probe_codes=("table_reading",),
        integrated_test_codes=("table_reading", "airborne_numerical"),
        direct_drill_codes=(
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
    PrimitiveDefinition(
        primitive_id="visual_scan_discipline",
        label="Visual Scan Discipline",
        domain_id="search_vigilance",
        leverage=1.20,
        benchmark_probe_codes=("visual_search",),
        integrated_test_codes=("visual_search", "vigilance", "target_recognition"),
        direct_drill_codes=(
            "vs_multi_target_class_search",
            "vs_priority_switch_search",
            "vs_matrix_routine_priority_switch",
            "vs_target_preview",
            "vs_clean_scan",
            "vs_family_run_letters",
            "vs_family_run_symbols",
            "vs_mixed_tempo",
            "vs_pressure_run",
        ),
        coarse_workout_codes=("visual_search_workout",),
        anchor_templates=("vs_multi_target_class_search",),
        tempo_templates=("vs_priority_switch_search", "vs_matrix_routine_priority_switch"),
        reset_templates=("vs_multi_target_class_search",),
        fatigue_templates=("vs_matrix_routine_priority_switch",),
    ),
    PrimitiveDefinition(
        primitive_id="symbolic_rule_extraction",
        label="Symbolic Rule Extraction",
        domain_id="system_logic",
        leverage=1.15,
        benchmark_probe_codes=("sl_graph_rule_anchor",),
        integrated_test_codes=("system_logic",),
        direct_drill_codes=(
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
    PrimitiveDefinition(
        primitive_id="tracking_stability_low_load",
        label="Tracking Stability Under Low Load",
        domain_id="psychomotor_tracking",
        leverage=1.00,
        benchmark_probe_codes=("rt_lock_anchor",),
        integrated_test_codes=("rapid_tracking", "sensory_motor_apparatus"),
        direct_drill_codes=(
            "rt_obscured_target_prediction",
            "rt_lock_anchor",
            "rt_building_handoff_prime",
            "rt_terrain_recovery_run",
            "rt_capture_timing_prime",
            "rt_ground_tempo_run",
            "rt_air_speed_run",
            "rt_mixed_tempo",
            "rt_pressure_run",
            "sma_split_axis_control",
            "sma_overshoot_recovery",
            "sma_joystick_horizontal_anchor",
            "sma_joystick_vertical_anchor",
            "sma_joystick_hold_run",
            "sma_split_horizontal_prime",
            "sma_split_coordination_run",
            "sma_mode_switch_run",
            "sma_disturbance_tempo",
            "sma_pressure_run",
            "sma_hold_center_lock",
            "sma_hold_band_horizontal",
            "sma_hold_band_vertical",
            "sma_recenter_run",
            "sma_turbulence_recovery",
            "sma_mixed_axis_run",
        ),
        coarse_workout_codes=("rapid_tracking_workout", "sensory_motor_apparatus_workout"),
        anchor_templates=("sma_split_axis_control",),
        tempo_templates=("sma_split_axis_control", "sma_overshoot_recovery"),
        reset_templates=("sma_split_axis_control",),
        fatigue_templates=("rt_obscured_target_prediction",),
    ),
    PrimitiveDefinition(
        primitive_id="dual_task_stability_fatigue",
        label="Dual-Task Stability Under Fatigue",
        domain_id="cln_multitask",
        leverage=1.30,
        benchmark_probe_codes=("cln_sequence_math_recall",),
        integrated_test_codes=(
            "colours_letters_numbers",
            "auditory_capacity",
            "cognitive_updating",
            "situational_awareness",
        ),
        direct_drill_codes=(
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
            "ac_tone_digit_anchor",
            "ac_filter_switch_run",
            "ac_digit_recall_mix",
            "ac_pressure_run",
            "cu_panel_focus_anchor",
            "cu_switch_density_run",
            "cu_revision_pressure",
            "sa_radio_recall_anchor",
            "sa_update_filter_run",
            "sa_pressure_run",
        ),
        coarse_workout_codes=(
            "colours_letters_numbers_workout",
            "auditory_capacity_workout",
            "cognitive_updating_workout",
            "situational_awareness_workout",
        ),
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

PRIMITIVES = _PRIMITIVES
PRIMITIVE_BY_ID = {primitive.primitive_id: primitive for primitive in _PRIMITIVES}
_BENCHMARK_PROBE_TO_PRIMITIVE = {
    code: primitive
    for primitive in _PRIMITIVES
    for code in primitive.benchmark_probe_codes
}
_DIRECT_DRILL_TO_PRIMITIVE = {
    code: primitive
    for primitive in _PRIMITIVES
    for code in primitive.direct_drill_codes
}
_INTEGRATED_TEST_TO_PRIMITIVE = {
    code: primitive
    for primitive in _PRIMITIVES
    for code in primitive.integrated_test_codes
}
_WORKOUT_TO_PRIMITIVE = {
    code: primitive
    for primitive in _PRIMITIVES
    for code in primitive.coarse_workout_codes
}
_CANONICAL_CODE_TO_PRIMITIVE_ID = {
    **{code: primitive.primitive_id for code, primitive in _BENCHMARK_PROBE_TO_PRIMITIVE.items()},
    **{code: primitive.primitive_id for code, primitive in _DIRECT_DRILL_TO_PRIMITIVE.items()},
    **{code: primitive.primitive_id for code, primitive in _INTEGRATED_TEST_TO_PRIMITIVE.items()},
    **{code: primitive.primitive_id for code, primitive in _WORKOUT_TO_PRIMITIVE.items()},
}


def canonical_ranked_primitive_id_for_code(test_code: str | None) -> str | None:
    token = str(test_code or "").strip().lower()
    if token == "":
        return None
    direct = _CANONICAL_CODE_TO_PRIMITIVE_ID.get(token)
    if direct is not None:
        return direct
    resolved = resolved_canonical_drill_code(token, for_adaptive=True)
    if resolved is None or resolved == token:
        return None
    return _CANONICAL_CODE_TO_PRIMITIVE_ID.get(resolved)


def collect_primitive_evidence(
    history: list[AttemptHistoryEntry],
    *,
    now_utc: str | None = None,
) -> list[PrimitiveEvidence]:
    _ = now_utc
    mapped: list[PrimitiveEvidence] = []
    for entry in history:
        mapped.extend(_evidence_from_attempt(entry))
    mapped.sort(key=lambda evidence: (evidence.completed_at_epoch_s, evidence.source_code), reverse=True)
    per_primitive_counts: dict[str, int] = defaultdict(int)
    limited: list[PrimitiveEvidence] = []
    for evidence in mapped:
        if len(limited) >= _MAX_EVIDENCE_TOTAL:
            break
        if per_primitive_counts[evidence.primitive_id] >= _MAX_EVIDENCE_PER_PRIMITIVE:
            continue
        limited.append(evidence)
        per_primitive_counts[evidence.primitive_id] += 1
    return limited


def rank_primitives(
    history: list[AttemptHistoryEntry],
    *,
    now_utc: str | None = None,
    seed: int = 0,
) -> PrimitiveRankingResult:
    now_token = now_utc or _utc_now_iso()
    now_epoch = _epoch_s(now_token)
    evidence = collect_primitive_evidence(history, now_utc=now_token)
    grouped: dict[str, list[PrimitiveEvidence]] = {primitive.primitive_id: [] for primitive in _PRIMITIVES}
    for item in evidence:
        grouped[item.primitive_id].append(item)

    primitive_states = tuple(
        _build_primitive_state(
            primitive=primitive,
            items=sorted(grouped.get(primitive.primitive_id, ()), key=lambda item: item.completed_at_epoch_s),
            now_epoch=now_epoch,
        )
        for primitive in _PRIMITIVES
    )
    ranked = _rank_items_from_states(primitive_states, seed=seed)
    domain_summaries = _domain_summaries_from_items(ranked)
    return PrimitiveRankingResult(
        generated_at_utc=now_token,
        weakest_primitives=tuple(ranked),
        domain_summaries=tuple(domain_summaries),
        primitive_states=primitive_states,
    )


def primitive_state_for_id(
    ranking: PrimitiveRankingResult,
    primitive_id: str,
) -> PrimitiveState | None:
    token = str(primitive_id).strip().lower()
    for state in ranking.primitive_states:
        if state.primitive_id == token:
            return state
    return None


def _build_primitive_state(
    *,
    primitive: PrimitiveDefinition,
    items: list[PrimitiveEvidence],
    now_epoch: float,
) -> PrimitiveState:
    mastery: float | None = None
    speed: float | None = None
    fatigue_penalty: float | None = None
    post_error_penalty: float | None = None
    instability_penalty: float | None = None
    lapse_penalty: float | None = None
    distractor_penalty: float | None = None
    switch_penalty: float | None = None
    control_penalty: float | None = None
    interference_penalty: float | None = None
    retention_ewma: float | None = None
    ewma_accuracy: float | None = None
    ewma_score_ratio: float | None = None
    ewma_timeout_rate: float | None = None
    ewma_mean_rt_ms: float | None = None
    confidence = 0.0
    level_evidence_confidence = 0.0
    last_successful_level: int | None = None
    last_meltdown_level: int | None = None
    last_start_level: int | None = None
    last_end_level: int | None = None
    last_trained_at_utc: str | None = None
    last_source_code: str | None = None
    benchmark_count = 0
    direct_count = 0
    mixed_count = 0
    previous_performance: PrimitiveEvidence | None = None
    recent_score_ratio: float | None = None
    recent_timeout_rate: float | None = None
    recent_fatigue = 0.0
    recent_post_error = 0.0
    recent_instability = 0.0
    recent_lapse = 0.0
    recent_distractor = 0.0
    recent_switch = 0.0
    recent_control = 0.0
    recent_interference = 0.0
    recent_level: int | None = None

    for item in items:
        alpha = float(item.alpha)
        mastery = _ewma(mastery, item.performance_score, alpha=alpha)
        speed = _ewma(speed, item.speed_score, alpha=alpha)
        fatigue_penalty = _ewma(fatigue_penalty, item.fatigue_penalty, alpha=alpha)
        post_error_penalty = _ewma(post_error_penalty, item.post_error_penalty, alpha=alpha)
        instability_penalty = _ewma(instability_penalty, item.instability_penalty, alpha=alpha)
        lapse_penalty = _ewma(lapse_penalty, item.lapse_penalty, alpha=alpha)
        distractor_penalty = _ewma(distractor_penalty, item.distractor_penalty, alpha=alpha)
        switch_penalty = _ewma(switch_penalty, item.switch_penalty, alpha=alpha)
        control_penalty = _ewma(control_penalty, item.control_penalty, alpha=alpha)
        interference_penalty = _ewma(interference_penalty, item.interference_penalty, alpha=alpha)
        ewma_accuracy = _ewma(ewma_accuracy, item.accuracy, alpha=alpha)
        ewma_score_ratio = _ewma(ewma_score_ratio, item.score_ratio, alpha=alpha)
        ewma_timeout_rate = _ewma(ewma_timeout_rate, item.timeout_rate, alpha=alpha)
        ewma_mean_rt_ms = _ewma(ewma_mean_rt_ms, item.mean_rt_ms, alpha=alpha)
        confidence = confidence + (alpha * (1.0 - confidence))
        last_trained_at_utc = item.completed_at_utc
        last_source_code = item.source_code
        if item.source_kind == "benchmark_probe":
            benchmark_count += 1
        elif item.source_kind == "direct":
            direct_count += 1
        else:
            mixed_count += 1
        if item.score_ratio is not None:
            recent_score_ratio = item.score_ratio
        if item.timeout_rate is not None:
            recent_timeout_rate = item.timeout_rate
        if item.fatigue_penalty is not None:
            recent_fatigue = float(item.fatigue_penalty)
        if item.post_error_penalty is not None:
            recent_post_error = float(item.post_error_penalty)
        if item.instability_penalty is not None:
            recent_instability = float(item.instability_penalty)
        if item.lapse_penalty is not None:
            recent_lapse = float(item.lapse_penalty)
        if item.distractor_penalty is not None:
            recent_distractor = float(item.distractor_penalty)
        if item.switch_penalty is not None:
            recent_switch = float(item.switch_penalty)
        if item.control_penalty is not None:
            recent_control = float(item.control_penalty)
        if item.interference_penalty is not None:
            recent_interference = float(item.interference_penalty)
        if item.difficulty_level is not None:
            level_evidence_confidence = level_evidence_confidence + (alpha * (1.0 - level_evidence_confidence))
            recent_level = item.difficulty_level
            last_start_level = item.difficulty_level_start
            last_end_level = item.difficulty_level_end
            if _is_successful_level_evidence(item):
                last_successful_level = item.difficulty_level
            if _is_meltdown_level_evidence(item):
                last_meltdown_level = item.difficulty_level
        if previous_performance is not None:
            if (
                previous_performance.performance_score is not None
                and item.performance_score is not None
            ):
                delta_h = (item.completed_at_epoch_s - previous_performance.completed_at_epoch_s) / 3600.0
                if 48.0 <= delta_h <= 72.0:
                    decay = _clamp(
                        float(previous_performance.performance_score) - float(item.performance_score)
                    )
                    retention_ewma = _ewma(retention_ewma, decay, alpha=alpha)
        if item.performance_score is not None:
            previous_performance = item

    mastery_value = _DEFAULT_MASTERY if mastery is None else _clamp(mastery)
    speed_value = _DEFAULT_SPEED if speed is None else _clamp(speed)
    fatigue_value = 0.0 if fatigue_penalty is None else _clamp(fatigue_penalty)
    post_error_value = 0.0 if post_error_penalty is None else _clamp(post_error_penalty)
    instability_value = 0.0 if instability_penalty is None else _clamp(instability_penalty)
    lapse_value = 0.0 if lapse_penalty is None else _clamp(lapse_penalty)
    distractor_value = 0.0 if distractor_penalty is None else _clamp(distractor_penalty)
    switch_value = 0.0 if switch_penalty is None else _clamp(switch_penalty)
    control_value = 0.0 if control_penalty is None else _clamp(control_penalty)
    interference_value = 0.0 if interference_penalty is None else _clamp(interference_penalty)
    retention_value = max(
        0.0 if retention_ewma is None else _clamp(retention_ewma),
        _age_retention_need(last_trained_at_utc, now_epoch=now_epoch),
    )
    composite_score = _clamp(
        (0.42 * mastery_value)
        + (0.15 * speed_value)
        + (0.08 * (1.0 - fatigue_value))
        + (0.07 * (1.0 - post_error_value))
        + (0.06 * (1.0 - lapse_value))
        + (0.05 * (1.0 - distractor_value))
        + (0.05 * (1.0 - switch_value))
        + (0.06 * (1.0 - control_value))
        + (0.06 * (1.0 - interference_value))
    )
    weakness = _clamp(1.0 - composite_score)
    recommendation = _recommend_level(
        current_level=recent_level or last_end_level or last_successful_level or 5,
        recent_score_ratio=recent_score_ratio,
        ewma_score_ratio=ewma_score_ratio,
        recent_timeout_rate=recent_timeout_rate,
        ewma_timeout_rate=ewma_timeout_rate,
        recent_fatigue=recent_fatigue,
        fatigue_penalty=fatigue_value,
        recent_post_error=recent_post_error,
        post_error_penalty=post_error_value,
        recent_instability=recent_instability,
        instability_penalty=instability_value,
        recent_lapse=recent_lapse,
        lapse_penalty=lapse_value,
        recent_distractor=recent_distractor,
        distractor_penalty=distractor_value,
        recent_switch=recent_switch,
        switch_penalty=switch_value,
        recent_control=recent_control,
        control_penalty=control_value,
        recent_interference=recent_interference,
        interference_penalty=interference_value,
        level_evidence_confidence=level_evidence_confidence,
        last_successful_level=last_successful_level,
        last_meltdown_level=last_meltdown_level,
    )
    return PrimitiveState(
        primitive_id=primitive.primitive_id,
        label=primitive.label,
        domain_id=primitive.domain_id,
        mastery=mastery_value,
        speed=speed_value,
        fatigue_penalty=fatigue_value,
        post_error_penalty=post_error_value,
        instability_penalty=instability_value,
        lapse_penalty=lapse_value,
        distractor_penalty=distractor_value,
        switch_penalty=switch_value,
        control_penalty=control_value,
        interference_penalty=interference_value,
        retention_need=retention_value,
        confidence=_clamp(confidence),
        leverage=float(primitive.leverage),
        recommended_level=int(recommendation.recommended_level),
        level_confidence=float(recommendation.level_confidence),
        last_successful_level=recommendation.last_successful_level,
        last_meltdown_level=recommendation.last_meltdown_level,
        last_start_level=last_start_level,
        last_end_level=last_end_level,
        ewma_accuracy=ewma_accuracy,
        ewma_score_ratio=ewma_score_ratio,
        ewma_timeout_rate=ewma_timeout_rate,
        ewma_mean_rt_ms=ewma_mean_rt_ms,
        composite_score=composite_score,
        weakness=weakness,
        evidence_count=len(items),
        benchmark_evidence_count=benchmark_count,
        direct_evidence_count=direct_count,
        mixed_evidence_count=mixed_count,
        last_trained_at_utc=last_trained_at_utc,
        last_source_code=last_source_code,
    )


def _recommend_level(
    *,
    current_level: int,
    recent_score_ratio: float | None,
    ewma_score_ratio: float | None,
    recent_timeout_rate: float | None,
    ewma_timeout_rate: float | None,
    recent_fatigue: float,
    fatigue_penalty: float,
    recent_post_error: float,
    post_error_penalty: float,
    recent_instability: float,
    instability_penalty: float,
    recent_lapse: float,
    lapse_penalty: float,
    recent_distractor: float,
    distractor_penalty: float,
    recent_switch: float,
    switch_penalty: float,
    recent_control: float,
    control_penalty: float,
    recent_interference: float,
    interference_penalty: float,
    level_evidence_confidence: float,
    last_successful_level: int | None = None,
    last_meltdown_level: int | None = None,
) -> PrimitiveLevelRecommendation:
    level = max(1, min(10, int(current_level)))
    recent_timeout = 0.0 if recent_timeout_rate is None else _clamp(recent_timeout_rate)
    ewma_timeout = 0.0 if ewma_timeout_rate is None else _clamp(ewma_timeout_rate)
    max_penalty = max(
        fatigue_penalty,
        post_error_penalty,
        instability_penalty,
        lapse_penalty,
        distractor_penalty,
        switch_penalty,
        control_penalty,
        interference_penalty,
    )
    level_confidence = _clamp(
        (0.6 * _clamp(level_evidence_confidence))
        + (0.4 * (1.0 - max_penalty))
    )

    promote = (
        recent_score_ratio is not None
        and ewma_score_ratio is not None
        and float(recent_score_ratio) >= 0.88
        and float(ewma_score_ratio) >= 0.88
        and recent_timeout <= 0.06
        and ewma_timeout <= 0.06
        and recent_fatigue <= 0.15
        and fatigue_penalty <= 0.15
        and recent_post_error <= 0.15
        and post_error_penalty <= 0.15
        and recent_instability <= 0.15
        and instability_penalty <= 0.15
        and recent_lapse <= 0.15
        and lapse_penalty <= 0.15
        and recent_distractor <= 0.15
        and distractor_penalty <= 0.15
        and recent_switch <= 0.15
        and switch_penalty <= 0.15
        and recent_control <= 0.15
        and control_penalty <= 0.15
        and recent_interference <= 0.15
        and interference_penalty <= 0.15
    )
    meltdown = (
        (recent_score_ratio is not None and float(recent_score_ratio) < 0.60)
        or recent_timeout > 0.20
        or recent_fatigue >= 0.35
        or recent_post_error >= 0.35
        or recent_instability >= 0.35
        or recent_lapse >= 0.35
        or recent_distractor >= 0.35
        or recent_switch >= 0.35
        or recent_control >= 0.35
        or recent_interference >= 0.35
    )
    demote = meltdown or (
        (recent_score_ratio is not None and float(recent_score_ratio) < 0.70)
        or recent_timeout > 0.16
        or recent_fatigue >= 0.28
        or recent_post_error >= 0.28
        or recent_instability >= 0.28
        or recent_lapse >= 0.28
        or recent_distractor >= 0.28
        or recent_switch >= 0.28
        or recent_control >= 0.28
        or recent_interference >= 0.28
    )

    if promote:
        recommended = min(10, level + 1)
    elif demote:
        recommended = max(1, level - 1)
    else:
        recommended = level

    return PrimitiveLevelRecommendation(
        recommended_level=int(recommended),
        level_confidence=float(level_confidence),
        last_successful_level=last_successful_level,
        last_meltdown_level=last_meltdown_level,
    )


def _is_successful_level_evidence(item: PrimitiveEvidence) -> bool:
    if item.difficulty_level is None or item.score_ratio is None:
        return False
    return bool(
        float(item.score_ratio) >= 0.80
        and (0.0 if item.timeout_rate is None else float(item.timeout_rate)) <= 0.10
        and (0.0 if item.fatigue_penalty is None else float(item.fatigue_penalty)) <= 0.20
        and (0.0 if item.post_error_penalty is None else float(item.post_error_penalty)) <= 0.20
        and (0.0 if item.instability_penalty is None else float(item.instability_penalty)) <= 0.20
        and (0.0 if item.lapse_penalty is None else float(item.lapse_penalty)) <= 0.20
        and (0.0 if item.distractor_penalty is None else float(item.distractor_penalty)) <= 0.20
        and (0.0 if item.switch_penalty is None else float(item.switch_penalty)) <= 0.20
        and (0.0 if item.control_penalty is None else float(item.control_penalty)) <= 0.20
        and (0.0 if item.interference_penalty is None else float(item.interference_penalty)) <= 0.20
    )


def _is_meltdown_level_evidence(item: PrimitiveEvidence) -> bool:
    if item.difficulty_level is None:
        return False
    if item.score_ratio is not None and float(item.score_ratio) < 0.60:
        return True
    if item.timeout_rate is not None and float(item.timeout_rate) > 0.20:
        return True
    if item.fatigue_penalty is not None and float(item.fatigue_penalty) >= 0.35:
        return True
    if item.post_error_penalty is not None and float(item.post_error_penalty) >= 0.35:
        return True
    if item.instability_penalty is not None and float(item.instability_penalty) >= 0.35:
        return True
    if item.lapse_penalty is not None and float(item.lapse_penalty) >= 0.35:
        return True
    if item.distractor_penalty is not None and float(item.distractor_penalty) >= 0.35:
        return True
    if item.switch_penalty is not None and float(item.switch_penalty) >= 0.35:
        return True
    if item.control_penalty is not None and float(item.control_penalty) >= 0.35:
        return True
    if item.interference_penalty is not None and float(item.interference_penalty) >= 0.35:
        return True
    return False


def _rank_items_from_states(
    states: tuple[PrimitiveState, ...],
    *,
    seed: int,
) -> list[PrimitiveRankingItem]:
    ranked = [
        _ranking_item_from_state(state)
        for state in states
    ]
    ranked.sort(key=lambda item: item.priority, reverse=True)
    rng = random.Random(int(seed))
    shuffled: list[PrimitiveRankingItem] = []
    start = 0
    while start < len(ranked):
        end = start + 1
        while end < len(ranked) and abs(ranked[end].priority - ranked[start].priority) <= _TIE_EPSILON:
            end += 1
        bucket = list(ranked[start:end])
        if len(bucket) > 1:
            rng.shuffle(bucket)
        shuffled.extend(bucket)
        start = end
    return shuffled


def _ranking_item_from_state(state: PrimitiveState) -> PrimitiveRankingItem:
    exploration_bonus = _clamp(1.0 - state.confidence)
    contributions = (
        ("weak", 0.28 * state.weakness),
        ("fatigue", 0.16 * state.fatigue_penalty),
        ("post-error", 0.12 * state.post_error_penalty),
        ("lapse", 0.10 * state.lapse_penalty),
        ("distractor", 0.08 * state.distractor_penalty),
        ("switch", 0.08 * state.switch_penalty),
        ("control", 0.08 * state.control_penalty),
        ("interference", 0.05 * state.interference_penalty),
        ("retention", 0.10 * state.retention_need),
        ("exploration", 0.05 * exploration_bonus),
    )
    priority = float(state.leverage) * sum(weight for _tag, weight in contributions)
    ordered_reasons = sorted(contributions, key=lambda pair: (pair[1], pair[0]), reverse=True)
    reason_tags = tuple(tag for tag, weight in ordered_reasons if weight >= 0.05)[:3]
    if not reason_tags:
        reason_tags = tuple(tag for tag, _weight in ordered_reasons[:2])
    return PrimitiveRankingItem(
        primitive_id=state.primitive_id,
        label=state.label,
        domain_id=state.domain_id,
        mastery=state.mastery,
        speed=state.speed,
        fatigue_penalty=state.fatigue_penalty,
        post_error_penalty=state.post_error_penalty,
        instability_penalty=state.instability_penalty,
        lapse_penalty=state.lapse_penalty,
        distractor_penalty=state.distractor_penalty,
        switch_penalty=state.switch_penalty,
        control_penalty=state.control_penalty,
        interference_penalty=state.interference_penalty,
        retention_need=state.retention_need,
        confidence=state.confidence,
        leverage=state.leverage,
        recommended_level=state.recommended_level,
        level_confidence=state.level_confidence,
        last_successful_level=state.last_successful_level,
        last_meltdown_level=state.last_meltdown_level,
        last_start_level=state.last_start_level,
        last_end_level=state.last_end_level,
        ewma_accuracy=state.ewma_accuracy,
        ewma_score_ratio=state.ewma_score_ratio,
        ewma_timeout_rate=state.ewma_timeout_rate,
        ewma_mean_rt_ms=state.ewma_mean_rt_ms,
        composite_score=state.composite_score,
        weakness=state.weakness,
        exploration_bonus=exploration_bonus,
        priority=priority,
        evidence_count=state.evidence_count,
        benchmark_evidence_count=state.benchmark_evidence_count,
        direct_evidence_count=state.direct_evidence_count,
        mixed_evidence_count=state.mixed_evidence_count,
        reason_tags=reason_tags,
        last_trained_at_utc=state.last_trained_at_utc,
        last_source_code=state.last_source_code,
    )


def _domain_summaries_from_items(
    ranked: list[PrimitiveRankingItem],
) -> list[PrimitiveDomainSummary]:
    grouped: dict[DifficultyFamilyId, list[PrimitiveRankingItem]] = defaultdict(list)
    for item in ranked:
        grouped[item.domain_id].append(item)
    summaries: list[PrimitiveDomainSummary] = []
    for domain_id, items in grouped.items():
        weakest = max(items, key=lambda item: item.priority)
        summaries.append(
            PrimitiveDomainSummary(
                domain_id=domain_id,
                label=_difficulty_family_label(domain_id),
                weakest_primitive_id=weakest.primitive_id,
                weakest_primitive_label=weakest.label,
                priority_sum=float(sum(item.priority for item in items)),
                mean_composite_score=float(sum(item.composite_score for item in items) / len(items)),
                mean_confidence=float(sum(item.confidence for item in items) / len(items)),
            )
        )
    summaries.sort(key=lambda item: item.priority_sum, reverse=True)
    return summaries


def _evidence_from_attempt(entry: AttemptHistoryEntry) -> list[PrimitiveEvidence]:
    if entry.test_code == "benchmark_battery":
        return _benchmark_evidence_from_attempt(entry)
    if entry.test_code in {"adaptive_session", "adaptive_session_short", "adaptive_session_micro"}:
        return _adaptive_block_evidence_from_attempt(entry)
    source = _source_for_code(entry.test_code)
    if source is None:
        return []
    primitive, source_kind = source
    evidence = _build_evidence(
        primitive=primitive,
        source_code=entry.test_code,
        source_kind=source_kind,
        completed_at_utc=entry.completed_at_utc,
        metrics=entry.metrics,
        difficulty_level_start=entry.difficulty_level_start,
        difficulty_level_end=entry.difficulty_level_end,
    )
    return [] if evidence is None else [evidence]


def _benchmark_evidence_from_attempt(entry: AttemptHistoryEntry) -> list[PrimitiveEvidence]:
    out: list[PrimitiveEvidence] = []
    for probe_code, primitive in _BENCHMARK_PROBE_TO_PRIMITIVE.items():
        prefix = f"probe.{probe_code}."
        if not _metric_bool(entry.metrics, f"{prefix}completed"):
            continue
        metrics = {
            key[len(prefix):]: value
            for key, value in entry.metrics.items()
            if key.startswith(prefix)
        }
        evidence = _build_evidence(
            primitive=primitive,
            source_code=probe_code,
            source_kind="benchmark_probe",
            completed_at_utc=entry.completed_at_utc,
            metrics=metrics,
            difficulty_level_start=_metric_level(metrics, "difficulty_level_start", "difficulty_level"),
            difficulty_level_end=_metric_level(metrics, "difficulty_level_end", "difficulty_level"),
        )
        if evidence is not None:
            out.append(evidence)
    return out


def _adaptive_block_evidence_from_attempt(entry: AttemptHistoryEntry) -> list[PrimitiveEvidence]:
    block_numbers = sorted(
        {
            key.split(".")[1]
            for key in entry.metrics
            if key.startswith("block.") and len(key.split(".")) >= 3
        }
    )
    out: list[PrimitiveEvidence] = []
    for number in block_numbers:
        prefix = f"block.{number}."
        primitive_id = str(entry.metrics.get(f"{prefix}primitive_id", "")).strip().lower()
        primitive = PRIMITIVE_BY_ID.get(primitive_id)
        if primitive is None:
            continue
        metrics = {
            key[len(prefix):]: value
            for key, value in entry.metrics.items()
            if key.startswith(prefix)
        }
        source_code = str(metrics.get("drill_code", f"adaptive_block_{number}"))
        evidence = _build_evidence(
            primitive=primitive,
            source_code=source_code,
            source_kind="adaptive_block",
            completed_at_utc=entry.completed_at_utc,
            metrics=metrics,
            difficulty_level_start=_metric_level(metrics, "difficulty_level_start", "difficulty_level"),
            difficulty_level_end=_metric_level(metrics, "difficulty_level_end", "difficulty_level"),
        )
        if evidence is not None:
            out.append(evidence)
    return out


def _source_for_code(test_code: str | None) -> tuple[PrimitiveDefinition, str] | None:
    token = str(test_code or "").strip().lower()
    if token in _DIRECT_DRILL_TO_PRIMITIVE:
        return (_DIRECT_DRILL_TO_PRIMITIVE[token], "direct")
    if token in _WORKOUT_TO_PRIMITIVE:
        return (_WORKOUT_TO_PRIMITIVE[token], "coarse_workout")
    if token in _INTEGRATED_TEST_TO_PRIMITIVE:
        return (_INTEGRATED_TEST_TO_PRIMITIVE[token], "integrated_test")
    if token in _BENCHMARK_PROBE_TO_PRIMITIVE:
        return (_BENCHMARK_PROBE_TO_PRIMITIVE[token], "benchmark_probe")
    return None


def _build_evidence(
    *,
    primitive: PrimitiveDefinition,
    source_code: str,
    source_kind: str,
    completed_at_utc: str,
    metrics: dict[str, str],
    difficulty_level_start: int | None,
    difficulty_level_end: int | None,
) -> PrimitiveEvidence | None:
    score_ratio = _metric_float(metrics, "score_ratio")
    accuracy = _metric_float(metrics, "accuracy")
    performance_score = score_ratio if score_ratio is not None else accuracy
    mean_rt_ms = _metric_float(metrics, "mean_rt_ms")
    median_rt_ms = _metric_float(metrics, "median_rt_ms")
    fatigue = _fatigue_penalty(metrics)
    post_error = _post_error_penalty(metrics)
    instability = _instability_penalty(metrics)
    lapse = _lapse_penalty(metrics)
    distractor = _distractor_penalty(metrics)
    switch = _switch_penalty(metrics)
    control = _control_penalty(metrics, instability_penalty=instability)
    interference = _interference_penalty(metrics)
    timeout_rate = _metric_float(metrics, "timeout_rate")
    if (
        performance_score is None
        and mean_rt_ms is None
        and median_rt_ms is None
        and fatigue is None
        and post_error is None
        and instability is None
        and lapse is None
        and distractor is None
        and switch is None
        and control is None
        and interference is None
        and timeout_rate is None
    ):
        return None
    level = difficulty_level_end if difficulty_level_end is not None else difficulty_level_start
    return PrimitiveEvidence(
        primitive_id=primitive.primitive_id,
        domain_id=primitive.domain_id,
        source_code=str(source_code),
        source_kind=str(source_kind),
        completed_at_utc=str(completed_at_utc),
        completed_at_epoch_s=_epoch_s(completed_at_utc),
        alpha=_alpha_for_source_kind(source_kind),
        accuracy=None if accuracy is None else _clamp(accuracy),
        score_ratio=None if score_ratio is None else _clamp(score_ratio),
        performance_score=None if performance_score is None else _clamp(performance_score),
        timeout_rate=None if timeout_rate is None else _clamp(timeout_rate),
        mean_rt_ms=None if mean_rt_ms is None else max(0.0, float(mean_rt_ms)),
        median_rt_ms=None if median_rt_ms is None else max(0.0, float(median_rt_ms)),
        speed_score=_speed_score(mean_rt_ms if mean_rt_ms is not None else median_rt_ms),
        fatigue_penalty=fatigue,
        post_error_penalty=post_error,
        instability_penalty=instability,
        lapse_penalty=lapse,
        distractor_penalty=distractor,
        switch_penalty=switch,
        control_penalty=control,
        interference_penalty=interference,
        difficulty_level_start=difficulty_level_start,
        difficulty_level_end=difficulty_level_end,
        difficulty_level=level,
        coarse=source_kind in {"coarse_workout", "adaptive_block", "integrated_test"},
    )
