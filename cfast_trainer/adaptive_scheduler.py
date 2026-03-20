from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import cast

from .ant_drills import AntDrillMode
from .adaptive_difficulty import (
    LaunchDifficultyMode,
    ResolvedDifficultyContext,
    build_resolved_difficulty_context,
    clamp_level,
)
from .ant_workouts import AntWorkoutBlockPlan, build_workout_block_engine
from .canonical_drill_registry import (
    CANONICAL_DRILL_BY_CODE,
    canonical_drill_spec,
    resolved_canonical_drill_code,
)
from .clock import Clock
from .cognitive_core import Phase
from .guide_skill_catalog import guide_ranking_primitive_id_for_code, guide_subskill_ids_for_code
from .persistence import AttemptHistoryEntry
from .primitive_ranking import (
    PRIMITIVES as _RANKED_PRIMITIVES,
    PRIMITIVE_BY_ID as _RANKED_PRIMITIVE_BY_ID,
    PrimitiveDomainSummary,
    PrimitiveRankingItem,
    collect_primitive_evidence,
    rank_primitives,
)
from .results import AttemptResult, attempt_result_from_engine
from .telemetry import TelemetryEvent, telemetry_analytics_from_events
from .training_modes import split_half_note_fragment, supports_training_mode

_SCHEDULER_VERSION = 2
_ROLE_SEQUENCE = (
    "target_anchor",
    "target_tempo",
    "adjacent_cross_train",
    "reassessment_probe",
    "target_pressure_fatigue",
    "late_repeat_transfer",
)
ADAPTIVE_SESSION_CODES = frozenset(
    {
        "adaptive_session",
        "adaptive_session_short",
        "adaptive_session_micro",
    }
)


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_utc_iso(value: str) -> datetime:
    return datetime.strptime(str(value), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


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
    form_factor: str = "short"
    target_area: str = ""
    linked_primitive_id: str | None = None
    comparable_level: int | None = None


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
    variant: str = "full"
    domain_summaries: tuple[PrimitiveDomainSummary, ...] = ()

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


_PRIMITIVES = _RANKED_PRIMITIVES
_PRIMITIVE_BY_ID = _RANKED_PRIMITIVE_BY_ID


@dataclass(frozen=True, slots=True)
class AdaptiveVariantSpec:
    variant: str
    code: str
    title: str
    block_duration_s: float
    allowed_form_factors: frozenset[str]


@dataclass(frozen=True, slots=True)
class AdaptiveDrillCandidate:
    drill_code: str
    primitive_id: str
    target_area: str
    form_factor: str
    role_support: tuple[str, ...]
    canonical_preferred: bool


@dataclass(frozen=True, slots=True)
class AdaptiveSkillLink:
    linked_primitive_id: str
    linked_target_areas: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class AdaptiveHistorySessionSummary:
    completed_at_utc: str
    completed_at_epoch_s: float
    primary_primitive_id: str | None
    primary_domain_id: str | None
    drill_codes: tuple[str, ...]
    target_areas: tuple[str, ...]


ADAPTIVE_VARIANT_SPECS = {
    "micro": AdaptiveVariantSpec(
        variant="micro",
        code="adaptive_session_micro",
        title="Adaptive Micro (15m)",
        block_duration_s=150.0,
        allowed_form_factors=frozenset({"micro"}),
    ),
    "short": AdaptiveVariantSpec(
        variant="short",
        code="adaptive_session_short",
        title="Adaptive Short (30m)",
        block_duration_s=300.0,
        allowed_form_factors=frozenset({"micro", "short"}),
    ),
    "full": AdaptiveVariantSpec(
        variant="full",
        code="adaptive_session",
        title="Adaptive Session (60m)",
        block_duration_s=600.0,
        allowed_form_factors=frozenset({"micro", "short", "block_component"}),
    ),
}


ADAPTIVE_SKILL_GRAPH = {
    "mental_arithmetic_automaticity": (
        AdaptiveSkillLink(
            linked_primitive_id="table_cross_reference_speed",
            linked_target_areas=(
                ("quantitative_core", "single_lookup"),
                ("written_extraction", "two_source_xref"),
                ("applied_rate_fuel", "distractor_table_pressure"),
                ("interference_resilience", "distractor_table_pressure"),
            ),
        ),
        AdaptiveSkillLink(
            linked_primitive_id="visual_scan_discipline",
            linked_target_areas=(
                ("quantitative_core", "class_search"),
                ("written_extraction", "priority_switch"),
                ("interference_resilience", "routine_priority_switch"),
            ),
        ),
        AdaptiveSkillLink(
            linked_primitive_id="dual_task_stability_fatigue",
            linked_target_areas=(
                ("applied_rate_fuel", "tracking_recall_bridge"),
                ("interference_resilience", "interference_recovery"),
            ),
        ),
    ),
    "table_cross_reference_speed": (
        AdaptiveSkillLink(
            linked_primitive_id="mental_arithmetic_automaticity",
            linked_target_areas=(
                ("single_lookup", "quantitative_core"),
                ("two_source_xref", "written_extraction"),
                ("distractor_table_pressure", "interference_resilience"),
            ),
        ),
        AdaptiveSkillLink(
            linked_primitive_id="visual_scan_discipline",
            linked_target_areas=(
                ("single_lookup", "class_search"),
                ("distractor_table_pressure", "priority_switch"),
                ("two_source_xref", "routine_priority_switch"),
            ),
        ),
        AdaptiveSkillLink(
            linked_primitive_id="symbolic_rule_extraction",
            linked_target_areas=(
                ("single_lookup", "single_rule"),
                ("two_source_xref", "two_source_reconcile"),
                ("distractor_table_pressure", "distractor_reject"),
            ),
        ),
    ),
    "visual_scan_discipline": (
        AdaptiveSkillLink(
            linked_primitive_id="table_cross_reference_speed",
            linked_target_areas=(
                ("class_search", "single_lookup"),
                ("priority_switch", "two_source_xref"),
                ("routine_priority_switch", "distractor_table_pressure"),
            ),
        ),
        AdaptiveSkillLink(
            linked_primitive_id="mental_arithmetic_automaticity",
            linked_target_areas=(
                ("class_search", "quantitative_core"),
                ("priority_switch", "written_extraction"),
                ("routine_priority_switch", "interference_resilience"),
            ),
        ),
        AdaptiveSkillLink(
            linked_primitive_id="dual_task_stability_fatigue",
            linked_target_areas=(
                ("class_search", "tracking_recall_bridge"),
                ("priority_switch", "command_filter"),
                ("routine_priority_switch", "filtered_digit_report"),
            ),
        ),
    ),
    "symbolic_rule_extraction": (
        AdaptiveSkillLink(
            linked_primitive_id="table_cross_reference_speed",
            linked_target_areas=(
                ("single_rule", "single_lookup"),
                ("two_source_reconcile", "two_source_xref"),
                ("distractor_reject", "distractor_table_pressure"),
            ),
        ),
        AdaptiveSkillLink(
            linked_primitive_id="mental_arithmetic_automaticity",
            linked_target_areas=(
                ("single_rule", "quantitative_core"),
                ("two_source_reconcile", "written_extraction"),
                ("distractor_reject", "interference_resilience"),
            ),
        ),
        AdaptiveSkillLink(
            linked_primitive_id="dual_task_stability_fatigue",
            linked_target_areas=(
                ("single_rule", "command_filter"),
                ("two_source_reconcile", "filtered_digit_report"),
                ("distractor_reject", "interference_recovery"),
            ),
        ),
    ),
    "tracking_stability_low_load": (
        AdaptiveSkillLink(
            linked_primitive_id="dual_task_stability_fatigue",
            linked_target_areas=(
                ("split_axis_control", "tracking_recall_bridge"),
                ("overshoot_recovery", "command_filter"),
                ("obscured_prediction", "interference_recovery"),
            ),
        ),
        AdaptiveSkillLink(
            linked_primitive_id="visual_scan_discipline",
            linked_target_areas=(
                ("split_axis_control", "class_search"),
                ("overshoot_recovery", "priority_switch"),
                ("obscured_prediction", "routine_priority_switch"),
            ),
        ),
    ),
    "dual_task_stability_fatigue": (
        AdaptiveSkillLink(
            linked_primitive_id="tracking_stability_low_load",
            linked_target_areas=(
                ("tracking_recall_bridge", "split_axis_control"),
                ("command_filter", "overshoot_recovery"),
                ("interference_recovery", "obscured_prediction"),
            ),
        ),
        AdaptiveSkillLink(
            linked_primitive_id="visual_scan_discipline",
            linked_target_areas=(
                ("tracking_recall_bridge", "class_search"),
                ("command_filter", "priority_switch"),
                ("filtered_digit_report", "routine_priority_switch"),
            ),
        ),
        AdaptiveSkillLink(
            linked_primitive_id="mental_arithmetic_automaticity",
            linked_target_areas=(
                ("tracking_recall_bridge", "quantitative_core"),
                ("filtered_digit_report", "written_extraction"),
                ("interference_recovery", "interference_resilience"),
            ),
        ),
    ),
}


def _candidate(
    drill_code: str,
    primitive_id: str,
    target_area: str,
    form_factor: str,
    role_support: tuple[str, ...],
) -> AdaptiveDrillCandidate:
    registry_spec = canonical_drill_spec(drill_code)
    mapped_primitive_id = guide_ranking_primitive_id_for_code(drill_code) or primitive_id
    mapped_subskills = guide_subskill_ids_for_code(drill_code)
    mapped_target_area = (
        registry_spec.primary_subskill
        if registry_spec is not None
        else (mapped_subskills[0] if mapped_subskills else target_area)
    )
    mapped_form_factor = registry_spec.granularity if registry_spec is not None else form_factor
    return AdaptiveDrillCandidate(
        drill_code=drill_code,
        primitive_id=str(mapped_primitive_id).strip().lower(),
        target_area=str(mapped_target_area).strip().lower(),
        form_factor=str(mapped_form_factor).strip().lower(),
        role_support=tuple(role_support),
        canonical_preferred=(registry_spec is not None or str(drill_code).strip().lower() in CANONICAL_DRILL_BY_CODE),
    )


ADAPTIVE_DRILL_CATALOG: tuple[AdaptiveDrillCandidate, ...] = (
    _candidate("ma_one_step_fluency", "mental_arithmetic_automaticity", "quantitative_core", "micro", ("target_anchor", "adjacent_cross_train", "reassessment_probe")),
    _candidate("ma_percentage_snap", "mental_arithmetic_automaticity", "quantitative_core", "micro", ("target_anchor", "target_tempo", "adjacent_cross_train", "reassessment_probe", "target_pressure_fatigue", "late_repeat_transfer")),
    _candidate("ma_written_numerical_extraction", "mental_arithmetic_automaticity", "written_extraction", "micro", ("target_anchor", "target_tempo", "adjacent_cross_train", "reassessment_probe", "target_pressure_fatigue", "late_repeat_transfer")),
    _candidate("ma_rate_time_distance", "mental_arithmetic_automaticity", "applied_rate_fuel", "short", ("target_tempo", "adjacent_cross_train", "reassessment_probe", "target_pressure_fatigue", "late_repeat_transfer")),
    _candidate("ma_fuel_endurance", "mental_arithmetic_automaticity", "applied_rate_fuel", "short", ("target_tempo", "adjacent_cross_train", "reassessment_probe", "target_pressure_fatigue", "late_repeat_transfer")),
    _candidate("ma_mixed_conversion_caps", "mental_arithmetic_automaticity", "interference_resilience", "block_component", ("adjacent_cross_train", "target_pressure_fatigue", "late_repeat_transfer")),
    _candidate("tbl_single_lookup_anchor", "table_cross_reference_speed", "single_lookup", "micro", ("target_anchor", "target_tempo", "adjacent_cross_train", "reassessment_probe", "target_pressure_fatigue", "late_repeat_transfer")),
    _candidate("tbl_two_table_xref", "table_cross_reference_speed", "two_source_xref", "short", ("target_tempo", "adjacent_cross_train", "reassessment_probe", "late_repeat_transfer")),
    _candidate("tbl_distractor_grid", "table_cross_reference_speed", "distractor_table_pressure", "short", ("target_tempo", "adjacent_cross_train", "target_pressure_fatigue", "late_repeat_transfer")),
    _candidate("tbl_lookup_compute", "table_cross_reference_speed", "two_source_xref", "block_component", ("adjacent_cross_train", "target_pressure_fatigue", "late_repeat_transfer")),
    _candidate("tbl_shrinking_cap_run", "table_cross_reference_speed", "distractor_table_pressure", "block_component", ("target_pressure_fatigue", "late_repeat_transfer")),
    _candidate("vs_multi_target_class_search", "visual_scan_discipline", "class_search", "micro", ("target_anchor", "target_tempo", "adjacent_cross_train", "reassessment_probe", "target_pressure_fatigue", "late_repeat_transfer")),
    _candidate("vs_priority_switch_search", "visual_scan_discipline", "priority_switch", "short", ("target_tempo", "adjacent_cross_train", "late_repeat_transfer")),
    _candidate("vs_matrix_routine_priority_switch", "visual_scan_discipline", "routine_priority_switch", "block_component", ("adjacent_cross_train", "target_pressure_fatigue", "late_repeat_transfer")),
    _candidate("sl_one_rule_identify", "symbolic_rule_extraction", "single_rule", "micro", ("target_anchor", "target_tempo", "adjacent_cross_train", "reassessment_probe", "target_pressure_fatigue", "late_repeat_transfer")),
    _candidate("sl_rule_match", "symbolic_rule_extraction", "single_rule", "short", ("target_tempo", "adjacent_cross_train", "reassessment_probe", "late_repeat_transfer")),
    _candidate("sl_two_source_reconcile", "symbolic_rule_extraction", "two_source_reconcile", "short", ("target_tempo", "adjacent_cross_train", "late_repeat_transfer")),
    _candidate("sl_missing_step_complete", "symbolic_rule_extraction", "two_source_reconcile", "block_component", ("adjacent_cross_train", "target_pressure_fatigue", "late_repeat_transfer")),
    _candidate("sl_fast_reject", "symbolic_rule_extraction", "distractor_reject", "block_component", ("adjacent_cross_train", "target_pressure_fatigue", "late_repeat_transfer")),
    _candidate("sma_split_axis_control", "tracking_stability_low_load", "split_axis_control", "micro", ("target_anchor", "target_tempo", "adjacent_cross_train", "reassessment_probe", "target_pressure_fatigue", "late_repeat_transfer")),
    _candidate("sma_overshoot_recovery", "tracking_stability_low_load", "overshoot_recovery", "short", ("target_tempo", "adjacent_cross_train", "target_pressure_fatigue", "late_repeat_transfer")),
    _candidate("rt_obscured_target_prediction", "tracking_stability_low_load", "obscured_prediction", "block_component", ("adjacent_cross_train", "target_pressure_fatigue", "late_repeat_transfer")),
    _candidate("dtb_tracking_recall", "dual_task_stability_fatigue", "tracking_recall_bridge", "micro", ("target_anchor", "target_tempo", "adjacent_cross_train", "reassessment_probe", "target_pressure_fatigue", "late_repeat_transfer")),
    _candidate("dtb_tracking_command_filter", "dual_task_stability_fatigue", "command_filter", "short", ("target_tempo", "adjacent_cross_train", "target_pressure_fatigue", "late_repeat_transfer")),
    _candidate("dtb_tracking_filter_digit_report", "dual_task_stability_fatigue", "filtered_digit_report", "short", ("target_tempo", "adjacent_cross_train", "target_pressure_fatigue", "late_repeat_transfer")),
    _candidate("dtb_tracking_interference_recovery", "dual_task_stability_fatigue", "interference_recovery", "block_component", ("adjacent_cross_train", "target_pressure_fatigue", "late_repeat_transfer")),
)


def collect_adaptive_evidence(
    history: list[AttemptHistoryEntry],
    *,
    now_utc: str | None = None,
) -> list[AdaptiveEvidence]:
    return cast(list[AdaptiveEvidence], collect_primitive_evidence(history, now_utc=now_utc))


def rank_adaptive_primitives(
    history: list[AttemptHistoryEntry],
    *,
    now_utc: str | None = None,
    seed: int = 0,
) -> tuple[AdaptivePriorityBreakdown, ...]:
    ranking = rank_primitives(history, now_utc=now_utc, seed=seed)
    return cast(tuple[AdaptivePriorityBreakdown, ...], ranking.weakest_primitives)


def _variant_spec(variant: str) -> AdaptiveVariantSpec:
    token = str(variant or "full").strip().lower()
    return ADAPTIVE_VARIANT_SPECS.get(token, ADAPTIVE_VARIANT_SPECS["full"])


def _hours_since(last_trained_at_utc: str | None, *, now_utc: str | None = None) -> float:
    if not last_trained_at_utc:
        return float("inf")
    try:
        baseline = _parse_utc_iso(now_utc) if now_utc else datetime.now(UTC)
        return max(0.0, (baseline - _parse_utc_iso(last_trained_at_utc)).total_seconds() / 3600.0)
    except Exception:
        return float("inf")


def _seed_tiebreak(token: str, *, seed: int) -> float:
    acc = int(seed) * 131
    for index, ch in enumerate(str(token)):
        acc += (index + 1) * ord(ch)
    return float((acc % 1000) / 1_000_000.0)


def _recent_adaptive_sessions(history: list[AttemptHistoryEntry]) -> list[AdaptiveHistorySessionSummary]:
    out: list[AdaptiveHistorySessionSummary] = []
    for entry in history:
        token = str(entry.test_code or "").strip().lower()
        if token not in ADAPTIVE_SESSION_CODES:
            continue
        drill_codes: list[str] = []
        target_areas: list[str] = []
        block_numbers = sorted(
            {
                key.split(".")[1]
                for key in entry.metrics
                if key.startswith("block.") and len(key.split(".")) >= 3
            }
        )
        primary_primitive_id = None
        for index, number in enumerate(block_numbers):
            prefix = f"block.{number}."
            primitive_id = str(entry.metrics.get(f"{prefix}primitive_id", "")).strip().lower() or None
            if index == 0:
                primary_primitive_id = primitive_id
            drill_code = str(entry.metrics.get(f"{prefix}drill_code", "")).strip().lower()
            if drill_code != "":
                drill_codes.append(drill_code)
            target_area = str(entry.metrics.get(f"{prefix}target_area", "")).strip().lower()
            if target_area != "":
                target_areas.append(target_area)
        primary_domain_id = None
        if primary_primitive_id in _PRIMITIVE_BY_ID:
            primary_domain_id = _PRIMITIVE_BY_ID[primary_primitive_id].domain_id
        out.append(
            AdaptiveHistorySessionSummary(
                completed_at_utc=entry.completed_at_utc,
                completed_at_epoch_s=_parse_utc_iso(entry.completed_at_utc).timestamp(),
                primary_primitive_id=primary_primitive_id,
                primary_domain_id=primary_domain_id,
                drill_codes=tuple(drill_codes),
                target_areas=tuple(target_areas),
            )
        )
    out.sort(key=lambda item: item.completed_at_epoch_s, reverse=True)
    return out


def _domain_dominated_in_recent_sessions(
    recent_sessions: list[AdaptiveHistorySessionSummary],
    *,
    domain_id: str,
    now_utc: str,
) -> bool:
    recent = [
        session
        for session in recent_sessions[:3]
        if _hours_since(session.completed_at_utc, now_utc=now_utc) <= 72.0
    ]
    if len(recent) < 3:
        return False
    return sum(1 for session in recent if session.primary_domain_id == domain_id) >= 2


def _adjusted_priority(
    item: PrimitiveRankingItem,
    *,
    recent_sessions: list[AdaptiveHistorySessionSummary],
    now_utc: str,
) -> float:
    adjusted = float(item.priority)
    recent_primary = recent_sessions[0] if recent_sessions else None
    if (
        recent_primary is not None
        and recent_primary.primary_primitive_id == item.primitive_id
        and _hours_since(recent_primary.completed_at_utc, now_utc=now_utc) <= 24.0
    ):
        adjusted -= 0.18
    if _domain_dominated_in_recent_sessions(recent_sessions, domain_id=item.domain_id, now_utc=now_utc):
        adjusted -= 0.08
    age_h = _hours_since(item.last_trained_at_utc, now_utc=now_utc)
    if 48.0 <= age_h <= 72.0 and item.retention >= 0.45:
        adjusted += 0.10
    return max(0.0, adjusted)


def _dominant_penalty_name(item: PrimitiveRankingItem) -> str:
    components = (
        ("fatigue", float(item.fatigue_penalty)),
        ("post_error", float(item.post_error_penalty)),
        ("lapse", float(item.lapse_penalty)),
        ("distractor", float(item.distractor_penalty)),
        ("switch", float(item.switch_penalty)),
        ("control", float(item.control_penalty)),
        ("interference", float(item.interference_penalty)),
        ("retention", float(item.retention)),
        ("weakness", float(item.weakness)),
    )
    ordered = sorted(components, key=lambda pair: (pair[1], pair[0]), reverse=True)
    return ordered[0][0]


def _target_area_for_primitive(item: PrimitiveRankingItem) -> str:
    dominant = _dominant_penalty_name(item)
    per_primitive = {
        "mental_arithmetic_automaticity": {
            "interference": "interference_resilience",
            "post_error": "interference_resilience",
            "lapse": "written_extraction",
            "distractor": "written_extraction",
            "fatigue": "applied_rate_fuel",
            "retention": "applied_rate_fuel",
            "weakness": "quantitative_core",
        },
        "table_cross_reference_speed": {
            "distractor": "distractor_table_pressure",
            "switch": "two_source_xref",
            "retention": "two_source_xref",
            "weakness": "single_lookup",
        },
        "visual_scan_discipline": {
            "switch": "priority_switch",
            "distractor": "routine_priority_switch",
            "fatigue": "routine_priority_switch",
            "weakness": "class_search",
        },
        "symbolic_rule_extraction": {
            "switch": "two_source_reconcile",
            "distractor": "distractor_reject",
            "interference": "distractor_reject",
            "weakness": "single_rule",
        },
        "tracking_stability_low_load": {
            "control": "overshoot_recovery",
            "fatigue": "obscured_prediction",
            "weakness": "split_axis_control",
        },
        "dual_task_stability_fatigue": {
            "interference": "interference_recovery",
            "switch": "command_filter",
            "distractor": "filtered_digit_report",
            "fatigue": "interference_recovery",
            "lapse": "tracking_recall_bridge",
            "weakness": "tracking_recall_bridge",
        },
    }
    defaults = {
        "mental_arithmetic_automaticity": "quantitative_core",
        "table_cross_reference_speed": "single_lookup",
        "visual_scan_discipline": "class_search",
        "symbolic_rule_extraction": "single_rule",
        "tracking_stability_low_load": "split_axis_control",
        "dual_task_stability_fatigue": "tracking_recall_bridge",
    }
    mapping = per_primitive.get(item.primitive_id, {})
    return mapping.get(dominant, defaults.get(item.primitive_id, ""))


def _select_primary_target(
    ranked: tuple[AdaptivePriorityBreakdown, ...],
    *,
    recent_sessions: list[AdaptiveHistorySessionSummary],
    now_utc: str,
) -> AdaptivePriorityBreakdown:
    adjusted = sorted(
        (
            (
                _adjusted_priority(item, recent_sessions=recent_sessions, now_utc=now_utc),
                float(item.priority),
                item,
            )
            for item in ranked
        ),
        key=lambda triple: (triple[0], triple[1], triple[2].primitive_id),
        reverse=True,
    )
    top = adjusted[0][2]
    recent_primary = recent_sessions[0] if recent_sessions else None
    if (
        recent_primary is not None
        and recent_primary.primary_primitive_id == top.primitive_id
        and _hours_since(recent_primary.completed_at_utc, now_utc=now_utc) <= 24.0
        and len(adjusted) > 1
    ):
        second_score = adjusted[1][0]
        if second_score > 0.0 and adjusted[0][0] < (second_score * 1.15):
            return adjusted[1][2]
    return top


def _adjacent_links_for_primitive(primitive_id: str) -> tuple[AdaptiveSkillLink, ...]:
    return tuple(ADAPTIVE_SKILL_GRAPH.get(primitive_id, ()))


def _link_for_target_area(
    primitive_id: str,
    *,
    target_area: str,
    ranked_by_id: dict[str, AdaptivePriorityBreakdown],
    recent_sessions: list[AdaptiveHistorySessionSummary],
    now_utc: str,
) -> tuple[AdaptiveSkillLink, str] | None:
    scored: list[tuple[float, str, AdaptiveSkillLink]] = []
    recent_primary = recent_sessions[0] if recent_sessions else None
    for link in _adjacent_links_for_primitive(primitive_id):
        neighbor = ranked_by_id.get(link.linked_primitive_id)
        if neighbor is None:
            continue
        linked_area = ""
        area_boost = 0.0
        for source_area, candidate_area in link.linked_target_areas:
            if source_area == target_area:
                linked_area = candidate_area
                area_boost = 0.18
                break
        if linked_area == "" and link.linked_target_areas:
            linked_area = link.linked_target_areas[0][1]
        score = _adjusted_priority(neighbor, recent_sessions=recent_sessions, now_utc=now_utc) + area_boost
        if (
            recent_primary is not None
            and recent_primary.primary_primitive_id == neighbor.primitive_id
            and _hours_since(recent_primary.completed_at_utc, now_utc=now_utc) <= 24.0
        ):
            score -= 0.05
        scored.append((score, linked_area, link))
    if not scored:
        return None
    scored.sort(key=lambda item: (item[0], item[1], item[2].linked_primitive_id), reverse=True)
    best = scored[0]
    return (best[2], best[1])


def _retention_repeat_needed(item: PrimitiveRankingItem, *, now_utc: str) -> bool:
    age_h = _hours_since(item.last_trained_at_utc, now_utc=now_utc)
    return bool(item.retention >= 0.45 or 48.0 <= age_h <= 72.0)


def _meltdown_cap_applies(item: PrimitiveRankingItem, *, base_level: int) -> bool:
    meltdown = item.last_meltdown_level
    return meltdown is not None and int(meltdown) >= int(base_level)


def _comparable_probe_level(item: PrimitiveRankingItem, *, base_level: int) -> int:
    candidate = item.last_successful_level
    if candidate is not None and abs(int(candidate) - int(base_level)) <= 1:
        return clamp_level(int(candidate))
    return clamp_level(int(base_level))


def _level_for_role(
    *,
    role: str,
    target_item: PrimitiveRankingItem,
    target_base_level: int,
    adjacent_item: PrimitiveRankingItem | None = None,
    retention_repeat: bool = False,
) -> tuple[int, int | None]:
    target_confidence = float(target_item.confidence)
    meltdown_cap = _meltdown_cap_applies(target_item, base_level=target_base_level)
    if role == "target_anchor":
        level = target_base_level - 2 if (target_confidence < 0.50 or meltdown_cap) else target_base_level - 1
        return (clamp_level(level), None)
    if role == "target_tempo":
        return (clamp_level(target_base_level), None)
    if role == "adjacent_cross_train":
        assert adjacent_item is not None
        adjacent_base = clamp_level(int(adjacent_item.recommended_level))
        level = min(int(target_base_level), int(adjacent_base))
        if float(adjacent_item.confidence) < 0.55 or target_confidence < 0.55:
            level -= 1
        return (clamp_level(level), None)
    if role == "reassessment_probe":
        comparable = _comparable_probe_level(target_item, base_level=target_base_level)
        return (clamp_level(comparable), int(comparable))
    if role == "target_pressure_fatigue":
        level = target_base_level + 1 if (float(target_item.level_confidence) >= 0.45 and not meltdown_cap) else target_base_level
        return (clamp_level(level), None)
    if role == "late_repeat_transfer":
        if retention_repeat:
            comparable = _comparable_probe_level(target_item, base_level=target_base_level)
            return (clamp_level(comparable), int(comparable))
        assert adjacent_item is not None
        return _level_for_role(
            role="adjacent_cross_train",
            target_item=target_item,
            target_base_level=target_base_level,
            adjacent_item=adjacent_item,
            retention_repeat=False,
        )
    return (clamp_level(target_base_level), None)


def _pressure_is_fatigue_dominant(item: PrimitiveRankingItem) -> bool:
    fatigue_cluster = max(float(item.fatigue_penalty), float(item.lapse_penalty))
    other_cluster = max(
        float(item.post_error_penalty),
        float(item.distractor_penalty),
        float(item.switch_penalty),
        float(item.control_penalty),
        float(item.interference_penalty),
    )
    return fatigue_cluster >= other_cluster


def _training_mode_for_role(
    drill_code: str,
    *,
    role: str,
    target_confidence: float | None = None,
    fatigue_dominant: bool = False,
    retention_repeat: bool = False,
) -> AntDrillMode:
    if role == "target_anchor":
        if supports_training_mode(drill_code, AntDrillMode.FRESH):
            return AntDrillMode.FRESH
        return AntDrillMode.BUILD
    if role == "target_tempo":
        return AntDrillMode.TEMPO if supports_training_mode(drill_code, AntDrillMode.TEMPO) else AntDrillMode.PRESSURE
    if role == "adjacent_cross_train":
        if target_confidence is not None and target_confidence < 0.55:
            return AntDrillMode.BUILD
        return AntDrillMode.TEMPO
    if role == "reassessment_probe":
        return AntDrillMode.BUILD
    if role == "target_pressure_fatigue":
        if fatigue_dominant and supports_training_mode(drill_code, AntDrillMode.FATIGUE_PROBE):
            return AntDrillMode.FATIGUE_PROBE
        return AntDrillMode.PRESSURE
    if role == "late_repeat_transfer":
        return AntDrillMode.BUILD if retention_repeat else AntDrillMode.TEMPO
    return AntDrillMode.BUILD


def _preferred_form_factors_for_role(
    role: str,
    *,
    variant_spec: AdaptiveVariantSpec,
    retention_repeat: bool,
) -> tuple[str, ...]:
    if variant_spec.variant == "micro":
        return ("micro",)
    if variant_spec.variant == "short":
        if role in {"target_anchor", "reassessment_probe"}:
            return ("micro", "short")
        return ("short", "micro")
    if role in {"target_anchor", "reassessment_probe"}:
        return ("micro", "short", "block_component")
    if role == "late_repeat_transfer" and retention_repeat:
        return ("short", "micro", "block_component")
    if role in {"adjacent_cross_train", "target_pressure_fatigue", "late_repeat_transfer"}:
        return ("block_component", "short", "micro")
    return ("short", "micro", "block_component")


def _select_candidate(
    *,
    primitive_id: str,
    role: str,
    target_area: str,
    variant_spec: AdaptiveVariantSpec,
    seed: int,
    recent_session: AdaptiveHistorySessionSummary | None,
    previous_target_area: str | None,
    retention_repeat: bool,
) -> AdaptiveDrillCandidate:
    candidates = [
        candidate
        for candidate in ADAPTIVE_DRILL_CATALOG
        if candidate.primitive_id == primitive_id and role in candidate.role_support
    ]
    if not candidates:
        candidates = [
            candidate
            for candidate in ADAPTIVE_DRILL_CATALOG
            if candidate.primitive_id == primitive_id
        ]
    allowed = variant_spec.allowed_form_factors
    filtered = [candidate for candidate in candidates if candidate.form_factor in allowed]
    if filtered:
        candidates = filtered
    preferred_forms = _preferred_form_factors_for_role(
        role,
        variant_spec=variant_spec,
        retention_repeat=retention_repeat,
    )
    recent_drill_codes = set() if recent_session is None else set(recent_session.drill_codes)
    scored: list[tuple[float, str, AdaptiveDrillCandidate]] = []
    for candidate in candidates:
        score = 0.0
        if candidate.target_area == target_area:
            score += 5.0
        form_index = preferred_forms.index(candidate.form_factor) if candidate.form_factor in preferred_forms else len(preferred_forms)
        score += max(0.0, 2.0 - (0.5 * float(form_index)))
        if candidate.canonical_preferred:
            score += 1.0
        if candidate.drill_code in recent_drill_codes:
            score -= 0.8
        if (
            previous_target_area
            and previous_target_area == candidate.target_area
            and role not in {"reassessment_probe"}
            and not (role == "late_repeat_transfer" and retention_repeat)
        ):
            score -= 0.6
        score += _seed_tiebreak(candidate.drill_code, seed=seed)
        scored.append((score, candidate.drill_code, candidate))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return scored[0][2]


def build_adaptive_session_plan(
    *,
    history: list[AttemptHistoryEntry],
    seed: int,
    now_utc: str | None = None,
    recommended_level: int | None = None,
    fixed_mode: bool = False,
    variant: str = "full",
) -> AdaptiveSessionPlan | None:
    now_token = now_utc or _utc_now_iso()
    variant_spec = _variant_spec(variant)
    ranking = rank_primitives(history, now_utc=now_token, seed=seed)
    ranked = cast(tuple[AdaptivePriorityBreakdown, ...], ranking.weakest_primitives)
    if not any(item.evidence_count > 0 for item in ranked):
        return None
    ranked_by_id = {item.primitive_id: item for item in ranked}
    recent_sessions = _recent_adaptive_sessions(history)
    recent_session = recent_sessions[0] if recent_sessions else None
    target = _select_primary_target(ranked, recent_sessions=recent_sessions, now_utc=now_token)
    target_area = _target_area_for_primitive(target)
    target_base_level = clamp_level(recommended_level or int(target.recommended_level))
    adjacent_selection = _link_for_target_area(
        target.primitive_id,
        target_area=target_area,
        ranked_by_id=ranked_by_id,
        recent_sessions=recent_sessions,
        now_utc=now_token,
    )
    adjacent_link = None if adjacent_selection is None else adjacent_selection[0]
    adjacent_area = "" if adjacent_selection is None else adjacent_selection[1]
    adjacent_item = None if adjacent_link is None else ranked_by_id.get(adjacent_link.linked_primitive_id)
    retention_repeat = _retention_repeat_needed(target, now_utc=now_token)
    blocks: list[AdaptiveSessionBlock] = []
    previous_target_area: str | None = None
    for block_index, role in enumerate(_ROLE_SEQUENCE):
        if role in {"target_anchor", "target_tempo", "reassessment_probe", "target_pressure_fatigue"}:
            block_item = target
            block_primitive = _PRIMITIVE_BY_ID[target.primitive_id]
            block_target_area = target_area
            linked_primitive_id = None
            candidate_retention_repeat = False
            block_adjacent_item = None
        elif role == "adjacent_cross_train":
            if adjacent_item is None:
                continue
            block_item = adjacent_item
            block_primitive = _PRIMITIVE_BY_ID[adjacent_item.primitive_id]
            block_target_area = adjacent_area or _target_area_for_primitive(adjacent_item)
            linked_primitive_id = target.primitive_id
            candidate_retention_repeat = False
            block_adjacent_item = adjacent_item
        else:
            if retention_repeat:
                block_item = target
                block_primitive = _PRIMITIVE_BY_ID[target.primitive_id]
                block_target_area = target_area
                linked_primitive_id = None
                candidate_retention_repeat = True
                block_adjacent_item = None
            else:
                late_link_selection = _link_for_target_area(
                    target.primitive_id,
                    target_area=target_area,
                    ranked_by_id=ranked_by_id,
                    recent_sessions=recent_sessions,
                    now_utc=now_token,
                )
                late_link = None if late_link_selection is None else late_link_selection[0]
                late_area = "" if late_link_selection is None else late_link_selection[1]
                late_item = adjacent_item if adjacent_item is not None else None
                if late_link is not None and late_link.linked_primitive_id in ranked_by_id:
                    late_item = ranked_by_id[late_link.linked_primitive_id]
                if late_item is None:
                    late_item = target
                block_item = late_item
                block_primitive = _PRIMITIVE_BY_ID[late_item.primitive_id]
                block_target_area = late_area or _target_area_for_primitive(late_item)
                linked_primitive_id = target.primitive_id if late_item.primitive_id != target.primitive_id else None
                candidate_retention_repeat = False
                block_adjacent_item = late_item if late_item.primitive_id != target.primitive_id else None

        difficulty_level, comparable_level = _level_for_role(
            role=role,
            target_item=target,
            target_base_level=target_base_level,
            adjacent_item=block_adjacent_item,
            retention_repeat=candidate_retention_repeat,
        )
        candidate = _select_candidate(
            primitive_id=block_item.primitive_id,
            role=role,
            target_area=block_target_area,
            variant_spec=variant_spec,
            seed=int(seed) + (block_index * 17),
            recent_session=recent_session,
            previous_target_area=previous_target_area,
            retention_repeat=candidate_retention_repeat,
        )
        drill_code = resolved_canonical_drill_code(candidate.drill_code, for_adaptive=True) or candidate.drill_code
        drill_mode = _training_mode_for_role(
            drill_code,
            role=role,
            target_confidence=float(target.confidence if role != "adjacent_cross_train" else min(target.confidence, block_item.confidence)),
            fatigue_dominant=_pressure_is_fatigue_dominant(target),
            retention_repeat=candidate_retention_repeat,
        )
        blocks.append(
            AdaptiveSessionBlock(
                block_index=block_index,
                primitive_id=block_primitive.primitive_id,
                primitive_label=block_primitive.label,
                drill_code=drill_code,
                mode=role,
                duration_s=variant_spec.block_duration_s,
                difficulty_level=difficulty_level,
                seed=int(seed) + ((block_index + 1) * 131),
                reason_tags=tuple(block_item.reason_tags),
                priority=float(block_item.priority),
                drill_mode=drill_mode,
                form_factor=candidate.form_factor,
                target_area=block_target_area,
                linked_primitive_id=linked_primitive_id,
                comparable_level=comparable_level,
            )
        )
        previous_target_area = block_target_area

    title = variant_spec.title
    description = (
        "Built from recent benchmark and training history. "
        "This session uses a fixed six-block sequence: target anchor, target tempo, adjacent cross-train, "
        "reassessment probe, target under pressure or fatigue, and a late repeat or mixed transfer."
    )
    if fixed_mode:
        description += " This launch is using fixed mode."
    notes = tuple(
        f"{item.label}: priority {item.priority:.2f} [{', '.join(item.reason_tags)}]"
        for item in ranked[:3]
    )
    return AdaptiveSessionPlan(
        code=variant_spec.code,
        title=title,
        version=_SCHEDULER_VERSION,
        generated_at_utc=now_token,
        description=description,
        notes=notes,
        ranked_primitives=tuple(ranked),
        variant=variant_spec.variant,
        domain_summaries=tuple(ranking.domain_summaries),
        blocks=tuple(blocks),
    )


class AdaptiveSession:
    _NOMINAL_DIFFICULTY = 0.5

    def __init__(
        self,
        *,
        clock: Clock,
        seed: int,
        plan: AdaptiveSessionPlan,
        difficulty_mode: LaunchDifficultyMode = "adaptive",
        difficulty_context: ResolvedDifficultyContext | None = None,
    ) -> None:
        self._clock = clock
        self._seed = int(seed)
        self._plan = plan
        self._difficulty_mode = "adaptive" if difficulty_mode == "adaptive" else "fixed"
        baseline_level = 5 if not self._plan.blocks else int(self._plan.blocks[0].difficulty_level)
        self._difficulty_context = difficulty_context or build_resolved_difficulty_context(
            self._plan.code,
            mode=cast(LaunchDifficultyMode, self._difficulty_mode),
            launch_level=baseline_level,
            fixed_level=baseline_level,
            adaptive_enabled=(self._difficulty_mode == "adaptive"),
        )
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
                "This block was chosen from weakness, fatigue, lapses, interference, and adjacent transfer data.",
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
            "adaptive_mode": str(self._difficulty_mode),
            "adaptive_start_level": str(int(self._difficulty_context.launch_level)),
            "adaptive_end_level": str(
                int(self.scored_summary().difficulty_level_end or self._difficulty_context.launch_level)
            ),
            "adaptive_change_count": str(int(self.scored_summary().difficulty_change_count)),
            "adaptive_scope_code": self._difficulty_context.code_scope_key,
            "adaptive_scope_primitive": self._difficulty_context.primitive_scope_key,
        }
        for breakdown in self._plan.ranked_primitives:
            metrics[f"scheduler.priority.{breakdown.primitive_id}"] = f"{breakdown.priority:.6f}"
            metrics[f"scheduler.reason.{breakdown.primitive_id}"] = ",".join(breakdown.reason_tags)
            metrics[f"scheduler.recommended_level.{breakdown.primitive_id}"] = str(
                int(breakdown.recommended_level)
            )
            metrics[f"scheduler.level_confidence.{breakdown.primitive_id}"] = (
                f"{breakdown.level_confidence:.6f}"
            )
        for summary in self._plan.domain_summaries:
            prefix = f"scheduler.domain.{summary.domain_id}."
            metrics[f"{prefix}priority_sum"] = f"{summary.priority_sum:.6f}"
            metrics[f"{prefix}mean_composite_score"] = f"{summary.mean_composite_score:.6f}"
            metrics[f"{prefix}mean_confidence"] = f"{summary.mean_confidence:.6f}"
            metrics[f"{prefix}weakest_primitive_id"] = summary.weakest_primitive_id

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
            metrics[f"{prefix}form_factor"] = block.form_factor
            metrics[f"{prefix}target_area"] = block.target_area
            metrics[f"{prefix}linked_primitive_id"] = block.linked_primitive_id or ""
            metrics[f"{prefix}comparable_level"] = (
                "" if block.comparable_level is None else str(int(block.comparable_level))
            )
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
                "training_mode": block.drill_mode.value,
                "form_factor": block.form_factor,
                "target_area": block.target_area,
                "linked_primitive_id": block.linked_primitive_id or "",
                "comparable_level": (
                    None if block.comparable_level is None else int(block.comparable_level)
                ),
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
            setattr(engine, "_difficulty_code", str(block.drill_code))
            setattr(
                engine,
                "_resolved_difficulty_context",
                self._block_difficulty_context(block),
            )
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
        engine = build_workout_block_engine(
            clock=self._clock,
            block_seed=block.seed,
            difficulty_level=block.difficulty_level,
            block=workout_block,
            block_index=0,
        )
        setattr(engine, "_difficulty_code", str(block.drill_code))
        setattr(engine, "_resolved_difficulty_context", self._block_difficulty_context(block))
        return engine

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
            "Block order: anchor -> tempo -> adjacent cross-train -> probe -> pressure/fatigue -> late repeat/transfer.",
            "",
            "Ranked primitives:",
        ]
        for breakdown in self._plan.ranked_primitives[:3]:
            lines.append(
                f"{breakdown.label}: {breakdown.priority:.2f} [{', '.join(breakdown.reason_tags)}], "
                f"level {breakdown.recommended_level} ({breakdown.level_confidence:.2f})"
            )
        if self._plan.domain_summaries:
            lines.append("")
            lines.append("Domain rollup:")
            for summary in self._plan.domain_summaries[:3]:
                lines.append(
                    f"{summary.label}: {summary.priority_sum:.2f} -> {summary.weakest_primitive_label}"
                )
        lines.append("")
        lines.append("Planned blocks:")
        for block in self._plan.blocks:
            lines.append(
                f"{block.block_index + 1}. {block.primitive_label} -> {block.drill_code} "
                f"({block.mode}, L{block.difficulty_level}, {block.form_factor}, {block.target_area})"
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
                f"lapse {breakdown.lapse:.2f}, control {breakdown.control:.2f}, retention {breakdown.retention:.2f}"
            )
        return tuple(lines)

    def _block_difficulty_context(self, block: AdaptiveSessionBlock) -> ResolvedDifficultyContext:
        return build_resolved_difficulty_context(
            block.drill_code,
            mode=cast(LaunchDifficultyMode, self._difficulty_mode),
            launch_level=int(block.difficulty_level),
            fixed_level=int(block.difficulty_level),
            adaptive_enabled=(self._difficulty_mode == "adaptive"),
        )

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
