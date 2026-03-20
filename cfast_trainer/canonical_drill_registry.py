from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Callable, Literal

from .adaptive_difficulty import DifficultyFamilyId
from .guide_skill_catalog import OFFICIAL_GUIDE_TESTS, SUBSKILL_BY_ID, guide_subskill_ids_for_code


CanonicalDrillBuilder = Callable[..., object]
ConvergenceAction = Literal["replace_when_new_drill_exists", "keep_but_merge_later"]
CanonicalDrillGranularity = Literal["micro", "short", "block_component"]
CanonicalDrillStatus = Literal[
    "canonical_keep",
    "keep_but_merge_later",
    "replace_when_new_drill_exists",
    "whole_test_only",
]


_LEVELS_1_TO_10 = tuple(range(1, 11))
_DEFAULT_MODES = ("fresh", "build", "tempo", "pressure", "fatigue_probe")
_BUILD_PRESSURE_MODES = ("build", "tempo", "pressure", "fatigue_probe")
_FAMILY_AXES: dict[DifficultyFamilyId, tuple[str, ...]] = {
    "quantitative": (
        "content_complexity",
        "time_pressure",
        "source_integration_depth",
        "switch_frequency",
    ),
    "angle_bearing": ("spatial_ambiguity", "time_pressure"),
    "auditory_multitask": (
        "multitask_concurrency",
        "memory_span_delay",
        "distractor_density",
        "switch_frequency",
    ),
    "cln_multitask": (
        "multitask_concurrency",
        "memory_span_delay",
        "switch_frequency",
    ),
    "instrument_orientation": (
        "spatial_ambiguity",
        "source_integration_depth",
        "time_pressure",
    ),
    "visual_memory_updating": (
        "memory_span_delay",
        "time_pressure",
        "switch_frequency",
    ),
    "situational_awareness": (
        "source_integration_depth",
        "memory_span_delay",
        "switch_frequency",
    ),
    "table_cross_reference": (
        "time_pressure",
        "distractor_density",
        "source_integration_depth",
    ),
    "system_logic": (
        "content_complexity",
        "source_integration_depth",
        "distractor_density",
    ),
    "search_vigilance": (
        "time_pressure",
        "distractor_density",
        "switch_frequency",
    ),
    "psychomotor_tracking": (
        "control_sensitivity",
        "time_pressure",
        "multitask_concurrency",
        "spatial_ambiguity",
    ),
    "spatial_integration_trace": (
        "spatial_ambiguity",
        "source_integration_depth",
        "control_sensitivity",
        "memory_span_delay",
        "switch_frequency",
    ),
}
_GUIDE_TEST_LINKS_BY_SUBSKILL: dict[str, tuple[str, ...]] = {
    subskill_id: tuple(
        sorted(
            test.test_code
            for test in OFFICIAL_GUIDE_TESTS
            if subskill_id in test.component_subskills
        )
    )
    for subskill_id in SUBSKILL_BY_ID
}


@dataclass(frozen=True, slots=True)
class CanonicalDrillSpec:
    drill_code: str
    title: str
    granularity: CanonicalDrillGranularity
    primary_subskill: str
    secondary_subskills: tuple[str, ...]
    guide_test_links: tuple[str, ...]
    supports_modes: tuple[str, ...]
    status: CanonicalDrillStatus
    difficulty_family_id: DifficultyFamilyId
    difficulty_axes_used: tuple[str, ...]
    supported_levels: tuple[int, ...]
    default_start_level: int
    difficulty_progression_notes: str
    builder_module: str | None = None
    builder_name: str | None = None
    wave: int = 0
    coverage: str = "direct"
    target_area: str = ""

    @property
    def code(self) -> str:
        return self.drill_code

    @property
    def label(self) -> str:
        return self.title

    @property
    def family_id(self) -> DifficultyFamilyId:
        return self.difficulty_family_id

    @property
    def notes(self) -> str:
        return self.difficulty_progression_notes

    def resolve_builder(self) -> CanonicalDrillBuilder:
        if not self.builder_module or not self.builder_name:
            raise LookupError(f"{self.drill_code} does not expose a standalone builder")
        module = import_module(self.builder_module)
        builder = getattr(module, self.builder_name)
        return builder


@dataclass(frozen=True, slots=True)
class CanonicalDrillConvergence:
    legacy_code: str
    canonical_code: str
    action: ConvergenceAction
    hide_from_adaptive: bool = False


def _normalize_code(code: str | None) -> str:
    return str(code or "").strip().lower()


def _subskill_links(subskill_id: str) -> tuple[str, ...]:
    return _GUIDE_TEST_LINKS_BY_SUBSKILL.get(str(subskill_id).strip().lower(), ())


def _subskill_default_level(subskill_id: str) -> int:
    spec = SUBSKILL_BY_ID.get(str(subskill_id).strip().lower())
    return 3 if spec is None else int(spec.suggested_start_level)


def _builder_spec(
    *,
    code: str,
    title: str,
    granularity: CanonicalDrillGranularity,
    family_id: DifficultyFamilyId,
    builder_module: str,
    builder_name: str,
    wave: int = 0,
    coverage: str = "direct",
    modes: tuple[str, ...] = _DEFAULT_MODES,
    notes: str = "",
) -> CanonicalDrillSpec:
    subskills = guide_subskill_ids_for_code(code)
    primary_subskill = subskills[0] if subskills else code
    secondary = tuple(subskills[1:])
    target_area = primary_subskill
    return CanonicalDrillSpec(
        drill_code=code,
        title=title,
        granularity=granularity,
        primary_subskill=primary_subskill,
        secondary_subskills=secondary,
        guide_test_links=_subskill_links(primary_subskill),
        supports_modes=tuple(modes),
        status="canonical_keep",
        difficulty_family_id=family_id,
        difficulty_axes_used=_FAMILY_AXES[family_id],
        supported_levels=_LEVELS_1_TO_10,
        default_start_level=_subskill_default_level(primary_subskill),
        difficulty_progression_notes=notes or f"{title} scales cleanly across L1-L10 within the {family_id} family.",
        builder_module=builder_module,
        builder_name=builder_name,
        wave=int(wave),
        coverage=str(coverage),
        target_area=target_area,
    )


_CANONICAL_KEEP_SPECS: tuple[CanonicalDrillSpec, ...] = (
    _builder_spec(
        code="ma_one_step_fluency",
        title="One-Step Fluency",
        granularity="micro",
        family_id="quantitative",
        builder_module="cfast_trainer.ma_drills",
        builder_name="build_ma_one_step_fluency_drill",
    ),
    _builder_spec(
        code="ma_percentage_snap",
        title="Percentage / Ratio Snap",
        granularity="micro",
        family_id="quantitative",
        builder_module="cfast_trainer.ma_drills",
        builder_name="build_ma_percentage_snap_drill",
        wave=1,
    ),
    _builder_spec(
        code="ma_written_numerical_extraction",
        title="Written Numerical Extraction",
        granularity="micro",
        family_id="quantitative",
        builder_module="cfast_trainer.ma_drills",
        builder_name="build_ma_written_numerical_extraction_drill",
        wave=1,
        coverage="new",
    ),
    _builder_spec(
        code="ma_rate_time_distance",
        title="Time / Speed / Distance Solve",
        granularity="short",
        family_id="quantitative",
        builder_module="cfast_trainer.ma_drills",
        builder_name="build_ma_rate_time_distance_drill",
        wave=1,
    ),
    _builder_spec(
        code="ma_fuel_endurance",
        title="Fuel / Endurance Solve",
        granularity="short",
        family_id="quantitative",
        builder_module="cfast_trainer.ma_drills",
        builder_name="build_ma_fuel_endurance_drill",
        wave=1,
    ),
    _builder_spec(
        code="ma_mixed_conversion_caps",
        title="Mixed Conversion Caps",
        granularity="block_component",
        family_id="quantitative",
        builder_module="cfast_trainer.ma_drills",
        builder_name="build_ma_mixed_conversion_caps_drill",
        modes=_BUILD_PRESSURE_MODES,
    ),
    _builder_spec(
        code="tbl_single_lookup_anchor",
        title="Single Lookup Anchor",
        granularity="micro",
        family_id="table_cross_reference",
        builder_module="cfast_trainer.tbl_drills",
        builder_name="build_tbl_single_lookup_anchor_drill",
    ),
    _builder_spec(
        code="tbl_two_table_xref",
        title="Two-Table Cross Reference",
        granularity="short",
        family_id="table_cross_reference",
        builder_module="cfast_trainer.tbl_drills",
        builder_name="build_tbl_two_table_xref_drill",
    ),
    _builder_spec(
        code="tbl_distractor_grid",
        title="Distractor Grid",
        granularity="short",
        family_id="table_cross_reference",
        builder_module="cfast_trainer.tbl_drills",
        builder_name="build_tbl_distractor_grid_drill",
        modes=_BUILD_PRESSURE_MODES,
    ),
    _builder_spec(
        code="tbl_lookup_compute",
        title="Lookup + Compute",
        granularity="block_component",
        family_id="table_cross_reference",
        builder_module="cfast_trainer.tbl_drills",
        builder_name="build_tbl_lookup_compute_drill",
        modes=_BUILD_PRESSURE_MODES,
    ),
    _builder_spec(
        code="tbl_shrinking_cap_run",
        title="Shrinking Cap Run",
        granularity="block_component",
        family_id="table_cross_reference",
        builder_module="cfast_trainer.tbl_drills",
        builder_name="build_tbl_shrinking_cap_run_drill",
        modes=_BUILD_PRESSURE_MODES,
    ),
    _builder_spec(
        code="sl_one_rule_identify",
        title="One-Rule Identify",
        granularity="micro",
        family_id="system_logic",
        builder_module="cfast_trainer.sl_drills",
        builder_name="build_sl_one_rule_identify_drill",
    ),
    _builder_spec(
        code="sl_rule_match",
        title="Rule Match",
        granularity="short",
        family_id="system_logic",
        builder_module="cfast_trainer.sl_drills",
        builder_name="build_sl_rule_match_drill",
    ),
    _builder_spec(
        code="sl_two_source_reconcile",
        title="Two-Source Reconcile",
        granularity="short",
        family_id="system_logic",
        builder_module="cfast_trainer.sl_drills",
        builder_name="build_sl_two_source_reconcile_drill",
    ),
    _builder_spec(
        code="sl_missing_step_complete",
        title="Missing-Step Complete",
        granularity="block_component",
        family_id="system_logic",
        builder_module="cfast_trainer.sl_drills",
        builder_name="build_sl_missing_step_complete_drill",
        modes=_BUILD_PRESSURE_MODES,
    ),
    _builder_spec(
        code="sl_fast_reject",
        title="Fast Reject",
        granularity="block_component",
        family_id="system_logic",
        builder_module="cfast_trainer.sl_drills",
        builder_name="build_sl_fast_reject_drill",
        modes=_BUILD_PRESSURE_MODES,
    ),
    _builder_spec(
        code="vs_multi_target_class_search",
        title="Multi-Target Class Search",
        granularity="micro",
        family_id="search_vigilance",
        builder_module="cfast_trainer.vs_drills",
        builder_name="build_vs_multi_target_class_search_drill",
        wave=1,
        coverage="new",
    ),
    _builder_spec(
        code="vs_priority_switch_search",
        title="Priority Switch Search",
        granularity="short",
        family_id="search_vigilance",
        builder_module="cfast_trainer.vs_drills",
        builder_name="build_vs_priority_switch_search_drill",
        wave=1,
        coverage="new",
    ),
    _builder_spec(
        code="vs_matrix_routine_priority_switch",
        title="Matrix Scan + Routine/Priority Switching",
        granularity="block_component",
        family_id="search_vigilance",
        builder_module="cfast_trainer.vs_drills",
        builder_name="build_vs_matrix_routine_priority_switch_drill",
        wave=1,
        coverage="new",
        modes=_BUILD_PRESSURE_MODES,
    ),
    _builder_spec(
        code="dr_visual_digit_query",
        title="Visual Digit Query",
        granularity="micro",
        family_id="visual_memory_updating",
        builder_module="cfast_trainer.dr_drills",
        builder_name="build_dr_visual_digit_query_drill",
        wave=1,
        coverage="new",
    ),
    _builder_spec(
        code="dr_recall_after_interference",
        title="Recall After Interference",
        granularity="short",
        family_id="visual_memory_updating",
        builder_module="cfast_trainer.dr_drills",
        builder_name="build_dr_recall_after_interference_drill",
        wave=1,
        coverage="new",
    ),
    _builder_spec(
        code="dtb_tracking_recall",
        title="Tracking + Recall",
        granularity="micro",
        family_id="psychomotor_tracking",
        builder_module="cfast_trainer.dtb_drills",
        builder_name="build_dtb_tracking_recall_drill",
        wave=1,
    ),
    _builder_spec(
        code="dtb_tracking_command_filter",
        title="Tracking + Command Filter",
        granularity="short",
        family_id="psychomotor_tracking",
        builder_module="cfast_trainer.dtb_drills",
        builder_name="build_dtb_tracking_command_filter_drill",
        wave=1,
    ),
    _builder_spec(
        code="dtb_tracking_filter_digit_report",
        title="Tracking + Filtered Digit Report",
        granularity="short",
        family_id="psychomotor_tracking",
        builder_module="cfast_trainer.dtb_drills",
        builder_name="build_dtb_tracking_filter_digit_report_drill",
        wave=1,
    ),
    _builder_spec(
        code="dtb_tracking_interference_recovery",
        title="Tracking + Interference Recovery",
        granularity="block_component",
        family_id="psychomotor_tracking",
        builder_module="cfast_trainer.dtb_drills",
        builder_name="build_dtb_tracking_interference_recovery_drill",
        wave=1,
        modes=_BUILD_PRESSURE_MODES,
    ),
    _builder_spec(
        code="abd_angle_anchor",
        title="Angle Anchor",
        granularity="micro",
        family_id="angle_bearing",
        builder_module="cfast_trainer.abd_drills",
        builder_name="build_abd_angle_anchor_drill",
        wave=2,
        coverage="new",
    ),
    _builder_spec(
        code="abd_angle_tempo",
        title="Angle Tempo",
        granularity="short",
        family_id="angle_bearing",
        builder_module="cfast_trainer.abd_drills",
        builder_name="build_abd_angle_calibration_drill",
        wave=2,
        notes="Canonical angle-tempo routing uses the existing angle calibration builder.",
    ),
    _builder_spec(
        code="abd_bearing_anchor",
        title="Bearing Anchor",
        granularity="micro",
        family_id="angle_bearing",
        builder_module="cfast_trainer.abd_drills",
        builder_name="build_abd_bearing_anchor_drill",
        wave=2,
        coverage="new",
    ),
    _builder_spec(
        code="abd_bearing_tempo",
        title="Bearing Tempo",
        granularity="short",
        family_id="angle_bearing",
        builder_module="cfast_trainer.abd_drills",
        builder_name="build_abd_bearing_calibration_drill",
        wave=2,
        notes="Canonical bearing-tempo routing uses the existing bearing calibration builder.",
    ),
    _builder_spec(
        code="ic_instrument_attitude_matching",
        title="Instrument Attitude Matching",
        granularity="short",
        family_id="instrument_orientation",
        builder_module="cfast_trainer.ic_drills",
        builder_name="build_ic_attitude_frame_drill",
        wave=2,
    ),
    _builder_spec(
        code="si_static_multiview_integration",
        title="Static Multiview Integration",
        granularity="short",
        family_id="spatial_integration_trace",
        builder_module="cfast_trainer.si_drills",
        builder_name="build_si_static_mixed_run_drill",
        wave=2,
    ),
    _builder_spec(
        code="si_moving_aircraft_multiview_integration",
        title="Moving-Aircraft Multiview Integration",
        granularity="block_component",
        family_id="spatial_integration_trace",
        builder_module="cfast_trainer.si_drills",
        builder_name="build_si_aircraft_multiview_integration_drill",
        wave=2,
        coverage="new",
        modes=_BUILD_PRESSURE_MODES,
    ),
    _builder_spec(
        code="trace_orientation_decode",
        title="Trace Orientation Decode",
        granularity="short",
        family_id="spatial_integration_trace",
        builder_module="cfast_trainer.trace_drills",
        builder_name="build_tt1_command_switch_run_drill",
        wave=2,
    ),
    _builder_spec(
        code="trace_movement_recall",
        title="Trace Movement Recall",
        granularity="short",
        family_id="spatial_integration_trace",
        builder_module="cfast_trainer.trace_drills",
        builder_name="build_tt2_position_recall_run_drill",
        wave=2,
        notes="Canonical movement-recall routing uses the existing TT2 position-recall builder.",
    ),
    _builder_spec(
        code="sma_split_axis_control",
        title="Split-Axis Control",
        granularity="micro",
        family_id="psychomotor_tracking",
        builder_module="cfast_trainer.sma_drills",
        builder_name="build_sma_split_axis_control_drill",
        wave=2,
        coverage="new",
    ),
    _builder_spec(
        code="sma_overshoot_recovery",
        title="Overshoot Recovery",
        granularity="short",
        family_id="psychomotor_tracking",
        builder_module="cfast_trainer.sma_drills",
        builder_name="build_sma_overshoot_recovery_drill",
        wave=2,
        coverage="new",
    ),
    _builder_spec(
        code="rt_obscured_target_prediction",
        title="Obscured Target Prediction",
        granularity="block_component",
        family_id="psychomotor_tracking",
        builder_module="cfast_trainer.rt_drills",
        builder_name="build_rt_obscured_target_prediction_drill",
        wave=2,
        coverage="new",
        modes=_BUILD_PRESSURE_MODES,
    ),
)


def _whole_test_only_specs() -> tuple[CanonicalDrillSpec, ...]:
    specs: list[CanonicalDrillSpec] = []
    for test in OFFICIAL_GUIDE_TESTS:
        primary_subskill = test.component_subskills[0] if test.component_subskills else test.test_code
        secondary_subskills = tuple(test.component_subskills[1:])
        default_start = min(
            (
                _subskill_default_level(subskill_id)
                for subskill_id in test.component_subskills
            ),
            default=3,
        )
        specs.append(
            CanonicalDrillSpec(
                drill_code=test.test_code,
                title=test.official_name,
                granularity="block_component",
                primary_subskill=primary_subskill,
                secondary_subskills=secondary_subskills,
                guide_test_links=(test.test_code,),
                supports_modes=("build",),
                status="whole_test_only",
                difficulty_family_id=test.difficulty_family_id,  # type: ignore[arg-type]
                difficulty_axes_used=tuple(test.difficulty_axes_used),
                supported_levels=_LEVELS_1_TO_10,
                default_start_level=default_start,
                difficulty_progression_notes=(
                    "Whole-test coverage only. Use the linked canonical drills for isolated "
                    "subskill work and level-specific progression probes."
                ),
                coverage="whole_test",
                target_area=primary_subskill,
            )
        )
    return tuple(specs)


CANONICAL_DRILL_REGISTRY: tuple[CanonicalDrillSpec, ...] = (
    _CANONICAL_KEEP_SPECS + _whole_test_only_specs()
)
CANONICAL_DRILL_BY_CODE = {spec.drill_code: spec for spec in CANONICAL_DRILL_REGISTRY}

CANONICAL_DRILL_CONVERGENCE: tuple[CanonicalDrillConvergence, ...] = (
    CanonicalDrillConvergence(
        legacy_code="abd_angle_calibration",
        canonical_code="abd_angle_tempo",
        action="replace_when_new_drill_exists",
    ),
    CanonicalDrillConvergence(
        legacy_code="abd_bearing_calibration",
        canonical_code="abd_bearing_tempo",
        action="replace_when_new_drill_exists",
    ),
    CanonicalDrillConvergence(
        legacy_code="ic_attitude_frame",
        canonical_code="ic_instrument_attitude_matching",
        action="replace_when_new_drill_exists",
    ),
    CanonicalDrillConvergence(
        legacy_code="si_static_mixed_run",
        canonical_code="si_static_multiview_integration",
        action="replace_when_new_drill_exists",
    ),
    CanonicalDrillConvergence(
        legacy_code="si_aircraft_multiview_integration",
        canonical_code="si_moving_aircraft_multiview_integration",
        action="replace_when_new_drill_exists",
    ),
    CanonicalDrillConvergence(
        legacy_code="tt1_command_switch_run",
        canonical_code="trace_orientation_decode",
        action="replace_when_new_drill_exists",
    ),
    CanonicalDrillConvergence(
        legacy_code="tt2_position_recall_run",
        canonical_code="trace_movement_recall",
        action="replace_when_new_drill_exists",
    ),
    CanonicalDrillConvergence(
        legacy_code="no_fact_prime",
        canonical_code="ma_one_step_fluency",
        action="keep_but_merge_later",
        hide_from_adaptive=True,
    ),
    CanonicalDrillConvergence(
        legacy_code="no_operator_ladders",
        canonical_code="ma_one_step_fluency",
        action="keep_but_merge_later",
        hide_from_adaptive=True,
    ),
    CanonicalDrillConvergence(
        legacy_code="no_clean_compute",
        canonical_code="ma_one_step_fluency",
        action="keep_but_merge_later",
        hide_from_adaptive=True,
    ),
    CanonicalDrillConvergence(
        legacy_code="no_mixed_tempo",
        canonical_code="ma_written_numerical_extraction",
        action="keep_but_merge_later",
        hide_from_adaptive=True,
    ),
    CanonicalDrillConvergence(
        legacy_code="no_pressure_run",
        canonical_code="ma_written_numerical_extraction",
        action="keep_but_merge_later",
        hide_from_adaptive=True,
    ),
    CanonicalDrillConvergence(
        legacy_code="mr_unit_relation_prime",
        canonical_code="ma_percentage_snap",
        action="keep_but_merge_later",
        hide_from_adaptive=True,
    ),
    CanonicalDrillConvergence(
        legacy_code="mr_one_step_solve",
        canonical_code="ma_rate_time_distance",
        action="keep_but_merge_later",
        hide_from_adaptive=True,
    ),
    CanonicalDrillConvergence(
        legacy_code="ant_snap_facts_sprint",
        canonical_code="ma_one_step_fluency",
        action="keep_but_merge_later",
        hide_from_adaptive=True,
    ),
    CanonicalDrillConvergence(
        legacy_code="ant_time_flip",
        canonical_code="ma_rate_time_distance",
        action="keep_but_merge_later",
        hide_from_adaptive=True,
    ),
    CanonicalDrillConvergence(
        legacy_code="ant_route_time_solve",
        canonical_code="ma_rate_time_distance",
        action="keep_but_merge_later",
        hide_from_adaptive=True,
    ),
    CanonicalDrillConvergence(
        legacy_code="ant_endurance_solve",
        canonical_code="ma_fuel_endurance",
        action="keep_but_merge_later",
        hide_from_adaptive=True,
    ),
    CanonicalDrillConvergence(
        legacy_code="ant_fuel_burn_solve",
        canonical_code="ma_fuel_endurance",
        action="keep_but_merge_later",
        hide_from_adaptive=True,
    ),
    CanonicalDrillConvergence(
        legacy_code="vs_target_preview",
        canonical_code="vs_multi_target_class_search",
        action="keep_but_merge_later",
        hide_from_adaptive=True,
    ),
    CanonicalDrillConvergence(
        legacy_code="vs_clean_scan",
        canonical_code="vs_multi_target_class_search",
        action="keep_but_merge_later",
        hide_from_adaptive=True,
    ),
    CanonicalDrillConvergence(
        legacy_code="vs_family_run_letters",
        canonical_code="vs_multi_target_class_search",
        action="keep_but_merge_later",
        hide_from_adaptive=True,
    ),
    CanonicalDrillConvergence(
        legacy_code="vs_family_run_symbols",
        canonical_code="vs_multi_target_class_search",
        action="keep_but_merge_later",
        hide_from_adaptive=True,
    ),
    CanonicalDrillConvergence(
        legacy_code="vs_mixed_tempo",
        canonical_code="vs_priority_switch_search",
        action="keep_but_merge_later",
        hide_from_adaptive=True,
    ),
    CanonicalDrillConvergence(
        legacy_code="vs_pressure_run",
        canonical_code="vs_matrix_routine_priority_switch",
        action="keep_but_merge_later",
        hide_from_adaptive=True,
    ),
    CanonicalDrillConvergence(
        legacy_code="rt_lock_anchor",
        canonical_code="rt_obscured_target_prediction",
        action="keep_but_merge_later",
        hide_from_adaptive=True,
    ),
    CanonicalDrillConvergence(
        legacy_code="rt_building_handoff_prime",
        canonical_code="rt_obscured_target_prediction",
        action="keep_but_merge_later",
        hide_from_adaptive=True,
    ),
    CanonicalDrillConvergence(
        legacy_code="rt_terrain_recovery_run",
        canonical_code="rt_obscured_target_prediction",
        action="keep_but_merge_later",
        hide_from_adaptive=True,
    ),
    CanonicalDrillConvergence(
        legacy_code="rt_capture_timing_prime",
        canonical_code="rt_obscured_target_prediction",
        action="keep_but_merge_later",
        hide_from_adaptive=True,
    ),
    CanonicalDrillConvergence(
        legacy_code="rt_ground_tempo_run",
        canonical_code="rt_obscured_target_prediction",
        action="keep_but_merge_later",
        hide_from_adaptive=True,
    ),
    CanonicalDrillConvergence(
        legacy_code="rt_air_speed_run",
        canonical_code="rt_obscured_target_prediction",
        action="keep_but_merge_later",
        hide_from_adaptive=True,
    ),
    CanonicalDrillConvergence(
        legacy_code="rt_mixed_tempo",
        canonical_code="rt_obscured_target_prediction",
        action="keep_but_merge_later",
        hide_from_adaptive=True,
    ),
    CanonicalDrillConvergence(
        legacy_code="rt_pressure_run",
        canonical_code="rt_obscured_target_prediction",
        action="keep_but_merge_later",
        hide_from_adaptive=True,
    ),
    CanonicalDrillConvergence(
        legacy_code="sma_joystick_horizontal_anchor",
        canonical_code="sma_split_axis_control",
        action="keep_but_merge_later",
        hide_from_adaptive=True,
    ),
    CanonicalDrillConvergence(
        legacy_code="sma_joystick_vertical_anchor",
        canonical_code="sma_split_axis_control",
        action="keep_but_merge_later",
        hide_from_adaptive=True,
    ),
    CanonicalDrillConvergence(
        legacy_code="sma_joystick_hold_run",
        canonical_code="sma_split_axis_control",
        action="keep_but_merge_later",
        hide_from_adaptive=True,
    ),
    CanonicalDrillConvergence(
        legacy_code="sma_split_horizontal_prime",
        canonical_code="sma_split_axis_control",
        action="keep_but_merge_later",
        hide_from_adaptive=True,
    ),
    CanonicalDrillConvergence(
        legacy_code="sma_split_coordination_run",
        canonical_code="sma_split_axis_control",
        action="keep_but_merge_later",
        hide_from_adaptive=True,
    ),
    CanonicalDrillConvergence(
        legacy_code="sma_mode_switch_run",
        canonical_code="sma_overshoot_recovery",
        action="keep_but_merge_later",
        hide_from_adaptive=True,
    ),
    CanonicalDrillConvergence(
        legacy_code="sma_disturbance_tempo",
        canonical_code="sma_overshoot_recovery",
        action="keep_but_merge_later",
        hide_from_adaptive=True,
    ),
    CanonicalDrillConvergence(
        legacy_code="sma_pressure_run",
        canonical_code="sma_overshoot_recovery",
        action="keep_but_merge_later",
        hide_from_adaptive=True,
    ),
    CanonicalDrillConvergence(
        legacy_code="sma_hold_center_lock",
        canonical_code="sma_split_axis_control",
        action="keep_but_merge_later",
        hide_from_adaptive=True,
    ),
    CanonicalDrillConvergence(
        legacy_code="sma_hold_band_horizontal",
        canonical_code="sma_split_axis_control",
        action="keep_but_merge_later",
        hide_from_adaptive=True,
    ),
    CanonicalDrillConvergence(
        legacy_code="sma_hold_band_vertical",
        canonical_code="sma_split_axis_control",
        action="keep_but_merge_later",
        hide_from_adaptive=True,
    ),
    CanonicalDrillConvergence(
        legacy_code="sma_recenter_run",
        canonical_code="sma_overshoot_recovery",
        action="keep_but_merge_later",
        hide_from_adaptive=True,
    ),
    CanonicalDrillConvergence(
        legacy_code="sma_turbulence_recovery",
        canonical_code="sma_overshoot_recovery",
        action="keep_but_merge_later",
        hide_from_adaptive=True,
    ),
    CanonicalDrillConvergence(
        legacy_code="sma_mixed_axis_run",
        canonical_code="sma_split_axis_control",
        action="keep_but_merge_later",
        hide_from_adaptive=True,
    ),
)
CANONICAL_DRILL_CONVERGENCE_BY_CODE = {
    entry.legacy_code: entry for entry in CANONICAL_DRILL_CONVERGENCE
}


def canonical_drill_spec(code: str | None) -> CanonicalDrillSpec | None:
    token = _normalize_code(code)
    if token == "":
        return None
    return CANONICAL_DRILL_BY_CODE.get(token)


def canonical_drill_convergence(code: str | None) -> CanonicalDrillConvergence | None:
    token = _normalize_code(code)
    if token == "":
        return None
    return CANONICAL_DRILL_CONVERGENCE_BY_CODE.get(token)


def resolved_canonical_drill_code(code: str | None, *, for_adaptive: bool = False) -> str | None:
    token = _normalize_code(code)
    if token == "":
        return None
    convergence = canonical_drill_convergence(token)
    if convergence is None:
        return token
    if convergence.action == "replace_when_new_drill_exists":
        return convergence.canonical_code
    if for_adaptive and convergence.hide_from_adaptive:
        return convergence.canonical_code
    return token


def resolved_canonical_drill_spec(code: str | None) -> CanonicalDrillSpec | None:
    token = resolved_canonical_drill_code(code)
    if token is None:
        return None
    return CANONICAL_DRILL_BY_CODE.get(token)


def is_hidden_redundant_drill(code: str | None) -> bool:
    convergence = canonical_drill_convergence(code)
    return bool(convergence is not None and convergence.hide_from_adaptive)


def canonical_wave_drills(*, wave: int = 1) -> tuple[CanonicalDrillSpec, ...]:
    requested = int(wave)
    return tuple(
        spec
        for spec in CANONICAL_DRILL_REGISTRY
        if spec.wave == requested and spec.status == "canonical_keep"
    )


def drills_for_subskill(subskill_id: str | None) -> tuple[CanonicalDrillSpec, ...]:
    token = str(subskill_id or "").strip().lower()
    if token == "":
        return ()
    matches = [
        spec
        for spec in CANONICAL_DRILL_REGISTRY
        if spec.primary_subskill == token or token in spec.secondary_subskills
    ]
    return tuple(sorted(matches, key=lambda spec: (spec.granularity, spec.drill_code)))


def canonical_replacement_for_drill(code: str | None) -> str | None:
    token = _normalize_code(code)
    if token == "":
        return None
    convergence = canonical_drill_convergence(token)
    if convergence is None:
        return token if token in CANONICAL_DRILL_BY_CODE else None
    return convergence.canonical_code


def supported_granularities_for_subskill(subskill_id: str | None) -> tuple[str, ...]:
    ordered: list[str] = []
    for spec in drills_for_subskill(subskill_id):
        if spec.granularity not in ordered:
            ordered.append(spec.granularity)
    return tuple(ordered)


def difficulty_family_for_drill(code: str | None) -> DifficultyFamilyId | None:
    spec = resolved_canonical_drill_spec(code)
    return None if spec is None else spec.difficulty_family_id


def difficulty_axes_for_drill(code: str | None) -> tuple[str, ...]:
    spec = resolved_canonical_drill_spec(code)
    return () if spec is None else tuple(spec.difficulty_axes_used)
