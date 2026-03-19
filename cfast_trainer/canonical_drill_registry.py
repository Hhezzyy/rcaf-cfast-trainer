from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Callable, Literal

from .adaptive_difficulty import DifficultyFamilyId


CanonicalDrillBuilder = Callable[..., object]
ConvergenceAction = Literal["replace_when_new_drill_exists", "keep_but_merge_later"]


@dataclass(frozen=True, slots=True)
class CanonicalDrillSpec:
    code: str
    label: str
    target_area: str
    family_id: DifficultyFamilyId
    builder_module: str
    builder_name: str
    wave: int = 1
    coverage: str = "direct"
    notes: str = ""

    def resolve_builder(self) -> CanonicalDrillBuilder:
        module = import_module(self.builder_module)
        builder = getattr(module, self.builder_name)
        return builder


@dataclass(frozen=True, slots=True)
class CanonicalDrillConvergence:
    legacy_code: str
    canonical_code: str
    action: ConvergenceAction
    hide_from_adaptive: bool = False


_WAVE_1_SPECS: tuple[CanonicalDrillSpec, ...] = (
    CanonicalDrillSpec(
        code="ma_percentage_snap",
        label="Percentage / Ratio Snap",
        target_area="quantitative",
        family_id="quantitative",
        builder_module="cfast_trainer.ma_drills",
        builder_name="build_ma_percentage_snap_drill",
        coverage="direct",
    ),
    CanonicalDrillSpec(
        code="ma_written_numerical_extraction",
        label="Written Numerical Extraction",
        target_area="quantitative",
        family_id="quantitative",
        builder_module="cfast_trainer.ma_drills",
        builder_name="build_ma_written_numerical_extraction_drill",
        coverage="new",
    ),
    CanonicalDrillSpec(
        code="ma_rate_time_distance",
        label="Time / Speed / Distance Solve",
        target_area="quantitative",
        family_id="quantitative",
        builder_module="cfast_trainer.ma_drills",
        builder_name="build_ma_rate_time_distance_drill",
        coverage="direct",
    ),
    CanonicalDrillSpec(
        code="ma_fuel_endurance",
        label="Fuel / Endurance Solve",
        target_area="quantitative",
        family_id="quantitative",
        builder_module="cfast_trainer.ma_drills",
        builder_name="build_ma_fuel_endurance_drill",
        coverage="direct",
    ),
    CanonicalDrillSpec(
        code="vs_multi_target_class_search",
        label="Multi-Target Class Search",
        target_area="scan_search_cross_reference",
        family_id="search_vigilance",
        builder_module="cfast_trainer.vs_drills",
        builder_name="build_vs_multi_target_class_search_drill",
        coverage="new",
    ),
    CanonicalDrillSpec(
        code="vs_priority_switch_search",
        label="Priority Switch Search",
        target_area="scan_search_cross_reference",
        family_id="search_vigilance",
        builder_module="cfast_trainer.vs_drills",
        builder_name="build_vs_priority_switch_search_drill",
        coverage="new",
    ),
    CanonicalDrillSpec(
        code="vs_matrix_routine_priority_switch",
        label="Matrix Scan + Routine/Priority Switching",
        target_area="scan_search_cross_reference",
        family_id="search_vigilance",
        builder_module="cfast_trainer.vs_drills",
        builder_name="build_vs_matrix_routine_priority_switch_drill",
        coverage="new",
    ),
    CanonicalDrillSpec(
        code="dr_visual_digit_query",
        label="Visual Digit Query",
        target_area="memory_interference",
        family_id="visual_memory_updating",
        builder_module="cfast_trainer.dr_drills",
        builder_name="build_dr_visual_digit_query_drill",
        coverage="new",
    ),
    CanonicalDrillSpec(
        code="dr_recall_after_interference",
        label="Recall After Interference",
        target_area="memory_interference",
        family_id="visual_memory_updating",
        builder_module="cfast_trainer.dr_drills",
        builder_name="build_dr_recall_after_interference_drill",
        coverage="new",
    ),
    CanonicalDrillSpec(
        code="dtb_tracking_recall",
        label="Tracking + Recall",
        target_area="bridge_multitask",
        family_id="psychomotor_tracking",
        builder_module="cfast_trainer.dtb_drills",
        builder_name="build_dtb_tracking_recall_drill",
        coverage="direct",
    ),
    CanonicalDrillSpec(
        code="dtb_tracking_command_filter",
        label="Tracking + Command Filter",
        target_area="bridge_multitask",
        family_id="psychomotor_tracking",
        builder_module="cfast_trainer.dtb_drills",
        builder_name="build_dtb_tracking_command_filter_drill",
        coverage="direct",
    ),
    CanonicalDrillSpec(
        code="dtb_tracking_filter_digit_report",
        label="Tracking + Filtered Digit Report",
        target_area="bridge_multitask",
        family_id="psychomotor_tracking",
        builder_module="cfast_trainer.dtb_drills",
        builder_name="build_dtb_tracking_filter_digit_report_drill",
        coverage="direct",
    ),
    CanonicalDrillSpec(
        code="dtb_tracking_interference_recovery",
        label="Tracking + Interference Recovery",
        target_area="bridge_multitask",
        family_id="psychomotor_tracking",
        builder_module="cfast_trainer.dtb_drills",
        builder_name="build_dtb_tracking_interference_recovery_drill",
        coverage="direct",
    ),
)

_WAVE_2_SPECS: tuple[CanonicalDrillSpec, ...] = (
    CanonicalDrillSpec(
        code="abd_angle_anchor",
        label="Angle Anchor",
        target_area="angle_bearing_judgment",
        family_id="angle_bearing",
        builder_module="cfast_trainer.abd_drills",
        builder_name="build_abd_angle_anchor_drill",
        wave=2,
        coverage="new",
    ),
    CanonicalDrillSpec(
        code="abd_angle_tempo",
        label="Angle Tempo",
        target_area="angle_bearing_judgment",
        family_id="angle_bearing",
        builder_module="cfast_trainer.abd_drills",
        builder_name="build_abd_angle_calibration_drill",
        wave=2,
        coverage="direct",
        notes="Canonical angle-tempo mapping uses the existing angle calibration run.",
    ),
    CanonicalDrillSpec(
        code="abd_bearing_anchor",
        label="Bearing Anchor",
        target_area="angle_bearing_judgment",
        family_id="angle_bearing",
        builder_module="cfast_trainer.abd_drills",
        builder_name="build_abd_bearing_anchor_drill",
        wave=2,
        coverage="new",
    ),
    CanonicalDrillSpec(
        code="abd_bearing_tempo",
        label="Bearing Tempo",
        target_area="angle_bearing_judgment",
        family_id="angle_bearing",
        builder_module="cfast_trainer.abd_drills",
        builder_name="build_abd_bearing_calibration_drill",
        wave=2,
        coverage="direct",
        notes="Canonical bearing-tempo mapping uses the existing bearing calibration run.",
    ),
    CanonicalDrillSpec(
        code="ic_instrument_attitude_matching",
        label="Instrument Attitude Matching",
        target_area="instrument_orientation",
        family_id="instrument_orientation",
        builder_module="cfast_trainer.ic_drills",
        builder_name="build_ic_attitude_frame_drill",
        wave=2,
        coverage="direct",
    ),
    CanonicalDrillSpec(
        code="si_static_multiview_integration",
        label="Static Multiview Integration",
        target_area="multi_view_spatial_integration",
        family_id="spatial_integration_trace",
        builder_module="cfast_trainer.si_drills",
        builder_name="build_si_static_mixed_run_drill",
        wave=2,
        coverage="direct",
    ),
    CanonicalDrillSpec(
        code="si_moving_aircraft_multiview_integration",
        label="Moving-Aircraft Multiview Integration",
        target_area="multi_view_spatial_integration",
        family_id="spatial_integration_trace",
        builder_module="cfast_trainer.si_drills",
        builder_name="build_si_aircraft_multiview_integration_drill",
        wave=2,
        coverage="new",
    ),
    CanonicalDrillSpec(
        code="trace_orientation_decode",
        label="Trace Orientation Decode",
        target_area="trace_orientation_change",
        family_id="spatial_integration_trace",
        builder_module="cfast_trainer.trace_drills",
        builder_name="build_tt1_command_switch_run_drill",
        wave=2,
        coverage="direct",
    ),
    CanonicalDrillSpec(
        code="trace_movement_recall",
        label="Trace Movement Recall",
        target_area="three_d_movement_recall",
        family_id="spatial_integration_trace",
        builder_module="cfast_trainer.trace_drills",
        builder_name="build_tt2_position_recall_run_drill",
        wave=2,
        coverage="direct",
        notes="Canonical movement-recall mapping uses the existing TT2 position-recall run.",
    ),
    CanonicalDrillSpec(
        code="sma_split_axis_control",
        label="Split-Axis Control",
        target_area="joystick_pedal_subskills",
        family_id="psychomotor_tracking",
        builder_module="cfast_trainer.sma_drills",
        builder_name="build_sma_split_axis_control_drill",
        wave=2,
        coverage="new",
    ),
    CanonicalDrillSpec(
        code="sma_overshoot_recovery",
        label="Overshoot Recovery",
        target_area="joystick_pedal_subskills",
        family_id="psychomotor_tracking",
        builder_module="cfast_trainer.sma_drills",
        builder_name="build_sma_overshoot_recovery_drill",
        wave=2,
        coverage="new",
    ),
    CanonicalDrillSpec(
        code="rt_obscured_target_prediction",
        label="Obscured Target Prediction",
        target_area="obscured_motion_prediction",
        family_id="psychomotor_tracking",
        builder_module="cfast_trainer.rt_drills",
        builder_name="build_rt_obscured_target_prediction_drill",
        wave=2,
        coverage="new",
    ),
)

CANONICAL_DRILL_REGISTRY: tuple[CanonicalDrillSpec, ...] = _WAVE_1_SPECS + _WAVE_2_SPECS
CANONICAL_DRILL_BY_CODE = {spec.code: spec for spec in CANONICAL_DRILL_REGISTRY}

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


def _normalize_code(code: str | None) -> str:
    return str(code or "").strip().lower()


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
    return tuple(spec for spec in CANONICAL_DRILL_REGISTRY if spec.wave == requested)
