from __future__ import annotations

from cfast_trainer.canonical_drill_registry import (
    CANONICAL_DRILL_REGISTRY,
    canonical_drill_convergence,
    canonical_replacement_for_drill,
    canonical_drill_spec,
    canonical_wave_drills,
    difficulty_axes_for_drill,
    difficulty_family_for_drill,
    drills_for_subskill,
    is_hidden_redundant_drill,
    resolved_canonical_drill_code,
    resolved_canonical_drill_spec,
    supported_granularities_for_subskill,
)
from cfast_trainer.guide_skill_catalog import subskill_coverage_expectations


def test_wave1_registry_contains_expected_high_leverage_codes() -> None:
    codes = {spec.code for spec in canonical_wave_drills(wave=1)}
    assert {
        "ma_percentage_snap",
        "ma_written_numerical_extraction",
        "ma_rate_time_distance",
        "ma_fuel_endurance",
        "vs_multi_target_class_search",
        "vs_priority_switch_search",
        "vs_matrix_routine_priority_switch",
        "dr_visual_digit_query",
        "dr_recall_after_interference",
        "dtb_tracking_recall",
        "dtb_tracking_command_filter",
        "dtb_tracking_filter_digit_report",
        "dtb_tracking_interference_recovery",
    } <= codes


def test_registry_resolves_new_wave1_builders() -> None:
    for code in (
        "ma_written_numerical_extraction",
        "vs_multi_target_class_search",
        "vs_priority_switch_search",
        "vs_matrix_routine_priority_switch",
        "dr_visual_digit_query",
        "dr_recall_after_interference",
    ):
        spec = canonical_drill_spec(code)
        assert spec is not None
        assert spec.coverage == "new"
        assert callable(spec.resolve_builder())


def test_wave2_registry_contains_expected_spatial_psychomotor_codes() -> None:
    codes = {spec.code for spec in canonical_wave_drills(wave=2)}
    assert {
        "abd_angle_anchor",
        "abd_angle_tempo",
        "abd_bearing_anchor",
        "abd_bearing_tempo",
        "ic_instrument_attitude_matching",
        "si_static_multiview_integration",
        "si_moving_aircraft_multiview_integration",
        "trace_orientation_decode",
        "trace_movement_recall",
        "sma_split_axis_control",
        "sma_overshoot_recovery",
        "rt_obscured_target_prediction",
    } <= codes


def test_registry_resolves_new_wave2_builders() -> None:
    for code in (
        "abd_angle_anchor",
        "abd_bearing_anchor",
        "si_moving_aircraft_multiview_integration",
        "sma_split_axis_control",
        "sma_overshoot_recovery",
        "rt_obscured_target_prediction",
    ):
        spec = canonical_drill_spec(code)
        assert spec is not None
        assert spec.coverage == "new"
        assert callable(spec.resolve_builder())


def test_registry_entries_have_unique_codes() -> None:
    codes = [spec.code for spec in CANONICAL_DRILL_REGISTRY]
    assert len(codes) == len(set(codes))


def test_replacement_aliases_resolve_to_canonical_specs() -> None:
    assert canonical_drill_convergence("ic_attitude_frame") is not None
    assert resolved_canonical_drill_code("ic_attitude_frame") == "ic_instrument_attitude_matching"

    spec = resolved_canonical_drill_spec("tt1_command_switch_run")

    assert spec is not None
    assert spec.code == "trace_orientation_decode"
    assert spec.builder_name == "build_tt1_command_switch_run_drill"


def test_hidden_redundant_drills_stay_executable_but_do_not_resolve_as_canonical_specs() -> None:
    assert is_hidden_redundant_drill("vs_target_preview") is True
    assert resolved_canonical_drill_code("vs_target_preview") == "vs_target_preview"
    assert resolved_canonical_drill_code("vs_target_preview", for_adaptive=True) == "vs_multi_target_class_search"
    assert resolved_canonical_drill_spec("vs_target_preview") is None


def test_registry_entries_cover_catalog_linked_subskills() -> None:
    for subskill_id in subskill_coverage_expectations():
        assert drills_for_subskill(subskill_id)


def test_registry_compatibility_aliases_and_helpers_remain_stable() -> None:
    spec = canonical_drill_spec("ma_percentage_snap")

    assert spec is not None
    assert spec.code == spec.drill_code
    assert spec.label == spec.title
    assert spec.family_id == spec.difficulty_family_id
    assert canonical_replacement_for_drill("ic_attitude_frame") == "ic_instrument_attitude_matching"
    assert supported_granularities_for_subskill("quantitative_core") == ("block_component", "micro")
    assert difficulty_family_for_drill("visual_search") == "search_vigilance"
    assert "time_pressure" in difficulty_axes_for_drill("vs_multi_target_class_search")
