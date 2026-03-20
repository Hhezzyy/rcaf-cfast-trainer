from __future__ import annotations

from dataclasses import fields

import pytest

from cfast_trainer.adaptive_difficulty import (
    AdaptiveDifficultyController,
    AdaptiveDifficultyState,
    DifficultyAxes,
    build_resolved_difficulty_context,
    difficulty_profile_for_family,
    difficulty_ratio_for_level,
    family_id_for_code,
    next_level_from_performance,
    policy_for_activity_kind,
    recommended_level_for_primitive,
)
from cfast_trainer.airborne_numerical import build_ant_airborne_difficulty_profile
from cfast_trainer.guide_skill_catalog import OfficialGuideTestSpec
from cfast_trainer.rapid_tracking import build_rapid_tracking_test


_FAMILIES = (
    "quantitative",
    "angle_bearing",
    "auditory_multitask",
    "cln_multitask",
    "instrument_orientation",
    "visual_memory_updating",
    "situational_awareness",
    "table_cross_reference",
    "system_logic",
    "search_vigilance",
    "psychomotor_tracking",
    "spatial_integration_trace",
)

_REPLACEMENT_PAIRS = (
    ("abd_angle_calibration", "abd_angle_tempo"),
    ("abd_bearing_calibration", "abd_bearing_tempo"),
    ("ic_attitude_frame", "ic_instrument_attitude_matching"),
    ("si_static_mixed_run", "si_static_multiview_integration"),
    ("si_aircraft_multiview_integration", "si_moving_aircraft_multiview_integration"),
    ("tt1_command_switch_run", "trace_orientation_decode"),
    ("tt2_position_recall_run", "trace_movement_recall"),
)


class _FakeClock:
    def __init__(self) -> None:
        self._now = 0.0

    def now(self) -> float:
        return self._now


@pytest.mark.parametrize("family_id", _FAMILIES)
def test_difficulty_profiles_are_deterministic_and_monotonic(family_id: str) -> None:
    a = difficulty_profile_for_family(family_id, 6, "build")
    b = difficulty_profile_for_family(family_id, 6, "build")
    low = difficulty_profile_for_family(family_id, 1, "build")
    high = difficulty_profile_for_family(family_id, 10, "build")

    assert a == b
    for axis in fields(DifficultyAxes):
        low_value = getattr(low.axes, axis.name)
        high_value = getattr(high.axes, axis.name)
        assert high_value >= low_value


def test_mode_offsets_shift_time_pressure_and_distractors_by_intended_use() -> None:
    anchor = difficulty_profile_for_family("table_cross_reference", 5, "anchor")
    build = difficulty_profile_for_family("table_cross_reference", 5, "build")
    tempo = difficulty_profile_for_family("table_cross_reference", 5, "tempo")
    pressure = difficulty_profile_for_family("table_cross_reference", 5, "pressure")
    fatigue = difficulty_profile_for_family("table_cross_reference", 5, "fatigue_probe")

    assert anchor.axes.time_pressure < build.axes.time_pressure < tempo.axes.time_pressure
    assert tempo.axes.time_pressure <= pressure.axes.time_pressure
    assert anchor.axes.distractor_density < build.axes.distractor_density
    assert fatigue.axes.time_pressure >= build.axes.time_pressure
    assert fatigue.axes.distractor_density >= build.axes.distractor_density


def test_scope_keys_for_code_prefers_catalog_backed_primitive_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "cfast_trainer.adaptive_difficulty.guide_ranking_primitive_id_for_code",
        lambda code: "visual_scan_discipline" if code == "mystery_catalog_code" else None,
    )

    from cfast_trainer.adaptive_difficulty import scope_keys_for_code

    assert scope_keys_for_code("mystery_catalog_code") == (
        "mystery_catalog_code",
        "visual_scan_discipline",
    )


def test_family_id_for_code_prefers_catalog_backed_official_test_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = OfficialGuideTestSpec(
        official_name="Mystery Test",
        test_code="mystery_guide_test",
        guide_duration_min=10,
        guide_prepability="moderate",
        devices=("keyboard",),
        component_skills=("visual_scan_discipline",),
        component_subskills=("class_search",),
        difficulty_family_id="search_vigilance",
        difficulty_axes_used=("time_pressure",),
        difficulty_description_by_axis={"time_pressure": "Faster pacing."},
        difficulty_notes="Synthetic catalog mapping for tests.",
        cognitive_domain_id="scan_search_and_monitoring",
        test_family_id="scan_search",
        ranking_primitive_id="visual_scan_discipline",
    )
    monkeypatch.setattr(
        "cfast_trainer.adaptive_difficulty.official_guide_test",
        lambda code: fake if code == "mystery_guide_test" else None,
    )

    assert family_id_for_code("mystery_guide_test") == "search_vigilance"


def test_next_level_from_performance_applies_hysteresis_and_session_caps() -> None:
    assert (
        next_level_from_performance(
            current_level=5,
            launch_level=5,
            block_start_level=5,
            intended_use="build",
            recent_accuracy=1.0,
            recent_score_ratio=1.0,
            recent_timeout_rate=0.0,
            medium_accuracy=0.96,
            medium_score_ratio=0.96,
            medium_timeout_rate=0.0,
            max_delta=3,
        )
        == 6
    )
    assert (
        next_level_from_performance(
            current_level=7,
            launch_level=5,
            block_start_level=7,
            intended_use="pressure",
            recent_accuracy=1.0,
            recent_score_ratio=1.0,
            recent_timeout_rate=0.0,
            medium_accuracy=0.96,
            medium_score_ratio=0.96,
            medium_timeout_rate=0.0,
            max_delta=3,
        )
        == 7
    )
    assert (
        next_level_from_performance(
            current_level=6,
            launch_level=5,
            block_start_level=6,
            intended_use="fatigue_probe",
            recent_accuracy=1.0,
            recent_score_ratio=1.0,
            recent_timeout_rate=0.0,
            medium_accuracy=0.96,
            medium_score_ratio=0.96,
            medium_timeout_rate=0.0,
            max_delta=3,
        )
        == 6
    )
    assert (
        next_level_from_performance(
            current_level=5,
            launch_level=5,
            block_start_level=5,
            intended_use="tempo",
            recent_accuracy=0.0,
            recent_score_ratio=0.0,
            recent_timeout_rate=1.0,
            medium_accuracy=0.10,
            medium_score_ratio=0.10,
            medium_timeout_rate=0.75,
            max_delta=3,
        )
        == 4
    )


def test_next_level_from_performance_limits_gain_to_one_level_per_block() -> None:
    assert (
        next_level_from_performance(
            current_level=6,
            launch_level=5,
            block_start_level=5,
            intended_use="build",
            recent_accuracy=1.0,
            recent_score_ratio=1.0,
            recent_timeout_rate=0.0,
            medium_accuracy=0.96,
            medium_score_ratio=0.96,
            medium_timeout_rate=0.0,
            max_delta=3,
        )
        == 6
    )


def test_adaptive_controller_allows_immediate_meltdown_demote_before_throttle() -> None:
    controller = AdaptiveDifficultyController(
        launch_level=5,
        policy=policy_for_activity_kind("drill", intended_use="pressure"),
    )

    decision = controller.record_item(
        is_correct=False,
        is_timeout=True,
        score_ratio=0.0,
        response_time_ms=5000.0,
    )

    assert decision is not None
    assert decision.old_level == 5
    assert decision.new_level == 4


def test_adaptive_controller_limits_promotion_to_one_level_per_block() -> None:
    controller = AdaptiveDifficultyController(
        launch_level=5,
        policy=policy_for_activity_kind("drill", intended_use="build"),
    )

    decisions = []
    for _ in range(12):
        decision = controller.record_item(
            is_correct=True,
            is_timeout=False,
            score_ratio=1.0,
            response_time_ms=500.0,
        )
        if decision is not None:
            decisions.append((decision.old_level, decision.new_level))

    assert decisions == [(5, 6)]
    assert controller.current_level == 6


def test_recommended_level_for_primitive_prefers_code_then_family_then_primitive() -> None:
    primitive_state = AdaptiveDifficultyState(
        scope_kind="primitive",
        scope_key="mental_arithmetic_automaticity",
        recommended_level=8,
        last_start_level=7,
        last_end_level=8,
        ewma_accuracy=0.8,
        ewma_score_ratio=0.8,
        ewma_timeout_rate=0.1,
        ewma_mean_rt_ms=900.0,
        sample_count=4,
        updated_at_utc="2026-03-18T00:00:00Z",
    )
    family_state = AdaptiveDifficultyState(
        scope_kind="family",
        scope_key=family_id_for_code("numerical_operations"),
        recommended_level=6,
        last_start_level=6,
        last_end_level=6,
        ewma_accuracy=0.85,
        ewma_score_ratio=0.85,
        ewma_timeout_rate=0.08,
        ewma_mean_rt_ms=850.0,
        sample_count=5,
        updated_at_utc="2026-03-18T00:00:00Z",
    )
    code_state = AdaptiveDifficultyState(
        scope_kind="code",
        scope_key="numerical_operations",
        recommended_level=4,
        last_start_level=4,
        last_end_level=4,
        ewma_accuracy=0.91,
        ewma_score_ratio=0.91,
        ewma_timeout_rate=0.02,
        ewma_mean_rt_ms=700.0,
        sample_count=6,
        updated_at_utc="2026-03-18T00:00:00Z",
    )

    assert (
        recommended_level_for_primitive(
            code_state=code_state,
            family_state=family_state,
            primitive_state=primitive_state,
            fallback_level=5,
        )
        == 4
    )
    assert (
        recommended_level_for_primitive(
            code_state=None,
            family_state=family_state,
            primitive_state=primitive_state,
            fallback_level=5,
        )
        == 6
    )
    assert (
        recommended_level_for_primitive(
            code_state=None,
            family_state=None,
            primitive_state=primitive_state,
            fallback_level=5,
        )
        == 8
    )


def test_airborne_numerical_profile_uses_shared_level_for_advanced_reference_modes() -> None:
    low = build_ant_airborne_difficulty_profile(1, family="full")
    high = build_ant_airborne_difficulty_profile(10, family="full")

    assert low.speed_minutes == (60,)
    assert low.fuel_minutes == (60,)
    assert low.parcel_reference_formats == ("table",)
    assert low.fuel_reference_formats == ("table",)

    assert high.speed_minutes == (60, 1)
    assert high.fuel_minutes == (60, 1)
    assert "chart" in high.parcel_reference_formats
    assert "chart" in high.fuel_reference_formats


def test_rapid_tracking_profile_derives_tier_and_pressure_from_shared_family_ladder() -> None:
    clock = _FakeClock()
    low = build_rapid_tracking_test(
        clock=clock,
        seed=121,
        difficulty=difficulty_ratio_for_level("rapid_tracking", 1),
    )._difficulty_profile()
    high = build_rapid_tracking_test(
        clock=clock,
        seed=122,
        difficulty=difficulty_ratio_for_level("rapid_tracking", 10),
    )._difficulty_profile()

    assert low.tier == "low"
    assert high.tier == "high"
    assert high.duration_scale < low.duration_scale
    assert high.turbulence_strength > low.turbulence_strength


@pytest.mark.parametrize(("legacy_code", "canonical_code"), _REPLACEMENT_PAIRS)
def test_replacement_aliases_preserve_level_five_difficulty_ratio(
    legacy_code: str,
    canonical_code: str,
) -> None:
    assert difficulty_ratio_for_level(legacy_code, 5) == pytest.approx(
        difficulty_ratio_for_level(canonical_code, 5)
    )


def test_replacement_aliases_share_canonical_code_scope_keys() -> None:
    context = build_resolved_difficulty_context(
        "ic_attitude_frame",
        mode="fixed",
        launch_level=5,
        fixed_level=5,
        adaptive_enabled=False,
    )

    assert context.test_code == "ic_attitude_frame"
    assert context.code_scope_key == "ic_instrument_attitude_matching"


@pytest.mark.parametrize(
    "code",
    ("adaptive_session", "adaptive_session_short", "adaptive_session_micro"),
)
def test_adaptive_session_variants_share_adaptive_scope_keys(code: str) -> None:
    context = build_resolved_difficulty_context(
        code,
        mode="adaptive",
        launch_level=5,
        fixed_level=5,
        adaptive_enabled=True,
    )

    assert context.test_code == code
    assert context.code_scope_key == "adaptive_session"
    assert context.primitive_scope_key == "adaptive_session"
