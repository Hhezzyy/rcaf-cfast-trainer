from __future__ import annotations

import pytest

from cfast_trainer.activity_runtime_catalog import (
    GODOT_ACTIVITY_RUNTIME_SOURCE,
    OFFICIAL_TEST_MENU_ORDER,
    OFFICIAL_TEST_RUNTIME_SOURCE,
    build_benchmark_probe_runtime,
    build_godot_owned_activity_runtime,
    build_official_test_runtime,
    is_official_test_code,
)
from cfast_trainer.godot_owned import GodotOwnedPayload
from cfast_trainer.guide_skill_catalog import OFFICIAL_GUIDE_TEST_CODES


class _FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return float(self.t)

    def advance(self, dt: float) -> None:
        self.t += float(dt)


def _metric_overrides(engine: object) -> dict[str, str]:
    overrides = getattr(engine, "_result_metrics_overrides", None)
    assert isinstance(overrides, dict)
    return {str(key): str(value) for key, value in overrides.items()}


def test_runtime_catalog_covers_the_20_official_tests_menu_codes() -> None:
    assert len(OFFICIAL_TEST_MENU_ORDER) == 20
    assert set(OFFICIAL_TEST_MENU_ORDER) == set(OFFICIAL_GUIDE_TEST_CODES)
    assert all(is_official_test_code(code) for code in OFFICIAL_GUIDE_TEST_CODES)


@pytest.mark.parametrize("test_code", OFFICIAL_TEST_MENU_ORDER)
def test_official_test_runtime_stamps_source_of_truth_metadata(test_code: str) -> None:
    engine = build_official_test_runtime(
        clock=_FakeClock(),
        test_code=test_code,
        seed=2026,
        difficulty=0.5,
    )

    metrics = _metric_overrides(engine)

    assert metrics["runtime_source"] == OFFICIAL_TEST_RUNTIME_SOURCE
    assert metrics["source_test_code"] == test_code
    assert metrics["canonical_activity_code"] == test_code
    assert metrics["runtime_mode"] == "standard"
    assert metrics["runtime_seed"] == "2026"


@pytest.mark.parametrize(
    ("test_code", "kind"),
    (
        ("auditory_capacity", "auditory_capacity"),
        ("rapid_tracking", "rapid_tracking"),
        ("spatial_integration", "spatial_integration"),
        ("trace_test_1", "trace_test_1"),
        ("trace_test_2", "trace_test_2"),
    ),
)
def test_official_godot_tests_use_catalog_wrapper_but_keep_godot_payload(
    test_code: str,
    kind: str,
) -> None:
    engine = build_official_test_runtime(
        clock=_FakeClock(),
        test_code=test_code,
        seed=17,
        difficulty=0.55,
        duration_s=45.0,
    )
    payload = engine.snapshot().payload

    assert isinstance(payload, GodotOwnedPayload)
    assert payload.spec.kind == kind
    assert payload.spec.test_code == test_code
    assert payload.spec.duration_s == pytest.approx(45.0)
    assert _metric_overrides(engine)["runtime_source"] == OFFICIAL_TEST_RUNTIME_SOURCE


def test_benchmark_probe_runtime_reuses_official_runtime_with_benchmark_labels() -> None:
    engine = build_benchmark_probe_runtime(
        clock=_FakeClock(),
        probe_code="rapid_tracking",
        seed=51,
        duration_s=30.0,
    )
    payload = engine.snapshot().payload
    metrics = _metric_overrides(engine)

    assert isinstance(payload, GodotOwnedPayload)
    assert payload.spec.mode == "benchmark"
    assert payload.spec.config["benchmark"] is True
    assert metrics["runtime_source"] == OFFICIAL_TEST_RUNTIME_SOURCE
    assert metrics["runtime_context"] == "benchmark_probe"
    assert metrics["difficulty_level"] == "5"


def test_godot_owned_activity_runtime_is_the_shared_drill_workout_adaptive_path() -> None:
    engine = build_godot_owned_activity_runtime(
        clock=_FakeClock(),
        seed=91,
        difficulty=0.5,
        test_code="rt_obscured_target_prediction",
        title="Rapid Tracking: Obscured Target Prediction",
        duration_s=25.0,
        mode="build",
        extra={"drill": True, "adaptive": True},
    )
    payload = engine.snapshot().payload
    metrics = _metric_overrides(engine)

    assert isinstance(payload, GodotOwnedPayload)
    assert payload.spec.kind == "rapid_tracking"
    assert payload.spec.config["drill"] is True
    assert payload.spec.config["adaptive"] is True
    assert metrics["runtime_source"] == GODOT_ACTIVITY_RUNTIME_SOURCE
    assert metrics["canonical_activity_code"] == "rt_obscured_target_prediction"
