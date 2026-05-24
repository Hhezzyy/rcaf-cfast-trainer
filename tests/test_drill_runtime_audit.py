"""Runtime catalog audit for drills, workouts, and official test families.

These tests deliberately exercise menu-visible drill codes, canonical drill
registry codes, and workout block codes together so stale builders or fallback
runtime configs fail close to the catalog entry that introduced them.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import pytest

from cfast_trainer.abd_workouts import build_abd_workout_plan
from cfast_trainer.ac_workouts import build_ac_workout_plan
from cfast_trainer.activity_runtime_catalog import (
    GODOT_ACTIVITY_RUNTIME_SOURCE,
    OFFICIAL_TEST_MENU_ORDER,
    OFFICIAL_TEST_RUNTIME_SOURCE,
    build_godot_owned_activity_runtime,
    build_official_test_runtime,
)
from cfast_trainer.ant_drills import AntDrillMode
from cfast_trainer.ant_workouts import (
    AntWorkoutBlockPlan,
    AntWorkoutPlan,
    build_ant_workout_plan,
    build_workout_block_engine,
)
from cfast_trainer.auditory_capacity import AuditoryCapacityPayload
from cfast_trainer.canonical_drill_registry import CANONICAL_DRILL_REGISTRY
from cfast_trainer.cln_workouts import build_cln_workout_plan
from cfast_trainer.cognitive_core import Phase
from cfast_trainer.cu_workouts import build_cu_workout_plan
from cfast_trainer.dr_workouts import build_dr_workout_plan
from cfast_trainer.godot_owned import GodotOwnedPayload, godot_kind_for_test_code
from cfast_trainer.ic_workouts import build_ic_workout_plan
from cfast_trainer.mr_workouts import build_mr_workout_plan
from cfast_trainer.no_workouts import build_no_workout_plan
from cfast_trainer.rt_workouts import build_rt_workout_plan
from cfast_trainer.sa_workouts import build_sa_workout_plan
from cfast_trainer.si_workouts import build_si_workout_plan
from cfast_trainer.sl_workouts import build_sl_workout_plan
from cfast_trainer.sma_workouts import build_sma_workout_plan
from cfast_trainer.tbl_workouts import build_tbl_workout_plan
from cfast_trainer.tr_workouts import build_tr_workout_plan
from cfast_trainer.trace_workouts import (
    build_trace_test_1_workout_plan,
    build_trace_test_2_workout_plan,
)
from cfast_trainer.vig_workouts import build_vig_workout_plan
from cfast_trainer.vs_workouts import build_vs_workout_plan


class _FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return float(self.t)

    def advance(self, dt: float) -> None:
        self.t += float(dt)


_APP_SOURCE = (Path(__file__).resolve().parents[1] / "cfast_trainer" / "app.py").read_text(
    encoding="utf-8"
)


def _drill_menu_codes_from_app() -> tuple[str, ...]:
    codes = set(
        re.findall(
            r'\(\s*"([a-z0-9_]+)"\s*,\s*"[^"]+"\s*,\s*open_[a-z0-9_]+',
            _APP_SOURCE,
        )
    )
    return tuple(sorted(codes))


def _workout_plans() -> tuple[AntWorkoutPlan, ...]:
    builders: tuple[Callable[[], AntWorkoutPlan], ...] = (
        lambda: build_ant_workout_plan("airborne_numerical_workout"),
        build_abd_workout_plan,
        build_no_workout_plan,
        build_mr_workout_plan,
        build_dr_workout_plan,
        build_cln_workout_plan,
        build_vs_workout_plan,
        build_ic_workout_plan,
        build_tr_workout_plan,
        build_sl_workout_plan,
        build_tbl_workout_plan,
        build_sma_workout_plan,
        build_ac_workout_plan,
        build_cu_workout_plan,
        build_sa_workout_plan,
        build_rt_workout_plan,
        build_si_workout_plan,
        build_trace_test_1_workout_plan,
        build_trace_test_2_workout_plan,
        build_vig_workout_plan,
    )
    return tuple(builder() for builder in builders)


def _representative_block(code: str) -> AntWorkoutBlockPlan:
    return AntWorkoutBlockPlan(
        block_id=f"audit_{code}",
        label=code.replace("_", " ").title(),
        description="Runtime audit block.",
        focus_skills=("audit",),
        drill_code=code,
        mode=AntDrillMode.BUILD,
        duration_min=0.05,
    )


def _phase(engine: object) -> Phase | None:
    value = getattr(engine, "phase", None)
    if isinstance(value, Phase):
        return value
    try:
        snap = engine.snapshot()
    except Exception:
        return None
    snap_phase = getattr(snap, "phase", None)
    return snap_phase if isinstance(snap_phase, Phase) else None


DRILL_MENU_CODES = _drill_menu_codes_from_app()
WORKOUT_BLOCK_CODES = tuple(
    sorted({block.drill_code for plan in _workout_plans() for block in plan.blocks})
)
CANONICAL_BUILDER_CODES = tuple(
    sorted(
        spec.drill_code
        for spec in CANONICAL_DRILL_REGISTRY
        if spec.builder_module and spec.builder_name
    )
)
AUDITED_ACTIVITY_CODES = tuple(
    sorted(set(DRILL_MENU_CODES) | set(WORKOUT_BLOCK_CODES) | set(CANONICAL_BUILDER_CODES))
)
GODOT_ACTIVITY_CODES = tuple(
    code for code in AUDITED_ACTIVITY_CODES if godot_kind_for_test_code(code) is not None
)


def test_drill_runtime_audit_enumerates_expected_catalogs() -> None:
    assert len(OFFICIAL_TEST_MENU_ORDER) == 20
    assert {
        "rt_lock_anchor",
        "rt_obscured_target_prediction",
        "rt_pressure_run",
    }.issubset(DRILL_MENU_CODES)
    assert {
        "ac_gate_anchor",
        "ac_mixed_tempo",
        "ac_pressure_run",
    }.issubset(DRILL_MENU_CODES)
    assert set(WORKOUT_BLOCK_CODES).issubset(set(AUDITED_ACTIVITY_CODES))
    assert set(CANONICAL_BUILDER_CODES).issubset(set(AUDITED_ACTIVITY_CODES))


@pytest.mark.parametrize("test_code", OFFICIAL_TEST_MENU_ORDER)
def test_official_tests_are_buildable_and_source_stamped(test_code: str) -> None:
    engine = build_official_test_runtime(
        clock=_FakeClock(),
        test_code=test_code,
        seed=2026,
        difficulty=0.5,
        duration_s=12.0,
    )
    metrics = getattr(engine, "_result_metrics_overrides", {})

    assert isinstance(metrics, dict)
    assert metrics["runtime_source"] == OFFICIAL_TEST_RUNTIME_SOURCE
    assert metrics["source_test_code"] == test_code


@pytest.mark.parametrize("spec", CANONICAL_DRILL_REGISTRY, ids=lambda spec: spec.drill_code)
def test_canonical_drill_builders_resolve(spec) -> None:
    if not spec.builder_module or not spec.builder_name:
        return
    assert callable(spec.resolve_builder())


@pytest.mark.parametrize("drill_code", AUDITED_ACTIVITY_CODES)
def test_drill_or_workout_block_codes_are_buildable_and_launchable(drill_code: str) -> None:
    engine = build_workout_block_engine(
        clock=_FakeClock(),
        block_seed=4242,
        difficulty_level=5,
        block=_representative_block(drill_code),
    )
    assert engine.snapshot().payload is not None or _phase(engine) is not None

    starter = getattr(engine, "start_practice", None)
    if callable(starter):
        starter()
    elif callable(getattr(engine, "start_scored", None)):
        engine.start_scored()

    assert _phase(engine) is not Phase.INSTRUCTIONS


@pytest.mark.parametrize("drill_code", GODOT_ACTIVITY_CODES)
def test_godot_owned_drill_routes_match_official_runtime_family(drill_code: str) -> None:
    expected_kind = godot_kind_for_test_code(drill_code)
    assert expected_kind is not None
    source_test_code = {
        "auditory_capacity": "auditory_capacity",
        "rapid_tracking": "rapid_tracking",
        "spatial_integration": "spatial_integration",
        "trace_test_1": "trace_test_1",
        "trace_test_2": "trace_test_2",
    }[expected_kind]
    official = build_official_test_runtime(
        clock=_FakeClock(),
        test_code=source_test_code,
        seed=2026,
        difficulty=0.5,
        duration_s=12.0,
    )
    official_payload = official.snapshot().payload
    assert isinstance(official_payload, GodotOwnedPayload)

    engine = build_godot_owned_activity_runtime(
        clock=_FakeClock(),
        seed=2027,
        difficulty=0.5,
        test_code=drill_code,
        title=drill_code.replace("_", " ").title(),
        duration_s=12.0,
        mode="build",
        extra={"drill": True},
    )
    payload = engine.snapshot().payload
    metrics = getattr(engine, "_result_metrics_overrides", {})

    assert isinstance(payload, GodotOwnedPayload)
    assert payload.spec.kind == official_payload.spec.kind == expected_kind
    assert payload.spec.test_code == drill_code
    assert metrics["runtime_source"] == GODOT_ACTIVITY_RUNTIME_SOURCE
    assert metrics["source_test_code"] == source_test_code

    config = payload.spec.config
    assert config["drill"] is True
    if expected_kind == "auditory_capacity":
        assert config["auditory_capacity"] is True
        assert config["difficulty_scaled"] is True
        assert config["segments"]
        assert config["active_channels"]
    elif expected_kind == "rapid_tracking":
        assert config["rapid_tracking"] is True
        assert config["difficulty_scaled"] is True
        assert isinstance(config["rapid_world"], dict)

    engine.apply_godot_authoritative_message(
        {
            "command": "godot_phase_advance",
            "run_key": payload.spec.run_key,
            "test_code": drill_code,
            "kind": expected_kind,
        }
    )
    assert engine.phase is not Phase.INSTRUCTIONS


@pytest.mark.parametrize("drill_code", ("ac_gate_anchor", "ac_pressure_run"))
def test_legacy_python_auditory_drill_wrappers_still_expose_control_for_regression(
    drill_code: str,
) -> None:
    engine = build_workout_block_engine(
        clock=_FakeClock(),
        block_seed=5151,
        difficulty_level=5,
        block=_representative_block(drill_code),
    )
    payload = engine.snapshot().payload

    assert isinstance(payload, GodotOwnedPayload)
    assert payload.spec.kind == "auditory_capacity"

    from cfast_trainer.ac_drills import AcDrillConfig
    from cfast_trainer.ac_drills import build_ac_gate_anchor_drill, build_ac_pressure_run_drill

    builder = {
        "ac_gate_anchor": build_ac_gate_anchor_drill,
        "ac_pressure_run": build_ac_pressure_run_drill,
    }[drill_code]
    legacy = builder(
        clock=_FakeClock(),
        seed=5151,
        difficulty=0.5,
        mode=AntDrillMode.BUILD,
        config=AcDrillConfig(scored_duration_s=2.0),
    )
    legacy.start_practice()
    legacy.set_control(horizontal=0.25, vertical=-0.5)
    legacy_payload = legacy.snapshot().payload

    assert isinstance(legacy_payload, AuditoryCapacityPayload)
    assert legacy_payload.control_x == pytest.approx(0.25)
    assert legacy_payload.control_y == pytest.approx(-0.5)
