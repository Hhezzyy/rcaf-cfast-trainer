from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from .adaptive_difficulty import difficulty_ratio_for_level
from .airborne_numerical import build_airborne_numerical_test
from .angles_bearings_degrees import AnglesBearingsDegreesConfig, build_angles_bearings_degrees_test
from .clock import Clock
from .cognitive_updating import CognitiveUpdatingConfig, build_cognitive_updating_test
from .colours_letters_numbers import build_colours_letters_numbers_test
from .digit_recognition import build_digit_recognition_test
from .godot_owned import (
    GODOT_OWNED_KINDS,
    auditory_capacity_godot_config,
    build_godot_owned_test,
    default_title_for_godot_kind,
    godot_kind_for_test_code,
    rapid_tracking_godot_config,
    spatial_integration_godot_config,
    trace_test_godot_config,
)
from .instrument_comprehension import InstrumentComprehensionConfig, build_instrument_comprehension_test
from .math_reasoning import MathReasoningConfig, build_math_reasoning_test
from .numerical_operations import NumericalOperationsConfig, build_numerical_operations_test
from .sensory_motor_apparatus import SensoryMotorApparatusConfig, build_sensory_motor_apparatus_test
from .situational_awareness import SituationalAwarenessConfig, build_situational_awareness_test
from .system_logic import SystemLogicConfig, build_system_logic_test
from .table_reading import TableReadingConfig, build_table_reading_test
from .target_recognition import TargetRecognitionConfig, build_target_recognition_test
from .vigilance import VigilanceConfig, build_vigilance_test
from .visual_search import VisualSearchConfig, build_visual_search_test


OFFICIAL_TEST_RUNTIME_SOURCE = "tests_menu_runtime_catalog"
GODOT_ACTIVITY_RUNTIME_SOURCE = "godot_owned_activity_catalog"

OFFICIAL_TEST_MENU_ORDER: tuple[str, ...] = (
    "numerical_operations",
    "math_reasoning",
    "airborne_numerical",
    "digit_recognition",
    "colours_letters_numbers",
    "angles_bearings_degrees",
    "visual_search",
    "instrument_comprehension",
    "target_recognition",
    "system_logic",
    "table_reading",
    "sensory_motor_apparatus",
    "auditory_capacity",
    "cognitive_updating",
    "situational_awareness",
    "rapid_tracking",
    "spatial_integration",
    "trace_test_1",
    "trace_test_2",
    "vigilance",
)

OFFICIAL_TEST_MENU_TITLES: dict[str, str] = {
    "numerical_operations": "Numerical Operations",
    "math_reasoning": "Mathematics Reasoning",
    "airborne_numerical": "Airborne Numerical Test",
    "digit_recognition": "Digit Recognition",
    "colours_letters_numbers": "Colours, Letters and Numbers",
    "angles_bearings_degrees": "Angles, Bearings and Degrees",
    "visual_search": "Visual Search",
    "instrument_comprehension": "Instrument Comprehension",
    "target_recognition": "Target Recognition",
    "system_logic": "System Logic",
    "table_reading": "Table Reading",
    "sensory_motor_apparatus": "Sensory Motor Apparatus",
    "auditory_capacity": "Auditory Capacity",
    "cognitive_updating": "Cognitive Updating",
    "situational_awareness": "Situational Awareness",
    "rapid_tracking": "Rapid Tracking",
    "spatial_integration": "Spatial Integration",
    "trace_test_1": "Trace Test 1",
    "trace_test_2": "Trace Test 2",
    "vigilance": "Vigilance",
}

_GODOT_FAMILY_LABELS = {
    "auditory_capacity": "auditory_capacity",
    "rapid_tracking": "rapid_tracking",
    "spatial_integration": "spatial_integration",
    "trace_test_1": "trace_test_1",
    "trace_test_2": "trace_test_2",
}


def official_test_title(test_code: str | None) -> str:
    token = _normalize_code(test_code)
    return OFFICIAL_TEST_MENU_TITLES.get(token, token.replace("_", " ").title())


def is_official_test_code(test_code: str | None) -> bool:
    return _normalize_code(test_code) in OFFICIAL_TEST_MENU_ORDER


def build_official_test_runtime(
    *,
    clock: Clock,
    test_code: str,
    seed: int,
    difficulty: float,
    mode: str = "standard",
    duration_s: float | None = None,
    practice_enabled: bool = True,
    review_mode_enabled: bool = False,
    auditory_extra: Mapping[str, object] | None = None,
    extra_config: Mapping[str, object] | None = None,
) -> object:
    """Build the authoritative runtime used by the Tests menu for official tests."""

    token = _normalize_code(test_code)
    normalized_mode = _normalize_mode(mode)
    title = official_test_title(token)
    if token not in OFFICIAL_TEST_MENU_ORDER:
        raise KeyError(f"unsupported official test runtime: {test_code}")
    godot_kind = godot_kind_for_test_code(token)
    if godot_kind in GODOT_OWNED_KINDS:
        engine = build_godot_owned_activity_runtime(
            clock=clock,
            seed=seed,
            difficulty=difficulty,
            test_code=token,
            title=title,
            kind=godot_kind,
            duration_s=duration_s,
            mode=normalized_mode,
            extra=extra_config,
            review_mode_enabled=review_mode_enabled,
            auditory_extra=auditory_extra,
        )
        _stamp_runtime_metadata(
            engine,
            runtime_source=OFFICIAL_TEST_RUNTIME_SOURCE,
            source_test_code=token,
            canonical_code=token,
            mode=normalized_mode,
            duration_s=duration_s,
            seed=seed,
            difficulty=difficulty,
        )
        return engine

    engine = _build_python_official_runtime(
        clock=clock,
        test_code=token,
        seed=seed,
        difficulty=difficulty,
        duration_s=duration_s,
        practice_enabled=practice_enabled,
    )
    _stamp_runtime_metadata(
        engine,
        runtime_source=OFFICIAL_TEST_RUNTIME_SOURCE,
        source_test_code=token,
        canonical_code=token,
        mode=normalized_mode,
        duration_s=duration_s,
        seed=seed,
        difficulty=difficulty,
    )
    return engine


def build_benchmark_probe_runtime(
    *,
    clock: Clock,
    probe_code: str,
    seed: int,
    duration_s: float,
) -> object:
    difficulty = difficulty_ratio_for_level(probe_code, 5)
    engine = build_official_test_runtime(
        clock=clock,
        test_code=probe_code,
        seed=seed,
        difficulty=difficulty,
        mode="benchmark",
        duration_s=duration_s,
        practice_enabled=False,
        extra_config={"benchmark": True},
    )
    _stamp_runtime_metadata(
        engine,
        runtime_source=OFFICIAL_TEST_RUNTIME_SOURCE,
        source_test_code=probe_code,
        canonical_code=probe_code,
        mode="benchmark",
        duration_s=duration_s,
        seed=seed,
        difficulty=difficulty,
        extra={
            "benchmark": "1",
            "runtime_context": "benchmark_probe",
            "difficulty_level": "5",
        },
    )
    return engine


def build_godot_owned_activity_runtime(
    *,
    clock: Clock,
    seed: int,
    difficulty: float,
    test_code: str,
    title: str | None = None,
    kind: str | None = None,
    duration_s: float | None = None,
    mode: str = "standard",
    extra: Mapping[str, object] | None = None,
    review_mode_enabled: bool = False,
    auditory_extra: Mapping[str, object] | None = None,
    practice_enabled: bool = True,
    practice_duration_s: float = 12.0,
) -> object:
    """Build a Godot-owned official test, drill, workout, or adaptive block consistently."""

    token = _normalize_code(test_code)
    resolved_kind = kind or godot_kind_for_test_code(token)
    if resolved_kind not in GODOT_OWNED_KINDS:
        raise ValueError(f"test code is not Godot-owned: {test_code}")
    normalized_mode = _normalize_mode(mode)
    config_extra = dict(extra or {})
    if resolved_kind == "auditory_capacity":
        merged_extra = {**dict(auditory_extra or {}), **config_extra}
        config = auditory_capacity_godot_config(
            test_code=token,
            mode=normalized_mode,
            difficulty=difficulty,
            duration_s=duration_s,
            review_mode_enabled=review_mode_enabled,
            extra=merged_extra,
        )
    elif resolved_kind == "rapid_tracking":
        config = rapid_tracking_godot_config(
            test_code=token,
            mode=normalized_mode,
            difficulty=difficulty,
            duration_s=duration_s,
            extra=config_extra,
        )
    elif resolved_kind == "spatial_integration":
        config = spatial_integration_godot_config(
            test_code=token,
            mode=normalized_mode,
            duration_s=duration_s,
            extra=config_extra,
        )
    else:
        config = trace_test_godot_config(
            test_code=token,
            mode=normalized_mode,
            difficulty=difficulty,
            duration_s=duration_s,
            extra=config_extra,
        )
    resolved_title = str(title or "").strip() or default_title_for_godot_kind(
        str(resolved_kind),
        token,
    )
    engine = build_godot_owned_test(
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        kind=str(resolved_kind),
        test_code=token,
        title=resolved_title,
        duration_s=duration_s,
        mode=normalized_mode,
        practice_enabled=practice_enabled,
        practice_duration_s=practice_duration_s,
        config=config,
    )
    _stamp_runtime_metadata(
        engine,
        runtime_source=GODOT_ACTIVITY_RUNTIME_SOURCE,
        source_test_code=_GODOT_FAMILY_LABELS.get(str(resolved_kind), str(resolved_kind)),
        canonical_code=token,
        mode=normalized_mode,
        duration_s=duration_s,
        seed=seed,
        difficulty=difficulty,
    )
    return engine


def _build_python_official_runtime(
    *,
    clock: Clock,
    test_code: str,
    seed: int,
    difficulty: float,
    duration_s: float | None,
    practice_enabled: bool,
) -> object:
    if test_code == "numerical_operations":
        config = _timed_config(
            NumericalOperationsConfig(),
            duration_s=duration_s,
            practice_field="practice_questions",
            practice_value=5 if practice_enabled else 0,
        )
        return build_numerical_operations_test(
            clock=clock,
            seed=seed,
            difficulty=difficulty,
            config=config,
        )
    if test_code == "math_reasoning":
        config = _timed_config(
            MathReasoningConfig(),
            duration_s=duration_s,
            practice_field="practice_questions",
            practice_value=3 if practice_enabled else 0,
        )
        return build_math_reasoning_test(
            clock=clock,
            seed=seed,
            difficulty=difficulty,
            config=config,
        )
    if test_code == "airborne_numerical":
        return build_airborne_numerical_test(
            clock=clock,
            seed=seed,
            practice=practice_enabled,
            difficulty=difficulty,
            scored_duration_s=35.0 * 60.0 if duration_s is None else float(duration_s),
        )
    if test_code == "digit_recognition":
        return build_digit_recognition_test(
            clock=clock,
            seed=seed,
            difficulty=difficulty,
            practice=practice_enabled,
            scored_duration_s=360.0 if duration_s is None else float(duration_s),
        )
    if test_code == "colours_letters_numbers":
        return build_colours_letters_numbers_test(
            clock=clock,
            seed=seed,
            difficulty=difficulty,
            practice=practice_enabled,
            scored_duration_s=duration_s,
        )
    if test_code == "angles_bearings_degrees":
        config = _timed_config(
            AnglesBearingsDegreesConfig(),
            duration_s=duration_s,
            practice_field="practice_questions",
            practice_value=3 if practice_enabled else 0,
        )
        return build_angles_bearings_degrees_test(
            clock=clock,
            seed=seed,
            difficulty=difficulty,
            config=config,
        )
    if test_code == "visual_search":
        config = _timed_config(
            VisualSearchConfig(),
            duration_s=duration_s,
            practice_field="practice_questions",
            practice_value=4 if practice_enabled else 0,
        )
        return build_visual_search_test(
            clock=clock,
            seed=seed,
            difficulty=difficulty,
            config=config,
        )
    if test_code == "instrument_comprehension":
        config = _timed_config(
            InstrumentComprehensionConfig(),
            duration_s=duration_s,
            practice_field="practice_questions",
            practice_value=1 if practice_enabled else 0,
        )
        return build_instrument_comprehension_test(
            clock=clock,
            seed=seed,
            difficulty=difficulty,
            config=config,
        )
    if test_code == "target_recognition":
        config = _timed_config(
            TargetRecognitionConfig(),
            duration_s=duration_s,
            practice_field="practice_questions",
            practice_value=4 if practice_enabled else 0,
        )
        return build_target_recognition_test(
            clock=clock,
            seed=seed,
            difficulty=difficulty,
            config=config,
        )
    if test_code == "system_logic":
        config = _timed_config(
            SystemLogicConfig(),
            duration_s=duration_s,
            practice_field="practice_questions",
            practice_value=3 if practice_enabled else 0,
        )
        return build_system_logic_test(
            clock=clock,
            seed=seed,
            difficulty=difficulty,
            config=config,
        )
    if test_code == "table_reading":
        config = _timed_config(
            TableReadingConfig(),
            duration_s=duration_s,
            practice_field="practice_questions",
            practice_value=2 if practice_enabled else 0,
        )
        return build_table_reading_test(
            clock=clock,
            seed=seed,
            difficulty=difficulty,
            config=config,
        )
    if test_code == "sensory_motor_apparatus":
        config = _timed_config(
            SensoryMotorApparatusConfig(),
            duration_s=duration_s,
            practice_field="practice_duration_s",
            practice_value=45.0 if practice_enabled else 0.0,
        )
        return build_sensory_motor_apparatus_test(
            clock=clock,
            seed=seed,
            difficulty=difficulty,
            config=config,
        )
    if test_code == "cognitive_updating":
        config = _timed_config(
            CognitiveUpdatingConfig(),
            duration_s=duration_s,
            practice_field="practice_questions",
            practice_value=3 if practice_enabled else 0,
        )
        return build_cognitive_updating_test(
            clock=clock,
            seed=seed,
            difficulty=difficulty,
            config=config,
        )
    if test_code == "situational_awareness":
        base = SituationalAwarenessConfig()
        config = replace(
            base,
            scored_duration_s=base.scored_duration_s if duration_s is None else float(duration_s),
            practice_scenarios=base.practice_scenarios if practice_enabled else 0,
            practice_scenario_duration_s=base.practice_scenario_duration_s if practice_enabled else 0.0,
            scored_scenario_duration_s=base.scored_scenario_duration_s if practice_enabled else 40.0,
        )
        return build_situational_awareness_test(
            clock=clock,
            seed=seed,
            difficulty=difficulty,
            config=config,
        )
    if test_code == "vigilance":
        config = _timed_config(
            VigilanceConfig(),
            duration_s=duration_s,
            practice_field="practice_duration_s",
            practice_value=45.0 if practice_enabled else 0.0,
        )
        return build_vigilance_test(
            clock=clock,
            seed=seed,
            difficulty=difficulty,
            config=config,
        )
    raise KeyError(f"unsupported official test runtime: {test_code}")


def _timed_config(
    config: object,
    *,
    duration_s: float | None,
    practice_field: str,
    practice_value: object,
) -> object:
    updates: dict[str, object] = {practice_field: practice_value}
    if duration_s is not None:
        updates["scored_duration_s"] = float(duration_s)
    return replace(config, **updates)


def _stamp_runtime_metadata(
    engine: object,
    *,
    runtime_source: str,
    source_test_code: str,
    canonical_code: str,
    mode: str,
    duration_s: float | None,
    seed: int,
    difficulty: float,
    extra: Mapping[str, object] | None = None,
) -> None:
    overrides = getattr(engine, "_result_metrics_overrides", None)
    if not isinstance(overrides, dict):
        overrides = {}
        setattr(engine, "_result_metrics_overrides", overrides)
    overrides["runtime_source"] = str(runtime_source)
    overrides["source_test_code"] = str(source_test_code)
    overrides["canonical_activity_code"] = str(canonical_code)
    overrides["runtime_mode"] = str(mode)
    overrides["runtime_seed"] = str(int(seed))
    overrides["runtime_difficulty"] = f"{float(difficulty):.6f}"
    if duration_s is not None:
        overrides["runtime_duration_s"] = f"{float(duration_s):.6f}"
    if extra:
        for key, value in extra.items():
            overrides[str(key)] = str(value)


def _normalize_code(value: str | None) -> str:
    return str(value or "").strip().lower()


def _normalize_mode(value: str | None) -> str:
    return str(value or "standard").strip().lower() or "standard"
