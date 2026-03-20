from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal, cast

from .guide_skill_catalog import official_guide_test, guide_ranking_primitive_id_for_code


LaunchDifficultyMode = Literal["adaptive", "fixed"]
ScopeKind = Literal["code", "family", "primitive"]
DifficultyFamilyId = Literal[
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
]
DifficultyIntendedUse = Literal["anchor", "build", "tempo", "pressure", "fatigue_probe"]

_BENCHMARK_CODE = "benchmark_battery"
_DEFAULT_LEVEL_RATIOS = (0.00, 0.08, 0.16, 0.26, 0.38, 0.50, 0.62, 0.74, 0.87, 1.00)
_MATH_LEVEL_RATIOS = (0.00, 0.03, 0.08, 0.15, 0.25, 0.38, 0.52, 0.68, 0.84, 1.00)
_LOOKUP_LEVEL_RATIOS = (0.00, 0.06, 0.14, 0.24, 0.36, 0.48, 0.60, 0.73, 0.86, 1.00)
_MEMORY_LEVEL_RATIOS = (0.00, 0.05, 0.12, 0.22, 0.34, 0.48, 0.62, 0.76, 0.89, 1.00)
_SCAN_LEVEL_RATIOS = (0.00, 0.07, 0.16, 0.26, 0.38, 0.50, 0.62, 0.74, 0.87, 1.00)
_TRACKING_LEVEL_RATIOS = (0.00, 0.10, 0.18, 0.28, 0.40, 0.52, 0.64, 0.76, 0.88, 1.00)
_SPATIAL_TRACE_LEVEL_RATIOS = (0.00, 0.07, 0.15, 0.24, 0.34, 0.46, 0.59, 0.72, 0.86, 1.00)


def clamp_level(level: int) -> int:
    return max(1, min(10, int(level)))


def clamp_ratio(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def linear_ratio_for_level(level: int) -> float:
    clamped = clamp_level(level)
    return float(clamped - 1) / 9.0


def level_from_ratio(value: float) -> int:
    ratio = clamp_ratio(value)
    return clamp_level(int(round(ratio * 9.0)) + 1)


@dataclass(frozen=True, slots=True)
class DifficultyAxes:
    content_complexity: float = 0.0
    time_pressure: float = 0.0
    distractor_density: float = 0.0
    multitask_concurrency: float = 0.0
    memory_span_delay: float = 0.0
    switch_frequency: float = 0.0
    control_sensitivity: float = 0.0
    spatial_ambiguity: float = 0.0
    source_integration_depth: float = 0.0

    def clamped(self) -> "DifficultyAxes":
        return DifficultyAxes(
            content_complexity=clamp_ratio(self.content_complexity),
            time_pressure=clamp_ratio(self.time_pressure),
            distractor_density=clamp_ratio(self.distractor_density),
            multitask_concurrency=clamp_ratio(self.multitask_concurrency),
            memory_span_delay=clamp_ratio(self.memory_span_delay),
            switch_frequency=clamp_ratio(self.switch_frequency),
            control_sensitivity=clamp_ratio(self.control_sensitivity),
            spatial_ambiguity=clamp_ratio(self.spatial_ambiguity),
            source_integration_depth=clamp_ratio(self.source_integration_depth),
        )

    def add(self, other: "DifficultyAxes") -> "DifficultyAxes":
        return DifficultyAxes(
            content_complexity=self.content_complexity + other.content_complexity,
            time_pressure=self.time_pressure + other.time_pressure,
            distractor_density=self.distractor_density + other.distractor_density,
            multitask_concurrency=self.multitask_concurrency + other.multitask_concurrency,
            memory_span_delay=self.memory_span_delay + other.memory_span_delay,
            switch_frequency=self.switch_frequency + other.switch_frequency,
            control_sensitivity=self.control_sensitivity + other.control_sensitivity,
            spatial_ambiguity=self.spatial_ambiguity + other.spatial_ambiguity,
            source_integration_depth=self.source_integration_depth + other.source_integration_depth,
        )

    def scale(self, factor: float) -> "DifficultyAxes":
        scaled = float(factor)
        return DifficultyAxes(
            content_complexity=self.content_complexity * scaled,
            time_pressure=self.time_pressure * scaled,
            distractor_density=self.distractor_density * scaled,
            multitask_concurrency=self.multitask_concurrency * scaled,
            memory_span_delay=self.memory_span_delay * scaled,
            switch_frequency=self.switch_frequency * scaled,
            control_sensitivity=self.control_sensitivity * scaled,
            spatial_ambiguity=self.spatial_ambiguity * scaled,
            source_integration_depth=self.source_integration_depth * scaled,
        )


@dataclass(frozen=True, slots=True)
class DifficultyProfile:
    family_id: DifficultyFamilyId
    level: int
    label: str
    intended_use: DifficultyIntendedUse
    axis_values: DifficultyAxes
    axis_weights: DifficultyAxes
    legacy_ratio: float

    @property
    def axes(self) -> DifficultyAxes:
        return self.axis_values


@dataclass(frozen=True, slots=True)
class ChallengeBand:
    key: str
    label: str
    retain_accuracy: float
    promote_accuracy: float
    max_timeout_rate: float
    promote_timeout_rate: float
    meltdown_accuracy: float
    meltdown_timeout_rate: float
    allow_promotion: bool


@dataclass(frozen=True, slots=True)
class DifficultyLadderSpec:
    scope_key: str
    label: str
    level_ratios: tuple[float, ...]

    def ratio_for_level(self, level: int) -> float:
        index = clamp_level(level) - 1
        ratios = self.level_ratios or _DEFAULT_LEVEL_RATIOS
        if index >= len(ratios):
            return float(ratios[-1])
        return clamp_ratio(float(ratios[index]))


@dataclass(frozen=True, slots=True)
class AdaptiveDifficultyState:
    scope_kind: ScopeKind
    scope_key: str
    recommended_level: int
    last_start_level: int | None
    last_end_level: int | None
    ewma_accuracy: float | None
    ewma_score_ratio: float | None
    ewma_timeout_rate: float | None
    ewma_mean_rt_ms: float | None
    sample_count: int
    updated_at_utc: str
    mastery: float | None = None
    speed: float | None = None
    fatigue_penalty: float | None = None
    post_error_penalty: float | None = None
    instability_penalty: float | None = None
    retention_need: float | None = None
    confidence: float | None = None
    leverage: float | None = None
    level_confidence: float | None = None
    last_successful_level: int | None = None
    last_meltdown_level: int | None = None


@dataclass(frozen=True, slots=True)
class AdaptiveDifficultyDecision:
    after_units: int
    old_level: int
    new_level: int
    reason: str


@dataclass(frozen=True, slots=True)
class ResolvedDifficultyContext:
    test_code: str
    mode: LaunchDifficultyMode
    fixed_level: int
    launch_level: int
    launch_ratio: float
    code_scope_key: str
    primitive_scope_key: str
    family_scope_key: str
    adaptive_enabled: bool


@dataclass(frozen=True, slots=True)
class AdaptivePolicy:
    short_window_size: int
    medium_window_size: int
    throttle_units: int
    max_delta: int
    session_gain_cap: int
    challenge_band: ChallengeBand


@dataclass(frozen=True, slots=True)
class _AdaptiveObservation:
    attempted: int
    correct: int
    timeout_rate: float
    score_ratio: float | None
    mean_rt_ms: float | None


@dataclass(frozen=True, slots=True)
class _FamilyDifficultyConfig:
    family_id: DifficultyFamilyId
    label: str
    legacy_ratios: tuple[float, ...]
    axis_weights: DifficultyAxes


_BUILD_BAND = ChallengeBand(
    key="build",
    label="Build",
    retain_accuracy=0.78,
    promote_accuracy=0.92,
    max_timeout_rate=0.18,
    promote_timeout_rate=0.05,
    meltdown_accuracy=0.22,
    meltdown_timeout_rate=0.55,
    allow_promotion=True,
)
_TEMPO_BAND = ChallengeBand(
    key="tempo",
    label="Tempo",
    retain_accuracy=0.66,
    promote_accuracy=0.86,
    max_timeout_rate=0.28,
    promote_timeout_rate=0.10,
    meltdown_accuracy=0.18,
    meltdown_timeout_rate=0.60,
    allow_promotion=True,
)
_FATIGUE_PROBE_BAND = ChallengeBand(
    key="fatigue_probe",
    label="Fatigue Probe",
    retain_accuracy=0.60,
    promote_accuracy=1.10,
    max_timeout_rate=0.35,
    promote_timeout_rate=0.0,
    meltdown_accuracy=0.18,
    meltdown_timeout_rate=0.60,
    allow_promotion=False,
)

_MODE_LEVEL_OFFSETS: dict[DifficultyIntendedUse, int] = {
    "anchor": -2,
    "build": 0,
    "tempo": 1,
    "pressure": 2,
    "fatigue_probe": 1,
}

_MODE_AXIS_BOOSTS: dict[DifficultyIntendedUse, DifficultyAxes] = {
    "anchor": DifficultyAxes(
        time_pressure=-0.12,
        distractor_density=-0.06,
        multitask_concurrency=-0.06,
        memory_span_delay=-0.05,
        switch_frequency=-0.05,
        control_sensitivity=-0.04,
        spatial_ambiguity=-0.05,
        source_integration_depth=-0.04,
    ),
    "build": DifficultyAxes(),
    "tempo": DifficultyAxes(
        time_pressure=0.08,
        distractor_density=0.03,
        switch_frequency=0.05,
        memory_span_delay=0.03,
    ),
    "pressure": DifficultyAxes(
        time_pressure=0.15,
        distractor_density=0.08,
        multitask_concurrency=0.06,
        memory_span_delay=0.05,
        switch_frequency=0.08,
        control_sensitivity=0.06,
        spatial_ambiguity=0.06,
        source_integration_depth=0.05,
    ),
    "fatigue_probe": DifficultyAxes(
        time_pressure=0.10,
        distractor_density=0.04,
        multitask_concurrency=0.04,
        memory_span_delay=0.05,
        switch_frequency=0.05,
        control_sensitivity=0.04,
    ),
}

_FAMILY_REGISTRY: dict[DifficultyFamilyId, _FamilyDifficultyConfig] = {
    "quantitative": _FamilyDifficultyConfig(
        family_id="quantitative",
        label="Quantitative",
        legacy_ratios=_MATH_LEVEL_RATIOS,
        axis_weights=DifficultyAxes(
            content_complexity=1.00,
            time_pressure=0.85,
            distractor_density=0.20,
            multitask_concurrency=0.12,
            memory_span_delay=0.18,
            switch_frequency=0.26,
            source_integration_depth=0.70,
        ),
    ),
    "angle_bearing": _FamilyDifficultyConfig(
        family_id="angle_bearing",
        label="Angle and Bearing",
        legacy_ratios=_MATH_LEVEL_RATIOS,
        axis_weights=DifficultyAxes(
            content_complexity=0.65,
            time_pressure=0.65,
            distractor_density=0.28,
            switch_frequency=0.24,
            spatial_ambiguity=0.88,
            source_integration_depth=0.30,
        ),
    ),
    "auditory_multitask": _FamilyDifficultyConfig(
        family_id="auditory_multitask",
        label="Auditory Multitask",
        legacy_ratios=_MEMORY_LEVEL_RATIOS,
        axis_weights=DifficultyAxes(
            content_complexity=0.20,
            time_pressure=0.76,
            distractor_density=0.64,
            multitask_concurrency=1.00,
            memory_span_delay=0.82,
            switch_frequency=0.74,
            source_integration_depth=0.26,
        ),
    ),
    "cln_multitask": _FamilyDifficultyConfig(
        family_id="cln_multitask",
        label="CLN Multitask",
        legacy_ratios=_MEMORY_LEVEL_RATIOS,
        axis_weights=DifficultyAxes(
            content_complexity=0.32,
            time_pressure=0.72,
            distractor_density=0.36,
            multitask_concurrency=0.92,
            memory_span_delay=0.96,
            switch_frequency=0.68,
            source_integration_depth=0.22,
        ),
    ),
    "instrument_orientation": _FamilyDifficultyConfig(
        family_id="instrument_orientation",
        label="Instrument Orientation",
        legacy_ratios=_LOOKUP_LEVEL_RATIOS,
        axis_weights=DifficultyAxes(
            content_complexity=0.26,
            time_pressure=0.58,
            distractor_density=0.30,
            switch_frequency=0.30,
            spatial_ambiguity=0.98,
            source_integration_depth=0.34,
        ),
    ),
    "visual_memory_updating": _FamilyDifficultyConfig(
        family_id="visual_memory_updating",
        label="Visual Memory Updating",
        legacy_ratios=_MEMORY_LEVEL_RATIOS,
        axis_weights=DifficultyAxes(
            content_complexity=0.36,
            time_pressure=0.55,
            distractor_density=0.28,
            memory_span_delay=1.00,
            switch_frequency=0.66,
            source_integration_depth=0.38,
        ),
    ),
    "situational_awareness": _FamilyDifficultyConfig(
        family_id="situational_awareness",
        label="Situational Awareness",
        legacy_ratios=_LOOKUP_LEVEL_RATIOS,
        axis_weights=DifficultyAxes(
            content_complexity=0.34,
            time_pressure=0.62,
            distractor_density=0.46,
            multitask_concurrency=0.40,
            memory_span_delay=0.36,
            switch_frequency=0.62,
            spatial_ambiguity=0.54,
            source_integration_depth=1.00,
        ),
    ),
    "table_cross_reference": _FamilyDifficultyConfig(
        family_id="table_cross_reference",
        label="Table Cross Reference",
        legacy_ratios=_LOOKUP_LEVEL_RATIOS,
        axis_weights=DifficultyAxes(
            content_complexity=0.24,
            time_pressure=0.74,
            distractor_density=0.76,
            switch_frequency=0.34,
            spatial_ambiguity=0.12,
            source_integration_depth=1.00,
        ),
    ),
    "system_logic": _FamilyDifficultyConfig(
        family_id="system_logic",
        label="System Logic",
        legacy_ratios=_LOOKUP_LEVEL_RATIOS,
        axis_weights=DifficultyAxes(
            content_complexity=0.86,
            time_pressure=0.58,
            distractor_density=0.52,
            switch_frequency=0.42,
            spatial_ambiguity=0.24,
            source_integration_depth=0.96,
        ),
    ),
    "search_vigilance": _FamilyDifficultyConfig(
        family_id="search_vigilance",
        label="Search and Vigilance",
        legacy_ratios=_SCAN_LEVEL_RATIOS,
        axis_weights=DifficultyAxes(
            content_complexity=0.24,
            time_pressure=0.72,
            distractor_density=1.00,
            multitask_concurrency=0.26,
            memory_span_delay=0.16,
            switch_frequency=0.52,
            spatial_ambiguity=0.44,
        ),
    ),
    "psychomotor_tracking": _FamilyDifficultyConfig(
        family_id="psychomotor_tracking",
        label="Psychomotor Tracking",
        legacy_ratios=_TRACKING_LEVEL_RATIOS,
        axis_weights=DifficultyAxes(
            content_complexity=0.14,
            time_pressure=0.66,
            distractor_density=0.24,
            multitask_concurrency=0.46,
            switch_frequency=0.56,
            control_sensitivity=1.00,
            spatial_ambiguity=0.28,
        ),
    ),
    "spatial_integration_trace": _FamilyDifficultyConfig(
        family_id="spatial_integration_trace",
        label="Spatial Integration and Trace",
        legacy_ratios=_SPATIAL_TRACE_LEVEL_RATIOS,
        axis_weights=DifficultyAxes(
            content_complexity=0.40,
            time_pressure=0.58,
            distractor_density=0.26,
            switch_frequency=0.46,
            control_sensitivity=0.70,
            spatial_ambiguity=1.00,
            source_integration_depth=0.56,
        ),
    ),
}


def is_benchmark_code(test_code: str | None) -> bool:
    return str(test_code or "").strip().lower() == _BENCHMARK_CODE


def default_launch_mode_for_code(test_code: str | None) -> LaunchDifficultyMode:
    return "fixed" if is_benchmark_code(test_code) else "adaptive"


def supports_adaptive_difficulty(test_code: str | None) -> bool:
    return not is_benchmark_code(test_code)


def scope_keys_for_code(test_code: str | None) -> tuple[str, str]:
    token = str(test_code or "").strip().lower()
    if token == "":
        return ("unknown", "unknown")
    if token == _BENCHMARK_CODE:
        return (token, token)
    if token in {"adaptive_session", "adaptive_session_short", "adaptive_session_micro"}:
        return ("adaptive_session", "adaptive_session")
    code_scope = token
    try:
        from .canonical_drill_registry import canonical_drill_convergence, resolved_canonical_drill_code
    except Exception:
        canonical_drill_convergence = None
        resolved_canonical_drill_code = None
    if canonical_drill_convergence is not None and resolved_canonical_drill_code is not None:
        convergence = canonical_drill_convergence(token)
        if convergence is not None and convergence.action == "replace_when_new_drill_exists":
            resolved = resolved_canonical_drill_code(token)
            if resolved:
                code_scope = resolved
    guide_primitive = guide_ranking_primitive_id_for_code(code_scope)
    if guide_primitive is None:
        guide_primitive = guide_ranking_primitive_id_for_code(token)
    if guide_primitive is not None:
        return (code_scope, guide_primitive)
    try:
        from .primitive_ranking import canonical_ranked_primitive_id_for_code
    except Exception:
        canonical_ranked_primitive_id_for_code = None
    if canonical_ranked_primitive_id_for_code is not None:
        ranked = canonical_ranked_primitive_id_for_code(token)
        if ranked is not None:
            return (code_scope, ranked)
    if token.startswith(("numerical_operations", "no_", "ma_")):
        return (code_scope, "mental_arithmetic_automaticity")
    if token.startswith(("math_reasoning", "mr_")):
        return (code_scope, "math_reasoning_reasoning")
    if token.startswith(("airborne_numerical", "ant_")):
        return (code_scope, "airborne_numerical_applied_math")
    if token.startswith(("system_logic", "sl_")):
        return (code_scope, "symbolic_rule_extraction")
    if token.startswith(("table_reading", "tbl_")):
        return (code_scope, "table_cross_reference_speed")
    if token.startswith(("visual_search", "vs_")):
        return (code_scope, "visual_scan_discipline")
    if token.startswith(("digit_recognition", "dr_")):
        return (code_scope, "digit_recognition_memory")
    if token.startswith(("colours_letters_numbers", "cln_")):
        return (code_scope, "dual_task_stability_fatigue")
    if token.startswith(("rapid_tracking", "rt_")):
        return (code_scope, "tracking_stability_low_load")
    if token.startswith(("dtb_", "auditory_capacity", "ac_")):
        return (code_scope, "dual_task_stability_fatigue")
    if token.startswith(("vigilance", "vig_")):
        return (code_scope, "vigilance_monitoring")
    if token.startswith(("target_recognition", "tr_")):
        return (code_scope, "target_recognition_scan")
    if token.startswith(("instrument_comprehension", "ic_")):
        return (code_scope, "instrument_orientation")
    if token.startswith(("angles_bearings_degrees", "abd_")):
        return (code_scope, "angles_bearings_degrees")
    if token.startswith(("spatial_integration", "si_")):
        return (code_scope, "spatial_integration")
    if token.startswith(("situational_awareness", "sa_")):
        return (code_scope, "situational_awareness")
    if token.startswith(("cognitive_updating", "cu_")):
        return (code_scope, "cognitive_updating")
    if token.startswith(("sensory_motor_apparatus", "sma_")):
        return (code_scope, "sensory_motor_control")
    if token.startswith(("trace_test_", "tt1_", "tt2_", "trace_")):
        return (code_scope, "trace_control")
    if token.endswith("_workout"):
        return (code_scope, token[:-8])
    return (code_scope, token)


def family_id_for_code(test_code: str | None) -> DifficultyFamilyId:
    token = str(test_code or "").strip().lower()
    if token in {"adaptive_session", "adaptive_session_short", "adaptive_session_micro"}:
        return "quantitative"
    try:
        from .canonical_drill_registry import canonical_drill_spec, resolved_canonical_drill_spec
    except Exception:
        canonical_drill_spec = None
        resolved_canonical_drill_spec = None
    if canonical_drill_spec is not None and resolved_canonical_drill_spec is not None:
        direct = canonical_drill_spec(token)
        if direct is not None:
            return direct.difficulty_family_id
        resolved = resolved_canonical_drill_spec(token)
        if resolved is not None:
            return resolved.difficulty_family_id
    guide_test = official_guide_test(token)
    if guide_test is not None:
        return cast(DifficultyFamilyId, guide_test.difficulty_family_id)
    if token.startswith(
        (
            "numerical_operations",
            "math_reasoning",
            "airborne_numerical",
            "no_",
            "mr_",
            "ma_",
            "ant_",
        )
    ):
        return "quantitative"
    if token.startswith(("angles_bearings_degrees", "abd_")):
        return "angle_bearing"
    if token.startswith(("auditory_capacity", "ac_")):
        return "auditory_multitask"
    if token.startswith(("colours_letters_numbers", "cln_")):
        return "cln_multitask"
    if token.startswith(("instrument_comprehension", "ic_")):
        return "instrument_orientation"
    if token.startswith(("cognitive_updating", "digit_recognition", "cu_", "dr_")):
        return "visual_memory_updating"
    if token.startswith(("situational_awareness", "sa_")):
        return "situational_awareness"
    if token.startswith(("table_reading", "tbl_")):
        return "table_cross_reference"
    if token.startswith(("system_logic", "sl_")):
        return "system_logic"
    if token.startswith(("visual_search", "vigilance", "target_recognition", "vs_", "vig_", "tr_")):
        return "search_vigilance"
    if token.startswith(("rapid_tracking", "sensory_motor_apparatus", "rt_", "sma_", "dtb_")):
        return "psychomotor_tracking"
    if token.startswith(("spatial_integration", "trace_test_1", "trace_test_2", "si_", "trace_", "tt1_", "tt2_")):
        return "spatial_integration_trace"
    code_scope, primitive_scope = scope_keys_for_code(token)
    for candidate in (token, code_scope, primitive_scope):
        lower = str(candidate).strip().lower()
        if "table" in lower:
            return "table_cross_reference"
        if "logic" in lower or "symbol" in lower:
            return "system_logic"
        if "tracking" in lower or "trace" in lower or "sma" in lower:
            return "psychomotor_tracking"
        if "spatial" in lower:
            return "spatial_integration_trace"
        if "visual" in lower or "search" in lower or "target" in lower or "vig" in lower:
            return "search_vigilance"
        if "situational" in lower:
            return "situational_awareness"
    return "quantitative"


def _normalize_intended_use(mode: object | None, *, default: DifficultyIntendedUse = "build") -> DifficultyIntendedUse:
    token = str(getattr(mode, "value", mode) or "").strip().lower()
    if token in {"anchor", "fresh"}:
        return "anchor"
    if token in {"build", "fixed", "adaptive", "recovery", ""}:
        return "build"
    if token == "tempo":
        return "tempo"
    if token in {"pressure", "stress"}:
        return "pressure"
    if token == "fatigue_probe":
        return "fatigue_probe"
    return default


def _challenge_band_for_use(intended_use: DifficultyIntendedUse) -> ChallengeBand:
    if intended_use in {"anchor", "build"}:
        return _BUILD_BAND
    if intended_use in {"tempo", "pressure"}:
        return _TEMPO_BAND
    return _FATIGUE_PROBE_BAND


def _mode_adjusted_level(level: int, intended_use: DifficultyIntendedUse) -> int:
    return clamp_level(clamp_level(level) + _MODE_LEVEL_OFFSETS[intended_use])


def _curve_content(progress: float) -> float:
    return clamp_ratio(progress ** 1.08)


def _curve_time(progress: float) -> float:
    return clamp_ratio((progress * 1.08) - 0.02)


def _curve_distractor(progress: float) -> float:
    return clamp_ratio((progress - 0.10) / 0.90)


def _curve_concurrency(progress: float) -> float:
    return clamp_ratio((progress - 0.15) / 0.85)


def _curve_memory(progress: float) -> float:
    return clamp_ratio((progress - 0.05) / 0.95)


def _curve_switch(progress: float) -> float:
    return clamp_ratio((progress - 0.08) / 0.92)


def _curve_control(progress: float) -> float:
    return clamp_ratio((progress - 0.12) / 0.88)


def _curve_spatial(progress: float) -> float:
    return clamp_ratio((progress - 0.05) / 0.95)


def _curve_source(progress: float) -> float:
    return clamp_ratio((progress - 0.10) / 0.90)


def _axis_value(base_curve: float, weight: float, boost: float) -> float:
    return clamp_ratio(clamp_ratio(base_curve + boost) * max(0.0, float(weight)))


def _axis_values_for_family(
    config: _FamilyDifficultyConfig,
    *,
    level: int,
    intended_use: DifficultyIntendedUse,
) -> DifficultyAxes:
    adjusted_level = _mode_adjusted_level(level, intended_use)
    progress = float(adjusted_level - 1) / 9.0
    boost = _MODE_AXIS_BOOSTS[intended_use]
    weights = config.axis_weights
    return DifficultyAxes(
        content_complexity=_axis_value(_curve_content(progress), weights.content_complexity, boost.content_complexity),
        time_pressure=_axis_value(_curve_time(progress), weights.time_pressure, boost.time_pressure),
        distractor_density=_axis_value(_curve_distractor(progress), weights.distractor_density, boost.distractor_density),
        multitask_concurrency=_axis_value(_curve_concurrency(progress), weights.multitask_concurrency, boost.multitask_concurrency),
        memory_span_delay=_axis_value(_curve_memory(progress), weights.memory_span_delay, boost.memory_span_delay),
        switch_frequency=_axis_value(_curve_switch(progress), weights.switch_frequency, boost.switch_frequency),
        control_sensitivity=_axis_value(_curve_control(progress), weights.control_sensitivity, boost.control_sensitivity),
        spatial_ambiguity=_axis_value(_curve_spatial(progress), weights.spatial_ambiguity, boost.spatial_ambiguity),
        source_integration_depth=_axis_value(_curve_source(progress), weights.source_integration_depth, boost.source_integration_depth),
    ).clamped()


def difficulty_profile_for_family(
    family_id: DifficultyFamilyId,
    level: int,
    mode: object | None = "build",
) -> DifficultyProfile:
    config = _FAMILY_REGISTRY[family_id]
    clamped_level = clamp_level(level)
    intended_use = _normalize_intended_use(mode)
    axis_values = _axis_values_for_family(config, level=clamped_level, intended_use=intended_use)
    legacy_ratio = DifficultyLadderSpec(
        scope_key=family_id,
        label=config.label,
        level_ratios=config.legacy_ratios,
    ).ratio_for_level(_mode_adjusted_level(clamped_level, intended_use))
    label = f"{config.label} L{clamped_level} {intended_use.replace('_', ' ').title()}"
    return DifficultyProfile(
        family_id=family_id,
        level=clamped_level,
        label=label,
        intended_use=intended_use,
        axis_values=axis_values,
        axis_weights=config.axis_weights.clamped(),
        legacy_ratio=float(legacy_ratio),
    )


def difficulty_profile_for_code(
    test_code: str | None,
    level: int,
    mode: object | None = "build",
) -> DifficultyProfile:
    return difficulty_profile_for_family(family_id_for_code(test_code), level, mode)


def ladder_spec_for_code(test_code: str | None) -> DifficultyLadderSpec:
    family_id = family_id_for_code(test_code)
    config = _FAMILY_REGISTRY[family_id]
    return DifficultyLadderSpec(
        scope_key=family_id,
        label=config.label,
        level_ratios=config.legacy_ratios,
    )


def difficulty_ratio_for_level(test_code: str | None, level: int, mode: object | None = "build") -> float:
    return float(difficulty_profile_for_code(test_code, level, mode).legacy_ratio)


def difficulty_level_for_ratio(test_code: str | None, value: float) -> int:
    ratio = clamp_ratio(value)
    spec = ladder_spec_for_code(test_code)
    distances = [
        (abs(ratio - float(level_ratio)), index + 1)
        for index, level_ratio in enumerate(spec.level_ratios or _DEFAULT_LEVEL_RATIOS)
    ]
    distances.sort(key=lambda item: (item[0], item[1]))
    return clamp_level(distances[0][1])


def build_resolved_difficulty_context(
    test_code: str | None,
    *,
    mode: LaunchDifficultyMode,
    launch_level: int,
    fixed_level: int | None = None,
    adaptive_enabled: bool | None = None,
) -> ResolvedDifficultyContext:
    token = str(test_code or "").strip()
    code_scope, primitive_scope = scope_keys_for_code(token)
    family_scope = family_id_for_code(token)
    resolved_launch = clamp_level(launch_level)
    resolved_fixed = resolved_launch if fixed_level is None else clamp_level(fixed_level)
    adaptive_flag = (
        bool(adaptive_enabled)
        if adaptive_enabled is not None
        else bool(mode == "adaptive" and supports_adaptive_difficulty(token))
    )
    return ResolvedDifficultyContext(
        test_code=token,
        mode=mode,
        fixed_level=resolved_fixed,
        launch_level=resolved_launch,
        launch_ratio=float(difficulty_ratio_for_level(token, resolved_launch)),
        code_scope_key=code_scope,
        primitive_scope_key=primitive_scope,
        family_scope_key=str(family_scope),
        adaptive_enabled=adaptive_flag,
    )


def apply_level_to_engine(engine: object, *, test_code: str | None, level: int) -> float:
    ratio = difficulty_ratio_for_level(test_code, level)
    if hasattr(engine, "_difficulty"):
        try:
            setattr(engine, "_difficulty", float(ratio))
        except Exception:
            pass
    return float(ratio)


def policy_for_activity_kind(
    activity_kind: str | None,
    *,
    intended_use: object | None = None,
) -> AdaptivePolicy:
    token = str(activity_kind or "").strip().lower()
    normalized_use = _normalize_intended_use(intended_use, default="tempo" if token in {"cognitive_test", "test"} else "build")
    max_delta = 2 if token in {"cognitive_test", "test"} else 3
    return AdaptivePolicy(
        short_window_size=3,
        medium_window_size=12,
        throttle_units=3,
        max_delta=max_delta,
        session_gain_cap=2,
        challenge_band=_challenge_band_for_use(normalized_use),
    )


def next_level_from_performance(
    *,
    current_level: int,
    launch_level: int,
    block_start_level: int | None = None,
    intended_use: object | None,
    recent_accuracy: float | None,
    recent_score_ratio: float | None,
    recent_timeout_rate: float | None,
    medium_accuracy: float | None,
    medium_score_ratio: float | None,
    medium_timeout_rate: float | None,
    max_delta: int,
    session_gain_cap: int = 2,
    severe_meltdown: bool = False,
) -> int:
    normalized_use = _normalize_intended_use(intended_use)
    band = _challenge_band_for_use(normalized_use)
    current = clamp_level(current_level)
    launch = clamp_level(launch_level)
    block_start = launch if block_start_level is None else clamp_level(block_start_level)
    recent_perf = recent_score_ratio if recent_score_ratio is not None else recent_accuracy
    medium_perf = medium_score_ratio if medium_score_ratio is not None else medium_accuracy
    recent_timeout = 0.0 if recent_timeout_rate is None else clamp_ratio(recent_timeout_rate)
    medium_timeout = 0.0 if medium_timeout_rate is None else clamp_ratio(medium_timeout_rate)

    if recent_perf is None and medium_perf is None:
        return current

    meltdown = bool(severe_meltdown)
    if recent_accuracy is not None and float(recent_accuracy) <= band.meltdown_accuracy:
        meltdown = True
    if recent_timeout >= band.meltdown_timeout_rate:
        meltdown = True
    if medium_accuracy is not None and float(medium_accuracy) <= max(0.0, band.meltdown_accuracy - 0.02):
        meltdown = True

    target = current
    if meltdown:
        target = current - 1
    else:
        below_band = False
        if recent_perf is not None and float(recent_perf) < band.retain_accuracy:
            below_band = True
        if medium_perf is not None and float(medium_perf) < band.retain_accuracy:
            below_band = True
        if recent_timeout > band.max_timeout_rate or medium_timeout > band.max_timeout_rate:
            below_band = True
        if below_band:
            target = current - 1
        elif band.allow_promotion:
            if (
                recent_perf is not None
                and medium_perf is not None
                and float(recent_perf) >= band.promote_accuracy
                and float(medium_perf) >= band.promote_accuracy
                and recent_timeout <= band.promote_timeout_rate
                and medium_timeout <= band.promote_timeout_rate
            ):
                target = current + 1

    bounded_target = clamp_level(target)
    demotion_floor = clamp_level(launch - int(max_delta))
    promotion_ceiling = clamp_level(
        min(
            launch + int(max_delta),
            launch + int(session_gain_cap),
            block_start + 1,
        )
    )
    if bounded_target > current:
        return max(current, min(bounded_target, promotion_ceiling))
    if bounded_target < current:
        return min(current, max(bounded_target, demotion_floor))
    return current


class AdaptiveDifficultyController:
    def __init__(
        self,
        *,
        launch_level: int,
        policy: AdaptivePolicy,
        block_start_level: int | None = None,
    ) -> None:
        self._launch_level = clamp_level(launch_level)
        self._block_start_level = (
            self._launch_level if block_start_level is None else clamp_level(block_start_level)
        )
        self._current_level = self._launch_level
        self._policy = policy
        self._observations: deque[_AdaptiveObservation] = deque(
            maxlen=max(1, int(policy.medium_window_size))
        )
        self._decisions: list[AdaptiveDifficultyDecision] = []
        self._units_since_change = 0
        self._total_units = 0

    @property
    def launch_level(self) -> int:
        return self._launch_level

    @property
    def current_level(self) -> int:
        return self._current_level

    @property
    def block_start_level(self) -> int:
        return self._block_start_level

    def decisions(self) -> list[AdaptiveDifficultyDecision]:
        return list(self._decisions)

    def record_item(
        self,
        *,
        is_correct: bool,
        is_timeout: bool,
        score_ratio: float | None,
        response_time_ms: float | None,
    ) -> AdaptiveDifficultyDecision | None:
        return self._record_observation(
            attempted=1,
            correct=1 if is_correct else 0,
            timeout_rate=1.0 if is_timeout else 0.0,
            score_ratio=score_ratio,
            mean_rt_ms=response_time_ms,
        )

    def record_window(
        self,
        *,
        attempted: int,
        correct: int,
        timeout_rate: float,
        score_ratio: float | None,
        mean_rt_ms: float | None,
    ) -> AdaptiveDifficultyDecision | None:
        return self._record_observation(
            attempted=attempted,
            correct=correct,
            timeout_rate=timeout_rate,
            score_ratio=score_ratio,
            mean_rt_ms=mean_rt_ms,
        )

    def _record_observation(
        self,
        *,
        attempted: int,
        correct: int,
        timeout_rate: float,
        score_ratio: float | None,
        mean_rt_ms: float | None,
    ) -> AdaptiveDifficultyDecision | None:
        attempted = max(0, int(attempted))
        if attempted <= 0:
            return None
        observation = _AdaptiveObservation(
            attempted=attempted,
            correct=max(0, min(attempted, int(correct))),
            timeout_rate=max(0.0, min(1.0, float(timeout_rate))),
            score_ratio=None if score_ratio is None else clamp_ratio(score_ratio),
            mean_rt_ms=None if mean_rt_ms is None else max(0.0, float(mean_rt_ms)),
        )
        self._observations.append(observation)
        self._total_units += attempted
        self._units_since_change += attempted
        return self._maybe_change_level()

    def _maybe_change_level(self) -> AdaptiveDifficultyDecision | None:
        short_metrics = self._aggregate_recent(self._policy.short_window_size)
        medium_metrics = self._aggregate_recent(self._policy.medium_window_size)
        meltdown = self._is_severe_meltdown(short_metrics, medium_metrics)
        if self._units_since_change < int(self._policy.throttle_units) and not meltdown:
            return None

        target_level = next_level_from_performance(
            current_level=self._current_level,
            launch_level=self._launch_level,
            block_start_level=self._block_start_level,
            intended_use=self._policy.challenge_band.key,
            recent_accuracy=short_metrics["accuracy"],
            recent_score_ratio=short_metrics["score_ratio"],
            recent_timeout_rate=short_metrics["timeout_rate"],
            medium_accuracy=medium_metrics["accuracy"],
            medium_score_ratio=medium_metrics["score_ratio"],
            medium_timeout_rate=medium_metrics["timeout_rate"],
            max_delta=self._policy.max_delta,
            session_gain_cap=self._policy.session_gain_cap,
            severe_meltdown=meltdown,
        )
        if target_level == self._current_level:
            return None

        reason = (
            f"{self._policy.challenge_band.label}: "
            f"short acc {short_metrics['accuracy']:.2f}, "
            f"short timeout {short_metrics['timeout_rate']:.2f}, "
            f"medium acc {medium_metrics['accuracy']:.2f}, "
            f"medium timeout {medium_metrics['timeout_rate']:.2f}"
        )
        decision = AdaptiveDifficultyDecision(
            after_units=int(self._total_units),
            old_level=int(self._current_level),
            new_level=int(target_level),
            reason=reason,
        )
        self._current_level = int(target_level)
        self._decisions.append(decision)
        self._units_since_change = 0
        return decision

    def _aggregate_recent(self, count: int) -> dict[str, float | None]:
        recent = list(self._observations)[-max(1, int(count)) :]
        attempted = sum(item.attempted for item in recent)
        if attempted <= 0:
            return {
                "attempted": 0.0,
                "accuracy": 0.0,
                "timeout_rate": 0.0,
                "score_ratio": None,
                "mean_rt_ms": None,
            }
        correct = sum(item.correct for item in recent)
        timeout_weighted = sum(float(item.timeout_rate) * float(item.attempted) for item in recent)
        score_weighted = 0.0
        score_units = 0
        rt_weighted = 0.0
        rt_units = 0
        for item in recent:
            if item.score_ratio is not None:
                score_weighted += float(item.score_ratio) * float(item.attempted)
                score_units += int(item.attempted)
            if item.mean_rt_ms is not None:
                rt_weighted += float(item.mean_rt_ms) * float(item.attempted)
                rt_units += int(item.attempted)
        return {
            "attempted": float(attempted),
            "accuracy": float(correct) / float(attempted),
            "timeout_rate": float(timeout_weighted) / float(attempted),
            "score_ratio": None if score_units <= 0 else float(score_weighted) / float(score_units),
            "mean_rt_ms": None if rt_units <= 0 else float(rt_weighted) / float(rt_units),
        }

    def _is_severe_meltdown(
        self,
        short_metrics: dict[str, float | None],
        medium_metrics: dict[str, float | None],
    ) -> bool:
        band = self._policy.challenge_band
        recent_accuracy = short_metrics["accuracy"]
        recent_timeout = short_metrics["timeout_rate"]
        medium_accuracy = medium_metrics["accuracy"]
        return bool(
            (recent_accuracy is not None and float(recent_accuracy) <= band.meltdown_accuracy)
            or (recent_timeout is not None and float(recent_timeout) >= band.meltdown_timeout_rate)
            or (medium_accuracy is not None and float(medium_accuracy) <= max(0.0, band.meltdown_accuracy - 0.02))
        )


def metric_float(metrics: dict[str, str], key: str) -> float | None:
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
    return value if value == value else None


def recommended_level_for_primitive(
    *,
    code_state: AdaptiveDifficultyState | None = None,
    family_state: AdaptiveDifficultyState | None = None,
    primitive_state: AdaptiveDifficultyState | None = None,
    fallback_level: int = 5,
) -> int:
    for state in (code_state, family_state, primitive_state):
        if state is None:
            continue
        return clamp_level(int(state.recommended_level))
    return clamp_level(fallback_level)


def merge_adaptive_state(
    previous: AdaptiveDifficultyState | None,
    *,
    scope_kind: ScopeKind,
    scope_key: str,
    start_level: int | None,
    end_level: int | None,
    accuracy: float | None,
    score_ratio: float | None,
    timeout_rate: float | None,
    mean_rt_ms: float | None,
    updated_at_utc: str,
    training_mode: str | None = None,
) -> AdaptiveDifficultyState:
    prev_level = 5 if previous is None else int(previous.recommended_level)
    sample_count = 1 if previous is None else int(previous.sample_count) + 1
    alpha = 0.35

    def _ewma(prev: float | None, current: float | None) -> float | None:
        if current is None:
            return prev
        if prev is None:
            return float(current)
        return (alpha * float(current)) + ((1.0 - alpha) * float(prev))

    ewma_accuracy = _ewma(None if previous is None else previous.ewma_accuracy, accuracy)
    ewma_score_ratio = _ewma(None if previous is None else previous.ewma_score_ratio, score_ratio)
    ewma_timeout = _ewma(None if previous is None else previous.ewma_timeout_rate, timeout_rate)
    ewma_rt = _ewma(None if previous is None else previous.ewma_mean_rt_ms, mean_rt_ms)

    baseline_level = clamp_level(end_level or start_level or prev_level)
    target_level = next_level_from_performance(
        current_level=baseline_level,
        launch_level=clamp_level(start_level or prev_level),
        block_start_level=clamp_level(start_level or prev_level),
        intended_use=training_mode,
        recent_accuracy=accuracy,
        recent_score_ratio=score_ratio,
        recent_timeout_rate=timeout_rate,
        medium_accuracy=ewma_accuracy,
        medium_score_ratio=ewma_score_ratio,
        medium_timeout_rate=ewma_timeout,
        max_delta=3,
        session_gain_cap=2,
        severe_meltdown=False,
    )
    recommended = clamp_level(
        int(round(((float(prev_level) * 1.0) + (float(target_level) * 2.0)) / 3.0))
    )
    return AdaptiveDifficultyState(
        scope_kind=cast(ScopeKind, scope_kind),
        scope_key=str(scope_key),
        recommended_level=int(recommended),
        last_start_level=None if start_level is None else clamp_level(start_level),
        last_end_level=None if end_level is None else clamp_level(end_level),
        ewma_accuracy=ewma_accuracy,
        ewma_score_ratio=ewma_score_ratio,
        ewma_timeout_rate=ewma_timeout,
        ewma_mean_rt_ms=ewma_rt,
        sample_count=int(sample_count),
        updated_at_utc=str(updated_at_utc),
    )
