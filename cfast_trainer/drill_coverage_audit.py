from __future__ import annotations

from dataclasses import dataclass

from .adaptive_difficulty import difficulty_profile_for_code
from .canonical_drill_registry import (
    CANONICAL_DRILL_CONVERGENCE,
    CANONICAL_DRILL_REGISTRY,
    CanonicalDrillConvergence,
    CanonicalDrillSpec,
)
from .guide_skill_catalog import GUIDE_CODE_MAPPINGS, TEST_DIFFICULTY_OPTIONS, subskill_coverage_expectations
from .primitive_ranking import PRIMITIVES


@dataclass(frozen=True, slots=True)
class DrillCoverageAudit:
    covered_subskills: tuple[str, ...]
    partial_subskills: tuple[str, ...]
    missing_subskills: tuple[str, ...]
    redundant_drill_codes: tuple[str, ...]
    orphan_drill_codes: tuple[str, ...]
    difficulty_integrated_drill_codes: tuple[str, ...]
    difficulty_stub_drill_codes: tuple[str, ...]
    difficulty_dead_scale_drill_codes: tuple[str, ...]


def _reachable_codes() -> tuple[str, ...]:
    codes = {code for code, _label in TEST_DIFFICULTY_OPTIONS}
    for primitive in PRIMITIVES:
        codes.update(primitive.benchmark_probe_codes)
        codes.update(primitive.integrated_test_codes)
        codes.update(primitive.direct_drill_codes)
        codes.update(primitive.coarse_workout_codes)
        codes.update(primitive.anchor_templates)
        codes.update(primitive.tempo_templates)
        codes.update(primitive.reset_templates)
        codes.update(primitive.fatigue_templates)
    return tuple(sorted(code for code in codes if str(code).strip() != ""))


def _has_stub_difficulty(spec: CanonicalDrillSpec) -> bool:
    if not spec.supported_levels:
        return False
    return bool(
        spec.difficulty_family_id == ""
        or not spec.difficulty_axes_used
        or str(spec.difficulty_progression_notes).strip() == ""
    )


def _level_signature(
    spec: CanonicalDrillSpec,
    level: int,
    *,
    behavior_probe_signatures: dict[str, dict[int, tuple[object, ...]]] | None,
) -> tuple[object, ...]:
    if behavior_probe_signatures:
        code_probes = behavior_probe_signatures.get(spec.drill_code)
        if code_probes and int(level) in code_probes:
            return tuple(code_probes[int(level)])
    profile = difficulty_profile_for_code(spec.drill_code, int(level), "build")
    axis_values = tuple(
        round(float(getattr(profile.axes, axis)), 6)
        for axis in spec.difficulty_axes_used
    )
    return (
        int(profile.level),
        str(profile.intended_use),
        *axis_values,
    )


def _is_dead_scale(
    spec: CanonicalDrillSpec,
    *,
    behavior_probe_signatures: dict[str, dict[int, tuple[object, ...]]] | None,
) -> bool:
    if 2 not in spec.supported_levels or 5 not in spec.supported_levels or 8 not in spec.supported_levels:
        return False
    left = _level_signature(spec, 2, behavior_probe_signatures=behavior_probe_signatures)
    middle = _level_signature(spec, 5, behavior_probe_signatures=behavior_probe_signatures)
    right = _level_signature(spec, 8, behavior_probe_signatures=behavior_probe_signatures)
    return left == middle == right


def build_drill_coverage_audit(
    *,
    registry: tuple[CanonicalDrillSpec, ...] | None = None,
    convergence: tuple[CanonicalDrillConvergence, ...] | None = None,
    reachable_codes: tuple[str, ...] | None = None,
    coverage_expectations: dict[str, tuple[str, ...]] | None = None,
    behavior_probe_signatures: dict[str, dict[int, tuple[object, ...]]] | None = None,
) -> DrillCoverageAudit:
    active_registry = registry or CANONICAL_DRILL_REGISTRY
    active_convergence = convergence or CANONICAL_DRILL_CONVERGENCE
    active_reachable = reachable_codes or _reachable_codes()
    expected_subskills = coverage_expectations or subskill_coverage_expectations()

    specs_by_subskill: dict[str, list[CanonicalDrillSpec]] = {
        subskill_id: []
        for subskill_id in sorted(expected_subskills)
    }
    for spec in active_registry:
        keys = (spec.primary_subskill, *spec.secondary_subskills)
        for subskill_id in keys:
            token = str(subskill_id).strip().lower()
            if token == "":
                continue
            specs_by_subskill.setdefault(token, []).append(spec)

    stub_codes = sorted(
        spec.drill_code
        for spec in active_registry
        if _has_stub_difficulty(spec)
    )
    dead_codes = sorted(
        spec.drill_code
        for spec in active_registry
        if not _has_stub_difficulty(spec)
        and _is_dead_scale(spec, behavior_probe_signatures=behavior_probe_signatures)
    )
    integrated_codes = sorted(
        spec.drill_code
        for spec in active_registry
        if spec.drill_code not in stub_codes and spec.drill_code not in dead_codes
    )

    covered: list[str] = []
    partial: list[str] = []
    missing: list[str] = []
    for subskill_id in sorted(expected_subskills):
        specs = tuple(specs_by_subskill.get(subskill_id, ()))
        direct_canonical = [
            spec
            for spec in specs
            if spec.status == "canonical_keep"
            and spec.drill_code in integrated_codes
            and spec.drill_code not in dead_codes
        ]
        if direct_canonical:
            covered.append(subskill_id)
            continue
        if specs:
            partial.append(subskill_id)
            continue
        missing.append(subskill_id)

    redundant_codes = sorted(
        entry.legacy_code
        for entry in active_convergence
        if entry.hide_from_adaptive or entry.action == "replace_when_new_drill_exists"
    )
    known_codes = {spec.drill_code for spec in active_registry}
    known_codes.update(GUIDE_CODE_MAPPINGS)
    known_codes.update(entry.legacy_code for entry in active_convergence)
    orphan_codes = sorted(
        code
        for code in active_reachable
        if code not in known_codes
    )

    return DrillCoverageAudit(
        covered_subskills=tuple(sorted(covered)),
        partial_subskills=tuple(sorted(partial)),
        missing_subskills=tuple(sorted(missing)),
        redundant_drill_codes=tuple(redundant_codes),
        orphan_drill_codes=tuple(orphan_codes),
        difficulty_integrated_drill_codes=tuple(integrated_codes),
        difficulty_stub_drill_codes=tuple(stub_codes),
        difficulty_dead_scale_drill_codes=tuple(dead_codes),
    )
