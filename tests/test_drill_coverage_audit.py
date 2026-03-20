from __future__ import annotations

from cfast_trainer.canonical_drill_registry import CanonicalDrillSpec
from cfast_trainer.drill_coverage_audit import build_drill_coverage_audit
from cfast_trainer.guide_skill_catalog import subskill_coverage_expectations


def _spec(
    *,
    code: str,
    primary_subskill: str,
    status: str = "canonical_keep",
    granularity: str = "micro",
    axes: tuple[str, ...] = ("content_complexity",),
    notes: str = "Progresses across L1-L10.",
) -> CanonicalDrillSpec:
    return CanonicalDrillSpec(
        drill_code=code,
        title=code.replace("_", " ").title(),
        granularity=granularity,  # type: ignore[arg-type]
        primary_subskill=primary_subskill,
        secondary_subskills=(),
        guide_test_links=("numerical_operations",),
        supports_modes=("build",),
        status=status,  # type: ignore[arg-type]
        difficulty_family_id="quantitative",
        difficulty_axes_used=axes,
        supported_levels=tuple(range(1, 11)),
        default_start_level=3,
        difficulty_progression_notes=notes,
    )


def test_coverage_audit_is_deterministic_and_partitions_known_subskills() -> None:
    first = build_drill_coverage_audit()
    second = build_drill_coverage_audit()
    expected = set(subskill_coverage_expectations())

    assert first == second
    assert set(first.covered_subskills) | set(first.partial_subskills) | set(first.missing_subskills) == expected
    assert not (
        set(first.covered_subskills)
        & set(first.partial_subskills)
        & set(first.missing_subskills)
    )


def test_coverage_audit_flags_known_redundant_and_orphan_codes() -> None:
    audit = build_drill_coverage_audit()

    assert "vs_target_preview" in audit.redundant_drill_codes
    assert "numerical_operations_workout" in audit.orphan_drill_codes


def test_coverage_audit_classifies_stub_and_dead_scale_entries() -> None:
    stub = _spec(code="stub_drill", primary_subskill="stub_skill", axes=(), notes="")
    dead = _spec(code="dead_drill", primary_subskill="dead_skill")
    live = _spec(code="live_drill", primary_subskill="live_skill")

    audit = build_drill_coverage_audit(
        registry=(stub, dead, live),
        convergence=(),
        reachable_codes=("stub_drill", "dead_drill", "live_drill"),
        coverage_expectations={
            "stub_skill": ("stub_drill",),
            "dead_skill": ("dead_drill",),
            "live_skill": ("live_drill",),
        },
        behavior_probe_signatures={
            "dead_drill": {2: ("same",), 5: ("same",), 8: ("same",)},
            "live_drill": {2: ("low",), 5: ("mid",), 8: ("high",)},
        },
    )

    assert "stub_drill" in audit.difficulty_stub_drill_codes
    assert "dead_drill" in audit.difficulty_dead_scale_drill_codes
    assert "live_drill" in audit.difficulty_integrated_drill_codes
    assert "live_skill" in audit.covered_subskills
    assert "stub_skill" in audit.partial_subskills
    assert "dead_skill" in audit.partial_subskills
