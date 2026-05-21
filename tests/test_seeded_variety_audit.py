from __future__ import annotations

from pathlib import Path

import pytest

from cfast_trainer.activity_runtime_catalog import OFFICIAL_TEST_MENU_ORDER
from cfast_trainer.guide_skill_catalog import TEST_DIFFICULTY_OPTIONS
from cfast_trainer.seeded_variety_audit import (
    adaptive_plan_signature,
    benchmark_plan_signature,
    builder_backed_canonical_drills,
    canonical_drill_replay_signature,
    difficulty_ladder_signatures,
    official_runtime_replay_signature,
    randomness_source_violations,
    seed_settings_activity_codes,
)


def test_seed_settings_catalog_covers_tests_drills_workouts_and_seeded_sessions() -> None:
    codes = set(seed_settings_activity_codes())

    assert set(OFFICIAL_TEST_MENU_ORDER) <= codes
    assert {code for code, _label in TEST_DIFFICULTY_OPTIONS} <= codes
    assert "adaptive_session" in codes
    assert "benchmark_battery" in codes


@pytest.mark.parametrize("code", seed_settings_activity_codes())
def test_all_activity_difficulty_ladders_have_meaningful_1_to_10_steps(code: str) -> None:
    ladder = difficulty_ladder_signatures(code)
    ratios = [row.ratio for row in ladder]
    axis_totals = [sum(value for _name, value in row.axes) for row in ladder]

    assert len(ladder) == 10
    assert len(set(ratios)) == 10
    assert ratios == sorted(ratios)
    assert axis_totals[-1] > axis_totals[0]


@pytest.mark.parametrize("test_code", OFFICIAL_TEST_MENU_ORDER)
def test_official_runtime_signatures_replay_with_same_seed_and_level(test_code: str) -> None:
    first = official_runtime_replay_signature(test_code, seed=2026, level=6)
    second = official_runtime_replay_signature(test_code, seed=2026, level=6)

    assert first.signature == second.signature


@pytest.mark.parametrize("spec", builder_backed_canonical_drills(), ids=lambda spec: spec.drill_code)
def test_canonical_drill_signatures_replay_with_same_seed_and_level(spec) -> None:
    first = canonical_drill_replay_signature(spec, seed=4242, level=5)
    second = canonical_drill_replay_signature(spec, seed=4242, level=5)

    assert first.signature == second.signature


def test_benchmark_and_adaptive_plans_replay_from_session_seed() -> None:
    assert benchmark_plan_signature(seed=777) == benchmark_plan_signature(seed=777)
    assert benchmark_plan_signature(seed=777) != benchmark_plan_signature(seed=778)

    assert adaptive_plan_signature(seed=777) == adaptive_plan_signature(seed=777)
    assert adaptive_plan_signature(seed=777) != adaptive_plan_signature(seed=778)


def test_entropy_sources_are_limited_to_launch_restart_seed_helpers() -> None:
    assert randomness_source_violations(Path.cwd()) == ()


def test_godot_owned_random_generators_are_seeded_from_session_streams() -> None:
    scripts = Path.cwd() / "godot" / "cfast_3d" / "scripts"
    checked = 0
    for path in (
        scripts / "auditory_runtime.gd",
        scripts / "rapid_tracking_runtime.gd",
        scripts / "main.gd",
        scripts / "chunk_map_generator.gd",
        scripts / "spatial_integration_runtime.gd",
        scripts / "godot_owned_runtime.gd",
    ):
        lines = path.read_text(encoding="utf-8").splitlines()
        source = "\n".join(lines)
        for index, line in enumerate(lines):
            if "RandomNumberGenerator.new()" not in line:
                continue
            checked += 1
            if path.name == "godot_owned_runtime.gd" and line.startswith("var rng"):
                assert "rng.seed = session_seed + _kind_salt(kind)" in source
                continue
            snippet = "\n".join(lines[index : index + 6])
            assert ".seed =" in snippet, f"{path}:{index + 1} creates an unseeded RNG"
    assert checked >= 10
