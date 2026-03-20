from __future__ import annotations

from cfast_trainer.guide_skill_catalog import (
    GUIDE_SKILL_CATALOG,
    OFFICIAL_GUIDE_TEST_CODES,
    OFFICIAL_GUIDE_TESTS,
    TEST_DIFFICULTY_OPTIONS,
    TEST_GUIDE_BRIEFS,
    guide_code_mapping,
    official_guide_tests,
)


def test_guide_catalog_is_deterministic() -> None:
    first = official_guide_tests()
    second = official_guide_tests()

    assert first == second
    assert GUIDE_SKILL_CATALOG == GUIDE_SKILL_CATALOG


def test_every_official_guide_test_is_present_once_with_required_metadata() -> None:
    tests = official_guide_tests()

    assert tuple(test.test_code for test in tests) == OFFICIAL_GUIDE_TEST_CODES
    assert len(OFFICIAL_GUIDE_TEST_CODES) == len(set(OFFICIAL_GUIDE_TEST_CODES))

    for test in tests:
        assert test.official_name.strip() != ""
        assert test.test_code.strip() != ""
        assert test.guide_duration_min > 0
        assert test.guide_prepability.strip() != ""
        assert test.devices
        assert test.component_skills
        assert test.component_subskills
        assert test.difficulty_family_id.strip() != ""
        assert test.difficulty_axes_used
        assert test.difficulty_notes.strip() != ""
        assert set(test.difficulty_description_by_axis) == set(test.difficulty_axes_used)


def test_each_skill_and_subskill_has_required_training_bands() -> None:
    for domain in GUIDE_SKILL_CATALOG:
        assert domain.domain_id.strip() != ""
        for family in domain.test_families:
            assert family.family_id.strip() != ""
            assert family.official_test_codes
            for skill in family.skills:
                assert 1 <= skill.suggested_start_level <= 10
                assert len(skill.recommended_training_band) == 2
                assert len(skill.safe_pressure_band) == 2
                for subskill in skill.subskills:
                    assert 1 <= subskill.suggested_start_level <= 10
                    assert len(subskill.recommended_training_band) == 2
                    assert len(subskill.safe_pressure_band) == 2
                    assert subskill.canonical_drill_codes


def test_guide_code_mapping_covers_official_tests_and_canonical_drills() -> None:
    assert guide_code_mapping("visual_search") is not None
    assert guide_code_mapping("visual_search").is_official_test is True
    assert guide_code_mapping("vs_multi_target_class_search") is not None
    assert guide_code_mapping("vs_multi_target_class_search").subskill_ids == ("class_search",)


def test_intro_briefing_truth_now_lives_in_catalog_module() -> None:
    expected_codes = {code for code, _label in TEST_DIFFICULTY_OPTIONS}

    assert expected_codes == set(TEST_GUIDE_BRIEFS)
