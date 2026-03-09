from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from cfast_trainer.app import TEST_DIFFICULTY_OPTIONS, TEST_GUIDE_BRIEFS


def test_every_test_has_a_guide_based_intro_briefing() -> None:
    expected_codes = {code for code, _label in TEST_DIFFICULTY_OPTIONS}
    assert expected_codes == set(TEST_GUIDE_BRIEFS)

    for code in expected_codes:
        briefing = TEST_GUIDE_BRIEFS[code]
        assert briefing.label.strip() != ""
        assert briefing.assessment.strip() != ""
        assert briefing.tasks
        assert all(task.strip() != "" for task in briefing.tasks)
        assert briefing.timing.strip() != ""
        assert briefing.prep.strip() != ""
        assert briefing.controls.strip() != ""
        assert briefing.app_flow.strip() != ""


def test_airborne_numerical_briefings_match_live_kernel_state() -> None:
    mixed = TEST_GUIDE_BRIEFS["ant_mixed_tempo_set"]
    assert "still only covers" not in mixed.app_flow.lower()
    assert "surrogate" not in TEST_GUIDE_BRIEFS["airborne_numerical_workout"].app_flow.lower()
