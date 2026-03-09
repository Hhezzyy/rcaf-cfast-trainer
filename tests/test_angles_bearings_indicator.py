from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from cfast_trainer.app import CognitiveTestScreen


def test_angle_indicator_bearings_follow_smaller_sweep() -> None:
    bearings = CognitiveTestScreen._angle_indicator_bearings(0, 300, "smaller")

    assert bearings[0] == 0.0
    assert bearings[-1] == 300.0
    # Smaller sweep from 000 to 300 should move counterclockwise via 330, not clockwise via 060.
    assert all((bearing >= 300.0 or bearing <= 0.0) for bearing in bearings)


def test_angle_indicator_bearings_support_larger_sweep() -> None:
    bearings = CognitiveTestScreen._angle_indicator_bearings(0, 300, "larger")

    assert bearings[0] == 0.0
    assert bearings[-1] == 300.0
    assert any(0.0 < bearing < 180.0 for bearing in bearings[1:-1])
