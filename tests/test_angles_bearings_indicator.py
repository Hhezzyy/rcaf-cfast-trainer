from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from cfast_trainer.angles_bearings_degrees import (
    AnglesBearingsDegreesPayload,
    AnglesBearingsOption,
    AnglesBearingsQuestionKind,
)
from cfast_trainer.app import App, CognitiveTestScreen, MenuItem, MenuScreen
from cfast_trainer.cognitive_core import Phase
from cfast_trainer.cognitive_core import TestSnapshot as SnapshotModel


class _FakeAnglesEngine:
    def snapshot(self) -> SnapshotModel:
        return SnapshotModel(
            title="Angles, Bearings and Degrees",
            phase=Phase.PRACTICE,
            prompt="",
            input_hint="",
            time_remaining_s=None,
            attempted_scored=0,
            correct_scored=0,
            payload=None,
        )

    def can_exit(self) -> bool:
        return True

    def start_practice(self) -> None:
        return

    def start_scored(self) -> None:
        return

    def submit_answer(self, raw: str) -> bool:
        _ = raw
        return True

    def update(self) -> None:
        return


def _build_screen() -> CognitiveTestScreen:
    pygame.init()
    surface = pygame.display.set_mode((960, 540))
    font = pygame.font.Font(None, 36)
    app = App(surface=surface, font=font)
    root = MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True)
    app.push(root)
    screen = CognitiveTestScreen(app, engine_factory=_FakeAnglesEngine)
    app.push(screen)
    return screen


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


def test_angle_trial_keeps_yellow_indicator_but_removes_blue_protractor_arc() -> None:
    screen = _build_screen()
    calls: list[tuple[int, int, int]] = []
    original_lines = pygame.draw.lines
    try:
        payload = AnglesBearingsDegreesPayload(
            kind=AnglesBearingsQuestionKind.ANGLE_BETWEEN_LINES,
            stem="Estimate the smaller angle.",
            reference_bearing_deg=35,
            target_bearing_deg=282,
            angle_measure="smaller",
            object_label="",
            options=(
                AnglesBearingsOption(code=1, text="113", value_deg=113),
                AnglesBearingsOption(code=2, text="146", value_deg=146),
                AnglesBearingsOption(code=3, text="247", value_deg=247),
                AnglesBearingsOption(code=4, text="214", value_deg=214),
                AnglesBearingsOption(code=5, text="102", value_deg=102),
            ),
            correct_code=2,
            correct_value_deg=113,
        )
        surface = pygame.Surface((640, 480))

        def fake_lines(surface, color, closed, points, width=1):
            _ = (surface, closed, points, width)
            calls.append(tuple(int(channel) for channel in color))
            return pygame.Rect(0, 0, 1, 1)

        pygame.draw.lines = fake_lines  # type: ignore[assignment]
        screen._draw_angle_trial(surface, pygame.Rect(20, 20, 360, 360), payload)

        assert (255, 208, 104) in calls
        assert (108, 128, 178) not in calls
    finally:
        pygame.draw.lines = original_lines  # type: ignore[assignment]
        pygame.quit()
