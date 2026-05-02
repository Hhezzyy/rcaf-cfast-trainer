from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from cfast_trainer.airborne_numerical import (
    TEMPLATES_BY_NAME,
    AirborneNumericalGenerator,
    AirborneScenario,
)
from cfast_trainer.app import App, CognitiveTestScreen, MenuItem, MenuScreen
from cfast_trainer.cognitive_core import Phase, SeededRng
from cfast_trainer.cognitive_core import TestSnapshot as SnapshotModel


@dataclass
class _FakeAirborneEngine:
    payload: AirborneScenario

    def snapshot(self) -> SnapshotModel:
        return SnapshotModel(
            title="Airborne Numerical",
            phase=Phase.PRACTICE,
            prompt="Estimate the route value.",
            input_hint="",
            time_remaining_s=None,
            attempted_scored=0,
            correct_scored=0,
            payload=self.payload,
        )

    def can_exit(self) -> bool:
        return True

    def start_practice(self) -> None:
        return

    def start_scored(self) -> None:
        return

    def submit_answer(self, raw: str) -> bool:
        return True

    def update(self) -> None:
        return


def _build_screen(engine: _FakeAirborneEngine) -> tuple[App, CognitiveTestScreen]:
    pygame.init()
    surface = pygame.display.set_mode((960, 540))
    font = pygame.font.Font(None, 36)
    app = App(surface=surface, font=font)
    root = MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True)
    app.push(root)
    screen = CognitiveTestScreen(app, engine_factory=lambda: engine)
    app.push(screen)
    return app, screen


def _sample_airborne_scenario() -> AirborneScenario:
    problem = AirborneNumericalGenerator(SeededRng(417)).generate()
    scenario = problem.payload
    assert isinstance(scenario, AirborneScenario)
    assert scenario.route
    return scenario


def test_airborne_map_omits_start_dot_and_parcel_icon(monkeypatch) -> None:
    scenario = _sample_airborne_scenario()
    _app, screen = _build_screen(_FakeAirborneEngine(scenario))
    circle_colors: list[tuple[int, int, int]] = []
    rect_colors: list[tuple[int, int, int]] = []
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        original_circle = pygame.draw.circle
        original_rect = pygame.draw.rect

        def wrapped_circle(*args, **kwargs):
            circle_colors.append(tuple(int(v) for v in tuple(args[1])[:3]))
            return original_circle(*args, **kwargs)

        def wrapped_rect(*args, **kwargs):
            rect_colors.append(tuple(int(v) for v in tuple(args[1])[:3]))
            return original_rect(*args, **kwargs)

        monkeypatch.setattr(pygame.draw, "circle", wrapped_circle)
        monkeypatch.setattr(pygame.draw, "rect", wrapped_rect)

        screen._draw_airborne_map_guide_panel(
            surface,
            pygame.Rect(24, 24, 560, 360),
            scenario=scenario,
            panel_bg=(248, 249, 252),
            text_main=(42, 42, 42),
        )

        node_count = len(TEMPLATES_BY_NAME[scenario.template_name].nodes)
        assert circle_colors.count((244, 244, 244)) >= node_count
        assert (255, 18, 24) not in circle_colors
        assert (128, 0, 0) not in circle_colors
        assert (72, 170, 214) not in rect_colors
        assert (28, 98, 136) not in rect_colors
        assert (214, 182, 132) not in rect_colors
        assert (118, 88, 48) not in rect_colors
    finally:
        pygame.quit()
