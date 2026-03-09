from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from cfast_trainer.app import App, CognitiveTestScreen, MenuItem, MenuScreen
from cfast_trainer.cognitive_core import Phase
from cfast_trainer.cognitive_core import TestSnapshot as SnapshotModel


class _PromptAdvanceEngine:
    def __init__(self) -> None:
        self._phase = Phase.PRACTICE
        self._prompt = "4 + 5 ="
        self._update_count = 0
        self._difficulty = 0.5

    def snapshot(self) -> SnapshotModel:
        return SnapshotModel(
            title="Generic Prompt Reset",
            phase=self._phase,
            prompt=self._prompt,
            input_hint="Type answer then Enter",
            time_remaining_s=20.0,
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
        self._update_count += 1
        if self._update_count >= 2:
            self._prompt = "6 + 7 ="


def test_generic_text_input_clears_when_prompt_auto_advances() -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        app = App(surface=surface, font=font)
        root = MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True)
        app.push(root)
        screen = CognitiveTestScreen(app, engine_factory=_PromptAdvanceEngine)
        app.push(screen)

        screen.render(surface)
        screen._input = "123"
        screen.render(surface)

        assert screen._input == ""
    finally:
        pygame.quit()
