from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from cfast_trainer.app import App, CognitiveTestScreen, MenuItem, MenuScreen
from cfast_trainer.cognitive_core import Phase
from cfast_trainer.cognitive_core import TestSnapshot as SnapshotModel


class _FakeEngine:
    def __init__(self, *, phase: Phase = Phase.PRACTICE) -> None:
        self._phase = phase
        self.update_count = 0

    def snapshot(self) -> SnapshotModel:
        return SnapshotModel(
            title="Fake Test",
            phase=self._phase,
            prompt="",
            input_hint="",
            time_remaining_s=60.0 if self._phase is Phase.SCORED else None,
            attempted_scored=0,
            correct_scored=0,
            payload=None,
        )

    def can_exit(self) -> bool:
        return self._phase in (
            Phase.INSTRUCTIONS,
            Phase.PRACTICE,
            Phase.PRACTICE_DONE,
            Phase.RESULTS,
        )

    def start_practice(self) -> None:
        if self._phase is Phase.INSTRUCTIONS:
            self._phase = Phase.PRACTICE

    def start_scored(self) -> None:
        if self._phase is Phase.PRACTICE_DONE:
            self._phase = Phase.SCORED

    def submit_answer(self, raw: str) -> bool:
        _ = raw
        return False

    def update(self) -> None:
        self.update_count += 1


def _build_app_and_screen(
    *, phase: Phase = Phase.PRACTICE
) -> tuple[App, CognitiveTestScreen, _FakeEngine]:
    pygame.init()
    surface = pygame.display.set_mode((960, 540))
    font = pygame.font.Font(None, 36)
    app = App(surface=surface, font=font)
    root = MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True)
    app.push(root)
    engine = _FakeEngine(phase=phase)
    screen = CognitiveTestScreen(app, engine_factory=lambda: engine)
    app.push(screen)
    return app, screen, engine


def test_pause_menu_escape_then_back_resumes_test() -> None:
    app, screen, _engine = _build_app_and_screen()
    try:
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "mod": 0, "unicode": ""})
        )
        assert screen._pause_menu_active is True

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": 0, "unicode": ""})
        )
        assert screen._pause_menu_active is False
        assert len(app._screens) == 2
    finally:
        pygame.quit()


def test_pause_menu_skip_advances_to_next_part_for_debugging() -> None:
    app, screen, engine = _build_app_and_screen(phase=Phase.PRACTICE)
    try:
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "mod": 0, "unicode": ""})
        )
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "mod": 0, "unicode": ""})
        )
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": 0, "unicode": ""})
        )

        assert screen._pause_menu_active is False
        assert engine.snapshot().phase is Phase.SCORED
        assert len(app._screens) == 2
    finally:
        pygame.quit()


def test_pause_menu_main_menu_returns_to_root() -> None:
    app, screen, _engine = _build_app_and_screen()
    try:
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "mod": 0, "unicode": ""})
        )
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "mod": 0, "unicode": ""})
        )
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "mod": 0, "unicode": ""})
        )
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": 0, "unicode": ""})
        )

        assert len(app._screens) == 1
    finally:
        pygame.quit()


def test_pause_menu_freezes_engine_updates_while_open() -> None:
    _app, screen, engine = _build_app_and_screen()
    try:
        surface = pygame.display.get_surface()
        assert surface is not None

        screen.render(surface)
        before_pause = engine.update_count
        assert before_pause > 0

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "mod": 0, "unicode": ""})
        )
        screen.render(surface)
        assert engine.update_count == before_pause
    finally:
        pygame.quit()
