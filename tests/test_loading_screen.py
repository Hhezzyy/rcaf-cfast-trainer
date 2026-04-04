from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from cfast_trainer.app import App, LoadingScreen, MenuItem, MenuScreen


class _TargetScreen:
    def __init__(self) -> None:
        self.render_count = 0

    def handle_event(self, event: pygame.event.Event) -> None:
        _ = event

    def render(self, surface: pygame.Surface) -> None:
        self.render_count += 1
        surface.fill((12, 18, 24))


def test_loading_screen_defers_target_construction_until_second_render() -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        app = App(surface=surface, font=font)
        app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))

        target = _TargetScreen()
        created: list[str] = []

        loading = LoadingScreen(
            app,
            title="Visual Search",
            detail="Preparing test",
            target_factory=lambda: created.append("loaded") or target,
        )
        app.push(loading)

        app.render()
        assert created == []
        assert isinstance(app._screens[-1], LoadingScreen)

        app.render()
        assert created == ["loaded"]
        assert app._screens[-1] is target

        app.render()
        assert target.render_count == 1
    finally:
        pygame.quit()


def test_loading_screen_escape_is_ignored_while_loading() -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        app = App(surface=surface, font=font)
        root = MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True)
        app.push(root)
        app.push(
            LoadingScreen(
                app,
                title="Trace Test 1",
                detail="Preparing test",
                target_factory=_TargetScreen,
            )
        )

        app.handle_event(
            pygame.event.Event(
                pygame.KEYDOWN,
                {"key": pygame.K_ESCAPE, "mod": 0, "unicode": ""},
            )
        )

        assert len(app._screens) == 2
        assert app._screens[-1] is not root
    finally:
        pygame.quit()
