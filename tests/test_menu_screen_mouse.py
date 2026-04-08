from __future__ import annotations

import os
import sys
from importlib.machinery import ModuleSpec
from types import ModuleType

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

if "moderngl" not in sys.modules:
    moderngl_stub = ModuleType("moderngl")
    moderngl_stub.__spec__ = ModuleSpec("moderngl", loader=None)
    sys.modules["moderngl"] = moderngl_stub

import pygame

from cfast_trainer.app import App, MenuItem, MenuScreen


class _PressedKeys:
    def __init__(self, active: set[int]) -> None:
        self._active = set(active)

    def __getitem__(self, key: int) -> int:
        return 1 if key in self._active else 0


def test_menu_screen_mouse_click_activates_clicked_item() -> None:
    pygame.init()
    called: list[str] = []
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        app = App(surface=surface, font=font)
        menu = MenuScreen(
            app,
            "Main Menu",
            [
                MenuItem("First", lambda: called.append("first")),
                MenuItem("Second", lambda: called.append("second")),
            ],
            is_root=True,
        )

        menu.render(surface)
        hitbox = menu._item_hitboxes[1]
        menu.handle_event(
            pygame.event.Event(
                pygame.MOUSEBUTTONDOWN,
                {"button": 1, "pos": hitbox.center},
            )
        )

        assert called == ["second"]
    finally:
        pygame.quit()


def test_menu_screen_mouse_click_activates_without_prior_render() -> None:
    pygame.init()
    called: list[str] = []
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        app = App(surface=surface, font=font)
        seeded_menu = MenuScreen(
            app,
            "Main Menu",
            [
                MenuItem("First", lambda: called.append("first")),
                MenuItem("Second", lambda: called.append("second")),
            ],
            is_root=True,
        )
        seeded_menu.render(surface)
        click_pos = seeded_menu._item_hitboxes[1].center

        menu = MenuScreen(
            app,
            "Main Menu",
            [
                MenuItem("First", lambda: called.append("first")),
                MenuItem("Second", lambda: called.append("second")),
            ],
            is_root=True,
        )
        menu.handle_event(
            pygame.event.Event(
                pygame.MOUSEBUTTONDOWN,
                {"button": 1, "pos": click_pos},
            )
        )

        assert called == ["second"]
    finally:
        pygame.quit()


def test_menu_screen_mouse_motion_updates_selection() -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        app = App(surface=surface, font=font)
        menu = MenuScreen(
            app,
            "Main Menu",
            [
                MenuItem("First", lambda: None),
                MenuItem("Second", lambda: None),
            ],
            is_root=True,
        )

        menu.render(surface)
        hitbox = menu._item_hitboxes[1]
        menu.handle_event(pygame.event.Event(pygame.MOUSEMOTION, {"pos": hitbox.center}))

        assert menu._selected == 1
    finally:
        pygame.quit()


def test_menu_screen_keyboard_hold_repeats_after_short_delay(monkeypatch) -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        app = App(surface=surface, font=font)
        menu = MenuScreen(
            app,
            "Main Menu",
            [
                MenuItem("One", lambda: None),
                MenuItem("Two", lambda: None),
                MenuItem("Three", lambda: None),
                MenuItem("Four", lambda: None),
            ],
            is_root=True,
        )
        app.push(menu)

        held_keys = {pygame.K_DOWN}
        now_ms = {"value": 0}
        monkeypatch.setattr(pygame.key, "get_pressed", lambda: _PressedKeys(held_keys))
        monkeypatch.setattr(pygame.time, "get_ticks", lambda: now_ms["value"])

        app.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "mod": 0, "unicode": ""})
        )
        assert menu._selected == 1

        app.render()
        assert menu._selected == 1

        now_ms["value"] = 220
        app.render()
        assert menu._selected == 1

        now_ms["value"] = 270
        app.render()
        assert menu._selected == 2

        now_ms["value"] = 390
        app.render()
        assert menu._selected == 3

        held_keys.clear()
        now_ms["value"] = 520
        app.render()
        assert menu._selected == 3
    finally:
        pygame.quit()


def test_app_ctrl_q_shortcut_quits_from_any_screen() -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        app = App(surface=surface, font=font)
        menu = MenuScreen(
            app,
            "Main Menu",
            [
                MenuItem("First", lambda: None),
                MenuItem("Second", lambda: None),
            ],
            is_root=True,
        )
        app.push(menu)

        app.handle_event(
            pygame.event.Event(
                pygame.KEYDOWN,
                {"key": pygame.K_q, "mod": pygame.KMOD_CTRL, "unicode": "q"},
            )
        )

        assert app.running is False
    finally:
        pygame.quit()
