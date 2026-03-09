from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from cfast_trainer.app import App, MenuItem, MenuScreen


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
