from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
import pytest

from cfast_trainer.app import (
    App,
    DifficultySettingsScreen,
    DifficultySettingsStore,
    MenuItem,
    MenuScreen,
)


def test_difficulty_settings_store_persists_global_and_per_test_levels(tmp_path) -> None:
    path = tmp_path / "difficulty-settings.json"
    store = DifficultySettingsStore(path)

    assert store.global_override_enabled() is False
    assert store.global_level() == 5
    assert store.test_level("rapid_tracking") == 5

    store.set_test_level(test_code="rapid_tracking", level=8)
    store.set_global_level(9)
    store.set_global_override_enabled(True)

    reloaded = DifficultySettingsStore(path)
    assert reloaded.global_override_enabled() is True
    assert reloaded.global_level() == 9
    assert reloaded.test_level("rapid_tracking") == 8
    assert reloaded.effective_level("rapid_tracking") == 9
    assert reloaded.effective_ratio("rapid_tracking") == pytest.approx((9 - 1) / 9.0)
    assert reloaded.intro_mode_label("rapid_tracking") == "Global Override"

    reloaded.set_global_override_enabled(False)
    final = DifficultySettingsStore(path)
    assert final.global_override_enabled() is False
    assert final.effective_level("rapid_tracking") == 8
    assert final.intro_mode_label("rapid_tracking") == "This Test"


def test_difficulty_settings_screen_updates_global_and_per_test_values(tmp_path) -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        store = DifficultySettingsStore(tmp_path / "difficulty-settings.json")
        app = App(surface=surface, font=font, difficulty_settings_store=store)
        root = MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True)
        app.push(root)
        screen = DifficultySettingsScreen(app)
        app.push(screen)

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RIGHT, "mod": 0, "unicode": ""})
        )
        assert store.global_override_enabled() is True

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "mod": 0, "unicode": ""})
        )
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RIGHT, "mod": 0, "unicode": ""})
        )
        assert store.global_level() == 6

        rows = screen._rows()
        rapid_tracking_index = next(
            idx for idx, (key, _label, _value) in enumerate(rows) if key == "test:rapid_tracking"
        )
        screen._selected = rapid_tracking_index
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RIGHT, "mod": 0, "unicode": ""})
        )

        assert store.test_level("rapid_tracking") == 6
    finally:
        pygame.quit()
