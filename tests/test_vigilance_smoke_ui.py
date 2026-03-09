from __future__ import annotations

import os


def test_ui_smoke_navigate_to_tests_and_open_vigilance() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    import pygame

    from cfast_trainer.app import run

    def inject(frame: int) -> None:
        # Main Menu: DOWN, DOWN -> Tests, ENTER
        # Tests menu: DOWN x19 -> Vigilance, ENTER
        # Instructions: ENTER to begin practice
        if frame in (1, 2):
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""})
            )
            return
        if frame == 3:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )
            return
        if 4 <= frame <= 22:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""})
            )
            return
        if frame in (23, 24):
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )

    assert run(max_frames=80, event_injector=inject) == 0
