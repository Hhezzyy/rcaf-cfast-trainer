from __future__ import annotations

import os


def test_ui_smoke_navigate_to_tests_and_open_numerical_ops() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    import pygame
    from cfast_trainer.app import run

    def inject(frame: int) -> None:
        # Main Menu: DOWN, DOWN -> Tests, ENTER
        # Tests menu: ENTER on Numerical Operations
        # Instructions: ENTER to begin practice
        if frame == 1:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        elif frame == 2:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        elif frame == 3:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))
        elif frame == 4:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))
        elif frame == 5:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))

    assert run(max_frames=30, event_injector=inject) == 0


def test_ui_smoke_navigate_to_tests_and_open_math_reasoning() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    import pygame
    from cfast_trainer.app import run

    def inject(frame: int) -> None:
        # Main Menu: DOWN, DOWN -> Tests, ENTER
        # Tests menu: DOWN -> Math Reasoning, ENTER
        # Instructions: ENTER to begin practice
        if frame == 1:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        elif frame == 2:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        elif frame == 3:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))
        elif frame == 4:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        elif frame == 5:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))
        elif frame == 6:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))

    assert run(max_frames=30, event_injector=inject) == 0


def test_ui_smoke_navigate_to_tests_and_open_airborne_numerical() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    import pygame
    from cfast_trainer.app import run

    def inject(frame: int) -> None:
        # Assumes Tests submenu order:
        # 1) Numerical Operations
        # 2) Mathematics Reasoning
        # 3) Airborne Numerical Test
        #
        # Main Menu: DOWN, DOWN -> Tests, ENTER
        # Tests menu: DOWN, DOWN -> Airborne Numerical Test, ENTER
        # Instructions: ENTER to begin practice
        if frame == 1:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        elif frame == 2:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        elif frame == 3:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))
        elif frame == 4:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        elif frame == 5:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        elif frame == 6:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))
        elif frame == 7:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))

    assert run(max_frames=35, event_injector=inject) == 0
