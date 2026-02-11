from __future__ import annotations

import os


def test_ui_smoke_navigate_to_tests_and_open_numerical_ops() -> None:
    # Headless SDL for CI.
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    import pygame

    from cfast_trainer.app import run

    def inject(frame: int) -> None:
        # Navigate: Main Menu -> Tests -> Numerical Operations -> begin practice
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

    assert run(max_frames=20, event_injector=inject) == 0

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

    assert run(max_frames=25, event_injector=inject) == 0
def test_ui_smoke_navigate_to_tests_and_open_math_reasoning() -> None:
    # Headless SDL for CI.
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    import pygame

    from cfast_trainer.app import run

    def inject(frame: int) -> None:
        # Main Menu: DOWN, DOWN -> Tests, ENTER
        # Tests menu: DOWN -> Mathematics Reasoning, ENTER
        # Test instructions: ENTER to begin practice (ensures screen is live)
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

    assert run(max_frames=25, event_injector=inject) == 0