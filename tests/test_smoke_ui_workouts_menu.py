from __future__ import annotations

import os


def test_ui_smoke_navigate_to_workouts_and_open_airborne_numerical_workout() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    import pygame

    from cfast_trainer.app import run

    def inject(frame: int) -> None:
        # Main menu: Enter on 90-minute workouts.
        # Workouts menu: Enter on Airborne Numerical Workout.
        # Loading screen needs two renders before the target is swapped in.
        # Workout intro: Right once, Enter, type two short reflections, then start the first block.
        if frame == 1:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )
        elif frame == 2:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )
        elif frame == 6:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RIGHT, "unicode": ""})
            )
        elif frame == 7:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )
        elif frame in (8, 9, 10):
            ch = "why"[frame - 8]
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": ord(ch), "unicode": ch})
            )
        elif frame == 11:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )
        elif frame in (12, 13, 14, 15):
            ch = "rule"[frame - 12]
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": ord(ch), "unicode": ch})
            )
        elif frame == 16:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )
        elif frame == 17:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )

    assert run(max_frames=60, event_injector=inject) == 0
