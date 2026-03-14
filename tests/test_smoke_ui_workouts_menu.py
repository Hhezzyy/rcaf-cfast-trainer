from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def enable_dev_tools(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CFAST_ENABLE_DEV_TOOLS", "1")
    yield


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


def test_ui_smoke_navigate_to_workouts_and_open_abd_workout() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    import pygame

    from cfast_trainer.app import run

    def inject(frame: int) -> None:
        if frame == 1:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )
        elif frame == 2:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""})
            )
        elif frame == 3:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )
        elif frame in (7, 8, 9, 10):
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_F10, "unicode": ""})
            )

    assert run(max_frames=60, event_injector=inject) == 0


def test_ui_smoke_navigate_to_workouts_and_open_no_workout() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    import pygame

    from cfast_trainer.app import run

    def inject(frame: int) -> None:
        if frame == 1:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )
        elif frame in (2, 3):
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""})
            )
        elif frame == 4:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )
        elif frame in (8, 9, 10, 11):
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_F10, "unicode": ""})
            )

    assert run(max_frames=60, event_injector=inject) == 0


def test_ui_smoke_navigate_to_workouts_and_open_mr_workout() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    import pygame

    from cfast_trainer.app import run

    def inject(frame: int) -> None:
        if frame == 1:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )
        elif frame in (2, 3, 4):
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""})
            )
        elif frame == 5:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )
        elif frame in (9, 10, 11, 12):
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_F10, "unicode": ""})
            )

    assert run(max_frames=64, event_injector=inject) == 0


def test_ui_smoke_navigate_to_workouts_and_open_dr_workout() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    import pygame

    from cfast_trainer.app import run

    def inject(frame: int) -> None:
        if frame == 1:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )
        elif frame in (2, 3, 4, 5):
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""})
            )
        elif frame == 6:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )
        elif frame in (10, 11, 12, 13):
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_F10, "unicode": ""})
            )

    assert run(max_frames=70, event_injector=inject) == 0


def test_ui_smoke_navigate_to_workouts_and_open_cln_workout() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    import pygame

    from cfast_trainer.app import run

    def inject(frame: int) -> None:
        if frame == 1:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )
        elif frame in (2, 3, 4, 5, 6):
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""})
            )
        elif frame == 7:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )
        elif frame in (11, 12, 13, 14):
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_F10, "unicode": ""})
            )

    assert run(max_frames=74, event_injector=inject) == 0


def test_ui_smoke_navigate_to_workouts_and_open_vs_workout() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    import pygame

    from cfast_trainer.app import run

    def inject(frame: int) -> None:
        if frame == 1:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )
        elif frame in (2, 3, 4, 5, 6, 7):
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""})
            )
        elif frame == 8:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )
        elif frame in (12, 13, 14, 15):
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_F10, "unicode": ""})
            )

    assert run(max_frames=78, event_injector=inject) == 0


def test_ui_smoke_navigate_to_workouts_and_open_table_reading_workout() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    import pygame

    from cfast_trainer.app import run

    def inject(frame: int) -> None:
        if frame == 1:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )
        elif frame in (2, 3, 4, 5, 6, 7, 8, 9, 10, 11):
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""})
            )
        elif frame == 12:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )
        elif frame in (16, 17, 18, 19):
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_F10, "unicode": ""})
            )

    assert run(max_frames=94, event_injector=inject) == 0


def test_ui_smoke_navigate_to_workouts_and_open_ic_workout() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    import pygame

    from cfast_trainer.app import run

    def inject(frame: int) -> None:
        if frame == 1:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )
        elif frame in (2, 3, 4, 5, 6, 7, 8):
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""})
            )
        elif frame == 9:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )
        elif frame in (13, 14, 15, 16):
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_F10, "unicode": ""})
            )

    assert run(max_frames=82, event_injector=inject) == 0


def test_ui_smoke_navigate_to_workouts_and_open_tr_workout() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    import pygame

    from cfast_trainer.app import run

    def inject(frame: int) -> None:
        if frame == 1:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )
        elif frame in (2, 3, 4, 5, 6, 7, 8, 9):
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""})
            )
        elif frame == 10:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )
        elif frame in (14, 15, 16, 17):
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_F10, "unicode": ""})
            )

    assert run(max_frames=86, event_injector=inject) == 0


def test_ui_smoke_navigate_to_workouts_and_open_system_logic_workout() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    import pygame

    from cfast_trainer.app import run

    def inject(frame: int) -> None:
        if frame == 1:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )
        elif frame in (2, 3, 4, 5, 6, 7, 8, 9, 10):
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""})
            )
        elif frame == 11:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )
        elif frame in (15, 16, 17, 18):
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_F10, "unicode": ""})
            )

    assert run(max_frames=90, event_injector=inject) == 0


def test_ui_smoke_navigate_to_workouts_and_open_sma_workout() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    import pygame

    from cfast_trainer.app import run

    def inject(frame: int) -> None:
        if frame == 1:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )
        elif frame in (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12):
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""})
            )
        elif frame == 13:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )
        elif frame in (17, 18, 19, 20):
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_F10, "unicode": ""})
            )

    assert run(max_frames=96, event_injector=inject) == 0


def test_ui_smoke_navigate_to_workouts_and_open_auditory_capacity_workout() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    import pygame

    from cfast_trainer.app import run

    def inject(frame: int) -> None:
        if frame == 1:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )
        elif frame in (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13):
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""})
            )
        elif frame == 14:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )
        elif frame in (18, 19, 20, 21):
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_F10, "unicode": ""})
            )

    assert run(max_frames=102, event_injector=inject) == 0


def test_ui_smoke_navigate_to_workouts_and_open_cognitive_updating_workout() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    import pygame

    from cfast_trainer.app import run

    def inject(frame: int) -> None:
        if frame == 1:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )
        elif frame in (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14):
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""})
            )
        elif frame == 15:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )
        elif frame in (19, 20, 21, 22):
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_F10, "unicode": ""})
            )

    assert run(max_frames=106, event_injector=inject) == 0


def test_ui_smoke_navigate_to_workouts_and_open_situational_awareness_workout() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    import pygame

    from cfast_trainer.app import run

    def inject(frame: int) -> None:
        if frame == 1:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )
        elif frame in (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15):
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""})
            )
        elif frame == 16:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )
        elif frame in (20, 21, 22, 23):
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_F10, "unicode": ""})
            )

    assert run(max_frames=110, event_injector=inject) == 0


def test_ui_smoke_navigate_to_workouts_and_open_rapid_tracking_workout() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    import pygame

    from cfast_trainer.app import run

    def inject(frame: int) -> None:
        if frame == 1:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )
        elif frame in (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16):
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""})
            )
        elif frame == 17:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )
        elif frame in (21, 22, 23, 24):
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_F10, "unicode": ""})
            )

    assert run(max_frames=114, event_injector=inject) == 0
