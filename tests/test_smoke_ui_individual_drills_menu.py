from __future__ import annotations

import os


def test_ui_smoke_navigate_to_individual_drills_and_open_ant_snap_facts_build() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    import pygame

    from cfast_trainer.app import run

    def inject(frame: int) -> None:
        # Main Menu: DOWN -> Individual drills, ENTER
        # Drills menu: ENTER on ANT Drills
        # ANT drills: ENTER on Snap Facts Sprint - Build
        # Instructions: ENTER to begin practice
        if frame == 1:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""})
            )
        elif frame == 2:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )
        elif frame == 3:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )
        elif frame == 4:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )
        elif frame == 5:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )

    assert run(max_frames=35, event_injector=inject) == 0


def test_ui_smoke_navigate_to_individual_drills_and_open_ant_time_flip_build() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    import pygame

    from cfast_trainer.app import run

    def inject(frame: int) -> None:
        # Main Menu: DOWN -> Individual drills, ENTER
        # Drills menu: ENTER on ANT Drills
        # ANT drills: DOWN x3 -> Time Flip - Build, ENTER
        # Instructions: ENTER to begin practice
        if frame == 1:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""})
            )
        elif frame == 2:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )
        elif frame == 3:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )
        elif frame in (4, 5, 6):
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""})
            )
        elif frame == 7:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )
        elif frame == 8:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )

    assert run(max_frames=45, event_injector=inject) == 0


def test_ui_smoke_navigate_to_individual_drills_and_open_ant_mixed_tempo_set_build() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    import pygame

    from cfast_trainer.app import run

    def inject(frame: int) -> None:
        # Main Menu: DOWN -> Drill, ENTER
        # Drill menu: ENTER on ANT Drills
        # ANT drills: DOWN x6 -> Mixed Tempo Set - Build, ENTER
        # Instructions: ENTER to begin practice
        if frame == 1:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""})
            )
        elif frame == 2:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )
        elif frame == 3:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )
        elif frame in (4, 5, 6, 7, 8, 9):
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""})
            )
        elif frame == 10:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )
        elif frame == 11:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )

    assert run(max_frames=55, event_injector=inject) == 0


def test_ui_smoke_navigate_to_individual_drills_and_open_ant_route_time_solve_build() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    import pygame

    from cfast_trainer.app import run

    def inject(frame: int) -> None:
        # Main Menu: DOWN -> Drill, ENTER
        # Drill menu: ENTER on ANT Drills
        # ANT drills: DOWN x9 -> Route Time Solve - Build, ENTER
        # Instructions: ENTER to begin practice
        if frame == 1:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""})
            )
        elif frame == 2:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )
        elif frame == 3:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )
        elif frame in (4, 5, 6, 7, 8, 9, 10, 11, 12):
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""})
            )
        elif frame == 13:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )
        elif frame == 14:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )

    assert run(max_frames=60, event_injector=inject) == 0


def test_ui_smoke_navigate_to_individual_drills_and_open_ant_fuel_burn_solve_build() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    import pygame

    from cfast_trainer.app import run

    def inject(frame: int) -> None:
        # Main Menu: DOWN -> Drill, ENTER
        # Drill menu: ENTER on ANT Drills
        # ANT drills: DOWN x12 -> Fuel Burn Solve - Build, ENTER
        # Instructions: ENTER to begin practice
        if frame == 1:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""})
            )
        elif frame == 2:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )
        elif frame == 3:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )
        elif frame in (4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15):
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""})
            )
        elif frame == 16:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )
        elif frame == 17:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )

    assert run(max_frames=68, event_injector=inject) == 0


def test_ui_smoke_navigate_to_individual_drills_and_open_ant_endurance_solve_build() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    import pygame

    from cfast_trainer.app import run

    def inject(frame: int) -> None:
        # Main Menu: DOWN -> Drill, ENTER
        # Drill menu: ENTER on ANT Drills
        # ANT drills: DOWN x15 -> Endurance Solve - Build, ENTER
        # Instructions: ENTER to begin practice
        if frame == 1:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""})
            )
        elif frame == 2:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )
        elif frame == 3:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )
        elif frame in (4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18):
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""})
            )
        elif frame == 19:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )
        elif frame == 20:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )

    assert run(max_frames=72, event_injector=inject) == 0
