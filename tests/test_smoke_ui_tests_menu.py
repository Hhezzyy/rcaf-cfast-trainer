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

def test_ui_smoke_navigate_to_tests_and_open_digit_recognition() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    import pygame
    from cfast_trainer.app import run

    def inject(frame: int) -> None:
        # Assumes Tests submenu order:
        # 1) Numerical Operations
        # 2) Mathematics Reasoning
        # 3) Airborne Numerical Test
        # 4) Digit Recognition
        # 5) Colours, Letters and Numbers
        # 6) Angles, Bearings and Degrees
        #
        # Main Menu: DOWN, DOWN -> Tests, ENTER
        # Tests menu: DOWN, DOWN, DOWN -> Digit Recognition, ENTER
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
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        elif frame == 7:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))
        elif frame == 8:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))

    assert run(max_frames=40, event_injector=inject) == 0


def test_ui_smoke_navigate_to_tests_and_open_colours_letters_numbers() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    import pygame
    from cfast_trainer.app import run

    def inject(frame: int) -> None:
        # Assumes Tests submenu order:
        # 1) Numerical Operations
        # 2) Mathematics Reasoning
        # 3) Airborne Numerical Test
        # 4) Digit Recognition
        # 5) Colours, Letters and Numbers
        #
        # Main Menu: DOWN, DOWN -> Tests, ENTER
        # Tests menu: DOWN x4 -> Colours, Letters and Numbers, ENTER
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
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        elif frame == 7:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        elif frame == 8:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))
        elif frame == 9:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))

    assert run(max_frames=45, event_injector=inject) == 0


def test_ui_smoke_navigate_to_tests_and_open_angles_bearings_degrees() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    import pygame
    from cfast_trainer.app import run

    def inject(frame: int) -> None:
        # Assumes Tests submenu order:
        # 1) Numerical Operations
        # 2) Mathematics Reasoning
        # 3) Airborne Numerical Test
        # 4) Digit Recognition
        # 5) Colours, Letters and Numbers
        # 6) Angles, Bearings and Degrees
        #
        # Main Menu: DOWN, DOWN -> Tests, ENTER
        # Tests menu: DOWN x5 -> Angles, Bearings and Degrees, ENTER
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
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        elif frame == 7:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        elif frame == 8:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        elif frame == 9:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))
        elif frame == 10:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))

    assert run(max_frames=50, event_injector=inject) == 0


def test_ui_smoke_navigate_to_tests_and_open_visual_search() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    import pygame
    from cfast_trainer.app import run

    def inject(frame: int) -> None:
        # Assumes Tests submenu order:
        # 1) Numerical Operations
        # 2) Mathematics Reasoning
        # 3) Airborne Numerical Test
        # 4) Digit Recognition
        # 5) Colours, Letters and Numbers
        # 6) Angles, Bearings and Degrees
        # 7) Visual Search
        #
        # Main Menu: DOWN, DOWN -> Tests, ENTER
        # Tests menu: DOWN x6 -> Visual Search, ENTER
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
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        elif frame == 7:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        elif frame == 8:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        elif frame == 9:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        elif frame == 10:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))
        elif frame == 11:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))

    assert run(max_frames=55, event_injector=inject) == 0


def test_ui_smoke_navigate_to_tests_and_open_instrument_comprehension() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    import pygame
    from cfast_trainer.app import run

    def inject(frame: int) -> None:
        # Assumes Tests submenu order:
        # 1) Numerical Operations
        # 2) Mathematics Reasoning
        # 3) Airborne Numerical Test
        # 4) Digit Recognition
        # 5) Colours, Letters and Numbers
        # 6) Angles, Bearings and Degrees
        # 7) Visual Search
        # 8) Instrument Comprehension
        #
        # Main Menu: DOWN, DOWN -> Tests, ENTER
        # Tests menu: DOWN x7 -> Instrument Comprehension, ENTER
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
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        elif frame == 7:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        elif frame == 8:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        elif frame == 9:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        elif frame == 10:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        elif frame == 11:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))
        elif frame == 12:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))

    assert run(max_frames=60, event_injector=inject) == 0


def test_ui_smoke_navigate_to_tests_and_open_target_recognition() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    import pygame
    from cfast_trainer.app import run

    def inject(frame: int) -> None:
        # Assumes Tests submenu order:
        # 1) Numerical Operations
        # 2) Mathematics Reasoning
        # 3) Airborne Numerical Test
        # 4) Digit Recognition
        # 5) Colours, Letters and Numbers
        # 6) Angles, Bearings and Degrees
        # 7) Visual Search
        # 8) Instrument Comprehension
        # 9) Target Recognition
        #
        # Main Menu: DOWN, DOWN -> Tests, ENTER
        # Tests menu: DOWN x8 -> Target Recognition, ENTER
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
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        elif frame == 7:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        elif frame == 8:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        elif frame == 9:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        elif frame == 10:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        elif frame == 11:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        elif frame == 12:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))
        elif frame == 13:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))

    assert run(max_frames=65, event_injector=inject) == 0


def test_ui_smoke_navigate_to_tests_and_open_system_logic() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    import pygame
    from cfast_trainer.app import run

    def inject(frame: int) -> None:
        # Assumes Tests submenu order:
        # 1) Numerical Operations
        # 2) Mathematics Reasoning
        # 3) Airborne Numerical Test
        # 4) Digit Recognition
        # 5) Colours, Letters and Numbers
        # 6) Angles, Bearings and Degrees
        # 7) Visual Search
        # 8) Instrument Comprehension
        # 9) Target Recognition
        # 10) System Logic
        #
        # Main Menu: DOWN, DOWN -> Tests, ENTER
        # Tests menu: DOWN x9 -> System Logic, ENTER
        # Instructions: ENTER to begin practice
        if frame == 1:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        elif frame == 2:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        elif frame == 3:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))
        elif frame in (4, 5, 6, 7, 8, 9, 10, 11, 12):
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        elif frame == 13:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))
        elif frame == 14:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))

    assert run(max_frames=70, event_injector=inject) == 0


def test_ui_smoke_navigate_to_tests_and_open_table_reading() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    import pygame
    from cfast_trainer.app import run

    def inject(frame: int) -> None:
        # Assumes Tests submenu order:
        # 1) Numerical Operations
        # 2) Mathematics Reasoning
        # 3) Airborne Numerical Test
        # 4) Digit Recognition
        # 5) Colours, Letters and Numbers
        # 6) Angles, Bearings and Degrees
        # 7) Visual Search
        # 8) Instrument Comprehension
        # 9) Target Recognition
        # 10) System Logic
        # 11) Table Reading
        #
        # Main Menu: DOWN, DOWN -> Tests, ENTER
        # Tests menu: DOWN x10 -> Table Reading, ENTER
        # Instructions: ENTER to begin practice
        if frame == 1:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        elif frame == 2:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        elif frame == 3:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))
        elif frame in (4, 5, 6, 7, 8, 9, 10, 11, 12, 13):
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        elif frame == 14:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))
        elif frame == 15:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))

    assert run(max_frames=75, event_injector=inject) == 0


def test_ui_smoke_navigate_to_tests_and_open_sensory_motor_apparatus() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    import pygame
    from cfast_trainer.app import run

    def inject(frame: int) -> None:
        # Assumes Tests submenu order:
        # 1) Numerical Operations
        # 2) Mathematics Reasoning
        # 3) Airborne Numerical Test
        # 4) Digit Recognition
        # 5) Colours, Letters and Numbers
        # 6) Angles, Bearings and Degrees
        # 7) Visual Search
        # 8) Instrument Comprehension
        # 9) Target Recognition
        # 10) System Logic
        # 11) Table Reading
        # 12) Sensory Motor Apparatus
        #
        # Main Menu: DOWN, DOWN -> Tests, ENTER
        # Tests menu: DOWN x11 -> Sensory Motor Apparatus, ENTER
        # Instructions: ENTER to begin practice
        if frame == 1:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        elif frame == 2:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        elif frame == 3:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))
        elif frame in (4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14):
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        elif frame == 15:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))
        elif frame == 16:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))

    assert run(max_frames=80, event_injector=inject) == 0


def test_ui_smoke_navigate_to_tests_and_open_auditory_capacity() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    import pygame
    from cfast_trainer.app import run

    def inject(frame: int) -> None:
        # Assumes Tests submenu order:
        # 1) Numerical Operations
        # 2) Mathematics Reasoning
        # 3) Airborne Numerical Test
        # 4) Digit Recognition
        # 5) Colours, Letters and Numbers
        # 6) Angles, Bearings and Degrees
        # 7) Visual Search
        # 8) Instrument Comprehension
        # 9) Target Recognition
        # 10) System Logic
        # 11) Table Reading
        # 12) Sensory Motor Apparatus
        # 13) Auditory Capacity
        #
        # Main Menu: DOWN, DOWN -> Tests, ENTER
        # Tests menu: DOWN x12 -> Auditory Capacity, ENTER
        # Instructions: ENTER to begin practice
        if frame == 1:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        elif frame == 2:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        elif frame == 3:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))
        elif frame in (4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15):
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        elif frame == 16:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))
        elif frame == 17:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))

    assert run(max_frames=84, event_injector=inject) == 0


def test_ui_smoke_target_recognition_mouse_select_and_submit() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    import pygame
    from cfast_trainer.app import run

    def inject(frame: int) -> None:
        # Main Menu: DOWN, DOWN -> Tests, ENTER
        # Tests menu: DOWN x8 -> Target Recognition, ENTER
        # Instructions: ENTER to begin practice
        # Practice: click one target strip, then click SUBMIT
        if frame == 1:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        elif frame == 2:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        elif frame == 3:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))
        elif frame in (4, 5, 6, 7, 8, 9, 10, 11):
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        elif frame == 12:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))
        elif frame == 13:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))
        elif frame == 14:
            pygame.event.post(
                pygame.event.Event(
                    pygame.MOUSEBUTTONDOWN,
                    {"button": 1, "pos": (110, 430)},
                )
            )
        elif frame == 15:
            pygame.event.post(
                pygame.event.Event(
                    pygame.MOUSEBUTTONDOWN,
                    {"button": 1, "pos": (870, 490)},
                )
            )

    assert run(max_frames=75, event_injector=inject) == 0


def test_ui_smoke_navigate_to_hotas_and_open_axis_calibration() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    import pygame
    from cfast_trainer.app import run

    def inject(frame: int) -> None:
        # Main Menu: DOWN x3 -> HOTAS, ENTER
        # HOTAS menu: ENTER on Axis Calibration
        if frame == 1:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        elif frame == 2:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        elif frame == 3:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        elif frame == 4:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))
        elif frame == 5:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))

    assert run(max_frames=30, event_injector=inject) == 0


def test_ui_smoke_navigate_to_hotas_and_open_axis_visualizer() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    import pygame
    from cfast_trainer.app import run

    def inject(frame: int) -> None:
        # Main Menu: DOWN x3 -> HOTAS, ENTER
        # HOTAS menu: DOWN -> Axis Visualizer, ENTER
        if frame == 1:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        elif frame == 2:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        elif frame == 3:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        elif frame == 4:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))
        elif frame == 5:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        elif frame == 6:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))

    assert run(max_frames=35, event_injector=inject) == 0


def test_ui_smoke_navigate_to_hotas_and_open_input_profiles() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ["CFAST_INPUT_PROFILES_PATH"] = "/tmp/cfast_input_profiles_smoke.json"

    import pygame
    from cfast_trainer.app import run

    def inject(frame: int) -> None:
        # Main Menu: DOWN x3 -> HOTAS, ENTER
        # HOTAS menu: DOWN x2 -> Input Profiles, ENTER
        # In profile screen: create a profile (N), then set active (Enter)
        if frame == 1:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        elif frame == 2:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        elif frame == 3:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        elif frame == 4:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))
        elif frame == 5:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        elif frame == 6:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        elif frame == 7:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))
        elif frame == 8:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_n, "unicode": "n"}))
        elif frame == 9:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))

    assert run(max_frames=45, event_injector=inject) == 0
