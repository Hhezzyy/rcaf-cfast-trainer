"""Pygame UI shell for the RCAF CFAST Trainer.

This task adds two cognitive tests under a "Tests" submenu:
- Numerical Operations (mental arithmetic)
- Mathematics Reasoning (time/speed/distance word problems)

Deterministic timing/scoring/RNG/state lives in cfast_trainer/* (core modules).
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import pygame

from .airborne_numerical import build_airborne_numerical_test
from .clock import RealClock
from .cognitive_core import Phase, TestSnapshot
from .math_reasoning import build_math_reasoning_test
from .numerical_operations import build_numerical_operations_test


class Screen(Protocol):
    def handle_event(self, event: pygame.event.Event) -> None:
        ...

    def render(self, surface: pygame.Surface) -> None:
        ...

class CognitiveEngine(Protocol):
    def snapshot(self) -> TestSnapshot: ...
    def can_exit(self) -> bool: ...
    def start_practice(self) -> None: ...
    def start_scored(self) -> None: ...
    def submit_answer(self, raw: str) -> bool: ...
    def update(self) -> None: ...

@dataclass(frozen=True, slots=True)
class MenuItem:
    label: str
    action: Callable[[], None]


@dataclass
class Settings:
    # Display
    fullscreen: bool = False
    max_fps: int = 60
    show_fps_overlay: bool = False

    # Input (used later by psychomotor tasks / mappings)
    invert_y_axis: bool = False


# Global settings instance (persistence later).
settings = Settings()


class App:
    def __init__(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        self._surface = surface
        self._font = font
        self._screens: list[Screen] = []
        self._running = True

    @property
    def running(self) -> bool:
        return self._running

    @property
    def font(self) -> pygame.font.Font:
        return self._font

    def push(self, screen: Screen) -> None:
        self._screens.append(screen)

    def pop(self) -> None:
        # Never pop the last/root screen; root handles its own quit/back behavior.
        if len(self._screens) > 1:
            self._screens.pop()

    def quit(self) -> None:
        self._running = False

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.QUIT:
            self.quit()
            return
        if not self._screens:
            return
        self._screens[-1].handle_event(event)

    def render(self) -> None:
        if not self._screens:
            return
        self._screens[-1].render(self._surface)

    def update_surface(self, new_surface: pygame.Surface) -> None:
        # Used when toggling fullscreen (display mode changes).
        self._surface = new_surface


class PlaceholderScreen:
    def __init__(self, app: App, title: str) -> None:
        self._app = app
        self._title = title

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type != pygame.KEYDOWN:
            return
        if event.key in (pygame.K_ESCAPE, pygame.K_BACKSPACE):
            self._app.pop()

    def render(self, surface: pygame.Surface) -> None:
        surface.fill((10, 10, 14))
        title = self._app.font.render(self._title, True, (235, 235, 245))
        hint = self._app.font.render("Placeholder. Press Esc to go back.", True, (180, 180, 190))
        surface.blit(title, (40, 40))
        surface.blit(hint, (40, 100))


class MenuScreen:
    def __init__(
        self, app: App, title: str, items: list[MenuItem], *, is_root: bool = False
    ) -> None:
        self._app = app
        self._title = title
        self._items = items
        self._selected = 0
        self._is_root = is_root

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            self._handle_key(event.key)
            return

        if event.type == pygame.JOYHATMOTION:
            # D-pad / hat navigation (works on many sticks).
            _, y = event.value
            if y == 1:
                self._move(-1)
            elif y == -1:
                self._move(1)
            return

        if event.type == pygame.JOYBUTTONDOWN:
            # Common mapping: 0 = select, 1 = back/cancel.
            if event.button == 0:
                self._activate()
            elif event.button == 1:
                self._back()

    def _handle_key(self, key: int) -> None:
        if key in (pygame.K_UP, pygame.K_w):
            self._move(-1)
        elif key in (pygame.K_DOWN, pygame.K_s):
            self._move(1)
        elif key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
            self._activate()
        elif key in (pygame.K_ESCAPE, pygame.K_BACKSPACE):
            self._back()

    def _move(self, delta: int) -> None:
        if not self._items:
            return
        self._selected = (self._selected + delta) % len(self._items)

    def _activate(self) -> None:
        if not self._items:
            return
        self._items[self._selected].action()

    def _back(self) -> None:
        if self._is_root:
            self._app.quit()
        else:
            self._app.pop()

    def render(self, surface: pygame.Surface) -> None:
        surface.fill((10, 10, 14))
        title = self._app.font.render(self._title, True, (235, 235, 245))
        surface.blit(title, (40, 40))

        y = 120
        for idx, item in enumerate(self._items):
            prefix = "> " if idx == self._selected else "  "
            color = (235, 235, 245) if idx == self._selected else (180, 180, 190)
            text = self._app.font.render(f"{prefix}{item.label}", True, color)
            surface.blit(text, (60, y))
            y += 42

        footer = "Enter/Space to select • Esc to back/quit • D-pad + Button0/1 supported"
        foot = pygame.font.Font(None, 22).render(footer, True, (140, 140, 150))
        surface.blit(foot, (40, surface.get_height() - 40))


class SettingsScreen:
    def __init__(
        self,
        app: App,
        *,
        axis_calibration: Screen,
        axis_visualizer: Screen,
        input_profiles: Screen,
        data_storage: Screen,
    ) -> None:
        self._app = app
        self._selected = 0
        self._fps_options = [30, 60, 120]
        self._axis_calibration = axis_calibration
        self._axis_visualizer = axis_visualizer
        self._input_profiles = input_profiles
        self._data_storage = data_storage

    def _toggle_fullscreen(self) -> None:
        settings.fullscreen = not settings.fullscreen
        flags = pygame.FULLSCREEN if settings.fullscreen else 0
        new_surface = pygame.display.set_mode((960, 540), flags)
        self._app.update_surface(new_surface)

    def _cycle_max_fps(self) -> None:
        try:
            idx = self._fps_options.index(settings.max_fps)
        except ValueError:
            idx = 0
        settings.max_fps = self._fps_options[(idx + 1) % len(self._fps_options)]

    def _toggle_show_fps(self) -> None:
        settings.show_fps_overlay = not settings.show_fps_overlay

    def _toggle_invert_y(self) -> None:
        settings.invert_y_axis = not settings.invert_y_axis

    def _entries(self) -> list[tuple[str, Callable[[], None]]]:
        return [
            (f"Fullscreen: {'ON' if settings.fullscreen else 'OFF'}", self._toggle_fullscreen),
            (f"Max FPS: {settings.max_fps}", self._cycle_max_fps),
            (
                f"Show FPS overlay: {'ON' if settings.show_fps_overlay else 'OFF'}",
                self._toggle_show_fps,
            ),
            (f"Invert Y axis: {'ON' if settings.invert_y_axis else 'OFF'}", self._toggle_invert_y),
            ("Axis Calibration", lambda: self._app.push(self._axis_calibration)),
            ("Axis Visualizer", lambda: self._app.push(self._axis_visualizer)),
            ("Input Profiles", lambda: self._app.push(self._input_profiles)),
            ("Data & Storage", lambda: self._app.push(self._data_storage)),
            ("Back", self._app.pop),
        ]

    def _move(self, delta: int) -> None:
        items = self._entries()
        if not items:
            return
        self._selected = (self._selected + delta) % len(items)

    def _activate(self) -> None:
        items = self._entries()
        if not items:
            return
        _, action = items[self._selected]
        action()

    def _back(self) -> None:
        self._app.pop()

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_UP, pygame.K_w):
                self._move(-1)
            elif event.key in (pygame.K_DOWN, pygame.K_s):
                self._move(1)
            elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                self._activate()
            elif event.key in (pygame.K_ESCAPE, pygame.K_BACKSPACE):
                self._back()
            return

        if event.type == pygame.JOYHATMOTION:
            _, y = event.value
            if y == 1:
                self._move(-1)
            elif y == -1:
                self._move(1)
            return

        if event.type == pygame.JOYBUTTONDOWN:
            if event.button == 0:
                self._activate()
            elif event.button == 1:
                self._back()

    def render(self, surface: pygame.Surface) -> None:
        surface.fill((10, 10, 14))
        title = self._app.font.render("Settings", True, (235, 235, 245))
        surface.blit(title, (40, 40))

        y = 120
        items = self._entries()
        for idx, (label, _) in enumerate(items):
            prefix = "> " if idx == self._selected else "  "
            color = (235, 235, 245) if idx == self._selected else (180, 180, 190)
            text = self._app.font.render(f"{prefix}{label}", True, color)
            surface.blit(text, (60, y))
            y += 42

        footer = "Enter/Space to select • Esc to back • D-pad + Button0/1 supported"
        foot = pygame.font.Font(None, 22).render(footer, True, (140, 140, 150))
        surface.blit(foot, (40, surface.get_height() - 40))


class CognitiveTestScreen:
    def __init__(self, app: App, *, engine_factory: Callable[[], CognitiveEngine]) -> None:
        self._app = app
        self._engine: CognitiveEngine = engine_factory()
        self._input = ""

        self._small_font = pygame.font.Font(None, 24)

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type != pygame.KEYDOWN:
            return

        key = event.key
        snap = self._engine.snapshot()

        if key in (pygame.K_ESCAPE, pygame.K_BACKSPACE) and self._engine.can_exit():
            self._app.pop()
            return

        if key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            if snap.phase is Phase.INSTRUCTIONS:
                self._engine.start_practice()
                self._input = ""
                return
            if snap.phase is Phase.PRACTICE_DONE:
                self._engine.start_scored()
                self._input = ""
                return
            if snap.phase is Phase.RESULTS:
                self._app.pop()
                return

            accepted = self._engine.submit_answer(self._input)
            if accepted:
                self._input = ""
            return

        if snap.phase not in (Phase.PRACTICE, Phase.SCORED):
            return

        if key == pygame.K_BACKSPACE:
            self._input = self._input[:-1]
            return

        ch = event.unicode
        if ch and (ch.isdigit() or (ch == "-" and self._input == "")):
            self._input += ch

    def render(self, surface: pygame.Surface) -> None:
        self._engine.update()
        snap = self._engine.snapshot()

        surface.fill((10, 10, 14))

        title = self._app.font.render(snap.title, True, (235, 235, 245))
        surface.blit(title, (40, 30))

        y_info = 80
        if snap.time_remaining_s is not None:
            rem = int(round(snap.time_remaining_s))
            mm = rem // 60
            ss = rem % 60
            timer = self._small_font.render(
                f"Time remaining: {mm:02d}:{ss:02d}", True, (200, 200, 210)
            )
            surface.blit(timer, (40, y_info))
            y_info += 28

        stats = self._small_font.render(
            f"Scored: {snap.correct_scored}/{snap.attempted_scored}", True, (180, 180, 190)
        )
        surface.blit(stats, (40, y_info))

        prompt_lines = str(snap.prompt).split("\n")
        y = 140
        for line in prompt_lines[:10]:
            txt = self._small_font.render(line, True, (235, 235, 245))
            surface.blit(txt, (40, y))
            y += 26

        if snap.phase in (Phase.PRACTICE, Phase.SCORED):
            box = pygame.Rect(40, surface.get_height() - 120, 400, 44)
            pygame.draw.rect(surface, (30, 30, 40), box)
            pygame.draw.rect(surface, (90, 90, 110), box, 2)

            caret = "|" if (pygame.time.get_ticks() // 500) % 2 == 0 else ""
            entry = self._app.font.render(self._input + caret, True, (235, 235, 245))
            surface.blit(entry, (box.x + 10, box.y + 8))

            hint = self._small_font.render(snap.input_hint, True, (140, 140, 150))
            surface.blit(hint, (40, surface.get_height() - 60))

        if not self._engine.can_exit() and snap.phase is Phase.SCORED:
            lock = self._small_font.render("Test in progress: cannot exit.", True, (140, 140, 150))
            surface.blit(lock, (460, surface.get_height() - 60))


def _init_joysticks() -> None:
    # Safe on platforms with no joystick support.
    try:
        count = pygame.joystick.get_count()
    except Exception:
        return

    for i in range(count):
        try:
            js = pygame.joystick.Joystick(i)
            js.init()
        except Exception:
            continue


def _new_seed() -> int:
    return random.SystemRandom().randint(1, 2**31 - 1)


def run(
    *, max_frames: int | None = None, event_injector: Callable[[int], None] | None = None
) -> int:
    pygame.init()
    _init_joysticks()

    pygame.display.set_caption("RCAF CFAST Trainer")
    surface = pygame.display.set_mode((960, 540), pygame.FULLSCREEN if settings.fullscreen else 0)

    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 20)
    clock = pygame.time.Clock()

    app = App(surface=surface, font=font)

    workouts = PlaceholderScreen(app, "90-minute workouts")
    drills = PlaceholderScreen(app, "Individual drills")

    axis_calibration = PlaceholderScreen(app, "Axis Calibration (placeholder)")
    axis_visualizer = PlaceholderScreen(app, "Axis Visualizer (placeholder)")
    input_profiles = PlaceholderScreen(app, "Input Profiles (placeholder)")
    data_storage = PlaceholderScreen(app, "Data & Storage (placeholder)")

    settings_screen = SettingsScreen(
        app,
        axis_calibration=axis_calibration,
        axis_visualizer=axis_visualizer,
        input_profiles=input_profiles,
        data_storage=data_storage,
    )

    real_clock = RealClock()

    def open_numerical_ops() -> None:
        seed = _new_seed()
        app.push(
            CognitiveTestScreen(
                app,
                engine_factory=lambda: build_numerical_operations_test(
                    clock=real_clock, seed=seed, difficulty=0.5
                ),
            )
        )

    def open_airborne_numerical() -> None:
        seed = _new_seed()
        app.push(
            CognitiveTestScreen(
                app,
                engine_factory=lambda: build_airborne_numerical_test(
                    clock=real_clock, seed=seed, difficulty=0.5
                ),
            )
        )

    def open_math_reasoning() -> None:
        seed = _new_seed()
        app.push(
            CognitiveTestScreen(
                app,
                engine_factory=lambda: build_math_reasoning_test(
                    clock=real_clock, seed=seed, difficulty=0.5
                ),
            )
        )

    tests_menu = MenuScreen(
        app,
        "Tests",
        [
            MenuItem("Numerical Operations", open_numerical_ops),
            MenuItem("Mathematics Reasoning", open_math_reasoning),
            MenuItem("Airborne Numerical Test", open_airborne_numerical),
            MenuItem("Back", app.pop),
        ],
    )

    main_items = [
        MenuItem("90-minute workouts", lambda: app.push(workouts)),
        MenuItem("Individual drills", lambda: app.push(drills)),
        MenuItem("Tests", lambda: app.push(tests_menu)),
        MenuItem("Settings", lambda: app.push(settings_screen)),
        MenuItem("Quit", app.quit),
    ]

    app.push(MenuScreen(app, "Main Menu", main_items, is_root=True))

    frame = 0
    try:
        while app.running:
            if event_injector is not None:
                event_injector(frame)

            for event in pygame.event.get():
                app.handle_event(event)

            app.render()

            # Always draw overlays on the current display surface (fullscreen toggle recreates it).
            display_surface = pygame.display.get_surface()
            if display_surface is not None and settings.show_fps_overlay:
                fps = int(clock.get_fps())
                fps_surf = small_font.render(f"{fps} FPS", True, (140, 220, 140))
                display_surface.blit(
                    fps_surf,
                    (display_surface.get_width() - fps_surf.get_width() - 10, 10),
                )

            pygame.display.flip()

            frame += 1
            if max_frames is not None and frame >= max_frames:
                break

            clock.tick(settings.max_fps)
    finally:
        pygame.quit()

    return 0