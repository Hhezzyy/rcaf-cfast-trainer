from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import pygame


class Screen(Protocol):
    def handle_event(self, event: pygame.event.Event) -> None: ...
    def render(self, surface: pygame.Surface) -> None: ...


@dataclass(frozen=True)
class MenuItem:
    label: str
    action: Callable[[], None]


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
    def __init__(self, app: App, title: str, items: list[MenuItem], *, is_root: bool = False) -> None:
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


def _init_joysticks() -> None:
    # Safe on platforms with no joystick support; pygame-ce typically exposes this on Win/macOS.
    try:
        count = pygame.joystick.get_count()
    except Exception:
        return
    for i in range(count):
        try:
            js = pygame.joystick.Joystick(i)
            js.init()
        except Exception:
            # Ignore any problematic device; keyboard remains primary input.
            continue


def run(*, max_frames: int | None = None, event_injector: Callable[[int], None] | None = None) -> int:
    pygame.init()
    _init_joysticks()

    pygame.display.set_caption("RCAF CFAST Trainer")
    surface = pygame.display.set_mode((960, 540))
    font = pygame.font.Font(None, 36)
    clock = pygame.time.Clock()

    app = App(surface=surface, font=font)

    workouts = PlaceholderScreen(app, "90-minute workouts")
    drills = PlaceholderScreen(app, "Individual drills")
    tests = PlaceholderScreen(app, "Individual tests")
    settings = PlaceholderScreen(app, "Settings")

    main_items = [
        MenuItem("90-minute workouts", lambda: app.push(workouts)),
        MenuItem("Individual drills", lambda: app.push(drills)),
        MenuItem("Individual tests", lambda: app.push(tests)),
        MenuItem("Settings", lambda: app.push(settings)),
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
            pygame.display.flip()

            frame += 1
            if max_frames is not None and frame >= max_frames:
                break
            clock.tick(60)
    finally:
        pygame.quit()

    return 0