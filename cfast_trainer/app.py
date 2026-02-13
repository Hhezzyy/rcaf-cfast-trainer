"""Pygame UI shell for the RCAF CFAST Trainer.

This task adds two cognitive tests under a "Tests" submenu:
- Numerical Operations (mental arithmetic)
- Mathematics Reasoning (time/speed/distance word problems)
- Airborne Numerical Test (HHMM timing with map/table UI)

Deterministic timing/scoring/RNG/state lives in cfast_trainer/* (core modules).
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import pygame

from .airborne_numerical import AirborneScenario, TEMPLATES_BY_NAME, build_airborne_numerical_test
from .clock import RealClock
from .cognitive_core import Phase, TestSnapshot
from .math_reasoning import build_math_reasoning_test
from .digit_recognition import DigitRecognitionPayload, build_digit_recognition_test
from .numerical_operations import build_numerical_operations_test


class Screen(Protocol):
    def handle_event(self, event: pygame.event.Event) -> None: ...
    def render(self, surface: pygame.Surface) -> None: ...


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
            (f"Show FPS overlay: {'ON' if settings.show_fps_overlay else 'OFF'}", self._toggle_show_fps),
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
        self._tiny_font = pygame.font.Font(None, 18)
        self._big_font = pygame.font.Font(None, 72)
        self._mid_font = pygame.font.Font(None, 52)

        # Airborne-specific UI state (hold-to-show overlays).
        self._air_overlay: str | None = None  # "intro" | "fuel" | "parcel"
        self._air_show_distances = False

    def handle_event(self, event: pygame.event.Event) -> None:
        snap = self._engine.snapshot()
        scenario = snap.payload if isinstance(snap.payload, AirborneScenario) else None
        dr = snap.payload if isinstance(snap.payload, DigitRecognitionPayload) else None

        # Airborne: hold-to-show overlays.
        if scenario is not None:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    self._air_overlay = "intro"
                elif event.key == pygame.K_d:
                    self._air_overlay = "fuel"
                elif event.key == pygame.K_f:
                    self._air_overlay = "parcel"
                elif event.key == pygame.K_a:
                    self._air_show_distances = True
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_a:
                    self._air_show_distances = False
                elif event.key == pygame.K_s and self._air_overlay == "intro":
                    self._air_overlay = None
                elif event.key == pygame.K_d and self._air_overlay == "fuel":
                    self._air_overlay = None
                elif event.key == pygame.K_f and self._air_overlay == "parcel":
                    self._air_overlay = None

        if event.type != pygame.KEYDOWN:
            return

        key = event.key

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
            if dr is not None and not dr.accepting_input:
                return
            
            accepted = self._engine.submit_answer(self._input)
            if accepted:
                self._input = ""
            return

        if snap.phase not in (Phase.PRACTICE, Phase.SCORED):
            return
        
        if dr is not None and not dr.accepting_input:
            return

        if key == pygame.K_BACKSPACE:
            # Airborne test: no backspace editing.
            if scenario is None:
                self._input = self._input[:-1]
            return

        ch = event.unicode
        if scenario is not None:
            if ch and ch.isdigit() and len(self._input) < 4:
                self._input += ch
            return

        if ch and (ch.isdigit() or (ch == "-" and self._input == "")):
            self._input += ch

    def render(self, surface: pygame.Surface) -> None:
        # If the timer expires mid-entry, auto-submit the digits typed so far (no backspace on the
        # real test; partial entry will simply score 0).
        snap_pre = self._engine.snapshot()
        if (
            snap_pre.phase is Phase.SCORED
            and snap_pre.time_remaining_s is not None
            and snap_pre.time_remaining_s <= 0.0
            and self._input.strip() != ""
        ):
            self._engine.submit_answer(self._input)
            self._input = ""

        self._engine.update()
        snap = self._engine.snapshot()
        scenario = snap.payload if isinstance(snap.payload, AirborneScenario) else None

        surface.fill((10, 10, 14))

        title = self._app.font.render(snap.title, True, (235, 235, 245))
        surface.blit(title, (40, 30))

        y_info = 80
        if snap.time_remaining_s is not None:
            rem = int(round(snap.time_remaining_s))
            mm = rem // 60
            ss = rem % 60
            timer = self._small_font.render(f"Time remaining: {mm:02d}:{ss:02d}", True, (200, 200, 210))
            surface.blit(timer, (40, y_info))
            y_info += 28

        stats = self._small_font.render(
            f"Scored: {snap.correct_scored}/{snap.attempted_scored}", True, (180, 180, 190)
        )
        surface.blit(stats, (40, y_info))

        if scenario is not None and snap.phase in (Phase.PRACTICE, Phase.SCORED):
            self._render_airborne_question(surface, snap, scenario)
        else:
            prompt_lines = str(snap.prompt).split("\n")
            y = 140
            for line in prompt_lines[:10]:
                txt = self._small_font.render(line, True, (235, 235, 245))
                surface.blit(txt, (40, y))
                y += 26

        if snap.phase in (Phase.PRACTICE, Phase.SCORED):
            if scenario is None:
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

    def _render_airborne_question(self, surface: pygame.Surface, snap: TestSnapshot, scenario: AirborneScenario) -> None:
        # Layout: menu on left, map, bottom table, answer/timer on the right.
        w, h = surface.get_size()
        menu_rect = pygame.Rect(20, 80, 220, h - 120)
        content_rect = pygame.Rect(menu_rect.right + 20, 80, w - menu_rect.right - 40, h - 160)
        map_rect = pygame.Rect(content_rect.x, content_rect.y, content_rect.w, 250)
        table_rect = pygame.Rect(content_rect.x, map_rect.bottom + 20, content_rect.w, content_rect.h - 270)
        answer_rect = pygame.Rect(content_rect.right - 210, 30, 190, 58)
        task_rect = pygame.Rect(content_rect.x, 30, content_rect.w - 230, 58)

        # Task / prompt.
        pygame.draw.rect(surface, (18, 18, 26), task_rect)
        pygame.draw.rect(surface, (70, 70, 85), task_rect, 2)
        task_lines = str(snap.prompt).split("\n")
        task = task_lines[0] if task_lines else ""
        surface.blit(self._small_font.render(task, True, (235, 235, 245)), (task_rect.x + 12, task_rect.y + 10))
        surface.blit(
            self._tiny_font.render("Answer HHMM (4 digits). No backspace.", True, (150, 150, 165)),
            (task_rect.x + 12, task_rect.y + 34),
        )

        # Answer box with 4 slots.
        pygame.draw.rect(surface, (18, 18, 26), answer_rect)
        pygame.draw.rect(surface, (70, 70, 85), answer_rect, 2)
        surface.blit(self._tiny_font.render("ANSWER", True, (150, 150, 165)), (answer_rect.x + 10, answer_rect.y + 6))
        slots_x = answer_rect.x + 10
        slots_y = answer_rect.y + 26
        for i in range(4):
            r = pygame.Rect(slots_x + i * 42, slots_y, 34, 26)
            pygame.draw.rect(surface, (30, 30, 40), r)
            pygame.draw.rect(surface, (90, 90, 110), r, 2)
            ch = self._input[i] if i < len(self._input) else ""
            if ch:
                surface.blit(self._small_font.render(ch, True, (235, 235, 245)), (r.x + 10, r.y + 2))

        # Menu panel.
        pygame.draw.rect(surface, (18, 18, 26), menu_rect)
        pygame.draw.rect(surface, (70, 70, 85), menu_rect, 2)
        surface.blit(self._app.font.render("Menu", True, (235, 235, 245)), (menu_rect.x + 14, menu_rect.y + 10))
        surface.blit(self._tiny_font.render("Hold S: Intro", True, (150, 150, 165)), (menu_rect.x + 14, menu_rect.y + 60))
        surface.blit(self._tiny_font.render("Hold D: Speed & Fuel", True, (150, 150, 165)), (menu_rect.x + 14, menu_rect.y + 84))
        surface.blit(self._tiny_font.render("Hold F: Speed & Parcel", True, (150, 150, 165)), (menu_rect.x + 14, menu_rect.y + 108))
        surface.blit(self._tiny_font.render("Hold A: Show distances", True, (150, 150, 165)), (menu_rect.x + 14, menu_rect.y + 140))

        # Map.
        pygame.draw.rect(surface, (12, 12, 18), map_rect)
        pygame.draw.rect(surface, (70, 70, 85), map_rect, 2)
        self._draw_airborne_map(surface, map_rect, scenario)

        # Bottom table.
        pygame.draw.rect(surface, (12, 12, 18), table_rect)
        pygame.draw.rect(surface, (70, 70, 85), table_rect, 2)
        self._draw_airborne_table(surface, table_rect, scenario)

        # Overlay panels (hold-to-show; intentionally occludes map/table).
        if self._air_overlay is not None:
            overlay = pygame.Surface((content_rect.w, content_rect.h), flags=pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 210))
            surface.blit(overlay, (content_rect.x, content_rect.y))
            panel = pygame.Rect(content_rect.x + 20, content_rect.y + 20, content_rect.w - 40, content_rect.h - 40)
            pygame.draw.rect(surface, (18, 18, 26), panel)
            pygame.draw.rect(surface, (120, 120, 140), panel, 2)

            title = {"intro": "Introduction", "fuel": "Speed & Fuel Consumption", "parcel": "Speed & Parcel Weight"}[
                self._air_overlay
            ]
            surface.blit(self._app.font.render(title, True, (235, 235, 245)), (panel.x + 16, panel.y + 14))

            lines: list[str] = []
            if self._air_overlay == "intro":
                route_names = " → ".join(scenario.node_names[i] for i in scenario.route)
                lines = [
                    f"Start time: {scenario.start_time_hhmm}",
                    f"Route: {route_names}",
                    "",
                    "Use the map and table to derive the required time.",
                    "Distances are hidden unless you hold A.",
                ]
            elif self._air_overlay == "fuel":
                lines = [
                    f"Speed: {getattr(scenario, 'speed_value', 0)} {getattr(scenario, 'speed_unit', '')}",
                    f"Fuel burn: {getattr(scenario, 'fuel_burn_per_hr', 0)} L/hr",
                    "",
                    "(Training note: fuel calculations will be added as question types.)",
                ]
            else:
                pw = getattr(scenario, "parcel_weight_kg", getattr(scenario, "parcel_weight", 0))
                lines = [
                    f"Speed: {getattr(scenario, 'speed_value', 0)} {getattr(scenario, 'speed_unit', '')}",
                    f"Parcel weight: {pw} kg",
                    "",
                    "(Training note: weight effects will be added as question types.)",
                ]

            y = panel.y + 70
            for line in lines:
                surface.blit(self._small_font.render(line, True, (235, 235, 245)), (panel.x + 16, y))
                y += 26

            graph_rect = pygame.Rect(panel.x + 16, panel.y + 140, panel.w - 32, panel.h - 160)
            if self._air_overlay == "fuel":
                self._draw_airborne_fuel_panel(surface, graph_rect, scenario)
            elif self._air_overlay == "parcel":
                self._draw_airborne_parcel_panel(surface, graph_rect, scenario)

    def _render_digit_recognition(self, surface: pygame.Surface, snap: TestSnapshot, payload: DigitRecognitionPayload) -> None:
        w, h = surface.get_size()

        if snap.prompt:
            surface.blit(self._app.font.render(str(snap.prompt), True, (235, 235, 245)), (40, 140))

        if payload.display_digits:
            txt = payload.display_digits
            surf = self._big_font.render(txt, True, (235, 235, 245))
            if surf.get_width() > int(w * 0.92):
                surf = self._mid_font.render(txt, True, (235, 235, 245))
            x = (w - surf.get_width()) // 2
            y = (h - surf.get_height()) // 2 - 10
            surface.blit(surf, (x, y))

    def _airborne_graph_seed(self, scenario: AirborneScenario) -> int:
        # Stable per-scenario seed (no Python hash()).
        seed = 2166136261

        def mix(x: int) -> None:
            nonlocal seed
            seed ^= (x & 0xFFFFFFFF)
            seed = (seed * 16777619) & 0xFFFFFFFF

        mix(int(getattr(scenario, "speed_value", 0)))
        mix(int(getattr(scenario, "fuel_burn_per_hr", 0)))
        mix(int(getattr(scenario, "parcel_weight_kg", getattr(scenario, "parcel_weight", 0))))
        for name in getattr(scenario, "node_names", ()):
            for ch in name:
                mix(ord(ch))
        for idx in getattr(scenario, "route", ()):
            mix(int(idx))
        return seed

    def _draw_airborne_bar_chart(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        *,
        title: str,
        x_labels: list[str],
        values: list[int],
        value_unit: str,
    ) -> None:
        pygame.draw.rect(surface, (18, 18, 26), rect)
        pygame.draw.rect(surface, (120, 120, 140), rect, 2)

        surface.blit(self._small_font.render(title, True, (235, 235, 245)), (rect.x + 12, rect.y + 10))

        chart = pygame.Rect(rect.x + 40, rect.y + 42, rect.w - 56, rect.h - 70)
        pygame.draw.line(surface, (140, 140, 150), (chart.x, chart.bottom), (chart.right, chart.bottom), 1)
        pygame.draw.line(surface, (140, 140, 150), (chart.x, chart.y), (chart.x, chart.bottom), 1)

        if not values:
            return

        vmax = max(values) or 1
        n = len(values)
        gap = 8
        bar_w = max(10, (chart.w - gap * (n + 1)) // n)

        for i, (lbl, v) in enumerate(zip(x_labels, values, strict=False)):
            x = chart.x + gap + i * (bar_w + gap)
            hh = int(round((v / vmax) * (chart.h - 20)))
            bar = pygame.Rect(x, chart.bottom - hh, bar_w, hh)
            pygame.draw.rect(surface, (90, 90, 110), bar)

            t = self._tiny_font.render(f"{v}{value_unit}", True, (200, 200, 210))
            surface.blit(t, t.get_rect(midbottom=(bar.centerx, bar.y - 2)))

            xl = self._tiny_font.render(lbl, True, (150, 150, 165))
            surface.blit(xl, xl.get_rect(midtop=(bar.centerx, chart.bottom + 4)))

    def _draw_airborne_table_small(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        *,
        title: str,
        headers: tuple[str, str],
        rows: list[tuple[str, str]],
    ) -> None:
        pygame.draw.rect(surface, (18, 18, 26), rect)
        pygame.draw.rect(surface, (120, 120, 140), rect, 2)
        surface.blit(self._small_font.render(title, True, (235, 235, 245)), (rect.x + 12, rect.y + 10))

        x1 = rect.x + 14
        x2 = rect.x + rect.w // 2 + 10
        y = rect.y + 46

        surface.blit(self._tiny_font.render(headers[0], True, (150, 150, 165)), (x1, y))
        surface.blit(self._tiny_font.render(headers[1], True, (150, 150, 165)), (x2, y))
        y += 20

        for a, b in rows[:10]:
            surface.blit(self._tiny_font.render(a, True, (235, 235, 245)), (x1, y))
            surface.blit(self._tiny_font.render(b, True, (235, 235, 245)), (x2, y))
            y += 20

    def _draw_airborne_fuel_panel(self, surface: pygame.Surface, rect: pygame.Rect, scenario: AirborneScenario) -> None:
        seed = self._airborne_graph_seed(scenario)
        base_speed = max(1, int(getattr(scenario, "speed_value", 1)))
        base_burn = int(getattr(scenario, "fuel_burn_per_hr", 0))
        speeds = [int(round(base_speed * f)) for f in (0.8, 0.9, 1.0, 1.1, 1.2)]
        exp = 2.0 if (seed & 2) == 0 else 2.2
        burns = [int(round(base_burn * ((s / base_speed) ** exp))) for s in speeds]
        labels = [str(s) for s in speeds]

        if (seed & 1) == 0:
            self._draw_airborne_bar_chart(
                surface, rect, title="Fuel burn vs speed", x_labels=labels, values=burns, value_unit=""
            )
        else:
            rows = [(f"{sp} {getattr(scenario, 'speed_unit', '')}", f"{bn} L/hr") for sp, bn in zip(speeds, burns, strict=False)]
            self._draw_airborne_table_small(surface, rect, title="Fuel burn table", headers=("SPEED", "BURN"), rows=rows)

    def _draw_airborne_parcel_panel(self, surface: pygame.Surface, rect: pygame.Rect, scenario: AirborneScenario) -> None:
        seed = self._airborne_graph_seed(scenario) ^ 0x9E3779B9
        base_speed = max(1, int(getattr(scenario, "speed_value", 1)))
        weights = [0, 200, 400, 600, 800, 1000]
        slope = 6 + (seed % 7)  # speed drop per 100kg
        speeds = [max(1, base_speed - int((w / 100) * slope)) for w in weights]
        wlabels = [str(w) for w in weights]

        if (seed & 1) == 0:
            self._draw_airborne_bar_chart(
                surface, rect, title="Speed vs parcel weight", x_labels=wlabels, values=speeds, value_unit=""
            )
        else:
            rows = [(f"{w} kg", f"{sp} {getattr(scenario, 'speed_unit', '')}") for w, sp in zip(weights, speeds, strict=False)]
            self._draw_airborne_table_small(surface, rect, title="Parcel weight table", headers=("WEIGHT", "SPEED"), rows=rows)

    def _draw_airborne_map(self, surface: pygame.Surface, rect: pygame.Rect, scenario: AirborneScenario) -> None:
        template = TEMPLATES_BY_NAME.get(scenario.template_name)
        if template is None:
            return

        node_px: list[tuple[int, int]] = []
        for nx, ny in template.nodes:
            x = int(rect.x + nx * rect.w)
            y = int(rect.y + ny * rect.h)
            node_px.append((x, y))

        for idx, (ea, eb) in enumerate(template.edges):
            a = node_px[ea]
            b = node_px[eb]
            pygame.draw.line(surface, (70, 70, 85), a, b, 2)

            if self._air_show_distances:
                midx = (a[0] + b[0]) / 2
                midy = (a[1] + b[1]) / 2
                dx = b[0] - a[0]
                dy = b[1] - a[1]
                length = max(1.0, (dx * dx + dy * dy) ** 0.5)
                ox = int(-dy / length * 10)
                oy = int(dx / length * 10)

                text = self._tiny_font.render(str(scenario.edge_distances[idx]), True, (12, 12, 18))
                bg = text.get_rect(center=(int(midx) + ox, int(midy) + oy))
                bg.inflate_ip(10, 6)
                pygame.draw.rect(surface, (235, 235, 245), bg)
                surface.blit(text, text.get_rect(center=bg.center))

        for i, (x, y) in enumerate(node_px):
            pygame.draw.circle(surface, (12, 12, 18), (x, y), 10)
            pygame.draw.circle(surface, (235, 235, 245), (x, y), 10, 2)

            lx = x + 18 if x < rect.right - 80 else x - 18
            ly = y
            label = self._tiny_font.render(scenario.node_names[i], True, (12, 12, 18))
            bg = label.get_rect(midleft=(lx, ly))
            bg.inflate_ip(10, 6)
            pygame.draw.rect(surface, (235, 235, 245), bg)
            surface.blit(label, label.get_rect(midleft=bg.midleft))

    def _draw_airborne_table(self, surface: pygame.Surface, rect: pygame.Rect, scenario: AirborneScenario) -> None:
        header = self._tiny_font.render("Journey", True, (235, 235, 245))
        surface.blit(header, (rect.x + 12, rect.y + 10))

        cols = [
            ("LEG", rect.x + 12),
            ("FROM", rect.x + 80),
            ("TO", rect.x + 210),
            ("DIST", rect.x + 340),
            ("SPEED", rect.x + 430),
            ("TIME", rect.x + 530),
            ("PARCEL", rect.x + 610),
        ]
        y = rect.y + 32
        for label, x in cols:
            surface.blit(self._tiny_font.render(label, True, (150, 150, 165)), (x, y))
        y += 22

        pw = getattr(scenario, "parcel_weight_kg", getattr(scenario, "parcel_weight", 0))

        for i in range(3):
            if i < len(scenario.legs):
                leg = scenario.legs[i]
                dist = str(getattr(leg, "distance", "----")) if self._air_show_distances else "----"
                speed = "----"
                t = "----"
                row = [
                    (str(i + 1), cols[0][1]),
                    (scenario.node_names[getattr(leg, "frm")], cols[1][1]),
                    (scenario.node_names[getattr(leg, "to")], cols[2][1]),
                    (dist, cols[3][1]),
                    (speed, cols[4][1]),
                    (t, cols[5][1]),
                    (str(pw), cols[6][1]),
                ]
            else:
                row = [("", x) for _, x in cols]

            for text, x in row:
                surface.blit(self._tiny_font.render(text, True, (235, 235, 245)), (x, y))
            y += 22


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


def run(*, max_frames: int | None = None, event_injector: Callable[[int], None] | None = None) -> int:
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
                engine_factory=lambda: build_numerical_operations_test(clock=real_clock, seed=seed, difficulty=0.5),
            )
        )

    def open_airborne_numerical() -> None:
        seed = _new_seed()
        app.push(
            CognitiveTestScreen(
                app,
                engine_factory=lambda: build_airborne_numerical_test(clock=real_clock, seed=seed, difficulty=0.5),
            )
        )

    def open_math_reasoning() -> None:
        seed = _new_seed()
        app.push(
            CognitiveTestScreen(
                app,
                engine_factory=lambda: build_math_reasoning_test(clock=real_clock, seed=seed, difficulty=0.5),
            )
        )
    def open_digit_recognition() -> None:
        seed = _new_seed()
        app.push(
            CognitiveTestScreen(
                app,
                engine_factory=lambda: build_digit_recognition_test(clock=real_clock, seed=seed, difficulty=0.5),
            )
        )
    tests_menu = MenuScreen(
        app,
        "Tests",
        [
            MenuItem("Numerical Operations", open_numerical_ops),
            MenuItem("Mathematics Reasoning", open_math_reasoning),
            MenuItem("Airborne Numerical Test", open_airborne_numerical),
            MenuItem("Digit Recognition", open_digit_recognition),
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