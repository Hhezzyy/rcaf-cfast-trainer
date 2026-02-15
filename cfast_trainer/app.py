"""Pygame UI shell for the RCAF CFAST Trainer.

This task adds two cognitive tests under a "Tests" submenu:
- Numerical Operations (mental arithmetic)
- Mathematics Reasoning (multiple-choice word problems)
- Airborne Numerical Test (HHMM timing with map/table UI)

Deterministic timing/scoring/RNG/state lives in cfast_trainer/* (core modules).
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast
from typing import Protocol

import pygame

from .airborne_numerical import AirborneScenario, TEMPLATES_BY_NAME, build_airborne_numerical_test
from .angles_bearings_degrees import (
    AnglesBearingsDegreesPayload,
    AnglesBearingsQuestionKind,
    build_angles_bearings_degrees_test,
)
from .clock import RealClock
from .cognitive_core import Phase, TestSnapshot
from .digit_recognition import DigitRecognitionPayload, build_digit_recognition_test
from .instrument_comprehension import (
    InstrumentComprehensionPayload,
    InstrumentComprehensionTrialKind,
    InstrumentState,
    build_instrument_comprehension_test,
)
from .math_reasoning import MathReasoningPayload, build_math_reasoning_test
from .numerical_operations import build_numerical_operations_test
from .visual_search import VisualSearchPayload, VisualSearchTaskKind, build_visual_search_test


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


WINDOW_SIZE = (960, 540)
TARGET_FPS = 60


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
        self._title_font = pygame.font.Font(None, 42)
        self._item_font = pygame.font.Font(None, 32)
        self._hint_font = pygame.font.Font(None, 22)

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

    def _fit_label(self, font: pygame.font.Font, label: str, max_width: int) -> str:
        if max_width <= 0:
            return ""
        if font.size(label)[0] <= max_width:
            return label
        clipped = label
        while clipped and font.size(f"{clipped}...")[0] > max_width:
            clipped = clipped[:-1]
        return f"{clipped}..." if clipped else "..."

    def render(self, surface: pygame.Surface) -> None:
        w, h = surface.get_size()
        bg = (3, 9, 78)
        panel_bg = (8, 18, 104)
        header_bg = (18, 30, 118)
        border = (226, 236, 255)
        text_main = (238, 245, 255)
        text_muted = (186, 200, 224)
        active_bg = (244, 248, 255)
        active_text = (14, 26, 74)

        surface.fill(bg)

        frame_margin = max(10, min(26, w // 34))
        frame = pygame.Rect(
            frame_margin,
            frame_margin,
            max(260, w - frame_margin * 2),
            max(220, h - frame_margin * 2),
        )
        pygame.draw.rect(surface, panel_bg, frame)
        pygame.draw.rect(surface, border, frame, 2)

        header_h = max(34, min(52, h // 8))
        header = pygame.Rect(frame.x + 2, frame.y + 2, frame.w - 4, header_h)
        pygame.draw.rect(surface, header_bg, header)
        pygame.draw.line(
            surface,
            border,
            (header.x, header.bottom),
            (header.right, header.bottom),
            1,
        )

        tag = self._hint_font.render("MENU", True, text_muted)
        surface.blit(tag, (header.x + 12, header.y + (header.h - tag.get_height()) // 2))

        title = self._title_font.render(self._title, True, text_main)
        surface.blit(title, title.get_rect(center=(frame.centerx, header.centery)))

        content_top = header.bottom + max(16, h // 30)
        content_bottom = frame.bottom - max(44, h // 12)
        list_rect = pygame.Rect(
            frame.x + max(14, w // 44),
            content_top,
            frame.w - max(28, w // 22),
            max(120, content_bottom - content_top),
        )
        pygame.draw.rect(surface, (6, 13, 92), list_rect)
        pygame.draw.rect(surface, (78, 102, 170), list_rect, 1)

        item_count = max(1, len(self._items))
        gap = max(4, min(10, list_rect.h // max(10, item_count * 3)))
        row_h = max(30, min(44, (list_rect.h - gap * (item_count + 1)) // item_count))
        total_h = row_h * item_count + gap * (item_count - 1)
        y = list_rect.y + max(8, (list_rect.h - total_h) // 2)

        for idx, item in enumerate(self._items):
            row = pygame.Rect(list_rect.x + 12, y, list_rect.w - 24, row_h)
            selected = idx == self._selected
            if selected:
                pygame.draw.rect(surface, active_bg, row)
                pygame.draw.rect(surface, (120, 142, 196), row, 2)
            else:
                pygame.draw.rect(surface, (9, 20, 106), row)
                pygame.draw.rect(surface, (62, 84, 152), row, 1)

            color = active_text if selected else text_main
            label = self._fit_label(self._item_font, item.label, row.w - 20)
            text = self._item_font.render(label, True, color)
            surface.blit(text, (row.x + 10, row.y + (row.h - text.get_height()) // 2))
            y += row_h + gap

        footer = "Enter/Space: Select  |  Esc/Backspace: Back  |  D-pad + Button0/1"
        foot = self._hint_font.render(footer, True, text_muted)
        surface.blit(foot, foot.get_rect(midbottom=(frame.centerx, frame.bottom - 10)))


class CognitiveTestScreen:
    def __init__(self, app: App, *, engine_factory: Callable[[], CognitiveEngine]) -> None:
        self._app = app
        self._engine: CognitiveEngine = engine_factory()
        self._input = ""
        self._math_choice = 1

        self._small_font = pygame.font.Font(None, 24)
        self._tiny_font = pygame.font.Font(None, 18)
        self._big_font = pygame.font.Font(None, 72)
        self._mid_font = pygame.font.Font(None, 52)
        self._num_header_font = pygame.font.Font(None, 28)
        self._num_prompt_fonts = [
            pygame.font.Font(None, 112),
            pygame.font.Font(None, 96),
            pygame.font.Font(None, 84),
            pygame.font.Font(None, 72),
        ]
        self._num_input_font = pygame.font.Font(None, 58)

        # Airborne-specific UI state (hold-to-show overlays).
        self._air_overlay: str | None = None  # "intro" | "fuel" | "parcel"
        self._air_show_distances = False

        # Cached procedural sprites for Instrument Comprehension dials.
        self._instrument_sprite_cache: dict[tuple[object, ...], pygame.Surface] = {}

    def handle_event(self, event: pygame.event.Event) -> None:
        snap = self._engine.snapshot()
        p = snap.payload
        scenario = p if isinstance(p, AirborneScenario) else None
        math_payload: MathReasoningPayload | None = p if isinstance(p, MathReasoningPayload) else None
        angles_payload: AnglesBearingsDegreesPayload | None = (
            p if isinstance(p, AnglesBearingsDegreesPayload) else None
        )

        # Emergency exit: allow a hard escape from any state (including SCORED).
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_F12:
                self._app.pop()
                return
            if event.key == pygame.K_ESCAPE and (event.mod & pygame.KMOD_SHIFT):
                self._app.pop()
                return

        dr: DigitRecognitionPayload | None = None
        if p is not None and hasattr(p, "display_digits") and hasattr(p, "accepting_input"):
            dr = cast(DigitRecognitionPayload, p)
        if dr is not None and not dr.accepting_input:
            return
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
                self._math_choice = 1
                return
            if snap.phase is Phase.PRACTICE_DONE:
                self._engine.start_scored()
                self._input = ""
                self._math_choice = 1
                return
            if snap.phase is Phase.RESULTS:
                self._app.pop()
                return
            if dr is not None and not dr.accepting_input:
                return
            if math_payload is not None and self._input == "":
                self._input = str(self._math_choice)
            if angles_payload is not None and self._input == "":
                self._input = str(self._math_choice)

            accepted = self._engine.submit_answer(self._input)
            if accepted:
                self._input = ""
                self._math_choice = 1
            return

        if snap.phase not in (Phase.PRACTICE, Phase.SCORED):
            return
        
        if dr is not None and not dr.accepting_input:
            return

        if math_payload is not None:
            option_count = max(1, len(math_payload.options))
            if key in (pygame.K_UP, pygame.K_w):
                self._math_choice = option_count if self._math_choice <= 1 else self._math_choice - 1
                self._input = str(self._math_choice)
                return
            if key in (pygame.K_DOWN, pygame.K_s):
                self._math_choice = 1 if self._math_choice >= option_count else self._math_choice + 1
                self._input = str(self._math_choice)
                return

            choice = self._choice_from_key(key)
            if choice is not None and 1 <= choice <= option_count:
                self._math_choice = choice
                self._input = str(choice)
            return

        if angles_payload is not None:
            option_count = max(1, len(angles_payload.options))
            if key in (pygame.K_UP, pygame.K_w):
                self._math_choice = option_count if self._math_choice <= 1 else self._math_choice - 1
                self._input = str(self._math_choice)
                return
            if key in (pygame.K_DOWN, pygame.K_s):
                self._math_choice = 1 if self._math_choice >= option_count else self._math_choice + 1
                self._input = str(self._math_choice)
                return

            choice = self._choice_from_key(key)
            if choice is not None and 1 <= choice <= option_count:
                self._math_choice = choice
                self._input = str(choice)
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
        # Update engine and take a fresh snapshot.
        self._engine.update()
        snap = self._engine.snapshot()

        # If the timer expires mid-entry, auto-submit what was typed so far.
        if (
            snap.phase is Phase.SCORED
            and snap.time_remaining_s is not None
            and snap.time_remaining_s <= 0.0
            and self._input.strip() != ""
        ):
            self._engine.submit_answer(self._input)
            self._input = ""
            self._engine.update()
            snap = self._engine.snapshot()

        # Identify payloads.
        p = snap.payload
        scenario: AirborneScenario | None = p if isinstance(p, AirborneScenario) else None
        abd: AnglesBearingsDegreesPayload | None = (
            p if isinstance(p, AnglesBearingsDegreesPayload) else None
        )
        ic: InstrumentComprehensionPayload | None = (
            p if isinstance(p, InstrumentComprehensionPayload) else None
        )
        vs: VisualSearchPayload | None = p if isinstance(p, VisualSearchPayload) else None
        mr: MathReasoningPayload | None = p if isinstance(p, MathReasoningPayload) else None
        dr: DigitRecognitionPayload | None = None
        if p is not None:
            if isinstance(p, DigitRecognitionPayload):
                dr = p
            elif hasattr(p, "display_digits") and hasattr(p, "accepting_input"):
                dr = cast(DigitRecognitionPayload, p)

        is_numerical_ops = snap.title == "Numerical Operations"
        is_math_reasoning = snap.title == "Mathematics Reasoning"
        is_angles_bearings = snap.title == "Angles, Bearings and Degrees"
        is_digit_recognition = snap.title == "Digit Recognition"
        is_instrument_comprehension = snap.title == "Instrument Comprehension"
        if is_numerical_ops and snap.phase in (Phase.PRACTICE, Phase.SCORED):
            self._render_numerical_operations_question(surface, snap)
        elif is_math_reasoning:
            self._render_math_reasoning(surface, snap, mr)
        elif is_angles_bearings:
            self._render_angles_bearings_screen(surface, snap, abd)
        elif is_digit_recognition:
            self._render_digit_recognition_screen(surface, snap, dr)
        elif is_instrument_comprehension:
            self._render_instrument_comprehension_screen(surface, snap, ic)
        elif vs is not None and snap.phase in (Phase.PRACTICE, Phase.SCORED):
            self._render_visual_search_question(surface, snap, vs)
        else:
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
            if is_numerical_ops:
                self._render_numerical_operations_answer_box(surface, snap)
            elif is_math_reasoning:
                pass
            elif is_angles_bearings:
                pass
            elif is_digit_recognition:
                self._render_digit_recognition_answer_box(surface, snap, dr)
            elif is_instrument_comprehension:
                pass
            elif vs is not None:
                pass
            elif scenario is None and (dr is None or dr.accepting_input):
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

    @staticmethod
    def _choice_from_key(key: int) -> int | None:
        mapping = {
            pygame.K_1: 1,
            pygame.K_2: 2,
            pygame.K_3: 3,
            pygame.K_4: 4,
            pygame.K_KP1: 1,
            pygame.K_KP2: 2,
            pygame.K_KP3: 3,
            pygame.K_KP4: 4,
        }
        return mapping.get(key)

    def _render_math_reasoning(
        self,
        surface: pygame.Surface,
        snap: TestSnapshot,
        payload: MathReasoningPayload | None,
    ) -> None:
        w, h = surface.get_size()
        bg = (4, 12, 84)
        panel_bg = (8, 18, 104)
        header_bg = (18, 30, 118)
        border = (226, 236, 255)
        text_main = (238, 245, 255)
        text_muted = (188, 204, 228)
        active_bg = (244, 248, 255)
        active_text = (16, 32, 88)

        surface.fill(bg)

        margin = max(10, min(24, w // 34))
        frame = pygame.Rect(margin, margin, max(280, w - margin * 2), max(220, h - margin * 2))
        pygame.draw.rect(surface, panel_bg, frame)
        pygame.draw.rect(surface, border, frame, 2)

        header_h = max(40, min(56, h // 7))
        header = pygame.Rect(frame.x + 2, frame.y + 2, frame.w - 4, header_h)
        pygame.draw.rect(surface, header_bg, header)
        pygame.draw.line(surface, border, (header.x, header.bottom), (header.right, header.bottom), 1)

        phase_label = {
            Phase.INSTRUCTIONS: "Instructions",
            Phase.PRACTICE: "Practice",
            Phase.PRACTICE_DONE: "Practice Complete",
            Phase.SCORED: "Timed Test",
            Phase.RESULTS: "Results",
        }.get(snap.phase, "Task")
        surface.blit(
            self._tiny_font.render(phase_label, True, text_muted),
            (header.x + 12, header.y + (header.h - self._tiny_font.get_height()) // 2),
        )

        title = self._small_font.render("Mathematics Reasoning", True, text_main)
        surface.blit(title, title.get_rect(midleft=(header.x + 145, header.centery)))

        stats_text = f"{snap.correct_scored}/{snap.attempted_scored}"
        stats = self._tiny_font.render(stats_text, True, text_muted)
        stats_rect = stats.get_rect(midright=(header.right - 12, header.centery))
        surface.blit(stats, stats_rect)
        stats_label = self._tiny_font.render("Scored", True, text_muted)
        surface.blit(stats_label, stats_label.get_rect(midright=(stats_rect.left - 6, header.centery)))

        if snap.time_remaining_s is not None:
            rem = int(round(snap.time_remaining_s))
            mm = rem // 60
            ss = rem % 60
            timer = self._small_font.render(f"{mm:02d}:{ss:02d}", True, text_main)
            surface.blit(timer, timer.get_rect(topright=(frame.right - 12, header.bottom + 8)))

        content = pygame.Rect(
            frame.x + max(14, w // 48),
            header.bottom + max(12, h // 36),
            frame.w - max(28, w // 24),
            frame.bottom - header.bottom - max(62, h // 9),
        )
        pygame.draw.rect(surface, (6, 13, 92), content)
        pygame.draw.rect(surface, (78, 102, 170), content, 1)

        if payload is not None and snap.phase in (Phase.PRACTICE, Phase.SCORED):
            domain_tag = self._tiny_font.render(payload.domain.upper(), True, text_muted)
            surface.blit(domain_tag, (content.x + 12, content.y + 10))

            stem_rect = pygame.Rect(content.x + 12, content.y + 30, content.w - 24, max(72, content.h // 3))
            self._draw_wrapped_text(
                surface,
                payload.stem,
                stem_rect,
                color=text_main,
                font=self._small_font,
                max_lines=4,
            )

            selected = self._math_choice
            if self._input in ("1", "2", "3", "4"):
                selected = int(self._input)

            top = stem_rect.bottom + 10
            gap = 8
            rows = max(1, len(payload.options))
            row_h = max(42, min(58, (content.bottom - top - gap * (rows + 1)) // rows))
            y = top + gap

            for option in payload.options:
                row = pygame.Rect(content.x + 12, y, content.w - 24, row_h)
                is_selected = option.code == selected
                if is_selected:
                    pygame.draw.rect(surface, active_bg, row)
                    pygame.draw.rect(surface, (124, 148, 202), row, 2)
                else:
                    pygame.draw.rect(surface, (9, 20, 106), row)
                    pygame.draw.rect(surface, (62, 84, 152), row, 1)

                text_color = active_text if is_selected else text_main
                label = self._small_font.render(f"{option.code}. {option.text}", True, text_color)
                surface.blit(label, (row.x + 12, row.y + (row.h - label.get_height()) // 2))
                y += row_h + gap
        else:
            self._draw_wrapped_text(
                surface,
                str(snap.prompt),
                content.inflate(-24, -24),
                color=text_main,
                font=self._small_font,
                max_lines=12,
            )

        if snap.phase in (Phase.PRACTICE, Phase.SCORED):
            footer = "1-4: Select option  |  Up/Down: Move  |  Enter: Submit"
        elif snap.phase in (Phase.INSTRUCTIONS, Phase.PRACTICE_DONE):
            footer = "Enter: Continue  |  Esc/Backspace: Back"
        else:
            footer = "Enter: Return to Tests"
        footer_text = self._tiny_font.render(footer, True, text_muted)
        surface.blit(footer_text, footer_text.get_rect(midbottom=(frame.centerx, frame.bottom - 12)))

    def _render_numerical_operations_question(
        self,
        surface: pygame.Surface,
        snap: TestSnapshot,
    ) -> None:
        w, h = surface.get_size()
        bg = (0, 0, 116)
        frame_color = (236, 243, 255)
        text_main = (244, 248, 255)
        text_muted = (214, 225, 244)
        accent = (164, 190, 235)

        surface.fill(bg)
        margin = max(8, min(18, w // 50))
        frame = pygame.Rect(margin, margin, w - margin * 2, h - margin * 2)
        pygame.draw.rect(surface, frame_color, frame, 1)

        phase_label = "Practice" if snap.phase is Phase.PRACTICE else "Timed Test"
        left = self._num_header_font.render(phase_label, True, text_muted)
        surface.blit(left, (frame.x + 12, frame.y + 10))

        title = self._num_header_font.render("Numerical Operations Test", True, text_main)
        surface.blit(title, title.get_rect(midtop=(frame.centerx, frame.y + 10)))

        if snap.time_remaining_s is not None:
            rem = int(round(snap.time_remaining_s))
            mm = rem // 60
            ss = rem % 60
            timer = self._num_header_font.render(f"{mm:02d}:{ss:02d}", True, text_muted)
            surface.blit(timer, timer.get_rect(topright=(frame.right - 12, frame.y + 10)))

        stats = self._small_font.render(
            f"Scored {snap.correct_scored}/{snap.attempted_scored}",
            True,
            accent,
        )
        surface.blit(stats, stats.get_rect(midtop=(frame.centerx, frame.y + 40)))

        prompt = str(snap.prompt).strip().split("\n", 1)[0]
        if prompt == "":
            prompt = "0 + 0 ="

        prompt_surface = None
        for f in self._num_prompt_fonts:
            candidate = f.render(prompt, True, text_main)
            if candidate.get_width() <= int(frame.w * 0.86):
                prompt_surface = candidate
                break
        if prompt_surface is None:
            prompt_surface = self._num_prompt_fonts[-1].render(prompt, True, text_main)

        prompt_y = frame.y + int(frame.h * 0.36)
        surface.blit(prompt_surface, prompt_surface.get_rect(center=(frame.centerx, prompt_y)))

    def _render_numerical_operations_answer_box(
        self,
        surface: pygame.Surface,
        snap: TestSnapshot,
    ) -> None:
        w, h = surface.get_size()
        text_main = (244, 248, 255)
        box_fill = (246, 250, 255)
        box_border = (142, 168, 210)
        input_color = (12, 26, 88)

        answer_label = self._small_font.render("Your Answer:", True, text_main)
        box_w = max(220, min(380, int(w * 0.42)))
        box_h = max(52, min(66, int(h * 0.12)))
        box_x = (w - box_w) // 2
        box_y = int(h * 0.62)
        surface.blit(answer_label, answer_label.get_rect(midbottom=(w // 2, box_y - 8)))

        box = pygame.Rect(box_x, box_y, box_w, box_h)
        pygame.draw.rect(surface, box_fill, box)
        pygame.draw.rect(surface, box_border, box, 2)

        caret = "|" if (pygame.time.get_ticks() // 500) % 2 == 0 else ""
        entry = self._num_input_font.render(self._input + caret, True, input_color)
        surface.blit(entry, (box.x + 12, box.y + max(2, (box.h - entry.get_height()) // 2)))

        hint = self._small_font.render(snap.input_hint, True, (194, 210, 236))
        surface.blit(hint, hint.get_rect(midtop=(w // 2, box.bottom + 10)))

    def _render_angles_bearings_screen(
        self,
        surface: pygame.Surface,
        snap: TestSnapshot,
        payload: AnglesBearingsDegreesPayload | None,
    ) -> None:
        w, h = surface.get_size()
        bg = (4, 12, 84)
        panel_bg = (8, 18, 104)
        header_bg = (18, 30, 118)
        border = (226, 236, 255)
        text_main = (238, 245, 255)
        text_muted = (188, 204, 228)
        active_bg = (244, 248, 255)
        active_text = (16, 32, 88)

        surface.fill(bg)

        margin = max(10, min(24, w // 34))
        frame = pygame.Rect(margin, margin, max(280, w - margin * 2), max(220, h - margin * 2))
        pygame.draw.rect(surface, panel_bg, frame)
        pygame.draw.rect(surface, border, frame, 2)

        header_h = max(40, min(56, h // 7))
        header = pygame.Rect(frame.x + 2, frame.y + 2, frame.w - 4, header_h)
        pygame.draw.rect(surface, header_bg, header)
        pygame.draw.line(surface, border, (header.x, header.bottom), (header.right, header.bottom), 1)

        phase_label = {
            Phase.INSTRUCTIONS: "Instructions",
            Phase.PRACTICE: "Practice",
            Phase.PRACTICE_DONE: "Practice Complete",
            Phase.SCORED: "Timed Test",
            Phase.RESULTS: "Results",
        }.get(snap.phase, "Task")
        surface.blit(
            self._tiny_font.render(phase_label, True, text_muted),
            (header.x + 12, header.y + (header.h - self._tiny_font.get_height()) // 2),
        )

        title = self._small_font.render("Angles, Bearings and Degrees", True, text_main)
        surface.blit(title, title.get_rect(midleft=(header.x + 145, header.centery)))

        stats_text = f"{snap.correct_scored}/{snap.attempted_scored}"
        stats = self._tiny_font.render(stats_text, True, text_muted)
        stats_rect = stats.get_rect(midright=(header.right - 12, header.centery))
        surface.blit(stats, stats_rect)
        stats_label = self._tiny_font.render("Scored", True, text_muted)
        surface.blit(stats_label, stats_label.get_rect(midright=(stats_rect.left - 6, header.centery)))

        if snap.time_remaining_s is not None:
            rem = int(round(snap.time_remaining_s))
            mm = rem // 60
            ss = rem % 60
            timer = self._small_font.render(f"{mm:02d}:{ss:02d}", True, text_main)
            surface.blit(timer, timer.get_rect(topright=(frame.right - 12, header.bottom + 8)))

        content = pygame.Rect(
            frame.x + max(14, w // 48),
            header.bottom + max(12, h // 36),
            frame.w - max(28, w // 24),
            frame.bottom - header.bottom - max(62, h // 9),
        )
        pygame.draw.rect(surface, (6, 13, 92), content)
        pygame.draw.rect(surface, (78, 102, 170), content, 1)

        if payload is not None and snap.phase in (Phase.PRACTICE, Phase.SCORED):
            stem_rect = pygame.Rect(content.x + 12, content.y + 10, content.w - 24, max(44, content.h // 6))
            self._draw_wrapped_text(
                surface,
                payload.stem,
                stem_rect,
                color=text_main,
                font=self._small_font,
                max_lines=2,
            )

            work = pygame.Rect(
                content.x + 12,
                stem_rect.bottom + 8,
                content.w - 24,
                content.bottom - stem_rect.bottom - 16,
            )
            left_w = max(240, min(int(work.w * 0.60), work.w - 180))
            diagram_rect = pygame.Rect(work.x, work.y, left_w, work.h)
            options_rect = pygame.Rect(diagram_rect.right + 10, work.y, work.w - left_w - 10, work.h)

            self._render_angles_bearings_question(surface, diagram_rect, payload)

            selected = self._math_choice
            if self._input in ("1", "2", "3", "4"):
                selected = int(self._input)

            pygame.draw.rect(surface, (9, 20, 106), options_rect)
            pygame.draw.rect(surface, (62, 84, 152), options_rect, 1)

            rows = max(1, len(payload.options))
            gap = 8
            row_h = max(36, min(58, (options_rect.h - gap * (rows + 1)) // rows))
            y = options_rect.y + gap
            for option in payload.options:
                row = pygame.Rect(options_rect.x + 8, y, options_rect.w - 16, row_h)
                is_selected = option.code == selected
                if is_selected:
                    pygame.draw.rect(surface, active_bg, row)
                    pygame.draw.rect(surface, (124, 148, 202), row, 2)
                else:
                    pygame.draw.rect(surface, (8, 18, 96), row)
                    pygame.draw.rect(surface, (62, 84, 152), row, 1)

                text_color = active_text if is_selected else text_main
                label = self._small_font.render(f"{option.code}. {option.text}", True, text_color)
                surface.blit(label, (row.x + 10, row.y + (row.h - label.get_height()) // 2))
                y += row_h + gap
        else:
            self._draw_wrapped_text(
                surface,
                str(snap.prompt),
                content.inflate(-24, -24),
                color=text_main,
                font=self._small_font,
                max_lines=12,
            )

        if snap.phase in (Phase.PRACTICE, Phase.SCORED):
            footer = "1-4: Select option  |  Up/Down: Move  |  Enter: Submit"
        elif snap.phase in (Phase.INSTRUCTIONS, Phase.PRACTICE_DONE):
            footer = "Enter: Continue  |  Esc/Backspace: Back"
        else:
            footer = "Enter: Return to Tests"
        footer_text = self._tiny_font.render(footer, True, text_muted)
        surface.blit(footer_text, footer_text.get_rect(midbottom=(frame.centerx, frame.bottom - 12)))

    def _render_angles_bearings_question(
        self,
        surface: pygame.Surface,
        panel: pygame.Rect,
        payload: AnglesBearingsDegreesPayload,
    ) -> None:
        pygame.draw.rect(surface, (16, 18, 64), panel)
        pygame.draw.rect(surface, (112, 134, 190), panel, 1)

        if payload.kind is AnglesBearingsQuestionKind.ANGLE_BETWEEN_LINES:
            self._draw_angle_trial(surface, panel, payload)
        else:
            self._draw_bearing_trial(surface, panel, payload)

    def _draw_angle_trial(
        self,
        surface: pygame.Surface,
        panel: pygame.Rect,
        payload: AnglesBearingsDegreesPayload,
    ) -> None:
        cx = panel.centerx
        cy = panel.centery
        radius = max(44, (min(panel.w, panel.h) // 2) - 20)

        pygame.draw.circle(surface, (35, 35, 48), (cx, cy), radius)
        pygame.draw.circle(surface, (90, 90, 110), (cx, cy), radius, 2)

        p1 = self._bearing_point(cx, cy, radius, payload.reference_bearing_deg)
        p2 = self._bearing_point(cx, cy, radius, payload.target_bearing_deg)
        pygame.draw.line(surface, (235, 235, 245), (cx, cy), p1, 4)
        pygame.draw.line(surface, (140, 220, 140), (cx, cy), p2, 4)
        pygame.draw.circle(surface, (235, 235, 245), (cx, cy), 6)

    def _draw_bearing_trial(
        self,
        surface: pygame.Surface,
        panel: pygame.Rect,
        payload: AnglesBearingsDegreesPayload,
    ) -> None:
        cx = panel.centerx
        cy = panel.centery
        radius = max(44, (min(panel.w, panel.h) // 2) - 20)

        pygame.draw.circle(surface, (35, 35, 48), (cx, cy), radius)
        pygame.draw.circle(surface, (90, 90, 110), (cx, cy), radius, 2)

        for bearing in (0, 90, 180, 270):
            end = self._bearing_point(cx, cy, radius, bearing)
            pygame.draw.line(surface, (70, 70, 85), (cx, cy), end, 1)

        for label, bearing in (("000", 0), ("090", 90), ("180", 180), ("270", 270)):
            tx, ty = self._bearing_point(cx, cy, radius + 24, bearing)
            surf = self._tiny_font.render(label, True, (150, 150, 165))
            rect = surf.get_rect(center=(tx, ty))
            surface.blit(surf, rect)

        target = self._bearing_point(cx, cy, radius - 8, payload.target_bearing_deg)
        pygame.draw.line(surface, (140, 220, 140), (cx, cy), target, 4)
        pygame.draw.circle(surface, (235, 235, 245), target, 6)
        lbl = self._small_font.render(payload.object_label, True, (235, 235, 245))
        surface.blit(lbl, (target[0] + 8, target[1] - 12))
        pygame.draw.circle(surface, (235, 235, 245), (cx, cy), 6)

    def _render_visual_search_question(
        self,
        surface: pygame.Surface,
        snap: TestSnapshot,
        payload: VisualSearchPayload,
    ) -> None:
        w, h = surface.get_size()
        bg = (2, 8, 114)
        frame_border = (232, 240, 255)
        panel_bg = (9, 20, 126)
        card_bg = (8, 14, 66)
        card_border = (150, 164, 198)
        text_main = (238, 245, 255)
        text_muted = (184, 200, 226)

        surface.fill(bg)
        margin = max(10, min(20, w // 40))
        frame = pygame.Rect(margin, margin, w - margin * 2, h - margin * 2)
        pygame.draw.rect(surface, frame_border, frame, 1)

        header_h = max(24, min(30, h // 18))
        header = pygame.Rect(frame.x + 2, frame.y + 2, frame.w - 4, header_h)
        pygame.draw.rect(surface, panel_bg, header)
        pygame.draw.line(
            surface,
            frame_border,
            (header.x, header.bottom),
            (header.right, header.bottom),
            1,
        )

        surface.blit(
            self._tiny_font.render("Target Recognition Test - Testing", True, text_main),
            (header.x + 10, header.y + 5),
        )
        if snap.time_remaining_s is not None:
            rem = int(round(snap.time_remaining_s))
            mm = rem // 60
            ss = rem % 60
            timer = self._tiny_font.render(f"{mm:02d}:{ss:02d}", True, text_main)
            surface.blit(timer, timer.get_rect(topright=(header.right - 10, header.y + 5)))

        work = pygame.Rect(frame.x + 10, header.bottom + 8, frame.w - 20, frame.h - header_h - 18)
        pygame.draw.rect(surface, (94, 99, 112), work)
        pygame.draw.rect(surface, (204, 214, 236), work, 1)

        top_row = pygame.Rect(work.x + 8, work.y + 8, work.w - 16, 30)
        pygame.draw.rect(surface, card_bg, top_row)
        pygame.draw.rect(surface, card_border, top_row, 1)
        surface.blit(self._tiny_font.render("Information", True, text_muted), (top_row.x + 10, top_row.y + 7))
        surface.blit(
            self._tiny_font.render(f"Scored {snap.correct_scored}/{snap.attempted_scored}", True, text_muted),
            (top_row.centerx - 42, top_row.y + 7),
        )
        surface.blit(
            self._tiny_font.render(f"Scan for target: {payload.target}", True, (210, 220, 244)),
            (top_row.x + 10, top_row.y + 18),
        )

        main = pygame.Rect(work.x + 8, top_row.bottom + 8, work.w - 16, work.h - 98)
        sys_w = max(180, min(260, main.w // 4))
        scan_rect = pygame.Rect(main.x, main.y, main.w - sys_w - 8, main.h)
        system_rect = pygame.Rect(scan_rect.right + 8, main.y, sys_w, main.h)
        pygame.draw.rect(surface, card_bg, scan_rect)
        pygame.draw.rect(surface, card_border, scan_rect, 1)
        pygame.draw.rect(surface, card_bg, system_rect)
        pygame.draw.rect(surface, card_border, system_rect, 1)

        surface.blit(self._tiny_font.render("Scan Panel", True, text_muted), (scan_rect.x + 8, scan_rect.y + 6))
        surface.blit(self._tiny_font.render("System Panel", True, text_muted), (system_rect.x + 8, system_rect.y + 6))

        grid_rect = scan_rect.inflate(-14, -28)
        pygame.draw.rect(surface, (20, 50, 44), grid_rect)
        pygame.draw.rect(surface, (82, 106, 140), grid_rect, 1)

        rows = max(1, int(payload.rows))
        cols = max(1, int(payload.cols))
        cell_size = min(max(20, grid_rect.w // cols), max(20, grid_rect.h // rows))
        grid_w = cols * cell_size
        grid_h = rows * cell_size
        start_x = grid_rect.x + max(0, (grid_rect.w - grid_w) // 2)
        start_y = grid_rect.y + max(0, (grid_rect.h - grid_h) // 2)
        token_font = pygame.font.Font(None, max(16, min(30, int(cell_size * 0.42))))
        code_font = pygame.font.Font(None, max(12, min(18, int(cell_size * 0.24))))

        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                token = payload.cells[idx] if idx < len(payload.cells) else ""
                code = payload.cell_codes[idx] if idx < len(payload.cell_codes) else 0
                cell = pygame.Rect(start_x + c * cell_size, start_y + r * cell_size, cell_size, cell_size)
                fill = self._visual_search_cell_color(payload.kind, token)
                pygame.draw.rect(surface, fill, cell)
                pygame.draw.rect(surface, (40, 76, 70), cell, 1)

                luminance = (fill[0] * 299 + fill[1] * 587 + fill[2] * 114) / 1000
                text_color = (20, 20, 24) if luminance > 145 else (235, 235, 245)
                code_surf = code_font.render(str(code), True, text_color)
                surface.blit(code_surf, (cell.x + 3, cell.y + 2))
                token_surface = token_font.render(str(token), True, text_color)
                token_rect = token_surface.get_rect(center=cell.center)
                surface.blit(token_surface, token_rect)

        # System list on the right (guide-like dense side panel).
        list_rect = system_rect.inflate(-10, -28)
        y = list_rect.y + 8
        step = 18
        max_lines = max(1, list_rect.h // step)
        for i in range(max_lines):
            idx = i if i < len(payload.cell_codes) else None
            if idx is None:
                line = ""
            else:
                code = payload.cell_codes[idx]
                tok = payload.cells[idx] if idx < len(payload.cells) else ""
                line = f"{code:>3}  {tok}"
            surface.blit(self._tiny_font.render(line, True, text_muted), (list_rect.x + 4, y))
            y += step

        # Bottom strip with scan target cards and answer entry.
        footer = pygame.Rect(work.x + 8, work.bottom - 42, work.w - 16, 34)
        pygame.draw.rect(surface, card_bg, footer)
        pygame.draw.rect(surface, card_border, footer, 1)
        surface.blit(self._tiny_font.render("Scan Target", True, text_muted), (footer.x + 8, footer.y + 10))

        target_chip = pygame.Rect(footer.x + 86, footer.y + 5, 72, 24)
        pygame.draw.rect(surface, (6, 42, 34), target_chip)
        pygame.draw.rect(surface, (108, 144, 124), target_chip, 1)
        t_surf = self._small_font.render(str(payload.target), True, text_main)
        surface.blit(t_surf, t_surf.get_rect(center=target_chip.center))

        answer_box = pygame.Rect(footer.centerx - 90, footer.y + 5, 180, 24)
        pygame.draw.rect(surface, (0, 0, 0), answer_box)
        pygame.draw.rect(surface, (132, 150, 190), answer_box, 1)
        answer_label = self._tiny_font.render("Answer", True, text_muted)
        surface.blit(answer_label, (answer_box.x - 46, answer_box.y + 5))
        caret = "|" if (pygame.time.get_ticks() // 500) % 2 == 0 else ""
        entry = self._tiny_font.render(self._input + caret, True, text_main)
        surface.blit(entry, (answer_box.x + 8, answer_box.y + 5))

        chips_start = footer.right - 142
        shown: list[str] = [str(payload.target)]
        for tok in payload.cells:
            if tok != payload.target and tok not in shown:
                shown.append(str(tok))
            if len(shown) >= 3:
                break
        while len(shown) < 3:
            shown.append("--")
        for i, tok in enumerate(shown[:3]):
            chip = pygame.Rect(chips_start + i * 44, footer.y + 5, 40, 24)
            pygame.draw.rect(surface, (6, 42, 34), chip)
            pygame.draw.rect(surface, (108, 144, 124), chip, 1)
            tok_s = self._tiny_font.render(tok, True, text_main)
            surface.blit(tok_s, tok_s.get_rect(center=chip.center))

    def _render_instrument_comprehension_screen(
        self,
        surface: pygame.Surface,
        snap: TestSnapshot,
        payload: InstrumentComprehensionPayload | None,
    ) -> None:
        w, h = surface.get_size()
        bg = (3, 11, 88)
        frame_bg = (8, 18, 104)
        header_bg = (16, 30, 126)
        content_bg = (70, 74, 84)
        border = (228, 238, 255)
        text_main = (236, 244, 255)
        text_muted = (186, 202, 228)
        card_bg = (44, 48, 58)
        card_border = (170, 184, 212)

        surface.fill(bg)

        margin = max(10, min(20, w // 40))
        frame = pygame.Rect(margin, margin, w - margin * 2, h - margin * 2)
        pygame.draw.rect(surface, frame_bg, frame)
        pygame.draw.rect(surface, border, frame, 1)

        header_h = max(28, min(36, h // 15))
        header = pygame.Rect(frame.x + 2, frame.y + 2, frame.w - 4, header_h)
        pygame.draw.rect(surface, header_bg, header)
        pygame.draw.line(surface, border, (header.x, header.bottom), (header.right, header.bottom), 1)

        phase_label = {
            Phase.INSTRUCTIONS: "Instructions",
            Phase.PRACTICE: "Practice",
            Phase.PRACTICE_DONE: "Practice Complete",
            Phase.SCORED: "Scored",
            Phase.RESULTS: "Results",
        }.get(snap.phase, "Task")

        surface.blit(
            self._tiny_font.render(f"Instrument Comprehension - {phase_label}", True, text_main),
            (header.x + 10, header.y + 6),
        )

        stats_text = f"{snap.correct_scored}/{snap.attempted_scored}"
        stats = self._tiny_font.render(stats_text, True, text_muted)
        stats_rect = stats.get_rect(midright=(header.right - 10, header.centery))
        surface.blit(stats, stats_rect)
        scored_label = self._tiny_font.render("Scored", True, text_muted)
        surface.blit(scored_label, scored_label.get_rect(midright=(stats_rect.left - 6, header.centery)))

        if snap.time_remaining_s is not None:
            rem = int(round(snap.time_remaining_s))
            timer = self._small_font.render(f"{rem // 60:02d}:{rem % 60:02d}", True, text_main)
            surface.blit(timer, timer.get_rect(topright=(frame.right - 14, header.bottom + 8)))

        work = pygame.Rect(frame.x + 8, header.bottom + 8, frame.w - 16, frame.bottom - header.bottom - 16)
        pygame.draw.rect(surface, content_bg, work)
        pygame.draw.rect(surface, border, work, 1)

        if snap.phase in (Phase.PRACTICE, Phase.SCORED) and payload is not None:
            answer_h = max(64, min(76, h // 7))
            question_rect = pygame.Rect(work.x + 8, work.y + 8, work.w - 16, work.h - answer_h - 16)
            self._render_instrument_comprehension_question(surface, snap, payload, question_rect)
            answer_rect = pygame.Rect(work.x + 8, question_rect.bottom + 8, work.w - 16, answer_h)
            self._render_instrument_answer_entry(surface, snap, answer_rect)
            return

        info_card = work.inflate(-16, -16)
        pygame.draw.rect(surface, card_bg, info_card)
        pygame.draw.rect(surface, card_border, info_card, 1)
        if snap.phase is Phase.INSTRUCTIONS:
            prompt = (
                "Instrument Comprehension\n\n"
                "Part 1: Read attitude and heading instruments, then choose the matching aircraft orientation.\n"
                "Part 2: Match between instrument panel and written flight description.\n\n"
                "Controls: Type 1-4, then press Enter.\n"
                "You will complete a short practice before the scored timed block."
            )
        elif snap.phase is Phase.PRACTICE_DONE:
            prompt = "Practice complete. Press Enter to start the scored timed block."
        else:
            prompt = str(snap.prompt)
        self._draw_wrapped_text(
            surface,
            prompt,
            info_card.inflate(-16, -14),
            color=text_main,
            font=self._small_font,
            max_lines=12,
        )

    def _render_instrument_answer_entry(
        self,
        surface: pygame.Surface,
        snap: TestSnapshot,
        rect: pygame.Rect,
    ) -> None:
        text_main = (236, 244, 255)
        text_muted = (184, 200, 224)
        card_bg = (44, 48, 58)
        card_border = (170, 184, 212)
        input_bg = (5, 10, 24)

        pygame.draw.rect(surface, card_bg, rect)
        pygame.draw.rect(surface, card_border, rect, 1)

        prompt = self._tiny_font.render("Answer (1-4):", True, text_muted)
        surface.blit(prompt, (rect.x + 10, rect.y + 10))

        entry_rect = pygame.Rect(rect.x + 114, rect.y + 8, max(100, rect.w // 3), 28)
        pygame.draw.rect(surface, input_bg, entry_rect)
        pygame.draw.rect(surface, card_border, entry_rect, 1)
        caret = "|" if (pygame.time.get_ticks() // 500) % 2 == 0 else ""
        entry = self._small_font.render(self._input + caret, True, text_main)
        surface.blit(entry, (entry_rect.x + 8, entry_rect.y + 4))

        hint = self._tiny_font.render(snap.input_hint, True, text_muted)
        surface.blit(hint, (rect.x + 10, rect.bottom - 24))

    def _render_instrument_comprehension_question(
        self,
        surface: pygame.Surface,
        snap: TestSnapshot,
        payload: InstrumentComprehensionPayload,
        panel: pygame.Rect,
    ) -> None:
        panel_bg = (54, 58, 68)
        panel_border = (170, 184, 212)
        prompt_bg = (36, 40, 50)
        text_main = (236, 244, 255)

        pygame.draw.rect(surface, panel_bg, panel)
        pygame.draw.rect(surface, panel_border, panel, 1)

        prompt_box = pygame.Rect(panel.x + 10, panel.y + 8, panel.w - 20, 34)
        pygame.draw.rect(surface, prompt_bg, prompt_box)
        pygame.draw.rect(surface, panel_border, prompt_box, 1)
        self._draw_wrapped_text(
            surface,
            str(snap.prompt),
            prompt_box.inflate(-10, -6),
            color=text_main,
            font=self._tiny_font,
            max_lines=2,
        )

        body = pygame.Rect(panel.x + 10, prompt_box.bottom + 8, panel.w - 20, panel.bottom - prompt_box.bottom - 16)
        if payload.kind is InstrumentComprehensionTrialKind.DESCRIPTION_TO_INSTRUMENTS:
            stimulus_h = max(60, int(body.h * 0.28))
        elif payload.kind is InstrumentComprehensionTrialKind.INSTRUMENTS_TO_DESCRIPTION:
            stimulus_h = max(80, int(body.h * 0.44))
        else:
            stimulus_h = max(74, int(body.h * 0.33))

        stimulus_rect = pygame.Rect(body.x, body.y, body.w, stimulus_h)
        options_rect = pygame.Rect(body.x, stimulus_rect.bottom + 8, body.w, body.bottom - stimulus_rect.bottom - 8)

        if payload.kind is InstrumentComprehensionTrialKind.DESCRIPTION_TO_INSTRUMENTS:
            pygame.draw.rect(surface, prompt_bg, stimulus_rect)
            pygame.draw.rect(surface, panel_border, stimulus_rect, 1)
            self._draw_wrapped_text(
                surface,
                payload.prompt_description,
                stimulus_rect.inflate(-14, -12),
                color=text_main,
                font=self._small_font,
                max_lines=4,
            )
        elif payload.kind is InstrumentComprehensionTrialKind.INSTRUMENTS_TO_AIRCRAFT:
            self._draw_orientation_prompt_dials(surface, stimulus_rect, payload.prompt_state)
        else:
            self._draw_instrument_cluster(surface, stimulus_rect, payload.prompt_state, compact=False)

        options = payload.options[:4]
        gap = 8
        if payload.kind is InstrumentComprehensionTrialKind.INSTRUMENTS_TO_DESCRIPTION:
            row_h = max(34, (options_rect.h - gap * 5) // 4)
            for idx, option in enumerate(options):
                card = pygame.Rect(
                    options_rect.x + gap,
                    options_rect.y + gap + idx * (row_h + gap),
                    options_rect.w - gap * 2,
                    row_h,
                )
                pygame.draw.rect(surface, prompt_bg, card)
                pygame.draw.rect(surface, panel_border, card, 1)

                badge = pygame.Rect(card.x + 6, card.y + 5, 26, card.h - 10)
                pygame.draw.rect(surface, (18, 30, 108), badge)
                pygame.draw.rect(surface, panel_border, badge, 1)
                num = self._small_font.render(str(option.code), True, text_main)
                surface.blit(num, num.get_rect(center=badge.center))

                self._draw_wrapped_text(
                    surface,
                    option.description,
                    pygame.Rect(card.x + 40, card.y + 6, card.w - 46, card.h - 10),
                    color=text_main,
                    font=self._tiny_font,
                    max_lines=2,
                )
            return

        rows = 2
        cols = 2
        card_w = (options_rect.w - gap * (cols + 1)) // cols
        card_h = (options_rect.h - gap * (rows + 1)) // rows
        for idx, option in enumerate(options):
            row = idx // cols
            col = idx % cols
            card = pygame.Rect(
                options_rect.x + gap + col * (card_w + gap),
                options_rect.y + gap + row * (card_h + gap),
                card_w,
                card_h,
            )
            pygame.draw.rect(surface, prompt_bg, card)
            pygame.draw.rect(surface, panel_border, card, 1)

            badge = pygame.Rect(card.x + 6, card.y + 5, 26, 18)
            pygame.draw.rect(surface, (18, 30, 108), badge)
            pygame.draw.rect(surface, panel_border, badge, 1)
            label = self._tiny_font.render(str(option.code), True, text_main)
            surface.blit(label, label.get_rect(center=badge.center))

            inner = pygame.Rect(card.x + 6, card.y + 26, card.w - 12, card.h - 32)
            if payload.kind is InstrumentComprehensionTrialKind.DESCRIPTION_TO_INSTRUMENTS:
                self._draw_instrument_cluster(surface, inner, option.state, compact=True)
            else:
                self._draw_aircraft_orientation_card(surface, inner, option.state)

    def _draw_orientation_prompt_dials(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        state: InstrumentState,
    ) -> None:
        panel_bg = (36, 40, 50)
        panel_border = (170, 184, 212)
        text_muted = (184, 200, 224)

        pygame.draw.rect(surface, panel_bg, rect)
        pygame.draw.rect(surface, panel_border, rect, 1)

        gap = 12
        dial_size = max(42, min(rect.h - 20, (rect.w - gap * 3) // 2))
        total_w = dial_size * 2 + gap
        start_x = rect.x + (rect.w - total_w) // 2
        y = rect.y + (rect.h - dial_size) // 2
        att_rect = pygame.Rect(start_x, y, dial_size, dial_size)
        hdg_rect = pygame.Rect(att_rect.right + gap, y, dial_size, dial_size)

        self._draw_attitude_dial(surface, att_rect, bank_deg=state.bank_deg, pitch_deg=state.pitch_deg)
        self._draw_heading_dial(surface, hdg_rect, state.heading_deg)

        att_label = self._tiny_font.render("ATTITUDE", True, text_muted)
        hdg_label = self._tiny_font.render("HEADING", True, text_muted)
        surface.blit(att_label, att_label.get_rect(midbottom=(att_rect.centerx, rect.bottom - 3)))
        surface.blit(hdg_label, hdg_label.get_rect(midbottom=(hdg_rect.centerx, rect.bottom - 3)))

    def _draw_wrapped_text(
        self,
        surface: pygame.Surface,
        text: str,
        rect: pygame.Rect,
        *,
        color: tuple[int, int, int],
        font: pygame.font.Font,
        max_lines: int,
    ) -> None:
        words = str(text).split()
        lines: list[str] = []
        cur = ""
        for word in words:
            trial = word if cur == "" else f"{cur} {word}"
            if font.size(trial)[0] <= rect.w:
                cur = trial
                continue
            if cur:
                lines.append(cur)
            cur = word
        if cur:
            lines.append(cur)

        y = rect.y
        line_h = font.get_linesize() + 2
        for line in lines[: max(0, max_lines)]:
            to_draw = line
            if font.size(to_draw)[0] > rect.w:
                while to_draw and font.size(f"{to_draw}...")[0] > rect.w:
                    to_draw = to_draw[:-1]
                to_draw = f"{to_draw}..." if to_draw else "..."
            surface.blit(font.render(to_draw, True, color), (rect.x, y))
            y += line_h

    def _draw_instrument_cluster(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        state: InstrumentState,
        *,
        compact: bool = False,
    ) -> None:
        panel_bg = (36, 40, 50)
        panel_border = (170, 184, 212)

        pygame.draw.rect(surface, panel_bg, rect)
        pygame.draw.rect(surface, panel_border, rect, 1)
        inner = rect.inflate(-8, -8)

        gap = 5 if compact else 8
        cols = 3
        rows = 2
        cell_w = max(46, (inner.w - gap * (cols - 1)) // cols)
        cell_h = max(46, (inner.h - gap * (rows - 1)) // rows)

        cells: list[pygame.Rect] = []
        for row in range(rows):
            for col in range(cols):
                x = inner.x + col * (cell_w + gap)
                y = inner.y + row * (cell_h + gap)
                cells.append(pygame.Rect(x, y, cell_w, cell_h))

        self._draw_speed_dial(surface, cells[0], state.speed_kts)
        self._draw_attitude_dial(surface, cells[1], bank_deg=state.bank_deg, pitch_deg=state.pitch_deg)
        self._draw_heading_dial(surface, cells[2], state.heading_deg)
        self._draw_altimeter_dial(surface, cells[3], state.altitude_ft)
        self._draw_vertical_dial(surface, cells[4], state.vertical_rate_fpm, state.slip)
        self._draw_slip_indicator(surface, cells[5], bank_deg=state.bank_deg, slip=state.slip)

    def _draw_speed_dial(self, surface: pygame.Surface, rect: pygame.Rect, speed_kts: int) -> None:
        self._draw_scalar_dial(surface, rect, "KNOTS", int(speed_kts), vmin=80, vmax=360)

    def _draw_altimeter_dial(self, surface: pygame.Surface, rect: pygame.Rect, altitude_ft: int) -> None:
        self._draw_scalar_dial(surface, rect, "ALT", int(altitude_ft), vmin=0, vmax=10000)

    def _draw_vertical_dial(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        vertical_rate_fpm: int,
        slip: int,
    ) -> None:
        _ = slip
        self._draw_scalar_dial(surface, rect, "V/S", int(vertical_rate_fpm), vmin=-2000, vmax=2000)

    def _draw_scalar_dial(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        title: str,
        value: int,
        *,
        vmin: int,
        vmax: int,
    ) -> None:
        dial_rect, cx, cy, _, face_r = self._dial_geometry(rect)
        size = dial_rect.w

        def build_base() -> pygame.Surface:
            base = pygame.Surface((size, size), pygame.SRCALPHA)
            c = size // 2
            outer_r = size // 2 - 1
            inner_ring = max(10, outer_r - 6)

            pygame.draw.circle(base, (198, 204, 214), (c, c), outer_r)
            pygame.draw.circle(base, (86, 96, 116), (c, c), outer_r, 2)
            pygame.draw.circle(base, (0, 0, 0), (c, c), inner_ring)
            pygame.draw.circle(base, (32, 38, 52), (c, c), inner_ring, 1)

            for idx in range(36):
                ang = math.radians(-130.0 + (260.0 / 35.0) * idx)
                outer = inner_ring - 1
                inner = inner_ring - (9 if idx % 6 == 0 else 5)
                ox = int(round(c + math.cos(ang) * outer))
                oy = int(round(c + math.sin(ang) * outer))
                ix = int(round(c + math.cos(ang) * inner))
                iy = int(round(c + math.sin(ang) * inner))
                pygame.draw.line(base, (188, 196, 212), (ix, iy), (ox, oy), 1)

            if size >= 64:
                for idx in range(6):
                    t = idx / 5.0
                    raw_v = int(round(float(vmin) + float(vmax - vmin) * t))
                    label = self._format_scalar_tick(title, raw_v)
                    ang = math.radians(-130.0 + 260.0 * t)
                    tx = int(round(c + math.cos(ang) * (inner_ring - 16)))
                    ty = int(round(c + math.sin(ang) * (inner_ring - 16)))
                    label_surf = self._tiny_font.render(label, True, (226, 232, 244))
                    base.blit(label_surf, label_surf.get_rect(center=(tx, ty)))

            title_surf = self._tiny_font.render(title, True, (166, 180, 206))
            base.blit(title_surf, title_surf.get_rect(center=(c, c - inner_ring + 12)))
            return base

        key = ("scalar_base", size, title, int(vmin), int(vmax))
        surface.blit(self._get_instrument_sprite(key, build_base), dial_rect.topleft)

        t = (float(value) - float(vmin)) / max(1.0, float(vmax - vmin))
        t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
        ang = math.radians(-130.0 + 260.0 * t)
        needle_len = max(6, face_r - 10)
        tip = (
            int(round(cx + math.cos(ang) * needle_len)),
            int(round(cy + math.sin(ang) * needle_len)),
        )
        tail = (
            int(round(cx - math.cos(ang) * max(5, face_r * 0.16))),
            int(round(cy - math.sin(ang) * max(5, face_r * 0.16))),
        )
        pygame.draw.line(surface, (246, 248, 252), tail, tip, 4 if size >= 84 else 3)
        pygame.draw.circle(surface, (10, 10, 12), (cx, cy), max(2, size // 18))
        pygame.draw.circle(surface, (246, 248, 252), (cx, cy), max(1, size // 24))


    def _draw_attitude_dial(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        *,
        bank_deg: int,
        pitch_deg: int,
    ) -> None:
        dial_rect, _, _, _, face_r = self._dial_geometry(rect)
        size = dial_rect.w

        # Dynamic horizon/pitch ladder layer.
        horizon_side = max(size * 3, 96)
        horizon = pygame.Surface((horizon_side, horizon_side), pygame.SRCALPHA)
        hc = horizon_side // 2
        pitch = max(-20.0, min(20.0, float(pitch_deg)))
        horizon_y = hc + int(round((pitch / 20.0) * (face_r * 0.90)))
        sky = (30, 176, 238)
        ground = (174, 108, 36)
        pygame.draw.rect(horizon, sky, pygame.Rect(0, 0, horizon_side, horizon_y))
        pygame.draw.rect(
            horizon,
            ground,
            pygame.Rect(0, horizon_y, horizon_side, max(0, horizon_side - horizon_y)),
        )
        pygame.draw.line(horizon, (246, 246, 248), (0, horizon_y), (horizon_side, horizon_y), 3)

        for mark in (-15, -10, -5, 5, 10, 15):
            y = horizon_y - int(round((mark / 20.0) * (face_r * 0.90)))
            half = int(round(face_r * (0.45 if mark % 10 == 0 else 0.30)))
            pygame.draw.line(horizon, (242, 244, 248), (hc - half, y), (hc + half, y), 2)

        rotated = pygame.transform.rotozoom(horizon, -float(bank_deg), 1.0)
        self._draw_circular_layer(surface, dial_rect, rotated, radius=face_r - 1)

        def build_overlay() -> pygame.Surface:
            overlay = pygame.Surface((size, size), pygame.SRCALPHA)
            c = size // 2
            outer_r = size // 2 - 1
            inner_ring = max(10, outer_r - 6)
            rim_w = max(3, outer_r - inner_ring + 1)

            # Draw only the bezel/rim so the dynamic horizon remains visible.
            pygame.draw.circle(overlay, (198, 204, 214), (c, c), outer_r, rim_w)
            pygame.draw.circle(overlay, (86, 96, 116), (c, c), outer_r, 2)
            pygame.draw.circle(overlay, (24, 30, 44), (c, c), inner_ring, 2)

            for deg in (-60, -45, -30, -20, -10, 0, 10, 20, 30, 45, 60):
                rad = math.radians(float(deg - 90))
                outer = inner_ring - 1
                inner = inner_ring - (9 if deg % 30 == 0 else 6)
                ox = int(round(c + math.cos(rad) * outer))
                oy = int(round(c + math.sin(rad) * outer))
                ix = int(round(c + math.cos(rad) * inner))
                iy = int(round(c + math.sin(rad) * inner))
                pygame.draw.line(overlay, (204, 214, 230), (ix, iy), (ox, oy), 1)

            # Fixed airplane cue.
            wing_y = c + int(round(inner_ring * 0.10))
            wing_half = max(8, int(round(inner_ring * 0.35)))
            pygame.draw.line(overlay, (248, 250, 255), (c - wing_half, wing_y), (c + wing_half, wing_y), 3)
            pygame.draw.line(overlay, (248, 250, 255), (c, wing_y - 6), (c, wing_y + 6), 2)

            cue = [
                (c - int(inner_ring * 0.42), c + int(inner_ring * 0.10)),
                (c - int(inner_ring * 0.26), c + int(inner_ring * 0.23)),
                (c - int(inner_ring * 0.14), c + int(inner_ring * 0.08)),
                (c, c + int(inner_ring * 0.22)),
                (c + int(inner_ring * 0.14), c + int(inner_ring * 0.08)),
                (c + int(inner_ring * 0.26), c + int(inner_ring * 0.23)),
                (c + int(inner_ring * 0.42), c + int(inner_ring * 0.10)),
            ]
            pygame.draw.lines(overlay, (242, 244, 248), False, cue, 2)
            return overlay

        overlay_key = ("attitude_overlay", size)
        surface.blit(self._get_instrument_sprite(overlay_key, build_overlay), dial_rect.topleft)

    def _draw_heading_dial(self, surface: pygame.Surface, rect: pygame.Rect, heading_deg: int) -> None:
        dial_rect, cx, cy, _, face_r = self._dial_geometry(rect)
        size = dial_rect.w

        def build_base() -> pygame.Surface:
            base = pygame.Surface((size, size), pygame.SRCALPHA)
            c = size // 2
            outer_r = size // 2 - 1
            inner_ring = max(10, outer_r - 6)

            pygame.draw.circle(base, (198, 204, 214), (c, c), outer_r)
            pygame.draw.circle(base, (86, 96, 116), (c, c), outer_r, 2)
            pygame.draw.circle(base, (0, 0, 0), (c, c), inner_ring)
            pygame.draw.circle(base, (32, 38, 52), (c, c), inner_ring, 1)

            for deg in range(0, 360, 15):
                rad = math.radians(float(deg - 90))
                outer = inner_ring - 1
                inner = inner_ring - (10 if deg % 90 == 0 else 6)
                ox = int(round(c + math.cos(rad) * outer))
                oy = int(round(c + math.sin(rad) * outer))
                ix = int(round(c + math.cos(rad) * inner))
                iy = int(round(c + math.sin(rad) * inner))
                pygame.draw.line(base, (192, 202, 220), (ix, iy), (ox, oy), 1)

            for label, deg in (("N", 0), ("E", 90), ("S", 180), ("W", 270)):
                rad = math.radians(float(deg - 90))
                tx = int(round(c + math.cos(rad) * (inner_ring - 18)))
                ty = int(round(c + math.sin(rad) * (inner_ring - 18)))
                label_font = self._small_font if size >= 88 else self._tiny_font
                surf = label_font.render(label, True, (236, 244, 255))
                base.blit(surf, surf.get_rect(center=(tx, ty)))
            return base

        base_key = ("heading_base", size)
        surface.blit(self._get_instrument_sprite(base_key, build_base), dial_rect.topleft)

        def build_aircraft_icon() -> pygame.Surface:
            icon = pygame.Surface((size, size), pygame.SRCALPHA)
            c = size // 2
            wing = max(7, int(round(face_r * 0.40)))
            body = max(12, int(round(face_r * 0.82)))
            aircraft = [
                (c, c - body),
                (c + 4, c - body + 8),
                (c + 4, c - 4),
                (c + wing, c - 1),
                (c + wing, c + 3),
                (c + 4, c + 2),
                (c + 4, c + body - 10),
                (c + 9, c + body - 2),
                (c + 9, c + body + 2),
                (c + 3, c + body),
                (c - 3, c + body),
                (c - 9, c + body + 2),
                (c - 9, c + body - 2),
                (c - 4, c + body - 10),
                (c - 4, c + 2),
                (c - wing, c + 3),
                (c - wing, c - 1),
                (c - 4, c - 4),
                (c - 4, c - body + 8),
            ]
            pygame.draw.polygon(icon, (245, 248, 255), aircraft)
            arrow = [(c, c - body - 12), (c - 5, c - body - 2), (c + 5, c - body - 2)]
            pygame.draw.polygon(icon, (245, 248, 255), arrow)
            return icon

        icon_key = ("heading_icon", size)
        icon = self._get_instrument_sprite(icon_key, build_aircraft_icon)
        rotation = -float(int(heading_deg) % 360)
        rot_icon = pygame.transform.rotozoom(icon, rotation, 1.0)
        surface.blit(rot_icon, rot_icon.get_rect(center=(cx, cy)))


    def _draw_slip_indicator(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        *,
        bank_deg: int,
        slip: int,
    ) -> None:
        dial_rect, cx, cy, _, face_r = self._dial_geometry(rect)
        size = dial_rect.w

        def build_base() -> pygame.Surface:
            base = pygame.Surface((size, size), pygame.SRCALPHA)
            c = size // 2
            outer_r = size // 2 - 1
            inner_ring = max(10, outer_r - 6)

            pygame.draw.circle(base, (198, 204, 214), (c, c), outer_r)
            pygame.draw.circle(base, (86, 96, 116), (c, c), outer_r, 2)
            pygame.draw.circle(base, (0, 0, 0), (c, c), inner_ring)
            pygame.draw.circle(base, (32, 38, 52), (c, c), inner_ring, 1)

            for deg in (-60, -45, -30, -15, 0, 15, 30, 45, 60):
                rad = math.radians(float(deg - 90))
                outer = inner_ring - 1
                inner = inner_ring - (9 if deg % 30 == 0 else 6)
                ox = int(round(c + math.cos(rad) * outer))
                oy = int(round(c + math.sin(rad) * outer))
                ix = int(round(c + math.cos(rad) * inner))
                iy = int(round(c + math.sin(rad) * inner))
                pygame.draw.line(base, (192, 202, 220), (ix, iy), (ox, oy), 1)

            left = self._tiny_font.render("L", True, (236, 244, 255))
            right = self._tiny_font.render("R", True, (236, 244, 255))
            base.blit(left, left.get_rect(center=(c - int(inner_ring * 0.55), c - int(inner_ring * 0.22))))
            base.blit(right, right.get_rect(center=(c + int(inner_ring * 0.55), c - int(inner_ring * 0.22))))
            return base

        base_key = ("slip_base", size)
        surface.blit(self._get_instrument_sprite(base_key, build_base), dial_rect.topleft)

        bank_norm = max(-1.0, min(1.0, float(bank_deg) / 35.0))
        angle = math.radians(-90.0 + bank_norm * 58.0)
        pointer_len = max(6, face_r - 8)
        tip = (
            int(round(cx + math.cos(angle) * pointer_len)),
            int(round(cy + math.sin(angle) * pointer_len)),
        )
        pygame.draw.line(surface, (246, 248, 252), (cx, cy), tip, 4 if size >= 84 else 3)
        pygame.draw.circle(surface, (246, 248, 252), (cx, cy), max(2, size // 24))

        tube_w = min(dial_rect.w - 8, max(14, int(round(face_r * 1.40))))
        tube_h = max(7, int(round(face_r * 0.46)))
        track = pygame.Rect(0, 0, tube_w, tube_h)
        track.centerx = cx
        track.centery = cy + int(round(face_r * 0.62))
        pygame.draw.rect(surface, (8, 10, 16), track)
        pygame.draw.rect(surface, (130, 146, 176), track, 1)
        pygame.draw.line(surface, (172, 184, 208), (track.centerx, track.y), (track.centerx, track.bottom), 1)

        ball_r = max(2, min(4, tube_h // 2 - 1))
        max_offset = max(1, track.w // 2 - ball_r - 2)
        offset = int(max(-1, min(1, int(slip))) * max_offset)
        ball_center = (track.centerx + offset, track.centery)
        pygame.draw.circle(surface, (244, 248, 255), ball_center, ball_r)
        pygame.draw.circle(surface, (120, 132, 156), ball_center, 1)

    def _dial_geometry(self, rect: pygame.Rect) -> tuple[pygame.Rect, int, int, int, int]:
        size = max(24, min(rect.w, rect.h))
        dial_rect = pygame.Rect(0, 0, size, size)
        dial_rect.center = rect.center
        cx = dial_rect.centerx
        cy = dial_rect.centery
        outer_r = size // 2 - 1
        face_r = max(8, outer_r - 7)
        return dial_rect, cx, cy, outer_r, face_r

    def _format_scalar_tick(self, title: str, value: int) -> str:
        if title == "ALT":
            return str((abs(int(value)) // 1000) % 10)
        if title == "V/S":
            v = int(round(float(value) / 1000.0))
            return "0" if v == 0 else f"{v:+d}"
        return str(int(value))

    def _get_instrument_sprite(
        self,
        key: tuple[object, ...],
        builder: Callable[[], pygame.Surface],
    ) -> pygame.Surface:
        cached = self._instrument_sprite_cache.get(key)
        if cached is not None:
            return cached
        built = builder()
        self._instrument_sprite_cache[key] = built
        return built

    def _draw_circular_layer(
        self,
        surface: pygame.Surface,
        dial_rect: pygame.Rect,
        layer: pygame.Surface,
        *,
        radius: int,
    ) -> None:
        face = pygame.Surface(dial_rect.size, pygame.SRCALPHA)
        face.blit(layer, layer.get_rect(center=(dial_rect.w // 2, dial_rect.h // 2)))
        mask = pygame.Surface(dial_rect.size, pygame.SRCALPHA)
        pygame.draw.circle(mask, (255, 255, 255, 255), (dial_rect.w // 2, dial_rect.h // 2), max(1, radius))
        face.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        surface.blit(face, dial_rect.topleft)

    def _draw_aircraft_orientation_card(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        state: InstrumentState,
    ) -> None:
        panel_bg = (30, 34, 44)
        border = (170, 184, 212)

        pygame.draw.rect(surface, panel_bg, rect)
        pitch = max(-20.0, min(20.0, float(state.pitch_deg)))
        bank = max(-45.0, min(45.0, float(state.bank_deg)))
        heading = float(int(state.heading_deg) % 360)

        pitch_norm = pitch / 20.0
        bank_norm = bank / 45.0
        horizon_y = rect.centery + int(round(-pitch_norm * rect.h * 0.22))
        slope = int(round(bank_norm * rect.h * 0.28))
        left = (rect.x - 2, horizon_y + slope)
        right = (rect.right + 2, horizon_y - slope)
        sky_poly = [rect.topleft, rect.topright, right, left]
        ground_poly = [left, right, rect.bottomright, rect.bottomleft]
        pygame.draw.polygon(surface, (36, 150, 232), sky_poly)
        pygame.draw.polygon(surface, (24, 102, 60), ground_poly)
        pygame.draw.rect(surface, border, rect, 1)

        cx = rect.centerx
        cy = rect.centery + int(round(rect.h * 0.05))
        scale = max(9.0, float(min(rect.w, rect.h)) * 0.16)

        fuselage = [
            (0.00, 2.90, 0.00),
            (0.40, 1.90, 0.24),
            (0.52, -1.95, 0.16),
            (0.00, -2.70, 0.10),
            (-0.52, -1.95, 0.16),
            (-0.40, 1.90, 0.24),
        ]
        wings = [
            (-2.70, 0.10, 0.02),
            (-0.60, 0.52, 0.10),
            (0.60, 0.52, 0.10),
            (2.70, 0.10, 0.02),
            (1.85, -0.35, -0.04),
            (-1.85, -0.35, -0.04),
        ]
        tailplane = [
            (-1.20, -1.98, 0.06),
            (-0.22, -1.76, 0.10),
            (0.22, -1.76, 0.10),
            (1.20, -1.98, 0.06),
            (0.65, -2.34, 0.02),
            (-0.65, -2.34, 0.02),
        ]
        fin = [
            (0.00, -2.20, 0.10),
            (0.00, -1.44, 1.12),
            (0.24, -2.02, 0.28),
            (-0.24, -2.02, 0.28),
        ]
        canopy = [(-0.20, 1.78, 0.34), (0.20, 1.78, 0.34), (0.16, 1.18, 0.44), (-0.16, 1.18, 0.44)]

        parts: list[
            tuple[str, list[tuple[float, float, float]], tuple[int, int, int], tuple[int, int, int]]
        ] = [
            ("wings", wings, (214, 52, 56), (236, 86, 90)),
            ("tailplane", tailplane, (206, 50, 54), (232, 82, 86)),
            ("fuselage", fuselage, (220, 58, 62), (244, 96, 100)),
            ("fin", fin, (198, 44, 50), (226, 78, 84)),
            ("canopy", canopy, (156, 226, 232), (104, 188, 208)),
        ]

        projected_parts: list[
            tuple[float, list[tuple[int, int]], tuple[int, int, int], tuple[int, int, int]]
        ] = []
        for _, local_pts, fill_color, edge_color in parts:
            depth_total = 0.0
            pts_2d: list[tuple[int, int]] = []
            for pt in local_pts:
                rot = self._rotate_aircraft_point(
                    pt,
                    heading_deg=heading,
                    pitch_deg=pitch,
                    bank_deg=bank,
                )
                sx, sy, depth = self._project_aircraft_point(rot, cx=cx, cy=cy, scale=scale)
                pts_2d.append((sx, sy))
                depth_total += depth
            avg_depth = depth_total / float(max(1, len(local_pts)))
            projected_parts.append((avg_depth, pts_2d, fill_color, edge_color))

        # Draw far geometry first.
        for _, pts_2d, fill_color, edge_color in sorted(projected_parts, key=lambda t: t[0], reverse=True):
            pygame.draw.polygon(surface, fill_color, pts_2d)
            pygame.draw.polygon(surface, edge_color, pts_2d, 1)

        def project_point(pt: tuple[float, float, float]) -> tuple[int, int]:
            rot = self._rotate_aircraft_point(
                pt,
                heading_deg=heading,
                pitch_deg=pitch,
                bank_deg=bank,
            )
            sx, sy, _ = self._project_aircraft_point(rot, cx=cx, cy=cy, scale=scale)
            return sx, sy

        nose = project_point((0.0, 2.95, 0.0))
        tail = project_point((0.0, -2.58, 0.08))
        pygame.draw.line(surface, (255, 238, 230), tail, nose, 1)
        pygame.draw.circle(surface, (240, 244, 255), nose, max(1, int(scale * 0.10)))

        gear_pairs = [
            ((0.00, 1.70, -0.18), (0.00, 1.70, -0.95)),
            ((-0.72, -0.20, -0.10), (-0.72, -0.20, -0.90)),
            ((0.72, -0.20, -0.10), (0.72, -0.20, -0.90)),
        ]
        for start, end in gear_pairs:
            p0 = project_point(start)
            p1 = project_point(end)
            pygame.draw.line(surface, (24, 24, 26), p0, p1, 2)
            pygame.draw.circle(surface, (16, 16, 18), p1, max(1, int(scale * 0.08)))

    def _rotate_aircraft_point(
        self,
        point: tuple[float, float, float],
        *,
        heading_deg: float,
        pitch_deg: float,
        bank_deg: float,
    ) -> tuple[float, float, float]:
        x, y, z = point

        # Roll around the aircraft longitudinal axis (forward/y).
        roll = math.radians(bank_deg)
        cos_r = math.cos(roll)
        sin_r = math.sin(roll)
        x1 = x * cos_r + z * sin_r
        y1 = y
        z1 = -x * sin_r + z * cos_r

        # Pitch around right axis (x).
        pitch = math.radians(pitch_deg)
        cos_p = math.cos(pitch)
        sin_p = math.sin(pitch)
        x2 = x1
        y2 = y1 * cos_p - z1 * sin_p
        z2 = y1 * sin_p + z1 * cos_p

        # Yaw around up axis (z). Heading is clockwise from North.
        yaw = math.radians(-heading_deg)
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        x3 = x2 * cos_y - y2 * sin_y
        y3 = x2 * sin_y + y2 * cos_y
        z3 = z2
        return x3, y3, z3

    def _project_aircraft_point(
        self,
        point: tuple[float, float, float],
        *,
        cx: int,
        cy: int,
        scale: float,
    ) -> tuple[int, int, float]:
        x, y, z = point
        sx = int(round(cx + (x + y * 0.10) * scale))
        sy = int(round(cy - (z + y * 0.34) * scale))
        return sx, sy, y

    def _visual_search_cell_color(
        self, kind: VisualSearchTaskKind, token: str
    ) -> tuple[int, int, int]:
        _ = (kind, token)
        # Keep scan field visually consistent across token types.
        return (36, 78, 70)

    def _color_pattern_cell_color(self, token: str) -> tuple[int, int, int]:
        palette = {
            "R": (200, 70, 70),
            "G": (70, 180, 100),
            "B": (80, 110, 200),
            "Y": (210, 190, 80),
            "W": (220, 220, 220),
        }
        t = str(token)
        c1 = palette.get(t[0], (90, 90, 110)) if len(t) >= 1 else (90, 90, 110)
        c2 = palette.get(t[1], c1) if len(t) >= 2 else c1
        return ((c1[0] + c2[0]) // 2, (c1[1] + c2[1]) // 2, (c1[2] + c2[2]) // 2)

    def _bearing_point(self, cx: int, cy: int, radius: int, bearing_deg: int) -> tuple[int, int]:
        rad = math.radians(float(bearing_deg))
        x = int(round(cx + math.sin(rad) * radius))
        y = int(round(cy - math.cos(rad) * radius))
        return x, y

    def _render_airborne_question(
        self, surface: pygame.Surface, snap: TestSnapshot, scenario: AirborneScenario
    ) -> None:
        w, h = surface.get_size()

        bg = (2, 8, 114)
        frame_border = (232, 240, 255)
        panel_fill = (96, 101, 111)
        panel_border = (200, 208, 226)
        card_fill = (64, 67, 74)
        card_border = (172, 182, 204)
        text_main = (238, 245, 255)
        text_muted = (196, 207, 228)
        text_dim = (155, 168, 194)

        surface.fill(bg)

        margin = max(10, min(20, w // 40))
        frame = pygame.Rect(margin, margin, w - margin * 2, h - margin * 2)
        pygame.draw.rect(surface, frame_border, frame, 1)

        header_h = max(24, min(30, h // 18))
        header = pygame.Rect(frame.x + 2, frame.y + 2, frame.w - 4, header_h)
        pygame.draw.rect(surface, (9, 20, 123), header)
        pygame.draw.line(
            surface,
            frame_border,
            (header.x, header.bottom),
            (header.right, header.bottom),
            1,
        )

        surface.blit(
            self._tiny_font.render("Airborne Numerical Test - Testing", True, text_main),
            (header.x + 10, header.y + 5),
        )
        phase_label = "Practice" if snap.phase is Phase.PRACTICE else "Scored"
        phase_surf = self._tiny_font.render(phase_label, True, text_muted)
        surface.blit(phase_surf, phase_surf.get_rect(topright=(header.right - 10, header.y + 5)))

        work = pygame.Rect(frame.x + 10, header.bottom + 8, frame.w - 20, frame.h - header_h - 18)
        pygame.draw.rect(surface, panel_fill, work)
        pygame.draw.rect(surface, panel_border, work, 1)

        side_w = max(210, min(300, work.w // 3))
        left = pygame.Rect(work.x + 8, work.y + 8, work.w - side_w - 24, work.h - 16)
        right = pygame.Rect(left.right + 8, work.y + 8, side_w, work.h - 16)

        map_h = int(left.h * 0.60)
        map_rect = pygame.Rect(left.x, left.y, left.w, map_h)
        table_rect = pygame.Rect(left.x, map_rect.bottom + 8, left.w, left.h - map_h - 8)

        pygame.draw.rect(surface, card_fill, map_rect)
        pygame.draw.rect(surface, card_border, map_rect, 1)
        pygame.draw.rect(surface, card_fill, table_rect)
        pygame.draw.rect(surface, card_border, table_rect, 1)

        self._draw_airborne_map(surface, map_rect, scenario)
        self._draw_airborne_table(surface, table_rect, scenario)

        pygame.draw.rect(surface, card_fill, right)
        pygame.draw.rect(surface, card_border, right, 1)

        prompt_text = str(snap.prompt).split("\n", 1)[0]
        self._draw_wrapped_text(
            surface,
            prompt_text,
            pygame.Rect(right.x + 10, right.y + 8, right.w - 20, 66),
            color=text_main,
            font=self._small_font,
            max_lines=2,
        )
        surface.blit(
            self._tiny_font.render("Enter HHMM (4 digits)", True, text_dim),
            (right.x + 10, right.y + 72),
        )

        info_lines = [
            f"Start: {scenario.start_time_hhmm}",
            f"Speed: {scenario.speed_value} {scenario.speed_unit}",
            f"Fuel burn: {scenario.fuel_burn_per_hr} L/hr",
            f"Parcel: {scenario.parcel_weight_kg} kg",
            f"Target: {scenario.target_label}",
        ]
        y = right.y + 98
        for line in info_lines:
            surface.blit(self._tiny_font.render(line, True, text_muted), (right.x + 10, y))
            y += 20

        timer_rect = pygame.Rect(right.x + 10, y + 4, right.w - 20, 34)
        pygame.draw.rect(surface, (52, 55, 61), timer_rect)
        pygame.draw.rect(surface, card_border, timer_rect, 1)
        if snap.time_remaining_s is not None:
            rem = int(round(snap.time_remaining_s))
            mm = rem // 60
            ss = rem % 60
            timer_label = f"TIME {mm:02d}:{ss:02d}"
        else:
            timer_label = "TIME --:--"
        timer_surf = self._small_font.render(timer_label, True, text_main)
        surface.blit(timer_surf, timer_surf.get_rect(center=timer_rect.center))

        answer_rect = pygame.Rect(right.x + 10, timer_rect.bottom + 8, right.w - 20, 52)
        pygame.draw.rect(surface, (52, 55, 61), answer_rect)
        pygame.draw.rect(surface, card_border, answer_rect, 1)
        surface.blit(
            self._tiny_font.render("ANSWER", True, text_dim),
            (answer_rect.x + 8, answer_rect.y + 5),
        )

        slots_x = answer_rect.x + 8
        slots_y = answer_rect.y + 20
        slot_w = max(24, min(34, (answer_rect.w - 22) // 4))
        gap = max(4, min(8, (answer_rect.w - (slot_w * 4) - 16) // 3))
        for i in range(4):
            slot = pygame.Rect(slots_x + i * (slot_w + gap), slots_y, slot_w, 24)
            pygame.draw.rect(surface, (14, 18, 38), slot)
            pygame.draw.rect(surface, (134, 150, 188), slot, 1)
            ch = self._input[i] if i < len(self._input) else ""
            if ch:
                txt = self._small_font.render(ch, True, text_main)
                surface.blit(txt, txt.get_rect(center=slot.center))

        control_text = "Hold A: Distances  |  Hold S/D/F: References"
        surface.blit(
            self._tiny_font.render(control_text, True, text_dim),
            (right.x + 10, answer_rect.bottom + 10),
        )

        if self._air_overlay is not None:
            overlay_rect = pygame.Rect(
                left.x + 16,
                left.y + 16,
                max(220, left.w - 32),
                max(160, left.h - 32),
            )
            pygame.draw.rect(surface, (42, 46, 54), overlay_rect)
            pygame.draw.rect(surface, (210, 218, 236), overlay_rect, 1)

            title = {
                "intro": "Introduction",
                "fuel": "Speed & Fuel",
                "parcel": "Speed & Parcel",
            }[self._air_overlay]
            surface.blit(
                self._small_font.render(title, True, text_main),
                (overlay_rect.x + 12, overlay_rect.y + 10),
            )

            lines: list[str]
            if self._air_overlay == "intro":
                route_names = " -> ".join(scenario.node_names[i] for i in scenario.route)
                lines = [
                    f"Route: {route_names}",
                    "Use map + journey table to compute requested time.",
                    "Distances are visible only while A is held.",
                ]
            elif self._air_overlay == "fuel":
                lines = [
                    f"Reference speed: {scenario.speed_value} {scenario.speed_unit}",
                    f"Fuel burn: {scenario.fuel_burn_per_hr} L/hr",
                    f"Start fuel: {scenario.start_fuel_liters} L",
                ]
            else:
                lines = [
                    f"Parcel weight: {scenario.parcel_weight_kg} kg",
                    f"Speed at parcel weight: {scenario.speed_value} {scenario.speed_unit}",
                    "Use parcel reference table for related prompts.",
                ]
            self._draw_wrapped_text(
                surface,
                "\n".join(lines),
                pygame.Rect(overlay_rect.x + 12, overlay_rect.y + 46, overlay_rect.w - 24, overlay_rect.h - 56),
                color=text_muted,
                font=self._tiny_font,
                max_lines=8,
            )

    def _render_digit_recognition_screen(
        self,
        surface: pygame.Surface,
        snap: TestSnapshot,
        payload: DigitRecognitionPayload | None,
    ) -> None:
        w, h = surface.get_size()
        bg = (2, 8, 114)
        frame_border = (232, 240, 255)
        panel_bg = (9, 20, 126)
        card_bg = (6, 14, 96)
        text_main = (238, 245, 255)
        text_muted = (190, 204, 232)

        surface.fill(bg)
        margin = max(10, min(20, w // 40))
        frame = pygame.Rect(margin, margin, w - margin * 2, h - margin * 2)
        pygame.draw.rect(surface, frame_border, frame, 1)

        header = pygame.Rect(frame.x + 2, frame.y + 2, frame.w - 4, max(24, min(30, h // 18)))
        pygame.draw.rect(surface, panel_bg, header)
        pygame.draw.line(
            surface,
            frame_border,
            (header.x, header.bottom),
            (header.right, header.bottom),
            1,
        )
        surface.blit(
            self._tiny_font.render("Digit Recognition Test", True, text_main),
            (header.x + 10, header.y + 5),
        )

        phase_label = {
            Phase.INSTRUCTIONS: "Instructions",
            Phase.PRACTICE: "Practice",
            Phase.PRACTICE_DONE: "Practice Complete",
            Phase.SCORED: "Timed Test",
            Phase.RESULTS: "Results",
        }.get(snap.phase, "Task")
        phase_surf = self._tiny_font.render(phase_label, True, text_muted)
        surface.blit(phase_surf, phase_surf.get_rect(topright=(header.right - 10, header.y + 5)))

        if snap.time_remaining_s is not None:
            rem = int(round(snap.time_remaining_s))
            mm = rem // 60
            ss = rem % 60
            timer_surf = self._tiny_font.render(f"{mm:02d}:{ss:02d}", True, text_main)
            surface.blit(timer_surf, timer_surf.get_rect(topright=(header.right - 10, header.bottom + 8)))

        stats = self._tiny_font.render(
            f"Scored {snap.correct_scored}/{snap.attempted_scored}",
            True,
            text_muted,
        )
        surface.blit(stats, (frame.x + 12, header.bottom + 8))

        content = pygame.Rect(
            frame.x + 18,
            header.bottom + 28,
            frame.w - 36,
            frame.h - header.h - 58,
        )
        pygame.draw.rect(surface, card_bg, content)
        pygame.draw.rect(surface, (120, 146, 202), content, 1)

        # Instructions, transition, and results screens use prompt text only.
        if snap.phase in (Phase.INSTRUCTIONS, Phase.PRACTICE_DONE, Phase.RESULTS):
            self._draw_wrapped_text(
                surface,
                str(snap.prompt),
                content.inflate(-26, -20),
                color=text_main,
                font=self._small_font,
                max_lines=14,
            )
            return

        stage_rect = pygame.Rect(
            content.x + 20,
            content.y + 64,
            content.w - 40,
            max(120, content.h - 140),
        )
        pygame.draw.rect(surface, panel_bg, stage_rect)
        pygame.draw.rect(surface, (146, 168, 214), stage_rect, 1)

        if payload is not None and payload.display_digits is not None:
            label = self._small_font.render("MEMORIZE", True, text_muted)
            surface.blit(label, label.get_rect(midtop=(stage_rect.centerx, stage_rect.y + 14)))

            txt = payload.display_digits
            digits = self._big_font.render(txt, True, text_main)
            if digits.get_width() > int(stage_rect.w * 0.92):
                digits = self._mid_font.render(txt, True, text_main)
            surface.blit(digits, digits.get_rect(center=stage_rect.center))
            return

        if payload is not None and not payload.accepting_input:
            label = self._small_font.render("MASK", True, text_muted)
            surface.blit(label, label.get_rect(midtop=(stage_rect.centerx, stage_rect.y + 14)))
            mask_txt = self._mid_font.render("XXXX XXXX XXXX", True, (128, 146, 186))
            surface.blit(mask_txt, mask_txt.get_rect(center=stage_rect.center))
            return

        # Question stage.
        prompt_box = pygame.Rect(stage_rect.x + 20, stage_rect.y + 22, stage_rect.w - 40, 74)
        pygame.draw.rect(surface, (7, 16, 108), prompt_box)
        pygame.draw.rect(surface, (112, 136, 188), prompt_box, 1)
        self._draw_wrapped_text(
            surface,
            str(snap.prompt),
            prompt_box.inflate(-14, -12),
            color=text_main,
            font=self._small_font,
            max_lines=2,
        )

    def _render_digit_recognition_answer_box(
        self,
        surface: pygame.Surface,
        snap: TestSnapshot,
        payload: DigitRecognitionPayload | None,
    ) -> None:
        if payload is None or not payload.accepting_input:
            return

        w, h = surface.get_size()
        text_main = (238, 245, 255)
        box_fill = (8, 18, 116)
        box_border = (138, 162, 210)

        box_w = max(260, min(460, int(w * 0.52)))
        box_h = 52
        box_x = (w - box_w) // 2
        box_y = h - 96
        box = pygame.Rect(box_x, box_y, box_w, box_h)
        pygame.draw.rect(surface, box_fill, box)
        pygame.draw.rect(surface, box_border, box, 2)

        label = self._tiny_font.render("YOUR ANSWER", True, (188, 204, 232))
        surface.blit(label, (box.x + 10, box.y + 6))

        caret = "|" if (pygame.time.get_ticks() // 500) % 2 == 0 else ""
        entry = self._small_font.render(self._input + caret, True, text_main)
        surface.blit(entry, (box.x + 10, box.y + 22))

        hint = self._tiny_font.render(snap.input_hint, True, (168, 186, 222))
        surface.blit(hint, hint.get_rect(midtop=(w // 2, box.bottom + 6)))

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
        text_main = (238, 245, 255)
        text_muted = (176, 192, 218)

        header = self._tiny_font.render("Journey Table", True, text_main)
        surface.blit(header, (rect.x + 10, rect.y + 8))

        inner = pygame.Rect(rect.x + 8, rect.y + 24, rect.w - 16, rect.h - 32)
        pygame.draw.rect(surface, (54, 58, 65), inner)
        pygame.draw.rect(surface, (156, 170, 198), inner, 1)

        # Responsive column anchors (fractions of inner width).
        col_defs = [
            ("LEG", 0.03),
            ("FROM", 0.13),
            ("TO", 0.30),
            ("DIST", 0.46),
            ("SPEED", 0.60),
            ("TIME", 0.75),
            ("PARCEL", 0.88),
        ]
        cols = [(name, inner.x + int(inner.w * frac)) for name, frac in col_defs]

        y = inner.y + 6
        for label, x in cols:
            surface.blit(self._tiny_font.render(label, True, text_muted), (x, y))
        pygame.draw.line(surface, (120, 132, 157), (inner.x + 4, y + 16), (inner.right - 4, y + 16), 1)
        y += 20

        pw = getattr(scenario, "parcel_weight_kg", getattr(scenario, "parcel_weight", 0))
        row_h = 20
        max_rows = max(3, min(5, (inner.h - 26) // row_h))
        for i in range(max_rows):
            if i < len(scenario.legs):
                leg = scenario.legs[i]
                dist = str(getattr(leg, "distance", "----")) if self._air_show_distances else "----"
                row = [
                    str(i + 1),
                    scenario.node_names[getattr(leg, "frm")],
                    scenario.node_names[getattr(leg, "to")],
                    dist,
                    "----",
                    "----",
                    str(pw),
                ]
            else:
                row = ["", "", "", "", "", "", ""]

            for (text, (_, x)) in zip(row, cols, strict=True):
                surface.blit(self._tiny_font.render(text, True, text_main), (x, y))
            y += row_h


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
    surface = pygame.display.set_mode(WINDOW_SIZE, pygame.RESIZABLE)

    font = pygame.font.Font(None, 36)
    clock = pygame.time.Clock()

    app = App(surface=surface, font=font)

    workouts = PlaceholderScreen(app, "90-minute workouts")
    drills = PlaceholderScreen(app, "Individual drills")

    axis_calibration = PlaceholderScreen(app, "Axis Calibration (placeholder)")
    axis_visualizer = PlaceholderScreen(app, "Axis Visualizer (placeholder)")
    input_profiles = PlaceholderScreen(app, "Input Profiles (placeholder)")

    settings_menu = MenuScreen(
        app,
        "HOTAS & Input",
        [
            MenuItem("Axis Calibration", lambda: app.push(axis_calibration)),
            MenuItem("Axis Visualizer", lambda: app.push(axis_visualizer)),
            MenuItem("Input Profiles", lambda: app.push(input_profiles)),
            MenuItem("Back", app.pop),
        ],
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

    def open_angles_bearings_degrees() -> None:
        seed = _new_seed()
        app.push(
            CognitiveTestScreen(
                app,
                engine_factory=lambda: build_angles_bearings_degrees_test(
                    clock=real_clock,
                    seed=seed,
                    difficulty=0.5,
                ),
            )
        )

    def open_visual_search() -> None:
        seed = _new_seed()
        app.push(
            CognitiveTestScreen(
                app,
                engine_factory=lambda: build_visual_search_test(
                    clock=real_clock,
                    seed=seed,
                    difficulty=0.5,
                ),
            )
        )

    def open_instrument_comprehension() -> None:
        seed = _new_seed()
        app.push(
            CognitiveTestScreen(
                app,
                engine_factory=lambda: build_instrument_comprehension_test(
                    clock=real_clock,
                    seed=seed,
                    difficulty=0.5,
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
            MenuItem("Digit Recognition", open_digit_recognition),
            MenuItem("Angles, Bearings and Degrees", open_angles_bearings_degrees),
            MenuItem("Visual Search (Target Recognition)", open_visual_search),
            MenuItem("Instrument Comprehension", open_instrument_comprehension),
            MenuItem("Back", app.pop),
        ],
    )

    main_items = [
        MenuItem("90-minute workouts", lambda: app.push(workouts)),
        MenuItem("Individual drills", lambda: app.push(drills)),
        MenuItem("Tests", lambda: app.push(tests_menu)),
        MenuItem("HOTAS", lambda: app.push(settings_menu)),
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

            clock.tick(TARGET_FPS)
    finally:
        pygame.quit()

    return 0
