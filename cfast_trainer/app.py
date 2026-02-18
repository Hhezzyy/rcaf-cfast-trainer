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
from .colours_letters_numbers import (
    ColoursLettersNumbersPayload,
    build_colours_letters_numbers_test,
)
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
from .target_recognition import (
    TargetRecognitionPayload,
    TargetRecognitionSceneEntity,
    build_target_recognition_test,
)
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


@dataclass(slots=True)
class _TargetRecognitionSceneGlyph:
    glyph_id: int
    kind: str  # "entity" | "beacon" | "unknown"
    entity: TargetRecognitionSceneEntity | None
    nx: float
    ny: float
    scale: float
    heading: float
    alpha: float
    max_alpha: float
    matching_labels: tuple[str, ...]


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

        # CLN mouse-selection hitboxes (code -> rect), refreshed during render.
        self._cln_option_hitboxes: dict[int, pygame.Rect] = {}

        # Target-recognition panel interaction + animation state.
        self._tr_selection_payload_id: int | None = None
        self._tr_selected_panels: set[str] = set()
        self._tr_selector_hitboxes: dict[str, pygame.Rect] = {}
        self._tr_light_payload_id: int | None = None
        self._tr_light_rng: random.Random | None = None
        self._tr_light_next_change_ms = 0
        self._tr_light_current_pattern: tuple[str, str, str] = ("G", "G", "G")
        self._tr_light_target_pattern_live: tuple[str, str, str] = ("G", "G", "G")
        self._tr_light_points = 0
        self._tr_light_hits = 0
        self._tr_light_early_presses = 0
        self._tr_light_button_hitbox: pygame.Rect | None = None

        self._tr_scan_payload_id: int | None = None
        self._tr_scan_rng: random.Random | None = None
        self._tr_scan_token_pool: tuple[str, ...] = ("<>", "[]", "/\\", "\\/")
        self._tr_scan_next_change_ms = 0
        self._tr_scan_current_pattern: tuple[str, str, str, str] = ("<>", "[]", "/\\", "\\/")
        self._tr_scan_target_pattern_live: tuple[str, str, str, str] = ("<>", "[]", "/\\", "\\/")
        self._tr_scan_points = 0
        self._tr_scan_hits = 0
        self._tr_scan_early_presses = 0
        self._tr_scan_button_hitbox: pygame.Rect | None = None
        self._tr_scan_reveal_index = 3
        self._tr_scan_next_step_ms = 0
        self._tr_scan_passes_left = 0

        self._tr_system_payload_id: int | None = None
        self._tr_system_rng: random.Random | None = None
        self._tr_system_columns: list[list[str]] = [[], [], []]
        self._tr_system_row_offset = 0
        self._tr_system_row_frac = 0.0
        self._tr_system_target_code = "----"
        self._tr_system_step_interval_ms = 1700
        self._tr_system_last_step_ms = 0
        self._tr_system_points = 0
        self._tr_system_hits = 0
        self._tr_system_string_hitboxes: list[tuple[pygame.Rect, str]] = []
        self._tr_scene_payload_id: int | None = None
        self._tr_scene_rng: random.Random | None = None
        self._tr_scene_glyphs: dict[int, _TargetRecognitionSceneGlyph] = {}
        self._tr_scene_glyph_order: list[int] = []
        self._tr_scene_symbol_hitboxes: list[tuple[pygame.Rect, int]] = []
        self._tr_scene_next_glyph_id = 1
        self._tr_scene_target_queue: list[str] = []
        self._tr_scene_active_targets: list[str] = []
        self._tr_scene_next_target_add_ms = 0
        self._tr_scene_had_active_targets = False
        self._tr_scene_complete = False
        self._tr_scene_points = 0
        self._tr_scene_hits = 0
        self._tr_scene_misses = 0
        self._tr_scene_beacon_hits = 0
        self._tr_scene_unknown_hits = 0
        self._tr_scene_anim_frame = 0
        self._tr_scene_last_update_ms = 0
        self._tr_scene_base_cache: pygame.Surface | None = None
        self._tr_scene_base_cache_size: tuple[int, int] = (0, 0)
        self._tr_scene_base_cache_seed = 0
        self._target_recognition_reset_practice_breakdown()
        self._target_recognition_reset_scene_subtask()
        self._target_recognition_reset_light_subtask()
        self._target_recognition_reset_scan_subtask()
        self._target_recognition_reset_system_subtask()

    def handle_event(self, event: pygame.event.Event) -> None:
        snap = self._engine.snapshot()
        p = snap.payload
        scenario = p if isinstance(p, AirborneScenario) else None
        math_payload: MathReasoningPayload | None = p if isinstance(p, MathReasoningPayload) else None
        angles_payload: AnglesBearingsDegreesPayload | None = (
            p if isinstance(p, AnglesBearingsDegreesPayload) else None
        )
        cln_payload: ColoursLettersNumbersPayload | None = (
            p if isinstance(p, ColoursLettersNumbersPayload) else None
        )
        tr_payload: TargetRecognitionPayload | None = (
            p if isinstance(p, TargetRecognitionPayload) else None
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

        if (
            event.type == pygame.MOUSEBUTTONDOWN
            and getattr(event, "button", 0) == 1
            and tr_payload is not None
            and snap.phase in (Phase.PRACTICE, Phase.SCORED)
        ):
            self._target_recognition_sync_selection(tr_payload)
            self._target_recognition_sync_light_stream(tr_payload)
            self._target_recognition_sync_scan_stream(tr_payload)
            self._target_recognition_sync_system_stream(tr_payload)
            pos = getattr(event, "pos", None)
            if pos is not None:
                expected = self._target_recognition_expected_panels(tr_payload)

                for hit_rect, glyph_id in reversed(self._tr_scene_symbol_hitboxes):
                    if not hit_rect.collidepoint(pos):
                        continue
                    scene_success = self._target_recognition_handle_scene_press(
                        tr_payload,
                        glyph_id=glyph_id,
                    )
                    if "scene" in expected:
                        if self._tr_scene_complete:
                            self._tr_selected_panels.add("scene")
                        else:
                            self._tr_selected_panels.discard("scene")
                        if scene_success and self._tr_selected_panels == expected:
                            selected_snapshot = set(self._tr_selected_panels)
                            accepted = self._engine.submit_answer(str(len(selected_snapshot)))
                            if accepted:
                                if snap.phase is Phase.PRACTICE:
                                    self._target_recognition_record_practice_trial(
                                        selected=selected_snapshot,
                                        expected=expected,
                                    )
                                self._tr_selected_panels.clear()
                    return

                if (
                    self._tr_light_button_hitbox is not None
                    and self._tr_light_button_hitbox.collidepoint(pos)
                ):
                    light_success = self._target_recognition_handle_light_press(tr_payload)
                    if "light" in expected:
                        if light_success:
                            self._tr_selected_panels.add("light")
                        else:
                            self._tr_selected_panels.discard("light")
                        if light_success and self._tr_selected_panels == expected:
                            selected_snapshot = set(self._tr_selected_panels)
                            accepted = self._engine.submit_answer(str(len(selected_snapshot)))
                            if accepted:
                                if snap.phase is Phase.PRACTICE:
                                    self._target_recognition_record_practice_trial(
                                        selected=selected_snapshot,
                                        expected=expected,
                                    )
                                self._tr_selected_panels.clear()
                    return

                if (
                    self._tr_scan_button_hitbox is not None
                    and self._tr_scan_button_hitbox.collidepoint(pos)
                ):
                    scan_success = self._target_recognition_handle_scan_press(tr_payload)
                    if "scan" in expected:
                        if scan_success:
                            self._tr_selected_panels.add("scan")
                        else:
                            self._tr_selected_panels.discard("scan")
                        if scan_success and self._tr_selected_panels == expected:
                            selected_snapshot = set(self._tr_selected_panels)
                            accepted = self._engine.submit_answer(str(len(selected_snapshot)))
                            if accepted:
                                if snap.phase is Phase.PRACTICE:
                                    self._target_recognition_record_practice_trial(
                                        selected=selected_snapshot,
                                        expected=expected,
                                    )
                                self._tr_selected_panels.clear()
                    return

                for hit_rect, hit_code in self._tr_system_string_hitboxes:
                    if hit_rect.collidepoint(pos):
                        system_success = self._target_recognition_handle_system_press(
                            tr_payload,
                            clicked_code=hit_code,
                        )
                        if "system" in expected:
                            if system_success:
                                self._tr_selected_panels.add("system")
                            else:
                                self._tr_selected_panels.discard("system")
                            if system_success and self._tr_selected_panels == expected:
                                selected_snapshot = set(self._tr_selected_panels)
                                accepted = self._engine.submit_answer(str(len(selected_snapshot)))
                                if accepted:
                                    if snap.phase is Phase.PRACTICE:
                                        self._target_recognition_record_practice_trial(
                                            selected=selected_snapshot,
                                            expected=expected,
                                        )
                                    self._tr_selected_panels.clear()
                        return

                for panel, rect in self._tr_selector_hitboxes.items():
                    if rect.collidepoint(pos):
                        if panel == "scene":
                            if bool(tr_payload.scene_has_target):
                                self._tr_selected_panels.add("scene")
                            else:
                                self._tr_selected_panels.discard("scene")
                        else:
                            if panel in self._tr_selected_panels:
                                self._tr_selected_panels.remove(panel)
                            else:
                                self._tr_selected_panels.add(panel)

                        if self._tr_selected_panels == expected:
                            selected_snapshot = set(self._tr_selected_panels)
                            accepted = self._engine.submit_answer(str(len(selected_snapshot)))
                            if accepted:
                                if snap.phase is Phase.PRACTICE:
                                    self._target_recognition_record_practice_trial(
                                        selected=selected_snapshot,
                                        expected=expected,
                                    )
                                self._tr_selected_panels.clear()
                        return

        if (
            event.type == pygame.MOUSEBUTTONDOWN
            and getattr(event, "button", 0) == 1
            and cln_payload is not None
            and snap.phase in (Phase.PRACTICE, Phase.SCORED)
            and cln_payload.options_active
        ):
            pos = getattr(event, "pos", None)
            if pos is not None:
                for code, rect in self._cln_option_hitboxes.items():
                    if rect.collidepoint(pos):
                        self._engine.submit_answer(f"MEM:{code}")
                        return

        if event.type != pygame.KEYDOWN:
            return

        key = event.key

        if key in (pygame.K_ESCAPE, pygame.K_BACKSPACE) and self._engine.can_exit():
            self._app.pop()
            return

        if key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            if snap.phase is Phase.INSTRUCTIONS:
                if snap.title == "Target Recognition":
                    self._target_recognition_reset_practice_breakdown()
                    self._target_recognition_reset_scene_subtask()
                    self._target_recognition_reset_light_subtask()
                    self._target_recognition_reset_scan_subtask()
                    self._target_recognition_reset_system_subtask()
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
            if tr_payload is not None:
                # Target Recognition answers are mouse-only in PRACTICE/SCORED.
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

        if cln_payload is not None:
            color_key = {
                pygame.K_q: "Q",
                pygame.K_w: "W",
                pygame.K_e: "E",
                pygame.K_r: "R",
            }.get(key)
            if color_key is not None:
                self._engine.submit_answer(f"CLR:{color_key}")
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

        if tr_payload is not None:
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
            and not isinstance(snap.payload, TargetRecognitionPayload)
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
        tr: TargetRecognitionPayload | None = p if isinstance(p, TargetRecognitionPayload) else None
        vs: VisualSearchPayload | None = p if isinstance(p, VisualSearchPayload) else None
        mr: MathReasoningPayload | None = p if isinstance(p, MathReasoningPayload) else None
        cln: ColoursLettersNumbersPayload | None = (
            p if isinstance(p, ColoursLettersNumbersPayload) else None
        )
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
        is_colours_letters_numbers = snap.title == "Colours, Letters and Numbers"
        is_instrument_comprehension = snap.title == "Instrument Comprehension"
        is_target_recognition = snap.title == "Target Recognition"
        if is_numerical_ops and snap.phase in (Phase.PRACTICE, Phase.SCORED):
            self._render_numerical_operations_question(surface, snap)
        elif is_math_reasoning:
            self._render_math_reasoning(surface, snap, mr)
        elif is_angles_bearings:
            self._render_angles_bearings_screen(surface, snap, abd)
        elif is_colours_letters_numbers:
            self._render_colours_letters_numbers_screen(surface, snap, cln)
        elif is_digit_recognition:
            self._render_digit_recognition_screen(surface, snap, dr)
        elif is_instrument_comprehension:
            self._render_instrument_comprehension_screen(surface, snap, ic)
        elif is_target_recognition:
            self._render_target_recognition_screen(surface, snap, tr)
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
            elif is_colours_letters_numbers:
                pass
            elif is_digit_recognition:
                self._render_digit_recognition_answer_box(surface, snap, dr)
            elif is_instrument_comprehension:
                pass
            elif is_target_recognition:
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

    @staticmethod
    def _fit_label(font: pygame.font.Font, label: str, max_width: int) -> str:
        if max_width <= 0:
            return ""
        if font.size(label)[0] <= max_width:
            return label
        clipped = label
        while clipped and font.size(f"{clipped}...")[0] > max_width:
            clipped = clipped[:-1]
        return f"{clipped}..." if clipped else "..."

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

    def _render_target_recognition_screen(
        self,
        surface: pygame.Surface,
        snap: TestSnapshot,
        payload: TargetRecognitionPayload | None,
    ) -> None:
        w, h = surface.get_size()
        bg = (2, 10, 108)
        frame_bg = (4, 14, 96)
        panel_bg = (8, 16, 80)
        panel_header = (130, 10, 22)
        strip_header = (20, 148, 26)
        border = (228, 238, 255)
        text_main = (236, 244, 255)
        text_muted = (182, 198, 226)
        text_dim = (132, 148, 178)

        surface.fill(bg)

        margin = max(8, min(16, w // 56))
        frame = pygame.Rect(margin, margin, w - margin * 2, h - margin * 2)
        pygame.draw.rect(surface, frame_bg, frame)
        pygame.draw.rect(surface, border, frame, 1)

        header_h = max(28, min(36, h // 15))
        header = pygame.Rect(frame.x + 1, frame.y + 1, frame.w - 2, header_h)
        pygame.draw.rect(surface, bg, header)
        pygame.draw.line(surface, border, (header.x, header.bottom), (header.right, header.bottom), 1)

        phase_label = {
            Phase.INSTRUCTIONS: "Instructions",
            Phase.PRACTICE: "Practice",
            Phase.PRACTICE_DONE: "Practice Complete",
            Phase.SCORED: "Timed Test",
            Phase.RESULTS: "Results",
        }.get(snap.phase, "Task")
        left = self._tiny_font.render(f"Target Recognition - {phase_label}", True, text_main)
        surface.blit(left, left.get_rect(midleft=(header.x + 10, header.centery)))

        stats = self._tiny_font.render(
            f"Scored {snap.correct_scored}/{snap.attempted_scored}",
            True,
            text_muted,
        )
        surface.blit(stats, stats.get_rect(midright=(header.right - 10, header.centery)))

        if snap.time_remaining_s is not None:
            rem = int(round(snap.time_remaining_s))
            timer = self._small_font.render(f"{rem // 60:02d}:{rem % 60:02d}", True, text_main)
            surface.blit(timer, timer.get_rect(topright=(frame.right - 12, header.bottom + 6)))

        content = pygame.Rect(frame.x + 8, header.bottom + 8, frame.w - 16, frame.bottom - header.bottom - 16)
        if payload is None or snap.phase not in (Phase.PRACTICE, Phase.SCORED):
            self._tr_selector_hitboxes = {}
            self._tr_light_button_hitbox = None
            self._tr_scan_button_hitbox = None
            self._tr_system_string_hitboxes = []
            self._tr_scene_symbol_hitboxes = []
            card = content.inflate(-8, -8)
            pygame.draw.rect(surface, panel_bg, card)
            pygame.draw.rect(surface, border, card, 1)
            if snap.phase is Phase.PRACTICE_DONE:
                prompt_rect = pygame.Rect(card.x + 14, card.y + 12, card.w - 28, 40)
                self._draw_wrapped_text(
                    surface,
                    str(snap.prompt),
                    prompt_rect,
                    color=text_main,
                    font=self._small_font,
                    max_lines=2,
                )
                title = self._small_font.render("Practice Category Breakdown", True, text_main)
                surface.blit(title, (card.x + 14, prompt_rect.bottom + 8))

                y = prompt_rect.bottom + 34
                step = self._tiny_font.get_linesize() + 2
                for line in self._target_recognition_practice_breakdown_lines():
                    row = self._tiny_font.render(line, True, text_muted)
                    surface.blit(row, (card.x + 16, y))
                    y += step
            else:
                self._draw_wrapped_text(
                    surface,
                    str(snap.prompt),
                    card.inflate(-14, -14),
                    color=text_main,
                    font=self._small_font,
                    max_lines=12,
                )
            footer = (
                "Enter: Continue  |  Esc/Backspace: Back"
                if snap.phase in (Phase.INSTRUCTIONS, Phase.PRACTICE_DONE)
                else "Enter: Return to Tests"
            )
            footer_surf = self._tiny_font.render(footer, True, text_muted)
            surface.blit(footer_surf, footer_surf.get_rect(midbottom=(frame.centerx, frame.bottom - 10)))
            return

        self._target_recognition_sync_selection(payload)
        self._tr_selector_hitboxes = {}
        self._tr_light_button_hitbox = None
        self._tr_scan_button_hitbox = None
        self._tr_system_string_hitboxes = []
        self._tr_scene_symbol_hitboxes = []
        self._target_recognition_sync_scene_stream(payload)
        self._target_recognition_sync_light_stream(payload)
        self._target_recognition_sync_scan_stream(payload)
        self._target_recognition_sync_system_stream(payload)
        live_light_pattern = self._tr_light_current_pattern
        live_light_target = self._tr_light_target_pattern_live
        live_scan_pattern = self._tr_scan_current_pattern
        live_scan_target = self._tr_scan_target_pattern_live

        active_system_target, system_columns, system_row_frac = self._target_recognition_system_view(
            payload
        )

        target_strip_h = max(92, min(130, h // 4))
        panels_h = max(140, content.h - target_strip_h - 6)
        panels = pygame.Rect(content.x, content.y, content.w, panels_h)
        targets = pygame.Rect(content.x, panels.bottom + 6, content.w, target_strip_h)

        right_w = max(170, min(230, int(panels.w * 0.28)))
        left_rect = pygame.Rect(panels.x, panels.y, panels.w - right_w - 8, panels.h)
        right_rect = pygame.Rect(left_rect.right + 8, panels.y, right_w, panels.h)

        top_h = max(76, min(106, int(left_rect.h * 0.26)))
        top_row = pygame.Rect(left_rect.x, left_rect.y, left_rect.w, top_h)
        scene_rect = pygame.Rect(left_rect.x, top_row.bottom + 8, left_rect.w, left_rect.h - top_h - 8)

        gap = 8
        col_w = max(80, (top_row.w - gap * 2) // 3)
        info_rect = pygame.Rect(top_row.x, top_row.y, col_w, top_row.h)
        light_rect = pygame.Rect(info_rect.right + gap, top_row.y, col_w, top_row.h)
        scan_rect = pygame.Rect(light_rect.right + gap, top_row.y, top_row.right - (light_rect.right + gap), top_row.h)

        def draw_panel(rect: pygame.Rect, title: str) -> pygame.Rect:
            pygame.draw.rect(surface, panel_bg, rect)
            pygame.draw.rect(surface, border, rect, 1)
            bar = pygame.Rect(rect.x + 1, rect.y + 1, rect.w - 2, 18)
            pygame.draw.rect(surface, panel_header, bar)
            lbl = self._tiny_font.render(title, True, text_main)
            surface.blit(lbl, lbl.get_rect(center=bar.center))
            return pygame.Rect(rect.x + 6, rect.y + 24, rect.w - 12, rect.h - 30)

        info_inner = draw_panel(info_rect, "Information")
        light_inner = draw_panel(light_rect, "Light Panel")
        scan_inner = draw_panel(scan_rect, "Scan Panel")
        scene_inner = draw_panel(scene_rect, "Map Panel")
        system_inner = draw_panel(right_rect, "System Panel")

        self._draw_target_recognition_info_legend(surface, info_inner)

        light_bg = light_inner.inflate(-1, -2)
        pygame.draw.rect(surface, (44, 44, 52), light_bg)
        pygame.draw.rect(surface, (86, 104, 150), light_bg, 1)
        bulb_y = light_bg.centery
        bulb_r = max(8, min(12, (light_bg.h // 2) - 4))
        step = max(20, light_bg.w // 4)
        start_x = light_bg.x + max(16, (light_bg.w - (step * 2)) // 2)
        for idx, code in enumerate(live_light_pattern):
            cx = start_x + idx * step
            color = self._target_recognition_light_color(code)
            pygame.draw.circle(surface, color, (cx, bulb_y), bulb_r)
            pygame.draw.circle(surface, (16, 22, 44), (cx, bulb_y), bulb_r, 2)

        light_btn_w = max(56, min(72, light_bg.w // 3))
        light_btn_h = max(24, min(32, light_bg.h - 8))
        light_btn = pygame.Rect(light_bg.right - light_btn_w - 4, light_bg.centery - (light_btn_h // 2), light_btn_w, light_btn_h)
        pygame.draw.rect(surface, (220, 174, 34), light_btn)
        pygame.draw.rect(surface, (248, 234, 184), light_btn, 1)
        light_btn_txt = self._tiny_font.render("PRESS", True, (28, 20, 6))
        surface.blit(light_btn_txt, light_btn_txt.get_rect(center=light_btn.center))
        self._tr_light_button_hitbox = light_btn

        scan_bg = scan_inner.inflate(-1, -2)
        pygame.draw.rect(surface, (28, 30, 36), scan_bg)
        pygame.draw.rect(surface, (86, 104, 150), scan_bg, 1)
        scan_token_w = max(22, min(32, (scan_bg.w - 84) // 4))
        scan_token_h = max(18, min(24, scan_bg.h - 8))
        scan_gap = 4
        scan_x0 = scan_bg.x + 4
        scan_y = scan_bg.centery - (scan_token_h // 2)
        for idx in range(4):
            tok_rect = pygame.Rect(scan_x0 + idx * (scan_token_w + scan_gap), scan_y, scan_token_w, scan_token_h)
            pygame.draw.rect(surface, (14, 20, 34), tok_rect)
            pygame.draw.rect(surface, (72, 92, 126), tok_rect, 1)
        reveal_idx = max(0, min(3, int(self._tr_scan_reveal_index)))
        show_rect = pygame.Rect(
            scan_x0 + reveal_idx * (scan_token_w + scan_gap),
            scan_y,
            scan_token_w,
            scan_token_h,
        )
        pygame.draw.rect(surface, (26, 42, 72), show_rect)
        pygame.draw.rect(surface, (118, 164, 226), show_rect, 1)
        tok_s = self._tiny_font.render(str(live_scan_pattern[reveal_idx]), True, text_main)
        surface.blit(tok_s, tok_s.get_rect(center=show_rect.center))

        scan_btn_w = max(56, min(72, scan_bg.w // 3))
        scan_btn_h = max(24, min(32, scan_bg.h - 8))
        scan_btn = pygame.Rect(scan_bg.right - scan_btn_w - 4, scan_bg.centery - (scan_btn_h // 2), scan_btn_w, scan_btn_h)
        pygame.draw.rect(surface, (220, 174, 34), scan_btn)
        pygame.draw.rect(surface, (248, 234, 184), scan_btn, 1)
        scan_btn_txt = self._tiny_font.render("PRESS", True, (28, 20, 6))
        surface.blit(scan_btn_txt, scan_btn_txt.get_rect(center=scan_btn.center))
        self._tr_scan_button_hitbox = scan_btn

        self._draw_target_recognition_scene(surface, scene_inner, payload)

        pygame.draw.rect(surface, (30, 30, 38), system_inner)
        pygame.draw.rect(surface, (86, 104, 150), system_inner, 1)

        cols = max(1, len(system_columns))
        gap_x = 8
        inner_top = system_inner.y + 4
        inner_h = max(20, system_inner.bottom - inner_top - 2)
        col_w = max(46, (system_inner.w - gap_x * (cols + 1)) // cols)
        row_h = self._tiny_font.get_linesize() + 2
        max_rows = max(1, inner_h // row_h)

        for col_idx, col_values in enumerate(system_columns):
            x = system_inner.x + gap_x + col_idx * (col_w + gap_x)
            col_rect = pygame.Rect(x, inner_top, col_w, inner_h)
            pygame.draw.rect(surface, (18, 20, 26), col_rect)
            pygame.draw.rect(surface, (70, 86, 124), col_rect, 1)
            if not col_values:
                continue

            clip = col_rect.inflate(-2, -2)
            prev_clip = surface.get_clip()
            surface.set_clip(clip)
            n_rows = len(col_values)
            for slot in range(-1, max_rows + 2):
                row = col_values[slot % n_rows]
                y = clip.y + int((slot + system_row_frac) * row_h)
                row_surf = self._tiny_font.render(str(row), True, text_main)
                surface.blit(row_surf, (clip.x + 3, y))
                hit = pygame.Rect(clip.x + 3, y, max(8, min(clip.w - 5, row_surf.get_width())), row_h)
                hit = hit.clip(clip)
                if hit.w > 0 and hit.h > 0:
                    self._tr_system_string_hitboxes.append((hit, str(row)))
            surface.set_clip(prev_clip)

        pygame.draw.rect(surface, panel_bg, targets)
        pygame.draw.rect(surface, border, targets, 1)

        controls_h = 26
        boxes_area = pygame.Rect(targets.x + 2, targets.y + 2, targets.w - 4, max(24, targets.h - controls_h - 4))
        controls = pygame.Rect(targets.x + 2, boxes_area.bottom + 2, targets.w - 4, controls_h)

        target_gap = 6
        target_w = max(80, (boxes_area.w - target_gap * 3) // 4)
        target_labels = (
            ("scene", "Map Targets", ""),
            ("light", "Light Target", "-".join(live_light_target)),
            ("scan", "Scan Target", " ".join(live_scan_target)),
            ("system", "System Target", active_system_target),
        )
        for idx, (panel_key, label, value) in enumerate(target_labels):
            x = boxes_area.x + idx * (target_w + target_gap)
            if idx == 3:
                box_w = boxes_area.right - x
            else:
                box_w = target_w
            box = pygame.Rect(x, boxes_area.y, box_w, boxes_area.h)
            pygame.draw.rect(surface, (4, 9, 36), box)
            pygame.draw.rect(surface, border, box, 1)
            bar = pygame.Rect(box.x + 1, box.y + 1, box.w - 2, 16)
            pygame.draw.rect(surface, strip_header, bar)
            label_surf = self._tiny_font.render(label, True, text_main)
            surface.blit(label_surf, label_surf.get_rect(center=bar.center))
            value_rect = pygame.Rect(box.x + 6, bar.bottom + 4, box.w - 12, box.bottom - bar.bottom - 8)
            if panel_key == "scene":
                lines = list(self._tr_scene_active_targets)[:5]
                y = value_rect.y
                line_h = self._tiny_font.get_linesize() + 1
                for line in lines:
                    surf = self._tiny_font.render(line, True, text_main)
                    surface.blit(surf, (value_rect.x, y))
                    y += line_h
            elif panel_key == "light":
                dot_r = max(6, min(10, value_rect.h // 3))
                dot_step = max(dot_r * 2 + 8, value_rect.w // 3)
                dots_x0 = value_rect.x + max(dot_r + 2, (value_rect.w - (dot_step * 2 + dot_r * 2)) // 2)
                cy = value_rect.centery
                for dot_idx, code in enumerate(live_light_target):
                    dcx = dots_x0 + dot_idx * dot_step
                    dcol = self._target_recognition_light_color(code)
                    pygame.draw.circle(surface, dcol, (dcx, cy), dot_r)
                    pygame.draw.circle(surface, (16, 22, 44), (dcx, cy), dot_r, 2)
            elif panel_key == "scan":
                tok_w = max(18, min(28, value_rect.w // 4))
                tok_h = max(18, min(24, value_rect.h))
                tok_gap = 4
                x0 = value_rect.x + max(2, (value_rect.w - (tok_w * 4 + tok_gap * 3)) // 2)
                y0 = value_rect.centery - (tok_h // 2)
                for tok_idx, tok in enumerate(live_scan_target):
                    tok_rect = pygame.Rect(x0 + tok_idx * (tok_w + tok_gap), y0, tok_w, tok_h)
                    pygame.draw.rect(surface, (14, 20, 34), tok_rect)
                    pygame.draw.rect(surface, (110, 132, 188), tok_rect, 1)
                    tok_s = self._tiny_font.render(str(tok), True, text_main)
                    surface.blit(tok_s, tok_s.get_rect(center=tok_rect.center))
            else:
                value_surf = self._small_font.render(value, True, text_main)
                value_pos = value_surf.get_rect(center=(value_rect.centerx, value_rect.centery))
                surface.blit(value_surf, value_pos)
            if panel_key not in ("scene", "light", "scan", "system"):
                self._tr_selector_hitboxes[panel_key] = box

        pygame.draw.rect(surface, (5, 12, 42), controls)
        pygame.draw.rect(surface, (72, 92, 138), controls, 1)

        hint = self._tiny_font.render(
            (
                "Mouse only: click active panel controls when matching. "
                f"Auto-advance on exact match.  "
                f"Scene Pts: {self._tr_scene_points}  "
                f"Light Pts: {self._tr_light_points}  "
                f"Scan Pts: {self._tr_scan_points}  "
                f"Sys Pts: {self._tr_system_points}"
            ),
            True,
            text_muted,
        )
        surface.blit(hint, (controls.x + 8, controls.y + 5))

    @staticmethod
    def _target_recognition_light_color(code: str) -> tuple[int, int, int]:
        key = str(code).strip().upper()
        if key == "G":
            return (42, 222, 68)
        if key == "B":
            return (64, 104, 242)
        if key == "Y":
            return (250, 214, 56)
        if key == "R":
            return (234, 72, 72)
        return (186, 190, 204)

    def _draw_target_recognition_info_legend(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        text = (220, 230, 246)
        muted = (156, 176, 206)
        row_h = max(14, rect.h // 4)
        y0 = rect.y + 2
        x_l = rect.x + 8
        x_r = rect.x + (rect.w // 2) + 2

        # Shape legend (top row).
        shape_defs = (
            ("Trucks", TargetRecognitionSceneEntity("truck", "friendly", False, False)),
            ("Tanks", TargetRecognitionSceneEntity("tank", "friendly", False, False)),
            ("Buildings", TargetRecognitionSceneEntity("building", "friendly", False, False)),
        )
        for idx, (label, entity) in enumerate(shape_defs):
            cy = y0 + 7
            self._draw_target_recognition_symbol(
                surface,
                entity=entity,
                cx=x_l + 4 + idx * max(44, rect.w // 3),
                cy=cy,
                size=6,
                color=(230, 230, 230, 255),
            )
            surf = self._tiny_font.render(label, True, text)
            surface.blit(surf, (x_l + 14 + idx * max(44, rect.w // 3), cy - 7))

        # Affiliation row.
        aff_defs = (
            ("Hostile", (226, 90, 92)),
            ("Friendly", (96, 176, 232)),
            ("Neutral", (214, 206, 88)),
        )
        for idx, (label, color) in enumerate(aff_defs):
            cy = y0 + row_h + 7
            sw = pygame.Rect(x_l + idx * max(56, rect.w // 3), cy - 5, 8, 8)
            pygame.draw.rect(surface, color, sw)
            surf = self._tiny_font.render(label, True, text)
            surface.blit(surf, (sw.right + 4, cy - 7))

        # Modifiers row.
        flags_y = y0 + (2 * row_h) + 6
        dmg = self._tiny_font.render("X Damaged", True, muted)
        pri = self._tiny_font.render("+- High Priority", True, muted)
        surface.blit(dmg, (x_l, flags_y))
        surface.blit(pri, (x_r - 8, flags_y))

        # Standalone filler symbols.
        bot_y = y0 + (3 * row_h) + 6
        self._draw_target_recognition_beacon(surface, cx=x_l + 5, cy=bot_y + 5, size=5, color=(226, 90, 92, 255))
        beacon = self._tiny_font.render("Beacon", True, muted)
        surface.blit(beacon, (x_l + 14, bot_y - 1))
        self._draw_target_recognition_unknown(surface, cx=x_r - 2, cy=bot_y + 5, size=5, color=(214, 206, 88, 255))
        unknown = self._tiny_font.render("Unknown", True, muted)
        surface.blit(unknown, (x_r + 8, bot_y - 1))

    def _draw_target_recognition_scene(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        payload: TargetRecognitionPayload,
    ) -> None:
        if rect.w <= 0 or rect.h <= 0:
            return

        seed = self._target_recognition_scene_seed(payload)
        if (
            self._tr_scene_base_cache is None
            or self._tr_scene_base_cache_size != (rect.w, rect.h)
            or self._tr_scene_base_cache_seed != seed
        ):
            self._tr_scene_base_cache = self._target_recognition_build_scene_base(rect.w, rect.h, seed)
            self._tr_scene_base_cache_size = (rect.w, rect.h)
            self._tr_scene_base_cache_seed = seed

        assert self._tr_scene_base_cache is not None
        scene = self._tr_scene_base_cache.copy()
        self._draw_target_recognition_scene_compass(scene)
        self._tr_scene_symbol_hitboxes = []

        for glyph_id in self._tr_scene_glyph_order:
            glyph = self._tr_scene_glyphs.get(glyph_id)
            if glyph is None:
                continue
            cx = int(glyph.nx * float(rect.w))
            cy = int(glyph.ny * float(rect.h))
            size = max(5, int(min(rect.w, rect.h) * glyph.scale))
            alpha = max(20, min(255, int(round(glyph.alpha))))
            if glyph.kind == "entity" and glyph.entity is not None:
                rc, gc, bc = self._target_recognition_affiliation_color(glyph.entity.affiliation)
                self._draw_target_recognition_symbol(
                    scene,
                    entity=glyph.entity,
                    cx=cx,
                    cy=cy,
                    size=size,
                    color=(rc, gc, bc, alpha),
                    heading=glyph.heading,
                )
            elif glyph.kind == "beacon":
                self._draw_target_recognition_beacon(
                    scene,
                    cx=cx,
                    cy=cy,
                    size=max(4, int(size * 0.68)),
                    color=(226, 90, 92, alpha),
                )
            else:
                self._draw_target_recognition_unknown(
                    scene,
                    cx=cx,
                    cy=cy,
                    size=max(4, int(size * 0.72)),
                    color=(214, 206, 88, alpha),
                )

            hit_r = max(8, int(size * 1.7))
            hit = pygame.Rect(rect.x + cx - hit_r, rect.y + cy - hit_r, hit_r * 2, hit_r * 2)
            self._tr_scene_symbol_hitboxes.append((hit, glyph_id))

        self._draw_target_recognition_clouds(
            scene,
            payload,
            phase_s=float(self._tr_scene_anim_frame) / 60.0,
        )

        surface.blit(scene, rect.topleft)
        pygame.draw.rect(surface, (78, 98, 138), rect, 1)

    def _target_recognition_build_scene_base(self, width: int, height: int, seed: int) -> pygame.Surface:
        base = pygame.Surface((width, height), pygame.SRCALPHA)
        rng = random.Random(seed ^ 0x7F4A7C15)
        base.fill((18, 34, 30, 255))

        for _ in range(24):
            cx = int(rng.uniform(0, width))
            cy = int(rng.uniform(0, height))
            radius = int(rng.uniform(max(36, width * 0.08), max(104, width * 0.25)))
            tone = int(rng.uniform(30, 78))
            alpha = int(rng.uniform(30, 74))
            pygame.draw.circle(base, (tone - 8, tone, tone - 11, alpha), (cx, cy), radius)
        return base

    @staticmethod
    def _draw_target_recognition_scene_compass(scene: pygame.Surface) -> None:
        ring_c = (212, 214, 220, 230)
        mark_c = (240, 102, 106, 230)
        txt_c = (220, 226, 238, 210)
        cx = 16
        cy = 16
        r = 11
        pygame.draw.circle(scene, (200, 206, 214, 65), (cx, cy), r + 2)
        pygame.draw.circle(scene, ring_c, (cx, cy), r, 1)
        pygame.draw.line(scene, mark_c, (cx, cy), (cx, cy - r + 2), 2)
        pygame.draw.line(scene, txt_c, (cx - 2, cy), (cx + 2, cy), 1)

    @staticmethod
    def _target_recognition_scene_seed(payload: TargetRecognitionPayload) -> int:
        # Stable seed per trial payload (do not use Python hash()).
        seed = 2166136261

        def mix(v: int) -> None:
            nonlocal seed
            seed ^= (v & 0xFFFFFFFF)
            seed = (seed * 16777619) & 0xFFFFFFFF

        mix(int(payload.scene_rows))
        mix(int(payload.scene_cols))
        for e in payload.scene_entities:
            for ch in f"{e.shape}:{e.affiliation}:{int(e.damaged)}:{int(e.high_priority)}":
                mix(ord(ch))
        for tok in payload.scene_cells:
            for ch in str(tok):
                mix(ord(ch))
        for opt in payload.scene_target_options:
            for ch in str(opt):
                mix(ord(ch))
        return seed

    def _draw_target_recognition_symbol(
        self,
        surface: pygame.Surface,
        entity: TargetRecognitionSceneEntity,
        *,
        cx: int,
        cy: int,
        size: int,
        color: tuple[int, int, int, int],
        heading: float = 0.0,
    ) -> None:
        line_w = 2 if size >= 8 else 1
        s = max(5, int(size))

        if entity.shape == "truck":
            pygame.draw.circle(surface, color, (cx, cy), s, line_w)
            dx = math.cos(heading)
            dy = math.sin(heading)
            ex = int(cx + dx * (s + 7))
            ey = int(cy + dy * (s + 7))
            pygame.draw.line(surface, color, (cx, cy), (ex, ey), line_w)
            px = -dy
            py = dx
            p1 = (ex, ey)
            p2 = (int(ex - dx * 5 + px * 3), int(ey - dy * 5 + py * 3))
            p3 = (int(ex - dx * 5 - px * 3), int(ey - dy * 5 - py * 3))
            pygame.draw.polygon(surface, color, (p1, p2, p3), line_w)
        elif entity.shape == "tank":
            box = pygame.Rect(cx - s, cy - s, s * 2, s * 2)
            pygame.draw.rect(surface, color, box, line_w)
            dx = math.cos(heading)
            dy = math.sin(heading)
            pygame.draw.line(
                surface,
                color,
                (int(cx - dx * (s + 3)), int(cy - dy * (s + 3))),
                (int(cx + dx * (s + 3)), int(cy + dy * (s + 3))),
                line_w,
            )
        elif entity.shape == "building":
            a0 = heading - (math.pi / 2.0)
            pts = (
                (int(cx + math.cos(a0) * (s + 1)), int(cy + math.sin(a0) * (s + 1))),
                (int(cx + math.cos(a0 + 2.12) * (s + 2)), int(cy + math.sin(a0 + 2.12) * (s + 2))),
                (int(cx + math.cos(a0 - 2.12) * (s + 2)), int(cy + math.sin(a0 - 2.12) * (s + 2))),
            )
            pygame.draw.polygon(surface, color, pts, line_w)
        else:
            points = []
            for i in range(6):
                ang = (math.tau * i) / 6.0
                points.append((int(cx + math.cos(ang) * s), int(cy + math.sin(ang) * s)))
            pygame.draw.polygon(surface, color, points, line_w)

        if entity.damaged:
            xw = max(1, line_w)
            pygame.draw.line(surface, color, (cx - s + 1, cy - s + 1), (cx + s - 1, cy + s - 1), xw)
            pygame.draw.line(surface, color, (cx + s - 1, cy - s + 1), (cx - s + 1, cy + s - 1), xw)
        if entity.high_priority:
            pw = max(1, line_w)
            pygame.draw.line(surface, color, (cx - s - 3, cy), (cx + s + 3, cy), pw)
            pygame.draw.line(surface, color, (cx, cy - s - 3), (cx, cy + s + 3), pw)

    @staticmethod
    def _draw_target_recognition_beacon(
        surface: pygame.Surface,
        *,
        cx: int,
        cy: int,
        size: int,
        color: tuple[int, int, int, int],
    ) -> None:
        s = max(3, int(size))
        box = pygame.Rect(cx - s, cy - s, s * 2, s * 2)
        pygame.draw.rect(surface, color, box)
        pygame.draw.rect(surface, (18, 22, 34, max(80, color[3])), box, 1)

    @staticmethod
    def _draw_target_recognition_unknown(
        surface: pygame.Surface,
        *,
        cx: int,
        cy: int,
        size: int,
        color: tuple[int, int, int, int],
    ) -> None:
        s = max(4, int(size))
        pts = ((cx, cy - s), (cx + s, cy), (cx, cy + s), (cx - s, cy))
        pygame.draw.polygon(surface, color, pts, 2)

    @staticmethod
    def _target_recognition_affiliation_color(affiliation: str) -> tuple[int, int, int]:
        key = str(affiliation).strip().lower()
        if key == "hostile":
            return (224, 88, 90)
        if key == "friendly":
            return (96, 176, 232)
        return (214, 206, 88)

    @staticmethod
    def _target_recognition_entity_from_code(code: str) -> TargetRecognitionSceneEntity:
        text = str(code).upper()
        shape_code = text
        side = "N"
        flags = ""
        if ":" in text:
            shape_code, rest = text.split(":", 1)
            if rest:
                side = rest[0]
                flags = rest[1:]
        shape = {
            "TRK": "truck",
            "TNK": "tank",
            "BLD": "building",
        }.get(shape_code, "truck")
        affiliation = {
            "H": "hostile",
            "F": "friendly",
            "N": "neutral",
        }.get(side, "neutral")
        return TargetRecognitionSceneEntity(
            shape=shape,
            affiliation=affiliation,
            damaged=("D" in flags),
            high_priority=("P" in flags),
        )

    def _draw_target_recognition_clouds(
        self,
        scene: pygame.Surface,
        payload: TargetRecognitionPayload,
        *,
        phase_s: float,
    ) -> None:
        w, h = scene.get_size()
        seed = self._target_recognition_scene_seed(payload) ^ 0x9E3779B9
        rng = random.Random(seed)
        t = max(0.0, float(phase_s))

        # Bright haze base.
        haze = pygame.Surface((w, h), pygame.SRCALPHA)
        haze.fill((182, 188, 184, 20))

        for _ in range(18):
            bx = rng.uniform(0, w)
            by = rng.uniform(0, h)
            radius = rng.uniform(max(46, w * 0.08), max(120, w * 0.24))
            phase = rng.uniform(0.0, math.tau)
            speed = rng.uniform(0.05, 0.14)
            drift = rng.uniform(8.0, 28.0)
            cx = bx + math.sin((t * speed) + phase) * drift
            cy = by + math.cos((t * speed * 0.8) + phase * 1.3) * drift

            for i in range(3):
                rr = int(radius * (1.0 - i * 0.22))
                alpha = max(22, int(76 - i * 18))
                shade = 184 - (i * 6)
                pygame.draw.circle(haze, (shade, shade + 6, shade, alpha), (int(cx), int(cy)), rr)
                pygame.draw.circle(
                    haze,
                    (shade, shade + 4, shade, max(18, alpha - 20)),
                    (int(cx + rr * 0.34), int(cy - rr * 0.18)),
                    int(rr * 0.72),
                )

        # Dark cloud pockets to block symbol visibility.
        for _ in range(12):
            bx = rng.uniform(0, w)
            by = rng.uniform(0, h)
            radius = rng.uniform(max(34, w * 0.07), max(96, w * 0.16))
            phase = rng.uniform(0.0, math.tau)
            speed = rng.uniform(0.04, 0.11)
            drift = rng.uniform(6.0, 20.0)
            cx = bx + math.sin((t * speed) + phase) * drift
            cy = by + math.cos((t * speed * 0.9) + phase * 0.9) * drift
            pygame.draw.circle(haze, (16, 22, 20, 72), (int(cx), int(cy)), int(radius))
            pygame.draw.circle(haze, (10, 14, 12, 58), (int(cx + radius * 0.2), int(cy - radius * 0.1)), int(radius * 0.68))

        scene.blit(haze, (0, 0))

    def _target_recognition_reset_scene_subtask(self) -> None:
        self._tr_scene_payload_id = None
        self._tr_scene_rng = None
        self._tr_scene_glyphs = {}
        self._tr_scene_glyph_order = []
        self._tr_scene_symbol_hitboxes = []
        self._tr_scene_next_glyph_id = 1
        self._tr_scene_target_queue = []
        self._tr_scene_active_targets = []
        self._tr_scene_next_target_add_ms = 0
        self._tr_scene_had_active_targets = False
        self._tr_scene_complete = False
        self._tr_scene_points = 0
        self._tr_scene_hits = 0
        self._tr_scene_misses = 0
        self._tr_scene_beacon_hits = 0
        self._tr_scene_unknown_hits = 0
        self._tr_scene_anim_frame = 0
        self._tr_scene_last_update_ms = 0
        self._tr_scene_base_cache = None
        self._tr_scene_base_cache_size = (0, 0)
        self._tr_scene_base_cache_seed = 0

    def _target_recognition_sync_scene_stream(self, payload: TargetRecognitionPayload) -> None:
        now_ms = pygame.time.get_ticks()
        pid = id(payload)
        if self._tr_scene_payload_id != pid:
            self._tr_scene_payload_id = pid
            self._tr_scene_rng = random.Random(self._target_recognition_scene_seed(payload) ^ 0xC0FFEE17)
            self._tr_scene_glyphs = {}
            self._tr_scene_glyph_order = []
            self._tr_scene_next_glyph_id = 1
            self._tr_scene_symbol_hitboxes = []
            self._tr_scene_target_queue = list(payload.scene_target_options) if payload.scene_has_target else []
            self._tr_scene_active_targets = []
            self._tr_scene_next_target_add_ms = now_ms + 1200
            self._tr_scene_had_active_targets = False
            self._tr_scene_complete = False
            self._tr_scene_anim_frame = 0
            self._tr_scene_last_update_ms = now_ms
            self._tr_scene_base_cache = None
            self._tr_scene_base_cache_size = (0, 0)
            self._tr_scene_base_cache_seed = 0

            scene_entities = payload.scene_entities
            if not scene_entities:
                scene_entities = tuple(
                    self._target_recognition_entity_from_code(code) for code in payload.scene_cells
                )
            rows = max(1, int(payload.scene_rows))
            cols = max(1, int(payload.scene_cols))
            max_items = min(rows * cols, len(scene_entities))
            assert self._tr_scene_rng is not None
            for idx in range(max_items):
                rr = idx // cols
                cc = idx % cols
                nx = (cc + 0.5) / float(cols)
                ny = (rr + 0.5) / float(rows)
                nx += (self._tr_scene_rng.random() - 0.5) * 0.08
                ny += (self._tr_scene_rng.random() - 0.5) * 0.08
                nx = max(0.04, min(0.96, nx))
                ny = max(0.04, min(0.96, ny))
                if self._target_recognition_scene_in_compass_zone(nx, ny):
                    ny = min(0.96, ny + 0.16)
                entity = scene_entities[idx]
                labels = self._target_recognition_scene_matching_labels(
                    entity=entity,
                    labels=payload.scene_target_options,
                )
                glyph_id = self._tr_scene_next_glyph_id
                self._tr_scene_next_glyph_id += 1
                self._tr_scene_glyphs[glyph_id] = _TargetRecognitionSceneGlyph(
                    glyph_id=glyph_id,
                    kind="entity",
                    entity=entity,
                    nx=nx,
                    ny=ny,
                    scale=float(self._tr_scene_rng.uniform(0.010, 0.024)),
                    heading=float(self._tr_scene_rng.uniform(0.0, math.tau)),
                    alpha=float(self._tr_scene_rng.uniform(88.0, 160.0)),
                    max_alpha=float(self._tr_scene_rng.uniform(128.0, 190.0)),
                    matching_labels=labels,
                )
                self._tr_scene_glyph_order.append(glyph_id)

            filler_count = max(2, min(5, (rows * cols) // 20))
            for _ in range(filler_count):
                self._target_recognition_scene_add_filler_glyph(kind="beacon")
                self._target_recognition_scene_add_filler_glyph(kind="unknown")

        if self._tr_scene_payload_id != pid:
            return
        dt_ms = max(0, min(120, now_ms - self._tr_scene_last_update_ms))
        self._tr_scene_last_update_ms = now_ms
        self._tr_scene_anim_frame += 1
        if dt_ms > 0:
            fade = float(dt_ms) * 0.040
            for glyph in self._tr_scene_glyphs.values():
                if glyph.alpha < glyph.max_alpha:
                    glyph.alpha = min(glyph.max_alpha, glyph.alpha + fade)

        if payload.scene_has_target and (not self._tr_scene_complete):
            while self._tr_scene_target_queue and now_ms >= self._tr_scene_next_target_add_ms:
                if len(self._tr_scene_active_targets) >= 3:
                    break
                nxt = self._tr_scene_target_queue.pop(0)
                if nxt not in self._tr_scene_active_targets:
                    self._tr_scene_active_targets.append(nxt)
                    self._tr_scene_had_active_targets = True
                assert self._tr_scene_rng is not None
                self._tr_scene_next_target_add_ms += int(round(self._tr_scene_rng.uniform(1600.0, 2800.0)))

            self._target_recognition_scene_prune_completed_targets()

    def _target_recognition_scene_prune_completed_targets(self) -> None:
        if not self._tr_scene_active_targets:
            return
        active: list[str] = []
        for label in self._tr_scene_active_targets:
            has_any = any(
                glyph.kind == "entity" and label in glyph.matching_labels
                for glyph in self._tr_scene_glyphs.values()
            )
            if has_any:
                active.append(label)
        self._tr_scene_active_targets = active
        if not self._tr_scene_active_targets and self._tr_scene_had_active_targets:
            self._tr_scene_complete = True
            self._tr_scene_target_queue.clear()
            self._tr_selected_panels.add("scene")

    def _target_recognition_handle_scene_press(
        self,
        payload: TargetRecognitionPayload,
        *,
        glyph_id: int,
    ) -> bool:
        glyph = self._tr_scene_glyphs.get(int(glyph_id))
        if glyph is None:
            return False

        if glyph.kind == "beacon":
            self._tr_scene_points += 1
            self._tr_scene_beacon_hits += 1
            self._target_recognition_scene_reseed_glyph(payload, glyph, kind="beacon", force_non_target=True)
            return True
        if glyph.kind == "unknown":
            self._tr_scene_points += 1
            self._tr_scene_unknown_hits += 1
            self._target_recognition_scene_reseed_glyph(payload, glyph, kind="unknown", force_non_target=True)
            return True

        active = set(self._tr_scene_active_targets)
        hit_target = bool(active.intersection(glyph.matching_labels))
        if hit_target:
            self._tr_scene_points += 1
            self._tr_scene_hits += 1
            self._target_recognition_scene_reseed_glyph(payload, glyph, kind="entity", force_non_target=True)
            self._target_recognition_scene_prune_completed_targets()
            return True

        self._tr_scene_points -= 1
        self._tr_scene_misses += 1
        return False

    def _target_recognition_scene_reseed_glyph(
        self,
        payload: TargetRecognitionPayload,
        glyph: _TargetRecognitionSceneGlyph,
        *,
        kind: str,
        force_non_target: bool,
    ) -> None:
        if self._tr_scene_rng is None:
            self._tr_scene_rng = random.Random(self._target_recognition_scene_seed(payload) ^ 0xC0FFEE17)

        glyph.kind = kind
        glyph.nx, glyph.ny = self._target_recognition_scene_random_position()
        glyph.scale = float(self._tr_scene_rng.uniform(0.010, 0.024))
        glyph.heading = float(self._tr_scene_rng.uniform(0.0, math.tau))
        glyph.alpha = 0.0
        glyph.max_alpha = float(self._tr_scene_rng.uniform(128.0, 186.0))
        glyph.matching_labels = ()
        glyph.entity = None

        if kind != "entity":
            return

        active = set(self._tr_scene_active_targets)
        for _ in range(72):
            candidate = TargetRecognitionSceneEntity(
                shape=str(self._tr_scene_rng.choice(("truck", "tank", "building"))),
                affiliation=str(self._tr_scene_rng.choice(("hostile", "friendly", "neutral"))),
                damaged=bool(self._tr_scene_rng.random() < 0.36),
                high_priority=bool(self._tr_scene_rng.random() < 0.28),
            )
            labels = self._target_recognition_scene_matching_labels(
                entity=candidate,
                labels=payload.scene_target_options,
            )
            if force_non_target and active.intersection(labels):
                continue
            glyph.entity = candidate
            glyph.matching_labels = labels
            return

        glyph.entity = TargetRecognitionSceneEntity("truck", "neutral", False, False)
        glyph.matching_labels = ()

    def _target_recognition_scene_add_filler_glyph(self, *, kind: str) -> None:
        if self._tr_scene_rng is None:
            return
        glyph_id = self._tr_scene_next_glyph_id
        self._tr_scene_next_glyph_id += 1
        nx, ny = self._target_recognition_scene_random_position()
        self._tr_scene_glyphs[glyph_id] = _TargetRecognitionSceneGlyph(
            glyph_id=glyph_id,
            kind=kind,
            entity=None,
            nx=nx,
            ny=ny,
            scale=float(self._tr_scene_rng.uniform(0.008, 0.016)),
            heading=0.0,
            alpha=float(self._tr_scene_rng.uniform(120.0, 188.0)),
            max_alpha=float(self._tr_scene_rng.uniform(140.0, 196.0)),
            matching_labels=(),
        )
        self._tr_scene_glyph_order.append(glyph_id)

    def _target_recognition_scene_random_position(self) -> tuple[float, float]:
        assert self._tr_scene_rng is not None
        for _ in range(48):
            nx = float(self._tr_scene_rng.uniform(0.04, 0.96))
            ny = float(self._tr_scene_rng.uniform(0.05, 0.96))
            if not self._target_recognition_scene_in_compass_zone(nx, ny):
                return nx, ny
        return 0.5, 0.52

    @staticmethod
    def _target_recognition_scene_in_compass_zone(nx: float, ny: float) -> bool:
        dx = nx - 0.055
        dy = ny - 0.065
        return (dx * dx + dy * dy) <= (0.058 * 0.058)

    def _target_recognition_scene_matching_labels(
        self,
        *,
        entity: TargetRecognitionSceneEntity,
        labels: tuple[str, ...],
    ) -> tuple[str, ...]:
        return tuple(label for label in labels if self._target_recognition_scene_label_matches(entity, label))

    @staticmethod
    def _target_recognition_scene_label_matches(entity: TargetRecognitionSceneEntity, label: str) -> bool:
        txt = str(label).upper()
        if "UNKNOWN" in txt or "BEACON" in txt:
            return False
        if "TRUCK" in txt:
            shape = "truck"
        elif "TANK" in txt:
            shape = "tank"
        elif "BUILDING" in txt:
            shape = "building"
        else:
            return False
        if "HOSTILE" in txt:
            affiliation = "hostile"
        elif "FRIENDLY" in txt:
            affiliation = "friendly"
        elif "NEUTRAL" in txt:
            affiliation = "neutral"
        else:
            return False
        if entity.shape != shape or entity.affiliation != affiliation:
            return False
        if "DAMAGED" in txt and not entity.damaged:
            return False
        if "HP" in txt and not entity.high_priority:
            return False
        return True

    def _target_recognition_sync_selection(self, payload: TargetRecognitionPayload) -> None:
        pid = id(payload)
        if self._tr_selection_payload_id == pid:
            return
        self._tr_selection_payload_id = pid
        self._tr_selected_panels.clear()
        self._input = ""

    def _target_recognition_reset_light_subtask(self) -> None:
        self._tr_light_payload_id = None
        self._tr_light_rng = None
        self._tr_light_next_change_ms = 0
        self._tr_light_current_pattern = ("G", "G", "G")
        self._tr_light_target_pattern_live = ("G", "G", "G")
        self._tr_light_points = 0
        self._tr_light_hits = 0
        self._tr_light_early_presses = 0
        self._tr_light_button_hitbox = None

    def _target_recognition_reset_scan_subtask(self) -> None:
        self._tr_scan_payload_id = None
        self._tr_scan_rng = None
        self._tr_scan_token_pool = ("<>", "[]", "/\\", "\\/")
        self._tr_scan_next_change_ms = 0
        self._tr_scan_current_pattern = ("<>", "[]", "/\\", "\\/")
        self._tr_scan_target_pattern_live = ("<>", "[]", "/\\", "\\/")
        self._tr_scan_points = 0
        self._tr_scan_hits = 0
        self._tr_scan_early_presses = 0
        self._tr_scan_button_hitbox = None
        self._tr_scan_reveal_index = 3
        self._tr_scan_next_step_ms = 0
        self._tr_scan_passes_left = 0

    def _target_recognition_sync_light_stream(self, payload: TargetRecognitionPayload) -> None:
        now_ms = pygame.time.get_ticks()
        pid = id(payload)
        if self._tr_light_payload_id != pid:
            self._tr_light_payload_id = pid
            self._tr_light_rng = random.Random(self._target_recognition_light_seed(payload))
            self._tr_light_current_pattern = self._target_recognition_light_triplet(payload.light_pattern)
            self._tr_light_target_pattern_live = self._target_recognition_light_triplet(payload.light_target_pattern)
            self._tr_light_next_change_ms = now_ms + self._target_recognition_light_interval_ms()

        while now_ms >= self._tr_light_next_change_ms:
            assert self._tr_light_rng is not None
            if self._tr_light_rng.random() < 0.38:
                self._tr_light_current_pattern = self._tr_light_target_pattern_live
            else:
                self._tr_light_current_pattern = self._target_recognition_next_light_pattern(
                    exclude=self._tr_light_target_pattern_live
                )
            self._tr_light_next_change_ms += self._target_recognition_light_interval_ms()

    def _target_recognition_handle_light_press(self, payload: TargetRecognitionPayload) -> bool:
        self._target_recognition_sync_light_stream(payload)
        if self._tr_light_current_pattern == self._tr_light_target_pattern_live:
            self._tr_light_points += 1
            self._tr_light_hits += 1
            self._tr_light_target_pattern_live = self._target_recognition_next_light_pattern(
                exclude=self._tr_light_current_pattern
            )
            return True

        self._tr_light_points -= 1
        self._tr_light_early_presses += 1
        return False

    def _target_recognition_sync_scan_stream(self, payload: TargetRecognitionPayload) -> None:
        now_ms = pygame.time.get_ticks()
        pid = id(payload)
        if self._tr_scan_payload_id != pid:
            self._tr_scan_payload_id = pid
            self._tr_scan_rng = random.Random(self._target_recognition_scan_seed(payload))
            pool = tuple(dict.fromkeys(str(tok) for tok in payload.scan_tokens))
            if payload.scan_target and str(payload.scan_target) not in pool:
                pool = (*pool, str(payload.scan_target))
            if len(pool) < 4:
                pool = ("<>", "<|", "|>", "[]", "{}", "()", "/\\", "\\/", "==", "=~")
            self._tr_scan_token_pool = tuple(pool)
            self._tr_scan_target_pattern_live = self._target_recognition_next_scan_pattern(exclude=None)
            self._tr_scan_current_pattern = self._target_recognition_next_scan_pattern(
                exclude=self._tr_scan_target_pattern_live
            )
            self._tr_scan_next_change_ms = now_ms + self._target_recognition_scan_interval_ms()
            self._tr_scan_reveal_index = 3
            self._tr_scan_next_step_ms = now_ms + 1000
            self._tr_scan_passes_left = self._target_recognition_scan_repeat_count()

        while now_ms >= self._tr_scan_next_change_ms:
            assert self._tr_scan_rng is not None
            if self._tr_scan_rng.random() < 0.34:
                self._tr_scan_current_pattern = self._tr_scan_target_pattern_live
            else:
                self._tr_scan_current_pattern = self._target_recognition_next_scan_pattern(
                    exclude=self._tr_scan_target_pattern_live
                )
            self._tr_scan_next_change_ms += self._target_recognition_scan_interval_ms()
            self._tr_scan_passes_left = self._target_recognition_scan_repeat_count()

        while now_ms >= self._tr_scan_next_step_ms:
            self._tr_scan_next_step_ms += 1000
            if self._tr_scan_reveal_index > 0:
                self._tr_scan_reveal_index -= 1
                continue

            self._tr_scan_reveal_index = 3
            self._tr_scan_passes_left -= 1
            if self._tr_scan_passes_left <= 0:
                assert self._tr_scan_rng is not None
                if self._tr_scan_rng.random() < 0.34:
                    self._tr_scan_current_pattern = self._tr_scan_target_pattern_live
                else:
                    self._tr_scan_current_pattern = self._target_recognition_next_scan_pattern(
                        exclude=self._tr_scan_target_pattern_live
                    )
                self._tr_scan_passes_left = self._target_recognition_scan_repeat_count()

    def _target_recognition_handle_scan_press(self, payload: TargetRecognitionPayload) -> bool:
        self._target_recognition_sync_scan_stream(payload)
        if self._tr_scan_current_pattern == self._tr_scan_target_pattern_live:
            self._tr_scan_points += 1
            self._tr_scan_hits += 1
            self._tr_scan_target_pattern_live = self._target_recognition_next_scan_pattern(
                exclude=self._tr_scan_current_pattern
            )
            return True

        self._tr_scan_points -= 1
        self._tr_scan_early_presses += 1
        return False

    def _target_recognition_next_scan_pattern(
        self,
        *,
        exclude: tuple[str, str, str, str] | None,
    ) -> tuple[str, str, str, str]:
        if self._tr_scan_rng is None:
            self._tr_scan_rng = random.Random(0)
        pool = self._tr_scan_token_pool or ("<>", "[]", "/\\", "\\/")
        for _ in range(64):
            cand = (
                self._target_recognition_scan_symbol(pool),
                self._target_recognition_scan_symbol(pool),
                self._target_recognition_scan_symbol(pool),
                self._target_recognition_scan_symbol(pool),
            )
            if exclude is None or cand != exclude:
                return cand
        if exclude is None:
            return (
                f"{pool[0]}{pool[1 % len(pool)]}",
                f"{pool[1 % len(pool)]}{pool[2 % len(pool)]}",
                f"{pool[2 % len(pool)]}{pool[0]}",
                f"{pool[0]}{pool[3 % len(pool)]}",
            )
        return (
            f"{pool[0]}{pool[1 % len(pool)]}",
            f"{pool[1 % len(pool)]}{pool[0]}",
            f"{pool[0]}{pool[0]}",
            f"{pool[2 % len(pool)]}{pool[3 % len(pool)]}",
        )

    def _target_recognition_scan_symbol(self, pool: tuple[str, ...]) -> str:
        assert self._tr_scan_rng is not None
        a = str(self._tr_scan_rng.choice(pool))
        b = str(self._tr_scan_rng.choice(pool))
        return f"{a}{b}"

    def _target_recognition_scan_repeat_count(self) -> int:
        if self._tr_scan_rng is None:
            return 3
        return int(self._tr_scan_rng.randint(2, 4))

    def _target_recognition_scan_interval_ms(self) -> int:
        if self._tr_scan_rng is None:
            return 6500
        return int(round(self._tr_scan_rng.uniform(5.0, 10.0) * 1000.0))

    @staticmethod
    def _target_recognition_scan_seed(payload: TargetRecognitionPayload) -> int:
        seed = 2166136261

        def mix(v: int) -> None:
            nonlocal seed
            seed ^= (v & 0xFFFFFFFF)
            seed = (seed * 16777619) & 0xFFFFFFFF

        for token in (
            payload.scan_target,
            payload.system_target,
            payload.scene_target,
            *payload.scan_tokens,
        ):
            for ch in str(token):
                mix(ord(ch))
        return seed ^ 0x13579BDF

    def _target_recognition_next_light_pattern(
        self,
        *,
        exclude: tuple[str, str, str] | None = None,
    ) -> tuple[str, str, str]:
        if self._tr_light_rng is None:
            self._tr_light_rng = random.Random(0)
        colors = ("G", "B", "Y", "R")
        for _ in range(48):
            cand = (
                str(self._tr_light_rng.choice(colors)),
                str(self._tr_light_rng.choice(colors)),
                str(self._tr_light_rng.choice(colors)),
            )
            if exclude is None or cand != exclude:
                return cand
        if exclude is None:
            return ("R", "G", "B")
        return ("R" if exclude[0] != "R" else "G", exclude[1], exclude[2])

    def _target_recognition_light_interval_ms(self) -> int:
        if self._tr_light_rng is None:
            return 7500
        return int(round(self._tr_light_rng.uniform(5.0, 10.0) * 1000.0))

    @staticmethod
    def _target_recognition_light_triplet(pattern: tuple[str, ...]) -> tuple[str, str, str]:
        vals = [str(v).strip().upper()[:1] for v in pattern]
        vals = [v if v in ("G", "B", "Y", "R") else "G" for v in vals]
        while len(vals) < 3:
            vals.append("G")
        return (vals[0], vals[1], vals[2])

    @staticmethod
    def _target_recognition_light_seed(payload: TargetRecognitionPayload) -> int:
        # Stable per-trial seed for light cadence and target switching.
        seed = 2166136261

        def mix(v: int) -> None:
            nonlocal seed
            seed ^= (v & 0xFFFFFFFF)
            seed = (seed * 16777619) & 0xFFFFFFFF

        mix(int(payload.scene_rows))
        mix(int(payload.scene_cols))
        for token in (
            *payload.light_pattern,
            *payload.light_target_pattern,
            payload.scan_target,
            payload.system_target,
            payload.scene_target,
        ):
            for ch in str(token):
                mix(ord(ch))
        return seed ^ 0xA5A55A5A

    def _target_recognition_reset_system_subtask(self) -> None:
        self._tr_system_payload_id = None
        self._tr_system_rng = None
        self._tr_system_columns = [[], [], []]
        self._tr_system_row_offset = 0
        self._tr_system_row_frac = 0.0
        self._tr_system_target_code = "----"
        self._tr_system_step_interval_ms = 1700
        self._tr_system_last_step_ms = 0
        self._tr_system_points = 0
        self._tr_system_hits = 0
        self._tr_system_string_hitboxes = []

    def _target_recognition_sync_system_stream(self, payload: TargetRecognitionPayload) -> None:
        now_ms = pygame.time.get_ticks()
        pid = id(payload)
        if self._tr_system_payload_id != pid:
            self._tr_system_payload_id = pid
            self._tr_system_rng = random.Random(self._target_recognition_system_seed(payload))
            self._tr_system_columns = self._target_recognition_build_system_columns(payload)
            row_count = max(1, len(self._tr_system_columns[0])) if self._tr_system_columns else 1
            self._tr_system_row_offset = 0
            self._tr_system_row_frac = 0.0
            self._tr_system_last_step_ms = now_ms
            assert self._tr_system_rng is not None
            self._tr_system_step_interval_ms = int(round(self._tr_system_rng.uniform(1450.0, 2300.0)))
            self._tr_system_target_code = self._target_recognition_pick_initial_system_target(payload)
            if row_count <= 0:
                return

        step = max(1000, int(self._tr_system_step_interval_ms))
        row_count = max(1, len(self._tr_system_columns[0])) if self._tr_system_columns else 1
        while now_ms - self._tr_system_last_step_ms >= step:
            self._tr_system_last_step_ms += step
            self._tr_system_row_offset = (self._tr_system_row_offset + 1) % row_count
        self._tr_system_row_frac = max(
            0.0,
            min(1.0, float(now_ms - self._tr_system_last_step_ms) / float(step)),
        )

    def _target_recognition_build_system_columns(
        self,
        payload: TargetRecognitionPayload,
    ) -> list[list[str]]:
        cols: list[list[str]] = []
        if payload.system_cycles and payload.system_cycles[0].columns:
            source_cols = payload.system_cycles[0].columns
            for col in source_cols[:3]:
                cols.append([str(v) for v in col])
        else:
            base = [str(v) for v in payload.system_rows]
            if not base:
                base = ["A1B2", "C3D4", "E5F6", "G7H8", "J9K1", "L2M3"]
            while len(cols) < 3:
                cols.append(list(base))

        while len(cols) < 3:
            cols.append(list(cols[-1] if cols else ["A1B2", "C3D4", "E5F6", "G7H8", "J9K1", "L2M3"]))

        max_len = max(6, max((len(c) for c in cols), default=6))
        for idx, col in enumerate(cols):
            if not col:
                col = ["A1B2", "C3D4", "E5F6", "G7H8", "J9K1", "L2M3"]
            while len(col) < max_len:
                col.append(col[len(col) % max(1, len(col))])
            cols[idx] = col[:max_len]
        return cols[:3]

    def _target_recognition_pick_initial_system_target(self, payload: TargetRecognitionPayload) -> str:
        pool = [code for col in self._tr_system_columns for code in col]
        if payload.system_target and str(payload.system_target) in pool:
            return str(payload.system_target)
        if not pool:
            return "A1B2"
        assert self._tr_system_rng is not None
        return str(self._tr_system_rng.choice(pool))

    def _target_recognition_handle_system_press(
        self,
        payload: TargetRecognitionPayload,
        *,
        clicked_code: str,
    ) -> bool:
        self._target_recognition_sync_system_stream(payload)
        if str(clicked_code) != str(self._tr_system_target_code):
            return False
        self._tr_system_points += 1
        self._tr_system_hits += 1
        next_target = self._target_recognition_pick_next_system_target()
        self._tr_system_target_code = next_target
        return True

    def _target_recognition_pick_next_system_target(self) -> str:
        pool = [code for col in self._tr_system_columns for code in col if code != self._tr_system_target_code]
        if not pool:
            return self._tr_system_target_code
        assert self._tr_system_rng is not None
        return str(self._tr_system_rng.choice(pool))

    @staticmethod
    def _target_recognition_system_seed(payload: TargetRecognitionPayload) -> int:
        seed = 2166136261

        def mix(v: int) -> None:
            nonlocal seed
            seed ^= (v & 0xFFFFFFFF)
            seed = (seed * 16777619) & 0xFFFFFFFF

        for token in (
            payload.system_target,
            payload.scene_target,
            payload.scan_target,
            *payload.light_pattern,
            *payload.light_target_pattern,
        ):
            for ch in str(token):
                mix(ord(ch))
        for cycle in payload.system_cycles:
            for col in cycle.columns:
                for row in col:
                    for ch in str(row):
                        mix(ord(ch))
        return seed ^ 0x5A5AA55A

    def _target_recognition_reset_practice_breakdown(self) -> None:
        self._tr_practice_trials = 0
        self._tr_practice_panel_correct = {
            "scene": 0,
            "light": 0,
            "scan": 0,
            "system": 0,
        }
        self._tr_practice_panel_present = {
            "scene": 0,
            "light": 0,
            "scan": 0,
            "system": 0,
        }
        self._tr_practice_panel_hits = {
            "scene": 0,
            "light": 0,
            "scan": 0,
            "system": 0,
        }

    def _target_recognition_record_practice_trial(
        self,
        *,
        selected: set[str],
        expected: set[str],
    ) -> None:
        self._tr_practice_trials += 1
        for panel in ("scene", "light", "scan", "system"):
            exp = panel in expected
            sel = panel in selected
            if exp:
                self._tr_practice_panel_present[panel] += 1
            if exp and sel:
                self._tr_practice_panel_hits[panel] += 1
            if exp == sel:
                self._tr_practice_panel_correct[panel] += 1

    def _target_recognition_practice_breakdown_lines(self) -> tuple[str, ...]:
        trials = max(0, int(self._tr_practice_trials))
        if trials == 0:
            return ("No practice data recorded.",)

        labels = (
            ("scene", "Map"),
            ("light", "Light"),
            ("scan", "Scan"),
            ("system", "System"),
        )
        lines: list[str] = []
        for key, label in labels:
            correct = int(self._tr_practice_panel_correct.get(key, 0))
            present = int(self._tr_practice_panel_present.get(key, 0))
            hits = int(self._tr_practice_panel_hits.get(key, 0))
            acc = (correct / float(trials)) * 100.0
            if present > 0:
                lines.append(
                    f"{label}: {correct}/{trials} ({acc:.0f}%)  Hits {hits}/{present}"
                )
            else:
                lines.append(f"{label}: {correct}/{trials} ({acc:.0f}%)  Hits n/a")
        return tuple(lines)

    @staticmethod
    def _target_recognition_expected_panels(payload: TargetRecognitionPayload) -> set[str]:
        expected: set[str] = set()
        if payload.scene_has_target:
            expected.add("scene")
        if payload.light_has_target:
            expected.add("light")
        if payload.scan_has_target:
            expected.add("scan")
        if payload.system_has_target:
            expected.add("system")
        return expected

    def _target_recognition_system_view(
        self,
        payload: TargetRecognitionPayload,
    ) -> tuple[str, tuple[tuple[str, ...], ...], float]:
        self._target_recognition_sync_system_stream(payload)
        columns = self._tr_system_columns
        if not columns:
            return payload.system_target, (payload.system_rows,), 0.0

        row_offset = self._tr_system_row_offset
        step_frac = self._tr_system_row_frac
        rotated_cols: list[tuple[str, ...]] = []
        for col in columns:
            if not col:
                rotated_cols.append(())
                continue
            n = len(col)
            rotated = tuple(col[(row - row_offset) % n] for row in range(n))
            rotated_cols.append(rotated)
        return self._tr_system_target_code, tuple(rotated_cols), step_frac

    def _render_visual_search_question(
        self,
        surface: pygame.Surface,
        snap: TestSnapshot,
        payload: VisualSearchPayload,
    ) -> None:
        w, h = surface.get_size()
        bg = (2, 8, 114)
        frame_border = (232, 240, 255)
        text_main = (238, 245, 255)
        text_muted = (188, 204, 228)
        panel_bg = (10, 18, 92)

        surface.fill(bg)
        margin = max(8, min(18, w // 48))
        frame = pygame.Rect(margin, margin, w - margin * 2, h - margin * 2)
        pygame.draw.rect(surface, bg, frame)
        pygame.draw.rect(surface, frame_border, frame, 1)

        header_h = max(28, min(36, h // 16))
        header = pygame.Rect(frame.x + 1, frame.y + 1, frame.w - 2, header_h)
        pygame.draw.rect(surface, bg, header)
        pygame.draw.line(surface, frame_border, (header.x, header.bottom), (header.right, header.bottom), 1)

        title = self._tiny_font.render("Visual Search Test", True, text_main)
        surface.blit(title, title.get_rect(midleft=(header.x + 10, header.centery)))

        if snap.time_remaining_s is not None:
            rem = int(round(snap.time_remaining_s))
            mm = rem // 60
            ss = rem % 60
            timer = self._tiny_font.render(f"{mm:02d}:{ss:02d}", True, text_main)
            surface.blit(timer, timer.get_rect(midright=(header.right - 10, header.centery)))

        info = pygame.Rect(frame.x + 12, header.bottom + 8, frame.w - 24, 40)
        pygame.draw.rect(surface, panel_bg, info)
        pygame.draw.rect(surface, frame_border, info, 1)
        surface.blit(
            self._tiny_font.render(f"Target: {payload.target}", True, text_main),
            (info.x + 10, info.y + 6),
        )
        surface.blit(
            self._tiny_font.render(
                f"Scored {snap.correct_scored}/{snap.attempted_scored}",
                True,
                text_muted,
            ),
            (info.x + 10, info.y + 20),
        )

        grid_panel = pygame.Rect(frame.x + 12, info.bottom + 8, frame.w - 24, frame.h - header_h - 116)
        pygame.draw.rect(surface, panel_bg, grid_panel)
        pygame.draw.rect(surface, frame_border, grid_panel, 1)

        grid_rect = grid_panel.inflate(-12, -12)
        pygame.draw.rect(surface, (248, 250, 255), grid_rect)
        pygame.draw.rect(surface, (180, 192, 220), grid_rect, 1)

        rows = max(1, int(payload.rows))
        cols = max(1, int(payload.cols))
        cell_size = min(max(20, grid_rect.w // cols), max(20, grid_rect.h // rows))
        grid_w = cols * cell_size
        grid_h = rows * cell_size
        start_x = grid_rect.x + max(0, (grid_rect.w - grid_w) // 2)
        start_y = grid_rect.y + max(0, (grid_rect.h - grid_h) // 2)

        token_font = pygame.font.Font(None, max(16, min(30, int(cell_size * 0.44))))
        code_font = pygame.font.Font(None, max(12, min(17, int(cell_size * 0.24))))

        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                token = payload.cells[idx] if idx < len(payload.cells) else ""
                code = payload.cell_codes[idx] if idx < len(payload.cell_codes) else 0
                cell = pygame.Rect(start_x + c * cell_size, start_y + r * cell_size, cell_size, cell_size)
                pygame.draw.rect(surface, (244, 247, 255), cell)
                pygame.draw.rect(surface, (162, 176, 208), cell, 1)

                code_surf = code_font.render(str(code), True, (80, 96, 132))
                surface.blit(code_surf, (cell.x + 2, cell.y + 1))
                token_surface = token_font.render(str(token), True, (20, 24, 34))
                surface.blit(token_surface, token_surface.get_rect(center=cell.center))

        footer = pygame.Rect(frame.x + 12, frame.bottom - 52, frame.w - 24, 34)
        pygame.draw.rect(surface, panel_bg, footer)
        pygame.draw.rect(surface, frame_border, footer, 1)

        answer_label = self._tiny_font.render("Answer:", True, text_muted)
        surface.blit(answer_label, (footer.x + 10, footer.y + 10))
        answer_box = pygame.Rect(footer.x + 60, footer.y + 5, 140, 24)
        pygame.draw.rect(surface, (0, 0, 0), answer_box)
        pygame.draw.rect(surface, (152, 170, 208), answer_box, 1)
        caret = "|" if (pygame.time.get_ticks() // 500) % 2 == 0 else ""
        entry = self._tiny_font.render(self._input + caret, True, text_main)
        surface.blit(entry, (answer_box.x + 6, answer_box.y + 5))

        hint = self._tiny_font.render("Enter target block number and press Enter", True, text_muted)
        surface.blit(hint, (answer_box.right + 10, footer.y + 10))

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

    def _render_colours_letters_numbers_screen(
        self,
        surface: pygame.Surface,
        snap: TestSnapshot,
        payload: ColoursLettersNumbersPayload | None,
    ) -> None:
        self._cln_option_hitboxes = {}

        w, h = surface.get_size()
        bg = (2, 8, 118)
        frame_edge = (228, 236, 255)
        gray_panel = (176, 176, 176)
        dark_panel = (8, 8, 10)
        text_light = (238, 245, 255)
        text_dark = (14, 14, 18)

        phase_label = {
            Phase.INSTRUCTIONS: "Instructions",
            Phase.PRACTICE: "Practice",
            Phase.PRACTICE_DONE: "Practice Complete",
            Phase.SCORED: "Timed Test",
            Phase.RESULTS: "Results",
        }.get(snap.phase, "Task")

        rem_txt = "--:--"
        if snap.time_remaining_s is not None:
            rem = int(round(snap.time_remaining_s))
            rem_txt = f"{rem // 60:02d}:{rem % 60:02d}"

        surface.fill(bg)
        margin = max(8, min(16, w // 56))
        frame = pygame.Rect(margin, margin, w - margin * 2, h - margin * 2)
        pygame.draw.rect(surface, bg, frame)
        pygame.draw.rect(surface, frame_edge, frame, 1)

        header = pygame.Rect(frame.x + 1, frame.y + 1, frame.w - 2, max(26, min(34, h // 18)))
        footer = pygame.Rect(frame.x + 1, frame.bottom - 34, frame.w - 2, 33)
        pygame.draw.rect(surface, bg, header)
        pygame.draw.line(surface, frame_edge, (header.x, header.bottom), (header.right, header.bottom), 1)
        pygame.draw.rect(surface, dark_panel, footer)
        pygame.draw.line(surface, frame_edge, (footer.x, footer.y), (footer.right, footer.y), 1)

        title = self._tiny_font.render(f"Colours, Letters and Numbers - {phase_label}", True, text_light)
        surface.blit(title, title.get_rect(center=header.center))

        body = pygame.Rect(frame.x + 1, header.bottom + 1, frame.w - 2, footer.y - header.bottom - 2)

        bar_colors = {
            "RED": (255, 44, 48),
            "YELLOW": (228, 232, 84),
            "GREEN": (92, 236, 96),
            "BLUE": (60, 114, 242),
        }
        key_for_color = {
            "RED": "R",
            "YELLOW": "E",
            "GREEN": "W",
            "BLUE": "Q",
        }

        if snap.phase is Phase.INSTRUCTIONS:
            pad_x = max(24, min(120, body.w // 8))
            pad_y = max(20, min(90, body.h // 8))
            panel = pygame.Rect(
                body.x + pad_x,
                body.y + pad_y,
                body.w - (pad_x * 2),
                body.h - (pad_y * 2),
            )
            if panel.w < 280 or panel.h < 180:
                panel = body.inflate(-20, -20)

            pygame.draw.rect(surface, dark_panel, panel)
            pygame.draw.rect(surface, frame_edge, panel, 1)

            head = self._small_font.render("How this test works", True, text_light)
            surface.blit(head, (panel.x + 14, panel.y + 12))

            help_text = "\n".join(
                [
                    "1) Memorize the letter sequence at the top.",
                    "2) Click the matching corner option (A-D).",
                    "3) Type the math answer and press Enter (no math timer).",
                    "4) Clear diamonds inside the color lanes.",
                    "5) Memory, math, and colours run independently.",
                    "6) Missed diamonds reduce your score.",
                ]
            )
            self._draw_wrapped_text(
                surface,
                help_text,
                pygame.Rect(panel.x + 14, panel.y + 44, panel.w - 28, max(64, panel.h - 122)),
                color=text_light,
                font=self._small_font,
                max_lines=8,
            )

            legend = pygame.Rect(panel.x + 14, panel.bottom - 70, panel.w - 28, 54)
            pygame.draw.rect(surface, (16, 18, 30), legend)
            pygame.draw.rect(surface, frame_edge, legend, 1)
            legend_title = self._tiny_font.render("Color keys (right -> left): Q / W / E / R", True, text_light)
            surface.blit(legend_title, (legend.x + 8, legend.y + 6))

            chips = [("RED", "R"), ("YELLOW", "E"), ("GREEN", "W"), ("BLUE", "Q")]
            chip_w = max(56, min(90, (legend.w - 18) // 4))
            chip_h = 22
            gap = max(4, (legend.w - (chip_w * 4)) // 5)
            cy = legend.bottom - chip_h - 8
            x = legend.x + gap
            for color_name, key_lbl in chips:
                chip = pygame.Rect(x, cy, chip_w, chip_h)
                pygame.draw.rect(surface, bar_colors.get(color_name, (120, 120, 120)), chip)
                pygame.draw.rect(surface, frame_edge, chip, 1)
                k = self._tiny_font.render(key_lbl, True, text_dark)
                surface.blit(k, k.get_rect(center=chip.center))
                x += chip_w + gap

        elif snap.phase in (Phase.PRACTICE_DONE, Phase.RESULTS):
            panel = body.inflate(-max(32, body.w // 10), -max(22, body.h // 10))
            pygame.draw.rect(surface, dark_panel, panel)
            pygame.draw.rect(surface, frame_edge, panel, 1)
            self._draw_wrapped_text(
                surface,
                str(snap.prompt),
                panel.inflate(-18, -16),
                color=text_light,
                font=self._small_font,
                max_lines=14,
            )

        else:
            corner_w = max(164, min(228, int(body.w * 0.25)))
            corner_h = max(96, min(142, int(body.h * 0.24)))
            top_left = pygame.Rect(body.x, body.y, corner_w, corner_h)
            top_right = pygame.Rect(body.right - corner_w, body.y, corner_w, corner_h)
            bottom_left = pygame.Rect(body.x, body.bottom - corner_h, corner_w, corner_h)
            bottom_right = pygame.Rect(body.right - corner_w, body.bottom - corner_h, corner_w, corner_h)

            option_rects = [top_left, top_right, bottom_left, bottom_right]
            labels = ["A", "B", "C", "D"]
            options = payload.options if payload is not None else tuple()
            options_visible = bool(payload is not None and payload.options_active)

            for i, rect in enumerate(option_rects):
                pygame.draw.rect(surface, gray_panel, rect)
                pygame.draw.rect(surface, frame_edge, rect, 1)

                shown_value = ""
                if options_visible and i < len(options):
                    shown_value = options[i].label
                    self._cln_option_hitboxes[i + 1] = rect.copy()
                if shown_value:
                    text = self._mid_font.render(shown_value, True, text_dark)
                    if text.get_width() > rect.w - 16:
                        text = self._small_font.render(shown_value, True, text_dark)
                    surface.blit(text, text.get_rect(center=(rect.centerx, rect.centery + 2)))

                badge = pygame.Rect(rect.right - 20, rect.bottom - 20, 18, 18)
                pygame.draw.rect(surface, dark_panel, badge)
                pygame.draw.rect(surface, frame_edge, badge, 1)
                badge_text = self._tiny_font.render(labels[i], True, text_light)
                surface.blit(badge_text, badge_text.get_rect(center=badge.center))

            max_center_w = max(280, body.w - (corner_w * 2) - max(16, body.w // 24))
            center_w = max(280, min(max_center_w, int(body.w * 0.50)))
            top_mid_w = max(220, min(340, center_w - 24))
            top_mid_h = max(78, min(96, int(body.h * 0.18)))
            top_mid = pygame.Rect(
                body.centerx - (top_mid_w // 2),
                body.y + max(12, body.h // 30),
                top_mid_w,
                top_mid_h,
            )
            pygame.draw.rect(surface, bg, top_mid)
            pygame.draw.rect(surface, frame_edge, top_mid, 1)

            eq_w = max(224, min(300, int(body.w * 0.28)))
            eq_h = max(52, min(74, int(corner_h * 0.58)))
            eq_rect = pygame.Rect(
                body.centerx - (eq_w // 2),
                body.bottom - corner_h + max(12, (corner_h - eq_h) // 2),
                eq_w,
                eq_h,
            )

            center_gap = max(10, body.h // 38)
            center_top = top_mid.bottom + center_gap
            center_bottom = eq_rect.y - center_gap
            center_h = max(192, min(int(body.h * 0.63), center_bottom - center_top))
            center_rect = pygame.Rect(body.centerx - (center_w // 2), center_top, center_w, center_h)

            pygame.draw.rect(surface, dark_panel, center_rect)
            pygame.draw.rect(surface, frame_edge, center_rect, 1)

            lane_colors = payload.lane_colors if payload is not None else ("RED", "YELLOW", "GREEN", "BLUE")
            lane_start_norm = payload.lane_start_norm if payload is not None else 0.54
            lane_end_norm = payload.lane_end_norm if payload is not None else 0.98
            lane_start_norm = max(0.0, min(1.0, lane_start_norm))
            lane_end_norm = max(lane_start_norm + 0.01, min(1.0, lane_end_norm))
            lane_start_x = center_rect.x + int(lane_start_norm * center_rect.w)
            lane_end_x = center_rect.x + int(lane_end_norm * center_rect.w)
            lane_zone = pygame.Rect(
                lane_start_x,
                center_rect.y + 1,
                max(1, lane_end_x - lane_start_x),
                center_rect.h - 2,
            )
            pygame.draw.rect(surface, frame_edge, lane_zone, 1)
            lane_count = max(1, len(lane_colors))
            for i, color_name in enumerate(lane_colors):
                color = bar_colors.get(color_name, (128, 128, 128))
                lx0 = lane_zone.x + int((i * lane_zone.w) / lane_count)
                lx1 = lane_zone.x + int(((i + 1) * lane_zone.w) / lane_count)
                lane = pygame.Rect(lx0, lane_zone.y, max(1, lx1 - lx0), lane_zone.h)
                pygame.draw.rect(surface, color, lane)
                key_lbl = key_for_color.get(color_name, "?")
                key_s = self._tiny_font.render(key_lbl, True, (14, 14, 18))
                surface.blit(key_s, key_s.get_rect(midtop=(lane.centerx, lane.y + 4)))

            diamonds = payload.diamonds if payload is not None else tuple()
            row_margin = max(42, min(74, center_rect.h // 4))
            row_y = [
                center_rect.y + row_margin,
                center_rect.centery,
                center_rect.bottom - row_margin,
            ]
            diamond_size = max(7, min(10, center_rect.h // 24))
            for d in diamonds:
                x = center_rect.x + int(d.x_norm * max(1, center_rect.w - 1))
                y = row_y[max(0, min(len(row_y) - 1, int(d.row)))]
                poly = [
                    (x, y - diamond_size),
                    (x + diamond_size, y),
                    (x, y + diamond_size),
                    (x - diamond_size, y),
                ]
                color = bar_colors.get(d.color, (180, 180, 180))
                pygame.draw.polygon(surface, color, poly)
                pygame.draw.polygon(surface, frame_edge, poly, 1)

            pygame.draw.rect(surface, (100, 100, 100), eq_rect)
            pygame.draw.rect(surface, frame_edge, eq_rect, 1)
            eq_text = payload.math_prompt if payload is not None else "0 + 0 ="
            eq_text = eq_text.replace("SOLVE:", "").strip()
            eq_s = self._mid_font.render(eq_text, True, text_dark)
            if eq_s.get_width() > eq_rect.w - 12:
                eq_s = self._small_font.render(eq_text, True, text_dark)
            surface.blit(eq_s, eq_s.get_rect(center=eq_rect.center))

            if payload is not None and payload.target_sequence is not None:
                seq = self._mid_font.render(payload.target_sequence, True, text_light)
                surface.blit(seq, seq.get_rect(center=top_mid.center))
                hint_text = "Memorize sequence"
            elif payload is not None and not payload.memory_answered:
                hint_text = "Click matching corner option"
            elif payload is not None and payload.memory_answered:
                hint_text = "Sequence selected"
            else:
                hint_text = ""
            if hint_text:
                hint = self._tiny_font.render(hint_text, True, text_light)
                surface.blit(hint, hint.get_rect(midbottom=(top_mid.centerx, top_mid.bottom - 6)))

        attempted = max(0, int(snap.attempted_scored))
        correct = max(0, int(snap.correct_scored))
        accuracy = 0.0 if attempted == 0 else (correct / attempted) * 100.0
        misses = payload.missed_diamonds if payload is not None else 0
        cleared = payload.cleared_diamonds if payload is not None else 0
        points = payload.points if payload is not None else 0.0

        if snap.phase in (Phase.PRACTICE, Phase.SCORED):
            caret = "|" if (pygame.time.get_ticks() // 500) % 2 == 0 else ""
            answer_value = self._input + caret
            control_text = "Mouse: A-D sequence  |  Keys: Q/W/E/R clear lanes right->left  |  Enter: math"
        elif snap.phase in (Phase.INSTRUCTIONS, Phase.PRACTICE_DONE, Phase.RESULTS):
            answer_value = "--"
            control_text = "Press Enter to continue"
        else:
            answer_value = "--"
            control_text = ""

        left = self._tiny_font.render(control_text, True, text_light)
        center = self._small_font.render(f"Math Answer: {answer_value}", True, text_light)
        right = self._tiny_font.render(
            f"Scored {correct}/{attempted} ({accuracy:.1f}%)  Clear {cleared}  Miss {misses}  Pts {points:.1f}  T {rem_txt}",
            True,
            text_light,
        )
        surface.blit(left, (footer.x + 8, footer.y + 3))
        surface.blit(center, center.get_rect(midleft=(footer.x + 10, footer.bottom - 10)))
        surface.blit(right, right.get_rect(midright=(footer.right - 8, footer.y + 9)))

    def _render_digit_recognition_screen(
        self,
        surface: pygame.Surface,
        snap: TestSnapshot,
        payload: DigitRecognitionPayload | None,
    ) -> None:
        w, h = surface.get_size()
        bg = (2, 8, 114)
        edge = (232, 240, 255)
        text_main = (236, 244, 255)
        text_muted = (184, 198, 224)

        surface.fill(bg)
        margin = max(8, min(16, w // 56))
        frame = pygame.Rect(margin, margin, w - margin * 2, h - margin * 2)
        pygame.draw.rect(surface, bg, frame)
        pygame.draw.rect(surface, edge, frame, 1)

        header_h = max(24, min(32, h // 18))
        footer_h = max(30, min(38, h // 14))
        header = pygame.Rect(frame.x + 1, frame.y + 1, frame.w - 2, header_h)
        footer = pygame.Rect(frame.x + 1, frame.bottom - footer_h - 1, frame.w - 2, footer_h)
        body = pygame.Rect(frame.x + 1, header.bottom + 1, frame.w - 2, footer.y - header.bottom - 2)

        pygame.draw.rect(surface, bg, header)
        pygame.draw.line(surface, edge, (header.x, header.bottom), (header.right, header.bottom), 1)
        pygame.draw.rect(surface, (0, 0, 0), footer)
        pygame.draw.line(surface, edge, (footer.x, footer.y), (footer.right, footer.y), 1)

        phase_label = {
            Phase.INSTRUCTIONS: "Instructions",
            Phase.PRACTICE: "Practice",
            Phase.PRACTICE_DONE: "Practice Complete",
            Phase.SCORED: "Timed Test",
            Phase.RESULTS: "Results",
        }.get(snap.phase, "Task")

        title = self._tiny_font.render(f"Digit Recognition - {phase_label}", True, text_main)
        surface.blit(title, title.get_rect(center=header.center))

        rem_txt = "--:--"
        if snap.time_remaining_s is not None:
            rem = int(round(snap.time_remaining_s))
            rem_txt = f"{rem // 60:02d}:{rem % 60:02d}"

        if snap.phase in (Phase.INSTRUCTIONS, Phase.PRACTICE_DONE, Phase.RESULTS):
            self._draw_wrapped_text(
                surface,
                str(snap.prompt),
                body.inflate(-40, -34),
                color=text_main,
                font=self._small_font,
                max_lines=12,
            )
        elif payload is not None and payload.display_digits is not None:
            digits = self._big_font.render(payload.display_digits, True, text_main)
            if digits.get_width() > int(body.w * 0.9):
                digits = self._mid_font.render(payload.display_digits, True, text_main)
            surface.blit(digits, digits.get_rect(center=body.center))
        elif payload is not None and not payload.accepting_input:
            mask = self._mid_font.render("X X X X X X X X", True, text_muted)
            surface.blit(mask, mask.get_rect(center=body.center))
        else:
            prompt_box = pygame.Rect(body.x + 20, body.y + 26, body.w - 40, 72)
            pygame.draw.rect(surface, (8, 18, 120), prompt_box)
            pygame.draw.rect(surface, edge, prompt_box, 1)
            self._draw_wrapped_text(
                surface,
                str(snap.prompt),
                prompt_box.inflate(-12, -10),
                color=text_main,
                font=self._small_font,
                max_lines=2,
            )

        info = self._tiny_font.render(
            f"Scored {snap.correct_scored}/{snap.attempted_scored}  |  Time Left: {rem_txt}",
            True,
            text_main,
        )
        surface.blit(info, (footer.x + 12, footer.y + (footer.h - info.get_height()) // 2))

    def _render_digit_recognition_answer_box(
        self,
        surface: pygame.Surface,
        snap: TestSnapshot,
        payload: DigitRecognitionPayload | None,
    ) -> None:
        if payload is None or not payload.accepting_input:
            return

        w, h = surface.get_size()
        box_w = max(240, min(460, int(w * 0.52)))
        box_h = 48
        box = pygame.Rect((w - box_w) // 2, h - 88, box_w, box_h)
        pygame.draw.rect(surface, (6, 15, 92), box)
        pygame.draw.rect(surface, (180, 196, 230), box, 2)

        caret = "|" if (pygame.time.get_ticks() // 500) % 2 == 0 else ""
        entry = self._small_font.render(self._input + caret, True, (236, 244, 255))
        surface.blit(entry, (box.x + 12, box.y + 11))

        hint = self._tiny_font.render(snap.input_hint, True, (168, 184, 214))
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

    def open_colours_letters_numbers() -> None:
        seed = _new_seed()
        app.push(
            CognitiveTestScreen(
                app,
                engine_factory=lambda: build_colours_letters_numbers_test(
                    clock=real_clock,
                    seed=seed,
                    difficulty=0.5,
                ),
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

    def open_target_recognition() -> None:
        seed = _new_seed()
        app.push(
            CognitiveTestScreen(
                app,
                engine_factory=lambda: build_target_recognition_test(
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
            MenuItem("Colours, Letters and Numbers", open_colours_letters_numbers),
            MenuItem("Angles, Bearings and Degrees", open_angles_bearings_degrees),
            MenuItem("Visual Search", open_visual_search),
            MenuItem("Instrument Comprehension", open_instrument_comprehension),
            MenuItem("Target Recognition", open_target_recognition),
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
