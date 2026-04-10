from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType

import pygame

if "moderngl" not in sys.modules:
    moderngl_stub = ModuleType("moderngl")
    moderngl_stub.__spec__ = ModuleSpec("moderngl", loader=None)
    sys.modules["moderngl"] = moderngl_stub

from cfast_trainer.app import App, CognitiveTestScreen, MenuItem, MenuScreen
from cfast_trainer.app import DifficultySettingsStore
from cfast_trainer.ant_drills import AntDrillMode
from cfast_trainer.vig_drills import VigilanceDrillConfig, build_vig_entry_anchor_drill
from cfast_trainer.vigilance import VigilancePayload


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


class _RecordingFont:
    def __init__(self, base: pygame.font.Font, sink: list[str]) -> None:
        self._base = base
        self._sink = sink

    def render(self, text: str, antialias: bool, color: object) -> pygame.Surface:
        self._sink.append(str(text))
        return self._base.render(text, antialias, color)

    def __getattr__(self, name: str) -> object:
        return getattr(self._base, name)


def _build_screen(
    *,
    engine_factory,
    test_code: str,
    review_mode: bool = False,
) -> CognitiveTestScreen:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    pygame.init()
    surface = pygame.Surface((960, 540))
    font = pygame.font.Font(None, 36)
    store = DifficultySettingsStore(Path("/tmp/cfast-vigilance-ui-difficulty-settings.json"))
    store.set_review_mode_enabled(review_mode)
    app = App(
        surface=surface,
        font=font,
        opengl_enabled=False,
        difficulty_settings_store=store,
    )
    app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))
    screen = CognitiveTestScreen(app, engine_factory=engine_factory, test_code=test_code)
    app.push(screen)
    return screen


def _install_recording_fonts(*fonts: object) -> list[str]:
    captured: list[str] = []
    for obj in fonts:
        for attr in ("_small_font", "_tiny_font", "_mid_font", "_big_font"):
            font = getattr(obj, attr, None)
            if isinstance(font, _RecordingFont) or font is None:
                continue
            setattr(obj, attr, _RecordingFont(font, captured))
    return captured


def _advance_until_symbol(
    *,
    screen: CognitiveTestScreen,
    clock: FakeClock,
    step_s: float = 0.05,
    timeout_s: float = 5.0,
    drive_render: bool = False,
) -> VigilancePayload:
    elapsed = 0.0
    while elapsed < timeout_s:
        clock.advance(step_s)
        if drive_render:
            screen.render(screen._app.surface)
        else:
            screen._engine.update()
        payload = screen._engine.snapshot().payload
        assert isinstance(payload, VigilancePayload)
        if payload.symbols:
            return payload
        elapsed += step_s
    raise AssertionError("expected a visible vigilance symbol before timeout")


def test_vigilance_custom_drill_title_routes_to_vigilance_renderer() -> None:
    clock = FakeClock()
    try:
        screen = _build_screen(
            engine_factory=lambda: build_vig_entry_anchor_drill(
                clock=clock,
                seed=73,
                difficulty=0.5,
                mode=AntDrillMode.BUILD,
            ),
            test_code="vig_entry_anchor",
        )
        screen._engine.start_practice()
        called = {"value": False}

        def _fake_render(surface, snap, payload) -> None:
            called["value"] = True

        screen._render_vigilance_screen = _fake_render  # type: ignore[method-assign]
        screen.render(pygame.Surface((960, 540)))

        assert called["value"] is True
    finally:
        pygame.quit()


def test_vigilance_custom_drill_title_keeps_row_col_input_active() -> None:
    clock = FakeClock()
    try:
        screen = _build_screen(
            engine_factory=lambda: build_vig_entry_anchor_drill(
                clock=clock,
                seed=79,
                difficulty=0.5,
                mode=AntDrillMode.BUILD,
                config=VigilanceDrillConfig(
                    practice_duration_s=5.0,
                    scored_duration_s=10.0,
                    spawn_interval_s=10.0,
                    max_active_symbols=4,
                ),
            ),
            test_code="vig_entry_anchor",
        )
        screen._engine.start_practice()

        payload: VigilancePayload | None = None
        for _ in range(50):
            clock.advance(0.1)
            screen._engine.update()
            snap_payload = screen._engine.snapshot().payload
            if isinstance(snap_payload, VigilancePayload) and snap_payload.symbols:
                payload = snap_payload
                break
        assert payload is not None

        symbol = payload.symbols[0]
        screen.handle_event(
            pygame.event.Event(
                pygame.KEYDOWN,
                {"key": ord(str(symbol.row)), "unicode": str(symbol.row)},
            )
        )
        screen.handle_event(
            pygame.event.Event(
                pygame.KEYDOWN,
                {"key": ord(str(symbol.col)), "unicode": str(symbol.col)},
            )
        )

        updated = screen._engine.snapshot().payload
        assert isinstance(updated, VigilancePayload)
        assert updated.points_total == symbol.points
        assert all(active.symbol_id != symbol.symbol_id for active in updated.symbols)
    finally:
        pygame.quit()


def test_vigilance_enter_submission_clears_inputs_even_with_review_mode_enabled() -> None:
    clock = FakeClock()
    try:
        screen = _build_screen(
            engine_factory=lambda: build_vig_entry_anchor_drill(
                clock=clock,
                seed=97,
                difficulty=0.5,
                mode=AntDrillMode.BUILD,
                config=VigilanceDrillConfig(
                    practice_duration_s=5.0,
                    scored_duration_s=10.0,
                    spawn_interval_s=10.0,
                    max_active_symbols=4,
                ),
            ),
            test_code="vig_entry_anchor",
            review_mode=True,
        )
        screen._engine.start_practice()
        payload = _advance_until_symbol(screen=screen, clock=clock)
        symbol = payload.symbols[0]

        screen._vigilance_row_input = str(symbol.row)
        screen._vigilance_col_input = str(symbol.col)
        screen.handle_event(
            pygame.event.Event(
                pygame.KEYDOWN,
                {"key": pygame.K_RETURN, "mod": 0, "unicode": "\r"},
            )
        )

        updated = screen._engine.snapshot().payload
        assert isinstance(updated, VigilancePayload)
        assert screen._review_state is None
        assert screen._vigilance_row_input == ""
        assert screen._vigilance_col_input == ""
        assert updated.points_total == symbol.points
        assert all(active.symbol_id != symbol.symbol_id for active in updated.symbols)
    finally:
        pygame.quit()


def test_vigilance_live_stream_keeps_advancing_after_enter_submission() -> None:
    clock = FakeClock()
    try:
        screen = _build_screen(
            engine_factory=lambda: build_vig_entry_anchor_drill(
                clock=clock,
                seed=101,
                difficulty=0.5,
                mode=AntDrillMode.BUILD,
                config=VigilanceDrillConfig(
                    practice_duration_s=5.0,
                    scored_duration_s=10.0,
                    spawn_interval_s=0.25,
                    max_active_symbols=1,
                ),
            ),
            test_code="vig_entry_anchor",
            review_mode=True,
        )
        screen._engine.start_practice()
        payload = _advance_until_symbol(screen=screen, clock=clock, drive_render=True)
        symbol = payload.symbols[0]

        screen._vigilance_row_input = str(symbol.row)
        screen._vigilance_col_input = str(symbol.col)
        screen.handle_event(
            pygame.event.Event(
                pygame.KEYDOWN,
                {"key": pygame.K_RETURN, "mod": 0, "unicode": "\r"},
            )
        )

        assert screen._review_state is None
        assert screen._review_clock is not None
        assert screen._review_clock.is_paused() is False

        updated = _advance_until_symbol(
            screen=screen,
            clock=clock,
            step_s=0.05,
            timeout_s=1.5,
            drive_render=True,
        )
        assert updated.symbols
        assert updated.symbols[0].symbol_id != symbol.symbol_id
    finally:
        pygame.quit()


def test_vigilance_live_screen_keeps_symbol_points_legend_without_live_tally_counters() -> None:
    clock = FakeClock()
    try:
        screen = _build_screen(
            engine_factory=lambda: build_vig_entry_anchor_drill(
                clock=clock,
                seed=131,
                difficulty=0.5,
                mode=AntDrillMode.BUILD,
                config=VigilanceDrillConfig(
                    practice_duration_s=5.0,
                    scored_duration_s=10.0,
                    spawn_interval_s=0.25,
                    max_active_symbols=2,
                ),
            ),
            test_code="vig_entry_anchor",
        )
        screen._engine.start_practice()
        _advance_until_symbol(screen=screen, clock=clock)
        captured = _install_recording_fonts(screen)

        screen.render(screen._app.surface)

        assert "Symbol Points" in captured
        assert not any(text.startswith("Captures ") for text in captured)
        assert not any(text.startswith("Points ") for text in captured)
        assert not any(text.startswith("Points: ") for text in captured)
        assert not any(text.startswith("Captured: ") for text in captured)
        assert not any(text.startswith("Missed: ") for text in captured)
        assert not any(text.startswith("Visible: ") for text in captured)
    finally:
        pygame.quit()
