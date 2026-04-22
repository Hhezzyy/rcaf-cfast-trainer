from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from importlib.machinery import ModuleSpec
from types import ModuleType

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
import pytest

if "moderngl" not in sys.modules:
    moderngl_stub = ModuleType("moderngl")
    moderngl_stub.__spec__ = ModuleSpec("moderngl", loader=None)
    sys.modules["moderngl"] = moderngl_stub

from cfast_trainer.app import App, CognitiveTestScreen, MenuItem, MenuScreen
from cfast_trainer.ma_drills import MaDrillConfig, build_ma_percentage_snap_drill


@dataclass
class _FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return float(self.t)

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


def _install_recording_fonts(app: App, screen: CognitiveTestScreen) -> list[str]:
    captured: list[str] = []
    for obj, attrs in (
        (app, ("_font",)),
        (screen, ("_small_font", "_tiny_font", "_mid_font", "_big_font")),
    ):
        for attr in attrs:
            font = getattr(obj, attr, None)
            if isinstance(font, _RecordingFont) or font is None:
                continue
            setattr(obj, attr, _RecordingFont(font, captured))
    return captured


def test_percentage_snap_scored_screen_hides_countdown_but_keeps_internal_time() -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        app = App(surface=surface, font=font)
        app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))
        clock = _FakeClock()
        engine = build_ma_percentage_snap_drill(
            clock=clock,
            seed=31,
            difficulty=0.5,
            config=MaDrillConfig(practice_questions=0, scored_duration_s=42.0),
        )
        engine.start_scored()
        assert engine.time_remaining_s() == pytest.approx(42.0)

        screen = CognitiveTestScreen(
            app,
            engine_factory=lambda: engine,
            test_code="ma_percentage_snap",
        )
        app.push(screen)
        captured = _install_recording_fonts(app, screen)

        screen.render(surface)
        rendered = "\n".join(captured)

        assert engine.time_remaining_s() is not None
        assert any("Mental Arithmetic: Percentage Snap" in text for text in captured)
        assert "Time remaining" not in rendered
        assert "Time Left" not in rendered
        assert not re.search(r"\b\d{2}:\d{2}\b", rendered)
    finally:
        pygame.quit()
