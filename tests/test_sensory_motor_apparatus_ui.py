from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from cfast_trainer.app import App, CognitiveTestScreen, MenuItem, MenuScreen
from cfast_trainer.cognitive_core import Phase
from cfast_trainer.cognitive_core import TestSnapshot as SnapshotModel
from cfast_trainer.sensory_motor_apparatus import SensoryMotorApparatusPayload


class _SpyFont:
    def __init__(self) -> None:
        self.rendered: list[str] = []

    def render(self, text: str, _antialias: bool, _color) -> pygame.Surface:
        value = str(text)
        self.rendered.append(value)
        width = max(8, len(value) * 6)
        surface = pygame.Surface((width, 16), pygame.SRCALPHA)
        surface.fill((255, 255, 255, 255))
        return surface

    def size(self, text: str) -> tuple[int, int]:
        return (max(8, len(str(text)) * 6), 16)

    def get_linesize(self) -> int:
        return 16


@dataclass
class _FakeSensoryMotorEngine:
    payload: SensoryMotorApparatusPayload
    phase: Phase = Phase.SCORED
    title: str = "Sensory Motor Apparatus"

    def snapshot(self) -> SnapshotModel:
        return SnapshotModel(
            title=self.title,
            phase=self.phase,
            prompt="Keep the dot centered.",
            input_hint="",
            time_remaining_s=42.0,
            attempted_scored=3,
            correct_scored=1,
            payload=self.payload,
        )

    def can_exit(self) -> bool:
        return True

    def start_practice(self) -> None:
        return

    def start_scored(self) -> None:
        return

    def submit_answer(self, raw: str) -> bool:
        return False

    def update(self) -> None:
        return

    def set_control(self, *, horizontal: float, vertical: float) -> None:
        return


def _build_payload(*, axis_focus: str = "both") -> SensoryMotorApparatusPayload:
    return SensoryMotorApparatusPayload(
        dot_x=0.22,
        dot_y=0.0 if axis_focus == "horizontal" else -0.18,
        control_x=0.30,
        control_y=0.0 if axis_focus == "horizontal" else -0.10,
        disturbance_x=0.12,
        disturbance_y=0.0 if axis_focus == "horizontal" else -0.08,
        control_mode="split",
        block_kind="scored",
        block_index=2,
        block_total=4,
        axis_focus=axis_focus,
        guide_band_half_width=0.12 if axis_focus != "both" else 0.0,
        segment_label="Tempo - Split Pulse",
        segment_index=2,
        segment_total=4,
        segment_time_remaining_s=29.4,
        phase_elapsed_s=12.0,
        mean_error=0.221,
        rms_error=0.247,
        on_target_s=4.5,
        on_target_ratio=0.375,
    )


def _build_screen(engine: _FakeSensoryMotorEngine) -> tuple[App, CognitiveTestScreen]:
    pygame.init()
    surface = pygame.display.set_mode((960, 540))
    font = pygame.font.Font(None, 36)
    app = App(surface=surface, font=font)
    root = MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True)
    app.push(root)
    screen = CognitiveTestScreen(app, engine_factory=lambda: engine)
    app.push(screen)
    return app, screen


def test_sma_renderer_is_used_for_titles_starting_with_sensory_motor_apparatus() -> None:
    engine = _FakeSensoryMotorEngine(
        payload=_build_payload(),
        title="Sensory Motor Apparatus: Pressure Run",
    )
    _app, screen = _build_screen(engine)
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        called: list[SnapshotModel] = []
        screen._render_sensory_motor_apparatus_screen = (
            lambda target, snap, payload=None: called.append(snap)
        )  # type: ignore[method-assign]

        screen.render(surface)

        assert len(called) == 1
        assert isinstance(called[0].payload, SensoryMotorApparatusPayload)
    finally:
        pygame.quit()


def test_sma_renderer_draws_guide_band_and_segment_metadata(monkeypatch) -> None:
    engine = _FakeSensoryMotorEngine(payload=_build_payload(axis_focus="horizontal"))
    _app, screen = _build_screen(engine)
    try:
        surface = pygame.display.get_surface()
        assert surface is not None

        spy_small = _SpyFont()
        spy_tiny = _SpyFont()
        screen._small_font = spy_small  # type: ignore[assignment]
        screen._tiny_font = spy_tiny  # type: ignore[assignment]

        band_draws: list[pygame.Rect] = []
        real_draw_rect = pygame.draw.rect

        def _spy_draw_rect(target, color, rect, width=0, border_radius=0, border_top_left_radius=-1, border_top_right_radius=-1, border_bottom_left_radius=-1, border_bottom_right_radius=-1):
            if color == (84, 182, 255, 34):
                band_draws.append(pygame.Rect(rect))
            return real_draw_rect(
                target,
                color,
                rect,
                width,
                border_radius,
                border_top_left_radius,
                border_top_right_radius,
                border_bottom_left_radius,
                border_bottom_right_radius,
            )

        monkeypatch.setattr(pygame.draw, "rect", _spy_draw_rect)

        screen.render(surface)

        assert band_draws
        assert "Tempo - Split Pulse" in spy_tiny.rendered
        assert "Tempo - Split Pulse 2/4" not in spy_tiny.rendered
        assert not any(text.startswith("Segment 00:") for text in spy_tiny.rendered)
        assert not any(text.startswith("Windows ") for text in spy_tiny.rendered)
    finally:
        pygame.quit()


def test_sma_renderer_labels_scored_handoff_as_timed_segment_complete() -> None:
    engine = _FakeSensoryMotorEngine(payload=_build_payload(), phase=Phase.PRACTICE_DONE)
    _app, screen = _build_screen(engine)
    try:
        surface = pygame.display.get_surface()
        assert surface is not None

        spy_small = _SpyFont()
        spy_tiny = _SpyFont()
        screen._small_font = spy_small  # type: ignore[assignment]
        screen._tiny_font = spy_tiny  # type: ignore[assignment]

        screen.render(surface)

        rendered = spy_small.rendered + spy_tiny.rendered
        assert "Sensory Motor Apparatus - Timed Segment Complete" in rendered
        assert "Enter: Continue Timed Test  |  Esc/Backspace: Pause" in rendered
    finally:
        pygame.quit()
