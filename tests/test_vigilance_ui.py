from __future__ import annotations

import os
from dataclasses import dataclass

import pygame

from cfast_trainer.app import App, CognitiveTestScreen, MenuItem, MenuScreen
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


def _build_screen(*, engine_factory, test_code: str) -> CognitiveTestScreen:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    pygame.init()
    surface = pygame.Surface((960, 540))
    font = pygame.font.Font(None, 36)
    app = App(surface=surface, font=font, opengl_enabled=False)
    app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))
    screen = CognitiveTestScreen(app, engine_factory=engine_factory, test_code=test_code)
    app.push(screen)
    return screen


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
