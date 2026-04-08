from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from cfast_trainer.app import App, CognitiveTestScreen, MenuItem, MenuScreen
from cfast_trainer.cognitive_core import Phase
from cfast_trainer.cognitive_core import TestSnapshot as SnapshotModel
from cfast_trainer.visual_search import VisualSearchPayload, VisualSearchTaskKind


@dataclass
class _FakeVisualSearchEngine:
    payload: VisualSearchPayload

    def snapshot(self) -> SnapshotModel:
        return SnapshotModel(
            title="Visual Search",
            phase=Phase.PRACTICE,
            prompt="",
            input_hint="",
            time_remaining_s=None,
            attempted_scored=0,
            correct_scored=0,
            payload=self.payload,
        )

    def can_exit(self) -> bool:
        return True

    def start_practice(self) -> None:
        return

    def start_scored(self) -> None:
        return

    def submit_answer(self, raw: str) -> bool:
        _ = raw
        return True

    def update(self) -> None:
        return


def _build_screen(payload: VisualSearchPayload) -> tuple[App, CognitiveTestScreen]:
    pygame.init()
    surface = pygame.display.set_mode((960, 540))
    font = pygame.font.Font(None, 36)
    app = App(surface=surface, font=font)
    root = MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True)
    app.push(root)
    screen = CognitiveTestScreen(app, engine_factory=lambda: _FakeVisualSearchEngine(payload))
    app.push(screen)
    return app, screen


def _payload(*, kind: VisualSearchTaskKind, rows: int, cols: int, target: str, cells: tuple[str, ...]) -> VisualSearchPayload:
    return VisualSearchPayload(
        kind=kind,
        rows=rows,
        cols=cols,
        target=target,
        cells=cells,
        cell_codes=tuple(range(10, 10 + (rows * cols))),
        full_credit_error=0,
        zero_credit_error=1,
    )


def test_visual_search_screen_renders_5x6_symbol_board_with_composite_tokens() -> None:
    payload = _payload(
        kind=VisualSearchTaskKind.SYMBOL_CODE,
        rows=5,
        cols=6,
        target="L_HOOK@TR",
        cells=tuple(
            f"{base}@{mark}"
            for base, mark in (
                ("L_HOOK", "TR"),
                ("PIN", "T+TR"),
                ("FORK", "R"),
                ("BOX", "B+BL"),
                ("TRIANGLE", "L"),
                ("RING_SPOKE", "C"),
                ("STAR", "TR"),
                ("BOLT", "T"),
                ("DOUBLE_CROSS", "R+BR"),
                ("X_MARK", "B"),
                ("S_BEND", "L+TL"),
                ("LOLLIPOP", "C+DOT"),
                ("L_HOOK", "T"),
                ("PIN", "R"),
                ("FORK", "B"),
                ("BOX", "L"),
                ("TRIANGLE", "C"),
                ("RING_SPOKE", "TR+DOT"),
                ("STAR", "T+TL"),
                ("BOLT", "R"),
                ("DOUBLE_CROSS", "B"),
                ("X_MARK", "L"),
                ("S_BEND", "C"),
                ("LOLLIPOP", "TR"),
                ("L_HOOK", "R+BR"),
                ("PIN", "B"),
                ("FORK", "L+BL"),
                ("BOX", "C+DOT"),
                ("TRIANGLE", "TR"),
                ("RING_SPOKE", "T+TR"),
            )
        ),
    )
    _app, screen = _build_screen(payload)
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        screen.render(surface)
        assert all(len(str(code)) == 2 for code in payload.cell_codes)
    finally:
        pygame.quit()


def test_visual_search_screen_renders_letter_variants_without_error() -> None:
    payload = _payload(
        kind=VisualSearchTaskKind.ALPHANUMERIC,
        rows=5,
        cols=5,
        target="E@C",
        cells=(
            "E@C",
            "F@T+TR",
            "H@R",
            "K@B+BR",
            "L@L",
            "A@C+DOT",
            "M@TR",
            "R@T",
            "B@R+BR",
            "G@B",
            "P@L+TL",
            "S@C",
            "E@TR+DOT",
            "F@C",
            "H@T",
            "K@R",
            "L@B",
            "A@L",
            "M@C+DOT",
            "R@TR",
            "B@T+TL",
            "G@R",
            "P@B+BL",
            "S@L",
            "E@T+TR",
        ),
    )
    _app, screen = _build_screen(payload)
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        screen.render(surface)
    finally:
        pygame.quit()


def test_visual_search_screen_renders_7x6_alphanumeric_string_board_without_error() -> None:
    payload = _payload(
        kind=VisualSearchTaskKind.ALPHANUMERIC,
        rows=7,
        cols=6,
        target="A00B",
        cells=tuple(f"A{row}{col}B" for row in range(7) for col in range(6)),
    )
    _app, screen = _build_screen(payload)
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        screen.render(surface)
    finally:
        pygame.quit()
