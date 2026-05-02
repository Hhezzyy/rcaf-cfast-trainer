from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from cfast_trainer.ant_drills import AntDrillMode
from cfast_trainer.app import App, CognitiveTestScreen, MenuItem, MenuScreen
from cfast_trainer.cognitive_core import Phase
from cfast_trainer.cognitive_core import TestSnapshot as SnapshotModel
from cfast_trainer.visual_search import VisualSearchPayload, VisualSearchTaskKind
from cfast_trainer.vs_drills import VsDrillConfig, build_vs_mixed_tempo_drill


@dataclass
class _FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


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


class _IntroEngine:
    def __init__(
        self,
        *,
        phase: Phase,
        title: str,
        prompt: str,
        input_hint: str = "Press Enter",
        difficulty_code: str | None = None,
    ) -> None:
        self._phase = phase
        self._title = title
        self._prompt = prompt
        self._input_hint = input_hint
        if difficulty_code is not None:
            self._difficulty_code = difficulty_code

    def snapshot(self) -> SnapshotModel:
        return SnapshotModel(
            title=self._title,
            phase=self._phase,
            prompt=self._prompt,
            input_hint=self._input_hint,
            time_remaining_s=None,
            attempted_scored=0,
            correct_scored=0,
            payload=None,
        )

    def can_exit(self) -> bool:
        return True

    def start_practice(self) -> None:
        self._phase = Phase.PRACTICE

    def start_scored(self) -> None:
        self._phase = Phase.SCORED

    def submit_answer(self, raw: str) -> bool:
        _ = raw
        return True

    def update(self) -> None:
        return


def _build_screen_for_engine(
    engine: object,
    *,
    test_code: str | None = None,
) -> tuple[App, CognitiveTestScreen]:
    pygame.init()
    surface = pygame.display.set_mode((960, 540))
    font = pygame.font.Font(None, 36)
    app = App(surface=surface, font=font)
    root = MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True)
    app.push(root)
    screen = CognitiveTestScreen(app, engine_factory=lambda: engine, test_code=test_code)
    app.push(screen)
    return app, screen


def _build_screen(payload: VisualSearchPayload) -> tuple[App, CognitiveTestScreen]:
    return _build_screen_for_engine(_FakeVisualSearchEngine(payload))


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


def test_visual_search_timeout_clears_partial_input_on_next_problem() -> None:
    clock = _FakeClock()
    engine = build_vs_mixed_tempo_drill(
        clock=clock,
        seed=41,
        difficulty=0.8,
        mode=AntDrillMode.TEMPO,
        config=VsDrillConfig(practice_questions=0, scored_duration_s=60.0),
    )
    engine.start_scored()

    _app, screen = _build_screen_for_engine(engine, test_code="vs_mixed_tempo")
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        screen.render(surface)
        assert isinstance(engine.snapshot().payload, VisualSearchPayload)
        screen._input = "9"

        cap_s = getattr(engine, "_current_cap_s")
        assert cap_s is not None
        clock.advance(float(cap_s) + 0.05)
        screen.render(surface)

        assert engine.snapshot().attempted_scored == 1
        assert isinstance(engine.snapshot().payload, VisualSearchPayload)
        assert screen._input == ""
    finally:
        pygame.quit()


def test_visual_search_payload_change_with_same_title_clears_partial_input() -> None:
    first = _payload(
        kind=VisualSearchTaskKind.ALPHANUMERIC,
        rows=2,
        cols=2,
        target="A",
        cells=("A", "B", "C", "D"),
    )
    second = _payload(
        kind=VisualSearchTaskKind.ALPHANUMERIC,
        rows=2,
        cols=2,
        target="A",
        cells=("A", "B", "C", "E"),
    )
    engine = _FakeVisualSearchEngine(first)
    _app, screen = _build_screen_for_engine(engine)
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        screen.render(surface)
        screen._input = "10"

        engine.payload = second
        screen.render(surface)

        assert screen._input == ""
    finally:
        pygame.quit()


def test_visual_search_guide_intro_draws_overlay_after_demo_board() -> None:
    engine = _IntroEngine(
        phase=Phase.INSTRUCTIONS,
        title="Visual Search: Mixed Tempo",
        prompt="Press Enter to begin practice.",
    )
    _app, screen = _build_screen_for_engine(engine, test_code="vs_mixed_tempo")
    events: list[str] = []
    section_titles: list[str] = []
    original_board = screen._draw_visual_search_board
    original_section = screen._draw_intro_section

    def record_board(surface, rect, payload, **kwargs):
        events.append("board")
        return original_board(surface, rect, payload, **kwargs)

    def record_section(surface, rect, **kwargs):
        events.append("section")
        section_titles.append(str(kwargs.get("title", "")))
        return original_section(surface, rect, **kwargs)

    screen._draw_visual_search_board = record_board
    screen._draw_intro_section = record_section

    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        screen.render(surface)

        assert "board" in events
        assert "section" in events
        assert events.index("board") < events.index("section")
        assert "What This Assesses" in section_titles
        assert "Guide Notes" in section_titles
    finally:
        pygame.quit()


def test_missing_intro_briefing_draws_generic_overlay_over_demo_board() -> None:
    engine = _IntroEngine(
        phase=Phase.INSTRUCTIONS,
        title="Visual Search: Missing Briefing",
        prompt="Read the board, find the target, and type the block number.",
        input_hint="Type the block number",
    )
    _app, screen = _build_screen_for_engine(engine)
    captured_lines: list[str] = []
    original_section = screen._draw_intro_section

    def record_section(surface, rect, **kwargs):
        captured_lines.extend(str(line) for line in kwargs.get("lines", ()))
        return original_section(surface, rect, **kwargs)

    screen._draw_intro_section = record_section

    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        screen.render(surface)

        assert "Runtime drill instructions." in captured_lines
        assert any("Read the board" in line for line in captured_lines)
        assert any("Type the block number" in line for line in captured_lines)
    finally:
        pygame.quit()


def test_practice_done_without_briefing_draws_generic_overlay() -> None:
    engine = _IntroEngine(
        phase=Phase.PRACTICE_DONE,
        title="Visual Search: Missing Briefing",
        prompt="Practice complete. Press Enter for the scored block.",
        input_hint="Press Enter",
    )
    _app, screen = _build_screen_for_engine(engine)
    section_titles: list[str] = []
    captured_lines: list[str] = []
    original_section = screen._draw_intro_section

    def record_section(surface, rect, **kwargs):
        section_titles.append(str(kwargs.get("title", "")))
        captured_lines.extend(str(line) for line in kwargs.get("lines", ()))
        return original_section(surface, rect, **kwargs)

    screen._draw_intro_section = record_section

    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        screen.render(surface)

        assert "Practice Summary" in section_titles
        assert "Runtime drill instructions." in captured_lines
        assert any("Practice complete" in line for line in captured_lines)
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
