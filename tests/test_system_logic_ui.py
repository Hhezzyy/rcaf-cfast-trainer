from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from cfast_trainer.app import App, CognitiveTestScreen, MenuItem, MenuScreen
from cfast_trainer.cognitive_core import Phase
from cfast_trainer.cognitive_core import TestSnapshot as SnapshotModel
from cfast_trainer.system_logic import (
    SystemLogicAnswerChoice,
    SystemLogicDocument,
    SystemLogicIndexEntry,
    SystemLogicPayload,
)


@dataclass
class _FakeSystemLogicEngine:
    payload: SystemLogicPayload
    phase: Phase = Phase.PRACTICE
    title: str = "System Logic"

    def __post_init__(self) -> None:
        self.submissions: list[str] = []

    def snapshot(self) -> SnapshotModel:
        return SnapshotModel(
            title=self.title,
            phase=self.phase,
            prompt=self.payload.question,
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
        self.submissions.append(str(raw))
        return True

    def update(self) -> None:
        return


def _build_payload() -> SystemLogicPayload:
    return SystemLogicPayload(
        scenario_code="SYS-999",
        system_family="fuel",
        index_entries=(
            SystemLogicIndexEntry(
                code=0,
                label="Tank Status",
                top_document=SystemLogicDocument(
                    title="Fuel Table",
                    kind="table",
                    table_headers=("Tank", "Litres"),
                    table_rows=(("Forward", "2800"), ("Aft", "2400"), ("Reserve", "500")),
                ),
                bottom_document=SystemLogicDocument(
                    title="Feed Selector",
                    kind="facts",
                    lines=("Crossfeed valve: OPEN", "Selected feed tank: FORWARD"),
                ),
            ),
            SystemLogicIndexEntry(
                code=1,
                label="Flow Rule",
                top_document=SystemLogicDocument(
                    title="Endurance Rule",
                    kind="equation",
                    lines=("accessible fuel = usable tanks - reserve", "minutes = floor(fuel / burn)"),
                ),
                bottom_document=SystemLogicDocument(
                    title="Feed Path",
                    kind="diagram",
                    diagram_paths=(("Forward", "Crossfeed", "Feed manifold", "Engine"),),
                ),
            ),
            SystemLogicIndexEntry(
                code=2,
                label="Mission Card",
                top_document=SystemLogicDocument(
                    title="Burn Reference",
                    kind="graph",
                    graph_points=(("Idle", 40), ("Cruise", 90), ("Climb", 110)),
                    graph_unit="L/min",
                ),
                bottom_document=SystemLogicDocument(
                    title="Mission Load",
                    kind="facts",
                    lines=("Current burn rate: 90 L/min",),
                ),
            ),
            SystemLogicIndexEntry(
                code=3,
                label="Answer Check",
                top_document=SystemLogicDocument(
                    title="Question Focus",
                    kind="facts",
                    lines=("Compute whole-minute endurance only.",),
                ),
                bottom_document=SystemLogicDocument(
                    title="Common Errors",
                    kind="facts",
                    lines=("Do not count locked reserve.",),
                ),
            ),
        ),
        question="How many whole minutes of fuel are available before the locked reserve is reached?",
        answer_choices=(
            SystemLogicAnswerChoice(1, "31 min"),
            SystemLogicAnswerChoice(2, "40 min"),
            SystemLogicAnswerChoice(3, "52 min"),
            SystemLogicAnswerChoice(4, "58 min"),
            SystemLogicAnswerChoice(5, "65 min"),
        ),
        correct_choice_code=4,
        reasoning_mode="quantitative_duration",
        required_index_codes=(0, 1, 2),
        required_document_kinds=("table", "equation"),
    )


def _build_screen(engine: _FakeSystemLogicEngine) -> CognitiveTestScreen:
    pygame.init()
    surface = pygame.display.set_mode((960, 540))
    font = pygame.font.Font(None, 36)
    app = App(surface=surface, font=font)
    root = MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True)
    app.push(root)
    screen = CognitiveTestScreen(app, engine_factory=lambda: engine)
    app.push(screen)
    return screen


def test_system_logic_layout_matches_guide_structure() -> None:
    engine = _FakeSystemLogicEngine(payload=_build_payload())
    screen = _build_screen(engine)
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        screen.render(surface)

        layout = screen._system_logic_layout
        assert layout is not None
        assert layout.top_pane.left == layout.bottom_pane.left
        assert layout.top_pane.right < layout.index_panel.left
        assert layout.top_pane.bottom < layout.bottom_pane.top
        assert len(layout.answer_rects) == 5
        assert [screen._system_logic_choice_key_label(code) for code in range(1, 6)] == [
            "A",
            "B",
            "C",
            "D",
            "E",
        ]
    finally:
        pygame.quit()


def test_system_logic_index_navigation_changes_rendered_pane_content() -> None:
    engine = _FakeSystemLogicEngine(payload=_build_payload())
    screen = _build_screen(engine)
    try:
        surface = pygame.display.get_surface()
        assert surface is not None

        screen.render(surface)
        before = pygame.image.tobytes(surface, "RGBA")

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "mod": 0, "unicode": ""})
        )
        assert screen._system_logic_index_index == 1

        screen.render(surface)
        after = pygame.image.tobytes(surface, "RGBA")
        assert after != before
    finally:
        pygame.quit()


def test_system_logic_choice_keys_wait_for_enter_then_submit() -> None:
    engine = _FakeSystemLogicEngine(payload=_build_payload())
    screen = _build_screen(engine)
    try:
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_c, "mod": 0, "unicode": "c"})
        )

        assert engine.submissions == []
        assert screen._system_logic_choice == 3
        assert screen._input == "3"

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": 0, "unicode": "\r"})
        )

        assert engine.submissions == ["3"]
    finally:
        pygame.quit()


def test_system_logic_renderer_handles_facts_tables_graphs_equations_and_diagrams() -> None:
    engine = _FakeSystemLogicEngine(payload=_build_payload())
    screen = _build_screen(engine)
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        for idx in range(4):
            screen._system_logic_index_index = idx
            screen.render(surface)
    finally:
        pygame.quit()


def test_system_logic_renderer_is_used_for_drill_titles_that_start_with_system_logic() -> None:
    engine = _FakeSystemLogicEngine(
        payload=_build_payload(),
        title="System Logic: Quantitative Anchor",
    )
    screen = _build_screen(engine)
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        screen.render(surface)

        layout = screen._system_logic_layout
        assert layout is not None
        assert layout.index_panel.width > 0
        assert len(layout.answer_rects) == 5
    finally:
        pygame.quit()
