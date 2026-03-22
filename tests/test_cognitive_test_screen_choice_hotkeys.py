from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from cfast_trainer.app import App, CognitiveTestScreen, MenuItem, MenuScreen
from cfast_trainer.cognitive_core import Phase
from cfast_trainer.cognitive_core import TestSnapshot as SnapshotModel
from cfast_trainer.math_reasoning import MathReasoningOption, MathReasoningPayload
from cfast_trainer.visual_search import VisualSearchPayload, VisualSearchTaskKind


@dataclass
class _ChoiceFakeEngine:
    payload: MathReasoningPayload
    phase: Phase = Phase.PRACTICE
    title: str = "Mathematics Reasoning"

    def __post_init__(self) -> None:
        self.submissions: list[str] = []

    def snapshot(self) -> SnapshotModel:
        return SnapshotModel(
            title=self.title,
            phase=self.phase,
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
        self.submissions.append(str(raw))
        return True

    def update(self) -> None:
        return


@dataclass
class _TypingFakeEngine:
    payload: object | None
    phase: Phase = Phase.PRACTICE
    title: str = "Visual Search"

    def __post_init__(self) -> None:
        self.submissions: list[str] = []

    def snapshot(self) -> SnapshotModel:
        return SnapshotModel(
            title=self.title,
            phase=self.phase,
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
        self.submissions.append(str(raw))
        return True

    def update(self) -> None:
        return


def _build_screen(engine: _ChoiceFakeEngine) -> CognitiveTestScreen:
    pygame.init()
    surface = pygame.display.set_mode((960, 540))
    font = pygame.font.Font(None, 36)
    app = App(surface=surface, font=font)
    root = MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True)
    app.push(root)
    screen = CognitiveTestScreen(app, engine_factory=lambda: engine)
    app.push(screen)
    return screen


def _math_payload() -> MathReasoningPayload:
    return MathReasoningPayload(
        domain="test",
        stem="pick one",
        options=(
            MathReasoningOption(code=1, text="A", value=10),
            MathReasoningOption(code=2, text="B", value=20),
            MathReasoningOption(code=3, text="C", value=30),
            MathReasoningOption(code=4, text="D", value=40),
            MathReasoningOption(code=5, text="E", value=50),
        ),
        correct_code=3,
        correct_value=30,
    )


def test_asdfg_choice_key_submits_without_enter() -> None:
    engine = _ChoiceFakeEngine(payload=_math_payload())
    screen = _build_screen(engine)
    try:
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_d, "mod": 0, "unicode": "d"})
        )

        assert engine.submissions == ["3"]
        assert screen._input == ""
        assert screen._math_choice == 1
    finally:
        pygame.quit()


def test_numeric_choice_key_submits_without_enter() -> None:
    engine = _ChoiceFakeEngine(payload=_math_payload())
    screen = _build_screen(engine)
    try:
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_3, "mod": 0, "unicode": "3"})
        )

        assert engine.submissions == ["3"]
        assert screen._input == ""
        assert screen._math_choice == 1
    finally:
        pygame.quit()


def test_visual_search_requires_enter_to_submit_typed_block_code() -> None:
    engine = _TypingFakeEngine(
        payload=VisualSearchPayload(
            kind=VisualSearchTaskKind.ALPHANUMERIC,
            rows=2,
            cols=3,
            target="R",
            cells=("A", "B", "C", "D", "E", "R"),
            cell_codes=(10, 11, 12, 13, 14, 15),
            full_credit_error=0,
            zero_credit_error=1,
        ),
    )
    screen = _build_screen(engine)
    try:
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_1, "mod": 0, "unicode": "1"})
        )
        assert engine.submissions == []

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_5, "mod": 0, "unicode": "5"})
        )

        assert engine.submissions == []
        assert screen._input == "15"

        screen.handle_event(
            pygame.event.Event(
                pygame.KEYDOWN,
                {"key": pygame.K_RETURN, "mod": 0, "unicode": "\r"},
            )
        )

        assert engine.submissions == ["15"]
        assert screen._input == ""
    finally:
        pygame.quit()


def test_visual_search_allows_extra_digits_before_enter_submission() -> None:
    engine = _TypingFakeEngine(
        payload=VisualSearchPayload(
            kind=VisualSearchTaskKind.ALPHANUMERIC,
            rows=2,
            cols=3,
            target="R",
            cells=("A", "B", "C", "D", "E", "R"),
            cell_codes=(10, 11, 12, 13, 14, 15),
            full_credit_error=0,
            zero_credit_error=1,
        ),
    )
    screen = _build_screen(engine)
    try:
        for key, digit in (
            (pygame.K_1, "1"),
            (pygame.K_5, "5"),
            (pygame.K_2, "2"),
        ):
            screen.handle_event(
                pygame.event.Event(pygame.KEYDOWN, {"key": key, "mod": 0, "unicode": digit})
            )

        assert engine.submissions == []
        assert screen._input == "152"

        screen.handle_event(
            pygame.event.Event(
                pygame.KEYDOWN,
                {"key": pygame.K_RETURN, "mod": 0, "unicode": "\r"},
            )
        )

        assert engine.submissions == ["152"]
        assert screen._input == ""
    finally:
        pygame.quit()


def test_manual_typed_numeric_input_ignores_backspace() -> None:
    engine = _TypingFakeEngine(payload=None, title="Numerical Operations")
    screen = _build_screen(engine)
    try:
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_1, "mod": 0, "unicode": "1"})
        )
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_2, "mod": 0, "unicode": "2"})
        )
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_BACKSPACE, "mod": 0, "unicode": ""})
        )

        assert screen._input == "12"
    finally:
        pygame.quit()
