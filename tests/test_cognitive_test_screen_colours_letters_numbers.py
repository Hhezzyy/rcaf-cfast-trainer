from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from cfast_trainer.app import App, CognitiveTestScreen, MenuItem, MenuScreen
from cfast_trainer.cognitive_core import Phase
from cfast_trainer.cognitive_core import TestSnapshot as SnapshotModel
from cfast_trainer.colours_letters_numbers import (
    ColoursLettersNumbersOption,
    ColoursLettersNumbersPayload,
    ColoursLettersNumbersRuntimePayload,
    ColoursLettersNumbersTrainingPayload,
)


class _FakeClnEngine:
    def __init__(self, payload: ColoursLettersNumbersRuntimePayload, *, title: str) -> None:
        self._payload = payload
        self._title = str(title)
        self.answers: list[str] = []

    def snapshot(self) -> SnapshotModel:
        return SnapshotModel(
            title=self._title,
            phase=Phase.PRACTICE,
            prompt="",
            input_hint="",
            time_remaining_s=None,
            attempted_scored=0,
            correct_scored=0,
            payload=self._payload,
        )

    def can_exit(self) -> bool:
        return True

    def start_practice(self) -> None:
        pass

    def start_scored(self) -> None:
        pass

    def submit_answer(self, raw: str) -> bool:
        self.answers.append(raw)
        return True

    def update(self) -> None:
        pass


def _build_screen(
    payload: ColoursLettersNumbersRuntimePayload,
    *,
    title: str = "Colours, Letters and Numbers",
) -> tuple[App, CognitiveTestScreen, _FakeClnEngine]:
    pygame.init()
    surface = pygame.display.set_mode((960, 540))
    font = pygame.font.Font(None, 36)
    app = App(surface=surface, font=font)
    root = MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True)
    app.push(root)
    engine = _FakeClnEngine(payload, title=title)
    screen = CognitiveTestScreen(app, engine_factory=lambda: engine)
    app.push(screen)
    return app, screen, engine


def _build_payload(*, options_active: bool = True) -> ColoursLettersNumbersPayload:
    return ColoursLettersNumbersPayload(
        target_sequence=None if options_active else "ABCDE",
        options=(
            ColoursLettersNumbersOption(code=1, label="ABCDE"),
            ColoursLettersNumbersOption(code=2, label="ABCDF"),
            ColoursLettersNumbersOption(code=3, label="ABGDE"),
            ColoursLettersNumbersOption(code=4, label="XBCDE"),
            ColoursLettersNumbersOption(code=5, label="ABCDX"),
        ),
        options_active=options_active,
        memory_answered=False,
        math_answered=False,
        math_prompt="2 + 2 =",
        lane_colors=("RED", "YELLOW", "GREEN", "BLUE"),
        lane_start_norm=0.54,
        lane_end_norm=0.98,
        diamonds=(),
        missed_diamonds=0,
        cleared_diamonds=0,
        points=0.0,
    )


def _build_training_payload() -> ColoursLettersNumbersTrainingPayload:
    return ColoursLettersNumbersTrainingPayload(
        target_sequence="ABCDE",
        options=(),
        options_active=False,
        memory_answered=False,
        math_answered=False,
        math_prompt="Type the full sequence and press Enter.",
        lane_colors=("RED", "YELLOW", "GREEN", "BLUE"),
        lane_start_norm=0.54,
        lane_end_norm=0.98,
        diamonds=(),
        missed_diamonds=0,
        cleared_diamonds=0,
        points=0.0,
        memory_input_active=True,
        memory_input_max_length=8,
        input_label="Sequence Entry",
        show_text_entry=True,
        static_text="--",
        control_hint="Type letters then Enter",
        top_hint_override="Type the full sequence while it remains visible.",
        colour_active=False,
        math_active=False,
        memory_active=True,
    )


def test_cln_memory_keys_use_asdfg_mapping() -> None:
    _app, screen, engine = _build_screen(_build_payload(options_active=True))
    try:
        for key, expected in (
            (pygame.K_a, "MEM:1"),
            (pygame.K_s, "MEM:2"),
            (pygame.K_d, "MEM:3"),
            (pygame.K_f, "MEM:4"),
            (pygame.K_g, "MEM:5"),
        ):
            screen.handle_event(
                pygame.event.Event(pygame.KEYDOWN, {"key": key, "mod": 0, "unicode": ""})
            )
            assert engine.answers[-1] == expected
    finally:
        pygame.quit()


def test_cln_memory_mouse_click_uses_rendered_option_hitboxes() -> None:
    _app, screen, engine = _build_screen(_build_payload(options_active=True))
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        screen.render(surface)

        hitbox = screen._cln_option_hitboxes[2]
        screen.handle_event(
            pygame.event.Event(
                pygame.MOUSEBUTTONDOWN,
                {"button": 1, "pos": hitbox.center},
            )
        )

        assert engine.answers == ["MEM:2"]
    finally:
        pygame.quit()


def test_cln_color_keys_use_qwer_mapping() -> None:
    _app, screen, engine = _build_screen(_build_payload(options_active=False))
    try:
        for key, expected in (
            (pygame.K_q, "CLR:Q"),
            (pygame.K_w, "CLR:W"),
            (pygame.K_e, "CLR:E"),
            (pygame.K_r, "CLR:R"),
        ):
            screen.handle_event(
                pygame.event.Event(pygame.KEYDOWN, {"key": key, "mod": 0, "unicode": ""})
            )
            assert engine.answers[-1] == expected
    finally:
        pygame.quit()


def test_cln_training_payload_accepts_typed_sequence_input_under_drill_title() -> None:
    _app, screen, engine = _build_screen(
        _build_training_payload(),
        title="Colours, Letters and Numbers: Sequence Copy",
    )
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        screen.render(surface)

        for key, ch in (
            (pygame.K_a, "a"),
            (pygame.K_b, "b"),
            (pygame.K_c, "c"),
            (pygame.K_d, "d"),
            (pygame.K_e, "e"),
        ):
            screen.handle_event(
                pygame.event.Event(pygame.KEYDOWN, {"key": key, "mod": 0, "unicode": ch})
            )

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": 0, "unicode": ""})
        )

        assert engine.answers == ["MEMSEQ:ABCDE"]
    finally:
        pygame.quit()
