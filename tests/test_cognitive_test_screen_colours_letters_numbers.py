from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from cfast_trainer.app import App, CognitiveTestScreen, MenuItem, MenuScreen
from cfast_trainer.cognitive_core import Phase
from cfast_trainer.cognitive_core import TestSnapshot as SnapshotModel
from cfast_trainer.colours_letters_numbers import (
    ColoursLettersNumbersDiamond,
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


def _install_recording_fonts(*fonts: object) -> list[str]:
    captured: list[str] = []
    for obj in fonts:
        for attr in ("_small_font", "_tiny_font", "_mid_font", "_big_font"):
            font = getattr(obj, attr, None)
            if isinstance(font, _RecordingFont) or font is None:
                continue
            setattr(obj, attr, _RecordingFont(font, captured))
    return captured


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
        lane_colors=("RED", "YELLOW", "GREEN"),
        lane_start_norm=0.48,
        lane_end_norm=0.92,
        diamonds=(),
        missed_diamonds=0,
        cleared_diamonds=0,
        points=0.0,
        memory_choice_keys=("A", "S", "D", "F", "G"),
    )


def _build_training_payload() -> ColoursLettersNumbersTrainingPayload:
    return ColoursLettersNumbersTrainingPayload(
        target_sequence="ABCDE",
        options=(),
        options_active=False,
        memory_answered=False,
        math_answered=False,
        math_prompt="Type the full sequence and press Enter.",
        lane_colors=("RED", "YELLOW", "GREEN"),
        lane_start_norm=0.48,
        lane_end_norm=0.92,
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


def _build_six_choice_payload() -> ColoursLettersNumbersPayload:
    return ColoursLettersNumbersPayload(
        target_sequence=None,
        options=(
            ColoursLettersNumbersOption(code=1, label="ABCDE"),
            ColoursLettersNumbersOption(code=2, label="ABCDG"),
            ColoursLettersNumbersOption(code=3, label="ABXDE"),
            ColoursLettersNumbersOption(code=4, label="AYCDE"),
            ColoursLettersNumbersOption(code=5, label="BBCDE"),
            ColoursLettersNumbersOption(code=6, label="QBCDE"),
        ),
        options_active=True,
        memory_answered=False,
        math_answered=False,
        math_prompt="7 + 5 =",
        lane_colors=("RED", "YELLOW", "GREEN"),
        lane_start_norm=0.48,
        lane_end_norm=0.92,
        diamonds=(),
        missed_diamonds=0,
        cleared_diamonds=0,
        points=0.0,
        memory_choice_keys=("A", "S", "D", "F", "G", "H"),
    )


def _build_dual_math_payload() -> ColoursLettersNumbersTrainingPayload:
    return ColoursLettersNumbersTrainingPayload(
        target_sequence="ABCDE",
        options=(),
        options_active=False,
        memory_answered=False,
        math_answered=False,
        math_prompt="12 + 4 =",
        lane_colors=("RED", "YELLOW", "GREEN"),
        lane_start_norm=0.48,
        lane_end_norm=0.92,
        diamonds=(),
        missed_diamonds=0,
        cleared_diamonds=0,
        points=0.0,
        memory_input_active=False,
        input_label="Main Math",
        show_text_entry=True,
        static_text="--",
        control_hint="Memory: A/S/D/F/G or mouse  |  Colour lanes: Q/W/E  |  Enter: main math  |  1-5 or mouse: bonus math",
        colour_active=True,
        math_active=True,
        memory_active=True,
        memory_choice_keys=("A", "S", "D", "F", "G"),
        secondary_math_prompt="3 + 4 =",
        secondary_math_options=(
            ColoursLettersNumbersOption(code=1, label="5"),
            ColoursLettersNumbersOption(code=2, label="6"),
            ColoursLettersNumbersOption(code=3, label="7"),
            ColoursLettersNumbersOption(code=4, label="8"),
            ColoursLettersNumbersOption(code=5, label="9"),
        ),
        secondary_math_choice_active=True,
        secondary_math_choice_keys=("1", "2", "3", "4", "5"),
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


def test_cln_memory_choice_hotkey_preserves_typed_math_buffer() -> None:
    _app, screen, engine = _build_screen(_build_payload(options_active=True))
    try:
        screen._input = "42"

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_s, "mod": 0, "unicode": ""})
        )

        assert engine.answers == ["MEM:2"]
        assert screen._input == "42"
    finally:
        pygame.quit()


def test_app_routes_scaled_mouse_click_to_cln_option_hitbox() -> None:
    app, screen, engine = _build_screen(_build_payload(options_active=True))
    try:
        display_surface = pygame.display.get_surface()
        assert display_surface is not None
        get_window_size = getattr(pygame.display, "get_window_size", None)
        window_size = (
            tuple(int(v) for v in get_window_size())
            if callable(get_window_size)
            else display_surface.get_size()
        )
        app_surface = pygame.Surface(
            (display_surface.get_width() * 2, display_surface.get_height() * 2),
            pygame.SRCALPHA,
        )
        app.set_surface(app_surface)
        screen.render(app_surface)

        hitbox = screen._cln_option_hitboxes[2]
        click_pos = (
            int(round(hitbox.centerx * window_size[0] / app_surface.get_width())),
            int(round(hitbox.centery * window_size[1] / app_surface.get_height())),
        )
        app.handle_event(
            pygame.event.Event(
                pygame.MOUSEBUTTONDOWN,
                {"button": 1, "pos": click_pos},
            )
        )

        assert engine.answers == ["MEM:2"]
    finally:
        pygame.quit()


def test_cln_live_screen_hides_footer_scoreboard() -> None:
    _app, screen, _engine = _build_screen(_build_payload(options_active=True))
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        captured = _install_recording_fonts(screen)

        screen.render(surface)

        assert not any(text.startswith("Scored ") for text in captured)
        assert not any(text.startswith("Clear ") for text in captured)
        assert not any(text.startswith("Miss ") for text in captured)
        assert not any(text.startswith("Pts ") for text in captured)
    finally:
        pygame.quit()


def test_cln_main_math_accepts_number_row_keys_without_unicode() -> None:
    _app, screen, engine = _build_screen(_build_payload(options_active=False))
    try:
        for key in (pygame.K_4, pygame.K_2):
            screen.handle_event(
                pygame.event.Event(pygame.KEYDOWN, {"key": key, "mod": 0, "unicode": ""})
            )

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": 0, "unicode": ""})
        )

        assert engine.answers == ["42"]
    finally:
        pygame.quit()


def test_cln_main_math_accepts_numeric_keypad_digits() -> None:
    _app, screen, engine = _build_screen(_build_payload(options_active=False))
    try:
        for key in (pygame.K_KP4, pygame.K_KP2):
            screen.handle_event(
                pygame.event.Event(pygame.KEYDOWN, {"key": key, "mod": 0, "unicode": ""})
            )

        screen.handle_event(
            pygame.event.Event(
                pygame.KEYDOWN,
                {"key": pygame.K_KP_ENTER, "mod": 0, "unicode": ""},
            )
        )

        assert engine.answers == ["42"]
    finally:
        pygame.quit()


def test_cln_color_keys_use_qwer_mapping() -> None:
    _app, screen, engine = _build_screen(_build_payload(options_active=False))
    try:
        for key, expected in (
            (pygame.K_q, "CLR:Q"),
            (pygame.K_w, "CLR:W"),
            (pygame.K_e, "CLR:E"),
        ):
            screen.handle_event(
                pygame.event.Event(pygame.KEYDOWN, {"key": key, "mod": 0, "unicode": ""})
            )
            assert engine.answers[-1] == expected
        before = list(engine.answers)
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_r, "mod": 0, "unicode": ""})
        )
        assert engine.answers == before
    finally:
        pygame.quit()


def test_cln_render_uses_shifted_lane_window_and_filled_diamonds_only() -> None:
    payload = _build_payload(options_active=False)
    payload = ColoursLettersNumbersPayload(
        target_sequence=payload.target_sequence,
        options=payload.options,
        options_active=payload.options_active,
        memory_answered=payload.memory_answered,
        math_answered=payload.math_answered,
        math_prompt=payload.math_prompt,
        lane_colors=payload.lane_colors,
        lane_start_norm=payload.lane_start_norm,
        lane_end_norm=payload.lane_end_norm,
        diamonds=(
            ColoursLettersNumbersDiamond(id=1, color="RED", row=1, x_norm=0.64),
        ),
        missed_diamonds=payload.missed_diamonds,
        cleared_diamonds=payload.cleared_diamonds,
        points=payload.points,
        memory_choice_keys=payload.memory_choice_keys,
    )
    _app, screen, _engine = _build_screen(payload)
    polygon_calls: list[int] = []
    original_polygon = pygame.draw.polygon
    try:
        surface = pygame.display.get_surface()
        assert surface is not None

        def fake_polygon(surface, color, points, width=0):
            polygon_calls.append(int(width))
            return pygame.Rect(0, 0, 1, 1)

        pygame.draw.polygon = fake_polygon  # type: ignore[assignment]
        screen.render(surface)

        assert payload.lane_start_norm == 0.48
        assert payload.lane_end_norm == 0.92
        assert polygon_calls == [0]
    finally:
        pygame.draw.polygon = original_polygon  # type: ignore[assignment]
        pygame.quit()


def test_cln_six_choice_payload_accepts_h_mapping() -> None:
    _app, screen, engine = _build_screen(_build_six_choice_payload())
    try:
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_h, "mod": 0, "unicode": ""})
        )
        assert engine.answers == ["MEM:6"]
    finally:
        pygame.quit()


def test_cln_dual_math_mouse_click_submits_secondary_math_choice() -> None:
    _app, screen, engine = _build_screen(
        _build_dual_math_payload(),
        title="Colours, Letters and Numbers: Overdrive Dual Math",
    )
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        screen.render(surface)

        hitbox = screen._cln_secondary_math_hitboxes[3]
        screen.handle_event(
            pygame.event.Event(
                pygame.MOUSEBUTTONDOWN,
                {"button": 1, "pos": hitbox.center},
            )
        )
        assert engine.answers == ["MATH2:3"]
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
