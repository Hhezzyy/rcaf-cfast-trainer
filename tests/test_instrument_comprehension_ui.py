from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from dataclasses import dataclass, replace

import pygame

from cfast_trainer.ant_drills import AntDrillMode
from cfast_trainer.ant_workouts import (
    AntWorkoutBlockPlan,
    AntWorkoutPlan,
    AntWorkoutSession,
)
from cfast_trainer.app import (
    AntWorkoutScreen,
    App,
    CognitiveTestScreen,
    DisplayBootstrapResult,
    MenuItem,
    MenuScreen,
    _apply_display_bootstrap_to_app,
)
from cfast_trainer.cognitive_core import Phase
from cfast_trainer.cognitive_core import TestSnapshot as SnapshotModel
from cfast_trainer.ic_drills import build_ic_description_run_drill
from cfast_trainer.instrument_aircraft_cards import aircraft_card_semantic_drift_tags
from cfast_trainer.instrument_comprehension import (
    InstrumentAircraftViewPreset,
    InstrumentComprehensionConfig,
    InstrumentComprehensionPayload,
    InstrumentComprehensionTrialKind,
    InstrumentHeadingDisplayMode,
    InstrumentOption,
    InstrumentOptionRenderMode,
    InstrumentState,
    build_instrument_comprehension_test,
    instrument_aircraft_view_preset_for_code,
)
from cfast_trainer.instrument_orientation_solver import (
    attitude_display_observation_from_bank_pitch,
    heading_display_observation_from_heading,
)


class _FakeInstrumentEngine:
    def __init__(self, payload: InstrumentComprehensionPayload) -> None:
        self._payload = payload

    def snapshot(self) -> SnapshotModel:
        return SnapshotModel(
            title="Instrument Comprehension",
            phase=Phase.PRACTICE,
            prompt="Read the instruments and choose the matching aircraft image (A/S/D/F/G).",
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
        return True

    def update(self) -> None:
        pass


@dataclass
class _FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class _FakeGlRenderer:
    def resize(self, *, window_size: tuple[int, int]) -> None:
        _ = window_size

    def render_frame(self, *, ui_surface: pygame.Surface, scene) -> None:
        _ = (ui_surface, scene)


class _RecordingFont:
    def __init__(self, base: pygame.font.Font, sink: list[str]) -> None:
        self._base = base
        self._sink = sink

    def render(self, text: str, antialias: bool, color: object) -> pygame.Surface:
        self._sink.append(str(text))
        return self._base.render(text, antialias, color)

    def __getattr__(self, name: str) -> object:
        return getattr(self._base, name)


def _base_state() -> InstrumentState:
    return InstrumentState(
        speed_kts=220,
        altitude_ft=5000,
        vertical_rate_fpm=0,
        bank_deg=-12,
        pitch_deg=6,
        heading_deg=78,
        slip=0,
    )


def _build_payload() -> InstrumentComprehensionPayload:
    base = _base_state()
    options = tuple(
        InstrumentOption(
            code=code,
            state=replace(
                base,
                heading_deg=(base.heading_deg + (code - 1) * 28) % 360,
                pitch_deg=base.pitch_deg + (code - 3),
                bank_deg=base.bank_deg + (code - 3) * 4,
            ),
            description=f"Option {code}",
            view_preset=instrument_aircraft_view_preset_for_code(code),
        )
        for code in range(1, 6)
    )
    return InstrumentComprehensionPayload(
        kind=InstrumentComprehensionTrialKind.INSTRUMENTS_TO_AIRCRAFT,
        prompt_state=base,
        prompt_description="",
        options=options,
        option_render_mode=InstrumentOptionRenderMode.AIRCRAFT,
        option_errors=(0, 12, 24, 36, 48),
        full_credit_error=0,
        zero_credit_error=90,
    )


def _build_reverse_payload() -> InstrumentComprehensionPayload:
    base = _base_state()
    options = tuple(
        InstrumentOption(
            code=code,
            state=replace(
                base,
                heading_deg=(base.heading_deg + (code - 1) * 32) % 360,
                pitch_deg=base.pitch_deg + (code - 3),
                bank_deg=base.bank_deg + (code - 3) * 5,
            ),
            description=f"Option {code}",
        )
        for code in range(1, 6)
    )
    return InstrumentComprehensionPayload(
        kind=InstrumentComprehensionTrialKind.AIRCRAFT_TO_INSTRUMENTS,
        prompt_state=options[2].state,
        prompt_description="",
        prompt_view_preset=InstrumentAircraftViewPreset.TOP_OBLIQUE,
        options=options,
        option_render_mode=InstrumentOptionRenderMode.INSTRUMENT_PANEL,
        option_errors=(40, 24, 0, 18, 36),
        full_credit_error=0,
        zero_credit_error=90,
    )


def _build_description_payload() -> InstrumentComprehensionPayload:
    base = _base_state()
    options = tuple(
        InstrumentOption(
            code=code,
            state=replace(
                base,
                speed_kts=base.speed_kts + ((code - 3) * 10),
                altitude_ft=base.altitude_ft + ((code - 3) * 1000),
                vertical_rate_fpm=(code - 3) * 400,
                heading_deg=(base.heading_deg + ((code - 3) * 90)) % 360,
            ),
            description=f"Description option {code}",
        )
        for code in range(1, 6)
    )
    return InstrumentComprehensionPayload(
        kind=InstrumentComprehensionTrialKind.INSTRUMENTS_TO_DESCRIPTION,
        prompt_state=base,
        prompt_description="",
        options=options,
        option_render_mode=InstrumentOptionRenderMode.DESCRIPTION,
        option_errors=(36, 18, 0, 16, 32),
        full_credit_error=0,
        zero_credit_error=90,
    )


def _build_screen(
    payload: InstrumentComprehensionPayload,
    *,
    size: tuple[int, int] = (960, 540),
) -> tuple[App, CognitiveTestScreen]:
    return _build_live_screen(_FakeInstrumentEngine(payload), size=size)


def _build_live_screen(
    engine: object,
    *,
    size: tuple[int, int] = (960, 540),
    test_code: str | None = None,
) -> tuple[App, CognitiveTestScreen]:
    pygame.init()
    surface = pygame.display.set_mode(size)
    font = pygame.font.Font(None, 36)
    app = App(surface=surface, font=font)
    root = MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True)
    app.push(root)
    screen = CognitiveTestScreen(app, engine_factory=lambda: engine, test_code=test_code)
    app.push(screen)
    return app, screen


def _install_recording_fonts(*fonts: object) -> list[str]:
    captured: list[str] = []
    for obj in fonts:
        for attr in ("_small_font", "_tiny_font", "_mid_font", "_big_font"):
            font = getattr(obj, attr, None)
            if isinstance(font, _RecordingFont) or font is None:
                continue
            setattr(obj, attr, _RecordingFont(font, captured))
    return captured


def test_instrument_screen_renders_after_display_bootstrap_sync() -> None:
    app, screen = _build_screen(_build_payload())
    try:
        display_surface = pygame.display.set_mode((1440, 900))
        bootstrap = DisplayBootstrapResult(
            display_surface=display_surface,
            app_surface=pygame.Surface(display_surface.get_size(), pygame.SRCALPHA),
            gl_renderer=_FakeGlRenderer(),
            active_window_flags=pygame.FULLSCREEN | pygame.OPENGL | pygame.DOUBLEBUF,
            gl_requested=True,
            gl_attempted=True,
        )
        _apply_display_bootstrap_to_app(
            app=app,
            bootstrap=bootstrap,
            window_mode="fullscreen",
        )

        app.render()

        assert app.surface.get_size() == (1440, 900)
        assert screen._instrument_part1_layout is not None
        assert app.current_run_state().display_mode == "FULLSCREEN"
    finally:
        pygame.quit()


def test_instrument_live_screen_hides_scored_counter() -> None:
    _app, screen = _build_screen(_build_payload())
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        captured = _install_recording_fonts(screen)

        screen.render(surface)

        assert not any(text.startswith("Scored") for text in captured)
    finally:
        pygame.quit()


def _assert_uniform_guide_card_grid(layout) -> None:
    assert layout is not None
    assert len(layout.card_rects) == 5
    assert layout.show_prompt_dial_labels is False
    assert len({(rect.width, rect.height) for rect in layout.card_rects}) == 1
    assert layout.card_rects[0].y == layout.card_rects[1].y
    assert layout.card_rects[2].y == layout.card_rects[3].y == layout.card_rects[4].y
    assert layout.card_rects[2].y > layout.card_rects[0].bottom
    assert layout.card_rects[0].x > layout.card_rects[2].x
    assert layout.card_rects[1].right < layout.card_rects[4].right


def _instrument_question_rect(surface: pygame.Surface) -> pygame.Rect:
    w, h = surface.get_size()
    margin = max(8, min(14, w // 72))
    frame = pygame.Rect(margin, margin, w - margin * 2, h - margin * 2)
    header_h = max(26, min(34, h // 16))
    header = pygame.Rect(frame.x + 2, frame.y + 2, frame.w - 4, header_h)
    work = pygame.Rect(frame.x + 2, header.bottom + 6, frame.w - 4, frame.bottom - header.bottom - 8)
    footer_h = 18
    return pygame.Rect(work.x + 3, work.y + 3, work.w - 6, work.h - footer_h - 4)


def _red_bounds(surface: pygame.Surface) -> tuple[int, int, int, int] | None:
    min_x = surface.get_width()
    min_y = surface.get_height()
    max_x = -1
    max_y = -1
    for y in range(surface.get_height()):
        for x in range(surface.get_width()):
            color = surface.get_at((x, y))
            if color.r <= 140 or color.r <= color.g + 20 or color.r <= color.b + 20:
                continue
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
    if max_x < min_x or max_y < min_y:
        return None
    return (min_x, min_y, max_x, max_y)


def _red_bounds_in_rect(
    surface: pygame.Surface, rect: pygame.Rect
) -> tuple[int, int, int, int] | None:
    min_x = rect.right
    min_y = rect.bottom
    max_x = rect.x - 1
    max_y = rect.y - 1
    for y in range(rect.y, rect.bottom):
        for x in range(rect.x, rect.right):
            color = surface.get_at((x, y))
            if color.a <= 0:
                continue
            if color.r <= 140 or color.r <= color.g + 20 or color.r <= color.b + 20:
                continue
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
    if max_x < min_x or max_y < min_y:
        return None
    return (min_x, min_y, max_x, max_y)


def _outer_ring_bytes(surface: pygame.Surface, rect: pygame.Rect) -> bytes:
    copy = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    cx, cy = rect.center
    radius = min(rect.w, rect.h) // 2
    inner_radius = int(round(radius * 0.62))
    outer_sq = radius * radius
    inner_sq = inner_radius * inner_radius
    for y in range(rect.y, rect.bottom):
        for x in range(rect.x, rect.right):
            dx = x - cx
            dy = y - cy
            dist_sq = dx * dx + dy * dy
            if dist_sq > outer_sq or dist_sq < inner_sq:
                continue
            copy.set_at((x, y), surface.get_at((x, y)))
    return pygame.image.tobytes(copy, "RGBA")


def _furthest_red_point(surface: pygame.Surface, rect: pygame.Rect) -> tuple[int, int] | None:
    center = rect.center
    best: tuple[int, tuple[int, int]] | None = None
    for y in range(rect.y, rect.bottom):
        for x in range(rect.x, rect.right):
            color = surface.get_at((x, y))
            if color.a <= 0:
                continue
            if color.r <= 140 or color.r <= color.g + 20 or color.r <= color.b + 20:
                continue
            dx = x - center[0]
            dy = y - center[1]
            dist_sq = dx * dx + dy * dy
            if best is None or dist_sq > best[0]:
                best = (dist_sq, (x, y))
    return None if best is None else best[1]


def _sky_boundary_y(surface: pygame.Surface, x: int, y_min: int, y_max: int) -> int | None:
    boundary: int | None = None
    for y in range(y_min, y_max):
        color = surface.get_at((x, y))
        if color.a <= 0:
            continue
        if color.b <= color.r + 20 or color.b <= color.g + 10:
            continue
        boundary = y
    return boundary


def _average_blue_y(
    surface: pygame.Surface,
    *,
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int,
) -> float | None:
    samples: list[int] = []
    for y in range(y_min, y_max):
        for x in range(x_min, x_max):
            color = surface.get_at((x, y))
            if color.a <= 0:
                continue
            if color.b <= color.r + 20 or color.b <= color.g + 10:
                continue
            samples.append(y)
    if not samples:
        return None
    return sum(samples) / len(samples)


def test_part1_layout_matches_guide_style_grid_and_hides_dial_captions() -> None:
    _app, screen = _build_screen(_build_payload())
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        screen.render(surface)

        layout = screen._instrument_part1_layout
        _assert_uniform_guide_card_grid(layout)
        assert [screen._choice_key_label(code) for code in range(1, 6)] == ["A", "S", "D", "F", "G"]
    finally:
        pygame.quit()


def test_reverse_part_layout_matches_uniform_guide_grid() -> None:
    _app, screen = _build_screen(_build_reverse_payload())
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        screen.render(surface)

        _assert_uniform_guide_card_grid(screen._instrument_part1_layout)
    finally:
        pygame.quit()


def test_instruction_screen_uses_standard_intro_overlay_without_part_preview(
    monkeypatch,
) -> None:
    clock = _FakeClock()
    engine = build_instrument_comprehension_test(
        clock=clock,
        seed=17,
        difficulty=0.5,
        config=InstrumentComprehensionConfig(scored_duration_s=20.0, practice_questions=1),
    )
    _app, screen = _build_live_screen(engine, test_code="instrument_comprehension")
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        calls = {"dials": 0, "cards": 0, "intro_sections": 0, "custom_intro": 0}

        def dials_stub(*args, **kwargs):
            _ = (args, kwargs)
            calls["dials"] += 1

        def cards_stub(*args, **kwargs):
            _ = (args, kwargs)
            calls["cards"] += 1

        def intro_stub(*args, **kwargs):
            _ = (args, kwargs)
            calls["intro_sections"] += 1

        def custom_intro_stub(*args, **kwargs):
            _ = (args, kwargs)
            calls["custom_intro"] += 1

        monkeypatch.setattr(screen, "_draw_orientation_prompt_dials", dials_stub)
        monkeypatch.setattr(screen, "_draw_aircraft_orientation_card", cards_stub)
        monkeypatch.setattr(screen, "_draw_intro_section", intro_stub)
        monkeypatch.setattr(screen, "_render_instrument_instruction_page", custom_intro_stub)

        screen.render(surface)

        assert calls["dials"] == 0
        assert calls["cards"] == 0
        assert calls["custom_intro"] == 0
        assert calls["intro_sections"] == 5
        assert screen._instrument_part1_layout is None
    finally:
        pygame.quit()


def test_transition_screen_uses_standard_intro_overlay_without_part_preview(monkeypatch) -> None:
    clock = _FakeClock()
    engine = build_instrument_comprehension_test(
        clock=clock,
        seed=19,
        difficulty=0.5,
        config=InstrumentComprehensionConfig(scored_duration_s=20.0, practice_questions=1),
    )
    engine.start_practice()
    assert engine.submit_answer("__skip_practice__") is True
    engine.start_scored()
    assert engine.submit_answer("__skip_section__") is True

    _app, screen = _build_live_screen(engine, test_code="instrument_comprehension")
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        calls = {"prompt": 0, "answers": 0, "intro_sections": 0, "custom_intro": 0}

        def prompt_stub(*args, **kwargs):
            _ = (args, kwargs)
            calls["prompt"] += 1

        def answer_stub(*args, **kwargs):
            _ = (args, kwargs)
            calls["answers"] += 1

        def intro_stub(*args, **kwargs):
            _ = (args, kwargs)
            calls["intro_sections"] += 1

        def custom_intro_stub(*args, **kwargs):
            _ = (args, kwargs)
            calls["custom_intro"] += 1

        monkeypatch.setattr(screen, "_draw_aircraft_prompt_card", prompt_stub)
        monkeypatch.setattr(screen, "_draw_instrument_panel_answer_card", answer_stub)
        monkeypatch.setattr(screen, "_draw_intro_section", intro_stub)
        monkeypatch.setattr(screen, "_render_instrument_instruction_page", custom_intro_stub)

        screen.render(surface)

        assert calls["prompt"] == 0
        assert calls["answers"] == 0
        assert calls["custom_intro"] == 0
        assert calls["intro_sections"] == 5
        assert screen._instrument_part1_layout is None
    finally:
        pygame.quit()


def test_heading_dial_keeps_red_aircraft_fixed_while_rose_rotates() -> None:
    _app, screen = _build_screen(_build_payload())
    try:
        surface_north = pygame.Surface((140, 140), pygame.SRCALPHA)
        surface_east = pygame.Surface((140, 140), pygame.SRCALPHA)
        rect = pygame.Rect(10, 10, 120, 120)

        screen._draw_heading_dial(
            surface_north,
            rect,
            0,
            mode=InstrumentHeadingDisplayMode.ROTATING_ROSE,
        )
        screen._draw_heading_dial(
            surface_east,
            rect,
            90,
            mode=InstrumentHeadingDisplayMode.ROTATING_ROSE,
        )

        assert _red_bounds(surface_north) == _red_bounds(surface_east)
        assert pygame.image.tobytes(surface_north, "RGBA") != pygame.image.tobytes(
            surface_east, "RGBA"
        )
    finally:
        pygame.quit()


def test_heading_dial_can_keep_rose_fixed_while_red_arrow_moves() -> None:
    _app, screen = _build_screen(_build_payload())
    try:
        surface_north = pygame.Surface((140, 140), pygame.SRCALPHA)
        surface_east = pygame.Surface((140, 140), pygame.SRCALPHA)
        rect = pygame.Rect(10, 10, 120, 120)

        screen._draw_heading_dial(
            surface_north,
            rect,
            0,
            mode=InstrumentHeadingDisplayMode.MOVING_ARROW,
        )
        screen._draw_heading_dial(
            surface_east,
            rect,
            90,
            mode=InstrumentHeadingDisplayMode.MOVING_ARROW,
        )

        assert _red_bounds(surface_north) != _red_bounds(surface_east)
        assert _outer_ring_bytes(surface_north, rect) == _outer_ring_bytes(surface_east, rect)
        assert pygame.image.tobytes(surface_north, "RGBA") != pygame.image.tobytes(
            surface_east, "RGBA"
        )
    finally:
        pygame.quit()


def test_heading_dial_semantics_match_cardinals_in_both_modes() -> None:
    _app, screen = _build_screen(_build_payload())
    try:
        rect = pygame.Rect(10, 10, 120, 120)

        moving_east = pygame.Surface((140, 140), pygame.SRCALPHA)
        moving_south = pygame.Surface((140, 140), pygame.SRCALPHA)
        rotating_east = pygame.Surface((140, 140), pygame.SRCALPHA)

        screen._draw_heading_dial(
            moving_east,
            rect,
            observation=heading_display_observation_from_heading(
                90,
                InstrumentHeadingDisplayMode.MOVING_ARROW,
            ),
        )
        screen._draw_heading_dial(
            moving_south,
            rect,
            observation=heading_display_observation_from_heading(
                180,
                InstrumentHeadingDisplayMode.MOVING_ARROW,
            ),
        )
        screen._draw_heading_dial(
            rotating_east,
            rect,
            observation=heading_display_observation_from_heading(
                90,
                InstrumentHeadingDisplayMode.ROTATING_ROSE,
            ),
        )

        east_tip = _furthest_red_point(moving_east, rect)
        south_tip = _furthest_red_point(moving_south, rect)
        rotating_tip = _furthest_red_point(rotating_east, rect)

        assert east_tip is not None
        assert south_tip is not None
        assert rotating_tip is not None
        assert east_tip[0] > rect.centerx + 16
        assert abs(east_tip[1] - rect.centery) < 18
        assert south_tip[1] > rect.centery + 16
        assert abs(south_tip[0] - rect.centerx) < 18
        assert rotating_tip[1] < rect.centery - 18
    finally:
        pygame.quit()


def test_attitude_dial_semantics_match_signed_pitch_and_bank() -> None:
    _app, screen = _build_screen(_build_payload())
    try:
        rect = pygame.Rect(10, 10, 140, 140)
        climb = pygame.Surface((160, 160), pygame.SRCALPHA)
        descent = pygame.Surface((160, 160), pygame.SRCALPHA)
        right_bank = pygame.Surface((160, 160), pygame.SRCALPHA)
        left_bank = pygame.Surface((160, 160), pygame.SRCALPHA)

        screen._draw_attitude_dial(
            climb,
            rect,
            observation=attitude_display_observation_from_bank_pitch(0, 10),
        )
        screen._draw_attitude_dial(
            descent,
            rect,
            observation=attitude_display_observation_from_bank_pitch(0, -10),
        )
        screen._draw_attitude_dial(
            right_bank,
            rect,
            observation=attitude_display_observation_from_bank_pitch(14, 0),
        )
        screen._draw_attitude_dial(
            left_bank,
            rect,
            observation=attitude_display_observation_from_bank_pitch(-14, 0),
        )

        center_x = rect.centerx
        center_band = rect.w // 10
        climb_average = _average_blue_y(
            climb,
            x_min=center_x - center_band,
            x_max=center_x + center_band,
            y_min=rect.y + 8,
            y_max=rect.bottom - 8,
        )
        descent_average = _average_blue_y(
            descent,
            x_min=center_x - center_band,
            x_max=center_x + center_band,
            y_min=rect.y + 8,
            y_max=rect.bottom - 8,
        )
        stripe_half = rect.w // 8
        left_x = rect.x + rect.w // 3
        right_x = rect.right - rect.w // 3
        right_bank_left = _average_blue_y(
            right_bank,
            x_min=left_x - stripe_half,
            x_max=left_x + stripe_half,
            y_min=rect.y + 8,
            y_max=rect.bottom - 8,
        )
        right_bank_right = _average_blue_y(
            right_bank,
            x_min=right_x - stripe_half,
            x_max=right_x + stripe_half,
            y_min=rect.y + 8,
            y_max=rect.bottom - 8,
        )
        left_bank_left = _average_blue_y(
            left_bank,
            x_min=left_x - stripe_half,
            x_max=left_x + stripe_half,
            y_min=rect.y + 8,
            y_max=rect.bottom - 8,
        )
        left_bank_right = _average_blue_y(
            left_bank,
            x_min=right_x - stripe_half,
            x_max=right_x + stripe_half,
            y_min=rect.y + 8,
            y_max=rect.bottom - 8,
        )

        assert climb_average is not None
        assert descent_average is not None
        assert right_bank_left is not None
        assert right_bank_right is not None
        assert left_bank_left is not None
        assert left_bank_right is not None

        assert climb_average > descent_average
        assert right_bank_right < right_bank_left
        assert left_bank_right > left_bank_left
    finally:
        pygame.quit()


def test_reverse_part_renders_aircraft_prompt_and_five_instrument_answer_cards(monkeypatch) -> None:
    _app, screen = _build_screen(_build_reverse_payload())
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        calls = {"prompt": 0, "answers": 0}

        def prompt_stub(*args, **kwargs):
            calls["prompt"] += 1

        def answer_stub(*args, **kwargs):
            calls["answers"] += 1

        monkeypatch.setattr(screen, "_draw_aircraft_prompt_card", prompt_stub)
        monkeypatch.setattr(screen, "_draw_instrument_panel_answer_card", answer_stub)
        screen.render(surface)

        assert calls["prompt"] == 1
        assert calls["answers"] == 5
        assert screen._instrument_part1_layout is not None
    finally:
        pygame.quit()


def test_part3_layout_uses_equal_rows_and_scales_with_surface() -> None:
    small_payload = _build_description_payload()
    _app, screen = _build_screen(small_payload)
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        screen.render(surface)

        small_layout = screen._instrument_part3_layout
        assert small_layout is not None
        assert len(small_layout.option_rects) == 5
        assert len({(rect.width, rect.height) for rect in small_layout.option_rects}) == 1
        small_question_rect = _instrument_question_rect(surface)
        assert small_layout.cluster_rect.top <= small_question_rect.top + 16
        assert small_question_rect.bottom - small_layout.option_rects[-1].bottom <= 16
        small_row_h = small_layout.option_rects[0].height
        small_cluster_h = small_layout.cluster_rect.height
    finally:
        pygame.quit()

    large_payload = _build_description_payload()
    _app, screen = _build_screen(large_payload, size=(1440, 900))
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        screen.render(surface)

        large_layout = screen._instrument_part3_layout
        assert large_layout is not None
        assert len({(rect.width, rect.height) for rect in large_layout.option_rects}) == 1
        large_question_rect = _instrument_question_rect(surface)
        assert large_layout.cluster_rect.top <= large_question_rect.top + 16
        assert large_question_rect.bottom - large_layout.option_rects[-1].bottom <= 16
        assert large_layout.option_rects[0].height > small_row_h
        assert large_layout.cluster_rect.height > small_cluster_h
    finally:
        pygame.quit()


def test_scored_instrument_screen_does_not_render_cannot_exit_copy() -> None:
    payload = _build_payload()

    class _NoExitEngine(_FakeInstrumentEngine):
        def snapshot(self) -> SnapshotModel:
            return SnapshotModel(
                title="Instrument Comprehension",
                phase=Phase.SCORED,
                prompt="Read the instruments and choose the matching aircraft image (A/S/D/F/G).",
                input_hint="",
                time_remaining_s=120.0,
                attempted_scored=2,
                correct_scored=1,
                payload=payload,
            )

        def can_exit(self) -> bool:
            return False

    class _FontSpy:
        def __init__(self, base, seen: list[str]) -> None:
            self._base = base
            self._seen = seen

        def render(self, text, antialias, color, background=None):
            self._seen.append(str(text))
            if background is None:
                return self._base.render(text, antialias, color)
            return self._base.render(text, antialias, color, background)

        def __getattr__(self, name: str):
            return getattr(self._base, name)

    _app, screen = _build_live_screen(_NoExitEngine(payload))
    seen: list[str] = []
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        for attr in ("_small_font", "_tiny_font", "_large_font"):
            if hasattr(screen, attr):
                setattr(screen, attr, _FontSpy(getattr(screen, attr), seen))
        screen.render(surface)

        assert not any("cannot exit" in text.lower() for text in seen)
    finally:
        pygame.quit()


def test_live_aircraft_cards_keep_aircraft_inside_small_wide_card_bounds() -> None:
    _app, screen = _build_screen(_build_payload())
    try:
        screen._instrument_card_bank._generation_failed = True
        screen._instrument_card_bank._allow_generation = False
        canvas = pygame.Surface((960, 300), pygame.SRCALPHA)
        rects = (
            pygame.Rect(20, 20, 420, 110),
            pygame.Rect(460, 20, 420, 110),
            pygame.Rect(20, 160, 260, 86),
            pygame.Rect(320, 160, 260, 86),
            pygame.Rect(620, 160, 260, 86),
        )
        for rect, preset in zip(rects, InstrumentAircraftViewPreset, strict=False):
            screen._draw_aircraft_orientation_card(canvas, rect, _base_state(), view_preset=preset)
            bounds = _red_bounds_in_rect(canvas, rect)
            assert bounds is not None, preset
            min_x, min_y, max_x, max_y = bounds
            inset = 6
            assert min_x >= rect.x + inset, preset
            assert min_y >= rect.y + inset, preset
            assert max_x <= rect.right - inset, preset
            assert max_y <= rect.bottom - inset, preset
    finally:
        pygame.quit()


def test_live_instrument_screen_uses_software_or_cached_cards_without_gl_generation(
    monkeypatch,
) -> None:
    _app, screen = _build_live_screen(_FakeInstrumentEngine(_build_payload()))
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        assert screen._instrument_card_bank._allow_generation is False

        def fail_renderer():
            raise AssertionError("standalone GL renderer should not be used during live IC rendering")

        monkeypatch.setattr(screen._instrument_card_bank, "_get_renderer", fail_renderer)

        screen.render(surface)

        assert screen._instrument_part1_layout is not None
    finally:
        pygame.quit()


def test_live_aircraft_cards_keep_reference_states_centered_and_semantic() -> None:
    _app, screen = _build_screen(_build_payload())
    try:
        screen._instrument_card_bank._generation_failed = True
        screen._instrument_card_bank._allow_generation = False
        canvas = pygame.Surface((980, 360), pygame.SRCALPHA)
        cases = (
            (
                pygame.Rect(20, 20, 420, 110),
                InstrumentState(
                    speed_kts=220,
                    altitude_ft=5000,
                    vertical_rate_fpm=0,
                    bank_deg=0,
                    pitch_deg=0,
                    heading_deg=90,
                    slip=0,
                ),
                InstrumentAircraftViewPreset.FRONT_LEFT,
            ),
            (
                pygame.Rect(460, 20, 420, 110),
                InstrumentState(
                    speed_kts=220,
                    altitude_ft=5000,
                    vertical_rate_fpm=0,
                    bank_deg=14,
                    pitch_deg=0,
                    heading_deg=90,
                    slip=0,
                ),
                InstrumentAircraftViewPreset.FRONT_LEFT,
            ),
            (
                pygame.Rect(20, 160, 420, 110),
                InstrumentState(
                    speed_kts=220,
                    altitude_ft=5000,
                    vertical_rate_fpm=0,
                    bank_deg=0,
                    pitch_deg=0,
                    heading_deg=0,
                    slip=0,
                ),
                InstrumentAircraftViewPreset.TOP_OBLIQUE,
            ),
        )
        for rect, state, preset in cases:
            screen._draw_aircraft_orientation_card(canvas, rect, state, view_preset=preset)
            bounds = _red_bounds_in_rect(canvas, rect)
            assert bounds is not None, preset
            min_x, _min_y, max_x, _max_y = bounds
            center_x = (min_x + max_x) / 2.0
            assert abs(center_x - rect.centerx) <= 18.0, (preset, center_x, rect.centerx)
            assert aircraft_card_semantic_drift_tags(state, view_preset=preset) == ()
    finally:
        pygame.quit()


def test_instrument_drill_title_still_routes_to_real_ic_renderer(monkeypatch) -> None:
    clock = _FakeClock()
    engine = build_ic_description_run_drill(
        clock=clock,
        seed=303,
        difficulty=0.5,
        mode=AntDrillMode.BUILD,
    )
    engine.start_scored()
    _app, screen = _build_live_screen(engine)
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        called = {"value": False}
        original = screen._render_instrument_comprehension_screen

        def wrapped(surface, snap, payload):
            called["value"] = True
            return original(surface, snap, payload)

        monkeypatch.setattr(screen, "_render_instrument_comprehension_screen", wrapped)
        screen.render(surface)

        assert called["value"] is True
    finally:
        pygame.quit()


def test_instrument_drill_instruction_uses_standard_intro_overlay(monkeypatch) -> None:
    clock = _FakeClock()
    engine = build_ic_description_run_drill(
        clock=clock,
        seed=304,
        difficulty=0.5,
        mode=AntDrillMode.BUILD,
    )
    _app, screen = _build_live_screen(engine, test_code="ic_description_run")
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        calls = {"intro_sections": 0, "custom_intro": 0}

        def intro_stub(*args, **kwargs):
            _ = (args, kwargs)
            calls["intro_sections"] += 1

        def custom_intro_stub(*args, **kwargs):
            _ = (args, kwargs)
            calls["custom_intro"] += 1

        monkeypatch.setattr(screen, "_draw_intro_section", intro_stub)
        monkeypatch.setattr(screen, "_render_instrument_instruction_page", custom_intro_stub)
        screen.render(surface)

        assert calls["custom_intro"] == 0
        assert calls["intro_sections"] == 5
        assert screen._instrument_part1_layout is None
    finally:
        pygame.quit()


def test_instrument_workout_block_uses_standard_ic_instruction_overlay(monkeypatch) -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        app = App(surface=surface, font=font)
        root = MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True)
        app.push(root)

        clock = _FakeClock()
        plan = AntWorkoutPlan(
            code="instrument_comprehension_workout",
            title="IC Workout UI",
            description="UI regression workout.",
            notes=("Untimed block setup.",),
            blocks=(
                AntWorkoutBlockPlan(
                    block_id="heading",
                    label="Heading Anchor",
                    description="Warm-up.",
                    focus_skills=("Heading anchoring",),
                    drill_code="ic_heading_anchor",
                    mode=AntDrillMode.BUILD,
                    duration_min=0.25,
                ),
            ),
        )
        session = AntWorkoutSession(
            clock=clock,
            seed=414,
            plan=plan,
            starting_level=5,
        )
        session.activate()
        session.activate()
        session.activate()
        session.activate()

        screen = AntWorkoutScreen(app, session=session, test_code="instrument_comprehension_workout")
        app.push(screen)
        screen.render(surface)

        runtime = screen._runtime_screen
        assert runtime is not None
        assert isinstance(runtime, CognitiveTestScreen)
        assert runtime._resolved_intro_test_code() == "ic_heading_anchor"
        assert runtime._engine.snapshot().phase is Phase.SCORED
        assert runtime._instrument_part1_layout is not None

        calls = {"intro_sections": 0, "custom_intro": 0}

        def intro_stub(*args, **kwargs):
            _ = (args, kwargs)
            calls["intro_sections"] += 1

        def custom_intro_stub(*args, **kwargs):
            _ = (args, kwargs)
            calls["custom_intro"] += 1

        monkeypatch.setattr(runtime, "_draw_intro_section", intro_stub)
        monkeypatch.setattr(runtime, "_render_instrument_instruction_page", custom_intro_stub)
        setattr(runtime._engine, "_phase", Phase.INSTRUCTIONS)
        runtime.render(surface)

        assert calls["custom_intro"] == 0
        assert calls["intro_sections"] == 5
        assert runtime._instrument_part1_layout is None
    finally:
        pygame.quit()
