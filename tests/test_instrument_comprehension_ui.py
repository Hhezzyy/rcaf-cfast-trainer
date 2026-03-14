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
from cfast_trainer.app import App, AntWorkoutScreen, CognitiveTestScreen, MenuItem, MenuScreen
from cfast_trainer.cognitive_core import Phase
from cfast_trainer.cognitive_core import TestSnapshot as SnapshotModel
from cfast_trainer.ic_drills import build_ic_description_run_drill
from cfast_trainer.instrument_comprehension import (
    InstrumentAircraftViewPreset,
    InstrumentComprehensionPayload,
    InstrumentComprehensionTrialKind,
    InstrumentOptionRenderMode,
    InstrumentOption,
    InstrumentState,
    instrument_aircraft_view_preset_for_code,
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


def _build_screen(payload: InstrumentComprehensionPayload) -> tuple[App, CognitiveTestScreen]:
    return _build_live_screen(_FakeInstrumentEngine(payload))


def _build_live_screen(engine: object) -> tuple[App, CognitiveTestScreen]:
    pygame.init()
    surface = pygame.display.set_mode((960, 540))
    font = pygame.font.Font(None, 36)
    app = App(surface=surface, font=font)
    root = MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True)
    app.push(root)
    screen = CognitiveTestScreen(app, engine_factory=lambda: engine)
    app.push(screen)
    return app, screen


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


def test_part1_layout_matches_guide_style_grid_and_hides_dial_captions() -> None:
    _app, screen = _build_screen(_build_payload())
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        screen.render(surface)

        layout = screen._instrument_part1_layout
        assert layout is not None
        assert layout.show_prompt_dial_labels is False
        assert len(layout.card_rects) == 5
        assert layout.card_rects[0].y == layout.card_rects[1].y
        assert layout.card_rects[2].y == layout.card_rects[3].y == layout.card_rects[4].y
        assert layout.card_rects[2].y > layout.card_rects[0].bottom
        assert layout.card_rects[0].width > layout.card_rects[2].width
        assert [screen._choice_key_label(code) for code in range(1, 6)] == ["A", "S", "D", "F", "G"]
    finally:
        pygame.quit()


def test_heading_dial_keeps_red_aircraft_fixed_while_rose_rotates() -> None:
    _app, screen = _build_screen(_build_payload())
    try:
        surface_north = pygame.Surface((140, 140), pygame.SRCALPHA)
        surface_east = pygame.Surface((140, 140), pygame.SRCALPHA)
        rect = pygame.Rect(10, 10, 120, 120)

        screen._draw_heading_dial(surface_north, rect, 0)
        screen._draw_heading_dial(surface_east, rect, 90)

        assert _red_bounds(surface_north) == _red_bounds(surface_east)
        assert pygame.image.tobytes(surface_north, "RGBA") != pygame.image.tobytes(
            surface_east, "RGBA"
        )
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


def test_instrument_workout_block_uses_real_ic_runtime_screen() -> None:
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
            notes=("Untimed reflections.",),
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
        session.append_text("focus")
        session.activate()
        session.append_text("reset")
        session.activate()
        session.activate()

        screen = AntWorkoutScreen(app, session=session, test_code="instrument_comprehension_workout")
        app.push(screen)
        screen.render(surface)

        runtime = screen._runtime_screen
        assert runtime is not None
        assert isinstance(runtime, CognitiveTestScreen)
        assert runtime._instrument_part1_layout is not None
    finally:
        pygame.quit()
