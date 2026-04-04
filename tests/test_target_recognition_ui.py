from __future__ import annotations

import os
import re
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from dataclasses import dataclass, replace
from importlib.machinery import ModuleSpec
from types import ModuleType

import pygame

if "moderngl" not in sys.modules:
    moderngl_stub = ModuleType("moderngl")
    moderngl_stub.__spec__ = ModuleSpec("moderngl", loader=None)
    sys.modules["moderngl"] = moderngl_stub

from cfast_trainer.ant_drills import AntDrillMode
from cfast_trainer.ant_workouts import (
    AntWorkoutBlockPlan,
    AntWorkoutPlan,
    AntWorkoutSession,
)
from cfast_trainer.app import App, AntWorkoutScreen, CognitiveTestScreen, MenuItem, MenuScreen
from cfast_trainer.cognitive_core import Phase
from cfast_trainer.cognitive_core import TestSnapshot as SnapshotModel
from cfast_trainer.target_recognition import (
    TargetRecognitionPayload,
    TargetRecognitionSceneEntity,
    TargetRecognitionSystemCycle,
)
from cfast_trainer.tr_drills import build_tr_mixed_tempo_drill


class _FakeTREngine:
    def __init__(self, payload: TargetRecognitionPayload, *, title: str) -> None:
        self._payload = payload
        self._title = title
        self.submit_calls: list[str] = []

    def snapshot(self) -> SnapshotModel:
        return SnapshotModel(
            title=self._title,
            phase=Phase.PRACTICE,
            prompt="Register matches only in the active panels.",
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
        self.submit_calls.append(str(raw))
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


@dataclass
class _FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _build_payload(*, active_panels: tuple[str, ...]) -> TargetRecognitionPayload:
    return TargetRecognitionPayload(
        scene_rows=2,
        scene_cols=2,
        scene_cells=("TRK:F", "BLD:N", "TRK:H", "TNK:F"),
        scene_entities=(
            TargetRecognitionSceneEntity("truck", "friendly", False, False),
            TargetRecognitionSceneEntity("building", "neutral", False, False),
            TargetRecognitionSceneEntity("truck", "hostile", False, False),
            TargetRecognitionSceneEntity("tank", "friendly", False, False),
        ),
        scene_target="Friendly Truck",
        scene_has_target=True,
        scene_target_options=("Friendly Truck", "Hostile Truck", "Neutral Building", "Friendly Tank"),
        light_pattern=("G", "B", "R"),
        light_target_pattern=("G", "B", "R"),
        light_has_target=True,
        scan_tokens=("<>", "[]", "/\\", "()"),
        scan_target="<>",
        scan_has_target=True,
        system_rows=("A1B2", "C3D4", "E5F6"),
        system_target="A1B2",
        system_has_target=True,
        system_cycles=(
            TargetRecognitionSystemCycle(
                target="A1B2",
                columns=(
                    ("C3D4", "E5F6", "G7H8"),
                    ("A1B2", "J9K1", "L2M3"),
                    ("N4P5", "Q6R7", "S8T9"),
                ),
            ),
        ),
        system_step_interval_s=1.4,
        full_credit_error=0,
        zero_credit_error=3,
        active_panels=active_panels,
        light_interval_range_s=(4.5, 6.5),
        scan_interval_range_s=(4.2, 6.2),
        scan_repeat_range=(2, 3),
    )


def _build_screen(engine: object) -> tuple[App, CognitiveTestScreen]:
    pygame.init()
    surface = pygame.display.set_mode((960, 540))
    font = pygame.font.Font(None, 36)
    app = App(surface=surface, font=font)
    root = MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True)
    app.push(root)
    screen = CognitiveTestScreen(app, engine_factory=lambda: engine)
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


def test_target_recognition_drill_title_still_routes_to_real_renderer(monkeypatch) -> None:
    clock = _FakeClock()
    engine = build_tr_mixed_tempo_drill(
        clock=clock,
        seed=505,
        difficulty=0.5,
        mode=AntDrillMode.BUILD,
    )
    engine.start_scored()
    _app, screen = _build_screen(engine)
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        called = {"value": False}
        original = screen._render_target_recognition_screen

        def wrapped(surface, snap, payload):
            called["value"] = True
            return original(surface, snap, payload)

        monkeypatch.setattr(screen, "_render_target_recognition_screen", wrapped)
        screen.render(surface)

        assert called["value"] is True
    finally:
        pygame.quit()


def test_target_recognition_focused_drill_only_exposes_active_hitboxes_and_dims_off_panels() -> None:
    active_payload = _build_payload(active_panels=("scene", "light", "scan", "system"))
    focused_payload = _build_payload(active_panels=("scene",))

    _app, screen = _build_screen(_FakeTREngine(focused_payload, title="Target Recognition: Scene Anchor"))
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        screen.render(surface)

        assert set(screen._tr_selector_hitboxes) == {"scene"}
        assert screen._tr_light_button_hitbox is None
        assert screen._tr_scan_button_hitbox is None
        assert not screen._tr_system_string_hitboxes

        focused_light_pixel = surface.get_at((360, 102))
        focused_scan_pixel = surface.get_at((585, 102))

        screen._engine = _FakeTREngine(active_payload, title="Target Recognition")
        screen.render(surface)
        active_light_pixel = surface.get_at((360, 102))
        active_scan_pixel = surface.get_at((585, 102))

        assert focused_light_pixel != active_light_pixel
        assert focused_scan_pixel != active_scan_pixel
    finally:
        pygame.quit()


def test_target_recognition_workout_block_uses_real_runtime_screen() -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        app = App(surface=surface, font=font)
        root = MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True)
        app.push(root)

        clock = _FakeClock()
        plan = AntWorkoutPlan(
            code="target_recognition_workout",
            title="TR Workout UI",
            description="UI regression workout.",
            notes=("Untimed reflections.",),
            blocks=(
                AntWorkoutBlockPlan(
                    block_id="scene",
                    label="Scene Anchor",
                    description="Warm-up.",
                    focus_skills=("Map discrimination",),
                    drill_code="tr_scene_anchor",
                    mode=AntDrillMode.BUILD,
                    duration_min=0.25,
                ),
            ),
        )
        session = AntWorkoutSession(
            clock=clock,
            seed=606,
            plan=plan,
            starting_level=5,
        )
        session.activate()
        session.append_text("focus")
        session.activate()
        session.append_text("reset")
        session.activate()
        session.activate()

        screen = AntWorkoutScreen(app, session=session, test_code="target_recognition_workout")
        app.push(screen)
        screen.render(surface)

        runtime = screen._runtime_screen
        assert runtime is not None
        assert runtime._tr_selector_hitboxes
        assert set(runtime._tr_selector_hitboxes) == {"scene"}
    finally:
        pygame.quit()


def test_target_recognition_workout_overlay_hides_timer_text() -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        app = App(surface=surface, font=font)
        root = MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True)
        app.push(root)

        clock = _FakeClock()
        plan = AntWorkoutPlan(
            code="target_recognition_workout",
            title="TR Workout UI",
            description="UI regression workout.",
            notes=("Untimed reflections.",),
            blocks=(
                AntWorkoutBlockPlan(
                    block_id="scene",
                    label="Scene Anchor",
                    description="Warm-up.",
                    focus_skills=("Map discrimination",),
                    drill_code="tr_scene_anchor",
                    mode=AntDrillMode.BUILD,
                    duration_min=0.25,
                ),
            ),
        )
        session = AntWorkoutSession(
            clock=clock,
            seed=606,
            plan=plan,
            starting_level=5,
        )
        session.activate()
        session.append_text("focus")
        session.activate()
        session.append_text("reset")
        session.activate()
        session.activate()

        screen = AntWorkoutScreen(app, session=session, test_code="target_recognition_workout")
        captured = _install_recording_fonts(screen)

        screen._render_block_status_overlay(surface, session.snapshot())

        assert not any("Workout time" in text for text in captured)
        assert not any(re.search(r"\b\d{2}:\d{2}\b", text) for text in captured)
    finally:
        pygame.quit()


def test_target_recognition_symbols_use_plain_truck_and_tank_shapes() -> None:
    _app, screen = _build_screen(_FakeTREngine(_build_payload(active_panels=("scene",)), title="Target Recognition"))
    try:
        truck_surface = pygame.Surface((48, 48), pygame.SRCALPHA)
        truck = TargetRecognitionSceneEntity("truck", "friendly", False, False)
        screen._draw_target_recognition_symbol(
            truck_surface,
            entity=truck,
            cx=20,
            cy=20,
            size=8,
            color=(255, 255, 255, 255),
            heading=0.0,
        )
        assert truck_surface.get_at((35, 20)).a == 0

        tank_surface = pygame.Surface((48, 48), pygame.SRCALPHA)
        tank = TargetRecognitionSceneEntity("tank", "friendly", False, False)
        screen._draw_target_recognition_symbol(
            tank_surface,
            entity=tank,
            cx=20,
            cy=20,
            size=8,
            color=(255, 255, 255, 255),
            heading=0.0,
        )
        assert tank_surface.get_at((20, 20)).a == 0
    finally:
        pygame.quit()


def test_target_recognition_scene_clear_all_objective_waits_for_last_target() -> None:
    payload = replace(
        _build_payload(active_panels=("scene",)),
        scene_rows=2,
        scene_cols=2,
        scene_cells=("TRK:FP", "TNK:HP", "BLD:N", "TRK:F"),
        scene_entities=(
            TargetRecognitionSceneEntity("truck", "friendly", False, True),
            TargetRecognitionSceneEntity("tank", "hostile", False, True),
            TargetRecognitionSceneEntity("building", "neutral", False, False),
            TargetRecognitionSceneEntity("truck", "friendly", False, False),
        ),
        scene_target="All Priority Targets",
        scene_has_target=True,
        scene_target_options=("Friendly Truck (HP)", "Hostile Tank (HP)"),
        scene_objective_label="All Priority Targets",
        scene_clear_all_targets=True,
    )
    engine = _FakeTREngine(payload, title="Target Recognition: Priority Sweep")
    _app, screen = _build_screen(engine)
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        screen.render(surface)

        assert screen._tr_scene_active_targets == list(payload.scene_target_options)

        def center_for_label(label: str) -> tuple[int, int]:
            for hit_rect, glyph_id in screen._tr_scene_symbol_hitboxes:
                glyph = screen._tr_scene_glyphs[glyph_id]
                if label in glyph.matching_labels:
                    return hit_rect.center
            raise AssertionError(f"missing scene glyph for {label}")

        first_target = center_for_label("Friendly Truck (HP)")
        screen.handle_event(
            pygame.event.Event(
                pygame.MOUSEBUTTONDOWN,
                {"button": 1, "pos": first_target},
            )
        )

        assert engine.submit_calls == []
        assert screen._tr_scene_active_targets == ["Hostile Tank (HP)"]

        screen.render(surface)
        second_target = center_for_label("Hostile Tank (HP)")
        screen.handle_event(
            pygame.event.Event(
                pygame.MOUSEBUTTONDOWN,
                {"button": 1, "pos": second_target},
            )
        )

        assert engine.submit_calls == ["1"]
        assert screen._tr_scene_active_targets == []
    finally:
        pygame.quit()
