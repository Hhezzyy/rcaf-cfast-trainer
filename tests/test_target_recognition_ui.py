from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from dataclasses import dataclass

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
