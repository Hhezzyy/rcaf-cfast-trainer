from __future__ import annotations

import os
from dataclasses import dataclass
from typing import cast

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from cfast_trainer.ant_drills import AntDrillMode
from cfast_trainer.ant_workouts import AntWorkoutBlockPlan, AntWorkoutPlan, AntWorkoutSession
from cfast_trainer.app import App, AntWorkoutScreen, CognitiveTestScreen, MenuItem, MenuScreen
from cfast_trainer.cognitive_core import Phase
from cfast_trainer.cognitive_core import TestSnapshot as SnapshotModel
from cfast_trainer.rapid_tracking import RapidTrackingPayload, build_rapid_tracking_test
from cfast_trainer.rt_drills import build_rt_mixed_tempo_drill


@dataclass
class _FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class _FakeRapidTrackingEngine:
    def __init__(self, payload: RapidTrackingPayload, *, title: str) -> None:
        self._payload = payload
        self._title = title
        self.submissions: list[str] = []

    def snapshot(self) -> SnapshotModel:
        return SnapshotModel(
            title=self._title,
            phase=Phase.PRACTICE,
            prompt="Track the target.",
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
        self.submissions.append(str(raw))
        return True

    def update(self) -> None:
        pass


def _sample_payload() -> RapidTrackingPayload:
    clock = _FakeClock()
    engine = build_rapid_tracking_test(clock=clock, seed=551, difficulty=0.5)
    engine.start_scored()
    payload = engine.snapshot().payload
    assert isinstance(payload, RapidTrackingPayload)
    return payload


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


def _build_live_rt_screen(*, clock: _FakeClock) -> CognitiveTestScreen:
    screen = _build_screen(
        build_rapid_tracking_test(
            clock=clock,
            seed=551,
            difficulty=0.5,
        )
    )[1]
    screen._engine.start_scored()
    screen._engine._target_x = 0.0
    screen._engine._target_y = 0.0
    screen._engine._target_terrain_occluded = False
    screen._engine._reset_camera_pose_to_target()
    return screen


def test_rapid_tracking_title_prefix_routes_to_real_renderer(monkeypatch) -> None:
    payload = _sample_payload()
    _app, screen = _build_screen(
        _FakeRapidTrackingEngine(payload, title="Rapid Tracking: Lock Anchor")
    )
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        called = {"value": False}
        original = screen._render_rapid_tracking_screen

        def wrapped(surface, snap, payload):
            called["value"] = True
            return original(surface, snap, payload)

        monkeypatch.setattr(screen, "_render_rapid_tracking_screen", wrapped)
        screen.render(surface)

        assert called["value"] is True
    finally:
        pygame.quit()


def test_rapid_tracking_workout_block_uses_real_runtime_screen() -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        app = App(surface=surface, font=font)
        root = MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True)
        app.push(root)

        clock = _FakeClock()
        plan = AntWorkoutPlan(
            code="rapid_tracking_workout",
            title="RT Workout UI",
            description="UI regression workout.",
            notes=("Untimed reflections.",),
            blocks=(
                AntWorkoutBlockPlan(
                    block_id="mixed",
                    label="Mixed Tempo",
                    description="Warm-up.",
                    focus_skills=("Tracking",),
                    drill_code="rt_mixed_tempo",
                    mode=AntDrillMode.BUILD,
                    duration_min=0.25,
                ),
            ),
        )
        session = AntWorkoutSession(clock=clock, seed=606, plan=plan, starting_level=5)
        session.activate()
        session.append_text("steady scan")
        session.activate()
        session.append_text("capture late")
        session.activate()
        session.activate()

        screen = AntWorkoutScreen(app, session=session, test_code="rapid_tracking_workout")
        app.push(screen)
        screen.render(surface)

        runtime = screen._runtime_screen
        assert runtime is not None
        assert runtime._engine is not None
        snap = runtime._engine.snapshot()
        assert str(snap.title).startswith("Rapid Tracking")
        assert isinstance(snap.payload, RapidTrackingPayload)
    finally:
        pygame.quit()


def test_rapid_tracking_real_drill_engine_exposes_focus_metadata_on_live_screen() -> None:
    clock = _FakeClock()
    engine = build_rt_mixed_tempo_drill(
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
        screen.render(surface)

        payload = engine.snapshot().payload
        assert isinstance(payload, RapidTrackingPayload)
        assert payload.focus_label == "Stable lock quality"
        assert payload.active_target_kinds == ("soldier", "truck")
    finally:
        pygame.quit()


def test_rapid_tracking_space_hold_starts_zoom_and_release_restores_view() -> None:
    clock = _FakeClock()
    screen = _build_live_rt_screen(clock=clock)
    try:
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_SPACE, "unicode": " "})
        )
        clock.advance(0.30)
        screen._engine.update()
        zoomed = screen._engine.snapshot().payload
        assert zoomed is not None
        assert zoomed.capture_zoom > 0.8

        screen.handle_event(
            pygame.event.Event(pygame.KEYUP, {"key": pygame.K_SPACE, "unicode": ""})
        )
        clock.advance(0.30)
        screen._engine.update()
        released = screen._engine.snapshot().payload
        assert released is not None
        assert released.capture_zoom < 0.2
    finally:
        pygame.quit()


def test_rapid_tracking_mouse_and_trigger_use_hold_tokens() -> None:
    payload = _sample_payload()
    _app, screen = _build_screen(
        _FakeRapidTrackingEngine(payload, title="Rapid Tracking: Lock Anchor")
    )
    try:
        screen.handle_event(
            pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"button": 1, "pos": (20, 20)})
        )
        screen.handle_event(
            pygame.event.Event(pygame.MOUSEBUTTONUP, {"button": 1, "pos": (20, 20)})
        )
        screen.handle_event(
            pygame.event.Event(pygame.JOYBUTTONDOWN, {"button": 0})
        )
        screen.handle_event(
            pygame.event.Event(pygame.JOYBUTTONUP, {"button": 0})
        )

        assert cast(_FakeRapidTrackingEngine, screen._engine).submissions == [
            "CAPTURE_HOLD_START",
            "CAPTURE_HOLD_END",
            "CAPTURE_HOLD_START",
            "CAPTURE_HOLD_END",
        ]
    finally:
        pygame.quit()
