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
from cfast_trainer.cognitive_updating import (
    CognitiveUpdatingGenerator,
    CognitiveUpdatingPayload,
    CognitiveUpdatingTrainingProfile,
)
from cfast_trainer.cu_drills import build_cu_mixed_tempo_drill


@dataclass
class _FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class _FakeCognitiveUpdatingEngine:
    def __init__(self, payload: CognitiveUpdatingPayload, *, title: str) -> None:
        self._payload = payload
        self._title = title
        self.submissions: list[str] = []

    def snapshot(self) -> SnapshotModel:
        return SnapshotModel(
            title=self._title,
            phase=Phase.PRACTICE,
            prompt=self._payload.question,
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


def _sample_payload(*, active_domains: tuple[str, ...], focus_label: str) -> CognitiveUpdatingPayload:
    problem = CognitiveUpdatingGenerator(seed=202).next_problem_for_selection(
        difficulty=0.5,
        training_profile=CognitiveUpdatingTrainingProfile(
            active_domains=active_domains,
            focus_label=focus_label,
        ),
        scenario_family="compressed",
    )
    return cast(CognitiveUpdatingPayload, problem.payload)


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


def test_cognitive_updating_drill_title_routes_to_real_renderer(monkeypatch) -> None:
    payload = _sample_payload(active_domains=("controls", "state_code"), focus_label="Controls")
    _app, screen = _build_screen(
        _FakeCognitiveUpdatingEngine(payload, title="Cognitive Updating: Controls Anchor")
    )
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        called = {"value": False}
        original = screen._render_cognitive_updating_screen

        def wrapped(surface, snap, payload):
            called["value"] = True
            return original(surface, snap, payload)

        monkeypatch.setattr(screen, "_render_cognitive_updating_screen", wrapped)
        screen.render(surface)

        assert called["value"] is True
    finally:
        pygame.quit()


def test_cognitive_updating_focused_payload_dims_inactive_panels_visually() -> None:
    focused_payload = _sample_payload(active_domains=("controls", "state_code"), focus_label="Controls")
    full_payload = _sample_payload(
        active_domains=("controls", "navigation", "engine", "sensors", "objectives", "state_code"),
        focus_label="Full Mixed",
    )
    _app, screen = _build_screen(
        _FakeCognitiveUpdatingEngine(focused_payload, title="Cognitive Updating: Controls Anchor")
    )
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        screen.render(surface)
        focused_bytes = pygame.image.tobytes(surface, "RGBA")

        screen._engine = _FakeCognitiveUpdatingEngine(full_payload, title="Cognitive Updating: Pressure Run")
        screen.render(surface)
        full_bytes = pygame.image.tobytes(surface, "RGBA")

        assert focused_bytes != full_bytes
    finally:
        pygame.quit()


def test_cognitive_updating_workout_block_uses_real_runtime_screen() -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        app = App(surface=surface, font=font)
        root = MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True)
        app.push(root)

        clock = _FakeClock()
        plan = AntWorkoutPlan(
            code="cognitive_updating_workout",
            title="CU Workout UI",
            description="UI regression workout.",
            notes=("Untimed reflections.",),
            blocks=(
                AntWorkoutBlockPlan(
                    block_id="mixed",
                    label="Mixed Tempo",
                    description="Warm-up.",
                    focus_skills=("Domain switching",),
                    drill_code="cu_mixed_tempo",
                    mode=AntDrillMode.BUILD,
                    duration_min=0.25,
                ),
            ),
        )
        session = AntWorkoutSession(clock=clock, seed=606, plan=plan, starting_level=5)
        session.activate()
        session.append_text("scan")
        session.activate()
        session.append_text("code")
        session.activate()
        session.activate()

        screen = AntWorkoutScreen(app, session=session, test_code="cognitive_updating_workout")
        app.push(screen)
        screen.render(surface)

        runtime = screen._runtime_screen
        assert runtime is not None
        assert runtime._engine is not None
        snap = runtime._engine.snapshot()
        assert str(snap.title).startswith("Cognitive Updating")
        assert isinstance(snap.payload, CognitiveUpdatingPayload)
    finally:
        pygame.quit()


def test_cognitive_updating_real_drill_engine_uses_payload_metadata_on_live_screen() -> None:
    clock = _FakeClock()
    engine = build_cu_mixed_tempo_drill(
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

        payload = engine._current.payload
        assert isinstance(payload, CognitiveUpdatingPayload)
        assert payload.focus_label == "Controls"
        assert payload.active_domains == ("controls", "state_code")
    finally:
        pygame.quit()
