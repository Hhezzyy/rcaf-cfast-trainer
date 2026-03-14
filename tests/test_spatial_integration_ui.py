from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from cfast_trainer.app import App, CognitiveTestScreen, MenuItem, MenuScreen
from cfast_trainer.cognitive_core import Phase
from cfast_trainer.cognitive_core import TestSnapshot as SnapshotModel
from cfast_trainer.spatial_integration import (
    SpatialIntegrationAnswerMode,
    SpatialIntegrationConfig,
    SpatialIntegrationPart,
    SpatialIntegrationPayload,
    SpatialIntegrationTrialStage,
    build_spatial_integration_test,
)


@dataclass
class _FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class _FakeSpatialEngine:
    def __init__(self, payload: SpatialIntegrationPayload, *, title: str = "Spatial Integration") -> None:
        self._payload = payload
        self._title = title
        self.submissions: list[str] = []

    def snapshot(self) -> SnapshotModel:
        return SnapshotModel(
            title=self._title,
            phase=Phase.PRACTICE,
            prompt=self._payload.stem,
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


def _payload_for(*, part: SpatialIntegrationPart, study: bool, question_index: int) -> SpatialIntegrationPayload:
    clock = _FakeClock()
    engine = build_spatial_integration_test(
        clock=clock,
        seed=77 if part is SpatialIntegrationPart.STATIC else 99,
        difficulty=0.55,
        config=SpatialIntegrationConfig(start_part="AIRCRAFT" if part is SpatialIntegrationPart.AIRCRAFT else "STATIC"),
    )
    engine.start_practice()
    for _ in range(100):
        snap = engine.snapshot()
        payload = snap.payload
        if isinstance(payload, SpatialIntegrationPayload):
            if study and payload.trial_stage is SpatialIntegrationTrialStage.STUDY and payload.part is part:
                return payload
            if (not study) and payload.trial_stage is SpatialIntegrationTrialStage.QUESTION and payload.part is part:
                if payload.question_index_in_scene == question_index:
                    return payload
                answer = payload.correct_answer_token if payload.answer_mode is SpatialIntegrationAnswerMode.GRID_CLICK else str(payload.correct_code)
                engine.submit_answer(answer)
                continue
        clock.advance(0.2)
        engine.update()
    raise AssertionError("Could not build requested payload")


def test_study_screen_renders_three_reference_panes_without_answer_hitboxes() -> None:
    payload = _payload_for(part=SpatialIntegrationPart.STATIC, study=True, question_index=1)
    _app, screen = _build_screen(_FakeSpatialEngine(payload))
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        screen.render(surface)

        assert screen._spatial_grid_hitboxes == {}
        assert screen._spatial_option_hitboxes == {}
        assert pygame.transform.average_color(surface)[:3] != (0, 0, 0)
    finally:
        pygame.quit()


def test_grid_question_click_submits_cell_token() -> None:
    payload = _payload_for(part=SpatialIntegrationPart.STATIC, study=False, question_index=1)
    engine = _FakeSpatialEngine(payload)
    _app, screen = _build_screen(engine)
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        screen.render(surface)

        hitbox = screen._spatial_grid_hitboxes[payload.correct_answer_token]
        screen.handle_event(
            pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"button": 1, "pos": hitbox.center})
        )

        assert engine.submissions == [payload.correct_answer_token]
    finally:
        pygame.quit()


def test_option_question_uses_numeric_keys_then_enter() -> None:
    payload = _payload_for(part=SpatialIntegrationPart.STATIC, study=False, question_index=3)
    engine = _FakeSpatialEngine(payload)
    _app, screen = _build_screen(engine)
    try:
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_2, "mod": 0, "unicode": "2"})
        )

        assert engine.submissions == []
        assert screen._input == "2"

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": 0, "unicode": "\r"})
        )

        assert engine.submissions == ["2"]
    finally:
        pygame.quit()


def test_spatial_renderer_is_used_for_titles_with_prefix() -> None:
    payload = _payload_for(part=SpatialIntegrationPart.AIRCRAFT, study=False, question_index=1)
    engine = _FakeSpatialEngine(payload, title="Spatial Integration: Aircraft Drill")
    _app, screen = _build_screen(engine)
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        called = {"value": False}
        original = screen._render_spatial_integration_screen

        def wrapped(surface, snap, payload):
            called["value"] = True
            return original(surface, snap, payload)

        screen._render_spatial_integration_screen = wrapped  # type: ignore[method-assign]
        screen.render(surface)

        assert called["value"] is True
    finally:
        pygame.quit()
