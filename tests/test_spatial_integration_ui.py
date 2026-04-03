from __future__ import annotations

import os
import sys
from dataclasses import replace
from dataclasses import dataclass
from importlib.machinery import ModuleSpec
from types import ModuleType

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

if "moderngl" not in sys.modules:
    moderngl_stub = ModuleType("moderngl")
    moderngl_stub.__spec__ = ModuleSpec("moderngl", loader=None)
    sys.modules["moderngl"] = moderngl_stub

from cfast_trainer.app import App, CognitiveTestScreen, MenuItem, MenuScreen
from cfast_trainer.ant_drills import AntDrillMode
from cfast_trainer.cognitive_core import Phase
from cfast_trainer.cognitive_core import TestSnapshot as SnapshotModel
from cfast_trainer.si_drills import SiDrillConfig, build_si_landmark_anchor_drill
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


def _study_scene_content_rect(*, size: tuple[int, int]) -> pygame.Rect:
    w, h = size
    margin = max(10, min(18, w // 42))
    frame = pygame.Rect(margin, margin, w - margin * 2, h - margin * 2)
    header_h = max(30, min(38, h // 15))
    header = pygame.Rect(frame.x + 2, frame.y + 2, frame.w - 4, header_h)
    work = pygame.Rect(
        frame.x + 8,
        header.bottom + 8,
        frame.w - 16,
        frame.bottom - header.bottom - 16,
    )
    panel = work.inflate(-12, -12)
    banner = pygame.Rect(panel.x, panel.y, panel.w, 56)
    body = pygame.Rect(panel.x, banner.bottom + 10, panel.w, panel.h - banner.h - 10)
    left = pygame.Rect(body.x, body.y, int(body.w * 0.72), body.h)
    crop = left.inflate(-24, -24)
    crop.y += 36
    crop.h -= 44
    return crop


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


def test_study_scene_is_static_across_renders(monkeypatch) -> None:
    payload = _payload_for(part=SpatialIntegrationPart.STATIC, study=True, question_index=1)
    _app, screen = _build_screen(_FakeSpatialEngine(payload))
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        crop = _study_scene_content_rect(size=surface.get_size())

        monkeypatch.setattr(pygame.time, "get_ticks", lambda: 1000)
        screen.render(surface)
        first = pygame.image.tobytes(surface.subsurface(crop).copy(), "RGBA")

        monkeypatch.setattr(pygame.time, "get_ticks", lambda: 6000)
        screen.render(surface)
        second = pygame.image.tobytes(surface.subsurface(crop).copy(), "RGBA")

        assert second == first
    finally:
        pygame.quit()


def test_study_scene_uses_same_scene_with_different_reference_view_heading() -> None:
    payload = _payload_for(part=SpatialIntegrationPart.STATIC, study=True, question_index=1)
    assert len(payload.reference_views) >= 2
    second_view_payload = replace(
        payload,
        study_view_index=2,
        active_reference_view=payload.reference_views[1],
    )

    _app, screen = _build_screen(_FakeSpatialEngine(second_view_payload))
    captured: dict[str, int] = {}
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        original_scene = screen._draw_spatial_terrain_scene
        original_compass = screen._draw_spatial_compass

        def wrapped_scene(surface, rect, *, payload, heading_deg_override=None, scene_view_override=None, title_override=None, allow_external_3d=True):
            captured["heading"] = 0 if heading_deg_override is None else int(heading_deg_override)
            return original_scene(
                surface,
                rect,
                payload=payload,
                heading_deg_override=heading_deg_override,
                scene_view_override=scene_view_override,
                title_override=title_override,
                allow_external_3d=allow_external_3d,
            )

        def wrapped_compass(surface, rect, *, north_deg):
            captured["north"] = int(north_deg)
            return original_compass(surface, rect, north_deg=north_deg)

        screen._draw_spatial_terrain_scene = wrapped_scene  # type: ignore[method-assign]
        screen._draw_spatial_compass = wrapped_compass  # type: ignore[method-assign]
        screen.render(surface)

        assert captured["heading"] == int(payload.reference_views[1].heading_deg)
        assert captured["north"] == (-int(payload.reference_views[1].heading_deg)) % 360
    finally:
        pygame.quit()


def test_study_scene_falls_back_to_builtin_renderer_when_external_3d_is_unavailable(
    monkeypatch,
) -> None:
    payload = _payload_for(part=SpatialIntegrationPart.STATIC, study=True, question_index=1)
    _app, screen = _build_screen(_FakeSpatialEngine(payload))
    asset_calls = {"count": 0}
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        crop = _study_scene_content_rect(size=surface.get_size())

        monkeypatch.setattr(
            screen,
            "_get_spatial_integration_panda_renderer",
            lambda *, size: None,
        )

        def fail_blocked_world(**_: object) -> None:
            raise AssertionError("Spatial Integration should use the built-in scene fallback.")

        monkeypatch.setattr(screen, "_render_scene_panda_blocked_world", fail_blocked_world)
        original_asset = screen._draw_spatial_scene_asset

        def wrapped_asset(*args, **kwargs):
            asset_calls["count"] += 1
            return original_asset(*args, **kwargs)

        screen._draw_spatial_scene_asset = wrapped_asset  # type: ignore[method-assign]
        screen.render(surface)

        assert asset_calls["count"] > 0
        assert pygame.transform.average_color(surface.subsurface(crop))[:3] != (0, 0, 0)
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


def test_option_question_uses_numeric_keys_immediately() -> None:
    payload = _payload_for(part=SpatialIntegrationPart.STATIC, study=False, question_index=3)
    engine = _FakeSpatialEngine(payload)
    _app, screen = _build_screen(engine)
    try:
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_2, "mod": 0, "unicode": "2"})
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


def test_spatial_renderer_is_used_for_real_si_drill_titles() -> None:
    clock = _FakeClock()
    engine = build_si_landmark_anchor_drill(
        clock=clock,
        seed=303,
        difficulty=0.5,
        mode=AntDrillMode.TEMPO,
        config=SiDrillConfig(practice_scenes_per_part=0, scored_duration_s=24.0),
    )
    engine.start_practice()
    for _ in range(240):
        snap = engine.snapshot()
        payload = snap.payload
        if isinstance(payload, SpatialIntegrationPayload) and payload.trial_stage is SpatialIntegrationTrialStage.QUESTION:
            break
        clock.advance(0.2)
        engine.update()
    else:
        raise AssertionError("Could not reach a Spatial Integration drill question stage.")

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
        assert engine.snapshot().title.startswith("Spatial Integration:")
    finally:
        pygame.quit()
