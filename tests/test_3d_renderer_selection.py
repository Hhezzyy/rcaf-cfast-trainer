from __future__ import annotations

from dataclasses import dataclass

import pygame
import pytest

from cfast_trainer.app import (
    App,
    CognitiveTestScreen,
    MenuItem,
    MenuScreen,
    _AuditoryPandaRequirementState,
)
from cfast_trainer.auditory_capacity import build_auditory_capacity_test
from cfast_trainer.rapid_tracking import build_rapid_tracking_test
from cfast_trainer.spatial_integration import build_spatial_integration_test
from cfast_trainer.trace_test_1 import build_trace_test_1_test
from cfast_trainer.trace_test_2 import build_trace_test_2_test


class _FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class _FakePandaRenderer:
    def __init__(self, *, size: tuple[int, int]) -> None:
        self.size = tuple(size)
        self.render_calls = 0

    def render(self, **_: object) -> pygame.Surface:
        self.render_calls += 1
        surface = pygame.Surface(self.size)
        surface.fill((72, 108, 144))
        return surface

    def close(self) -> None:
        return None

    def target_overlay_state(self):
        return None


@dataclass(frozen=True, slots=True)
class _SceneSpec:
    name: str
    test_code: str
    getter_name: str


_SCENES = (
    _SceneSpec(
        name="auditory",
        test_code="auditory_capacity",
        getter_name="_get_auditory_panda_renderer",
    ),
    _SceneSpec(
        name="rapid_tracking",
        test_code="rapid_tracking",
        getter_name="_get_rapid_tracking_panda_renderer",
    ),
    _SceneSpec(
        name="spatial_integration",
        test_code="spatial_integration",
        getter_name="_get_spatial_integration_panda_renderer",
    ),
    _SceneSpec(
        name="trace_test_1",
        test_code="trace_test_1",
        getter_name="_get_trace_test_1_panda_renderer",
    ),
    _SceneSpec(
        name="trace_test_2",
        test_code="trace_test_2",
        getter_name="_get_trace_test_2_panda_renderer",
    ),
)

_GL_SCENE_TYPE_BY_NAME = {
    "auditory": "AuditoryGlScene",
    "rapid_tracking": "RapidTrackingGlScene",
    "spatial_integration": "SpatialIntegrationGlScene",
    "trace_test_1": "TraceTest1GlScene",
    "trace_test_2": "TraceTest2GlScene",
}

_PANDA_WHEN_GL_DISABLED = {
    "spatial_integration",
    "trace_test_1",
    "trace_test_2",
}


@pytest.fixture
def pygame_headless(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    pygame.quit()
    pygame.init()
    yield
    pygame.quit()


def _build_screen(*, spec: _SceneSpec, opengl_enabled: bool) -> tuple[App, CognitiveTestScreen]:
    display = pygame.display.set_mode((960, 540))
    font = pygame.font.Font(None, 36)
    app = App(surface=display, font=font, opengl_enabled=opengl_enabled)
    app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))

    if spec.name == "auditory":
        clock = _FakeClock()
        screen = CognitiveTestScreen(
            app,
            engine_factory=lambda: build_auditory_capacity_test(
                clock=clock,
                seed=17,
                difficulty=0.58,
            ),
            test_code=spec.test_code,
        )
        app.push(screen)
        screen._engine.start_practice()
        clock.advance(0.2)
        screen._engine.update()
        return app, screen

    if spec.name == "rapid_tracking":
        clock = _FakeClock()
        screen = CognitiveTestScreen(
            app,
            engine_factory=lambda: build_rapid_tracking_test(
                clock=clock,
                seed=551,
                difficulty=0.63,
            ),
            test_code=spec.test_code,
        )
        app.push(screen)
        screen._engine.start_scored()
        clock.advance(0.6)
        screen._engine.update()
        return app, screen

    if spec.name == "trace_test_1":
        clock = _FakeClock()
        screen = CognitiveTestScreen(
            app,
            engine_factory=lambda: build_trace_test_1_test(
                clock=clock,
                seed=37,
                difficulty=0.6,
            ),
            test_code=spec.test_code,
        )
        app.push(screen)
        screen._engine.start_practice()
        clock.advance(0.1)
        screen._engine.update()
        return app, screen

    if spec.name == "spatial_integration":
        clock = _FakeClock()
        screen = CognitiveTestScreen(
            app,
            engine_factory=lambda: build_spatial_integration_test(
                clock=clock,
                seed=61,
                difficulty=0.58,
            ),
            test_code=spec.test_code,
        )
        app.push(screen)
        screen._engine.start_practice()
        clock.advance(0.1)
        screen._engine.update()
        return app, screen

    if spec.name == "trace_test_2":
        clock = _FakeClock()
        screen = CognitiveTestScreen(
            app,
            engine_factory=lambda: build_trace_test_2_test(
                clock=clock,
                seed=71,
                difficulty=0.58,
            ),
            test_code=spec.test_code,
        )
        app.push(screen)
        screen._engine.start_practice()
        clock.advance(0.1)
        screen._engine.update()
        return app, screen

    raise ValueError(f"unknown scene: {spec.name}")


def _mark_auditory_panda_ready(screen: CognitiveTestScreen) -> None:
    screen._auditory_panda_requirement = _AuditoryPandaRequirementState(
        checked=True,
        ready=True,
    )
    screen._auditory_panda_failed = False


def _queued_gl_scene_type(app: App) -> str | None:
    scene = app.consume_gl_scene()
    return None if scene is None else type(scene).__name__


@pytest.mark.parametrize("spec", _SCENES, ids=lambda spec: spec.name)
def test_complex_scene_queues_gl_scene_when_opengl_enabled(
    pygame_headless,
    monkeypatch,
    spec: _SceneSpec,
) -> None:
    app, screen = _build_screen(spec=spec, opengl_enabled=True)
    fake_renderer = _FakePandaRenderer(size=(320, 200))
    if spec.name == "auditory":
        _mark_auditory_panda_ready(screen)
        monkeypatch.setattr(screen, "_sync_auditory_audio", lambda **_: None)

    def fake_getter(*, size: tuple[int, int]):
        fake_renderer.size = tuple(size)
        return fake_renderer

    monkeypatch.setattr(screen, spec.getter_name, fake_getter)

    app.render()

    assert fake_renderer.render_calls == 0
    assert _queued_gl_scene_type(app) == _GL_SCENE_TYPE_BY_NAME[spec.name]


@pytest.mark.parametrize("spec", _SCENES, ids=lambda spec: spec.name)
def test_complex_scene_uses_non_gl_fallbacks_when_opengl_disabled(
    pygame_headless,
    monkeypatch,
    spec: _SceneSpec,
) -> None:
    app, screen = _build_screen(spec=spec, opengl_enabled=False)
    fake_renderer = _FakePandaRenderer(size=(320, 200))
    if spec.name == "auditory":
        _mark_auditory_panda_ready(screen)
        monkeypatch.setattr(screen, "_sync_auditory_audio", lambda **_: None)

    def fake_getter(*, size: tuple[int, int]):
        fake_renderer.size = tuple(size)
        return fake_renderer

    monkeypatch.setattr(screen, spec.getter_name, fake_getter)

    app.render()

    assert _queued_gl_scene_type(app) is None
    if spec.name in _PANDA_WHEN_GL_DISABLED:
        assert fake_renderer.render_calls == 1
    else:
        assert fake_renderer.render_calls == 0


@pytest.mark.parametrize("spec", _SCENES, ids=lambda spec: spec.name)
def test_renderer_env_preference_does_not_override_opengl_standard_path(
    pygame_headless,
    monkeypatch,
    spec: _SceneSpec,
) -> None:
    env_var = f"CFAST_{spec.name.upper()}_RENDERER"
    app, screen = _build_screen(spec=spec, opengl_enabled=True)
    fake_renderer = _FakePandaRenderer(size=(320, 200))
    monkeypatch.setenv(env_var, "pygame")
    if spec.name == "auditory":
        _mark_auditory_panda_ready(screen)
        monkeypatch.setattr(screen, "_sync_auditory_audio", lambda **_: None)

    def fake_getter(*, size: tuple[int, int]):
        fake_renderer.size = tuple(size)
        return fake_renderer

    monkeypatch.setattr(screen, spec.getter_name, fake_getter)

    app.render()

    assert fake_renderer.render_calls == 0
    assert _queued_gl_scene_type(app) == _GL_SCENE_TYPE_BY_NAME[spec.name]
