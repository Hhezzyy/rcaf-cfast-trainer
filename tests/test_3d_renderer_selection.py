from __future__ import annotations

from dataclasses import dataclass

import pygame
import pytest

from cfast_trainer.app import App, CognitiveTestScreen, MenuItem, MenuScreen
from cfast_trainer.auditory_capacity import build_auditory_capacity_test
from cfast_trainer.gl_scenes import (
    AuditoryGlScene,
    RapidTrackingGlScene,
    TraceTest1GlScene,
    TraceTest2GlScene,
)
from cfast_trainer.rapid_tracking import build_rapid_tracking_test
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
    gl_scene_type: type[object]


_SCENES = (
    _SceneSpec(
        name="auditory",
        test_code="auditory_capacity",
        getter_name="_get_auditory_panda_renderer",
        gl_scene_type=AuditoryGlScene,
    ),
    _SceneSpec(
        name="rapid_tracking",
        test_code="rapid_tracking",
        getter_name="_get_rapid_tracking_panda_renderer",
        gl_scene_type=RapidTrackingGlScene,
    ),
    _SceneSpec(
        name="trace_test_1",
        test_code="trace_test_1",
        getter_name="_get_trace_test_1_panda_renderer",
        gl_scene_type=TraceTest1GlScene,
    ),
    _SceneSpec(
        name="trace_test_2",
        test_code="trace_test_2",
        getter_name="_get_trace_test_2_panda_renderer",
        gl_scene_type=TraceTest2GlScene,
    ),
)


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


@pytest.mark.parametrize("spec", _SCENES, ids=lambda spec: spec.name)
@pytest.mark.parametrize("opengl_enabled", [False, True], ids=["gl-off", "gl-on"])
def test_complex_scene_prefers_panda_renderer_when_available(
    pygame_headless,
    monkeypatch,
    spec: _SceneSpec,
    opengl_enabled: bool,
) -> None:
    app, screen = _build_screen(spec=spec, opengl_enabled=opengl_enabled)
    fake_renderer = _FakePandaRenderer(size=(320, 200))

    def fake_getter(*, size: tuple[int, int]):
        fake_renderer.size = tuple(size)
        return fake_renderer

    monkeypatch.setattr(screen, spec.getter_name, fake_getter)

    app.render()

    assert fake_renderer.render_calls == 1
    assert app.consume_gl_scene() is None


@pytest.mark.parametrize("spec", _SCENES, ids=lambda spec: spec.name)
def test_complex_scene_falls_back_to_gl_when_panda_unavailable_and_gl_enabled(
    pygame_headless,
    monkeypatch,
    spec: _SceneSpec,
) -> None:
    app, screen = _build_screen(spec=spec, opengl_enabled=True)
    monkeypatch.setattr(screen, spec.getter_name, lambda **_: None)

    app.render()

    assert isinstance(app.consume_gl_scene(), spec.gl_scene_type)
