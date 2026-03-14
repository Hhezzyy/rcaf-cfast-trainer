from __future__ import annotations

import pygame
import pytest

from cfast_trainer.app import _initialize_display_surfaces


@pytest.fixture
def pygame_headless(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    pygame.quit()
    pygame.init()
    yield
    pygame.quit()


def test_initialize_display_surfaces_enables_gl_when_renderer_starts(
    pygame_headless,
    monkeypatch,
) -> None:
    calls: list[int] = []
    display_surface = pygame.Surface((320, 240))

    def fake_set_mode(size, flags):
        _ = size
        calls.append(int(flags))
        return display_surface

    class _FakeRenderer:
        def __init__(self, *, window_size):
            self.window_size = tuple(window_size)

    monkeypatch.setattr("pygame.display.set_mode", fake_set_mode)
    monkeypatch.setattr("cfast_trainer.app._OpenGLSceneRenderer", _FakeRenderer)

    display, app_surface, renderer, active_flags = _initialize_display_surfaces(
        window_size=(320, 240),
        window_flags=pygame.RESIZABLE,
        video_driver="metal",
        want_gl=True,
    )

    assert display is display_surface
    assert app_surface is not display_surface
    assert renderer is not None
    assert renderer.window_size == (320, 240)
    assert active_flags == (pygame.RESIZABLE | pygame.OPENGL | pygame.DOUBLEBUF)
    assert calls == [pygame.RESIZABLE | pygame.OPENGL | pygame.DOUBLEBUF]


def test_initialize_display_surfaces_falls_back_when_gl_renderer_init_fails(
    pygame_headless,
    monkeypatch,
) -> None:
    calls: list[int] = []
    display_surface = pygame.Surface((320, 240))

    def fake_set_mode(size, flags):
        _ = size
        calls.append(int(flags))
        return display_surface

    class _BoomRenderer:
        def __init__(self, *, window_size):
            _ = window_size
            raise RuntimeError("gl unavailable")

    monkeypatch.setattr("pygame.display.set_mode", fake_set_mode)
    monkeypatch.setattr("cfast_trainer.app._OpenGLSceneRenderer", _BoomRenderer)

    display, app_surface, renderer, active_flags = _initialize_display_surfaces(
        window_size=(320, 240),
        window_flags=pygame.RESIZABLE,
        video_driver="metal",
        want_gl=True,
    )

    assert display is display_surface
    assert app_surface is display_surface
    assert renderer is None
    assert active_flags == pygame.RESIZABLE
    assert calls == [
        pygame.RESIZABLE | pygame.OPENGL | pygame.DOUBLEBUF,
        pygame.RESIZABLE,
    ]


def test_initialize_display_surfaces_skips_gl_for_dummy_driver(
    pygame_headless,
    monkeypatch,
) -> None:
    calls: list[int] = []
    display_surface = pygame.Surface((320, 240))

    monkeypatch.setattr(
        "pygame.display.set_mode",
        lambda size, flags: (calls.append(int(flags)) or display_surface),
    )

    display, app_surface, renderer, active_flags = _initialize_display_surfaces(
        window_size=(320, 240),
        window_flags=pygame.RESIZABLE,
        video_driver="dummy",
        want_gl=True,
    )

    assert display is display_surface
    assert app_surface is display_surface
    assert renderer is None
    assert active_flags == pygame.RESIZABLE
    assert calls == [pygame.RESIZABLE]
