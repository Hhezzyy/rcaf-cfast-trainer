from __future__ import annotations

import pygame

from cfast_trainer.app import (
    DisplayLifecycleState,
    _resolve_display_rebootstrap,
    _resolve_use_opengl,
    _resolve_window_mode,
)


def test_window_mode_defaults_to_windowed_even_on_macos(monkeypatch) -> None:
    monkeypatch.delenv("CFAST_WINDOW_MODE", raising=False)
    monkeypatch.delenv("CFAST_FULLSCREEN", raising=False)

    assert _resolve_window_mode(video_driver="", platform_name="darwin") == "windowed"


def test_window_mode_honors_fullscreen_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("CFAST_WINDOW_MODE", raising=False)
    monkeypatch.setenv("CFAST_FULLSCREEN", "1")

    assert _resolve_window_mode(video_driver="", platform_name="darwin") == "borderless"
    assert _resolve_window_mode(video_driver="", platform_name="linux") == "fullscreen"


def test_window_mode_explicit_env_override_wins(monkeypatch) -> None:
    monkeypatch.setenv("CFAST_WINDOW_MODE", "fullscreen")
    monkeypatch.setenv("CFAST_FULLSCREEN", "0")

    assert _resolve_window_mode(video_driver="", platform_name="darwin") == "fullscreen"


def test_dummy_video_driver_forces_windowed(monkeypatch) -> None:
    monkeypatch.setenv("CFAST_WINDOW_MODE", "fullscreen")
    monkeypatch.setenv("CFAST_FULLSCREEN", "1")

    assert _resolve_window_mode(video_driver="dummy", platform_name="darwin") == "windowed"


def test_window_mode_ignores_stored_fullscreen_default_when_env_is_absent(monkeypatch) -> None:
    monkeypatch.delenv("CFAST_WINDOW_MODE", raising=False)
    monkeypatch.delenv("CFAST_FULLSCREEN", raising=False)

    assert (
        _resolve_window_mode(
            video_driver="",
            platform_name="linux",
            stored_mode="borderless",
        )
        == "windowed"
    )


def test_use_opengl_honors_env_override_and_stored_default(monkeypatch) -> None:
    monkeypatch.setenv("CFAST_USE_OPENGL", "0")
    assert _resolve_use_opengl(stored_default=True) is False

    monkeypatch.setenv("CFAST_USE_OPENGL", "1")
    assert _resolve_use_opengl(stored_default=False) is True

    monkeypatch.delenv("CFAST_USE_OPENGL", raising=False)
    assert _resolve_use_opengl(stored_default=False) is False
    assert _resolve_use_opengl(stored_default=None) is True


def test_display_rebootstrap_detects_stale_native_fullscreen_transition() -> None:
    state = DisplayLifecycleState(
        window_size=(1440, 900),
        surface_size=(960, 540),
        desktop_size=(1440, 900),
        active_window_flags=pygame.RESIZABLE | pygame.OPENGL | pygame.DOUBLEBUF,
        window_mode="windowed",
    )

    decision = _resolve_display_rebootstrap(state)

    assert decision is not None
    assert decision.window_mode == "fullscreen"
    assert decision.window_size == (1440, 900)


def test_display_rebootstrap_detects_restore_from_controlled_fullscreen() -> None:
    state = DisplayLifecycleState(
        window_size=(1440, 872),
        surface_size=(1440, 900),
        desktop_size=(1440, 900),
        active_window_flags=pygame.FULLSCREEN | pygame.OPENGL | pygame.DOUBLEBUF,
        window_mode="fullscreen",
    )

    decision = _resolve_display_rebootstrap(state)

    assert decision is not None
    assert decision.window_mode == "windowed"
    assert decision.window_size == (1440, 872)


def test_display_rebootstrap_ignores_healthy_large_window_surface_pairing() -> None:
    state = DisplayLifecycleState(
        window_size=(1440, 872),
        surface_size=(1440, 900),
        desktop_size=(1440, 900),
        active_window_flags=pygame.RESIZABLE | pygame.OPENGL | pygame.DOUBLEBUF,
        window_mode="windowed",
    )

    assert _resolve_display_rebootstrap(state) is None
