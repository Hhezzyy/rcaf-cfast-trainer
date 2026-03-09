from __future__ import annotations

from cfast_trainer.app import _resolve_window_mode


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
