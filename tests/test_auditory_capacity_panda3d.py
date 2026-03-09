from __future__ import annotations

from cfast_trainer.auditory_capacity_panda3d import panda3d_auditory_rendering_available


def test_panda3d_auditory_rendering_disabled_for_dummy_video(monkeypatch) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.delenv("CFAST_AUDITORY_RENDERER", raising=False)

    assert panda3d_auditory_rendering_available() is False


def test_panda3d_auditory_rendering_disabled_when_forced_to_pygame(monkeypatch) -> None:
    monkeypatch.setenv("CFAST_AUDITORY_RENDERER", "pygame")
    monkeypatch.delenv("SDL_VIDEODRIVER", raising=False)

    assert panda3d_auditory_rendering_available() is False
