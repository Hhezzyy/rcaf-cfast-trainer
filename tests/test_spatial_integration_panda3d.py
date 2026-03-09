from __future__ import annotations

from cfast_trainer.spatial_integration_panda3d import (
    panda3d_spatial_integration_rendering_available,
)


def test_panda3d_spatial_integration_rendering_disabled_for_dummy_video(monkeypatch) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.delenv("CFAST_SPATIAL_INTEGRATION_RENDERER", raising=False)
    assert panda3d_spatial_integration_rendering_available() is False


def test_panda3d_spatial_integration_rendering_disabled_when_forced_to_pygame(
    monkeypatch,
) -> None:
    monkeypatch.setenv("CFAST_SPATIAL_INTEGRATION_RENDERER", "pygame")
    monkeypatch.delenv("SDL_VIDEODRIVER", raising=False)
    assert panda3d_spatial_integration_rendering_available() is False
