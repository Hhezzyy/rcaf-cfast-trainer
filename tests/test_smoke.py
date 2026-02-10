"""Smoke tests for the pygame UI.

These tests verify that the application's main loop can initialise and
execute a handful of frames without crashing when the SDL dummy video
driver is used.  They do not attempt to exercise user interaction or
rendering correctness; rather they ensure that the integration points
between pygame and the application do not raise exceptions in a headless
environment.
"""

from __future__ import annotations

import os

# Use the dummy drivers before importing pygame or the application
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


def test_app_runs_headless() -> None:
    """Ensure the application can start and run a few frames headlessly."""
    # Import inside the test so that environment variables take effect
    from cfast_trainer.app import run

    exit_code = run(max_frames=3)
    assert exit_code == 0