from __future__ import annotations

import os
from dataclasses import dataclass
from types import SimpleNamespace

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from cfast_trainer.cognitive_core import Phase
from cfast_trainer.gl_scenes import RapidTrackingGlScene
from cfast_trainer.rapid_tracking import (
    RapidTrackingConfig,
    RapidTrackingPayload,
    build_rapid_tracking_test,
)


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


@dataclass
class FakeApp:
    opengl_enabled: bool = True

    def __post_init__(self) -> None:
        self.queued: list[RapidTrackingGlScene] = []

    def queue_gl_scene(self, scene: RapidTrackingGlScene) -> None:
        self.queued.append(scene)

    def dev_tools_enabled(self) -> bool:
        return True


def _sample_sequence(
    engine,
    *,
    clock: FakeClock,
    steps: int = 8,
) -> list[tuple[float, float, float, float, float]]:
    controls = [(0.35, -0.18), (0.0, 0.0), (-0.22, 0.16), (0.18, -0.08)]
    samples: list[tuple[float, float, float, float, float]] = []
    for idx in range(steps):
        cx, cy = controls[idx % len(controls)]
        engine.set_control(horizontal=cx, vertical=cy)
        clock.advance(0.25)
        engine.update()
        payload = engine.snapshot().payload
        assert isinstance(payload, RapidTrackingPayload)
        samples.append(
            (
                float(payload.target_world_x),
                float(payload.target_world_y),
                float(payload.camera_yaw_deg),
                float(payload.camera_pitch_deg),
                float(payload.scene_progress),
            )
        )
    return samples


def test_same_seed_reset_replays_same_scored_sequence() -> None:
    clock = FakeClock()
    engine = build_rapid_tracking_test(
        clock=clock,
        seed=551,
        difficulty=0.63,
        config=RapidTrackingConfig(
            practice_duration_s=0.0,
            scored_duration_s=8.0,
            tick_hz=120.0,
        ),
    )
    engine.start_scored()

    first = _sample_sequence(engine, clock=clock)
    engine.reset()
    assert engine.phase is Phase.SCORED
    second = _sample_sequence(engine, clock=clock)

    assert first == second


def test_reseed_advances_seed_and_keeps_current_mode() -> None:
    clock = FakeClock()
    engine = build_rapid_tracking_test(
        clock=clock,
        seed=551,
        difficulty=0.5,
        config=RapidTrackingConfig(practice_duration_s=0.0, scored_duration_s=10.0),
    )
    engine.start_scored()
    original_scene_seed = engine.scene_seed

    next_seed = engine.reseed()
    snap = engine.snapshot()
    assert engine.phase is Phase.SCORED
    assert next_seed == 552
    assert engine.seed == 552
    assert isinstance(snap.payload, RapidTrackingPayload)
    assert snap.payload.session_seed == 552
    assert snap.payload.scene_seed != original_scene_seed


def test_dev_hotkeys_toggle_debug_and_support_instruction_reset() -> None:
    clock = FakeClock()
    engine = build_rapid_tracking_test(
        clock=clock,
        seed=600,
        difficulty=0.5,
    )
    engine.set_dev_tools_enabled(True)
    engine.start_practice()
    clock.advance(0.3)
    engine.update()

    assert engine.handle_event(SimpleNamespace(type=pygame.KEYDOWN, key=pygame.K_F2, mod=0)) is True
    assert engine.debug_state.overlay_enabled is True

    assert engine.handle_event(SimpleNamespace(type=pygame.KEYDOWN, key=pygame.K_F3, mod=0)) is True
    assert engine.debug_state.diagnostics_enabled is True

    reset_count = engine.debug_state.reset_count
    assert engine.handle_event(SimpleNamespace(type=pygame.KEYDOWN, key=pygame.K_F5, mod=0)) is True
    assert engine.phase is Phase.PRACTICE
    assert engine.debug_state.reset_count == reset_count + 1

    assert (
        engine.handle_event(
            SimpleNamespace(type=pygame.KEYDOWN, key=pygame.K_F5, mod=pygame.KMOD_SHIFT)
        )
        is True
    )
    assert engine.phase is Phase.INSTRUCTIONS


def test_scene_render_queues_modern_gl_and_exposes_dev_buttons() -> None:
    pygame.init()
    try:
        clock = FakeClock()
        app = FakeApp(opengl_enabled=True)
        engine = build_rapid_tracking_test(clock=clock, seed=551, difficulty=0.5)
        engine.bind_screen_context(
            app=app,
            small_font=pygame.font.Font(None, 24),
            tiny_font=pygame.font.Font(None, 18),
        )
        engine.enter()
        surface = pygame.Surface((960, 540))

        engine.render(surface)
        assert app.queued
        queued = app.queued[-1]
        assert isinstance(queued, RapidTrackingGlScene)
        assert isinstance(queued.payload, RapidTrackingPayload)
        assert set(engine._dev_button_hitboxes) >= {
            "reset",
            "reseed",
            "instructions",
            "practice",
            "scored",
            "debug",
            "camera",
        }

        scored_rect = engine._dev_button_hitboxes["scored"]
        clicked = engine.handle_event(
            SimpleNamespace(
                type=pygame.MOUSEBUTTONDOWN,
                button=1,
                pos=scored_rect.center,
            )
        )
        assert clicked is True
        assert engine.phase is Phase.SCORED
    finally:
        pygame.quit()
