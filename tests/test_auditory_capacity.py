from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import wave

import pytest

from cfast_trainer.auditory_capacity import (
    AUDITORY_GATE_RETIRE_X_NORM,
    AUDITORY_GATE_SPAWN_X_NORM,
    AUDITORY_TRIANGLE_GATE_POINTS,
    AuditoryCapacityConfig,
    build_auditory_capacity_test,
    project_inside_tube,
    tube_contact_ratio,
)
from cfast_trainer.auditory_capacity_panda3d import AuditoryCapacityPanda3DRenderer


@dataclass
class _FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


def test_default_config_values_match_auditory_overhaul() -> None:
    cfg = AuditoryCapacityConfig()

    assert cfg.tube_half_width == pytest.approx(0.82)
    assert cfg.tube_half_height == pytest.approx(0.60)
    assert cfg.gate_spawn_rate == pytest.approx(0.24)
    assert cfg.gate_interval_s == pytest.approx(4.2)


def test_engine_and_panda_renderer_share_gate_anchors() -> None:
    assert AUDITORY_GATE_SPAWN_X_NORM == pytest.approx(1.65)
    assert AUDITORY_GATE_RETIRE_X_NORM == pytest.approx(-1.25)
    assert AuditoryCapacityPanda3DRenderer._GATE_SPAWN_X_NORM == pytest.approx(
        AUDITORY_GATE_SPAWN_X_NORM
    )
    assert AuditoryCapacityPanda3DRenderer._GATE_RETIRE_X_NORM == pytest.approx(
        AUDITORY_GATE_RETIRE_X_NORM
    )


def test_default_gate_interval_comes_from_spawn_rate() -> None:
    clock = _FakeClock()
    engine = build_auditory_capacity_test(clock=clock, seed=17, difficulty=0.5)

    assert engine._gate_interval_s() == pytest.approx(1.0 / 0.24)


def test_spawned_gate_uses_shared_far_spawn_anchor() -> None:
    clock = _FakeClock()
    engine = build_auditory_capacity_test(clock=clock, seed=17, difficulty=0.5)
    engine.start_practice()
    engine._next_gate_at_s = 0.0

    engine._update_gates(0.0)

    assert len(engine._gates) == 1
    assert engine._gates[0].x_norm == pytest.approx(AUDITORY_GATE_SPAWN_X_NORM)


def test_triangle_gate_profile_is_equilateral() -> None:
    sides: list[float] = []
    for idx, (x0, y0) in enumerate(AUDITORY_TRIANGLE_GATE_POINTS):
        x1, y1 = AUDITORY_TRIANGLE_GATE_POINTS[(idx + 1) % len(AUDITORY_TRIANGLE_GATE_POINTS)]
        sides.append(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5)

    assert max(sides) - min(sides) <= 0.01


def test_taller_wider_more_circular_default_tunnel_is_more_forgiving_than_previous_shape() -> None:
    point_x = 0.78
    point_y = 0.18

    old_ratio = tube_contact_ratio(x=point_x, y=point_y, tube_half_width=0.74, tube_half_height=0.44)
    new_ratio = tube_contact_ratio(
        x=point_x,
        y=point_y,
        tube_half_width=AuditoryCapacityConfig().tube_half_width,
        tube_half_height=AuditoryCapacityConfig().tube_half_height,
    )
    px, py, raw_ratio = project_inside_tube(
        x=point_x,
        y=point_y,
        tube_half_width=AuditoryCapacityConfig().tube_half_width,
        tube_half_height=AuditoryCapacityConfig().tube_half_height,
    )

    assert old_ratio > 1.0
    assert new_ratio < 1.0
    assert raw_ratio == pytest.approx(new_ratio)
    assert px == pytest.approx(point_x)
    assert py == pytest.approx(point_y)


def test_converted_runtime_audio_assets_exist_and_load_cleanly() -> None:
    asset_dir = Path(__file__).resolve().parents[1] / "assets" / "audio" / "auditory_capacity"
    expected = (
        "background_noise_1.wav",
        "background_noise_2.wav",
        "background_noise_3.wav",
        "background_noise_4.wav",
        "background_noise_5.wav",
        "background_noise_6.wav",
        "mature_1.wav",
        "mature_2.wav",
        "mature_distraction_1.wav",
        "mature_distraction_2.wav",
        "mature_distraction_3.wav",
        "mature_distraction_4.wav",
        "mature_distraction_5.wav",
        "mature_distraction_6.wav",
        "mature_distraction_7.wav",
        "mature_distraction_8.wav",
        "mature_distraction_9.wav",
    )

    for name in expected:
        path = asset_dir / name
        assert path.exists(), name
        with wave.open(str(path), "rb") as wav_file:
            assert wav_file.getframerate() == 22050
            assert wav_file.getnchannels() == 1
            assert wav_file.getsampwidth() == 2
            assert wav_file.getnframes() > 0
