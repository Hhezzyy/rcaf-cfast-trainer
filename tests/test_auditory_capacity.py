from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import wave

import pytest

from cfast_trainer.auditory_capacity import (
    AUDITORY_GATE_PLAYER_X_NORM,
    AUDITORY_GATE_RETIRE_X_NORM,
    AUDITORY_GATE_SPAWN_X_NORM,
    AUDITORY_TRIANGLE_GATE_POINTS,
    AuditoryCapacityConfig,
    build_auditory_capacity_test,
    project_inside_tube,
    tube_contact_ratio,
)
from cfast_trainer.auditory_capacity_panda3d import AuditoryCapacityPanda3DRenderer
from cfast_trainer.auditory_capacity_view import (
    gate_distance_from_x_norm,
    run_start_distance,
    run_travel_distance,
)


@dataclass
class _FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


def _difficulty_ratio(level: int) -> float:
    return max(0.0, min(1.0, float(level - 1) / 9.0))


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


def test_default_profile_stays_bounded_low_and_overloads_high() -> None:
    low_clock = _FakeClock()
    high_clock = _FakeClock()
    low_engine = build_auditory_capacity_test(
        clock=low_clock,
        seed=17,
        difficulty=_difficulty_ratio(1),
    )
    high_engine = build_auditory_capacity_test(
        clock=high_clock,
        seed=17,
        difficulty=_difficulty_ratio(10),
    )

    low_engine.start_practice()
    low_engine.start_scored()
    high_engine.start_practice()
    high_engine.start_scored()

    low_digits = "".join(
        ch
        for ch in str(low_engine._build_runtime_digit_sequence_event(timestamp_s=1.0).payload)
        if ch.isdigit()
    )
    high_digits = "".join(
        ch
        for ch in str(high_engine._build_runtime_digit_sequence_event(timestamp_s=1.0).payload)
        if ch.isdigit()
    )

    assert len(low_digits) in (4, 5)
    assert len(high_digits) == 10
    assert len(low_engine._assigned_callsigns) == 3
    assert len(high_engine._assigned_callsigns) == 4
    assert low_engine._effective_gate_interval_s() == pytest.approx(4.2, abs=0.05)
    assert high_engine._effective_gate_interval_s() == pytest.approx(1.6, abs=0.05)
    assert high_engine._effective_gate_interval_s() < low_engine._effective_gate_interval_s()


def test_spawned_gate_uses_shared_far_spawn_anchor() -> None:
    clock = _FakeClock()
    engine = build_auditory_capacity_test(clock=clock, seed=17, difficulty=0.5)
    engine.start_practice()
    engine._next_gate_at_s = 0.0

    engine._update_gates(0.0)

    assert len(engine._gates) == 1
    assert engine._gates[0].x_norm == pytest.approx(AUDITORY_GATE_SPAWN_X_NORM)


def test_live_tunnel_travel_stays_monotonic_even_when_ball_forward_norm_relaxes() -> None:
    clock = _FakeClock()
    engine = build_auditory_capacity_test(clock=clock, seed=17, difficulty=0.58)
    engine.start_practice()

    samples: list[tuple[float, float, float]] = []
    for step in range(16 * 120):
        clock.advance(1.0 / 120.0)
        engine.update()
        if step % 60 != 0:
            continue
        payload = engine.snapshot().payload
        assert payload is not None
        samples.append(
            (
                float(payload.phase_elapsed_s),
                float(payload.ball_forward_norm),
                run_travel_distance(
                    session_seed=int(payload.session_seed),
                    phase_elapsed_s=float(payload.phase_elapsed_s),
                ),
            )
        )

    assert any(
        samples[idx + 1][1] + 0.20 < samples[idx][1]
        for idx in range(len(samples) - 1)
    )
    assert all(
        samples[idx + 1][2] >= samples[idx][2]
        for idx in range(len(samples) - 1)
    )


def test_live_tunnel_run_start_is_seeded_repeatable_and_varies_between_runs() -> None:
    first = run_start_distance(17)
    second = run_start_distance(17)
    other = run_start_distance(99)

    assert first == pytest.approx(second)
    assert 0.0 <= first
    assert other != pytest.approx(first)


def test_correct_pass_gate_flashes_white_on_the_gate() -> None:
    clock = _FakeClock()
    engine = build_auditory_capacity_test(clock=clock, seed=17, difficulty=0.58)
    engine.start_practice()
    engine._next_gate_at_s = 0.0
    engine._update_gates(0.0)

    assert len(engine._gates) == 1
    gate = engine._gates[0]
    gate.x_norm = float(AUDITORY_GATE_PLAYER_X_NORM)
    gate.y_norm = float(engine._ball_y)
    gate.aperture_norm = 0.22

    engine._update_gates(0.0)
    payload = engine.snapshot().payload

    assert payload is not None
    flashed = next(g for g in payload.gates if g.gate_id == gate.gate_id)
    assert flashed.flash_color == "WHITE"
    assert flashed.flash_strength > 0.95
    assert payload.gate_hits == 1
    assert payload.gate_misses == 0
    assert payload.forbidden_gate_hits == 0


def test_forbidden_pass_flashes_distinct_error_red_on_the_gate() -> None:
    clock = _FakeClock()
    engine = build_auditory_capacity_test(clock=clock, seed=17, difficulty=0.58)
    engine.start_practice()
    engine._next_gate_at_s = 0.0
    engine._update_gates(0.0)

    assert len(engine._gates) == 1
    gate = engine._gates[0]
    engine._forbidden_gate_color = str(gate.color)
    gate.x_norm = float(AUDITORY_GATE_PLAYER_X_NORM)
    gate.y_norm = float(engine._ball_y)
    gate.aperture_norm = 0.22

    engine._update_gates(0.0)
    payload = engine.snapshot().payload

    assert payload is not None
    flashed = next(g for g in payload.gates if g.gate_id == gate.gate_id)
    assert flashed.flash_color == "ERROR_RED"
    assert flashed.flash_strength > 0.95
    assert payload.gate_hits == 0
    assert payload.gate_misses == 1
    assert payload.forbidden_gate_hits == 1


def test_gate_distance_comes_from_live_x_progress_and_preserves_spacing() -> None:
    travel_distance = 18.0
    far = gate_distance_from_x_norm(
        AUDITORY_GATE_SPAWN_X_NORM,
        travel_distance=travel_distance,
        spawn_x_norm=AUDITORY_GATE_SPAWN_X_NORM,
        player_x_norm=AUDITORY_GATE_PLAYER_X_NORM,
        retire_x_norm=AUDITORY_GATE_RETIRE_X_NORM,
    )
    mid = gate_distance_from_x_norm(
        1.15,
        travel_distance=travel_distance,
        spawn_x_norm=AUDITORY_GATE_SPAWN_X_NORM,
        player_x_norm=AUDITORY_GATE_PLAYER_X_NORM,
        retire_x_norm=AUDITORY_GATE_RETIRE_X_NORM,
    )
    near = gate_distance_from_x_norm(
        0.55,
        travel_distance=travel_distance,
        spawn_x_norm=AUDITORY_GATE_SPAWN_X_NORM,
        player_x_norm=AUDITORY_GATE_PLAYER_X_NORM,
        retire_x_norm=AUDITORY_GATE_RETIRE_X_NORM,
    )
    contact = gate_distance_from_x_norm(
        AUDITORY_GATE_PLAYER_X_NORM,
        travel_distance=travel_distance,
        spawn_x_norm=AUDITORY_GATE_SPAWN_X_NORM,
        player_x_norm=AUDITORY_GATE_PLAYER_X_NORM,
        retire_x_norm=AUDITORY_GATE_RETIRE_X_NORM,
    )
    behind = gate_distance_from_x_norm(
        -0.80,
        travel_distance=travel_distance,
        spawn_x_norm=AUDITORY_GATE_SPAWN_X_NORM,
        player_x_norm=AUDITORY_GATE_PLAYER_X_NORM,
        retire_x_norm=AUDITORY_GATE_RETIRE_X_NORM,
    )

    assert far > mid > near > contact > behind
    assert (far - mid) > 6.0
    assert (mid - near) > 6.0
    assert (near - contact) > 6.0


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
