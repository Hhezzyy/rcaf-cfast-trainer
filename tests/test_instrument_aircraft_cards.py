from __future__ import annotations

import pygame
import pytest

from cfast_trainer.instrument_aircraft_cards import (
    InstrumentAircraftCardKey,
    aircraft_card_pose_signature,
    aircraft_card_pitch_cue_px,
    aircraft_card_pose_distance,
    aircraft_card_projected_heading_deg,
    aircraft_card_semantic_drift_tags,
    aircraft_card_wing_tilt_px,
    InstrumentAircraftCardSpriteBank,
)
from cfast_trainer.instrument_comprehension import InstrumentAircraftViewPreset, InstrumentState


def _state(*, heading_deg: int, pitch_deg: int, bank_deg: int) -> InstrumentState:
    return InstrumentState(
        speed_kts=220,
        altitude_ft=5000,
        vertical_rate_fpm=0,
        bank_deg=bank_deg,
        pitch_deg=pitch_deg,
        heading_deg=heading_deg,
        slip=0,
    )


def test_card_key_normalizes_state_values() -> None:
    key = InstrumentAircraftCardKey.from_state(
        _state(heading_deg=721, pitch_deg=29, bank_deg=-63),
        view_preset=InstrumentAircraftViewPreset.FRONT_LEFT,
    )

    assert key.heading_deg == 1
    assert key.pitch_deg == 20
    assert key.bank_deg == -45
    assert key.view_preset is InstrumentAircraftViewPreset.FRONT_LEFT
    assert key.filename() == "v20_front_left_h001_pp20_bm45.png"


def test_sprite_bank_loads_cached_png_without_generation(tmp_path) -> None:
    cache_dir = tmp_path / "cards"
    cache_dir.mkdir()

    state = _state(heading_deg=90, pitch_deg=4, bank_deg=-12)
    key = InstrumentAircraftCardKey.from_state(state)
    cached_path = cache_dir / key.filename()

    pygame.init()
    try:
        source = pygame.Surface((32, 20), pygame.SRCALPHA)
        source.fill((210, 40, 60, 255))
        pygame.image.save(source, str(cached_path))

        bank = InstrumentAircraftCardSpriteBank(cache_dir=cache_dir, allow_generation=False)
        loaded = bank.get_surface(state=state)

        assert loaded is not None
        assert loaded.get_size() == (32, 20)
        scaled = bank.get_scaled_surface(state=state, size=(64, 40))
        assert scaled is not None
        assert scaled.get_size() == (64, 40)
    finally:
        pygame.quit()


def test_pose_signature_is_stable_and_changes_with_orientation() -> None:
    base = _state(heading_deg=90, pitch_deg=4, bank_deg=-12)
    same = _state(heading_deg=90, pitch_deg=4, bank_deg=-12)
    different = _state(heading_deg=180, pitch_deg=-6, bank_deg=18)

    sig_a = aircraft_card_pose_signature(base, view_preset=InstrumentAircraftViewPreset.FRONT_LEFT)
    sig_b = aircraft_card_pose_signature(same, view_preset=InstrumentAircraftViewPreset.FRONT_LEFT)
    sig_c = aircraft_card_pose_signature(different, view_preset=InstrumentAircraftViewPreset.FRONT_LEFT)

    assert sig_a == sig_b
    assert aircraft_card_pose_distance(sig_a, sig_b) == pytest.approx(0.0)
    assert aircraft_card_pose_distance(sig_a, sig_c) > 0.0


def test_card_presets_keep_cardinal_heading_projection_reference_aligned() -> None:
    expected_angles = {
        0: 90.0,
        90: 0.0,
        180: -90.0,
        270: 180.0,
    }
    for preset in InstrumentAircraftViewPreset:
        for heading_deg, expected in expected_angles.items():
            signature = aircraft_card_pose_signature(
                _state(heading_deg=heading_deg, pitch_deg=0, bank_deg=0),
                view_preset=preset,
            )
            projected = aircraft_card_projected_heading_deg(signature)
            error = abs(((projected - expected + 180.0) % 360.0) - 180.0)
            assert error <= 8.0, (preset, heading_deg, projected, expected)
            assert "heading_axis" not in aircraft_card_semantic_drift_tags(
                _state(heading_deg=heading_deg, pitch_deg=0, bank_deg=0),
                view_preset=preset,
            )


def test_neutral_bank_and_signed_bank_deltas_stay_semantically_aligned() -> None:
    for preset in InstrumentAircraftViewPreset:
        neutral_state = _state(heading_deg=90, pitch_deg=0, bank_deg=0)
        left_bank_state = _state(heading_deg=90, pitch_deg=0, bank_deg=-20)
        right_bank_state = _state(heading_deg=90, pitch_deg=0, bank_deg=20)

        neutral_sig = aircraft_card_pose_signature(neutral_state, view_preset=preset)
        left_sig = aircraft_card_pose_signature(left_bank_state, view_preset=preset)
        right_sig = aircraft_card_pose_signature(right_bank_state, view_preset=preset)

        assert "bank_neutral" not in aircraft_card_semantic_drift_tags(
            neutral_state,
            view_preset=preset,
        )
        assert "bank_sign" not in aircraft_card_semantic_drift_tags(
            left_bank_state,
            view_preset=preset,
        )
        assert "bank_sign" not in aircraft_card_semantic_drift_tags(
            right_bank_state,
            view_preset=preset,
        )

        neutral_tilt = aircraft_card_wing_tilt_px(neutral_sig)
        left_delta = aircraft_card_wing_tilt_px(left_sig) - neutral_tilt
        right_delta = aircraft_card_wing_tilt_px(right_sig) - neutral_tilt
        assert left_delta > 0.0, (preset, left_delta)
        assert right_delta < 0.0, (preset, right_delta)
        assert abs(abs(left_delta) - abs(right_delta)) <= 12.0, (preset, left_delta, right_delta)


def test_pitch_metric_moves_symmetrically_around_level_flight() -> None:
    for preset in InstrumentAircraftViewPreset:
        neutral_sig = aircraft_card_pose_signature(
            _state(heading_deg=90, pitch_deg=0, bank_deg=0),
            view_preset=preset,
        )
        descent_sig = aircraft_card_pose_signature(
            _state(heading_deg=90, pitch_deg=-10, bank_deg=0),
            view_preset=preset,
        )
        climb_sig = aircraft_card_pose_signature(
            _state(heading_deg=90, pitch_deg=10, bank_deg=0),
            view_preset=preset,
        )
        neutral_pitch = aircraft_card_pitch_cue_px(neutral_sig)
        descent_delta = aircraft_card_pitch_cue_px(descent_sig) - neutral_pitch
        climb_delta = aircraft_card_pitch_cue_px(climb_sig) - neutral_pitch

        assert "pitch_sign" not in aircraft_card_semantic_drift_tags(
            _state(heading_deg=90, pitch_deg=-10, bank_deg=0),
            view_preset=preset,
        )
        assert "pitch_sign" not in aircraft_card_semantic_drift_tags(
            _state(heading_deg=90, pitch_deg=10, bank_deg=0),
            view_preset=preset,
        )
        assert descent_delta < 0.0, (preset, descent_delta)
        assert climb_delta > 0.0, (preset, climb_delta)
        assert abs(abs(descent_delta) - abs(climb_delta)) <= 2.5, (
            preset,
            descent_delta,
            climb_delta,
        )


def _red_bounds(surface: pygame.Surface) -> tuple[int, int, int, int] | None:
    min_x = surface.get_width()
    min_y = surface.get_height()
    max_x = -1
    max_y = -1
    for y in range(surface.get_height()):
        for x in range(surface.get_width()):
            color = surface.get_at((x, y))
            if color.a <= 0:
                continue
            if color.r <= 120 or color.r <= color.g + 25 or color.r <= color.b + 25:
                continue
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
    if max_x < min_x or max_y < min_y:
        return None
    return (min_x, min_y, max_x, max_y)


def test_generated_card_presets_keep_aircraft_inside_safe_inset(tmp_path, monkeypatch) -> None:
    cache_dir = tmp_path / "generated-cards"
    state = _state(heading_deg=82, pitch_deg=7, bank_deg=-14)
    bank = InstrumentAircraftCardSpriteBank(cache_dir=cache_dir, allow_generation=True)

    class _FakeRenderer:
        def render_card(self, *, key, destination, draw_callback, size) -> None:
            _ = (key, draw_callback)
            surface = pygame.Surface(size, pygame.SRCALPHA)
            surface.fill((232, 232, 232, 255))
            pygame.draw.polygon(
                surface,
                (210, 40, 60, 255),
                [
                    (size[0] // 2, 36),
                    (size[0] // 2 + 58, size[1] // 2),
                    (size[0] // 2, size[1] - 36),
                    (size[0] // 2 - 58, size[1] // 2),
                ],
            )
            pygame.image.save(surface, str(destination))

    monkeypatch.setattr(bank, "_get_renderer", lambda: _FakeRenderer())

    for preset in InstrumentAircraftViewPreset:
        surface = bank.get_surface(state=state, view_preset=preset)
        assert surface is not None
        bounds = _red_bounds(surface)
        assert bounds is not None
        min_x, min_y, max_x, max_y = bounds
        inset = 4
        assert min_x >= inset, preset
        assert min_y >= inset, preset
        assert max_x <= surface.get_width() - inset, preset
        assert max_y <= surface.get_height() - inset, preset
        for sample in (
            (2, 2),
            (surface.get_width() - 3, 2),
            (2, surface.get_height() - 3),
            (surface.get_width() - 3, surface.get_height() - 3),
        ):
            assert surface.get_at(sample).a > 0, preset


def test_generated_card_uses_software_fallback_when_renderer_generation_fails(tmp_path, monkeypatch) -> None:
    bank = InstrumentAircraftCardSpriteBank(cache_dir=tmp_path / "generated-cards", allow_generation=True)

    class _FailingRenderer:
        def render_card(self, *, key, destination, draw_callback, size) -> None:
            _ = (key, destination, draw_callback, size)
            raise RuntimeError("renderer boom")

    monkeypatch.setattr(bank, "_get_renderer", lambda: _FailingRenderer())

    pygame.init()
    try:
        surface = bank.get_surface(state=_state(heading_deg=90, pitch_deg=0, bank_deg=0))
        assert surface is not None
        assert bank._generation_failed is True
        assert _red_bounds(surface) is not None
    finally:
        pygame.quit()
