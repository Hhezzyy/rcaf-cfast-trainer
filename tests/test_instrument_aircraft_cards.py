from __future__ import annotations

import pygame
import pytest

from cfast_trainer.instrument_aircraft_cards import (
    InstrumentAircraftCardKey,
    InstrumentAircraftCardSpriteBank,
    aircraft_card_pitch_cue_px,
    aircraft_card_pose_distance,
    aircraft_card_pose_signature,
    aircraft_card_projected_heading_deg,
    aircraft_card_semantic_drift_tags,
    aircraft_card_wing_tilt_px,
)
from cfast_trainer.instrument_comprehension import (
    InstrumentAircraftViewPreset,
    InstrumentHeadingDisplayMode,
    InstrumentState,
)
from cfast_trainer.instrument_orientation_solver import (
    attitude_display_observation_from_state,
    heading_display_observation_from_state,
)


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
    assert key.filename() == "mesh_v3_mechanical_front_left_h001_pp20_bm45.png"


def test_aircraft_cards_do_not_embed_instrument_overlay_pixels() -> None:
    bank = InstrumentAircraftCardSpriteBank()
    state = _state(heading_deg=90, pitch_deg=8, bank_deg=-16)

    pygame.init()
    try:
        surface = bank.get_surface(
            state=state,
            view_preset=InstrumentAircraftViewPreset.FRONT_LEFT,
        )
        sample = pygame.Rect(8, 8, 130, 56)
        instrument_like_pixels = 0
        for y in range(sample.y, sample.bottom):
            for x in range(sample.x, sample.right):
                color = surface.get_at((x, y))
                is_heading_yellow = color.r > 230 and color.g > 190 and color.b < 150
                is_attitude_sky = color.r < 90 and color.g > 120 and color.b > 180
                is_attitude_ground = color.r > 120 and 60 < color.g < 140 and color.b < 90
                if is_heading_yellow or is_attitude_sky or is_attitude_ground:
                    instrument_like_pixels += 1

        assert instrument_like_pixels == 0
    finally:
        pygame.quit()


def test_pose_signature_is_stable_and_changes_with_orientation() -> None:
    base = _state(heading_deg=90, pitch_deg=4, bank_deg=-12)
    same = _state(heading_deg=90, pitch_deg=4, bank_deg=-12)
    different = _state(heading_deg=180, pitch_deg=-6, bank_deg=18)

    sig_a = aircraft_card_pose_signature(base, view_preset=InstrumentAircraftViewPreset.FRONT_LEFT)
    sig_b = aircraft_card_pose_signature(same, view_preset=InstrumentAircraftViewPreset.FRONT_LEFT)
    sig_c = aircraft_card_pose_signature(
        different,
        view_preset=InstrumentAircraftViewPreset.FRONT_LEFT,
    )

    assert sig_a == sig_b
    assert aircraft_card_pose_distance(sig_a, sig_b) == pytest.approx(0.0)
    assert aircraft_card_pose_distance(sig_a, sig_c) > 60.0


def test_top_down_heading_projection_preserves_cardinals() -> None:
    expected_angles = {
        0: 90.0,
        90: 0.0,
        180: -90.0,
        270: 180.0,
    }
    for heading_deg, expected in expected_angles.items():
        signature = aircraft_card_pose_signature(
            _state(heading_deg=heading_deg, pitch_deg=0, bank_deg=0),
            view_preset=InstrumentAircraftViewPreset.TOP_DOWN,
        )
        projected = aircraft_card_projected_heading_deg(signature)
        error = abs(((projected - expected + 180.0) % 360.0) - 180.0)
        assert error <= 10.0, (heading_deg, projected, expected)


def test_canonical_heading_projection_matches_gauge_cardinals_in_both_modes() -> None:
    for mode in InstrumentHeadingDisplayMode:
        for heading_deg in (0, 90, 180, 270):
            state = _state(heading_deg=heading_deg, pitch_deg=0, bank_deg=0)
            observation = heading_display_observation_from_state(state, mode)
            signature = aircraft_card_pose_signature(
                state,
                view_preset=InstrumentAircraftViewPreset.FRONT_LEFT,
            )
            projected = aircraft_card_projected_heading_deg(signature)
            expected = ((90.0 - float(observation.heading_deg) + 180.0) % 360.0) - 180.0
            if expected == -180.0:
                expected = 180.0
            error = abs(((projected - expected + 180.0) % 360.0) - 180.0)
            assert error <= 3.0, (mode, heading_deg, projected, expected)


def test_signed_bank_deltas_stay_semantically_aligned() -> None:
    presets = (
        InstrumentAircraftViewPreset.FRONT_LEFT,
        InstrumentAircraftViewPreset.FRONT_RIGHT,
        InstrumentAircraftViewPreset.PROFILE_LEFT,
        InstrumentAircraftViewPreset.PROFILE_RIGHT,
    )
    for preset in presets:
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


def test_canonical_bank_sign_matches_attitude_observation() -> None:
    neutral = _state(heading_deg=90, pitch_deg=0, bank_deg=0)
    neutral_tilt = aircraft_card_wing_tilt_px(
        aircraft_card_pose_signature(
            neutral,
            view_preset=InstrumentAircraftViewPreset.FRONT_LEFT,
        )
    )
    for bank_deg in (-20, 20):
        state = _state(heading_deg=90, pitch_deg=0, bank_deg=bank_deg)
        observation = attitude_display_observation_from_state(state)
        signature = aircraft_card_pose_signature(
            state,
            view_preset=InstrumentAircraftViewPreset.FRONT_LEFT,
        )
        delta = aircraft_card_wing_tilt_px(signature) - neutral_tilt
        assert (delta < 0.0) is (observation.bank_deg > 0)
        assert (delta > 0.0) is (observation.bank_deg < 0)


def test_pitch_metric_moves_symmetrically_around_level_flight() -> None:
    presets = (
        InstrumentAircraftViewPreset.FRONT_LEFT,
        InstrumentAircraftViewPreset.FRONT_RIGHT,
        InstrumentAircraftViewPreset.PROFILE_LEFT,
        InstrumentAircraftViewPreset.PROFILE_RIGHT,
    )
    for preset in presets:
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


def test_canonical_pitch_sign_matches_attitude_observation() -> None:
    neutral = _state(heading_deg=90, pitch_deg=0, bank_deg=0)
    neutral_pitch = aircraft_card_pitch_cue_px(
        aircraft_card_pose_signature(
            neutral,
            view_preset=InstrumentAircraftViewPreset.FRONT_LEFT,
        )
    )
    for pitch_deg in (-10, 10):
        state = _state(heading_deg=90, pitch_deg=pitch_deg, bank_deg=0)
        observation = attitude_display_observation_from_state(state)
        signature = aircraft_card_pose_signature(
            state,
            view_preset=InstrumentAircraftViewPreset.FRONT_LEFT,
        )
        delta = aircraft_card_pitch_cue_px(signature) - neutral_pitch
        assert (delta > 0.0) is (observation.pitch_deg > 0)
        assert (delta < 0.0) is (observation.pitch_deg < 0)


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


def test_card_bank_renders_mesh_cards_and_caches_scaled_surfaces() -> None:
    bank = InstrumentAircraftCardSpriteBank()
    state = _state(heading_deg=82, pitch_deg=7, bank_deg=-14)

    pygame.init()
    try:
        for preset in InstrumentAircraftViewPreset:
            surface = bank.get_scaled_surface(state=state, view_preset=preset, size=(224, 140))
            cached = bank.get_scaled_surface(state=state, view_preset=preset, size=(224, 140))
            assert cached is surface
            bounds = _red_bounds(surface)
            assert bounds is not None
            min_x, min_y, max_x, max_y = bounds
            assert min_x >= 2, preset
            assert min_y >= 2, preset
            assert max_x <= surface.get_width() - 3, preset
            assert max_y <= surface.get_height() - 3, preset
    finally:
        pygame.quit()


def test_card_bank_rendered_views_are_distinct() -> None:
    bank = InstrumentAircraftCardSpriteBank()
    state = _state(heading_deg=82, pitch_deg=7, bank_deg=-14)

    pygame.init()
    try:
        front = bank.get_scaled_surface(
            state=state,
            view_preset=InstrumentAircraftViewPreset.FRONT_LEFT,
            size=(224, 140),
        )
        top = bank.get_scaled_surface(
            state=state,
            view_preset=InstrumentAircraftViewPreset.TOP_DOWN,
            size=(224, 140),
        )

        assert pygame.image.tobytes(front, "RGBA") != pygame.image.tobytes(top, "RGBA")
    finally:
        pygame.quit()
