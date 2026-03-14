from __future__ import annotations

import pygame
import pytest

from cfast_trainer.instrument_aircraft_cards import (
    InstrumentAircraftCardKey,
    InstrumentAircraftCardSpriteBank,
    panda3d_card_rendering_available,
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
    assert key.filename() == "v19_front_left_h001_pp20_bm45.png"


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


@pytest.mark.skipif(not panda3d_card_rendering_available(), reason="Panda3D card renderer unavailable")
def test_generated_card_presets_keep_aircraft_inside_safe_inset(tmp_path) -> None:
    cache_dir = tmp_path / "generated-cards"
    state = _state(heading_deg=82, pitch_deg=7, bank_deg=-14)
    bank = InstrumentAircraftCardSpriteBank(cache_dir=cache_dir, allow_generation=True)

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
