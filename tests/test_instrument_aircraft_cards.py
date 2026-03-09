from __future__ import annotations

import pygame

from cfast_trainer.instrument_aircraft_cards import (
    InstrumentAircraftCardKey,
    InstrumentAircraftCardSpriteBank,
)
from cfast_trainer.instrument_comprehension import InstrumentState


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
        _state(heading_deg=721, pitch_deg=29, bank_deg=-63)
    )

    assert key.heading_deg == 1
    assert key.pitch_deg == 20
    assert key.bank_deg == -45
    assert key.filename() == "v2_h001_pp20_bm45.png"


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
