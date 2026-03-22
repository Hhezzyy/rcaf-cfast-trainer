from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from .cognitive_core import clamp01

if TYPE_CHECKING:
    from .instrument_comprehension import (
        InstrumentComprehensionTrialKind,
        InstrumentHeadingDisplayMode,
        InstrumentState,
    )


INSTRUMENT_COMMON_MISREAD_TAGS: tuple[str, ...] = (
    "misread_other_heading_mode",
    "misread_north_up",
)

INSTRUMENT_DISTRACTOR_FALLBACKS: tuple[str, ...] = (
    "reciprocal_heading",
    "quarter_turn_left",
    "quarter_turn_right",
    "bank_flip",
    "pitch_flip",
    "attitude_mirror",
)

_SPEED_RANGE = (120, 360)
_ALTITUDE_RANGE = (1000, 9500)
_VERTICAL_RATE_RANGE = (-2500, 2500)
_BANK_RANGE = (-45, 45)
_PITCH_RANGE = (-20, 20)


@dataclass(frozen=True, slots=True)
class InstrumentHeadingDisplayObservation:
    mode: InstrumentHeadingDisplayMode
    heading_deg: int
    rose_rotation_deg: int
    arrow_heading_deg: int
    north_up_heading_deg: int


@dataclass(frozen=True, slots=True)
class InstrumentAttitudeDisplayObservation:
    bank_deg: int
    pitch_deg: int
    horizon_rotation_deg: float
    horizon_offset_norm: float


@dataclass(frozen=True, slots=True)
class InstrumentDisplayObservation:
    heading: InstrumentHeadingDisplayObservation
    attitude: InstrumentAttitudeDisplayObservation
    speed_kts: int
    airspeed_turn: float
    altitude_ft: int
    altimeter_thousands_turn: float
    altimeter_hundreds_turn: float
    vertical_rate_fpm: int
    slip: int


def normalize_heading_deg(deg: int) -> int:
    return int(deg) % 360


def heading_error(a: int, b: int) -> int:
    aa = normalize_heading_deg(a)
    bb = normalize_heading_deg(b)
    diff = abs(aa - bb)
    return min(diff, 360 - diff)


def heading_cardinal_8(heading_deg: int) -> str:
    h = float(normalize_heading_deg(heading_deg))
    idx = int(((h + 22.5) % 360) // 45)
    labels = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
    return labels[idx]


def turn_phrase(bank_deg: int) -> str:
    bank = int(bank_deg)
    if abs(bank) <= 4:
        return "maintaining direction"
    return "turning left" if bank < 0 else "turning right"


def altitude_phrase(*, altitude_ft: int, vertical_rate_fpm: int) -> str:
    altitude = int(altitude_ft)
    vertical_rate = int(vertical_rate_fpm)
    if abs(vertical_rate) < 100:
        return f"maintaining height at {altitude} feet"
    if vertical_rate > 0:
        return f"climbing through {altitude} feet"
    return f"descending through {altitude} feet"


def normalize_instrument_state(state: InstrumentState) -> InstrumentState:
    return replace(
        state,
        speed_kts=_clamp(int(state.speed_kts), *_SPEED_RANGE),
        altitude_ft=_clamp(int(state.altitude_ft), *_ALTITUDE_RANGE),
        vertical_rate_fpm=_clamp(int(state.vertical_rate_fpm), *_VERTICAL_RATE_RANGE),
        bank_deg=_clamp(int(round(state.bank_deg)), *_BANK_RANGE),
        pitch_deg=_clamp(int(round(state.pitch_deg)), *_PITCH_RANGE),
        heading_deg=normalize_heading_deg(int(state.heading_deg)),
        slip=_clamp(int(state.slip), -1, 1),
    )


def heading_display_observation_from_heading(
    heading_deg: int,
    heading_display_mode: InstrumentHeadingDisplayMode,
) -> InstrumentHeadingDisplayObservation:
    heading = normalize_heading_deg(int(heading_deg))
    if heading_display_mode.value == "rotating_rose":
        return InstrumentHeadingDisplayObservation(
            mode=heading_display_mode,
            heading_deg=heading,
            rose_rotation_deg=heading,
            arrow_heading_deg=0,
            north_up_heading_deg=0,
        )
    return InstrumentHeadingDisplayObservation(
        mode=heading_display_mode,
        heading_deg=heading,
        rose_rotation_deg=0,
        arrow_heading_deg=heading,
        north_up_heading_deg=0,
    )


def heading_display_observation_from_state(
    state: InstrumentState,
    heading_display_mode: InstrumentHeadingDisplayMode,
) -> InstrumentHeadingDisplayObservation:
    normalized = normalize_instrument_state(state)
    return heading_display_observation_from_heading(
        int(normalized.heading_deg),
        heading_display_mode,
    )


def attitude_display_observation_from_bank_pitch(
    bank_deg: int,
    pitch_deg: int,
) -> InstrumentAttitudeDisplayObservation:
    bank = _clamp(int(round(bank_deg)), *_BANK_RANGE)
    pitch = _clamp(int(round(pitch_deg)), *_PITCH_RANGE)
    return InstrumentAttitudeDisplayObservation(
        bank_deg=bank,
        pitch_deg=pitch,
        horizon_rotation_deg=float(bank),
        horizon_offset_norm=float(pitch) / 20.0,
    )


def attitude_display_observation_from_state(
    state: InstrumentState,
) -> InstrumentAttitudeDisplayObservation:
    normalized = normalize_instrument_state(state)
    return attitude_display_observation_from_bank_pitch(
        int(normalized.bank_deg),
        int(normalized.pitch_deg),
    )


def display_observation_from_state(
    state: InstrumentState,
    heading_display_mode: InstrumentHeadingDisplayMode,
) -> InstrumentDisplayObservation:
    normalized = normalize_instrument_state(state)
    thousands_turn, hundreds_turn = _altimeter_hand_turns(int(normalized.altitude_ft))
    return InstrumentDisplayObservation(
        heading=heading_display_observation_from_state(normalized, heading_display_mode),
        attitude=attitude_display_observation_from_state(normalized),
        speed_kts=int(normalized.speed_kts),
        airspeed_turn=_airspeed_turn(int(normalized.speed_kts)),
        altitude_ft=int(normalized.altitude_ft),
        altimeter_thousands_turn=thousands_turn,
        altimeter_hundreds_turn=hundreds_turn,
        vertical_rate_fpm=int(normalized.vertical_rate_fpm),
        slip=int(normalized.slip),
    )


def interpreted_heading_from_display(
    observation: InstrumentHeadingDisplayObservation,
    *,
    assumed_mode: InstrumentHeadingDisplayMode,
) -> int:
    if assumed_mode.value == "rotating_rose":
        return normalize_heading_deg(int(round(observation.rose_rotation_deg)))
    return normalize_heading_deg(int(round(observation.arrow_heading_deg)))


def north_up_heading_from_display(observation: InstrumentHeadingDisplayObservation) -> int:
    return normalize_heading_deg(int(round(observation.north_up_heading_deg)))


def describe_instrument_state(state: InstrumentState) -> str:
    state = normalize_instrument_state(state)
    altitude_text = altitude_phrase(
        altitude_ft=int(state.altitude_ft),
        vertical_rate_fpm=int(state.vertical_rate_fpm),
    )
    return (
        f"Flying at {int(state.speed_kts)} kt, "
        f"{turn_phrase(int(state.bank_deg))}, "
        f"heading {heading_cardinal_8(int(state.heading_deg))}, "
        f"{altitude_text}."
    )


def solve_instrument_interpretation(
    *,
    prompt_state: InstrumentState,
    kind: InstrumentComprehensionTrialKind,
    heading_display_mode: InstrumentHeadingDisplayMode,
) -> InstrumentState:
    _ = kind
    observation = display_observation_from_state(prompt_state, heading_display_mode)
    return normalize_instrument_state(
        replace(
            normalize_instrument_state(prompt_state),
            speed_kts=int(observation.speed_kts),
            altitude_ft=int(observation.altitude_ft),
            vertical_rate_fpm=int(observation.vertical_rate_fpm),
            bank_deg=int(observation.attitude.bank_deg),
            pitch_deg=int(observation.attitude.pitch_deg),
            heading_deg=int(observation.heading.heading_deg),
            slip=int(observation.slip),
        )
    )


def display_match_error(
    prompt_observation: InstrumentDisplayObservation,
    candidate_state: InstrumentState,
    *,
    heading_display_mode: InstrumentHeadingDisplayMode,
    kind: InstrumentComprehensionTrialKind,
) -> int:
    candidate_observation = display_observation_from_state(candidate_state, heading_display_mode)
    heading_err = heading_error(
        int(prompt_observation.heading.heading_deg),
        int(candidate_observation.heading.heading_deg),
    )
    bank_err = abs(
        int(prompt_observation.attitude.bank_deg) - int(candidate_observation.attitude.bank_deg)
    )
    pitch_err = abs(
        int(prompt_observation.attitude.pitch_deg) - int(candidate_observation.attitude.pitch_deg)
    )

    if kind.value in {"instruments_to_aircraft", "aircraft_to_instruments"}:
        return int(heading_err + (bank_err * 4) + (pitch_err * 5))

    speed_err = abs(int(prompt_observation.speed_kts) - int(candidate_observation.speed_kts)) // 5
    alt_err = abs(int(prompt_observation.altitude_ft) - int(candidate_observation.altitude_ft)) // 100
    vs_err = abs(
        int(prompt_observation.vertical_rate_fpm) - int(candidate_observation.vertical_rate_fpm)
    ) // 100
    return int(speed_err + alt_err + vs_err + (bank_err * 2) + (pitch_err * 3) + heading_err)


def interpretation_error(
    true: InstrumentState,
    other: InstrumentState,
    *,
    kind: InstrumentComprehensionTrialKind,
    heading_display_mode: InstrumentHeadingDisplayMode,
) -> int:
    prompt_observation = display_observation_from_state(true, heading_display_mode)
    return display_match_error(
        prompt_observation,
        other,
        heading_display_mode=heading_display_mode,
        kind=kind,
    )


def apply_distractor_profile(
    state: InstrumentState,
    *,
    profile_tag: str,
    kind: InstrumentComprehensionTrialKind,
    heading_display_mode: InstrumentHeadingDisplayMode,
    difficulty: float,
) -> InstrumentState:
    state = normalize_instrument_state(state)
    d = clamp01(float(difficulty))
    orientation_only = kind.value in {"instruments_to_aircraft", "aircraft_to_instruments"}
    heading_step = 90 if d < 0.75 else 45
    speed_step = _quantized_delta(d, easy=45, hard=25, step=5)
    altitude_step = _quantized_delta(d, easy=1800, hard=800, step=100)
    vertical_step = _quantized_delta(d, easy=1200, hard=600, step=100)
    heading_observation = heading_display_observation_from_state(state, heading_display_mode)

    if profile_tag == "misread_other_heading_mode":
        other_mode = (
            "moving_arrow"
            if heading_display_mode.value == "rotating_rose"
            else "rotating_rose"
        )
        assumed_mode = type(heading_display_mode)(other_mode)
        return _with_heading(
            state,
            interpreted_heading_from_display(
                heading_observation,
                assumed_mode=assumed_mode,
            ),
        )
    if profile_tag == "misread_north_up":
        return _with_heading(state, north_up_heading_from_display(heading_observation))
    if profile_tag == "reciprocal_heading":
        return _with_heading(state, int(state.heading_deg) + 180)
    if profile_tag == "quarter_turn_left":
        return _with_heading(state, int(state.heading_deg) - heading_step)
    if profile_tag == "quarter_turn_right":
        return _with_heading(state, int(state.heading_deg) + heading_step)
    if profile_tag == "bank_flip":
        return _with_bank(state, -int(state.bank_deg))
    if profile_tag == "pitch_flip":
        return _with_pitch(state, -int(state.pitch_deg))
    if profile_tag == "attitude_mirror":
        return _with_pitch(_with_bank(state, -int(state.bank_deg)), -int(state.pitch_deg))
    if orientation_only:
        return state
    if profile_tag == "speed_shift_up":
        return normalize_instrument_state(
            replace(state, speed_kts=_clamp(int(state.speed_kts) + speed_step, *_SPEED_RANGE))
        )
    if profile_tag == "speed_shift_down":
        return normalize_instrument_state(
            replace(state, speed_kts=_clamp(int(state.speed_kts) - speed_step, *_SPEED_RANGE))
        )
    if profile_tag == "altitude_shift_up":
        return normalize_instrument_state(
            replace(
                state,
                altitude_ft=_clamp(int(state.altitude_ft) + altitude_step, *_ALTITUDE_RANGE),
            )
        )
    if profile_tag == "altitude_shift_down":
        return normalize_instrument_state(
            replace(
                state,
                altitude_ft=_clamp(int(state.altitude_ft) - altitude_step, *_ALTITUDE_RANGE),
            )
        )
    if profile_tag == "vertical_rate_flip":
        flipped = -int(state.vertical_rate_fpm)
        if flipped == 0:
            flipped = vertical_step if int(state.pitch_deg) >= 0 else -vertical_step
        return normalize_instrument_state(
            replace(
                state,
                vertical_rate_fpm=_clamp(flipped, *_VERTICAL_RATE_RANGE),
                pitch_deg=_pitch_for_vertical_rate(flipped, fallback=int(state.pitch_deg)),
            )
        )
    raise ValueError(f"Unsupported instrument distractor profile: {profile_tag}")


def lower_band_profile_pool(
    *,
    kind: InstrumentComprehensionTrialKind,
    difficulty: float,
) -> tuple[str, ...]:
    band = difficulty_band(difficulty)
    ladders = _profile_bands_for_kind(kind)
    source_band = max(0, band - 1)
    return ladders[source_band]


def nearest_profile_candidates(
    *,
    kind: InstrumentComprehensionTrialKind,
    difficulty: float,
) -> tuple[str, ...]:
    ladders = _profile_bands_for_kind(kind)
    band = difficulty_band(difficulty)
    ordered: list[str] = []
    for idx in range(band, -1, -1):
        for tag in ladders[idx]:
            if tag not in ordered:
                ordered.append(tag)
    for tag in ladders[-1]:
        if tag not in ordered:
            ordered.append(tag)
    for tag in INSTRUMENT_DISTRACTOR_FALLBACKS:
        if tag not in ordered:
            ordered.append(tag)
    return tuple(ordered)


def difficulty_band(difficulty: float) -> int:
    d = clamp01(float(difficulty))
    if d < 0.34:
        return 0
    if d < 0.67:
        return 1
    return 2


def _profile_bands_for_kind(kind: InstrumentComprehensionTrialKind) -> tuple[tuple[str, ...], ...]:
    if kind.value in {"instruments_to_aircraft", "aircraft_to_instruments"}:
        return (
            ("quarter_turn_left", "quarter_turn_right", "bank_flip"),
            ("quarter_turn_left", "quarter_turn_right", "bank_flip", "pitch_flip", "reciprocal_heading"),
            ("bank_flip", "pitch_flip", "attitude_mirror", "reciprocal_heading", "quarter_turn_left", "quarter_turn_right"),
        )
    return (
        ("speed_shift_up", "altitude_shift_up", "quarter_turn_left", "bank_flip"),
        ("speed_shift_up", "altitude_shift_up", "quarter_turn_left", "quarter_turn_right", "bank_flip", "pitch_flip"),
        ("vertical_rate_flip", "reciprocal_heading", "pitch_flip", "bank_flip", "altitude_shift_down", "speed_shift_down"),
    )


def _airspeed_turn(speed_kts: int) -> float:
    speed = max(0, min(360, int(speed_kts)))
    return speed / 360.0


def _altimeter_hand_turns(altitude_ft: int) -> tuple[float, float]:
    altitude = max(0, int(altitude_ft))
    thousands_turn = (altitude % 10000) / 10000.0
    hundreds_turn = (altitude % 1000) / 1000.0
    return thousands_turn, hundreds_turn


def _with_heading(state: InstrumentState, heading_deg: int) -> InstrumentState:
    return normalize_instrument_state(replace(state, heading_deg=normalize_heading_deg(heading_deg)))


def _with_bank(state: InstrumentState, bank_deg: int) -> InstrumentState:
    return normalize_instrument_state(replace(state, bank_deg=_clamp(int(bank_deg), *_BANK_RANGE)))


def _with_pitch(state: InstrumentState, pitch_deg: int) -> InstrumentState:
    pitch = _clamp(int(pitch_deg), *_PITCH_RANGE)
    return normalize_instrument_state(
        replace(
            state,
            pitch_deg=pitch,
            vertical_rate_fpm=_coherent_vertical_rate(
                pitch_deg=pitch,
                current_vertical_rate=int(state.vertical_rate_fpm),
            ),
        )
    )


def _coherent_vertical_rate(*, pitch_deg: int, current_vertical_rate: int) -> int:
    pitch = int(pitch_deg)
    if pitch == 0:
        return 0
    magnitude = abs(int(current_vertical_rate))
    if magnitude < 100:
        magnitude = 700
    magnitude = _clamp((magnitude // 100) * 100, 100, _VERTICAL_RATE_RANGE[1])
    return magnitude if pitch > 0 else -magnitude


def _pitch_for_vertical_rate(vertical_rate_fpm: int, *, fallback: int) -> int:
    vertical_rate = int(vertical_rate_fpm)
    if abs(vertical_rate) < 100:
        return 0
    if vertical_rate > 0:
        return max(2, int(abs(fallback)) or 4)
    return -max(2, int(abs(fallback)) or 4)


def _quantized_delta(difficulty: float, *, easy: int, hard: int, step: int) -> int:
    raw = int(round(easy + ((hard - easy) * clamp01(float(difficulty)))))
    quantized = max(step, (raw // step) * step)
    return int(quantized)


def _clamp(value: int, lo: int, hi: int) -> int:
    return int(lo if value < lo else hi if value > hi else value)


def _sign(value: int) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0
