from __future__ import annotations

from cfast_trainer.instrument_comprehension import (
    InstrumentComprehensionTrialKind,
    InstrumentHeadingDisplayMode,
    InstrumentState,
)
from cfast_trainer.instrument_orientation_solver import (
    apply_distractor_profile,
    attitude_display_observation_from_bank_pitch,
    describe_instrument_state,
    display_match_error,
    display_observation_from_state,
    heading_display_observation_from_heading,
    heading_display_observation_from_state,
    interpreted_heading_from_display,
    north_up_heading_from_display,
    solve_instrument_interpretation,
)


def _state(
    *,
    speed_kts: int = 240,
    altitude_ft: int = 5200,
    vertical_rate_fpm: int = 800,
    bank_deg: int = -14,
    pitch_deg: int = 5,
    heading_deg: int = 78,
    slip: int = 0,
) -> InstrumentState:
    return InstrumentState(
        speed_kts=speed_kts,
        altitude_ft=altitude_ft,
        vertical_rate_fpm=vertical_rate_fpm,
        bank_deg=bank_deg,
        pitch_deg=pitch_deg,
        heading_deg=heading_deg,
        slip=slip,
    )


def test_heading_display_observation_tracks_cardinals_in_both_modes() -> None:
    rotating = {
        heading: heading_display_observation_from_heading(
            heading,
            InstrumentHeadingDisplayMode.ROTATING_ROSE,
        )
        for heading in (0, 90, 180, 270)
    }
    moving = {
        heading: heading_display_observation_from_heading(
            heading,
            InstrumentHeadingDisplayMode.MOVING_ARROW,
        )
        for heading in (0, 90, 180, 270)
    }

    assert rotating[0].rose_rotation_deg == 0
    assert rotating[90].rose_rotation_deg == 90
    assert rotating[180].rose_rotation_deg == 180
    assert rotating[270].rose_rotation_deg == 270
    assert all(observation.arrow_heading_deg == 0 for observation in rotating.values())

    assert moving[0].arrow_heading_deg == 0
    assert moving[90].arrow_heading_deg == 90
    assert moving[180].arrow_heading_deg == 180
    assert moving[270].arrow_heading_deg == 270
    assert all(observation.rose_rotation_deg == 0 for observation in moving.values())


def test_attitude_display_observation_tracks_signed_pitch_and_bank() -> None:
    right_bank_climb = attitude_display_observation_from_bank_pitch(18, 7)
    left_bank_descent = attitude_display_observation_from_bank_pitch(-22, -6)

    assert right_bank_climb.horizon_rotation_deg > 0.0
    assert right_bank_climb.horizon_offset_norm > 0.0
    assert left_bank_descent.horizon_rotation_deg < 0.0
    assert left_bank_descent.horizon_offset_norm < 0.0


def test_display_observation_matches_combined_canonical_states() -> None:
    east_climb = display_observation_from_state(
        _state(bank_deg=15, pitch_deg=6, heading_deg=90),
        InstrumentHeadingDisplayMode.ROTATING_ROSE,
    )
    southwest_descent = display_observation_from_state(
        _state(bank_deg=-18, pitch_deg=-5, heading_deg=225),
        InstrumentHeadingDisplayMode.MOVING_ARROW,
    )

    assert east_climb.heading.heading_deg == 90
    assert east_climb.attitude.bank_deg == 15
    assert east_climb.attitude.pitch_deg == 6
    assert east_climb.airspeed_turn == 240 / 360.0

    assert southwest_descent.heading.heading_deg == 225
    assert southwest_descent.heading.arrow_heading_deg == 225
    assert southwest_descent.attitude.bank_deg == -18
    assert southwest_descent.attitude.pitch_deg == -5


def test_solver_round_trips_state_through_display_model() -> None:
    raw = _state(heading_deg=438, bank_deg=17, pitch_deg=-4)
    solved = solve_instrument_interpretation(
        prompt_state=raw,
        kind=InstrumentComprehensionTrialKind.INSTRUMENTS_TO_AIRCRAFT,
        heading_display_mode=InstrumentHeadingDisplayMode.MOVING_ARROW,
    )

    assert solved.heading_deg == 78
    assert solved.bank_deg == 17
    assert solved.pitch_deg == -4


def test_misread_profiles_are_derived_from_prompt_display() -> None:
    base = _state(heading_deg=78)
    heading_observation = heading_display_observation_from_state(
        base,
        InstrumentHeadingDisplayMode.ROTATING_ROSE,
    )

    other_mode = apply_distractor_profile(
        base,
        profile_tag="misread_other_heading_mode",
        kind=InstrumentComprehensionTrialKind.INSTRUMENTS_TO_AIRCRAFT,
        heading_display_mode=InstrumentHeadingDisplayMode.ROTATING_ROSE,
        difficulty=0.6,
    )
    north_up = apply_distractor_profile(
        base,
        profile_tag="misread_north_up",
        kind=InstrumentComprehensionTrialKind.INSTRUMENTS_TO_AIRCRAFT,
        heading_display_mode=InstrumentHeadingDisplayMode.ROTATING_ROSE,
        difficulty=0.6,
    )

    assert other_mode.heading_deg == interpreted_heading_from_display(
        heading_observation,
        assumed_mode=InstrumentHeadingDisplayMode.MOVING_ARROW,
    )
    assert north_up.heading_deg == north_up_heading_from_display(heading_observation)
    assert other_mode.heading_deg != (-base.heading_deg) % 360


def test_part1_display_match_error_is_orientation_grounded() -> None:
    base = _state()
    prompt_observation = display_observation_from_state(
        base,
        InstrumentHeadingDisplayMode.ROTATING_ROSE,
    )
    speed_only = apply_distractor_profile(
        base,
        profile_tag="speed_shift_up",
        kind=InstrumentComprehensionTrialKind.INSTRUMENTS_TO_DESCRIPTION,
        heading_display_mode=InstrumentHeadingDisplayMode.ROTATING_ROSE,
        difficulty=0.6,
    )
    mirrored = apply_distractor_profile(
        base,
        profile_tag="attitude_mirror",
        kind=InstrumentComprehensionTrialKind.INSTRUMENTS_TO_AIRCRAFT,
        heading_display_mode=InstrumentHeadingDisplayMode.ROTATING_ROSE,
        difficulty=0.6,
    )

    assert (
        display_match_error(
            prompt_observation,
            speed_only,
            kind=InstrumentComprehensionTrialKind.INSTRUMENTS_TO_AIRCRAFT,
            heading_display_mode=InstrumentHeadingDisplayMode.ROTATING_ROSE,
        )
        == 0
    )
    assert (
        display_match_error(
            prompt_observation,
            mirrored,
            kind=InstrumentComprehensionTrialKind.INSTRUMENTS_TO_AIRCRAFT,
            heading_display_mode=InstrumentHeadingDisplayMode.ROTATING_ROSE,
        )
        > 0
    )


def test_description_text_tracks_solver_state() -> None:
    description = describe_instrument_state(_state())

    assert "240 kt" in description
    assert "turning left" in description
    assert "heading E" in description
    assert "climbing through 5200 feet" in description
