from __future__ import annotations

import pytest

from cfast_trainer.aircraft_art import (
    build_fixed_wing_mesh,
    fixed_wing_heading_from_screen_heading,
    instrument_card_pygame_palette,
    panda3d_fixed_wing_hpr_from_world_hpr,
    panda3d_fixed_wing_hpr_from_world_tangent,
    project_fixed_wing_faces,
    screen_heading_deg_from_world_tangent,
)


def test_fixed_wing_mesh_contains_expected_roles_and_volume() -> None:
    faces = build_fixed_wing_mesh()
    roles = {face.role for face in faces}
    all_points = [point for face in faces for point in face.points]
    z_values = [point[2] for point in all_points]

    assert roles == {"body", "accent", "canopy", "engine"}
    assert len(faces) >= 30
    assert max(z_values) - min(z_values) >= 2.05


def test_projected_fixed_wing_faces_are_deterministic_for_same_attitude() -> None:
    first = project_fixed_wing_faces(
        heading_deg=45.0,
        pitch_deg=8.0,
        bank_deg=-18.0,
        cx=200,
        cy=160,
        scale=24.0,
    )
    second = project_fixed_wing_faces(
        heading_deg=45.0,
        pitch_deg=8.0,
        bank_deg=-18.0,
        cx=200,
        cy=160,
        scale=24.0,
    )

    assert first == second
    assert len(first) >= 20


def test_projected_fixed_wing_faces_change_with_attitude() -> None:
    level = project_fixed_wing_faces(
        heading_deg=0.0,
        pitch_deg=0.0,
        bank_deg=0.0,
        cx=220,
        cy=150,
        scale=22.0,
    )
    banked = project_fixed_wing_faces(
        heading_deg=120.0,
        pitch_deg=10.0,
        bank_deg=28.0,
        cx=220,
        cy=150,
        scale=22.0,
    )

    assert level != banked


def test_instrument_palette_keeps_distinct_aircraft_roles() -> None:
    palette = instrument_card_pygame_palette()

    assert palette.body != palette.canopy
    assert palette.body != palette.engine


def test_fixed_wing_heading_from_screen_heading_preserves_cardinal_motion() -> None:
    assert fixed_wing_heading_from_screen_heading(-90.0) == 0.0
    assert fixed_wing_heading_from_screen_heading(0.0) == 90.0
    assert fixed_wing_heading_from_screen_heading(90.0) == 180.0
    assert fixed_wing_heading_from_screen_heading(180.0) == 270.0


def test_screen_heading_from_world_tangent_matches_oblique_projection_direction() -> None:
    heading = screen_heading_deg_from_world_tangent((-1.0, 1.0, 0.0))

    assert heading is not None
    assert -180.0 < heading < -120.0


def test_screen_heading_from_world_tangent_returns_none_for_zero_motion() -> None:
    assert screen_heading_deg_from_world_tangent((0.0, 0.0, 0.0)) is None


def test_screen_heading_from_world_tangent_is_deterministic() -> None:
    first = screen_heading_deg_from_world_tangent((0.8, 2.0, -0.4))
    second = screen_heading_deg_from_world_tangent((0.8, 2.0, -0.4))

    assert first == second


def test_panda3d_fixed_wing_hpr_from_world_hpr_preserves_heading_and_roll_but_flips_pitch() -> None:
    hpr = panda3d_fixed_wing_hpr_from_world_hpr(
        heading_deg=42.0,
        pitch_deg=9.5,
        roll_deg=-17.0,
    )

    assert hpr == (42.0, -9.5, -17.0)


def test_panda3d_fixed_wing_hpr_from_world_tangent_uses_world_heading_and_climb_convention() -> None:
    east = panda3d_fixed_wing_hpr_from_world_tangent((1.0, 0.0, 0.0), roll_deg=12.0)
    climb = panda3d_fixed_wing_hpr_from_world_tangent((0.0, 1.0, 1.0), roll_deg=-8.0)

    assert east == (90.0, -0.0, 12.0)
    assert climb[0] == 0.0
    assert climb[1] < 0.0
    assert climb[2] == -8.0


def test_panda3d_fixed_wing_hpr_from_world_tangent_rejects_zero_motion() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        panda3d_fixed_wing_hpr_from_world_tangent(
            (0.0, 0.0, 0.0),
            roll_deg=22.0,
        )
