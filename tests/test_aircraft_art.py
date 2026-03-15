from __future__ import annotations

from cfast_trainer.aircraft_art import (
    build_fixed_wing_mesh,
    fixed_wing_heading_from_screen_heading,
    instrument_card_pygame_palette,
    project_fixed_wing_faces,
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
