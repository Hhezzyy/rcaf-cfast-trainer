from __future__ import annotations

from dataclasses import dataclass

import pytest

from cfast_trainer.aircraft_art import panda3d_fixed_wing_hpr_from_world_tangent
from cfast_trainer.spatial_integration_panda3d import SpatialIntegrationPanda3DRenderer


@dataclass
class _FakeNode:
    pos: tuple[float, float, float] | None = None
    hpr: tuple[float, float, float] | None = None
    scale: float | None = None
    visible: bool = True

    def setPos(self, x: float, y: float, z: float) -> None:
        self.pos = (float(x), float(y), float(z))

    def setHpr(self, h: float, p: float, r: float) -> None:
        self.hpr = (float(h), float(p), float(r))

    def setScale(self, scale: float) -> None:
        self.scale = float(scale)

    def show(self) -> None:
        self.visible = True

    def hide(self) -> None:
        self.visible = False


def test_update_aircraft_uses_static_scene_positions() -> None:
    renderer = object.__new__(SpatialIntegrationPanda3DRenderer)
    renderer._aircraft = _FakeNode()
    renderer._aircraft_prev = _FakeNode()
    renderer._aircraft_pred = _FakeNode()
    renderer._aircraft_hpr = (0.0, 0.0, 0.0)

    def fake_grid_to_world(*, x: int, y: int, z: int, **_: object) -> tuple[float, float, float, float]:
        return (float(x * 10), float(y * 20), float(z * 5), 0.0)

    renderer._grid_to_world = fake_grid_to_world  # type: ignore[method-assign]

    focus_a = SpatialIntegrationPanda3DRenderer._update_aircraft(
        renderer,
        now_point=(2, 3, 1),
        prev_point=(1, 2, 1),
        velocity=(1, 1, 0),
        show_motion=True,
        grid_cols=6,
        grid_rows=6,
        alt_levels=4,
    )
    focus_b = SpatialIntegrationPanda3DRenderer._update_aircraft(
        renderer,
        now_point=(2, 3, 1),
        prev_point=(1, 2, 1),
        velocity=(1, 1, 0),
        show_motion=True,
        grid_cols=6,
        grid_rows=6,
        alt_levels=4,
    )

    assert focus_a == (20.0, 60.0, 5.0)
    assert focus_b == focus_a
    assert renderer._aircraft.pos == (20.0, 60.0, 5.0)
    assert renderer._aircraft_prev.pos == (10.0, 40.0, 5.0)
    assert renderer._aircraft_pred.pos == (20.6, 60.6, 5.0)
    assert renderer._aircraft.hpr == pytest.approx(
        panda3d_fixed_wing_hpr_from_world_tangent(
            tangent=(10.0, 20.0, 0.0),
            roll_deg=((10.0 / (500.0 ** 0.5)) * 26.0),
        )
    )
    assert renderer._aircraft.scale == 1.24


def test_update_clouds_keeps_static_positions() -> None:
    renderer = object.__new__(SpatialIntegrationPanda3DRenderer)
    renderer._clouds = [_FakeNode(), _FakeNode()]
    renderer._cloud_positions = ((-80.0, 34.0, 66.0), (-42.0, 52.0, 69.0))

    SpatialIntegrationPanda3DRenderer._update_clouds(renderer)
    first_pass = [cloud.pos for cloud in renderer._clouds]

    SpatialIntegrationPanda3DRenderer._update_clouds(renderer)
    second_pass = [cloud.pos for cloud in renderer._clouds]

    assert first_pass == [(-80.0, 34.0, 66.0), (-42.0, 52.0, 69.0)]
    assert second_pass == first_pass


def test_update_aircraft_uses_neutral_hpr_when_motion_is_zero() -> None:
    renderer = object.__new__(SpatialIntegrationPanda3DRenderer)
    renderer._aircraft = _FakeNode()
    renderer._aircraft_prev = _FakeNode()
    renderer._aircraft_pred = _FakeNode()
    renderer._aircraft_hpr = (12.0, -6.0, 8.0)

    def fake_grid_to_world(*, x: int, y: int, z: int, **_: object) -> tuple[float, float, float, float]:
        return (float(x * 10), float(y * 20), float(z * 5), 0.0)

    renderer._grid_to_world = fake_grid_to_world  # type: ignore[method-assign]

    focus = SpatialIntegrationPanda3DRenderer._update_aircraft(
        renderer,
        now_point=(2, 3, 1),
        prev_point=(2, 3, 1),
        velocity=(0, 0, 0),
        show_motion=False,
        grid_cols=6,
        grid_rows=6,
        alt_levels=4,
    )

    assert focus == (20.0, 60.0, 5.0)
    assert renderer._aircraft.hpr == (0.0, 0.0, 0.0)
    assert renderer._aircraft_prev.visible is False
    assert renderer._aircraft_pred.visible is False
