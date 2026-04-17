from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from cfast_trainer.aircraft_art import panda3d_fixed_wing_hpr_from_world_tangent
from cfast_trainer.spatial_integration_panda3d import SpatialIntegrationPanda3DRenderer


@dataclass
class _FakeNode:
    pos: tuple[float, float, float] | None = None
    hpr: tuple[float, float, float] | None = None
    scale: float | None = None
    parent: object | None = None
    color_scale: tuple[float, float, float, float] | None = None
    visible: bool = True
    removed: bool = False

    def setPos(self, x: float, y: float, z: float) -> None:
        self.pos = (float(x), float(y), float(z))

    def setHpr(self, h: float, p: float, r: float) -> None:
        self.hpr = (float(h), float(p), float(r))

    def setScale(self, scale: float) -> None:
        self.scale = float(scale)

    def reparentTo(self, parent: object) -> None:
        self.parent = parent

    def setColorScale(self, r: float, g: float, b: float, a: float) -> None:
        self.color_scale = (float(r), float(g), float(b), float(a))

    def show(self) -> None:
        self.visible = True

    def hide(self) -> None:
        self.visible = False

    def removeNode(self) -> None:
        self.removed = True


@dataclass
class _FakeEntry:
    fallback: str = "box"
    transformed: list[tuple[tuple[float, float, float], tuple[float, float, float], float]] | None = None

    def apply_loaded_model_transform(
        self,
        node,
        *,
        pos: tuple[float, float, float],
        hpr: tuple[float, float, float] = (0.0, 0.0, 0.0),
        scale: float = 1.0,
    ) -> None:
        node.setPos(*pos)
        node.setHpr(*hpr)
        node.setScale(scale)
        if self.transformed is not None:
            self.transformed.append((pos, hpr, float(scale)))


class _FakeCatalog:
    def __init__(self, *, entry: _FakeEntry | None, resolved: Path | None) -> None:
        self._entry = entry
        self._resolved = resolved

    def entry(self, asset_id: str) -> _FakeEntry | None:
        return self._entry

    def resolve_path(self, asset_id: str) -> Path | None:
        return self._resolved


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


def test_load_asset_or_fallback_records_loaded_asset_ids_and_applies_transform() -> None:
    renderer = object.__new__(SpatialIntegrationPanda3DRenderer)
    entry = _FakeEntry(fallback="hangar", transformed=[])
    renderer._catalog = _FakeCatalog(entry=entry, resolved=Path("/tmp/hangar.obj"))
    renderer._loaded_asset_ids = set()
    renderer._fallback_asset_ids = set()
    renderer._load_model = lambda path: _FakeNode()  # type: ignore[method-assign]
    renderer._build_fallback_model = lambda *, kind, color: (_ for _ in ()).throw(AssertionError("fallback should not be used"))  # type: ignore[method-assign]

    parent = _FakeNode()
    node = SpatialIntegrationPanda3DRenderer._load_asset_or_fallback(
        renderer,
        asset_id="building_hangar",
        fallback="hangar",
        color=(1.0, 1.0, 1.0, 1.0),
        scale=0.9,
        parent=parent,
        pos=(1.0, 2.0, 3.0),
        hpr=(90.0, 0.0, 0.0),
    )

    assert isinstance(node, _FakeNode)
    assert node.parent is parent
    assert renderer.loaded_asset_ids() == ("building_hangar",)
    assert renderer.fallback_asset_ids() == ()
    assert entry.transformed == [((1.0, 2.0, 3.0), (90.0, 0.0, 0.0), 0.9)]


def test_load_asset_or_fallback_records_fallback_asset_ids_when_missing() -> None:
    renderer = object.__new__(SpatialIntegrationPanda3DRenderer)
    renderer._catalog = _FakeCatalog(entry=_FakeEntry(fallback="box"), resolved=None)
    renderer._loaded_asset_ids = set()
    renderer._fallback_asset_ids = set()
    renderer._load_model = lambda path: _FakeNode()  # type: ignore[method-assign]
    fallback_calls: list[str] = []
    renderer._build_fallback_model = lambda *, kind, color: (fallback_calls.append(str(kind)) or _FakeNode())  # type: ignore[method-assign]

    parent = _FakeNode()
    node = SpatialIntegrationPanda3DRenderer._load_asset_or_fallback(
        renderer,
        asset_id="spatial_tent_canvas",
        fallback="spatial_tent",
        color=(1.0, 1.0, 1.0, 1.0),
        scale=0.8,
        parent=parent,
    )

    assert isinstance(node, _FakeNode)
    assert node.parent is parent
    assert renderer.loaded_asset_ids() == ()
    assert renderer.fallback_asset_ids() == ("spatial_tent_canvas",)
    assert fallback_calls == ["spatial_tent"]


def test_update_landmarks_applies_query_highlight_to_real_landmark_nodes() -> None:
    renderer = object.__new__(SpatialIntegrationPanda3DRenderer)
    renderer._landmark_nodes = []
    renderer._landmark_root = _FakeNode()

    def fake_grid_to_world(*, x: int, y: int, z: int, **_: object) -> tuple[float, float, float, float]:
        return (float(x * 10), float(y * 20), float(z * 5), 1.5)

    renderer._grid_to_world = fake_grid_to_world  # type: ignore[method-assign]
    first = _FakeNode()
    second = _FakeNode()
    renderer._ensure_landmark_node = lambda *, idx, label, kind: (first if idx == 0 else second)  # type: ignore[method-assign]

    SpatialIntegrationPanda3DRenderer._update_landmarks(
        renderer,
        landmarks=(
            ("BLD1", "building", 1, 2),
            ("TRK1", "truck", 3, 4),
        ),
        query_label="TRK1",
        grid_cols=6,
        grid_rows=6,
        alt_levels=4,
    )

    assert first.pos == (10.0, 40.0, 1.66)
    assert second.pos == (30.0, 80.0, 1.66)
    assert first.color_scale == (1.0, 1.0, 1.0, 1.0)
    assert second.color_scale == pytest.approx((1.08, 1.02, 0.74, 1.0))
