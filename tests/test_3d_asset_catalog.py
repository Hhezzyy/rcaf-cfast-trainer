from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from cfast_trainer.panda3d_assets import Panda3DAssetCatalog, default_asset_root


@dataclass
class _FakeNode:
    pos: tuple[float, float, float] | None = None
    hpr: tuple[float, float, float] | None = None
    scale: float | None = None

    def setPos(self, x: float, y: float, z: float) -> None:
        self.pos = (float(x), float(y), float(z))

    def setHpr(self, h: float, p: float, r: float) -> None:
        self.hpr = (float(h), float(p), float(r))

    def setScale(self, scale: float) -> None:
        self.scale = float(scale)


def test_catalog_resolves_existing_asset_and_applies_offsets(tmp_path) -> None:
    asset_root = tmp_path / "assets"
    model_dir = asset_root / "aircraft"
    model_dir.mkdir(parents=True)
    model_path = model_dir / "plane_red.obj"
    model_path.write_text("# placeholder\n", encoding="utf-8")

    manifest = {
        "assets": {
            "plane_red": {
                "category": "aircraft",
                "fallback": "plane",
                "scale": 1.0,
                "hpr_offset_deg": [180.0, 0.0, 0.0],
                "pos_offset": [1.5, 0.0, 2.0],
                "candidates": ["aircraft/plane_red.obj"],
            }
        }
    }
    (asset_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    catalog = Panda3DAssetCatalog(asset_root=asset_root)

    assert catalog.asset_ids() == ("plane_red",)
    assert catalog.resolve_path("plane_red") == model_path
    assert catalog.available_asset_ids() == ("plane_red",)
    entry = catalog.entry("plane_red")
    assert entry is not None
    assert entry.hpr_offset_deg == (180.0, 0.0, 0.0)
    assert entry.pos_offset == (1.5, 0.0, 2.0)

    node = _FakeNode()
    entry.apply_loaded_model_transform(
        node,
        pos=(10.0, 20.0, 30.0),
        hpr=(45.0, 5.0, -2.0),
        scale=1.25,
    )
    assert node.pos == (11.5, 20.0, 32.0)
    assert node.hpr == (225.0, 5.0, -2.0)
    assert node.scale == 1.25


def test_catalog_handles_missing_manifest(tmp_path) -> None:
    catalog = Panda3DAssetCatalog(asset_root=tmp_path / "missing")

    assert catalog.asset_ids() == ()
    assert catalog.resolve_path("plane_red") is None


def test_repo_manifest_includes_auditory_assets() -> None:
    catalog = Panda3DAssetCatalog(asset_root=default_asset_root())

    expected = {
        "auditory_ball",
        "auditory_tunnel_rib",
        "auditory_tunnel_segment",
    }

    assert expected.issubset(set(catalog.asset_ids()))
    for asset_id in expected:
        resolved = catalog.resolve_path(asset_id)
        assert isinstance(resolved, Path)
        assert resolved is not None
        assert resolved.exists()


def test_repo_manifest_includes_rapid_tracking_foliage_assets() -> None:
    catalog = Panda3DAssetCatalog(asset_root=default_asset_root())

    expected = {
        "shrubs_low_cluster",
        "trees_field_cluster",
        "forest_canopy_patch",
    }

    assert expected.issubset(set(catalog.asset_ids()))
    for asset_id in expected:
        resolved = catalog.resolve_path(asset_id)
        assert isinstance(resolved, Path)
        assert resolved is not None
        assert resolved.exists()
