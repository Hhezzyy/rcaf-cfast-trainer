from __future__ import annotations

import json

from cfast_trainer.panda3d_assets import Panda3DAssetCatalog


def test_catalog_resolves_existing_asset(tmp_path) -> None:
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
                "candidates": ["aircraft/plane_red.obj"],
            }
        }
    }
    (asset_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    catalog = Panda3DAssetCatalog(asset_root=asset_root)

    assert catalog.asset_ids() == ("plane_red",)
    assert catalog.resolve_path("plane_red") == model_path
    assert catalog.available_asset_ids() == ("plane_red",)


def test_catalog_handles_missing_manifest(tmp_path) -> None:
    catalog = Panda3DAssetCatalog(asset_root=tmp_path / "missing")

    assert catalog.asset_ids() == ()
    assert catalog.resolve_path("plane_red") is None
