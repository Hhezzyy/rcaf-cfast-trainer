from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


def _float3(value: object, *, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if isinstance(value, list | tuple) and len(value) >= 3:
        return (float(value[0]), float(value[1]), float(value[2]))
    return default


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_asset_root() -> Path:
    return _repo_root() / "assets" / "panda3d"


@dataclass(frozen=True, slots=True)
class Panda3DAssetEntry:
    asset_id: str
    category: str
    fallback: str
    scale: float
    candidates: tuple[str, ...]
    hpr_offset_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def resolved_path(self, *, asset_root: Path) -> Path | None:
        for rel in self.candidates:
            path = asset_root / rel
            if path.exists():
                return path
        return None

    def apply_loaded_model_transform(
        self,
        node,
        *,
        pos: tuple[float, float, float],
        hpr: tuple[float, float, float] = (0.0, 0.0, 0.0),
        scale: float = 1.0,
    ) -> None:
        node.setPos(
            float(pos[0]) + self.pos_offset[0],
            float(pos[1]) + self.pos_offset[1],
            float(pos[2]) + self.pos_offset[2],
        )
        node.setHpr(
            float(hpr[0]) + self.hpr_offset_deg[0],
            float(hpr[1]) + self.hpr_offset_deg[1],
            float(hpr[2]) + self.hpr_offset_deg[2],
        )
        node.setScale(float(scale) * self.scale)


class Panda3DAssetCatalog:
    def __init__(
        self,
        *,
        asset_root: Path | None = None,
        manifest_path: Path | None = None,
    ) -> None:
        self.asset_root = (asset_root or default_asset_root()).resolve()
        self.manifest_path = manifest_path or (self.asset_root / "manifest.json")
        self._entries = self._load_manifest()

    def _load_manifest(self) -> dict[str, Panda3DAssetEntry]:
        if not self.manifest_path.exists():
            return {}
        data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        assets = dict(data.get("assets", {}))
        entries: dict[str, Panda3DAssetEntry] = {}
        for asset_id, raw in assets.items():
            item = dict(raw)
            entries[str(asset_id)] = Panda3DAssetEntry(
                asset_id=str(asset_id),
                category=str(item.get("category", "misc")),
                fallback=str(item.get("fallback", "box")),
                scale=float(item.get("scale", 1.0)),
                candidates=tuple(str(v) for v in item.get("candidates", ())),
                hpr_offset_deg=_float3(
                    item.get("hpr_offset_deg"),
                    default=(0.0, 0.0, 0.0),
                ),
                pos_offset=_float3(
                    item.get("pos_offset"),
                    default=(0.0, 0.0, 0.0),
                ),
            )
        return entries

    def asset_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))

    def entry(self, asset_id: str) -> Panda3DAssetEntry | None:
        return self._entries.get(str(asset_id))

    def resolve_path(self, asset_id: str) -> Path | None:
        entry = self.entry(asset_id)
        if entry is None:
            return None
        return entry.resolved_path(asset_root=self.asset_root)

    def available_asset_ids(self) -> tuple[str, ...]:
        available = [
            asset_id
            for asset_id in self.asset_ids()
            if self.resolve_path(asset_id) is not None
        ]
        return tuple(available)
