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
    return _repo_root() / "assets" / "render"


@dataclass(frozen=True, slots=True)
class RenderAssetEntry:
    asset_id: str
    category: str
    scale: float
    candidates: tuple[str, ...]
    builtin_kind: str | None = None
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


@dataclass(frozen=True, slots=True)
class RenderAssetSource:
    asset_id: str
    entry: RenderAssetEntry
    path: Path | None
    builtin_kind: str | None

    @property
    def is_builtin(self) -> bool:
        return self.builtin_kind is not None


@dataclass(frozen=True, slots=True)
class RenderAssetResolutionError(RuntimeError):
    asset_ids: tuple[str, ...]
    details: tuple[str, ...]

    def __str__(self) -> str:
        if not self.asset_ids:
            return "Required render assets are missing."
        joined = "; ".join(self.details)
        return f"Missing required render assets: {', '.join(self.asset_ids)}. {joined}".strip()


class RenderAssetCatalog:
    def __init__(
        self,
        *,
        asset_root: Path | None = None,
        manifest_path: Path | None = None,
    ) -> None:
        self.asset_root = (asset_root or default_asset_root()).resolve()
        self.manifest_path = manifest_path or (self.asset_root / "manifest.json")
        self._entries = self._load_manifest()

    def _load_manifest(self) -> dict[str, RenderAssetEntry]:
        if not self.manifest_path.exists():
            return {}
        data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        assets = dict(data.get("assets", {}))
        entries: dict[str, RenderAssetEntry] = {}
        for asset_id, raw in assets.items():
            item = dict(raw)
            builtin = item.get("builtin_kind")
            builtin_kind = None if builtin in (None, "") else str(builtin)
            entries[str(asset_id)] = RenderAssetEntry(
                asset_id=str(asset_id),
                category=str(item.get("category", "misc")),
                scale=float(item.get("scale", 1.0)),
                candidates=tuple(str(v) for v in item.get("candidates", ())),
                builtin_kind=builtin_kind,
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

    def entry(self, asset_id: str) -> RenderAssetEntry | None:
        return self._entries.get(str(asset_id))

    def resolve_path(self, asset_id: str) -> Path | None:
        entry = self.entry(asset_id)
        if entry is None:
            return None
        return entry.resolved_path(asset_root=self.asset_root)

    def resolve(self, asset_id: str) -> RenderAssetSource | None:
        entry = self.entry(asset_id)
        if entry is None:
            return None
        path = entry.resolved_path(asset_root=self.asset_root)
        if path is not None:
            return RenderAssetSource(
                asset_id=str(asset_id),
                entry=entry,
                path=path,
                builtin_kind=None,
            )
        if entry.builtin_kind is not None:
            return RenderAssetSource(
                asset_id=str(asset_id),
                entry=entry,
                path=None,
                builtin_kind=entry.builtin_kind,
            )
        return None

    def available_asset_ids(self) -> tuple[str, ...]:
        available = [
            asset_id
            for asset_id in self.asset_ids()
            if self.resolve(asset_id) is not None
        ]
        return tuple(available)

    def require(self, asset_id: str) -> RenderAssetSource:
        resolved = self.resolve(asset_id)
        if resolved is not None:
            return resolved
        entry = self.entry(asset_id)
        if entry is None:
            raise RenderAssetResolutionError(
                asset_ids=(str(asset_id),),
                details=(f"{asset_id}: not declared in {self.manifest_path}",),
            )
        attempted = ", ".join(str(self.asset_root / rel) for rel in entry.candidates) or "no candidate paths"
        raise RenderAssetResolutionError(
            asset_ids=(str(asset_id),),
            details=(f"{asset_id}: {attempted}",),
        )

    def require_many(self, asset_ids: tuple[str, ...] | list[str] | set[str]) -> tuple[RenderAssetSource, ...]:
        resolved: list[RenderAssetSource] = []
        missing_ids: list[str] = []
        details: list[str] = []
        for asset_id in tuple(str(token) for token in asset_ids):
            source = self.resolve(asset_id)
            if source is not None:
                resolved.append(source)
                continue
            missing_ids.append(asset_id)
            entry = self.entry(asset_id)
            if entry is None:
                details.append(f"{asset_id}: not declared in {self.manifest_path}")
                continue
            attempted = ", ".join(str(self.asset_root / rel) for rel in entry.candidates) or "no candidate paths"
            details.append(f"{asset_id}: {attempted}")
        if missing_ids:
            raise RenderAssetResolutionError(
                asset_ids=tuple(missing_ids),
                details=tuple(details),
            )
        return tuple(resolved)
