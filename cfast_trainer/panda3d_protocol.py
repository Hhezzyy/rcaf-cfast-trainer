from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Panda3DScene(StrEnum):
    AUDITORY_CAPACITY = "auditory_capacity"
    RAPID_TRACKING = "rapid_tracking"
    SPATIAL_INTEGRATION = "spatial_integration"


@dataclass(frozen=True, slots=True)
class Panda3DRequest:
    scene: Panda3DScene
    seed: int = 0
    duration_s: float = 15.0
    practice: bool = True
    asset_root: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene": self.scene.value,
            "seed": int(self.seed),
            "duration_s": float(self.duration_s),
            "practice": bool(self.practice),
            "asset_root": self.asset_root,
            "payload": dict(self.payload),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Panda3DRequest:
        return cls(
            scene=Panda3DScene(str(raw["scene"])),
            seed=int(raw.get("seed", 0)),
            duration_s=float(raw.get("duration_s", 15.0)),
            practice=bool(raw.get("practice", True)),
            asset_root=str(raw["asset_root"]) if raw.get("asset_root") is not None else None,
            payload=dict(raw.get("payload", {})),
        )


@dataclass(frozen=True, slots=True)
class Panda3DResult:
    ok: bool
    scene: Panda3DScene
    summary: str
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "scene": self.scene.value,
            "summary": str(self.summary),
            "metrics": {str(k): float(v) for k, v in self.metrics.items()},
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Panda3DResult:
        return cls(
            ok=bool(raw.get("ok", False)),
            scene=Panda3DScene(str(raw["scene"])),
            summary=str(raw.get("summary", "")),
            metrics={str(k): float(v) for k, v in dict(raw.get("metrics", {})).items()},
        )
