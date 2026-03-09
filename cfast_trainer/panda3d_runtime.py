from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

from .panda3d_assets import Panda3DAssetCatalog
from .panda3d_protocol import Panda3DRequest, Panda3DResult, Panda3DScene


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optional Panda3D runtime for true 3D test scenes."
    )
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


@dataclass(frozen=True, slots=True)
class _SceneStats:
    frames: int = 0
    loaded_assets: int = 0
    fallback_assets: int = 0


def _load_request(path: Path) -> Panda3DRequest:
    return Panda3DRequest.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _write_result(path: Path, result: Panda3DResult) -> None:
    path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")


def _configure_panda3d(*, headless: bool) -> None:
    from panda3d.core import loadPrcFileData

    loadPrcFileData("", "window-title CFAST Panda3D Runtime")
    loadPrcFileData("", "sync-video false")
    loadPrcFileData("", "audio-library-name null")
    if headless:
        loadPrcFileData("", "window-type offscreen")
    else:
        loadPrcFileData("", "win-size 1280 720")


def _simplepbr_enabled() -> bool:
    import os

    return os.environ.get("CFAST_PANDA_SIMPLEPBR", "").strip().lower() in {
        "1",
        "true",
        "on",
        "yes",
    }


class _RuntimeApp:
    def __init__(self, request: Panda3DRequest, *, headless: bool) -> None:
        _configure_panda3d(headless=headless)

        from direct.showbase.ShowBase import ShowBase
        from direct.showbase.ShowBaseGlobal import globalClock

        self._request = request
        self._stats = _SceneStats()
        self._catalog = Panda3DAssetCatalog(
            asset_root=Path(request.asset_root).resolve() if request.asset_root else None
        )
        self._base = ShowBase()
        self._clock = globalClock
        self._headless = headless

        self._configure_render_pipeline()
        self._configure_camera()
        self._configure_lights()
        self._build_scene()
        self._elapsed_s = 0.0
        self._base.taskMgr.add(self._tick, "cfast-panda3d-tick")

    def _configure_render_pipeline(self) -> None:
        if not _simplepbr_enabled():
            return
        try:
            import simplepbr

            simplepbr.init(
                render_node=self._base.render,
                window=self._base.win,
                camera_node=self._base.cam,
                taskmgr=self._base.taskMgr,
                enable_shadows=False,
                enable_fog=False,
            )
        except Exception:
            pass

    def _configure_camera(self) -> None:
        self._base.disableMouse()
        self._base.cam.setPos(0.0, -26.0, 8.0)
        self._base.cam.lookAt(0.0, 0.0, 2.5)
        self._base.setBackgroundColor(0.06, 0.10, 0.24, 1.0)

    def _configure_lights(self) -> None:
        from panda3d.core import AmbientLight, DirectionalLight, Vec4

        ambient = AmbientLight("ambient")
        ambient.setColor(Vec4(0.48, 0.50, 0.58, 1.0))
        ambient_np = self._base.render.attachNewNode(ambient)
        self._base.render.setLight(ambient_np)

        sun = DirectionalLight("sun")
        sun.setColor(Vec4(0.94, 0.92, 0.86, 1.0))
        sun_np = self._base.render.attachNewNode(sun)
        sun_np.setHpr(35.0, -35.0, 0.0)
        self._base.render.setLight(sun_np)

    def _build_scene(self) -> None:
        if self._request.scene is Panda3DScene.RAPID_TRACKING:
            self._build_rapid_tracking_scene()
        elif self._request.scene is Panda3DScene.SPATIAL_INTEGRATION:
            self._build_spatial_integration_scene()
        else:
            self._build_auditory_capacity_scene()

    def _make_box(
        self,
        *,
        size: tuple[float, float, float],
        color: tuple[float, float, float, float],
    ):
        from panda3d.core import CardMaker, NodePath

        sx, sy, sz = size
        root = NodePath("box")

        def add_card(
            name: str,
            frame: tuple[float, float, float, float],
            hpr: tuple[float, float, float],
            pos: tuple[float, float, float],
        ) -> None:
            maker = CardMaker(name)
            maker.setFrame(*frame)
            node = root.attachNewNode(maker.generate())
            node.setPos(*pos)
            node.setHpr(*hpr)
            node.setColor(*color)
            node.setTwoSided(True)

        add_card(
            "top",
            (-sx / 2.0, sx / 2.0, -sy / 2.0, sy / 2.0),
            (0.0, -90.0, 0.0),
            (0.0, 0.0, sz / 2.0),
        )
        add_card(
            "bottom",
            (-sx / 2.0, sx / 2.0, -sy / 2.0, sy / 2.0),
            (0.0, 90.0, 0.0),
            (0.0, 0.0, -sz / 2.0),
        )
        add_card(
            "front",
            (-sx / 2.0, sx / 2.0, -sz / 2.0, sz / 2.0),
            (0.0, 0.0, 0.0),
            (0.0, sy / 2.0, 0.0),
        )
        add_card(
            "back",
            (-sx / 2.0, sx / 2.0, -sz / 2.0, sz / 2.0),
            (180.0, 0.0, 0.0),
            (0.0, -sy / 2.0, 0.0),
        )
        add_card(
            "left",
            (-sy / 2.0, sy / 2.0, -sz / 2.0, sz / 2.0),
            (90.0, 0.0, 0.0),
            (-sx / 2.0, 0.0, 0.0),
        )
        add_card(
            "right",
            (-sy / 2.0, sy / 2.0, -sz / 2.0, sz / 2.0),
            (-90.0, 0.0, 0.0),
            (sx / 2.0, 0.0, 0.0),
        )
        root.setTransparency(True)
        return root

    def _load_asset_or_fallback(
        self,
        *,
        asset_id: str,
        fallback: str,
        pos: tuple[float, float, float],
        hpr: tuple[float, float, float] = (0.0, 0.0, 0.0),
        scale: float = 1.0,
        color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    ):
        from panda3d.core import NodePath

        entry = self._catalog.entry(asset_id)
        resolved = self._catalog.resolve_path(asset_id)
        node: NodePath
        if resolved is not None:
            node = self._load_model(resolved)
            self._stats = _SceneStats(
                frames=self._stats.frames,
                loaded_assets=self._stats.loaded_assets + 1,
                fallback_assets=self._stats.fallback_assets,
            )
            if entry is not None:
                scale *= entry.scale
        else:
            fallback_kind = entry.fallback if entry is not None else fallback
            node = self._build_fallback_model(kind=fallback_kind, color=color)
            self._stats = _SceneStats(
                frames=self._stats.frames,
                loaded_assets=self._stats.loaded_assets,
                fallback_assets=self._stats.fallback_assets + 1,
            )
        node.reparentTo(self._base.render)
        node.setPos(*pos)
        node.setHpr(*hpr)
        node.setScale(scale)
        return node

    def _load_model(self, path: Path):
        suffix = path.suffix.lower()
        if suffix in {".glb", ".gltf"}:
            import gltf

            return gltf.load_model(path)
        return self._base.loader.loadModel(str(path))

    def _build_fallback_model(self, *, kind: str, color: tuple[float, float, float, float]):
        from panda3d.core import NodePath

        root = NodePath(kind)
        if kind == "plane":
            fuselage = self._make_box(size=(0.7, 3.2, 0.55), color=color)
            wing = self._make_box(size=(3.8, 0.55, 0.10), color=color)
            tail = self._make_box(size=(1.4, 0.36, 0.10), color=color)
            fin = self._make_box(size=(0.10, 0.55, 0.55), color=(0.88, 0.88, 0.90, 1.0))
            wing.setZ(0.04)
            tail.setY(-1.28)
            tail.setZ(0.18)
            fin.setY(-1.35)
            fin.setZ(0.42)
            for child in (fuselage, wing, tail, fin):
                child.reparentTo(root)
        elif kind == "helicopter":
            body = self._make_box(size=(0.9, 1.8, 0.8), color=color)
            tail = self._make_box(size=(0.14, 1.5, 0.14), color=color)
            rotor = self._make_box(size=(3.0, 0.08, 0.03), color=(0.18, 0.18, 0.20, 1.0))
            tail.setY(-1.5)
            rotor.setZ(0.52)
            for child in (body, tail, rotor):
                child.reparentTo(root)
        elif kind == "truck":
            cab = self._make_box(size=(0.9, 0.9, 0.8), color=color)
            bed = self._make_box(size=(1.4, 1.2, 0.65), color=(0.36, 0.42, 0.24, 1.0))
            cab.setX(0.34)
            bed.setX(-0.38)
            for child in (cab, bed):
                child.reparentTo(root)
        elif kind == "hangar":
            shell = self._make_box(size=(6.0, 4.8, 2.8), color=(0.62, 0.64, 0.70, 1.0))
            door = self._make_box(size=(2.6, 0.08, 1.7), color=(0.16, 0.20, 0.28, 1.0))
            door.setY(2.45)
            door.setZ(-0.35)
            for child in (shell, door):
                child.reparentTo(root)
        elif kind == "tower":
            shaft = self._make_box(size=(0.8, 0.8, 4.4), color=(0.72, 0.72, 0.76, 1.0))
            cab = self._make_box(size=(1.5, 1.5, 0.9), color=(0.24, 0.34, 0.42, 1.0))
            cab.setZ(2.48)
            for child in (shaft, cab):
                child.reparentTo(root)
        elif kind == "trees":
            for offset in (-0.8, 0.0, 0.8):
                trunk = self._make_box(size=(0.14, 0.14, 0.8), color=(0.38, 0.24, 0.12, 1.0))
                crown = self._make_box(size=(0.9, 0.9, 1.1), color=(0.14, 0.40, 0.18, 0.96))
                trunk.setX(offset)
                trunk.setZ(0.4)
                crown.setX(offset)
                crown.setZ(1.25)
                trunk.reparentTo(root)
                crown.reparentTo(root)
        elif kind == "soldiers":
            for offset in (-0.35, 0.0, 0.35):
                body = self._make_box(size=(0.12, 0.12, 0.48), color=color)
                head = self._make_box(size=(0.16, 0.16, 0.16), color=(0.78, 0.72, 0.62, 1.0))
                body.setX(offset)
                body.setZ(0.24)
                head.setX(offset)
                head.setZ(0.60)
                body.reparentTo(root)
                head.reparentTo(root)
        else:
            self._make_box(size=(1.0, 1.0, 1.0), color=color).reparentTo(root)
        return root

    def _add_ground(self) -> None:
        ground = self._make_box(size=(90.0, 90.0, 0.1), color=(0.20, 0.28, 0.18, 1.0))
        ground.reparentTo(self._base.render)
        ground.setPos(0.0, 22.0, -0.08)

    def _build_rapid_tracking_scene(self) -> None:
        self._add_ground()
        self._load_asset_or_fallback(
            asset_id="building_hangar",
            fallback="hangar",
            pos=(0.0, 16.0, 1.4),
            scale=1.0,
        )
        self._rapid_target = self._load_asset_or_fallback(
            asset_id="plane_red",
            fallback="plane",
            pos=(-5.0, 6.0, 3.0),
            hpr=(18.0, 6.0, -8.0),
            scale=1.1,
            color=(0.86, 0.18, 0.16, 1.0),
        )
        self._rapid_decoy = self._load_asset_or_fallback(
            asset_id="plane_blue",
            fallback="plane",
            pos=(5.5, 13.0, 4.0),
            hpr=(-20.0, -4.0, 6.0),
            scale=0.95,
            color=(0.20, 0.42, 0.88, 1.0),
        )
        self._rapid_path = 0.0

    def _build_spatial_integration_scene(self) -> None:
        self._add_ground()
        self._load_asset_or_fallback(
            asset_id="building_tower",
            fallback="tower",
            pos=(-5.5, 9.0, 2.2),
            scale=1.0,
        )
        self._load_asset_or_fallback(
            asset_id="trees_pine_cluster",
            fallback="trees",
            pos=(6.5, 11.0, 0.0),
            scale=1.4,
        )
        self._load_asset_or_fallback(
            asset_id="truck_olive",
            fallback="truck",
            pos=(0.0, 8.5, 0.35),
            hpr=(90.0, 0.0, 0.0),
            scale=1.0,
            color=(0.42, 0.48, 0.24, 1.0),
        )
        self._load_asset_or_fallback(
            asset_id="soldiers_patrol",
            fallback="soldiers",
            pos=(-2.8, 6.6, 0.0),
            scale=1.0,
            color=(0.18, 0.22, 0.16, 1.0),
        )
        self._spatial_air = self._load_asset_or_fallback(
            asset_id="helicopter_green",
            fallback="helicopter",
            pos=(3.0, 7.0, 3.8),
            hpr=(150.0, 0.0, 0.0),
            scale=1.0,
            color=(0.24, 0.54, 0.32, 1.0),
        )
        self._spatial_path = 0.0

    def _build_auditory_capacity_scene(self) -> None:
        self._base.cam.setPos(0.0, -12.0, 1.0)
        self._base.cam.lookAt(0.0, 10.0, 1.5)
        ring_count = 12
        for idx in range(ring_count):
            y = idx * 2.5
            x = math.sin(idx * 0.52) * 2.8
            z = 1.0 + (math.cos(idx * 0.37) * 1.1)
            outer = self._make_box(size=(4.8, 0.12, 4.8), color=(0.16, 0.28, 0.62, 0.36))
            inner = self._make_box(size=(2.6, 0.24, 2.6), color=(0.06, 0.08, 0.14, 1.0))
            outer.reparentTo(self._base.render)
            inner.reparentTo(self._base.render)
            outer.setPos(x, y, z)
            inner.setPos(x, y, z)
        self._auditory_ball = self._make_box(size=(0.45, 0.45, 0.45), color=(0.94, 0.24, 0.20, 1.0))
        self._auditory_ball.reparentTo(self._base.render)
        self._auditory_path = 0.0

    def _tick(self, task):
        dt = self._clock.getDt()
        self._elapsed_s += dt
        self._stats = _SceneStats(
            frames=self._stats.frames + 1,
            loaded_assets=self._stats.loaded_assets,
            fallback_assets=self._stats.fallback_assets,
        )

        if self._request.scene is Panda3DScene.RAPID_TRACKING:
            self._animate_rapid_tracking()
        elif self._request.scene is Panda3DScene.SPATIAL_INTEGRATION:
            self._animate_spatial_integration()
        else:
            self._animate_auditory_capacity()

        if self._elapsed_s >= max(0.05, float(self._request.duration_s)):
            self._base.taskMgr.stop()
            return task.done
        return task.cont

    def _animate_rapid_tracking(self) -> None:
        self._rapid_path += 0.018
        t = self._rapid_path
        x = math.sin(t * 2.3) * 7.5
        y = 10.0 + math.cos(t * 1.6) * 4.2
        z = 3.4 + math.sin(t * 3.0) * 1.2
        self._rapid_target.setPos(x, y, z)
        self._rapid_target.setHpr(
            (t * 180.0) % 360.0,
            math.sin(t * 2.4) * 12.0,
            math.cos(t * 1.9) * 18.0,
        )
        self._rapid_decoy.setPos(
            -x * 0.72,
            15.0 + math.sin(t * 1.4) * 3.2,
            4.0 + math.cos(t * 2.1) * 1.1,
        )
        self._rapid_decoy.setHpr(
            (-t * 120.0) % 360.0,
            math.cos(t * 1.8) * 10.0,
            math.sin(t * 1.5) * 16.0,
        )

    def _animate_spatial_integration(self) -> None:
        self._spatial_path += 0.014
        t = self._spatial_path
        self._spatial_air.setPos(
            4.0 + math.sin(t * 2.0) * 4.5,
            8.0 + math.cos(t * 1.6) * 5.2,
            3.8 + math.sin(t * 2.8) * 1.2,
        )
        self._spatial_air.setHpr(
            140.0 + math.sin(t * 1.6) * 80.0,
            math.cos(t * 1.8) * 8.0,
            math.sin(t * 1.2) * 12.0,
        )

    def _animate_auditory_capacity(self) -> None:
        self._auditory_path += 0.026
        t = self._auditory_path
        self._auditory_ball.setPos(
            math.sin(t * 1.4) * 2.6,
            (t * 8.0) % 28.0,
            1.0 + math.cos(t * 1.1) * 1.05,
        )
        self._auditory_ball.setHpr((t * 240.0) % 360.0, (t * 120.0) % 360.0, (t * 80.0) % 360.0)

    def run(self) -> Panda3DResult:
        self._base.run()
        summary = (
            f"Ran {self._request.scene.value} for {self._elapsed_s:.2f}s "
            f"with {self._stats.loaded_assets} external assets and "
            f"{self._stats.fallback_assets} fallbacks."
        )
        return Panda3DResult(
            ok=True,
            scene=self._request.scene,
            summary=summary,
            metrics={
                "duration_s": self._elapsed_s,
                "frames": float(self._stats.frames),
                "loaded_assets": float(self._stats.loaded_assets),
                "fallback_assets": float(self._stats.fallback_assets),
            },
        )


def main() -> int:
    args = _parse_args()
    request = _load_request(args.request)
    app = _RuntimeApp(request, headless=bool(args.headless))
    result = app.run()
    _write_result(args.result, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
