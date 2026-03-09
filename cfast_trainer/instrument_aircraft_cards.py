from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pygame

from .instrument_comprehension import InstrumentState
from .panda3d_assets import Panda3DAssetCatalog

_CANONICAL_CARD_SIZE = (448, 280)
_CARD_SPRITE_VERSION = "v2"


def panda3d_card_rendering_available() -> bool:
    return importlib.util.find_spec("direct.showbase.ShowBase") is not None


def _default_cache_dir() -> Path:
    env = os.environ.get("CFAST_INSTRUMENT_CARD_CACHE")
    if env:
        return Path(env).expanduser().resolve()
    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Caches" / "cfast_trainer" / "instrument_cards"
    if os.name == "nt":
        local = os.environ.get("LOCALAPPDATA")
        base = Path(local) if local else (home / "AppData" / "Local")
        return base / "cfast_trainer" / "instrument_cards"
    return home / ".cache" / "cfast_trainer" / "instrument_cards"


@dataclass(frozen=True, slots=True)
class InstrumentAircraftCardKey:
    heading_deg: int
    pitch_deg: int
    bank_deg: int

    @classmethod
    def from_state(cls, state: InstrumentState) -> InstrumentAircraftCardKey:
        return cls(
            heading_deg=int(state.heading_deg) % 360,
            pitch_deg=max(-20, min(20, int(round(state.pitch_deg)))),
            bank_deg=max(-45, min(45, int(round(state.bank_deg)))),
        )

    def filename(self) -> str:
        return (
            f"{_CARD_SPRITE_VERSION}_"
            f"h{self.heading_deg:03d}_p{self.pitch_deg:+03d}_b{self.bank_deg:+03d}.png"
            .replace("+", "p")
            .replace("-", "m")
        )


class InstrumentAircraftCardSpriteBank:
    def __init__(
        self,
        *,
        cache_dir: Path | None = None,
        asset_catalog: Panda3DAssetCatalog | None = None,
        allow_generation: bool = True,
    ) -> None:
        self._cache_dir = (cache_dir or _default_cache_dir()).resolve()
        self._asset_catalog = asset_catalog or Panda3DAssetCatalog()
        self._allow_generation = bool(allow_generation)
        self._surface_cache: dict[InstrumentAircraftCardKey, pygame.Surface] = {}
        self._scaled_cache: dict[tuple[InstrumentAircraftCardKey, int, int], pygame.Surface] = {}
        self._renderer: _Panda3DInstrumentCardRenderer | None = None
        self._generation_failed = False

    def get_scaled_surface(
        self,
        *,
        state: InstrumentState,
        size: tuple[int, int],
    ) -> pygame.Surface | None:
        key = InstrumentAircraftCardKey.from_state(state)
        cache_key = (key, int(size[0]), int(size[1]))
        cached = self._scaled_cache.get(cache_key)
        if cached is not None:
            return cached

        source = self.get_surface(state=state)
        if source is None:
            return None
        scaled = pygame.transform.smoothscale(source, size)
        self._scaled_cache[cache_key] = scaled
        return scaled

    def get_surface(self, *, state: InstrumentState) -> pygame.Surface | None:
        key = InstrumentAircraftCardKey.from_state(state)
        cached = self._surface_cache.get(key)
        if cached is not None:
            return cached

        cache_path = self._cache_dir / key.filename()
        if cache_path.exists():
            loaded = self._load_surface(cache_path)
            self._surface_cache[key] = loaded
            return loaded

        if not self._allow_generation or self._generation_failed:
            return None
        if not panda3d_card_rendering_available():
            self._generation_failed = True
            return None

        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            renderer = self._get_renderer()
            renderer.render_card(key=key, destination=cache_path)
            loaded = self._load_surface(cache_path)
            self._surface_cache[key] = loaded
            return loaded
        except Exception:
            self._generation_failed = True
            return None

    def cache_path_for(self, *, state: InstrumentState) -> Path:
        return self._cache_dir / InstrumentAircraftCardKey.from_state(state).filename()

    def _get_renderer(self) -> _Panda3DInstrumentCardRenderer:
        if self._renderer is None:
            self._renderer = _Panda3DInstrumentCardRenderer(asset_catalog=self._asset_catalog)
        return self._renderer

    @staticmethod
    def _load_surface(path: Path) -> pygame.Surface:
        loaded = pygame.image.load(str(path))
        if pygame.display.get_init() and pygame.display.get_surface() is not None:
            return loaded.convert_alpha()
        return loaded.copy()


class _Panda3DInstrumentCardRenderer:
    def __init__(self, *, asset_catalog: Panda3DAssetCatalog) -> None:
        self._asset_catalog = asset_catalog

        from panda3d.core import AmbientLight, DirectionalLight, Vec4, loadPrcFileData

        loadPrcFileData("", "window-type offscreen")
        loadPrcFileData("", "audio-library-name null")
        loadPrcFileData("", f"win-size {_CANONICAL_CARD_SIZE[0]} {_CANONICAL_CARD_SIZE[1]}")
        loadPrcFileData("", "sync-video false")

        from direct.showbase.ShowBase import ShowBase

        self._base = ShowBase()
        self._base.disableMouse()
        self._base.setBackgroundColor(0.84, 0.85, 0.87, 1.0)
        self._base.cam.setPos(0.0, -10.4, 0.7)
        self._base.cam.lookAt(0.0, 0.0, 0.4)

        ambient = AmbientLight("ambient")
        ambient.setColor(Vec4(0.56, 0.58, 0.62, 1.0))
        ambient_np = self._base.render.attachNewNode(ambient)
        self._base.render.setLight(ambient_np)

        sun = DirectionalLight("sun")
        sun.setColor(Vec4(0.96, 0.94, 0.90, 1.0))
        sun_np = self._base.render.attachNewNode(sun)
        sun_np.setHpr(25.0, -32.0, 0.0)
        self._base.render.setLight(sun_np)

        self._build_room()
        self._plane_root = self._base.render.attachNewNode("instrument-plane-root")
        plane = self._load_plane_asset()
        plane.reparentTo(self._plane_root)
        plane.setScale(1.2)
        plane.setPos(0.0, 0.0, 0.10)

    def render_card(self, *, key: InstrumentAircraftCardKey, destination: Path) -> None:
        from panda3d.core import PNMImage

        self._plane_root.setHpr(-float(key.heading_deg), float(key.pitch_deg), -float(key.bank_deg))

        for _ in range(3):
            self._base.graphicsEngine.renderFrame()

        image = PNMImage()
        ok = self._base.win.getScreenshot(image)
        if not ok:
            raise RuntimeError("Failed to capture Instrument Comprehension card sprite.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.write(str(destination))

    def _build_room(self) -> None:
        room = self._make_open_room()
        room.reparentTo(self._base.render)
        room.setPos(0.0, 2.8, 0.0)

    def _load_plane_asset(self):
        asset_path = self._asset_catalog.resolve_path("plane_red")
        if asset_path is not None:
            suffix = asset_path.suffix.lower()
            if suffix in {".glb", ".gltf"}:
                import gltf

                node = gltf.load_model(asset_path)
            else:
                node = self._base.loader.loadModel(str(asset_path))
            if not node.isEmpty():
                return node
        return self._build_fallback_plane()

    def _make_open_room(self):
        from panda3d.core import CardMaker, NodePath

        root = NodePath("room")

        def add_panel(
            name: str,
            *,
            frame: tuple[float, float, float, float],
            pos: tuple[float, float, float],
            hpr: tuple[float, float, float],
            color: tuple[float, float, float, float],
        ) -> None:
            maker = CardMaker(name)
            maker.setFrame(*frame)
            node = root.attachNewNode(maker.generate())
            node.setPos(*pos)
            node.setHpr(*hpr)
            node.setColor(*color)
            node.setTwoSided(True)

        depth = 5.8
        width = 7.2
        height = 4.2
        add_panel(
            "back",
            frame=(-width / 2.0, width / 2.0, -height / 2.0, height / 2.0),
            pos=(0.0, depth, 0.0),
            hpr=(180.0, 0.0, 0.0),
            color=(0.76, 0.77, 0.79, 1.0),
        )
        add_panel(
            "floor",
            frame=(-width / 2.0, width / 2.0, -depth / 2.0, depth / 2.0),
            pos=(0.0, depth / 2.0, -height / 2.0),
            hpr=(0.0, 90.0, 0.0),
            color=(0.64, 0.64, 0.66, 1.0),
        )
        add_panel(
            "ceiling",
            frame=(-width / 2.0, width / 2.0, -depth / 2.0, depth / 2.0),
            pos=(0.0, depth / 2.0, height / 2.0),
            hpr=(0.0, -90.0, 0.0),
            color=(0.88, 0.88, 0.90, 1.0),
        )
        add_panel(
            "left",
            frame=(-depth / 2.0, depth / 2.0, -height / 2.0, height / 2.0),
            pos=(-width / 2.0, depth / 2.0, 0.0),
            hpr=(90.0, 0.0, 0.0),
            color=(0.72, 0.72, 0.74, 1.0),
        )
        add_panel(
            "right",
            frame=(-depth / 2.0, depth / 2.0, -height / 2.0, height / 2.0),
            pos=(width / 2.0, depth / 2.0, 0.0),
            hpr=(-90.0, 0.0, 0.0),
            color=(0.70, 0.70, 0.72, 1.0),
        )

        self._add_floor_guides(root=root, width=width, depth=depth, floor_z=-(height / 2.0) + 0.002)
        return root

    def _add_floor_guides(self, *, root, width: float, depth: float, floor_z: float) -> None:
        from panda3d.core import LineSegs

        guides = LineSegs("floor-guides")
        guides.setThickness(1.0)
        guides.setColor(0.52, 0.52, 0.54, 1.0)
        for lane in range(-3, 4):
            x = lane * (width / 8.0)
            guides.moveTo(x, 0.0, floor_z)
            guides.drawTo(x * 0.38, depth, floor_z)
        for row in range(1, 5):
            y = row * (depth / 5.0)
            scale = 1.0 - (row * 0.12)
            guides.moveTo(-(width / 2.0) * scale, y, floor_z)
            guides.drawTo((width / 2.0) * scale, y, floor_z)
        root.attachNewNode(guides.create())

    def _build_fallback_plane(self):
        from panda3d.core import NodePath

        root = NodePath("plane")
        plane_red = (0.84, 0.18, 0.16, 1.0)
        plane_dark = (0.48, 0.05, 0.06, 1.0)
        canopy = (0.72, 0.88, 0.94, 0.92)

        fuselage_main = self._make_box(size=(0.72, 2.28, 0.60), color=plane_red)
        fuselage_mid = self._make_box(size=(0.64, 1.10, 0.56), color=plane_red)
        nose = self._make_box(size=(0.48, 0.92, 0.46), color=(0.92, 0.36, 0.32, 1.0))
        nose_tip = self._make_box(size=(0.26, 0.42, 0.24), color=(0.96, 0.44, 0.40, 1.0))
        tail_boom = self._make_box(size=(0.48, 1.12, 0.42), color=plane_red)
        tail_tip = self._make_box(size=(0.24, 0.36, 0.22), color=plane_dark)

        wing_center = self._make_box(size=(2.10, 0.40, 0.12), color=plane_red)
        wing_left = self._make_box(size=(1.56, 0.42, 0.10), color=plane_red)
        wing_right = self._make_box(size=(1.56, 0.42, 0.10), color=plane_red)
        wing_tip_l = self._make_box(size=(0.16, 0.40, 0.08), color=plane_dark)
        wing_tip_r = self._make_box(size=(0.16, 0.40, 0.08), color=plane_dark)

        tail_center = self._make_box(size=(0.86, 0.22, 0.08), color=plane_red)
        tail_left = self._make_box(size=(0.62, 0.20, 0.06), color=plane_red)
        tail_right = self._make_box(size=(0.62, 0.20, 0.06), color=plane_red)
        fin = self._make_box(size=(0.10, 0.42, 0.72), color=plane_dark)
        canopy_node = self._make_box(size=(0.32, 0.86, 0.28), color=canopy)
        engine_l = self._make_box(size=(0.26, 0.62, 0.24), color=plane_dark)
        engine_r = self._make_box(size=(0.26, 0.62, 0.24), color=plane_dark)

        fuselage_mid.setY(1.46)
        nose.setY(2.44)
        nose_tip.setY(3.06)
        tail_boom.setY(-1.70)
        tail_tip.setY(-2.42)

        wing_center.setY(0.54)
        wing_center.setZ(0.02)
        wing_left.setPos(-1.74, 0.64, 0.04)
        wing_right.setPos(1.74, 0.64, 0.04)
        wing_left.setHpr(-12.0, 0.0, 2.0)
        wing_right.setHpr(12.0, 0.0, -2.0)
        wing_tip_l.setPos(-2.66, 0.48, 0.06)
        wing_tip_r.setPos(2.66, 0.48, 0.06)
        wing_tip_l.setHpr(-18.0, 0.0, 3.0)
        wing_tip_r.setHpr(18.0, 0.0, -3.0)

        tail_center.setPos(0.0, -2.00, 0.16)
        tail_left.setPos(-0.70, -2.04, 0.18)
        tail_right.setPos(0.70, -2.04, 0.18)
        tail_left.setHpr(-10.0, 0.0, 0.0)
        tail_right.setHpr(10.0, 0.0, 0.0)
        fin.setY(-2.12)
        fin.setZ(0.50)

        canopy_node.setY(1.06)
        canopy_node.setZ(0.28)
        engine_l.setPos(-0.84, 0.82, -0.10)
        engine_r.setPos(0.84, 0.82, -0.10)

        for node in (
            fuselage_main,
            fuselage_mid,
            nose,
            nose_tip,
            tail_boom,
            tail_tip,
            wing_center,
            wing_left,
            wing_right,
            wing_tip_l,
            wing_tip_r,
            tail_center,
            tail_left,
            tail_right,
            fin,
            canopy_node,
            engine_l,
            engine_r,
        ):
            node.reparentTo(root)
        return root

    @staticmethod
    def _make_box(*, size: tuple[float, float, float], color: tuple[float, float, float, float]):
        from panda3d.core import CardMaker, NodePath

        sx, sy, sz = size
        root = NodePath("box")

        def add_face(
            name: str,
            frame: tuple[float, float, float, float],
            pos: tuple[float, float, float],
            hpr: tuple[float, float, float],
        ) -> None:
            maker = CardMaker(name)
            maker.setFrame(*frame)
            node = root.attachNewNode(maker.generate())
            node.setPos(*pos)
            node.setHpr(*hpr)
            node.setColor(*color)
            node.setTwoSided(True)

        add_face(
            "top",
            (-sx / 2.0, sx / 2.0, -sy / 2.0, sy / 2.0),
            (0.0, 0.0, sz / 2.0),
            (0.0, -90.0, 0.0),
        )
        add_face(
            "bottom",
            (-sx / 2.0, sx / 2.0, -sy / 2.0, sy / 2.0),
            (0.0, 0.0, -sz / 2.0),
            (0.0, 90.0, 0.0),
        )
        add_face(
            "front",
            (-sx / 2.0, sx / 2.0, -sz / 2.0, sz / 2.0),
            (0.0, sy / 2.0, 0.0),
            (0.0, 0.0, 0.0),
        )
        add_face(
            "back",
            (-sx / 2.0, sx / 2.0, -sz / 2.0, sz / 2.0),
            (0.0, -sy / 2.0, 0.0),
            (180.0, 0.0, 0.0),
        )
        add_face(
            "left",
            (-sy / 2.0, sy / 2.0, -sz / 2.0, sz / 2.0),
            (-sx / 2.0, 0.0, 0.0),
            (90.0, 0.0, 0.0),
        )
        add_face(
            "right",
            (-sy / 2.0, sy / 2.0, -sz / 2.0, sz / 2.0),
            (sx / 2.0, 0.0, 0.0),
            (-90.0, 0.0, 0.0),
        )
        return root
