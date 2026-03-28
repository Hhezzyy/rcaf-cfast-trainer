from __future__ import annotations

import importlib.util
import math
import os

import pygame

from .aircraft_art import (
    build_panda_palette,
    build_panda3d_fixed_wing_model,
    panda3d_fixed_wing_hpr_from_world_tangent,
)
from .trace_test_2 import (
    TraceTest2AircraftTrack,
    TraceTest2Payload,
    trace_test_2_track_position,
    trace_test_2_track_tangent,
)


def panda3d_trace_test_2_rendering_available() -> bool:
    if os.environ.get("SDL_VIDEODRIVER", "").strip().lower() == "dummy":
        return False
    return importlib.util.find_spec("direct.showbase.ShowBase") is not None


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(lo if v < lo else hi if v > hi else v)


class TraceTest2Panda3DRenderer:
    def __init__(self, *, size: tuple[int, int]) -> None:
        from panda3d.core import AmbientLight, DirectionalLight, GraphicsOutput, Texture, Vec4, loadPrcFileData

        width = max(320, int(size[0]))
        height = max(200, int(size[1]))
        self._size = (width, height)
        self._elapsed_s = 0.0
        self._last_render_ms = pygame.time.get_ticks()

        loadPrcFileData("", "window-type offscreen")
        loadPrcFileData("", "audio-library-name null")
        loadPrcFileData("", "sync-video false")
        loadPrcFileData("", f"win-size {width} {height}")

        from direct.showbase.ShowBase import ShowBase

        self._base = ShowBase(windowType="offscreen")
        self._base.disableMouse()
        self._base.setBackgroundColor(0.52, 0.66, 0.86, 1.0)
        self._texture = Texture()
        self._texture.setKeepRamImage(True)
        self._base.win.addRenderTexture(self._texture, GraphicsOutput.RTMCopyRam)
        self._base.camLens.setFov(44.0)
        self._base.camLens.setNearFar(0.1, 640.0)

        ambient = AmbientLight("trace2-ambient")
        ambient.setColor(Vec4(0.82, 0.86, 0.90, 1.0))
        ambient_np = self._base.render.attachNewNode(ambient)
        self._base.render.setLight(ambient_np)

        sun = DirectionalLight("trace2-sun")
        sun.setColor(Vec4(0.98, 0.96, 0.92, 1.0))
        sun_np = self._base.render.attachNewNode(sun)
        sun_np.setHpr(24.0, -26.0, 0.0)
        self._base.render.setLight(sun_np)

        self._world_root = self._base.render.attachNewNode("trace2-world")
        self._aircraft_root = self._world_root.attachNewNode("trace2-aircraft")
        self._aircraft_nodes: dict[int, object] = {}

        self._build_world()
        self._update_camera()

    @property
    def size(self) -> tuple[int, int]:
        return self._size

    def close(self) -> None:
        try:
            self._base.destroy()
        except Exception:
            pass

    def render(self, *, payload: TraceTest2Payload | None) -> pygame.Surface:
        now_ms = pygame.time.get_ticks()
        dt_s = max(0.0, min(0.05, (now_ms - self._last_render_ms) / 1000.0))
        self._last_render_ms = now_ms
        self._elapsed_s += dt_s

        if payload is None:
            tracks = ()
            progress = float((self._elapsed_s / 8.0) % 1.0)
        else:
            tracks = payload.aircraft
            progress = float(payload.observe_progress)

        self._update_aircraft(tracks=tracks, progress=progress)
        self._base.graphicsEngine.renderFrame()

        ram = self._texture.getRamImageAs("RGBA")
        raw = bytes(ram)
        surface = pygame.image.frombuffer(raw, self._size, "RGBA").copy()
        return pygame.transform.flip(surface, False, True)

    def _build_world(self) -> None:
        ground = self._make_box(size=(340.0, 420.0, 0.7), color=(0.42, 0.46, 0.52, 1.0))
        ground.setPos(0.0, 122.0, -19.0)
        ground.reparentTo(self._world_root)

    def _update_camera(self) -> None:
        self._base.cam.setPos(0.0, -18.0, 14.0)
        self._base.cam.lookAt(0.0, 96.0, 9.0)

    def _update_aircraft(
        self,
        *,
        tracks: tuple[TraceTest2AircraftTrack, ...],
        progress: float,
    ) -> None:
        active_codes = {int(track.code) for track in tracks}
        for code, node in self._aircraft_nodes.items():
            if code not in active_codes:
                node.hide()

        for track in tracks:
            node = self._aircraft_nodes.get(int(track.code))
            if node is None:
                node = self._build_aircraft_model(color_rgb=track.color_rgb)
                node.reparentTo(self._aircraft_root)
                self._aircraft_nodes[int(track.code)] = node
            node.show()

            pos = trace_test_2_track_position(track=track, progress=progress)
            tangent = self._track_tangent(track=track, progress=progress)
            hpr = self._aircraft_hpr_for_track(
                track=track,
                progress=progress,
                size=self._size,
                tangent=tangent,
            )

            node.setPos(float(pos.x), float(pos.y), float(pos.z))
            node.setHpr(*hpr)
            node.setScale(0.98)

    @staticmethod
    def _aircraft_hpr_from_tangent(
        *,
        tangent: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        dx, dy, _dz = tangent
        horiz = max(1e-6, math.sqrt((dx * dx) + (dy * dy)))
        bank_deg = _clamp((dx / horiz) * 28.0, -34.0, 34.0)
        return panda3d_fixed_wing_hpr_from_world_tangent(
            tangent=tangent,
            roll_deg=bank_deg,
        )

    @classmethod
    def _aircraft_hpr_for_track(
        cls,
        *,
        track: TraceTest2AircraftTrack,
        progress: float,
        size: tuple[int, int],
        tangent: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        _ = (track, progress, size)
        return cls._aircraft_hpr_from_tangent(tangent=tangent)

    @staticmethod
    def _track_tangent(
        *,
        track: TraceTest2AircraftTrack,
        progress: float,
        sample_step: float = 0.03,
    ) -> tuple[float, float, float]:
        _ = sample_step
        dx, dy, dz = trace_test_2_track_tangent(
            track=track,
            progress=_clamp(progress, 0.0, 1.0),
        )
        if (dx * dx) + (dy * dy) + (dz * dz) <= 1e-8:
            raise ValueError(
                f"trace_test_2 track tangent must be non-zero for code={int(track.code)} progress={float(progress):.3f}"
            )
        return (float(dx), float(dy), float(dz))

    def _build_aircraft_model(self, *, color_rgb: tuple[int, int, int]):
        return build_panda3d_fixed_wing_model(
            palette=build_panda_palette(
                body_color=(
                    color_rgb[0] / 255.0,
                    color_rgb[1] / 255.0,
                    color_rgb[2] / 255.0,
                    1.0,
                ),
                canopy_color=(0.94, 0.96, 1.0, 1.0),
            )
        )

    def _make_box(self, *, size: tuple[float, float, float], color: tuple[float, float, float, float]):
        from panda3d.core import CardMaker, NodePath

        sx, sy, _sz = size
        maker = CardMaker("trace2-box")
        maker.setFrame(-(sx * 0.5), sx * 0.5, -(sy * 0.5), sy * 0.5)
        node = NodePath(maker.generate())
        node.setP(-90.0)
        node.setColor(*color)
        return node
