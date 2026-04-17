from __future__ import annotations

import importlib.util
import os

import pygame

from .aircraft_art import (
    build_panda_palette,
    build_panda3d_fixed_wing_model,
    panda3d_fixed_wing_hpr_from_world_hpr,
    panda3d_fixed_wing_hpr_from_world_tangent,
)
from .trace_scene_3d import TraceAircraftPose, TraceCameraPose, build_trace_test_2_scene3d
from .trace_test_2 import (
    TraceTest2AircraftTrack,
    TraceTest2Payload,
    trace_test_2_track_tangent,
)


def panda3d_trace_test_2_rendering_available() -> bool:
    if os.environ.get("SDL_VIDEODRIVER", "").strip().lower() == "dummy":
        return False
    try:
        return importlib.util.find_spec("direct.showbase.ShowBase") is not None
    except ModuleNotFoundError:
        return False


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
        self._base.camLens.setFov(52.0, 40.0)
        self._base.camLens.setNearFar(0.12, 1200.0)

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
            self._update_aircraft(poses=())
        else:
            snapshot = self._scene_snapshot(payload=payload)
            self._apply_camera(camera=snapshot.camera)
            self._update_aircraft(poses=snapshot.aircraft)
        self._base.graphicsEngine.renderFrame()

        ram = self._texture.getRamImageAs("RGBA")
        raw = bytes(ram)
        surface = pygame.image.frombuffer(raw, self._size, "RGBA").copy()
        return pygame.transform.flip(surface, False, True)

    def _build_world(self) -> None:
        ground = self._make_box(size=(340.0, 420.0, 0.7), color=(0.42, 0.46, 0.52, 1.0))
        ground.setPos(0.0, 122.0, -19.0)
        ground.reparentTo(self._world_root)

    def _update_aircraft(
        self,
        *,
        poses: tuple[TraceAircraftPose, ...],
    ) -> None:
        active_codes = {int(pose.code) for pose in poses}
        for code, node in self._aircraft_nodes.items():
            if code not in active_codes:
                node.hide()

        for pose in poses:
            node = self._aircraft_nodes.get(int(pose.code))
            if node is None:
                node = self._build_aircraft_model(color_rgba=self._pose_color_rgba(pose))
                node.reparentTo(self._aircraft_root)
                self._aircraft_nodes[int(pose.code)] = node
            node.show()
            node.setPos(
                float(pose.position[0]),
                float(pose.position[1]),
                float(pose.position[2]),
            )
            node.setHpr(*self._aircraft_hpr_for_pose(pose))
            node.setScale(
                float(pose.scale[0]),
                float(pose.scale[1]),
                float(pose.scale[2]),
            )

    @staticmethod
    def _aircraft_hpr_from_tangent(
        *,
        tangent: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        return panda3d_fixed_wing_hpr_from_world_tangent(
            tangent=tangent,
            roll_deg=0.0,
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
    def _aircraft_hpr_for_pose(pose: TraceAircraftPose) -> tuple[float, float, float]:
        return panda3d_fixed_wing_hpr_from_world_hpr(
            heading_deg=float(pose.hpr_deg[0]),
            pitch_deg=float(pose.hpr_deg[1]),
            roll_deg=float(pose.hpr_deg[2]),
        )

    @staticmethod
    def _scene_snapshot(*, payload: TraceTest2Payload):
        return build_trace_test_2_scene3d(payload=payload)

    def _apply_camera(self, *, camera: TraceCameraPose) -> None:
        self._base.camLens.setFov(float(camera.h_fov_deg), float(camera.v_fov_deg))
        self._base.camLens.setNearFar(float(camera.near_clip), float(camera.far_clip))
        self._base.cam.setPos(
            float(camera.position[0]),
            float(camera.position[1]),
            float(camera.position[2]),
        )
        self._base.cam.setHpr(
            float(camera.heading_deg),
            float(camera.pitch_deg),
            0.0,
        )

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

    @staticmethod
    def _pose_color_rgba(pose: TraceAircraftPose) -> tuple[float, float, float, float]:
        if pose.color_rgba is not None:
            return tuple(float(channel) for channel in pose.color_rgba)
        fallback = {
            "plane_red": (0.86, 0.25, 0.28, 1.0),
            "plane_blue": (0.28, 0.50, 0.88, 1.0),
            "plane_yellow": (0.90, 0.74, 0.24, 1.0),
            "plane_green": (0.42, 0.72, 0.48, 1.0),
        }
        return fallback.get(str(pose.asset_id), (0.28, 0.50, 0.88, 1.0))

    def _build_aircraft_model(self, *, color_rgba: tuple[float, float, float, float]):
        return build_panda3d_fixed_wing_model(
            palette=build_panda_palette(
                body_color=color_rgba,
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
