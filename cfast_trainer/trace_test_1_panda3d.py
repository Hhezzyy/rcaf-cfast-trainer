from __future__ import annotations

import importlib.util
import os

import pygame

from .aircraft_art import (
    build_panda_palette,
    build_panda3d_fixed_wing_model,
    panda3d_fixed_wing_hpr_from_world_hpr,
)
from .trace_scene_3d import TraceAircraftPose, TraceCameraPose, build_trace_test_1_scene3d
from .trace_test_1 import TraceTest1Command, TraceTest1Payload, trace_test_1_aircraft_hpr


def panda3d_trace_test_1_rendering_available() -> bool:
    if os.environ.get("SDL_VIDEODRIVER", "").strip().lower() == "dummy":
        return False
    try:
        return importlib.util.find_spec("direct.showbase.ShowBase") is not None
    except ModuleNotFoundError:
        return False


class TraceTest1Panda3DRenderer:
    def __init__(self, *, size: tuple[int, int]) -> None:
        from panda3d.core import (
            AmbientLight,
            GraphicsOutput,
            Texture,
            Vec4,
            loadPrcFileData,
        )

        width = max(320, int(size[0]))
        height = max(200, int(size[1]))
        self._size = (width, height)

        loadPrcFileData("", "window-type offscreen")
        loadPrcFileData("", "audio-library-name null")
        loadPrcFileData("", "sync-video false")
        loadPrcFileData("", f"win-size {width} {height}")

        from direct.showbase.ShowBase import ShowBase

        self._base = ShowBase(windowType="offscreen")
        self._base.disableMouse()
        self._base.setBackgroundColor(0.45, 0.60, 0.82, 1.0)
        self._base.camLens.setFov(48.0, 38.0)
        self._base.camLens.setNearFar(0.12, 1200.0)

        self._texture = Texture()
        self._texture.setKeepRamImage(True)
        self._base.win.addRenderTexture(self._texture, GraphicsOutput.RTMCopyRam)

        ambient = AmbientLight("trace1-ambient")
        ambient.setColor(Vec4(0.96, 0.97, 1.0, 1.0))
        ambient_np = self._base.render.attachNewNode(ambient)
        self._base.render.setLight(ambient_np)

        self._world_root = self._base.render.attachNewNode("trace1-world")
        self._aircraft_root = self._world_root.attachNewNode("trace1-aircraft")
        self._red_anchor = self._aircraft_root.attachNewNode("trace1-red")
        self._red_aircraft = self._build_aircraft_model(
            body_color=(0.90, 0.24, 0.26, 1.0),
            canopy_color=(1.0, 0.92, 0.92, 1.0),
        )
        self._red_aircraft.reparentTo(self._red_anchor)
        self._blue_anchors: list[object] = []
        self._blue_models: list[object] = []

    @property
    def size(self) -> tuple[int, int]:
        return self._size

    def close(self) -> None:
        try:
            self._base.destroy()
        except Exception:
            pass

    def render(
        self,
        *,
        payload: TraceTest1Payload | None,
    ) -> pygame.Surface:
        if payload is None:
            surface = pygame.Surface(self._size)
            surface.fill((86, 118, 160))
            return surface

        snapshot = self._scene_snapshot(payload=payload)
        self._apply_camera(camera=snapshot.camera)
        self._ensure_blue_anchors(count=max(0, len(snapshot.aircraft) - 1))
        self._update_aircraft(
            anchor=self._red_anchor,
            pose=snapshot.aircraft[0],
        )
        for idx, pose in enumerate(snapshot.aircraft[1:]):
            self._update_aircraft(
                anchor=self._blue_anchors[idx],
                pose=pose,
            )
        self._base.graphicsEngine.renderFrame()

        ram = self._texture.getRamImageAs("RGBA")
        raw = bytes(ram)
        surface = pygame.image.frombuffer(raw, self._size, "RGBA").copy()
        return pygame.transform.flip(surface, False, True)

    def _ensure_blue_anchors(self, *, count: int) -> None:
        while len(self._blue_anchors) < count:
            anchor = self._aircraft_root.attachNewNode(f"trace1-blue-{len(self._blue_anchors)}")
            model = self._build_aircraft_model(
                body_color=(0.34, 0.52, 0.90, 1.0),
                canopy_color=(0.90, 0.96, 1.0, 1.0),
            )
            model.reparentTo(anchor)
            self._blue_anchors.append(anchor)
            self._blue_models.append(model)
        for idx, anchor in enumerate(self._blue_anchors):
            anchor.show() if idx < count else anchor.hide()

    def _update_aircraft(
        self,
        *,
        anchor,
        pose: TraceAircraftPose,
    ) -> None:
        anchor.setPos(
            float(pose.position[0]),
            float(pose.position[1]),
            float(pose.position[2]),
        )
        anchor.setScale(
            float(pose.scale[0]),
            float(pose.scale[1]),
            float(pose.scale[2]),
        )
        anchor.setHpr(*self._aircraft_hpr_for_pose(pose))

    @staticmethod
    def _aircraft_hpr_for_frame(
        *,
        frame,
        command: TraceTest1Command,
        observe_progress: float,
        answer_open_progress: float,
        size: tuple[int, int],
    ) -> tuple[float, float, float]:
        _ = (command, observe_progress, answer_open_progress, size)
        heading_deg, pitch_deg, roll_deg = trace_test_1_aircraft_hpr(frame)
        return panda3d_fixed_wing_hpr_from_world_hpr(
            heading_deg=float(heading_deg),
            pitch_deg=float(pitch_deg),
            roll_deg=float(roll_deg),
        )

    @staticmethod
    def _aircraft_hpr_for_pose(pose: TraceAircraftPose) -> tuple[float, float, float]:
        return panda3d_fixed_wing_hpr_from_world_hpr(
            heading_deg=float(pose.hpr_deg[0]),
            pitch_deg=float(pose.hpr_deg[1]),
            roll_deg=float(pose.hpr_deg[2]),
        )

    @staticmethod
    def _scene_snapshot(*, payload: TraceTest1Payload):
        return build_trace_test_1_scene3d(payload=payload)

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

    def _build_aircraft_model(
        self,
        *,
        body_color: tuple[float, float, float, float],
        canopy_color: tuple[float, float, float, float],
    ):
        return build_panda3d_fixed_wing_model(
            palette=build_panda_palette(
                body_color=body_color,
                canopy_color=canopy_color,
            )
        )
