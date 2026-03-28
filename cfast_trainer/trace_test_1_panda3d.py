from __future__ import annotations

import importlib.util
import os

import pygame

from .aircraft_art import (
    build_panda_palette,
    build_panda3d_fixed_wing_model,
    panda3d_fixed_wing_hpr_from_screen_heading,
)
from .trace_test_1 import TraceTest1Command, TraceTest1Payload
from .trace_test_1_gl import project_scene_position, screen_heading_deg


def panda3d_trace_test_1_rendering_available() -> bool:
    if os.environ.get("SDL_VIDEODRIVER", "").strip().lower() == "dummy":
        return False
    return importlib.util.find_spec("direct.showbase.ShowBase") is not None


class TraceTest1Panda3DRenderer:
    def __init__(self, *, size: tuple[int, int]) -> None:
        from panda3d.core import (
            AmbientLight,
            GraphicsOutput,
            OrthographicLens,
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
        lens = OrthographicLens()
        lens.setFilmSize(width, height)
        lens.setNearFar(-500.0, 500.0)
        self._base.cam.node().setLens(lens)
        self._base.cam.setPos(0.0, -120.0, 0.0)
        self._base.cam.setHpr(0.0, 0.0, 0.0)

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

        self._ensure_blue_anchors(count=len(payload.scene.blue_frames))
        self._update_aircraft(
            anchor=self._red_anchor,
            frame=payload.scene.red_frame,
            command=payload.active_command,
            observe_progress=payload.observe_progress,
            answer_open_progress=payload.answer_open_progress,
            scale=1.08,
        )
        for idx, (frame, command) in enumerate(
            zip(payload.scene.blue_frames, payload.blue_commands, strict=True)
        ):
            self._update_aircraft(
                anchor=self._blue_anchors[idx],
                frame=frame,
                command=command,
                observe_progress=payload.observe_progress,
                answer_open_progress=payload.answer_open_progress,
                scale=0.88,
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
        frame,
        command: TraceTest1Command,
        observe_progress: float,
        answer_open_progress: float,
        scale: float,
    ) -> None:
        center, projected_scale = project_scene_position(frame.position, size=self._size)
        anchor.setPos(
            float(center[0] - (self._size[0] * 0.5)),
            0.0,
            float((self._size[1] * 0.5) - center[1]),
        )
        anchor.setScale(scale * projected_scale * 11.0)
        anchor.setHpr(
            *self._aircraft_hpr_for_frame(
                frame=frame,
                command=command,
                observe_progress=observe_progress,
                answer_open_progress=answer_open_progress,
                size=self._size,
            )
        )

    @staticmethod
    def _aircraft_hpr_for_frame(
        *,
        frame,
        command: TraceTest1Command,
        observe_progress: float,
        answer_open_progress: float,
        size: tuple[int, int],
    ) -> tuple[float, float, float]:
        return panda3d_fixed_wing_hpr_from_screen_heading(
            screen_heading_deg=screen_heading_deg(
                frame,
                command=command,
                observe_progress=observe_progress,
                answer_open_progress=answer_open_progress,
                size=size,
            ),
            pitch_deg=0.0,
            roll_deg=0.0,
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
