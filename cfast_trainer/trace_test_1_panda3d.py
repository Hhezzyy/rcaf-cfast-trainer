from __future__ import annotations

import importlib.util
import math
import os

import pygame

from .aircraft_art import (
    build_panda_palette,
    build_panda3d_fixed_wing_model,
    panda3d_fixed_wing_hpr,
)
from .trace_test_1 import TraceTest1Attitude, TraceTest1SceneFrame, trace_test_1_scene_frames


def panda3d_trace_test_1_rendering_available() -> bool:
    pref = os.environ.get("CFAST_TRACE_TEST_1_RENDERER", "panda").strip().lower()
    if pref in {"pygame", "2d", "off"}:
        return False
    if os.environ.get("SDL_VIDEODRIVER", "").strip().lower() == "dummy":
        return False
    return importlib.util.find_spec("direct.showbase.ShowBase") is not None


class TraceTest1Panda3DRenderer:
    def __init__(self, *, size: tuple[int, int]) -> None:
        from panda3d.core import (
            AmbientLight,
            DirectionalLight,
            GraphicsOutput,
            Texture,
            Vec4,
            loadPrcFileData,
        )

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
        self._base.setBackgroundColor(0.50, 0.66, 0.86, 1.0)
        self._texture = Texture()
        self._texture.setKeepRamImage(True)
        self._base.win.addRenderTexture(self._texture, GraphicsOutput.RTMCopyRam)

        self._base.camLens.setFov(58.0)
        self._base.camLens.setNearFar(0.1, 500.0)

        ambient = AmbientLight("trace-ambient")
        ambient.setColor(Vec4(0.80, 0.84, 0.90, 1.0))
        ambient_np = self._base.render.attachNewNode(ambient)
        self._base.render.setLight(ambient_np)

        sun = DirectionalLight("trace-sun")
        sun.setColor(Vec4(1.0, 0.98, 0.92, 1.0))
        sun_np = self._base.render.attachNewNode(sun)
        sun_np.setHpr(24.0, -36.0, 0.0)
        self._base.render.setLight(sun_np)

        self._world_root = self._base.render.attachNewNode("trace-world")
        self._cloud_root = self._world_root.attachNewNode("trace-clouds")
        self._terrain_root = self._world_root.attachNewNode("trace-terrain")
        self._guide_root = self._world_root.attachNewNode("trace-guides")
        self._aircraft_root = self._world_root.attachNewNode("trace-aircraft")

        self._target_anchor = self._aircraft_root.attachNewNode("trace-target-anchor")
        self._distractor_anchors = tuple(
            self._aircraft_root.attachNewNode(f"trace-distractor-anchor-{idx}") for idx in range(3)
        )

        self._target_aircraft = self._build_aircraft_model(
            body_color=(0.90, 0.24, 0.26, 1.0),
            canopy_color=(1.0, 0.92, 0.92, 1.0),
        )
        self._target_aircraft.reparentTo(self._target_anchor)
        # Keep the target readable even when terrain would otherwise occlude it.
        self._target_anchor.setBin("fixed", 30)
        self._target_anchor.setDepthTest(False)
        self._target_anchor.setDepthWrite(False)
        distractor_colours = (
            ((0.38, 0.50, 0.70, 1.0), (0.92, 0.96, 1.0, 1.0)),
            ((0.44, 0.46, 0.54, 1.0), (0.90, 0.94, 0.98, 1.0)),
            ((0.34, 0.46, 0.62, 1.0), (0.92, 0.95, 1.0, 1.0)),
        )
        self._distractor_aircraft = []
        for anchor, (body_colour, canopy_colour) in zip(
            self._distractor_anchors, distractor_colours, strict=False
        ):
            model = self._build_aircraft_model(
                body_color=body_colour,
                canopy_color=canopy_colour,
            )
            model.reparentTo(anchor)
            self._distractor_aircraft.append(model)

        self._build_world()

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
        reference: TraceTest1Attitude,
        candidate: TraceTest1Attitude,
        correct_code: int,
        viewpoint_bearing_deg: int,
        scene_turn_index: int = 0,
        animate: bool = True,
        progress: float | None = None,
    ) -> pygame.Surface:
        now_ms = pygame.time.get_ticks()
        dt_s = max(0.0, min(0.05, (now_ms - self._last_render_ms) / 1000.0))
        self._last_render_ms = now_ms
        self._elapsed_s += dt_s

        if progress is not None:
            scene_progress = max(0.0, min(1.0, float(progress)))
        elif animate:
            cycle_s = 7.2
            scene_progress = (self._elapsed_s / cycle_s) % 1.0
        else:
            scene_progress = 0.0

        target_frame, distractor_frames = trace_test_1_scene_frames(
            reference=reference,
            candidate=candidate,
            correct_code=correct_code,
            progress=scene_progress,
            scene_turn_index=scene_turn_index,
        )
        self._update_camera(target_frame=target_frame)
        self._update_aircraft(
            anchor=self._target_anchor,
            frame=target_frame,
            viewpoint_bearing_deg=viewpoint_bearing_deg,
            scale=1.08,
        )
        distractor_scales = (0.82, 0.76, 0.88)
        for anchor, frame, scale in zip(
            self._distractor_anchors, distractor_frames, distractor_scales, strict=False
        ):
            self._update_aircraft(
                anchor=anchor,
                frame=frame,
                viewpoint_bearing_deg=viewpoint_bearing_deg,
                scale=scale,
            )
        self._base.graphicsEngine.renderFrame()

        ram = self._texture.getRamImageAs("RGBA")
        raw = bytes(ram)
        surface = pygame.image.frombuffer(raw, self._size, "RGBA").copy()
        return pygame.transform.flip(surface, False, True)

    def _build_world(self) -> None:
        self._build_terrain()
        self._build_guides()
        self._clouds = [self._build_cloud(scale=s) for s in (1.0, 1.3, 0.8, 1.15, 0.9)]
        for idx, cloud in enumerate(self._clouds):
            cloud.reparentTo(self._cloud_root)
            ang = (idx / max(1, len(self._clouds))) * math.tau
            radius = 56.0 + (idx * 12.0)
            cloud.setPos(math.cos(ang) * radius, 50.0 + (idx * 16.0), 34.0 + (idx * 2.0))

    def _build_terrain(self) -> None:
        ground = self._make_box(size=(260.0, 360.0, 1.2), color=(0.48, 0.54, 0.60, 1.0))
        ground.setPos(0.0, 120.0, -18.5)
        ground.reparentTo(self._terrain_root)

        runway = self._make_box(size=(18.0, 240.0, 0.12), color=(0.30, 0.34, 0.40, 1.0))
        runway.setPos(0.0, 116.0, -17.82)
        runway.reparentTo(self._terrain_root)

        ridge_specs = (
            (-92.0, 168.0, -6.0, 58.0, 52.0, 22.0),
            (-40.0, 192.0, -4.5, 74.0, 64.0, 18.0),
            (30.0, 176.0, -5.0, 70.0, 58.0, 20.0),
            (84.0, 186.0, -6.4, 64.0, 54.0, 21.0),
        )
        for x, y, z, sx, sy, sz in ridge_specs:
            ridge = self._make_box(size=(sx, sy, sz), color=(0.38, 0.42, 0.48, 1.0))
            ridge.setPos(x, y, z)
            ridge.reparentTo(self._terrain_root)

    def _build_guides(self) -> None:
        from panda3d.core import LineSegs

        lines = LineSegs("trace-guides")
        lines.setThickness(1.4)
        lines.setColor(0.90, 0.92, 0.96, 0.34)

        for lane in (-3, -2, -1, 0, 1, 2, 3):
            x0 = lane * 7.0
            x1 = lane * 16.0
            lines.moveTo(x0, 18.0, -17.6)
            lines.drawTo(x1, 210.0, -17.6)

        guide_np = self._guide_root.attachNewNode(lines.create())
        guide_np.setTransparency(True)

    def _update_camera(self, *, target_frame: TraceTest1SceneFrame) -> None:
        # Follow the target with a zoomed-out side camera so it remains visible.
        tx, ty, tz = target_frame.position
        self._base.cam.setPos(tx - 220.0, ty - 14.0, tz + 96.0)
        self._base.cam.lookAt(tx + 30.0, ty, tz + 4.0)

    def _update_aircraft(
        self,
        *,
        anchor,
        frame: TraceTest1SceneFrame,
        viewpoint_bearing_deg: int,
        scale: float,
    ) -> None:
        _ = viewpoint_bearing_deg
        anchor.setPos(*frame.position)
        anchor.setScale(scale)
        anchor.setHpr(*self._aircraft_hpr_for_frame(frame))

    @staticmethod
    def _aircraft_hpr_for_frame(frame: TraceTest1SceneFrame) -> tuple[float, float, float]:
        attitude = frame.attitude
        return panda3d_fixed_wing_hpr(
            heading_deg=float(frame.travel_heading_deg),
            pitch_deg=float(attitude.pitch_deg),
            roll_deg=float(attitude.roll_deg),
        )

    def _build_cloud(self, *, scale: float):
        from panda3d.core import CardMaker, NodePath

        root = NodePath("trace-cloud")
        for idx, offset in enumerate((-1.0, 0.0, 1.0)):
            maker = CardMaker(f"trace-cloud-{idx}")
            maker.setFrame(-4.0, 4.0, -1.5, 1.5)
            card = root.attachNewNode(maker.generate())
            card.setBillboardPointEye()
            card.setTransparency(True)
            card.setColor(0.98, 0.98, 1.0, 0.28)
            card.setPos(offset * 4.0, idx * 0.12, (1 - idx) * 0.2)
        root.setScale(scale)
        return root

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
            ),
            name="trace-aircraft-model",
        )

    def _make_box(
        self,
        *,
        size: tuple[float, float, float],
        color: tuple[float, float, float, float],
    ):
        from panda3d.core import CardMaker, NodePath

        sx, sy, sz = size
        root = NodePath("trace-box")

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
