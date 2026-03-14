from __future__ import annotations

import importlib.util
import math
import os

import pygame

from .aircraft_art import (
    build_panda_palette,
    build_panda3d_fixed_wing_model,
    panda3d_fixed_wing_hpr_from_tangent,
)
from .trace_test_2 import (
    TraceTest2AircraftTrack,
    TraceTest2Payload,
    TraceTest2Point3,
    trace_test_2_track_position,
)


def panda3d_trace_test_2_rendering_available() -> bool:
    pref = os.environ.get("CFAST_TRACE_TEST_2_RENDERER", "panda").strip().lower()
    if pref in {"pygame", "2d", "off"}:
        return False
    if os.environ.get("SDL_VIDEODRIVER", "").strip().lower() == "dummy":
        return False
    return importlib.util.find_spec("direct.showbase.ShowBase") is not None


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(lo if v < lo else hi if v > hi else v)


def _loop_progress(elapsed_s: float, *, cycle_s: float) -> float:
    if cycle_s <= 1e-6:
        return 0.0
    return float((elapsed_s / cycle_s) % 1.0)


class TraceTest2Panda3DRenderer:
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
        self._base.setBackgroundColor(0.52, 0.67, 0.86, 1.0)
        self._texture = Texture()
        self._texture.setKeepRamImage(True)
        self._base.win.addRenderTexture(self._texture, GraphicsOutput.RTMCopyRam)
        self._base.camLens.setFov(43.0)
        self._base.camLens.setNearFar(0.1, 640.0)

        ambient = AmbientLight("trace2-ambient")
        ambient.setColor(Vec4(0.82, 0.86, 0.92, 1.0))
        ambient_np = self._base.render.attachNewNode(ambient)
        self._base.render.setLight(ambient_np)

        sun = DirectionalLight("trace2-sun")
        sun.setColor(Vec4(1.0, 0.98, 0.92, 1.0))
        sun_np = self._base.render.attachNewNode(sun)
        sun_np.setHpr(28.0, -38.0, 0.0)
        self._base.render.setLight(sun_np)

        self._world_root = self._base.render.attachNewNode("trace2-world")
        self._terrain_root = self._world_root.attachNewNode("trace2-terrain")
        self._cloud_root = self._world_root.attachNewNode("trace2-clouds")
        self._aircraft_root = self._world_root.attachNewNode("trace2-aircraft")
        self._aircraft_nodes: dict[int, object] = {}
        self._aircraft_orientation_by_code: dict[int, tuple[float, float, float]] = {}

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
            tracks = self._demo_tracks()
            progress = _loop_progress(self._elapsed_s, cycle_s=8.0)
            active_motion = True
        else:
            tracks = payload.aircraft
            progress = float(payload.observe_progress)
            active_motion = float(payload.observe_progress) < 1.0

        self._update_camera()
        self._update_clouds()
        self._update_aircraft(tracks=tracks, progress=progress, active_motion=active_motion)
        self._base.graphicsEngine.renderFrame()

        ram = self._texture.getRamImageAs("RGBA")
        raw = bytes(ram)
        surface = pygame.image.frombuffer(raw, self._size, "RGBA").copy()
        return pygame.transform.flip(surface, False, True)

    def _build_world(self) -> None:
        ground = self._make_box(size=(320.0, 420.0, 1.4), color=(0.44, 0.50, 0.58, 1.0))
        ground.setPos(0.0, 124.0, -20.6)
        ground.reparentTo(self._terrain_root)

        for x, y, z, sx, sy, sz in (
            (-102.0, 170.0, -7.0, 66.0, 52.0, 24.0),
            (-36.0, 196.0, -4.0, 82.0, 64.0, 18.0),
            (36.0, 182.0, -6.0, 72.0, 58.0, 20.0),
            (106.0, 192.0, -7.0, 70.0, 60.0, 23.0),
        ):
            ridge = self._make_box(size=(sx, sy, sz), color=(0.35, 0.40, 0.46, 1.0))
            ridge.setPos(x, y, z)
            ridge.reparentTo(self._terrain_root)

        runway = self._make_box(size=(24.0, 260.0, 0.16), color=(0.30, 0.34, 0.40, 1.0))
        runway.setPos(0.0, 112.0, -19.82)
        runway.reparentTo(self._terrain_root)

        self._clouds = [self._build_cloud(scale=s) for s in (1.0, 1.25, 0.82, 1.08)]
        for idx, cloud in enumerate(self._clouds):
            cloud.reparentTo(self._cloud_root)
            cloud.setPos((-54.0 + (idx * 34.0)), 44.0 + (idx * 16.0), 34.0 + (idx * 2.0))

    def _update_camera(self) -> None:
        self._base.cam.setPos(-146.0, 90.0, 18.0)
        self._base.cam.lookAt(34.0, 90.0, 18.0)

    def _update_clouds(self) -> None:
        for idx, cloud in enumerate(self._clouds):
            cloud.setPos((-54.0 + (idx * 34.0)), 44.0 + (idx * 16.0), 34.0 + (idx * 2.0))

    def _update_aircraft(
        self,
        *,
        tracks: tuple[TraceTest2AircraftTrack, ...],
        progress: float,
        active_motion: bool,
    ) -> None:
        active_codes = {int(track.code) for track in tracks}
        for code, node in self._aircraft_nodes.items():
            if code not in active_codes:
                node.hide()
        for code in tuple(self._aircraft_orientation_by_code):
            if code not in active_codes:
                del self._aircraft_orientation_by_code[code]

        for track in tracks:
            node = self._aircraft_nodes.get(int(track.code))
            if node is None:
                node = self._build_aircraft_model(color_rgb=track.color_rgb)
                node.reparentTo(self._aircraft_root)
                self._aircraft_nodes[int(track.code)] = node
            node.show()

            pos = trace_test_2_track_position(track=track, progress=progress)
            tangent = self._track_tangent(
                track=track,
                progress=progress,
                sample_step=(0.03 if active_motion else 0.018),
            )
            if tangent is None:
                hpr = self._aircraft_orientation_by_code.get(
                    int(track.code),
                    (0.0, 0.0, 0.0),
                )
            else:
                hpr = self._aircraft_hpr_from_tangent(
                    tangent=tangent,
                    default_hpr=self._aircraft_orientation_by_code.get(
                        int(track.code),
                        (0.0, 0.0, 0.0),
                    ),
                )
                self._aircraft_orientation_by_code[int(track.code)] = hpr

            node.setPos(float(pos.x), float(pos.y), float(pos.z))
            node.setHpr(*hpr)
            node.setScale(1.10)

    @staticmethod
    def _aircraft_hpr_from_tangent(
        *,
        tangent: tuple[float, float, float],
        default_hpr: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> tuple[float, float, float]:
        dx, dy, _dz = tangent
        horiz = max(1e-6, math.sqrt((dx * dx) + (dy * dy)))
        bank_deg = _clamp((dx / horiz) * 34.0, -40.0, 40.0)
        return panda3d_fixed_wing_hpr_from_tangent(
            tangent=tangent,
            bank_deg=bank_deg,
            default_hpr=default_hpr,
        )

    @staticmethod
    def _track_tangent(
        *,
        track: TraceTest2AircraftTrack,
        progress: float,
        sample_step: float,
    ) -> tuple[float, float, float] | None:
        t = _clamp(progress, 0.0, 1.0)
        step = max(0.005, float(sample_step))
        pos = trace_test_2_track_position(track=track, progress=t)
        ahead = trace_test_2_track_position(track=track, progress=min(1.0, t + step))
        dx = ahead.x - pos.x
        dy = ahead.y - pos.y
        dz = ahead.z - pos.z
        if (dx * dx) + (dy * dy) + (dz * dz) <= 1e-8:
            behind = trace_test_2_track_position(track=track, progress=max(0.0, t - step))
            dx = pos.x - behind.x
            dy = pos.y - behind.y
            dz = pos.z - behind.z
        if (dx * dx) + (dy * dy) + (dz * dz) <= 1e-8:
            return None
        return (float(dx), float(dy), float(dz))

    @staticmethod
    def _demo_tracks() -> tuple[TraceTest2AircraftTrack, ...]:
        return (
            TraceTest2AircraftTrack(
                code=1,
                color_name="Red",
                color_rgb=(228, 54, 56),
                waypoints=(
                    TraceTest2Point3(-34.0, 84.0, 14.0),
                    TraceTest2Point3(-10.0, 84.0, 14.0),
                    TraceTest2Point3(-10.0, 104.0, 14.0),
                    TraceTest2Point3(-32.0, 104.0, 13.0),
                ),
                left_turns=2,
                visible_fraction=0.0,
                visible_at_end=False,
                started_screen_y=0.0,
                ended_screen_x=-32.0,
            ),
            TraceTest2AircraftTrack(
                code=2,
                color_name="Blue",
                color_rgb=(76, 136, 244),
                waypoints=(
                    TraceTest2Point3(-10.0, 58.0, 1.5),
                    TraceTest2Point3(12.0, 58.0, 1.5),
                    TraceTest2Point3(34.0, 58.0, 4.0),
                ),
                left_turns=0,
                visible_fraction=0.0,
                visible_at_end=False,
                started_screen_y=0.0,
                ended_screen_x=34.0,
            ),
            TraceTest2AircraftTrack(
                code=3,
                color_name="Silver",
                color_rgb=(202, 208, 222),
                waypoints=(
                    TraceTest2Point3(-38.0, 98.0, 20.0),
                    TraceTest2Point3(-18.0, 98.0, 20.0),
                    TraceTest2Point3(-18.0, 114.0, 18.0),
                ),
                left_turns=1,
                visible_fraction=0.0,
                visible_at_end=False,
                started_screen_y=0.0,
                ended_screen_x=-18.0,
            ),
            TraceTest2AircraftTrack(
                code=4,
                color_name="Yellow",
                color_rgb=(236, 210, 92),
                waypoints=(
                    TraceTest2Point3(32.0, 92.0, 10.0),
                    TraceTest2Point3(14.0, 92.0, 10.0),
                    TraceTest2Point3(34.0, 106.0, 12.0),
                ),
                left_turns=0,
                visible_fraction=0.0,
                visible_at_end=False,
                started_screen_y=0.0,
                ended_screen_x=34.0,
            ),
        )

    def _build_cloud(self, *, scale: float):
        from panda3d.core import CardMaker, NodePath

        root = NodePath("trace2-cloud")
        for idx, offset in enumerate((-1.0, 0.0, 1.0)):
            maker = CardMaker(f"trace2-cloud-{idx}")
            maker.setFrame(-4.2, 4.2, -1.5, 1.5)
            card = root.attachNewNode(maker.generate())
            card.setBillboardPointEye()
            card.setTransparency(True)
            card.setColor(0.98, 0.98, 1.0, 0.25)
            card.setPos(offset * 4.2, idx * 0.12, (1 - idx) * 0.22)
        root.setScale(scale)
        return root

    def _build_aircraft_model(self, *, color_rgb: tuple[int, int, int]):
        body = tuple(channel / 255.0 for channel in color_rgb) + (1.0,)
        return build_panda3d_fixed_wing_model(
            palette=build_panda_palette(body_color=body),
            name="trace2-aircraft",
        )

    def _make_box(
        self,
        *,
        size: tuple[float, float, float],
        color: tuple[float, float, float, float],
    ):
        from panda3d.core import CardMaker, NodePath

        sx, sy, sz = size
        root = NodePath("trace2-box")

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
            "right",
            (-sy / 2.0, sy / 2.0, -sz / 2.0, sz / 2.0),
            (90.0, 0.0, 0.0),
            (sx / 2.0, 0.0, 0.0),
        )
        add_card(
            "left",
            (-sy / 2.0, sy / 2.0, -sz / 2.0, sz / 2.0),
            (-90.0, 0.0, 0.0),
            (-sx / 2.0, 0.0, 0.0),
        )
        return root
