from __future__ import annotations

import importlib.util
import math
import os

import pygame

from .spatial_integration import (
    SpatialIntegrationPayload,
    SpatialIntegrationSceneView,
)


def panda3d_spatial_integration_rendering_available() -> bool:
    pref = os.environ.get("CFAST_SPATIAL_INTEGRATION_RENDERER", "panda").strip().lower()
    if pref in {"pygame", "2d", "off"}:
        return False
    if os.environ.get("SDL_VIDEODRIVER", "").strip().lower() == "dummy":
        return False
    return importlib.util.find_spec("direct.showbase.ShowBase") is not None


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(lo if v < lo else hi if v > hi else v)


class SpatialIntegrationPanda3DRenderer:
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
        height = max(220, int(size[1]))
        self._size = (width, height)
        self._elapsed_s = 0.0
        self._last_render_ms = pygame.time.get_ticks()
        self._grid_signature = (0, 0)
        self._grid_node = None
        self._landmark_nodes: list[tuple[object, object]] = []
        self._aircraft_heading_deg = 0.0

        loadPrcFileData("", "window-type offscreen")
        loadPrcFileData("", "audio-library-name null")
        loadPrcFileData("", "sync-video false")
        loadPrcFileData("", f"win-size {width} {height}")

        from direct.showbase.ShowBase import ShowBase

        self._base = ShowBase(windowType="offscreen")
        self._base.disableMouse()
        self._base.setBackgroundColor(0.52, 0.69, 0.88, 1.0)
        self._texture = Texture()
        self._texture.setKeepRamImage(True)
        self._base.win.addRenderTexture(self._texture, GraphicsOutput.RTMCopyRam)
        self._base.camLens.setFov(52.0)
        self._base.camLens.setNearFar(0.1, 900.0)

        ambient = AmbientLight("spatial-ambient")
        ambient.setColor(Vec4(0.82, 0.86, 0.92, 1.0))
        ambient_np = self._base.render.attachNewNode(ambient)
        self._base.render.setLight(ambient_np)

        sun = DirectionalLight("spatial-sun")
        sun.setColor(Vec4(1.0, 0.98, 0.90, 1.0))
        sun_np = self._base.render.attachNewNode(sun)
        sun_np.setHpr(34.0, -38.0, 0.0)
        self._base.render.setLight(sun_np)

        self._world_root = self._base.render.attachNewNode("spatial-world")
        self._terrain_root = self._world_root.attachNewNode("spatial-terrain")
        self._grid_root = self._world_root.attachNewNode("spatial-grid")
        self._landmark_root = self._world_root.attachNewNode("spatial-landmarks")
        self._aircraft_root = self._world_root.attachNewNode("spatial-aircraft")
        self._cloud_root = self._world_root.attachNewNode("spatial-clouds")

        self._aircraft = self._build_aircraft_model(color=(0.28, 0.72, 0.88, 1.0))
        self._aircraft.reparentTo(self._aircraft_root)
        self._aircraft_prev = self._build_aircraft_model(color=(0.84, 0.88, 0.96, 0.44))
        self._aircraft_prev.reparentTo(self._aircraft_root)
        self._aircraft_prev.setTransparency(True)
        self._aircraft_pred = self._build_aircraft_model(color=(0.98, 0.94, 0.72, 0.44))
        self._aircraft_pred.reparentTo(self._aircraft_root)
        self._aircraft_pred.setTransparency(True)

        self._build_world()

    @property
    def size(self) -> tuple[int, int]:
        return self._size

    def close(self) -> None:
        try:
            self._base.destroy()
        except Exception:
            pass

    def render(self, *, payload: SpatialIntegrationPayload | None) -> pygame.Surface:
        now_ms = pygame.time.get_ticks()
        dt_s = max(0.0, min(0.05, (now_ms - self._last_render_ms) / 1000.0))
        self._last_render_ms = now_ms
        self._elapsed_s += dt_s

        if payload is None:
            scene_view = SpatialIntegrationSceneView.OBLIQUE
            grid_cols = 5
            grid_rows = 5
            alt_levels = 4
            landmarks = (("HGR", 1, 0), ("TWR", 3, 1), ("WDM", 4, 2), ("VLG", 0, 3))
            query_label = "TWR"
            now_point = (2, 1, 1)
            prev_point = (1, 0, 1)
            velocity = (1, 1, 0)
            show_motion = True
        else:
            scene_view = payload.scene_view
            grid_cols = max(1, int(payload.grid_cols))
            grid_rows = max(1, int(payload.grid_rows))
            alt_levels = max(1, int(payload.alt_levels))
            landmarks = tuple(
                (str(landmark.label), int(landmark.x), int(landmark.y))
                for landmark in payload.landmarks
            )
            query_label = str(payload.query_label)
            now_point = (
                int(payload.aircraft_now.x),
                int(payload.aircraft_now.y),
                int(payload.aircraft_now.z),
            )
            prev_point = (
                int(payload.aircraft_prev.x),
                int(payload.aircraft_prev.y),
                int(payload.aircraft_prev.z),
            )
            velocity = (
                int(payload.velocity.dx),
                int(payload.velocity.dy),
                int(payload.velocity.dz),
            )
            show_motion = bool(payload.show_aircraft_motion) and prev_point != now_point

        self._update_grid(grid_cols=grid_cols, grid_rows=grid_rows, alt_levels=alt_levels)
        self._update_landmarks(
            landmarks=landmarks,
            query_label=query_label,
            grid_cols=grid_cols,
            grid_rows=grid_rows,
            alt_levels=alt_levels,
        )
        focus = self._update_aircraft(
            now_point=now_point,
            prev_point=prev_point,
            velocity=velocity,
            show_motion=show_motion,
            grid_cols=grid_cols,
            grid_rows=grid_rows,
            alt_levels=alt_levels,
        )
        self._update_camera(scene_view=scene_view, focus=focus)
        self._update_clouds()
        self._base.graphicsEngine.renderFrame()

        ram = self._texture.getRamImageAs("RGBA")
        raw = bytes(ram)
        surface = pygame.image.frombuffer(raw, self._size, "RGBA").copy()
        return pygame.transform.flip(surface, False, True)

    def _build_world(self) -> None:
        ground = self._make_box(size=(240.0, 320.0, 1.5), color=(0.48, 0.56, 0.44, 1.0))
        ground.setPos(0.0, 128.0, -2.6)
        ground.reparentTo(self._terrain_root)

        for x, y, z, sx, sy, sz in (
            (-86.0, 210.0, 10.0, 96.0, 54.0, 22.0),
            (18.0, 208.0, 8.0, 106.0, 62.0, 26.0),
            (96.0, 182.0, 7.0, 84.0, 58.0, 20.0),
            (-120.0, 164.0, 8.0, 72.0, 52.0, 18.0),
        ):
            ridge = self._make_box(size=(sx, sy, sz), color=(0.44, 0.48, 0.42, 1.0))
            ridge.setPos(x, y, z)
            ridge.reparentTo(self._terrain_root)

        runway = self._make_box(size=(22.0, 196.0, 0.15), color=(0.34, 0.36, 0.40, 1.0))
        runway.setPos(0.0, 118.0, 0.05)
        runway.reparentTo(self._terrain_root)

        self._clouds = [self._build_cloud(scale=s) for s in (0.95, 1.22, 0.84, 1.08, 0.90)]
        for idx, cloud in enumerate(self._clouds):
            cloud.reparentTo(self._cloud_root)
            cloud.setPos(-80.0 + (idx * 38.0), 34.0 + (idx * 18.0), 66.0 + (idx * 3.0))

    def _update_grid(self, *, grid_cols: int, grid_rows: int, alt_levels: int) -> None:
        _ = alt_levels
        signature = (int(grid_cols), int(grid_rows))
        if signature == self._grid_signature and self._grid_node is not None:
            return

        from panda3d.core import LineSegs

        if self._grid_node is not None:
            self._grid_node.removeNode()

        lines = LineSegs("spatial-grid-lines")
        lines.setThickness(1.2)
        lines.setColor(0.84, 0.90, 0.98, 0.22)

        cols = max(1, int(grid_cols))
        rows = max(1, int(grid_rows))
        for col in range(cols):
            x0, y0, _, t0 = self._grid_to_world(
                x=col,
                y=0,
                z=0,
                grid_cols=cols,
                grid_rows=rows,
                alt_levels=3,
            )
            x1, y1, _, t1 = self._grid_to_world(
                x=col,
                y=rows - 1,
                z=0,
                grid_cols=cols,
                grid_rows=rows,
                alt_levels=3,
            )
            lines.moveTo(x0, y0, t0 + 0.22)
            lines.drawTo(x1, y1, t1 + 0.22)

        for row in range(rows):
            x0, y0, _, t0 = self._grid_to_world(
                x=0,
                y=row,
                z=0,
                grid_cols=cols,
                grid_rows=rows,
                alt_levels=3,
            )
            x1, y1, _, t1 = self._grid_to_world(
                x=cols - 1,
                y=row,
                z=0,
                grid_cols=cols,
                grid_rows=rows,
                alt_levels=3,
            )
            lines.moveTo(x0, y0, t0 + 0.22)
            lines.drawTo(x1, y1, t1 + 0.22)

        self._grid_node = self._grid_root.attachNewNode(lines.create())
        self._grid_node.setTransparency(True)
        self._grid_signature = signature

    def _update_landmarks(
        self,
        *,
        landmarks: tuple[tuple[str, int, int], ...],
        query_label: str,
        grid_cols: int,
        grid_rows: int,
        alt_levels: int,
    ) -> None:
        while len(self._landmark_nodes) < len(landmarks):
            marker, label_node = self._build_landmark_marker()
            marker.reparentTo(self._landmark_root)
            self._landmark_nodes.append((marker, label_node))

        for idx, (label, gx, gy) in enumerate(landmarks):
            marker, label_node = self._landmark_nodes[idx]
            wx, wy, _, terrain = self._grid_to_world(
                x=gx,
                y=gy,
                z=0,
                grid_cols=grid_cols,
                grid_rows=grid_rows,
                alt_levels=alt_levels,
            )
            marker.show()
            marker.setPos(wx, wy, terrain + 0.16)
            is_query = str(label).upper() == str(query_label).upper()
            if is_query:
                marker.setColorScale(1.06, 1.00, 0.74, 1.0)
                label_node.node().setTextColor(0.98, 0.94, 0.70, 1.0)
            else:
                marker.setColorScale(0.96, 0.98, 1.0, 1.0)
                label_node.node().setTextColor(0.94, 0.97, 1.0, 1.0)
            label_node.node().setText(str(label))

        for marker, _label_node in self._landmark_nodes[len(landmarks) :]:
            marker.hide()

    def _update_aircraft(
        self,
        *,
        now_point: tuple[int, int, int],
        prev_point: tuple[int, int, int],
        velocity: tuple[int, int, int],
        show_motion: bool,
        grid_cols: int,
        grid_rows: int,
        alt_levels: int,
    ) -> tuple[float, float, float]:
        now_wx, now_wy, now_wz, _ = self._grid_to_world(
            x=now_point[0],
            y=now_point[1],
            z=now_point[2],
            grid_cols=grid_cols,
            grid_rows=grid_rows,
            alt_levels=alt_levels,
        )
        prev_wx, prev_wy, prev_wz, _ = self._grid_to_world(
            x=prev_point[0],
            y=prev_point[1],
            z=prev_point[2],
            grid_cols=grid_cols,
            grid_rows=grid_rows,
            alt_levels=alt_levels,
        )

        live_x = now_wx
        live_y = now_wy
        live_z = now_wz
        if show_motion:
            phase = (self._elapsed_s * 0.32) % 1.0
            future_x = now_wx + (now_wx - prev_wx)
            future_y = now_wy + (now_wy - prev_wy)
            future_z = now_wz + (now_wz - prev_wz)
            if phase < 0.5:
                t = phase / 0.5
                live_x = prev_wx + ((now_wx - prev_wx) * t)
                live_y = prev_wy + ((now_wy - prev_wy) * t)
                live_z = prev_wz + ((now_wz - prev_wz) * t)
            else:
                t = (phase - 0.5) / 0.5
                live_x = now_wx + ((future_x - now_wx) * t)
                live_y = now_wy + ((future_y - now_wy) * t)
                live_z = now_wz + ((future_z - now_wz) * t)

            self._aircraft_prev.show()
            self._aircraft_prev.setPos(prev_wx, prev_wy, prev_wz)
            self._aircraft_pred.show()
            self._aircraft_pred.setPos(
                now_wx + (0.60 * velocity[0]),
                now_wy + (0.60 * velocity[1]),
                now_wz + (0.48 * velocity[2]),
            )
        else:
            self._aircraft_prev.hide()
            self._aircraft_pred.hide()

        dir_x = live_x - prev_wx
        dir_y = live_y - prev_wy
        dir_z = live_z - prev_wz
        if (dir_x * dir_x) + (dir_y * dir_y) + (dir_z * dir_z) > 1e-8:
            horiz = max(1e-6, math.sqrt((dir_x * dir_x) + (dir_y * dir_y)))
            self._aircraft_heading_deg = math.degrees(math.atan2(dir_x, dir_y))
            pitch_deg = -math.degrees(math.atan2(dir_z, horiz))
            bank_deg = _clamp((dir_x / horiz) * 26.0, -34.0, 34.0)
        else:
            pitch_deg = 0.0
            bank_deg = 0.0

        bob = 0.32 * math.sin(self._elapsed_s * 3.1)
        self._aircraft.setPos(live_x, live_y, live_z + bob)
        self._aircraft.setHpr(self._aircraft_heading_deg, pitch_deg, bank_deg)
        self._aircraft.setScale(1.24)
        return (live_x, live_y, live_z)

    def _update_camera(
        self,
        *,
        scene_view: SpatialIntegrationSceneView,
        focus: tuple[float, float, float],
    ) -> None:
        fx, fy, fz = focus
        if scene_view is SpatialIntegrationSceneView.TOPDOWN:
            self._base.camLens.setFov(44.0)
            self._base.cam.setPos(fx * 0.22, fy + 8.0, 244.0)
            self._base.cam.lookAt(fx * 0.22, fy, max(0.0, fz - 8.0))
            return
        self._base.camLens.setFov(52.0)
        self._base.cam.setPos(-184.0, 34.0, 88.0)
        self._base.cam.lookAt(fx * 0.35, fy + 34.0, max(8.0, fz + 8.0))

    def _update_clouds(self) -> None:
        for idx, cloud in enumerate(self._clouds):
            drift = (self._elapsed_s * (4.0 + (idx * 0.7))) % 240.0
            cloud.setPos(-120.0 + drift, 28.0 + (idx * 20.0), 64.0 + (idx * 2.8))

    def _grid_to_world(
        self,
        *,
        x: int,
        y: int,
        z: int,
        grid_cols: int,
        grid_rows: int,
        alt_levels: int,
    ) -> tuple[float, float, float, float]:
        cols = max(1, int(grid_cols))
        rows = max(1, int(grid_rows))
        levels = max(1, int(alt_levels))

        x_norm = 0.5 if cols <= 1 else float(x) / float(cols - 1)
        y_norm = 0.5 if rows <= 1 else float(y) / float(rows - 1)
        z_norm = 0.0 if levels <= 1 else float(z) / float(levels - 1)

        wx = (x_norm - 0.5) * 72.0
        wy = 44.0 + (y_norm * 176.0)
        terrain = self._terrain_height(wx=wx, wy=wy)
        wz = terrain + 2.4 + (z_norm * 24.0)
        return wx, wy, wz, terrain

    def _terrain_height(self, *, wx: float, wy: float) -> float:
        ridge = 4.2 * math.sin((wx * 0.030) + 0.7) * math.exp(-((wy - 126.0) ** 2) * 0.00008)
        hill_a = 7.6 * math.exp(-(((wx + 24.0) ** 2) * 0.0018) - (((wy - 98.0) ** 2) * 0.00060))
        hill_b = 6.8 * math.exp(-(((wx - 30.0) ** 2) * 0.0016) - (((wy - 152.0) ** 2) * 0.00048))
        return float(ridge + hill_a + hill_b)

    def _build_cloud(self, *, scale: float):
        from panda3d.core import CardMaker, NodePath

        root = NodePath("spatial-cloud")
        for idx, offset in enumerate((-1.0, 0.0, 1.0)):
            maker = CardMaker(f"spatial-cloud-{idx}")
            maker.setFrame(-5.0, 5.0, -1.8, 1.8)
            card = root.attachNewNode(maker.generate())
            card.setBillboardPointEye()
            card.setTransparency(True)
            card.setColor(0.98, 0.98, 1.0, 0.26)
            card.setPos(offset * 5.2, idx * 0.15, (1 - idx) * 0.30)
        root.setScale(scale)
        return root

    def _build_landmark_marker(self):
        from panda3d.core import TextNode

        marker = self._make_box(size=(2.2, 2.2, 0.7), color=(0.92, 0.94, 0.98, 1.0))
        pole = self._make_box(size=(0.32, 0.32, 2.4), color=(0.94, 0.88, 0.52, 1.0))
        pole.setPos(0.0, 0.0, 1.4)
        pole.reparentTo(marker)

        label = TextNode("spatial-landmark-label")
        label.setText("LMK")
        label.setAlign(TextNode.ACenter)
        label.setTextColor(0.94, 0.97, 1.0, 1.0)
        label_np = marker.attachNewNode(label)
        label_np.setPos(0.0, 0.0, 4.8)
        label_np.setScale(8.0)
        label_np.setBillboardPointEye()
        label_np.setDepthTest(False)
        label_np.setDepthWrite(False)
        return marker, label_np

    def _build_aircraft_model(self, *, color: tuple[float, float, float, float]):
        from panda3d.core import NodePath

        root = NodePath("spatial-aircraft-model")
        fuselage = self._make_box(size=(1.0, 7.2, 0.9), color=color)
        wing = self._make_box(size=(9.2, 1.0, 0.20), color=color)
        tail_plane = self._make_box(size=(3.1, 0.7, 0.18), color=color)
        fin = self._make_box(size=(0.18, 1.0, 1.0), color=color)
        canopy = self._make_box(size=(0.64, 1.4, 0.44), color=(0.94, 0.97, 1.0, color[3]))
        nose = self._make_box(size=(0.64, 0.60, 0.56), color=(0.94, 0.97, 1.0, color[3]))

        wing.setY(-0.18)
        wing.setZ(0.05)
        tail_plane.setY(-2.48)
        tail_plane.setZ(0.26)
        fin.setY(-2.74)
        fin.setZ(0.70)
        canopy.setY(1.65)
        canopy.setZ(0.36)
        nose.setY(3.4)
        nose.setZ(0.12)

        for child in (fuselage, wing, tail_plane, fin, canopy, nose):
            child.reparentTo(root)
        return root

    def _make_box(
        self,
        *,
        size: tuple[float, float, float],
        color: tuple[float, float, float, float],
    ):
        from panda3d.core import CardMaker, NodePath

        sx, sy, sz = size
        root = NodePath("spatial-box")

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
