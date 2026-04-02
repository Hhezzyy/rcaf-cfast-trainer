from __future__ import annotations

import importlib.util
import math
import os

import pygame

from .auditory_capacity import (
    AUDITORY_GATE_PLAYER_X_NORM,
    AUDITORY_GATE_RETIRE_X_NORM,
    AUDITORY_GATE_SPAWN_X_NORM,
    AUDITORY_TRIANGLE_GATE_POINTS,
    AuditoryCapacityPayload,
)
from .auditory_capacity_view import (
    BALL_FORWARD_IDLE_NORM,
    TUBE_PATH_POINTS as _TUBE_PATH_POINTS,
    TUBE_PATH_SPAN,
    TUNNEL_CAMERA_H_FOV_DEG,
    TUNNEL_GEOMETRY_END_DISTANCE,
    TUNNEL_GEOMETRY_START_DISTANCE,
    fixed_camera_pose,
    forward_norm_to_distance,
    slot_distance,
    tube_center_at_distance as _tube_center_at_distance,
    tube_frame as _tube_frame,
    vec_add as _vec_add,
    vec_cross as _vec_cross,
    vec_dot as _vec_dot,
    vec_norm as _vec_norm,
    vec_normalize as _vec_normalize,
    vec_scale as _vec_scale,
)
from .panda3d_assets import Panda3DAssetCatalog

def panda3d_auditory_rendering_available() -> bool:
    if os.environ.get("SDL_VIDEODRIVER", "").strip().lower() == "dummy":
        return False
    return importlib.util.find_spec("direct.showbase.ShowBase") is not None


def _mix_rgb(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    *,
    mix: float,
) -> tuple[float, float, float]:
    m = max(0.0, min(1.0, float(mix)))
    return (
        (a[0] * (1.0 - m)) + (b[0] * m),
        (a[1] * (1.0 - m)) + (b[1] * m),
        (a[2] * (1.0 - m)) + (b[2] * m),
    )


class AuditoryCapacityPanda3DRenderer:
    _GATE_SPAWN_X_NORM = AUDITORY_GATE_SPAWN_X_NORM
    _GATE_PLAYER_X_NORM = AUDITORY_GATE_PLAYER_X_NORM
    _GATE_RETIRE_X_NORM = AUDITORY_GATE_RETIRE_X_NORM
    _TUNNEL_SEGMENT_LENGTH = 4.0
    _TUNNEL_RIB_STEP = 4.0
    _BALL_COLOR_MAP = {
        "RED": (0.94, 0.34, 0.38),
        "GREEN": (0.34, 0.88, 0.56),
        "BLUE": (0.36, 0.62, 0.94),
        "YELLOW": (0.94, 0.82, 0.34),
        "WHITE": (0.94, 0.96, 1.0),
    }

    def __init__(self, *, size: tuple[int, int]) -> None:
        from panda3d.core import (
            AmbientLight,
            Fog,
            DirectionalLight,
            GraphicsOutput,
            Texture,
            Vec4,
            loadPrcFileData,
        )

        width = max(320, int(size[0]))
        height = max(200, int(size[1]))
        self._size = (width, height)
        self._span = float(TUBE_PATH_SPAN)
        self._tube_rx = 2.24
        self._tube_rz = 1.64
        self._ring_count = 118
        self._ring_steps = 72
        self._travel_offset = 0.0
        self._tube_geometry_start_distance = float(TUNNEL_GEOMETRY_START_DISTANCE)
        self._tube_geometry_end_distance = float(TUNNEL_GEOMETRY_END_DISTANCE)
        self._last_render_ms = pygame.time.get_ticks()
        self._gate_nodes: dict[int, object] = {}
        self._asset_catalog = Panda3DAssetCatalog()
        self._loaded_asset_ids: set[str] = set()
        self._fallback_asset_ids: set[str] = set()

        loadPrcFileData("", "window-type offscreen")
        loadPrcFileData("", "audio-library-name null")
        loadPrcFileData("", "sync-video false")
        loadPrcFileData("", f"win-size {width} {height}")

        from direct.showbase.ShowBase import ShowBase

        self._base = ShowBase(windowType="offscreen")
        self._base.disableMouse()
        self._base.setBackgroundColor(0.01, 0.02, 0.08, 1.0)
        self._prototype_root = self._base.render.attachNewNode("auditory-prototypes")
        self._prototype_root.hide()
        self._texture = Texture()
        self._texture.setKeepRamImage(True)
        self._base.win.addRenderTexture(self._texture, GraphicsOutput.RTMCopyRam)

        self._base.camLens.setFov(float(TUNNEL_CAMERA_H_FOV_DEG))

        ambient = AmbientLight("auditory-ambient")
        ambient.setColor(Vec4(0.82, 0.84, 0.92, 1.0))
        ambient_np = self._base.render.attachNewNode(ambient)
        self._base.render.setLight(ambient_np)

        sun = DirectionalLight("auditory-sun")
        sun.setColor(Vec4(0.92, 0.95, 1.0, 1.0))
        sun_np = self._base.render.attachNewNode(sun)
        sun_np.setHpr(18.0, -24.0, 0.0)
        self._base.render.setLight(sun_np)

        fog = Fog("auditory-fog")
        fog.setColor(0.01, 0.02, 0.08)
        fog.setExpDensity(0.038)
        self._base.render.setFog(fog)

        self._tube_root = self._base.render.attachNewNode("tube-root")
        self._gate_root = self._base.render.attachNewNode("gate-root")
        self._ball_root = self._base.render.attachNewNode("ball-root")
        self._build_tube()
        self._build_ball()

    @property
    def size(self) -> tuple[int, int]:
        return self._size

    def close(self) -> None:
        try:
            self._base.destroy()
        except Exception:
            pass

    @property
    def loaded_asset_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._loaded_asset_ids))

    @property
    def fallback_asset_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._fallback_asset_ids))

    def render(
        self,
        *,
        payload: AuditoryCapacityPayload | None,
        advance_animation: bool = True,
    ) -> pygame.Surface:
        _ = advance_animation
        self._last_render_ms = pygame.time.get_ticks()
        self._update_ball(payload=payload)
        self._update_gates(payload=payload)
        self._base.graphicsEngine.renderFrame()

        ram = self._texture.getRamImageAs("RGBA")
        raw = bytes(ram)
        surface = pygame.image.frombuffer(raw, self._size, "RGBA").copy()
        return pygame.transform.flip(surface, False, True)

    def _build_tube(self) -> None:
        segment_proto = self._load_catalog_model(asset_id="auditory_tunnel_segment")
        rib_proto = self._load_catalog_model(asset_id="auditory_tunnel_rib")
        if segment_proto is not None and rib_proto is not None:
            self._build_asset_tube(segment_proto=segment_proto, rib_proto=rib_proto)
            return
        self._build_fallback_tube()

    def _build_asset_tube(self, *, segment_proto, rib_proto) -> None:
        from panda3d.core import Point3, Vec3

        geometry_span = self._tube_geometry_end_distance - self._tube_geometry_start_distance
        segment_count = int(math.ceil(geometry_span / self._TUNNEL_SEGMENT_LENGTH))
        for idx in range(segment_count):
            distance = (
                self._tube_geometry_start_distance
                + (idx * self._TUNNEL_SEGMENT_LENGTH)
                + (self._TUNNEL_SEGMENT_LENGTH * 0.5)
            )
            center, tangent, _right, up = _tube_frame(distance, span=self._span)
            anchor = self._tube_root.attachNewNode(f"tube-segment-{idx}")
            segment_proto.instanceTo(anchor)
            anchor.setPos(*center)
            anchor.lookAt(
                Point3(*(center[0] + tangent[0], center[1] + tangent[1], center[2] + tangent[2])),
                Vec3(*up),
            )
            anchor.setScale(self._tube_rx, 1.0, self._tube_rz)
            anchor.setTwoSided(True)
            anchor.setLightOff()
            anchor.setColorScale(0.08, 0.14, 0.38, 1.0)

        rib_count = int(math.ceil(geometry_span / self._TUNNEL_RIB_STEP)) + 1
        for idx in range(rib_count):
            distance = self._tube_geometry_start_distance + (idx * self._TUNNEL_RIB_STEP)
            center, tangent, _right, up = _tube_frame(distance, span=self._span)
            anchor = self._tube_root.attachNewNode(f"tube-rib-{idx}")
            rib_proto.instanceTo(anchor)
            anchor.setPos(*center)
            anchor.lookAt(
                Point3(*(center[0] + tangent[0], center[1] + tangent[1], center[2] + tangent[2])),
                Vec3(*up),
            )
            anchor.setScale(self._tube_rx, 1.0, self._tube_rz)
            anchor.setTwoSided(True)
            anchor.setLightOff()
            anchor.setColorScale(0.34, 0.52, 0.98, 1.0)

    def _load_catalog_model(self, *, asset_id: str):
        entry = self._asset_catalog.entry(asset_id)
        resolved = self._asset_catalog.resolve_path(asset_id)
        if resolved is not None:
            try:
                model = self._base.loader.loadModel(str(resolved))
                if not model.isEmpty():
                    model.reparentTo(self._prototype_root)
                    if entry is not None:
                        entry.apply_loaded_model_transform(
                            model,
                            pos=(0.0, 0.0, 0.0),
                            hpr=(0.0, 0.0, 0.0),
                            scale=1.0,
                        )
                    self._loaded_asset_ids.add(asset_id)
                    return model
            except Exception:
                pass
        self._fallback_asset_ids.add(asset_id)
        return None

    def _build_fallback_tube(self) -> None:
        from panda3d.core import (
            Geom,
            GeomNode,
            GeomTriangles,
            GeomVertexData,
            GeomVertexFormat,
            GeomVertexWriter,
            LineSegs,
        )

        fmt = GeomVertexFormat.getV3n3t2()
        vdata = GeomVertexData("auditory-tube", fmt, Geom.UHStatic)
        vertex = GeomVertexWriter(vdata, "vertex")
        normal = GeomVertexWriter(vdata, "normal")
        texcoord = GeomVertexWriter(vdata, "texcoord")
        tris = GeomTriangles(Geom.UHStatic)

        geometry_span = self._tube_geometry_end_distance - self._tube_geometry_start_distance
        next_index = 0
        for ring_idx in range(self._ring_count - 1):
            d0 = self._tube_geometry_start_distance + (
                (ring_idx / float(self._ring_count - 1)) * geometry_span
            )
            d1 = self._tube_geometry_start_distance + (
                ((ring_idx + 1) / float(self._ring_count - 1)) * geometry_span
            )
            c0, _t0, r0, u0 = _tube_frame(d0, span=self._span)
            c1, _t1, r1, u1 = _tube_frame(d1, span=self._span)
            for seg_idx in range(self._ring_steps):
                a0 = (seg_idx / float(self._ring_steps)) * math.tau
                a1 = ((seg_idx + 1) / float(self._ring_steps)) * math.tau
                radial00 = _vec_add(
                    _vec_scale(r0, math.cos(a0) * self._tube_rx),
                    _vec_scale(u0, math.sin(a0) * self._tube_rz),
                )
                radial01 = _vec_add(
                    _vec_scale(r0, math.cos(a1) * self._tube_rx),
                    _vec_scale(u0, math.sin(a1) * self._tube_rz),
                )
                radial10 = _vec_add(
                    _vec_scale(r1, math.cos(a1) * self._tube_rx),
                    _vec_scale(u1, math.sin(a1) * self._tube_rz),
                )
                radial11 = _vec_add(
                    _vec_scale(r1, math.cos(a0) * self._tube_rx),
                    _vec_scale(u1, math.sin(a0) * self._tube_rz),
                )
                p00 = _vec_add(c0, radial00)
                p01 = _vec_add(c0, radial01)
                p10 = _vec_add(c1, radial10)
                p11 = _vec_add(c1, radial11)
                n00 = _vec_normalize(_vec_scale(radial00, -1.0))
                n01 = _vec_normalize(_vec_scale(radial01, -1.0))
                n10 = _vec_normalize(_vec_scale(radial10, -1.0))
                n11 = _vec_normalize(_vec_scale(radial11, -1.0))
                t0 = (d0 - self._tube_geometry_start_distance) / geometry_span
                t1 = (d1 - self._tube_geometry_start_distance) / geometry_span
                u_start = seg_idx / float(self._ring_steps)
                u_end = (seg_idx + 1) / float(self._ring_steps)
                for point, norm, uv in (
                    (p00, n00, (u_start, t0)),
                    (p01, n01, (u_end, t0)),
                    (p10, n10, (u_end, t1)),
                    (p11, n11, (u_start, t1)),
                ):
                    vertex.addData3f(*point)
                    normal.addData3f(*norm)
                    texcoord.addData2f(*uv)
                tris.addVertices(next_index, next_index + 1, next_index + 2)
                tris.addVertices(next_index, next_index + 2, next_index + 3)
                next_index += 4

        geom = Geom(vdata)
        geom.addPrimitive(tris)
        geom_node = GeomNode("auditory-tube-geom")
        geom_node.addGeom(geom)
        tube = self._tube_root.attachNewNode(geom_node)
        tube.setTwoSided(True)
        tube.setTexture(self._build_vortex_texture())

        rings = LineSegs("auditory-rings")
        rings.setThickness(1.0)
        for ring_idx in range(self._ring_count):
            if ring_idx % 6 != 0:
                continue
            d = self._tube_geometry_start_distance + (
                (ring_idx / float(self._ring_count - 1)) * geometry_span
            )
            center, _tangent, right, up = _tube_frame(d, span=self._span)
            t = (d - self._tube_geometry_start_distance) / geometry_span
            lum = 0.52 - (0.30 * t)
            rings.setColor(0.04 * lum, 0.18 * lum, 0.70 * lum, 0.16)
            first_point: tuple[float, float, float] | None = None
            prev_point: tuple[float, float, float] | None = None
            for seg_idx in range(self._ring_steps + 1):
                angle = (seg_idx / float(self._ring_steps)) * math.tau
                point = _vec_add(
                    center,
                    _vec_add(
                        _vec_scale(right, math.cos(angle) * self._tube_rx),
                        _vec_scale(up, math.sin(angle) * self._tube_rz),
                    ),
                )
                if first_point is None:
                    first_point = point
                if prev_point is not None:
                    rings.moveTo(*prev_point)
                    rings.drawTo(*point)
                prev_point = point
            if first_point is not None and prev_point is not None:
                rings.moveTo(*prev_point)
                rings.drawTo(*first_point)
        ring_np = self._tube_root.attachNewNode(rings.create())
        ring_np.setLightOff()

    def _build_vortex_texture(self):
        from panda3d.core import PNMImage, Texture

        size = 512
        image = PNMImage(size, size, 4)
        for y in range(size):
            v = y / float(size - 1)
            for x in range(size):
                u = x / float(size - 1)
                angle = u * math.tau
                swirl = angle + (v * math.tau * 2.8)
                streak = 0.5 + (0.5 * math.sin((swirl * 5.2) + (v * 18.0)))
                wave = 0.5 + (0.5 * math.cos((swirl * 2.6) - (v * 11.0)))
                wall = 0.35 + (0.65 * abs(math.cos(angle)))
                r = 0.01 + (0.01 * wave)
                g = 0.05 + (0.08 * wall) + (0.05 * streak)
                b = 0.16 + (0.18 * wall) + (0.20 * streak)
                image.setXelA(x, y, r, g, b, 1.0)

        texture = Texture("auditory-vortex")
        texture.load(image)
        texture.setWrapU(Texture.WM_repeat)
        texture.setWrapV(Texture.WM_repeat)
        texture.setMinfilter(Texture.FT_linear_mipmap_linear)
        texture.setMagfilter(Texture.FT_linear)
        return texture

    def _build_ball(self) -> None:
        sphere = self._load_catalog_model(asset_id="auditory_ball")
        if sphere is not None:
            sphere.instanceTo(self._ball_root)
        else:
            self._load_ball_model().reparentTo(self._ball_root)
        self._ball_root.setScale(0.11)
        self._ball_root.setColor(0.94, 0.96, 1.0, 1.0)

    def _load_ball_model(self):
        try:
            model = self._base.loader.loadModel("models/misc/sphere")
            if not model.isEmpty():
                return model
        except Exception:
            pass
        return self._make_uv_sphere(radius=1.0, color=(0.94, 0.96, 1.0, 1.0))

    def _make_uv_sphere(
        self,
        *,
        radius: float,
        color: tuple[float, float, float, float],
        lat_steps: int = 18,
        lon_steps: int = 30,
    ):
        from panda3d.core import (
            Geom,
            GeomNode,
            GeomTriangles,
            GeomVertexData,
            GeomVertexFormat,
            GeomVertexWriter,
            NodePath,
        )

        fmt = GeomVertexFormat.getV3n3c4()
        vdata = GeomVertexData("auditory-sphere", fmt, Geom.UHStatic)
        vertex = GeomVertexWriter(vdata, "vertex")
        normal = GeomVertexWriter(vdata, "normal")
        color_writer = GeomVertexWriter(vdata, "color")
        tris = GeomTriangles(Geom.UHStatic)

        ring_vertices = lon_steps + 1
        for lat_idx in range(lat_steps + 1):
            v = lat_idx / float(lat_steps)
            phi = (v * math.pi) - (math.pi / 2.0)
            ring_r = math.cos(phi) * radius
            y = math.sin(phi) * radius
            for lon_idx in range(lon_steps + 1):
                u = lon_idx / float(lon_steps)
                theta = u * math.tau
                x = math.cos(theta) * ring_r
                z = math.sin(theta) * ring_r
                vertex.addData3f(x, y, z)
                nx, ny, nz = _vec_normalize((x, y, z))
                normal.addData3f(nx, ny, nz)
                color_writer.addData4f(*color)

        for lat_idx in range(lat_steps):
            for lon_idx in range(lon_steps):
                a = (lat_idx * ring_vertices) + lon_idx
                b = a + 1
                c = a + ring_vertices
                d = c + 1
                tris.addVertices(a, c, b)
                tris.addVertices(b, c, d)

        geom = Geom(vdata)
        geom.addPrimitive(tris)
        geom_node = GeomNode("auditory-sphere-geom")
        geom_node.addGeom(geom)
        root = NodePath("auditory-sphere")
        root.attachNewNode(geom_node)
        return root

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

    def _update_ball(self, *, payload: AuditoryCapacityPayload | None) -> None:
        forward_norm = (
            float(payload.ball_forward_norm) if payload is not None else float(BALL_FORWARD_IDLE_NORM)
        )
        distance = forward_norm_to_distance(forward_norm)
        center, tangent, right, up = _tube_frame(distance, span=self._span)
        ball_pos = center
        danger = False
        if payload is not None:
            x_half_span = max(0.08, float(payload.tube_half_width))
            z_half_span = max(0.08, float(payload.tube_half_height))
            local_x = max(-1.0, min(1.0, payload.ball_x / x_half_span)) * (self._tube_rx * 0.72)
            local_z = max(-1.0, min(1.0, payload.ball_y / z_half_span)) * (self._tube_rz * 0.72)
            ball_pos = _vec_add(
                ball_pos,
                _vec_add(_vec_scale(right, local_x), _vec_scale(up, local_z)),
            )
            danger = float(payload.ball_contact_ratio) >= 1.0
        self._ball_root.setPos(*ball_pos)
        ball_rgb = self._BALL_COLOR_MAP["RED"]
        if not danger:
            target_rgb = self._BALL_COLOR_MAP.get(
                str(payload.ball_visual_color).upper() if payload is not None else "WHITE",
                self._BALL_COLOR_MAP["WHITE"],
            )
            strength = 0.0 if payload is None else float(payload.ball_visual_strength)
            ball_rgb = _mix_rgb(self._BALL_COLOR_MAP["WHITE"], target_rgb, mix=strength)
        self._ball_root.setColor(ball_rgb[0], ball_rgb[1], ball_rgb[2], 1.0)
        self._ball_root.setHpr((forward_norm * 540.0) % 360.0, 0.0, 0.0)

        cam_pos, look_target = fixed_camera_pose()
        self._base.cam.setPos(*cam_pos)
        self._base.cam.lookAt(*look_target)

    def _update_gates(self, *, payload: AuditoryCapacityPayload | None) -> None:
        from panda3d.core import Point3, Vec3

        if payload is None:
            for node in self._gate_nodes.values():
                node.removeNode()
            self._gate_nodes.clear()
            return

        visible_gates = [
            gate for gate in payload.gates if gate.visual_slot_index is not None
        ]
        visible_gates.sort(key=lambda gate: int(gate.visual_slot_index), reverse=True)
        y_half_span = max(0.08, float(payload.tube_half_height))
        keep_ids: set[int] = set()
        for gate in visible_gates[:14]:
            assert gate.visual_slot_index is not None
            distance = slot_distance(int(gate.visual_slot_index))
            keep_ids.add(int(gate.gate_id))
            center, tangent, right, up = _tube_frame(distance, span=self._span)
            depth_t = max(0.0, min(1.0, float(gate.visual_slot_index + 1) / 8.0))
            local_x = 0.0
            local_z = max(-1.0, min(1.0, gate.y_norm / y_half_span)) * (self._tube_rz * 0.62)
            radius = max(0.16, (float(gate.aperture_norm) / y_half_span) * (self._tube_rz * 0.82))
            rgba = self._depth_gate_rgba(base=self._gate_rgba(gate.color), depth_t=depth_t)
            node = self._gate_nodes.get(int(gate.gate_id))
            if node is None:
                node = self._build_gate_shape_node(shape=gate.shape)
                node.reparentTo(self._gate_root)
                self._gate_nodes[int(gate.gate_id)] = node
            gate_pos = _vec_add(
                center,
                _vec_add(_vec_scale(right, local_x), _vec_scale(up, local_z)),
            )
            node.setScale(radius)
            node.setColorScale(*rgba)
            node.setPos(*gate_pos)
            look_target = _vec_add(gate_pos, tangent)
            node.lookAt(
                Point3(*look_target),
                Vec3(up[0], up[1], up[2]),
            )
            node.setR(-6.0)
            node.show()

        stale_ids = [gate_id for gate_id in self._gate_nodes if gate_id not in keep_ids]
        for gate_id in stale_ids:
            self._gate_nodes[gate_id].removeNode()
            del self._gate_nodes[gate_id]

    def _advance_travel_offset(self, dt_s: float) -> None:
        _ = dt_s
        return None

    def _build_gate_shape_node(
        self,
        *,
        shape: str,
    ):
        from panda3d.core import (
            Geom,
            GeomNode,
            GeomTriangles,
            GeomVertexData,
            GeomVertexFormat,
            GeomVertexWriter,
            NodePath,
        )

        root = NodePath(f"gate-{shape.lower()}")
        token = str(shape).upper()
        points: list[tuple[float, float]]
        if token == "CIRCLE":
            points = []
            steps = 96
            for idx in range(steps):
                angle = (idx / float(steps)) * math.tau
                points.append((math.cos(angle), math.sin(angle)))
        elif token == "TRIANGLE":
            points = list(AUDITORY_TRIANGLE_GATE_POINTS)
        else:
            points = [
                (-1.10, 1.10),
                (1.10, 1.10),
                (1.10, -1.10),
                (-1.10, -1.10),
            ]

        fmt = GeomVertexFormat.getV3n3c4()
        vdata = GeomVertexData(f"gate-{token.lower()}-geom", fmt, Geom.UHStatic)
        vertex = GeomVertexWriter(vdata, "vertex")
        normal = GeomVertexWriter(vdata, "normal")
        color_writer = GeomVertexWriter(vdata, "color")
        tris = GeomTriangles(Geom.UHStatic)

        next_index = 0
        half_width = 0.095 if token == "CIRCLE" else 0.108
        half_depth = 0.080
        last_idx = len(points)
        for idx in range(last_idx):
            x0, z0 = points[idx]
            x1, z1 = points[(idx + 1) % last_idx]
            next_index = self._append_gate_segment_prism(
                vertex=vertex,
                normal=normal,
                color_writer=color_writer,
                tris=tris,
                next_index=next_index,
                start=(x0, z0),
                end=(x1, z1),
                half_width=half_width,
                half_depth=half_depth,
                rgba=(1.0, 1.0, 1.0, 1.0),
            )

        geom = Geom(vdata)
        geom.addPrimitive(tris)
        geom_node = GeomNode(f"gate-{token.lower()}-node")
        geom_node.addGeom(geom)
        geom_np = root.attachNewNode(geom_node)
        geom_np.setLightOff()
        geom_np.setTransparency(True)
        root.setTransparency(True)
        root.setTwoSided(True)
        return root

    @staticmethod
    def _append_gate_segment_prism(
        *,
        vertex,
        normal,
        color_writer,
        tris,
        next_index: int,
        start: tuple[float, float],
        end: tuple[float, float],
        half_width: float,
        half_depth: float,
        rgba: tuple[float, float, float, float],
    ) -> int:
        dx = end[0] - start[0]
        dz = end[1] - start[1]
        length = math.hypot(dx, dz)
        if length <= 1e-5:
            return next_index

        tx = dx / length
        tz = dz / length
        nx = -tz * half_width
        nz = tx * half_width

        a = (start[0] + nx, start[1] + nz)
        b = (start[0] - nx, start[1] - nz)
        c = (end[0] - nx, end[1] - nz)
        d = (end[0] + nx, end[1] + nz)

        faces = (
            (
                (
                    (a[0], -half_depth, a[1]),
                    (b[0], -half_depth, b[1]),
                    (c[0], -half_depth, c[1]),
                    (d[0], -half_depth, d[1]),
                ),
                (0.0, -1.0, 0.0),
            ),
            (
                (
                    (a[0], half_depth, a[1]),
                    (d[0], half_depth, d[1]),
                    (c[0], half_depth, c[1]),
                    (b[0], half_depth, b[1]),
                ),
                (0.0, 1.0, 0.0),
            ),
            (
                (
                    (a[0], -half_depth, a[1]),
                    (d[0], -half_depth, d[1]),
                    (d[0], half_depth, d[1]),
                    (a[0], half_depth, a[1]),
                ),
                (tz, 0.0, -tx),
            ),
            (
                (
                    (b[0], -half_depth, b[1]),
                    (b[0], half_depth, b[1]),
                    (c[0], half_depth, c[1]),
                    (c[0], -half_depth, c[1]),
                ),
                (-tz, 0.0, tx),
            ),
            (
                (
                    (a[0], -half_depth, a[1]),
                    (a[0], half_depth, a[1]),
                    (b[0], half_depth, b[1]),
                    (b[0], -half_depth, b[1]),
                ),
                (-tx, 0.0, -tz),
            ),
            (
                (
                    (d[0], -half_depth, d[1]),
                    (c[0], -half_depth, c[1]),
                    (c[0], half_depth, c[1]),
                    (d[0], half_depth, d[1]),
                ),
                (tx, 0.0, tz),
            ),
        )

        for quad, face_normal in faces:
            for point in quad:
                vertex.addData3f(*point)
                normal.addData3f(*face_normal)
                color_writer.addData4f(*rgba)
            tris.addVertices(next_index, next_index + 1, next_index + 2)
            tris.addVertices(next_index, next_index + 2, next_index + 3)
            next_index += 4
        return next_index

    def _load_cube_model(self):
        try:
            model = self._base.loader.loadModel("models/box")
            if not model.isEmpty():
                return model
        except Exception:
            pass
        try:
            model = self._base.loader.loadModel("models/misc/rgbCube")
            if not model.isEmpty():
                return model
        except Exception:
            pass
        return self._make_box(size=(1.0, 1.0, 1.0), color=(1.0, 1.0, 1.0, 1.0))

    @staticmethod
    def _gate_rgba(name: str) -> tuple[float, float, float, float]:
        palette = {
            "RED": (0.92, 0.28, 0.34, 0.96),
            "GREEN": (0.32, 0.86, 0.50, 0.96),
            "BLUE": (0.38, 0.58, 0.96, 0.96),
            "YELLOW": (0.96, 0.84, 0.32, 0.96),
        }
        return palette.get(str(name).upper(), (0.90, 0.92, 0.98, 0.96))

    @staticmethod
    def _depth_gate_rgba(
        *,
        base: tuple[float, float, float, float],
        depth_t: float,
    ) -> tuple[float, float, float, float]:
        t = max(0.0, min(1.0, float(depth_t)))
        visibility = 1.0 - (0.78 * t)
        brighten = 0.24 * (1.0 - t)
        return (
            min(1.0, (base[0] * visibility) + brighten),
            min(1.0, (base[1] * visibility) + brighten),
            min(1.0, (base[2] * visibility) + brighten),
            max(0.16, min(1.0, base[3] * (0.20 + (0.80 * (1.0 - t))))),
        )
