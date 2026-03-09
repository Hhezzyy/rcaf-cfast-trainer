from __future__ import annotations

import importlib.util
import math
import os
from dataclasses import dataclass
from pathlib import Path

import pygame

from .panda3d_assets import Panda3DAssetCatalog
from .rapid_tracking import RapidTrackingPayload


def panda3d_rapid_tracking_rendering_available() -> bool:
    if os.environ.get("CFAST_RAPID_TRACKING_RENDERER", "panda").strip().lower() in {
        "pygame",
        "2d",
        "off",
    }:
        return False
    if os.environ.get("SDL_VIDEODRIVER", "").strip().lower() == "dummy":
        return False
    return importlib.util.find_spec("direct.showbase.ShowBase") is not None


@dataclass(slots=True)
class _RapidTrackingDecoy:
    node: object
    kind: str
    phase: float
    speed: float
    lateral_bias: float
    depth_bias: float
    altitude: float
    route: str = "air_loop"
    activation_progress: float = 0.0


@dataclass(frozen=True, slots=True)
class RapidTrackingCameraRigState:
    cam_world_x: float
    cam_world_y: float
    cam_world_z: float
    carrier_heading_deg: float
    heading_deg: float
    pitch_deg: float
    roll_deg: float
    fov_deg: float
    orbit_weight: float
    orbit_radius: float
    altitude_agl: float


@dataclass(frozen=True, slots=True)
class RapidTrackingOverlayState:
    screen_x: float
    screen_y: float
    on_screen: bool
    in_front: bool
    target_visible: bool


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _smoothstep(edge0: float, edge1: float, value: float) -> float:
    if edge0 == edge1:
        return 1.0 if value >= edge1 else 0.0
    t = _clamp((value - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - (2.0 * t))


def _lerp(a: float, b: float, t: float) -> float:
    return a + ((b - a) * t)


def _lerp_angle_deg(a: float, b: float, t: float) -> float:
    delta = ((b - a + 180.0) % 360.0) - 180.0
    return a + (delta * t)


def _loop01(value: float) -> float:
    return float(value) - math.floor(float(value))


def _triangle01(value: float) -> float:
    phase = _loop01(value)
    if phase <= 0.5:
        return phase * 2.0
    return 2.0 - (phase * 2.0)


class RapidTrackingPanda3DRenderer:
    _WORLD_EXTENT_SCALE = 5.0
    _CAMERA_SWEEP_LIMIT_DEG = 66.0
    _TARGET_SWEEP_LIMIT_DEG = 78.0
    _SCENE_X_SPAN = 3.0
    _TERRAIN_HALF_SPAN = 260.0 * _WORLD_EXTENT_SCALE
    _GROUND_INTERSECT_MAX_DISTANCE = 320.0

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
        self._catalog = Panda3DAssetCatalog()
        self._active_target_node = None
        self._active_target_kind = ""
        self._target_overlay_state: RapidTrackingOverlayState | None = None

        loadPrcFileData("", "window-type offscreen")
        loadPrcFileData("", "audio-library-name null")
        loadPrcFileData("", "sync-video false")
        loadPrcFileData("", f"win-size {width} {height}")

        from direct.showbase.ShowBase import ShowBase

        self._base = ShowBase(windowType="offscreen")
        self._base.disableMouse()
        self._base.setBackgroundColor(0.56, 0.68, 0.82, 1.0)
        self._texture = Texture()
        self._texture.setKeepRamImage(True)
        self._base.win.addRenderTexture(self._texture, GraphicsOutput.RTMCopyRam)

        self._configure_render_pipeline()

        self._base.camLens.setFov(58.0)
        self._base.camLens.setNearFar(0.1, 1600.0)

        ambient = AmbientLight("rapid-ambient")
        ambient.setColor(Vec4(0.70, 0.72, 0.76, 1.0))
        ambient_np = self._base.render.attachNewNode(ambient)
        self._base.render.setLight(ambient_np)

        sun = DirectionalLight("rapid-sun")
        sun.setColor(Vec4(0.98, 0.95, 0.88, 1.0))
        sun_np = self._base.render.attachNewNode(sun)
        sun_np.setHpr(28.0, -30.0, 0.0)
        self._base.render.setLight(sun_np)

        self._world_root = self._base.render.attachNewNode("rapid-world")
        self._terrain_root = self._world_root.attachNewNode("rapid-terrain")
        self._scenery_root = self._world_root.attachNewNode("rapid-scenery")
        self._decoy_root = self._world_root.attachNewNode("rapid-decoys")
        self._target_root = self._world_root.attachNewNode("rapid-targets")
        self._cloud_root = self._world_root.attachNewNode("rapid-clouds")
        self._direction_probe = self._base.render.attachNewNode("rapid-direction-probe")

        self._build_world()

    @property
    def size(self) -> tuple[int, int]:
        return self._size

    def target_overlay_state(self) -> RapidTrackingOverlayState | None:
        return self._target_overlay_state

    def close(self) -> None:
        try:
            self._base.destroy()
        except Exception:
            pass

    @staticmethod
    def _path_point(*, t: float) -> tuple[float, float]:
        # Preloaded procedural run-in path for the late stage of the test.
        u = _clamp(float(t), 0.0, 1.0)
        x = -24.0 + (68.0 * u) + (math.sin(u * math.pi * 1.6) * 12.0)
        y = -34.0 + (136.0 * u) + (math.sin((u * math.pi * 2.2) + 0.35) * 8.0)
        return float(x), float(y)

    @classmethod
    def _camera_rig_state(
        cls,
        *,
        elapsed_s: float,
        progress: float,
        cam_x: float,
        cam_y: float,
        zoom: float,
        target_kind: str,
        target_rel_x: float,
        target_rel_y: float,
        assist_strength: float,
        turbulence_strength: float,
    ) -> RapidTrackingCameraRigState:
        p = _clamp(float(progress), 0.0, 1.0)
        assist = _clamp(float(assist_strength), 0.0, 1.0)
        turbulence = max(0.0, float(turbulence_strength))
        orbit_phase = 0.42 + (elapsed_s * 0.15) + (p * 0.24)
        orbit_radius = _lerp(58.0, 42.0, _smoothstep(0.0, 0.52, p))
        orbit_x = math.cos(orbit_phase) * orbit_radius
        orbit_y = math.sin(orbit_phase) * (orbit_radius * 0.86)
        orbit_heading = (90.0 - math.degrees(math.atan2(-orbit_y, -orbit_x))) % 360.0

        path_transition = _smoothstep(0.56, 0.84, p)
        path_t = _smoothstep(0.60, 1.0, p)
        path_x, path_y = cls._path_point(t=path_t)
        next_x, next_y = cls._path_point(t=min(1.0, path_t + 0.015))
        path_dx = next_x - path_x
        path_dy = next_y - path_y
        if abs(path_dx) < 1e-6 and abs(path_dy) < 1e-6:
            path_heading = orbit_heading
        else:
            path_heading = (90.0 - math.degrees(math.atan2(path_dy, path_dx))) % 360.0

        right_mount = 1.35
        heading_base = _lerp_angle_deg(orbit_heading, path_heading, path_transition)
        heading_rad = math.radians(heading_base)
        right_vec_x = math.cos(heading_rad)
        right_vec_y = -math.sin(heading_rad)

        base_x = _lerp(orbit_x, path_x, path_transition)
        base_y = _lerp(orbit_y, path_y, path_transition)
        cam_world_x = base_x + (right_vec_x * right_mount)
        cam_world_y = base_y + (right_vec_y * right_mount)

        ground_z = cls._terrain_height(cam_world_x, cam_world_y)
        altitude_agl = _lerp(24.0, 5.8, _smoothstep(0.0, 1.0, p))
        bob = (
            math.sin((elapsed_s * 2.1) + 0.25) * _lerp(0.18, 0.08, path_transition) * turbulence
            + math.sin((elapsed_s * 4.6) + 1.1) * _lerp(0.09, 0.04, path_transition) * turbulence
        )
        cam_world_z = ground_z + altitude_agl + bob + (zoom * 0.22)

        local_yaw_deg = _clamp(
            (cam_x / 5.2) * cls._CAMERA_SWEEP_LIMIT_DEG,
            -cls._CAMERA_SWEEP_LIMIT_DEG,
            cls._CAMERA_SWEEP_LIMIT_DEG,
        )
        air_target = str(target_kind).strip().lower() in {"jet", "helicopter"}
        late_air_assist = (_smoothstep(0.68, 1.0, p) * assist) if air_target else 0.0
        heading_deg = (
            heading_base
            + (local_yaw_deg * (1.0 - (assist * 0.25)))
            + (assist * target_rel_x * 8.0)
            + (late_air_assist * target_rel_x * 12.0)
        )

        pitch_base = _lerp(-12.0, 3.0, _smoothstep(0.0, 0.86, p))
        pitch_from_input = -cam_y * 9.6
        pitch_from_air = late_air_assist * (8.0 + max(0.0, -float(target_rel_y)) * 10.0)
        pitch_deg = _clamp(pitch_base + pitch_from_input + pitch_from_air, -34.0, 26.0)

        orbit_bank = _lerp(7.5, 2.0, path_transition)
        roll_deg = (
            orbit_bank
            + (
                math.sin((elapsed_s * 1.9) + 0.4)
                * _lerp(2.0, 0.9, path_transition)
                * max(0.15, turbulence)
            )
            + (-cam_x * 2.4)
        )

        fov_deg = _clamp(52.0 - (zoom * 6.0), 44.0, 52.0)
        orbit_weight = 1.0 - path_transition

        return RapidTrackingCameraRigState(
            cam_world_x=float(cam_world_x),
            cam_world_y=float(cam_world_y),
            cam_world_z=float(cam_world_z),
            carrier_heading_deg=float(heading_base % 360.0),
            heading_deg=float(heading_deg),
            pitch_deg=float(pitch_deg),
            roll_deg=float(roll_deg),
            fov_deg=float(fov_deg),
            orbit_weight=float(orbit_weight),
            orbit_radius=float(orbit_radius),
            altitude_agl=float(altitude_agl),
        )

    def render(self, *, payload: RapidTrackingPayload | None) -> pygame.Surface:
        now_ms = pygame.time.get_ticks()
        dt_s = max(0.0, min(0.05, (now_ms - self._last_render_ms) / 1000.0))
        self._last_render_ms = now_ms
        self._elapsed_s += dt_s

        self._update_camera(payload=payload)
        self._update_target(payload=payload)
        self._update_decoys(payload=payload)
        self._update_clouds()
        self._update_overlay_state(payload=payload)
        self._base.graphicsEngine.renderFrame()

        ram = self._texture.getRamImageAs("RGBA")
        raw = bytes(ram)
        surface = pygame.image.frombuffer(raw, self._size, "RGBA").copy()
        return pygame.transform.flip(surface, False, True)

    @staticmethod
    def _heading_from_velocity(
        *,
        vx: float,
        vy: float,
        default_heading: float = 0.0,
    ) -> float:
        if abs(vx) <= 1e-6 and abs(vy) <= 1e-6:
            return float(default_heading) % 360.0
        return float((-math.degrees(math.atan2(vx, vy))) % 360.0)

    @classmethod
    def _ground_route_pose(
        cls,
        *,
        elapsed_s: float,
        phase: float,
        speed: float,
        lateral_bias: float,
        depth_bias: float,
        route: str,
        tank_spin: bool = False,
    ) -> tuple[float, float, float]:
        cycle_rate = 0.16 if route == "ground_convoy" else 0.20
        cycle = (float(elapsed_s) * max(0.05, float(speed)) * cycle_rate) + (
            float(phase) / math.tau
        )
        phase_local = _loop01(cycle)
        forward = 1.0 if phase_local < 0.5 else -1.0
        travel = (_triangle01(cycle) * 2.0) - 1.0

        if route == "ground_convoy":
            x = float(lateral_bias)
            y = (travel * (20.0 + (float(depth_bias) * 0.55))) + 12.0
            heading = 0.0 if forward > 0.0 else 180.0
        elif route == "tank_hold":
            x = float(lateral_bias)
            y = 26.0 + float(depth_bias)
            heading = (
                (float(phase) * 57.0) + (float(elapsed_s) * max(0.05, float(speed)) * 140.0)
            ) % 360.0
        else:
            x = travel * (30.0 + (float(depth_bias) * 0.36))
            y = 24.0 + float(depth_bias) + (float(lateral_bias) * 0.08)
            heading = 270.0 if forward > 0.0 else 90.0

        if tank_spin and route != "tank_hold":
            heading = (
                heading
                + (math.sin((float(elapsed_s) * 0.9) + float(phase)) * 16.0)
                + (float(elapsed_s) * 40.0)
            ) % 360.0

        return float(x), float(y), float(heading)

    @staticmethod
    def _camera_space_to_viewport(
        *,
        cam_x: float,
        cam_y: float,
        cam_z: float,
        size: tuple[int, int],
        h_fov_deg: float,
        v_fov_deg: float,
    ) -> tuple[float, float, bool, bool]:
        width = max(1, int(size[0]))
        height = max(1, int(size[1]))
        in_front = float(cam_y) > 1e-3
        depth = max(1e-3, abs(float(cam_y)))
        tan_h = max(1e-4, math.tan(math.radians(float(h_fov_deg) * 0.5)))
        tan_v = max(1e-4, math.tan(math.radians(float(v_fov_deg) * 0.5)))

        norm_x = float(cam_x) / (depth * tan_h)
        norm_y = float(cam_z) / (depth * tan_v)
        if not in_front:
            norm_x = -norm_x

        screen_x = (norm_x + 1.0) * 0.5 * width
        screen_y = (1.0 - norm_y) * 0.5 * height
        on_screen = in_front and abs(norm_x) <= 1.0 and abs(norm_y) <= 1.0
        return float(screen_x), float(screen_y), bool(on_screen), bool(in_front)

    def _configure_render_pipeline(self) -> None:
        enable_simplepbr = os.environ.get("CFAST_PANDA_SIMPLEPBR", "").strip().lower() in {
            "1",
            "true",
            "on",
            "yes",
        }
        if not enable_simplepbr:
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

    def _build_world(self) -> None:
        terrain = self._build_terrain_mesh()
        terrain.reparentTo(self._terrain_root)
        self._build_central_compound()

        self._place_scenery(
            asset_id="building_hangar",
            fallback="hangar",
            pos=(-15.0, 62.0, self._terrain_height(-15.0, 62.0) + 1.4),
            scale=1.2,
        )
        self._place_scenery(
            asset_id="building_tower",
            fallback="tower",
            pos=(18.0, 56.0, self._terrain_height(18.0, 56.0) + 2.2),
            scale=0.95,
        )
        self._place_scenery(
            asset_id="trees_pine_cluster",
            fallback="trees",
            pos=(-24.0, 44.0, self._terrain_height(-24.0, 44.0)),
            scale=1.4,
        )
        self._place_scenery(
            asset_id="trees_pine_cluster",
            fallback="trees",
            pos=(25.0, 72.0, self._terrain_height(25.0, 72.0)),
            scale=1.6,
        )
        self._place_scenery(
            asset_id="building_hangar",
            fallback="hangar",
            pos=(21.0, 88.0, self._terrain_height(21.0, 88.0) + 1.1),
            scale=0.82,
        )
        self._place_scenery(
            asset_id="building_tower",
            fallback="tower",
            pos=(-28.0, 102.0, self._terrain_height(-28.0, 102.0) + 2.0),
            scale=0.88,
        )
        self._place_scenery(
            asset_id="trees_pine_cluster",
            fallback="trees",
            pos=(-12.0, 118.0, self._terrain_height(-12.0, 118.0)),
            scale=2.0,
        )
        self._place_scenery(
            asset_id="trees_pine_cluster",
            fallback="trees",
            pos=(34.0, 126.0, self._terrain_height(34.0, 126.0)),
            scale=1.8,
        )
        self._place_scenery(
            asset_id="truck_olive",
            fallback="truck",
            pos=(-7.0, 58.0, self._terrain_height(-7.0, 58.0) + 0.35),
            scale=0.94,
        )
        self._place_scenery(
            asset_id="soldiers_patrol",
            fallback="soldiers",
            pos=(-14.0, 66.0, self._terrain_height(-14.0, 66.0)),
            scale=1.0,
        )
        self._place_scenery(
            asset_id="soldiers_patrol",
            fallback="soldiers",
            pos=(14.0, 62.0, self._terrain_height(14.0, 62.0)),
            scale=0.94,
        )

        self._target_jet = self._load_asset_or_fallback(
            asset_id="plane_red",
            fallback="plane",
            color=(0.88, 0.20, 0.16, 1.0),
            scale=1.56,
            parent=self._target_root,
        )
        self._target_helicopter = self._load_asset_or_fallback(
            asset_id="helicopter_green",
            fallback="helicopter",
            color=(0.24, 0.54, 0.32, 1.0),
            scale=1.34,
            parent=self._target_root,
        )
        self._target_truck = self._load_asset_or_fallback(
            asset_id="truck_olive",
            fallback="truck",
            color=(0.38, 0.42, 0.24, 1.0),
            scale=1.24,
            parent=self._target_root,
        )
        self._target_hangar = self._load_asset_or_fallback(
            asset_id="building_hangar",
            fallback="hangar",
            color=(0.72, 0.76, 0.82, 1.0),
            scale=0.98,
            parent=self._target_root,
        )
        self._target_tower = self._load_asset_or_fallback(
            asset_id="building_tower",
            fallback="tower",
            color=(0.74, 0.78, 0.82, 1.0),
            scale=1.08,
            parent=self._target_root,
        )
        self._target_soldier = self._build_fallback_model(
            kind="soldier",
            color=(0.74, 0.28, 0.24, 1.0),
        )
        self._target_soldier.reparentTo(self._target_root)
        self._target_soldier.setScale(1.06)
        self._soldier_group = [
            self._build_fallback_model(kind="soldier", color=(0.18, 0.24, 0.16, 1.0))
            for _ in range(4)
        ]
        for node in self._soldier_group:
            node.reparentTo(self._target_root)
            node.setScale(0.92)

        self._target_nodes = (
            self._target_jet,
            self._target_helicopter,
            self._target_truck,
            self._target_hangar,
            self._target_tower,
            self._target_soldier,
            *self._soldier_group,
        )
        for node in self._target_nodes:
            node.setTransparency(True)
        self._decoys = self._build_traffic_pool()
        self._build_surround_scenery()

        self._clouds = [
            self._build_cloud(scale=scale) for scale in (1.0, 1.4, 0.8, 1.2, 0.95, 1.55, 1.75)
        ]
        for idx, cloud in enumerate(self._clouds):
            ang = (idx / max(1, len(self._clouds))) * math.tau
            radius = self._world_extent_distance(108.0 + (idx * 14.0))
            cloud.reparentTo(self._cloud_root)
            cloud.setPos(
                math.cos(ang) * radius,
                math.sin(ang) * radius,
                25.0 + (idx * 2.2),
            )

        self._hide_target_nodes()

    def _build_traffic_pool(self) -> list[_RapidTrackingDecoy]:
        decoys = [
            _RapidTrackingDecoy(
                node=self._load_asset_or_fallback(
                    asset_id="plane_blue",
                    fallback="plane",
                    color=(0.22, 0.46, 0.88, 1.0),
                    scale=0.96,
                    parent=self._decoy_root,
                ),
                kind="plane",
                phase=0.15,
                speed=0.78,
                lateral_bias=-18.0,
                depth_bias=0.0,
                altitude=18.0,
                route="air_loop",
                activation_progress=0.0,
            ),
            _RapidTrackingDecoy(
                node=self._load_asset_or_fallback(
                    asset_id="plane_green",
                    fallback="plane",
                    color=(0.26, 0.60, 0.34, 1.0),
                    scale=0.90,
                    parent=self._decoy_root,
                ),
                kind="plane",
                phase=1.80,
                speed=0.62,
                lateral_bias=14.0,
                depth_bias=14.0,
                altitude=22.0,
                route="air_cross",
                activation_progress=0.0,
            ),
            _RapidTrackingDecoy(
                node=self._load_asset_or_fallback(
                    asset_id="plane_yellow",
                    fallback="plane",
                    color=(0.86, 0.72, 0.22, 1.0),
                    scale=0.88,
                    parent=self._decoy_root,
                ),
                kind="plane",
                phase=0.62,
                speed=0.74,
                lateral_bias=-6.0,
                depth_bias=22.0,
                altitude=16.0,
                route="air_loop",
                activation_progress=0.24,
            ),
            _RapidTrackingDecoy(
                node=self._load_asset_or_fallback(
                    asset_id="helicopter_green",
                    fallback="helicopter",
                    color=(0.28, 0.54, 0.34, 1.0),
                    scale=0.86,
                    parent=self._decoy_root,
                ),
                kind="helicopter",
                phase=2.18,
                speed=0.55,
                lateral_bias=22.0,
                depth_bias=28.0,
                altitude=12.0,
                route="helo_arc",
                activation_progress=0.38,
            ),
            _RapidTrackingDecoy(
                node=self._build_fallback_model(
                    kind="tank",
                    color=(0.46, 0.44, 0.24, 1.0),
                ),
                kind="tank",
                phase=0.55,
                speed=0.38,
                lateral_bias=-11.0,
                depth_bias=18.0,
                altitude=0.0,
                route="tank_hold",
                activation_progress=0.52,
            ),
            _RapidTrackingDecoy(
                node=self._load_asset_or_fallback(
                    asset_id="truck_olive",
                    fallback="truck",
                    color=(0.38, 0.42, 0.24, 1.0),
                    scale=0.92,
                    parent=self._decoy_root,
                ),
                kind="truck",
                phase=2.10,
                speed=0.36,
                lateral_bias=12.0,
                depth_bias=26.0,
                altitude=0.0,
                route="ground_side",
                activation_progress=0.66,
            ),
            _RapidTrackingDecoy(
                node=self._build_fallback_model(
                    kind="plane",
                    color=(0.78, 0.30, 0.22, 1.0),
                ),
                kind="plane",
                phase=1.34,
                speed=0.82,
                lateral_bias=4.0,
                depth_bias=40.0,
                altitude=27.0,
                route="air_cross",
                activation_progress=0.79,
            ),
            _RapidTrackingDecoy(
                node=self._build_fallback_model(
                    kind="truck",
                    color=(0.28, 0.36, 0.22, 1.0),
                ),
                kind="truck",
                phase=2.92,
                speed=0.34,
                lateral_bias=-18.0,
                depth_bias=34.0,
                altitude=0.0,
                route="ground_convoy",
                activation_progress=0.90,
            ),
        ]
        for decoy in decoys:
            decoy.node.reparentTo(self._decoy_root)
            decoy.node.setTransparency(True)
            if decoy.kind in {"tank", "truck"}:
                scale = decoy.node.getScale()
                decoy.node.setScale(scale.x * 0.94, scale.y * 0.94, scale.z * 0.94)
        return decoys

    def _hide_target_nodes(self) -> None:
        self._active_target_node = None
        self._active_target_kind = ""
        for node in self._target_nodes:
            node.hide()
            node.setAlphaScale(0.0)

    def _build_surround_scenery(self) -> None:
        ring_items = (
            ("trees_pine_cluster", "trees", 44.0, 18.0, 1.4),
            ("building_hangar", "hangar", 56.0, 62.0, 0.92),
            ("building_tower", "tower", 68.0, 118.0, 0.96),
            ("trees_pine_cluster", "trees", 82.0, 160.0, 1.9),
            ("truck_olive", "truck", 38.0, 210.0, 0.84),
            ("trees_pine_cluster", "trees", 74.0, 252.0, 1.6),
            ("building_hangar", "hangar", 64.0, 302.0, 0.88),
            ("building_tower", "tower", 52.0, 338.0, 0.84),
            ("trees_pine_cluster", "trees", 92.0, 36.0, 2.2),
            ("trees_pine_cluster", "trees", 88.0, 84.0, 1.8),
            ("building_hangar", "hangar", 98.0, 142.0, 0.86),
            ("trees_pine_cluster", "trees", 108.0, 196.0, 2.0),
            ("building_tower", "tower", 116.0, 228.0, 0.82),
            ("trees_pine_cluster", "trees", 104.0, 276.0, 1.9),
            ("truck_olive", "truck", 58.0, 318.0, 0.82),
            ("trees_pine_cluster", "trees", 122.0, 352.0, 2.4),
            ("building_hangar", "hangar", 144.0, 14.0, 0.96),
            ("trees_pine_cluster", "trees", 158.0, 54.0, 2.8),
            ("building_tower", "tower", 172.0, 102.0, 0.88),
            ("truck_olive", "truck", 164.0, 138.0, 0.86),
            ("trees_pine_cluster", "trees", 176.0, 184.0, 2.6),
            ("building_hangar", "hangar", 168.0, 226.0, 0.90),
            ("trees_pine_cluster", "trees", 182.0, 274.0, 2.7),
            ("building_tower", "tower", 170.0, 318.0, 0.86),
        )
        for radius, bearing_deg in (
            (96.0, 28.0),
            (118.0, 94.0),
            (104.0, 182.0),
            (126.0, 266.0),
            (136.0, 48.0),
            (142.0, 132.0),
            (138.0, 224.0),
            (146.0, 312.0),
            (178.0, 24.0),
            (194.0, 86.0),
            (188.0, 152.0),
            (202.0, 214.0),
            (196.0, 286.0),
            (208.0, 334.0),
        ):
            scaled_radius = self._world_extent_distance(radius)
            x = math.cos(math.radians(bearing_deg)) * scaled_radius
            y = math.sin(math.radians(bearing_deg)) * scaled_radius
            self._place_scenery(
                asset_id="trees_pine_cluster",
                fallback="trees",
                pos=(x, y, self._terrain_height(x, y)),
                scale=2.3,
            )

        for asset_id, fallback, radius, bearing_deg, scale in ring_items:
            ang = math.radians(bearing_deg)
            scaled_radius = self._world_extent_distance(radius)
            x = math.cos(ang) * scaled_radius
            y = math.sin(ang) * scaled_radius
            z = self._terrain_height(x, y)
            if fallback in {"hangar", "tower"}:
                z += 1.0 if fallback == "hangar" else 2.0
            elif fallback == "truck":
                z += 0.35
            self._place_scenery(
                asset_id=asset_id,
                fallback=fallback,
                pos=(x, y, z),
                scale=scale,
            )

    def _build_central_compound(self) -> None:
        compound_items = (
            ("building_hangar", "hangar", -24.0, 36.0, 1.18),
            ("building_hangar", "hangar", -2.0, 34.0, 1.08),
            ("building_tower", "tower", 22.0, 38.0, 0.94),
            ("truck_olive", "truck", -30.0, 52.0, 0.88),
            ("soldiers_patrol", "soldiers", -18.0, 50.0, 0.98),
            ("building_hangar", "hangar", 6.0, 56.0, 0.92),
            ("building_tower", "tower", 28.0, 58.0, 0.84),
            ("soldiers_patrol", "soldiers", 18.0, 48.0, 0.90),
            ("trees_pine_cluster", "trees", -38.0, 66.0, 1.34),
            ("truck_olive", "truck", 36.0, 68.0, 0.84),
            ("building_hangar", "hangar", -18.0, 76.0, 0.86),
            ("trees_pine_cluster", "trees", 2.0, 76.0, 1.42),
            ("building_hangar", "hangar", 22.0, 78.0, 0.84),
            ("building_tower", "tower", -34.0, 88.0, 0.80),
            ("soldiers_patrol", "soldiers", -8.0, 92.0, 0.84),
            ("truck_olive", "truck", 10.0, 94.0, 0.80),
            ("trees_pine_cluster", "trees", 32.0, 96.0, 1.26),
            ("building_hangar", "hangar", -12.0, 108.0, 0.78),
            ("soldiers_patrol", "soldiers", 18.0, 110.0, 0.80),
            ("trees_pine_cluster", "trees", -30.0, 116.0, 1.52),
        )

        for asset_id, fallback, x, y, scale in compound_items:
            z = self._terrain_height(x, y)
            if fallback == "hangar":
                z += 1.1
            elif fallback == "tower":
                z += 2.1
            elif fallback == "truck":
                z += 0.34
            self._place_scenery(
                asset_id=asset_id,
                fallback=fallback,
                pos=(x, y, z),
                scale=scale,
            )

        for x, y, scale, heading in (
            (-6.0, 44.0, 0.88, 0.0),
            (30.0, 86.0, 0.82, 92.0),
            (-28.0, 104.0, 0.76, 188.0),
        ):
            tank = self._build_fallback_model(kind="tank", color=(0.42, 0.40, 0.22, 1.0))
            tank.reparentTo(self._scenery_root)
            tank.setScale(scale)
            tank.setPos(x, y, self._terrain_height(x, y) + 0.28)
            tank.setHpr(heading, 0.0, 0.0)

    def _update_camera(self, *, payload: RapidTrackingPayload | None) -> None:
        cam_x = 0.0 if payload is None else float(payload.camera_x)
        cam_y = 0.0 if payload is None else float(payload.camera_y)
        progress = 0.0 if payload is None else float(payload.scene_progress)
        zoom = 0.0 if payload is None else float(payload.capture_zoom)
        assist_strength = 0.0 if payload is None else float(payload.camera_assist_strength)
        turbulence_strength = 0.65 if payload is None else float(payload.turbulence_strength)
        target_kind = "" if payload is None else str(payload.target_kind)
        target_rel_x = 0.0 if payload is None else float(payload.target_rel_x)
        target_rel_y = 0.0 if payload is None else float(payload.target_rel_y)
        rig = self._camera_rig_state(
            elapsed_s=self._elapsed_s,
            progress=progress,
            cam_x=cam_x,
            cam_y=cam_y,
            zoom=zoom,
            target_kind=target_kind,
            target_rel_x=target_rel_x,
            target_rel_y=target_rel_y,
            assist_strength=assist_strength,
            turbulence_strength=turbulence_strength,
        )

        self._base.camLens.setFov(rig.fov_deg)
        self._base.cam.setPos(rig.cam_world_x, rig.cam_world_y, rig.cam_world_z)
        self._base.cam.setHpr(
            rig.heading_deg,
            rig.pitch_deg,
            rig.roll_deg,
        )

    def _update_target(self, *, payload: RapidTrackingPayload | None) -> None:
        from panda3d.core import Vec3

        self._hide_target_nodes()

        if payload is None:
            self._target_jet.show()
            self._target_jet.setAlphaScale(1.0)
            self._target_jet.setPos(
                math.sin(self._elapsed_s * 0.9) * 8.0,
                62.0,
                16.0 + (math.sin(self._elapsed_s * 1.4) * 2.2),
            )
            self._target_jet.setHpr(
                math.sin(self._elapsed_s * 0.8) * 20.0,
                math.cos(self._elapsed_s * 1.1) * 6.0,
                math.sin(self._elapsed_s * 1.6) * 18.0,
            )
            self._active_target_node = self._target_jet
            self._active_target_kind = "jet"
            return

        kind = str(payload.target_kind).strip().lower()
        variant = str(payload.target_variant).strip().lower()
        progress = float(payload.scene_progress)
        scene_x = float(payload.target_rel_x + payload.camera_x)
        scene_y = float(payload.target_rel_y + payload.camera_y)
        cam_pos = self._base.cam.getPos(self._base.render)
        target_direction = self._scene_direction(
            scene_x=scene_x,
            scene_y=scene_y,
            progress=progress,
        )
        visible_alpha = 1.0 if bool(payload.target_visible) else 0.76

        def ground_target(*, clearance: float, distance_bias: float) -> tuple[float, float, float]:
            target_pos = self._world_point_on_terrain_along_direction(
                origin=cam_pos,
                direction=target_direction,
                clearance=clearance,
            )
            if target_pos is None:
                flat_direction = Vec3(target_direction.x, target_direction.y, 0.0)
                if flat_direction.lengthSquared() <= 1e-6:
                    flat_direction = Vec3(0.0, 1.0, 0.0)
                else:
                    flat_direction.normalize()
                ground_distance = distance_bias + _clamp((0.58 - scene_y) * 18.0, -10.0, 16.0)
                fallback = cam_pos + (flat_direction * ground_distance)
                return (
                    float(fallback.x),
                    float(fallback.y),
                    self._terrain_height(float(fallback.x), float(fallback.y)) + clearance,
                )
            return target_pos

        if kind == "soldier":
            x, y, z = ground_target(clearance=0.05, distance_bias=40.0)
            heading = self._heading_from_velocity(
                vx=float(payload.target_vx),
                vy=float(payload.target_vy),
            )
            self._target_soldier.setPos(x, y, z + 0.26)
            self._target_soldier.setHpr(heading, 0.0, 0.0)
            self._target_soldier.setAlphaScale(visible_alpha)
            self._target_soldier.show()
            self._active_target_node = self._target_soldier
            self._active_target_kind = kind
            if visible_alpha > 0.0:
                offsets = ((-1.1, 0.8), (1.0, 0.7), (-0.9, -0.8), (1.1, -0.9))
                for node, (ox, oy) in zip(self._soldier_group, offsets):
                    gx = x + ox
                    gy = y + oy
                    gz = self._terrain_height(gx, gy) + 0.24
                    node.setPos(gx, gy, gz)
                    node.setHpr(heading + (oy * 8.0), 0.0, 0.0)
                    node.setAlphaScale(0.92)
                    node.show()
            return
        if kind == "truck":
            x, y, z = ground_target(clearance=0.34, distance_bias=42.0)
            self._target_truck.setPos(x, y, z)
            self._target_truck.setHpr(
                self._heading_from_velocity(
                    vx=float(payload.target_vx),
                    vy=float(payload.target_vy),
                ),
                0.0,
                0.0,
            )
            self._target_truck.setAlphaScale(visible_alpha)
            self._target_truck.show()
            self._active_target_node = self._target_truck
            self._active_target_kind = kind
            return

        if kind == "building":
            x, y, z = ground_target(clearance=0.0, distance_bias=44.0)
            if variant == "tower":
                node = self._target_tower
                z += 2.2
            else:
                node = self._target_hangar
                z += 1.25
            node.setPos(x, y, z)
            node.setHpr(0.0, 0.0, 0.0)
            node.setAlphaScale(0.96 if visible_alpha > 0.0 else 0.0)
            node.show()
            self._active_target_node = node
            self._active_target_kind = kind
            return

        if kind == "helicopter":
            air_distance = 46.0 + _clamp((-scene_y) * 12.0, -10.0, 16.0)
            air_pos = cam_pos + (target_direction * air_distance)
            x = float(air_pos.x)
            y = float(air_pos.y)
            z = max(self._terrain_height(x, y) + 5.8, min(34.0, float(air_pos.z)))
            self._target_helicopter.setPos(x, y, z)
            self._target_helicopter.setHpr(
                -math.degrees(math.atan2(float(payload.target_vx) * 16.0, 1.0)),
                _clamp(-float(payload.target_vy) * 60.0, -12.0, 12.0),
                _clamp(-float(payload.target_vx) * 80.0, -18.0, 18.0),
            )
            self._target_helicopter.setAlphaScale(visible_alpha)
            self._target_helicopter.show()
            self._active_target_node = self._target_helicopter
            self._active_target_kind = kind
            return

        air_distance = 60.0 + _clamp((-scene_y) * 14.0, -12.0, 18.0)
        air_pos = cam_pos + (target_direction * air_distance)
        x = float(air_pos.x)
        y = float(air_pos.y)
        z = max(self._terrain_height(x, y) + 10.0, min(42.0, float(air_pos.z)))
        self._target_jet.setPos(x, y, z)
        self._target_jet.setHpr(
            -math.degrees(math.atan2(float(payload.target_vx) * 18.0, 1.0)),
            _clamp(-float(payload.target_vy) * 80.0, -16.0, 16.0),
            _clamp(-float(payload.target_vx) * 120.0, -30.0, 30.0),
        )
        self._target_jet.setAlphaScale(visible_alpha)
        self._target_jet.show()
        self._active_target_node = self._target_jet
        self._active_target_kind = kind

    def _update_decoys(self, *, payload: RapidTrackingPayload | None) -> None:
        t = self._elapsed_s
        progress = 0.0 if payload is None else float(payload.scene_progress)
        for decoy in self._decoys:
            fade = _clamp((progress - decoy.activation_progress + 0.18) / 0.18, 0.0, 1.0)
            if decoy.activation_progress > 0.0 and fade <= 0.0:
                decoy.node.hide()
                continue

            decoy.node.show()
            decoy.node.setAlphaScale(fade)
            if decoy.route == "air_cross":
                direction = -1.0 if decoy.lateral_bias < 0.0 else 1.0
                angle = (t * decoy.speed * 0.34 * math.tau * direction) + decoy.phase
                radius = 62.0 + decoy.depth_bias + (math.sin((t * 0.44) + decoy.phase) * 8.0)
                x = math.cos(angle) * radius + math.sin((t * 1.2) + decoy.phase) * 6.0
                y = math.sin(angle) * radius + math.cos((t * 0.9) + decoy.phase) * 4.0
                z = max(
                    self._terrain_height(x, y) + 9.0,
                    12.0 + decoy.altitude + (math.cos((t * 1.1) + decoy.phase) * 3.4),
                )
                heading = 90.0 - math.degrees(angle) + (180.0 if direction < 0.0 else 0.0)
                decoy.node.setPos(x, y, z)
                decoy.node.setHpr(
                    heading,
                    math.cos((t * 1.3) + decoy.phase) * 10.0,
                    math.sin((t * 1.6) + decoy.phase) * 22.0,
                )
            elif decoy.route == "helo_arc":
                direction = -1.0 if decoy.lateral_bias < 0.0 else 1.0
                angle = (t * decoy.speed * 0.22 * math.tau * direction) + decoy.phase
                radius_x = 34.0 + (decoy.depth_bias * 0.55)
                radius_y = 28.0 + (decoy.depth_bias * 0.42)
                x = math.cos(angle) * radius_x + math.sin((t * 0.8) + decoy.phase) * 5.0
                y = math.sin(angle) * radius_y + math.cos((t * 0.6) + decoy.phase) * 3.5
                z = max(
                    self._terrain_height(x, y) + 5.5,
                    8.0 + decoy.altitude + math.sin((t * 1.7) + decoy.phase) * 1.5,
                )
                heading = 90.0 - math.degrees(angle) + (180.0 if direction < 0.0 else 0.0)
                decoy.node.setPos(x, y, z)
                decoy.node.setHpr(
                    heading,
                    math.cos((t * 1.1) + decoy.phase) * 6.0,
                    math.sin((t * 2.8) + decoy.phase) * 8.0,
                )
            elif decoy.route == "ground_side":
                x, y, heading = self._ground_route_pose(
                    elapsed_s=t,
                    phase=decoy.phase,
                    speed=decoy.speed,
                    lateral_bias=decoy.lateral_bias,
                    depth_bias=decoy.depth_bias,
                    route=decoy.route,
                )
                z = self._terrain_height(x, y) + 0.34
                decoy.node.setPos(x, y, z)
                decoy.node.setHpr(heading, 0.0, 0.0)
            elif decoy.route in {"ground_convoy", "tank_hold"}:
                x, y, heading = self._ground_route_pose(
                    elapsed_s=t,
                    phase=decoy.phase,
                    speed=decoy.speed,
                    lateral_bias=decoy.lateral_bias,
                    depth_bias=decoy.depth_bias,
                    route=decoy.route,
                    tank_spin=decoy.kind == "tank",
                )
                z = self._terrain_height(x, y) + 0.34
                decoy.node.setPos(x, y, z)
                decoy.node.setHpr(heading, 0.0, 0.0)
            else:
                direction = -1.0 if decoy.lateral_bias < 0.0 else 1.0
                angle = (t * decoy.speed * 0.28 * math.tau * direction) + decoy.phase
                radius_x = 54.0 + abs(decoy.lateral_bias) + (math.sin((t * 0.5) + decoy.phase) * 5.0)
                radius_y = 48.0 + decoy.depth_bias + (math.cos((t * 0.4) + decoy.phase) * 6.0)
                x = math.cos(angle) * radius_x
                y = math.sin(angle) * radius_y
                z = max(
                    self._terrain_height(x, y) + 7.0,
                    10.0 + decoy.altitude + (math.cos((t * 1.1) + decoy.phase) * 2.8),
                )
                heading = 90.0 - math.degrees(angle) + (180.0 if direction < 0.0 else 0.0)
                decoy.node.setPos(x, y, z)
                decoy.node.setHpr(
                    heading,
                    math.cos((t * 1.2) + decoy.phase) * 8.0,
                    math.sin((t * 1.4) + decoy.phase) * 20.0,
                )

    def _update_clouds(self) -> None:
        for idx, cloud in enumerate(self._clouds):
            drift = (self._elapsed_s * (0.10 + (idx * 0.018))) + (idx * 0.7)
            radius = self._world_extent_distance(86.0 + (idx * 9.0))
            cloud.setPos(
                math.cos(drift) * radius,
                math.sin(drift) * radius,
                24.0 + (idx * 2.0) + (math.sin(self._elapsed_s + idx) * 0.9),
            )

    def _update_overlay_state(self, *, payload: RapidTrackingPayload | None) -> None:
        from panda3d.core import Point3

        node = self._active_target_node
        if node is None:
            self._target_overlay_state = None
            return

        world_pos = node.getPos(self._base.render)
        kind = str(self._active_target_kind).strip().lower()
        if kind == "soldier":
            world_pos = Point3(world_pos.x, world_pos.y, world_pos.z + 0.48)
        elif kind == "truck":
            world_pos = Point3(world_pos.x, world_pos.y, world_pos.z + 0.42)
        elif kind == "building":
            world_pos = Point3(world_pos.x, world_pos.y, world_pos.z + 1.8)
        elif kind == "helicopter":
            world_pos = Point3(world_pos.x, world_pos.y, world_pos.z + 0.8)
        else:
            world_pos = Point3(world_pos.x, world_pos.y, world_pos.z + 0.6)

        cam_pos = self._base.cam.getRelativePoint(self._base.render, world_pos)
        fov = self._base.camLens.getFov()
        screen_x, screen_y, on_screen, in_front = self._camera_space_to_viewport(
            cam_x=float(cam_pos.x),
            cam_y=float(cam_pos.y),
            cam_z=float(cam_pos.z),
            size=self._size,
            h_fov_deg=float(fov[0]),
            v_fov_deg=float(fov[1]),
        )
        target_visible = bool(payload.target_visible) if payload is not None else True
        self._target_overlay_state = RapidTrackingOverlayState(
            screen_x=float(screen_x),
            screen_y=float(screen_y),
            on_screen=bool(on_screen),
            in_front=bool(in_front),
            target_visible=bool(target_visible),
        )

    def _camera_basis(self):
        from panda3d.core import Vec3

        cam_pos = self._base.cam.getPos(self._base.render)
        forward = self._base.render.getRelativeVector(self._base.cam, Vec3(0.0, 1.0, 0.0))
        right = self._base.render.getRelativeVector(self._base.cam, Vec3(1.0, 0.0, 0.0))
        up = self._base.render.getRelativeVector(self._base.cam, Vec3(0.0, 0.0, 1.0))
        if forward.lengthSquared() > 1e-6:
            forward.normalize()
        if right.lengthSquared() > 1e-6:
            right.normalize()
        if up.lengthSquared() > 1e-6:
            up.normalize()
        return cam_pos, forward, right, up

    def _scene_direction(self, *, scene_x: float, scene_y: float, progress: float):
        from panda3d.core import Vec3

        local_yaw_deg = _clamp(
            (float(scene_x) / self._SCENE_X_SPAN) * self._TARGET_SWEEP_LIMIT_DEG,
            -self._TARGET_SWEEP_LIMIT_DEG,
            self._TARGET_SWEEP_LIMIT_DEG,
        )
        yaw_deg = self._helicopter_heading_deg(progress=float(progress)) + local_yaw_deg
        pitch_deg = _clamp((-float(scene_y) * 11.5) - (float(progress) * 3.0), -72.0, 52.0)
        self._direction_probe.setPos(0.0, 0.0, 0.0)
        self._direction_probe.setHpr(yaw_deg, pitch_deg, 0.0)
        direction = self._base.render.getRelativeVector(self._direction_probe, Vec3(0.0, 1.0, 0.0))
        if direction.lengthSquared() > 1e-6:
            direction.normalize()
        return direction

    def _helicopter_heading_deg(self, *, progress: float) -> float:
        rig = self._camera_rig_state(
            elapsed_s=self._elapsed_s,
            progress=progress,
            cam_x=0.0,
            cam_y=0.0,
            zoom=0.0,
            target_kind="",
            target_rel_x=0.0,
            target_rel_y=0.0,
            assist_strength=0.0,
            turbulence_strength=0.65,
        )
        return rig.carrier_heading_deg

    def _world_ray_from_lens(self, *, lens_x: float, lens_y: float):
        from panda3d.core import Point2, Point3

        near_point = Point3()
        far_point = Point3()
        if not self._base.camLens.extrude(Point2(lens_x, lens_y), near_point, far_point):
            return None
        world_near = self._base.render.getRelativePoint(self._base.cam, near_point)
        world_far = self._base.render.getRelativePoint(self._base.cam, far_point)
        return world_near, world_far

    def _world_point_on_plane(self, *, lens_x: float, lens_y: float, plane):
        from panda3d.core import Point3

        ray = self._world_ray_from_lens(lens_x=lens_x, lens_y=lens_y)
        if ray is None:
            return None
        world_near, world_far = ray
        intersection = Point3()
        if not plane.intersectsLine(intersection, world_near, world_far):
            return None
        return (float(intersection.x), float(intersection.y), float(intersection.z))

    def _world_point_on_terrain(self, *, lens_x: float, lens_y: float, clearance: float):
        ray = self._world_ray_from_lens(lens_x=lens_x, lens_y=lens_y)
        if ray is None:
            return None
        world_near, world_far = ray
        prev_point: tuple[float, float, float] | None = None
        prev_gap: float | None = None
        steps = 120
        for idx in range(steps + 1):
            t = idx / float(steps)
            x = float(world_near.x + ((world_far.x - world_near.x) * t))
            y = float(world_near.y + ((world_far.y - world_near.y) * t))
            z = float(world_near.z + ((world_far.z - world_near.z) * t))
            gap = z - (self._terrain_height(x, y) + clearance)
            if gap <= 0.0:
                if prev_point is None or prev_gap is None:
                    return (x, y, self._terrain_height(x, y) + clearance)
                blend = 0.0 if (prev_gap - gap) == 0.0 else prev_gap / (prev_gap - gap)
                blend = _clamp(blend, 0.0, 1.0)
                hit_x = prev_point[0] + ((x - prev_point[0]) * blend)
                hit_y = prev_point[1] + ((y - prev_point[1]) * blend)
                return (hit_x, hit_y, self._terrain_height(hit_x, hit_y) + clearance)
            prev_point = (x, y, z)
            prev_gap = gap
        return None

    def _world_point_on_terrain_along_direction(self, *, origin, direction, clearance: float):
        from panda3d.core import Vec3

        ray_direction = Vec3(direction)
        if ray_direction.lengthSquared() <= 1e-6:
            return None
        ray_direction.normalize()
        prev_point: tuple[float, float, float] | None = None
        prev_gap: float | None = None
        max_distance = self._GROUND_INTERSECT_MAX_DISTANCE
        steps = 140
        for idx in range(1, steps + 1):
            distance = (idx / float(steps)) * max_distance
            x = float(origin.x + (ray_direction.x * distance))
            y = float(origin.y + (ray_direction.y * distance))
            z = float(origin.z + (ray_direction.z * distance))
            gap = z - (self._terrain_height(x, y) + clearance)
            if gap <= 0.0:
                if prev_point is None or prev_gap is None:
                    return (x, y, self._terrain_height(x, y) + clearance)
                blend = 0.0 if (prev_gap - gap) == 0.0 else prev_gap / (prev_gap - gap)
                blend = _clamp(blend, 0.0, 1.0)
                hit_x = prev_point[0] + ((x - prev_point[0]) * blend)
                hit_y = prev_point[1] + ((y - prev_point[1]) * blend)
                return (hit_x, hit_y, self._terrain_height(hit_x, hit_y) + clearance)
            prev_point = (x, y, z)
            prev_gap = gap
        return None

    def _build_terrain_mesh(self):
        from panda3d.core import (
            Geom,
            GeomNode,
            GeomTriangles,
            GeomVertexData,
            GeomVertexFormat,
            GeomVertexWriter,
        )

        x_steps = 108
        y_steps = 108
        x_min = -self._TERRAIN_HALF_SPAN
        x_span = self._TERRAIN_HALF_SPAN * 2.0
        y_min = -self._TERRAIN_HALF_SPAN
        y_span = self._TERRAIN_HALF_SPAN * 2.0

        fmt = GeomVertexFormat.getV3n3c4()
        vdata = GeomVertexData("rapid-terrain", fmt, Geom.UHStatic)
        vertex = GeomVertexWriter(vdata, "vertex")
        normal = GeomVertexWriter(vdata, "normal")
        color = GeomVertexWriter(vdata, "color")
        tris = GeomTriangles(Geom.UHStatic)

        def index(ix: int, iy: int) -> int:
            return iy * (x_steps + 1) + ix

        for iy in range(y_steps + 1):
            y_t = iy / float(y_steps)
            world_y = y_min + (y_t * y_span)
            for ix in range(x_steps + 1):
                x_t = ix / float(x_steps)
                world_x = x_min + (x_t * x_span)
                world_z = self._terrain_height(world_x, world_y)
                vertex.addData3f(world_x, world_y, world_z)

                eps = 0.35
                dh_dx = (
                    self._terrain_height(world_x + eps, world_y)
                    - self._terrain_height(world_x - eps, world_y)
                ) / (eps * 2.0)
                dh_dy = (
                    self._terrain_height(world_x, world_y + eps)
                    - self._terrain_height(world_x, world_y - eps)
                ) / (eps * 2.0)
                nx = -dh_dx
                ny = -dh_dy
                nz = 1.0
                normal_len = math.sqrt((nx * nx) + (ny * ny) + (nz * nz))
                normal.addData3f(nx / normal_len, ny / normal_len, nz / normal_len)

                depth = _clamp((world_y - y_min) / y_span, 0.0, 1.0)
                green = 0.46 - (depth * 0.18)
                red = 0.26 - (depth * 0.08)
                blue = 0.18 - (depth * 0.06)
                if world_z > 1.5:
                    red += 0.08
                    green += 0.06
                    blue += 0.04
                color.addData4f(red, green, blue, 1.0)

        for iy in range(y_steps):
            for ix in range(x_steps):
                i0 = index(ix, iy)
                i1 = index(ix + 1, iy)
                i2 = index(ix + 1, iy + 1)
                i3 = index(ix, iy + 1)
                tris.addVertices(i0, i1, i2)
                tris.addVertices(i0, i2, i3)

        geom = Geom(vdata)
        geom.addPrimitive(tris)
        geom_node = GeomNode("rapid-terrain-node")
        geom_node.addGeom(geom)
        return self._terrain_root.attachNewNode(geom_node)

    @staticmethod
    def _terrain_height(x: float, y: float) -> float:
        px = float(x)
        py = float(y)
        dist = math.sqrt((px * px) + (py * py))
        ang = math.atan2(py, px)
        ring = _clamp((dist - 18.0) / 92.0, 0.0, 1.0)
        far_ring = _clamp((dist - 54.0) / 86.0, 0.0, 1.0)
        undulation = (
            math.sin((px * 0.09) + (py * 0.04))
            + math.cos((py * 0.08) - (px * 0.03))
        ) * 0.10 * max(0.0, 1.0 - (dist / 120.0))
        ridge_a = math.sin((ang * 3.6) + (dist * 0.046) + 0.4) * (0.7 + (1.6 * ring))
        ridge_b = math.cos((ang * 5.1) - (dist * 0.034) + 1.3) * (0.4 + (1.4 * far_ring))
        ridge_c = math.sin((ang * 7.3) + (dist * 0.024) - 0.8) * (0.2 + (0.9 * far_ring))
        bowl = -0.16 + (0.0012 * min(dist, 42.0))
        return bowl + undulation + (ridge_a * ring * 1.5) + (ridge_b * far_ring * 1.1) + (ridge_c * far_ring * 0.8)

    def _place_scenery(
        self,
        *,
        asset_id: str,
        fallback: str,
        pos: tuple[float, float, float],
        scale: float,
    ) -> None:
        self._load_asset_or_fallback(
            asset_id=asset_id,
            fallback=fallback,
            color=(1.0, 1.0, 1.0, 1.0),
            scale=scale,
            parent=self._scenery_root,
            pos=pos,
        )

    def _build_cloud(self, *, scale: float):
        from panda3d.core import CardMaker, NodePath

        root = NodePath("cloud")
        for idx, offset in enumerate((-1.0, 0.0, 1.0)):
            maker = CardMaker(f"cloud-{idx}")
            maker.setFrame(-3.6, 3.6, -1.4, 1.4)
            card = root.attachNewNode(maker.generate())
            card.setBillboardPointEye()
            card.setTransparency(True)
            card.setColor(0.96, 0.98, 1.0, 0.22)
            card.setPos(offset * 3.8, idx * 0.08, (1 - idx) * 0.25)
        root.setScale(scale)
        return root

    @classmethod
    def _world_extent_distance(cls, value: float) -> float:
        return float(value) * cls._WORLD_EXTENT_SCALE

    def _load_asset_or_fallback(
        self,
        *,
        asset_id: str,
        fallback: str,
        color: tuple[float, float, float, float],
        scale: float,
        parent,
        pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        entry = self._catalog.entry(asset_id)
        resolved = self._catalog.resolve_path(asset_id)
        node = None
        if resolved is not None:
            try:
                node = self._load_model(resolved)
                if entry is not None:
                    scale *= entry.scale
            except Exception:
                node = None
        if node is None:
            fallback_kind = entry.fallback if entry is not None else fallback
            node = self._build_fallback_model(kind=fallback_kind, color=color)
        node.reparentTo(parent)
        node.setPos(*pos)
        node.setScale(scale)
        return node

    def _load_model(self, path: Path):
        suffix = path.suffix.lower()
        if suffix in {".glb", ".gltf"}:
            import gltf

            return gltf.load_model(path)
        return self._base.loader.loadModel(str(path))

    def _build_fallback_model(
        self,
        *,
        kind: str,
        color: tuple[float, float, float, float],
    ):
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
        elif kind == "tank":
            hull = self._make_box(size=(1.8, 2.8, 0.6), color=color)
            turret = self._make_box(size=(0.9, 1.0, 0.42), color=(0.58, 0.60, 0.30, 1.0))
            barrel = self._make_box(size=(0.14, 1.4, 0.12), color=(0.32, 0.34, 0.18, 1.0))
            track_left = self._make_box(size=(0.28, 2.6, 0.42), color=(0.18, 0.18, 0.18, 1.0))
            track_right = self._make_box(size=(0.28, 2.6, 0.42), color=(0.18, 0.18, 0.18, 1.0))
            turret.setZ(0.42)
            barrel.setPos(0.0, 1.25, 0.46)
            track_left.setX(-0.82)
            track_right.setX(0.82)
            for child in (hull, turret, barrel, track_left, track_right):
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
        elif kind == "soldier":
            body = self._make_box(size=(0.24, 0.16, 0.54), color=color)
            vest = self._make_box(size=(0.14, 0.18, 0.24), color=(0.92, 0.78, 0.24, 1.0))
            head = self._make_box(size=(0.18, 0.18, 0.18), color=(0.80, 0.72, 0.62, 1.0))
            leg_left = self._make_box(size=(0.08, 0.08, 0.30), color=(0.16, 0.20, 0.14, 1.0))
            leg_right = self._make_box(size=(0.08, 0.08, 0.30), color=(0.16, 0.20, 0.14, 1.0))
            body.setZ(0.44)
            vest.setY(0.04)
            vest.setZ(0.44)
            head.setZ(0.82)
            leg_left.setX(-0.06)
            leg_left.setZ(0.15)
            leg_right.setX(0.06)
            leg_right.setZ(0.15)
            for child in (body, vest, head, leg_left, leg_right):
                child.reparentTo(root)
        elif kind == "soldiers":
            for offset in (-0.32, 0.0, 0.32):
                unit = self._build_fallback_model(kind="soldier", color=color)
                unit.setX(offset)
                unit.reparentTo(root)
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
        else:
            self._make_box(size=(1.0, 1.0, 1.0), color=color).reparentTo(root)
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
