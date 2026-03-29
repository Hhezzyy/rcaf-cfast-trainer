from __future__ import annotations

import importlib.util
import math
import os
from dataclasses import dataclass
from pathlib import Path

import pygame

from .aircraft_art import (
    build_panda_palette,
    build_panda3d_fixed_wing_model,
)
from .panda3d_assets import Panda3DAssetCatalog
from .rapid_tracking import (
    RapidTrackingCompoundLayout,
    RapidTrackingPayload,
    build_rapid_tracking_compound_layout,
)
from .rapid_tracking_view import (
    RapidTrackingCameraRigState,
    camera_rig_state as shared_camera_rig_state,
    rapid_tracking_seed_unit,
)


def panda3d_rapid_tracking_rendering_available() -> bool:
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
    _TARGET_VIEW_LIMIT = 1.3
    _TERRAIN_HALF_SPAN = 260.0 * _WORLD_EXTENT_SCALE
    _GROUND_INTERSECT_MAX_DISTANCE = 320.0
    _RECENTER_ENTER_H_RATIO = 0.88
    _RECENTER_ENTER_V_RATIO = 0.82
    _RECENTER_EXIT_H_RATIO = 0.62
    _RECENTER_EXIT_V_RATIO = 0.62
    _AIR_MOTION_SAMPLE_DT_S = 0.08

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
        self._session_seed = 0
        self._layout: RapidTrackingCompoundLayout | None = None
        self._catalog = Panda3DAssetCatalog()
        self._active_target_node = None
        self._active_target_kind = ""
        self._active_scenery_target_node = None
        self._selected_target_node = None
        self._selected_target_kind = ""
        self._selected_target_variant = ""
        self._target_overlay_state: RapidTrackingOverlayState | None = None
        self._last_camera_rig: RapidTrackingCameraRigState | None = None
        self._scenery_nodes_by_kind: dict[str, list[object]] = {
            "hangar": [],
            "tower": [],
            "truck": [],
            "soldiers": [],
        }

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
        self._road_root = self._world_root.attachNewNode("rapid-roads")
        self._scenery_root = self._world_root.attachNewNode("rapid-scenery")
        self._decoy_root = self._world_root.attachNewNode("rapid-decoys")
        self._target_root = self._world_root.attachNewNode("rapid-targets")
        self._cloud_root = self._world_root.attachNewNode("rapid-clouds")
        self._direction_probe = self._base.render.attachNewNode("rapid-direction-probe")
        self._road_segment_nodes: list[object] = []
        self._road_intersection_nodes: list[object] = []

        self._build_world()

    @property
    def size(self) -> tuple[int, int]:
        return self._size

    def target_overlay_state(self) -> RapidTrackingOverlayState | None:
        return self._target_overlay_state

    def last_camera_rig_state(self) -> RapidTrackingCameraRigState | None:
        return self._last_camera_rig

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
        seed: int = 0,
        progress: float,
        camera_yaw_deg: float | None,
        camera_pitch_deg: float | None,
        zoom: float,
        target_kind: str,
        target_world_x: float = 0.0,
        target_world_y: float = 0.0,
        focus_world_x: float = 0.0,
        focus_world_y: float = 70.0,
        turbulence_strength: float,
    ) -> RapidTrackingCameraRigState:
        return shared_camera_rig_state(
            elapsed_s=elapsed_s,
            seed=seed,
            progress=progress,
            camera_yaw_deg=camera_yaw_deg,
            camera_pitch_deg=camera_pitch_deg,
            zoom=zoom,
            target_kind=target_kind,
            target_world_x=target_world_x,
            target_world_y=target_world_y,
            focus_world_x=focus_world_x,
            focus_world_y=focus_world_y,
            turbulence_strength=turbulence_strength,
        )

    def render(self, *, payload: RapidTrackingPayload | None) -> pygame.Surface:
        if payload is None:
            now_ms = pygame.time.get_ticks()
            dt_s = max(0.0, min(0.05, (now_ms - self._last_render_ms) / 1000.0))
            self._last_render_ms = now_ms
            self._elapsed_s += dt_s
            self._ensure_seeded_layout(seed=0)
        else:
            self._last_render_ms = pygame.time.get_ticks()
            self._elapsed_s = max(0.0, float(payload.phase_elapsed_s))
            self._ensure_seeded_layout(seed=int(payload.session_seed))

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
        return float(math.degrees(math.atan2(vx, vy)) % 360.0)

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
            heading = 90.0 if forward > 0.0 else 270.0

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

    @staticmethod
    def _basis_vectors_from_angles(
        *,
        heading_deg: float,
        pitch_deg: float,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
        heading_rad = math.radians(float(heading_deg))
        pitch_rad = math.radians(float(pitch_deg))
        right = (math.cos(heading_rad), -math.sin(heading_rad), 0.0)
        forward = (
            math.sin(heading_rad) * math.cos(pitch_rad),
            math.cos(heading_rad) * math.cos(pitch_rad),
            math.sin(pitch_rad),
        )
        up = (
            (right[1] * forward[2]) - (right[2] * forward[1]),
            (right[2] * forward[0]) - (right[0] * forward[2]),
            (right[0] * forward[1]) - (right[1] * forward[0]),
        )
        return right, forward, up

    @classmethod
    def _world_vector_from_camera_delta(
        cls,
        *,
        rig: RapidTrackingCameraRigState,
        cam_dx: float,
        cam_dy: float,
        cam_dz: float,
    ) -> tuple[float, float, float]:
        right, forward, up = cls._basis_vectors_from_angles(
            heading_deg=float(rig.view_heading_deg),
            pitch_deg=float(rig.view_pitch_deg),
        )
        return (
            float((right[0] * cam_dx) + (forward[0] * cam_dy) + (up[0] * cam_dz)),
            float((right[1] * cam_dx) + (forward[1] * cam_dy) + (up[1] * cam_dz)),
            float((right[2] * cam_dx) + (forward[2] * cam_dy) + (up[2] * cam_dz)),
        )

    @staticmethod
    def _world_heading_pitch_from_vector(
        *,
        dx: float,
        dy: float,
        dz: float,
        default_heading: float,
        default_pitch: float = 0.0,
    ) -> tuple[float, float]:
        horiz = math.hypot(float(dx), float(dy))
        magnitude = math.sqrt((float(dx) * float(dx)) + (float(dy) * float(dy)) + (float(dz) * float(dz)))
        if magnitude <= 1e-6:
            return float(default_heading) % 360.0, float(default_pitch)
        heading = float(default_heading) % 360.0 if horiz <= 1e-3 else float(
            math.degrees(math.atan2(float(dx), float(dy))) % 360.0
        )
        pitch = float(default_pitch) if magnitude <= 1e-6 else -math.degrees(
            math.atan2(float(dz), max(1e-6, horiz))
        )
        return float(heading), float(pitch)

    def _sample_camera_rig(
        self,
        *,
        payload: RapidTrackingPayload | None,
        elapsed_s: float,
    ) -> RapidTrackingCameraRigState:
        delta_s = max(0.0, float(elapsed_s) - float(self._elapsed_s))
        if payload is None:
            return self._camera_rig_state(
                elapsed_s=float(elapsed_s),
                seed=self._session_seed,
                progress=0.0,
                camera_yaw_deg=None,
                camera_pitch_deg=None,
                zoom=0.0,
                target_kind="",
                target_world_x=0.0,
                target_world_y=0.0,
                focus_world_x=self._carrier_focus_world()[0],
                focus_world_y=self._carrier_focus_world()[1],
                turbulence_strength=0.65,
            )
        return self._camera_rig_state(
            elapsed_s=float(elapsed_s),
            seed=int(payload.session_seed),
            progress=float(payload.scene_progress),
            camera_yaw_deg=float(payload.camera_yaw_deg),
            camera_pitch_deg=float(payload.camera_pitch_deg),
            zoom=float(payload.capture_zoom),
            target_kind=str(payload.target_kind),
            target_world_x=float(payload.target_world_x) + (float(payload.target_vx) * delta_s),
            target_world_y=float(payload.target_world_y) + (float(payload.target_vy) * delta_s),
            focus_world_x=float(payload.focus_world_x),
            focus_world_y=float(payload.focus_world_y),
            turbulence_strength=float(payload.turbulence_strength),
        )

    def _airborne_apparent_hpr(
        self,
        *,
        cache_key: int,
        current_pos: tuple[float, float, float],
        next_pos: tuple[float, float, float],
        current_rig: RapidTrackingCameraRigState,
        next_rig: RapidTrackingCameraRigState,
        default_heading: float,
        pitch_flair: float,
        roll_flair: float,
        pitch_limit: float,
        roll_limit: float,
    ) -> tuple[float, float, float]:
        _ = cache_key
        world_dx = float(next_pos[0] - current_pos[0])
        world_dy = float(next_pos[1] - current_pos[1])
        world_dz = float(next_pos[2] - current_pos[2])
        heading_deg, base_pitch = self._world_heading_pitch_from_vector(
            dx=world_dx,
            dy=world_dy,
            dz=world_dz,
            default_heading=float(default_heading),
            default_pitch=0.0,
        )
        hpr = (
            float(heading_deg),
            _clamp((float(base_pitch) * 0.55) + float(pitch_flair), -float(pitch_limit), float(pitch_limit)),
            _clamp(float(roll_flair), -float(roll_limit), float(roll_limit)),
        )
        return hpr

    @staticmethod
    def _apply_dynamic_target_visibility(*, node, visible: bool) -> None:
        if bool(visible):
            node.setAlphaScale(1.0)
            node.show()
            return
        node.setAlphaScale(0.0)
        node.hide()

    def _decoy_air_pose(
        self,
        *,
        decoy: _RapidTrackingDecoy,
        elapsed_s: float,
    ) -> tuple[tuple[float, float, float], float, float, float]:
        t = float(elapsed_s)
        direction = -1.0 if decoy.lateral_bias < 0.0 else 1.0
        if decoy.route == "air_cross":
            angle = (t * decoy.speed * 0.34 * math.tau * direction) + decoy.phase
            radius = 62.0 + decoy.depth_bias + (math.sin((t * 0.44) + decoy.phase) * 8.0)
            x = math.cos(angle) * radius + math.sin((t * 1.2) + decoy.phase) * 6.0
            y = math.sin(angle) * radius + math.cos((t * 0.9) + decoy.phase) * 4.0
            z = max(
                self._terrain_height(x, y) + 9.0,
                12.0 + decoy.altitude + (math.cos((t * 1.1) + decoy.phase) * 3.4),
            )
            return (
                (float(x), float(y), float(z)),
                float(90.0 - math.degrees(angle) + (180.0 if direction < 0.0 else 0.0)),
                float(math.cos((t * 1.3) + decoy.phase) * 10.0),
                float(math.sin((t * 1.6) + decoy.phase) * 22.0),
            )
        if decoy.route == "helo_arc":
            angle = (t * decoy.speed * 0.22 * math.tau * direction) + decoy.phase
            radius_x = 34.0 + (decoy.depth_bias * 0.55)
            radius_y = 28.0 + (decoy.depth_bias * 0.42)
            x = math.cos(angle) * radius_x + math.sin((t * 0.8) + decoy.phase) * 5.0
            y = math.sin(angle) * radius_y + math.cos((t * 0.6) + decoy.phase) * 3.5
            z = max(
                self._terrain_height(x, y) + 5.5,
                8.0 + decoy.altitude + math.sin((t * 1.7) + decoy.phase) * 1.5,
            )
            return (
                (float(x), float(y), float(z)),
                float(90.0 - math.degrees(angle) + (180.0 if direction < 0.0 else 0.0)),
                float(math.cos((t * 1.1) + decoy.phase) * 6.0),
                float(math.sin((t * 2.8) + decoy.phase) * 8.0),
            )
        angle = (t * decoy.speed * 0.28 * math.tau * direction) + decoy.phase
        radius_x = 54.0 + abs(decoy.lateral_bias) + (math.sin((t * 0.5) + decoy.phase) * 5.0)
        radius_y = 48.0 + decoy.depth_bias + (math.cos((t * 0.4) + decoy.phase) * 6.0)
        x = math.cos(angle) * radius_x
        y = math.sin(angle) * radius_y
        z = max(
            self._terrain_height(x, y) + 7.0,
            10.0 + decoy.altitude + (math.cos((t * 1.1) + decoy.phase) * 2.8),
        )
        return (
            (float(x), float(y), float(z)),
            float(90.0 - math.degrees(angle) + (180.0 if direction < 0.0 else 0.0)),
            float(math.cos((t * 1.2) + decoy.phase) * 8.0),
            float(math.sin((t * 1.4) + decoy.phase) * 20.0),
        )

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

        self._jet_pool = [
            self._load_asset_or_fallback(
                asset_id="plane_red",
                fallback="plane",
                color=(0.88, 0.20, 0.16, 1.0),
                scale=1.56,
                parent=self._target_root,
            ),
            self._load_asset_or_fallback(
                asset_id="plane_yellow",
                fallback="plane",
                color=(0.90, 0.76, 0.24, 1.0),
                scale=1.42,
                parent=self._target_root,
            ),
        ]
        self._helicopter_pool = [
            self._load_asset_or_fallback(
                asset_id="helicopter_green",
                fallback="helicopter",
                color=(0.24, 0.54, 0.32, 1.0),
                scale=1.34,
                parent=self._target_root,
            ),
            self._load_asset_or_fallback(
                asset_id="helicopter_green",
                fallback="helicopter",
                color=(0.18, 0.44, 0.28, 1.0),
                scale=1.22,
                parent=self._target_root,
            ),
        ]
        self._truck_pool = [
            self._load_asset_or_fallback(
                asset_id="truck_olive",
                fallback="truck",
                color=(0.38, 0.42, 0.24, 1.0),
                scale=1.24,
                parent=self._target_root,
            ),
            self._load_asset_or_fallback(
                asset_id="truck_olive",
                fallback="truck",
                color=(0.30, 0.36, 0.20, 1.0),
                scale=1.16,
                parent=self._target_root,
            ),
        ]
        self._tracked_pool = [
            self._build_fallback_model(
                kind="tank",
                color=(0.44, 0.48, 0.28, 1.0),
            ),
            self._build_fallback_model(
                kind="tank",
                color=(0.34, 0.38, 0.22, 1.0),
            ),
        ]
        for node in self._tracked_pool:
            node.reparentTo(self._target_root)
        self._soldier_pool = [
            self._build_soldier_squad(
                leader_color=(0.74, 0.28, 0.24, 1.0),
                wing_color=(0.18, 0.24, 0.16, 1.0),
            ),
            self._build_soldier_squad(
                leader_color=(0.78, 0.34, 0.28, 1.0),
                wing_color=(0.20, 0.26, 0.18, 1.0),
            ),
        ]
        for node in self._soldier_pool:
            node.reparentTo(self._target_root)

        self._dynamic_target_nodes = (
            *self._jet_pool,
            *self._helicopter_pool,
            *self._truck_pool,
            *self._tracked_pool,
            *self._soldier_pool,
        )
        for node in self._dynamic_target_nodes:
            node.setTransparency(True)
        self._decoys = []
        self._clouds = []
        self._ensure_seeded_layout(seed=0)
        self._hide_target_nodes()

    def _ensure_seeded_layout(self, *, seed: int) -> None:
        normalized_seed = int(seed)
        if self._layout is not None and self._session_seed == normalized_seed:
            return
        self._session_seed = normalized_seed
        self._layout = build_rapid_tracking_compound_layout(seed=normalized_seed)
        self._rebuild_seeded_scenery(layout=self._layout)

    @staticmethod
    def _clear_node_children(root) -> None:
        for child in list(root.getChildren()):
            child.removeNode()

    def _scene_track_world_xy(self, *, track_x: float, track_y: float) -> tuple[float, float]:
        lateral_bias = 0.0 if self._layout is None else float(self._layout.path_lateral_bias)
        x = (float(track_x) * 34.0) + (lateral_bias * 14.0)
        y = 70.0 + ((float(track_y) + 0.18) * 56.0)
        return float(x), float(y)

    def _world_pos_from_anchor(self, anchor, *, y_bias: float = 0.0) -> tuple[float, float, float]:
        x, y = self._scene_track_world_xy(track_x=float(anchor.x), track_y=float(anchor.y))
        y += y_bias
        z = self._terrain_height(x, y)
        return float(x), float(y), float(z)

    def _ground_target_world_pos(
        self,
        *,
        world_x: float,
        world_y: float,
        clearance: float,
    ) -> tuple[float, float, float]:
        return (
            float(world_x),
            float(world_y),
            float(self._terrain_height(world_x, world_y) + clearance),
        )

    def _air_target_world_pos(
        self,
        *,
        kind: str,
        world_x: float,
        world_y: float,
        phase_elapsed_s: float,
        scene_progress: float,
    ) -> tuple[float, float, float]:
        terrain_z = float(self._terrain_height(world_x, world_y))
        phase = (
            (float(phase_elapsed_s) * (1.4 if str(kind).strip().lower() == "helicopter" else 2.0))
            + (float(world_x) * 0.018)
            + (float(world_y) * 0.014)
            + (self._session_seed * 0.00073)
        )
        if str(kind).strip().lower() == "helicopter":
            clearance = 9.0 + (math.sin(phase) * 1.4) + (math.cos((phase * 0.58) + 0.35) * 0.9)
            clearance += _lerp(-0.4, 0.7, _smoothstep(0.0, 1.0, float(scene_progress)))
            minimum = 6.4
        else:
            clearance = 17.0 + (math.sin(phase) * 1.9) + (math.cos((phase * 0.52) + 0.7) * 1.2)
            clearance += _lerp(0.0, 2.8, _smoothstep(0.18, 1.0, float(scene_progress)))
            minimum = 12.0
        return (
            world_x,
            world_y,
            float(max(terrain_z + minimum, terrain_z + clearance)),
        )

    def _carrier_focus_world(self) -> tuple[float, float]:
        if self._layout is None:
            return (0.0, 70.0)
        return self._scene_track_world_xy(
            track_x=float(self._layout.compound_center_x),
            track_y=float(self._layout.compound_center_y),
        )

    def _rebuild_seeded_scenery(self, *, layout: RapidTrackingCompoundLayout) -> None:
        self._clear_node_children(self._road_root)
        self._clear_node_children(self._scenery_root)
        self._clear_node_children(self._decoy_root)
        self._clear_node_children(self._cloud_root)
        self._road_segment_nodes = []
        self._road_intersection_nodes = []
        self._scenery_nodes_by_kind = {
            "hangar": [],
            "tower": [],
            "truck": [],
            "soldiers": [],
        }
        self._selected_target_node = None
        self._selected_target_kind = ""
        self._selected_target_variant = ""
        self._build_seeded_roads(layout=layout)
        self._build_seeded_obstacles(layout=layout)
        self._build_seeded_compound(layout=layout)
        self._build_seeded_foliage(layout=layout)
        self._build_seeded_ring(layout=layout)
        self._decoys = self._build_traffic_pool(layout=layout)
        self._clouds = [
            self._build_cloud(scale=scale) for scale in (1.0, 1.35, 0.92, 1.22, 1.62, 1.85)
        ]
        for idx, cloud in enumerate(self._clouds):
            ang = layout.orbit_phase_offset + (layout.orbit_direction * idx * 0.78)
            radius = self._world_extent_distance(92.0 + (idx * 12.0 * layout.orbit_radius_scale))
            cloud.reparentTo(self._cloud_root)
            cloud.setPos(
                math.cos(ang) * radius,
                math.sin(ang) * radius,
                23.0 + layout.altitude_bias + (idx * 2.3),
            )

    def _build_seeded_foliage(self, *, layout: RapidTrackingCompoundLayout) -> None:
        for cluster in (*layout.shrub_clusters, *layout.tree_clusters, *layout.forest_clusters):
            for idx in range(max(1, int(cluster.count))):
                angle = rapid_tracking_seed_unit(
                    seed=layout.seed,
                    salt=f"{cluster.cluster_id}:angle:{idx}",
                ) * math.tau
                radius = math.sqrt(
                    rapid_tracking_seed_unit(
                        seed=layout.seed,
                        salt=f"{cluster.cluster_id}:radius:{idx}",
                    )
                ) * float(cluster.radius)
                jitter_x = math.cos(angle) * radius
                jitter_y = math.sin(angle) * radius * (0.72 if cluster.fallback == "forest" else 1.0)
                world_x, world_y = self._scene_track_world_xy(
                    track_x=float(cluster.x) + jitter_x,
                    track_y=float(cluster.y) + jitter_y,
                )
                scale = float(cluster.scale) * _lerp(
                    0.82,
                    1.18,
                    rapid_tracking_seed_unit(
                        seed=layout.seed,
                        salt=f"{cluster.cluster_id}:scale:{idx}",
                    ),
                )
                node = self._place_scenery(
                    asset_id=str(cluster.asset_id),
                    fallback=str(cluster.fallback),
                    pos=(world_x, world_y, self._terrain_height(world_x, world_y)),
                    scale=scale,
                )
                node.setHpr(
                    rapid_tracking_seed_unit(
                        seed=layout.seed,
                        salt=f"{cluster.cluster_id}:heading:{idx}",
                    )
                    * 360.0,
                    0.0,
                    0.0,
                )

    @staticmethod
    def _seeded_road_segment_ids() -> tuple[tuple[str, str], ...]:
        return ()

    def _build_seeded_roads(self, *, layout: RapidTrackingCompoundLayout) -> None:
        for road_segment in layout.road_segments:
            self._build_road_segment_points(
                points=tuple((float(x), float(y)) for x, y in road_segment.points),
                surface=str(road_segment.surface),
                seed=layout.seed,
            )

        road_anchor_lookup = {anchor.anchor_id: anchor for anchor in layout.road_anchors}
        seen_intersections: set[str] = set()
        for road_segment in layout.road_segments:
            for anchor_id in (road_segment.start_anchor_id, road_segment.end_anchor_id):
                if anchor_id in seen_intersections:
                    continue
                anchor = road_anchor_lookup.get(anchor_id)
                if anchor is None:
                    continue
                seen_intersections.add(anchor_id)
                self._build_road_intersection(anchor=anchor, seed=layout.seed)

    def _build_road_segment(self, *, start_anchor, end_anchor, seed: int) -> None:
        self._build_road_segment_points(
            points=((float(start_anchor.x), float(start_anchor.y)), (float(end_anchor.x), float(end_anchor.y))),
            surface="paved",
            seed=seed,
        )

    def _build_road_segment_points(
        self,
        *,
        points: tuple[tuple[float, float], ...],
        surface: str,
        seed: int,
    ) -> None:
        from panda3d.core import NodePath

        if len(points) < 2:
            return

        wear = rapid_tracking_seed_unit(seed=int(seed), salt=f"road:{surface}:{len(points)}:wear")
        paved = str(surface).strip().lower() == "paved"
        road_width = _lerp(8.1, 9.6, wear) if paved else _lerp(5.4, 6.8, wear)
        shoulder_width = road_width + _lerp(2.6, 3.4, wear)
        stripe_width = _lerp(0.26, 0.38, wear) if paved else 0.0
        piece_overlap = 1.2 if paved else 0.8

        root = NodePath(f"rapid-road-segment-{surface}-{len(self._road_segment_nodes)}")
        root.reparentTo(self._road_root)
        for idx, (start_xy, end_xy) in enumerate(zip(points, points[1:], strict=False)):
            seg_start_x, seg_start_y = self._scene_track_world_xy(track_x=float(start_xy[0]), track_y=float(start_xy[1]))
            seg_end_x, seg_end_y = self._scene_track_world_xy(track_x=float(end_xy[0]), track_y=float(end_xy[1]))
            seg_start_z = self._terrain_height(seg_start_x, seg_start_y)
            seg_end_z = self._terrain_height(seg_end_x, seg_end_y)
            seg_dx = seg_end_x - seg_start_x
            seg_dy = seg_end_y - seg_start_y
            seg_length = max(1.0, math.hypot(seg_dx, seg_dy))
            seg_heading_deg = self._heading_from_velocity(vx=seg_dx, vy=seg_dy)
            seg_pitch_deg = -math.degrees(
                math.atan2(seg_end_z - seg_start_z, max(1e-3, seg_length))
            )
            seg_mid_x = (seg_start_x + seg_end_x) * 0.5
            seg_mid_y = (seg_start_y + seg_end_y) * 0.5
            seg_mid_z = max(seg_start_z, seg_end_z, self._terrain_height(seg_mid_x, seg_mid_y)) + 0.035
            asphalt_tint = _lerp(0.0, 0.035, (idx % 2) / 1.0)

            piece_root = NodePath(f"road-piece-{idx}")
            piece_root.reparentTo(root)
            piece_root.setPos(seg_mid_x, seg_mid_y, seg_mid_z)
            piece_root.setHpr(seg_heading_deg, seg_pitch_deg, 0.0)

            shoulder = self._make_box(
                size=(shoulder_width, seg_length + piece_overlap + 1.2, 0.06),
                color=(0.36, 0.30, 0.22, 1.0) if paved else (0.34, 0.28, 0.20, 1.0),
            )
            shoulder.setZ(-0.012)
            shoulder.reparentTo(piece_root)

            asphalt = self._make_box(
                size=(road_width, seg_length + piece_overlap, 0.05),
                color=(0.13 + asphalt_tint, 0.14 + asphalt_tint, 0.15 + asphalt_tint, 1.0)
                if paved
                else (0.30, 0.24, 0.18, 1.0),
            )
            asphalt.reparentTo(piece_root)

            if paved and idx % 2 == 0:
                stripe = self._make_box(
                    size=(stripe_width, max(2.8, seg_length * 0.62), 0.014),
                    color=(0.84, 0.78, 0.54, 0.96),
                )
                stripe.setPos(0.0, 0.0, 0.035)
                stripe.reparentTo(piece_root)

        self._road_segment_nodes.append(root)

    def _build_seeded_obstacles(self, *, layout: RapidTrackingCompoundLayout) -> None:
        for obstacle in layout.obstacles:
            world_x, world_y = self._scene_track_world_xy(track_x=float(obstacle.x), track_y=float(obstacle.y))
            terrain_z = self._terrain_height(world_x, world_y)
            if obstacle.kind == "lake":
                node = self._make_box(
                    size=(float(obstacle.radius_x) * 34.0 * 2.0, float(obstacle.radius_y) * 56.0 * 2.0, 0.08),
                    color=(0.18, 0.34, 0.52, 0.92),
                )
                node.reparentTo(self._scenery_root)
                node.setPos(world_x, world_y, terrain_z - 0.08)
                node.setHpr(float(obstacle.rotation_deg), 0.0, 0.0)
            elif obstacle.kind == "hill":
                node = self._make_box(
                    size=(float(obstacle.radius_x) * 34.0 * 1.7, float(obstacle.radius_y) * 56.0 * 1.5, float(obstacle.height) * 5.5),
                    color=(0.30, 0.42, 0.24, 1.0),
                )
                node.reparentTo(self._scenery_root)
                node.setPos(world_x, world_y, terrain_z + (float(obstacle.height) * 1.2))
                node.setHpr(float(obstacle.rotation_deg), 0.0, 0.0)
            else:
                node = self._build_fallback_model(kind="tank", color=(0.42, 0.40, 0.38, 1.0))
                node.reparentTo(self._scenery_root)
                node.setScale(max(0.8, float(obstacle.radius_x) * 2.0))
                node.setPos(world_x, world_y, terrain_z + 0.24)
                node.setHpr(float(obstacle.rotation_deg), 0.0, 0.0)

    def _build_road_intersection(self, *, anchor, seed: int) -> None:
        from panda3d.core import NodePath

        x, y, z = self._world_pos_from_anchor(anchor)
        wear = rapid_tracking_seed_unit(
            seed=int(seed),
            salt=f"road:{anchor.anchor_id}:pad",
        )
        pad_size = _lerp(8.2, 9.6, wear)
        root = NodePath(f"rapid-road-junction-{anchor.anchor_id}")
        root.reparentTo(self._road_root)
        root.setPos(x, y, z + 0.034)
        shoulder = self._make_box(
            size=(pad_size + 2.8, pad_size + 2.8, 0.06),
            color=(0.34, 0.29, 0.22, 1.0),
        )
        shoulder.setZ(-0.012)
        shoulder.reparentTo(root)
        pad = self._make_box(
            size=(pad_size, pad_size, 0.052),
            color=(0.14, 0.15, 0.16, 1.0),
        )
        pad.reparentTo(root)
        diamond = self._make_box(
            size=(pad_size * 0.68, pad_size * 0.68, 0.018),
            color=(0.18, 0.19, 0.20, 0.96),
        )
        diamond.setHpr(45.0, 0.0, 0.0)
        diamond.setZ(0.016)
        diamond.reparentTo(root)
        self._road_intersection_nodes.append(root)

    def _build_seeded_compound(self, *, layout: RapidTrackingCompoundLayout) -> None:
        for anchor in layout.building_anchors:
            wx, wy, wz = self._world_pos_from_anchor(anchor)
            if anchor.variant == "tower":
                self._place_scenery(
                    asset_id="building_tower",
                    fallback="tower",
                    pos=(wx, wy, wz + 2.1),
                    scale=0.76 + (rapid_tracking_seed_unit(seed=layout.seed, salt=anchor.anchor_id) * 0.18),
                )
            else:
                self._place_scenery(
                    asset_id="building_hangar",
                    fallback="hangar",
                    pos=(wx, wy, wz + 1.1),
                    scale=0.76 + (rapid_tracking_seed_unit(seed=layout.seed, salt=anchor.anchor_id) * 0.24),
                )

        for idx, anchor in enumerate(layout.patrol_anchors):
            wx, wy, wz = self._world_pos_from_anchor(anchor, y_bias=6.0)
            if idx % 2 == 0:
                self._place_scenery(
                    asset_id="soldiers_patrol",
                    fallback="soldiers",
                    pos=(wx, wy, wz),
                    scale=0.82 + (0.08 * (idx % 3)),
                )
            else:
                self._place_scenery(
                    asset_id="trees_field_cluster",
                    fallback="trees",
                    pos=(wx, wy, wz),
                    scale=1.14 + (0.16 * (idx % 4)),
                )

        for idx, anchor in enumerate(layout.road_anchors):
            wx, wy, wz = self._world_pos_from_anchor(anchor, y_bias=4.0)
            if idx % 3 == 0:
                self._place_scenery(
                    asset_id="truck_olive",
                    fallback="truck",
                    pos=(wx, wy, wz + 0.34),
                    scale=0.78 + (0.10 * ((idx + 1) % 3)),
                )
            elif idx % 3 == 1:
                self._place_scenery(
                    asset_id="shrubs_low_cluster",
                    fallback="shrub",
                    pos=(wx, wy, wz),
                    scale=0.88 + (0.10 * (idx % 3)),
                )
            else:
                self._place_scenery(
                    asset_id="soldiers_patrol",
                    fallback="soldiers",
                    pos=(wx, wy, wz),
                    scale=0.84,
                )

        for idx, anchor in enumerate(layout.ridge_anchors):
            wx, wy, wz = self._world_pos_from_anchor(anchor, y_bias=18.0 + (idx * 8.0))
            self._place_scenery(
                asset_id="forest_canopy_patch" if idx % 2 == 0 else "trees_field_cluster",
                fallback="forest" if idx % 2 == 0 else "trees",
                pos=(wx, wy, wz),
                scale=(2.1 + (0.30 * idx)) if idx % 2 == 0 else (1.5 + (0.20 * idx)),
            )

    def _build_seeded_ring(self, *, layout: RapidTrackingCompoundLayout) -> None:
        for idx in range(14):
            ang = layout.orbit_phase_offset + (idx * (math.tau / 14.0))
            radius = self._world_extent_distance(96.0 + (idx * 7.0 * layout.orbit_radius_scale))
            x = math.cos(ang) * radius
            y = math.sin(ang) * radius
            z = self._terrain_height(x, y)
            if idx % 5 == 0:
                self._place_scenery(
                    asset_id="building_hangar",
                    fallback="hangar",
                    pos=(x, y, z + 1.0),
                    scale=0.78 + (0.06 * (idx % 3)),
                )
            elif idx % 5 == 1:
                self._place_scenery(
                    asset_id="building_tower",
                    fallback="tower",
                    pos=(x, y, z + 2.0),
                    scale=0.74 + (0.05 * (idx % 4)),
                )
            else:
                self._place_scenery(
                    asset_id="forest_canopy_patch" if idx % 4 == 2 else "trees_field_cluster",
                    fallback="forest" if idx % 4 == 2 else "trees",
                    pos=(x, y, z),
                    scale=(1.9 + (0.18 * (idx % 4))) if idx % 4 == 2 else (1.5 + (0.2 * (idx % 4))),
                )

    def _build_traffic_pool(self, *, layout: RapidTrackingCompoundLayout) -> list[_RapidTrackingDecoy]:
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
                phase=layout.orbit_phase_offset + 0.15,
                speed=0.66 + (rapid_tracking_seed_unit(seed=layout.seed, salt="decoy-plane-blue") * 0.18),
                lateral_bias=-18.0 * layout.orbit_direction,
                depth_bias=2.0,
                altitude=18.0 + layout.altitude_bias,
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
                phase=layout.orbit_phase_offset + 1.80,
                speed=0.58 + (rapid_tracking_seed_unit(seed=layout.seed, salt="decoy-plane-green") * 0.16),
                lateral_bias=14.0,
                depth_bias=14.0 + (layout.path_lateral_bias * 8.0),
                altitude=22.0 + layout.altitude_bias,
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
                phase=layout.orbit_phase_offset + 0.62,
                speed=0.68 + (rapid_tracking_seed_unit(seed=layout.seed, salt="decoy-plane-yellow") * 0.16),
                lateral_bias=-6.0,
                depth_bias=22.0,
                altitude=16.0 + layout.altitude_bias,
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
                phase=layout.orbit_phase_offset + 2.18,
                speed=0.46 + (rapid_tracking_seed_unit(seed=layout.seed, salt="decoy-helo") * 0.18),
                lateral_bias=22.0,
                depth_bias=28.0,
                altitude=12.0 + layout.altitude_bias,
                route="helo_arc",
                activation_progress=0.38,
            ),
            _RapidTrackingDecoy(
                node=self._build_fallback_model(
                    kind="tank",
                    color=(0.46, 0.44, 0.24, 1.0),
                ),
                kind="tank",
                phase=layout.orbit_phase_offset + 0.55,
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
                phase=layout.orbit_phase_offset + 2.10,
                speed=0.30 + (rapid_tracking_seed_unit(seed=layout.seed, salt="decoy-truck") * 0.14),
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
                phase=layout.orbit_phase_offset + 1.34,
                speed=0.76 + (rapid_tracking_seed_unit(seed=layout.seed, salt="decoy-plane-red") * 0.18),
                lateral_bias=4.0 * layout.orbit_direction,
                depth_bias=40.0,
                altitude=27.0 + layout.altitude_bias,
                route="air_cross",
                activation_progress=0.79,
            ),
            _RapidTrackingDecoy(
                node=self._build_fallback_model(
                    kind="truck",
                    color=(0.28, 0.36, 0.22, 1.0),
                ),
                kind="truck",
                phase=layout.orbit_phase_offset + 2.92,
                speed=0.28 + (rapid_tracking_seed_unit(seed=layout.seed, salt="decoy-convoy") * 0.12),
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
        if self._active_scenery_target_node is not None:
            self._active_scenery_target_node.setAlphaScale(1.0)
            self._active_scenery_target_node = None
        for node in self._dynamic_target_nodes:
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

        # Forward-corridor buildings give the engine-driven building segments
        # stable pre-existing scenery to hand off to while the camera is
        # looking down the late run-in path.
        for asset_id, fallback, x, y, scale in (
            ("building_hangar", "hangar", 128.0, 22.0, 0.84),
            ("building_hangar", "hangar", 60.0, -16.0, 0.78),
            ("building_hangar", "hangar", 44.0, -20.0, 0.74),
            ("building_tower", "tower", 46.0, -36.0, 0.82),
            ("building_tower", "tower", 128.0, -64.0, 0.78),
        ):
            z = self._terrain_height(x, y)
            if fallback == "hangar":
                z += 1.0
            else:
                z += 2.0
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

    def _build_soldier_squad(
        self,
        *,
        leader_color: tuple[float, float, float, float],
        wing_color: tuple[float, float, float, float],
    ):
        from panda3d.core import NodePath

        root = NodePath("rapid-soldier-squad")
        leader = self._build_fallback_model(kind="soldier", color=leader_color)
        leader.reparentTo(root)
        leader.setZ(0.26)
        leader.setScale(1.06)

        for ox, oy in ((-1.1, 0.8), (1.0, 0.7), (-0.9, -0.8), (1.1, -0.9)):
            wing = self._build_fallback_model(kind="soldier", color=wing_color)
            wing.reparentTo(root)
            wing.setPos(ox, oy, 0.24)
            wing.setScale(0.92)
        return root

    @staticmethod
    def _normalized_building_variant(variant: str) -> str:
        return "tower" if str(variant).strip().lower() == "tower" else "hangar"

    def _select_target_node_from_candidates(
        self,
        *,
        kind: str,
        variant: str,
        candidates: list[object],
        desired_pos: tuple[float, float, float],
        desired_screen: tuple[float, float] | None = None,
        overlay_kind: str | None = None,
    ):
        if not candidates:
            return None
        current_selected = (
            self._selected_target_node in candidates
            and self._selected_target_kind == kind
            and self._selected_target_variant == variant
        )
        if current_selected and desired_screen is None:
            return self._selected_target_node

        desired_x, desired_y, desired_z = desired_pos

        def distance_sq(node) -> float:
            pos = node.getPos(self._base.render)
            dx = float(pos.x) - desired_x
            dy = float(pos.y) - desired_y
            dz = float(pos.z) - desired_z
            return (dx * dx) + (dy * dy) + (dz * dz)

        if desired_screen is None:
            selected = min(candidates, key=distance_sq)
        else:
            desired_screen_x, desired_screen_y = desired_screen
            projected_kind = overlay_kind or kind

            def projected_score(node):
                projection = self._project_target_node(node=node, kind=projected_kind)
                offscreen_dx = max(0.0, -projection.screen_x) + max(
                    0.0,
                    projection.screen_x - self._size[0],
                )
                offscreen_dy = max(0.0, -projection.screen_y) + max(
                    0.0,
                    projection.screen_y - self._size[1],
                )
                screen_dx = projection.screen_x - desired_screen_x
                screen_dy = projection.screen_y - desired_screen_y
                visibility_rank = 0 if projection.on_screen else (1 if projection.in_front else 2)
                return (
                    visibility_rank,
                    offscreen_dx + offscreen_dy,
                    (screen_dx * screen_dx) + (screen_dy * screen_dy),
                    distance_sq(node),
                )

            selected = min(candidates, key=projected_score)
            if current_selected:
                current_score = projected_score(self._selected_target_node)
                selected_score = projected_score(selected)
                if current_score[0] == 0 and selected_score[0] == 0:
                    return self._selected_target_node
                if current_score[0] < selected_score[0]:
                    return self._selected_target_node
                if current_score[0] == selected_score[0] and current_score[1] <= (
                    selected_score[1] + 36.0
                ):
                    return self._selected_target_node
        self._selected_target_node = selected
        self._selected_target_kind = kind
        self._selected_target_variant = variant
        return selected

    @classmethod
    def _target_rel_to_lens(cls, *, target_rel_x: float, target_rel_y: float) -> tuple[float, float]:
        lens_x = (
            _clamp(float(target_rel_x), -cls._TARGET_VIEW_LIMIT, cls._TARGET_VIEW_LIMIT)
            / cls._TARGET_VIEW_LIMIT
        )
        lens_y = -(
            _clamp(float(target_rel_y), -cls._TARGET_VIEW_LIMIT, cls._TARGET_VIEW_LIMIT)
            / cls._TARGET_VIEW_LIMIT
        )
        return (
            _clamp(lens_x, -0.95, 0.95),
            _clamp(lens_y, -0.95, 0.95),
        )

    def _target_rel_to_viewport(
        self,
        *,
        target_rel_x: float,
        target_rel_y: float,
    ) -> tuple[float, float]:
        clamped_x = _clamp(float(target_rel_x), -self._TARGET_VIEW_LIMIT, self._TARGET_VIEW_LIMIT)
        clamped_y = _clamp(float(target_rel_y), -self._TARGET_VIEW_LIMIT, self._TARGET_VIEW_LIMIT)
        width = max(1, self._size[0])
        height = max(1, self._size[1])
        return (
            ((clamped_x + self._TARGET_VIEW_LIMIT) / (self._TARGET_VIEW_LIMIT * 2.0)) * width,
            ((clamped_y + self._TARGET_VIEW_LIMIT) / (self._TARGET_VIEW_LIMIT * 2.0)) * height,
        )

    def _desired_tracking_anchor(
        self,
        *,
        target_rel_x: float,
        target_rel_y: float,
        distance: float,
    ) -> tuple[float, float, float]:
        cam_pos, forward, right, up = self._camera_basis()
        lateral = _clamp(float(target_rel_x), -1.5, 1.5) * 18.0
        vertical = _clamp(-float(target_rel_y), -1.25, 1.25) * 10.0
        anchor = cam_pos + (forward * max(10.0, float(distance))) + (right * lateral) + (up * vertical)
        return (float(anchor.x), float(anchor.y), float(anchor.z))

    def _ground_target_anchor(
        self,
        *,
        target_rel_x: float,
        target_rel_y: float,
        clearance: float,
        distance_bias: float,
        ray_target,
    ) -> tuple[float, float, float]:
        lens_x, lens_y = self._target_rel_to_lens(
            target_rel_x=target_rel_x,
            target_rel_y=target_rel_y,
        )
        terrain_anchor = self._world_point_on_terrain(
            lens_x=lens_x,
            lens_y=lens_y,
            clearance=clearance,
        )
        if terrain_anchor is not None:
            return terrain_anchor

        desired = self._desired_tracking_anchor(
            target_rel_x=target_rel_x,
            target_rel_y=target_rel_y,
            distance=distance_bias + _clamp((0.42 - target_rel_y) * 12.0, -6.0, 14.0),
        )
        if ray_target is None:
            return (
                desired[0],
                desired[1],
                self._terrain_height(desired[0], desired[1]) + clearance,
            )
        x = _lerp(ray_target[0], desired[0], 0.72)
        y = _lerp(ray_target[1], desired[1], 0.72)
        return (
            x,
            y,
            self._terrain_height(x, y) + clearance,
        )

    def _update_camera(self, *, payload: RapidTrackingPayload | None) -> None:
        progress = 0.0 if payload is None else float(payload.scene_progress)
        zoom = 0.0 if payload is None else float(payload.capture_zoom)
        turbulence_strength = 0.65 if payload is None else float(payload.turbulence_strength)
        target_kind = "" if payload is None else str(payload.target_kind)
        target_world_x = 0.0 if payload is None else float(payload.target_world_x)
        target_world_y = 0.0 if payload is None else float(payload.target_world_y)
        focus_world_x = (
            self._carrier_focus_world()[0] if payload is None else float(payload.focus_world_x)
        )
        focus_world_y = (
            self._carrier_focus_world()[1] if payload is None else float(payload.focus_world_y)
        )

        rig = self._camera_rig_state(
            elapsed_s=self._elapsed_s,
            seed=0 if payload is None else int(payload.session_seed),
            progress=progress,
            camera_yaw_deg=None if payload is None else float(payload.camera_yaw_deg),
            camera_pitch_deg=None if payload is None else float(payload.camera_pitch_deg),
            zoom=zoom,
            target_kind=target_kind,
            target_world_x=target_world_x,
            target_world_y=target_world_y,
            focus_world_x=focus_world_x,
            focus_world_y=focus_world_y,
            turbulence_strength=turbulence_strength,
        )
        self._last_camera_rig = rig

        self._base.camLens.setFov(rig.fov_deg)
        self._base.cam.setPos(rig.cam_world_x, rig.cam_world_y, rig.cam_world_z)
        self._base.cam.setHpr(
            rig.view_heading_deg,
            rig.view_pitch_deg,
            rig.roll_deg,
        )

    def _update_target(self, *, payload: RapidTrackingPayload | None) -> None:
        self._hide_target_nodes()

        if payload is None:
            demo_node = self._select_target_node_from_candidates(
                kind="jet",
                variant="demo",
                candidates=self._jet_pool,
                desired_pos=(0.0, 62.0, 16.0),
            )
            if demo_node is None:
                return
            demo_node.show()
            demo_node.setAlphaScale(1.0)
            demo_node.setPos(
                math.sin(self._elapsed_s * 0.9) * 8.0,
                62.0,
                16.0 + (math.sin(self._elapsed_s * 1.4) * 2.2),
            )
            demo_node.setHpr(
                90.0 + (math.sin(self._elapsed_s * 0.8) * 20.0),
                math.cos(self._elapsed_s * 1.1) * 6.0,
                math.sin(self._elapsed_s * 1.6) * 18.0,
            )
            self._active_target_node = demo_node
            self._active_target_kind = "jet"
            return

        kind = str(payload.target_kind).strip().lower()
        variant = str(payload.target_variant).strip().lower()
        progress = float(payload.scene_progress)
        scene_x = float(payload.target_rel_x)
        scene_y = float(payload.target_rel_y)
        track_x = float(payload.target_world_x)
        track_y = float(payload.target_world_y)
        target_visible = bool(payload.target_visible)

        if kind == "soldier":
            x, y, z = self._ground_target_world_pos(world_x=track_x, world_y=track_y, clearance=0.05)
            node = self._select_target_node_from_candidates(
                kind="soldier",
                variant="target",
                candidates=self._soldier_pool,
                desired_pos=(x, y, z),
            )
            if node is None:
                return
            heading = self._heading_from_velocity(
                vx=float(payload.target_vx),
                vy=float(payload.target_vy),
            )
            node.setPos(x, y, z)
            node.setHpr(heading, 0.0, 0.0)
            self._apply_dynamic_target_visibility(node=node, visible=target_visible)
            self._active_target_node = node
            self._active_target_kind = kind
            return
        if kind == "truck":
            x, y, z = self._ground_target_world_pos(world_x=track_x, world_y=track_y, clearance=0.34)
            tracked_variant = variant in {"tracked", "armor", "armored"}
            node = self._select_target_node_from_candidates(
                kind="truck",
                variant="tracked" if tracked_variant else "olive",
                candidates=self._tracked_pool if tracked_variant else self._truck_pool,
                desired_pos=(x, y, z),
            )
            if node is None:
                return
            node.setPos(x, y, z)
            node.setHpr(
                self._heading_from_velocity(
                    vx=float(payload.target_vx),
                    vy=float(payload.target_vy),
                ),
                0.0,
                0.0,
            )
            self._apply_dynamic_target_visibility(node=node, visible=target_visible)
            self._active_target_node = node
            self._active_target_kind = kind
            return

        if kind == "building":
            building_variant = self._normalized_building_variant(variant)
            x, y, z = self._ground_target_world_pos(
                world_x=track_x,
                world_y=track_y,
                clearance=2.2 if building_variant == "tower" else 1.25,
            )
            node = self._select_target_node_from_candidates(
                kind="building",
                variant=building_variant,
                candidates=self._scenery_nodes_by_kind[building_variant],
                desired_pos=(x, y, z),
            )
            if node is None:
                return
            node.setAlphaScale(0.96 if target_visible else 0.76)
            self._active_scenery_target_node = node
            self._active_target_node = node
            self._active_target_kind = kind
            return

        if kind == "helicopter":
            x, y, z = self._air_target_world_pos(
                kind=kind,
                world_x=track_x,
                world_y=track_y,
                phase_elapsed_s=float(payload.phase_elapsed_s),
                scene_progress=progress,
            )
            node = self._select_target_node_from_candidates(
                kind="helicopter",
                variant="green",
                candidates=self._helicopter_pool,
                desired_pos=(x, y, z),
            )
            if node is None:
                return
            current_rig = self._last_camera_rig or self._sample_camera_rig(
                payload=payload,
                elapsed_s=self._elapsed_s,
            )
            next_rig = self._sample_camera_rig(
                payload=payload,
                elapsed_s=self._elapsed_s + self._AIR_MOTION_SAMPLE_DT_S,
            )
            next_x, next_y, next_z = self._air_target_world_pos(
                kind=kind,
                world_x=track_x + (float(payload.target_vx) * self._AIR_MOTION_SAMPLE_DT_S),
                world_y=track_y + (float(payload.target_vy) * self._AIR_MOTION_SAMPLE_DT_S),
                phase_elapsed_s=float(payload.phase_elapsed_s) + self._AIR_MOTION_SAMPLE_DT_S,
                scene_progress=progress,
            )
            node.setPos(x, y, z)
            node.setHpr(
                *self._airborne_apparent_hpr(
                    cache_key=10_001,
                    current_pos=(x, y, z),
                    next_pos=(next_x, next_y, next_z),
                    current_rig=current_rig,
                    next_rig=next_rig,
                    default_heading=self._helicopter_heading_deg(progress=progress),
                    pitch_flair=_clamp(-float(payload.target_vy) * 60.0, -12.0, 12.0),
                    roll_flair=_clamp(-float(payload.target_vx) * 80.0, -18.0, 18.0),
                    pitch_limit=14.0,
                    roll_limit=18.0,
                ),
            )
            self._apply_dynamic_target_visibility(node=node, visible=target_visible)
            self._active_target_node = node
            self._active_target_kind = kind
            return

        x, y, z = self._air_target_world_pos(
            kind=kind,
            world_x=track_x,
            world_y=track_y,
            phase_elapsed_s=float(payload.phase_elapsed_s),
            scene_progress=progress,
        )
        node = self._select_target_node_from_candidates(
            kind="jet",
            variant="fast-pass",
            candidates=self._jet_pool,
            desired_pos=(x, y, z),
        )
        if node is None:
            return
        current_rig = self._last_camera_rig or self._sample_camera_rig(
            payload=payload,
            elapsed_s=self._elapsed_s,
        )
        next_rig = self._sample_camera_rig(
            payload=payload,
            elapsed_s=self._elapsed_s + self._AIR_MOTION_SAMPLE_DT_S,
        )
        next_x, next_y, next_z = self._air_target_world_pos(
            kind=kind,
            world_x=track_x + (float(payload.target_vx) * self._AIR_MOTION_SAMPLE_DT_S),
            world_y=track_y + (float(payload.target_vy) * self._AIR_MOTION_SAMPLE_DT_S),
            phase_elapsed_s=float(payload.phase_elapsed_s) + self._AIR_MOTION_SAMPLE_DT_S,
            scene_progress=progress,
        )
        node.setPos(x, y, z)
        node.setHpr(
            *self._airborne_apparent_hpr(
                cache_key=10_002,
                current_pos=(x, y, z),
                next_pos=(next_x, next_y, next_z),
                current_rig=current_rig,
                next_rig=next_rig,
                default_heading=self._helicopter_heading_deg(progress=progress),
                pitch_flair=_clamp(-float(payload.target_vy) * 80.0, -16.0, 16.0),
                roll_flair=_clamp(-float(payload.target_vx) * 120.0, -30.0, 30.0),
                pitch_limit=18.0,
                roll_limit=30.0,
            ),
        )
        self._apply_dynamic_target_visibility(node=node, visible=target_visible)
        self._active_target_node = node
        self._active_target_kind = kind

    def _update_decoys(self, *, payload: RapidTrackingPayload | None) -> None:
        t = self._elapsed_s
        progress = 0.0 if payload is None else float(payload.scene_progress)
        current_rig = self._last_camera_rig or self._sample_camera_rig(
            payload=payload,
            elapsed_s=t,
        )
        next_rig = self._sample_camera_rig(
            payload=payload,
            elapsed_s=t + self._AIR_MOTION_SAMPLE_DT_S,
        )
        for idx, decoy in enumerate(self._decoys):
            fade = _clamp((progress - decoy.activation_progress + 0.18) / 0.18, 0.0, 1.0)
            if decoy.activation_progress > 0.0 and fade <= 0.0:
                decoy.node.hide()
                continue

            decoy.node.show()
            decoy.node.setAlphaScale(fade)
            if decoy.route == "air_cross":
                current_pos, default_heading, pitch_flair, roll_flair = self._decoy_air_pose(
                    decoy=decoy,
                    elapsed_s=t,
                )
                next_pos, _next_heading, _next_pitch, _next_roll = self._decoy_air_pose(
                    decoy=decoy,
                    elapsed_s=t + self._AIR_MOTION_SAMPLE_DT_S,
                )
                decoy.node.setPos(*current_pos)
                decoy.node.setHpr(
                    *self._airborne_apparent_hpr(
                        cache_key=20_000 + idx,
                        current_pos=current_pos,
                        next_pos=next_pos,
                        current_rig=current_rig,
                        next_rig=next_rig,
                        default_heading=default_heading,
                        pitch_flair=pitch_flair,
                        roll_flair=roll_flair,
                        pitch_limit=16.0,
                        roll_limit=24.0,
                    ),
                )
            elif decoy.route == "helo_arc":
                current_pos, default_heading, pitch_flair, roll_flair = self._decoy_air_pose(
                    decoy=decoy,
                    elapsed_s=t,
                )
                next_pos, _next_heading, _next_pitch, _next_roll = self._decoy_air_pose(
                    decoy=decoy,
                    elapsed_s=t + self._AIR_MOTION_SAMPLE_DT_S,
                )
                decoy.node.setPos(*current_pos)
                decoy.node.setHpr(
                    *self._airborne_apparent_hpr(
                        cache_key=20_000 + idx,
                        current_pos=current_pos,
                        next_pos=next_pos,
                        current_rig=current_rig,
                        next_rig=next_rig,
                        default_heading=default_heading,
                        pitch_flair=pitch_flair,
                        roll_flair=roll_flair,
                        pitch_limit=12.0,
                        roll_limit=12.0,
                    ),
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
                current_pos, default_heading, pitch_flair, roll_flair = self._decoy_air_pose(
                    decoy=decoy,
                    elapsed_s=t,
                )
                next_pos, _next_heading, _next_pitch, _next_roll = self._decoy_air_pose(
                    decoy=decoy,
                    elapsed_s=t + self._AIR_MOTION_SAMPLE_DT_S,
                )
                decoy.node.setPos(*current_pos)
                decoy.node.setHpr(
                    *self._airborne_apparent_hpr(
                        cache_key=20_000 + idx,
                        current_pos=current_pos,
                        next_pos=next_pos,
                        current_rig=current_rig,
                        next_rig=next_rig,
                        default_heading=default_heading,
                        pitch_flair=pitch_flair,
                        roll_flair=roll_flair,
                        pitch_limit=14.0,
                        roll_limit=22.0,
                    ),
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
        node = self._active_target_node
        if node is None:
            self._target_overlay_state = None
            return

        kind = str(self._active_target_kind).strip().lower()
        projection = self._project_target_node(node=node, kind=kind)
        target_visible = bool(payload.target_visible) if payload is not None else True
        self._target_overlay_state = RapidTrackingOverlayState(
            screen_x=float(projection.screen_x),
            screen_y=float(projection.screen_y),
            on_screen=bool(projection.on_screen),
            in_front=bool(projection.in_front),
            target_visible=bool(target_visible),
        )

    def _target_overlay_world_pos(self, *, node, kind: str):
        from panda3d.core import Point3

        world_pos = node.getPos(self._base.render)
        if kind == "soldier":
            lift = 0.48
        elif kind == "truck":
            lift = 0.42
        elif kind == "building":
            lift = 1.8
        elif kind == "helicopter":
            lift = 0.8
        else:
            lift = 0.6
        return Point3(world_pos.x, world_pos.y, world_pos.z + lift)

    def _project_target_node(self, *, node, kind: str) -> RapidTrackingOverlayState:
        world_pos = self._target_overlay_world_pos(node=node, kind=kind)
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
        return RapidTrackingOverlayState(
            screen_x=float(screen_x),
            screen_y=float(screen_y),
            on_screen=bool(on_screen),
            in_front=bool(in_front),
            target_visible=True,
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
            seed=self._session_seed,
            progress=progress,
            camera_yaw_deg=None,
            camera_pitch_deg=None,
            zoom=0.0,
            target_kind="",
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
    ):
        node = self._load_asset_or_fallback(
            asset_id=asset_id,
            fallback=fallback,
            color=(1.0, 1.0, 1.0, 1.0),
            scale=scale,
            parent=self._scenery_root,
            pos=pos,
        )
        node.setTransparency(True)
        if fallback in self._scenery_nodes_by_kind:
            self._scenery_nodes_by_kind[fallback].append(node)
        return node

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
        hpr: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        entry = self._catalog.entry(asset_id)
        resolved = self._catalog.resolve_path(asset_id)
        fallback_kind = entry.fallback if entry is not None else fallback
        node = None
        if resolved is not None and fallback_kind != "plane":
            try:
                node = self._load_model(resolved)
            except Exception:
                node = None
        if node is None:
            node = self._build_fallback_model(kind=fallback_kind, color=color)
        node.reparentTo(parent)
        if resolved is not None and entry is not None and fallback_kind != "plane":
            entry.apply_loaded_model_transform(node, pos=pos, hpr=hpr, scale=scale)
        else:
            node.setPos(*pos)
            node.setHpr(*hpr)
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
            return build_panda3d_fixed_wing_model(
                palette=build_panda_palette(body_color=color),
                name="rapid-plane",
            )
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
        elif kind == "shrub":
            for offset_x, offset_y, size in ((-0.26, 0.06, 0.42), (0.04, -0.10, 0.48), (0.30, 0.08, 0.36)):
                crown = self._make_box(
                    size=(size, size * 0.9, size * 0.58),
                    color=(0.18, 0.46, 0.20, 0.96),
                )
                crown.setPos(offset_x, offset_y, size * 0.30)
                crown.reparentTo(root)
        elif kind == "forest":
            for offset_x, offset_y, scale in (
                (-1.3, -0.3, 1.0),
                (-0.4, 0.4, 1.2),
                (0.5, -0.2, 0.9),
                (1.2, 0.3, 1.1),
            ):
                trunk = self._make_box(
                    size=(0.18 * scale, 0.18 * scale, 1.0 * scale),
                    color=(0.34, 0.22, 0.12, 1.0),
                )
                crown = self._make_box(
                    size=(1.1 * scale, 1.1 * scale, 1.4 * scale),
                    color=(0.12, 0.34, 0.14, 0.94),
                )
                trunk.setPos(offset_x, offset_y, 0.5 * scale)
                crown.setPos(offset_x, offset_y, 1.5 * scale)
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
