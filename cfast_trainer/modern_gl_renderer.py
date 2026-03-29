from __future__ import annotations

import math
from dataclasses import dataclass

import moderngl
import numpy
import pygame

from .aircraft_art import (
    build_fixed_wing_mesh,
    build_pygame_palette,
    fixed_wing_heading_from_screen_heading,
    project_fixed_wing_faces,
    rotate_fixed_wing_point,
)
from .auditory_capacity_panda3d import _TUBE_PATH_POINTS, _tube_frame
from .auditory_capacity import (
    AUDITORY_GATE_PLAYER_X_NORM,
    AUDITORY_GATE_RETIRE_X_NORM,
    AUDITORY_GATE_SPAWN_X_NORM,
    AUDITORY_TRIANGLE_GATE_POINTS,
)
from .gl_scenes import (
    AuditoryGlScene,
    GlScene,
    RapidTrackingGlScene,
    SpatialIntegrationGlScene,
    TraceTest1GlScene,
    TraceTest2GlScene,
    gl_scene_name,
)
from .rapid_tracking import RapidTrackingPayload, build_rapid_tracking_compound_layout
from .rapid_tracking_gl import build_scene_target as build_rapid_tracking_scene_target
from .rapid_tracking_gl import camera_rig_state as rapid_tracking_camera_rig_state
from .rapid_tracking_view import (
    camera_space_to_viewport,
    estimated_target_world_z,
    rapid_tracking_seed_unit,
    terrain_height as rapid_tracking_terrain_height,
    track_to_world_xy as rapid_tracking_track_to_world_xy,
    world_to_camera_space,
)
from .render_assets import RenderAssetCatalog, RenderAssetResolutionError
from .spatial_integration import SpatialIntegrationSceneView
from .spatial_integration_gl import build_scene_layout as build_spatial_integration_scene_layout
from .trace_test_1 import TraceTest1Payload
from .trace_test_1_gl import (
    aircraft_screen_poses_for_payload as trace_test_1_aircraft_screen_poses_for_payload,
    project_scene_position as trace_test_1_project_scene_position,
)
from .trace_test_2 import TraceTest2Payload, trace_test_2_track_position, trace_test_2_track_tangent
from .trace_test_2_gl import (
    aircraft_screen_pose_for_track as trace_test_2_aircraft_screen_pose_for_track,
    project_point as trace_test_2_project_point,
)


class RendererBootstrapError(RuntimeError):
    pass


class RendererRenderError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class _ColorVertex:
    x: float
    y: float
    r: float
    g: float
    b: float
    a: float


@dataclass(frozen=True, slots=True)
class _TexVertex:
    x: float
    y: float
    u: float
    v: float
    r: float
    g: float
    b: float
    a: float


Point3 = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class _SceneCamera:
    position: Point3
    heading_deg: float
    pitch_deg: float
    h_fov_deg: float
    v_fov_deg: float
    near_clip: float = 0.12
    far_clip: float = 1200.0


@dataclass(frozen=True, slots=True)
class _AssetInstance:
    asset_id: str
    position: Point3
    hpr_deg: Point3 = (0.0, 0.0, 0.0)
    scale: Point3 = (1.0, 1.0, 1.0)
    color: tuple[float, float, float, float] | None = None


@dataclass(frozen=True, slots=True)
class _ProjectedOverlayPrimitive:
    kind: str
    points: tuple[tuple[float, float], ...] = ()
    center: tuple[float, float] | None = None
    radius: float = 0.0
    color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    filled: bool = True
    width: float = 1.0


@dataclass(frozen=True, slots=True)
class _ProjectedAircraftMarkerFace:
    role: str
    shade: float
    points: tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True)
class _ScenePlan:
    kind: str
    rect: pygame.Rect
    camera: _SceneCamera | None
    asset_instances: tuple[_AssetInstance, ...]
    overlay_primitives: tuple[_ProjectedOverlayPrimitive, ...]
    asset_ids: tuple[str, ...]
    entity_count: int


@dataclass(frozen=True, slots=True)
class _MeshTriangle:
    role: str
    points: tuple[Point3, Point3, Point3]
    normal: Point3


@dataclass(slots=True)
class _AssetMesh:
    asset_id: str
    triangles: tuple[_MeshTriangle, ...]
    role_palette: dict[str, tuple[float, float, float]]

    def color_for_role(self, role: str) -> tuple[float, float, float]:
        return self.role_palette.get(role, self.role_palette.get("body", (0.82, 0.84, 0.88)))


_AUDITORY_TUBE_SEGMENT_LENGTH = 4.0
_AUDITORY_TUBE_RIB_STEP = 4.0
_AUDITORY_CAMERA_BACK_DISTANCE = 8.2
_AUDITORY_CAMERA_LOOKAHEAD_DISTANCE = 13.0
_AUDITORY_CAMERA_FOLLOW_RATIO = 0.60
_AUDITORY_LOOK_TARGET_FOLLOW_RATIO = 0.90
_AUDITORY_TUNNEL_BACKFILL_DISTANCE = 14.0
_AUDITORY_TUNNEL_LOOKAHEAD_DISTANCE = 24.0
_AUDITORY_TRAVEL_WRAP_MARGIN = 0.6
_AUDITORY_BALL_ANCHOR_DISTANCE = 9.5
_AUDITORY_TUBE_RX = 2.24
_AUDITORY_TUBE_RZ = 1.64
_AUDITORY_PATH_SPAN = float(_TUBE_PATH_POINTS[-1][0])
_AUDITORY_TRAVEL_WRAP_THRESHOLD = _AUDITORY_PATH_SPAN - _AUDITORY_BALL_ANCHOR_DISTANCE - _AUDITORY_TRAVEL_WRAP_MARGIN
_AUDITORY_TUBE_GEOMETRY_START = -(
    _AUDITORY_BALL_ANCHOR_DISTANCE
    + _AUDITORY_CAMERA_BACK_DISTANCE
    + _AUDITORY_TUNNEL_BACKFILL_DISTANCE
)
_AUDITORY_TUBE_GEOMETRY_END = _AUDITORY_PATH_SPAN + _AUDITORY_TUNNEL_LOOKAHEAD_DISTANCE


def _wrap_heading_deg(dx: float, dy: float) -> float:
    return (math.degrees(math.atan2(float(dx), float(dy))) + 360.0) % 360.0


def _pitch_deg(dx: float, dy: float, dz: float) -> float:
    horizontal = max(1e-6, math.hypot(float(dx), float(dy)))
    return math.degrees(math.atan2(float(dz), horizontal))


def _look_at_camera(
    *,
    position: Point3,
    target: Point3,
    h_fov_deg: float,
    v_fov_deg: float,
    near_clip: float = 0.12,
    far_clip: float = 1200.0,
) -> _SceneCamera:
    dx = float(target[0]) - float(position[0])
    dy = float(target[1]) - float(position[1])
    dz = float(target[2]) - float(position[2])
    return _SceneCamera(
        position=(float(position[0]), float(position[1]), float(position[2])),
        heading_deg=_wrap_heading_deg(dx, dy),
        pitch_deg=_pitch_deg(dx, dy, dz),
        h_fov_deg=float(h_fov_deg),
        v_fov_deg=float(v_fov_deg),
        near_clip=float(near_clip),
        far_clip=float(far_clip),
    )


def _world_hpr_from_tangent(
    tangent: Point3,
    *,
    roll_deg: float = 0.0,
) -> Point3:
    dx, dy, dz = (float(tangent[0]), float(tangent[1]), float(tangent[2]))
    horiz = max(1e-6, math.hypot(dx, dy))
    return (
        math.degrees(math.atan2(dx, dy)) % 360.0,
        math.degrees(math.atan2(dz, horiz)),
        float(roll_deg),
    )


def _vec_add(a: Point3, b: Point3) -> Point3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _vec_scale(v: Point3, scale: float) -> Point3:
    return (v[0] * scale, v[1] * scale, v[2] * scale)


def _vec_norm(v: Point3) -> float:
    return math.sqrt((v[0] * v[0]) + (v[1] * v[1]) + (v[2] * v[2]))


def _vec_normalize(v: Point3) -> Point3:
    norm = _vec_norm(v)
    if norm <= 1e-6:
        return (0.0, 0.0, 0.0)
    return (v[0] / norm, v[1] / norm, v[2] / norm)


def _vec_sub(a: Point3, b: Point3) -> Point3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vec_cross(a: Point3, b: Point3) -> Point3:
    return (
        (a[1] * b[2]) - (a[2] * b[1]),
        (a[2] * b[0]) - (a[0] * b[2]),
        (a[0] * b[1]) - (a[1] * b[0]),
    )


def _triangle_normal(points: tuple[Point3, Point3, Point3]) -> Point3:
    ab = _vec_sub(points[1], points[0])
    ac = _vec_sub(points[2], points[0])
    return _vec_normalize(_vec_cross(ab, ac))


def _triangulate_points(
    *,
    role: str,
    points: tuple[Point3, ...] | list[Point3],
) -> list[_MeshTriangle]:
    if len(points) < 3:
        return []
    origin = tuple(float(v) for v in points[0])
    triangles: list[_MeshTriangle] = []
    for idx in range(1, len(points) - 1):
        tri_points = (
            origin,
            tuple(float(v) for v in points[idx]),
            tuple(float(v) for v in points[idx + 1]),
        )
        triangles.append(
            _MeshTriangle(
                role=role,
                points=tri_points,
                normal=_triangle_normal(tri_points),
            )
        )
    return triangles


def _shift_triangles(
    triangles: tuple[_MeshTriangle, ...] | list[_MeshTriangle],
    *,
    offset: Point3,
) -> tuple[_MeshTriangle, ...]:
    shifted: list[_MeshTriangle] = []
    for tri in triangles:
        shifted_points = tuple(_vec_add(point, offset) for point in tri.points)
        shifted.append(
            _MeshTriangle(
                role=tri.role,
                points=shifted_points,
                normal=tri.normal,
            )
        )
    return tuple(shifted)


def _box_triangles(
    *,
    role: str,
    size: Point3,
    center: Point3 = (0.0, 0.0, 0.0),
) -> tuple[_MeshTriangle, ...]:
    sx = float(size[0]) * 0.5
    sy = float(size[1]) * 0.5
    sz = float(size[2]) * 0.5
    cx, cy, cz = center
    corners = {
        "lbf": (cx - sx, cy + sy, cz - sz),
        "rbf": (cx + sx, cy + sy, cz - sz),
        "lbb": (cx - sx, cy - sy, cz - sz),
        "rbb": (cx + sx, cy - sy, cz - sz),
        "ltf": (cx - sx, cy + sy, cz + sz),
        "rtf": (cx + sx, cy + sy, cz + sz),
        "ltb": (cx - sx, cy - sy, cz + sz),
        "rtb": (cx + sx, cy - sy, cz + sz),
    }
    faces = (
        ("front", ("lbf", "rbf", "rtf", "ltf")),
        ("back", ("rbb", "lbb", "ltb", "rtb")),
        ("left", ("lbb", "lbf", "ltf", "ltb")),
        ("right", ("rbf", "rbb", "rtb", "rtf")),
        ("top", ("ltf", "rtf", "rtb", "ltb")),
        ("bottom", ("lbb", "rbb", "rbf", "lbf")),
    )
    triangles: list[_MeshTriangle] = []
    for _name, keys in faces:
        triangles.extend(
            _triangulate_points(role=role, points=tuple(corners[key] for key in keys))
        )
    return tuple(triangles)


def _pyramid_triangles(
    *,
    role: str,
    size: Point3,
    center: Point3 = (0.0, 0.0, 0.0),
) -> tuple[_MeshTriangle, ...]:
    sx = float(size[0]) * 0.5
    sy = float(size[1]) * 0.5
    sz = float(size[2]) * 0.5
    cx, cy, cz = center
    base = (
        (cx - sx, cy - sy, cz - sz),
        (cx + sx, cy - sy, cz - sz),
        (cx + sx, cy + sy, cz - sz),
        (cx - sx, cy + sy, cz - sz),
    )
    apex = (cx, cy, cz + sz)
    triangles = list(_triangulate_points(role=role, points=base))
    for start, end in zip(base, base[1:] + base[:1], strict=False):
        triangles.extend(_triangulate_points(role=role, points=(start, end, apex)))
    return tuple(triangles)


def _roof_prism_triangles(
    *,
    role: str,
    size: Point3,
    center: Point3 = (0.0, 0.0, 0.0),
) -> tuple[_MeshTriangle, ...]:
    sx = float(size[0]) * 0.5
    sy = float(size[1]) * 0.5
    sz = float(size[2]) * 0.5
    cx, cy, cz = center
    pts = {
        "fl": (cx - sx, cy + sy, cz - sz),
        "fr": (cx + sx, cy + sy, cz - sz),
        "bl": (cx - sx, cy - sy, cz - sz),
        "br": (cx + sx, cy - sy, cz - sz),
        "rtl": (cx - sx, cy, cz + sz),
        "rtr": (cx + sx, cy, cz + sz),
    }
    faces = (
        ("body", ("fl", "fr", "br", "bl")),
        ("body", ("fl", "rtl", "rtr", "fr")),
        ("body", ("bl", "br", "rtr", "rtl")),
        ("body", ("fl", "bl", "rtl")),
        ("body", ("fr", "rtr", "br")),
    )
    triangles: list[_MeshTriangle] = []
    for face_role, keys in faces:
        triangles.extend(
            _triangulate_points(
                role=face_role if role == "body" else role,
                points=tuple(pts[key] for key in keys),
            )
        )
    return tuple(triangles)


def _disc_triangles(
    *,
    role: str,
    radius: float,
    steps: int = 20,
    z: float = 0.0,
) -> tuple[_MeshTriangle, ...]:
    pts = tuple(
        (
            math.cos((idx / float(max(3, steps))) * math.tau) * float(radius),
            math.sin((idx / float(max(3, steps))) * math.tau) * float(radius),
            float(z),
        )
        for idx in range(max(3, int(steps)))
    )
    return tuple(_triangulate_points(role=role, points=pts))


def _auditory_gate_points(shape: str) -> tuple[tuple[float, float], ...]:
    token = str(shape).upper()
    if token == "TRIANGLE":
        return tuple((float(px), float(py)) for px, py in AUDITORY_TRIANGLE_GATE_POINTS)
    if token == "SQUARE":
        return (
            (-1.10, 1.10),
            (1.10, 1.10),
            (1.10, -1.10),
            (-1.10, -1.10),
        )
    steps = 24
    return tuple(
        (
            math.cos((idx / float(steps)) * math.tau),
            math.sin((idx / float(steps)) * math.tau),
        )
        for idx in range(steps)
    )


def _gate_segment_prism_triangles(
    *,
    role: str,
    start: tuple[float, float],
    end: tuple[float, float],
    half_width: float,
    half_depth: float,
) -> tuple[_MeshTriangle, ...]:
    dx = float(end[0] - start[0])
    dz = float(end[1] - start[1])
    length = math.hypot(dx, dz)
    if length <= 1e-6:
        return ()

    tx = dx / length
    tz = dz / length
    nx = -tz * float(half_width)
    nz = tx * float(half_width)

    a = (float(start[0]) + nx, -float(half_depth), float(start[1]) + nz)
    b = (float(start[0]) - nx, -float(half_depth), float(start[1]) - nz)
    c = (float(end[0]) - nx, -float(half_depth), float(end[1]) - nz)
    d = (float(end[0]) + nx, -float(half_depth), float(end[1]) + nz)
    ap = (a[0], float(half_depth), a[2])
    bp = (b[0], float(half_depth), b[2])
    cp = (c[0], float(half_depth), c[2])
    dp = (d[0], float(half_depth), d[2])

    triangles: list[_MeshTriangle] = []
    quads = (
        (a, b, c, d),
        (ap, dp, cp, bp),
        (a, d, dp, ap),
        (b, bp, cp, c),
        (a, ap, bp, b),
        (d, c, cp, dp),
    )
    for quad in quads:
        triangles.extend(_triangulate_points(role=role, points=quad))
    return tuple(triangles)


def _auditory_gate_mesh_triangles(shape: str) -> tuple[_MeshTriangle, ...]:
    token = str(shape).upper()
    half_width = 0.095 if token == "CIRCLE" else 0.108
    half_depth = 0.080
    points = _auditory_gate_points(token)
    triangles: list[_MeshTriangle] = []
    for idx in range(len(points)):
        triangles.extend(
            _gate_segment_prism_triangles(
                role="body",
                start=points[idx],
                end=points[(idx + 1) % len(points)],
                half_width=half_width,
                half_depth=half_depth,
            )
        )
    return tuple(triangles)


def _auditory_gate_asset_id(shape: str) -> str:
    token = str(shape).upper()
    if token == "TRIANGLE":
        return "auditory_gate_triangle"
    if token == "SQUARE":
        return "auditory_gate_square"
    return "auditory_gate_circle"


def _auditory_gate_rgba(name: str) -> tuple[float, float, float, float]:
    palette = {
        "RED": (0.92, 0.28, 0.34, 0.96),
        "GREEN": (0.32, 0.86, 0.50, 0.96),
        "BLUE": (0.38, 0.58, 0.96, 0.96),
        "YELLOW": (0.96, 0.84, 0.32, 0.96),
    }
    return palette.get(str(name).upper(), (0.90, 0.92, 0.98, 0.96))


def _auditory_depth_gate_rgba(
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


def _auditory_gate_ahead_distance_for_x_norm(x_norm: float) -> float:
    def lerp(a: float, b: float, t: float) -> float:
        return float(a) + ((float(b) - float(a)) * float(t))

    rel = float(x_norm)
    if rel >= AUDITORY_GATE_PLAYER_X_NORM:
        t = max(
            0.0,
            min(
                1.0,
                (rel - AUDITORY_GATE_PLAYER_X_NORM)
                / (AUDITORY_GATE_SPAWN_X_NORM - AUDITORY_GATE_PLAYER_X_NORM),
            ),
        )
        return lerp(0.8, 38.0, t)
    t = max(
        0.0,
        min(
            1.0,
            (AUDITORY_GATE_PLAYER_X_NORM - rel)
            / (AUDITORY_GATE_PLAYER_X_NORM - AUDITORY_GATE_RETIRE_X_NORM),
        ),
    )
    return lerp(0.8, -6.0, t)


def _asset_palette(asset_id: str) -> dict[str, tuple[float, float, float]]:
    palettes: dict[str, dict[str, tuple[float, float, float]]] = {
        "plane_red": {
            "body": (0.92, 0.24, 0.24),
            "accent": (0.74, 0.18, 0.18),
            "canopy": (0.94, 0.96, 1.0),
            "engine": (0.56, 0.12, 0.12),
        },
        "plane_blue": {
            "body": (0.34, 0.52, 0.90),
            "accent": (0.24, 0.40, 0.78),
            "canopy": (0.90, 0.96, 1.0),
            "engine": (0.20, 0.28, 0.54),
        },
        "plane_green": {
            "body": (0.30, 0.74, 0.48),
            "accent": (0.20, 0.56, 0.36),
            "canopy": (0.90, 0.96, 1.0),
            "engine": (0.18, 0.34, 0.24),
        },
        "plane_yellow": {
            "body": (0.94, 0.82, 0.34),
            "accent": (0.78, 0.60, 0.20),
            "canopy": (0.94, 0.96, 1.0),
            "engine": (0.46, 0.34, 0.12),
        },
        "helicopter_green": {
            "body": (0.42, 0.76, 0.58),
            "accent": (0.24, 0.52, 0.38),
            "canopy": (0.90, 0.96, 1.0),
            "engine": (0.18, 0.26, 0.20),
        },
        "truck_olive": {
            "body": (0.56, 0.54, 0.28),
            "accent": (0.36, 0.32, 0.16),
            "engine": (0.16, 0.16, 0.16),
            "wheel": (0.10, 0.10, 0.12),
        },
        "vehicle_tracked": {
            "body": (0.42, 0.48, 0.28),
            "accent": (0.30, 0.34, 0.20),
            "engine": (0.18, 0.18, 0.18),
            "wheel": (0.12, 0.12, 0.14),
        },
        "building_hangar": {
            "body": (0.62, 0.70, 0.74),
            "roof": (0.42, 0.48, 0.52),
        },
        "building_tower": {
            "body": (0.72, 0.78, 0.82),
            "roof": (0.48, 0.56, 0.62),
        },
        "road_paved_segment": {
            "body": (0.18, 0.19, 0.21),
            "accent": (0.84, 0.78, 0.56),
        },
        "road_dirt_segment": {
            "body": (0.42, 0.34, 0.24),
            "accent": (0.56, 0.46, 0.32),
        },
        "terrain_lake_patch": {
            "body": (0.20, 0.44, 0.66),
            "accent": (0.34, 0.62, 0.84),
        },
        "terrain_hill_mound": {
            "body": (0.38, 0.48, 0.28),
            "accent": (0.48, 0.58, 0.34),
        },
        "terrain_rock_cluster": {
            "body": (0.48, 0.46, 0.42),
            "accent": (0.34, 0.32, 0.30),
        },
        "trees_pine_cluster": {
            "body": (0.14, 0.20, 0.12),
            "canopy": (0.14, 0.42, 0.22),
        },
        "trees_field_cluster": {
            "body": (0.12, 0.18, 0.10),
            "canopy": (0.16, 0.48, 0.24),
        },
        "forest_canopy_patch": {
            "body": (0.12, 0.18, 0.10),
            "canopy": (0.16, 0.40, 0.22),
        },
        "shrubs_low_cluster": {
            "body": (0.20, 0.38, 0.18),
        },
        "soldiers_patrol": {
            "body": (0.26, 0.34, 0.20),
            "accent": (0.84, 0.76, 0.66),
        },
        "auditory_ball": {
            "body": (0.92, 0.95, 1.0),
        },
        "auditory_tunnel_segment": {
            "tunnel_wall": (0.08, 0.14, 0.38),
            "body": (0.08, 0.14, 0.38),
        },
        "auditory_tunnel_rib": {
            "tunnel_rib": (0.34, 0.52, 0.98),
            "body": (0.34, 0.52, 0.98),
        },
        "auditory_gate_circle": {
            "body": (0.92, 0.92, 0.98),
        },
        "auditory_gate_triangle": {
            "body": (0.92, 0.92, 0.98),
        },
        "auditory_gate_square": {
            "body": (0.92, 0.92, 0.98),
        },
    }
    return dict(palettes.get(asset_id, {"body": (0.78, 0.80, 0.84)}))


class _SceneAssetLibrary:
    def __init__(self, catalog: RenderAssetCatalog) -> None:
        self._catalog = catalog
        self._mesh_cache: dict[str, _AssetMesh] = {}

    def require_many(self, asset_ids: tuple[str, ...] | list[str] | set[str]) -> tuple[_AssetMesh, ...]:
        self._catalog.require_many(asset_ids)
        return tuple(self.mesh(asset_id) for asset_id in tuple(asset_ids))

    def mesh(self, asset_id: str) -> _AssetMesh:
        key = str(asset_id)
        cached = self._mesh_cache.get(key)
        if cached is not None:
            return cached
        source = self._catalog.require(key)
        if source.is_builtin:
            mesh = self._build_builtin_mesh(asset_id=key, builtin_kind=str(source.builtin_kind))
        else:
            assert source.path is not None
            mesh = self._load_obj_mesh(asset_id=key, path=source.path)
        self._mesh_cache[key] = mesh
        return mesh

    def _load_obj_mesh(self, *, asset_id: str, path) -> _AssetMesh:
        vertices: list[Point3] = []
        triangles: list[_MeshTriangle] = []
        role = "body"
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line == "" or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
                continue
            if parts[0] == "usemtl" and len(parts) >= 2:
                role = str(parts[1])
                continue
            if parts[0] != "f" or len(parts) < 4:
                continue
            face_vertices: list[Point3] = []
            for token in parts[1:]:
                index_token = token.split("/")[0]
                if index_token == "":
                    continue
                index = int(index_token)
                if index < 0:
                    index = len(vertices) + index + 1
                face_vertices.append(vertices[index - 1])
            triangles.extend(_triangulate_points(role=role, points=tuple(face_vertices)))
        return _AssetMesh(asset_id=asset_id, triangles=tuple(triangles), role_palette=_asset_palette(asset_id))

    def _build_builtin_mesh(self, *, asset_id: str, builtin_kind: str) -> _AssetMesh:
        token = str(builtin_kind).strip().lower()
        if token == "fixed_wing":
            triangles: list[_MeshTriangle] = []
            for face in build_fixed_wing_mesh():
                triangles.extend(_triangulate_points(role=face.role, points=face.points))
            return _AssetMesh(asset_id=asset_id, triangles=tuple(triangles), role_palette=_asset_palette(asset_id))
        if token == "helicopter":
            triangles = [
                *_box_triangles(role="body", size=(1.6, 2.0, 0.7)),
                *_box_triangles(role="body", size=(0.26, 2.6, 0.18), center=(0.0, -2.2, 0.0)),
                *_box_triangles(role="canopy", size=(1.0, 0.8, 0.5), center=(0.0, 0.9, 0.2)),
                *_box_triangles(role="accent", size=(3.6, 0.12, 0.06), center=(0.0, 0.0, 0.55)),
                *_box_triangles(role="accent", size=(0.12, 3.0, 0.06), center=(0.0, 0.0, 0.55)),
                *_box_triangles(role="accent", size=(0.9, 0.10, 0.06), center=(0.0, -3.35, 0.0)),
            ]
            return _AssetMesh(asset_id=asset_id, triangles=tuple(triangles), role_palette=_asset_palette(asset_id))
        if token == "truck":
            triangles = [
                *_box_triangles(role="body", size=(1.2, 2.4, 0.55), center=(0.0, -0.10, 0.12)),
                *_box_triangles(role="accent", size=(1.0, 0.8, 0.78), center=(0.0, 1.22, 0.24)),
                *_box_triangles(role="engine", size=(1.35, 0.26, 0.18), center=(0.0, 0.0, -0.18)),
            ]
            return _AssetMesh(asset_id=asset_id, triangles=tuple(triangles), role_palette=_asset_palette(asset_id))
        if token == "tracked_vehicle":
            triangles = [
                *_box_triangles(role="body", size=(1.8, 2.7, 0.56), center=(0.0, -0.02, 0.18)),
                *_box_triangles(role="accent", size=(0.94, 1.00, 0.42), center=(0.0, 0.22, 0.56)),
                *_box_triangles(role="engine", size=(0.16, 1.42, 0.12), center=(0.0, 1.30, 0.54)),
                *_box_triangles(role="wheel", size=(0.30, 2.54, 0.40), center=(-0.82, -0.02, 0.08)),
                *_box_triangles(role="wheel", size=(0.30, 2.54, 0.40), center=(0.82, -0.02, 0.08)),
            ]
            return _AssetMesh(asset_id=asset_id, triangles=tuple(triangles), role_palette=_asset_palette(asset_id))
        if token == "hangar":
            triangles = [
                *_box_triangles(role="body", size=(5.2, 4.6, 1.8), center=(0.0, 0.0, 0.9)),
                *_roof_prism_triangles(role="roof", size=(5.6, 4.8, 1.6), center=(0.0, 0.0, 2.2)),
            ]
            return _AssetMesh(asset_id=asset_id, triangles=tuple(triangles), role_palette=_asset_palette(asset_id))
        if token == "tower":
            triangles = [
                *_box_triangles(role="body", size=(0.9, 0.9, 4.8), center=(0.0, 0.0, 2.4)),
                *_box_triangles(role="roof", size=(2.1, 1.8, 0.9), center=(0.0, 0.0, 5.3)),
            ]
            return _AssetMesh(asset_id=asset_id, triangles=tuple(triangles), role_palette=_asset_palette(asset_id))
        if token == "road_paved":
            triangles = [
                *_box_triangles(role="body", size=(1.00, 1.00, 0.06), center=(0.0, 0.0, 0.0)),
                *_box_triangles(role="accent", size=(0.08, 0.56, 0.012), center=(0.0, 0.0, 0.04)),
            ]
            return _AssetMesh(asset_id=asset_id, triangles=tuple(triangles), role_palette=_asset_palette(asset_id))
        if token == "road_dirt":
            triangles = [
                *_box_triangles(role="body", size=(1.00, 1.00, 0.05), center=(0.0, 0.0, 0.0)),
                *_box_triangles(role="accent", size=(0.82, 0.82, 0.010), center=(0.0, 0.0, 0.03)),
            ]
            return _AssetMesh(asset_id=asset_id, triangles=tuple(triangles), role_palette=_asset_palette(asset_id))
        if token == "lake_patch":
            triangles = [
                *_disc_triangles(role="body", radius=1.0, steps=24, z=0.0),
                *_disc_triangles(role="accent", radius=0.74, steps=20, z=0.02),
            ]
            return _AssetMesh(asset_id=asset_id, triangles=tuple(triangles), role_palette=_asset_palette(asset_id))
        if token == "hill_mound":
            triangles = [
                *_pyramid_triangles(role="body", size=(2.6, 2.4, 1.6), center=(0.0, 0.0, 0.8)),
                *_pyramid_triangles(role="accent", size=(1.6, 1.4, 0.9), center=(0.0, 0.0, 1.2)),
            ]
            return _AssetMesh(asset_id=asset_id, triangles=tuple(triangles), role_palette=_asset_palette(asset_id))
        if token == "rock_cluster":
            triangles = [
                *_box_triangles(role="body", size=(0.72, 0.84, 0.52), center=(-0.22, -0.08, 0.24)),
                *_box_triangles(role="body", size=(0.56, 0.52, 0.46), center=(0.28, 0.14, 0.22)),
                *_pyramid_triangles(role="accent", size=(0.92, 0.76, 0.54), center=(0.0, 0.0, 0.34)),
            ]
            return _AssetMesh(asset_id=asset_id, triangles=tuple(triangles), role_palette=_asset_palette(asset_id))
        if token == "trees_cluster":
            triangles = []
            tree_offsets = ((-1.0, -0.4), (0.0, 0.0), (1.1, 0.6))
            for off_x, off_y in tree_offsets:
                triangles.extend(
                    _box_triangles(
                        role="body",
                        size=(0.18, 0.18, 0.9),
                        center=(off_x, off_y, 0.45),
                    )
                )
                triangles.extend(
                    _pyramid_triangles(
                        role="canopy",
                        size=(1.0, 1.0, 1.2),
                        center=(off_x, off_y, 1.35),
                    )
                )
                triangles.extend(
                    _pyramid_triangles(
                        role="canopy",
                        size=(0.78, 0.78, 1.0),
                        center=(off_x, off_y, 2.0),
                    )
                )
            return _AssetMesh(asset_id=asset_id, triangles=tuple(triangles), role_palette=_asset_palette(asset_id))
        if token == "soldier_squad":
            triangles = []
            offsets = (-0.35, 0.0, 0.35)
            for offset in offsets:
                triangles.extend(
                    _box_triangles(
                        role="body",
                        size=(0.12, 0.12, 0.48),
                        center=(offset, 0.0, 0.24),
                    )
                )
                triangles.extend(
                    _box_triangles(
                        role="accent",
                        size=(0.14, 0.14, 0.14),
                        center=(offset, 0.0, 0.58),
                    )
                )
            return _AssetMesh(asset_id=asset_id, triangles=tuple(triangles), role_palette=_asset_palette(asset_id))
        if token == "auditory_gate_circle":
            return _AssetMesh(
                asset_id=asset_id,
                triangles=_auditory_gate_mesh_triangles("CIRCLE"),
                role_palette=_asset_palette(asset_id),
            )
        if token == "auditory_gate_triangle":
            return _AssetMesh(
                asset_id=asset_id,
                triangles=_auditory_gate_mesh_triangles("TRIANGLE"),
                role_palette=_asset_palette(asset_id),
            )
        if token == "auditory_gate_square":
            return _AssetMesh(
                asset_id=asset_id,
                triangles=_auditory_gate_mesh_triangles("SQUARE"),
                role_palette=_asset_palette(asset_id),
            )
        raise RendererBootstrapError(f"Unsupported builtin render asset kind: {builtin_kind}")


def _scene_local_top_left_to_screen(
    *,
    rect: pygame.Rect,
    window_height: int,
    x: float,
    y: float,
) -> tuple[float, float]:
    return (
        float(rect.x) + float(x),
        float(int(window_height) - (rect.y + float(y))),
    )


def _project_aircraft_marker_polygons(
    *,
    rect: pygame.Rect,
    window_height: int,
    center_top_left: tuple[float, float],
    heading_deg: float,
    size: float,
    color: tuple[float, float, float, float],
    outline: tuple[float, float, float, float],
    pitch_deg: float = 0.0,
    bank_deg: float = 0.0,
    view_yaw_deg: float = 0.0,
    view_pitch_deg: float = 20.0,
) -> tuple[_ProjectedAircraftMarkerFace, ...]:
    _ = (color, outline)
    projected = project_fixed_wing_faces(
        heading_deg=fixed_wing_heading_from_screen_heading(heading_deg),
        pitch_deg=float(pitch_deg),
        bank_deg=float(bank_deg),
        cx=int(round(center_top_left[0])),
        cy=int(round(center_top_left[1])),
        scale=max(8.0, float(size)),
        view_yaw_deg=view_yaw_deg,
        view_pitch_deg=view_pitch_deg,
    )
    return tuple(
        _ProjectedAircraftMarkerFace(
            role=face.role,
            shade=float(face.shade),
            points=tuple(
                _scene_local_top_left_to_screen(
                    rect=rect,
                    window_height=window_height,
                    x=float(px),
                    y=float(py),
                )
                for px, py in face.points
            )
        )
        for face in projected
    )


def _rapid_tracking_target_asset_id(target_kind: str, variant: str) -> str:
    token = str(target_kind).strip().lower()
    if token == "building":
        return "building_tower" if str(variant).strip().lower() == "tower" else "building_hangar"
    if token == "truck":
        return "vehicle_tracked" if str(variant).strip().lower() in {"tracked", "armor", "armored"} else "truck_olive"
    if token == "soldier":
        return "soldiers_patrol"
    if token == "helicopter":
        return "helicopter_green"
    variant_token = str(variant).strip().lower()
    if variant_token in {"red", "blue", "green", "yellow"}:
        return f"plane_{variant_token}"
    return "plane_yellow"


def _build_auditory_scene_plan(scene: AuditoryGlScene) -> _ScenePlan:
    payload = scene.payload
    phase_elapsed_s = 0.0 if payload is None else float(payload.phase_elapsed_s)
    travel_offset = (phase_elapsed_s * 5.2) % max(0.001, _AUDITORY_TRAVEL_WRAP_THRESHOLD)
    ball_distance = _AUDITORY_BALL_ANCHOR_DISTANCE + travel_offset
    local_x = 0.0
    local_z = 0.0
    ball_color = (0.94, 0.96, 1.0, 0.98)
    if payload is not None:
        x_half_span = max(0.08, float(payload.tube_half_width))
        z_half_span = max(0.08, float(payload.tube_half_height))
        local_x = max(-1.0, min(1.0, float(payload.ball_x) / x_half_span)) * (_AUDITORY_TUBE_RX * 0.72)
        local_z = max(-1.0, min(1.0, float(payload.ball_y) / z_half_span)) * (_AUDITORY_TUBE_RZ * 0.72)
        if float(payload.ball_contact_ratio) >= 1.0:
            ball_color = (0.95, 0.35, 0.38, 0.98)
        else:
            ball_color = {
                "RED": (0.92, 0.34, 0.38, 0.98),
                "GREEN": (0.34, 0.88, 0.56, 0.98),
                "BLUE": (0.36, 0.62, 0.94, 0.98),
                "YELLOW": (0.94, 0.82, 0.34, 0.98),
            }.get(str(payload.ball_visual_color).upper(), (0.94, 0.96, 1.0, 0.98))
    camera_distance = max(_AUDITORY_TUBE_GEOMETRY_START + 0.4, ball_distance - _AUDITORY_CAMERA_BACK_DISTANCE)
    cam_center, _cam_tangent, cam_right, cam_up = _tube_frame(camera_distance, span=_AUDITORY_PATH_SPAN)
    look_distance = ball_distance + _AUDITORY_CAMERA_LOOKAHEAD_DISTANCE
    look_center, tangent, look_right, look_up = _tube_frame(look_distance, span=_AUDITORY_PATH_SPAN)
    cam_follow = _vec_add(
        _vec_scale(cam_right, local_x * _AUDITORY_CAMERA_FOLLOW_RATIO),
        _vec_scale(cam_up, local_z * _AUDITORY_CAMERA_FOLLOW_RATIO),
    )
    look_follow = _vec_add(
        _vec_scale(look_right, local_x * _AUDITORY_LOOK_TARGET_FOLLOW_RATIO),
        _vec_scale(look_up, local_z * _AUDITORY_LOOK_TARGET_FOLLOW_RATIO),
    )
    cam_pos = _vec_add(_vec_add(cam_center, cam_follow), _vec_scale(cam_up, 0.08))
    look_target = _vec_add(
        _vec_add(look_center, look_follow),
        _vec_add(_vec_scale(tangent, 0.45), _vec_scale(look_up, 0.03)),
    )
    camera = _look_at_camera(
        position=cam_pos,
        target=look_target,
        h_fov_deg=50.0,
        v_fov_deg=38.0,
        far_clip=420.0,
    )

    instances: list[_AssetInstance] = []
    geometry_start = ball_distance - (_AUDITORY_CAMERA_BACK_DISTANCE + _AUDITORY_TUNNEL_BACKFILL_DISTANCE)
    geometry_end = ball_distance + _AUDITORY_TUNNEL_LOOKAHEAD_DISTANCE

    distance = geometry_start + (_AUDITORY_TUBE_SEGMENT_LENGTH * 0.5)
    while distance <= geometry_end:
        center, tangent, _right, _up = _tube_frame(distance, span=_AUDITORY_PATH_SPAN)
        hpr = _world_hpr_from_tangent(tangent=tangent, roll_deg=0.0)
        instances.append(
            _AssetInstance(
                asset_id="auditory_tunnel_segment",
                position=center,
                hpr_deg=hpr,
                scale=(_AUDITORY_TUBE_RX, 1.0, _AUDITORY_TUBE_RZ),
                color=(0.08, 0.14, 0.38, 0.78),
            )
        )
        distance += _AUDITORY_TUBE_SEGMENT_LENGTH

    distance = geometry_start
    while distance <= geometry_end:
        center, tangent, _right, _up = _tube_frame(distance, span=_AUDITORY_PATH_SPAN)
        hpr = _world_hpr_from_tangent(tangent=tangent, roll_deg=0.0)
        instances.append(
            _AssetInstance(
                asset_id="auditory_tunnel_rib",
                position=center,
                hpr_deg=hpr,
                scale=(_AUDITORY_TUBE_RX, 1.0, _AUDITORY_TUBE_RZ),
                color=(0.34, 0.52, 0.98, 0.94),
            )
        )
        distance += _AUDITORY_TUBE_RIB_STEP

    ball_center, _ball_tangent, right, up = _tube_frame(ball_distance, span=_AUDITORY_PATH_SPAN)
    gate_asset_ids: set[str] = set()
    if payload is not None:
        ball_center = _vec_add(ball_center, _vec_add(_vec_scale(right, local_x), _vec_scale(up, local_z)))
    instances.append(
        _AssetInstance(
            asset_id="auditory_ball",
            position=ball_center,
            hpr_deg=((travel_offset * 125.0) % 360.0, 0.0, 0.0),
            scale=(0.11, 0.11, 0.11),
            color=ball_color,
        )
    )

    if payload is not None:
        y_half_span = max(0.08, float(payload.tube_half_height))
        visible_gates = [gate for gate in payload.gates if float(gate.x_norm) >= AUDITORY_GATE_RETIRE_X_NORM]
        visible_gates.sort(key=lambda gate: float(gate.x_norm), reverse=True)
        for gate in visible_gates[:14]:
            ahead = _auditory_gate_ahead_distance_for_x_norm(float(gate.x_norm))
            distance = ball_distance + ahead
            if distance < (ball_distance - 7.0):
                continue
            center, tangent, right, up = _tube_frame(distance, span=_AUDITORY_PATH_SPAN)
            depth_t = max(0.0, min(1.0, (ahead - 0.8) / (38.0 - 0.8)))
            local_z = max(-1.0, min(1.0, float(gate.y_norm) / y_half_span)) * (_AUDITORY_TUBE_RZ * 0.62)
            radius = max(0.16, (float(gate.aperture_norm) / y_half_span) * (_AUDITORY_TUBE_RZ * 0.82))
            gate_pos = _vec_add(center, _vec_scale(up, local_z))
            asset_id = _auditory_gate_asset_id(gate.shape)
            gate_asset_ids.add(asset_id)
            instances.append(
                _AssetInstance(
                    asset_id=asset_id,
                    position=gate_pos,
                    hpr_deg=_world_hpr_from_tangent(tangent=tangent, roll_deg=-6.0),
                    scale=(radius, radius, radius),
                    color=_auditory_depth_gate_rgba(
                        base=_auditory_gate_rgba(gate.color),
                        depth_t=depth_t,
                    ),
                )
            )

    return _ScenePlan(
        kind="auditory",
        rect=pygame.Rect(scene.world),
        camera=camera,
        asset_instances=tuple(instances),
        overlay_primitives=(),
        asset_ids=(
            "auditory_ball",
            *tuple(sorted(gate_asset_ids)),
            "auditory_tunnel_rib",
            "auditory_tunnel_segment",
        ),
        entity_count=len(instances),
    )


def _build_rapid_tracking_scene_plan(scene: RapidTrackingGlScene) -> _ScenePlan:
    rect = pygame.Rect(scene.world)
    payload = scene.payload
    if payload is None:
        return _ScenePlan(
            kind="rapid_tracking",
            rect=rect,
            camera=None,
            asset_instances=(),
            overlay_primitives=(),
            asset_ids=(
                "building_hangar",
                "building_tower",
                "forest_canopy_patch",
                "helicopter_green",
                "plane_blue",
                "plane_green",
                "plane_red",
                "plane_yellow",
                "road_dirt_segment",
                "road_paved_segment",
                "shrubs_low_cluster",
                "soldiers_patrol",
                "terrain_hill_mound",
                "terrain_lake_patch",
                "terrain_rock_cluster",
                "trees_field_cluster",
                "trees_pine_cluster",
                "truck_olive",
                "vehicle_tracked",
            ),
            entity_count=0,
        )

    rig = rapid_tracking_camera_rig_state(
        elapsed_s=float(payload.phase_elapsed_s),
        seed=int(payload.session_seed),
        progress=float(payload.scene_progress),
        camera_yaw_deg=float(payload.camera_yaw_deg),
        camera_pitch_deg=float(payload.camera_pitch_deg),
        zoom=float(payload.capture_zoom),
        target_kind=str(payload.target_kind),
        target_world_x=float(payload.target_world_x),
        target_world_y=float(payload.target_world_y),
        focus_world_x=float(payload.focus_world_x),
        focus_world_y=float(payload.focus_world_y),
        turbulence_strength=float(payload.turbulence_strength),
    )
    camera = _SceneCamera(
        position=(float(rig.cam_world_x), float(rig.cam_world_y), float(rig.cam_world_z)),
        heading_deg=float(rig.view_heading_deg),
        pitch_deg=float(rig.view_pitch_deg),
        h_fov_deg=float(rig.fov_deg),
        v_fov_deg=max(18.0, float(rig.fov_deg) * 0.78),
        near_clip=0.4,
        far_clip=1800.0,
    )

    target = build_rapid_tracking_scene_target(payload=payload, size=(rect.w, rect.h))
    target_asset_id = _rapid_tracking_target_asset_id(target.kind, target.variant)
    layout = build_rapid_tracking_compound_layout(seed=int(payload.session_seed))

    def layout_world_xy(track_x: float, track_y: float) -> tuple[float, float]:
        return rapid_tracking_track_to_world_xy(
            track_x=float(track_x),
            track_y=float(track_y),
            path_lateral_bias=float(layout.path_lateral_bias),
        )

    def anchored_world_pos(*, anchor, y_bias: float = 0.0, clearance: float = 0.0) -> tuple[float, float, float]:
        wx, wy = layout_world_xy(float(anchor.x), float(anchor.y))
        wy += float(y_bias)
        wz = float(rapid_tracking_terrain_height(float(wx), float(wy))) + float(clearance)
        return float(wx), float(wy), float(wz)

    def road_piece_instance(
        *,
        asset_id: str,
        start_xy: tuple[float, float],
        end_xy: tuple[float, float],
        width: float,
    ) -> _AssetInstance:
        start_wx, start_wy = layout_world_xy(float(start_xy[0]), float(start_xy[1]))
        end_wx, end_wy = layout_world_xy(float(end_xy[0]), float(end_xy[1]))
        center_x = (start_wx + end_wx) * 0.5
        center_y = (start_wy + end_wy) * 0.5
        start_z = float(rapid_tracking_terrain_height(float(start_wx), float(start_wy)))
        end_z = float(rapid_tracking_terrain_height(float(end_wx), float(end_wy)))
        length = max(1.0, math.hypot(float(end_wx - start_wx), float(end_wy - start_wy)))
        heading = _wrap_heading_deg(float(end_wx - start_wx), float(end_wy - start_wy))
        pitch = -math.degrees(math.atan2(end_z - start_z, max(1e-6, length)))
        return _AssetInstance(
            asset_id=asset_id,
            position=(center_x, center_y, max(start_z, end_z) + 0.05),
            hpr_deg=(heading, pitch, 0.0),
            scale=(float(width), float(length) + 2.0, 1.0),
        )

    def layout_heading(token: str) -> float:
        return float(
            rapid_tracking_seed_unit(
                seed=int(layout.seed),
                salt=str(token),
            )
            * 360.0
        )

    target_world_z = estimated_target_world_z(
        kind=str(payload.target_kind),
        target_world_x=float(payload.target_world_x),
        target_world_y=float(payload.target_world_y),
        elapsed_s=float(payload.phase_elapsed_s),
        scene_progress=float(payload.scene_progress),
        seed=int(payload.session_seed),
    )
    target_ground_z = float(
        rapid_tracking_terrain_height(
            float(payload.target_world_x),
            float(payload.target_world_y),
        )
    )
    target_heading = 0.0
    if target.kind in {"jet", "helicopter"}:
        target_heading = _wrap_heading_deg(float(payload.target_vx), float(payload.target_vy))
    if target_asset_id in {"building_hangar", "building_tower"}:
        target_world_z = target_ground_z
    elif target_asset_id in {"truck_olive", "vehicle_tracked"}:
        target_world_z = target_ground_z + 0.82
    elif target_asset_id == "soldiers_patrol":
        target_world_z = target_ground_z
    scale_factor = (
        7.6
        if target_asset_id == "building_hangar"
        else 4.2
        if target_asset_id == "building_tower"
        else 2.8
        if target_asset_id in {"truck_olive", "vehicle_tracked"}
        else 2.2
        if target_asset_id == "soldiers_patrol"
        else 2.6
        if target_asset_id == "helicopter_green"
        else 2.1
    )

    instances = [
        _AssetInstance(
            asset_id=target_asset_id,
            position=(float(payload.target_world_x), float(payload.target_world_y), float(target_world_z)),
            hpr_deg=(target_heading, 0.0, 0.0),
            scale=(scale_factor, scale_factor, scale_factor),
        )
    ]

    for obstacle in layout.obstacles:
        wx, wy = layout_world_xy(float(obstacle.x), float(obstacle.y))
        wz = float(rapid_tracking_terrain_height(float(wx), float(wy)))
        scale_x = abs(layout_world_xy(float(obstacle.x + obstacle.radius_x), float(obstacle.y))[0] - wx)
        scale_y = abs(layout_world_xy(float(obstacle.x), float(obstacle.y + obstacle.radius_y))[1] - wy)
        if obstacle.kind == "lake":
            instances.append(
                _AssetInstance(
                    asset_id="terrain_lake_patch",
                    position=(float(wx), float(wy), float(wz) - 0.14),
                    hpr_deg=(float(obstacle.rotation_deg), 0.0, 0.0),
                    scale=(max(6.0, scale_x), max(6.0, scale_y), 1.0),
                )
            )
        elif obstacle.kind == "hill":
            instances.append(
                _AssetInstance(
                    asset_id="terrain_hill_mound",
                    position=(float(wx), float(wy), float(wz) + (float(obstacle.height) * 0.24)),
                    hpr_deg=(float(obstacle.rotation_deg), 0.0, 0.0),
                    scale=(max(5.0, scale_x * 1.3), max(5.0, scale_y * 1.2), max(2.0, float(obstacle.height) * 4.2)),
                )
            )
        else:
            instances.append(
                _AssetInstance(
                    asset_id="terrain_rock_cluster",
                    position=(float(wx), float(wy), float(wz) + 0.18),
                    hpr_deg=(float(obstacle.rotation_deg), 0.0, 0.0),
                    scale=(max(1.4, scale_x * 0.9), max(1.4, scale_y * 0.9), max(1.0, float(obstacle.height) * 2.2)),
                )
            )

    for road_segment in layout.road_segments:
        road_asset = "road_paved_segment" if str(road_segment.surface) == "paved" else "road_dirt_segment"
        road_width = 7.8 if road_asset == "road_paved_segment" else 6.0
        for start_xy, end_xy in zip(road_segment.points, road_segment.points[1:], strict=False):
            instances.append(
                road_piece_instance(
                    asset_id=road_asset,
                    start_xy=start_xy,
                    end_xy=end_xy,
                    width=road_width,
                )
            )

    for anchor in layout.building_anchors:
        asset_id = "building_tower" if str(anchor.variant) == "tower" else "building_hangar"
        wx, wy, wz = anchored_world_pos(
            anchor=anchor,
            clearance=2.1 if asset_id == "building_tower" else 1.1,
        )
        scale = (
            3.6 + (rapid_tracking_seed_unit(seed=int(layout.seed), salt=str(anchor.anchor_id)) * 0.8)
            if asset_id == "building_tower"
            else 6.4 + (rapid_tracking_seed_unit(seed=int(layout.seed), salt=str(anchor.anchor_id)) * 1.2)
        )
        instances.append(
            _AssetInstance(
                asset_id=asset_id,
                position=(wx, wy, wz),
                hpr_deg=(layout_heading(f"{anchor.anchor_id}:heading"), 0.0, 0.0),
                scale=(scale, scale, scale),
            )
        )

    for idx, anchor in enumerate(layout.patrol_anchors):
        asset_id = "soldiers_patrol" if idx % 2 == 0 else "vehicle_tracked" if idx % 5 == 0 else "trees_field_cluster"
        clearance = 0.82 if asset_id == "vehicle_tracked" else 0.0
        scale = 1.8 + (0.12 * (idx % 3)) if asset_id == "soldiers_patrol" else 2.2 if asset_id == "vehicle_tracked" else 2.8 + (0.22 * (idx % 4))
        wx, wy, wz = anchored_world_pos(anchor=anchor, y_bias=6.0, clearance=clearance)
        instances.append(
            _AssetInstance(
                asset_id=asset_id,
                position=(wx, wy, wz),
                hpr_deg=(layout_heading(f"{anchor.anchor_id}:heading"), 0.0, 0.0),
                scale=(scale, scale, scale),
            )
        )

    for idx, anchor in enumerate(layout.road_anchors[:: max(1, len(layout.road_anchors) // 12 or 1)]):
        asset_id = "truck_olive" if idx % 2 == 0 else "vehicle_tracked" if idx % 5 == 0 else "shrubs_low_cluster"
        clearance = 0.82 if asset_id in {"truck_olive", "vehicle_tracked"} else 0.0
        scale = 2.8 if asset_id == "truck_olive" else 2.4 if asset_id == "vehicle_tracked" else 3.6
        wx, wy, wz = anchored_world_pos(anchor=anchor, y_bias=4.0, clearance=clearance)
        instances.append(
            _AssetInstance(
                asset_id=asset_id,
                position=(wx, wy, wz),
                hpr_deg=(layout_heading(f"{anchor.anchor_id}:heading"), 0.0, 0.0),
                scale=(scale, scale, scale),
            )
        )

    for idx, anchor in enumerate(layout.helicopter_anchors):
        wx, wy, terrain = anchored_world_pos(anchor=anchor)
        altitude = 12.0 + float(layout.altitude_bias) + (idx % 3) * 1.6
        instances.append(
            _AssetInstance(
                asset_id="helicopter_green",
                position=(wx, wy, terrain + altitude),
                hpr_deg=(layout_heading(f"{anchor.anchor_id}:air-heading"), 0.0, 0.0),
                scale=(2.3, 2.3, 2.3),
            )
        )

    jet_assets = ("plane_blue", "plane_green", "plane_yellow", "plane_red")
    for idx, anchor in enumerate(layout.jet_anchors):
        wx, wy, terrain = anchored_world_pos(anchor=anchor)
        altitude = 18.0 + float(layout.altitude_bias) + (idx % 4) * 1.8
        asset_id = jet_assets[idx % len(jet_assets)]
        instances.append(
            _AssetInstance(
                asset_id=asset_id,
                position=(wx, wy, terrain + altitude),
                hpr_deg=(layout_heading(f"{anchor.anchor_id}:air-heading"), 0.0, 0.0),
                scale=(1.9, 1.9, 1.9),
            )
        )

    for idx, cluster in enumerate(layout.scenic_clusters):
        wx, wy = layout_world_xy(float(cluster.x), float(cluster.y))
        wz = float(rapid_tracking_terrain_height(float(wx), float(wy)))
        asset_id = str(cluster.asset_id)
        if asset_id == "trees_field_cluster" and idx % 2 == 1:
            asset_id = "trees_pine_cluster"
        scale = (
            3.0 if cluster.asset_id == "forest_canopy_patch" else 2.6 if "tree" in cluster.asset_id else 3.2
        ) * float(cluster.scale) * (1.0 + (0.04 * max(0, int(cluster.count) - 1)))
        instances.append(
            _AssetInstance(
                asset_id=asset_id,
                position=(float(wx), float(wy), float(wz)),
                hpr_deg=(layout_heading(f"{cluster.cluster_id}:heading"), 0.0, 0.0),
                scale=(scale, scale, scale),
            )
        )

    unique_asset_ids = tuple(
        sorted(
            {
                "building_hangar",
                "building_tower",
                "forest_canopy_patch",
                "helicopter_green",
                "plane_blue",
                "plane_green",
                "plane_red",
                "plane_yellow",
                "road_dirt_segment",
                "road_paved_segment",
                "shrubs_low_cluster",
                "soldiers_patrol",
                "terrain_hill_mound",
                "terrain_lake_patch",
                "terrain_rock_cluster",
                "trees_field_cluster",
                "trees_pine_cluster",
                "truck_olive",
                "vehicle_tracked",
                target_asset_id,
            }
        )
    )
    return _ScenePlan(
        kind="rapid_tracking",
        rect=rect,
        camera=camera,
        asset_instances=tuple(instances),
        overlay_primitives=(),
        asset_ids=unique_asset_ids,
        entity_count=len(instances),
    )


def _spatial_terrain_height(*, wx: float, wy: float) -> float:
    ridge = 4.2 * math.sin((wx * 0.030) + 0.7) * math.exp(-((wy - 126.0) ** 2) * 0.00008)
    hill_a = 7.6 * math.exp(-(((wx + 24.0) ** 2) * 0.0018) - (((wy - 98.0) ** 2) * 0.00060))
    hill_b = 6.8 * math.exp(-(((wx - 30.0) ** 2) * 0.0016) - (((wy - 152.0) ** 2) * 0.00048))
    return float(ridge + hill_a + hill_b)


def _spatial_grid_to_world(
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
    terrain = _spatial_terrain_height(wx=wx, wy=wy)
    wz = terrain + 2.4 + (z_norm * 24.0)
    return wx, wy, wz, terrain


def _build_spatial_integration_scene_plan(scene: SpatialIntegrationGlScene) -> _ScenePlan:
    rect = pygame.Rect(scene.world)
    payload = scene.payload
    if payload is None:
        return _ScenePlan(
            kind="spatial_integration",
            rect=rect,
            camera=None,
            asset_instances=(),
            overlay_primitives=(),
            asset_ids=("building_tower", "forest_canopy_patch", "plane_blue", "trees_field_cluster"),
            entity_count=0,
        )

    now_wx, now_wy, now_wz, _ = _spatial_grid_to_world(
        x=int(payload.aircraft_now.x),
        y=int(payload.aircraft_now.y),
        z=int(payload.aircraft_now.z),
        grid_cols=int(payload.grid_cols),
        grid_rows=int(payload.grid_rows),
        alt_levels=int(payload.alt_levels),
    )
    prev_wx, prev_wy, prev_wz, _ = _spatial_grid_to_world(
        x=int(payload.aircraft_prev.x),
        y=int(payload.aircraft_prev.y),
        z=int(payload.aircraft_prev.z),
        grid_cols=int(payload.grid_cols),
        grid_rows=int(payload.grid_rows),
        alt_levels=int(payload.alt_levels),
    )
    dir_x = now_wx - prev_wx
    dir_y = now_wy - prev_wy
    dir_z = now_wz - prev_wz
    if (dir_x * dir_x) + (dir_y * dir_y) + (dir_z * dir_z) > 1e-8:
        horiz = max(1e-6, math.hypot(dir_x, dir_y))
        bank_deg = max(-34.0, min(34.0, (dir_x / horiz) * 26.0))
        plane_hpr = _world_hpr_from_tangent(
            tangent=(dir_x, dir_y, dir_z),
            roll_deg=bank_deg,
        )
    else:
        plane_hpr = (0.0, 0.0, 0.0)

    if payload.scene_view is SpatialIntegrationSceneView.TOPDOWN:
        cam_pos = (now_wx * 0.22, now_wy + 8.0, 244.0)
        look_target = (now_wx * 0.22, now_wy, max(0.0, now_wz - 8.0))
        camera = _look_at_camera(
            position=cam_pos,
            target=look_target,
            h_fov_deg=44.0,
            v_fov_deg=44.0,
            far_clip=520.0,
        )
    else:
        cam_pos = (-184.0, 34.0, 88.0)
        look_target = (now_wx * 0.35, now_wy + 34.0, max(8.0, now_wz + 8.0))
        camera = _look_at_camera(
            position=cam_pos,
            target=look_target,
            h_fov_deg=52.0,
            v_fov_deg=41.0,
            far_clip=620.0,
        )

    instances: list[_AssetInstance] = [
        _AssetInstance(
            asset_id="plane_blue",
            position=(now_wx, now_wy, now_wz),
            hpr_deg=plane_hpr,
            scale=(1.6, 1.6, 1.6),
        )
    ]
    if bool(payload.show_aircraft_motion):
        instances.append(
            _AssetInstance(
                asset_id="plane_green",
                position=(prev_wx, prev_wy, prev_wz),
                hpr_deg=plane_hpr,
                scale=(1.3, 1.3, 1.3),
                color=(0.68, 0.88, 0.74, 0.42),
            )
        )
        future_wx, future_wy, future_wz, _ = _spatial_grid_to_world(
            x=int(payload.aircraft_now.x + payload.velocity.dx),
            y=int(payload.aircraft_now.y + payload.velocity.dy),
            z=int(payload.aircraft_now.z + payload.velocity.dz),
            grid_cols=int(payload.grid_cols),
            grid_rows=int(payload.grid_rows),
            alt_levels=int(payload.alt_levels),
        )
        instances.append(
            _AssetInstance(
                asset_id="plane_yellow",
                position=(future_wx, future_wy, future_wz),
                hpr_deg=plane_hpr,
                scale=(1.2, 1.2, 1.2),
                color=(0.94, 0.88, 0.54, 0.34),
            )
        )

    landmark_specs = tuple(payload.landmarks)
    for idx, landmark in enumerate(landmark_specs):
        wx, wy, wz, terrain = _spatial_grid_to_world(
            x=int(landmark.x),
            y=int(landmark.y),
            z=0,
            grid_cols=int(payload.grid_cols),
            grid_rows=int(payload.grid_rows),
            alt_levels=int(payload.alt_levels),
        )
        if str(landmark.label).upper() == str(payload.query_label).upper():
            instances.append(
                _AssetInstance(
                    asset_id="building_tower",
                    position=(wx, wy, terrain),
                    scale=(3.4, 3.4, 3.4),
                )
            )
            continue
        asset_id = "trees_field_cluster" if idx % 2 == 0 else "forest_canopy_patch"
        scale = 2.8 if asset_id == "trees_field_cluster" else 3.2
        instances.append(
            _AssetInstance(
                asset_id=asset_id,
                position=(wx, wy, terrain),
                scale=(scale, scale, scale),
            )
        )

    return _ScenePlan(
        kind="spatial_integration",
        rect=rect,
        camera=camera,
        asset_instances=tuple(instances),
        overlay_primitives=(),
        asset_ids=("building_tower", "forest_canopy_patch", "plane_blue", "plane_green", "plane_yellow", "trees_field_cluster"),
        entity_count=len(instances),
    )


def _build_trace_test_1_scene_plan(scene: TraceTest1GlScene) -> _ScenePlan:
    rect = pygame.Rect(scene.world)
    payload = scene.payload
    if payload is None:
        return _ScenePlan(
            kind="trace_test_1",
            rect=rect,
            camera=None,
            asset_instances=(),
            overlay_primitives=(),
            asset_ids=("plane_blue", "plane_green", "plane_red", "plane_yellow"),
            entity_count=0,
        )

    camera = _look_at_camera(
        position=(0.0, -208.0, 30.0),
        target=(0.0, 104.0, 15.0),
        h_fov_deg=24.0,
        v_fov_deg=22.0,
        far_clip=720.0,
    )
    instances: list[_AssetInstance] = [
        _AssetInstance(
            asset_id="plane_red",
            position=payload.scene.red_frame.position,
            hpr_deg=(
                float(payload.scene.red_frame.attitude.yaw_deg),
                float(payload.scene.red_frame.attitude.pitch_deg),
                float(payload.scene.red_frame.attitude.roll_deg),
            ),
            scale=(1.7, 1.7, 1.7),
        )
    ]
    blue_assets = ("plane_blue", "plane_green", "plane_yellow", "plane_blue")
    blue_colors = (
        (0.34, 0.52, 0.90, 0.96),
        (0.42, 0.76, 0.54, 0.96),
        (0.94, 0.82, 0.34, 0.96),
        (0.66, 0.76, 0.94, 0.96),
    )
    for idx, frame in enumerate(payload.scene.blue_frames):
        instances.append(
            _AssetInstance(
                asset_id=blue_assets[idx % len(blue_assets)],
                position=frame.position,
                hpr_deg=(
                    float(frame.attitude.yaw_deg),
                    float(frame.attitude.pitch_deg),
                    float(frame.attitude.roll_deg),
                ),
                scale=(1.35, 1.35, 1.35),
                color=blue_colors[idx % len(blue_colors)],
            )
        )
    return _ScenePlan(
        kind="trace_test_1",
        rect=rect,
        camera=camera,
        asset_instances=tuple(instances),
        overlay_primitives=(),
        asset_ids=("plane_blue", "plane_green", "plane_red", "plane_yellow"),
        entity_count=len(instances),
    )


def _trace_test_2_asset(track) -> tuple[str, tuple[float, float, float, float] | None]:
    token = str(track.color_name).strip().lower()
    if token == "red":
        return "plane_red", None
    if token == "blue":
        return "plane_blue", None
    if token == "yellow":
        return "plane_yellow", None
    if token == "silver":
        return "plane_green", (0.80, 0.84, 0.90, 0.96)
    return "plane_green", None


def _build_trace_test_2_scene_plan(scene: TraceTest2GlScene) -> _ScenePlan:
    rect = pygame.Rect(scene.world)
    payload = scene.payload
    if payload is None:
        return _ScenePlan(
            kind="trace_test_2",
            rect=rect,
            camera=None,
            asset_instances=(),
            overlay_primitives=(),
            asset_ids=("plane_blue", "plane_green", "plane_red", "plane_yellow"),
            entity_count=0,
        )

    camera = _look_at_camera(
        position=(0.0, -72.0, 28.0),
        target=(0.0, 126.0, 12.0),
        h_fov_deg=38.0,
        v_fov_deg=30.0,
        far_clip=820.0,
    )
    progress = float(payload.observe_progress)
    instances: list[_AssetInstance] = []
    for track in payload.aircraft:
        position = trace_test_2_track_position(track=track, progress=progress)
        tangent = trace_test_2_track_tangent(track=track, progress=progress)
        asset_id, color = _trace_test_2_asset(track)
        instances.append(
            _AssetInstance(
                asset_id=asset_id,
                position=(float(position.x), float(position.y), float(position.z)),
                hpr_deg=_world_hpr_from_tangent(tangent=tangent, roll_deg=0.0),
                scale=(1.18, 1.18, 1.18),
                color=color,
            )
        )
    return _ScenePlan(
        kind="trace_test_2",
        rect=rect,
        camera=camera,
        asset_instances=tuple(instances),
        overlay_primitives=(),
        asset_ids=("plane_blue", "plane_green", "plane_red", "plane_yellow"),
        entity_count=len(instances),
    )


class _GeometryBatch:
    def __init__(self) -> None:
        self.triangles: list[_ColorVertex] = []
        self.lines: list[_ColorVertex] = []
        self.textured: list[_TexVertex] = []

    def clear(self) -> None:
        self.triangles.clear()
        self.lines.clear()
        self.textured.clear()


class ModernSceneRenderer:
    def __init__(
        self,
        *,
        window_size: tuple[int, int],
        asset_catalog: RenderAssetCatalog | None = None,
    ) -> None:
        try:
            self._ctx = moderngl.create_context(require=330)
        except Exception as exc:
            raise RendererBootstrapError(
                f"ModernGL context creation failed with {type(exc).__name__}: {exc}"
            ) from exc

        self._win_w = max(1, int(window_size[0]))
        self._win_h = max(1, int(window_size[1]))
        self._asset_catalog = asset_catalog or RenderAssetCatalog()
        self._scene_assets = _SceneAssetLibrary(self._asset_catalog)
        self._batch = _GeometryBatch()
        self._last_scene_debug: dict[str, object] = {
            "kind": "none",
            "entity_count": 0,
            "viewport": (self._win_w, self._win_h),
        }

        self._ctx.enable(moderngl.BLEND)
        self._ctx.blend_func = (
            moderngl.SRC_ALPHA,
            moderngl.ONE_MINUS_SRC_ALPHA,
            moderngl.ONE,
            moderngl.ONE_MINUS_SRC_ALPHA,
        )
        self._ctx.disable(moderngl.CULL_FACE)
        self._ctx.disable(moderngl.DEPTH_TEST)

        self._color_program = self._ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_pos;
                in vec4 in_color;
                uniform vec2 u_viewport;
                out vec4 v_color;
                void main() {
                    vec2 ndc = vec2(
                        (in_pos.x / u_viewport.x) * 2.0 - 1.0,
                        (in_pos.y / u_viewport.y) * 2.0 - 1.0
                    );
                    gl_Position = vec4(ndc, 0.0, 1.0);
                    v_color = in_color;
                }
            """,
            fragment_shader="""
                #version 330
                in vec4 v_color;
                out vec4 f_color;
                void main() {
                    f_color = v_color;
                }
            """,
        )
        self._color_program["u_viewport"].value = (float(self._win_w), float(self._win_h))

        self._texture_program = self._ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_pos;
                in vec2 in_uv;
                in vec4 in_color;
                uniform vec2 u_viewport;
                out vec2 v_uv;
                out vec4 v_color;
                void main() {
                    vec2 ndc = vec2(
                        (in_pos.x / u_viewport.x) * 2.0 - 1.0,
                        (in_pos.y / u_viewport.y) * 2.0 - 1.0
                    );
                    gl_Position = vec4(ndc, 0.0, 1.0);
                    v_uv = in_uv;
                    v_color = in_color;
                }
            """,
            fragment_shader="""
                #version 330
                uniform sampler2D u_texture;
                in vec2 v_uv;
                in vec4 v_color;
                out vec4 f_color;
                void main() {
                    f_color = texture(u_texture, v_uv) * v_color;
                }
            """,
        )
        self._texture_program["u_viewport"].value = (float(self._win_w), float(self._win_h))
        self._texture_program["u_texture"].value = 0

        vortex_bytes, vortex_size = self._build_vortex_rgba(size=640)
        self._vortex_texture = self._ctx.texture((vortex_size, vortex_size), 4, vortex_bytes)
        self._vortex_texture.repeat_x = True
        self._vortex_texture.repeat_y = True
        self._vortex_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)

        self._ui_texture: moderngl.Texture | None = None
        self._ui_tex_size: tuple[int, int] = (0, 0)

    def resize(self, *, window_size: tuple[int, int]) -> None:
        self._win_w = max(1, int(window_size[0]))
        self._win_h = max(1, int(window_size[1]))
        self._color_program["u_viewport"].value = (float(self._win_w), float(self._win_h))
        self._texture_program["u_viewport"].value = (float(self._win_w), float(self._win_h))

    def debug_last_scene(self) -> dict[str, object]:
        return dict(self._last_scene_debug)

    def render_frame(self, *, ui_surface: pygame.Surface, scene: GlScene | None) -> None:
        try:
            self._ctx.screen.use()
            self._ctx.viewport = (0, 0, self._win_w, self._win_h)
            self._ctx.clear(0.01, 0.02, 0.06, 1.0)
            self._batch.clear()

            if scene is not None:
                self._draw_scene(scene=scene)
            else:
                self._last_scene_debug = {
                    "kind": "none",
                    "entity_count": 0,
                    "viewport": (self._win_w, self._win_h),
                }

            self._flush_scene_textures()
            self._flush_color_geometry()
            self._draw_ui_surface(ui_surface=ui_surface)
        except RenderAssetResolutionError:
            raise
        except Exception as exc:
            raise RendererRenderError(
                f"ModernGL frame render failed with {type(exc).__name__}: {exc}"
            ) from exc

    def _draw_scene(self, *, scene: GlScene) -> None:
        kind = gl_scene_name(scene)
        if isinstance(scene, AuditoryGlScene):
            scene_plan = _build_auditory_scene_plan(scene)
            self._scene_assets.require_many(scene_plan.asset_ids)
            entity_count = self._draw_auditory_scene(scene=scene, scene_plan=scene_plan)
        elif isinstance(scene, RapidTrackingGlScene):
            scene_plan = _build_rapid_tracking_scene_plan(scene)
            self._scene_assets.require_many(scene_plan.asset_ids)
            entity_count = self._draw_rapid_tracking_scene(scene=scene, scene_plan=scene_plan)
        elif isinstance(scene, SpatialIntegrationGlScene):
            scene_plan = _build_spatial_integration_scene_plan(scene)
            self._scene_assets.require_many(scene_plan.asset_ids)
            entity_count = self._draw_spatial_integration_scene(scene=scene, scene_plan=scene_plan)
        elif isinstance(scene, TraceTest1GlScene):
            scene_plan = _build_trace_test_1_scene_plan(scene)
            self._scene_assets.require_many(scene_plan.asset_ids)
            entity_count = self._draw_trace_test_1_scene(scene=scene, scene_plan=scene_plan)
        else:
            scene_plan = _build_trace_test_2_scene_plan(scene)
            self._scene_assets.require_many(scene_plan.asset_ids)
            entity_count = self._draw_trace_test_2_scene(scene=scene, scene_plan=scene_plan)
        self._last_scene_debug = {
            "kind": kind,
            "entity_count": int(entity_count),
            "viewport": (self._win_w, self._win_h),
        }

    def _scene_origin(self, rect: pygame.Rect) -> tuple[float, float]:
        return float(rect.x), float(self._win_h - rect.bottom)

    def _to_screen(self, rect: pygame.Rect, x: float, y: float) -> tuple[float, float]:
        ox, oy = self._scene_origin(rect)
        return ox + float(x), oy + float(y)

    def _add_triangle(
        self,
        a: tuple[float, float],
        b: tuple[float, float],
        c: tuple[float, float],
        color: tuple[float, float, float, float],
    ) -> None:
        self._batch.triangles.extend(
            [
                _ColorVertex(a[0], a[1], *color),
                _ColorVertex(b[0], b[1], *color),
                _ColorVertex(c[0], c[1], *color),
            ]
        )

    def _add_quad(
        self,
        a: tuple[float, float],
        b: tuple[float, float],
        c: tuple[float, float],
        d: tuple[float, float],
        color: tuple[float, float, float, float],
    ) -> None:
        self._add_triangle(a, b, c, color)
        self._add_triangle(a, c, d, color)

    def _add_line(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        color: tuple[float, float, float, float],
        width: float = 1.0,
    ) -> None:
        half = max(0.5, float(width) * 0.5)
        dx = float(end[0] - start[0])
        dy = float(end[1] - start[1])
        length = math.hypot(dx, dy)
        if length <= 1e-6:
            return
        nx = -dy / length
        ny = dx / length
        a = (start[0] + (nx * half), start[1] + (ny * half))
        b = (start[0] - (nx * half), start[1] - (ny * half))
        c = (end[0] - (nx * half), end[1] - (ny * half))
        d = (end[0] + (nx * half), end[1] + (ny * half))
        self._add_quad(a, b, c, d, color)

    def _add_polyline(
        self,
        points: list[tuple[float, float]] | tuple[tuple[float, float], ...],
        color: tuple[float, float, float, float],
        *,
        closed: bool = False,
        width: float = 1.0,
    ) -> None:
        if len(points) < 2:
            return
        for idx in range(len(points) - 1):
            self._add_line(points[idx], points[idx + 1], color, width=width)
        if closed:
            self._add_line(points[-1], points[0], color, width=width)

    def _add_filled_polygon(
        self,
        points: list[tuple[float, float]] | tuple[tuple[float, float], ...],
        color: tuple[float, float, float, float],
    ) -> None:
        if len(points) < 3:
            return
        origin = points[0]
        for idx in range(1, len(points) - 1):
            self._add_triangle(origin, points[idx], points[idx + 1], color)

    def _add_circle(
        self,
        *,
        center: tuple[float, float],
        radius: float,
        color: tuple[float, float, float, float],
        filled: bool = True,
        segments: int = 28,
        width: float = 1.0,
    ) -> None:
        pts = []
        for idx in range(segments):
            angle = (idx / float(max(1, segments))) * math.tau
            pts.append(
                (
                    center[0] + (math.cos(angle) * float(radius)),
                    center[1] + (math.sin(angle) * float(radius)),
                )
            )
        if filled:
            self._add_filled_polygon(pts, color)
            return
        self._add_polyline(pts, color, closed=True, width=width)

    def _add_textured_quad(
        self,
        *,
        a: tuple[float, float],
        b: tuple[float, float],
        c: tuple[float, float],
        d: tuple[float, float],
        uv0: tuple[float, float] = (0.0, 0.0),
        uv1: tuple[float, float] = (1.0, 0.0),
        uv2: tuple[float, float] = (1.0, 1.0),
        uv3: tuple[float, float] = (0.0, 1.0),
        color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    ) -> None:
        self._batch.textured.extend(
            [
                _TexVertex(a[0], a[1], uv0[0], uv0[1], *color),
                _TexVertex(b[0], b[1], uv1[0], uv1[1], *color),
                _TexVertex(c[0], c[1], uv2[0], uv2[1], *color),
                _TexVertex(a[0], a[1], uv0[0], uv0[1], *color),
                _TexVertex(c[0], c[1], uv2[0], uv2[1], *color),
                _TexVertex(d[0], d[1], uv3[0], uv3[1], *color),
            ]
        )

    def _flush_color_geometry(self) -> None:
        if self._batch.triangles:
            tri_data = numpy.array(
                [[v.x, v.y, v.r, v.g, v.b, v.a] for v in self._batch.triangles],
                dtype="f4",
            )
            buffer = self._ctx.buffer(tri_data.tobytes())
            vao = self._ctx.vertex_array(
                self._color_program,
                [(buffer, "2f 4f", "in_pos", "in_color")],
            )
            vao.render(moderngl.TRIANGLES)
            vao.release()
            buffer.release()

    def _flush_scene_textures(self) -> None:
        if not self._batch.textured:
            return
        self._flush_textured(self._vortex_texture)
        self._batch.textured.clear()

    def _draw_ui_surface(self, *, ui_surface: pygame.Surface) -> None:
        pixels = pygame.image.tostring(ui_surface, "RGBA", True)
        size = (max(1, ui_surface.get_width()), max(1, ui_surface.get_height()))
        if self._ui_texture is None or self._ui_tex_size != size:
            if self._ui_texture is not None:
                self._ui_texture.release()
            self._ui_texture = self._ctx.texture(size, 4, pixels)
            self._ui_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._ui_tex_size = size
        else:
            self._ui_texture.write(pixels)

        if self._ui_texture is None:
            return
        quad = [
            _TexVertex(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0),
            _TexVertex(float(self._win_w), 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0),
            _TexVertex(float(self._win_w), float(self._win_h), 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            _TexVertex(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0),
            _TexVertex(float(self._win_w), float(self._win_h), 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            _TexVertex(0.0, float(self._win_h), 0.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        ]
        self._flush_textured(self._ui_texture, vertices=quad)

    def _flush_textured(
        self,
        texture: moderngl.Texture,
        *,
        vertices: list[_TexVertex] | None = None,
    ) -> None:
        data_vertices = self._batch.textured if vertices is None else vertices
        if not data_vertices:
            return
        data = numpy.array(
            [[v.x, v.y, v.u, v.v, v.r, v.g, v.b, v.a] for v in data_vertices],
            dtype="f4",
        )
        buffer = self._ctx.buffer(data.tobytes())
        vao = self._ctx.vertex_array(
            self._texture_program,
            [(buffer, "2f 2f 4f", "in_pos", "in_uv", "in_color")],
        )
        texture.use(location=0)
        vao.render(moderngl.TRIANGLES)
        vao.release()
        buffer.release()

    @staticmethod
    def _rotate_hpr_point(point: Point3, hpr_deg: Point3) -> Point3:
        return rotate_fixed_wing_point(
            point,
            heading_deg=float(hpr_deg[0]),
            pitch_deg=float(hpr_deg[1]),
            bank_deg=float(hpr_deg[2]),
        )

    @classmethod
    def _transform_instance_point(cls, point: Point3, instance: _AssetInstance) -> Point3:
        scaled = (
            float(point[0]) * float(instance.scale[0]),
            float(point[1]) * float(instance.scale[1]),
            float(point[2]) * float(instance.scale[2]),
        )
        rotated = cls._rotate_hpr_point(scaled, instance.hpr_deg)
        return (
            rotated[0] + float(instance.position[0]),
            rotated[1] + float(instance.position[1]),
            rotated[2] + float(instance.position[2]),
        )

    @classmethod
    def _transform_instance_normal(cls, normal: Point3, instance: _AssetInstance) -> Point3:
        rotated = cls._rotate_hpr_point(normal, instance.hpr_deg)
        return _vec_normalize(rotated)

    def _project_world_to_screen(
        self,
        *,
        rect: pygame.Rect,
        camera: _SceneCamera,
        point: Point3,
    ) -> tuple[float, float, float] | None:
        cam_x, cam_y, cam_z = world_to_camera_space(
            cam_world_x=float(camera.position[0]),
            cam_world_y=float(camera.position[1]),
            cam_world_z=float(camera.position[2]),
            heading_deg=float(camera.heading_deg),
            pitch_deg=float(camera.pitch_deg),
            target_world_x=float(point[0]),
            target_world_y=float(point[1]),
            target_world_z=float(point[2]),
        )
        if cam_y <= float(camera.near_clip) or cam_y >= float(camera.far_clip):
            return None
        screen_x, screen_y, _on_screen, in_front = camera_space_to_viewport(
            cam_x=cam_x,
            cam_y=cam_y,
            cam_z=cam_z,
            size=(max(1, int(rect.w)), max(1, int(rect.h))),
            h_fov_deg=float(camera.h_fov_deg),
            v_fov_deg=float(camera.v_fov_deg),
        )
        if not in_front:
            return None
        screen_point = _scene_local_top_left_to_screen(
            rect=rect,
            window_height=self._win_h,
            x=screen_x,
            y=screen_y,
        )
        return screen_point[0], screen_point[1], float(cam_y)

    def _render_scene_plan(self, *, scene_plan: _ScenePlan) -> None:
        camera = scene_plan.camera
        if camera is None or not scene_plan.asset_instances:
            return

        light_dir = _vec_normalize((0.34, 0.58, 0.74))
        projected_triangles: list[
            tuple[float, tuple[tuple[float, float], tuple[float, float], tuple[float, float]], tuple[float, float, float, float]]
        ] = []

        for instance in scene_plan.asset_instances:
            mesh = self._scene_assets.mesh(instance.asset_id)
            base_override = instance.color
            rotated_normal_cache: dict[Point3, Point3] = {}
            for triangle in mesh.triangles:
                projected_points: list[tuple[float, float]] = []
                depths: list[float] = []
                for point in triangle.points:
                    world_point = self._transform_instance_point(point, instance)
                    projected = self._project_world_to_screen(
                        rect=scene_plan.rect,
                        camera=camera,
                        point=world_point,
                    )
                    if projected is None:
                        projected_points = []
                        break
                    projected_points.append((projected[0], projected[1]))
                    depths.append(projected[2])
                if len(projected_points) != 3:
                    continue
                area = (
                    (projected_points[1][0] - projected_points[0][0]) * (projected_points[2][1] - projected_points[0][1])
                    - (projected_points[1][1] - projected_points[0][1]) * (projected_points[2][0] - projected_points[0][0])
                )
                if abs(area) <= 0.35:
                    continue
                world_normal = rotated_normal_cache.get(triangle.normal)
                if world_normal is None:
                    world_normal = self._transform_instance_normal(triangle.normal, instance)
                    rotated_normal_cache[triangle.normal] = world_normal
                light = max(0.0, min(1.0, (world_normal[0] * light_dir[0]) + (world_normal[1] * light_dir[1]) + (world_normal[2] * light_dir[2])))
                shade = 0.38 + (light * 0.62)
                if base_override is None:
                    base_rgb = mesh.color_for_role(triangle.role)
                    alpha = 1.0
                else:
                    base_rgb = tuple(float(channel) for channel in base_override[:3])
                    alpha = float(base_override[3])
                fill_color = (
                    max(0.0, min(1.0, base_rgb[0] * shade)),
                    max(0.0, min(1.0, base_rgb[1] * shade)),
                    max(0.0, min(1.0, base_rgb[2] * shade)),
                    max(0.0, min(1.0, alpha)),
                )
                projected_triangles.append(
                    (
                        sum(depths) / 3.0,
                        (
                            projected_points[0],
                            projected_points[1],
                            projected_points[2],
                        ),
                        fill_color,
                    )
                )

        projected_triangles.sort(key=lambda item: item[0], reverse=True)
        for _depth, points, color in projected_triangles:
            self._add_filled_polygon(points, color)

        for primitive in scene_plan.overlay_primitives:
            if primitive.kind == "polygon":
                if primitive.filled:
                    self._add_filled_polygon(primitive.points, primitive.color)
                else:
                    self._add_polyline(
                        primitive.points,
                        primitive.color,
                        closed=True,
                        width=primitive.width,
                    )
                continue
            if primitive.kind == "circle" and primitive.center is not None:
                self._add_circle(
                    center=primitive.center,
                    radius=primitive.radius,
                    color=primitive.color,
                    filled=primitive.filled,
                    width=primitive.width,
                )

    @staticmethod
    def _build_vortex_rgba(*, size: int) -> tuple[bytes, int]:
        n = max(64, int(size))
        cx = (n - 1) * 0.5
        cy = (n - 1) * 0.5
        inv = 1.0 / max(1.0, float(n - 1))
        out = bytearray(n * n * 4)
        for y in range(n):
            ny = ((float(y) - cy) * 2.0) * inv
            for x in range(n):
                nx = ((float(x) - cx) * 2.0) * inv
                radius = math.sqrt((nx * nx) + (ny * ny))
                angle = math.atan2(ny, nx)
                swirl = (
                    math.sin((radius * 23.0) - (angle * 5.6))
                    + math.cos((radius * 31.0) + (angle * 7.3))
                    + math.sin((nx * 17.0) - (ny * 13.0))
                )
                swirl_n = (swirl + 3.0) / 6.0
                falloff = max(0.0, min(1.0, 1.25 - (radius * 0.95)))
                grain = math.sin((x * 0.173) + (y * 0.217)) * 0.09
                lum = max(0.0, min(1.0, (swirl_n * 0.64) + (falloff * 0.30) + grain))
                i = ((y * n) + x) * 4
                out[i] = max(0, min(255, int(round(16 + (lum * 38)))))
                out[i + 1] = max(0, min(255, int(round(36 + (lum * 84)))))
                out[i + 2] = max(0, min(255, int(round(68 + (lum * 146)))))
                out[i + 3] = 255
        return bytes(out), n

    @staticmethod
    def _screen_heading_to_fixed_wing_heading(screen_heading_deg: float) -> float:
        return fixed_wing_heading_from_screen_heading(screen_heading_deg)

    @staticmethod
    def _tube_offset_at_depth(*, z: float, z_near: float, z_far: float) -> tuple[float, float]:
        span = max(0.001, float(z_far) - float(z_near))
        depth = max(0.0, min(1.0, ((-float(z)) - float(z_near)) / span))
        curve = depth**1.18
        x_wave = (math.sin((depth * math.tau * 1.05) + 0.45) * 0.74) + (
            math.sin((depth * math.tau * 2.25) - 0.35) * 0.26
        )
        y_wave = (math.sin((depth * math.tau * 1.45) + 1.45) * 0.72) + (
            math.sin((depth * math.tau * 2.55) + 0.80) * 0.28
        )
        return x_wave * (0.58 * curve), y_wave * (0.35 * curve)

    @staticmethod
    def _gate_color(name: str) -> tuple[float, float, float]:
        palette = {
            "RED": (0.92, 0.32, 0.34),
            "GREEN": (0.34, 0.88, 0.56),
            "BLUE": (0.36, 0.62, 0.94),
            "YELLOW": (0.94, 0.82, 0.34),
            "WHITE": (0.92, 0.95, 1.0),
        }
        return palette.get(str(name).upper(), (0.86, 0.88, 0.92))

    @staticmethod
    def _mix_rgb3(
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

    def _project_3d(
        self,
        *,
        x: float,
        y: float,
        z: float,
        fovy_deg: float,
        aspect: float,
        rect: pygame.Rect,
    ) -> tuple[float, float] | None:
        depth = -float(z)
        if depth <= 0.01:
            return None
        tan_half = math.tan(math.radians(float(fovy_deg) * 0.5))
        ndc_x = float(x) / max(1e-6, depth * tan_half * float(aspect))
        ndc_y = float(y) / max(1e-6, depth * tan_half)
        if abs(ndc_x) > 2.4 or abs(ndc_y) > 2.4:
            return None
        sx = ((ndc_x + 1.0) * 0.5) * float(rect.w)
        sy = ((ndc_y + 1.0) * 0.5) * float(rect.h)
        return self._to_screen(rect, sx, sy)

    def _add_aircraft_marker(
        self,
        *,
        center: tuple[float, float],
        heading_deg: float,
        size: float,
        color: tuple[float, float, float, float],
        outline: tuple[float, float, float, float],
        pitch_deg: float = 0.0,
        bank_deg: float = 0.0,
        view_yaw_deg: float = 0.0,
        view_pitch_deg: float = 20.0,
        scene_rect: pygame.Rect | None = None,
        center_is_scene_top_left: bool = False,
    ) -> None:
        palette = build_pygame_palette(
            body_color=tuple(max(0, min(255, int(round(channel * 255.0)))) for channel in color[:3]),
            outline_color=tuple(max(0, min(255, int(round(channel * 255.0)))) for channel in outline[:3]),
        )
        role_colors = {
            "body": palette.body,
            "accent": palette.accent,
            "canopy": palette.canopy,
            "engine": palette.engine,
        }
        if center_is_scene_top_left:
            if scene_rect is None:
                raise ValueError("scene_rect is required when center_is_scene_top_left=True")
            projected_faces = _project_aircraft_marker_polygons(
                rect=scene_rect,
                window_height=self._win_h,
                center_top_left=center,
                heading_deg=heading_deg,
                size=size,
                color=color,
                outline=outline,
                pitch_deg=pitch_deg,
                bank_deg=bank_deg,
                view_yaw_deg=view_yaw_deg,
                view_pitch_deg=view_pitch_deg,
            )
        else:
            projected_faces = tuple(
                _ProjectedAircraftMarkerFace(
                    role=face.role,
                    shade=float(face.shade),
                    points=tuple((float(px), float(py)) for px, py in face.points),
                )
                for face in project_fixed_wing_faces(
                    heading_deg=self._screen_heading_to_fixed_wing_heading(heading_deg),
                    pitch_deg=float(pitch_deg),
                    bank_deg=float(bank_deg),
                    cx=int(round(center[0])),
                    cy=int(round(center[1])),
                    scale=max(8.0, float(size)),
                    view_yaw_deg=view_yaw_deg,
                    view_pitch_deg=view_pitch_deg,
                )
            )
        for face in projected_faces:
            base = role_colors.get(face.role, palette.body)
            fill = tuple(max(0, min(255, int(round(channel * float(face.shade))))) for channel in base)
            fill_color = (fill[0] / 255.0, fill[1] / 255.0, fill[2] / 255.0, color[3])
            points = [(float(px), float(py)) for px, py in face.points]
            self._add_filled_polygon(points, fill_color)
            self._add_polyline(points, outline, closed=True, width=1.0)

    def _draw_auditory_scene(self, *, scene: AuditoryGlScene, scene_plan: _ScenePlan) -> int:
        rect = scene.world
        payload = scene.payload
        time_fill_ratio = scene.time_fill_ratio
        vw = max(1, int(rect.w))
        vh = max(1, int(rect.h))
        origin = self._to_screen(rect, 0.0, 0.0)
        self._add_textured_quad(
            a=origin,
            b=self._to_screen(rect, float(vw), 0.0),
            c=self._to_screen(rect, float(vw), float(vh)),
            d=self._to_screen(rect, 0.0, float(vh)),
        )
        self._render_scene_plan(scene_plan=scene_plan)

        bar_w = 132.0
        bar_h = 15.0
        bar_x = float(vw) - bar_w - 16.0
        bar_y = 12.0
        self._add_quad(
            self._to_screen(rect, bar_x, bar_y),
            self._to_screen(rect, bar_x + bar_w, bar_y),
            self._to_screen(rect, bar_x + bar_w, bar_y + bar_h),
            self._to_screen(rect, bar_x, bar_y + bar_h),
            (0.46, 0.50, 0.62, 0.70),
        )
        inner_x = bar_x + 4.0
        inner_y = bar_y + 4.0
        inner_w = bar_w - 8.0
        inner_h = bar_h - 8.0
        self._add_quad(
            self._to_screen(rect, inner_x, inner_y),
            self._to_screen(rect, inner_x + inner_w, inner_y),
            self._to_screen(rect, inner_x + inner_w, inner_y + inner_h),
            self._to_screen(rect, inner_x, inner_y + inner_h),
            (0.12, 0.14, 0.18, 0.84),
        )
        fill_ratio = 0.72 if time_fill_ratio is None else max(0.0, min(1.0, time_fill_ratio))
        fill_w = max(0.0, inner_w * fill_ratio)
        if fill_w > 0.0:
            self._add_quad(
                self._to_screen(rect, inner_x, inner_y),
                self._to_screen(rect, inner_x + fill_w, inner_y),
                self._to_screen(rect, inner_x + fill_w, inner_y + inner_h),
                self._to_screen(rect, inner_x, inner_y + inner_h),
                (0.76, 0.80, 0.86, 0.90),
            )
        return int(scene_plan.entity_count)

    def _draw_rapid_tracking_scene(self, *, scene: RapidTrackingGlScene, scene_plan: _ScenePlan) -> int:
        rect = scene.world
        payload = scene.payload
        vw = max(1, int(rect.w))
        vh = max(1, int(rect.h))
        horizon = vh * 0.48
        if payload is not None:
            rig = rapid_tracking_camera_rig_state(
                elapsed_s=float(payload.phase_elapsed_s),
                seed=int(payload.session_seed),
                progress=float(payload.scene_progress),
                camera_yaw_deg=float(payload.camera_yaw_deg),
                camera_pitch_deg=float(payload.camera_pitch_deg),
                zoom=float(payload.capture_zoom),
                target_kind=str(payload.target_kind),
                target_world_x=float(payload.target_world_x),
                target_world_y=float(payload.target_world_y),
                focus_world_x=float(payload.focus_world_x),
                focus_world_y=float(payload.focus_world_y),
                turbulence_strength=float(payload.turbulence_strength),
            )
            horizon = max(vh * 0.28, min(vh * 0.70, vh * (0.60 - (rig.view_pitch_deg / 90.0))))
        band_h = max(1.0, vh / 10.0)
        for idx in range(10):
            y0 = vh - ((idx + 1) * band_h)
            y1 = vh - (idx * band_h)
            mix = idx / 9.0
            self._add_quad(
                self._to_screen(rect, 0.0, y0),
                self._to_screen(rect, float(vw), y0),
                self._to_screen(rect, float(vw), y1),
                self._to_screen(rect, 0.0, y1),
                (0.08 + (0.12 * mix), 0.20 + (0.20 * mix), 0.28 + (0.18 * mix), 1.0),
            )
        self._add_quad(
            self._to_screen(rect, 0.0, 0.0),
            self._to_screen(rect, float(vw), 0.0),
            self._to_screen(rect, float(vw), horizon),
            self._to_screen(rect, 0.0, horizon),
            (0.10, 0.22, 0.16, 1.0),
        )
        self._render_scene_plan(scene_plan=scene_plan)
        if payload is None:
            return 0

        target = build_rapid_tracking_scene_target(payload=payload, size=(vw, vh))
        target_center = self._to_screen(rect, float(target.overlay.screen_x), float(vh - target.overlay.screen_y))
        self._add_circle(center=target_center, radius=30.0, color=(0.92, 0.98, 0.96, 0.64), filled=False)
        return max(1, int(scene_plan.entity_count))

    def _draw_spatial_integration_scene(self, *, scene: SpatialIntegrationGlScene, scene_plan: _ScenePlan) -> int:
        rect = scene.world
        payload = scene.payload
        vw = max(1, int(rect.w))
        vh = max(1, int(rect.h))
        layout = build_spatial_integration_scene_layout(payload=payload, size=(vw, vh))
        horizon = float(vh - layout.horizon_y)
        if layout.scene_view is SpatialIntegrationSceneView.TOPDOWN:
            self._add_quad(
                self._to_screen(rect, 0.0, 0.0),
                self._to_screen(rect, float(vw), 0.0),
                self._to_screen(rect, float(vw), float(vh)),
                self._to_screen(rect, 0.0, float(vh)),
                (0.24, 0.42, 0.22, 1.0),
            )
        else:
            self._add_quad(
                self._to_screen(rect, 0.0, horizon),
                self._to_screen(rect, float(vw), horizon),
                self._to_screen(rect, float(vw), float(vh)),
                self._to_screen(rect, 0.0, float(vh)),
                (0.42, 0.60, 0.82, 1.0),
            )
            self._add_quad(
                self._to_screen(rect, 0.0, 0.0),
                self._to_screen(rect, float(vw), 0.0),
                self._to_screen(rect, float(vw), horizon),
                self._to_screen(rect, 0.0, horizon),
                (0.30, 0.44, 0.22, 1.0),
            )
        self._render_scene_plan(scene_plan=scene_plan)
        for marker in layout.landmarks:
            center = self._to_screen(rect, float(marker.screen_x), float(vh - marker.screen_y))
            self._add_circle(center=center, radius=7.0, color=(0.98, 0.88, 0.42, 0.94))
            self._add_circle(center=center, radius=11.0, color=(0.98, 0.94, 0.72, 0.44), filled=False)
        prev = self._to_screen(rect, float(layout.aircraft_prev[0]), float(vh - layout.aircraft_prev[1]))
        now = self._to_screen(rect, float(layout.aircraft_now[0]), float(vh - layout.aircraft_now[1]))
        self._add_line(prev, now, (0.94, 0.78, 0.32, 0.82), width=2.0)
        if layout.aircraft_future is not None:
            future = self._to_screen(rect, float(layout.aircraft_future[0]), float(vh - layout.aircraft_future[1]))
            self._add_line(now, future, (0.94, 0.56, 0.30, 0.82), width=2.0)
        return max(int(scene_plan.entity_count), len(layout.landmarks) + 1)

    def _draw_trace_test_1_scene(self, *, scene: TraceTest1GlScene, scene_plan: _ScenePlan) -> int:
        rect = scene.world
        vw = max(1, int(rect.w))
        vh = max(1, int(rect.h))
        self._add_quad(
            self._to_screen(rect, 0.0, 0.0),
            self._to_screen(rect, float(vw), 0.0),
            self._to_screen(rect, float(vw), float(vh)),
            self._to_screen(rect, 0.0, float(vh)),
            (0.56, 0.70, 0.88, 1.0),
        )
        self._add_quad(
            self._to_screen(rect, 0.0, float(vh * 0.52)),
            self._to_screen(rect, float(vw), float(vh * 0.52)),
            self._to_screen(rect, float(vw), float(vh)),
            self._to_screen(rect, 0.0, float(vh)),
            (0.48, 0.60, 0.74, 1.0),
        )
        self._render_scene_plan(scene_plan=scene_plan)
        return int(scene_plan.entity_count)

    def _draw_trace_test_2_scene(self, *, scene: TraceTest2GlScene, scene_plan: _ScenePlan) -> int:
        rect = scene.world
        vw = max(1, int(rect.w))
        vh = max(1, int(rect.h))
        horizon = vh * 0.26
        self._add_quad(
            self._to_screen(rect, 0.0, horizon),
            self._to_screen(rect, float(vw), horizon),
            self._to_screen(rect, float(vw), float(vh)),
            self._to_screen(rect, 0.0, float(vh)),
            (0.50, 0.66, 0.86, 1.0),
        )
        self._add_quad(
            self._to_screen(rect, 0.0, 0.0),
            self._to_screen(rect, float(vw), 0.0),
            self._to_screen(rect, float(vw), horizon),
            self._to_screen(rect, 0.0, horizon),
            (0.40, 0.46, 0.52, 1.0),
        )
        self._render_scene_plan(scene_plan=scene_plan)
        return int(scene_plan.entity_count)


class ModernInstrumentCardRenderer:
    def __init__(self) -> None:
        try:
            self._ctx = moderngl.create_standalone_context(require=330)
        except Exception as exc:
            raise RendererBootstrapError(
                f"ModernGL standalone context creation failed with {type(exc).__name__}: {exc}"
            ) from exc
        self._ctx.enable(moderngl.BLEND)
        self._ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self._program = self._ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_pos;
                in vec4 in_color;
                uniform vec2 u_viewport;
                out vec4 v_color;
                void main() {
                    vec2 ndc = vec2(
                        (in_pos.x / u_viewport.x) * 2.0 - 1.0,
                        (in_pos.y / u_viewport.y) * 2.0 - 1.0
                    );
                    gl_Position = vec4(ndc, 0.0, 1.0);
                    v_color = in_color;
                }
            """,
            fragment_shader="""
                #version 330
                in vec4 v_color;
                out vec4 f_color;
                void main() {
                    f_color = v_color;
                }
            """,
        )

    def render_card(
        self,
        *,
        key,
        destination,
        draw_callback,
        size: tuple[int, int],
    ) -> None:
        width = max(1, int(size[0]))
        height = max(1, int(size[1]))
        self._program["u_viewport"].value = (float(width), float(height))
        texture = self._ctx.texture((width, height), 4)
        framebuffer = self._ctx.framebuffer(color_attachments=[texture])
        framebuffer.use()
        self._ctx.clear(0.92, 0.92, 0.92, 1.0)
        batch = _GeometryBatch()
        draw_callback(batch, key, width, height)
        if batch.triangles:
            data = numpy.array(
                [[v.x, v.y, v.r, v.g, v.b, v.a] for v in batch.triangles],
                dtype="f4",
            )
            buffer = self._ctx.buffer(data.tobytes())
            vao = self._ctx.vertex_array(
                self._program,
                [(buffer, "2f 4f", "in_pos", "in_color")],
            )
            vao.render(moderngl.TRIANGLES)
            vao.release()
            buffer.release()
        raw = framebuffer.read(components=4, alignment=1)
        surface = pygame.image.frombuffer(raw, (width, height), "RGBA").copy()
        surface = pygame.transform.flip(surface, False, True)
        pygame.image.save(surface, str(destination))
        framebuffer.release()
        texture.release()
