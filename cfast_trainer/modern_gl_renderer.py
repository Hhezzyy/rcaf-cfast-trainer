from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

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
from .auditory_capacity import (
    AUDITORY_GATE_PLAYER_X_NORM,
    AUDITORY_GATE_RETIRE_X_NORM,
    AUDITORY_GATE_SPAWN_X_NORM,
    AUDITORY_TRIANGLE_GATE_POINTS,
)
from .auditory_capacity_view import (
    AUDITORY_BALL_ANCHOR_DISTANCE,
    AUDITORY_GATE_BEHIND_DISTANCE,
    TUNNEL_CAMERA_H_FOV_DEG,
    TUNNEL_CAMERA_V_FOV_DEG,
    TUNNEL_GEOMETRY_END_DISTANCE,
    TUNNEL_GEOMETRY_START_DISTANCE,
    fixed_camera_pose_at_distance,
    gate_depth_ratio_from_distance,
    gate_distance_from_x_norm,
    run_travel_distance,
)
from .auditory_capacity_view import (
    tube_frame as _tube_frame,
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
from .rapid_tracking import (
    RapidTrackingCompoundLayout,
    build_distant_terrain_ring,
    build_rapid_tracking_compound_layout,
)
from .rapid_tracking_gl import build_scene_target as build_rapid_tracking_scene_target
from .rapid_tracking_gl import camera_rig_state as rapid_tracking_camera_rig_state
from .rapid_tracking_view import (
    TARGET_VIEW_LIMIT as RAPID_TRACKING_TARGET_VIEW_LIMIT,
)
from .rapid_tracking_view import (
    camera_space_to_viewport,
    estimated_target_world_z,
    rapid_tracking_seed_unit,
    world_to_camera_space,
)
from .rapid_tracking_view import (
    terrain_height as rapid_tracking_terrain_height,
)
from .rapid_tracking_view import (
    track_to_world_xy as rapid_tracking_track_to_world_xy,
)
from .render_assets import RenderAssetCatalog, RenderAssetResolutionError
from .spatial_integration import SpatialIntegrationLandmark, SpatialIntegrationSceneView
from .spatial_integration_gl import (
    build_scene_layout as build_spatial_integration_scene_layout,
)
from .spatial_integration_gl import landmark_visual_cell_offsets
from .trace_scene_3d import (
    TraceAircraftPose,
    TraceCameraPose,
    TraceScene3dSnapshot,
    build_trace_test_1_scene3d,
    build_trace_test_2_scene3d,
)
from .trace_test_1_gl import (
    aircraft_screen_poses_for_payload as trace_test_1_aircraft_screen_poses_for_payload,
)
from .trace_test_1_gl import (
    project_scene_position as trace_test_1_project_scene_position,
)
from .trace_test_2 import trace_test_2_track_position
from .trace_test_2_gl import (
    aircraft_screen_pose_for_track as trace_test_2_aircraft_screen_pose_for_track,
)
from .trace_test_2_gl import project_point as trace_test_2_project_point


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


@dataclass(frozen=True, slots=True)
class _DepthColorVertex:
    x: float
    y: float
    depth: float
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
    bucket: str = "playfield"


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
    static_groups: tuple[_RapidTrackingStaticGroup, ...] = ()
    backdrop_groups: tuple[_RapidTrackingStaticGroup, ...] = ()
    playfield_groups: tuple[_RapidTrackingStaticGroup, ...] = ()


@dataclass(frozen=True, slots=True)
class _RapidTrackingBackdropPlan:
    horizon_y: float
    sky_top_rgb: tuple[float, float, float]
    sky_bottom_rgb: tuple[float, float, float]
    ground_horizon_rgb: tuple[float, float, float]
    ground_base_rgb: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class _RapidTrackingStaticScene:
    layout: RapidTrackingCompoundLayout
    core_instances: tuple[_AssetInstance, ...]
    ambient_instances: tuple[_AssetInstance, ...]
    asset_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _PreparedWorldTriangle:
    points: tuple[Point3, Point3, Point3]
    normal: Point3
    base_rgb: tuple[float, float, float]
    alpha: float = 1.0


@dataclass(frozen=True, slots=True)
class _ProjectedWorldTriangle2D:
    depth: float
    points: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    camera_depths: tuple[float, float, float]
    extent_px: float


@dataclass(frozen=True, slots=True)
class _RapidTrackingStaticGroup:
    group_id: str
    layer: str
    center: Point3
    radius: float
    triangles: tuple[_PreparedWorldTriangle, ...]
    instance_count: int


@dataclass(frozen=True, slots=True)
class _RapidTrackingStaticBundle:
    groups: tuple[_RapidTrackingStaticGroup, ...]
    asset_ids: tuple[str, ...]
    instance_count: int
    triangle_count: int


@dataclass(frozen=True, slots=True)
class _TrackPointAnchor:
    x: float
    y: float


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
    bounds: tuple[Point3, Point3]
    base_z: float

    def color_for_role(self, role: str) -> tuple[float, float, float]:
        return self.role_palette.get(role, self.role_palette.get("body", (0.82, 0.84, 0.88)))


_AUDITORY_TUBE_SEGMENT_LENGTH = 4.0
_AUDITORY_TUBE_RIB_STEP = 4.0
_AUDITORY_TUBE_RX = 2.24
_AUDITORY_TUBE_RZ = 1.64
_AUDITORY_TUBE_GEOMETRY_START = float(TUNNEL_GEOMETRY_START_DISTANCE)
_AUDITORY_TUBE_GEOMETRY_END = float(TUNNEL_GEOMETRY_END_DISTANCE)


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


def _scene_camera_from_trace_camera(camera: TraceCameraPose) -> _SceneCamera:
    return _SceneCamera(
        position=tuple(float(value) for value in camera.position),
        heading_deg=float(camera.heading_deg),
        pitch_deg=float(camera.pitch_deg),
        h_fov_deg=float(camera.h_fov_deg),
        v_fov_deg=float(camera.v_fov_deg),
        near_clip=float(camera.near_clip),
        far_clip=float(camera.far_clip),
    )


def _asset_instance_from_trace_aircraft(pose: TraceAircraftPose) -> _AssetInstance:
    return _AssetInstance(
        asset_id=str(pose.asset_id),
        position=tuple(float(value) for value in pose.position),
        hpr_deg=tuple(float(value) for value in pose.hpr_deg),
        scale=tuple(float(value) for value in pose.scale),
        color=pose.color_rgba,
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


def _rotate_hpr_point(point: Point3, hpr_deg: Point3) -> Point3:
    return rotate_fixed_wing_point(
        point,
        heading_deg=float(hpr_deg[0]),
        pitch_deg=float(hpr_deg[1]),
        bank_deg=float(hpr_deg[2]),
    )


def _transform_asset_point(
    point: Point3,
    *,
    position: Point3,
    hpr_deg: Point3,
    scale: Point3,
) -> Point3:
    scaled = (
        float(point[0]) * float(scale[0]),
        float(point[1]) * float(scale[1]),
        float(point[2]) * float(scale[2]),
    )
    rotated = _rotate_hpr_point(scaled, hpr_deg)
    return (
        rotated[0] + float(position[0]),
        rotated[1] + float(position[1]),
        rotated[2] + float(position[2]),
    )


def _transform_asset_normal(normal: Point3, *, hpr_deg: Point3) -> Point3:
    return _vec_normalize(_rotate_hpr_point(normal, hpr_deg))


def _mix_rgb(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
    t: float,
) -> tuple[float, float, float]:
    blend = max(0.0, min(1.0, float(t)))
    inv = 1.0 - blend
    return (
        (left[0] * inv) + (right[0] * blend),
        (left[1] * inv) + (right[1] * blend),
        (left[2] * inv) + (right[2] * blend),
    )


def _lerp(a: float, b: float, t: float) -> float:
    return float(a) + ((float(b) - float(a)) * float(t))


def _distance_sq(ax: float, ay: float, bx: float, by: float) -> float:
    dx = float(bx) - float(ax)
    dy = float(by) - float(ay)
    return (dx * dx) + (dy * dy)


def _points_center_radius(points: tuple[Point3, ...] | list[Point3]) -> tuple[Point3, float]:
    min_x = min(point[0] for point in points)
    min_y = min(point[1] for point in points)
    min_z = min(point[2] for point in points)
    max_x = max(point[0] for point in points)
    max_y = max(point[1] for point in points)
    max_z = max(point[2] for point in points)
    center = (
        (min_x + max_x) * 0.5,
        (min_y + max_y) * 0.5,
        (min_z + max_z) * 0.5,
    )
    radius = max(
        math.sqrt(
            ((point[0] - center[0]) ** 2)
            + ((point[1] - center[1]) ** 2)
            + ((point[2] - center[2]) ** 2)
        )
        for point in points
    )
    return center, float(radius)


def _mesh_bounds_from_triangles(
    triangles: tuple[_MeshTriangle, ...] | list[_MeshTriangle],
) -> tuple[tuple[Point3, Point3], float]:
    if not triangles:
        zero = (0.0, 0.0, 0.0)
        return (zero, zero), 0.0
    points = [point for triangle in triangles for point in triangle.points]
    min_x = min(point[0] for point in points)
    min_y = min(point[1] for point in points)
    min_z = min(point[2] for point in points)
    max_x = max(point[0] for point in points)
    max_y = max(point[1] for point in points)
    max_z = max(point[2] for point in points)
    return (
        (
            (float(min_x), float(min_y), float(min_z)),
            (float(max_x), float(max_y), float(max_z)),
        ),
        float(min_z),
    )


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
        triangles.extend(_triangulate_points(role=role, points=tuple(corners[key] for key in keys)))
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
        "ERROR_RED": (0.98, 0.44, 0.22, 0.98),
        "GREEN": (0.32, 0.86, 0.50, 0.96),
        "BLUE": (0.38, 0.58, 0.96, 0.96),
        "YELLOW": (0.96, 0.84, 0.32, 0.96),
    }
    return palette.get(str(name).upper(), (0.90, 0.92, 0.98, 0.96))


def _mix_rgba(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
    t: float,
) -> tuple[float, float, float, float]:
    blend = max(0.0, min(1.0, float(t)))
    inv = 1.0 - blend
    return (
        (left[0] * inv) + (right[0] * blend),
        (left[1] * inv) + (right[1] * blend),
        (left[2] * inv) + (right[2] * blend),
        (left[3] * inv) + (right[3] * blend),
    )


def _auditory_gate_display_rgba(gate: object) -> tuple[float, float, float, float]:
    base = _auditory_gate_rgba(str(getattr(gate, "color", "WHITE")))
    flash_color = getattr(gate, "flash_color", None)
    flash_strength = float(getattr(gate, "flash_strength", 0.0))
    if flash_color is None or flash_strength <= 0.0:
        return base
    return _mix_rgba(base, _auditory_gate_rgba(str(flash_color)), flash_strength)


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


def _asset_palette(asset_id: str) -> dict[str, tuple[float, float, float]]:
    palettes: dict[str, dict[str, tuple[float, float, float]]] = {
        "plane_red": {
            "body": (0.82, 0.30, 0.28),
            "accent": (0.58, 0.18, 0.18),
            "canopy": (0.94, 0.96, 1.0),
            "engine": (0.28, 0.18, 0.18),
        },
        "plane_blue": {
            "body": (0.40, 0.56, 0.78),
            "accent": (0.24, 0.36, 0.52),
            "canopy": (0.90, 0.96, 1.0),
            "engine": (0.22, 0.28, 0.36),
        },
        "plane_green": {
            "body": (0.42, 0.62, 0.42),
            "accent": (0.24, 0.40, 0.26),
            "canopy": (0.90, 0.96, 1.0),
            "engine": (0.20, 0.26, 0.20),
        },
        "plane_yellow": {
            "body": (0.80, 0.70, 0.34),
            "accent": (0.56, 0.46, 0.20),
            "canopy": (0.94, 0.96, 1.0),
            "engine": (0.32, 0.26, 0.14),
        },
        "helicopter_green": {
            "body": (0.42, 0.52, 0.36),
            "accent": (0.24, 0.30, 0.22),
            "canopy": (0.90, 0.96, 1.0),
            "engine": (0.22, 0.24, 0.20),
        },
        "truck_olive": {
            "body": (0.48, 0.46, 0.26),
            "accent": (0.30, 0.28, 0.18),
            "engine": (0.18, 0.18, 0.18),
            "wheel": (0.10, 0.10, 0.10),
        },
        "vehicle_tracked": {
            "body": (0.40, 0.44, 0.26),
            "accent": (0.26, 0.30, 0.18),
            "engine": (0.20, 0.20, 0.20),
            "wheel": (0.11, 0.11, 0.12),
        },
        "building_hangar": {
            "body": (0.66, 0.68, 0.62),
            "roof": (0.38, 0.42, 0.40),
            "accent": (0.22, 0.26, 0.30),
        },
        "building_tower": {
            "body": (0.72, 0.72, 0.68),
            "roof": (0.34, 0.38, 0.40),
            "accent": (0.24, 0.28, 0.30),
        },
        "spatial_tent_canvas": {
            "body": (0.76, 0.70, 0.46),
            "accent": (0.44, 0.34, 0.20),
        },
        "spatial_sheep_flock": {
            "body": (0.92, 0.92, 0.88),
            "accent": (0.24, 0.24, 0.22),
        },
        "road_paved_segment": {
            "body": (0.20, 0.20, 0.20),
            "accent": (0.78, 0.74, 0.60),
        },
        "road_dirt_segment": {
            "body": (0.40, 0.32, 0.22),
            "accent": (0.52, 0.42, 0.30),
        },
        "terrain_lake_patch": {
            "body": (0.22, 0.40, 0.54),
            "accent": (0.34, 0.58, 0.70),
        },
        "terrain_hill_mound": {
            "body": (0.44, 0.46, 0.30),
            "accent": (0.56, 0.58, 0.40),
        },
        "terrain_rock_cluster": {
            "body": (0.46, 0.44, 0.40),
            "accent": (0.32, 0.30, 0.28),
        },
        "trees_pine_cluster": {
            "body": (0.18, 0.16, 0.10),
            "canopy": (0.18, 0.34, 0.20),
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
            "body": (0.28, 0.34, 0.22),
            "accent": (0.72, 0.68, 0.58),
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

    def require_many(
        self, asset_ids: tuple[str, ...] | list[str] | set[str]
    ) -> tuple[_AssetMesh, ...]:
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

    @staticmethod
    def _mesh_from_triangles(
        *,
        asset_id: str,
        triangles: list[_MeshTriangle] | tuple[_MeshTriangle, ...],
    ) -> _AssetMesh:
        triangle_tuple = tuple(triangles)
        bounds, base_z = _mesh_bounds_from_triangles(triangle_tuple)
        return _AssetMesh(
            asset_id=asset_id,
            triangles=triangle_tuple,
            role_palette=_asset_palette(asset_id),
            bounds=bounds,
            base_z=base_z,
        )

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
        return self._mesh_from_triangles(asset_id=asset_id, triangles=triangles)

    def _build_builtin_mesh(self, *, asset_id: str, builtin_kind: str) -> _AssetMesh:
        token = str(builtin_kind).strip().lower()
        if token == "fixed_wing":
            triangles: list[_MeshTriangle] = []
            for face in build_fixed_wing_mesh():
                triangles.extend(_triangulate_points(role=face.role, points=face.points))
            return self._mesh_from_triangles(asset_id=asset_id, triangles=triangles)
        if token == "helicopter":
            triangles = [
                *_box_triangles(role="body", size=(1.6, 2.0, 0.7)),
                *_box_triangles(role="body", size=(0.26, 2.6, 0.18), center=(0.0, -2.2, 0.0)),
                *_box_triangles(role="canopy", size=(1.0, 0.8, 0.5), center=(0.0, 0.9, 0.2)),
                *_box_triangles(role="accent", size=(3.6, 0.12, 0.06), center=(0.0, 0.0, 0.55)),
                *_box_triangles(role="accent", size=(0.12, 3.0, 0.06), center=(0.0, 0.0, 0.55)),
                *_box_triangles(role="accent", size=(0.9, 0.10, 0.06), center=(0.0, -3.35, 0.0)),
            ]
            return self._mesh_from_triangles(asset_id=asset_id, triangles=triangles)
        if token == "truck":
            triangles = [
                *_box_triangles(role="body", size=(1.2, 2.4, 0.55), center=(0.0, -0.10, 0.12)),
                *_box_triangles(role="accent", size=(1.0, 0.8, 0.78), center=(0.0, 1.22, 0.24)),
                *_box_triangles(role="engine", size=(1.35, 0.26, 0.18), center=(0.0, 0.0, -0.18)),
            ]
            return self._mesh_from_triangles(asset_id=asset_id, triangles=triangles)
        if token == "tracked_vehicle":
            triangles = [
                *_box_triangles(role="body", size=(1.8, 2.7, 0.56), center=(0.0, -0.02, 0.18)),
                *_box_triangles(role="accent", size=(0.94, 1.00, 0.42), center=(0.0, 0.22, 0.56)),
                *_box_triangles(role="engine", size=(0.16, 1.42, 0.12), center=(0.0, 1.30, 0.54)),
                *_box_triangles(role="wheel", size=(0.30, 2.54, 0.40), center=(-0.82, -0.02, 0.08)),
                *_box_triangles(role="wheel", size=(0.30, 2.54, 0.40), center=(0.82, -0.02, 0.08)),
            ]
            return self._mesh_from_triangles(asset_id=asset_id, triangles=triangles)
        if token == "hangar":
            triangles = [
                *_box_triangles(role="body", size=(5.2, 4.6, 1.8), center=(0.0, 0.0, 0.9)),
                *_roof_prism_triangles(role="roof", size=(5.6, 4.8, 1.6), center=(0.0, 0.0, 2.2)),
            ]
            return self._mesh_from_triangles(asset_id=asset_id, triangles=triangles)
        if token == "tower":
            triangles = [
                *_box_triangles(role="body", size=(0.9, 0.9, 4.8), center=(0.0, 0.0, 2.4)),
                *_box_triangles(role="roof", size=(2.1, 1.8, 0.9), center=(0.0, 0.0, 5.3)),
            ]
            return self._mesh_from_triangles(asset_id=asset_id, triangles=triangles)
        if token == "spatial_tent":
            triangles = [
                *_roof_prism_triangles(role="body", size=(2.4, 2.8, 1.7), center=(0.0, 0.0, 0.85)),
                *_box_triangles(role="accent", size=(0.16, 2.6, 0.22), center=(0.0, 0.0, 0.11)),
                *_box_triangles(role="accent", size=(1.8, 0.16, 0.14), center=(0.0, 1.16, 0.07)),
                *_box_triangles(role="accent", size=(1.8, 0.16, 0.14), center=(0.0, -1.16, 0.07)),
            ]
            return self._mesh_from_triangles(asset_id=asset_id, triangles=triangles)
        if token == "spatial_sheep":
            triangles = []
            flock_offsets = ((-0.62, -0.16), (0.58, 0.20))
            for offset_x, offset_y in flock_offsets:
                triangles.extend(
                    _box_triangles(
                        role="body",
                        size=(0.84, 1.18, 0.58),
                        center=(offset_x, offset_y, 0.36),
                    )
                )
                triangles.extend(
                    _box_triangles(
                        role="accent",
                        size=(0.30, 0.30, 0.24),
                        center=(offset_x + 0.46, offset_y + 0.10, 0.42),
                    )
                )
                triangles.extend(
                    _box_triangles(
                        role="accent",
                        size=(0.10, 0.10, 0.26),
                        center=(offset_x - 0.20, offset_y - 0.18, 0.05),
                    )
                )
                triangles.extend(
                    _box_triangles(
                        role="accent",
                        size=(0.10, 0.10, 0.26),
                        center=(offset_x + 0.20, offset_y - 0.18, 0.05),
                    )
                )
            return self._mesh_from_triangles(asset_id=asset_id, triangles=triangles)
        if token == "road_paved":
            triangles = [
                *_box_triangles(role="body", size=(1.00, 1.00, 0.06), center=(0.0, 0.0, 0.0)),
                *_box_triangles(role="accent", size=(0.08, 0.56, 0.012), center=(0.0, 0.0, 0.04)),
            ]
            return self._mesh_from_triangles(asset_id=asset_id, triangles=triangles)
        if token == "road_dirt":
            triangles = [
                *_box_triangles(role="body", size=(1.00, 1.00, 0.05), center=(0.0, 0.0, 0.0)),
                *_box_triangles(role="accent", size=(0.82, 0.82, 0.010), center=(0.0, 0.0, 0.03)),
            ]
            return self._mesh_from_triangles(asset_id=asset_id, triangles=triangles)
        if token == "lake_patch":
            triangles = [
                *_disc_triangles(role="body", radius=1.0, steps=24, z=0.0),
                *_disc_triangles(role="accent", radius=0.74, steps=20, z=0.02),
            ]
            return self._mesh_from_triangles(asset_id=asset_id, triangles=triangles)
        if token == "hill_mound":
            triangles = [
                *_pyramid_triangles(role="body", size=(2.6, 2.4, 1.6), center=(0.0, 0.0, 0.8)),
                *_pyramid_triangles(role="accent", size=(1.6, 1.4, 0.9), center=(0.0, 0.0, 1.2)),
            ]
            return self._mesh_from_triangles(asset_id=asset_id, triangles=triangles)
        if token == "rock_cluster":
            triangles = [
                *_box_triangles(role="body", size=(0.72, 0.84, 0.52), center=(-0.22, -0.08, 0.24)),
                *_box_triangles(role="body", size=(0.56, 0.52, 0.46), center=(0.28, 0.14, 0.22)),
                *_pyramid_triangles(
                    role="accent", size=(0.92, 0.76, 0.54), center=(0.0, 0.0, 0.34)
                ),
            ]
            return self._mesh_from_triangles(asset_id=asset_id, triangles=triangles)
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
            return self._mesh_from_triangles(asset_id=asset_id, triangles=triangles)
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
            return self._mesh_from_triangles(asset_id=asset_id, triangles=triangles)
        if token == "auditory_gate_circle":
            return self._mesh_from_triangles(
                asset_id=asset_id, triangles=_auditory_gate_mesh_triangles("CIRCLE")
            )
        if token == "auditory_gate_triangle":
            return self._mesh_from_triangles(
                asset_id=asset_id, triangles=_auditory_gate_mesh_triangles("TRIANGLE")
            )
        if token == "auditory_gate_square":
            return self._mesh_from_triangles(
                asset_id=asset_id, triangles=_auditory_gate_mesh_triangles("SQUARE")
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


def _scene_screen_bounds(
    *,
    rect: pygame.Rect,
    window_height: int,
) -> tuple[float, float, float, float]:
    top = float(int(window_height) - int(rect.bottom))
    return (
        float(rect.x),
        float(top),
        float(rect.x + rect.w),
        float(top + rect.h),
    )


def _world_point_to_camera_space(
    *,
    camera: _SceneCamera,
    point: Point3,
) -> Point3:
    return world_to_camera_space(
        cam_world_x=float(camera.position[0]),
        cam_world_y=float(camera.position[1]),
        cam_world_z=float(camera.position[2]),
        heading_deg=float(camera.heading_deg),
        pitch_deg=float(camera.pitch_deg),
        target_world_x=float(point[0]),
        target_world_y=float(point[1]),
        target_world_z=float(point[2]),
    )


def _camera_point_inside_frustum(
    *,
    camera: _SceneCamera,
    point: Point3,
    tolerance: float = 1e-6,
) -> bool:
    cam_x, cam_y, cam_z = point
    if not all(math.isfinite(value) for value in point):
        return False
    tan_h = max(1e-4, math.tan(math.radians(float(camera.h_fov_deg) * 0.5)))
    tan_v = max(1e-4, math.tan(math.radians(float(camera.v_fov_deg) * 0.5)))
    tol = float(tolerance)
    return (
        float(cam_y) >= (float(camera.near_clip) - tol)
        and float(cam_y) <= (float(camera.far_clip) + tol)
        and abs(float(cam_x)) <= ((float(cam_y) * tan_h) + tol)
        and abs(float(cam_z)) <= ((float(cam_y) * tan_v) + tol)
    )


def _interpolate_point3(a: Point3, b: Point3, t: float) -> Point3:
    return (
        float(a[0] + ((b[0] - a[0]) * t)),
        float(a[1] + ((b[1] - a[1]) * t)),
        float(a[2] + ((b[2] - a[2]) * t)),
    )


def _clip_polygon_against_plane(
    polygon: tuple[Point3, ...],
    *,
    plane_value,
) -> tuple[Point3, ...]:
    if not polygon:
        return ()
    clipped: list[Point3] = []
    prev = polygon[-1]
    prev_value = float(plane_value(prev))
    prev_inside = prev_value >= 0.0
    for current in polygon:
        curr_value = float(plane_value(current))
        curr_inside = curr_value >= 0.0
        if curr_inside != prev_inside:
            denom = prev_value - curr_value
            if abs(denom) > 1e-9:
                t = max(0.0, min(1.0, prev_value / denom))
                clipped.append(_interpolate_point3(prev, current, t))
        if curr_inside:
            clipped.append(current)
        prev = current
        prev_value = curr_value
        prev_inside = curr_inside
    return tuple(clipped)


def _clip_camera_polygon_to_frustum(
    *,
    camera: _SceneCamera,
    polygon: tuple[Point3, ...],
) -> tuple[Point3, ...]:
    if not polygon:
        return ()
    tan_h = max(1e-4, math.tan(math.radians(float(camera.h_fov_deg) * 0.5)))
    tan_v = max(1e-4, math.tan(math.radians(float(camera.v_fov_deg) * 0.5)))
    clipped = tuple(polygon)
    for plane in (
        lambda point: float(point[1]) - float(camera.near_clip),
        lambda point: float(camera.far_clip) - float(point[1]),
        lambda point: float(point[0]) + (float(point[1]) * tan_h),
        lambda point: (float(point[1]) * tan_h) - float(point[0]),
        lambda point: float(point[2]) + (float(point[1]) * tan_v),
        lambda point: (float(point[1]) * tan_v) - float(point[2]),
    ):
        clipped = _clip_polygon_against_plane(clipped, plane_value=plane)
        if len(clipped) < 3:
            return ()
    return clipped


def _project_clipped_camera_polygon(
    *,
    rect: pygame.Rect,
    window_height: int,
    camera: _SceneCamera,
    polygon: tuple[Point3, ...],
    tolerance_px: float = 1e-3,
) -> tuple[tuple[float, float, float], ...] | None:
    if len(polygon) < 3:
        return None
    bounds = _scene_screen_bounds(rect=rect, window_height=window_height)
    projected: list[tuple[float, float, float]] = []
    for point in polygon:
        cam_x, cam_y, cam_z = point
        screen_x, screen_y, _on_screen, in_front = camera_space_to_viewport(
            cam_x=float(cam_x),
            cam_y=float(cam_y),
            cam_z=float(cam_z),
            size=(max(1, int(rect.w)), max(1, int(rect.h))),
            h_fov_deg=float(camera.h_fov_deg),
            v_fov_deg=float(camera.v_fov_deg),
        )
        if not in_front or not all(math.isfinite(value) for value in (screen_x, screen_y)):
            return None
        proj_x, proj_y = _scene_local_top_left_to_screen(
            rect=rect,
            window_height=window_height,
            x=float(screen_x),
            y=float(screen_y),
        )
        if not all(math.isfinite(value) for value in (proj_x, proj_y, cam_y)):
            return None
        if (
            proj_x < (bounds[0] - float(tolerance_px))
            or proj_x > (bounds[2] + float(tolerance_px))
            or proj_y < (bounds[1] - float(tolerance_px))
            or proj_y > (bounds[3] + float(tolerance_px))
        ):
            return None
        projected.append(
            (
                max(bounds[0], min(bounds[2], float(proj_x))),
                max(bounds[1], min(bounds[3], float(proj_y))),
                float(cam_y),
            )
        )
    return tuple(projected)


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
            ),
        )
        for face in projected
    )


def _aircraft_marker_primitives(
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
) -> tuple[_ProjectedOverlayPrimitive, ...]:
    palette = build_pygame_palette(
        body_color=tuple(
            max(0, min(255, int(round(channel * 255.0)))) for channel in color[:3]
        ),
        outline_color=tuple(
            max(0, min(255, int(round(channel * 255.0)))) for channel in outline[:3]
        ),
    )
    role_colors = {
        "body": palette.body,
        "accent": palette.accent,
        "canopy": palette.canopy,
        "engine": palette.engine,
    }
    projected_faces = _project_aircraft_marker_polygons(
        rect=rect,
        window_height=window_height,
        center_top_left=center_top_left,
        heading_deg=heading_deg,
        size=size,
        color=color,
        outline=outline,
        pitch_deg=pitch_deg,
        bank_deg=bank_deg,
        view_yaw_deg=view_yaw_deg,
        view_pitch_deg=view_pitch_deg,
    )
    primitives: list[_ProjectedOverlayPrimitive] = []
    for face in projected_faces:
        base = role_colors.get(face.role, palette.body)
        fill = tuple(max(0, min(255, int(round(channel * float(face.shade))))) for channel in base)
        fill_color = (
            fill[0] / 255.0,
            fill[1] / 255.0,
            fill[2] / 255.0,
            float(color[3]),
        )
        primitives.append(
            _ProjectedOverlayPrimitive(
                kind="polygon",
                points=tuple((float(px), float(py)) for px, py in face.points),
                color=fill_color,
                filled=True,
            )
        )
        primitives.append(
            _ProjectedOverlayPrimitive(
                kind="polygon",
                points=tuple((float(px), float(py)) for px, py in face.points),
                color=tuple(float(channel) for channel in outline),
                filled=False,
                width=1.0,
            )
        )
    return tuple(primitives)


def _trace_test_1_marker_primitives(
    *,
    rect: pygame.Rect,
    window_height: int,
    payload,
) -> tuple[_ProjectedOverlayPrimitive, ...]:
    vw = max(1, int(rect.w))
    vh = max(1, int(rect.h))
    target_pose, blue_poses = trace_test_1_aircraft_screen_poses_for_payload(
        payload,
        size=(vw, vh),
    )
    blue_colors = (
        (0.34, 0.52, 0.90, 0.96),
        (0.38, 0.60, 0.94, 0.96),
        (0.42, 0.66, 0.96, 0.96),
        (0.30, 0.46, 0.82, 0.96),
    )
    blue_outlines = (
        (0.92, 0.96, 1.0, 0.88),
        (0.90, 0.96, 1.0, 0.86),
        (0.88, 0.94, 0.98, 0.84),
        (0.84, 0.92, 0.98, 0.84),
    )
    primitives: list[_ProjectedOverlayPrimitive] = []
    for idx, (blue_frame, blue_pose) in enumerate(
        zip(payload.scene.blue_frames, blue_poses, strict=True)
    ):
        blue_center, blue_scale = trace_test_1_project_scene_position(
            blue_frame.position,
            size=(vw, vh),
        )
        primitives.extend(
            _aircraft_marker_primitives(
                rect=rect,
                window_height=window_height,
                center_top_left=blue_center,
                heading_deg=float(blue_pose[0]),
                size=max(8.6, blue_scale * 10.2),
                color=blue_colors[idx % len(blue_colors)],
                outline=blue_outlines[idx % len(blue_outlines)],
                pitch_deg=float(blue_pose[1]),
                bank_deg=float(blue_pose[2]),
                view_pitch_deg=0.0,
            )
        )

    target_center, target_scale = trace_test_1_project_scene_position(
        payload.scene.red_frame.position,
        size=(vw, vh),
    )
    primitives.extend(
        _aircraft_marker_primitives(
            rect=rect,
            window_height=window_height,
            center_top_left=target_center,
            heading_deg=float(target_pose[0]),
            size=max(11.6, target_scale * 13.0),
            color=(0.92, 0.24, 0.24, 0.98),
            outline=(1.0, 0.92, 0.92, 0.90),
            pitch_deg=float(target_pose[1]),
            bank_deg=float(target_pose[2]),
            view_pitch_deg=0.0,
        )
    )
    return tuple(primitives)


def _trace_test_2_marker_primitives(
    *,
    rect: pygame.Rect,
    window_height: int,
    payload,
) -> tuple[_ProjectedOverlayPrimitive, ...]:
    vw = max(1, int(rect.w))
    vh = max(1, int(rect.h))
    progress = float(payload.observe_progress)
    primitives: list[_ProjectedOverlayPrimitive] = []
    for track in payload.aircraft:
        center = trace_test_2_project_point(
            trace_test_2_track_position(track=track, progress=progress),
            size=(vw, vh),
        )
        pose = trace_test_2_aircraft_screen_pose_for_track(
            track=track,
            progress=progress,
            size=(vw, vh),
        )
        primitives.extend(
            _aircraft_marker_primitives(
                rect=rect,
                window_height=window_height,
                center_top_left=center,
                heading_deg=float(pose[0]),
                size=max(10.0, 14.0 - (abs(float(pose[1])) * 0.05)),
                color=(
                    track.color_rgb[0] / 255.0,
                    track.color_rgb[1] / 255.0,
                    track.color_rgb[2] / 255.0,
                    0.96,
                ),
                outline=(0.96, 0.98, 1.0, 0.88),
                pitch_deg=float(pose[1]),
                bank_deg=float(pose[2]),
                view_pitch_deg=22.0,
            )
        )
    return tuple(primitives)


def _rapid_tracking_target_asset_id(target_kind: str, variant: str) -> str:
    token = str(target_kind).strip().lower()
    if token == "building":
        return "building_tower" if str(variant).strip().lower() == "tower" else "building_hangar"
    if token == "truck":
        return (
            "vehicle_tracked"
            if str(variant).strip().lower() in {"tracked", "armor", "armored"}
            else "truck_olive"
        )
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
    if payload is None:
        travel_distance = run_travel_distance(session_seed=0, phase_elapsed_s=0.0)
    else:
        travel_distance = float(
            getattr(
                payload,
                "presentation_travel_distance",
                run_travel_distance(
                    session_seed=int(payload.session_seed),
                    phase_elapsed_s=float(payload.phase_elapsed_s),
                ),
            )
        )
    ball_distance = travel_distance + AUDITORY_BALL_ANCHOR_DISTANCE
    local_x = 0.0
    local_z = 0.0
    ball_color = (0.94, 0.96, 1.0, 0.98)
    if payload is not None:
        x_half_span = max(0.08, float(payload.tube_half_width))
        z_half_span = max(0.08, float(payload.tube_half_height))
        local_x = max(-1.0, min(1.0, float(payload.ball_x) / x_half_span)) * (
            _AUDITORY_TUBE_RX * 0.72
        )
        local_z = max(-1.0, min(1.0, float(payload.ball_y) / z_half_span)) * (
            _AUDITORY_TUBE_RZ * 0.72
        )
        if float(payload.ball_contact_ratio) >= 1.0:
            ball_color = (0.95, 0.35, 0.38, 0.98)
        else:
            ball_color = {
                "RED": (0.92, 0.34, 0.38, 0.98),
                "GREEN": (0.34, 0.88, 0.56, 0.98),
                "BLUE": (0.36, 0.62, 0.94, 0.98),
                "YELLOW": (0.94, 0.82, 0.34, 0.98),
            }.get(str(payload.ball_visual_color).upper(), (0.94, 0.96, 1.0, 0.98))
    cam_pos, look_target = fixed_camera_pose_at_distance(ball_distance)
    camera = _look_at_camera(
        position=cam_pos,
        target=look_target,
        h_fov_deg=float(TUNNEL_CAMERA_H_FOV_DEG),
        v_fov_deg=float(TUNNEL_CAMERA_V_FOV_DEG),
        far_clip=420.0,
    )

    instances: list[_AssetInstance] = []
    geometry_start = travel_distance + _AUDITORY_TUBE_GEOMETRY_START
    geometry_end = travel_distance + _AUDITORY_TUBE_GEOMETRY_END

    distance = geometry_start + (_AUDITORY_TUBE_SEGMENT_LENGTH * 0.5)
    while distance <= geometry_end:
        center, tangent, _right, _up = _tube_frame(distance)
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
        center, tangent, _right, _up = _tube_frame(distance)
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

    ball_center, _ball_tangent, right, up = _tube_frame(ball_distance)
    gate_asset_ids: set[str] = set()
    if payload is not None:
        ball_center = _vec_add(
            ball_center, _vec_add(_vec_scale(right, local_x), _vec_scale(up, local_z))
        )
    instances.append(
        _AssetInstance(
            asset_id="auditory_ball",
            position=ball_center,
            hpr_deg=((travel_distance * 9.0) % 360.0, 0.0, 0.0),
            scale=(0.11, 0.11, 0.11),
            color=ball_color,
        )
    )

    if payload is not None:
        y_half_span = max(0.08, float(payload.tube_half_height))
        visible_gates: list[tuple[float, object]] = []
        for gate in payload.gates:
            distance = gate_distance_from_x_norm(
                float(gate.x_norm),
                travel_distance=travel_distance,
                spawn_x_norm=float(AUDITORY_GATE_SPAWN_X_NORM),
                player_x_norm=float(AUDITORY_GATE_PLAYER_X_NORM),
                retire_x_norm=float(AUDITORY_GATE_RETIRE_X_NORM),
            )
            if distance < (travel_distance + AUDITORY_GATE_BEHIND_DISTANCE):
                continue
            if distance > (travel_distance + _AUDITORY_TUBE_GEOMETRY_END + 0.5):
                continue
            visible_gates.append((distance, gate))
        visible_gates.sort(key=lambda item: float(item[0]), reverse=True)
        for distance, gate in visible_gates[:14]:
            center, tangent, right, up = _tube_frame(distance)
            depth_t = gate_depth_ratio_from_distance(distance, travel_distance=travel_distance)
            local_z = max(-1.0, min(1.0, float(gate.y_norm) / y_half_span)) * (
                _AUDITORY_TUBE_RZ * 0.62
            )
            radius = max(
                0.16, (float(gate.aperture_norm) / y_half_span) * (_AUDITORY_TUBE_RZ * 0.82)
            )
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
                        base=_auditory_gate_display_rgba(gate),
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
        seed=int(payload.scene_seed),
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
    static_scene = _rapid_tracking_static_scene(int(payload.scene_seed))
    static_bundle = _rapid_tracking_static_bundle(int(payload.scene_seed))

    target_world_z = estimated_target_world_z(
        kind=str(payload.target_kind),
        target_world_x=float(payload.target_world_x),
        target_world_y=float(payload.target_world_y),
        elapsed_s=float(payload.phase_elapsed_s),
        scene_progress=float(payload.scene_progress),
        seed=int(payload.scene_seed),
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
    target_scale = (scale_factor, scale_factor, scale_factor)
    if target_asset_id in {
        "building_hangar",
        "building_tower",
        "truck_olive",
        "vehicle_tracked",
        "soldiers_patrol",
    }:
        target_world_z = _grounded_asset_world_z(
            asset_id=target_asset_id,
            terrain_z=target_ground_z,
            scale=target_scale,
        )

    focus_x = float(payload.focus_world_x)
    focus_y = float(payload.focus_world_y)
    camera_x = float(camera.position[0])
    camera_y = float(camera.position[1])
    readable_scene = int(payload.scene_seed) != int(payload.session_seed)
    visible_ambient = sorted(
        (
            instance
            for instance in static_scene.ambient_instances
            if _rapid_tracking_static_instance_visible(
                instance,
                focus_x=focus_x,
                focus_y=focus_y,
                camera_x=camera_x,
                camera_y=camera_y,
            )
        ),
        key=lambda instance: _rapid_tracking_visibility_rank(
            instance,
            focus_x=focus_x,
            focus_y=focus_y,
            camera_x=camera_x,
            camera_y=camera_y,
        ),
    )
    ambient_limit = 4 if readable_scene else 6
    filtered_ambient: list[_AssetInstance] = []
    forest_canopy_count = 0
    for instance in visible_ambient:
        if instance.asset_id == "forest_canopy_patch":
            if forest_canopy_count >= 1:
                continue
            forest_canopy_count += 1
        filtered_ambient.append(instance)
        if len(filtered_ambient) >= ambient_limit:
            break
    visible_ambient = filtered_ambient

    instances = [*visible_ambient]
    instances.append(
        _AssetInstance(
            asset_id=target_asset_id,
            position=(
                float(payload.target_world_x),
                float(payload.target_world_y),
                float(target_world_z),
            ),
            hpr_deg=(target_heading, 0.0, 0.0),
            scale=target_scale,
        )
    )

    backdrop_groups = tuple(group for group in static_bundle.groups if group.layer == "far")
    playfield_groups = tuple(group for group in static_bundle.groups if group.layer != "far")

    unique_asset_ids = tuple(
        sorted(
            {
                *static_bundle.asset_ids,
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
        entity_count=int(static_bundle.instance_count + len(instances)),
        static_groups=static_bundle.groups,
        backdrop_groups=backdrop_groups,
        playfield_groups=playfield_groups,
    )


def _rapid_tracking_backdrop_plan(
    *,
    viewport_size: tuple[int, int],
    payload: object | None,
) -> _RapidTrackingBackdropPlan:
    vh = max(1, int(viewport_size[1]))
    horizon_ratio = 0.46
    if payload is not None:
        rig = rapid_tracking_camera_rig_state(
            elapsed_s=float(getattr(payload, "phase_elapsed_s", 0.0)),
            seed=int(getattr(payload, "scene_seed", 0)),
            progress=float(getattr(payload, "scene_progress", 0.0)),
            camera_yaw_deg=float(getattr(payload, "camera_yaw_deg", 0.0)),
            camera_pitch_deg=float(getattr(payload, "camera_pitch_deg", 0.0)),
            zoom=float(getattr(payload, "capture_zoom", 0.0)),
            target_kind=str(getattr(payload, "target_kind", "soldier")),
            target_world_x=float(getattr(payload, "target_world_x", 0.0)),
            target_world_y=float(getattr(payload, "target_world_y", 0.0)),
            focus_world_x=float(getattr(payload, "focus_world_x", 0.0)),
            focus_world_y=float(getattr(payload, "focus_world_y", 70.0)),
            turbulence_strength=float(getattr(payload, "turbulence_strength", 0.0)),
        )
        clamped_pitch = max(-82.0, min(82.0, float(rig.view_pitch_deg)))
        pitch_t = (clamped_pitch + 82.0) / 164.0
        horizon_ratio = _lerp(0.76, 0.08, pitch_t)
        if clamped_pitch >= 68.0:
            # Let the skyline move fully off-screen when the player looks almost straight up.
            horizon_ratio = _lerp(horizon_ratio, -0.08, (clamped_pitch - 68.0) / 14.0)
        elif clamped_pitch <= -68.0:
            horizon_ratio = _lerp(horizon_ratio, 0.82, (-68.0 - clamped_pitch) / 14.0)
    horizon_y = max(-(vh * 0.12), min(vh * 0.82, vh * horizon_ratio))
    return _RapidTrackingBackdropPlan(
        horizon_y=float(horizon_y),
        sky_top_rgb=(0.56, 0.68, 0.82),
        sky_bottom_rgb=(0.22, 0.38, 0.46),
        ground_horizon_rgb=(0.34, 0.42, 0.26),
        ground_base_rgb=(0.56, 0.64, 0.38),
    )


def _rapid_tracking_layout_world_xy(
    layout: RapidTrackingCompoundLayout,
    *,
    track_x: float,
    track_y: float,
) -> tuple[float, float]:
    return rapid_tracking_track_to_world_xy(
        track_x=float(track_x),
        track_y=float(track_y),
        path_lateral_bias=float(layout.path_lateral_bias),
    )


def _rapid_tracking_ground_clearance(asset_id: str) -> float:
    token = str(asset_id)
    if token.startswith("building_"):
        return 0.0
    if token.startswith("road_"):
        return 0.01
    if token == "terrain_lake_patch":
        return -0.02
    if token == "terrain_rock_cluster":
        return 0.0
    if token == "soldiers_patrol":
        return 0.01
    if token in {"truck_olive", "vehicle_tracked"}:
        return 0.02
    if token in {
        "forest_canopy_patch",
        "trees_field_cluster",
        "trees_pine_cluster",
        "shrubs_low_cluster",
    }:
        return 0.0
    return 0.0


def _asset_axis_extent(*, asset_id: str, axis: int) -> float:
    mesh = _rapid_tracking_static_asset_library().mesh(asset_id)
    return max(1e-6, float(mesh.bounds[1][axis]) - float(mesh.bounds[0][axis]))


def _grounded_asset_world_z(
    *,
    asset_id: str,
    terrain_z: float,
    scale: Point3 = (1.0, 1.0, 1.0),
    clearance: float | None = None,
) -> float:
    mesh = _rapid_tracking_static_asset_library().mesh(asset_id)
    z_offset = float(mesh.base_z) * float(scale[2])
    lift = _rapid_tracking_ground_clearance(asset_id) if clearance is None else float(clearance)
    return float(terrain_z + lift - z_offset)


def _rapid_tracking_local_hill_scale(
    *,
    radius_x_world: float,
    radius_y_world: float,
    height: float,
) -> Point3:
    mesh_extent_x = _asset_axis_extent(asset_id="terrain_hill_mound", axis=0)
    mesh_extent_y = _asset_axis_extent(asset_id="terrain_hill_mound", axis=1)
    scale_x = min(48.0 / mesh_extent_x, max(5.5 / mesh_extent_x, float(radius_x_world)))
    scale_y = min(44.0 / mesh_extent_y, max(5.5 / mesh_extent_y, float(radius_y_world)))
    scale_z = min(4.2, max(2.0, float(height) * 4.0))
    return float(scale_x), float(scale_y), float(scale_z)


def _rapid_tracking_anchor_world_pos(
    layout: RapidTrackingCompoundLayout,
    *,
    anchor,
    asset_id: str | None = None,
    scale: Point3 = (1.0, 1.0, 1.0),
    y_bias: float = 0.0,
    clearance: float = 0.0,
) -> tuple[float, float, float]:
    wx, wy = _rapid_tracking_layout_world_xy(
        layout, track_x=float(anchor.x), track_y=float(anchor.y)
    )
    wy += float(y_bias)
    terrain_z = float(rapid_tracking_terrain_height(float(wx), float(wy)))
    if asset_id is None:
        wz = terrain_z + float(clearance)
    else:
        wz = _grounded_asset_world_z(
            asset_id=asset_id,
            terrain_z=terrain_z,
            scale=scale,
            clearance=clearance,
        )
    return float(wx), float(wy), float(wz)


def _rapid_tracking_road_piece_instance(
    layout: RapidTrackingCompoundLayout,
    *,
    asset_id: str,
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    width: float,
) -> _AssetInstance:
    start_wx, start_wy = _rapid_tracking_layout_world_xy(
        layout, track_x=float(start_xy[0]), track_y=float(start_xy[1])
    )
    end_wx, end_wy = _rapid_tracking_layout_world_xy(
        layout, track_x=float(end_xy[0]), track_y=float(end_xy[1])
    )
    center_x = (start_wx + end_wx) * 0.5
    center_y = (start_wy + end_wy) * 0.5
    start_z = float(rapid_tracking_terrain_height(float(start_wx), float(start_wy)))
    mid_z = float(rapid_tracking_terrain_height(float(center_x), float(center_y)))
    end_z = float(rapid_tracking_terrain_height(float(end_wx), float(end_wy)))
    length = max(1.0, math.hypot(float(end_wx - start_wx), float(end_wy - start_wy)))
    heading = _wrap_heading_deg(float(end_wx - start_wx), float(end_wy - start_wy))
    pitch = -math.degrees(math.atan2(end_z - start_z, max(1e-6, length)))
    scale = (float(width), float(length) + 2.0, 1.0)
    return _AssetInstance(
        asset_id=asset_id,
        position=(
            center_x,
            center_y,
            _grounded_asset_world_z(
                asset_id=asset_id,
                terrain_z=(start_z + mid_z + end_z) / 3.0,
                scale=scale,
            ),
        ),
        hpr_deg=(heading, pitch, 0.0),
        scale=scale,
    )


def _rapid_tracking_render_polyline(
    layout: RapidTrackingCompoundLayout,
    points: tuple[tuple[float, float], ...],
    *,
    max_world_piece_length: float = 24.0,
) -> tuple[tuple[float, float], ...]:
    if len(points) <= 1:
        return tuple(points)
    sampled: list[tuple[float, float]] = [tuple(points[0])]
    for start_xy, end_xy in zip(points, points[1:], strict=False):
        start_wx, start_wy = _rapid_tracking_layout_world_xy(
            layout,
            track_x=float(start_xy[0]),
            track_y=float(start_xy[1]),
        )
        end_wx, end_wy = _rapid_tracking_layout_world_xy(
            layout,
            track_x=float(end_xy[0]),
            track_y=float(end_xy[1]),
        )
        distance = math.hypot(float(end_wx - start_wx), float(end_wy - start_wy))
        steps = max(1, int(math.ceil(distance / max(1.0, float(max_world_piece_length)))))
        for step_idx in range(1, steps + 1):
            t = float(step_idx) / float(steps)
            point = (
                _lerp(float(start_xy[0]), float(end_xy[0]), t),
                _lerp(float(start_xy[1]), float(end_xy[1]), t),
            )
            if _distance_sq(sampled[-1][0], sampled[-1][1], point[0], point[1]) > 1e-9:
                sampled.append(point)
    return tuple((float(x), float(y)) for x, y in sampled)


def _rapid_tracking_layout_heading(layout: RapidTrackingCompoundLayout, token: str) -> float:
    return float(rapid_tracking_seed_unit(seed=int(layout.seed), salt=str(token)) * 360.0)


def _rapid_tracking_cluster_instance(
    layout: RapidTrackingCompoundLayout,
    *,
    cluster,
    asset_id: str | None = None,
) -> _AssetInstance:
    wx, wy = _rapid_tracking_layout_world_xy(
        layout, track_x=float(cluster.x), track_y=float(cluster.y)
    )
    wz = float(rapid_tracking_terrain_height(float(wx), float(wy)))
    resolved_asset_id = str(asset_id or cluster.asset_id)
    scale = (
        (
            3.0
            if resolved_asset_id == "forest_canopy_patch"
            else 2.6
            if "tree" in resolved_asset_id
            else 3.2
        )
        * float(cluster.scale)
        * (1.0 + (0.04 * max(0, int(cluster.count) - 1)))
    )
    return _AssetInstance(
        asset_id=resolved_asset_id,
        position=(
            float(wx),
            float(wy),
            _grounded_asset_world_z(
                asset_id=resolved_asset_id,
                terrain_z=float(wz),
                scale=(scale, scale, scale),
            ),
        ),
        hpr_deg=(_rapid_tracking_layout_heading(layout, f"{cluster.cluster_id}:heading"), 0.0, 0.0),
        scale=(scale, scale, scale),
    )


def _rapid_tracking_visibility_rank(
    instance: _AssetInstance,
    *,
    focus_x: float,
    focus_y: float,
    camera_x: float,
    camera_y: float,
) -> float:
    px, py, _pz = instance.position
    return min(
        ((float(px) - float(focus_x)) ** 2) + ((float(py) - float(focus_y)) ** 2),
        ((float(px) - float(camera_x)) ** 2) + ((float(py) - float(camera_y)) ** 2),
    )


def _rapid_tracking_static_instance_visible(
    instance: _AssetInstance,
    *,
    focus_x: float,
    focus_y: float,
    camera_x: float,
    camera_y: float,
) -> bool:
    limit = 420.0 if instance.asset_id in {"plane_blue", "helicopter_green"} else 290.0
    return _rapid_tracking_visibility_rank(
        instance,
        focus_x=focus_x,
        focus_y=focus_y,
        camera_x=camera_x,
        camera_y=camera_y,
    ) <= (limit * limit)


@lru_cache(maxsize=24)
def _rapid_tracking_static_scene(seed: int) -> _RapidTrackingStaticScene:
    layout = build_rapid_tracking_compound_layout(seed=int(seed))
    core_instances: list[_AssetInstance] = []
    ambient_instances: list[_AssetInstance] = []

    for terrain in build_distant_terrain_ring(layout=layout):
        terrain_z = float(
            rapid_tracking_terrain_height(float(terrain.world_x), float(terrain.world_y))
        )
        core_instances.append(
            _AssetInstance(
                asset_id="terrain_hill_mound",
                position=(
                    float(terrain.world_x),
                    float(terrain.world_y),
                    _grounded_asset_world_z(
                        asset_id="terrain_hill_mound",
                        terrain_z=terrain_z,
                        scale=(
                            float(terrain.scale_x),
                            float(terrain.scale_y),
                            float(terrain.scale_z),
                        ),
                    ),
                ),
                hpr_deg=(float(terrain.heading_deg), 0.0, 0.0),
                scale=(
                    float(terrain.scale_x),
                    float(terrain.scale_y),
                    float(terrain.scale_z),
                ),
                bucket="backdrop",
            )
        )

    for obstacle in layout.obstacles:
        wx, wy = _rapid_tracking_layout_world_xy(
            layout, track_x=float(obstacle.x), track_y=float(obstacle.y)
        )
        wz = float(rapid_tracking_terrain_height(float(wx), float(wy)))
        scale_x = abs(
            _rapid_tracking_layout_world_xy(
                layout, track_x=float(obstacle.x + obstacle.radius_x), track_y=float(obstacle.y)
            )[0]
            - wx
        )
        scale_y = abs(
            _rapid_tracking_layout_world_xy(
                layout, track_x=float(obstacle.x), track_y=float(obstacle.y + obstacle.radius_y)
            )[1]
            - wy
        )
        if obstacle.kind == "lake":
            scale = (max(7.0, scale_x), max(7.0, scale_y), 1.0)
            core_instances.append(
                _AssetInstance(
                    asset_id="terrain_lake_patch",
                    position=(
                        float(wx),
                        float(wy),
                        _grounded_asset_world_z(
                            asset_id="terrain_lake_patch",
                            terrain_z=float(wz),
                            scale=scale,
                        ),
                    ),
                    hpr_deg=(float(obstacle.rotation_deg), 0.0, 0.0),
                    scale=scale,
                )
            )
        elif obstacle.kind == "hill":
            scale = _rapid_tracking_local_hill_scale(
                radius_x_world=float(scale_x),
                radius_y_world=float(scale_y),
                height=float(obstacle.height),
            )
            core_instances.append(
                _AssetInstance(
                    asset_id="terrain_hill_mound",
                    position=(
                        float(wx),
                        float(wy),
                        _grounded_asset_world_z(
                            asset_id="terrain_hill_mound",
                            terrain_z=float(wz),
                            scale=scale,
                        ),
                    ),
                    hpr_deg=(float(obstacle.rotation_deg), 0.0, 0.0),
                    scale=scale,
                )
            )
        else:
            scale = (
                max(1.6, scale_x * 0.88),
                max(1.6, scale_y * 0.88),
                max(1.0, float(obstacle.height) * 2.2),
            )
            core_instances.append(
                _AssetInstance(
                    asset_id="terrain_rock_cluster",
                    position=(
                        float(wx),
                        float(wy),
                        _grounded_asset_world_z(
                            asset_id="terrain_rock_cluster",
                            terrain_z=float(wz),
                            scale=scale,
                        ),
                    ),
                    hpr_deg=(float(obstacle.rotation_deg), 0.0, 0.0),
                    scale=scale,
                )
            )

    for road_segment in layout.road_segments:
        road_asset = (
            "road_paved_segment" if str(road_segment.surface) == "paved" else "road_dirt_segment"
        )
        road_width = 8.6 if road_asset == "road_paved_segment" else 6.6
        render_points = _rapid_tracking_render_polyline(
            layout,
            road_segment.points,
        )
        for start_xy, end_xy in zip(render_points, render_points[1:], strict=False):
            core_instances.append(
                _rapid_tracking_road_piece_instance(
                    layout,
                    asset_id=road_asset,
                    start_xy=start_xy,
                    end_xy=end_xy,
                    width=road_width,
                )
            )

    for anchor in layout.building_anchors:
        asset_id = "building_tower" if str(anchor.variant) == "tower" else "building_hangar"
        scale = (
            3.7
            + (rapid_tracking_seed_unit(seed=int(layout.seed), salt=str(anchor.anchor_id)) * 0.8)
            if asset_id == "building_tower"
            else 6.5
            + (rapid_tracking_seed_unit(seed=int(layout.seed), salt=str(anchor.anchor_id)) * 1.1)
        )
        wx, wy, wz = _rapid_tracking_anchor_world_pos(
            layout,
            anchor=anchor,
            asset_id=asset_id,
            scale=(scale, scale, scale),
        )
        core_instances.append(
            _AssetInstance(
                asset_id=asset_id,
                position=(wx, wy, wz),
                hpr_deg=(
                    _rapid_tracking_layout_heading(layout, f"{anchor.anchor_id}:heading"),
                    0.0,
                    0.0,
                ),
                scale=(scale, scale, scale),
            )
        )

    for idx, base in enumerate(layout.bases):
        parked_anchor = _TrackPointAnchor(
            x=float(base.x + (0.26 if idx % 2 == 0 else -0.22)),
            y=float(base.y - (0.22 if idx % 2 == 0 else 0.18)),
        )
        asset_id = "vehicle_tracked" if idx == 1 else "truck_olive"
        scale = 2.5 if asset_id == "vehicle_tracked" else 2.9
        wx, wy, wz = _rapid_tracking_anchor_world_pos(
            layout,
            anchor=parked_anchor,
            asset_id=asset_id,
            scale=(scale, scale, scale),
        )
        ambient_instances.append(
            _AssetInstance(
                asset_id=asset_id,
                position=(wx, wy, wz),
                hpr_deg=(_rapid_tracking_layout_heading(layout, f"base-vehicle:{idx}"), 0.0, 0.0),
                scale=(scale, scale, scale),
            )
        )

    road_sample = layout.road_anchors[:: max(1, len(layout.road_anchors) // 3 or 1)]
    for idx, anchor in enumerate(road_sample[:2]):
        wx, wy, wz = _rapid_tracking_anchor_world_pos(
            layout,
            anchor=anchor,
            asset_id="soldiers_patrol",
            scale=(1.9, 1.9, 1.9),
            y_bias=2.0,
        )
        ambient_instances.append(
            _AssetInstance(
                asset_id="soldiers_patrol",
                position=(wx, wy, wz),
                hpr_deg=(_rapid_tracking_layout_heading(layout, f"road-patrol:{idx}"), 0.0, 0.0),
                scale=(1.9, 1.9, 1.9),
            )
        )

    scenic_clusters = (
        *layout.forest_clusters[:2],
        *layout.tree_clusters[:2],
        *layout.shrub_clusters[:2],
    )
    for idx, cluster in enumerate(scenic_clusters):
        asset_id = (
            "trees_pine_cluster" if "tree" in str(cluster.asset_id) and idx % 2 == 1 else None
        )
        ambient_instances.append(
            _rapid_tracking_cluster_instance(
                layout,
                cluster=cluster,
                asset_id=asset_id,
            )
        )

    scenic_pines = [
        cluster
        for cluster in layout.scenic_clusters
        if cluster.asset_id in {"trees_pine_cluster", "trees_field_cluster"}
    ]
    for idx, cluster in enumerate(scenic_pines[:2]):
        ambient_instances.append(
            _rapid_tracking_cluster_instance(
                layout,
                cluster=cluster,
                asset_id="trees_pine_cluster" if idx % 2 == 0 else "trees_field_cluster",
            )
        )

    deduped_ambient: list[_AssetInstance] = []
    seen_tree_positions: set[tuple[str, int, int]] = set()
    for instance in ambient_instances:
        if instance.asset_id in {"trees_pine_cluster", "trees_field_cluster"}:
            key = (
                "trees",
                int(round(float(instance.position[0]) * 100.0)),
                int(round(float(instance.position[1]) * 100.0)),
            )
            if key in seen_tree_positions:
                continue
            seen_tree_positions.add(key)
        deduped_ambient.append(instance)
    ambient_instances = deduped_ambient

    if layout.helicopter_anchors:
        anchor = layout.helicopter_anchors[0]
        wx, wy, terrain = _rapid_tracking_anchor_world_pos(layout, anchor=anchor)
        ambient_instances.append(
            _AssetInstance(
                asset_id="helicopter_green",
                position=(wx, wy, terrain + 12.5 + float(layout.altitude_bias)),
                hpr_deg=(
                    _rapid_tracking_layout_heading(layout, f"{anchor.anchor_id}:air-heading"),
                    0.0,
                    0.0,
                ),
                scale=(2.2, 2.2, 2.2),
            )
        )
    if layout.jet_anchors:
        anchor = layout.jet_anchors[0]
        wx, wy, terrain = _rapid_tracking_anchor_world_pos(layout, anchor=anchor)
        ambient_instances.append(
            _AssetInstance(
                asset_id="plane_blue",
                position=(wx, wy, terrain + 20.0 + float(layout.altitude_bias)),
                hpr_deg=(
                    _rapid_tracking_layout_heading(layout, f"{anchor.anchor_id}:air-heading"),
                    0.0,
                    0.0,
                ),
                scale=(1.9, 1.9, 1.9),
            )
        )

    return _RapidTrackingStaticScene(
        layout=layout,
        core_instances=tuple(core_instances),
        ambient_instances=tuple(ambient_instances),
        asset_ids=tuple(
            sorted(
                {
                    "building_hangar",
                    "building_tower",
                    "forest_canopy_patch",
                    "helicopter_green",
                    "plane_blue",
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
                }
            )
        ),
    )


_RAPID_TRACKING_STATIC_GROUP_CELL_SIZE = 80.0
_RAPID_TRACKING_STATIC_GROUP_MAX_RADIUS = 80.0


@lru_cache(maxsize=1)
def _rapid_tracking_static_asset_library() -> _SceneAssetLibrary:
    return _SceneAssetLibrary(RenderAssetCatalog())


def _rapid_tracking_static_group_family(asset_id: str) -> str:
    token = str(asset_id)
    if token.startswith("road_"):
        return "roads"
    if token.startswith("building_"):
        return "buildings"
    if token.startswith("terrain_"):
        return "terrain"
    return "scenery"


def _rapid_tracking_static_group_layer(
    instance: _AssetInstance,
    *,
    center_x: float,
    center_y: float,
) -> str:
    _ = (center_x, center_y)
    if str(instance.bucket) == "backdrop":
        return "far"
    return "near"


def _rapid_tracking_static_group_key(
    instance: _AssetInstance,
    *,
    layer: str,
    index: int,
    center_x: float,
    center_y: float,
) -> str:
    if layer == "far":
        return f"far:{index:03d}:{instance.asset_id}"
    family = _rapid_tracking_static_group_family(instance.asset_id)
    cell_x = math.floor(
        (float(instance.position[0]) - float(center_x)) / _RAPID_TRACKING_STATIC_GROUP_CELL_SIZE
    )
    cell_y = math.floor(
        (float(instance.position[1]) - float(center_y)) / _RAPID_TRACKING_STATIC_GROUP_CELL_SIZE
    )
    return f"near:{family}:{cell_x}:{cell_y}"


def _rapid_tracking_static_group_visible(
    *,
    camera: _SceneCamera,
    group: _RapidTrackingStaticGroup,
) -> bool:
    cam_x, cam_y, cam_z = world_to_camera_space(
        cam_world_x=float(camera.position[0]),
        cam_world_y=float(camera.position[1]),
        cam_world_z=float(camera.position[2]),
        heading_deg=float(camera.heading_deg),
        pitch_deg=float(camera.pitch_deg),
        target_world_x=float(group.center[0]),
        target_world_y=float(group.center[1]),
        target_world_z=float(group.center[2]),
    )
    radius = max(1.0, float(group.radius))
    if cam_y <= (-radius) or (cam_y + radius) <= float(camera.near_clip):
        return False
    if (cam_y - radius) >= float(camera.far_clip):
        return False
    safe_depth = max(1.0, cam_y)
    angle_margin = math.degrees(math.atan2(radius, safe_depth))
    yaw = abs(math.degrees(math.atan2(cam_x, safe_depth)))
    pitch = abs(math.degrees(math.atan2(cam_z, safe_depth)))
    return yaw <= ((float(camera.h_fov_deg) * 0.5) + angle_margin + 6.0) and pitch <= (
        (float(camera.v_fov_deg) * 0.5) + angle_margin + 6.0
    )


def _world_triangle_centroid(triangle: _PreparedWorldTriangle) -> Point3:
    return (
        sum(point[0] for point in triangle.points) / 3.0,
        sum(point[1] for point in triangle.points) / 3.0,
        sum(point[2] for point in triangle.points) / 3.0,
    )


def _rapid_tracking_split_static_group(
    *,
    group_id: str,
    layer: str,
    triangles: tuple[_PreparedWorldTriangle, ...] | list[_PreparedWorldTriangle],
    instance_count: int,
    cell_size: float = _RAPID_TRACKING_STATIC_GROUP_CELL_SIZE,
    depth: int = 0,
) -> tuple[_RapidTrackingStaticGroup, ...]:
    triangle_tuple = tuple(triangles)
    if not triangle_tuple:
        return ()
    points = [point for triangle in triangle_tuple for point in triangle.points]
    center, radius = _points_center_radius(points)
    if radius <= _RAPID_TRACKING_STATIC_GROUP_MAX_RADIUS or depth >= 6:
        return (
            _RapidTrackingStaticGroup(
                group_id=group_id,
                layer=layer,
                center=center,
                radius=radius,
                triangles=triangle_tuple,
                instance_count=instance_count,
            ),
        )
    grouped_triangles: dict[tuple[int, int], list[_PreparedWorldTriangle]] = {}
    for triangle in triangle_tuple:
        centroid = _world_triangle_centroid(triangle)
        cell_x = math.floor(float(centroid[0]) / max(1.0, float(cell_size)))
        cell_y = math.floor(float(centroid[1]) / max(1.0, float(cell_size)))
        grouped_triangles.setdefault((cell_x, cell_y), []).append(triangle)
    if len(grouped_triangles) <= 1:
        if len(triangle_tuple) <= 1:
            return (
                _RapidTrackingStaticGroup(
                    group_id=group_id,
                    layer=layer,
                    center=center,
                    radius=radius,
                    triangles=triangle_tuple,
                    instance_count=instance_count,
                ),
            )
        x_span = max(point[0] for point in points) - min(point[0] for point in points)
        y_span = max(point[1] for point in points) - min(point[1] for point in points)
        axis = 0 if x_span >= y_span else 1
        sorted_triangles = sorted(
            triangle_tuple,
            key=lambda triangle: _world_triangle_centroid(triangle)[axis],
        )
        midpoint = max(1, len(sorted_triangles) // 2)
        grouped_triangles = {
            (0, 0): sorted_triangles[:midpoint],
            (1, 0): sorted_triangles[midpoint:],
        }
        if not grouped_triangles[(0, 0)] or not grouped_triangles[(1, 0)]:
            return (
                _RapidTrackingStaticGroup(
                    group_id=group_id,
                    layer=layer,
                    center=center,
                    radius=radius,
                    triangles=triangle_tuple,
                    instance_count=instance_count,
                ),
            )
    split_groups: list[_RapidTrackingStaticGroup] = []
    next_cell_size = max(24.0, float(cell_size) * 0.5)
    for (cell_x, cell_y), group_triangles in sorted(grouped_triangles.items()):
        split_groups.extend(
            _rapid_tracking_split_static_group(
                group_id=f"{group_id}:split:{depth}:{cell_x}:{cell_y}",
                layer=layer,
                triangles=tuple(group_triangles),
                instance_count=instance_count,
                cell_size=next_cell_size,
                depth=depth + 1,
            )
        )
    return tuple(split_groups)


@lru_cache(maxsize=24)
def _rapid_tracking_static_bundle(seed: int) -> _RapidTrackingStaticBundle:
    static_scene = _rapid_tracking_static_scene(int(seed))
    layout = static_scene.layout
    center_x, center_y = _rapid_tracking_layout_world_xy(
        layout,
        track_x=float(layout.compound_center_x),
        track_y=float(layout.compound_center_y),
    )
    library = _rapid_tracking_static_asset_library()
    grouped_instances: dict[str, list[_AssetInstance]] = {}
    grouped_layers: dict[str, str] = {}

    for idx, instance in enumerate(static_scene.core_instances):
        layer = _rapid_tracking_static_group_layer(
            instance,
            center_x=center_x,
            center_y=center_y,
        )
        key = _rapid_tracking_static_group_key(
            instance,
            layer=layer,
            index=idx,
            center_x=center_x,
            center_y=center_y,
        )
        grouped_instances.setdefault(key, []).append(instance)
        grouped_layers[key] = layer

    groups: list[_RapidTrackingStaticGroup] = []
    triangle_count = 0
    for key in sorted(grouped_instances):
        instances = grouped_instances[key]
        layer = grouped_layers[key]
        world_triangles: list[_PreparedWorldTriangle] = []
        for instance in instances:
            mesh = library.mesh(instance.asset_id)
            mesh_triangles = (
                mesh.triangles[::2]
                if layer == "far" and len(mesh.triangles) > 12
                else mesh.triangles
            )
            for triangle in mesh_triangles:
                world_points = tuple(
                    _transform_asset_point(
                        point,
                        position=instance.position,
                        hpr_deg=instance.hpr_deg,
                        scale=instance.scale,
                    )
                    for point in triangle.points
                )
                base_override = instance.color
                if base_override is None:
                    base_rgb = mesh.color_for_role(triangle.role)
                    alpha = 1.0
                else:
                    base_rgb = tuple(float(channel) for channel in base_override[:3])
                    alpha = float(base_override[3])
                world_triangles.append(
                    _PreparedWorldTriangle(
                        points=world_points,
                        normal=_transform_asset_normal(triangle.normal, hpr_deg=instance.hpr_deg),
                        base_rgb=base_rgb,
                        alpha=alpha,
                    )
                )
        if not world_triangles:
            continue
        triangle_count += len(world_triangles)
        groups.extend(
            _rapid_tracking_split_static_group(
                group_id=key,
                layer=layer,
                triangles=tuple(world_triangles),
                instance_count=len(instances),
            )
        )

    return _RapidTrackingStaticBundle(
        groups=tuple(groups),
        asset_ids=static_scene.asset_ids,
        instance_count=len(static_scene.core_instances),
        triangle_count=triangle_count,
    )


def _spatial_terrain_height(*, wx: float, wy: float) -> float:
    ridge = 4.2 * math.sin((wx * 0.030) + 0.7) * math.exp(-((wy - 126.0) ** 2) * 0.00008)
    hill_a = 7.6 * math.exp(-(((wx + 24.0) ** 2) * 0.0018) - (((wy - 98.0) ** 2) * 0.00060))
    hill_b = 6.8 * math.exp(-(((wx - 30.0) ** 2) * 0.0016) - (((wy - 152.0) ** 2) * 0.00048))
    return float(ridge + hill_a + hill_b)


def _spatial_grid_to_world(
    *,
    x: float,
    y: float,
    z: float,
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


def _spatial_landmark_seed(*tokens: object) -> int:
    joined = "|".join(str(token) for token in tokens)
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(joined))


def _spatial_landmark_asset_id(landmark: SpatialIntegrationLandmark) -> str:
    token = str(landmark.kind).strip().lower()
    if token == "building":
        return "building_hangar"
    if token == "tower":
        return "building_tower"
    if token == "truck":
        return "truck_olive"
    if token == "foot_soldiers":
        return "soldiers_patrol"
    if token == "forest":
        return (
            "trees_field_cluster"
            if (_spatial_landmark_seed(landmark.label, landmark.kind) % 2) == 0
            else "forest_canopy_patch"
        )
    if token == "tent":
        return "spatial_tent_canvas"
    if token == "sheep":
        return "spatial_sheep_flock"
    raise RendererBootstrapError(f"Unsupported Spatial Integration landmark kind: {landmark.kind}")


def _spatial_landmark_scale(asset_id: str, *, label: str, kind: str) -> tuple[float, float, float]:
    base = {
        "building_hangar": 2.4,
        "building_tower": 3.2,
        "truck_olive": 1.9,
        "soldiers_patrol": 1.8,
        "trees_field_cluster": 2.8,
        "forest_canopy_patch": 3.1,
        "spatial_tent_canvas": 1.7,
        "spatial_sheep_flock": 1.5,
    }.get(asset_id)
    if base is None:
        raise RendererBootstrapError(f"Unsupported Spatial Integration landmark asset: {asset_id}")
    variation = 0.94 + ((_spatial_landmark_seed(label, kind, asset_id) % 11) * 0.012)
    scale = base * variation
    return (scale, scale, scale)


def _spatial_landmark_heading_deg(asset_id: str, *, label: str, kind: str) -> tuple[float, float, float]:
    seed = _spatial_landmark_seed(label, kind, asset_id)
    if asset_id in {"building_hangar", "spatial_tent_canvas"}:
        heading = float((seed % 4) * 90)
    elif asset_id == "truck_olive":
        heading = float((seed % 8) * 45)
    elif asset_id == "soldiers_patrol":
        heading = float((seed % 12) * 30)
    else:
        heading = float(seed % 360)
    return (heading, 0.0, 0.0)


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
            asset_ids=(
                "building_tower",
                "forest_canopy_patch",
                "plane_blue",
                "trees_field_cluster",
            ),
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

    landmark_offsets = landmark_visual_cell_offsets(landmarks=tuple(payload.landmarks))
    for landmark in tuple(payload.landmarks):
        offset_x, offset_y = landmark_offsets.get(str(landmark.label), (0.0, 0.0))
        wx, wy, wz, terrain = _spatial_grid_to_world(
            x=float(landmark.x) + float(offset_x),
            y=float(landmark.y) + float(offset_y),
            z=0,
            grid_cols=int(payload.grid_cols),
            grid_rows=int(payload.grid_rows),
            alt_levels=int(payload.alt_levels),
        )
        asset_id = _spatial_landmark_asset_id(landmark)
        instances.append(
            _AssetInstance(
                asset_id=asset_id,
                position=(wx, wy, terrain),
                hpr_deg=_spatial_landmark_heading_deg(
                    asset_id,
                    label=str(landmark.label),
                    kind=str(landmark.kind),
                ),
                scale=_spatial_landmark_scale(
                    asset_id,
                    label=str(landmark.label),
                    kind=str(landmark.kind),
                ),
            )
        )

    asset_ids = tuple(sorted({instance.asset_id for instance in instances}))
    return _ScenePlan(
        kind="spatial_integration",
        rect=rect,
        camera=camera,
        asset_instances=tuple(instances),
        overlay_primitives=(),
        asset_ids=asset_ids,
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
            asset_ids=(),
            entity_count=0,
        )
    snapshot = build_trace_test_1_scene3d(payload=payload)
    return _ScenePlan(
        kind="trace_test_1",
        rect=rect,
        camera=_scene_camera_from_trace_camera(snapshot.camera),
        asset_instances=(),
        overlay_primitives=(),
        asset_ids=(),
        entity_count=1 + len(payload.scene.blue_frames),
    )


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
            asset_ids=(),
            entity_count=0,
        )
    snapshot = build_trace_test_2_scene3d(
        payload=payload,
        practice_mode=bool(scene.practice_mode),
    )
    asset_instances = tuple(
        _asset_instance_from_trace_aircraft(pose)
        for pose in snapshot.ghosts
    )
    return _ScenePlan(
        kind="trace_test_2",
        rect=rect,
        camera=_scene_camera_from_trace_camera(snapshot.camera),
        asset_instances=asset_instances,
        overlay_primitives=(),
        asset_ids=tuple(sorted({instance.asset_id for instance in asset_instances})),
        entity_count=len(payload.aircraft) + len(asset_instances),
    )


class _GeometryBatch:
    def __init__(self) -> None:
        self.triangles: list[_ColorVertex] = []
        self.scene_triangles: list[_DepthColorVertex] = []
        self.lines: list[_ColorVertex] = []
        self.textured: list[_TexVertex] = []

    def clear(self) -> None:
        self.triangles.clear()
        self.scene_triangles.clear()
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
        self._last_rt_world_debug: dict[str, object] = {}

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
        self._scene_program = self._ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_pos;
                in float in_depth;
                in vec4 in_color;
                uniform vec2 u_viewport;
                out vec4 v_color;
                void main() {
                    vec2 ndc_xy = vec2(
                        (in_pos.x / u_viewport.x) * 2.0 - 1.0,
                        (in_pos.y / u_viewport.y) * 2.0 - 1.0
                    );
                    float ndc_z = (clamp(in_depth, 0.0, 1.0) * 2.0) - 1.0;
                    gl_Position = vec4(ndc_xy, ndc_z, 1.0);
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
        self._scene_program["u_viewport"].value = (float(self._win_w), float(self._win_h))

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
        self._scene_program["u_viewport"].value = (float(self._win_w), float(self._win_h))
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
        scene_debug: dict[str, object] = {}
        if isinstance(scene, AuditoryGlScene):
            scene_plan = _build_auditory_scene_plan(scene)
            self._scene_assets.require_many(scene_plan.asset_ids)
            entity_count = self._draw_auditory_scene(scene=scene, scene_plan=scene_plan)
        elif isinstance(scene, RapidTrackingGlScene):
            self._last_rt_world_debug = {}
            scene_plan = _build_rapid_tracking_scene_plan(scene)
            self._scene_assets.require_many(scene_plan.asset_ids)
            entity_count = self._draw_rapid_tracking_scene(scene=scene, scene_plan=scene_plan)
            scene_debug = dict(self._last_rt_world_debug)
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
        self._last_scene_debug.update(scene_debug)

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

    def _add_scene_triangle(
        self,
        a: tuple[float, float],
        b: tuple[float, float],
        c: tuple[float, float],
        *,
        depth_a: float,
        depth_b: float,
        depth_c: float,
        color: tuple[float, float, float, float],
    ) -> None:
        self._batch.scene_triangles.extend(
            [
                _DepthColorVertex(a[0], a[1], float(depth_a), *color),
                _DepthColorVertex(b[0], b[1], float(depth_b), *color),
                _DepthColorVertex(c[0], c[1], float(depth_c), *color),
            ]
        )

    @staticmethod
    def _camera_depth_norm(*, depth: float, camera: _SceneCamera) -> float:
        span = max(1e-6, float(camera.far_clip) - float(camera.near_clip))
        return max(0.0, min(1.0, (float(depth) - float(camera.near_clip)) / span))

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
        if not self._batch.triangles:
            return
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
        self._batch.triangles.clear()

    def _flush_world_geometry(self) -> None:
        if not self._batch.scene_triangles:
            return
        tri_data = numpy.array(
            [[v.x, v.y, v.depth, v.r, v.g, v.b, v.a] for v in self._batch.scene_triangles],
            dtype="f4",
        )
        buffer = self._ctx.buffer(tri_data.tobytes())
        vao = self._ctx.vertex_array(
            self._scene_program,
            [(buffer, "2f 1f 4f", "in_pos", "in_depth", "in_color")],
        )
        self._ctx.clear(depth=1.0)
        self._ctx.enable(moderngl.DEPTH_TEST)
        vao.render(moderngl.TRIANGLES)
        self._ctx.disable(moderngl.DEPTH_TEST)
        vao.release()
        buffer.release()
        self._batch.scene_triangles.clear()

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
        return _rotate_hpr_point(point, hpr_deg)

    @classmethod
    def _transform_instance_point(cls, point: Point3, instance: _AssetInstance) -> Point3:
        return _transform_asset_point(
            point,
            position=instance.position,
            hpr_deg=instance.hpr_deg,
            scale=instance.scale,
        )

    @classmethod
    def _transform_instance_normal(cls, normal: Point3, instance: _AssetInstance) -> Point3:
        return _transform_asset_normal(normal, hpr_deg=instance.hpr_deg)

    def _project_world_to_screen(
        self,
        *,
        rect: pygame.Rect,
        camera: _SceneCamera,
        point: Point3,
    ) -> tuple[float, float, float] | None:
        cam_x, cam_y, cam_z = _world_point_to_camera_space(camera=camera, point=point)
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

    def _project_world_triangle_clipped(
        self,
        *,
        rect: pygame.Rect,
        camera: _SceneCamera,
        points: tuple[Point3, Point3, Point3],
    ) -> tuple[list[_ProjectedWorldTriangle2D], bool, int]:
        camera_points: list[Point3] = []
        clipped = False
        for point in points:
            camera_point = _world_point_to_camera_space(camera=camera, point=point)
            if not all(math.isfinite(value) for value in camera_point):
                return [], False, 1
            if not _camera_point_inside_frustum(camera=camera, point=camera_point):
                clipped = True
            camera_points.append(camera_point)
        clipped_polygon = _clip_camera_polygon_to_frustum(
            camera=camera,
            polygon=tuple(camera_points),
        )
        if len(clipped_polygon) < 3:
            return [], clipped, 1
        projected_polygon = _project_clipped_camera_polygon(
            rect=rect,
            window_height=self._win_h,
            camera=camera,
            polygon=clipped_polygon,
        )
        if projected_polygon is None or len(projected_polygon) < 3:
            return [], clipped, 1
        bounds = _scene_screen_bounds(rect=rect, window_height=self._win_h)
        fan_origin = projected_polygon[0]
        emitted: list[_ProjectedWorldTriangle2D] = []
        rejected = 0
        for idx in range(1, len(projected_polygon) - 1):
            tri = (
                fan_origin,
                projected_polygon[idx],
                projected_polygon[idx + 1],
            )
            screen_points = (
                (float(tri[0][0]), float(tri[0][1])),
                (float(tri[1][0]), float(tri[1][1])),
                (float(tri[2][0]), float(tri[2][1])),
            )
            min_x = min(point[0] for point in screen_points)
            max_x = max(point[0] for point in screen_points)
            min_y = min(point[1] for point in screen_points)
            max_y = max(point[1] for point in screen_points)
            if max_x < bounds[0] or min_x > bounds[2] or max_y < bounds[1] or min_y > bounds[3]:
                rejected += 1
                continue
            area = (screen_points[1][0] - screen_points[0][0]) * (
                screen_points[2][1] - screen_points[0][1]
            ) - (screen_points[1][1] - screen_points[0][1]) * (
                screen_points[2][0] - screen_points[0][0]
            )
            if abs(area) <= 0.35:
                rejected += 1
                continue
            camera_depths = (
                float(tri[0][2]),
                float(tri[1][2]),
                float(tri[2][2]),
            )
            emitted.append(
                _ProjectedWorldTriangle2D(
                    depth=sum(camera_depths) / 3.0,
                    points=screen_points,
                    camera_depths=camera_depths,
                    extent_px=max(max_x - min_x, max_y - min_y),
                )
            )
        if not emitted:
            return [], clipped, max(1, rejected)
        return emitted, clipped, rejected

    def _render_scene_plan(self, *, scene_plan: _ScenePlan) -> None:
        camera = scene_plan.camera
        use_depth_world = scene_plan.kind == "rapid_tracking"
        clip_world_triangles = scene_plan.kind in {"rapid_tracking", "auditory"}
        rt_world_debug = {
            "world_triangles_input": 0,
            "world_triangles_emitted": 0,
            "world_triangles_clipped": 0,
            "world_triangles_rejected": 0,
            "max_projected_extent_px": 0.0,
        }
        if use_depth_world:
            self._last_rt_world_debug = dict(rt_world_debug)
        if camera is None or (not scene_plan.asset_instances and not scene_plan.static_groups):
            return

        light_dir = _vec_normalize((0.34, 0.58, 0.74))
        static_groups = (
            (*scene_plan.backdrop_groups, *scene_plan.playfield_groups)
            if use_depth_world and (scene_plan.backdrop_groups or scene_plan.playfield_groups)
            else scene_plan.static_groups
        )
        projected_triangles: list[
            tuple[
                float,
                tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
                tuple[float, float, float] | None,
                tuple[float, float, float, float],
            ]
        ] = []

        for group in static_groups:
            if use_depth_world and not _rapid_tracking_static_group_visible(camera=camera, group=group):
                continue
            is_far = group.layer == "far"
            for triangle in group.triangles:
                rt_world_debug["world_triangles_input"] += 1
                if clip_world_triangles:
                    projected_world_triangles, was_clipped, rejected = (
                        self._project_world_triangle_clipped(
                            rect=scene_plan.rect,
                            camera=camera,
                            points=triangle.points,
                        )
                    )
                    if was_clipped:
                        rt_world_debug["world_triangles_clipped"] += 1
                    rt_world_debug["world_triangles_rejected"] += int(rejected)
                    if not projected_world_triangles:
                        continue
                else:
                    projected_points: list[tuple[float, float]] = []
                    depths: list[float] = []
                    for point in triangle.points:
                        projected = self._project_world_to_screen(
                            rect=scene_plan.rect,
                            camera=camera,
                            point=point,
                        )
                        if projected is None:
                            projected_points = []
                            break
                        projected_points.append((projected[0], projected[1]))
                        depths.append(projected[2])
                    if len(projected_points) != 3:
                        continue
                    projected_world_triangles = [
                        _ProjectedWorldTriangle2D(
                            depth=sum(depths) / 3.0,
                            points=(
                                projected_points[0],
                                projected_points[1],
                                projected_points[2],
                            ),
                            camera_depths=(depths[0], depths[1], depths[2]),
                            extent_px=max(
                                max(point[0] for point in projected_points)
                                - min(point[0] for point in projected_points),
                                max(point[1] for point in projected_points)
                                - min(point[1] for point in projected_points),
                            ),
                        )
                    ]
                for projected_world_triangle in projected_world_triangles:
                    projected_points = projected_world_triangle.points
                    depths = projected_world_triangle.camera_depths
                    area = (projected_points[1][0] - projected_points[0][0]) * (
                        projected_points[2][1] - projected_points[0][1]
                    ) - (projected_points[1][1] - projected_points[0][1]) * (
                        projected_points[2][0] - projected_points[0][0]
                    )
                    if abs(area) <= 0.35:
                        rt_world_debug["world_triangles_rejected"] += 1
                        continue
                    light = max(
                        0.0,
                        min(
                            1.0,
                            (triangle.normal[0] * light_dir[0])
                            + (triangle.normal[1] * light_dir[1])
                            + (triangle.normal[2] * light_dir[2]),
                        ),
                    )
                    if is_far:
                        depth_t = max(0.0, min(1.0, (sum(depths) / 3.0 - 220.0) / 900.0))
                        if use_depth_world:
                            shade = 0.66 + (light * 0.28)
                            base_rgb = _mix_rgb(
                                triangle.base_rgb, (0.62, 0.70, 0.66), 0.26 + (0.20 * depth_t)
                            )
                            alpha = max(0.56, triangle.alpha * (0.90 - (0.10 * depth_t)))
                        else:
                            shade = 0.30 + (light * 0.44)
                            base_rgb = _mix_rgb(
                                triangle.base_rgb, (0.50, 0.60, 0.68), 0.30 + (0.30 * depth_t)
                            )
                            alpha = max(0.42, triangle.alpha * (0.82 - (0.22 * depth_t)))
                    else:
                        depth_t = max(0.0, min(1.0, (sum(depths) / 3.0 - 120.0) / 700.0))
                        if use_depth_world:
                            shade = 0.70 + (light * 0.32)
                            base_rgb = _mix_rgb(
                                triangle.base_rgb, (0.52, 0.60, 0.48), 0.16 + (0.12 * depth_t)
                            )
                            alpha = triangle.alpha
                        else:
                            shade = 0.38 + (light * 0.62)
                            base_rgb = _mix_rgb(
                                triangle.base_rgb, (0.40, 0.48, 0.44), 0.06 * depth_t
                            )
                            alpha = triangle.alpha
                    fill_color = (
                        max(0.0, min(1.0, base_rgb[0] * shade)),
                        max(0.0, min(1.0, base_rgb[1] * shade)),
                        max(0.0, min(1.0, base_rgb[2] * shade)),
                        max(0.0, min(1.0, alpha)),
                    )
                    projected_triangles.append(
                        (
                            projected_world_triangle.depth,
                            projected_points,
                            (
                                self._camera_depth_norm(depth=depths[0], camera=camera),
                                self._camera_depth_norm(depth=depths[1], camera=camera),
                                self._camera_depth_norm(depth=depths[2], camera=camera),
                            )
                            if use_depth_world
                            else None,
                            fill_color,
                        )
                    )
                    rt_world_debug["world_triangles_emitted"] += 1
                    rt_world_debug["max_projected_extent_px"] = max(
                        float(rt_world_debug["max_projected_extent_px"]),
                        float(projected_world_triangle.extent_px),
                    )

        for instance in scene_plan.asset_instances:
            mesh = self._scene_assets.mesh(instance.asset_id)
            base_override = instance.color
            rotated_normal_cache: dict[Point3, Point3] = {}
            for triangle in mesh.triangles:
                world_points = tuple(
                    self._transform_instance_point(point, instance) for point in triangle.points
                )
                rt_world_debug["world_triangles_input"] += 1
                if clip_world_triangles:
                    projected_world_triangles, was_clipped, rejected = (
                        self._project_world_triangle_clipped(
                            rect=scene_plan.rect,
                            camera=camera,
                            points=world_points,
                        )
                    )
                    if was_clipped:
                        rt_world_debug["world_triangles_clipped"] += 1
                    rt_world_debug["world_triangles_rejected"] += int(rejected)
                    if not projected_world_triangles:
                        continue
                else:
                    projected_points: list[tuple[float, float]] = []
                    depths: list[float] = []
                    for point in world_points:
                        projected = self._project_world_to_screen(
                            rect=scene_plan.rect,
                            camera=camera,
                            point=point,
                        )
                        if projected is None:
                            projected_points = []
                            break
                        projected_points.append((projected[0], projected[1]))
                        depths.append(projected[2])
                    if len(projected_points) != 3:
                        continue
                    projected_world_triangles = [
                        _ProjectedWorldTriangle2D(
                            depth=sum(depths) / 3.0,
                            points=(
                                projected_points[0],
                                projected_points[1],
                                projected_points[2],
                            ),
                            camera_depths=(depths[0], depths[1], depths[2]),
                            extent_px=max(
                                max(point[0] for point in projected_points)
                                - min(point[0] for point in projected_points),
                                max(point[1] for point in projected_points)
                                - min(point[1] for point in projected_points),
                            ),
                        )
                    ]
                world_normal = rotated_normal_cache.get(triangle.normal)
                if world_normal is None:
                    world_normal = self._transform_instance_normal(triangle.normal, instance)
                    rotated_normal_cache[triangle.normal] = world_normal
                light = max(
                    0.0,
                    min(
                        1.0,
                        (world_normal[0] * light_dir[0])
                        + (world_normal[1] * light_dir[1])
                        + (world_normal[2] * light_dir[2]),
                    ),
                )
                shade = (0.52 + (light * 0.48)) if use_depth_world else (0.38 + (light * 0.62))
                if base_override is None:
                    base_rgb = mesh.color_for_role(triangle.role)
                    alpha = 1.0
                else:
                    base_rgb = tuple(float(channel) for channel in base_override[:3])
                    alpha = float(base_override[3])
                if use_depth_world:
                    base_rgb = _mix_rgb(base_rgb, (0.50, 0.58, 0.46), 0.12)
                    shade = 0.64 + (light * 0.32)
                fill_color = (
                    max(0.0, min(1.0, base_rgb[0] * shade)),
                    max(0.0, min(1.0, base_rgb[1] * shade)),
                    max(0.0, min(1.0, base_rgb[2] * shade)),
                    max(0.0, min(1.0, alpha)),
                )
                for projected_world_triangle in projected_world_triangles:
                    depths = projected_world_triangle.camera_depths
                    projected_triangles.append(
                        (
                            projected_world_triangle.depth,
                            projected_world_triangle.points,
                            (
                                self._camera_depth_norm(depth=depths[0], camera=camera),
                                self._camera_depth_norm(depth=depths[1], camera=camera),
                                self._camera_depth_norm(depth=depths[2], camera=camera),
                            )
                            if use_depth_world
                            else None,
                            fill_color,
                        )
                    )
                    rt_world_debug["world_triangles_emitted"] += 1
                    rt_world_debug["max_projected_extent_px"] = max(
                        float(rt_world_debug["max_projected_extent_px"]),
                        float(projected_world_triangle.extent_px),
                    )

        projected_triangles.sort(key=lambda item: item[0], reverse=True)
        for _depth, points, depth_norms, color in projected_triangles:
            if depth_norms is None:
                self._add_filled_polygon(points, color)
                continue
            self._add_scene_triangle(
                points[0],
                points[1],
                points[2],
                depth_a=depth_norms[0],
                depth_b=depth_norms[1],
                depth_c=depth_norms[2],
                color=color,
            )

        self._render_overlay_primitives(scene_plan.overlay_primitives)
        if use_depth_world:
            self._last_rt_world_debug = {
                "world_triangles_input": int(rt_world_debug["world_triangles_input"]),
                "world_triangles_emitted": int(rt_world_debug["world_triangles_emitted"]),
                "world_triangles_clipped": int(rt_world_debug["world_triangles_clipped"]),
                "world_triangles_rejected": int(rt_world_debug["world_triangles_rejected"]),
                "max_projected_extent_px": float(rt_world_debug["max_projected_extent_px"]),
            }

    def _render_overlay_primitives(
        self,
        primitives: tuple[_ProjectedOverlayPrimitive, ...] | list[_ProjectedOverlayPrimitive],
    ) -> None:
        for primitive in primitives:
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
            body_color=tuple(
                max(0, min(255, int(round(channel * 255.0)))) for channel in color[:3]
            ),
            outline_color=tuple(
                max(0, min(255, int(round(channel * 255.0)))) for channel in outline[:3]
            ),
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
            fill = tuple(
                max(0, min(255, int(round(channel * float(face.shade))))) for channel in base
            )
            fill_color = (fill[0] / 255.0, fill[1] / 255.0, fill[2] / 255.0, color[3])
            points = [(float(px), float(py)) for px, py in face.points]
            self._add_filled_polygon(points, fill_color)
            self._add_polyline(points, outline, closed=True, width=1.0)

    def _draw_auditory_scene(self, *, scene: AuditoryGlScene, scene_plan: _ScenePlan) -> int:
        rect = scene.world
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

    def _draw_rapid_tracking_scene(
        self, *, scene: RapidTrackingGlScene, scene_plan: _ScenePlan
    ) -> int:
        rect = scene.world
        payload = scene.payload
        vw = max(1, int(rect.w))
        vh = max(1, int(rect.h))
        backdrop = _rapid_tracking_backdrop_plan(viewport_size=(vw, vh), payload=payload)
        horizon = float(backdrop.horizon_y)
        ground_top = max(0.0, min(float(vh), horizon))
        if ground_top < float(vh):
            sky_span = max(1.0, float(vh) - ground_top)
            for idx in range(8):
                t0 = idx / 8.0
                t1 = (idx + 1) / 8.0
                y0 = ground_top + (sky_span * t0)
                y1 = ground_top + (sky_span * t1)
                mix = (t0 + t1) * 0.5
                sky_rgb = _mix_rgb(backdrop.sky_bottom_rgb, backdrop.sky_top_rgb, mix)
                self._add_quad(
                    self._to_screen(rect, 0.0, y0),
                    self._to_screen(rect, float(vw), y0),
                    self._to_screen(rect, float(vw), y1),
                    self._to_screen(rect, 0.0, y1),
                    (sky_rgb[0], sky_rgb[1], sky_rgb[2], 1.0),
                )
        if ground_top > 0.0:
            for idx in range(7):
                t0 = idx / 7.0
                t1 = (idx + 1) / 7.0
                y0 = ground_top * t0
                y1 = ground_top * t1
                mix = (t0 + t1) * 0.5
                ground_rgb = _mix_rgb(backdrop.ground_base_rgb, backdrop.ground_horizon_rgb, mix)
                self._add_quad(
                    self._to_screen(rect, 0.0, y0),
                    self._to_screen(rect, float(vw), y0),
                    self._to_screen(rect, float(vw), y1),
                    self._to_screen(rect, 0.0, y1),
                    (ground_rgb[0], ground_rgb[1], ground_rgb[2], 1.0),
                )
        if 0.0 < horizon < float(vh):
            haze_half_h = max(1.0, float(vh) / 90.0)
            self._add_quad(
                self._to_screen(rect, 0.0, max(0.0, horizon - haze_half_h)),
                self._to_screen(rect, float(vw), max(0.0, horizon - haze_half_h)),
                self._to_screen(rect, float(vw), min(float(vh), horizon + haze_half_h)),
                self._to_screen(rect, 0.0, min(float(vh), horizon + haze_half_h)),
                (0.72, 0.78, 0.72, 0.18),
            )
        self._flush_color_geometry()
        self._render_scene_plan(scene_plan=scene_plan)
        self._flush_world_geometry()
        if payload is None:
            return 0

        target = build_rapid_tracking_scene_target(payload=payload, size=(vw, vh))
        target_center = self._to_screen(
            rect, float(target.overlay.screen_x), float(vh - target.overlay.screen_y)
        )
        center = self._to_screen(rect, float(vw) * 0.5, float(vh) * 0.5)
        box_half_w = max(
            18.0,
            (((float(vw) * 0.5) - 12.0) * float(payload.capture_box_half_width)),
        )
        box_half_h = max(
            14.0,
            ((((float(vh) * 0.5) - 12.0) * 0.90) * float(payload.capture_box_half_height)),
        )
        box_color = (
            (0.34, 0.84, 0.62, 0.92)
            if bool(payload.target_in_capture_box) and bool(payload.target_visible)
            else (0.54, 0.72, 0.66, 0.82)
        )
        self._add_polyline(
            [
                (center[0] - box_half_w, center[1] - box_half_h),
                (center[0] + box_half_w, center[1] - box_half_h),
                (center[0] + box_half_w, center[1] + box_half_h),
                (center[0] - box_half_w, center[1] + box_half_h),
            ],
            box_color,
            closed=True,
            width=2.0,
        )
        cross_len = max(18.0, min(float(vw), float(vh)) / 14.0)
        self._add_line(
            (center[0] - cross_len, center[1]),
            (center[0] + cross_len, center[1]),
            (0.88, 0.36, 0.28, 0.90),
            width=2.0,
        )
        self._add_line(
            (center[0], center[1] - cross_len),
            (center[0], center[1] + cross_len),
            (0.88, 0.36, 0.28, 0.90),
            width=2.0,
        )
        reticle_x = (
            (float(payload.reticle_x) + float(RAPID_TRACKING_TARGET_VIEW_LIMIT))
            / (float(RAPID_TRACKING_TARGET_VIEW_LIMIT) * 2.0)
        ) * float(vw)
        reticle_y = (
            (float(payload.reticle_y) + float(RAPID_TRACKING_TARGET_VIEW_LIMIT))
            / (float(RAPID_TRACKING_TARGET_VIEW_LIMIT) * 2.0)
        ) * float(vh)
        reticle_center = self._to_screen(rect, reticle_x, float(vh) - reticle_y)
        self._add_circle(
            center=reticle_center,
            radius=8.0,
            color=(0.98, 0.95, 0.84, 0.88),
            filled=False,
            width=1.5,
            segments=24,
        )
        self._add_line(
            (reticle_center[0] - 12.0, reticle_center[1]),
            (reticle_center[0] + 12.0, reticle_center[1]),
            (0.98, 0.95, 0.84, 0.66),
            width=1.0,
        )
        self._add_line(
            (reticle_center[0], reticle_center[1] - 12.0),
            (reticle_center[0], reticle_center[1] + 12.0),
            (0.98, 0.95, 0.84, 0.66),
            width=1.0,
        )
        self._add_circle(
            center=target_center, radius=30.0, color=(0.92, 0.98, 0.96, 0.64), filled=False
        )
        return max(1, int(scene_plan.entity_count))

    def _draw_spatial_integration_scene(
        self, *, scene: SpatialIntegrationGlScene, scene_plan: _ScenePlan
    ) -> int:
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
        prev = self._to_screen(
            rect, float(layout.aircraft_prev[0]), float(vh - layout.aircraft_prev[1])
        )
        now = self._to_screen(
            rect, float(layout.aircraft_now[0]), float(vh - layout.aircraft_now[1])
        )
        self._add_line(prev, now, (0.94, 0.78, 0.32, 0.82), width=2.0)
        if layout.aircraft_future is not None:
            future = self._to_screen(
                rect, float(layout.aircraft_future[0]), float(vh - layout.aircraft_future[1])
            )
            self._add_line(now, future, (0.94, 0.56, 0.30, 0.82), width=2.0)
        return int(scene_plan.entity_count)

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
        if scene.payload is None:
            return 0
        self._render_overlay_primitives(
            _trace_test_1_marker_primitives(
                rect=rect,
                window_height=self._win_h,
                payload=scene.payload,
            )
        )
        if scene.practice_mode:
            horizon_y = float(vh * 0.52)
            left = _scene_local_top_left_to_screen(
                rect=rect,
                window_height=self._win_h,
                x=0.0,
                y=horizon_y,
            )
            right = _scene_local_top_left_to_screen(
                rect=rect,
                window_height=self._win_h,
                x=float(vw),
                y=horizon_y,
            )
            center = _scene_local_top_left_to_screen(
                rect=rect,
                window_height=self._win_h,
                x=float(vw * 0.5),
                y=float(vh * 0.5),
            )
            self._add_line(left, right, (0.92, 0.96, 1.0, 0.28), width=1.0)
            self._add_circle(
                center=center,
                radius=max(16.0, min(vw, vh) * 0.045),
                color=(0.92, 0.96, 1.0, 0.18),
                filled=False,
                width=1.2,
            )
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
        if scene.payload is None:
            return 0
        if scene_plan.asset_instances:
            self._render_scene_plan(scene_plan=scene_plan)
        self._render_overlay_primitives(
            _trace_test_2_marker_primitives(
                rect=rect,
                window_height=self._win_h,
                payload=scene.payload,
            )
        )
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
