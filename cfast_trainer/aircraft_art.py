from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


Point3 = tuple[float, float, float]
INSTRUMENT_AIRCRAFT_MESH_PATH = (
    Path(__file__).resolve().parent.parent / "assets" / "instrument_aircraft" / "fixed_wing.obj"
)


@dataclass(frozen=True, slots=True)
class FixedWingMeshFace:
    role: str
    points: tuple[Point3, ...]


@dataclass(frozen=True, slots=True)
class FixedWingProjectedFace:
    role: str
    points: tuple[tuple[int, int], ...]
    avg_depth: float
    shade: float


@dataclass(frozen=True, slots=True)
class FixedWingPygamePalette:
    body: tuple[int, int, int]
    wing: tuple[int, int, int]
    tail: tuple[int, int, int]
    accent: tuple[int, int, int]
    canopy: tuple[int, int, int]
    engine: tuple[int, int, int]
    outline: tuple[int, int, int] = (242, 246, 252)


_DEFAULT_PYGAME_INSTRUMENT_PALETTE = FixedWingPygamePalette(
    body=(218, 56, 62),
    wing=(204, 44, 50),
    tail=(182, 34, 44),
    accent=(134, 24, 34),
    canopy=(138, 218, 232),
    engine=(94, 94, 102),
    outline=(244, 248, 255),
)
_ROLE_ALIASES = {
    "body": "body",
    "fuselage": "body",
    "nose": "accent",
    "accent": "accent",
    "canopy": "canopy",
    "glass": "canopy",
    "engine": "engine",
    "nacelle": "engine",
    "wing": "wing",
    "tail": "tail",
    "tailplane": "tail",
    "vertical": "tail",
}


def instrument_card_pygame_palette() -> FixedWingPygamePalette:
    return _DEFAULT_PYGAME_INSTRUMENT_PALETTE


def build_pygame_palette(
    *,
    body_color: tuple[int, int, int],
    canopy_color: tuple[int, int, int] | None = None,
    accent_color: tuple[int, int, int] | None = None,
    engine_color: tuple[int, int, int] | None = None,
    outline_color: tuple[int, int, int] = (242, 246, 252),
) -> FixedWingPygamePalette:
    body = _rgb(body_color)

    def scaled(color: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
        return tuple(max(0, min(255, int(round(float(channel) * factor)))) for channel in color)

    return FixedWingPygamePalette(
        body=body,
        wing=scaled(body, 0.94),
        tail=scaled(body, 0.84),
        accent=accent_color or scaled(body, 0.66),
        canopy=canopy_color or (138, 218, 232),
        engine=engine_color or (94, 94, 102),
        outline=_rgb(outline_color),
    )


def fixed_wing_hpr(
    *,
    heading_deg: float,
    pitch_deg: float,
    roll_deg: float,
) -> tuple[float, float, float]:
    return (float(heading_deg), float(pitch_deg), float(roll_deg))


def fixed_wing_hpr_from_world_hpr(
    *,
    heading_deg: float,
    pitch_deg: float,
    roll_deg: float,
) -> tuple[float, float, float]:
    return fixed_wing_hpr(
        heading_deg=float(heading_deg),
        pitch_deg=-float(pitch_deg),
        roll_deg=float(roll_deg),
    )


def fixed_wing_heading_from_screen_heading(screen_heading_deg: float) -> float:
    return (float(screen_heading_deg) + 90.0) % 360.0


def screen_motion_heading_deg(
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    minimum_distance: float = 0.2,
) -> float | None:
    dx = float(end[0] - start[0])
    dy = float(end[1] - start[1])
    if math.hypot(dx, dy) < max(0.0, float(minimum_distance)):
        return None
    return float(math.degrees(math.atan2(dy, dx)))


def screen_heading_deg_from_world_tangent(
    tangent: Point3,
    *,
    forward_x_mix: float = 0.10,
    forward_y_mix: float = 0.26,
    minimum_distance: float = 1e-4,
) -> float | None:
    screen_dx = float(tangent[0]) + (float(tangent[1]) * float(forward_x_mix))
    screen_dy = -(float(tangent[2]) + (float(tangent[1]) * float(forward_y_mix)))
    return screen_motion_heading_deg(
        (0.0, 0.0),
        (screen_dx, screen_dy),
        minimum_distance=minimum_distance,
    )


def fixed_wing_hpr_from_screen_heading(
    *,
    screen_heading_deg: float,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
    camera_heading_deg: float = 0.0,
) -> tuple[float, float, float]:
    return fixed_wing_hpr(
        heading_deg=fixed_wing_heading_from_screen_heading(screen_heading_deg)
        + float(camera_heading_deg),
        pitch_deg=float(pitch_deg),
        roll_deg=float(roll_deg),
    )


def fixed_wing_hpr_from_tangent(
    tangent: Point3,
    *,
    bank_deg: float = 0.0,
) -> tuple[float, float, float]:
    return fixed_wing_hpr_from_world_tangent(tangent=tangent, roll_deg=float(bank_deg))


def fixed_wing_hpr_from_world_tangent(
    tangent: Point3,
    *,
    roll_deg: float = 0.0,
) -> tuple[float, float, float]:
    dx, dy, dz = (float(tangent[0]), float(tangent[1]), float(tangent[2]))
    if (dx * dx) + (dy * dy) + (dz * dz) <= 1e-8:
        raise ValueError("world tangent must be non-zero")

    horiz = max(1e-6, math.sqrt((dx * dx) + (dy * dy)))
    return fixed_wing_hpr_from_world_hpr(
        heading_deg=math.degrees(math.atan2(dx, dy)) % 360.0,
        pitch_deg=math.degrees(math.atan2(dz, horiz)),
        roll_deg=float(roll_deg),
    )


@lru_cache(maxsize=1)
def build_fixed_wing_mesh() -> tuple[FixedWingMeshFace, ...]:
    return load_fixed_wing_obj(INSTRUMENT_AIRCRAFT_MESH_PATH)


def load_fixed_wing_obj(path: Path | str) -> tuple[FixedWingMeshFace, ...]:
    vertices: list[Point3] = []
    faces: list[FixedWingMeshFace] = []
    current_role = "body"
    source = Path(path)
    for raw_line in source.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        head = parts[0]
        if head == "v" and len(parts) >= 4:
            vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            continue
        if head in {"o", "g", "usemtl"} and len(parts) >= 2:
            current_role = _role_from_token(" ".join(parts[1:]))
            continue
        if head == "f" and len(parts) >= 4:
            points: list[Point3] = []
            for token in parts[1:]:
                idx = int(token.split("/", 1)[0])
                if idx < 0:
                    idx = len(vertices) + idx + 1
                points.append(vertices[idx - 1])
            faces.append(FixedWingMeshFace(role=current_role, points=tuple(points)))
    if not faces:
        raise ValueError(f"fixed-wing mesh has no faces: {source}")
    return tuple(faces)


def project_fixed_wing_faces(
    *,
    heading_deg: float,
    pitch_deg: float,
    bank_deg: float,
    cx: int,
    cy: int,
    scale: float,
    view_yaw_deg: float = 0.0,
    view_pitch_deg: float = 0.0,
    view_roll_deg: float = 0.0,
    forward_x_mix: float = 0.10,
    forward_y_mix: float = 0.26,
) -> tuple[FixedWingProjectedFace, ...]:
    projected: list[FixedWingProjectedFace] = []
    for face in build_fixed_wing_mesh():
        rotated = tuple(
            apply_fixed_wing_view_rotation(
                rotate_fixed_wing_point(
                    point,
                    heading_deg=heading_deg,
                    pitch_deg=pitch_deg,
                    bank_deg=bank_deg,
                ),
                view_yaw_deg=view_yaw_deg,
                view_pitch_deg=view_pitch_deg,
                view_roll_deg=view_roll_deg,
            )
            for point in face.points
        )
        points_2d: list[tuple[int, int]] = []
        depth_sum = 0.0
        for point in rotated:
            sx, sy, depth = project_fixed_wing_point(
                point,
                cx=cx,
                cy=cy,
                scale=scale,
                forward_x_mix=forward_x_mix,
                forward_y_mix=forward_y_mix,
            )
            points_2d.append((sx, sy))
            depth_sum += depth
        if _polygon_area(points_2d) < 1.0:
            continue
        projected.append(
            FixedWingProjectedFace(
                role=face.role,
                points=tuple(points_2d),
                avg_depth=depth_sum / float(len(rotated)),
                shade=_face_shade(rotated),
            )
        )
    projected.sort(key=lambda item: item.avg_depth, reverse=True)
    return tuple(projected)


def draw_fixed_wing_pygame(
    surface,
    *,
    heading_deg: float,
    pitch_deg: float,
    bank_deg: float,
    cx: int,
    cy: int,
    scale: float,
    palette: FixedWingPygamePalette | None = None,
    view_yaw_deg: float = 0.0,
    view_pitch_deg: float = 0.0,
    view_roll_deg: float = 0.0,
    forward_x_mix: float = 0.10,
    forward_y_mix: float = 0.26,
) -> None:
    import pygame

    paint = palette or _DEFAULT_PYGAME_INSTRUMENT_PALETTE
    role_colors = {
        "body": paint.body,
        "wing": paint.wing,
        "tail": paint.tail,
        "accent": paint.accent,
        "canopy": paint.canopy,
        "engine": paint.engine,
    }
    for face in project_fixed_wing_faces(
        heading_deg=heading_deg,
        pitch_deg=pitch_deg,
        bank_deg=bank_deg,
        cx=cx,
        cy=cy,
        scale=scale,
        view_yaw_deg=view_yaw_deg,
        view_pitch_deg=view_pitch_deg,
        view_roll_deg=view_roll_deg,
        forward_x_mix=forward_x_mix,
        forward_y_mix=forward_y_mix,
    ):
        base = role_colors.get(face.role, paint.body)
        fill = _shade_rgb(base, face.shade)
        pygame.draw.polygon(surface, fill, face.points)
        pygame.draw.polygon(surface, paint.outline, face.points, 1)


def rotate_fixed_wing_point(
    point: Point3,
    *,
    heading_deg: float,
    pitch_deg: float,
    bank_deg: float,
) -> Point3:
    x, y, z = point

    roll = math.radians(float(bank_deg))
    cos_r = math.cos(roll)
    sin_r = math.sin(roll)
    x1 = x * cos_r + z * sin_r
    y1 = y
    z1 = -x * sin_r + z * cos_r

    pitch = math.radians(float(pitch_deg))
    cos_p = math.cos(pitch)
    sin_p = math.sin(pitch)
    x2 = x1
    y2 = y1 * cos_p - z1 * sin_p
    z2 = y1 * sin_p + z1 * cos_p

    yaw = math.radians(-float(heading_deg))
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    x3 = x2 * cos_y - y2 * sin_y
    y3 = x2 * sin_y + y2 * cos_y
    return (x3, y3, z2)


def apply_fixed_wing_view_rotation(
    point: Point3,
    *,
    view_yaw_deg: float = 0.0,
    view_pitch_deg: float = 0.0,
    view_roll_deg: float = 0.0,
) -> Point3:
    x, y, z = point

    yaw = math.radians(-float(view_yaw_deg))
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    x1 = x * cos_y - y * sin_y
    y1 = x * sin_y + y * cos_y
    z1 = z

    pitch = math.radians(float(view_pitch_deg))
    cos_p = math.cos(pitch)
    sin_p = math.sin(pitch)
    x2 = x1
    y2 = y1 * cos_p - z1 * sin_p
    z2 = y1 * sin_p + z1 * cos_p

    roll = math.radians(float(view_roll_deg))
    cos_r = math.cos(roll)
    sin_r = math.sin(roll)
    x3 = x2 * cos_r + z2 * sin_r
    y3 = y2
    z3 = -x2 * sin_r + z2 * cos_r
    return (x3, y3, z3)


def project_fixed_wing_point(
    point: Point3,
    *,
    cx: int,
    cy: int,
    scale: float,
    forward_x_mix: float = 0.10,
    forward_y_mix: float = 0.26,
) -> tuple[int, int, float]:
    x, y, z = point
    sx = int(round(cx + (x + (y * float(forward_x_mix))) * scale))
    sy = int(round(cy - (z + (y * float(forward_y_mix))) * scale))
    return sx, sy, y


def _role_from_token(token: str) -> str:
    normalized = token.strip().lower().replace("-", "_")
    for key, role in _ROLE_ALIASES.items():
        if key in normalized:
            return role
    return "body"


def _rgb(color: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(max(0, min(255, int(channel))) for channel in color)


def _shade_rgb(color: tuple[int, int, int], shade: float) -> tuple[int, int, int]:
    return tuple(max(0, min(255, int(round(channel * shade)))) for channel in color)


def _polygon_area(points: list[tuple[int, int]]) -> float:
    if len(points) < 3:
        return 0.0
    total = 0.0
    for current, nxt in zip(points, (*points[1:], points[0]), strict=False):
        total += (current[0] * nxt[1]) - (nxt[0] * current[1])
    return abs(total) * 0.5


def _face_shade(points: tuple[Point3, ...]) -> float:
    normal = _face_normal(points)
    light = _normalize((-0.42, -0.34, 0.84))
    dot = max(0.0, _dot(normal, light))
    return max(0.58, min(1.20, 0.70 + (dot * 0.45)))


def _face_normal(points: tuple[Point3, ...]) -> Point3:
    if len(points) < 3:
        return (0.0, 0.0, 1.0)
    a = points[0]
    b = points[1]
    c = points[2]
    return _normalize(_cross(_sub(b, a), _sub(c, a)))


def _sub(a: Point3, b: Point3) -> Point3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a: Point3, b: Point3) -> Point3:
    return (
        (a[1] * b[2]) - (a[2] * b[1]),
        (a[2] * b[0]) - (a[0] * b[2]),
        (a[0] * b[1]) - (a[1] * b[0]),
    )


def _dot(a: Point3, b: Point3) -> float:
    return (a[0] * b[0]) + (a[1] * b[1]) + (a[2] * b[2])


def _normalize(vec: Point3) -> Point3:
    mag = math.sqrt((vec[0] * vec[0]) + (vec[1] * vec[1]) + (vec[2] * vec[2]))
    if mag <= 1e-8:
        return (0.0, 0.0, 1.0)
    return (vec[0] / mag, vec[1] / mag, vec[2] / mag)
