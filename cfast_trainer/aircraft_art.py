from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache


Point3 = tuple[float, float, float]


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
    accent: tuple[int, int, int]
    canopy: tuple[int, int, int]
    engine: tuple[int, int, int]
    outline: tuple[int, int, int] = (242, 246, 252)


_DEFAULT_PYGAME_INSTRUMENT_PALETTE = FixedWingPygamePalette(
    body=(222, 66, 68),
    accent=(184, 34, 40),
    canopy=(164, 226, 234),
    engine=(134, 18, 24),
    outline=(244, 248, 255),
)
_FIXED_WING_MODEL_HPR_OFFSET_DEG = (0.0, 0.0, 0.0)


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
    body = tuple(max(0, min(255, int(channel))) for channel in body_color)

    def scaled(color: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
        return tuple(max(0, min(255, int(round(float(channel) * factor)))) for channel in color)

    return FixedWingPygamePalette(
        body=body,
        accent=accent_color or scaled(body, 0.78),
        canopy=canopy_color or (164, 226, 234),
        engine=engine_color or scaled(body, 0.58),
        outline=tuple(max(0, min(255, int(channel))) for channel in outline_color),
    )


def fixed_wing_hpr(
    *,
    heading_deg: float,
    pitch_deg: float,
    roll_deg: float,
) -> tuple[float, float, float]:
    offset_h, offset_p, offset_r = _FIXED_WING_MODEL_HPR_OFFSET_DEG
    return (
        float(heading_deg) + offset_h,
        float(pitch_deg) + offset_p,
        float(roll_deg) + offset_r,
    )


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
    """Convert a 2-D screen tangent angle into the fixed-wing heading convention.

    Screen headings use ``atan2(dy, dx)`` with 0 degrees pointing right and
    -90 degrees pointing up. The fixed-wing mesh uses 0 degrees as straight
    ahead/up on screen, 90 right, 180 down, and 270 left.
    """

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
    forward_x_mix: float = 0.11,
    forward_y_mix: float = 0.31,
    minimum_distance: float = 1e-4,
) -> float | None:
    screen_dx = float(tangent[0]) + (float(tangent[1]) * float(forward_x_mix))
    screen_dy = -(float(tangent[2]) + (float(tangent[1]) * float(forward_y_mix)))
    heading = screen_motion_heading_deg(
        (0.0, 0.0),
        (screen_dx, screen_dy),
        minimum_distance=minimum_distance,
    )
    return heading


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
    return fixed_wing_hpr_from_world_tangent(
        tangent=tangent,
        roll_deg=float(bank_deg),
    )


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
    faces: list[FixedWingMeshFace] = []

    fuselage_stations = _scale_station_heights(
        (
        (-3.15, 0.07, 0.10, -0.04),
        (-2.50, 0.18, 0.20, -0.08),
        (-1.70, 0.32, 0.30, -0.14),
        (-0.35, 0.46, 0.38, -0.18),
        (1.05, 0.52, 0.34, -0.16),
        (2.20, 0.42, 0.28, -0.14),
        (3.00, 0.20, 0.18, -0.08),
        (3.42, 0.08, 0.10, -0.04),
        ),
        top_scale=1.52,
        bottom_scale=1.65,
    )
    canopy_stations = _scale_station_heights(
        (
        (0.72, 0.16, 0.46, 0.28),
        (1.42, 0.22, 0.64, 0.36),
        (2.04, 0.17, 0.54, 0.33),
        ),
        top_scale=1.56,
        bottom_scale=1.30,
        top_bias=0.08,
    )
    engine_stations = _scale_station_heights(
        (
        (-0.04, 0.16, 0.10, -0.08),
        (0.56, 0.20, 0.14, -0.12),
        (1.20, 0.16, 0.10, -0.08),
        ),
        top_scale=1.34,
        bottom_scale=1.36,
    )

    faces.extend(_loft_body(stations=fuselage_stations, role="body"))
    faces.extend(_loft_body(stations=canopy_stations, role="canopy"))
    faces.extend(_loft_body(stations=engine_stations, role="engine", x_center=-1.06))
    faces.extend(_loft_body(stations=engine_stations, role="engine", x_center=1.06))

    faces.extend(
        _prism_from_surface(
            role="body",
            top_points=(
                (-0.18, 0.36, 0.10),
                (-3.86, 0.56, 0.16),
                (-3.14, -0.54, 0.07),
                (-0.62, -0.12, 0.08),
            ),
            offset=(0.0, 0.0, -0.10),
        )
    )
    faces.extend(
        _prism_from_surface(
            role="body",
            top_points=(
                (0.18, 0.36, 0.10),
                (0.62, -0.12, 0.08),
                (3.14, -0.54, 0.07),
                (3.86, 0.56, 0.16),
            ),
            offset=(0.0, 0.0, -0.10),
        )
    )
    faces.extend(
        _prism_from_surface(
            role="body",
            top_points=(
                (-0.16, -1.94, 0.22),
                (-1.56, -2.18, 0.28),
                (-1.10, -2.78, 0.18),
                (-0.22, -2.44, 0.20),
            ),
            offset=(0.0, 0.0, -0.08),
        )
    )
    faces.extend(
        _prism_from_surface(
            role="body",
            top_points=(
                (0.16, -1.94, 0.22),
                (0.22, -2.44, 0.20),
                (1.10, -2.78, 0.18),
                (1.56, -2.18, 0.28),
            ),
            offset=(0.0, 0.0, -0.08),
        )
    )
    faces.extend(
        _prism_from_surface(
            role="accent",
            top_points=(
                (0.0, -2.42, 0.30),
                (0.0, -2.04, 0.60),
                (0.0, -1.66, 1.90),
                (0.0, -2.10, 1.12),
            ),
            offset=(0.16, 0.0, 0.0),
        )
    )

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
    forward_x_mix: float = 0.11,
    forward_y_mix: float = 0.31,
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
        shade = _face_shade(rotated)
        projected.append(
            FixedWingProjectedFace(
                role=face.role,
                points=tuple(points_2d),
                avg_depth=depth_sum / float(len(rotated)),
                shade=shade,
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
    forward_x_mix: float = 0.11,
    forward_y_mix: float = 0.31,
) -> None:
    import pygame

    paint = palette or _DEFAULT_PYGAME_INSTRUMENT_PALETTE
    role_colors = {
        "body": paint.body,
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

    roll = math.radians(bank_deg)
    cos_r = math.cos(roll)
    sin_r = math.sin(roll)
    x1 = x * cos_r + z * sin_r
    y1 = y
    z1 = -x * sin_r + z * cos_r

    pitch = math.radians(pitch_deg)
    cos_p = math.cos(pitch)
    sin_p = math.sin(pitch)
    x2 = x1
    y2 = y1 * cos_p - z1 * sin_p
    z2 = y1 * sin_p + z1 * cos_p

    yaw = math.radians(-heading_deg)
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    x3 = x2 * cos_y - y2 * sin_y
    y3 = x2 * sin_y + y2 * cos_y
    z3 = z2
    return (x3, y3, z3)


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
    forward_x_mix: float = 0.11,
    forward_y_mix: float = 0.31,
) -> tuple[int, int, float]:
    x, y, z = point
    sx = int(round(cx + (x + (y * float(forward_x_mix))) * scale))
    sy = int(round(cy - (z + (y * float(forward_y_mix))) * scale))
    return sx, sy, y


def _loft_body(
    *,
    stations: tuple[tuple[float, float, float, float], ...],
    role: str,
    x_center: float = 0.0,
) -> tuple[FixedWingMeshFace, ...]:
    faces: list[FixedWingMeshFace] = []
    for left, right in zip(stations, stations[1:], strict=False):
        y0, half_w0, z_top0, z_bottom0 = left
        y1, half_w1, z_top1, z_bottom1 = right
        left_top0 = (x_center - half_w0, y0, z_top0)
        right_top0 = (x_center + half_w0, y0, z_top0)
        left_bottom0 = (x_center - half_w0, y0, z_bottom0)
        right_bottom0 = (x_center + half_w0, y0, z_bottom0)
        left_top1 = (x_center - half_w1, y1, z_top1)
        right_top1 = (x_center + half_w1, y1, z_top1)
        left_bottom1 = (x_center - half_w1, y1, z_bottom1)
        right_bottom1 = (x_center + half_w1, y1, z_bottom1)

        faces.append(
            FixedWingMeshFace(role=role, points=(left_top0, right_top0, right_top1, left_top1))
        )
        faces.append(
            FixedWingMeshFace(
                role=role,
                points=(left_bottom0, left_bottom1, right_bottom1, right_bottom0),
            )
        )
        faces.append(
            FixedWingMeshFace(
                role=role,
                points=(left_top0, left_top1, left_bottom1, left_bottom0),
            )
        )
        faces.append(
            FixedWingMeshFace(
                role=role,
                points=(right_top0, right_bottom0, right_bottom1, right_top1),
            )
        )

    y_tail, half_w_tail, z_top_tail, z_bottom_tail = stations[0]
    y_nose, half_w_nose, z_top_nose, z_bottom_nose = stations[-1]
    faces.append(
        FixedWingMeshFace(
            role=role,
            points=(
                (x_center - half_w_tail, y_tail, z_top_tail),
                (x_center + half_w_tail, y_tail, z_top_tail),
                (x_center + half_w_tail, y_tail, z_bottom_tail),
                (x_center - half_w_tail, y_tail, z_bottom_tail),
            ),
        )
    )
    faces.append(
        FixedWingMeshFace(
            role=role,
            points=(
                (x_center - half_w_nose, y_nose, z_top_nose),
                (x_center - half_w_nose, y_nose, z_bottom_nose),
                (x_center + half_w_nose, y_nose, z_bottom_nose),
                (x_center + half_w_nose, y_nose, z_top_nose),
            ),
        )
    )
    return tuple(faces)


def _scale_station_heights(
    stations: tuple[tuple[float, float, float, float], ...],
    *,
    top_scale: float = 1.0,
    bottom_scale: float = 1.0,
    top_bias: float = 0.0,
) -> tuple[tuple[float, float, float, float], ...]:
    scaled: list[tuple[float, float, float, float]] = []
    for y, half_w, z_top, z_bottom in stations:
        scaled.append(
            (
                y,
                half_w,
                (float(z_top) * float(top_scale)) + float(top_bias),
                float(z_bottom) * float(bottom_scale),
            )
        )
    return tuple(scaled)


def _prism_from_surface(
    *,
    role: str,
    top_points: tuple[Point3, ...],
    offset: Point3,
) -> tuple[FixedWingMeshFace, ...]:
    bottom_points = tuple(
        (point[0] + offset[0], point[1] + offset[1], point[2] + offset[2]) for point in top_points
    )
    faces = [
        FixedWingMeshFace(role=role, points=top_points),
        FixedWingMeshFace(role=role, points=tuple(reversed(bottom_points))),
    ]
    for idx in range(len(top_points)):
        next_idx = (idx + 1) % len(top_points)
        faces.append(
            FixedWingMeshFace(
                role=role,
                points=(
                    top_points[idx],
                    top_points[next_idx],
                    bottom_points[next_idx],
                    bottom_points[idx],
                ),
            )
        )
    return tuple(faces)


def _face_shade(points: tuple[Point3, ...]) -> float:
    normal = _face_normal(points)
    light = _normalize((-0.42, -0.34, 0.84))
    dot = max(0.0, _dot(normal, light))
    return max(0.62, min(1.18, 0.72 + (dot * 0.42)))


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


def _shade_rgb(color: tuple[int, int, int], shade: float) -> tuple[int, int, int]:
    return tuple(max(0, min(255, int(round(channel * shade)))) for channel in color)


def _polygon_area(points: list[tuple[int, int]]) -> float:
    if len(points) < 3:
        return 0.0
    total = 0.0
    for current, nxt in zip(points, points[1:] + points[:1], strict=False):
        total += (current[0] * nxt[1]) - (nxt[0] * current[1])
    return abs(total) * 0.5

