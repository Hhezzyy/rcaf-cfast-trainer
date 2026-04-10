from __future__ import annotations

import math

from .cognitive_core import clamp01

Point3 = tuple[float, float, float]

TUBE_PATH_POINTS: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.00, 0.00),
    (10.0, 0.00, 0.00),
    (22.0, 0.20, 0.04),
    (34.0, 2.10, 0.48),
    (44.0, 4.40, 0.72),
    (54.0, 4.80, 0.18),
    (64.0, 4.65, -1.10),
    (74.0, 2.40, -2.00),
    (84.0, -0.60, -2.20),
    (94.0, -3.45, -1.55),
    (104.0, -4.85, -0.10),
    (114.0, -4.70, 1.10),
    (124.0, -2.10, 2.04),
    (134.0, 1.10, 2.20),
    (144.0, 3.70, 1.20),
    (154.0, 4.60, -0.22),
    (164.0, 4.15, -1.34),
    (174.0, 2.30, -0.60),
    (184.0, 0.55, -0.08),
    (194.0, 0.00, 0.00),
)
TUBE_PATH_SPAN = float(TUBE_PATH_POINTS[-1][0])

GATE_DEPTH_SLOTS_NORM: tuple[float, ...] = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.88, 0.93)
BALL_FORWARD_START_NORM = 0.14
BALL_FORWARD_IDLE_NORM = 0.18
TUNNEL_EXIT_NORM = 1.0

TUNNEL_ENTRANCE_DISTANCE = 4.0
TUNNEL_PRESENTATION_SPAN = 40.0
TUNNEL_EXIT_DISTANCE = TUNNEL_ENTRANCE_DISTANCE + TUNNEL_PRESENTATION_SPAN
TUNNEL_GEOMETRY_START_DISTANCE = 0.0
TUNNEL_GEOMETRY_END_DISTANCE = TUNNEL_EXIT_DISTANCE + 4.0
TUNNEL_CAMERA_DISTANCE = 1.2
TUNNEL_CAMERA_LOOK_DISTANCE = TUNNEL_EXIT_DISTANCE - 0.8
TUNNEL_CAMERA_FOLLOW_BACK_DISTANCE = 10.0
TUNNEL_CAMERA_LOOK_AHEAD_DISTANCE = 14.0
TUNNEL_CAMERA_UP_OFFSET = 0.10
TUNNEL_CAMERA_TARGET_FORWARD_OFFSET = 0.45
TUNNEL_CAMERA_TARGET_UP_OFFSET = 0.04
TUNNEL_CAMERA_H_FOV_DEG = 50.0
TUNNEL_CAMERA_V_FOV_DEG = 38.0
AUDITORY_RUN_TRAVEL_SPEED_DISTANCE_PER_S = 0.16
AUDITORY_RUN_MAX_SCORING_DURATION_S = 13.0 * 60.0
AUDITORY_RUN_TRAVEL_MARGIN_DISTANCE = 2.0


def smoothstep01(value: float) -> float:
    t = clamp01(value)
    return t * t * (3.0 - (2.0 * t))


def lerp(a: float, b: float, t: float) -> float:
    return float(a) + ((float(b) - float(a)) * float(t))


def catmull_rom(a: float, b: float, c: float, d: float, t: float) -> float:
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        (2.0 * b)
        + ((-a + c) * t)
        + (((2.0 * a) - (5.0 * b) + (4.0 * c) - d) * t2)
        + (((-a) + (3.0 * b) - (3.0 * c) + d) * t3)
    )


def vec_add(a: Point3, b: Point3) -> Point3:
    return (float(a[0]) + float(b[0]), float(a[1]) + float(b[1]), float(a[2]) + float(b[2]))


def vec_scale(v: Point3, scalar: float) -> Point3:
    return (float(v[0]) * float(scalar), float(v[1]) * float(scalar), float(v[2]) * float(scalar))


def vec_dot(a: Point3, b: Point3) -> float:
    return (float(a[0]) * float(b[0])) + (float(a[1]) * float(b[1])) + (float(a[2]) * float(b[2]))


def vec_cross(a: Point3, b: Point3) -> Point3:
    return (
        (float(a[1]) * float(b[2])) - (float(a[2]) * float(b[1])),
        (float(a[2]) * float(b[0])) - (float(a[0]) * float(b[2])),
        (float(a[0]) * float(b[1])) - (float(a[1]) * float(b[0])),
    )


def vec_norm(v: Point3) -> float:
    return math.sqrt(vec_dot(v, v))


def vec_normalize(v: Point3) -> Point3:
    n = vec_norm(v)
    if n <= 1e-6:
        return (0.0, 0.0, 0.0)
    return (float(v[0]) / n, float(v[1]) / n, float(v[2]) / n)


def tube_center_at_distance(distance: float, *, span: float = TUBE_PATH_SPAN) -> tuple[float, float]:
    path_span = max(0.001, float(span))
    d = float(distance) % path_span
    unique_count = max(1, len(TUBE_PATH_POINTS) - 1)
    segment_idx = len(TUBE_PATH_POINTS) - 2
    for idx in range(len(TUBE_PATH_POINTS) - 1):
        d1, _x1, _z1 = TUBE_PATH_POINTS[idx]
        d2, _x2, _z2 = TUBE_PATH_POINTS[idx + 1]
        if d <= d2:
            segment_idx = idx
            break
    d1, x1, z1 = TUBE_PATH_POINTS[segment_idx]
    d2, x2, z2 = TUBE_PATH_POINTS[segment_idx + 1]
    p0 = TUBE_PATH_POINTS[(segment_idx - 1) % unique_count]
    p1 = (d1, x1, z1)
    p2 = (d2, x2, z2)
    p3 = TUBE_PATH_POINTS[(segment_idx + 2) % unique_count]
    t = 0.0 if d2 <= d1 else (d - d1) / (d2 - d1)
    return (
        catmull_rom(p0[1], p1[1], p2[1], p3[1], t),
        catmull_rom(p0[2], p1[2], p2[2], p3[2], t),
    )


def tube_frame(
    distance: float,
    *,
    span: float = TUBE_PATH_SPAN,
) -> tuple[Point3, Point3, Point3, Point3]:
    d = float(distance)
    eps = 0.12
    cx, cz = tube_center_at_distance(d, span=span)
    prev_x, prev_z = tube_center_at_distance(d - eps, span=span)
    next_x, next_z = tube_center_at_distance(d + eps, span=span)
    tangent = vec_normalize((next_x - prev_x, 2.0 * eps, next_z - prev_z))
    world_up = (0.0, 0.0, 1.0)
    right = vec_cross(tangent, world_up)
    if vec_norm(right) <= 1e-5:
        right = (1.0, 0.0, 0.0)
    right = vec_normalize(right)
    up = vec_normalize(vec_cross(right, tangent))
    return ((cx, d, cz), tangent, right, up)


def forward_norm_to_distance(forward_norm: float) -> float:
    clamped = clamp01(forward_norm)
    return TUNNEL_ENTRANCE_DISTANCE + (clamped * TUNNEL_PRESENTATION_SPAN)


def slot_forward_norm(slot_index: int) -> float:
    idx = max(0, min(len(GATE_DEPTH_SLOTS_NORM) - 1, int(slot_index)))
    return float(GATE_DEPTH_SLOTS_NORM[idx])


def slot_distance(slot_index: int) -> float:
    return forward_norm_to_distance(slot_forward_norm(slot_index))


AUDITORY_BALL_ANCHOR_DISTANCE = forward_norm_to_distance(BALL_FORWARD_IDLE_NORM)
AUDITORY_GATE_FAR_DISTANCE = slot_distance(len(GATE_DEPTH_SLOTS_NORM) - 1)
AUDITORY_GATE_BEHIND_DISTANCE = max(
    TUNNEL_GEOMETRY_START_DISTANCE + 1.0,
    AUDITORY_BALL_ANCHOR_DISTANCE - 6.0,
)
AUDITORY_RUN_MAX_START_DISTANCE = max(
    0.0,
    TUBE_PATH_SPAN
    - TUNNEL_GEOMETRY_END_DISTANCE
    - (AUDITORY_RUN_TRAVEL_SPEED_DISTANCE_PER_S * AUDITORY_RUN_MAX_SCORING_DURATION_S)
    - AUDITORY_RUN_TRAVEL_MARGIN_DISTANCE,
)


def _seed_unit_float(seed: int) -> float:
    value = int(seed) & 0xFFFFFFFF
    value ^= value >> 16
    value = (value * 0x7FEB352D) & 0xFFFFFFFF
    value ^= value >> 15
    value = (value * 0x846CA68B) & 0xFFFFFFFF
    value ^= value >> 16
    return float(value) / float(0xFFFFFFFF)


def run_start_distance(session_seed: int, *, span: float = TUBE_PATH_SPAN) -> float:
    usable = min(
        max(0.0, float(AUDITORY_RUN_MAX_START_DISTANCE)),
        max(
            0.0,
            float(span) - TUNNEL_GEOMETRY_END_DISTANCE - AUDITORY_RUN_TRAVEL_MARGIN_DISTANCE,
        ),
    )
    if usable <= 1e-6:
        return 0.0
    return usable * _seed_unit_float(int(session_seed))


def run_travel_distance(
    *,
    session_seed: int,
    phase_elapsed_s: float,
    span: float = TUBE_PATH_SPAN,
) -> float:
    max_visible_start = max(
        0.0,
        float(span) - TUNNEL_GEOMETRY_END_DISTANCE - AUDITORY_RUN_TRAVEL_MARGIN_DISTANCE,
    )
    start = min(max_visible_start, run_start_distance(int(session_seed), span=span))
    distance = start + (
        max(0.0, float(phase_elapsed_s)) * AUDITORY_RUN_TRAVEL_SPEED_DISTANCE_PER_S
    )
    return min(distance, max_visible_start)


def gate_distance_from_x_norm(
    x_norm: float,
    *,
    travel_distance: float,
    spawn_x_norm: float,
    player_x_norm: float,
    retire_x_norm: float,
) -> float:
    spawn = float(spawn_x_norm)
    player = float(player_x_norm)
    retire = float(retire_x_norm)
    x = float(x_norm)
    if x >= player:
        span = max(1e-6, spawn - player)
        t = clamp01((spawn - x) / span)
        relative = lerp(AUDITORY_GATE_FAR_DISTANCE, AUDITORY_BALL_ANCHOR_DISTANCE, t)
    else:
        span = max(1e-6, player - retire)
        t = clamp01((player - x) / span)
        relative = lerp(AUDITORY_BALL_ANCHOR_DISTANCE, AUDITORY_GATE_BEHIND_DISTANCE, t)
    return float(travel_distance) + relative


def gate_depth_ratio_from_distance(
    distance: float,
    *,
    travel_distance: float,
) -> float:
    near = float(travel_distance) + AUDITORY_BALL_ANCHOR_DISTANCE
    far = float(travel_distance) + AUDITORY_GATE_FAR_DISTANCE
    span = max(1e-6, far - near)
    return clamp01((float(distance) - near) / span)


def fixed_camera_pose_at_distance(ball_distance: float) -> tuple[Point3, Point3]:
    cam_distance = max(
        TUNNEL_CAMERA_DISTANCE,
        min(
            TUNNEL_GEOMETRY_END_DISTANCE - 6.0,
            float(ball_distance) - TUNNEL_CAMERA_FOLLOW_BACK_DISTANCE,
        ),
    )
    look_distance = max(
        cam_distance + 4.0,
        min(
            TUNNEL_GEOMETRY_END_DISTANCE - 0.4,
            float(ball_distance) + TUNNEL_CAMERA_LOOK_AHEAD_DISTANCE,
        ),
    )
    cam_center, _cam_tangent, _cam_right, cam_up = tube_frame(cam_distance)
    look_center, tangent, _look_right, look_up = tube_frame(look_distance)
    cam_pos = vec_add(cam_center, vec_scale(cam_up, TUNNEL_CAMERA_UP_OFFSET))
    look_target = vec_add(
        look_center,
        vec_add(
            vec_scale(tangent, TUNNEL_CAMERA_TARGET_FORWARD_OFFSET),
            vec_scale(look_up, TUNNEL_CAMERA_TARGET_UP_OFFSET),
        ),
    )
    return cam_pos, look_target


def fixed_camera_pose(
    *,
    forward_norm: float = BALL_FORWARD_IDLE_NORM,
) -> tuple[Point3, Point3]:
    return fixed_camera_pose_at_distance(forward_norm_to_distance(forward_norm))
