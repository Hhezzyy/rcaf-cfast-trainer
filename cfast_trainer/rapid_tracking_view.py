from __future__ import annotations

import math
from dataclasses import dataclass


TARGET_VIEW_LIMIT = 1.3
WORLD_EXTENT_SCALE = 5.0
TERRAIN_HALF_SPAN = 260.0 * WORLD_EXTENT_SCALE
CAMERA_SWEEP_LIMIT_DEG = 66.0
CAMERA_VERTICAL_SWEEP_LIMIT_DEG = 34.0


@dataclass(frozen=True, slots=True)
class RapidTrackingCameraRigState:
    cam_world_x: float
    cam_world_y: float
    cam_world_z: float
    carrier_heading_deg: float
    heading_deg: float
    pitch_deg: float
    view_heading_deg: float
    view_pitch_deg: float
    roll_deg: float
    fov_deg: float
    orbit_weight: float
    orbit_radius: float
    altitude_agl: float
    neutral_heading_deg: float
    neutral_pitch_deg: float


@dataclass(frozen=True, slots=True)
class RapidTrackingTargetProjection:
    target_rel_x: float
    target_rel_y: float
    screen_x: float
    screen_y: float
    on_screen: bool
    in_front: bool


def rapid_tracking_seed_unit(*, seed: int, salt: str) -> float:
    total = int(seed) & 0xFFFFFFFF
    for ch in str(salt):
        total ^= ord(ch) & 0xFFFFFFFF
        total = (total * 16777619) & 0xFFFFFFFF
        total ^= (total >> 13) & 0xFFFFFFFF
    return float(total & 0xFFFFFFFF) / float(0xFFFFFFFF)


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


def _angle_delta_deg(a: float, b: float) -> float:
    return ((float(a) - float(b) + 180.0) % 360.0) - 180.0


def _layered_oscillation(
    *,
    elapsed_s: float,
    seed: int,
    salt: str,
    amplitudes: tuple[float, ...],
    frequencies: tuple[float, ...],
) -> float:
    total = 0.0
    for idx, (amp, freq) in enumerate(zip(amplitudes, frequencies, strict=True)):
        phase = rapid_tracking_seed_unit(seed=seed, salt=f"{salt}:{idx}") * math.tau
        total += math.sin((float(elapsed_s) * float(freq)) + phase) * float(amp)
    return float(total)


def _path_point(*, t: float) -> tuple[float, float]:
    u = _clamp(float(t), 0.0, 1.0)
    x = -24.0 + (68.0 * u) + (math.sin(u * math.pi * 1.6) * 12.0)
    y = -34.0 + (136.0 * u) + (math.sin((u * math.pi * 2.2) + 0.35) * 8.0)
    return float(x), float(y)


def track_to_world_xy(
    *,
    track_x: float,
    track_y: float,
    path_lateral_bias: float = 0.0,
) -> tuple[float, float]:
    world_x = (float(track_x) * 34.0) + (float(path_lateral_bias) * 14.0)
    world_y = 70.0 + ((float(track_y) + 0.18) * 56.0)
    return float(world_x), float(world_y)


def terrain_height(x: float, y: float) -> float:
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


def estimated_target_world_z(
    *,
    kind: str,
    target_world_x: float,
    target_world_y: float,
    elapsed_s: float,
    scene_progress: float,
    seed: int,
) -> float:
    terrain_z = float(terrain_height(target_world_x, target_world_y))
    target_kind = str(kind).strip().lower()
    if target_kind == "soldier":
        return terrain_z + 0.05
    if target_kind == "truck":
        return terrain_z + 0.34
    if target_kind == "building":
        return terrain_z + 1.8
    if target_kind not in {"helicopter", "jet"}:
        return terrain_z + 0.6

    phase = (
        (float(elapsed_s) * (1.4 if target_kind == "helicopter" else 2.0))
        + (float(target_world_x) * 0.018)
        + (float(target_world_y) * 0.014)
        + (int(seed) * 0.00073)
    )
    if target_kind == "helicopter":
        clearance = 9.0 + (math.sin(phase) * 1.4) + (math.cos((phase * 0.58) + 0.35) * 0.9)
        clearance += _lerp(-0.4, 0.7, _smoothstep(0.0, 1.0, float(scene_progress)))
        minimum = 6.4
    else:
        clearance = 17.0 + (math.sin(phase) * 1.9) + (math.cos((phase * 0.52) + 0.7) * 1.2)
        clearance += _lerp(0.0, 2.8, _smoothstep(0.18, 1.0, float(scene_progress)))
        minimum = 12.0
    return float(max(terrain_z + minimum, terrain_z + clearance))


def _bearing_deg(*, from_x: float, from_y: float, to_x: float, to_y: float) -> float:
    return (90.0 - math.degrees(math.atan2(float(to_y) - float(from_y), float(to_x) - float(from_x)))) % 360.0


def _pitch_deg_to_target(
    *,
    from_x: float,
    from_y: float,
    from_z: float,
    to_x: float,
    to_y: float,
    to_z: float,
) -> float:
    horizontal_distance = math.hypot(float(to_x) - float(from_x), float(to_y) - float(from_y))
    return math.degrees(math.atan2(float(to_z) - float(from_z), max(1e-3, horizontal_distance)))


def camera_rig_state(
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
    p = _clamp(float(progress), 0.0, 1.0)
    turbulence = max(0.0, float(turbulence_strength))
    seed_dir = -1.0 if rapid_tracking_seed_unit(seed=seed, salt="orbit-direction") < 0.5 else 1.0
    phase_offset = rapid_tracking_seed_unit(seed=seed, salt="orbit-phase") * math.tau
    orbit_radius_scale = 0.92 + (rapid_tracking_seed_unit(seed=seed, salt="orbit-radius") * 0.20)
    altitude_bias = _lerp(-1.0, 1.4, rapid_tracking_seed_unit(seed=seed, salt="altitude-bias"))
    path_bias = _lerp(-8.0, 8.0, rapid_tracking_seed_unit(seed=seed, salt="path-bias"))
    bob_phase_a = rapid_tracking_seed_unit(seed=seed, salt="bob-a") * math.tau
    bob_phase_b = rapid_tracking_seed_unit(seed=seed, salt="bob-b") * math.tau

    orbit_phase = phase_offset + (seed_dir * elapsed_s * 0.15) + (p * 0.24)
    orbit_radius = _lerp(58.0, 42.0, _smoothstep(0.0, 0.52, p)) * orbit_radius_scale
    orbit_local_x = math.cos(orbit_phase) * orbit_radius
    orbit_local_y = math.sin(orbit_phase) * (orbit_radius * 0.86)
    orbit_x = float(focus_world_x) + orbit_local_x
    orbit_y = float(focus_world_y) + orbit_local_y
    orbit_heading = (90.0 - math.degrees(math.atan2(-orbit_local_y, -orbit_local_x))) % 360.0

    path_transition = _smoothstep(0.56, 0.84, p)
    path_t = _smoothstep(0.60, 1.0, p)
    path_x, path_y = _path_point(t=path_t)
    next_x, next_y = _path_point(t=min(1.0, path_t + 0.015))
    path_x += path_bias + float(focus_world_x)
    next_x += path_bias + float(focus_world_x)
    path_y += float(focus_world_y) - 70.0
    next_y += float(focus_world_y) - 70.0
    path_dx = next_x - path_x
    path_dy = next_y - path_y
    if abs(path_dx) < 1e-6 and abs(path_dy) < 1e-6:
        movement_heading = orbit_heading
    else:
        movement_heading = (90.0 - math.degrees(math.atan2(path_dy, path_dx))) % 360.0

    base_x = _lerp(orbit_x, path_x, path_transition)
    base_y = _lerp(orbit_y, path_y, path_transition)
    motion_heading_rad = math.radians(movement_heading)
    forward_vec_x = math.sin(motion_heading_rad)
    forward_vec_y = math.cos(motion_heading_rad)
    motion_right_x = math.cos(motion_heading_rad)
    motion_right_y = -math.sin(motion_heading_rad)
    shake_envelope = turbulence * _lerp(1.0, 0.62, path_transition)
    lateral_shake = _layered_oscillation(
        elapsed_s=elapsed_s,
        seed=seed,
        salt="cam:lateral",
        amplitudes=(0.72, 0.34, 0.16),
        frequencies=(2.7, 6.4, 10.8),
    ) * shake_envelope
    forward_shake = _layered_oscillation(
        elapsed_s=elapsed_s,
        seed=seed,
        salt="cam:forward",
        amplitudes=(0.34, 0.18, 0.09),
        frequencies=(2.3, 5.4, 9.6),
    ) * shake_envelope
    heading_shake = _layered_oscillation(
        elapsed_s=elapsed_s,
        seed=seed,
        salt="cam:heading",
        amplitudes=(1.8, 0.82, 0.34),
        frequencies=(3.5, 8.5, 14.2),
    ) * shake_envelope
    pitch_shake = _layered_oscillation(
        elapsed_s=elapsed_s,
        seed=seed,
        salt="cam:pitch",
        amplitudes=(1.4, 0.58, 0.22),
        frequencies=(3.2, 7.8, 12.6),
    ) * shake_envelope
    vertical_shake = _layered_oscillation(
        elapsed_s=elapsed_s,
        seed=seed,
        salt="cam:vertical",
        amplitudes=(0.86, 0.34, 0.14),
        frequencies=(2.8, 6.6, 11.8),
    ) * shake_envelope
    roll_shake = _layered_oscillation(
        elapsed_s=elapsed_s,
        seed=seed,
        salt="cam:roll",
        amplitudes=(5.4, 2.2, 0.8),
        frequencies=(3.0, 7.2, 12.2),
    ) * shake_envelope
    base_x += (motion_right_x * lateral_shake) + (forward_vec_x * forward_shake)
    base_y += (motion_right_y * lateral_shake) + (forward_vec_y * forward_shake)
    carrier_heading_deg = movement_heading
    carrier_heading_rad = math.radians(carrier_heading_deg)
    right_mount = 1.35
    right_vec_x = math.cos(carrier_heading_rad)
    right_vec_y = -math.sin(carrier_heading_rad)
    cam_world_x = base_x + (right_vec_x * right_mount)
    cam_world_y = base_y + (right_vec_y * right_mount)

    ground_z = terrain_height(cam_world_x, cam_world_y)
    altitude_agl = _lerp(24.0, 5.8, _smoothstep(0.0, 1.0, p)) + altitude_bias
    bob = (
        math.sin((elapsed_s * 2.1) + bob_phase_a) * _lerp(0.18, 0.08, path_transition) * turbulence
        + math.sin((elapsed_s * 4.6) + bob_phase_b) * _lerp(0.09, 0.04, path_transition) * turbulence
    )
    cam_world_z = ground_z + altitude_agl + bob + vertical_shake + (zoom * 0.22)

    focus_terrain_z = terrain_height(focus_world_x, focus_world_y)
    neutral_heading_deg = _bearing_deg(
        from_x=cam_world_x,
        from_y=cam_world_y,
        to_x=focus_world_x,
        to_y=focus_world_y,
    )
    neutral_pitch_deg = _pitch_deg_to_target(
        from_x=cam_world_x,
        from_y=cam_world_y,
        from_z=cam_world_z,
        to_x=focus_world_x,
        to_y=focus_world_y,
        to_z=focus_terrain_z + 0.4,
    )

    heading_deg = neutral_heading_deg if camera_yaw_deg is None else float(camera_yaw_deg)
    pitch_deg = neutral_pitch_deg if camera_pitch_deg is None else float(camera_pitch_deg)
    pitch_deg = _clamp(pitch_deg, -89.0, 89.0)
    view_heading_deg = heading_deg + heading_shake
    view_pitch_deg = _clamp(pitch_deg + pitch_shake, -89.0, 89.0)

    orbit_bank = _lerp(7.5, 2.0, path_transition)
    roll_deg = orbit_bank + (
        math.sin((elapsed_s * 1.9) + 0.4)
        * _lerp(2.0, 0.9, path_transition)
        * max(0.15, turbulence)
    )
    roll_deg += roll_shake
    fov_deg = _clamp(_lerp(130.0, 39.0, zoom), 36.0, 136.0)
    orbit_weight = 1.0 - path_transition

    return RapidTrackingCameraRigState(
        cam_world_x=float(cam_world_x),
        cam_world_y=float(cam_world_y),
        cam_world_z=float(cam_world_z),
        carrier_heading_deg=float(carrier_heading_deg % 360.0),
        heading_deg=float(heading_deg % 360.0),
        pitch_deg=float(pitch_deg),
        view_heading_deg=float(view_heading_deg % 360.0),
        view_pitch_deg=float(view_pitch_deg),
        roll_deg=float(roll_deg),
        fov_deg=float(fov_deg),
        orbit_weight=float(orbit_weight),
        orbit_radius=float(orbit_radius),
        altitude_agl=float(altitude_agl),
        neutral_heading_deg=float(neutral_heading_deg % 360.0),
        neutral_pitch_deg=float(neutral_pitch_deg),
    )


def world_to_camera_space(
    *,
    cam_world_x: float,
    cam_world_y: float,
    cam_world_z: float,
    heading_deg: float,
    pitch_deg: float,
    target_world_x: float,
    target_world_y: float,
    target_world_z: float,
) -> tuple[float, float, float]:
    dx = float(target_world_x) - float(cam_world_x)
    dy = float(target_world_y) - float(cam_world_y)
    dz = float(target_world_z) - float(cam_world_z)

    heading_rad = math.radians(float(heading_deg))
    pitch_rad = math.radians(float(pitch_deg))

    right_x = math.cos(heading_rad)
    right_y = -math.sin(heading_rad)
    right_z = 0.0

    forward_x = math.sin(heading_rad) * math.cos(pitch_rad)
    forward_y = math.cos(heading_rad) * math.cos(pitch_rad)
    forward_z = math.sin(pitch_rad)

    up_x = (right_y * forward_z) - (right_z * forward_y)
    up_y = (right_z * forward_x) - (right_x * forward_z)
    up_z = (right_x * forward_y) - (right_y * forward_x)

    cam_x = (dx * right_x) + (dy * right_y) + (dz * right_z)
    cam_y = (dx * forward_x) + (dy * forward_y) + (dz * forward_z)
    cam_z = (dx * up_x) + (dy * up_y) + (dz * up_z)
    return float(cam_x), float(cam_y), float(cam_z)


def camera_space_to_viewport(
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


def target_projection(
    *,
    rig: RapidTrackingCameraRigState,
    target_kind: str,
    target_world_x: float,
    target_world_y: float,
    elapsed_s: float,
    scene_progress: float,
    seed: int,
    size: tuple[int, int] = (1, 1),
) -> RapidTrackingTargetProjection:
    target_world_z = estimated_target_world_z(
        kind=target_kind,
        target_world_x=target_world_x,
        target_world_y=target_world_y,
        elapsed_s=elapsed_s,
        scene_progress=scene_progress,
        seed=seed,
    )
    cam_x, cam_y, cam_z = world_to_camera_space(
        cam_world_x=rig.cam_world_x,
        cam_world_y=rig.cam_world_y,
        cam_world_z=rig.cam_world_z,
        heading_deg=rig.view_heading_deg,
        pitch_deg=rig.view_pitch_deg,
        target_world_x=target_world_x,
        target_world_y=target_world_y,
        target_world_z=target_world_z,
    )
    screen_x, screen_y, on_screen, in_front = camera_space_to_viewport(
        cam_x=cam_x,
        cam_y=cam_y,
        cam_z=cam_z,
        size=size,
        h_fov_deg=rig.fov_deg,
        v_fov_deg=max(18.0, rig.fov_deg * 0.78),
    )

    depth = max(1e-3, abs(cam_y))
    tan_h = max(1e-4, math.tan(math.radians(rig.fov_deg * 0.5)))
    tan_v = max(1e-4, math.tan(math.radians(max(18.0, rig.fov_deg * 0.78) * 0.5)))
    rel_x = (cam_x / (depth * tan_h)) * TARGET_VIEW_LIMIT
    rel_y = (-cam_z / (depth * tan_v)) * TARGET_VIEW_LIMIT
    if not in_front:
        rel_x = math.copysign(abs(rel_x) + TARGET_VIEW_LIMIT, rel_x if abs(rel_x) > 1e-6 else 1.0)
        rel_y = math.copysign(abs(rel_y) + (TARGET_VIEW_LIMIT * 0.5), rel_y if abs(rel_y) > 1e-6 else 1.0)

    return RapidTrackingTargetProjection(
        target_rel_x=float(rel_x),
        target_rel_y=float(rel_y),
        screen_x=float(screen_x),
        screen_y=float(screen_y),
        on_screen=bool(on_screen),
        in_front=bool(in_front),
    )


def camera_pose_compat(
    *,
    heading_deg: float,
    pitch_deg: float,
    neutral_heading_deg: float,
    neutral_pitch_deg: float,
) -> tuple[float, float]:
    camera_x = _angle_delta_deg(float(heading_deg), float(neutral_heading_deg)) / (
        CAMERA_SWEEP_LIMIT_DEG / 5.2
    )
    camera_y = -(float(pitch_deg) - float(neutral_pitch_deg)) / 9.6
    return float(camera_x), float(camera_y)
