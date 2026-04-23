from __future__ import annotations

import math
from dataclasses import dataclass

from .auditory_capacity_view import (
    AUDITORY_BALL_ANCHOR_DISTANCE,
    AUDITORY_GATE_BEHIND_DISTANCE_OFFSET,
    AUDITORY_GATE_FAR_DISTANCE_OFFSET,
    Point3,
    lerp,
    run_start_distance,
    tube_frame,
    vec_add,
    vec_dot,
    vec_scale,
)


@dataclass(frozen=True, slots=True)
class AuditorySphereBound:
    center: Point3
    radius: float


@dataclass(frozen=True, slots=True)
class AuditoryTunnelWallBound:
    center: Point3
    tangent: Point3
    right: Point3
    up: Point3
    half_length: float
    inner_rx: float
    inner_rz: float


@dataclass(frozen=True, slots=True)
class AuditoryTunnelFollowerConfig:
    session_seed: int
    speed_distance_per_s: float
    ball_anchor_distance: float = AUDITORY_BALL_ANCHOR_DISTANCE
    ball_radius: float = 0.11
    tunnel_inner_rx: float = 2.24
    tunnel_inner_rz: float = 1.64
    wall_half_length: float = 4.0
    roll_deg_per_distance: float = 9.0


@dataclass(frozen=True, slots=True)
class AuditoryTunnelFollowerSnapshot:
    session_seed: int
    travel_distance: float
    ball_distance: float
    center: Point3
    tangent: Point3
    right: Point3
    up: Point3
    ball_roll_deg: float
    collision_penalties: int = 0
    collision_active: bool = False
    ball_contact_ratio: float = 0.0


class AuditoryTunnelFollower:
    def __init__(self, config: AuditoryTunnelFollowerConfig) -> None:
        self._config = config
        self._travel_distance = run_start_distance(int(config.session_seed))
        self._collision_penalties = 0
        self._collision_active = False
        self._ball_contact_ratio = 0.0

    @property
    def config(self) -> AuditoryTunnelFollowerConfig:
        return self._config

    @property
    def travel_distance(self) -> float:
        return float(self._travel_distance)

    def reset(self) -> AuditoryTunnelFollowerSnapshot:
        self._travel_distance = run_start_distance(int(self._config.session_seed))
        self._collision_penalties = 0
        self._collision_active = False
        self._ball_contact_ratio = 0.0
        return self.snapshot()

    def update(self, dt_s: float) -> AuditoryTunnelFollowerSnapshot:
        dt = max(0.0, float(dt_s))
        self._travel_distance += max(0.0, float(self._config.speed_distance_per_s)) * dt
        return self.snapshot()

    def set_collision_state(
        self,
        *,
        active: bool,
        contact_ratio: float,
    ) -> AuditoryTunnelFollowerSnapshot:
        active = bool(active)
        if active and not self._collision_active:
            self._collision_penalties += 1
        self._collision_active = active
        self._ball_contact_ratio = max(0.0, float(contact_ratio))
        return self.snapshot()

    def snapshot(self) -> AuditoryTunnelFollowerSnapshot:
        ball_distance = float(self._travel_distance)
        center, tangent, right, up = tube_frame(ball_distance)
        roll = auditory_ball_roll_deg(
            ball_distance=ball_distance,
            roll_deg_per_distance=float(self._config.roll_deg_per_distance),
        )
        return AuditoryTunnelFollowerSnapshot(
            session_seed=int(self._config.session_seed),
            travel_distance=float(self._travel_distance),
            ball_distance=float(ball_distance),
            center=center,
            tangent=tangent,
            right=right,
            up=up,
            ball_roll_deg=float(roll),
            collision_penalties=int(self._collision_penalties),
            collision_active=bool(self._collision_active),
            ball_contact_ratio=float(self._ball_contact_ratio),
        )


def auditory_ball_roll_deg(
    *,
    ball_distance: float,
    roll_deg_per_distance: float,
) -> float:
    return (float(ball_distance) * float(roll_deg_per_distance)) % 360.0


def auditory_travel_speed_from_gate_speed(
    *,
    gate_speed_x_norm_per_s: float,
    spawn_x_norm: float,
    player_x_norm: float,
) -> float:
    x_span = max(1e-6, float(spawn_x_norm) - float(player_x_norm))
    distance_span = float(AUDITORY_GATE_FAR_DISTANCE_OFFSET)
    return max(0.0, float(gate_speed_x_norm_per_s)) * (distance_span / x_span)


def auditory_gate_world_distance_from_x_norm(
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
        t = (spawn - x) / span
        relative = lerp(AUDITORY_GATE_FAR_DISTANCE_OFFSET, 0.0, t)
    else:
        span = max(1e-6, player - retire)
        t = (player - x) / span
        relative = lerp(0.0, AUDITORY_GATE_BEHIND_DISTANCE_OFFSET, t)
    return float(travel_distance) + float(relative)


def auditory_gate_x_norm_from_world_distance(
    world_distance: float,
    *,
    travel_distance: float,
    spawn_x_norm: float,
    player_x_norm: float,
    retire_x_norm: float,
) -> float:
    spawn = float(spawn_x_norm)
    player = float(player_x_norm)
    retire = float(retire_x_norm)
    relative = float(world_distance) - float(travel_distance)
    if relative >= 0.0:
        span = max(1e-6, float(AUDITORY_GATE_FAR_DISTANCE_OFFSET))
        t = (float(AUDITORY_GATE_FAR_DISTANCE_OFFSET) - relative) / span
        return lerp(spawn, player, t)
    span = max(1e-6, 0.0 - float(AUDITORY_GATE_BEHIND_DISTANCE_OFFSET))
    t = (0.0 - relative) / span
    return lerp(player, retire, t)


def auditory_tunnel_wall_bound(
    snapshot: AuditoryTunnelFollowerSnapshot,
    *,
    half_length: float,
    inner_rx: float,
    inner_rz: float,
) -> AuditoryTunnelWallBound:
    return AuditoryTunnelWallBound(
        center=snapshot.center,
        tangent=snapshot.tangent,
        right=snapshot.right,
        up=snapshot.up,
        half_length=max(0.01, float(half_length)),
        inner_rx=max(0.05, float(inner_rx)),
        inner_rz=max(0.05, float(inner_rz)),
    )


def auditory_ball_bound(
    snapshot: AuditoryTunnelFollowerSnapshot,
    *,
    x: float,
    y: float,
    tube_half_width: float,
    tube_half_height: float,
    radius: float,
    inner_rx: float,
    inner_rz: float,
) -> AuditorySphereBound:
    half_w = max(1e-6, float(tube_half_width))
    half_h = max(1e-6, float(tube_half_height))
    ball_radius = max(0.0, float(radius))
    local_x = (float(x) / half_w) * max(0.05, float(inner_rx) - ball_radius)
    local_z = (float(y) / half_h) * max(0.05, float(inner_rz) - ball_radius)
    center = vec_add(
        snapshot.center,
        vec_add(vec_scale(snapshot.right, local_x), vec_scale(snapshot.up, local_z)),
    )
    return AuditorySphereBound(center=center, radius=ball_radius)


def auditory_wall_contact_ratio(
    *,
    ball_bound: AuditorySphereBound,
    wall_bound: AuditoryTunnelWallBound,
) -> float:
    rel = (
        float(ball_bound.center[0]) - float(wall_bound.center[0]),
        float(ball_bound.center[1]) - float(wall_bound.center[1]),
        float(ball_bound.center[2]) - float(wall_bound.center[2]),
    )
    along = vec_dot(rel, wall_bound.tangent)
    if abs(along) > (float(wall_bound.half_length) + float(ball_bound.radius)):
        return 0.0
    local_x = vec_dot(rel, wall_bound.right)
    local_z = vec_dot(rel, wall_bound.up)
    inner_rx = max(0.05, float(wall_bound.inner_rx) - float(ball_bound.radius))
    inner_rz = max(0.05, float(wall_bound.inner_rz) - float(ball_bound.radius))
    return math.sqrt(((local_x / inner_rx) ** 2) + ((local_z / inner_rz) ** 2))


def auditory_wall_collision_active(
    *,
    ball_bound: AuditorySphereBound,
    wall_bound: AuditoryTunnelWallBound,
) -> bool:
    return auditory_wall_contact_ratio(ball_bound=ball_bound, wall_bound=wall_bound) >= 1.0
