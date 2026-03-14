from __future__ import annotations

import math
from dataclasses import dataclass

from .rapid_tracking import RapidTrackingPayload
from .rapid_tracking_view import (
    CAMERA_SWEEP_LIMIT_DEG as _CAMERA_SWEEP_LIMIT_DEG,
    TARGET_VIEW_LIMIT as _TARGET_VIEW_LIMIT,
    TERRAIN_HALF_SPAN as _TERRAIN_HALF_SPAN,
    WORLD_EXTENT_SCALE as _WORLD_EXTENT_SCALE,
    RapidTrackingCameraRigState,
    camera_rig_state,
    camera_space_to_viewport,
    target_projection,
)

_BUILDING_SCREEN_POOLS: dict[str, tuple[tuple[str, float, float], ...]] = {
    "hangar": (
        ("hangar-a", -1.10, 0.54),
        ("hangar-b", -0.22, 0.12),
        ("hangar-c", 0.20, -0.22),
    ),
    "tower": (
        ("tower-a", -0.64, 0.06),
        ("tower-b", 0.16, -0.28),
        ("tower-c", 0.82, -0.42),
    ),
}


@dataclass(frozen=True, slots=True)
class RapidTrackingOverlayState:
    screen_x: float
    screen_y: float
    on_screen: bool
    in_front: bool
    target_visible: bool


@dataclass(frozen=True, slots=True)
class RapidTrackingSceneTarget:
    kind: str
    variant: str
    source: str
    scenery_id: str | None
    overlay: RapidTrackingOverlayState


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _loop01(value: float) -> float:
    return float(value) - math.floor(float(value))


def _triangle01(value: float) -> float:
    phase = _loop01(value)
    if phase <= 0.5:
        return phase * 2.0
    return 2.0 - (phase * 2.0)


def ground_route_pose(
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


def normalized_building_variant(variant: str) -> str:
    return "tower" if str(variant).strip().lower() == "tower" else "hangar"


def overlay_from_target_rel(
    *,
    target_rel_x: float,
    target_rel_y: float,
    size: tuple[int, int],
    target_visible: bool = True,
) -> RapidTrackingOverlayState:
    width = max(1, int(size[0]))
    height = max(1, int(size[1]))
    clamped_x = _clamp(float(target_rel_x), -_TARGET_VIEW_LIMIT, _TARGET_VIEW_LIMIT)
    clamped_y = _clamp(float(target_rel_y), -_TARGET_VIEW_LIMIT, _TARGET_VIEW_LIMIT)
    screen_x = ((clamped_x + _TARGET_VIEW_LIMIT) / (_TARGET_VIEW_LIMIT * 2.0)) * width
    screen_y = ((clamped_y + _TARGET_VIEW_LIMIT) / (_TARGET_VIEW_LIMIT * 2.0)) * height
    on_screen = 0.0 <= screen_x <= width and 0.0 <= screen_y <= height
    return RapidTrackingOverlayState(
        screen_x=float(screen_x),
        screen_y=float(screen_y),
        on_screen=bool(on_screen),
        in_front=True,
        target_visible=bool(target_visible),
    )


def select_building_scenery(
    *,
    variant: str,
    target_rel_x: float,
    target_rel_y: float,
) -> str:
    normalized = normalized_building_variant(variant)
    candidates = _BUILDING_SCREEN_POOLS[normalized]
    best_id = candidates[0][0]
    best_dist = float("inf")
    for scenery_id, anchor_x, anchor_y in candidates:
        dx = float(target_rel_x) - anchor_x
        dy = float(target_rel_y) - anchor_y
        dist = (dx * dx) + (dy * dy)
        if dist < best_dist:
            best_id = scenery_id
            best_dist = dist
    return best_id


def build_scene_target(
    *,
    payload: RapidTrackingPayload,
    size: tuple[int, int],
) -> RapidTrackingSceneTarget:
    kind = str(payload.target_kind).strip().lower()
    variant = str(payload.target_variant).strip().lower()
    overlay = overlay_from_target_rel(
        target_rel_x=float(payload.target_rel_x),
        target_rel_y=float(payload.target_rel_y),
        size=size,
        target_visible=bool(payload.target_visible),
    )
    if kind == "building":
        return RapidTrackingSceneTarget(
            kind=kind,
            variant=normalized_building_variant(variant),
            source="scenery",
            scenery_id=select_building_scenery(
                variant=variant,
                target_rel_x=float(payload.target_rel_x),
                target_rel_y=float(payload.target_rel_y),
            ),
            overlay=overlay,
        )
    return RapidTrackingSceneTarget(
        kind=kind,
        variant=variant,
        source="dynamic",
        scenery_id=None,
        overlay=overlay,
    )
