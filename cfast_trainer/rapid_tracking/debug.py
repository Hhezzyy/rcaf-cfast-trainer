from __future__ import annotations

from dataclasses import dataclass

from ..cognitive_core import Phase, TestSnapshot
from .entities import RapidTrackingPayload


@dataclass(slots=True)
class RapidTrackingDebugState:
    overlay_enabled: bool = False
    diagnostics_enabled: bool = False
    last_reset_seed: int | None = None
    last_reset_reason: str = "launch"
    reset_count: int = 0
    viewport_size: tuple[int, int] = (0, 0)
    last_update_dt_s: float = 0.0
    active_mode: str = "instructions"
    route_id: str = ""
    rail_id: str = ""


def rapid_tracking_debug_lines(
    *,
    snap: TestSnapshot,
    state: RapidTrackingDebugState,
) -> tuple[str, ...]:
    payload = snap.payload if isinstance(snap.payload, RapidTrackingPayload) else None
    fps = 0.0 if state.last_update_dt_s <= 1e-6 else 1.0 / float(state.last_update_dt_s)
    lines = [
        f"phase={snap.phase.value} mode={state.active_mode}",
        (
            f"viewport={state.viewport_size[0]}x{state.viewport_size[1]} "
            f"dt={state.last_update_dt_s:.4f}s fps={fps:.1f}"
        ),
        f"reset={state.reset_count} last={state.last_reset_reason} seed={state.last_reset_seed}",
    ]
    if payload is None:
        return tuple(lines)

    lines.extend(
        (
            (
                f"session_seed={payload.session_seed} scene_seed={payload.scene_seed} "
                f"scene_progress={payload.scene_progress:.3f}"
            ),
            f"route={state.route_id or payload.segment_label} rail={state.rail_id or 'n/a'}",
            (
                f"target={payload.target_kind}/{payload.target_variant} "
                f"visible={int(payload.target_visible)} "
                f"moving={int(payload.target_is_moving)} cover={payload.target_cover_state}"
            ),
            (
                f"target_rel=({payload.target_rel_x:+.3f},{payload.target_rel_y:+.3f}) "
                f"reticle=({payload.reticle_x:+.3f},{payload.reticle_y:+.3f})"
            ),
            (
                f"camera=({payload.camera_x:+.3f},{payload.camera_y:+.3f}) "
                f"yaw={payload.camera_yaw_deg:.1f} pitch={payload.camera_pitch_deg:.1f}"
            ),
            (
                f"capture box=({payload.capture_box_half_width:.3f},"
                f"{payload.capture_box_half_height:.3f}) "
                f"zoom={payload.capture_zoom:.2f} lock={payload.lock_progress:.2f}"
            ),
            (
                f"error mean={payload.mean_error:.3f} rms={payload.rms_error:.3f} "
                f"on_target={payload.on_target_ratio:.3f}"
            ),
        )
    )
    if state.diagnostics_enabled and snap.phase in (Phase.PRACTICE, Phase.SCORED):
        lines.extend(
            (
                (
                    f"focus=({payload.focus_world_x:.1f},{payload.focus_world_y:.1f}) "
                    f"world=({payload.target_world_x:.1f},{payload.target_world_y:.1f})"
                ),
                (
                    f"target_vel=({payload.target_vx:+.2f},{payload.target_vy:+.2f}) "
                    f"turbulence={payload.turbulence_strength:.2f} "
                    f"assist={payload.camera_assist_strength:.2f}"
                ),
                (
                    f"segment={payload.segment_label} "
                    f"{payload.segment_index}/{payload.segment_total} "
                    f"rem={payload.segment_time_remaining_s:.1f}s"
                ),
            )
        )
    return tuple(lines)


__all__ = [
    "RapidTrackingDebugState",
    "rapid_tracking_debug_lines",
]
