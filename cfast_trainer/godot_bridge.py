"""Local UDP bridge for the optional Godot companion renderer.

Current 3D tests run as Godot-authoritative live scenes: Python sends a
deterministic start spec and receives sanitized lifecycle/result packets back.
Legacy snapshot serializers remain for non-normal-use compatibility tests.
"""

from __future__ import annotations

import errno
import json
import math
import os
import shutil
import socket
import subprocess
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .auditory_capacity import (
    _AUDITORY_BALL_BOUND_RADIUS,
    _AUDITORY_TUNNEL_INNER_RX,
    _AUDITORY_TUNNEL_INNER_RZ,
    AUDITORY_GATE_PLAYER_X_NORM,
    AUDITORY_GATE_RETIRE_X_NORM,
    AUDITORY_GATE_SPAWN_X_NORM,
    AuditoryCapacityConfig,
    AuditoryCapacityGate,
    AuditoryCapacityPayload,
)
from .auditory_capacity_motion import auditory_gate_world_distance_from_x_norm
from .auditory_capacity_view import (
    AUDITORY_GATE_BEHIND_DISTANCE_OFFSET,
    AUDITORY_GATE_FAR_DISTANCE_OFFSET,
    AUDITORY_TUNNEL_GEOMETRY_END_OFFSET_DISTANCE,
    Point3,
    tube_frame,
    tube_twist_angle,
    vec_add,
    vec_scale,
)
from .cognitive_core import TestSnapshot
from .godot_owned import GodotOwnedPayload
from .rapid_tracking.entities import RapidTrackingPayload
from .spatial_integration import SpatialIntegrationPayload, SpatialIntegrationPoint
from .trace_test_1 import TraceTest1Payload, TraceTest1SceneFrame
from .trace_test_2 import TraceTest2AircraftTrack, TraceTest2Payload, trace_test_2_track_position

GODOT_BACKEND_NAME = "godot_4"
GODOT_DEFAULT_BIN = "/Applications/Godot.app/Contents/MacOS/Godot"
GODOT_PROJECT_PATH = Path(__file__).resolve().parent.parent / "godot" / "cfast_3d"
AUDITORY_CAPACITY_AUDIO_ASSET_PATH = (
    Path(__file__).resolve().parent.parent / "assets" / "audio" / "auditory_capacity"
)
GODOT_HOST = "127.0.0.1"
GODOT_WINDOW_RESOLUTION = "960x540"
GODOT_MAX_FPS = "60"
GODOT_SCHEMA_VERSION = 1
GODOT_WINDOW_MODE_ENV = "CFAST_GODOT_WINDOW_MODE"
AUDITORY_GODOT_WORLD_SCALE = 0.42
AUDITORY_GODOT_Y_OFFSET = 1.15

KIND_AUDITORY_CAPACITY = "auditory_capacity"
KIND_RAPID_TRACKING = "rapid_tracking"
KIND_SPATIAL_INTEGRATION = "spatial_integration"
KIND_TRACE_TEST_1 = "trace_test_1"
KIND_TRACE_TEST_2 = "trace_test_2"

GODOT_TARGET_KINDS = {
    KIND_AUDITORY_CAPACITY,
    KIND_RAPID_TRACKING,
    KIND_SPATIAL_INTEGRATION,
    KIND_TRACE_TEST_1,
    KIND_TRACE_TEST_2,
}

GODOT_CONTROL_COMMANDS = {
    "activate_action",
    "activate_setting",
    "adjust_setting",
    "auditory_complete",
    "auditory_error",
    "auditory_event",
    "auditory_progress",
    "auditory_ready",
    "back_to_tests",
    "complete",
    "error",
    "event",
    "godot_complete",
    "godot_error",
    "godot_event",
    "godot_phase_advance",
    "godot_phase_complete",
    "godot_progress",
    "godot_ready",
    "progress",
    "ready",
    "main_menu",
    "menu_back",
    "menu_down",
    "menu_left",
    "menu_right",
    "menu_select",
    "menu_up",
    "pause_toggle",
    "resume",
    "set_window_mode",
    "settings_back",
}

GODOT_PERFORMANCE_DEFAULTS: dict[str, object] = {
    "resolution_scale": 0.67,
    "texture_atlas_size": 512,
    "mesh_style": "low_poly",
    "target_count_min": 3,
    "target_count_max": 8,
    "draw_distance": "medium",
    "fog": True,
    "lighting": "ambient_plus_directional",
    "shadows": False,
    "reflections": False,
    "antialiasing": "off",
}


def _normalize_godot_window_mode(value: object) -> str:
    token = str(value or "").strip().lower()
    if token in {"fullscreen", "exclusive", "full_screen"}:
        return "fullscreen"
    if token in {"borderless", "windowed_fullscreen", "desktop"}:
        return "fullscreen"
    if token in {"maximized", "maximize"}:
        return "maximized"
    return "windowed"


def _enum_value(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    return value


def _finite_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(number):
        return float(default)
    return number


def _json_safe(value: object) -> object:
    value = _enum_value(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return _finite_float(value)
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {field.name: _json_safe(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    return str(value)


def _position_tuple(value: object) -> dict[str, float]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        coords = list(value)
        x = coords[0] if len(coords) > 0 else 0.0
        y = coords[1] if len(coords) > 1 else 0.0
        z = coords[2] if len(coords) > 2 else 0.0
        return {"x": _finite_float(x), "y": _finite_float(y), "z": _finite_float(z)}
    return {
        "x": _finite_float(getattr(value, "x", 0.0)),
        "y": _finite_float(getattr(value, "y", 0.0)),
        "z": _finite_float(getattr(value, "z", 0.0)),
    }


def _spatial_point(point: SpatialIntegrationPoint | object) -> dict[str, int]:
    return {
        "x": int(getattr(point, "x", 0)),
        "y": int(getattr(point, "y", 0)),
        "z": int(getattr(point, "z", 0)),
    }


def _auditory_point_to_godot(point: Point3, *, origin: Point3) -> dict[str, float]:
    return {
        "x": _finite_float((float(point[0]) - float(origin[0])) * AUDITORY_GODOT_WORLD_SCALE),
        "y": _finite_float(
            AUDITORY_GODOT_Y_OFFSET
            + ((float(point[2]) - float(origin[2])) * AUDITORY_GODOT_WORLD_SCALE)
        ),
        "z": _finite_float(-(float(point[1]) - float(origin[1])) * AUDITORY_GODOT_WORLD_SCALE),
    }


def _auditory_vector_to_godot(vector: Point3) -> dict[str, float]:
    return {
        "x": _finite_float(vector[0]),
        "y": _finite_float(vector[2]),
        "z": _finite_float(-float(vector[1])),
    }


def _auditory_frame_payload(
    *,
    distance: float,
    origin: Point3,
    travel_distance: float,
    session_seed: int,
    curvature_intensity: float,
    twist_intensity: float,
) -> dict[str, object]:
    center, tangent, right, up = tube_frame(
        float(distance),
        session_seed=int(session_seed),
        curvature_intensity=float(curvature_intensity),
        twist_intensity=float(twist_intensity),
    )
    far_span = max(1e-6, float(AUDITORY_GATE_FAR_DISTANCE_OFFSET))
    depth_norm = (float(distance) - float(travel_distance)) / far_span
    return {
        "distance": _finite_float(distance),
        "offset_distance": _finite_float(float(distance) - float(travel_distance)),
        "depth_norm": _finite_float(max(-0.35, min(1.15, depth_norm))),
        "pos": _auditory_point_to_godot(center, origin=origin),
        "tangent": _auditory_vector_to_godot(tangent),
        "right": _auditory_vector_to_godot(right),
        "up": _auditory_vector_to_godot(up),
        "twist_angle_rad": _finite_float(
            tube_twist_angle(
                float(distance),
                intensity=float(twist_intensity),
                session_seed=int(session_seed),
            )
        ),
    }


def _auditory_position_in_frame(
    *,
    distance: float,
    local_right: float,
    local_up: float,
    origin: Point3,
    session_seed: int,
    curvature_intensity: float,
    twist_intensity: float,
) -> dict[str, float]:
    center, _tangent, right, up = tube_frame(
        float(distance),
        session_seed=int(session_seed),
        curvature_intensity=float(curvature_intensity),
        twist_intensity=float(twist_intensity),
    )
    point = vec_add(center, vec_add(vec_scale(right, float(local_right)), vec_scale(up, float(local_up))))
    return _auditory_point_to_godot(point, origin=origin)


def _serialize_auditory_tunnel(payload: AuditoryCapacityPayload) -> dict[str, object]:
    travel_distance = _finite_float(payload.presentation_travel_distance)
    session_seed = int(payload.session_seed)
    curvature_intensity = _finite_float(getattr(payload, "tunnel_curvature_intensity", 1.0))
    twist_intensity = _finite_float(getattr(payload, "tunnel_twist_intensity", 0.0))
    origin, _origin_tangent, _origin_right, _origin_up = tube_frame(
        travel_distance,
        session_seed=session_seed,
        curvature_intensity=curvature_intensity,
        twist_intensity=twist_intensity,
    )
    far_offset = max(
        float(AUDITORY_GATE_FAR_DISTANCE_OFFSET),
        float(AUDITORY_TUNNEL_GEOMETRY_END_OFFSET_DISTANCE),
    )
    sample_count = 24
    offsets = [-5.0, -2.5] + [
        (float(idx) / float(sample_count - 1)) * far_offset for idx in range(sample_count)
    ]
    samples = [
        _auditory_frame_payload(
            distance=travel_distance + offset,
            origin=origin,
            travel_distance=travel_distance,
            session_seed=session_seed,
            curvature_intensity=curvature_intensity,
            twist_intensity=twist_intensity,
        )
        for offset in offsets
    ]

    camera_distance = travel_distance - 9.0
    look_distance = travel_distance + min(18.0, far_offset * 0.55)
    camera_center, camera_tangent, _camera_right, camera_up = tube_frame(
        camera_distance,
        session_seed=session_seed,
        curvature_intensity=curvature_intensity,
        twist_intensity=twist_intensity,
    )
    camera_point = vec_add(camera_center, vec_add(vec_scale(camera_up, 1.25), vec_scale(camera_tangent, -1.2)))
    look_center, look_tangent, _look_right, look_up = tube_frame(
        look_distance,
        session_seed=session_seed,
        curvature_intensity=curvature_intensity,
        twist_intensity=twist_intensity,
    )
    look_point = vec_add(look_center, vec_add(vec_scale(look_tangent, 0.9), vec_scale(look_up, 0.18)))

    return {
        "origin_distance": _finite_float(travel_distance),
        "scale": _finite_float(AUDITORY_GODOT_WORLD_SCALE),
        "y_offset": _finite_float(AUDITORY_GODOT_Y_OFFSET),
        "inner_rx": _finite_float(_AUDITORY_TUNNEL_INNER_RX * AUDITORY_GODOT_WORLD_SCALE),
        "inner_rz": _finite_float(_AUDITORY_TUNNEL_INNER_RZ * AUDITORY_GODOT_WORLD_SCALE),
        "twist_intensity": _finite_float(twist_intensity),
        "curvature_intensity": _finite_float(curvature_intensity),
        "samples": samples,
        "camera": {
            "position": _auditory_point_to_godot(camera_point, origin=origin),
            "target": _auditory_point_to_godot(look_point, origin=origin),
        },
    }


def _gate_payload(
    gate: AuditoryCapacityGate,
    *,
    payload: AuditoryCapacityPayload | None = None,
    origin: Point3 | None = None,
    session_seed: int = 0,
    curvature_intensity: float = 1.0,
    twist_intensity: float = 0.0,
) -> dict[str, object]:
    data: dict[str, object] = {
        "gate_id": int(gate.gate_id),
        "x_norm": _finite_float(gate.x_norm),
        "y_norm": _finite_float(gate.y_norm),
        "color": str(gate.color),
        "shape": str(gate.shape),
        "aperture_norm": _finite_float(gate.aperture_norm),
        "world_distance": None if gate.world_distance is None else _finite_float(gate.world_distance),
        "visual_slot_index": gate.visual_slot_index,
        "flash_color": gate.flash_color,
        "flash_strength": _finite_float(gate.flash_strength),
    }
    if payload is None or origin is None:
        return data

    travel_distance = _finite_float(payload.presentation_travel_distance)
    gate_distance = (
        _finite_float(gate.world_distance)
        if gate.world_distance is not None
        else auditory_gate_world_distance_from_x_norm(
            float(gate.x_norm),
            travel_distance=travel_distance,
            spawn_x_norm=float(AUDITORY_GATE_SPAWN_X_NORM),
            player_x_norm=float(AUDITORY_GATE_PLAYER_X_NORM),
            retire_x_norm=float(AUDITORY_GATE_RETIRE_X_NORM),
        )
    )
    half_height = max(1e-6, _finite_float(payload.tube_half_height))
    local_up = (_finite_float(gate.y_norm) / half_height) * float(_AUDITORY_TUNNEL_INNER_RZ)
    aperture_radius = max(
        0.22,
        (_finite_float(gate.aperture_norm) / half_height)
        * float(_AUDITORY_TUNNEL_INNER_RZ)
        * float(AUDITORY_GODOT_WORLD_SCALE)
        * 1.18,
    )
    data["pose"] = _auditory_frame_payload(
        distance=gate_distance,
        origin=origin,
        travel_distance=travel_distance,
        session_seed=int(session_seed),
        curvature_intensity=float(curvature_intensity),
        twist_intensity=twist_intensity,
    )
    data["position"] = _auditory_position_in_frame(
        distance=gate_distance,
        local_right=0.0,
        local_up=local_up,
        origin=origin,
        session_seed=int(session_seed),
        curvature_intensity=float(curvature_intensity),
        twist_intensity=twist_intensity,
    )
    data["aperture_radius"] = _finite_float(aperture_radius)
    data["visible_depth_norm"] = _finite_float(
        max(
            -0.35,
            min(
                1.15,
                (gate_distance - (travel_distance + float(AUDITORY_GATE_BEHIND_DISTANCE_OFFSET)))
                / max(
                    1e-6,
                    float(AUDITORY_GATE_FAR_DISTANCE_OFFSET)
                    - float(AUDITORY_GATE_BEHIND_DISTANCE_OFFSET),
                ),
            ),
        )
    )
    return data


def _serialize_auditory_godot_runtime(
    payload: AuditoryCapacityPayload,
    *,
    phase: object | None = None,
    time_remaining_s: object | None = None,
) -> dict[str, object]:
    cfg = AuditoryCapacityConfig()
    difficulty = _finite_float(getattr(payload, "difficulty", 0.0))
    phase_token = str(_enum_value(phase) or "").strip().lower()
    duration_s: float | None = None
    if time_remaining_s is not None:
        remaining = _finite_float(time_remaining_s, default=-1.0)
        if remaining >= 0.0:
            duration_s = _finite_float(payload.phase_elapsed_s) + remaining
    if duration_s is None or duration_s <= 0.0:
        duration_s = _finite_float(payload.segment_time_remaining_s) + _finite_float(
            payload.phase_elapsed_s
        )
    active_channels = [str(channel) for channel in payload.active_channels]
    return {
        "command": "auditory_start",
        "authority": "godot",
        "run_key": f"{int(payload.session_seed)}:{phase_token or 'unknown'}",
        "session_seed": int(payload.session_seed),
        "difficulty": difficulty,
        "phase": phase_token,
        "phase_elapsed_s": _finite_float(payload.phase_elapsed_s),
        "duration_s": _finite_float(duration_s),
        "time_remaining_s": (
            None if time_remaining_s is None else _finite_float(time_remaining_s)
        ),
        "asset_root": str(AUDITORY_CAPACITY_AUDIO_ASSET_PATH),
        "assigned_callsigns": [str(value) for value in payload.assigned_callsigns],
        "active_channels": active_channels,
        "segment": {
            "label": str(payload.segment_label),
            "index": int(payload.segment_index),
            "total": int(payload.segment_total),
            "time_remaining_s": _finite_float(payload.segment_time_remaining_s),
        },
        "config": {
            "tick_hz": _finite_float(cfg.tick_hz),
            "control_gain": _finite_float(cfg.control_gain),
            "disturbance_gain": _finite_float(cfg.disturbance_gain),
            "tube_half_width": _finite_float(payload.tube_half_width),
            "tube_half_height": _finite_float(payload.tube_half_height),
            "tunnel_curvature_intensity": _finite_float(payload.tunnel_curvature_intensity),
            "tunnel_twist_intensity": _finite_float(payload.tunnel_twist_intensity),
            "gate_speed_norm_per_s": _finite_float(cfg.gate_speed_norm_per_s),
            "gate_spawn_rate": _finite_float(cfg.gate_spawn_rate),
            "gate_interval_s": _finite_float(cfg.gate_interval_s),
            "command_rate": _finite_float(cfg.command_rate),
            "distractor_rate": _finite_float(cfg.distractor_rate),
            "callsign_count": max(1, len(payload.assigned_callsigns)),
            "digit_sequence_min_len": max(1, int(payload.recall_target_length or 4)),
            "digit_sequence_max_len": max(1, int(cfg.digit_sequence_max_len)),
            "response_window_seconds": _finite_float(cfg.response_window_seconds),
            "sequence_display_s": _finite_float(cfg.sequence_display_s),
            "sequence_response_s": _finite_float(cfg.sequence_response_s),
            "beep_interval_s": _finite_float(cfg.beep_interval_s),
            "inner_rx": _finite_float(_AUDITORY_TUNNEL_INNER_RX),
            "inner_rz": _finite_float(_AUDITORY_TUNNEL_INNER_RZ),
            "ball_radius": _finite_float(_AUDITORY_BALL_BOUND_RADIUS),
        },
        "audio": {
            "tts_required": True,
            "background_noise_level": _finite_float(payload.background_noise_level),
            "background_distortion_level": _finite_float(payload.background_distortion_level),
            "instructor_noise_level": _finite_float(payload.instructor_noise_level),
            "instructor_distortion_level": _finite_float(payload.instructor_distortion_level),
            "instructor_rate_wpm": int(payload.instructor_rate_wpm),
            "ambient_layer_target": int(payload.ambient_layer_target),
            "background_noise_source": payload.background_noise_source,
        },
        "constants": {
            "gate_spawn_x_norm": _finite_float(AUDITORY_GATE_SPAWN_X_NORM),
            "gate_player_x_norm": _finite_float(AUDITORY_GATE_PLAYER_X_NORM),
            "gate_retire_x_norm": _finite_float(AUDITORY_GATE_RETIRE_X_NORM),
            "colors": ["RED", "BLUE", "YELLOW"],
            "shapes": ["CIRCLE", "TRIANGLE", "SQUARE"],
        },
        "metrics": {
            "gate_hits": int(payload.gate_hits),
            "gate_misses": int(payload.gate_misses),
            "forbidden_gate_hits": int(payload.forbidden_gate_hits),
            "collisions": int(payload.collisions),
            "correct_command_executions": int(payload.correct_command_executions),
            "missed_valid_commands": int(payload.missed_valid_commands),
            "false_responses_to_distractors": int(payload.false_responses_to_distractors),
            "digit_recall_attempts": int(payload.digit_recall_attempts),
            "digit_recall_accuracy": _finite_float(payload.digit_recall_accuracy),
            "points": _finite_float(payload.points),
        },
    }


def _serialize_godot_owned(payload: GodotOwnedPayload) -> dict[str, object]:
    spec = payload.spec
    data = _json_safe(spec)
    if not isinstance(data, dict):
        data = {}
    data["command"] = "godot_start"
    data["authority"] = "godot"
    data["kind"] = str(spec.kind)
    data["test_code"] = str(spec.test_code)
    data["title"] = str(spec.title)
    data["session_seed"] = int(spec.seed)
    data["seed"] = int(spec.seed)
    data["difficulty"] = _finite_float(spec.difficulty)
    data["duration_s"] = _finite_float(spec.duration_s)
    if str(spec.phase) == "practice" and spec.practice_duration_s > 0.0:
        data["duration_s"] = _finite_float(spec.practice_duration_s)
    data["mode"] = str(spec.mode)
    if str(spec.kind) == KIND_AUDITORY_CAPACITY:
        assets = dict(data.get("assets", {}) if isinstance(data.get("assets"), Mapping) else {})
        assets.setdefault("audio_root", str(AUDITORY_CAPACITY_AUDIO_ASSET_PATH))
        data["assets"] = assets
        data.setdefault("asset_root", str(AUDITORY_CAPACITY_AUDIO_ASSET_PATH))
        config = data.get("config", {})
        config_audio: dict[str, object] = {}
        if isinstance(config, Mapping) and isinstance(config.get("audio"), Mapping):
            config_audio = dict(config.get("audio", {}))
        audio = dict(config_audio)
        audio.update(dict(data.get("audio", {}) if isinstance(data.get("audio"), Mapping) else {}))
        audio.setdefault("tts_required", True)
        data["audio"] = audio
    return {
        "godot_start": data,
        "progress": _json_safe(payload.progress),
        "error": None if payload.error is None else _json_safe(payload.error),
    }


def _serialize_auditory_capacity(
    payload: AuditoryCapacityPayload | None,
    *,
    phase: object | None = None,
    time_remaining_s: object | None = None,
) -> dict[str, object]:
    if payload is None:
        return {}
    travel_distance = _finite_float(payload.presentation_travel_distance)
    session_seed = int(payload.session_seed)
    curvature_intensity = _finite_float(getattr(payload, "tunnel_curvature_intensity", 1.0))
    twist_intensity = _finite_float(getattr(payload, "tunnel_twist_intensity", 0.0))
    origin, _origin_tangent, _origin_right, _origin_up = tube_frame(
        travel_distance,
        session_seed=session_seed,
        curvature_intensity=curvature_intensity,
        twist_intensity=twist_intensity,
    )
    half_width = max(1e-6, _finite_float(payload.tube_half_width))
    half_height = max(1e-6, _finite_float(payload.tube_half_height))
    ball_local_right = (
        (_finite_float(payload.ball_x) / half_width)
        * max(0.05, float(_AUDITORY_TUNNEL_INNER_RX) - float(_AUDITORY_BALL_BOUND_RADIUS))
    )
    ball_local_up = (
        (_finite_float(payload.ball_y) / half_height)
        * max(0.05, float(_AUDITORY_TUNNEL_INNER_RZ) - float(_AUDITORY_BALL_BOUND_RADIUS))
    )
    return {
        "session_seed": int(payload.session_seed),
        "godot_runtime": _serialize_auditory_godot_runtime(
            payload,
            phase=phase,
            time_remaining_s=time_remaining_s,
        ),
        "phase_elapsed_s": _finite_float(payload.phase_elapsed_s),
        "presentation_travel_distance": travel_distance,
        "segment_label": str(payload.segment_label),
        "segment_index": int(payload.segment_index),
        "segment_total": int(payload.segment_total),
        "ball": {
            "x": _finite_float(payload.ball_x),
            "y": _finite_float(payload.ball_y),
            "forward_norm": _finite_float(payload.ball_forward_norm),
            "contact_ratio": _finite_float(payload.ball_contact_ratio),
            "color": str(payload.ball_visual_color or payload.ball_color),
            "color_strength": _finite_float(payload.ball_visual_strength),
            "number": int(payload.ball_number),
            "position": _auditory_position_in_frame(
                distance=travel_distance,
                local_right=ball_local_right,
                local_up=ball_local_up,
                origin=origin,
                session_seed=session_seed,
                curvature_intensity=curvature_intensity,
                twist_intensity=twist_intensity,
            ),
            "pose": _auditory_frame_payload(
                distance=travel_distance,
                origin=origin,
                travel_distance=travel_distance,
                session_seed=session_seed,
                curvature_intensity=curvature_intensity,
                twist_intensity=twist_intensity,
            ),
            "visual_radius": _finite_float(max(0.24, float(_AUDITORY_BALL_BOUND_RADIUS) * 2.4)),
        },
        "control": {
            "x": _finite_float(payload.control_x),
            "y": _finite_float(payload.control_y),
            "disturbance_x": _finite_float(payload.disturbance_x),
            "disturbance_y": _finite_float(payload.disturbance_y),
        },
        "tube": {
            "half_width": _finite_float(payload.tube_half_width),
            "half_height": _finite_float(payload.tube_half_height),
            "inner_rx": _finite_float(_AUDITORY_TUNNEL_INNER_RX),
            "inner_rz": _finite_float(_AUDITORY_TUNNEL_INNER_RZ),
        },
        "tunnel": _serialize_auditory_tunnel(payload),
        "instruction": {
            "text": payload.instruction_text,
            "uid": payload.instruction_uid,
            "command_type": payload.instruction_command_type,
            "target_gate_id": payload.target_gate_id,
            "target_gate_action": payload.target_gate_action,
            "forbidden_gate_color": payload.forbidden_gate_color,
            "forbidden_gate_shape": payload.forbidden_gate_shape,
        },
        "gates": [
            _gate_payload(
                gate,
                payload=payload,
                origin=origin,
                session_seed=session_seed,
                curvature_intensity=curvature_intensity,
                twist_intensity=twist_intensity,
            )
            for gate in payload.gates[:12]
        ],
        "metrics": {
            "gate_hits": int(payload.gate_hits),
            "gate_misses": int(payload.gate_misses),
            "collisions": int(payload.collisions),
            "points": _finite_float(payload.points),
        },
    }


def _serialize_rapid_tracking(payload: RapidTrackingPayload | None) -> dict[str, object]:
    if payload is None:
        return {}
    return {
        "session_seed": int(payload.session_seed),
        "scene_seed": int(payload.scene_seed),
        "phase_elapsed_s": _finite_float(payload.phase_elapsed_s),
        "difficulty_tier": str(payload.difficulty_tier),
        "segment_label": str(payload.segment_label),
        "target": {
            "rel_x": _finite_float(payload.target_rel_x),
            "rel_y": _finite_float(payload.target_rel_y),
            "world_x": _finite_float(payload.target_world_x),
            "world_y": _finite_float(payload.target_world_y),
            "vx": _finite_float(payload.target_vx),
            "vy": _finite_float(payload.target_vy),
            "visible": bool(payload.target_visible),
            "cover_state": str(payload.target_cover_state),
            "kind": str(payload.target_kind),
            "variant": str(payload.target_variant),
            "handoff_mode": str(payload.target_handoff_mode),
            "moving": bool(payload.target_is_moving),
        },
        "reticle": {
            "x": _finite_float(payload.reticle_x),
            "y": _finite_float(payload.reticle_y),
        },
        "camera": {
            "x": _finite_float(payload.camera_x),
            "y": _finite_float(payload.camera_y),
            "yaw_deg": _finite_float(payload.camera_yaw_deg),
            "pitch_deg": _finite_float(payload.camera_pitch_deg),
            "focus_world_x": _finite_float(payload.focus_world_x),
            "focus_world_y": _finite_float(payload.focus_world_y),
        },
        "capture": {
            "half_width": _finite_float(payload.capture_box_half_width),
            "half_height": _finite_float(payload.capture_box_half_height),
            "target_in_capture_box": bool(payload.target_in_capture_box),
            "zoom": _finite_float(payload.capture_zoom),
            "points": int(payload.capture_points),
            "hits": int(payload.capture_hits),
            "attempts": int(payload.capture_attempts),
            "feedback": str(payload.capture_feedback),
            "flash_s": _finite_float(payload.capture_flash_s),
        },
        "scene": {
            "progress": _finite_float(payload.scene_progress),
            "terrain_ridge_y": _finite_float(payload.terrain_ridge_y),
            "active_target_kinds": list(payload.active_target_kinds[:8]),
            "active_challenges": list(payload.active_challenges[:8]),
        },
        "metrics": {
            "mean_error": _finite_float(payload.mean_error),
            "rms_error": _finite_float(payload.rms_error),
            "on_target_ratio": _finite_float(payload.on_target_ratio),
            "obscured_tracking_ratio": _finite_float(payload.obscured_tracking_ratio),
        },
    }


def _serialize_spatial_integration(payload: SpatialIntegrationPayload | None) -> dict[str, object]:
    if payload is None:
        return {}
    return {
        "part": _enum_value(payload.part),
        "trial_stage": _enum_value(payload.trial_stage),
        "block_kind": str(payload.block_kind),
        "scene_id": int(payload.scene_id),
        "scene_index_in_block": int(payload.scene_index_in_block),
        "study_view_index": int(payload.study_view_index),
        "study_views_in_scene": int(payload.study_views_in_scene),
        "question_index_in_scene": int(payload.question_index_in_scene),
        "questions_in_scene": int(payload.questions_in_scene),
        "stage_time_remaining_s": (
            None if payload.stage_time_remaining_s is None else _finite_float(payload.stage_time_remaining_s)
        ),
        "kind": _enum_value(payload.kind),
        "answer_mode": _enum_value(payload.answer_mode),
        "stem": str(payload.stem),
        "query_label": str(payload.query_label),
        "north_arrow_deg": int(payload.north_arrow_deg),
        "scene_view": _enum_value(payload.scene_view),
        "grid": {
            "cols": int(payload.grid_cols),
            "rows": int(payload.grid_rows),
            "alt_levels": int(payload.alt_levels),
        },
        "reference_views": [_json_safe(view) for view in payload.reference_views],
        "active_reference_view": _json_safe(payload.active_reference_view),
        "hills": [_json_safe(hill) for hill in payload.hills],
        "landmarks": [_json_safe(landmark) for landmark in payload.landmarks],
        "route_points": [_spatial_point(point) for point in payload.route_points],
        "route_current_index": int(payload.route_current_index),
        "aircraft": {
            "previous": _spatial_point(payload.aircraft_prev),
            "current": _spatial_point(payload.aircraft_now),
            "velocity": _json_safe(payload.velocity),
            "show_motion": bool(payload.show_aircraft_motion),
        },
        "options": [_json_safe(option) for option in payload.options[:6]],
        "correct_point": _spatial_point(payload.correct_point),
    }


def _trace1_frame(frame: TraceTest1SceneFrame, *, role: str, index: int) -> dict[str, object]:
    return {
        "role": role,
        "index": int(index),
        "position": _position_tuple(frame.position),
        "attitude": _json_safe(frame.attitude),
        "travel_heading_deg": _finite_float(frame.travel_heading_deg),
        "world_forward": _position_tuple(frame.world_forward),
        "world_up": _position_tuple(frame.world_up),
        "world_tangent": _position_tuple(frame.world_tangent),
    }


def _serialize_trace_test_1(payload: TraceTest1Payload | None) -> dict[str, object]:
    if payload is None:
        return {}
    frames = [_trace1_frame(payload.scene.red_frame, role="red", index=0)]
    frames.extend(
        _trace1_frame(frame, role="blue", index=idx)
        for idx, frame in enumerate(payload.scene.blue_frames, start=1)
    )
    return {
        "trial_stage": _enum_value(payload.trial_stage),
        "stage_time_remaining_s": (
            None if payload.stage_time_remaining_s is None else _finite_float(payload.stage_time_remaining_s)
        ),
        "observe_progress": _finite_float(payload.observe_progress),
        "prompt_index": int(payload.prompt_index),
        "active_command": _enum_value(payload.active_command),
        "blue_commands": [_enum_value(command) for command in payload.blue_commands],
        "frames": frames,
        "options": [_json_safe(option) for option in payload.options],
        "correct_code": int(payload.correct_code),
        "answer_open_progress": _finite_float(payload.answer_open_progress),
        "speed_multiplier": _finite_float(payload.speed_multiplier),
        "viewpoint_bearing_deg": int(payload.viewpoint_bearing_deg),
    }


def _trace2_track(track: TraceTest2AircraftTrack, *, progress: float) -> dict[str, object]:
    current = trace_test_2_track_position(track=track, progress=progress)
    return {
        "code": int(track.code),
        "color_name": str(track.color_name),
        "color_rgb": [int(part) for part in track.color_rgb],
        "motion_kind": _enum_value(track.motion_kind),
        "direction_changed": bool(track.direction_changed),
        "ended_screen_x": _finite_float(track.ended_screen_x),
        "ended_altitude_z": _finite_float(track.ended_altitude_z),
        "current_position": _position_tuple(current),
        "waypoints": [_position_tuple(point) for point in track.waypoints],
    }


def _serialize_trace_test_2(payload: TraceTest2Payload | None) -> dict[str, object]:
    if payload is None:
        return {}
    progress = _finite_float(payload.observe_progress)
    return {
        "trial_stage": _enum_value(payload.trial_stage),
        "stage_time_remaining_s": (
            None if payload.stage_time_remaining_s is None else _finite_float(payload.stage_time_remaining_s)
        ),
        "observe_progress": progress,
        "block_kind": str(payload.block_kind),
        "trial_index_in_block": int(payload.trial_index_in_block),
        "trials_in_block": int(payload.trials_in_block),
        "question_kind": _enum_value(payload.question_kind),
        "stem": str(payload.stem),
        "viewpoint_bearing_deg": int(payload.viewpoint_bearing_deg),
        "aircraft": [_trace2_track(track, progress=progress) for track in payload.aircraft[:8]],
        "options": [_json_safe(option) for option in payload.options],
        "correct_code": int(payload.correct_code),
    }


def godot_kind_for_snapshot(snap: TestSnapshot) -> str | None:
    payload = snap.payload
    if isinstance(payload, GodotOwnedPayload):
        return str(payload.spec.kind)
    if isinstance(payload, AuditoryCapacityPayload):
        return KIND_AUDITORY_CAPACITY
    if isinstance(payload, RapidTrackingPayload):
        return KIND_RAPID_TRACKING
    if isinstance(payload, SpatialIntegrationPayload):
        return KIND_SPATIAL_INTEGRATION
    if isinstance(payload, TraceTest1Payload):
        return KIND_TRACE_TEST_1
    if isinstance(payload, TraceTest2Payload):
        return KIND_TRACE_TEST_2

    title = str(snap.title)
    if title.startswith("Auditory Capacity"):
        return KIND_AUDITORY_CAPACITY
    if title.startswith("Rapid Tracking"):
        return KIND_RAPID_TRACKING
    if title.startswith("Spatial Integration"):
        return KIND_SPATIAL_INTEGRATION
    if title.startswith("Trace Test 1"):
        return KIND_TRACE_TEST_1
    if title.startswith("Trace Test 2"):
        return KIND_TRACE_TEST_2
    return None


def serialize_godot_state(snap: TestSnapshot, payload: object | None = None) -> dict[str, object] | None:
    kind = godot_kind_for_snapshot(snap)
    if kind is None:
        return None
    resolved_payload = snap.payload if payload is None else payload
    if isinstance(resolved_payload, GodotOwnedPayload):
        kind = str(resolved_payload.spec.kind)
        visual_payload = _serialize_godot_owned(resolved_payload)
    elif kind == KIND_AUDITORY_CAPACITY:
        visual_payload = _serialize_auditory_capacity(
            resolved_payload if isinstance(resolved_payload, AuditoryCapacityPayload) else None,
            phase=snap.phase,
            time_remaining_s=snap.time_remaining_s,
        )
    elif kind == KIND_RAPID_TRACKING:
        visual_payload = _serialize_rapid_tracking(
            resolved_payload if isinstance(resolved_payload, RapidTrackingPayload) else None
        )
    elif kind == KIND_SPATIAL_INTEGRATION:
        visual_payload = _serialize_spatial_integration(
            resolved_payload if isinstance(resolved_payload, SpatialIntegrationPayload) else None
        )
    elif kind == KIND_TRACE_TEST_1:
        visual_payload = _serialize_trace_test_1(
            resolved_payload if isinstance(resolved_payload, TraceTest1Payload) else None
        )
    elif kind == KIND_TRACE_TEST_2:
        visual_payload = _serialize_trace_test_2(
            resolved_payload if isinstance(resolved_payload, TraceTest2Payload) else None
        )
    else:
        return None

    return {
        "schema": GODOT_SCHEMA_VERSION,
        "kind": kind,
        "active": True,
        "title": str(snap.title),
        "phase": _enum_value(snap.phase),
        "prompt": str(snap.prompt),
        "time_remaining_s": (
            None if snap.time_remaining_s is None else _finite_float(snap.time_remaining_s)
        ),
        "attempted_scored": int(snap.attempted_scored),
        "correct_scored": int(snap.correct_scored),
        "renderer_backend": GODOT_BACKEND_NAME,
        "performance": dict(GODOT_PERFORMANCE_DEFAULTS),
        "payload": visual_payload,
    }


class GodotBridgeManager:
    """Launches Godot lazily and streams visual snapshots over UDP JSON."""

    def __init__(
        self,
        *,
        headless: bool = False,
        project_path: Path | str | None = None,
        godot_bin: str | None = None,
        host: str = GODOT_HOST,
        port: int | None = None,
        env: Mapping[str, str] | None = None,
        window_mode: str = "fullscreen",
        popen_factory: Callable[..., object] = subprocess.Popen,
        socket_factory: Callable[..., Any] = socket.socket,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._env = os.environ if env is None else env
        self._headless = bool(headless)
        self._disabled = self._headless or self._env_flag_enabled("CFAST_DISABLE_GODOT")
        self._project_path = Path(project_path) if project_path is not None else GODOT_PROJECT_PATH
        self._configured_bin = godot_bin
        self._window_mode = _normalize_godot_window_mode(
            self._env.get(GODOT_WINDOW_MODE_ENV, window_mode)
        )
        self._host = str(host)
        self._port = port
        self._popen_factory = popen_factory
        self._socket_factory = socket_factory
        self._time_fn = time_fn
        self._process: object | None = None
        self._socket: Any | None = None
        self._control_socket: Any | None = None
        self._control_port: int | None = None
        self._session_id = uuid.uuid4().hex
        self._active_kind: str | None = None
        self._last_error: str | None = None
        self._used_fallback = False
        self._restart_attempted_for_kind: set[str] = set()
        self._last_send_at = 0.0
        self._min_send_interval_s = 1.0 / 60.0
        self._menu_state: Mapping[str, object] = {"active": False}

    def _env_flag_enabled(self, name: str) -> bool:
        return str(self._env.get(name, "")).strip().lower() in {"1", "true", "on", "yes"}

    @property
    def window_mode(self) -> str:
        return self._window_mode

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def control_port(self) -> int | None:
        return self._control_port

    def set_window_mode(self, window_mode: str) -> None:
        if self._env.get(GODOT_WINDOW_MODE_ENV):
            self._window_mode = _normalize_godot_window_mode(self._env.get(GODOT_WINDOW_MODE_ENV))
            return
        self._window_mode = _normalize_godot_window_mode(window_mode)

    def set_menu_state(self, menu_state: Mapping[str, object] | None) -> None:
        if menu_state is None:
            self._menu_state = {"active": False}
            return
        safe = _json_safe(dict(menu_state))
        self._menu_state = safe if isinstance(safe, Mapping) else {"active": False}

    @property
    def active_kind(self) -> str | None:
        if self._process_alive():
            return self._active_kind
        return None

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def is_active_for(self, kind: str) -> bool:
        return self.active_kind == str(kind)

    def used_fallback(self) -> bool:
        return bool(self._used_fallback)

    def renderer_backend_for(self, kind: str) -> str:
        return GODOT_BACKEND_NAME if self.is_active_for(kind) else "pygame_2d"

    def sync(self, snap: TestSnapshot, payload: object | None = None) -> bool:
        state = serialize_godot_state(snap, payload)
        if state is None:
            self.idle()
            return False
        kind = str(state["kind"])
        state["window_mode"] = self._window_mode
        state["session_id"] = self._session_id
        state["menu"] = self._menu_state
        if self._disabled:
            self._active_kind = None
            self._last_error = "Godot companion disabled"
            return False
        if not self._ensure_started(kind):
            self._active_kind = None
            return False
        now = self._time_fn()
        if now - self._last_send_at < self._min_send_interval_s and self._active_kind == kind:
            return True
        if self._send(state):
            self._active_kind = kind
            self._last_send_at = now
            return True
        self._active_kind = None
        return False

    def idle(self) -> None:
        if self._process_alive():
            self._send(
                {
                    "schema": GODOT_SCHEMA_VERSION,
                    "kind": "idle",
                    "active": False,
                    "renderer_backend": GODOT_BACKEND_NAME,
                    "window_mode": self._window_mode,
                    "session_id": self._session_id,
                    "performance": dict(GODOT_PERFORMANCE_DEFAULTS),
                    "menu": {"active": False},
                }
            )
        self._active_kind = None

    def close(self) -> None:
        if self._process_alive():
            self._send({"schema": GODOT_SCHEMA_VERSION, "command": "quit"})
        process = self._process
        self._active_kind = None
        self._process = None
        self._close_socket()
        self._close_control_socket()
        if process is None:
            return
        terminate = getattr(process, "terminate", None)
        if callable(terminate):
            try:
                terminate()
            except Exception:
                pass
        wait = getattr(process, "wait", None)
        if callable(wait):
            try:
                wait(timeout=1.0)
            except Exception:
                kill = getattr(process, "kill", None)
                if callable(kill):
                    try:
                        kill()
                    except Exception:
                        pass

    def _ensure_started(self, kind: str) -> bool:
        if self._process_alive():
            return True
        if self._process is not None:
            if kind in self._restart_attempted_for_kind:
                self._used_fallback = True
                self._last_error = "Godot companion exited"
                self._process = None
                self._close_socket()
                self._close_control_socket()
                return False
            self._restart_attempted_for_kind.add(kind)
            self._process = None
            self._close_socket()
            self._close_control_socket()
        return self._start_process(kind)

    def _start_process(self, kind: str) -> bool:
        godot_bin = self._resolve_godot_bin()
        if godot_bin is None:
            self._used_fallback = True
            self._last_error = "Godot binary not found"
            return False
        if not self._project_path.exists():
            self._used_fallback = True
            self._last_error = f"Godot project not found: {self._project_path}"
            return False
        port = self._port if self._port is not None else self._allocate_udp_port()
        self._port = int(port)
        if not self._ensure_control_socket():
            return False
        command = [
            godot_bin,
            "--path",
            str(self._project_path),
            *self._window_args(),
            "--max-fps",
            GODOT_MAX_FPS,
            "--",
            "--listen-port",
            str(self._port),
            "--control-host",
            self._host,
            "--control-port",
            str(self._control_port),
            "--session-id",
            self._session_id,
            "--initial-kind",
            str(kind),
        ]
        try:
            self._process = self._popen_factory(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._socket = self._socket_factory(socket.AF_INET, socket.SOCK_DGRAM)
        except Exception as exc:
            self._process = None
            self._close_socket()
            self._close_control_socket()
            self._used_fallback = True
            self._last_error = f"Godot launch failed: {exc}"
            return False
        return True

    def _window_args(self) -> list[str]:
        if self._window_mode == "fullscreen":
            return ["--fullscreen"]
        if self._window_mode == "maximized":
            return ["--maximized"]
        return ["--windowed", "--resolution", GODOT_WINDOW_RESOLUTION]

    def _resolve_godot_bin(self) -> str | None:
        candidates: list[str] = []
        configured = self._configured_bin or self._env.get("CFAST_GODOT_BIN")
        if configured:
            candidates.append(str(configured))
        candidates.append(GODOT_DEFAULT_BIN)
        for name in ("godot", "godot4"):
            found = shutil.which(name)
            if found:
                candidates.append(found)
        for candidate in candidates:
            if Path(candidate).is_file() and os.access(candidate, os.X_OK):
                return candidate
        return None

    def _allocate_udp_port(self) -> int:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.bind((self._host, 0))
            return int(sock.getsockname()[1])
        finally:
            sock.close()

    def _ensure_control_socket(self) -> bool:
        if self._control_socket is not None and self._control_port is not None:
            return True
        try:
            sock = self._socket_factory(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind((self._host, 0))
            setblocking = getattr(sock, "setblocking", None)
            if callable(setblocking):
                setblocking(False)
            self._control_port = int(sock.getsockname()[1])
            self._control_socket = sock
            return True
        except Exception as exc:
            self._control_socket = None
            self._control_port = None
            self._used_fallback = True
            self._last_error = f"Godot control socket failed: {exc}"
            return False

    def poll_control_commands(self, *, max_commands: int = 24) -> list[dict[str, object]]:
        sock = self._control_socket
        if sock is None:
            return []
        commands: list[dict[str, object]] = []
        for _ in range(max(1, int(max_commands))):
            try:
                payload, _addr = sock.recvfrom(65536)
            except BlockingIOError:
                break
            except TimeoutError:
                break
            except OSError as exc:
                if getattr(exc, "errno", None) in {errno.EAGAIN, errno.EWOULDBLOCK}:
                    break
                self._last_error = f"Godot control receive failed: {exc}"
                break
            parsed = self._parse_control_payload(payload)
            if parsed is not None:
                commands.append(parsed)
        return commands

    def _parse_control_payload(self, payload: bytes | bytearray | str) -> dict[str, object] | None:
        try:
            text = payload.decode("utf-8") if isinstance(payload, (bytes, bytearray)) else str(payload)
            message = json.loads(text)
        except Exception:
            return None
        if not isinstance(message, dict):
            return None
        try:
            schema = int(message.get("schema", 0))
        except Exception:
            schema = 0
        if schema != GODOT_SCHEMA_VERSION:
            return None
        if str(message.get("session_id", "")) != self._session_id:
            return None
        command = str(message.get("command", "")).strip().lower()
        if command not in GODOT_CONTROL_COMMANDS:
            return None
        parsed: dict[str, object] = {"command": command}
        for key in (
            "action",
            "key",
            "mode",
            "window_mode",
            "direction",
            "run_key",
            "phase",
            "event",
            "metrics",
            "summary",
            "result",
            "state",
            "reason",
            "detail",
            "events",
            "kind",
            "progress",
            "test_code",
        ):
            if key in message:
                parsed[key] = _json_safe(message[key])
        if command == "set_window_mode":
            requested = message.get("window_mode", message.get("mode", ""))
            self.set_window_mode(str(requested))
            parsed["window_mode"] = self._window_mode
        return parsed

    def _send(self, message: Mapping[str, object]) -> bool:
        if self._socket is None or self._port is None:
            return False
        try:
            payload = json.dumps(_json_safe(message), separators=(",", ":"), ensure_ascii=True).encode(
                "utf-8"
            )
            self._socket.sendto(payload, (self._host, int(self._port)))
            return True
        except Exception as exc:
            self._used_fallback = True
            self._last_error = f"Godot UDP send failed: {exc}"
            return False

    def _process_alive(self) -> bool:
        process = self._process
        if process is None:
            return False
        poll = getattr(process, "poll", None)
        if not callable(poll):
            return True
        try:
            return poll() is None
        except Exception:
            return False

    def _close_socket(self) -> None:
        sock = self._socket
        self._socket = None
        close = getattr(sock, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    def _close_control_socket(self) -> None:
        sock = self._control_socket
        self._control_socket = None
        self._control_port = None
        close = getattr(sock, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
