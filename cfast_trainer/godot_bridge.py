"""Local UDP bridge for the optional Godot companion renderer.

The pygame runtime remains authoritative for timing, scoring, input, and
results.  Godot only receives whitelisted visual snapshots for the tests that
currently need 3D presentation.
"""

from __future__ import annotations

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

from .auditory_capacity import AuditoryCapacityGate, AuditoryCapacityPayload
from .cognitive_core import TestSnapshot
from .rapid_tracking.entities import RapidTrackingPayload
from .spatial_integration import SpatialIntegrationPayload, SpatialIntegrationPoint
from .trace_test_1 import TraceTest1Payload, TraceTest1SceneFrame
from .trace_test_2 import TraceTest2AircraftTrack, TraceTest2Payload, trace_test_2_track_position

GODOT_BACKEND_NAME = "godot_4"
GODOT_DEFAULT_BIN = "/Applications/Godot.app/Contents/MacOS/Godot"
GODOT_PROJECT_PATH = Path(__file__).resolve().parent.parent / "godot" / "cfast_3d"
GODOT_HOST = "127.0.0.1"
GODOT_WINDOW_RESOLUTION = "960x540"
GODOT_MAX_FPS = "60"
GODOT_SCHEMA_VERSION = 1
GODOT_WINDOW_MODE_ENV = "CFAST_GODOT_WINDOW_MODE"

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


def _gate_payload(gate: AuditoryCapacityGate) -> dict[str, object]:
    return {
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


def _serialize_auditory_capacity(payload: AuditoryCapacityPayload | None) -> dict[str, object]:
    if payload is None:
        return {}
    return {
        "session_seed": int(payload.session_seed),
        "phase_elapsed_s": _finite_float(payload.phase_elapsed_s),
        "presentation_travel_distance": _finite_float(payload.presentation_travel_distance),
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
        },
        "instruction": {
            "text": payload.instruction_text,
            "uid": payload.instruction_uid,
            "command_type": payload.instruction_command_type,
            "target_gate_id": payload.target_gate_id,
            "target_gate_action": payload.target_gate_action,
            "forbidden_gate_color": payload.forbidden_gate_color,
            "forbidden_gate_shape": payload.forbidden_gate_shape,
        },
        "gates": [_gate_payload(gate) for gate in payload.gates[:12]],
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
    if kind == KIND_AUDITORY_CAPACITY:
        visual_payload = _serialize_auditory_capacity(
            resolved_payload if isinstance(resolved_payload, AuditoryCapacityPayload) else None
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
        window_mode: str = "windowed",
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
        self._session_id = uuid.uuid4().hex
        self._active_kind: str | None = None
        self._last_error: str | None = None
        self._used_fallback = False
        self._restart_attempted_for_kind: set[str] = set()
        self._last_send_at = 0.0
        self._min_send_interval_s = 1.0 / 60.0

    def _env_flag_enabled(self, name: str) -> bool:
        return str(self._env.get(name, "")).strip().lower() in {"1", "true", "on", "yes"}

    @property
    def window_mode(self) -> str:
        return self._window_mode

    def set_window_mode(self, window_mode: str) -> None:
        if self._env.get(GODOT_WINDOW_MODE_ENV):
            self._window_mode = _normalize_godot_window_mode(self._env.get(GODOT_WINDOW_MODE_ENV))
            return
        self._window_mode = _normalize_godot_window_mode(window_mode)

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
        if self._disabled:
            self._active_kind = None
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
                    "performance": dict(GODOT_PERFORMANCE_DEFAULTS),
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
                return False
            self._restart_attempted_for_kind.add(kind)
            self._process = None
            self._close_socket()
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
