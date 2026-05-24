from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
import pytest

from cfast_trainer.app import (
    AdaptiveSessionScreen,
    AntWorkoutScreen,
    App,
    CognitiveTestScreen,
    MenuItem,
    MenuScreen,
)
from cfast_trainer.ac_drills import AcDrillConfig, build_ac_gate_anchor_drill
from cfast_trainer.adaptive_scheduler import (
    AdaptiveSession,
    AdaptiveSessionBlock,
    AdaptiveSessionPlan,
    AdaptiveStage,
)
from cfast_trainer.ant_drills import AntDrillMode
from cfast_trainer.ant_workouts import (
    AntWorkoutBlockPlan,
    AntWorkoutPlan,
    AntWorkoutSession,
    AntWorkoutStage,
)
from cfast_trainer.auditory_capacity import AuditoryCapacityPayload, build_auditory_capacity_test
from cfast_trainer.cognitive_core import Phase
from cfast_trainer.cognitive_core import TestSnapshot as SnapshotModel
from cfast_trainer.godot_bridge import (
    GODOT_BACKEND_NAME,
    GODOT_DEFAULT_BIN,
    GODOT_PROJECT_PATH,
    GodotBridgeManager,
    godot_kind_for_snapshot,
    serialize_godot_state,
)
from cfast_trainer.godot_owned import (
    GodotOwnedPayload,
    auditory_capacity_godot_config,
    build_godot_owned_test,
    rapid_tracking_godot_config,
    spatial_integration_godot_config,
    trace_test_godot_config,
)
from cfast_trainer.instrument_comprehension import (
    InstrumentComprehensionGenerator,
    InstrumentComprehensionPayload,
    InstrumentComprehensionTrialKind,
)
from cfast_trainer.rapid_tracking import RapidTrackingPayload, build_rapid_tracking_test
from cfast_trainer.spatial_integration import (
    SpatialIntegrationPayload,
    build_spatial_integration_test,
)
from cfast_trainer.trace_test_1 import TraceTest1Payload, build_trace_test_1_test
from cfast_trainer.trace_test_2 import TraceTest2Payload, build_trace_test_2_test


@dataclass
class _FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


class _FakeProcess:
    def __init__(self) -> None:
        self.poll_result: int | None = None
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return self.poll_result

    def terminate(self) -> None:
        self.terminated = True
        self.poll_result = 0

    def wait(self, *, timeout: float | None = None) -> int:
        _ = timeout
        self.poll_result = 0
        return 0

    def kill(self) -> None:
        self.killed = True
        self.poll_result = 0


class _FakePopenFactory:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []
        self.processes: list[_FakeProcess] = []

    def __call__(self, command: list[str], **kwargs: object) -> _FakeProcess:
        _ = kwargs
        process = _FakeProcess()
        self.commands.append(list(command))
        self.processes.append(process)
        return process


class _FakeUdpSocket:
    _next_port = 56200

    def __init__(self, sink: list[dict[str, Any]]) -> None:
        self.sink = sink
        self.closed = False
        self.bound_addr: tuple[str, int] | None = None
        self.blocking: bool | None = None
        self.incoming: list[bytes] = []

    def bind(self, addr: tuple[str, int]) -> None:
        host, port = addr
        if int(port) <= 0:
            port = _FakeUdpSocket._next_port
            _FakeUdpSocket._next_port += 1
        self.bound_addr = (str(host), int(port))

    def getsockname(self) -> tuple[str, int]:
        if self.bound_addr is None:
            return ("127.0.0.1", 0)
        return self.bound_addr

    def setblocking(self, flag: bool) -> None:
        self.blocking = bool(flag)

    def sendto(self, payload: bytes, addr: tuple[str, int]) -> int:
        message = json.loads(payload.decode("utf-8"))
        message["_addr"] = [addr[0], addr[1]]
        self.sink.append(message)
        return len(payload)

    def recvfrom(self, bufsize: int) -> tuple[bytes, tuple[str, int]]:
        _ = bufsize
        if not self.incoming:
            raise BlockingIOError()
        return self.incoming.pop(0), ("127.0.0.1", 61111)

    def queue_json(self, message: dict[str, object]) -> None:
        self.incoming.append(json.dumps(message).encode("utf-8"))

    def close(self) -> None:
        self.closed = True


class _FakeBridge:
    def __init__(self) -> None:
        self.sync_calls: list[tuple[SnapshotModel, object | None]] = []
        self.idle_calls = 0
        self.close_calls = 0
        self.active_kind: str | None = None
        self.menu_states: list[dict[str, object]] = []
        self._window_mode = "fullscreen"
        self.control_commands: list[dict[str, object]] = []

    @property
    def window_mode(self) -> str:
        return self._window_mode

    def set_menu_state(self, menu_state: dict[str, object] | None) -> None:
        self.menu_states.append({} if menu_state is None else dict(menu_state))

    def set_window_mode(self, window_mode: str) -> None:
        self._window_mode = str(window_mode).strip().lower() or "windowed"

    def poll_control_commands(self) -> list[dict[str, object]]:
        commands = list(self.control_commands)
        self.control_commands.clear()
        return commands

    def sync(self, snap: SnapshotModel, payload: object | None = None) -> bool:
        self.sync_calls.append((snap, payload))
        self.active_kind = godot_kind_for_snapshot(snap)
        return self.active_kind is not None

    def idle(self) -> None:
        self.idle_calls += 1
        self.active_kind = None

    def close(self) -> None:
        self.close_calls += 1
        self.active_kind = None

    def used_fallback(self) -> bool:
        return False

    def renderer_backend_for(self, kind: str) -> str:
        return GODOT_BACKEND_NAME if self.active_kind == kind else "pygame_2d"


class _SnapshotEngine:
    def __init__(self, snap: SnapshotModel) -> None:
        self._snap = snap
        self.submissions: list[str] = []

    def snapshot(self) -> SnapshotModel:
        return self._snap

    def can_exit(self) -> bool:
        return True

    def start_practice(self) -> None:
        pass

    def start_scored(self) -> None:
        pass

    def submit_answer(self, raw: str) -> bool:
        self.submissions.append(str(raw))
        return True

    def update(self) -> None:
        pass


def _snapshot(title: str, payload: object) -> SnapshotModel:
    return SnapshotModel(
        title=title,
        phase=Phase.PRACTICE,
        prompt="Practice prompt",
        input_hint="",
        time_remaining_s=12.0,
        attempted_scored=1,
        correct_scored=1,
        payload=payload,
    )


def _payload_snapshots() -> list[tuple[str, SnapshotModel, object]]:
    clock = _FakeClock()
    auditory = build_auditory_capacity_test(clock=clock, seed=17, difficulty=0.55)
    auditory.start_practice()
    clock.advance(0.2)
    auditory.update()
    auditory_payload = auditory.snapshot().payload
    assert isinstance(auditory_payload, AuditoryCapacityPayload)

    clock = _FakeClock()
    rapid = build_rapid_tracking_test(clock=clock, seed=551, difficulty=0.5)
    rapid.start_scored()
    rapid_payload = rapid.snapshot().payload
    assert isinstance(rapid_payload, RapidTrackingPayload)

    clock = _FakeClock()
    spatial = build_spatial_integration_test(clock=clock, seed=77, difficulty=0.55)
    spatial.start_practice()
    spatial_payload = spatial.snapshot().payload
    assert isinstance(spatial_payload, SpatialIntegrationPayload)

    clock = _FakeClock()
    trace_1 = build_trace_test_1_test(clock=clock, seed=43, difficulty=0.5)
    trace_1.start_practice()
    trace_1_payload = trace_1.snapshot().payload
    assert isinstance(trace_1_payload, TraceTest1Payload)

    clock = _FakeClock()
    trace_2 = build_trace_test_2_test(clock=clock, seed=44, difficulty=0.5)
    trace_2.start_practice()
    trace_2_payload = trace_2.snapshot().payload
    assert isinstance(trace_2_payload, TraceTest2Payload)

    return [
        ("auditory_capacity", _snapshot("Auditory Capacity", auditory_payload), auditory_payload),
        ("rapid_tracking", _snapshot("Rapid Tracking", rapid_payload), rapid_payload),
        ("spatial_integration", _snapshot("Spatial Integration", spatial_payload), spatial_payload),
        ("trace_test_1", _snapshot("Trace Test 1", trace_1_payload), trace_1_payload),
        ("trace_test_2", _snapshot("Trace Test 2", trace_2_payload), trace_2_payload),
    ]


def _manager(
    tmp_path: Path,
    *,
    popen_factory: _FakePopenFactory | None = None,
    sent: list[dict[str, Any]] | None = None,
    now: float = 1.0,
    window_mode: str = "fullscreen",
    env: dict[str, str] | None = None,
) -> GodotBridgeManager:
    project_path = tmp_path / "godot_project"
    project_path.mkdir()
    (project_path / "project.godot").write_text("config_version=5\n", encoding="utf-8")
    sink = [] if sent is None else sent
    popen = _FakePopenFactory() if popen_factory is None else popen_factory
    return GodotBridgeManager(
        project_path=project_path,
        godot_bin="/bin/echo",
        port=55123,
        env=env,
        window_mode=window_mode,
        popen_factory=popen,
        socket_factory=lambda *args, **kwargs: _FakeUdpSocket(sink),
        time_fn=lambda: now,
    )


def test_serializes_all_five_companion_godot_payloads() -> None:
    for expected_kind, snap, payload in _payload_snapshots():
        state = serialize_godot_state(snap, payload)

        assert state is not None
        assert state["kind"] == expected_kind
        assert state["schema"] == 1
        assert state["renderer_backend"] == GODOT_BACKEND_NAME
        assert state["performance"]["resolution_scale"] == 0.67
        assert isinstance(state["payload"], dict)
        assert state["payload"]
        json.dumps(state)


def test_serializes_godot_owned_start_spec_without_python_frame_payload() -> None:
    clock = _FakeClock()
    config = auditory_capacity_godot_config(
        test_code="auditory_capacity",
        mode="standard",
        difficulty=0.6,
    )
    engine = build_godot_owned_test(
        clock=clock,
        seed=2024,
        difficulty=0.6,
        kind="auditory_capacity",
        test_code="auditory_capacity",
        title="Auditory Capacity",
        config=config,
    )
    engine.start_practice()
    engine.start_scored()
    snap = engine.snapshot()
    payload = snap.payload
    assert isinstance(payload, GodotOwnedPayload)

    state = serialize_godot_state(snap, payload)

    assert state is not None
    assert state["kind"] == "auditory_capacity"
    owned_payload = state["payload"]
    assert isinstance(owned_payload, dict)
    assert set(owned_payload) == {"godot_start", "progress", "error"}
    start = owned_payload["godot_start"]
    assert isinstance(start, dict)
    assert start["command"] == "godot_start"
    assert start["authority"] == "godot"
    assert start["session_seed"] == 2024
    assert start["kind"] == "auditory_capacity"
    assert start["test_code"] == "auditory_capacity"
    assert start["config"]["tunnel_curvature_intensity"] == pytest.approx(0.88)
    assert start["config"]["difficulty_scaled"] is True
    assert start["config"]["base_tube_half_width"] == pytest.approx(0.8496)
    assert start["config"]["base_tube_half_height"] == pytest.approx(0.6136)
    assert start["config"]["tube_half_width"] == pytest.approx(0.8496)
    assert start["config"]["tube_half_height"] == pytest.approx(0.6136)
    assert start["config"]["inner_rx"] == pytest.approx(2.86)
    assert start["config"]["inner_rz"] == pytest.approx(2.10)
    assert start["config"]["active_channels"] == [
        "gates",
        "state_commands",
        "gate_directives",
        "digit_recall",
        "trigger",
        "distractors",
    ]
    assert start["config"]["segments"][0]["label"] == "Full Mixed"
    assert start["config"]["segments"][0]["effective"]["gate_interval_s"] < 3.85
    assert start["config"]["segments"][0]["effective"]["digit_sequence_max_len"] >= 6
    assert start["config"]["beep_frequency_hz"] == pytest.approx(1120.0)
    assert start["config"]["review_mode_enabled"] is False
    assert start["config"]["ambient_volume_db"] == pytest.approx(-14.0)
    assert start["config"]["ambient_layer_drop_db"] == pytest.approx(-2.5)
    assert start["config"]["primary_voice_volume"] == pytest.approx(62.0)
    assert start["config"]["distractor_voice_volume"] == pytest.approx(76.0)
    assert start["config"]["filler_voice_volume"] == pytest.approx(40.0)
    assert start["config"]["beep_volume_db"] == pytest.approx(-6.0)
    assert start["config"]["secondary_voice_min_difficulty"] == pytest.approx(0.67)
    assert start["config"]["filler_narrator_interval_s"] == pytest.approx(13.0)
    assert start["config"]["tube_chunk_count"] == 22
    assert start["config"]["tube_chunk_length"] == pytest.approx(2.0)
    assert start["config"]["tube_radial_segments"] == 16
    assert start["config"]["physics_ball_radius"] == pytest.approx(0.28)
    assert start["config"]["wall_bounce_factor"] == pytest.approx(0.42)
    assert start["audio"]["tts_required"] is True
    assert start["audio"]["ambient_layer_target"] >= 1
    assert "ball" not in owned_payload
    assert "tunnel" not in owned_payload


def test_auditory_capacity_godot_config_serializes_tube_physics_overrides() -> None:
    config = auditory_capacity_godot_config(
        test_code="auditory_capacity",
        mode="standard",
        difficulty=0.6,
        extra={
            "tube_chunk_count": 12,
            "tube_chunk_length": 3.25,
            "tube_radial_segments": 10,
            "physics_ball_radius": 0.34,
            "wall_bounce_factor": 0.55,
        },
    )
    engine = build_godot_owned_test(
        clock=_FakeClock(),
        seed=2024,
        difficulty=0.6,
        kind="auditory_capacity",
        test_code="auditory_capacity",
        title="Auditory Capacity",
        config=config,
    )
    engine.start_practice()
    engine.start_scored()
    payload = engine.snapshot().payload
    assert isinstance(payload, GodotOwnedPayload)

    state = serialize_godot_state(engine.snapshot(), payload)
    start = state["payload"]["godot_start"]

    assert start["config"]["tube_chunk_count"] == 12
    assert start["config"]["tube_chunk_length"] == pytest.approx(3.25)
    assert start["config"]["tube_radial_segments"] == 10
    assert start["config"]["physics_ball_radius"] == pytest.approx(0.34)
    assert start["config"]["wall_bounce_factor"] == pytest.approx(0.55)


def test_trace_godot_owned_start_serializes_large_slow_trace_and_phase_screens() -> None:
    clock = _FakeClock()
    engine = build_godot_owned_test(
        clock=clock,
        seed=77,
        difficulty=0.5,
        kind="trace_test_1",
        test_code="trace_test_1",
        title="Trace Test 1",
        duration_s=60.0,
        config=trace_test_godot_config(
            test_code="trace_test_1",
            mode="standard",
            difficulty=0.5,
            duration_s=60.0,
        ),
    )

    instructions_state = serialize_godot_state(engine.snapshot(), engine.snapshot().payload)
    assert instructions_state is not None
    start = instructions_state["payload"]["godot_start"]

    assert start["phase"] == "instructions"
    assert start["practice_enabled"] is True
    assert start["practice_duration_s"] == pytest.approx(12.0)
    assert set(start["phase_screens"]) >= {"instructions", "practice", "practice_done", "results"}
    assert start["config"]["trace_grid_half_x"] == 7
    assert start["config"]["trace_grid_half_z"] == 7
    assert start["config"]["trace_grid_max_y"] == 5
    assert start["config"]["trace_move_duration_s"] == pytest.approx(1.48)
    assert start["config"]["trace_pause_duration_s"] == pytest.approx(0.98)
    assert start["config"]["trace_camera_style"] == "fixed_square_orthographic"
    assert start["config"]["trace_panel_visual_scale"] == pytest.approx(2.0)
    assert start["config"]["trace1_response_window_s"] == pytest.approx(2.725)
    instructions = start["phase_screens"]["instructions"]
    assert "Watch the red aircraft" in instructions["body"]
    assert "stick movement" in instructions["body"]
    assert "Up/forward stick pushes the nose down" in instructions["controls"]
    assert "Down/back stick pulls the nose up" in instructions["controls"]
    assert "response window" in instructions["controls"]

    engine.start_practice()
    practice_snap = engine.snapshot()
    practice_state = serialize_godot_state(practice_snap, practice_snap.payload)
    practice_start = practice_state["payload"]["godot_start"]

    assert practice_start["phase"] == "practice"
    assert practice_start["duration_s"] == pytest.approx(12.0)


def test_godot_phase_messages_advance_godot_owned_python_shell() -> None:
    engine = build_godot_owned_test(
        clock=_FakeClock(),
        seed=77,
        difficulty=0.5,
        kind="trace_test_2",
        test_code="trace_test_2",
        title="Trace Test 2",
        config=trace_test_godot_config(test_code="trace_test_2", difficulty=0.5),
    )

    assert engine.phase is Phase.INSTRUCTIONS

    engine.apply_godot_authoritative_message({"command": "godot_phase_advance"})
    assert engine.phase is Phase.PRACTICE

    engine.apply_godot_authoritative_message({"command": "godot_phase_complete", "phase": "practice"})
    assert engine.phase is Phase.PRACTICE_DONE

    engine.apply_godot_authoritative_message({"command": "godot_phase_advance"})
    assert engine.phase is Phase.SCORED


def test_trace_test_2_godot_start_serializes_clip_question_instructions() -> None:
    engine = build_godot_owned_test(
        clock=_FakeClock(),
        seed=78,
        difficulty=0.5,
        kind="trace_test_2",
        test_code="trace_test_2",
        title="Trace Test 2",
        duration_s=60.0,
        config=trace_test_godot_config(
            test_code="trace_test_2",
            mode="standard",
            difficulty=0.5,
            duration_s=60.0,
        ),
    )

    state = serialize_godot_state(engine.snapshot(), engine.snapshot().payload)
    start = state["payload"]["godot_start"]
    instructions = start["phase_screens"]["instructions"]

    assert start["config"]["trace2_observe_s"] == pytest.approx(4.725)
    assert start["config"]["trace2_clip_min_s"] == pytest.approx(3.8)
    assert start["config"]["trace2_clip_max_s"] == pytest.approx(6.5)
    assert start["config"]["trace2_question_window_s"] == pytest.approx(6.4)
    assert start["config"]["trace2_offscreen_margin_cells"] == pytest.approx(2.25)
    assert start["config"]["trace2_close_camera_scale"] == pytest.approx(0.64)
    assert start["config"]["trace2_aircraft_scale"] == pytest.approx(1.45)
    assert start["config"]["trace2_question_overlay_enabled"] is True
    assert start["config"]["trace2_new_clip_after_answer"] is True
    assert "short clip" in instructions["body"]
    assert "full-screen multiple-choice question" in instructions["body"]
    assert "1-4 or A/S/D/F" in instructions["controls"]
    assert "started on top" in instructions["controls"]
    assert "left or right turn" in instructions["controls"]
    assert "ended off screen" in instructions["controls"]


def test_auditory_capacity_godot_config_serializes_review_mode() -> None:
    config = auditory_capacity_godot_config(
        test_code="auditory_capacity",
        mode="standard",
        difficulty=0.6,
        review_mode_enabled=True,
    )

    assert config["review_mode_enabled"] is True


def test_auditory_capacity_godot_config_scales_with_difficulty() -> None:
    low = auditory_capacity_godot_config(
        test_code="auditory_capacity",
        mode="standard",
        difficulty=0.0,
    )
    high = auditory_capacity_godot_config(
        test_code="auditory_capacity",
        mode="standard",
        difficulty=1.0,
    )

    assert low["callsign_count"] == 3
    assert high["callsign_count"] == 4
    assert low["digit_sequence_min_len"] == 4
    assert high["digit_sequence_min_len"] == 10
    assert high["gate_interval_s"] < low["gate_interval_s"]
    assert high["audio"]["ambient_layer_target"] > low["audio"]["ambient_layer_target"]


def test_auditory_capacity_godot_config_serializes_drill_channels() -> None:
    config = auditory_capacity_godot_config(
        test_code="ac_trigger_cue_anchor",
        mode="build",
        difficulty=0.55,
        duration_s=45.0,
        extra={"drill": True},
    )

    assert config["drill"] is True
    assert config["active_channels"] == ["gates", "trigger"]
    assert config["segments"][0]["label"] == "Trigger Cues"
    assert config["segments"][0]["duration_s"] == pytest.approx(45.0)
    assert config["segments"][0]["effective"]["beep_interval_s"] > config["base_beep_interval_s"]
    assert config["segments"][0]["effective"]["tube_half_width"] > config["base_tube_half_width"]


def test_auditory_capacity_workout_config_contains_focused_segments() -> None:
    config = auditory_capacity_godot_config(
        test_code="auditory_capacity_workout",
        mode="workout",
        difficulty=0.55,
        duration_s=90.0 * 60.0,
    )
    labels = [str(segment["label"]) for segment in config["segments"]]

    assert config["workout"] is True
    assert labels[:6] == [
        "Gate Flight",
        "State Commands",
        "Gate Directives",
        "Digit Recall",
        "Trigger Cues",
        "Callsign Filter",
    ]
    assert "Pressure Run" in labels


def test_serializes_rapid_tracking_godot_owned_start_spec_for_dedicated_runtime() -> None:
    clock = _FakeClock()
    config = rapid_tracking_godot_config(
        test_code="rt_building_handoff_prime",
        mode="adaptive",
        difficulty=0.62,
        duration_s=45.0,
        extra={"capture_box_half_width": 0.12, "capture_cooldown_s": 0.3},
    )
    engine = build_godot_owned_test(
        clock=clock,
        seed=551,
        difficulty=0.62,
        kind="rapid_tracking",
        test_code="rt_building_handoff_prime",
        title="Rapid Tracking: Building Handoff Prime",
        duration_s=45.0,
        mode="adaptive",
        config=config,
    )
    engine.start_practice()
    engine.start_scored()
    snap = engine.snapshot()
    payload = snap.payload
    assert isinstance(payload, GodotOwnedPayload)

    state = serialize_godot_state(snap, payload)

    assert state is not None
    assert state["kind"] == "rapid_tracking"
    owned_payload = state["payload"]
    assert isinstance(owned_payload, dict)
    assert set(owned_payload) == {"godot_start", "progress", "error"}
    start = owned_payload["godot_start"]
    assert isinstance(start, dict)
    assert start["command"] == "godot_start"
    assert start["authority"] == "godot"
    assert start["kind"] == "rapid_tracking"
    assert start["test_code"] == "rt_building_handoff_prime"
    assert start["session_seed"] == 551
    assert start["duration_s"] == 45.0
    assert start["mode"] == "adaptive"
    assert start["config"]["rapid_tracking"] is True
    assert start["config"]["difficulty_scaled"] is True
    assert start["config"]["active_target_mix"] == "ground_heavy"
    assert start["config"]["reticle_mode"] == "fixed_center"
    assert start["config"]["unlimited_gimbal"] is True
    assert start["config"]["capture_box_half_width"] == 0.12
    assert start["config"]["capture_cooldown_s"] == 0.3
    assert start["config"]["rapid_world"]["scene_style"] == "low_poly_large"
    assert start["config"]["rapid_world"]["vehicle_count"] >= 10
    assert start["config"]["rapid_world"]["pedestrian_count"] >= 6
    assert start["config"]["rapid_world"]["tunnel_count"] >= 1
    assert start["config"]["rapid_world"]["ground_target_weight"] > 0.5
    assert start["config"]["rapid_world"]["chunked_generation"] is True
    assert start["config"]["rapid_world"]["world_size_m"] == 300.0
    assert start["config"]["rapid_world"]["chunk_grid_cols"] == 50
    assert start["config"]["rapid_world"]["chunk_grid_rows"] == 50
    assert start["config"]["rapid_world"]["chunk_cell_size_m"] == 6.0
    assert start["config"]["rapid_world"]["chunk_pack"] == "rural_mixed_v1"
    assert start["config"]["rapid_world"]["asset_spawn_policy"] == "socketed"
    assert start["config"]["rapid_world"]["road_topology"] == "organic_looped"
    assert start["config"]["rapid_world"]["road_buffer_cells"] == 1
    assert start["config"]["rapid_world"]["terrain_pipeline"] == "terrain_first_v3"
    assert "target" not in owned_payload
    assert "camera" not in owned_payload


def test_rapid_tracking_godot_config_scales_world_with_difficulty() -> None:
    low = rapid_tracking_godot_config(
        test_code="rapid_tracking",
        mode="standard",
        difficulty=0.0,
    )
    high = rapid_tracking_godot_config(
        test_code="rapid_tracking",
        mode="standard",
        difficulty=1.0,
    )

    assert high["rapid_world"]["world_size_m"] == low["rapid_world"]["world_size_m"] == 300.0
    assert low["rapid_world"]["chunk_grid_cols"] == 50
    assert low["rapid_world"]["chunk_grid_rows"] == 50
    assert low["rapid_world"]["chunk_cell_size_m"] == 6.0
    assert low["rapid_world"]["terrain_pipeline"] == "terrain_first_v3"
    assert high["rapid_world"]["town_count"] > low["rapid_world"]["town_count"]
    assert high["rapid_world"]["vehicle_count"] > low["rapid_world"]["vehicle_count"]
    assert high["rapid_world"]["pedestrian_count"] > low["rapid_world"]["pedestrian_count"]
    assert high["rapid_world"]["forest_patch_count"] > low["rapid_world"]["forest_patch_count"]
    assert high["rapid_world"]["occlusion_density"] > low["rapid_world"]["occlusion_density"]
    assert high["handoff_interval_s"] < low["handoff_interval_s"]
    assert low["handoff_interval_s"] == pytest.approx(15.0)
    assert high["handoff_interval_s"] == pytest.approx(8.2)
    assert low["target_speed_scale"] == pytest.approx(0.52)
    assert high["target_speed_scale"] > low["target_speed_scale"]
    assert low["zoom_fov"] == pytest.approx(16.0)
    assert low["target_zoom_time_scale"] == pytest.approx(0.42)
    assert low["zoom_bonus_points"] == pytest.approx(1.0)
    assert low["target_guide_mode"] == "offscreen_pip"
    assert high["zoom_bonus_interval_s"] < low["zoom_bonus_interval_s"]


def test_spatial_integration_godot_config_defaults_to_large_scene_questions() -> None:
    config = spatial_integration_godot_config(
        test_code="spatial_integration",
        mode="standard",
    )

    assert config["grid_cols"] == 24
    assert config["grid_rows"] == 24
    assert config["answer_grid_cols"] == 3
    assert config["answer_grid_rows"] == 3
    assert config["question_time_limit_s"] == 0.0
    assert config["chunk_grid_cols"] == 24
    assert config["chunk_grid_rows"] == 24
    assert config["terrain_pipeline"] == "si_large_scene_v2"
    assert "scene_presence" in config["allowed_question_kinds"]
    assert "viewpoint_match" in config["allowed_question_kinds"]
    assert "object_count" in config["allowed_question_kinds"]
    assert "object_relation" in config["allowed_question_kinds"]
    assert "aircraft_color_route_selection" in config["allowed_question_kinds"]
    assert "aircraft_count" in config["allowed_question_kinds"]
    assert "aircraft_presence" in config["allowed_question_kinds"]
    assert "aircraft_order" in config["allowed_question_kinds"]
    assert "landmark_grid" not in config["allowed_question_kinds"]
    assert "object_kind_at_cell" not in config["allowed_question_kinds"]
    assert "aircraft_location_grid" not in config["allowed_question_kinds"]
    assert "aircraft_color_location_grid" not in config["allowed_question_kinds"]

    aircraft = spatial_integration_godot_config(
        test_code="si_moving_aircraft_multiview_integration",
        mode="drill",
    )
    assert "scene_presence" not in aircraft["allowed_question_kinds"]
    assert "viewpoint_match" not in aircraft["allowed_question_kinds"]
    assert "aircraft_color_route_selection" in aircraft["allowed_question_kinds"]
    assert "aircraft_count" in aircraft["allowed_question_kinds"]
    assert "aircraft_presence" in aircraft["allowed_question_kinds"]
    assert "aircraft_order" in aircraft["allowed_question_kinds"]
    assert "aircraft_location_grid" not in aircraft["allowed_question_kinds"]
    assert "aircraft_color_location_grid" not in aircraft["allowed_question_kinds"]


def test_serializes_spatial_integration_godot_owned_start_spec_for_dedicated_runtime() -> None:
    clock = _FakeClock()
    config = spatial_integration_godot_config(
        test_code="si_route_anchor",
        mode="drill",
        duration_s=60.0,
        extra={"question_limit": 5},
    )
    engine = build_godot_owned_test(
        clock=clock,
        seed=713,
        difficulty=0.58,
        kind="spatial_integration",
        test_code="si_route_anchor",
        title="Spatial Integration: Route Anchor",
        duration_s=60.0,
        mode="drill",
        config=config,
    )
    engine.start_practice()
    engine.start_scored()
    snap = engine.snapshot()
    payload = snap.payload
    assert isinstance(payload, GodotOwnedPayload)

    state = serialize_godot_state(snap, payload)

    assert state is not None
    assert state["kind"] == "spatial_integration"
    owned_payload = state["payload"]
    assert isinstance(owned_payload, dict)
    assert set(owned_payload) == {"godot_start", "progress", "error"}
    start = owned_payload["godot_start"]
    assert isinstance(start, dict)
    assert start["command"] == "godot_start"
    assert start["authority"] == "godot"
    assert start["kind"] == "spatial_integration"
    assert start["test_code"] == "si_route_anchor"
    assert start["session_seed"] == 713
    assert start["duration_s"] == 60.0
    assert start["mode"] == "drill"
    assert start["config"]["parts"] == ["aircraft"]
    assert start["config"]["allowed_question_kinds"] == ["aircraft_route_selection"]
    assert start["config"]["static_study_s"] == 12.0
    assert start["config"]["aircraft_study_s"] == 15.0
    assert start["config"]["question_time_limit_s"] == 0.0
    assert start["config"]["chunked_generation"] is True
    assert start["config"]["grid_cols"] == 24
    assert start["config"]["grid_rows"] == 24
    assert start["config"]["answer_grid_cols"] == 3
    assert start["config"]["answer_grid_rows"] == 3
    assert start["config"]["chunk_grid_cols"] == 24
    assert start["config"]["chunk_grid_rows"] == 24
    assert start["config"]["chunk_pack"] == "rural_mixed_v1"
    assert start["config"]["asset_spawn_policy"] == "socketed"
    assert start["config"]["terrain_pipeline"] == "si_large_scene_v2"
    assert start["config"]["question_limit"] == 5
    assert "scene" not in owned_payload
    assert "questions" not in owned_payload


def test_godot_project_routes_rapid_tracking_to_dedicated_runtime() -> None:
    scripts_dir = GODOT_PROJECT_PATH / "scripts"
    main_source = (scripts_dir / "main.gd").read_text(encoding="utf-8")
    runtime_source = (scripts_dir / "rapid_tracking_runtime.gd").read_text(encoding="utf-8")

    assert 'preload("res://scripts/rapid_tracking_runtime.gd")' in main_source
    assert 'godot_owned_runtime.name = "RapidTrackingRuntime"' in main_source
    assert "func rapid_tracking_runtime_marker()" in runtime_source
    assert "rural_house_clusters" in runtime_source
    assert "rapid_world_config" in runtime_source
    assert "_generate_road_graph" in runtime_source
    assert "_find_path_nodes" in runtime_source
    assert '"pathfinding": "road_graph"' in runtime_source
    assert "SeededHeightfieldTerrain" in runtime_source
    assert "Lake" in runtime_source
    assert "River" in runtime_source
    assert "TunnelPortal" in runtime_source
    assert "PathGraphMover" in runtime_source
    assert "person" in runtime_source
    assert "road_graph_hash" in runtime_source
    assert "route_hash" in runtime_source
    assert 'preload("res://scripts/chunk_map_generator.gd")' in runtime_source
    assert "ChunkMapGenerator.generate" in runtime_source
    assert "asset_spawn_policy" in runtime_source
    assert "spawn_socket" in runtime_source
    assert "tank" in runtime_source
    assert "chunk_map_hash" in runtime_source
    assert "road_component_count" in runtime_source
    assert "road_dead_end_count" in runtime_source
    assert "road_buffer_violation_count" in runtime_source
    assert "water_feature_count" in runtime_source
    assert "aim_angles" in runtime_source
    assert "wrapf(aim_angles.x" in runtime_source
    assert "deg_to_rad(-86.0)" in runtime_source
    assert "base_focus" in runtime_source
    assert "turbulence_strength" in runtime_source
    assert "Vector2.ZERO - target_screen_pos" in runtime_source
    assert "_norm_to_pixel(Vector2.ZERO" in runtime_source
    assert '"response": "camera_aim"' in runtime_source
    assert "KEY_KP_PERIOD" in runtime_source
    assert "KEY_KP_ENTER" in runtime_source
    assert "godot_complete" in runtime_source


def test_godot_project_routes_spatial_integration_to_dedicated_runtime() -> None:
    scripts_dir = GODOT_PROJECT_PATH / "scripts"
    main_source = (scripts_dir / "main.gd").read_text(encoding="utf-8")
    runtime_source = (scripts_dir / "spatial_integration_runtime.gd").read_text(
        encoding="utf-8"
    )

    assert 'preload("res://scripts/spatial_integration_runtime.gd")' in main_source
    assert 'godot_owned_runtime.name = "SpatialIntegrationRuntime"' in main_source
    assert "func spatial_integration_runtime_marker()" in runtime_source
    assert "object_relation" in runtime_source
    assert "scene_reconstruction" in runtime_source
    assert "aircraft_route_selection" in runtime_source
    assert "aircraft_continuation_selection" in runtime_source
    assert "aircraft_count" in runtime_source
    assert "aircraft_presence" in runtime_source
    assert "aircraft_order" in runtime_source
    assert "scene_hash" in runtime_source
    assert "route_hash" in runtime_source
    assert 'preload("res://scripts/chunk_map_generator.gd")' in runtime_source
    assert "ChunkMapGenerator.generate" in runtime_source
    assert "_draw_chunked_grid_terrain" in runtime_source
    assert "_chunk_aircraft_route" in runtime_source
    assert "_chunk_landmarks" in runtime_source
    assert "chunk_map_hash" in runtime_source
    assert "question_order_hash" in runtime_source
    assert "option_order_hash" in runtime_source
    assert 'var show_scene := stage != "question"' in runtime_source
    assert "NorthSceneMarker" in runtime_source
    assert "_build_north_scene_marker" in runtime_source
    assert "_update_north_marker" in runtime_source
    assert 'north_marker_root.visible = active and stage != "question"' in runtime_source
    assert "study_orientation_index" in runtime_source
    assert "_study_camera_position" in runtime_source
    assert "_record_answer(\"TIMEOUT\"" not in runtime_source
    assert "KEY_KP_PERIOD" in runtime_source
    assert "KEY_KP_ENTER" in runtime_source
    assert "godot_complete" in runtime_source


def test_godot_owned_trace_aircraft_use_discrete_grid_step_motion() -> None:
    runtime_source = (GODOT_PROJECT_PATH / "scripts" / "godot_owned_runtime.gd").read_text(
        encoding="utf-8"
    )

    assert "TRACE_DEFAULT_GRID_HALF_X" in runtime_source
    assert "trace_grid_half_x" in runtime_source
    assert "TRACE_DIRS" in runtime_source
    assert "func _init_trace_tracks()" in runtime_source
    assert "func _step_trace_tracks" in runtime_source
    assert "func _trace_begin_move" in runtime_source
    assert "func _trace_choose_next_dir" in runtime_source
    assert "func _trace_forward_blocked" in runtime_source
    assert "forced_edge_turn" in runtime_source
    assert '"phase": "pause"' in runtime_source
    assert 'track["phase"] = "move"' in runtime_source
    assert "absf(dot) > 0.05" in runtime_source
    assert "return current_idx" in runtime_source
    assert "root.look_at(pos + look_dir, up)" in runtime_source


def test_auditory_capacity_payload_includes_twisted_tunnel_and_gate_poses() -> None:
    clock = _FakeClock()
    engine = build_auditory_capacity_test(clock=clock, seed=17, difficulty=0.55)
    engine.start_practice()
    engine._next_gate_at_s = 0.0
    engine._update_gates(0.0)
    payload = engine.snapshot().payload
    assert isinstance(payload, AuditoryCapacityPayload)
    snap = _snapshot("Auditory Capacity", payload)
    state = serialize_godot_state(snap, payload)

    assert state is not None
    ac = state["payload"]
    assert isinstance(ac, dict)
    tunnel = ac["tunnel"]
    assert isinstance(tunnel, dict)
    assert tunnel["twist_intensity"] > 0.0
    assert tunnel["curvature_intensity"] > 0.0
    assert len(tunnel["samples"]) >= 18
    first_sample = tunnel["samples"][0]
    assert {"pos", "tangent", "right", "up", "twist_angle_rad"} <= set(first_sample)
    assert {"position", "pose", "visual_radius"} <= set(ac["ball"])

    gates = ac["gates"]
    assert isinstance(gates, list)
    assert gates
    for gate in gates:
        assert gate["color"] in {"RED", "BLUE", "YELLOW"}
        assert gate["shape"] in {"CIRCLE", "TRIANGLE", "SQUARE"}
        assert "position" in gate
        assert "pose" in gate
        assert gate["aperture_radius"] > 0.0

    json.dumps(state)


def test_auditory_capacity_payload_includes_godot_authoritative_run_spec() -> None:
    clock = _FakeClock()
    engine = build_auditory_capacity_test(clock=clock, seed=17, difficulty=0.55)
    engine.start_practice()
    clock.advance(0.2)
    engine.update()
    payload = engine.snapshot().payload
    assert isinstance(payload, AuditoryCapacityPayload)
    state = serialize_godot_state(_snapshot("Auditory Capacity", payload), payload)

    assert state is not None
    ac = state["payload"]
    assert isinstance(ac, dict)
    runtime = ac["godot_runtime"]
    assert isinstance(runtime, dict)
    assert runtime["command"] == "auditory_start"
    assert runtime["authority"] == "godot"
    assert runtime["session_seed"] == 17
    assert runtime["difficulty"] == pytest.approx(0.55)
    assert runtime["asset_root"].endswith("assets/audio/auditory_capacity")
    assert runtime["constants"]["colors"] == ["RED", "BLUE", "YELLOW"]
    assert runtime["constants"]["shapes"] == ["CIRCLE", "TRIANGLE", "SQUARE"]
    assert runtime["config"]["tunnel_curvature_intensity"] > 0.0
    assert runtime["config"]["tunnel_twist_intensity"] > 0.0
    json.dumps(state)


def test_auditory_capacity_godot_payload_uses_seeded_tunnel_path() -> None:
    def auditory_state(seed: int) -> dict[str, object]:
        clock = _FakeClock()
        engine = build_auditory_capacity_test(clock=clock, seed=seed, difficulty=0.55)
        engine.start_practice()
        engine._next_gate_at_s = 0.0
        engine._update_gates(0.0)
        clock.advance(0.25)
        engine.update()
        payload = engine.snapshot().payload
        assert isinstance(payload, AuditoryCapacityPayload)
        state = serialize_godot_state(_snapshot("Auditory Capacity", payload), payload)
        assert state is not None
        ac = state["payload"]
        assert isinstance(ac, dict)
        return ac

    first = auditory_state(17)
    repeated = auditory_state(17)
    other_seed = auditory_state(99)

    first_tunnel = first["tunnel"]
    repeated_tunnel = repeated["tunnel"]
    other_tunnel = other_seed["tunnel"]
    assert isinstance(first_tunnel, dict)
    assert isinstance(repeated_tunnel, dict)
    assert isinstance(other_tunnel, dict)

    first_samples = first_tunnel["samples"]
    repeated_samples = repeated_tunnel["samples"]
    other_samples = other_tunnel["samples"]
    assert isinstance(first_samples, list)
    assert isinstance(repeated_samples, list)
    assert isinstance(other_samples, list)
    assert first_samples[4]["pos"] == repeated_samples[4]["pos"]
    assert first_samples[4]["pos"] != other_samples[4]["pos"]

    first_gates = first["gates"]
    repeated_gates = repeated["gates"]
    other_gates = other_seed["gates"]
    assert isinstance(first_gates, list)
    assert isinstance(repeated_gates, list)
    assert isinstance(other_gates, list)
    assert first_gates[0]["position"] == repeated_gates[0]["position"]
    assert first_gates[0]["position"] != other_gates[0]["position"]


@pytest.mark.parametrize("kind", list(InstrumentComprehensionTrialKind))
def test_instrument_comprehension_payloads_stay_pygame_only(
    kind: InstrumentComprehensionTrialKind,
) -> None:
    payload = InstrumentComprehensionGenerator(seed=117).next_problem_for_kind(
        kind=kind,
        difficulty=0.55,
    ).payload

    assert isinstance(payload, InstrumentComprehensionPayload)
    snap = _snapshot("Instrument Comprehension", payload)

    assert godot_kind_for_snapshot(snap) is None
    assert serialize_godot_state(snap, payload) is None


def test_bridge_launches_godot_and_sends_udp_json(tmp_path: Path) -> None:
    sent: list[dict[str, Any]] = []
    popen = _FakePopenFactory()
    manager = _manager(tmp_path, popen_factory=popen, sent=sent)
    _kind, snap, payload = _payload_snapshots()[1]

    assert manager.sync(snap, payload) is True

    assert len(popen.commands) == 1
    command = popen.commands[0]
    assert command[:4] == ["/bin/echo", "--path", str(tmp_path / "godot_project"), "--fullscreen"]
    assert "--listen-port" in command
    assert "--control-port" in command
    assert str(manager.control_port) in command
    assert sent[-1]["kind"] == "rapid_tracking"
    assert sent[-1]["renderer_backend"] == GODOT_BACKEND_NAME
    assert sent[-1]["window_mode"] == "fullscreen"
    assert sent[-1]["session_id"] == manager.session_id
    assert sent[-1]["menu"] == {"active": False}
    assert manager.is_active_for("rapid_tracking")


def test_bridge_launches_windowed_when_requested(tmp_path: Path) -> None:
    sent: list[dict[str, Any]] = []
    popen = _FakePopenFactory()
    manager = _manager(tmp_path, popen_factory=popen, sent=sent, window_mode="windowed")
    _kind, snap, payload = _payload_snapshots()[1]

    assert manager.sync(snap, payload) is True

    command = popen.commands[0]
    assert "--windowed" in command
    assert "--resolution" in command
    assert "960x540" in command
    assert sent[-1]["window_mode"] == "windowed"


def test_bridge_can_force_companion_window_mode_with_env(tmp_path: Path) -> None:
    sent: list[dict[str, Any]] = []
    popen = _FakePopenFactory()
    manager = _manager(
        tmp_path,
        popen_factory=popen,
        sent=sent,
        window_mode="windowed",
        env={"CFAST_GODOT_WINDOW_MODE": "maximized"},
    )
    _kind, snap, payload = _payload_snapshots()[1]

    assert manager.window_mode == "maximized"
    manager.set_window_mode("fullscreen")
    assert manager.window_mode == "maximized"
    assert manager.sync(snap, payload) is True

    assert "--maximized" in popen.commands[0]
    assert sent[-1]["window_mode"] == "maximized"


def test_bridge_streams_window_mode_changes_to_running_godot(tmp_path: Path) -> None:
    sent: list[dict[str, Any]] = []
    popen = _FakePopenFactory()
    clock = _FakeClock(t=1.0)
    project_path = tmp_path / "godot_project"
    project_path.mkdir()
    (project_path / "project.godot").write_text("config_version=5\n", encoding="utf-8")
    manager = GodotBridgeManager(
        project_path=project_path,
        godot_bin="/bin/echo",
        port=55123,
        window_mode="windowed",
        popen_factory=popen,
        socket_factory=lambda *args, **kwargs: _FakeUdpSocket(sent),
        time_fn=clock.now,
    )
    _kind, snap, payload = _payload_snapshots()[1]

    assert manager.sync(snap, payload) is True
    manager.set_window_mode("fullscreen")
    clock.advance(1.0)
    assert manager.sync(snap, payload) is True

    assert len(popen.commands) == 1
    assert sent[-1]["window_mode"] == "fullscreen"


def test_bridge_polls_validated_godot_control_packets(tmp_path: Path) -> None:
    sent: list[dict[str, Any]] = []
    popen = _FakePopenFactory()
    manager = _manager(tmp_path, popen_factory=popen, sent=sent)
    _kind, snap, payload = _payload_snapshots()[1]

    assert manager.sync(snap, payload) is True
    control_socket = manager._control_socket
    assert isinstance(control_socket, _FakeUdpSocket)
    control_socket.queue_json(
        {
            "schema": 1,
            "session_id": manager.session_id,
            "command": "set_window_mode",
            "window_mode": "windowed",
        }
    )
    control_socket.queue_json(
        {
            "schema": 1,
            "session_id": "wrong-session",
            "command": "back_to_tests",
        }
    )
    control_socket.queue_json(
        {
            "schema": 1,
            "session_id": manager.session_id,
            "command": "godot_phase_advance",
            "run_key": "trace_test_1:77:standard:launch:instructions",
            "phase": "instructions",
            "kind": "trace_test_1",
            "test_code": "trace_test_1",
        }
    )

    assert manager.poll_control_commands() == [
        {"command": "set_window_mode", "window_mode": "windowed"},
        {
            "command": "godot_phase_advance",
            "run_key": "trace_test_1:77:standard:launch:instructions",
            "phase": "instructions",
            "kind": "trace_test_1",
            "test_code": "trace_test_1",
        },
    ]
    assert manager.window_mode == "windowed"


def test_bridge_polls_auditory_lifecycle_packets(tmp_path: Path) -> None:
    sent: list[dict[str, Any]] = []
    popen = _FakePopenFactory()
    manager = _manager(tmp_path, popen_factory=popen, sent=sent)
    _kind, snap, payload = _payload_snapshots()[0]

    assert manager.sync(snap, payload) is True
    control_socket = manager._control_socket
    assert isinstance(control_socket, _FakeUdpSocket)
    control_socket.queue_json(
        {
            "schema": 1,
            "session_id": manager.session_id,
            "command": "auditory_complete",
            "run_key": "17:practice",
            "phase": "results",
            "result": {
                "summary": {"attempted": 4, "correct": 3, "total_score": 3.0, "max_score": 4.0},
                "metrics": {"gate_hits": 2, "gate_misses": 1, "collisions": 0},
            },
        }
    )

    commands = manager.poll_control_commands()

    assert commands == [
        {
            "command": "auditory_complete",
            "run_key": "17:practice",
            "phase": "results",
            "result": {
                "summary": {"attempted": 4, "correct": 3, "total_score": 3.0, "max_score": 4.0},
                "metrics": {"gate_hits": 2, "gate_misses": 1, "collisions": 0},
            },
        }
    ]


def test_bridge_polls_generic_godot_lifecycle_packets(tmp_path: Path) -> None:
    sent: list[dict[str, Any]] = []
    popen = _FakePopenFactory()
    manager = _manager(tmp_path, popen_factory=popen, sent=sent)
    engine = build_godot_owned_test(
        clock=_FakeClock(),
        seed=9,
        difficulty=0.5,
        kind="rapid_tracking",
        test_code="rapid_tracking",
        title="Rapid Tracking",
    )
    engine.start_practice()
    engine.start_scored()
    snap = engine.snapshot()

    assert manager.sync(snap, snap.payload) is True
    control_socket = manager._control_socket
    assert isinstance(control_socket, _FakeUdpSocket)
    control_socket.queue_json(
        {
            "schema": 1,
            "session_id": manager.session_id,
            "command": "godot_complete",
            "kind": "rapid_tracking",
            "test_code": "rapid_tracking",
            "run_key": "rapid_tracking:9:standard:scored",
            "phase": "results",
            "result": {
                "summary": {"attempted": 2, "correct": 1, "total_score": 1.0, "max_score": 2.0},
                "metrics": {"mean_tracking_error": 0.25},
                "events": [{"kind": "sample", "is_correct": True, "occurred_at_ms": 100}],
            },
        }
    )

    commands = manager.poll_control_commands()

    assert commands == [
        {
            "command": "godot_complete",
            "kind": "rapid_tracking",
            "test_code": "rapid_tracking",
            "run_key": "rapid_tracking:9:standard:scored",
            "phase": "results",
            "result": {
                "summary": {"attempted": 2, "correct": 1, "total_score": 1.0, "max_score": 2.0},
                "metrics": {"mean_tracking_error": 0.25},
                "events": [{"kind": "sample", "is_correct": True, "occurred_at_ms": 100}],
            },
        }
    ]


def test_bridge_restarts_crashed_process_once_then_falls_back(tmp_path: Path) -> None:
    sent: list[dict[str, Any]] = []
    popen = _FakePopenFactory()
    manager = _manager(tmp_path, popen_factory=popen, sent=sent, now=1.0)
    _kind, snap, payload = _payload_snapshots()[1]

    assert manager.sync(snap, payload) is True
    popen.processes[-1].poll_result = 1
    assert manager.sync(snap, payload) is True
    popen.processes[-1].poll_result = 1
    assert manager.sync(snap, payload) is False

    assert len(popen.commands) == 2
    assert manager.used_fallback() is True


def test_bridge_closes_process_on_screen_exit(tmp_path: Path) -> None:
    sent: list[dict[str, Any]] = []
    popen = _FakePopenFactory()
    manager = _manager(tmp_path, popen_factory=popen, sent=sent)
    _kind, snap, payload = _payload_snapshots()[1]

    assert manager.sync(snap, payload) is True
    manager.close()

    assert sent[-1]["command"] == "quit"
    assert popen.processes[-1].terminated is True
    assert manager.active_kind is None


def test_headless_bridge_never_launches(tmp_path: Path) -> None:
    popen = _FakePopenFactory()
    _kind, snap, payload = _payload_snapshots()[1]
    manager = GodotBridgeManager(
        headless=True,
        project_path=tmp_path,
        godot_bin="/bin/echo",
        popen_factory=popen,
    )

    assert manager.sync(snap, payload) is False
    assert popen.commands == []


def test_default_app_bridge_is_suppressed_under_dummy_video_driver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.delenv("CFAST_ENABLE_GODOT_IN_TESTS", raising=False)
    monkeypatch.setenv("CFAST_GODOT_BIN", "/bin/echo")
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        app = App(surface=surface, font=font)
        _kind, snap, payload = _payload_snapshots()[1]

        assert app.godot_bridge().sync(snap, payload) is False
        assert app.current_run_state().renderer_path == "PYGAME_2D"
    finally:
        pygame.quit()


def test_app_display_mode_is_independent_from_godot_bridge_window_mode() -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        app = App(surface=surface, font=font, window_mode="windowed")

        assert app.godot_bridge().window_mode == "fullscreen"

        app.set_window_mode("fullscreen")

        assert app.godot_bridge().window_mode == "fullscreen"

        app.set_godot_window_mode("windowed")

        assert app.godot_bridge().window_mode == "windowed"
    finally:
        pygame.quit()


@pytest.mark.parametrize(
    ("kind", "method_name"),
    [
        ("auditory_capacity", "_render_auditory_capacity_screen"),
        ("rapid_tracking", "_render_rapid_tracking_screen"),
        ("spatial_integration", "_render_spatial_integration_screen"),
        ("trace_test_1", "_render_trace_test_1_screen"),
        ("trace_test_2", "_render_trace_test_2_screen"),
    ],
)
def test_cognitive_screen_updates_godot_bridge_and_keeps_pygame_renderer(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    method_name: str,
) -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        app = App(surface=surface, font=font)
        fake_bridge = _FakeBridge()
        app._godot_bridge = fake_bridge
        app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))
        payload_map = {payload_kind: (snap, payload) for payload_kind, snap, payload in _payload_snapshots()}
        snap, payload = payload_map[kind]
        screen = CognitiveTestScreen(app, engine_factory=lambda: _SnapshotEngine(snap))
        app.push(screen)
        called = {"value": False}

        def fake_render(*args: object, **kwargs: object) -> None:
            _ = args, kwargs
            called["value"] = True

        monkeypatch.setattr(screen, method_name, fake_render)
        screen.render(surface)

        assert fake_bridge.sync_calls[-1] == (snap, payload)
        assert called["value"] is True
        assert app.current_run_state().renderer_path == "GODOT_4"
    finally:
        pygame.quit()


def test_godot_pause_toggle_opens_existing_pause_menu() -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        app = App(surface=surface, font=font)
        fake_bridge = _FakeBridge()
        app._godot_bridge = fake_bridge
        app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))
        _kind, snap, _payload = _payload_snapshots()[1]
        screen = CognitiveTestScreen(
            app,
            engine_factory=lambda: _SnapshotEngine(snap),
            test_code="rapid_tracking",
        )
        app.push(screen)

        fake_bridge.control_commands.append({"command": "pause_toggle"})
        app.render()

        assert app.shell_pause_overlay_active() is True
        assert screen.shell_pause_menu_active() is True
        assert fake_bridge.menu_states[-1]["active"] is True
    finally:
        pygame.quit()


def test_godot_menu_commands_reuse_pause_settings_handlers() -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        app = App(surface=surface, font=font)
        fake_bridge = _FakeBridge()
        app._godot_bridge = fake_bridge
        app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))
        _kind, snap, _payload = _payload_snapshots()[1]
        screen = CognitiveTestScreen(
            app,
            engine_factory=lambda: _SnapshotEngine(snap),
            test_code="rapid_tracking",
        )
        app.push(screen)

        fake_bridge.control_commands.extend(
            [
                {"command": "pause_toggle"},
                {"command": "activate_action", "action": "settings"},
                {"command": "adjust_setting", "key": "difficulty", "direction": 1},
            ]
        )
        app.render()

        assert screen.shell_pause_menu_active() is True
        assert screen._pause_menu_mode == "settings"
        assert screen._pause_settings_rows()[screen._pause_settings_selected][0] == "difficulty"
        assert screen._staged_difficulty_level == 6
    finally:
        pygame.quit()


def test_godot_phase_advance_reaches_nested_rapid_tracking_workout_engine() -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        app = App(surface=surface, font=font)
        fake_bridge = _FakeBridge()
        app._godot_bridge = fake_bridge
        app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))
        clock = _FakeClock()
        session = AntWorkoutSession(
            clock=clock,
            seed=2026,
            plan=AntWorkoutPlan(
                code="rapid_tracking_workout",
                title="Rapid Tracking Workout",
                description="Nested route regression.",
                notes=(),
                blocks=(
                    AntWorkoutBlockPlan(
                        block_id="rt",
                        label="RT",
                        description="RT block.",
                        focus_skills=("Tracking",),
                        drill_code="rt_lock_anchor",
                        mode=AntDrillMode.BUILD,
                        duration_min=0.25,
                    ),
                ),
            ),
            starting_level=5,
        )
        session.activate()
        session.activate()
        assert session.stage is AntWorkoutStage.BLOCK
        engine = session.current_engine()
        assert engine is not None
        payload = engine.snapshot().payload
        assert isinstance(payload, GodotOwnedPayload)
        assert engine.phase is Phase.SCORED
        screen = AntWorkoutScreen(
            app,
            session=session,
            test_code="rapid_tracking_workout",
        )
        app.push(screen)

        fake_bridge.control_commands.append(
            {
                "command": "complete",
                "run_key": payload.spec.run_key,
                "test_code": "rt_lock_anchor",
                "kind": "rapid_tracking",
                "summary": {
                    "attempted": 3,
                    "correct": 2,
                    "accuracy": 0.6666667,
                    "duration_s": 15.0,
                    "total_score": 2.0,
                    "max_score": 3.0,
                    "score_ratio": 0.6666667,
                },
            }
        )
        app.render()

        assert engine.phase is Phase.RESULTS
        assert session.stage is AntWorkoutStage.BLOCK_RESULTS
    finally:
        pygame.quit()


def test_godot_phase_advance_reaches_nested_adaptive_godot_engine() -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        app = App(surface=surface, font=font)
        fake_bridge = _FakeBridge()
        app._godot_bridge = fake_bridge
        app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))
        clock = _FakeClock()
        block = AdaptiveSessionBlock(
            block_index=0,
            primitive_id="tracking_stability_low_load",
            primitive_label="Tracking Stability",
            drill_code="rt_obscured_target_prediction",
            mode="block_component",
            duration_s=15.0,
            difficulty_level=5,
            seed=2027,
            reason_tags=("regression",),
            priority=0.8,
            drill_mode=AntDrillMode.BUILD,
            form_factor="micro",
            target_area="tracking_stability",
        )
        session = AdaptiveSession(
            clock=clock,
            seed=2027,
            plan=AdaptiveSessionPlan(
                code="adaptive_session",
                title="Adaptive Route",
                version=1,
                generated_at_utc="2026-01-01T00:00:00Z",
                description="Nested route regression.",
                notes=(),
                ranked_primitives=(),
                variant="adaptive",
                blocks=(block,),
            ),
        )
        session.activate()
        assert session.stage is AdaptiveStage.BLOCK
        engine = session.current_engine()
        assert engine is not None
        payload = engine.snapshot().payload
        assert isinstance(payload, GodotOwnedPayload)
        assert payload.spec.phase == "instructions"
        screen = AdaptiveSessionScreen(
            app,
            session=session,
            test_code="adaptive_session",
        )
        app.push(screen)

        fake_bridge.control_commands.append(
            {
                "command": "godot_phase_advance",
                "run_key": payload.spec.run_key,
                "test_code": "rt_obscured_target_prediction",
                "kind": "rapid_tracking",
            }
        )
        app.render()

        routed_payload = engine.snapshot().payload
        assert isinstance(routed_payload, GodotOwnedPayload)
        assert routed_payload.spec.phase == "practice"
    finally:
        pygame.quit()


def test_auditory_capacity_wrapper_drills_receive_screen_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        app = App(surface=surface, font=font)
        app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))
        clock = _FakeClock()

        def build_engine() -> object:
            engine = build_ac_gate_anchor_drill(
                clock=clock,
                seed=2028,
                difficulty=0.5,
                mode=AntDrillMode.BUILD,
                config=AcDrillConfig(scored_duration_s=2.0),
            )
            engine.start_practice()
            return engine

        screen = CognitiveTestScreen(
            app,
            engine_factory=build_engine,
            test_code="ac_gate_anchor",
        )
        app.push(screen)
        monkeypatch.setattr(
            screen,
            "_read_sensory_motor_control",
            lambda **_kwargs: (0.42, -0.58),
        )

        screen.render(surface)

        payload = screen._engine.snapshot().payload
        assert isinstance(payload, AuditoryCapacityPayload)
        assert payload.control_x == pytest.approx(0.42)
        assert payload.control_y == pytest.approx(-0.58)
    finally:
        pygame.quit()


def test_godot_back_to_tests_returns_to_previous_menu() -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        app = App(surface=surface, font=font)
        fake_bridge = _FakeBridge()
        app._godot_bridge = fake_bridge
        root = MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True)
        tests_menu = MenuScreen(app, "Tests", [MenuItem("Back", app.pop)])
        app.push(root)
        app.push(tests_menu)
        _kind, snap, _payload = _payload_snapshots()[1]
        screen = CognitiveTestScreen(
            app,
            engine_factory=lambda: _SnapshotEngine(snap),
            test_code="rapid_tracking",
        )
        app.push(screen)

        fake_bridge.control_commands.append({"command": "back_to_tests"})
        app.render()

        assert app._screens[-1] is tests_menu
        assert fake_bridge.close_calls == 1
    finally:
        pygame.quit()


@pytest.mark.parametrize("kind", list(InstrumentComprehensionTrialKind))
def test_instrument_comprehension_modes_idle_godot_bridge(
    monkeypatch: pytest.MonkeyPatch,
    kind: InstrumentComprehensionTrialKind,
) -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        app = App(surface=surface, font=font)
        fake_bridge = _FakeBridge()
        app._godot_bridge = fake_bridge
        app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))
        payload = InstrumentComprehensionGenerator(seed=118).next_problem_for_kind(
            kind=kind,
            difficulty=0.55,
        ).payload
        assert isinstance(payload, InstrumentComprehensionPayload)
        snap = _snapshot("Instrument Comprehension", payload)
        screen = CognitiveTestScreen(app, engine_factory=lambda: _SnapshotEngine(snap))
        app.push(screen)
        called = {"value": False}

        def fake_render(*args: object, **kwargs: object) -> None:
            _ = args, kwargs
            called["value"] = True

        monkeypatch.setattr(screen, "_render_instrument_comprehension_screen", fake_render)
        screen.render(surface)

        assert fake_bridge.sync_calls == []
        assert fake_bridge.idle_calls == 1
        assert called["value"] is True
        assert app.current_run_state().renderer_path == "PYGAME_2D"
    finally:
        pygame.quit()


def test_godot_project_import_smoke_skips_without_binary() -> None:
    godot_bin = Path(os.environ.get("CFAST_GODOT_BIN", GODOT_DEFAULT_BIN))
    if not godot_bin.is_file():
        pytest.skip("Godot binary is not installed")
    completed = subprocess.run(
        [str(godot_bin), "--headless", "--path", str(GODOT_PROJECT_PATH), "--import"],
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )
    combined_output = f"{completed.stdout}\n{completed.stderr}"
    assert "SCRIPT ERROR" not in combined_output
    assert "Failed to load script" not in combined_output
