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

from cfast_trainer.app import App, CognitiveTestScreen, MenuItem, MenuScreen
from cfast_trainer.auditory_capacity import AuditoryCapacityPayload, build_auditory_capacity_test
from cfast_trainer.cognitive_core import Phase, TestSnapshot as SnapshotModel
from cfast_trainer.godot_bridge import (
    GODOT_BACKEND_NAME,
    GODOT_DEFAULT_BIN,
    GODOT_PROJECT_PATH,
    GodotBridgeManager,
    godot_kind_for_snapshot,
    serialize_godot_state,
)
from cfast_trainer.instrument_comprehension import (
    InstrumentComprehensionGenerator,
    InstrumentComprehensionPayload,
    InstrumentComprehensionTrialKind,
)
from cfast_trainer.rapid_tracking import RapidTrackingPayload, build_rapid_tracking_test
from cfast_trainer.spatial_integration import SpatialIntegrationPayload, build_spatial_integration_test
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
    def __init__(self, sink: list[dict[str, Any]]) -> None:
        self.sink = sink
        self.closed = False

    def sendto(self, payload: bytes, addr: tuple[str, int]) -> int:
        message = json.loads(payload.decode("utf-8"))
        message["_addr"] = [addr[0], addr[1]]
        self.sink.append(message)
        return len(payload)

    def close(self) -> None:
        self.closed = True


class _FakeBridge:
    def __init__(self) -> None:
        self.sync_calls: list[tuple[SnapshotModel, object | None]] = []
        self.idle_calls = 0
        self.close_calls = 0
        self.active_kind: str | None = None

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
    window_mode: str = "windowed",
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
    assert command[:5] == ["/bin/echo", "--path", str(tmp_path / "godot_project"), "--windowed", "--resolution"]
    assert "960x540" in command
    assert "--listen-port" in command
    assert sent[-1]["kind"] == "rapid_tracking"
    assert sent[-1]["renderer_backend"] == GODOT_BACKEND_NAME
    assert sent[-1]["window_mode"] == "windowed"
    assert manager.is_active_for("rapid_tracking")


def test_bridge_launches_fullscreen_when_app_window_mode_is_fullscreen(tmp_path: Path) -> None:
    sent: list[dict[str, Any]] = []
    popen = _FakePopenFactory()
    manager = _manager(tmp_path, popen_factory=popen, sent=sent, window_mode="fullscreen")
    _kind, snap, payload = _payload_snapshots()[1]

    assert manager.sync(snap, payload) is True

    command = popen.commands[0]
    assert "--fullscreen" in command
    assert "--windowed" not in command
    assert "--resolution" not in command
    assert sent[-1]["window_mode"] == "fullscreen"


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


def test_app_window_mode_is_passed_to_godot_bridge() -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        app = App(surface=surface, font=font, window_mode="fullscreen")

        assert app.godot_bridge().window_mode == "fullscreen"

        app.set_window_mode("windowed")

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
