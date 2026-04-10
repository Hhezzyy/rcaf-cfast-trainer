from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from importlib.machinery import ModuleSpec
from types import SimpleNamespace
from types import ModuleType
from typing import cast

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

if "moderngl" not in sys.modules:
    moderngl_stub = ModuleType("moderngl")
    moderngl_stub.__spec__ = ModuleSpec("moderngl", loader=None)
    sys.modules["moderngl"] = moderngl_stub

import pygame
import pytest

from cfast_trainer.ant_drills import AntDrillMode
from cfast_trainer.ant_workouts import AntWorkoutBlockPlan, AntWorkoutPlan, AntWorkoutSession
from cfast_trainer.app import (
    AnalogBinding,
    AntWorkoutScreen,
    App,
    AxisCalibrationSettings,
    CognitiveTestScreen,
    DigitalBinding,
    InputProfilesStore,
    MenuItem,
    MenuScreen,
    RapidTrackingSettingsStore,
)
from cfast_trainer.cognitive_core import Phase
from cfast_trainer.cognitive_core import TestSnapshot as SnapshotModel
from cfast_trainer.rapid_tracking import RapidTrackingPayload, build_rapid_tracking_test
from cfast_trainer.rapid_tracking.renderer import _rapid_tracking_fallback_notice_lines
from cfast_trainer.rt_drills import (
    build_rt_ground_tempo_run_drill,
    build_rt_mixed_tempo_drill,
    build_rt_rudder_horizontal_prime_drill,
)


@dataclass
class _FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class _PressedKeys:
    def __init__(self, active: set[int]) -> None:
        self._active = set(active)

    def __getitem__(self, key: int) -> int:
        return 1 if key in self._active else 0


class _FakeRapidTrackingEngine:
    def __init__(self, payload: RapidTrackingPayload, *, title: str) -> None:
        self._payload = payload
        self._title = title
        self.submissions: list[str] = []
        self.last_control: tuple[float, float] | None = None

    def snapshot(self) -> SnapshotModel:
        return SnapshotModel(
            title=self._title,
            phase=Phase.PRACTICE,
            prompt="Track the target.",
            input_hint="",
            time_remaining_s=None,
            attempted_scored=0,
            correct_scored=0,
            payload=self._payload,
        )

    def can_exit(self) -> bool:
        return True

    def start_practice(self) -> None:
        pass

    def start_scored(self) -> None:
        pass

    def set_control(self, *, horizontal: float, vertical: float) -> None:
        self.last_control = (float(horizontal), float(vertical))

    def submit_answer(self, raw: str) -> bool:
        self.submissions.append(str(raw))
        return True

    def update(self) -> None:
        pass


class _RecordingFont:
    def __init__(self, base: pygame.font.Font, sink: list[str]) -> None:
        self._base = base
        self._sink = sink

    def render(self, text: str, antialias: bool, color: object) -> pygame.Surface:
        self._sink.append(str(text))
        return self._base.render(text, antialias, color)

    def __getattr__(self, name: str) -> object:
        return getattr(self._base, name)


def _sample_payload() -> RapidTrackingPayload:
    clock = _FakeClock()
    engine = build_rapid_tracking_test(clock=clock, seed=551, difficulty=0.5)
    engine.start_scored()
    payload = engine.snapshot().payload
    assert isinstance(payload, RapidTrackingPayload)
    return payload


def _build_screen(engine: object) -> tuple[App, CognitiveTestScreen]:
    pygame.init()
    surface = pygame.display.set_mode((960, 540))
    font = pygame.font.Font(None, 36)
    app = App(surface=surface, font=font)
    root = MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True)
    app.push(root)
    screen = CognitiveTestScreen(app, engine_factory=lambda: engine)
    app.push(screen)
    return app, screen


def _install_recording_fonts(*fonts: object) -> list[str]:
    captured: list[str] = []
    for obj in fonts:
        for attr in ("_small_font", "_tiny_font", "_mid_font", "_big_font"):
            font = getattr(obj, attr, None)
            if isinstance(font, _RecordingFont) or font is None:
                continue
            setattr(obj, attr, _RecordingFont(font, captured))
    return captured


def _build_live_rt_screen(*, clock: _FakeClock) -> CognitiveTestScreen:
    screen = _build_screen(
        build_rapid_tracking_test(
            clock=clock,
            seed=551,
            difficulty=0.5,
        )
    )[1]
    screen._engine.start_scored()
    screen._engine._target_x = 0.0
    screen._engine._target_y = 0.0
    screen._engine._target_terrain_occluded = False
    screen._engine._reset_camera_pose_to_target()
    return screen


class _AxisJoystick:
    def __init__(self, *, name: str, axes: dict[int, float], guid: str) -> None:
        self._name = name
        self._axes = dict(axes)
        self._guid = guid

    def get_name(self) -> str:
        return self._name

    def get_guid(self) -> str:
        return self._guid

    def get_numaxes(self) -> int:
        return (max(self._axes) + 1) if self._axes else 0

    def get_axis(self, idx: int) -> float:
        return float(self._axes.get(idx, 0.0))


class _ButtonJoystick:
    def __init__(self, *, name: str, guid: str, buttons: dict[int, int]) -> None:
        self._name = name
        self._guid = guid
        self._buttons = dict(buttons)

    def get_name(self) -> str:
        return self._name

    def get_guid(self) -> str:
        return self._guid

    def get_numaxes(self) -> int:
        return 0

    def get_axis(self, idx: int) -> float:
        _ = idx
        return 0.0

    def get_numbuttons(self) -> int:
        return (max(self._buttons) + 1) if self._buttons else 0

    def get_button(self, idx: int) -> int:
        return int(self._buttons.get(idx, 0))

    def get_numhats(self) -> int:
        return 0


def test_rapid_tracking_title_prefix_routes_to_real_renderer(monkeypatch) -> None:
    payload = _sample_payload()
    _app, screen = _build_screen(
        _FakeRapidTrackingEngine(payload, title="Rapid Tracking: Lock Anchor")
    )
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        called = {"value": False}
        original = screen._render_rapid_tracking_screen

        def wrapped(surface, snap, payload):
            called["value"] = True
            return original(surface, snap, payload)

        monkeypatch.setattr(screen, "_render_rapid_tracking_screen", wrapped)
        screen.render(surface)

        assert called["value"] is True
    finally:
        pygame.quit()


def test_rapid_tracking_workout_block_uses_real_runtime_screen() -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        app = App(surface=surface, font=font)
        root = MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True)
        app.push(root)

        clock = _FakeClock()
        plan = AntWorkoutPlan(
            code="rapid_tracking_workout",
            title="RT Workout UI",
            description="UI regression workout.",
            notes=("Untimed reflections.",),
            blocks=(
                AntWorkoutBlockPlan(
                    block_id="mixed",
                    label="Mixed Tempo",
                    description="Warm-up.",
                    focus_skills=("Tracking",),
                    drill_code="rt_mixed_tempo",
                    mode=AntDrillMode.BUILD,
                    duration_min=0.25,
                ),
            ),
        )
        session = AntWorkoutSession(clock=clock, seed=606, plan=plan, starting_level=5)
        session.activate()
        session.append_text("steady scan")
        session.activate()
        session.append_text("capture late")
        session.activate()
        session.activate()

        screen = AntWorkoutScreen(app, session=session, test_code="rapid_tracking_workout")
        app.push(screen)
        screen.render(surface)

        runtime = screen._runtime_screen
        assert runtime is not None
        assert runtime._engine is not None
        snap = runtime._engine.snapshot()
        assert str(snap.title).startswith("Rapid Tracking")
        assert isinstance(snap.payload, RapidTrackingPayload)
    finally:
        pygame.quit()


def test_rapid_tracking_real_drill_engine_exposes_focus_metadata_on_live_screen() -> None:
    clock = _FakeClock()
    engine = build_rt_mixed_tempo_drill(
        clock=clock,
        seed=505,
        difficulty=0.5,
        mode=AntDrillMode.BUILD,
    )
    engine.start_scored()
    _app, screen = _build_screen(engine)
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        screen.render(surface)

        payload = engine.snapshot().payload
        assert isinstance(payload, RapidTrackingPayload)
        assert payload.focus_label == "Stable lock quality"
        assert payload.active_target_kinds == ("soldier", "truck")
        assert payload.control_scheme == "joystick_only"
    finally:
        pygame.quit()


def test_rapid_tracking_live_screen_hides_window_and_segment_progress_counters() -> None:
    clock = _FakeClock()
    screen = _build_live_rt_screen(clock=clock)
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        payload = screen._engine.snapshot().payload
        assert isinstance(payload, RapidTrackingPayload)
        captured = _install_recording_fonts(screen)

        screen.render(surface)

        assert payload.segment_label in captured
        assert not any(text.startswith("Windows ") for text in captured)
        assert f"{payload.segment_label} {payload.segment_index}/{payload.segment_total}" not in captured
    finally:
        pygame.quit()


def test_rapid_tracking_main_runtime_uses_joystick_x_and_y(monkeypatch) -> None:
    clock = _FakeClock()
    screen = _build_live_rt_screen(clock=clock)
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        monkeypatch.setattr(
            "cfast_trainer.app._iter_connected_joysticks",
            lambda: [
                _AxisJoystick(
                    name="primary stick",
                    guid="primary-guid",
                    axes={0: 0.46, 1: -0.28, 3: 0.91},
                ),
                _AxisJoystick(
                    name="rudder pedals",
                    guid="rudder-guid",
                    axes={3: -0.83},
                ),
            ],
        )
        monkeypatch.setattr(pygame.key, "get_pressed", lambda: _PressedKeys(set()))

        screen.render(surface)

        assert screen._engine.control_scheme == "joystick_only"
        assert screen._engine._control_x == pytest.approx(0.46)
        assert screen._engine._control_y == pytest.approx(-0.28)
    finally:
        pygame.quit()


def test_rapid_tracking_prefers_primary_stick_when_pedals_are_listed_first(monkeypatch) -> None:
    clock = _FakeClock()
    screen = _build_live_rt_screen(clock=clock)
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        monkeypatch.setattr(
            "cfast_trainer.app._iter_connected_joysticks",
            lambda: [
                _AxisJoystick(
                    name="rudder pedals",
                    guid="rudder-guid",
                    axes={0: -0.91, 1: 0.84, 3: -0.77},
                ),
                _AxisJoystick(
                    name="VKB Gladiator NXT EVO joystick",
                    guid="vkb-guid",
                    axes={0: 0.33, 1: -0.52, 3: 0.48},
                ),
            ],
        )
        monkeypatch.setattr(pygame.key, "get_pressed", lambda: _PressedKeys(set()))

        screen.render(surface)

        assert screen._engine.control_scheme == "joystick_only"
        assert screen._engine._control_x == pytest.approx(0.33)
        assert screen._engine._control_y == pytest.approx(-0.52)
    finally:
        pygame.quit()


def test_existing_rt_drill_runtime_uses_joystick_horizontal_control(monkeypatch) -> None:
    clock = _FakeClock()
    drill = build_rt_ground_tempo_run_drill(
        clock=clock,
        seed=771,
        difficulty=0.5,
        mode=AntDrillMode.BUILD,
    )
    drill.start_scored()
    _app, screen = _build_screen(drill)
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        monkeypatch.setattr(
            "cfast_trainer.app._iter_connected_joysticks",
            lambda: [
                _AxisJoystick(
                    name="primary stick",
                    guid="primary-guid",
                    axes={0: -0.35, 1: 0.22, 3: 0.77},
                ),
                _AxisJoystick(
                    name="rudder pedals",
                    guid="rudder-guid",
                    axes={3: -0.66},
                ),
            ],
        )
        monkeypatch.setattr(pygame.key, "get_pressed", lambda: _PressedKeys(set()))

        screen.render(surface)
        payload = drill.snapshot().payload

        assert isinstance(payload, RapidTrackingPayload)
        assert payload.control_scheme == "joystick_only"
        assert drill._control_x == pytest.approx(-0.35)
        assert drill._control_y == pytest.approx(0.22)
    finally:
        pygame.quit()


def test_rt_rudder_horizontal_prefers_pedals_for_horizontal_and_stick_for_vertical_when_pedals_listed_first(
    monkeypatch,
) -> None:
    clock = _FakeClock()
    drill = build_rt_rudder_horizontal_prime_drill(
        clock=clock,
        seed=912,
        difficulty=0.5,
        mode=AntDrillMode.BUILD,
    )
    drill.start_scored()
    _app, screen = _build_screen(drill)
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        monkeypatch.setattr(
            "cfast_trainer.app._iter_connected_joysticks",
            lambda: [
                _AxisJoystick(
                    name="Logitech Saitek Pro Rudder Pedals",
                    guid="rudder-guid",
                    axes={0: 0.41, 1: 0.92, 3: -0.72},
                ),
                _AxisJoystick(
                    name="VKB Gladiator NXT EVO joystick",
                    guid="vkb-guid",
                    axes={0: 0.55, 1: 0.24, 3: 0.81},
                ),
            ],
        )
        monkeypatch.setattr(pygame.key, "get_pressed", lambda: _PressedKeys(set()))

        screen.render(surface)
        payload = drill.snapshot().payload

        assert isinstance(payload, RapidTrackingPayload)
        assert payload.control_scheme == "rudder_horizontal"
        assert drill._control_x == pytest.approx(-0.72)
        assert drill._control_y == pytest.approx(0.24)
    finally:
        pygame.quit()


def test_rapid_tracking_wrapper_payload_routes_controls_without_direct_rt_engine(monkeypatch) -> None:
    payload = _sample_payload()
    engine = _FakeRapidTrackingEngine(payload, title="Wrapper Runtime")
    _app, screen = _build_screen(engine)
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        monkeypatch.setattr(
            "cfast_trainer.app._iter_connected_joysticks",
            lambda: [_AxisJoystick(name="primary stick", guid="primary-guid", axes={0: 0.31, 1: -0.44})],
        )
        monkeypatch.setattr(pygame.key, "get_pressed", lambda: _PressedKeys(set()))

        screen.render(surface)

        assert engine.last_control is not None
        assert engine.last_control[0] == pytest.approx(0.31)
        assert engine.last_control[1] == pytest.approx(-0.44)
    finally:
        pygame.quit()


def test_rapid_tracking_missing_explicit_primary_binding_falls_back_to_joystick(monkeypatch, tmp_path) -> None:
    profiles = InputProfilesStore(tmp_path / "input-profiles.json")
    profile_id = profiles.active_profile().profile_id
    profiles.set_axis_role_binding(
        profile_id=profile_id,
        role="primary_horizontal",
        binding=AnalogBinding(device_key="missing stick|guid-missing", axis_index=2),
    )
    neutral = AxisCalibrationSettings(deadzone=0.0, curve=1.0)
    profiles.set_axis_calibration(
        profile_id=profile_id,
        axis_key="primary stick|primary-guid::axis0",
        settings=neutral,
    )
    profiles.set_axis_calibration(
        profile_id=profile_id,
        axis_key="primary stick|primary-guid::axis1",
        settings=neutral,
    )

    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        app = App(surface=surface, font=font, input_profiles_store=profiles)
        app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))
        clock = _FakeClock()
        screen = CognitiveTestScreen(
            app,
            engine_factory=lambda: build_rapid_tracking_test(clock=clock, seed=654, difficulty=0.5),
            test_code="rapid_tracking",
        )
        app.push(screen)
        screen._engine.start_scored()
        monkeypatch.setattr(
            "cfast_trainer.app._iter_connected_joysticks",
            lambda: [_AxisJoystick(name="primary stick", guid="primary-guid", axes={0: 0.42, 1: -0.18})],
        )
        monkeypatch.setattr(pygame.key, "get_pressed", lambda: _PressedKeys(set()))

        app._joystick_binding_router.poll()
        screen.render(surface)

        assert screen._engine._control_x == pytest.approx(0.42)
        assert screen._engine._control_y == pytest.approx(-0.18)
    finally:
        pygame.quit()


def test_rapid_tracking_fallback_vertical_axis_respects_profile_inversion(monkeypatch, tmp_path) -> None:
    profiles = InputProfilesStore(tmp_path / "input-profiles.json")
    profile_id = profiles.active_profile().profile_id
    neutral = AxisCalibrationSettings(deadzone=0.0, curve=1.0)
    profiles.set_axis_calibration(
        profile_id=profile_id,
        axis_key="primary stick|primary-guid::axis0",
        settings=neutral,
    )
    profiles.set_axis_calibration(
        profile_id=profile_id,
        axis_key="primary stick|primary-guid::axis1",
        settings=AxisCalibrationSettings(deadzone=0.0, invert=True, curve=1.0),
    )

    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        app = App(surface=surface, font=font, input_profiles_store=profiles)
        app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))
        clock = _FakeClock()
        screen = CognitiveTestScreen(
            app,
            engine_factory=lambda: build_rapid_tracking_test(clock=clock, seed=655, difficulty=0.5),
            test_code="rapid_tracking",
        )
        app.push(screen)
        screen._engine.start_scored()
        monkeypatch.setattr(
            "cfast_trainer.app._iter_connected_joysticks",
            lambda: [_AxisJoystick(name="primary stick", guid="primary-guid", axes={0: 0.22, 1: 0.41})],
        )
        monkeypatch.setattr(pygame.key, "get_pressed", lambda: _PressedKeys(set()))

        app._joystick_binding_router.poll()
        screen.render(surface)

        assert screen._engine._control_x == pytest.approx(0.22)
        assert screen._engine._control_y == pytest.approx(-0.41)
    finally:
        pygame.quit()


def test_rapid_tracking_explicit_primary_vertical_binding_respects_profile_inversion(
    monkeypatch, tmp_path
) -> None:
    profiles = InputProfilesStore(tmp_path / "input-profiles.json")
    profile_id = profiles.active_profile().profile_id
    profiles.set_axis_role_binding(
        profile_id=profile_id,
        role="primary_vertical",
        binding=AnalogBinding(device_key="primary stick|primary-guid", axis_index=3),
    )
    neutral = AxisCalibrationSettings(deadzone=0.0, curve=1.0)
    profiles.set_axis_calibration(
        profile_id=profile_id,
        axis_key="primary stick|primary-guid::axis0",
        settings=neutral,
    )
    profiles.set_axis_calibration(
        profile_id=profile_id,
        axis_key="primary stick|primary-guid::axis3",
        settings=AxisCalibrationSettings(deadzone=0.0, invert=True, curve=1.0),
    )

    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        app = App(surface=surface, font=font, input_profiles_store=profiles)
        app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))
        clock = _FakeClock()
        screen = CognitiveTestScreen(
            app,
            engine_factory=lambda: build_rapid_tracking_test(clock=clock, seed=656, difficulty=0.5),
            test_code="rapid_tracking",
        )
        app.push(screen)
        screen._engine.start_scored()
        monkeypatch.setattr(
            "cfast_trainer.app._iter_connected_joysticks",
            lambda: [
                _AxisJoystick(
                    name="primary stick",
                    guid="primary-guid",
                    axes={0: 0.18, 1: 0.35, 3: 0.56},
                )
            ],
        )
        monkeypatch.setattr(pygame.key, "get_pressed", lambda: _PressedKeys(set()))

        app._joystick_binding_router.poll()
        screen.render(surface)

        assert screen._engine._control_x == pytest.approx(0.18)
        assert screen._engine._control_y == pytest.approx(-0.56)
    finally:
        pygame.quit()


def test_rapid_tracking_capture_hold_tracks_all_explicit_bindings_across_devices(
    monkeypatch, tmp_path
) -> None:
    profile_store = InputProfilesStore(tmp_path / "input-profiles.json")
    profile_id = profile_store.active_profile().profile_id
    profile_store.set_action_binding(
        profile_id=profile_id,
        action="rapid_tracking_capture",
        slot_index=0,
        binding=DigitalBinding(kind="button", device_key="capture stick a|guid-a", control_index=4),
    )
    profile_store.set_action_binding(
        profile_id=profile_id,
        action="rapid_tracking_capture",
        slot_index=1,
        binding=DigitalBinding(kind="button", device_key="capture stick b|guid-b", control_index=1),
    )

    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        app = App(surface=surface, font=font, input_profiles_store=profile_store)
        app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))
        clock = _FakeClock()
        screen = CognitiveTestScreen(
            app,
            engine_factory=lambda: build_rapid_tracking_test(clock=clock, seed=8123, difficulty=0.5),
            test_code="rapid_tracking",
        )
        app.push(screen)
        screen._engine.start_practice()
        screen._engine._target_x = 0.0
        screen._engine._target_y = 0.0
        screen._engine._target_terrain_occluded = False
        screen._engine._reset_camera_pose_to_target()

        stick_a = _ButtonJoystick(name="capture stick a", guid="guid-a", buttons={})
        stick_b = _ButtonJoystick(name="capture stick b", guid="guid-b", buttons={})
        monkeypatch.setattr("cfast_trainer.app._iter_connected_joysticks", lambda: [stick_a, stick_b])
        monkeypatch.setattr(pygame.key, "get_pressed", lambda: _PressedKeys(set()))

        stick_a._buttons[4] = 1
        app.render()
        first = screen._engine.snapshot().payload
        assert first is not None
        assert first.capture_attempts == 1
        assert first.capture_zoom > 0.0

        stick_b._buttons[1] = 1
        clock.advance(0.10)
        app.render()
        both_held = screen._engine.snapshot().payload
        assert both_held is not None
        assert both_held.capture_attempts == 1
        assert both_held.capture_zoom > 0.5

        stick_a._buttons[4] = 0
        clock.advance(0.30)
        app.render()
        still_held = screen._engine.snapshot().payload
        assert still_held is not None
        assert still_held.capture_attempts == 1
        assert still_held.capture_zoom > 0.8

        stick_b._buttons[1] = 0
        clock.advance(0.30)
        app.render()
        released = screen._engine.snapshot().payload
        assert released is not None
        assert released.capture_zoom < 0.2
    finally:
        pygame.quit()


def test_rapid_tracking_global_invert_pitch_applies_after_axis_calibration(
    monkeypatch, tmp_path
) -> None:
    profiles = InputProfilesStore(tmp_path / "input-profiles.json")
    profile_id = profiles.active_profile().profile_id
    neutral = AxisCalibrationSettings(deadzone=0.0, curve=1.0)
    profiles.set_axis_calibration(
        profile_id=profile_id,
        axis_key="primary stick|primary-guid::axis0",
        settings=neutral,
    )
    profiles.set_axis_calibration(
        profile_id=profile_id,
        axis_key="primary stick|primary-guid::axis1",
        settings=AxisCalibrationSettings(deadzone=0.0, invert=True, curve=1.0),
    )
    rapid_tracking_settings = RapidTrackingSettingsStore(tmp_path / "rapid-tracking-settings.json")
    rapid_tracking_settings.set_invert_pitch(True)

    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        app = App(
            surface=surface,
            font=font,
            input_profiles_store=profiles,
            rapid_tracking_settings_store=rapid_tracking_settings,
        )
        app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))
        clock = _FakeClock()
        screen = CognitiveTestScreen(
            app,
            engine_factory=lambda: build_rapid_tracking_test(clock=clock, seed=657, difficulty=0.5),
            test_code="rapid_tracking",
        )
        app.push(screen)
        screen._engine.start_scored()
        monkeypatch.setattr(
            "cfast_trainer.app._iter_connected_joysticks",
            lambda: [_AxisJoystick(name="primary stick", guid="primary-guid", axes={0: 0.22, 1: 0.41})],
        )
        monkeypatch.setattr(pygame.key, "get_pressed", lambda: _PressedKeys(set()))

        app._joystick_binding_router.poll()
        screen.render(surface)

        assert screen._engine._control_x == pytest.approx(0.22)
        assert screen._engine._control_y == pytest.approx(0.41)
    finally:
        pygame.quit()


def test_rt_rudder_horizontal_prime_uses_rudder_left_right_and_joystick_y(monkeypatch) -> None:
    clock = _FakeClock()
    drill = build_rt_rudder_horizontal_prime_drill(
        clock=clock,
        seed=909,
        difficulty=0.5,
        mode=AntDrillMode.BUILD,
    )
    drill.start_scored()
    _app, screen = _build_screen(drill)
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        monkeypatch.setattr(
            "cfast_trainer.app._iter_connected_joysticks",
            lambda: [
                _AxisJoystick(
                    name="primary stick",
                    guid="primary-guid",
                    axes={0: 0.55, 1: 0.24, 3: 0.81},
                ),
                _AxisJoystick(
                    name="rudder pedals",
                    guid="rudder-guid",
                    axes={3: -0.72},
                ),
            ],
        )
        monkeypatch.setattr(pygame.key, "get_pressed", lambda: _PressedKeys(set()))

        screen.render(surface)
        payload = drill.snapshot().payload

        assert isinstance(payload, RapidTrackingPayload)
        assert payload.control_scheme == "rudder_horizontal"
        assert drill._control_x == pytest.approx(-0.72)
        assert drill._control_y == pytest.approx(0.24)
    finally:
        pygame.quit()


def test_rapid_tracking_joybutton_hold_starts_zoom_and_release_restores_view() -> None:
    clock = _FakeClock()
    screen = _build_live_rt_screen(clock=clock)
    try:
        screen.handle_event(
            pygame.event.Event(pygame.JOYBUTTONDOWN, {"button": 0})
        )
        clock.advance(0.30)
        screen._engine.update()
        zoomed = screen._engine.snapshot().payload
        assert zoomed is not None
        assert zoomed.capture_zoom > 0.8

        screen.handle_event(
            pygame.event.Event(pygame.JOYBUTTONUP, {"button": 0})
        )
        clock.advance(0.30)
        screen._engine.update()
        released = screen._engine.snapshot().payload
        assert released is not None
        assert released.capture_zoom < 0.2
    finally:
        pygame.quit()


def test_rapid_tracking_legacy_capture_hold_waits_for_last_button_release() -> None:
    clock = _FakeClock()
    screen = _build_live_rt_screen(clock=clock)
    try:
        screen.handle_event(
            pygame.event.Event(pygame.JOYBUTTONDOWN, {"button": 0})
        )
        screen.handle_event(
            pygame.event.Event(pygame.JOYBUTTONDOWN, {"button": 1})
        )
        clock.advance(0.30)
        screen._engine.update()
        held = screen._engine.snapshot().payload
        assert held is not None
        assert held.capture_zoom > 0.8
        assert held.capture_attempts == 1

        screen.handle_event(
            pygame.event.Event(pygame.JOYBUTTONUP, {"button": 0})
        )
        clock.advance(0.30)
        screen._engine.update()
        still_held = screen._engine.snapshot().payload
        assert still_held is not None
        assert still_held.capture_zoom > 0.8
        assert still_held.capture_attempts == 1

        screen.handle_event(
            pygame.event.Event(pygame.JOYBUTTONUP, {"button": 1})
        )
        clock.advance(0.30)
        screen._engine.update()
        released = screen._engine.snapshot().payload
        assert released is not None
        assert released.capture_zoom < 0.2
    finally:
        pygame.quit()


def test_rapid_tracking_ignores_mouse_and_keyboard_capture_inputs() -> None:
    payload = _sample_payload()
    _app, screen = _build_screen(
        _FakeRapidTrackingEngine(payload, title="Rapid Tracking: Lock Anchor")
    )
    try:
        screen.handle_event(
            pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"button": 1, "pos": (20, 20)})
        )
        screen.handle_event(
            pygame.event.Event(pygame.MOUSEBUTTONUP, {"button": 1, "pos": (20, 20)})
        )
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_SPACE, "unicode": " "})
        )
        screen.handle_event(
            pygame.event.Event(pygame.KEYUP, {"key": pygame.K_SPACE, "unicode": ""})
        )
        screen.handle_event(
            pygame.event.Event(pygame.JOYBUTTONDOWN, {"button": 0})
        )
        screen.handle_event(
            pygame.event.Event(pygame.JOYBUTTONUP, {"button": 0})
        )

        assert cast(_FakeRapidTrackingEngine, screen._engine).submissions == [
            "CAPTURE_HOLD_START",
            "CAPTURE_HOLD_END",
        ]
    finally:
        pygame.quit()


def test_rapid_tracking_render_ignores_keyboard_camera_fallback(monkeypatch) -> None:
    clock = _FakeClock()
    screen = _build_live_rt_screen(clock=clock)
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        monkeypatch.setattr(pygame.key, "get_pressed", lambda: _PressedKeys({pygame.K_LEFT, pygame.K_UP}))
        monkeypatch.setattr("cfast_trainer.app._iter_connected_joysticks", lambda: [])

        before = screen._engine.snapshot().payload
        assert before is not None

        clock.advance(0.30)
        screen.render(surface)

        after = screen._engine.snapshot().payload
        assert after is not None
        assert after.camera_yaw_deg == pytest.approx(before.camera_yaw_deg)
        assert after.camera_pitch_deg == pytest.approx(before.camera_pitch_deg)
    finally:
        pygame.quit()


def test_rapid_tracking_fallback_notice_uses_opengl_wording_not_panda() -> None:
    app = SimpleNamespace(
        opengl_enabled=False,
        renderer_gl_requested=lambda: True,
        renderer_gl_attempted=lambda: True,
        renderer_bootstrap_failure=lambda: None,
    )

    lines = _rapid_tracking_fallback_notice_lines(
        app=app,
        diagnostic_code="RT-RUNT-FALL",
    )
    text = " ".join(lines)

    assert "OpenGL" in text
    assert "2D fallback" in text
    assert "RT-RUNT-FALL" in text
    assert "Panda" not in text
