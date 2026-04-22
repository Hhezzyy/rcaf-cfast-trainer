from __future__ import annotations

import os
import sys
from importlib.machinery import ModuleSpec
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

from cfast_trainer.ac_drills import AcDrillConfig, build_ac_gate_anchor_drill
from cfast_trainer.airborne_numerical import build_airborne_numerical_test
from cfast_trainer.ant_drills import AntDrillMode
from cfast_trainer.ant_workouts import (
    AntWorkoutBlockPlan,
    AntWorkoutPlan,
    AntWorkoutSession,
    AntWorkoutStage,
)
from cfast_trainer.app import (
    INTRO_LOADING_MIN_FRAMES,
    AnalogBinding,
    App,
    AntWorkoutScreen,
    AxisCalibrationSettings,
    CognitiveTestScreen,
    DifficultySettingsStore,
    DigitalBinding,
    InputProfilesStore,
    JoystickBindingsScreen,
    MenuItem,
    MenuScreen,
    RapidTrackingSettingsScreen,
    RapidTrackingSettingsStore,
)
from cfast_trainer.clock import Clock
from cfast_trainer.cognitive_core import Phase
from cfast_trainer.cognitive_core import TestSnapshot as SnapshotModel
from cfast_trainer.gl_scenes import RapidTrackingGlScene
from cfast_trainer.numerical_operations import build_numerical_operations_test
from cfast_trainer.persistence import ResultsStore
from cfast_trainer.rapid_tracking import build_rapid_tracking_test
from cfast_trainer.runtime_defaults import RuntimeDefaultsStore
from cfast_trainer.sensory_motor_apparatus import (
    SensoryMotorApparatusConfig,
    build_sensory_motor_apparatus_test,
)


class _PressedKeys:
    def __init__(self, active: set[int]) -> None:
        self._active = set(active)

    def __getitem__(self, key: int) -> int:
        return 1 if key in self._active else 0


class _FakeEngine:
    def __init__(self, *, phase: Phase = Phase.PRACTICE, title: str = "Fake Test") -> None:
        self._phase = phase
        self._title = title
        self.update_count = 0
        self._difficulty = 0.5
        self._noise_level_override: float | None = None
        self._distortion_level_override: float | None = None
        self._noise_source_override: str | None = None

    def snapshot(self) -> SnapshotModel:
        return SnapshotModel(
            title=self._title,
            phase=self._phase,
            prompt="",
            input_hint="",
            time_remaining_s=60.0 if self._phase is Phase.SCORED else None,
            attempted_scored=0,
            correct_scored=0,
            payload=None,
        )

    def can_exit(self) -> bool:
        return self._phase in (
            Phase.INSTRUCTIONS,
            Phase.PRACTICE,
            Phase.PRACTICE_DONE,
            Phase.RESULTS,
        )

    def start_practice(self) -> None:
        if self._phase is Phase.INSTRUCTIONS:
            self._phase = Phase.PRACTICE

    def start_scored(self) -> None:
        if self._phase is Phase.PRACTICE_DONE:
            self._phase = Phase.SCORED

    def submit_answer(self, raw: str) -> bool:
        token = str(raw).strip().lower()
        if token in {"__skip_practice__", "skip_practice"} and self._phase is Phase.PRACTICE:
            self._phase = Phase.PRACTICE_DONE
            return True
        if token in {"__skip_section__", "skip_section", "__skip_all__", "skip_all"} and (
            self._phase is Phase.SCORED
        ):
            self._phase = Phase.RESULTS
            return True
        return False

    def update(self) -> None:
        self.update_count += 1

    def set_audio_overrides(
        self,
        *,
        noise_level: float | None = None,
        distortion_level: float | None = None,
        noise_source: str | None = None,
    ) -> None:
        self._noise_level_override = noise_level
        self._distortion_level_override = distortion_level
        self._noise_source_override = noise_source


class _SpyFont:
    def __init__(self) -> None:
        self.rendered: list[str] = []

    def render(self, text: str, _antialias: bool, _color) -> pygame.Surface:
        value = str(text)
        self.rendered.append(value)
        width = max(4, len(value) * 7)
        surface = pygame.Surface((width, 16))
        surface.fill((255, 255, 255))
        return surface

    def size(self, text: str) -> tuple[int, int]:
        return (max(4, len(str(text)) * 7), 16)

    def get_linesize(self) -> int:
        return 16


class _FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class _PressedKeys:
    def __init__(self, pressed: set[int]) -> None:
        self._pressed = set(pressed)

    def __getitem__(self, key: int) -> int:
        return 1 if key in self._pressed else 0


def _build_app_and_screen(
    *,
    phase: Phase = Phase.PRACTICE,
    title: str = "Fake Test",
    test_code: str | None = None,
    difficulty_settings_store: DifficultySettingsStore | None = None,
    input_profiles_store: InputProfilesStore | None = None,
    rapid_tracking_settings_store: RapidTrackingSettingsStore | None = None,
    runtime_defaults_store: RuntimeDefaultsStore | None = None,
) -> tuple[App, CognitiveTestScreen, list[_FakeEngine]]:
    pygame.init()
    surface = pygame.display.set_mode((960, 540))
    font = pygame.font.Font(None, 36)
    app = App(
        surface=surface,
        font=font,
        difficulty_settings_store=difficulty_settings_store,
        input_profiles_store=input_profiles_store,
        rapid_tracking_settings_store=rapid_tracking_settings_store,
        runtime_defaults_store=runtime_defaults_store,
    )
    root = MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True)
    app.push(root)
    created: list[_FakeEngine] = []

    def factory() -> _FakeEngine:
        engine_phase = phase if not created else Phase.INSTRUCTIONS
        engine = _FakeEngine(phase=engine_phase, title=title)
        if test_code is not None:
            engine._difficulty = (app.effective_difficulty_level(test_code) - 1) / 9.0
        created.append(engine)
        return engine

    screen = CognitiveTestScreen(app, engine_factory=factory, test_code=test_code)
    app.push(screen)
    return app, screen, created


def _build_single_block_no_workout_plan() -> AntWorkoutPlan:
    return AntWorkoutPlan(
        code="numerical_operations_workout",
        title="Pause Workout",
        description="Short deterministic workout for pause tests.",
        notes=("Block setup is untimed.",),
        blocks=(
            AntWorkoutBlockPlan(
                block_id="prime",
                label="Fact Prime",
                description="Warm-up.",
                focus_skills=("Arithmetic fact retrieval",),
                drill_code="no_fact_prime",
                mode=AntDrillMode.BUILD,
                duration_min=0.05,
            ),
        ),
    )


def _build_airborne_screen() -> tuple[App, CognitiveTestScreen, Clock]:
    pygame.init()
    surface = pygame.display.set_mode((960, 540))
    font = pygame.font.Font(None, 36)
    app = App(surface=surface, font=font)
    root = MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True)
    app.push(root)
    clock = _FakeClock()

    def factory():
        engine = build_airborne_numerical_test(clock, seed=12345, practice=True)
        engine.start()
        return engine

    screen = CognitiveTestScreen(
        app,
        engine_factory=factory,
        test_code="airborne_numerical",
    )
    app.push(screen)
    return app, screen, clock


def test_airborne_distance_reveal_tracks_current_held_a_key(monkeypatch) -> None:
    app, screen, _clock = _build_airborne_screen()
    try:
        monkeypatch.setattr(pygame.key, "get_pressed", lambda: _PressedKeys({pygame.K_a}))

        screen.render(app.surface)
        screen.render(app.surface)

        assert screen._air_show_distances is True

        monkeypatch.setattr(pygame.key, "get_pressed", lambda: _PressedKeys(set()))
        screen.render(app.surface)

        assert screen._air_show_distances is False
    finally:
        pygame.quit()


def test_airborne_reference_overlays_follow_held_reference_keys(monkeypatch) -> None:
    app, screen, _clock = _build_airborne_screen()
    try:
        monkeypatch.setattr(pygame.key, "get_pressed", lambda: _PressedKeys({pygame.K_d}))
        screen.render(app.surface)
        assert screen._air_overlay == "fuel"

        monkeypatch.setattr(pygame.key, "get_pressed", lambda: _PressedKeys({pygame.K_f}))
        screen.render(app.surface)
        assert screen._air_overlay == "parcel"

        monkeypatch.setattr(pygame.key, "get_pressed", lambda: _PressedKeys({pygame.K_s}))
        screen.render(app.surface)
        assert screen._air_overlay == "intro"

        monkeypatch.setattr(pygame.key, "get_pressed", lambda: _PressedKeys(set()))
        screen.render(app.surface)
        assert screen._air_overlay is None
    finally:
        pygame.quit()


def test_airborne_live_screen_hides_practice_progress_and_scored_counter() -> None:
    app, screen, _clock = _build_airborne_screen()
    try:
        tiny_font = _SpyFont()
        small_font = _SpyFont()
        app_font = _SpyFont()
        screen._tiny_font = tiny_font
        screen._small_font = small_font
        screen._app._font = app_font

        screen.render(app.surface)
        rendered = app_font.rendered + small_font.rendered + tiny_font.rendered

        assert "Practice" in rendered
        assert not any(text.startswith("Practice:") for text in rendered)
        assert not any(text.startswith("Scored:") for text in rendered)
    finally:
        pygame.quit()


def test_pause_menu_escape_then_resume_resumes_test() -> None:
    app, screen, _engines = _build_app_and_screen()
    try:
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "mod": 0, "unicode": ""})
        )
        assert screen._pause_menu_active is True

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": 0, "unicode": ""})
        )
        assert screen._pause_menu_active is False
        assert len(app._screens) == 2
    finally:
        pygame.quit()


def test_pause_menu_escape_opens_from_instructions() -> None:
    app, screen, _engines = _build_app_and_screen(phase=Phase.INSTRUCTIONS)
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        for _ in range(8):
            screen.render(surface)
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "mod": 0, "unicode": ""})
        )
        assert screen._pause_menu_active is True
        assert len(app._screens) == 2
    finally:
        pygame.quit()


def test_pause_menu_escape_opens_from_results() -> None:
    app, screen, _engines = _build_app_and_screen(phase=Phase.RESULTS)
    try:
        assert screen.shell_pause_available() is True

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "mod": 0, "unicode": ""})
        )

        assert screen._pause_menu_active is True
    finally:
        pygame.quit()


def test_pause_menu_main_menu_returns_to_root() -> None:
    app, screen, _engines = _build_app_and_screen()
    try:
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "mod": 0, "unicode": ""})
        )
        main_menu_index = screen._pause_menu_options().index("Main Menu")
        for _ in range(main_menu_index):
            screen.handle_event(
                pygame.event.Event(
                    pygame.KEYDOWN,
                    {"key": pygame.K_DOWN, "mod": 0, "unicode": ""},
                )
            )
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": 0, "unicode": ""})
        )

        assert len(app._screens) == 1
    finally:
        pygame.quit()


def test_pause_menu_freezes_engine_updates_while_open() -> None:
    _app, screen, engines = _build_app_and_screen()
    try:
        surface = pygame.display.get_surface()
        assert surface is not None

        engine = engines[-1]
        screen.render(surface)
        before_pause = engine.update_count
        assert before_pause > 0

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "mod": 0, "unicode": ""})
        )
        screen.render(surface)
        assert engine.update_count == before_pause
    finally:
        pygame.quit()


def test_pause_menu_mouse_click_activates_main_menu_row() -> None:
    app, screen, _engines = _build_app_and_screen(phase=Phase.INSTRUCTIONS)
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        for _ in range(8):
            screen.render(surface)

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "mod": 0, "unicode": ""})
        )
        screen.render(surface)
        main_menu_index = screen._pause_menu_options().index("Main Menu")
        hitbox = screen._pause_menu_hitboxes[main_menu_index]
        screen.handle_event(
            pygame.event.Event(
                pygame.MOUSEBUTTONDOWN,
                {"button": 1, "pos": hitbox.center},
            )
        )

        assert len(app._screens) == 1
    finally:
        pygame.quit()


def test_pause_menu_mouse_click_activates_without_prior_pause_render() -> None:
    app, screen, _engines = _build_app_and_screen(phase=Phase.INSTRUCTIONS)
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        for _ in range(4):
            screen.render(surface)

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "mod": 0, "unicode": ""})
        )
        screen.render(surface)
        main_menu_index = screen._pause_menu_options().index("Main Menu")
        click_pos = screen._pause_menu_hitboxes[main_menu_index].center

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "mod": 0, "unicode": ""})
        )
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "mod": 0, "unicode": ""})
        )
        screen.handle_event(
            pygame.event.Event(
                pygame.MOUSEBUTTONDOWN,
                {"button": 1, "pos": click_pos},
            )
        )

        assert len(app._screens) == 1
    finally:
        pygame.quit()


def test_pause_menu_keyboard_hold_repeats_after_short_delay(monkeypatch) -> None:
    _app, screen, _engines = _build_app_and_screen(phase=Phase.PRACTICE)
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        screen.render(surface)
        screen._set_pause_menu_state(True)

        held_keys = {pygame.K_DOWN}
        now_ms = {"value": 0}
        monkeypatch.setattr(pygame.key, "get_pressed", lambda: _PressedKeys(held_keys))
        monkeypatch.setattr(pygame.time, "get_ticks", lambda: now_ms["value"])

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "mod": 0, "unicode": ""})
        )
        assert screen._pause_menu_selected == 1

        _app.render()
        assert screen._pause_menu_selected == 1

        now_ms["value"] = 220
        _app.render()
        assert screen._pause_menu_selected == 1

        now_ms["value"] = 270
        _app.render()
        assert screen._pause_menu_selected == 2

        now_ms["value"] = 390
        _app.render()
        assert screen._pause_menu_selected == 3

        held_keys.clear()
        now_ms["value"] = 520
        _app.render()
        assert screen._pause_menu_selected == 3
    finally:
        pygame.quit()


def test_pause_menu_shows_unified_actions_without_dev_tools() -> None:
    _app, screen, _engines = _build_app_and_screen(phase=Phase.PRACTICE)
    try:
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "mod": 0, "unicode": ""})
        )

        assert screen._pause_menu_options() == (
            "Resume",
            "Skip Current Segment",
            "Restart Current",
            "Settings",
            "Main Menu",
        )
    finally:
        pygame.quit()


def test_pause_menu_skip_current_segment_advances_practice_to_practice_done() -> None:
    _app, screen, engines = _build_app_and_screen(phase=Phase.PRACTICE)
    try:
        engine = engines[-1]
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "mod": 0, "unicode": ""})
        )
        skip_index = screen._pause_menu_options().index("Skip Current Segment")
        for _ in range(skip_index):
            screen.handle_event(
                pygame.event.Event(
                    pygame.KEYDOWN,
                    {"key": pygame.K_DOWN, "mod": 0, "unicode": ""},
                )
            )
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": 0, "unicode": ""})
        )

        assert engine.snapshot().phase is Phase.PRACTICE_DONE
        assert screen._pause_menu_active is False
    finally:
        pygame.quit()


def test_pause_menu_skip_current_segment_advances_scored_to_results() -> None:
    _app, screen, engines = _build_app_and_screen(phase=Phase.SCORED)
    try:
        engine = engines[-1]
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "mod": 0, "unicode": ""})
        )
        skip_index = screen._pause_menu_options().index("Skip Current Segment")
        for _ in range(skip_index):
            screen.handle_event(
                pygame.event.Event(
                    pygame.KEYDOWN,
                    {"key": pygame.K_DOWN, "mod": 0, "unicode": ""},
                )
            )
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": 0, "unicode": ""})
        )

        assert engine.snapshot().phase is Phase.RESULTS
    finally:
        pygame.quit()


def test_pause_settings_include_review_mode_row(tmp_path) -> None:
    store = DifficultySettingsStore(tmp_path / "difficulty-settings.json")
    _app, screen, _engines = _build_app_and_screen(
        phase=Phase.PRACTICE,
        difficulty_settings_store=store,
    )
    try:
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "mod": 0, "unicode": ""})
        )
        settings_index = screen._pause_menu_options().index("Settings")
        for _ in range(settings_index):
            screen.handle_event(
                pygame.event.Event(
                    pygame.KEYDOWN,
                    {"key": pygame.K_DOWN, "mod": 0, "unicode": ""},
                )
            )
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": 0, "unicode": ""})
        )

        rows = screen._pause_settings_rows()
        assert any(key == "review_mode" for key, _label, _value in rows)
    finally:
        pygame.quit()


def test_pause_settings_seed_value_uses_enter_for_manual_input(tmp_path) -> None:
    store = DifficultySettingsStore(tmp_path / "difficulty-settings.json")
    _app, screen, _engines = _build_app_and_screen(
        phase=Phase.PRACTICE,
        test_code="numerical_operations",
        difficulty_settings_store=store,
    )
    try:
        surface = pygame.display.get_surface()
        assert surface is not None

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "mod": 0, "unicode": ""})
        )
        settings_index = screen._pause_menu_options().index("Settings")
        for _ in range(settings_index):
            screen.handle_event(
                pygame.event.Event(
                    pygame.KEYDOWN,
                    {"key": pygame.K_DOWN, "mod": 0, "unicode": ""},
                )
            )
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": 0, "unicode": ""})
        )

        rows = screen._pause_settings_rows()
        seed_index = next(idx for idx, (key, _label, _value) in enumerate(rows) if key == "seed_value")
        screen._pause_settings_selected = seed_index
        screen.render(surface)

        assert (seed_index, "dec") not in screen._pause_settings_control_hitboxes
        assert (seed_index, "inc") not in screen._pause_settings_control_hitboxes

        original = screen._pause_settings_rows()[seed_index][2]
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RIGHT, "mod": 0, "unicode": ""})
        )
        assert screen._pause_settings_rows()[seed_index][2] == original
        assert screen._pause_seed_editing is False

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": 0, "unicode": ""})
        )

        assert screen._pause_seed_editing is True
        assert screen._pause_seed_manual_enabled is True

        for digit in "4242":
            screen.handle_event(
                pygame.event.Event(
                    pygame.KEYDOWN,
                    {"key": ord(digit), "mod": 0, "unicode": digit},
                )
            )
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": 0, "unicode": ""})
        )

        assert screen._pause_seed_editing is False
        assert screen._pause_settings_rows()[seed_index][2] == "4242"
    finally:
        pygame.quit()


def test_pause_settings_open_joystick_bindings_screen_and_return_to_pause_settings(tmp_path) -> None:
    profile_store = InputProfilesStore(tmp_path / "input-profiles.json")
    _app, screen, _engines = _build_app_and_screen(
        phase=Phase.PRACTICE,
        input_profiles_store=profile_store,
    )
    try:
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "mod": 0, "unicode": ""})
        )
        settings_index = screen._pause_menu_options().index("Settings")
        for _ in range(settings_index):
            screen.handle_event(
                pygame.event.Event(
                    pygame.KEYDOWN,
                    {"key": pygame.K_DOWN, "mod": 0, "unicode": ""},
                )
            )
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": 0, "unicode": ""})
        )

        rows = screen._pause_settings_rows()
        binding_index = next(
            idx for idx, (key, _label, _value) in enumerate(rows) if key == "joystick_bindings"
        )
        while screen._pause_settings_selected != binding_index:
            screen.handle_event(
                pygame.event.Event(
                    pygame.KEYDOWN,
                    {"key": pygame.K_DOWN, "mod": 0, "unicode": ""},
                )
            )
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": 0, "unicode": ""})
        )

        assert isinstance(screen._app._screens[-1], JoystickBindingsScreen)
        assert screen._pause_menu_active is True
        assert screen._pause_menu_mode == "settings"

        screen._app.pop()
        assert screen._app._screens[-1] is screen
        assert screen._pause_menu_active is True
        assert screen._pause_menu_mode == "settings"
    finally:
        pygame.quit()


def test_rapid_tracking_pause_settings_open_dedicated_rt_settings_screen(tmp_path) -> None:
    rapid_tracking_store = RapidTrackingSettingsStore(tmp_path / "rapid-tracking-settings.json")
    _app, screen, _engines = _build_app_and_screen(
        phase=Phase.PRACTICE,
        title="Rapid Tracking",
        test_code="rapid_tracking",
        rapid_tracking_settings_store=rapid_tracking_store,
    )
    try:
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "mod": 0, "unicode": ""})
        )
        settings_index = screen._pause_menu_options().index("Settings")
        for _ in range(settings_index):
            screen.handle_event(
                pygame.event.Event(
                    pygame.KEYDOWN,
                    {"key": pygame.K_DOWN, "mod": 0, "unicode": ""},
                )
            )
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": 0, "unicode": ""})
        )

        rows = screen._pause_settings_rows()
        rt_settings_index = next(
            idx for idx, (key, _label, _value) in enumerate(rows) if key == "rapid_tracking_settings"
        )
        assert rows[rt_settings_index][2] == "Pitch Invert OFF"

        while screen._pause_settings_selected != rt_settings_index:
            screen.handle_event(
                pygame.event.Event(
                    pygame.KEYDOWN,
                    {"key": pygame.K_DOWN, "mod": 0, "unicode": ""},
                )
            )
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": 0, "unicode": ""})
        )

        assert isinstance(screen._app._screens[-1], RapidTrackingSettingsScreen)
        assert screen._pause_menu_active is True
        assert screen._pause_menu_mode == "settings"
    finally:
        pygame.quit()


def test_dev_skip_practice_hotkey_advances_from_instructions_to_practice_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CFAST_ENABLE_DEV_TOOLS", "1")
    _app, screen, engines = _build_app_and_screen(phase=Phase.INSTRUCTIONS)
    try:
        engine = engines[-1]
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_F10, "mod": 0, "unicode": ""})
        )

        assert engine.snapshot().phase is Phase.PRACTICE_DONE
    finally:
        pygame.quit()


def test_dev_skip_section_hotkey_advances_from_scored_to_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CFAST_ENABLE_DEV_TOOLS", "1")
    _app, screen, engines = _build_app_and_screen(phase=Phase.SCORED)
    try:
        engine = engines[-1]
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_F11, "mod": 0, "unicode": ""})
        )

        assert engine.snapshot().phase is Phase.RESULTS
    finally:
        pygame.quit()


def test_dev_skip_all_hotkey_advances_from_practice_to_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CFAST_ENABLE_DEV_TOOLS", "1")
    _app, screen, engines = _build_app_and_screen(phase=Phase.PRACTICE)
    try:
        engine = engines[-1]
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_F8, "mod": 0, "unicode": ""})
        )

        assert engine.snapshot().phase is Phase.RESULTS
    finally:
        pygame.quit()


def test_dev_skip_hotkeys_do_nothing_without_dev_tools() -> None:
    _app, screen, engines = _build_app_and_screen(phase=Phase.PRACTICE)
    try:
        engine = engines[-1]
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_F8, "mod": 0, "unicode": ""})
        )

        assert engine.snapshot().phase is Phase.PRACTICE
    finally:
        pygame.quit()


def test_pause_menu_freezes_scored_timer_until_resumed() -> None:
    pygame.init()
    surface = pygame.display.set_mode((960, 540))
    font = pygame.font.Font(None, 36)
    app = App(surface=surface, font=font)
    app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))
    clock = _FakeClock()
    screen = CognitiveTestScreen(
        app,
        engine_factory=lambda: build_numerical_operations_test(
            clock=clock,
            seed=7,
            difficulty=0.5,
        ),
        test_code="numerical_operations",
    )
    app.push(screen)
    try:
        screen._engine.start_practice()
        screen._engine.start_scored()
        before = screen._engine.snapshot().time_remaining_s
        assert before is not None

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "mod": 0, "unicode": ""})
        )
        clock.advance(5.0)
        paused = screen._engine.snapshot().time_remaining_s

        assert paused == pytest.approx(before)

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "mod": 0, "unicode": ""})
        )
        clock.advance(2.0)
        resumed = screen._engine.snapshot().time_remaining_s

        assert resumed == pytest.approx(before - 2.0)
    finally:
        pygame.quit()


def test_shell_pause_blocks_long_pause_from_falling_through_to_results(tmp_path) -> None:
    pygame.init()
    surface = pygame.display.set_mode((960, 540))
    font = pygame.font.Font(None, 36)
    store = ResultsStore(tmp_path / "results.sqlite3")
    app = App(surface=surface, font=font, results_store=store)
    app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))
    clock = _FakeClock()
    screen = CognitiveTestScreen(
        app,
        engine_factory=lambda: build_numerical_operations_test(
            clock=clock,
            seed=17,
            difficulty=0.5,
        ),
        test_code="numerical_operations",
    )
    app.push(screen)
    try:
        screen._engine.start_practice()
        screen._engine.start_scored()
        before = screen._engine.snapshot().time_remaining_s
        assert before is not None

        app.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "mod": 0, "unicode": ""})
        )
        assert app.shell_pause_overlay_active() is True
        clock.advance(float(before) + 30.0)
        app.render()

        paused_snap = screen._engine.snapshot()
        assert paused_snap.phase is Phase.SCORED
        assert paused_snap.time_remaining_s == pytest.approx(before)
        assert screen._results_persisted is False
        session = store.session_summary()
        assert session is not None
        assert session.attempt_count == 0

        app.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "mod": 0, "unicode": ""})
        )
        assert app.shell_pause_overlay_active() is False
        clock.advance(1.0)
        app.render()

        resumed_snap = screen._engine.snapshot()
        assert resumed_snap.phase is Phase.SCORED
        assert resumed_snap.time_remaining_s == pytest.approx(float(before) - 1.0)
    finally:
        pygame.quit()


def test_workout_block_pause_freezes_child_engine_before_runtime_screen_exists() -> None:
    pygame.init()
    surface = pygame.display.set_mode((960, 540))
    font = pygame.font.Font(None, 36)
    app = App(surface=surface, font=font)
    app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))
    clock = _FakeClock()
    session = AntWorkoutSession(
        clock=clock,
        seed=19,
        plan=_build_single_block_no_workout_plan(),
        starting_level=5,
    )
    session.activate()
    assert session.stage is AntWorkoutStage.BLOCK_SETUP
    session.activate()
    assert session.stage is AntWorkoutStage.BLOCK
    screen = AntWorkoutScreen(app, session=session, test_code="numerical_operations_workout")
    app.push(screen)
    try:
        engine = session.current_engine()
        assert engine is not None
        before = engine.snapshot().time_remaining_s
        assert before is not None

        app.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "mod": 0, "unicode": ""})
        )
        assert app.shell_pause_overlay_active() is True
        clock.advance(float(before) + 10.0)
        app.render()

        assert session.stage is AntWorkoutStage.BLOCK
        paused_engine = session.current_engine()
        assert paused_engine is engine
        paused_snap = paused_engine.snapshot()
        assert paused_snap.phase is Phase.SCORED
        assert paused_snap.time_remaining_s == pytest.approx(before)

        app.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "mod": 0, "unicode": ""})
        )
        clock.advance(1.0)
        app.render()

        assert session.stage is AntWorkoutStage.BLOCK
        resumed_snap = engine.snapshot()
        assert resumed_snap.time_remaining_s == pytest.approx(float(before) - 1.0)
    finally:
        pygame.quit()


def test_rapid_tracking_gl_scene_elapsed_freezes_while_shell_paused() -> None:
    pygame.init()
    surface = pygame.display.set_mode((960, 540))
    font = pygame.font.Font(None, 36)
    app = App(surface=surface, font=font)
    app.set_opengl_enabled(True)
    app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))
    clock = _FakeClock()
    screen = CognitiveTestScreen(
        app,
        engine_factory=lambda: build_rapid_tracking_test(
            clock=clock,
            seed=23,
            difficulty=0.5,
        ),
        test_code="rapid_tracking",
    )
    app.push(screen)
    try:
        screen._engine.start_scored()
        app.render()
        scene = app.consume_gl_scene()
        assert isinstance(scene, RapidTrackingGlScene)
        assert scene.payload is not None
        before_elapsed = scene.payload.phase_elapsed_s
        before_remaining = screen._engine.snapshot().time_remaining_s
        assert before_remaining is not None

        app.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "mod": 0, "unicode": ""})
        )
        clock.advance(12.0)
        app.render()
        paused_scene = app.consume_gl_scene()
        assert isinstance(paused_scene, RapidTrackingGlScene)
        assert paused_scene.payload is not None
        paused_snap = screen._engine.snapshot()

        assert paused_scene.payload.phase_elapsed_s == pytest.approx(before_elapsed)
        assert paused_snap.phase is Phase.SCORED
        assert paused_snap.time_remaining_s == pytest.approx(before_remaining)

        app.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "mod": 0, "unicode": ""})
        )
        clock.advance(1.0)
        app.render()
        resumed_scene = app.consume_gl_scene()
        assert isinstance(resumed_scene, RapidTrackingGlScene)
        assert resumed_scene.payload is not None
        resumed_snap = screen._engine.snapshot()

        assert resumed_scene.payload.phase_elapsed_s > paused_scene.payload.phase_elapsed_s
        assert resumed_snap.time_remaining_s == pytest.approx(float(before_remaining) - 1.0, abs=0.05)
    finally:
        pygame.quit()


def test_pause_menu_skip_does_not_persist_attempt(tmp_path) -> None:
    pygame.init()
    surface = pygame.display.set_mode((960, 540))
    font = pygame.font.Font(None, 36)
    store = ResultsStore(tmp_path / "results.sqlite3")
    app = App(surface=surface, font=font, results_store=store)
    app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))
    clock = _FakeClock()
    screen = CognitiveTestScreen(
        app,
        engine_factory=lambda: build_numerical_operations_test(
            clock=clock,
            seed=11,
            difficulty=0.5,
        ),
        test_code="numerical_operations",
    )
    app.push(screen)
    try:
        screen._engine.start_practice()
        screen._force_engine_phase(Phase.PRACTICE_DONE)
        screen._engine.start_scored()
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "mod": 0, "unicode": ""})
        )
        skip_index = screen._pause_menu_options().index("Skip Current Segment")
        for _ in range(skip_index):
            screen.handle_event(
                pygame.event.Event(
                    pygame.KEYDOWN,
                    {"key": pygame.K_DOWN, "mod": 0, "unicode": ""},
                )
            )
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": 0, "unicode": ""})
        )
        screen.render(surface)

        session = store.session_summary()
        assert session is not None
        assert session.activity_count == 1
        assert session.completed_activity_count == 0
        assert session.aborted_activity_count == 1
        assert session.attempt_count == 0
        assert screen._results_persistence_lines == ["Local save skipped in dev mode."]
    finally:
        pygame.quit()


def test_pause_settings_apply_and_restart_rebuilds_screen_with_new_difficulty(tmp_path) -> None:
    store = DifficultySettingsStore(tmp_path / "difficulty-settings.json")
    app, screen, engines = _build_app_and_screen(
        phase=Phase.PRACTICE,
        test_code="rapid_tracking",
        difficulty_settings_store=store,
    )
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        first_engine = engines[-1]

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "mod": 0, "unicode": ""})
        )
        settings_index = screen._pause_menu_options().index("Settings")
        for _ in range(settings_index):
            screen.handle_event(
                pygame.event.Event(
                    pygame.KEYDOWN,
                    {"key": pygame.K_DOWN, "mod": 0, "unicode": ""},
                )
            )
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": 0, "unicode": ""})
        )
        screen.render(surface)
        assert screen._pause_menu_mode == "settings"

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RIGHT, "mod": 0, "unicode": ""})
        )
        assert store.test_level("rapid_tracking") == 5
        assert first_engine._difficulty == pytest.approx((5 - 1) / 9.0)

        rows = screen._pause_settings_rows()
        apply_index = next(
            idx for idx, (key, _label, _value) in enumerate(rows) if key == "apply_restart"
        )
        while screen._pause_settings_selected != apply_index:
            screen.handle_event(
                pygame.event.Event(
                    pygame.KEYDOWN,
                    {"key": pygame.K_DOWN, "mod": 0, "unicode": ""},
                )
            )
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": 0, "unicode": ""})
        )

        new_screen = cast(CognitiveTestScreen, app._screens[-1])
        assert new_screen is not screen
        assert store.test_level("rapid_tracking") == 6
        assert app.effective_difficulty_level("rapid_tracking") == 6
        assert len(engines) == 2
        assert engines[-1]._difficulty == pytest.approx((6 - 1) / 9.0)
    finally:
        pygame.quit()


def test_intro_difficulty_is_staged_until_enter_starts_practice(tmp_path) -> None:
    store = DifficultySettingsStore(tmp_path / "difficulty-settings.json")
    app, screen, engines = _build_app_and_screen(
        phase=Phase.INSTRUCTIONS,
        test_code="rapid_tracking",
        difficulty_settings_store=store,
    )
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        first_engine = engines[-1]

        for _ in range(8):
            screen.render(surface)
        assert screen._get_intro_difficulty_level() == 5
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RIGHT, "mod": 0, "unicode": ""})
        )

        assert store.test_level("rapid_tracking") == 5
        assert first_engine._difficulty == pytest.approx((5 - 1) / 9.0)

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": 0, "unicode": ""})
        )

        new_screen = cast(CognitiveTestScreen, app._screens[-1])
        assert new_screen is not screen
        assert store.test_level("rapid_tracking") == 6
        assert app.effective_difficulty_level("rapid_tracking") == 6
        assert len(engines) == 2
        assert engines[-1]._difficulty == pytest.approx((6 - 1) / 9.0)
        assert engines[-1].snapshot().phase is Phase.PRACTICE
    finally:
        pygame.quit()


def test_intro_difficulty_change_from_practice_done_restarts_to_beginning(tmp_path) -> None:
    store = DifficultySettingsStore(tmp_path / "difficulty-settings.json")
    store.set_test_level(test_code="rapid_tracking", level=7)
    app, screen, engines = _build_app_and_screen(
        phase=Phase.PRACTICE_DONE,
        test_code="rapid_tracking",
        difficulty_settings_store=store,
    )
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        first_engine = engines[-1]

        screen.render(surface)
        for _ in range(8):
            screen.render(surface)

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_LEFT, "mod": 0, "unicode": ""})
        )
        assert store.test_level("rapid_tracking") == 7
        assert engines[-1]._difficulty == pytest.approx((7 - 1) / 9.0)

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": 0, "unicode": ""})
        )

        new_screen = cast(CognitiveTestScreen, app._screens[-1])
        assert new_screen is not screen
        assert store.test_level("rapid_tracking") == 6
        assert first_engine.snapshot().phase is Phase.PRACTICE_DONE
        assert first_engine._difficulty == pytest.approx((7 - 1) / 9.0)
        assert engines[-1] is not first_engine
        assert engines[-1].snapshot().phase is Phase.INSTRUCTIONS
        assert engines[-1]._difficulty == pytest.approx((6 - 1) / 9.0)
    finally:
        pygame.quit()


def test_numerical_operations_intro_segments_support_next_and_back_without_affecting_live_stage() -> None:
    _app, screen, engines = _build_app_and_screen(
        phase=Phase.INSTRUCTIONS,
        test_code="numerical_operations",
    )
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        engine = engines[-1]

        screen.render(surface)
        for _ in range(INTRO_LOADING_MIN_FRAMES):
            screen.render(surface)

        assert screen._intro_segment_index == 0

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_TAB, "mod": 0, "unicode": ""})
        )
        assert screen._intro_segment_index == 1

        screen.handle_event(
            pygame.event.Event(
                pygame.KEYDOWN,
                {"key": pygame.K_TAB, "mod": pygame.KMOD_SHIFT, "unicode": ""},
            )
        )
        assert screen._intro_segment_index == 0

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_PAGEDOWN, "mod": 0, "unicode": ""})
        )
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_PAGEDOWN, "mod": 0, "unicode": ""})
        )
        assert screen._intro_segment_index == 2

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": 0, "unicode": ""})
        )
        assert engine.snapshot().phase is Phase.PRACTICE

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_TAB, "mod": 0, "unicode": ""})
        )
        assert engine.snapshot().phase is Phase.PRACTICE
    finally:
        pygame.quit()


def test_intro_loading_blocks_enter_until_practice_stage_is_ready() -> None:
    _app, screen, engines = _build_app_and_screen(phase=Phase.INSTRUCTIONS)
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        engine = engines[-1]

        screen.render(surface)
        assert screen._intro_loading_complete(Phase.INSTRUCTIONS) is False

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": 0, "unicode": ""})
        )
        assert engine.snapshot().phase is Phase.INSTRUCTIONS

        for _ in range(8):
            screen.render(surface)

        assert screen._intro_loading_complete(Phase.INSTRUCTIONS) is True
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": 0, "unicode": ""})
        )
        assert engine.snapshot().phase is Phase.PRACTICE
    finally:
        pygame.quit()


def test_practice_done_loading_blocks_enter_until_scored_stage_is_ready() -> None:
    _app, screen, engines = _build_app_and_screen(phase=Phase.PRACTICE_DONE)
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        engine = engines[-1]

        screen.render(surface)
        assert screen._intro_loading_complete(Phase.PRACTICE_DONE) is False

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": 0, "unicode": ""})
        )
        assert engine.snapshot().phase is Phase.PRACTICE_DONE

        for _ in range(8):
            screen.render(surface)

        assert screen._intro_loading_complete(Phase.PRACTICE_DONE) is True
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": 0, "unicode": ""})
        )
        assert engine.snapshot().phase is Phase.SCORED
    finally:
        pygame.quit()


def test_auditory_intro_loading_shows_callsigns_and_rule_recap() -> None:
    _app, screen, engines = _build_app_and_screen(
        phase=Phase.INSTRUCTIONS,
        title="Auditory Capacity",
    )
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        engine = engines[-1]
        engine._assigned_callsigns = ("RAVEN", "EAGLE", "VIPER")

        tiny_font = _SpyFont()
        small_font = _SpyFont()
        app_font = _SpyFont()
        screen._tiny_font = tiny_font
        screen._small_font = small_font
        screen._app._font = app_font

        screen.render(surface)
        rendered_text = "\n".join(app_font.rendered + small_font.rendered + tiny_font.rendered)

        assert "Call signs: RAVEN, EAGLE, VIPER" in rendered_text
        assert "Digit groups are" in rendered_text
        assert "Type them with Enter." in rendered_text
        assert "Press Space or trigger once when you hear the beep." in rendered_text
        assert "Next-gate directives apply to the next matching gate only." in rendered_text
    finally:
        pygame.quit()


def test_auditory_prefixed_title_uses_live_auditory_renderer_path() -> None:
    _app, screen, _engines = _build_app_and_screen(
        phase=Phase.PRACTICE,
        title="Auditory Capacity: Gate Anchor",
    )
    try:
        surface = pygame.display.get_surface()
        assert surface is not None

        tiny_font = _SpyFont()
        small_font = _SpyFont()
        app_font = _SpyFont()
        screen._tiny_font = tiny_font
        screen._small_font = small_font
        screen._app._font = app_font

        screen.render(surface)
        rendered_text = "\n".join(app_font.rendered + small_font.rendered + tiny_font.rendered)

        assert "Auditory Capacity - Practice" in rendered_text
        assert "Q/W/E/R: colour" in rendered_text
        assert "Scored " not in rendered_text
    finally:
        pygame.quit()


def test_auditory_testing_menu_toggles_for_prefixed_title() -> None:
    _app, screen, _engines = _build_app_and_screen(
        phase=Phase.PRACTICE,
        title="Auditory Capacity: Gate Anchor",
    )
    try:
        starting_state = screen._auditory_testing_menu
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_F9, "mod": 0, "unicode": ""})
        )
        assert screen._auditory_testing_menu is (not starting_state)
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_F9, "mod": 0, "unicode": ""})
        )
        assert screen._auditory_testing_menu is starting_state
    finally:
        pygame.quit()


def test_rapid_tracking_instructions_show_session_seed() -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        app = App(surface=surface, font=font)
        root = MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True)
        app.push(root)

        clock = _FakeClock()
        screen = CognitiveTestScreen(
            app,
            engine_factory=lambda: build_rapid_tracking_test(
                clock=clock,
                seed=4321,
                difficulty=0.5,
            ),
            test_code="rapid_tracking",
        )
        app.push(screen)

        tiny_font = _SpyFont()
        small_font = _SpyFont()
        app_font = _SpyFont()
        screen._tiny_font = tiny_font
        screen._small_font = small_font
        screen._app._font = app_font

        screen.render(surface)
        rendered_text = "\n".join(app_font.rendered + small_font.rendered + tiny_font.rendered)

        assert "Session Seed 4321" in rendered_text
    finally:
        pygame.quit()


def test_keyboard_camera_controls_match_left_right_directions(monkeypatch) -> None:
    _app, screen, _engines = _build_app_and_screen(
        phase=Phase.PRACTICE,
        title="Rapid Tracking",
        test_code="rapid_tracking",
    )
    try:
        monkeypatch.setattr("cfast_trainer.app._iter_connected_joysticks", lambda: [])

        monkeypatch.setattr(
            pygame.key,
            "get_pressed",
            lambda: _PressedKeys({pygame.K_LEFT}),
        )
        left_horizontal, left_vertical = screen._read_sensory_motor_control()
        assert left_horizontal == 1.0
        assert left_vertical == 0.0

        monkeypatch.setattr(
            pygame.key,
            "get_pressed",
            lambda: _PressedKeys({pygame.K_RIGHT}),
        )
        right_horizontal, right_vertical = screen._read_sensory_motor_control()
        assert right_horizontal == -1.0
        assert right_vertical == 0.0

        monkeypatch.setattr(
            pygame.key,
            "get_pressed",
            lambda: _PressedKeys({pygame.K_a}),
        )
        a_horizontal, _a_vertical = screen._read_sensory_motor_control()
        assert a_horizontal == 1.0

        monkeypatch.setattr(
            pygame.key,
            "get_pressed",
            lambda: _PressedKeys({pygame.K_d}),
        )
        d_horizontal, _d_vertical = screen._read_sensory_motor_control()
        assert d_horizontal == -1.0
    finally:
        pygame.quit()


def test_keyboard_camera_controls_follow_key_events_even_without_pressed_snapshot(monkeypatch) -> None:
    _app, screen, _engines = _build_app_and_screen(
        phase=Phase.PRACTICE,
        title="Rapid Tracking",
        test_code="rapid_tracking",
    )
    try:
        monkeypatch.setattr("cfast_trainer.app._iter_connected_joysticks", lambda: [])
        monkeypatch.setattr(pygame.key, "get_pressed", lambda: _PressedKeys(set()))

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_LEFT, "mod": 0, "unicode": ""})
        )
        left_horizontal, left_vertical = screen._read_sensory_motor_control()
        assert left_horizontal == 1.0
        assert left_vertical == 0.0

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_UP, "mod": 0, "unicode": ""})
        )
        up_horizontal, up_vertical = screen._read_sensory_motor_control()
        assert up_horizontal == 1.0
        assert up_vertical == -1.0

        screen.handle_event(
            pygame.event.Event(pygame.KEYUP, {"key": pygame.K_LEFT, "mod": 0, "unicode": ""})
        )
        screen.handle_event(
            pygame.event.Event(pygame.KEYUP, {"key": pygame.K_UP, "mod": 0, "unicode": ""})
        )
        released_horizontal, released_vertical = screen._read_sensory_motor_control()
        assert released_horizontal == 0.0
        assert released_vertical == 0.0
    finally:
        pygame.quit()


def test_keyboard_vertical_controls_override_connected_joystick(monkeypatch) -> None:
    _app, screen, _engines = _build_app_and_screen(
        phase=Phase.PRACTICE,
        title="Rapid Tracking",
        test_code="rapid_tracking",
    )
    try:
        class _FakeJoystick:
            def get_numaxes(self) -> int:
                return 4

            def get_axis(self, idx: int) -> float:
                if idx == 1:
                    return -0.65
                if idx == 3:
                    return 0.40
                return 0.0

            def get_name(self) -> str:
                return "test stick"

        monkeypatch.setattr("cfast_trainer.app._iter_connected_joysticks", lambda: [_FakeJoystick()])
        monkeypatch.setattr(
            pygame.key,
            "get_pressed",
            lambda: _PressedKeys({pygame.K_DOWN}),
        )

        horizontal, vertical = screen._read_sensory_motor_control()
        assert horizontal == pytest.approx(0.40)
        assert vertical == 1.0
    finally:
        pygame.quit()


def test_app_escape_opens_pause_for_sensory_motor_apparatus_screen() -> None:
    app, screen, _engines = _build_app_and_screen(
        phase=Phase.PRACTICE,
        title="Sensory Motor Apparatus",
        test_code="sensory_motor_apparatus",
    )
    try:
        app.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "mod": 0, "unicode": ""})
        )

        assert app.shell_pause_overlay_active() is True
        assert screen._pause_menu_active is True
        assert app._current_screen() is screen
    finally:
        pygame.quit()


def test_sensory_motor_intro_footer_mentions_pause_not_back() -> None:
    _app, screen, _engines = _build_app_and_screen(
        phase=Phase.INSTRUCTIONS,
        title="Sensory Motor Apparatus",
        test_code="sensory_motor_apparatus",
    )
    try:
        surface = pygame.display.get_surface()
        assert surface is not None

        tiny_font = _SpyFont()
        small_font = _SpyFont()
        app_font = _SpyFont()
        screen._tiny_font = tiny_font
        screen._small_font = small_font
        screen._app._font = app_font

        screen.render(surface)
        rendered_text = "\n".join(app_font.rendered + small_font.rendered + tiny_font.rendered)

        assert "Esc/Backspace: Pause" in rendered_text
        assert "Esc/Backspace: Back" not in rendered_text
    finally:
        pygame.quit()


def test_sensory_motor_joystick_only_uses_stick_x_and_ignores_rudder(monkeypatch) -> None:
    _app, screen, _engines = _build_app_and_screen(
        phase=Phase.PRACTICE,
        title="Sensory Motor Apparatus",
        test_code="sensory_motor_apparatus",
    )
    try:
        class _PrimaryStick:
            def get_numaxes(self) -> int:
                return 4

            def get_axis(self, idx: int) -> float:
                return {
                    0: -0.55,
                    1: 0.25,
                    3: 0.90,
                }.get(idx, 0.0)

            def get_name(self) -> str:
                return "primary stick"

        class _RudderPedals:
            def get_numaxes(self) -> int:
                return 4

            def get_axis(self, idx: int) -> float:
                return {
                    3: -0.80,
                    2: -0.80,
                }.get(idx, 0.0)

            def get_name(self) -> str:
                return "rudder pedals"

        monkeypatch.setattr(
            "cfast_trainer.app._iter_connected_joysticks",
            lambda: [_PrimaryStick(), _RudderPedals()],
        )
        monkeypatch.setattr(pygame.key, "get_pressed", lambda: _PressedKeys(set()))

        joystick_only_horizontal, joystick_only_vertical = screen._read_sensory_motor_control(
            control_mode="joystick_only"
        )
        split_horizontal, split_vertical = screen._read_sensory_motor_control(
            control_mode="split"
        )

        assert joystick_only_horizontal == pytest.approx(-0.55)
        assert joystick_only_vertical == pytest.approx(0.25)
        assert split_horizontal == pytest.approx(-0.80)
        assert split_vertical == pytest.approx(0.25)
    finally:
        pygame.quit()


def test_sensory_motor_fallback_vertical_axis_respects_profile_inversion(
    monkeypatch, tmp_path
) -> None:
    profile_store = InputProfilesStore(tmp_path / "input-profiles.json")
    profile_id = profile_store.active_profile().profile_id
    neutral = AxisCalibrationSettings(deadzone=0.0, curve=1.0)
    profile_store.set_axis_calibration(
        profile_id=profile_id,
        axis_key="primary stick|guid-1::axis0",
        settings=neutral,
    )
    profile_store.set_axis_calibration(
        profile_id=profile_id,
        axis_key="primary stick|guid-1::axis1",
        settings=AxisCalibrationSettings(deadzone=0.0, invert=True, curve=1.0),
    )
    profile_store.set_axis_calibration(
        profile_id=profile_id,
        axis_key="primary stick|guid-1::axis3",
        settings=neutral,
    )

    _app, screen, _engines = _build_app_and_screen(
        phase=Phase.PRACTICE,
        title="Sensory Motor Apparatus",
        test_code="sensory_motor_apparatus",
        input_profiles_store=profile_store,
    )
    try:
        class _PrimaryStick:
            def get_name(self) -> str:
                return "primary stick"

            def get_guid(self) -> str:
                return "guid-1"

            def get_numaxes(self) -> int:
                return 4

            def get_axis(self, idx: int) -> float:
                return {
                    0: 0.15,
                    1: 0.60,
                    3: -0.25,
                }.get(idx, 0.0)

        monkeypatch.setattr("cfast_trainer.app._iter_connected_joysticks", lambda: [_PrimaryStick()])
        monkeypatch.setattr(pygame.key, "get_pressed", lambda: _PressedKeys(set()))

        joystick_only_horizontal, joystick_only_vertical = screen._read_sensory_motor_control(
            control_mode="joystick_only"
        )
        split_horizontal, split_vertical = screen._read_sensory_motor_control(
            control_mode="split"
        )

        assert joystick_only_horizontal == pytest.approx(0.15)
        assert joystick_only_vertical == pytest.approx(-0.60)
        assert split_horizontal == pytest.approx(-0.25)
        assert split_vertical == pytest.approx(-0.60)
    finally:
        pygame.quit()


def test_sensory_motor_axis_focus_zeroes_inactive_axis(monkeypatch) -> None:
    _app, screen, _engines = _build_app_and_screen(
        phase=Phase.PRACTICE,
        title="Sensory Motor Apparatus",
        test_code="sensory_motor_apparatus",
    )
    try:
        monkeypatch.setattr(
            pygame.key,
            "get_pressed",
            lambda: _PressedKeys({pygame.K_LEFT, pygame.K_UP}),
        )

        horizontal_only = screen._read_sensory_motor_control(
            control_mode="split",
            axis_focus="horizontal",
        )
        vertical_only = screen._read_sensory_motor_control(
            control_mode="split",
            axis_focus="vertical",
        )

        assert horizontal_only[1] == 0.0
        assert abs(horizontal_only[0]) == 1.0
        assert vertical_only[0] == 0.0
        assert vertical_only[1] == -1.0
    finally:
        pygame.quit()


def test_explicit_axis_role_bindings_override_legacy_joystick_heuristics(monkeypatch, tmp_path) -> None:
    profile_store = InputProfilesStore(tmp_path / "input-profiles.json")
    profile_id = profile_store.active_profile().profile_id
    profile_store.set_axis_role_binding(
        profile_id=profile_id,
        role="primary_horizontal",
        binding=AnalogBinding(device_key="bound stick|guid-1", axis_index=2),
    )
    profile_store.set_axis_role_binding(
        profile_id=profile_id,
        role="primary_vertical",
        binding=AnalogBinding(device_key="bound stick|guid-1", axis_index=3),
    )
    profile_store.set_axis_role_binding(
        profile_id=profile_id,
        role="rudder_horizontal",
        binding=AnalogBinding(device_key="bound stick|guid-1", axis_index=1),
    )
    neutral_calibration = AxisCalibrationSettings(deadzone=0.0, curve=1.0)
    for axis_key in (
        "bound stick|guid-1::axis1",
        "bound stick|guid-1::axis2",
        "bound stick|guid-1::axis3",
    ):
        profile_store.set_axis_calibration(
            profile_id=profile_id,
            axis_key=axis_key,
            settings=neutral_calibration,
        )

    _app, screen, _engines = _build_app_and_screen(
        phase=Phase.PRACTICE,
        title="Rapid Tracking",
        test_code="rapid_tracking",
        input_profiles_store=profile_store,
    )
    try:
        class _FakeJoystick:
            def get_name(self) -> str:
                return "bound stick"

            def get_guid(self) -> str:
                return "guid-1"

            def get_numaxes(self) -> int:
                return 4

            def get_axis(self, idx: int) -> float:
                return {
                    0: 0.95,
                    1: -0.40,
                    2: 0.22,
                    3: -0.58,
                }.get(idx, 0.0)

        monkeypatch.setattr("cfast_trainer.app._iter_connected_joysticks", lambda: [_FakeJoystick()])
        monkeypatch.setattr(pygame.key, "get_pressed", lambda: _PressedKeys(set()))

        screen._app._joystick_binding_router.poll()
        joystick_only_horizontal, joystick_only_vertical = screen._read_sensory_motor_control(
            control_mode="joystick_only"
        )
        split_horizontal, split_vertical = screen._read_sensory_motor_control(
            control_mode="split"
        )

        assert joystick_only_horizontal == pytest.approx(0.22)
        assert joystick_only_vertical == pytest.approx(-0.58)
        assert split_horizontal == pytest.approx(-0.40)
        assert split_vertical == pytest.approx(-0.58)
    finally:
        pygame.quit()


def test_explicit_rapid_tracking_capture_binding_disables_legacy_joystick_button_fallback(
    monkeypatch, tmp_path
) -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        profile_store = InputProfilesStore(tmp_path / "input-profiles.json")
        profile_id = profile_store.active_profile().profile_id
        profile_store.set_action_binding(
            profile_id=profile_id,
            action="rapid_tracking_capture",
            slot_index=0,
            binding=DigitalBinding(kind="button", device_key="capture stick|guid-1", control_index=4),
        )

        app = App(surface=surface, font=font, input_profiles_store=profile_store)
        app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))
        clock = _FakeClock()
        screen = CognitiveTestScreen(
            app,
            engine_factory=lambda: build_rapid_tracking_test(
                clock=clock,
                seed=4321,
                difficulty=0.5,
            ),
            test_code="rapid_tracking",
        )
        app.push(screen)

        class _FakeJoystick:
            def __init__(self) -> None:
                self.buttons = [0] * 8

            def get_name(self) -> str:
                return "capture stick"

            def get_guid(self) -> str:
                return "guid-1"

            def get_numaxes(self) -> int:
                return 4

            def get_axis(self, idx: int) -> float:
                return 0.0

            def get_numbuttons(self) -> int:
                return len(self.buttons)

            def get_button(self, idx: int) -> int:
                return int(self.buttons[idx])

            def get_numhats(self) -> int:
                return 0

        fake = _FakeJoystick()
        monkeypatch.setattr("cfast_trainer.app._iter_connected_joysticks", lambda: [fake])
        engine = cast(object, screen._engine)
        engine.start_practice()
        base_payload = engine.snapshot().payload
        assert base_payload is not None
        assert base_payload.capture_attempts == 0

        screen.handle_event(
            pygame.event.Event(pygame.JOYBUTTONDOWN, {"button": 0, "instance_id": 0})
        )
        after_legacy_payload = engine.snapshot().payload
        assert after_legacy_payload is not None
        assert after_legacy_payload.capture_attempts == 0

        fake.buttons[4] = 1
        app.render()
        after_bound_payload = engine.snapshot().payload
        assert after_bound_payload is not None
        assert after_bound_payload.capture_attempts == 1
    finally:
        pygame.quit()


def test_explicit_rapid_tracking_capture_binding_supports_hold_zoom_and_capture(
    monkeypatch, tmp_path
) -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        profile_store = InputProfilesStore(tmp_path / "input-profiles.json")
        profile_id = profile_store.active_profile().profile_id
        profile_store.set_action_binding(
            profile_id=profile_id,
            action="rapid_tracking_capture",
            slot_index=0,
            binding=DigitalBinding(kind="button", device_key="capture stick|guid-1", control_index=4),
        )

        app = App(surface=surface, font=font, input_profiles_store=profile_store)
        app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))
        clock = _FakeClock()
        screen = CognitiveTestScreen(
            app,
            engine_factory=lambda: build_rapid_tracking_test(
                clock=clock,
                seed=9876,
                difficulty=0.5,
            ),
            test_code="rapid_tracking",
        )
        app.push(screen)

        class _FakeJoystick:
            def __init__(self) -> None:
                self.buttons = [0] * 8

            def get_name(self) -> str:
                return "capture stick"

            def get_guid(self) -> str:
                return "guid-1"

            def get_numaxes(self) -> int:
                return 4

            def get_axis(self, idx: int) -> float:
                return 0.0

            def get_numbuttons(self) -> int:
                return len(self.buttons)

            def get_button(self, idx: int) -> int:
                return int(self.buttons[idx])

            def get_numhats(self) -> int:
                return 0

        fake = _FakeJoystick()
        monkeypatch.setattr("cfast_trainer.app._iter_connected_joysticks", lambda: [fake])
        monkeypatch.setattr(pygame.key, "get_pressed", lambda: _PressedKeys(set()))
        engine = cast(object, screen._engine)
        engine.start_practice()
        engine._target_x = 0.0
        engine._target_y = 0.0
        engine._target_terrain_occluded = False
        engine._reset_camera_pose_to_target()

        fake.buttons[4] = 1
        app.render()
        first = engine.snapshot().payload
        assert first is not None
        assert first.capture_attempts == 1

        clock.advance(0.30)
        app.render()
        held = engine.snapshot().payload
        assert held is not None
        assert held.capture_zoom > 0.8
        assert held.capture_points >= 2

        fake.buttons[4] = 0
        clock.advance(0.30)
        app.render()
        released = engine.snapshot().payload
        assert released is not None
        assert released.capture_zoom < 0.2
    finally:
        pygame.quit()


def test_explicit_rapid_tracking_capture_binding_waits_for_last_bound_control_release(
    monkeypatch, tmp_path
) -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        profile_store = InputProfilesStore(tmp_path / "input-profiles.json")
        profile_id = profile_store.active_profile().profile_id
        profile_store.set_action_binding(
            profile_id=profile_id,
            action="rapid_tracking_capture",
            slot_index=0,
            binding=DigitalBinding(kind="button", device_key="capture stick|guid-1", control_index=4),
        )
        profile_store.set_action_binding(
            profile_id=profile_id,
            action="rapid_tracking_capture",
            slot_index=1,
            binding=DigitalBinding(kind="button", device_key="capture stick|guid-1", control_index=5),
        )

        app = App(surface=surface, font=font, input_profiles_store=profile_store)
        app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))
        clock = _FakeClock()
        screen = CognitiveTestScreen(
            app,
            engine_factory=lambda: build_rapid_tracking_test(
                clock=clock,
                seed=9877,
                difficulty=0.5,
            ),
            test_code="rapid_tracking",
        )
        app.push(screen)

        class _FakeJoystick:
            def __init__(self) -> None:
                self.buttons = [0] * 8

            def get_name(self) -> str:
                return "capture stick"

            def get_guid(self) -> str:
                return "guid-1"

            def get_numaxes(self) -> int:
                return 4

            def get_axis(self, idx: int) -> float:
                return 0.0

            def get_numbuttons(self) -> int:
                return len(self.buttons)

            def get_button(self, idx: int) -> int:
                return int(self.buttons[idx])

            def get_numhats(self) -> int:
                return 0

        fake = _FakeJoystick()
        monkeypatch.setattr("cfast_trainer.app._iter_connected_joysticks", lambda: [fake])
        monkeypatch.setattr(pygame.key, "get_pressed", lambda: _PressedKeys(set()))
        engine = cast(object, screen._engine)
        engine.start_practice()
        engine._target_x = 0.0
        engine._target_y = 0.0
        engine._target_terrain_occluded = False
        engine._reset_camera_pose_to_target()

        fake.buttons[4] = 1
        app.render()
        first = engine.snapshot().payload
        assert first is not None
        assert first.capture_attempts == 1
        assert first.capture_zoom > 0.0

        fake.buttons[5] = 1
        clock.advance(0.10)
        app.render()
        both_held = engine.snapshot().payload
        assert both_held is not None
        assert both_held.capture_attempts == 1
        assert both_held.capture_zoom > 0.5

        fake.buttons[4] = 0
        clock.advance(0.30)
        app.render()
        still_held = engine.snapshot().payload
        assert still_held is not None
        assert still_held.capture_attempts == 1
        assert still_held.capture_zoom > 0.8

        fake.buttons[5] = 0
        clock.advance(0.30)
        app.render()
        released = engine.snapshot().payload
        assert released is not None
        assert released.capture_zoom < 0.2
    finally:
        pygame.quit()


def test_sensory_motor_practice_done_enter_advances_to_next_internal_block() -> None:
    pygame.init()
    surface = pygame.display.set_mode((960, 540))
    font = pygame.font.Font(None, 36)
    app = App(surface=surface, font=font)
    app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))
    clock = _FakeClock()
    screen = CognitiveTestScreen(
        app,
        engine_factory=lambda: build_sensory_motor_apparatus_test(
            clock=clock,
            seed=77,
            difficulty=0.5,
            config=SensoryMotorApparatusConfig(
                practice_duration_s=1.0,
                scored_duration_s=2.0,
                tick_hz=120.0,
            ),
        ),
        test_code="sensory_motor_apparatus",
    )
    app.push(screen)
    try:
        engine = cast(object, screen._engine)
        engine.start_practice()

        clock.advance(0.5)
        engine.update()
        snap = engine.snapshot()
        assert snap.phase is Phase.PRACTICE_DONE
        assert snap.payload is not None
        assert snap.payload.block_kind == "practice"
        assert snap.payload.block_index == 2

        screen._sync_intro_loading_state(snap.phase)
        screen._intro_loading_ready = True
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": 0, "unicode": ""})
        )
        snap = engine.snapshot()
        assert snap.phase is Phase.PRACTICE
        assert snap.payload is not None
        assert snap.payload.control_mode == "split"
        assert snap.payload.block_index == 2

        clock.advance(0.5)
        engine.update()
        snap = engine.snapshot()
        assert snap.phase is Phase.PRACTICE_DONE
        assert snap.payload is not None
        assert snap.payload.block_kind == "scored"
        assert snap.payload.block_index == 1

        screen._sync_intro_loading_state(snap.phase)
        screen._intro_loading_ready = True
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": 0, "unicode": ""})
        )
        snap = engine.snapshot()
        assert snap.phase is Phase.SCORED
        assert snap.payload is not None
        assert snap.payload.control_mode == "joystick_only"
        assert snap.payload.block_index == 1
    finally:
        pygame.quit()


def test_auditory_intro_loading_reuses_frozen_world_frame() -> None:
    app, screen, engines = _build_app_and_screen(
        phase=Phase.INSTRUCTIONS,
        title="Auditory Capacity",
        test_code="auditory_capacity",
    )
    try:
        app.set_opengl_enabled(True)
        surface = pygame.display.get_surface()
        assert surface is not None
        engines[-1]._assigned_callsigns = ("RAVEN", "EAGLE", "VIPER")
        draw_calls = 0

        def _fake_world_renderer(**kwargs) -> None:
            nonlocal draw_calls
            draw_calls += 1
            target_surface = kwargs["surface"]
            target_surface.fill((draw_calls * 40, 0, 0))

        screen._render_auditory_capacity_tube_chase_view = _fake_world_renderer  # type: ignore[method-assign]

        screen.render(surface)
        first_color = screen._auditory_frozen_world_frame.get_at((10, 10))
        screen.render(surface)
        second_color = screen._auditory_frozen_world_frame.get_at((10, 10))

        assert draw_calls == 1
        assert first_color == second_color
    finally:
        pygame.quit()


def test_auditory_pause_reuses_last_live_world_frame_without_advancing() -> None:
    app, screen, engines = _build_app_and_screen(
        phase=Phase.PRACTICE,
        title="Auditory Capacity",
        test_code="auditory_capacity",
    )
    try:
        app.set_opengl_enabled(True)
        surface = pygame.display.get_surface()
        assert surface is not None
        draw_calls = 0

        def _fake_world_renderer(**kwargs) -> None:
            nonlocal draw_calls
            draw_calls += 1
            target_surface = kwargs["surface"]
            target_surface.fill((draw_calls * 50, 0, 0))

        screen._render_auditory_capacity_tube_chase_view = _fake_world_renderer  # type: ignore[method-assign]

        screen.render(surface)
        assert screen._auditory_live_world_frame is not None
        live_color = screen._auditory_live_world_frame.get_at((10, 10))
        assert draw_calls == 1

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "mod": 0, "unicode": ""})
        )
        screen.render(surface)
        assert screen._auditory_frozen_world_frame is not None
        paused_color = screen._auditory_frozen_world_frame.get_at((10, 10))
        screen.render(surface)
        paused_color_again = screen._auditory_frozen_world_frame.get_at((10, 10))

        assert draw_calls == 1
        assert paused_color == live_color
        assert paused_color_again == live_color
    finally:
        pygame.quit()


def test_pause_settings_changes_auditory_mix_controls(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_defaults_store = RuntimeDefaultsStore(tmp_path / "runtime-defaults.json")
    _app, screen, engines = _build_app_and_screen(
        title="Auditory Capacity",
        runtime_defaults_store=runtime_defaults_store,
    )
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        engine = engines[-1]

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "mod": 0, "unicode": ""})
        )
        settings_index = screen._pause_menu_options().index("Settings")
        for _ in range(settings_index):
            screen.handle_event(
                pygame.event.Event(
                    pygame.KEYDOWN,
                    {"key": pygame.K_DOWN, "mod": 0, "unicode": ""},
                )
            )
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": 0, "unicode": ""})
        )
        screen.render(surface)

        rows = screen._pause_settings_rows()
        noise_index = next(
            idx for idx, (key, _label, _value) in enumerate(rows) if key == "auditory_noise"
        )
        distortion_index = next(
            idx for idx, (key, _label, _value) in enumerate(rows) if key == "auditory_distortion"
        )
        source_index = next(
            idx for idx, (key, _label, _value) in enumerate(rows) if key == "auditory_source"
        )

        while screen._pause_settings_selected != noise_index:
            screen.handle_event(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "mod": 0, "unicode": ""})
            )
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RIGHT, "mod": 0, "unicode": ""})
        )
        assert engine._noise_level_override == 0.1
        assert runtime_defaults_store.stored_auditory_noise_level() == pytest.approx(0.1)

        while screen._pause_settings_selected != distortion_index:
            screen.handle_event(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "mod": 0, "unicode": ""})
            )
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RIGHT, "mod": 0, "unicode": ""})
        )
        assert engine._distortion_level_override == 0.1
        assert runtime_defaults_store.stored_auditory_distortion_level() == pytest.approx(0.1)

        while screen._pause_settings_selected != source_index:
            screen.handle_event(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "mod": 0, "unicode": ""})
            )
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RIGHT, "mod": 0, "unicode": ""})
        )
        assert engine._noise_source_override is not None
        assert runtime_defaults_store.stored_auditory_noise_source() == engine._noise_source_override
    finally:
        pygame.quit()


def test_runtime_defaults_apply_to_auditory_engine_on_launch(tmp_path) -> None:
    runtime_defaults_store = RuntimeDefaultsStore(tmp_path / "runtime-defaults.json")
    runtime_defaults_store.set_auditory_noise_level(0.4)
    runtime_defaults_store.set_auditory_distortion_level(0.2)
    runtime_defaults_store.set_auditory_noise_source("pink")

    _app, _screen, engines = _build_app_and_screen(
        title="Auditory Capacity",
        runtime_defaults_store=runtime_defaults_store,
    )
    try:
        engine = engines[-1]
        assert engine._noise_level_override == pytest.approx(0.4)
        assert engine._distortion_level_override == pytest.approx(0.2)
        assert engine._noise_source_override == "pink"
    finally:
        pygame.quit()


def test_auditory_drill_wrapper_supports_pause_audio_settings() -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        app = App(surface=surface, font=font)
        app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))
        clock = _FakeClock()
        screen = CognitiveTestScreen(
            app,
            engine_factory=lambda: build_ac_gate_anchor_drill(
                clock=clock,
                seed=77,
                difficulty=0.5,
                config=AcDrillConfig(scored_duration_s=4.0),
            ),
            test_code="ac_gate_anchor",
        )
        app.push(screen)

        assert screen._supports_auditory_pause_settings() is True
        screen._apply_auditory_pause_settings(noise_step=1, distortion_step=2)

        assert getattr(screen._engine, "_noise_level_override", None) == pytest.approx(0.1)
        assert getattr(screen._engine, "_distortion_level_override", None) == pytest.approx(0.2)
    finally:
        pygame.quit()
