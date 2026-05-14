from __future__ import annotations

import os
from typing import cast

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

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
    _set_engine_clock_paused,
)
from cfast_trainer.clock import Clock, PausableClock
from cfast_trainer.cognitive_core import Phase
from cfast_trainer.cognitive_core import TestSnapshot as SnapshotModel
from cfast_trainer.numerical_operations import build_numerical_operations_test
from cfast_trainer.persistence import ResultsStore
from cfast_trainer.rapid_tracking import build_rapid_tracking_test
from cfast_trainer.runtime_defaults import RuntimeDefaultsStore
from cfast_trainer.sensory_motor_apparatus import (
    SensoryMotorApparatusConfig,
    build_sensory_motor_apparatus_test,
)
from cfast_trainer.training_modes import FatigueProbeConfig, maybe_build_fatigue_probe_drill
from cfast_trainer.vs_drills import VsDrillConfig, build_vs_mixed_tempo_drill


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


class _NestedClockEngine:
    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self._current_engine: _NestedClockEngine | None = None
        self._engine: _NestedClockEngine | None = None

    def current_engine(self) -> "_NestedClockEngine | None":
        return self._current_engine


class _PressedKeys:
    def __init__(self, pressed: set[int]) -> None:
        self._pressed = set(pressed)

    def __getitem__(self, key: int) -> int:
        return 1 if key in self._pressed else 0


def test_recursive_pause_helper_freezes_nested_engine_clocks_without_rewrapping() -> None:
    clock = _FakeClock()
    parent = _NestedClockEngine(clock)
    child = _NestedClockEngine(clock)
    inner = _NestedClockEngine(clock)
    parent._current_engine = child
    child._engine = inner
    inner._current_engine = parent

    returned = _set_engine_clock_paused(parent, True)

    assert returned is parent._clock
    assert isinstance(parent._clock, PausableClock)
    assert isinstance(child._clock, PausableClock)
    assert isinstance(inner._clock, PausableClock)
    assert parent._clock.is_paused() is True
    assert child._clock.is_paused() is True
    assert inner._clock.is_paused() is True

    clock_ids = (id(parent._clock), id(child._clock), id(inner._clock))
    paused_times = (parent._clock.now(), child._clock.now(), inner._clock.now())
    _set_engine_clock_paused(parent, True)
    assert (id(parent._clock), id(child._clock), id(inner._clock)) == clock_ids

    clock.advance(5.0)
    assert (parent._clock.now(), child._clock.now(), inner._clock.now()) == paused_times

    _set_engine_clock_paused(parent, False)
    clock.advance(2.0)
    assert parent._clock.now() == pytest.approx(paused_times[0] + 2.0)
    assert child._clock.now() == pytest.approx(paused_times[1] + 2.0)
    assert inner._clock.now() == pytest.approx(paused_times[2] + 2.0)


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


def _build_visual_search_fatigue_probe(clock: Clock):
    engine = maybe_build_fatigue_probe_drill(
        mode=AntDrillMode.FATIGUE_PROBE,
        title_base="Visual Search: Fatigue Probe",
        clock=clock,
        seed=37,
        difficulty=0.55,
        config=FatigueProbeConfig(
            baseline_duration_s=10.0,
            loader_duration_s=10.0,
            late_duration_s=10.0,
        ),
        build_segment=lambda segment_mode, segment_seed, segment_duration_s: build_vs_mixed_tempo_drill(
            clock=clock,
            seed=segment_seed,
            difficulty=0.55,
            mode=AntDrillMode(str(segment_mode)),
            config=VsDrillConfig(
                practice_questions=0,
                scored_duration_s=float(segment_duration_s),
            ),
        ),
    )
    assert engine is not None
    return engine


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


def test_pause_menu_freezes_nested_fatigue_probe_child_timer() -> None:
    pygame.init()
    surface = pygame.display.set_mode((960, 540))
    font = pygame.font.Font(None, 36)
    app = App(surface=surface, font=font)
    app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))
    clock = _FakeClock()
    engine = _build_visual_search_fatigue_probe(clock)
    screen = CognitiveTestScreen(
        app,
        engine_factory=lambda: engine,
        test_code="vs_mixed_tempo",
    )
    app.push(screen)
    try:
        engine.start_practice()
        engine.start_scored()
        engine.update()
        child = engine._current_engine
        assert child is not None
        before_total = engine.snapshot().time_remaining_s
        before_child = child.snapshot().time_remaining_s
        assert before_total is not None
        assert before_child is not None

        app.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "mod": 0, "unicode": ""})
        )
        assert app.shell_pause_overlay_active() is True
        clock.advance(3.0)
        app.render()

        assert engine.snapshot().time_remaining_s == pytest.approx(before_total)
        assert child.snapshot().time_remaining_s == pytest.approx(before_child)
        assert isinstance(child._clock, PausableClock)
        assert child._clock.is_paused() is True

        app.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "mod": 0, "unicode": ""})
        )
        assert app.shell_pause_overlay_active() is False
        clock.advance(1.0)
        app.render()

        assert engine.snapshot().time_remaining_s == pytest.approx(float(before_total) - 1.0)
        assert child.snapshot().time_remaining_s == pytest.approx(float(before_child) - 1.0)
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


def test_auditory_intro_loading_reuses_frozen_world_frame() -> None:
    app, screen, engines = _build_app_and_screen(
        phase=Phase.INSTRUCTIONS,
        title="Auditory Capacity",
        test_code="auditory_capacity",
    )
    try:
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
        ambient_index = next(
            idx for idx, (key, _label, _value) in enumerate(rows) if key == "auditory_ambient_volume"
        )
        primary_voice_index = next(
            idx for idx, (key, _label, _value) in enumerate(rows) if key == "auditory_primary_voice_volume"
        )
        decoy_voice_index = next(
            idx for idx, (key, _label, _value) in enumerate(rows) if key == "auditory_decoy_voice_volume"
        )
        filler_voice_index = next(
            idx for idx, (key, _label, _value) in enumerate(rows) if key == "auditory_filler_voice_volume"
        )
        beep_index = next(
            idx for idx, (key, _label, _value) in enumerate(rows) if key == "auditory_beep_volume"
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

        for target_index, stored_getter in (
            (ambient_index, runtime_defaults_store.stored_auditory_ambient_volume),
            (primary_voice_index, runtime_defaults_store.stored_auditory_primary_voice_volume),
            (decoy_voice_index, runtime_defaults_store.stored_auditory_decoy_voice_volume),
            (filler_voice_index, runtime_defaults_store.stored_auditory_filler_voice_volume),
            (beep_index, runtime_defaults_store.stored_auditory_beep_volume),
        ):
            while screen._pause_settings_selected != target_index:
                screen.handle_event(
                    pygame.event.Event(
                        pygame.KEYDOWN,
                        {"key": pygame.K_DOWN, "mod": 0, "unicode": ""},
                    )
                )
            before = screen._pause_settings_rows()[target_index][2]
            screen.handle_event(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RIGHT, "mod": 0, "unicode": ""})
            )
            after = screen._pause_settings_rows()[target_index][2]
            assert after != before
            assert stored_getter() is not None
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
