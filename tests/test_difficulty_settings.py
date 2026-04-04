from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
import pytest

from cfast_trainer.app import (
    AnalogBinding,
    App,
    DigitalBinding,
    DifficultySettingsScreen,
    DifficultySettingsStore,
    InputProfilesStore,
    JoystickBindingsScreen,
    MenuItem,
    MenuScreen,
    OpenGLFailureInfo,
    OpenGLFailureScreen,
    TestSeedSettingsScreen,
    TestSeedSettingsStore,
)
from cfast_trainer.runtime_defaults import RuntimeDefaultsStore


def test_difficulty_settings_store_persists_global_and_per_test_levels(tmp_path) -> None:
    path = tmp_path / "difficulty-settings.json"
    store = DifficultySettingsStore(path)

    assert store.global_override_enabled() is False
    assert store.global_level() == 5
    assert store.review_mode_enabled() is False
    assert store.test_level("rapid_tracking") == 5

    store.set_test_level(test_code="rapid_tracking", level=8)
    store.set_global_level(9)
    store.set_global_override_enabled(True)
    store.set_review_mode_enabled(True)

    reloaded = DifficultySettingsStore(path)
    assert reloaded.global_override_enabled() is True
    assert reloaded.global_level() == 9
    assert reloaded.review_mode_enabled() is True
    assert reloaded.test_level("rapid_tracking") == 8
    assert reloaded.effective_level("rapid_tracking") == 9
    assert reloaded.effective_ratio("rapid_tracking") == pytest.approx((9 - 1) / 9.0)
    assert reloaded.intro_mode_label("rapid_tracking") == "Global Override"

    reloaded.set_global_override_enabled(False)
    final = DifficultySettingsStore(path)
    assert final.global_override_enabled() is False
    assert final.effective_level("rapid_tracking") == 8
    assert final.intro_mode_label("rapid_tracking") == "This Test"


def test_difficulty_settings_screen_updates_global_and_per_test_values(tmp_path) -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        store = DifficultySettingsStore(tmp_path / "difficulty-settings.json")
        app = App(surface=surface, font=font, difficulty_settings_store=store)
        root = MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True)
        app.push(root)
        screen = DifficultySettingsScreen(app)
        app.push(screen)

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RIGHT, "mod": 0, "unicode": ""})
        )
        assert store.global_override_enabled() is True

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "mod": 0, "unicode": ""})
        )
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RIGHT, "mod": 0, "unicode": ""})
        )
        assert store.global_level() == 6

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "mod": 0, "unicode": ""})
        )
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RIGHT, "mod": 0, "unicode": ""})
        )
        assert store.review_mode_enabled() is True

        rows = screen._rows()
        rapid_tracking_index = next(
            idx for idx, (key, _label, _value) in enumerate(rows) if key == "test:rapid_tracking"
        )
        screen._selected = rapid_tracking_index
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RIGHT, "mod": 0, "unicode": ""})
        )

        assert store.test_level("rapid_tracking") == 6
    finally:
        pygame.quit()


def test_difficulty_settings_screen_lists_spatial_trace_and_vigilance_drill_and_workout_codes(
    tmp_path,
) -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        store = DifficultySettingsStore(tmp_path / "difficulty-settings.json")
        app = App(surface=surface, font=font, difficulty_settings_store=store)
        root = MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True)
        app.push(root)
        screen = DifficultySettingsScreen(app)
        app.push(screen)

        row_keys = {key for key, _label, _value in screen._rows()}
        assert "test:si_landmark_anchor" in row_keys
        assert "test:si_pressure_run" in row_keys
        assert "test:spatial_integration_workout" in row_keys
        assert "test:tt1_lateral_anchor" in row_keys
        assert "test:trace_pressure_run" in row_keys
        assert "test:trace_test_1_workout" in row_keys
        assert "test:trace_test_2_workout" in row_keys
        assert "test:vig_entry_anchor" in row_keys
        assert "test:vig_pressure_run" in row_keys
        assert "test:vigilance_workout" in row_keys
    finally:
        pygame.quit()


def test_test_seed_settings_store_persists_override_and_seed_value(tmp_path) -> None:
    path = tmp_path / "test-seed-settings.json"
    store = TestSeedSettingsStore(path)

    assert store.rapid_tracking_seed_override_enabled() is False
    assert store.rapid_tracking_seed_value() == 551

    store.set_rapid_tracking_seed_override_enabled(True)
    store.set_rapid_tracking_seed_value(424242)

    reloaded = TestSeedSettingsStore(path)
    assert reloaded.rapid_tracking_seed_override_enabled() is True
    assert reloaded.rapid_tracking_seed_value() == 424242


def test_test_seed_settings_screen_toggles_override_and_saves_typed_seed(tmp_path) -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        difficulty_store = DifficultySettingsStore(tmp_path / "difficulty-settings.json")
        seed_store = TestSeedSettingsStore(tmp_path / "test-seed-settings.json")
        app = App(
            surface=surface,
            font=font,
            difficulty_settings_store=difficulty_store,
            test_seed_settings_store=seed_store,
        )
        root = MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True)
        app.push(root)
        screen = TestSeedSettingsScreen(app)
        app.push(screen)

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": 0, "unicode": ""})
        )
        assert seed_store.rapid_tracking_seed_override_enabled() is True

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "mod": 0, "unicode": ""})
        )
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": 0, "unicode": ""})
        )
        assert screen._editing_seed is True

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_BACKSPACE, "mod": 0, "unicode": ""})
        )
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_BACKSPACE, "mod": 0, "unicode": ""})
        )
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_BACKSPACE, "mod": 0, "unicode": ""})
        )
        for ch in "12345":
            screen.handle_event(
                pygame.event.Event(
                    pygame.KEYDOWN,
                    {"key": ord(ch), "mod": 0, "unicode": ch},
                )
            )
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": 0, "unicode": ""})
        )

        assert seed_store.rapid_tracking_seed_value() == 12345
        assert screen._editing_seed is False
    finally:
        pygame.quit()


def test_app_resolves_rapid_tracking_launch_seed_from_override(tmp_path, monkeypatch) -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        seed_store = TestSeedSettingsStore(tmp_path / "test-seed-settings.json")
        app = App(surface=surface, font=font, test_seed_settings_store=seed_store)

        monkeypatch.setattr("cfast_trainer.app._new_seed", lambda: 987654)
        assert app.resolved_rapid_tracking_launch_seed() == 987654

        seed_store.set_rapid_tracking_seed_override_enabled(True)
        seed_store.set_rapid_tracking_seed_value(246810)
        assert app.resolved_rapid_tracking_launch_seed() == 246810
    finally:
        pygame.quit()


def test_input_profiles_store_persists_axis_and_action_bindings(tmp_path) -> None:
    path = tmp_path / "input-profiles.json"
    store = InputProfilesStore(path)
    profile_id = store.active_profile().profile_id

    store.set_axis_role_binding(
        profile_id=profile_id,
        role="primary_horizontal",
        binding=AnalogBinding(device_key="test-stick|abc", axis_index=2),
    )
    store.set_action_binding(
        profile_id=profile_id,
        action="rapid_tracking_capture",
        slot_index=0,
        binding=DigitalBinding(kind="button", device_key="test-stick|abc", control_index=5),
    )
    store.set_action_binding(
        profile_id=profile_id,
        action="menu_select",
        slot_index=1,
        binding=DigitalBinding(
            kind="hat",
            device_key="test-stick|abc",
            control_index=0,
            direction="up",
        ),
    )

    reloaded = InputProfilesStore(path)
    reloaded_profile_id = reloaded.active_profile().profile_id
    axis_binding = reloaded.get_axis_role_binding(
        profile_id=reloaded_profile_id,
        role="primary_horizontal",
    )
    assert axis_binding is not None
    assert axis_binding.device_key == "test-stick|abc"
    assert axis_binding.axis_index == 2

    action_slots = reloaded.get_action_binding_slots(
        profile_id=reloaded_profile_id,
        action="rapid_tracking_capture",
    )
    assert action_slots[0] is not None
    assert action_slots[0].kind == "button"
    assert action_slots[0].control_index == 5

    menu_slots = reloaded.get_action_binding_slots(
        profile_id=reloaded_profile_id,
        action="menu_select",
    )
    assert menu_slots[0] is None
    assert menu_slots[1] is not None
    assert menu_slots[1].kind == "hat"
    assert menu_slots[1].direction == "up"


def test_joystick_bindings_screen_captures_axis_and_button(tmp_path, monkeypatch) -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        store = InputProfilesStore(tmp_path / "input-profiles.json")
        app = App(surface=surface, font=font, input_profiles_store=store)
        app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))
        screen = JoystickBindingsScreen(app, profiles=store)
        app.push(screen)

        class _FakeJoystick:
            def __init__(self) -> None:
                self.axes = [0.0, 0.0, 0.0, 0.0]
                self.buttons = [0] * 8

            def get_name(self) -> str:
                return "bind stick"

            def get_guid(self) -> str:
                return "guid-1"

            def get_numaxes(self) -> int:
                return len(self.axes)

            def get_axis(self, idx: int) -> float:
                return float(self.axes[idx])

            def get_numbuttons(self) -> int:
                return len(self.buttons)

            def get_button(self, idx: int) -> int:
                return int(self.buttons[idx])

            def get_numhats(self) -> int:
                return 1

            def get_hat(self, idx: int) -> tuple[int, int]:
                return (0, 0)

        fake = _FakeJoystick()
        monkeypatch.setattr("cfast_trainer.app._iter_connected_joysticks", lambda: [fake])

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": 0, "unicode": ""})
        )
        fake.axes[2] = 0.82
        screen.render(surface)

        axis_binding = store.get_axis_role_binding(
            profile_id=store.active_profile().profile_id,
            role="primary_horizontal",
        )
        assert axis_binding is not None
        assert axis_binding.device_key.startswith("bind stick|")
        assert axis_binding.axis_index == 2

        capture_index = next(
            idx
            for idx, row in enumerate(screen._rows())
            if row.row_type == "digital" and row.key == "rapid_tracking_capture" and row.slot_index == 0
        )
        screen._selected = capture_index
        fake.axes[2] = 0.0
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": 0, "unicode": ""})
        )
        fake.buttons[4] = 1
        screen.render(surface)

        slots = store.get_action_binding_slots(
            profile_id=store.active_profile().profile_id,
            action="rapid_tracking_capture",
        )
        assert slots[0] is not None
        assert slots[0].kind == "button"
        assert slots[0].control_index == 4
    finally:
        pygame.quit()


def test_runtime_defaults_store_persists_display_and_auditory_values(tmp_path) -> None:
    path = tmp_path / "runtime-defaults.json"
    store = RuntimeDefaultsStore(path)

    assert store.stored_window_mode() is None
    assert store.stored_use_opengl() is None
    assert store.stored_auditory_noise_level() is None
    assert store.stored_auditory_distortion_level() is None
    assert store.stored_auditory_noise_source() is None

    store.set_window_mode("borderless")
    store.set_use_opengl(True)
    store.set_auditory_noise_level(0.3)
    store.set_auditory_distortion_level(0.6)
    store.set_auditory_noise_source("pink")

    reloaded = RuntimeDefaultsStore(path)
    assert reloaded.stored_window_mode() == "borderless"
    assert reloaded.stored_use_opengl() is True
    assert reloaded.stored_auditory_noise_level() == pytest.approx(0.3)
    assert reloaded.stored_auditory_distortion_level() == pytest.approx(0.6)
    assert reloaded.stored_auditory_noise_source() == "pink"


def test_runtime_defaults_store_honors_env_overrides_and_default_path(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "runtime-defaults.json"
    monkeypatch.setenv("CFAST_RUNTIME_DEFAULTS_PATH", str(path))

    store = RuntimeDefaultsStore(RuntimeDefaultsStore.default_path())
    store.set_auditory_noise_level(0.1)
    store.set_auditory_distortion_level(0.2)
    store.set_auditory_noise_source("brown")

    monkeypatch.setenv("CFAST_AUDITORY_NOISE_LEVEL", "0.8")
    monkeypatch.setenv("CFAST_AUDITORY_DISTORTION_LEVEL", "0.9")
    monkeypatch.setenv("CFAST_AUDITORY_NOISE_SOURCE", "white")

    reloaded = RuntimeDefaultsStore(RuntimeDefaultsStore.default_path())
    assert RuntimeDefaultsStore.default_path() == path
    assert reloaded.resolved_auditory_noise_level() == pytest.approx(0.8)
    assert reloaded.resolved_auditory_distortion_level() == pytest.approx(0.9)
    assert reloaded.resolved_auditory_noise_source() == "white"


def test_app_applies_and_saves_runtime_defaults(tmp_path) -> None:
    class _AudioEngine:
        def __init__(self) -> None:
            self.noise_level = None
            self.distortion_level = None
            self.noise_source = None

        def set_audio_overrides(
            self,
            *,
            noise_level: float | None = None,
            distortion_level: float | None = None,
            noise_source: str | None = None,
        ) -> None:
            self.noise_level = noise_level
            self.distortion_level = distortion_level
            self.noise_source = noise_source

    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        runtime_defaults = RuntimeDefaultsStore(tmp_path / "runtime-defaults.json")
        runtime_defaults.set_auditory_noise_level(0.4)
        runtime_defaults.set_auditory_distortion_level(0.1)
        runtime_defaults.set_auditory_noise_source("pink")

        app = App(surface=surface, font=font, runtime_defaults_store=runtime_defaults)
        engine = _AudioEngine()
        app.apply_runtime_defaults_to_engine(engine)

        assert engine.noise_level == pytest.approx(0.4)
        assert engine.distortion_level == pytest.approx(0.1)
        assert engine.noise_source == "pink"

        app.save_auditory_runtime_defaults(
            noise_level=0.6,
            distortion_level=0.3,
            noise_source="brown",
        )

        reloaded = RuntimeDefaultsStore(tmp_path / "runtime-defaults.json")
        assert reloaded.stored_auditory_noise_level() == pytest.approx(0.6)
        assert reloaded.stored_auditory_distortion_level() == pytest.approx(0.3)
        assert reloaded.stored_auditory_noise_source() == "brown"
    finally:
        pygame.quit()


def test_app_uses_stored_opengl_preference(tmp_path, monkeypatch) -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        runtime_defaults = RuntimeDefaultsStore(tmp_path / "runtime-defaults.json")
        runtime_defaults.set_use_opengl(False)
        app = App(surface=surface, font=font, runtime_defaults_store=runtime_defaults)

        assert runtime_defaults.stored_use_opengl() is False
        assert app.stored_use_opengl() is False
        app.set_stored_use_opengl(True)
        assert app.stored_use_opengl() is True
        assert runtime_defaults.stored_use_opengl() is True

        monkeypatch.setenv("CFAST_USE_OPENGL", "0")
        assert app.use_opengl_env_override() == "0"
        assert app.use_opengl_env_forces_enabled() is False

        monkeypatch.setenv("CFAST_USE_OPENGL", "1")
        assert app.use_opengl_env_override() == "1"
        assert app.use_opengl_env_forces_enabled() is True
    finally:
        pygame.quit()


def test_renderer_failure_screen_only_exposes_quit_action() -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        app = App(surface=surface, font=font)
        root = MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True)
        app.push(root)
        screen = OpenGLFailureScreen(
            app,
            failure=OpenGLFailureInfo(
                stage="render",
                summary="Renderer failed.",
                detail="The app could not continue while rendering the frame.",
                requested=True,
                attempted=True,
            ),
        )
        app.push(screen)

        rows = screen._rows()
        assert rows == [("quit", "Quit", "Exit the app.", True)]

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": 0, "unicode": ""})
        )

        assert app.running is False
        assert app.exit_reason == "renderer_failure_quit"
    finally:
        pygame.quit()


def test_bound_menu_actions_navigate_and_select_menu_items(tmp_path, monkeypatch) -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        store = InputProfilesStore(tmp_path / "input-profiles.json")
        profile_id = store.active_profile().profile_id
        store.set_action_binding(
            profile_id=profile_id,
            action="menu_down",
            slot_index=0,
            binding=DigitalBinding(kind="button", device_key="menu stick|guid-1", control_index=2),
        )
        store.set_action_binding(
            profile_id=profile_id,
            action="menu_select",
            slot_index=0,
            binding=DigitalBinding(kind="button", device_key="menu stick|guid-1", control_index=3),
        )

        app = App(surface=surface, font=font, input_profiles_store=store)
        called: list[str] = []
        screen = MenuScreen(
            app,
            "Main Menu",
            [
                MenuItem("First", lambda: called.append("first")),
                MenuItem("Second", lambda: called.append("second")),
            ],
            is_root=True,
        )
        app.push(screen)

        class _FakeJoystick:
            def __init__(self) -> None:
                self.buttons = [0] * 8

            def get_name(self) -> str:
                return "menu stick"

            def get_guid(self) -> str:
                return "guid-1"

            def get_numaxes(self) -> int:
                return 0

            def get_numbuttons(self) -> int:
                return len(self.buttons)

            def get_button(self, idx: int) -> int:
                return int(self.buttons[idx])

            def get_numhats(self) -> int:
                return 0

        fake = _FakeJoystick()
        monkeypatch.setattr("cfast_trainer.app._iter_connected_joysticks", lambda: [fake])

        fake.buttons[2] = 1
        app.render()
        fake.buttons[2] = 0
        app.render()
        assert screen._selected == 1

        fake.buttons[3] = 1
        app.render()
        assert called == ["second"]
    finally:
        pygame.quit()
