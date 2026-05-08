from __future__ import annotations

import json
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from cfast_trainer.__main__ import main as cli_main
from cfast_trainer.app import (
    App,
    CognitiveTestScreen,
    MenuItem,
    MenuScreen,
    run,
    run_headless_sim,
)
from cfast_trainer.cognitive_core import Phase
from cfast_trainer.cognitive_core import TestSnapshot as SnapshotModel
from cfast_trainer.rapid_tracking import build_rapid_tracking_test


class _FakeEngine:
    def __init__(self, *, phase: Phase = Phase.PRACTICE, title: str = "Fake Test") -> None:
        self._phase = phase
        self._title = title
        self._difficulty = 0.5

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
        return True

    def start_practice(self) -> None:
        if self._phase is Phase.INSTRUCTIONS:
            self._phase = Phase.PRACTICE

    def start_scored(self) -> None:
        if self._phase is Phase.PRACTICE_DONE:
            self._phase = Phase.SCORED

    def submit_answer(self, _raw: str) -> bool:
        return True

    def update(self) -> None:
        return None


class _FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t


class _FailingRunScreen:
    def __init__(self, app: App, *, fail_in: str) -> None:
        self._app = app
        self._fail_in = str(fail_in)
        self._paused = False
        self.reasons: list[str] = []

    def handle_event(self, _event: pygame.event.Event) -> None:
        if self._fail_in == "input":
            raise RuntimeError("input boom")

    def render(self, _surface: pygame.Surface) -> None:
        if self._fail_in == "render":
            raise RuntimeError("render boom")

    def shell_activity_active(self) -> bool:
        return True

    def shell_pause_available(self) -> bool:
        return True

    def shell_pause_set_active(self, active: bool) -> None:
        self._paused = bool(active)

    def shell_pause_restart(self) -> None:
        self._paused = False

    def shell_pause_main_menu(self) -> None:
        self._paused = False
        self._app.pop_to_root()

    def shell_emergency_exit(self, reason: str) -> None:
        self._paused = False
        self.reasons.append(str(reason))
        self._app.pop_to_root()

    def shell_activity_label(self) -> str:
        return "Failing Run"


def _build_app_and_screen() -> tuple[App, CognitiveTestScreen]:
    pygame.init()
    surface = pygame.display.set_mode((960, 540))
    font = pygame.font.Font(None, 36)
    app = App(surface=surface, font=font, window_mode="windowed")
    root = MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True)
    app.push(root)
    screen = CognitiveTestScreen(
        app,
        engine_factory=lambda: _FakeEngine(phase=Phase.PRACTICE, title="Numerical Operations"),
        test_code="numerical_operations",
    )
    app.push(screen)
    return app, screen


def _build_intro_app_and_screen() -> tuple[App, CognitiveTestScreen, _FakeEngine]:
    pygame.init()
    surface = pygame.display.set_mode((960, 540))
    font = pygame.font.Font(None, 36)
    app = App(surface=surface, font=font, window_mode="windowed")
    root = MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True)
    app.push(root)
    engine = _FakeEngine(phase=Phase.INSTRUCTIONS, title="Numerical Operations")
    screen = CognitiveTestScreen(
        app,
        engine_factory=lambda: engine,
        test_code="numerical_operations",
    )
    screen._intro_loading_ready = True
    app.push(screen)
    return app, screen, engine


def test_app_escape_opens_shell_pause_and_resume_updates_run_state() -> None:
    app, _screen = _build_app_and_screen()
    try:
        app.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "mod": 0, "unicode": ""})
        )
        assert app.shell_pause_overlay_active() is True
        assert app.current_run_state().shell_state == "PAUSED"

        app.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": 0, "unicode": ""})
        )
        assert app.shell_pause_overlay_active() is False
        assert app.current_run_state().shell_state == "RUNNING"
    finally:
        pygame.quit()


def test_keypad_period_acts_as_enter_for_runtime_submit() -> None:
    app, _screen, engine = _build_intro_app_and_screen()
    try:
        app.handle_event(
            pygame.event.Event(
                pygame.KEYDOWN,
                {"key": pygame.K_KP_PERIOD, "mod": 0, "unicode": "."},
            )
        )
        assert engine.snapshot().phase is Phase.PRACTICE
    finally:
        pygame.quit()


def test_keypad_delete_scancode_acts_as_enter_but_regular_delete_does_not() -> None:
    app, _screen, engine = _build_intro_app_and_screen()
    try:
        app.handle_event(
            pygame.event.Event(
                pygame.KEYDOWN,
                {"key": pygame.K_DELETE, "mod": 0, "unicode": ""},
            )
        )
        assert engine.snapshot().phase is Phase.INSTRUCTIONS

        app.handle_event(
            pygame.event.Event(
                pygame.KEYDOWN,
                {
                    "key": pygame.K_DELETE,
                    "scancode": pygame.KSCAN_KP_PERIOD,
                    "mod": 0,
                    "unicode": "",
                },
            )
        )
        assert engine.snapshot().phase is Phase.PRACTICE
    finally:
        pygame.quit()


def test_keypad_enter_is_suppressed_for_runtime_submit() -> None:
    app, _screen, engine = _build_intro_app_and_screen()
    try:
        app.handle_event(
            pygame.event.Event(
                pygame.KEYDOWN,
                {"key": pygame.K_KP_ENTER, "mod": 0, "unicode": ""},
            )
        )
        assert engine.snapshot().phase is Phase.INSTRUCTIONS

        app.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": 0, "unicode": ""})
        )
        assert engine.snapshot().phase is Phase.PRACTICE
    finally:
        pygame.quit()


def test_keypad_enter_does_not_activate_menu_selection_but_keypad_period_does() -> None:
    pygame.init()
    surface = pygame.display.set_mode((960, 540))
    font = pygame.font.Font(None, 36)
    app = App(surface=surface, font=font, window_mode="windowed")
    activations: list[str] = []
    root = MenuScreen(
        app,
        "Main Menu",
        [MenuItem("Start", lambda: activations.append("start"))],
        is_root=True,
    )
    app.push(root)
    try:
        root.handle_event(
            pygame.event.Event(
                pygame.KEYDOWN,
                {"key": pygame.K_KP_ENTER, "mod": 0, "unicode": ""},
            )
        )
        assert activations == []

        app.handle_event(
            pygame.event.Event(
                pygame.KEYDOWN,
                {"key": pygame.K_KP_PERIOD, "mod": 0, "unicode": "."},
            )
        )
        assert activations == ["start"]
    finally:
        pygame.quit()


def test_app_render_hides_run_state_indicator_during_normal_use() -> None:
    pygame.init()
    surface = pygame.display.set_mode((960, 540))
    font = pygame.font.Font(None, 36)
    app = App(surface=surface, font=font, window_mode="windowed")
    root = MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True)
    screen = _FailingRunScreen(app, fail_in="none")
    app.push(root)
    app.push(screen)
    calls: list[tuple[int, int]] = []
    app._render_run_state_indicator = lambda target: calls.append(target.get_size())
    try:
        app.render()
        assert calls == []
    finally:
        pygame.quit()


def test_app_render_shows_run_state_indicator_in_dev_mode() -> None:
    pygame.init()
    surface = pygame.display.set_mode((960, 540))
    font = pygame.font.Font(None, 36)
    app = App(surface=surface, font=font, window_mode="windowed")
    app._dev_tools_enabled = True
    root = MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True)
    screen = _FailingRunScreen(app, fail_in="none")
    app.push(root)
    app.push(screen)
    calls: list[tuple[int, int]] = []
    app._render_run_state_indicator = lambda target: calls.append(target.get_size())
    try:
        app.render()
        assert calls == [surface.get_size()]
    finally:
        pygame.quit()


def test_run_headless_sim_drill_pause_cycle_returns_menu_summary() -> None:
    result = run_headless_sim("drill_pause_cycle")
    assert result.success is True
    assert result.final_shell_state == "MENU"
    assert result.display_mode == "WINDOWED"
    assert result.activity_label is None


def test_run_headless_sim_benchmark_intro_returns_to_menu_summary() -> None:
    result = run_headless_sim("benchmark_intro")
    assert result.success is True
    assert result.final_shell_state == "MENU"
    assert result.activity_label is None


def test_headless_cli_prints_json_summary(capsys) -> None:
    code = cli_main(["--headless-sim", "boot"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 0
    assert payload["scenario"] == "boot"
    assert payload["display_mode"] == "WINDOWED"
