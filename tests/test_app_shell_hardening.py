from __future__ import annotations

import json
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from cfast_trainer.__main__ import main as cli_main
from cfast_trainer.app import App, CognitiveTestScreen, MenuItem, MenuScreen, run_headless_sim
from cfast_trainer.cognitive_core import Phase
from cfast_trainer.cognitive_core import TestSnapshot as SnapshotModel


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


def test_app_emergency_hotkey_aborts_active_run_and_returns_to_root() -> None:
    app, screen = _build_app_and_screen()
    try:
        app.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_F12, "mod": 0, "unicode": ""})
        )
        assert len(app._screens) == 1
        assert screen._activity_finalized is True
        assert app.current_run_state().shell_state == "MENU"
    finally:
        pygame.quit()


def test_input_failure_recovers_to_root_menu() -> None:
    pygame.init()
    surface = pygame.display.set_mode((960, 540))
    font = pygame.font.Font(None, 36)
    app = App(surface=surface, font=font, window_mode="windowed")
    root = MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True)
    failing = _FailingRunScreen(app, fail_in="input")
    app.push(root)
    app.push(failing)
    try:
        app.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_a, "mod": 0, "unicode": "a"})
        )
        assert len(app._screens) == 1
        assert failing.reasons == ["input_failure_abort"]
        assert app.used_failure_recovery() is True
        assert app.current_run_state().shell_state == "MENU"
    finally:
        pygame.quit()


def test_render_failure_recovers_to_root_menu() -> None:
    pygame.init()
    surface = pygame.display.set_mode((960, 540))
    font = pygame.font.Font(None, 36)
    app = App(surface=surface, font=font, window_mode="windowed")
    root = MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True)
    failing = _FailingRunScreen(app, fail_in="render")
    app.push(root)
    app.push(failing)
    try:
        app.render()
        assert len(app._screens) == 1
        assert failing.reasons == ["runtime_failure_abort"]
        assert app.used_failure_recovery() is True
        assert app.current_run_state().shell_state == "MENU"
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
