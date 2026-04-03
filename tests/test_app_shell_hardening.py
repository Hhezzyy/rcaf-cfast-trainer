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
    DisplayBootstrapResult,
    DisplayLifecycleState,
    MenuItem,
    MenuScreen,
    OpenGLFailureInfo,
    OpenGLFailureScreen,
    _present_display_transition_frame,
    run,
    run_headless_sim,
)
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


class _RecordingGlRenderer:
    def __init__(self, bucket: dict[str, list[object]]) -> None:
        self._bucket = bucket

    def resize(self, *, window_size: tuple[int, int]) -> None:
        self._bucket.setdefault("resize", []).append(tuple(window_size))

    def render_frame(self, *, ui_surface: pygame.Surface, scene) -> None:
        self._bucket.setdefault("ui", []).append(ui_surface.get_size())
        sample_point = (ui_surface.get_width() // 2, ui_surface.get_height() // 2)
        self._bucket.setdefault("frames", []).append(
            {
                "ui": ui_surface.get_size(),
                "scene_is_none": scene is None,
                "sample": tuple(ui_surface.get_at(sample_point)[:3]),
            }
        )


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


def test_app_render_shows_run_state_indicator_for_renderer_warning() -> None:
    pygame.init()
    surface = pygame.display.set_mode((960, 540))
    font = pygame.font.Font(None, 36)
    app = App(surface=surface, font=font, window_mode="windowed")
    root = MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True)
    screen = _FailingRunScreen(app, fail_in="none")
    app.push(root)
    app.push(screen)
    app.note_renderer_fallback()
    calls: list[tuple[int, int]] = []
    app._render_run_state_indicator = lambda target: calls.append(target.get_size())
    try:
        assert app.current_run_state().warning == "FALLBACK"
        app.render()
        assert calls == [surface.get_size()]
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


def test_present_renderer_failure_aborts_active_run_and_pushes_failure_screen() -> None:
    pygame.init()
    surface = pygame.display.set_mode((960, 540))
    font = pygame.font.Font(None, 36)
    app = App(surface=surface, font=font, window_mode="windowed")
    root = MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True)
    failing = _FailingRunScreen(app, fail_in="render")
    app.push(root)
    app.push(failing)
    try:
        app.present_renderer_failure(
            OpenGLFailureInfo(
                stage="render",
                summary="OpenGL renderer failed.",
                detail="The app could not continue while rendering the OpenGL frame.",
                requested=True,
                attempted=True,
            )
        )
        assert failing.reasons == ["renderer_failure_abort"]
        assert len(app._screens) == 2
        assert isinstance(app._screens[-1], OpenGLFailureScreen)
    finally:
        pygame.quit()


def test_present_display_transition_frame_scales_cached_frame_for_gl_renderer(
    monkeypatch,
) -> None:
    pygame.init()
    surface = pygame.display.set_mode((1440, 900))
    snapshot = pygame.Surface((960, 540), pygame.SRCALPHA)
    snapshot.fill((24, 48, 112))
    render_log: dict[str, list[object]] = {}
    renderer = _RecordingGlRenderer(render_log)
    flips: list[str] = []
    monkeypatch.setattr("pygame.display.flip", lambda: flips.append("flip"))
    try:
        assert (
            _present_display_transition_frame(
                display_surface=surface,
                gl_renderer=renderer,
                transition_frame=snapshot,
            )
            is True
        )
        assert render_log["resize"][-1] == (1440, 900)
        frame = render_log["frames"][-1]
        assert frame == {
            "ui": (1440, 900),
            "scene_is_none": True,
            "sample": (24, 48, 112),
        }
        assert flips == ["flip"]
    finally:
        pygame.quit()


def test_run_startup_renderer_failure_shows_fatal_screen_and_quits(monkeypatch) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")

    def fake_initialize_display_surfaces(*, window_size, window_flags, video_driver, want_gl):
        _ = (video_driver, want_gl)
        display_surface = pygame.display.set_mode(window_size, window_flags)
        return DisplayBootstrapResult(
            display_surface=display_surface,
            app_surface=display_surface,
            gl_renderer=None,
            active_window_flags=window_flags,
            gl_requested=True,
            gl_attempted=True,
            gl_failure=OpenGLFailureInfo(
                stage="renderer_init",
                summary="Renderer failed.",
                detail="The app could not continue while initializing the ModernGL renderer.",
                requested=True,
                attempted=True,
            ),
        )

    monkeypatch.setattr("cfast_trainer.app._initialize_display_surfaces", fake_initialize_display_surfaces)

    def inject(frame: int) -> None:
        if frame == 0:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))

    assert run(max_frames=4, event_injector=inject) == 1


def test_run_rebootstraps_stale_fullscreen_drawable_and_keeps_ui_surface_synced(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")

    render_log: dict[str, list[tuple[int, int]]] = {}
    lifecycle_calls = {"count": 0}
    decisions: list[tuple[str, tuple[int, int]]] = []

    def fake_initialize_display_surfaces(*, window_size, window_flags, video_driver, want_gl):
        _ = (window_flags, video_driver, want_gl)
        display_surface = pygame.display.set_mode(window_size)
        return DisplayBootstrapResult(
            display_surface=display_surface,
            app_surface=pygame.Surface(display_surface.get_size(), pygame.SRCALPHA),
            gl_renderer=_RecordingGlRenderer(render_log),
            active_window_flags=pygame.RESIZABLE | pygame.OPENGL | pygame.DOUBLEBUF,
            gl_requested=True,
            gl_attempted=True,
        )

    def fake_read_display_lifecycle_state(*, display_surface, active_window_flags, window_mode):
        lifecycle_calls["count"] += 1
        if lifecycle_calls["count"] == 1:
            return DisplayLifecycleState(
                window_size=(1440, 900),
                surface_size=display_surface.get_size(),
                desktop_size=(1440, 900),
                active_window_flags=active_window_flags,
                window_mode=window_mode,
            )
        return DisplayLifecycleState(
            window_size=(1440, 900),
            surface_size=(1440, 900),
            desktop_size=(1440, 900),
            active_window_flags=active_window_flags,
            window_mode=window_mode,
        )

    def fake_rebootstrap_display_for_transition(*, decision, video_driver, want_gl):
        _ = (video_driver, want_gl)
        decisions.append((decision.window_mode, decision.window_size))
        display_surface = pygame.display.set_mode(decision.window_size)
        return DisplayBootstrapResult(
            display_surface=display_surface,
            app_surface=pygame.Surface(display_surface.get_size(), pygame.SRCALPHA),
            gl_renderer=_RecordingGlRenderer(render_log),
            active_window_flags=pygame.FULLSCREEN | pygame.OPENGL | pygame.DOUBLEBUF,
            gl_requested=True,
            gl_attempted=True,
        )

    monkeypatch.setattr("cfast_trainer.app._initialize_display_surfaces", fake_initialize_display_surfaces)
    monkeypatch.setattr(
        "cfast_trainer.app._read_display_lifecycle_state",
        fake_read_display_lifecycle_state,
    )
    monkeypatch.setattr(
        "cfast_trainer.app._rebootstrap_display_for_transition",
        fake_rebootstrap_display_for_transition,
    )

    summary: dict[str, object] = {}
    assert run(max_frames=1, summary_sink=summary) == 0

    assert decisions == [("fullscreen", (1440, 900))]
    assert render_log["resize"][-1] == (1440, 900)
    assert render_log["ui"][-1] == (1440, 900)
    assert summary["display_mode"] == "FULLSCREEN"


def test_run_presents_cached_frame_before_runtime_fullscreen_rebootstrap(monkeypatch) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")

    render_log: dict[str, list[object]] = {}
    lifecycle_calls = {"count": 0}
    decisions: list[tuple[str, tuple[int, int]]] = []
    captured_sizes: list[tuple[int, int]] = []
    presented: list[dict[str, object]] = []

    def fake_initialize_display_surfaces(*, window_size, window_flags, video_driver, want_gl):
        _ = (window_flags, video_driver, want_gl)
        display_surface = pygame.display.set_mode(window_size)
        return DisplayBootstrapResult(
            display_surface=display_surface,
            app_surface=pygame.Surface(display_surface.get_size(), pygame.SRCALPHA),
            gl_renderer=_RecordingGlRenderer(render_log),
            active_window_flags=pygame.RESIZABLE | pygame.OPENGL | pygame.DOUBLEBUF,
            gl_requested=True,
            gl_attempted=True,
        )

    def fake_read_display_lifecycle_state(*, display_surface, active_window_flags, window_mode):
        lifecycle_calls["count"] += 1
        if lifecycle_calls["count"] == 1:
            return DisplayLifecycleState(
                window_size=display_surface.get_size(),
                surface_size=display_surface.get_size(),
                desktop_size=(1440, 900),
                active_window_flags=active_window_flags,
                window_mode=window_mode,
            )
        if lifecycle_calls["count"] == 2:
            return DisplayLifecycleState(
                window_size=(1440, 900),
                surface_size=display_surface.get_size(),
                desktop_size=(1440, 900),
                active_window_flags=active_window_flags,
                window_mode=window_mode,
            )
        return DisplayLifecycleState(
            window_size=(1440, 900),
            surface_size=(1440, 900),
            desktop_size=(1440, 900),
            active_window_flags=active_window_flags,
            window_mode=window_mode,
        )

    def fake_rebootstrap_display_for_transition(*, decision, video_driver, want_gl):
        _ = (video_driver, want_gl)
        decisions.append((decision.window_mode, decision.window_size))
        display_surface = pygame.display.set_mode(decision.window_size)
        return DisplayBootstrapResult(
            display_surface=display_surface,
            app_surface=pygame.Surface(display_surface.get_size(), pygame.SRCALPHA),
            gl_renderer=_RecordingGlRenderer(render_log),
            active_window_flags=pygame.FULLSCREEN | pygame.OPENGL | pygame.DOUBLEBUF,
            gl_requested=True,
            gl_attempted=True,
        )

    def fake_capture_display_transition_frame(display_surface):
        captured_sizes.append(display_surface.get_size())
        frame = pygame.Surface(display_surface.get_size(), pygame.SRCALPHA)
        frame.fill((18, 36, 90))
        return frame

    def fake_present_display_transition_frame(*, display_surface, gl_renderer, transition_frame):
        presented.append(
            {
                "display": display_surface.get_size(),
                "has_gl": gl_renderer is not None,
                "frame": None if transition_frame is None else transition_frame.get_size(),
                "sample": None
                if transition_frame is None
                else tuple(transition_frame.get_at((0, 0))[:3]),
            }
        )
        return True

    monkeypatch.setattr("cfast_trainer.app._initialize_display_surfaces", fake_initialize_display_surfaces)
    monkeypatch.setattr(
        "cfast_trainer.app._read_display_lifecycle_state",
        fake_read_display_lifecycle_state,
    )
    monkeypatch.setattr(
        "cfast_trainer.app._rebootstrap_display_for_transition",
        fake_rebootstrap_display_for_transition,
    )
    monkeypatch.setattr(
        "cfast_trainer.app._capture_display_transition_frame",
        fake_capture_display_transition_frame,
    )
    monkeypatch.setattr(
        "cfast_trainer.app._present_display_transition_frame",
        fake_present_display_transition_frame,
    )

    summary: dict[str, object] = {}
    assert run(max_frames=2, summary_sink=summary) == 0

    assert decisions == [("fullscreen", (1440, 900))]
    assert captured_sizes == [(960, 540)]
    assert presented == [
        {
            "display": (1440, 900),
            "has_gl": True,
            "frame": (960, 540),
            "sample": (18, 36, 90),
        }
    ]
    assert summary["display_mode"] == "FULLSCREEN"


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
