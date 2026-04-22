from __future__ import annotations

import os
import sys
from dataclasses import dataclass, replace
from importlib.machinery import ModuleSpec
from types import ModuleType
from typing import cast

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

if "moderngl" not in sys.modules:
    moderngl_stub = ModuleType("moderngl")
    moderngl_stub.__spec__ = ModuleSpec("moderngl", loader=None)
    sys.modules["moderngl"] = moderngl_stub

from cfast_trainer.ant_drills import AntDrillMode
from cfast_trainer.ant_workouts import AntWorkoutBlockPlan, AntWorkoutPlan, AntWorkoutSession
from cfast_trainer.app import App, AntWorkoutScreen, CognitiveTestScreen, MenuItem, MenuScreen
from cfast_trainer.cognitive_core import Phase
from cfast_trainer.cognitive_core import TestSnapshot as SnapshotModel
from cfast_trainer.cognitive_updating import (
    CognitiveUpdatingGenerator,
    CognitiveUpdatingPayload,
    CognitiveUpdatingTrainingProfile,
)
from cfast_trainer.cu_drills import build_cu_mixed_tempo_drill


@dataclass
class _FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class _FakeCognitiveUpdatingEngine:
    def __init__(
        self,
        payload: CognitiveUpdatingPayload,
        *,
        title: str,
        clock: _FakeClock | None = None,
    ) -> None:
        self._payload = payload
        self._title = title
        self.clock = clock if clock is not None else _FakeClock()
        self.submissions: list[str] = []

    def snapshot(self) -> SnapshotModel:
        return SnapshotModel(
            title=self._title,
            phase=Phase.PRACTICE,
            prompt=self._payload.question,
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

    def submit_answer(self, raw: str) -> bool:
        self.submissions.append(str(raw))
        return True

    def update(self) -> None:
        pass


def _sample_payload(*, active_domains: tuple[str, ...], focus_label: str) -> CognitiveUpdatingPayload:
    problem = CognitiveUpdatingGenerator(seed=202).next_problem_for_selection(
        difficulty=0.5,
        training_profile=CognitiveUpdatingTrainingProfile(
            active_domains=active_domains,
            focus_label=focus_label,
        ),
        scenario_family="compressed",
    )
    return cast(CognitiveUpdatingPayload, problem.payload)


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


class _RecordingFont:
    def __init__(self, base: pygame.font.Font, sink: list[str]) -> None:
        self._base = base
        self._sink = sink

    def render(self, text: str, antialias: bool, color: object) -> pygame.Surface:
        self._sink.append(str(text))
        return self._base.render(text, antialias, color)

    def __getattr__(self, name: str) -> object:
        return getattr(self._base, name)


def _install_recording_fonts(app: App, screen: CognitiveTestScreen) -> list[str]:
    captured: list[str] = []
    for obj, attrs in (
        (app, ("_font",)),
        (screen, ("_small_font", "_tiny_font", "_mid_font", "_big_font")),
    ):
        for attr in attrs:
            font = getattr(obj, attr, None)
            if isinstance(font, _RecordingFont) or font is None:
                continue
            setattr(obj, attr, _RecordingFont(font, captured))
    return captured


def test_cognitive_updating_drill_title_routes_to_real_renderer(monkeypatch) -> None:
    payload = _sample_payload(active_domains=("controls", "state_code"), focus_label="Controls")
    _app, screen = _build_screen(
        _FakeCognitiveUpdatingEngine(payload, title="Cognitive Updating: Controls Anchor")
    )
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        called = {"value": False}
        original = screen._render_cognitive_updating_screen

        def wrapped(surface, snap, payload):
            called["value"] = True
            return original(surface, snap, payload)

        monkeypatch.setattr(screen, "_render_cognitive_updating_screen", wrapped)
        screen.render(surface)

        assert called["value"] is True
    finally:
        pygame.quit()


def test_cognitive_updating_focused_payload_dims_inactive_panels_visually() -> None:
    focused_payload = _sample_payload(active_domains=("controls", "state_code"), focus_label="Controls")
    full_payload = _sample_payload(
        active_domains=("controls", "navigation", "engine", "sensors", "objectives", "state_code"),
        focus_label="Full Mixed",
    )
    _app, screen = _build_screen(
        _FakeCognitiveUpdatingEngine(focused_payload, title="Cognitive Updating: Controls Anchor")
    )
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        screen.render(surface)
        focused_bytes = pygame.image.tobytes(surface, "RGBA")

        screen._engine = _FakeCognitiveUpdatingEngine(full_payload, title="Cognitive Updating: Pressure Run")
        screen.render(surface)
        full_bytes = pygame.image.tobytes(surface, "RGBA")

        assert focused_bytes != full_bytes
    finally:
        pygame.quit()


def test_cognitive_updating_workout_block_uses_real_runtime_screen() -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        app = App(surface=surface, font=font)
        root = MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True)
        app.push(root)

        clock = _FakeClock()
        plan = AntWorkoutPlan(
            code="cognitive_updating_workout",
            title="CU Workout UI",
            description="UI regression workout.",
            notes=("Untimed block setup.",),
            blocks=(
                AntWorkoutBlockPlan(
                    block_id="mixed",
                    label="Mixed Tempo",
                    description="Warm-up.",
                    focus_skills=("Domain switching",),
                    drill_code="cu_mixed_tempo",
                    mode=AntDrillMode.BUILD,
                    duration_min=0.25,
                ),
            ),
        )
        session = AntWorkoutSession(clock=clock, seed=606, plan=plan, starting_level=5)
        session.activate()
        session.activate()
        session.activate()
        session.activate()

        screen = AntWorkoutScreen(app, session=session, test_code="cognitive_updating_workout")
        app.push(screen)
        screen.render(surface)

        runtime = screen._runtime_screen
        assert runtime is not None
        assert runtime._engine is not None
        snap = runtime._engine.snapshot()
        assert str(snap.title).startswith("Cognitive Updating")
        assert isinstance(snap.payload, CognitiveUpdatingPayload)
    finally:
        pygame.quit()


def test_cognitive_updating_real_drill_engine_uses_payload_metadata_on_live_screen() -> None:
    clock = _FakeClock()
    engine = build_cu_mixed_tempo_drill(
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

        payload = engine._current.payload
        assert isinstance(payload, CognitiveUpdatingPayload)
        assert payload.focus_label == "Controls"
        assert payload.active_domains == ("controls", "state_code")
    finally:
        pygame.quit()


def test_cognitive_updating_messages_page_skips_cleared_objective_rows() -> None:
    clock = _FakeClock()
    payload = replace(
        _sample_payload(
            active_domains=("controls", "navigation", "engine", "sensors", "objectives", "state_code"),
            focus_label="Full Mixed",
        ),
        parcel_target=(725880, 292172, 180742),
        objective_deadline_s=40,
    )
    engine = _FakeCognitiveUpdatingEngine(
        payload,
        title="Cognitive Updating: Full Mixed",
        clock=clock,
    )
    app, screen = _build_screen(engine)
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        captured = _install_recording_fonts(app, screen)

        screen.render(surface)
        screen._cognitive_updating_upper_tab_index = 0
        clock.advance(23.0)
        captured.clear()
        screen.render(surface)
        rendered_before = "\n".join(captured)
        assert "Latitude: 725880" in rendered_before
        assert "Longitude: 292172" in rendered_before
        assert "Time: 180742" in rendered_before

        runtime = screen._cognitive_updating_runtime
        assert runtime is not None
        for ch in "725880":
            runtime.append_parcel_digit(ch)
        for ch in "292172":
            runtime.append_parcel_digit(ch)
        for ch in "180742":
            runtime.append_parcel_digit(ch)
        runtime.activate_dispenser()

        captured.clear()
        screen.render(surface)
        rendered_after = "\n".join(captured)
        assert "Latitude: 725880" not in rendered_after
        assert "Longitude: 292172" not in rendered_after
        assert "Time: 180742" not in rendered_after
    finally:
        pygame.quit()


def test_cognitive_updating_messages_page_removes_stale_sensor_line_after_sensor_action() -> None:
    clock = _FakeClock()
    payload = replace(
        _sample_payload(
            active_domains=("controls", "navigation", "engine", "sensors", "objectives", "state_code"),
            focus_label="Full Mixed",
        ),
        alpha_camera_due_s=45,
        bravo_camera_due_s=48,
        air_sensor_due_s=8,
        ground_sensor_due_s=10,
    )
    engine = _FakeCognitiveUpdatingEngine(
        payload,
        title="Cognitive Updating: Full Mixed",
        clock=clock,
    )
    app, screen = _build_screen(engine)
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        captured = _install_recording_fonts(app, screen)

        screen.render(surface)
        screen._cognitive_updating_upper_tab_index = 0
        captured.clear()
        screen.render(surface)
        rendered_before = "\n".join(captured)
        assert "Air Sensor due in: 08s" in rendered_before
        assert "Ground Sensor due in: 10s" in rendered_before

        runtime = screen._cognitive_updating_runtime
        assert runtime is not None
        runtime.toggle_sensor("air")

        captured.clear()
        screen.render(surface)
        rendered_after = "\n".join(captured)
        assert "Air Sensor due in:" not in rendered_after
        assert "Ground Sensor due in: 10s" in rendered_after
    finally:
        pygame.quit()


def test_cognitive_updating_comms_code_status_moves_from_controls_page_to_messages_page() -> None:
    clock = _FakeClock()
    payload = replace(
        _sample_payload(active_domains=("controls", "state_code"), focus_label="Controls"),
        comms_time_limit_s=30,
    )
    engine = _FakeCognitiveUpdatingEngine(
        payload,
        title="Cognitive Updating: Controls Anchor",
        clock=clock,
    )
    app, screen = _build_screen(engine)
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        captured = _install_recording_fonts(app, screen)

        screen.render(surface)
        screen._cognitive_updating_upper_tab_index = 0
        screen._cognitive_updating_lower_tab_index = 2
        captured.clear()
        screen.render(surface)

        rendered = "\n".join(captured)
        assert rendered.count(f"Current code: {payload.comms_code}") == 1
        assert rendered.count("Code changes in: 00:30") == 1

        runtime = screen._cognitive_updating_runtime
        assert runtime is not None

        clock.advance(29.5)
        captured.clear()
        screen.render(surface)
        rendered_pending = "\n".join(captured)
        assert rendered_pending.count(f"Current code: {payload.comms_code}") == 1
        assert rendered_pending.count("Code changes in: 00:01") == 1

        clock.advance(0.5)
        captured.clear()
        screen.render(surface)
        rendered_rolled = "\n".join(captured)
        rolled = runtime.snapshot()
        assert rolled.current_comms_code != payload.comms_code
        assert rendered_rolled.count(f"Current code: {rolled.current_comms_code}") == 1
        assert rendered_rolled.count("Code changes in: 00:30") == 1
    finally:
        pygame.quit()


def test_cognitive_updating_upcoming_comms_code_renders_only_on_messages_page() -> None:
    clock = _FakeClock()
    payload = replace(
        _sample_payload(active_domains=("controls", "state_code"), focus_label="Controls"),
        comms_time_limit_s=30,
        message_reveal_comms_s=5.0,
    )
    engine = _FakeCognitiveUpdatingEngine(
        payload,
        title="Cognitive Updating: Controls Anchor",
        clock=clock,
    )
    app, screen = _build_screen(engine)
    try:
        surface = pygame.display.get_surface()
        assert surface is not None
        captured = _install_recording_fonts(app, screen)

        screen.render(surface)
        clock.advance(25.0)

        screen._cognitive_updating_upper_tab_index = 0
        screen._cognitive_updating_lower_tab_index = 0
        captured.clear()
        screen.render(surface)
        rendered_messages = "\n".join(captured)
        assert "New Comms Code:" in rendered_messages
        assert f"Current code: {payload.comms_code}" in rendered_messages
        assert "Code changes in: 00:05" in rendered_messages

        screen._cognitive_updating_upper_tab_index = 2
        screen._cognitive_updating_lower_tab_index = 2
        captured.clear()
        screen.render(surface)
        rendered_controls = "\n".join(captured)
        assert "New Comms Code:" not in rendered_controls
        assert "Current code:" not in rendered_controls
        assert "Code changes in:" not in rendered_controls
    finally:
        pygame.quit()
