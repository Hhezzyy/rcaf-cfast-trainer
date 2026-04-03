from __future__ import annotations

from dataclasses import replace
import sys
from importlib.machinery import ModuleSpec
from types import ModuleType
import os

import pygame
import pytest

if "moderngl" not in sys.modules:
    moderngl_stub = ModuleType("moderngl")
    moderngl_stub.__spec__ = ModuleSpec("moderngl", loader=None)
    sys.modules["moderngl"] = moderngl_stub

from cfast_trainer.app import App, CognitiveTestScreen, MenuItem, MenuScreen
from cfast_trainer.trace_lattice import TraceLatticeAction, trace_lattice_state
from cfast_trainer.trace_test_1 import (
    TraceTest1AircraftPlan,
    TraceTest1Attitude,
    TraceTest1Command,
    TraceTest1Generator,
    TraceTest1PromptPlan,
    _tt1_action_for_command,
    _tt1_aircraft_state_from_lattice_state,
    build_trace_test_1_test,
    trace_test_1_scene_frames,
)
from cfast_trainer.trace_test_1_panda3d import (
    TraceTest1Panda3DRenderer,
    panda3d_trace_test_1_rendering_available,
)


class _FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _prompt_start_state(command: TraceTest1Command):
    starts = {
        TraceTest1Command.LEFT: trace_lattice_state(col=4, row=1, level=2, forward=(0, 1, 0), up=(0, 0, 1)),
        TraceTest1Command.RIGHT: trace_lattice_state(col=2, row=1, level=2, forward=(0, 1, 0), up=(0, 0, 1)),
        TraceTest1Command.PUSH: trace_lattice_state(col=3, row=1, level=3, forward=(0, 1, 0), up=(0, 0, 1)),
        TraceTest1Command.PULL: trace_lattice_state(col=3, row=1, level=1, forward=(0, 1, 0), up=(0, 0, 1)),
    }
    return starts[command]


def _manual_prompt(command: TraceTest1Command) -> TraceTest1PromptPlan:
    state = _prompt_start_state(command)
    return TraceTest1PromptPlan(
        prompt_index=0,
        answer_open_progress=0.36,
        speed_multiplier=1.15,
        red_plan=TraceTest1AircraftPlan(
            start_state=_tt1_aircraft_state_from_lattice_state(state),
            command=command,
            lead_distance=5.0,
            maneuver_distance=4.0,
            altitude_delta=2.0 if command is TraceTest1Command.PULL else -2.0,
            lattice_start=state,
            lattice_actions=(
                TraceLatticeAction.STRAIGHT,
                _tt1_action_for_command(command),
                TraceLatticeAction.STRAIGHT,
            ),
        ),
        blue_plans=(),
    )


def _panda_hpr_for_prompt(
    prompt: TraceTest1PromptPlan,
    *,
    progress: float,
) -> tuple[float, float, float]:
    frame = trace_test_1_scene_frames(prompt=prompt, progress=progress).red_frame
    return TraceTest1Panda3DRenderer._aircraft_hpr_for_frame(
        frame=frame,
        command=prompt.red_plan.command,
        observe_progress=progress,
        answer_open_progress=prompt.answer_open_progress,
        size=(640, 640),
    )


def _run_probe() -> dict[str, object]:
    previous_driver = os.environ.get("SDL_VIDEODRIVER")
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()
    screen: CognitiveTestScreen | None = None
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        app = App(surface=surface, font=font, opengl_enabled=True)
        app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))
        clock = _FakeClock()
        screen = CognitiveTestScreen(
            app,
            engine_factory=lambda: build_trace_test_1_test(
                clock=clock,
                seed=37,
                difficulty=0.6,
            ),
            test_code="trace_test_1",
        )
        app.push(screen)
        screen._engine.start_practice()
        clock.advance(0.1)
        screen._engine.update()
        app.render()
        scene = app.consume_gl_scene()
        assert scene is not None
        payload = scene.payload
        return {
            "scene_type": type(scene).__name__,
            "correct_code": int(payload.correct_code),
            "active_command": str(payload.active_command),
            "blue_count": len(payload.scene.blue_frames),
            "viewpoint_bearing_deg": int(payload.viewpoint_bearing_deg),
        }
    finally:
        if screen is not None:
            close = getattr(screen, "close", None)
            if callable(close):
                close()
        pygame.quit()
        if previous_driver is None:
            os.environ.pop("SDL_VIDEODRIVER", None)
        else:
            os.environ["SDL_VIDEODRIVER"] = previous_driver


def test_panda3d_trace_test_1_rendering_disabled_for_dummy_video(monkeypatch) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.delenv("CFAST_TRACE_TEST_1_RENDERER", raising=False)

    assert panda3d_trace_test_1_rendering_available() is False


def test_panda3d_trace_test_1_rendering_ignores_non_panda_preference(monkeypatch) -> None:
    monkeypatch.setenv("CFAST_TRACE_TEST_1_RENDERER", "pygame")
    monkeypatch.delenv("SDL_VIDEODRIVER", raising=False)
    monkeypatch.setattr(
        "cfast_trainer.trace_test_1_panda3d.importlib.util.find_spec",
        lambda _name: object(),
    )

    assert panda3d_trace_test_1_rendering_available() is True


def test_trace_test_1_panda_hpr_tracks_trace_screen_pose_for_all_commands() -> None:
    assert _panda_hpr_for_prompt(_manual_prompt(TraceTest1Command.LEFT), progress=0.68) == pytest.approx((270.0, 0.0, 0.0))
    assert _panda_hpr_for_prompt(_manual_prompt(TraceTest1Command.RIGHT), progress=0.68) == pytest.approx((90.0, 0.0, 0.0))
    assert _panda_hpr_for_prompt(_manual_prompt(TraceTest1Command.PUSH), progress=0.68) == pytest.approx((180.0, 90.0, 0.0))
    assert _panda_hpr_for_prompt(_manual_prompt(TraceTest1Command.PULL), progress=0.68) == pytest.approx((0.0, -90.0, 0.0))


def test_trace_test_1_panda_hpr_matches_shared_lead_in_screen_heading() -> None:
    assert _panda_hpr_for_prompt(_manual_prompt(TraceTest1Command.LEFT), progress=0.18) == pytest.approx((19.536654938128393, 0.0, 0.0))
    assert _panda_hpr_for_prompt(_manual_prompt(TraceTest1Command.PUSH), progress=0.18) == pytest.approx((19.536654938128393, 0.0, 0.0))


def test_panda3d_trace_test_1_rendering_returns_false_when_direct_is_missing(monkeypatch) -> None:
    monkeypatch.delenv("SDL_VIDEODRIVER", raising=False)

    def _missing(_name: str):
        raise ModuleNotFoundError("direct")

    monkeypatch.setattr(
        "cfast_trainer.trace_test_1_panda3d.importlib.util.find_spec",
        _missing,
    )

    assert panda3d_trace_test_1_rendering_available() is False


def test_trace_test_1_panda_hpr_sampling_is_deterministic_for_same_seed() -> None:
    progresses = (0.18, 0.42, 0.66, 0.84)
    prompt_a = TraceTest1Generator(seed=44).next_problem(difficulty=0.82).payload
    prompt_b = TraceTest1Generator(seed=44).next_problem(difficulty=0.82).payload

    assert isinstance(prompt_a, TraceTest1PromptPlan)
    assert isinstance(prompt_b, TraceTest1PromptPlan)

    frames_a = tuple(
        trace_test_1_scene_frames(prompt=prompt_a, progress=progress).red_frame
        for progress in progresses
    )
    frames_b = tuple(
        trace_test_1_scene_frames(prompt=prompt_b, progress=progress).red_frame
        for progress in progresses
    )
    samples_a = tuple(
        TraceTest1Panda3DRenderer._aircraft_hpr_for_frame(
            frame=frame,
            command=prompt_a.red_plan.command,
            observe_progress=progress,
            answer_open_progress=prompt_a.answer_open_progress,
            size=(640, 640),
        )
        for frame, progress in zip(frames_a, progresses, strict=True)
    )
    samples_b = tuple(
        TraceTest1Panda3DRenderer._aircraft_hpr_for_frame(
            frame=frame,
            command=prompt_b.red_plan.command,
            observe_progress=progress,
            answer_open_progress=prompt_b.answer_open_progress,
            size=(640, 640),
        )
        for frame, progress in zip(frames_b, progresses, strict=True)
    )

    assert frames_a == frames_b
    assert samples_a == samples_b


def test_trace_test_1_panda_hpr_uses_trace_command_fallback_when_frame_motion_decays() -> None:
    prompt = _manual_prompt(TraceTest1Command.LEFT)
    progress = 0.68
    frame = trace_test_1_scene_frames(prompt=prompt, progress=progress).red_frame
    decayed_frame = replace(
        frame,
        attitude=TraceTest1Attitude(roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0),
        world_tangent=(0.0, 0.0, 0.0),
        world_forward=(0.0, 0.0, 0.0),
    )

    hpr = TraceTest1Panda3DRenderer._aircraft_hpr_for_frame(
        frame=decayed_frame,
        command=prompt.red_plan.command,
        observe_progress=progress,
        answer_open_progress=prompt.answer_open_progress,
        size=(640, 640),
    )

    assert hpr == pytest.approx((290.0, 0.0, 0.0))


def test_trace_test_1_screen_queues_modern_gl_runtime() -> None:
    probe = _run_probe()
    code_for_command = {
        "LEFT": 1,
        "RIGHT": 2,
        "PUSH": 3,
        "PULL": 4,
    }

    assert probe["scene_type"] == "TraceTest1GlScene"
    assert probe["correct_code"] == code_for_command[probe["active_command"]]
    assert probe["blue_count"] >= 1
    assert probe["viewpoint_bearing_deg"] == 180
