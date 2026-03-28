from __future__ import annotations

from dataclasses import replace
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cfast_trainer.aircraft_art import panda3d_fixed_wing_hpr_from_screen_heading
from cfast_trainer.trace_test_1 import (
    TraceTest1AircraftPlan,
    TraceTest1AircraftState,
    TraceTest1Attitude,
    TraceTest1Command,
    TraceTest1Generator,
    TraceTest1PromptPlan,
    trace_test_1_scene_frames,
)
from cfast_trainer.trace_test_1_panda3d import (
    TraceTest1Panda3DRenderer,
    panda3d_trace_test_1_rendering_available,
)


_HELPER = Path(__file__).with_name("_panda3d_runtime_probe.py")


def _manual_prompt(command: TraceTest1Command) -> TraceTest1PromptPlan:
    altitude_delta = {
        TraceTest1Command.PUSH: -8.0,
        TraceTest1Command.PULL: 8.0,
    }.get(command, 0.0)
    return TraceTest1PromptPlan(
        prompt_index=0,
        answer_open_progress=0.42,
        speed_multiplier=1.0,
        red_plan=TraceTest1AircraftPlan(
            start_state=TraceTest1AircraftState(position=(0.0, 8.0, 12.0), heading_deg=0.0),
            command=command,
            lead_distance=18.0,
            maneuver_distance=18.0,
            altitude_delta=altitude_delta,
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


def _run_probe(scene_name: str) -> dict[str, object]:
    env = dict(os.environ)
    env.pop("SDL_VIDEODRIVER", None)
    env.setdefault("SDL_AUDIODRIVER", "dummy")
    env.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    result = subprocess.run(
        [sys.executable, str(_HELPER), scene_name],
        capture_output=True,
        text=True,
        env=env,
        cwd=Path(__file__).resolve().parents[1],
    )
    if result.returncode == 77:
        pytest.skip(result.stdout.strip() or result.stderr.strip() or "Panda3D unavailable")
    assert result.returncode == 0, result.stdout + result.stderr
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert lines, "subprocess produced no output"
    return json.loads(lines[-1])


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


def test_trace_test_1_panda_hpr_uses_exact_left_anchor() -> None:
    prompt = _manual_prompt(TraceTest1Command.LEFT)
    expected_hpr = panda3d_fixed_wing_hpr_from_screen_heading(
        screen_heading_deg=-160.0,
        pitch_deg=0.0,
        roll_deg=0.0,
    )

    hpr = _panda_hpr_for_prompt(prompt, progress=0.70)

    assert hpr == pytest.approx(expected_hpr)
    assert hpr == pytest.approx((290.0, 0.0, 0.0))


def test_trace_test_1_panda_hpr_stays_neutral_during_lead_in() -> None:
    assert _panda_hpr_for_prompt(_manual_prompt(TraceTest1Command.LEFT), progress=0.30) == pytest.approx((0.0, 0.0, 0.0))
    assert _panda_hpr_for_prompt(_manual_prompt(TraceTest1Command.PUSH), progress=0.30) == pytest.approx((0.0, 0.0, 0.0))


def test_trace_test_1_panda_turn_headings_use_exact_anchors_across_samples() -> None:
    left_prompt = _manual_prompt(TraceTest1Command.LEFT)
    right_prompt = _manual_prompt(TraceTest1Command.RIGHT)
    left_headings = [
        _panda_hpr_for_prompt(left_prompt, progress=progress)[0]
        for progress in (0.52, 0.60, 0.68, 0.76)
    ]
    right_headings = [
        _panda_hpr_for_prompt(right_prompt, progress=progress)[0]
        for progress in (0.52, 0.60, 0.68, 0.76)
    ]

    assert all(heading == pytest.approx(290.0) for heading in left_headings)
    assert all(heading == pytest.approx(90.0) for heading in right_headings)


def test_trace_test_1_panda_push_and_pull_use_exact_anchors() -> None:
    push_hpr = _panda_hpr_for_prompt(_manual_prompt(TraceTest1Command.PUSH), progress=0.82)
    pull_hpr = _panda_hpr_for_prompt(_manual_prompt(TraceTest1Command.PULL), progress=0.82)

    assert push_hpr[0] == pytest.approx(180.0)
    assert pull_hpr[0] == pytest.approx(0.0)
    assert push_hpr[1] == pytest.approx(0.0)
    assert pull_hpr[1] == pytest.approx(0.0)
    assert push_hpr[2] == pytest.approx(0.0)
    assert pull_hpr[2] == pytest.approx(0.0)


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


def test_trace_test_1_panda_hpr_stays_on_command_pose_when_frame_attitude_decays() -> None:
    prompt = _manual_prompt(TraceTest1Command.LEFT)
    progress = 0.70
    frame = trace_test_1_scene_frames(prompt=prompt, progress=progress).red_frame
    decayed_frame = replace(
        frame,
        attitude=TraceTest1Attitude(roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0),
        world_tangent=(0.0, 0.0, 0.0),
    )

    hpr = TraceTest1Panda3DRenderer._aircraft_hpr_for_frame(
        frame=decayed_frame,
        command=prompt.red_plan.command,
        observe_progress=progress,
        answer_open_progress=prompt.answer_open_progress,
        size=(640, 640),
    )

    assert hpr == pytest.approx((290.0, 0.0, 0.0))


def test_trace_test_1_screen_prefers_panda3d_runtime() -> None:
    probe = _run_probe("trace_test_1")

    assert probe["kind"] == "trace_test_1"
    assert probe["renderer_type"] == "TraceTest1Panda3DRenderer"
    assert probe["gl_scene_type"] is None
    assert probe["renderer_size"][0] > 0
    assert probe["renderer_size"][1] > 0
    assert probe["aircraft_count"] >= 2
    assert probe["blue_count"] >= 1
    assert len(probe["red_hpr"]) == 3
    assert probe["blue_hpr_count"] == probe["blue_count"]
    assert sum(probe["avg_color"]) > 60
