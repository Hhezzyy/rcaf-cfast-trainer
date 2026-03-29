from __future__ import annotations

import math

import pygame
import pytest

from cfast_trainer.trace_lattice import TraceLatticeAction, trace_lattice_state
from cfast_trainer.trace_test_1 import (
    TraceTest1AircraftPlan,
    TraceTest1Command,
    TraceTest1Generator,
    TraceTest1Payload,
    TraceTest1PromptPlan,
    TraceTest1TrialStage,
    _tt1_action_for_command,
    _tt1_aircraft_state_from_lattice_state,
    trace_test_1_scene_frames,
)
from cfast_trainer.trace_test_1_gl import (
    aircraft_hpr_for_frame,
    aircraft_screen_poses_for_payload,
    aircraft_screen_pose_for_frame,
    build_scene_frames,
    project_scene_position,
    screen_heading_deg,
)


def _sample_prompt(*, difficulty: float = 0.82) -> TraceTest1PromptPlan:
    payload = TraceTest1Generator(seed=44).next_problem(difficulty=difficulty).payload
    assert isinstance(payload, TraceTest1PromptPlan)
    return payload


def _prompt_start_state(command: TraceTest1Command):
    starts = {
        TraceTest1Command.LEFT: trace_lattice_state(col=4, row=1, level=2, forward=(0, 1, 0), up=(0, 0, 1)),
        TraceTest1Command.RIGHT: trace_lattice_state(col=2, row=1, level=2, forward=(0, 1, 0), up=(0, 0, 1)),
        TraceTest1Command.PUSH: trace_lattice_state(col=3, row=1, level=3, forward=(0, 1, 0), up=(0, 0, 1)),
        TraceTest1Command.PULL: trace_lattice_state(col=3, row=1, level=1, forward=(0, 1, 0), up=(0, 0, 1)),
    }
    return starts[command]


def _manual_plan(command: TraceTest1Command) -> TraceTest1AircraftPlan:
    state = _prompt_start_state(command)
    return TraceTest1AircraftPlan(
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
    )


def _manual_prompt(command: TraceTest1Command) -> TraceTest1PromptPlan:
    return TraceTest1PromptPlan(
        prompt_index=0,
        answer_open_progress=0.36,
        speed_multiplier=1.15,
        red_plan=_manual_plan(command),
        blue_plans=(),
    )


def _screen_heading_for_prompt(prompt: TraceTest1PromptPlan, *, progress: float) -> float:
    frame = trace_test_1_scene_frames(prompt=prompt, progress=progress).red_frame
    return screen_heading_deg(
        frame,
        command=prompt.red_plan.command,
        observe_progress=progress,
        answer_open_progress=prompt.answer_open_progress,
        size=(960, 540),
    )


def _screen_pose_for_prompt(
    prompt: TraceTest1PromptPlan,
    *,
    progress: float,
) -> tuple[float, float, float]:
    frame = trace_test_1_scene_frames(prompt=prompt, progress=progress).red_frame
    return aircraft_screen_pose_for_frame(
        frame,
        command=prompt.red_plan.command,
        observe_progress=progress,
        answer_open_progress=prompt.answer_open_progress,
        size=(960, 540),
    )


def test_trace_test_1_aircraft_hpr_matches_heading_pitch_and_roll() -> None:
    prompt = _sample_prompt()
    scene = build_scene_frames(prompt=prompt, progress=0.78)
    hpr = aircraft_hpr_for_frame(scene.red_frame)

    assert len(hpr) == 3
    assert hpr[0] == scene.red_frame.travel_heading_deg
    assert hpr[1] == scene.red_frame.attitude.pitch_deg
    assert hpr[2] == scene.red_frame.attitude.roll_deg


def test_trace_test_1_projection_keeps_red_on_screen_and_projects_blues() -> None:
    prompt = _sample_prompt(difficulty=0.95)
    scene = build_scene_frames(prompt=prompt, progress=0.48)

    red_center, red_scale = project_scene_position(scene.red_frame.position, size=(960, 540))
    assert 0.0 <= red_center[0] <= 960.0
    assert 0.0 <= red_center[1] <= 540.0
    assert red_scale > 0.0

    for blue_frame in scene.blue_frames:
        blue_center, blue_scale = project_scene_position(blue_frame.position, size=(960, 540))
        assert math.isfinite(blue_center[0])
        assert math.isfinite(blue_center[1])
        assert blue_scale > 0.0


def test_trace_test_1_lattice_lead_in_screen_heading_is_shared_across_commands() -> None:
    headings = [
        _screen_heading_for_prompt(_manual_prompt(command), progress=0.18)
        for command in TraceTest1Command
    ]

    assert all(math.isfinite(heading) for heading in headings)
    assert max(headings) - min(headings) <= 0.01


def test_trace_test_1_screen_pose_matches_lattice_command_orientations() -> None:
    assert _screen_pose_for_prompt(_manual_prompt(TraceTest1Command.LEFT), progress=0.68) == pytest.approx((-180.0, 0.0, 0.0))
    assert _screen_pose_for_prompt(_manual_prompt(TraceTest1Command.RIGHT), progress=0.68) == pytest.approx((0.0, 0.0, 0.0))
    assert _screen_pose_for_prompt(_manual_prompt(TraceTest1Command.PUSH), progress=0.68) == pytest.approx((90.0, -90.0, 0.0))
    assert _screen_pose_for_prompt(_manual_prompt(TraceTest1Command.PULL), progress=0.68) == pytest.approx((-90.0, 90.0, 0.0))


def test_trace_test_1_screen_pose_sampling_is_deterministic_for_same_seed() -> None:
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
    poses_a = tuple(
        aircraft_screen_pose_for_frame(
            frame,
            command=prompt_a.red_plan.command,
            observe_progress=progress,
            answer_open_progress=prompt_a.answer_open_progress,
            size=(960, 540),
        )
        for frame, progress in zip(frames_a, progresses, strict=True)
    )
    poses_b = tuple(
        aircraft_screen_pose_for_frame(
            frame,
            command=prompt_b.red_plan.command,
            observe_progress=progress,
            answer_open_progress=prompt_b.answer_open_progress,
            size=(960, 540),
        )
        for frame, progress in zip(frames_b, progresses, strict=True)
    )

    assert frames_a == frames_b
    assert poses_a == poses_b


def test_trace_test_1_payload_blue_commands_drive_blue_screen_poses() -> None:
    prompt = TraceTest1PromptPlan(
        prompt_index=0,
        answer_open_progress=0.36,
        speed_multiplier=1.15,
        red_plan=_manual_plan(TraceTest1Command.LEFT),
        blue_plans=(
            _manual_plan(TraceTest1Command.RIGHT),
            _manual_plan(TraceTest1Command.PUSH),
        ),
    )
    progress = 0.68
    scene = trace_test_1_scene_frames(prompt=prompt, progress=progress)
    payload = TraceTest1Payload(
        trial_stage=TraceTest1TrialStage.ANSWER_OPEN,
        stage_time_remaining_s=1.0,
        observe_progress=progress,
        prompt_index=prompt.prompt_index,
        active_command=prompt.red_plan.command,
        blue_commands=tuple(blue_plan.command for blue_plan in prompt.blue_plans),
        scene=scene,
        options=(),
        correct_code=1,
        prompt_window_s=4.3,
        answer_open_progress=prompt.answer_open_progress,
        speed_multiplier=prompt.speed_multiplier,
        viewpoint_bearing_deg=180,
    )

    red_pose, blue_poses = aircraft_screen_poses_for_payload(payload, size=(960, 540))

    assert payload.blue_commands == (TraceTest1Command.RIGHT, TraceTest1Command.PUSH)
    assert red_pose == pytest.approx((-180.0, 0.0, 0.0))
    assert len(blue_poses) == 2
    assert blue_poses[0] == pytest.approx((0.0, 0.0, 0.0))
    assert blue_poses[1] == pytest.approx((90.0, -90.0, 0.0))


def test_trace_test_1_forward_depth_changes_scale_and_vertical_position() -> None:
    near_center, near_scale = project_scene_position((0.0, 26.0, 12.0), size=(960, 540))
    far_center, far_scale = project_scene_position((0.0, 40.0, 12.0), size=(960, 540))

    assert near_center[0] == pytest.approx(far_center[0], abs=0.01)
    assert far_center[1] < near_center[1]
    assert near_scale > far_scale


def test_trace_test_1_altitude_changes_vertical_position_without_depth_scale_shift() -> None:
    low_center, low_scale = project_scene_position((0.0, 40.0, 6.0), size=(960, 540))
    high_center, high_scale = project_scene_position((0.0, 40.0, 18.0), size=(960, 540))

    assert low_center[0] == pytest.approx(high_center[0], abs=0.01)
    assert low_center[1] > high_center[1]
    assert low_scale == pytest.approx(high_scale, abs=0.01)


def test_trace_test_1_lateral_turns_translate_horizontally_after_rotation_phase() -> None:
    left_prompt = _manual_prompt(TraceTest1Command.LEFT)
    right_prompt = _manual_prompt(TraceTest1Command.RIGHT)

    left_centers = [
        project_scene_position(
            trace_test_1_scene_frames(prompt=left_prompt, progress=progress).red_frame.position,
            size=(960, 540),
        )[0]
        for progress in (0.36, 0.40, 0.68, 1.0)
    ]
    right_centers = [
        project_scene_position(
            trace_test_1_scene_frames(prompt=right_prompt, progress=progress).red_frame.position,
            size=(960, 540),
        )[0]
        for progress in (0.36, 0.40, 0.68, 1.0)
    ]

    assert left_centers[0] == pytest.approx(left_centers[1], abs=0.01)
    assert left_centers[2][0] < left_centers[1][0]
    assert left_centers[3][0] < left_centers[2][0]

    assert right_centers[0] == pytest.approx(right_centers[1], abs=0.01)
    assert right_centers[2][0] > right_centers[1][0]
    assert right_centers[3][0] > right_centers[2][0]
