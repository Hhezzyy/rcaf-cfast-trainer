from __future__ import annotations

import math
from dataclasses import replace

import pytest

from cfast_trainer.trace_test_1 import (
    TraceTest1AircraftPlan,
    TraceTest1AircraftState,
    TraceTest1Attitude,
    TraceTest1Command,
    TraceTest1Generator,
    TraceTest1Payload,
    TraceTest1PromptPlan,
    TraceTest1TrialStage,
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


def test_trace_test_1_screen_heading_returns_finite_anchor_value() -> None:
    prompt = _sample_prompt()
    scene = build_scene_frames(prompt=prompt, progress=0.62)

    heading = screen_heading_deg(
        scene.red_frame,
        command=prompt.red_plan.command,
        observe_progress=0.62,
        answer_open_progress=prompt.answer_open_progress,
        size=(960, 540),
    )

    assert math.isfinite(heading)
    assert abs(heading) > 1.0


def test_trace_test_1_screen_heading_stays_neutral_during_lead_in() -> None:
    lead_in_progress = 0.30

    assert _screen_heading_for_prompt(_manual_prompt(TraceTest1Command.LEFT), progress=lead_in_progress) == pytest.approx(-90.0)
    assert _screen_heading_for_prompt(_manual_prompt(TraceTest1Command.RIGHT), progress=lead_in_progress) == pytest.approx(-90.0)
    assert _screen_heading_for_prompt(_manual_prompt(TraceTest1Command.PUSH), progress=lead_in_progress) == pytest.approx(-90.0)
    assert _screen_heading_for_prompt(_manual_prompt(TraceTest1Command.PULL), progress=lead_in_progress) == pytest.approx(-90.0)


def test_trace_test_1_screen_heading_uses_exact_command_anchors_after_maneuver_start() -> None:
    assert _screen_heading_for_prompt(_manual_prompt(TraceTest1Command.LEFT), progress=0.70) == pytest.approx(-160.0)
    assert _screen_heading_for_prompt(_manual_prompt(TraceTest1Command.RIGHT), progress=0.70) == pytest.approx(0.0)
    assert _screen_heading_for_prompt(_manual_prompt(TraceTest1Command.PUSH), progress=0.82) == pytest.approx(90.0)
    assert _screen_heading_for_prompt(_manual_prompt(TraceTest1Command.PULL), progress=0.82) == pytest.approx(-90.0)


def test_trace_test_1_screen_headings_stay_on_exact_command_anchors_across_samples() -> None:
    left_prompt = _manual_prompt(TraceTest1Command.LEFT)
    right_prompt = _manual_prompt(TraceTest1Command.RIGHT)
    left_headings = [
        _screen_heading_for_prompt(left_prompt, progress=progress)
        for progress in (0.52, 0.60, 0.68, 0.76)
    ]
    right_headings = [
        _screen_heading_for_prompt(right_prompt, progress=progress)
        for progress in (0.52, 0.60, 0.68, 0.76)
    ]

    assert all(heading == pytest.approx(-160.0) for heading in left_headings)
    assert all(heading == pytest.approx(0.0) for heading in right_headings)


def test_trace_test_1_screen_pose_uses_exact_heading_anchors_and_no_tilt() -> None:
    left_pose = _screen_pose_for_prompt(_manual_prompt(TraceTest1Command.LEFT), progress=0.70)
    push_pose = _screen_pose_for_prompt(_manual_prompt(TraceTest1Command.PUSH), progress=0.82)
    pull_pose = _screen_pose_for_prompt(_manual_prompt(TraceTest1Command.PULL), progress=0.82)

    assert left_pose == pytest.approx((-160.0, 0.0, 0.0))
    assert push_pose == pytest.approx((90.0, 0.0, 0.0))
    assert pull_pose == pytest.approx((-90.0, 0.0, 0.0))
    assert push_pose[2] == pytest.approx(0.0)
    assert pull_pose[2] == pytest.approx(0.0)


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


def test_trace_test_1_screen_pose_stays_on_command_pose_when_frame_attitude_decays() -> None:
    prompt = _manual_prompt(TraceTest1Command.LEFT)
    progress = 0.70
    frame = trace_test_1_scene_frames(prompt=prompt, progress=progress).red_frame
    decayed_frame = replace(
        frame,
        attitude=TraceTest1Attitude(roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0),
        world_tangent=(0.0, 0.0, 0.0),
    )

    pose = aircraft_screen_pose_for_frame(
        decayed_frame,
        command=prompt.red_plan.command,
        observe_progress=progress,
        answer_open_progress=prompt.answer_open_progress,
        size=(960, 540),
    )

    assert pose == pytest.approx((-160.0, 0.0, 0.0))


def test_trace_test_1_payload_blue_commands_drive_blue_screen_poses() -> None:
    prompt = TraceTest1PromptPlan(
        prompt_index=0,
        answer_open_progress=0.42,
        speed_multiplier=1.0,
        red_plan=TraceTest1AircraftPlan(
            start_state=TraceTest1AircraftState(position=(0.0, 8.0, 12.0), heading_deg=0.0),
            command=TraceTest1Command.LEFT,
            lead_distance=18.0,
            maneuver_distance=18.0,
            altitude_delta=0.0,
        ),
        blue_plans=(
            TraceTest1AircraftPlan(
                start_state=TraceTest1AircraftState(position=(-6.0, 8.0, 12.0), heading_deg=0.0),
                command=TraceTest1Command.RIGHT,
                lead_distance=18.0,
                maneuver_distance=18.0,
                altitude_delta=0.0,
            ),
            TraceTest1AircraftPlan(
                start_state=TraceTest1AircraftState(position=(6.0, 8.0, 12.0), heading_deg=0.0),
                command=TraceTest1Command.PUSH,
                lead_distance=18.0,
                maneuver_distance=18.0,
                altitude_delta=-8.0,
            ),
        ),
    )
    progress = 0.70
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
    assert red_pose == pytest.approx((-160.0, 0.0, 0.0))
    assert len(blue_poses) == 2
    assert blue_poses[0] == pytest.approx((0.0, 0.0, 0.0))
    assert blue_poses[1] == pytest.approx((90.0, 0.0, 0.0))


def test_trace_test_1_forward_depth_changes_scale_and_vertical_position() -> None:
    near_center, near_scale = project_scene_position((0.0, 8.0, 12.0), size=(960, 540))
    far_center, far_scale = project_scene_position((0.0, 40.0, 12.0), size=(960, 540))

    assert near_center[0] == pytest.approx(far_center[0], abs=0.01)
    assert far_center[1] < near_center[1]
    assert near_scale > far_scale


def test_trace_test_1_altitude_changes_vertical_position_without_depth_scale_shift() -> None:
    low_center, low_scale = project_scene_position((0.0, 20.0, 8.0), size=(960, 540))
    high_center, high_scale = project_scene_position((0.0, 20.0, 18.0), size=(960, 540))

    assert low_center[0] == pytest.approx(high_center[0], abs=0.01)
    assert low_center[1] > high_center[1]
    assert low_scale == pytest.approx(high_scale, abs=0.01)


def test_trace_test_1_left_and_right_projected_motion_include_forward_depth_drift() -> None:
    left_prompt = _manual_prompt(TraceTest1Command.LEFT)
    right_prompt = _manual_prompt(TraceTest1Command.RIGHT)

    left_centers = [
        project_scene_position(
            trace_test_1_scene_frames(prompt=left_prompt, progress=progress).red_frame.position,
            size=(960, 540),
        )[0]
        for progress in (0.44, 0.56, 0.72, 0.88)
    ]
    right_centers = [
        project_scene_position(
            trace_test_1_scene_frames(prompt=right_prompt, progress=progress).red_frame.position,
            size=(960, 540),
        )[0]
        for progress in (0.44, 0.56, 0.72, 0.88)
    ]

    assert all(later[0] < earlier[0] for earlier, later in zip(left_centers, left_centers[1:]))
    assert all(later[1] < earlier[1] for earlier, later in zip(left_centers, left_centers[1:]))
    assert all(later[0] > earlier[0] for earlier, later in zip(right_centers, right_centers[1:]))
    assert all(later[1] < earlier[1] for earlier, later in zip(right_centers, right_centers[1:]))
