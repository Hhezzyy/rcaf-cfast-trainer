from __future__ import annotations

import pytest

from cfast_trainer.trace_lattice import TraceLatticeAction, trace_lattice_state
from cfast_trainer.trace_scene_3d import (
    build_trace_test_1_scene3d,
    classify_trace_test_1_view_maneuver,
    trace_test_1_camera_space_delta,
)
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


def _manual_plan(command: TraceTest1Command) -> TraceTest1AircraftPlan:
    state = {
        TraceTest1Command.LEFT: trace_lattice_state(col=4, row=1, level=2, forward=(0, 1, 0), up=(0, 0, 1)),
        TraceTest1Command.RIGHT: trace_lattice_state(col=2, row=1, level=2, forward=(0, 1, 0), up=(0, 0, 1)),
        TraceTest1Command.PUSH: trace_lattice_state(col=3, row=1, level=3, forward=(0, 1, 0), up=(0, 0, 1)),
        TraceTest1Command.PULL: trace_lattice_state(col=3, row=1, level=1, forward=(0, 1, 0), up=(0, 0, 1)),
    }[command]
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


def _sample_payload(*, difficulty: float = 0.82, progress: float = 0.68) -> TraceTest1Payload:
    prompt = TraceTest1Generator(seed=44).next_problem(difficulty=difficulty).payload
    assert isinstance(prompt, TraceTest1PromptPlan)
    return TraceTest1Payload(
        trial_stage=TraceTest1TrialStage.ANSWER_OPEN,
        stage_time_remaining_s=1.0,
        observe_progress=progress,
        prompt_index=prompt.prompt_index,
        active_command=prompt.red_plan.command,
        blue_commands=tuple(blue_plan.command for blue_plan in prompt.blue_plans),
        scene=trace_test_1_scene_frames(prompt=prompt, progress=progress),
        options=(),
        correct_code=int(
            {
                TraceTest1Command.LEFT: 1,
                TraceTest1Command.RIGHT: 2,
                TraceTest1Command.PUSH: 3,
                TraceTest1Command.PULL: 4,
            }[classify_trace_test_1_view_maneuver(prompt=prompt)]
        ),
        prompt_window_s=4.3,
        answer_open_progress=prompt.answer_open_progress,
        speed_multiplier=prompt.speed_multiplier,
        viewpoint_bearing_deg=180,
    )


def test_trace_test_1_projected_view_delta_separates_manual_commands() -> None:
    expected = {
        TraceTest1Command.LEFT: (-1.0, 0.0),
        TraceTest1Command.RIGHT: (1.0, 0.0),
        TraceTest1Command.PUSH: (0.0, 1.0),
        TraceTest1Command.PULL: (0.0, -1.0),
    }

    for command, (expected_x_sign, expected_y_sign) in expected.items():
        delta = trace_test_1_camera_space_delta(prompt=_manual_prompt(command))
        if expected_x_sign != 0.0:
            assert delta[0] * expected_x_sign > 0.0
            assert abs(delta[0]) > abs(delta[1])
        else:
            assert delta[1] * expected_y_sign > 0.0
            assert abs(delta[1]) > abs(delta[0])


def test_trace_test_1_scene3d_builder_is_deterministic_for_same_payload() -> None:
    payload = _sample_payload()

    first = build_trace_test_1_scene3d(payload=payload)
    second = build_trace_test_1_scene3d(payload=payload)

    assert first == second
    assert len(first.aircraft) == 1 + len(payload.scene.blue_frames)
    assert first.aircraft[0].asset_id == "plane_red"
    assert all(aircraft.asset_id == "plane_blue" for aircraft in first.aircraft[1:])


def test_trace_test_1_scene3d_camera_tracks_real_world_aircraft_pose() -> None:
    payload = _sample_payload(progress=0.24)
    snapshot = build_trace_test_1_scene3d(payload=payload)

    assert snapshot.camera.position != snapshot.camera.target
    assert snapshot.camera.target[1] > snapshot.camera.position[1]
    assert snapshot.aircraft[0].position == pytest.approx(payload.scene.red_frame.position)
