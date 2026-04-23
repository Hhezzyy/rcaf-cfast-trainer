from __future__ import annotations

import pytest

from cfast_trainer.trace_lattice import (
    DEFAULT_TRACE_LATTICE_SPEC,
    TraceLatticeAction,
    TraceLatticeMotionConfig,
    TraceLatticeMotionPhase,
    TraceLatticeMotionPlayer,
    TraceLatticeOrientation,
    trace_lattice_build_path,
    trace_lattice_center_state,
    trace_lattice_execute_step,
    trace_lattice_right,
    trace_lattice_rotate,
    trace_lattice_sample_path,
    trace_lattice_state,
)


def test_trace_lattice_rotations_are_self_relative() -> None:
    orientation = TraceLatticeOrientation(forward=(0, 1, 0), up=(0, 0, 1))

    assert trace_lattice_right(orientation) == (1, 0, 0)
    assert trace_lattice_rotate(orientation, TraceLatticeAction.LEFT) == TraceLatticeOrientation(
        forward=(-1, 0, 0),
        up=(0, 0, 1),
    )
    assert trace_lattice_rotate(orientation, TraceLatticeAction.RIGHT) == TraceLatticeOrientation(
        forward=(1, 0, 0),
        up=(0, 0, 1),
    )
    pulled = trace_lattice_rotate(orientation, TraceLatticeAction.PULL)
    assert pulled == TraceLatticeOrientation(forward=(0, 0, 1), up=(0, -1, 0))
    assert trace_lattice_rotate(pulled, TraceLatticeAction.RIGHT) == TraceLatticeOrientation(
        forward=(1, 0, 0),
        up=(0, -1, 0),
    )


def test_trace_lattice_samples_non_straight_steps_with_move_then_pivot() -> None:
    path = trace_lattice_build_path(
        start_state=trace_lattice_center_state(),
        actions=(TraceLatticeAction.RIGHT,),
    )

    early_pose = trace_lattice_sample_path(path, progress=0.10, turn_phase_ratio=0.35)
    mid_pose = trace_lattice_sample_path(path, progress=0.30, turn_phase_ratio=0.35)
    dwell_pose = trace_lattice_sample_path(path, progress=0.52, turn_phase_ratio=0.35)
    turn_pose = trace_lattice_sample_path(path, progress=0.75, turn_phase_ratio=0.35)
    final_hold_pose = trace_lattice_sample_path(path, progress=0.96, turn_phase_ratio=0.35)

    assert early_pose.position[0] == pytest.approx(0.0)
    assert early_pose.position[1] > 3.0
    assert early_pose.position[2] == pytest.approx(0.0)
    assert early_pose.rotated is False
    assert early_pose.forward == pytest.approx((0.0, 1.0, 0.0), abs=1e-6)

    assert mid_pose.position[0] == pytest.approx(0.0)
    assert mid_pose.position[1] > early_pose.position[1]
    assert mid_pose.position[2] == pytest.approx(0.0)
    assert mid_pose.forward == pytest.approx(early_pose.forward, abs=1e-6)

    assert dwell_pose.position == pytest.approx((0.0, 4.0, 0.0), abs=1e-6)
    assert dwell_pose.forward == pytest.approx((0.0, 1.0, 0.0), abs=1e-6)

    assert turn_pose.position == pytest.approx(dwell_pose.position)
    assert turn_pose.rotated is True
    assert turn_pose.forward[0] > 0.0
    assert turn_pose.forward[1] > 0.0

    assert final_hold_pose.position == pytest.approx(dwell_pose.position)
    assert final_hold_pose.forward == pytest.approx((1.0, 0.0, 0.0), abs=1e-6)


def test_trace_lattice_sample_reaches_exact_cardinal_orientation_after_node_hold() -> None:
    path = trace_lattice_build_path(
        start_state=trace_lattice_center_state(),
        actions=(TraceLatticeAction.RIGHT,),
    )

    node_pose = trace_lattice_sample_path(path, progress=0.49, turn_phase_ratio=0.35)
    dwell_pose = trace_lattice_sample_path(path, progress=0.54, turn_phase_ratio=0.35)
    pivot_pose = trace_lattice_sample_path(path, progress=0.74, turn_phase_ratio=0.35)
    after_pivot_pose = trace_lattice_sample_path(path, progress=0.93, turn_phase_ratio=0.35)

    assert node_pose.position == pytest.approx((0.0, 4.0, 0.0), abs=1e-6)
    assert node_pose.forward == pytest.approx((0.0, 1.0, 0.0), abs=1e-6)
    assert dwell_pose.position == pytest.approx(node_pose.position)
    assert dwell_pose.forward == pytest.approx(node_pose.forward, abs=1e-6)
    assert pivot_pose.position == pytest.approx(node_pose.position)
    assert 0.0 < pivot_pose.forward[0] < 1.0
    assert after_pivot_pose.position == pytest.approx(node_pose.position)
    assert after_pivot_pose.forward == pytest.approx((1.0, 0.0, 0.0), abs=1e-6)


def test_trace_lattice_samples_stay_on_axis_aligned_edges_without_curves() -> None:
    path = trace_lattice_build_path(
        start_state=trace_lattice_center_state(),
        actions=(
            TraceLatticeAction.RIGHT,
            TraceLatticeAction.STRAIGHT,
            TraceLatticeAction.PULL,
            TraceLatticeAction.STRAIGHT,
        ),
    )
    samples = [trace_lattice_sample_path(path, progress=idx / 80.0).position for idx in range(81)]

    for position in samples:
        integer_axes = sum(
            1
            for component in position
            if abs(float(component) - round(float(component))) <= 1e-6
        )
        assert integer_axes >= 2

    for start, end in zip(samples, samples[1:], strict=False):
        changed_axes = sum(
            1
            for left, right in zip(start, end, strict=True)
            if abs(float(right) - float(left)) > 1e-6
        )
        assert changed_axes <= 1


def test_trace_lattice_turns_are_exact_right_angles_at_nodes() -> None:
    path = trace_lattice_build_path(
        start_state=trace_lattice_center_state(),
        actions=(TraceLatticeAction.RIGHT, TraceLatticeAction.PULL),
    )

    first = path.steps[0]
    second = path.steps[1]

    assert (first.end_state.node.col, first.end_state.node.row, first.end_state.node.level) == (3, 4, 2)
    assert first.travel_orientation.forward == (0, 1, 0)
    assert first.rotated_orientation.forward == (1, 0, 0)
    assert sum(
        a * b
        for a, b in zip(
            first.travel_orientation.forward,
            first.rotated_orientation.forward,
            strict=True,
        )
    ) == 0

    assert second.travel_orientation.forward == (1, 0, 0)
    assert second.rotated_orientation.forward == (0, 0, 1)
    assert sum(
        a * b
        for a, b in zip(
            second.travel_orientation.forward,
            second.rotated_orientation.forward,
            strict=True,
        )
    ) == 0


def test_trace_lattice_motion_player_moves_holds_then_rotates_one_cell() -> None:
    player = TraceLatticeMotionPlayer(
        start_state=trace_lattice_center_state(),
        actions=(TraceLatticeAction.RIGHT,),
        config=TraceLatticeMotionConfig(
            command_duration_s=2.0,
            turn_phase_ratio=0.25,
            max_update_dt_s=10.0,
        ),
    )

    halfway_translation = player.update(0.50)
    assert halfway_translation.phase is TraceLatticeMotionPhase.TRANSLATING
    assert halfway_translation.pose.position[1] > 3.0
    assert halfway_translation.pose.forward == pytest.approx((0.0, 1.0, 0.0), abs=1e-6)

    node_hold = player.update(0.68)
    assert node_hold.phase is TraceLatticeMotionPhase.HOLDING
    assert node_hold.pose.position == pytest.approx((0.0, 4.0, 0.0))
    assert node_hold.pose.forward == pytest.approx((0.0, 1.0, 0.0), abs=1e-6)

    halfway_rotation = player.update(0.41)
    assert halfway_rotation.phase is TraceLatticeMotionPhase.ROTATING
    assert halfway_rotation.pose.position == pytest.approx((0.0, 4.0, 0.0))
    assert halfway_rotation.pose.forward[0] > 0.0
    assert halfway_rotation.pose.forward[1] > 0.0

    finished_rotation = player.update(0.25)
    assert finished_rotation.phase is TraceLatticeMotionPhase.HOLDING
    assert finished_rotation.pose.position == pytest.approx((0.0, 4.0, 0.0))
    assert finished_rotation.pose.forward == pytest.approx((1.0, 0.0, 0.0), abs=1e-6)

    complete = player.update(0.16)
    assert complete.phase is TraceLatticeMotionPhase.COMPLETE
    assert complete.pose.position == pytest.approx((0.0, 4.0, 0.0))
    assert complete.pose.forward == pytest.approx((1.0, 0.0, 0.0), abs=1e-6)
    assert complete.completed_commands == 1


def test_trace_lattice_motion_player_queues_commands_sequentially() -> None:
    player = TraceLatticeMotionPlayer(
        start_state=trace_lattice_center_state(),
        actions=(TraceLatticeAction.RIGHT, TraceLatticeAction.LEFT),
        config=TraceLatticeMotionConfig(
            command_duration_s=1.0,
            turn_phase_ratio=0.25,
            max_update_dt_s=10.0,
        ),
    )

    after_first_command = player.update(1.0)
    assert after_first_command.completed_commands == 1
    assert after_first_command.phase is TraceLatticeMotionPhase.TRANSLATING
    assert after_first_command.pose.active_step_index == 1
    assert after_first_command.pose.position == pytest.approx((0.0, 4.0, 0.0))

    finished = player.update(1.0)
    assert finished.phase is TraceLatticeMotionPhase.COMPLETE
    assert finished.pose.position == pytest.approx((1.0, 4.0, 0.0))
    assert finished.completed_commands == 2


def test_trace_lattice_motion_player_clamps_large_delta_to_one_command() -> None:
    player = TraceLatticeMotionPlayer(
        start_state=trace_lattice_center_state(),
        actions=(
            TraceLatticeAction.RIGHT,
            TraceLatticeAction.STRAIGHT,
            TraceLatticeAction.STRAIGHT,
        ),
        config=TraceLatticeMotionConfig(
            command_duration_s=1.0,
            turn_phase_ratio=0.25,
            max_update_dt_s=1.0,
        ),
    )

    snapshot = player.update(10.0)

    assert snapshot.completed_commands == 1
    assert snapshot.pose.active_step_index == 1
    assert snapshot.pose.position == pytest.approx((0.0, 4.0, 0.0))


def test_trace_lattice_motion_player_uses_boundary_override_commands() -> None:
    player = TraceLatticeMotionPlayer(
        start_state=trace_lattice_state(col=3, row=3, level=0, forward=(0, 1, 0), up=(0, 0, 1)),
        actions=(TraceLatticeAction.STRAIGHT,),
        config=TraceLatticeMotionConfig(
            command_duration_s=1.0,
            turn_phase_ratio=0.25,
            max_update_dt_s=10.0,
        ),
    )

    assert player.path.steps[0].effective_action is TraceLatticeAction.PULL
    complete = player.update(1.0)

    assert complete.phase is TraceLatticeMotionPhase.COMPLETE
    assert complete.pose.position == pytest.approx((0.0, 3.0, -1.0))
    assert player.path.end_state.node.level == 1


def test_trace_lattice_boundary_overrides_force_inward_actions() -> None:
    bottom = trace_lattice_execute_step(
        trace_lattice_state(col=3, row=3, level=0, forward=(0, 1, 0), up=(0, 0, 1)),
        action=TraceLatticeAction.STRAIGHT,
    )
    top = trace_lattice_execute_step(
        trace_lattice_state(col=3, row=3, level=4, forward=(0, 1, 0), up=(0, 0, 1)),
        action=TraceLatticeAction.STRAIGHT,
    )
    left = trace_lattice_execute_step(
        trace_lattice_state(col=0, row=3, level=2, forward=(0, 1, 0), up=(0, 0, 1)),
        action=TraceLatticeAction.STRAIGHT,
    )
    right = trace_lattice_execute_step(
        trace_lattice_state(col=6, row=3, level=2, forward=(0, 1, 0), up=(0, 0, 1)),
        action=TraceLatticeAction.STRAIGHT,
    )
    near = trace_lattice_execute_step(
        trace_lattice_state(col=3, row=0, level=2, forward=(1, 0, 0), up=(0, 0, 1)),
        action=TraceLatticeAction.STRAIGHT,
    )
    far = trace_lattice_execute_step(
        trace_lattice_state(col=3, row=6, level=2, forward=(1, 0, 0), up=(0, 0, 1)),
        action=TraceLatticeAction.STRAIGHT,
    )
    far_outward = trace_lattice_execute_step(
        trace_lattice_state(col=3, row=6, level=2, forward=(0, 1, 0), up=(0, 0, 1)),
        action=TraceLatticeAction.STRAIGHT,
    )

    assert bottom.effective_action is TraceLatticeAction.PULL
    assert bottom.override_reason == "bottom"
    assert bottom.end_state.node.level == 1

    assert top.effective_action is TraceLatticeAction.PUSH
    assert top.override_reason == "top"
    assert top.end_state.node.level == 3

    assert left.effective_action is TraceLatticeAction.RIGHT
    assert left.override_reason == "left"
    assert left.end_state.node.col == 1

    assert right.effective_action is TraceLatticeAction.LEFT
    assert right.override_reason == "right"
    assert right.end_state.node.col == 5

    assert near.override_reason == "near"
    assert near.end_state.node.row == 1

    assert far.override_reason == "far"
    assert far.end_state.node.row == 5

    assert far_outward.override_reason == "far"
    assert far_outward.end_state.node.row == 5
    assert far_outward.travel_orientation.forward == (0, -1, 0)


def test_trace_lattice_corner_recovery_remains_in_bounds() -> None:
    path = trace_lattice_build_path(
        start_state=trace_lattice_state(col=0, row=0, level=0, forward=(-1, 0, 0), up=(0, 0, 1)),
        actions=(TraceLatticeAction.STRAIGHT,) * 4,
    )

    for step in path.steps:
        assert step.end_state.node.col in range(DEFAULT_TRACE_LATTICE_SPEC.cols)
        assert step.end_state.node.row in range(DEFAULT_TRACE_LATTICE_SPEC.rows)
        assert step.end_state.node.level in range(DEFAULT_TRACE_LATTICE_SPEC.levels)

    assert path.steps[0].override_reason == "bottom"
    assert path.steps[1].override_reason == "left"
    assert path.end_state.node.col >= 1
    assert path.end_state.node.row >= 1
