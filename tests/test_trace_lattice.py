from __future__ import annotations

import pytest

from cfast_trainer.trace_lattice import (
    DEFAULT_TRACE_LATTICE_SPEC,
    TraceLatticeAction,
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


def test_trace_lattice_samples_rotate_before_translating() -> None:
    path = trace_lattice_build_path(
        start_state=trace_lattice_center_state(),
        actions=(TraceLatticeAction.RIGHT,),
    )

    early_pose = trace_lattice_sample_path(path, progress=0.10, turn_phase_ratio=0.35)
    late_pose = trace_lattice_sample_path(path, progress=0.70, turn_phase_ratio=0.35)

    assert early_pose.position == pytest.approx((0.0, 3.0, 0.0))
    assert early_pose.rotated is False
    assert early_pose.forward[0] > 0.0
    assert early_pose.forward[1] > 0.0

    assert late_pose.position[0] > early_pose.position[0]
    assert late_pose.position[1] == pytest.approx(3.0)
    assert late_pose.rotated is True
    assert late_pose.forward == pytest.approx((1.0, 0.0, 0.0))


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
