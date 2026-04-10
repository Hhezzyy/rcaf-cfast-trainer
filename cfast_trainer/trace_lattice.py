from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum


Int3 = tuple[int, int, int]
Point3 = tuple[float, float, float]


class TraceLatticeAction(StrEnum):
    STRAIGHT = "straight"
    LEFT = "left"
    RIGHT = "right"
    PUSH = "push"
    PULL = "pull"


@dataclass(frozen=True, slots=True)
class TraceLatticeNode:
    col: int
    row: int
    level: int


@dataclass(frozen=True, slots=True)
class TraceLatticeOrientation:
    forward: Int3
    up: Int3


@dataclass(frozen=True, slots=True)
class TraceLatticeState:
    node: TraceLatticeNode
    orientation: TraceLatticeOrientation


@dataclass(frozen=True, slots=True)
class TraceLatticeSpec:
    cols: int = 7
    rows: int = 7
    levels: int = 5


@dataclass(frozen=True, slots=True)
class TraceLatticeStep:
    start_state: TraceLatticeState
    requested_action: TraceLatticeAction
    effective_action: TraceLatticeAction
    rotated_orientation: TraceLatticeOrientation
    end_state: TraceLatticeState
    overridden: bool = False
    override_reason: str | None = None


@dataclass(frozen=True, slots=True)
class TraceLatticePath:
    spec: TraceLatticeSpec
    start_state: TraceLatticeState
    steps: tuple[TraceLatticeStep, ...]

    @property
    def end_state(self) -> TraceLatticeState:
        if not self.steps:
            return self.start_state
        return self.steps[-1].end_state


@dataclass(frozen=True, slots=True)
class TraceLatticePose:
    position: Point3
    forward: Point3
    up: Point3
    active_step_index: int
    effective_action: TraceLatticeAction
    rotated: bool


DEFAULT_TRACE_LATTICE_SPEC = TraceLatticeSpec(cols=7, rows=7, levels=5)
_TRACE_LATTICE_TURN_CURVE_TANGENT_SCALE = 0.4


def _cross(a: Point3, b: Point3) -> Point3:
    return (
        (float(a[1]) * float(b[2])) - (float(a[2]) * float(b[1])),
        (float(a[2]) * float(b[0])) - (float(a[0]) * float(b[2])),
        (float(a[0]) * float(b[1])) - (float(a[1]) * float(b[0])),
    )


def _normalize(v: Point3) -> Point3:
    norm = math.sqrt((float(v[0]) ** 2) + (float(v[1]) ** 2) + (float(v[2]) ** 2))
    if norm <= 1e-8:
        return (0.0, 0.0, 0.0)
    return (float(v[0]) / norm, float(v[1]) / norm, float(v[2]) / norm)


def _neg(v: Int3) -> Int3:
    return (-int(v[0]), -int(v[1]), -int(v[2]))


def _add_node(node: TraceLatticeNode, forward: Int3) -> TraceLatticeNode:
    return TraceLatticeNode(
        col=int(node.col) + int(forward[0]),
        row=int(node.row) + int(forward[1]),
        level=int(node.level) + int(forward[2]),
    )


def trace_lattice_right(orientation: TraceLatticeOrientation) -> Int3:
    right = _cross(orientation.forward, orientation.up)
    return (int(round(right[0])), int(round(right[1])), int(round(right[2])))


def trace_lattice_center_state(
    *,
    spec: TraceLatticeSpec = DEFAULT_TRACE_LATTICE_SPEC,
    facing: Int3 = (0, 1, 0),
    up: Int3 = (0, 0, 1),
) -> TraceLatticeState:
    return TraceLatticeState(
        node=TraceLatticeNode(
            col=int(spec.cols // 2),
            row=int(spec.rows // 2),
            level=int(spec.levels // 2),
        ),
        orientation=TraceLatticeOrientation(forward=facing, up=up),
    )


def trace_lattice_state(
    *,
    col: int,
    row: int,
    level: int,
    forward: Int3 = (0, 1, 0),
    up: Int3 = (0, 0, 1),
) -> TraceLatticeState:
    return TraceLatticeState(
        node=TraceLatticeNode(col=int(col), row=int(row), level=int(level)),
        orientation=TraceLatticeOrientation(forward=forward, up=up),
    )


def trace_lattice_rotate(
    orientation: TraceLatticeOrientation,
    action: TraceLatticeAction,
) -> TraceLatticeOrientation:
    if action is TraceLatticeAction.STRAIGHT:
        return orientation
    forward = orientation.forward
    up = orientation.up
    right = trace_lattice_right(orientation)
    if action is TraceLatticeAction.LEFT:
        return TraceLatticeOrientation(forward=_neg(right), up=up)
    if action is TraceLatticeAction.RIGHT:
        return TraceLatticeOrientation(forward=right, up=up)
    if action is TraceLatticeAction.PULL:
        return TraceLatticeOrientation(forward=up, up=_neg(forward))
    return TraceLatticeOrientation(forward=_neg(up), up=forward)


def _orientation_faces_inward_row(
    orientation: TraceLatticeOrientation,
    *,
    desired_row_delta: int,
) -> bool:
    return int(orientation.forward[1]) == int(desired_row_delta)


def _yaw_override_toward_row(
    orientation: TraceLatticeOrientation,
    *,
    desired_row_delta: int,
) -> TraceLatticeAction:
    candidates = (TraceLatticeAction.LEFT, TraceLatticeAction.RIGHT)
    scores: list[tuple[int, int, TraceLatticeAction]] = []
    for idx, action in enumerate(candidates):
        rotated = trace_lattice_rotate(orientation, action)
        row_delta = int(rotated.forward[1])
        scores.append((abs(desired_row_delta - row_delta), idx, action))
        if row_delta == int(desired_row_delta):
            return action
    scores.sort(key=lambda item: (item[0], item[1]))
    return scores[0][2]


def trace_lattice_boundary_override(
    state: TraceLatticeState,
    *,
    spec: TraceLatticeSpec = DEFAULT_TRACE_LATTICE_SPEC,
) -> tuple[TraceLatticeAction, str] | None:
    node = state.node
    if int(node.level) <= 0:
        return TraceLatticeAction.PULL, "bottom"
    if int(node.level) >= int(spec.levels - 1):
        return TraceLatticeAction.PUSH, "top"
    if int(node.col) <= 0:
        return TraceLatticeAction.RIGHT, "left"
    if int(node.col) >= int(spec.cols - 1):
        return TraceLatticeAction.LEFT, "right"
    if int(node.row) <= 0:
        return _yaw_override_toward_row(state.orientation, desired_row_delta=1), "near"
    if int(node.row) >= int(spec.rows - 1):
        return _yaw_override_toward_row(state.orientation, desired_row_delta=-1), "far"
    return None


def trace_lattice_execute_step(
    state: TraceLatticeState,
    *,
    action: TraceLatticeAction,
    spec: TraceLatticeSpec = DEFAULT_TRACE_LATTICE_SPEC,
) -> TraceLatticeStep:
    override = trace_lattice_boundary_override(state, spec=spec)
    effective = action if override is None else override[0]
    rotated = trace_lattice_rotate(state.orientation, effective)
    end_state = TraceLatticeState(
        node=_add_node(state.node, rotated.forward),
        orientation=rotated,
    )
    return TraceLatticeStep(
        start_state=state,
        requested_action=action,
        effective_action=effective,
        rotated_orientation=rotated,
        end_state=end_state,
        overridden=override is not None,
        override_reason=None if override is None else override[1],
    )


def trace_lattice_build_path(
    *,
    start_state: TraceLatticeState,
    actions: tuple[TraceLatticeAction, ...],
    spec: TraceLatticeSpec = DEFAULT_TRACE_LATTICE_SPEC,
) -> TraceLatticePath:
    state = start_state
    steps: list[TraceLatticeStep] = []
    for action in actions:
        step = trace_lattice_execute_step(state, action=action, spec=spec)
        steps.append(step)
        state = step.end_state
    return TraceLatticePath(
        spec=spec,
        start_state=start_state,
        steps=tuple(steps),
    )


def trace_lattice_node_point(
    node: TraceLatticeNode,
    *,
    spec: TraceLatticeSpec = DEFAULT_TRACE_LATTICE_SPEC,
) -> Point3:
    return (
        float(node.col) - (float(spec.cols - 1) * 0.5),
        float(node.row),
        float(node.level) - (float(spec.levels - 1) * 0.5),
    )


def _rotate_vector_about_axis(
    vector: Point3,
    axis: Point3,
    *,
    degrees: float,
) -> Point3:
    radians = math.radians(float(degrees))
    cos_a = math.cos(radians)
    sin_a = math.sin(radians)
    axis_n = _normalize(axis)
    dot = (
        (float(vector[0]) * float(axis_n[0]))
        + (float(vector[1]) * float(axis_n[1]))
        + (float(vector[2]) * float(axis_n[2]))
    )
    cross = _cross(axis_n, vector)
    return (
        (float(vector[0]) * cos_a)
        + (cross[0] * sin_a)
        + (axis_n[0] * dot * (1.0 - cos_a)),
        (float(vector[1]) * cos_a)
        + (cross[1] * sin_a)
        + (axis_n[1] * dot * (1.0 - cos_a)),
        (float(vector[2]) * cos_a)
        + (cross[2] * sin_a)
        + (axis_n[2] * dot * (1.0 - cos_a)),
    )


def _interpolated_orientation(
    step: TraceLatticeStep,
    *,
    turn_progress: float,
) -> tuple[Point3, Point3]:
    start_forward = tuple(float(v) for v in step.start_state.orientation.forward)
    start_up = tuple(float(v) for v in step.start_state.orientation.up)
    if step.effective_action is TraceLatticeAction.STRAIGHT:
        return start_forward, start_up

    if step.effective_action is TraceLatticeAction.LEFT:
        axis = tuple(float(v) for v in step.start_state.orientation.up)
        angle_deg = 90.0 * float(turn_progress)
        return (
            _rotate_vector_about_axis(start_forward, axis, degrees=angle_deg),
            start_up,
        )
    if step.effective_action is TraceLatticeAction.RIGHT:
        axis = tuple(float(v) for v in step.start_state.orientation.up)
        angle_deg = -90.0 * float(turn_progress)
        return (
            _rotate_vector_about_axis(start_forward, axis, degrees=angle_deg),
            start_up,
        )

    axis = tuple(float(v) for v in trace_lattice_right(step.start_state.orientation))
    angle_deg = 90.0 * float(turn_progress)
    if step.effective_action is TraceLatticeAction.PUSH:
        angle_deg *= -1.0
    return (
        _rotate_vector_about_axis(start_forward, axis, degrees=angle_deg),
        _rotate_vector_about_axis(start_up, axis, degrees=angle_deg),
    )


def _point_lerp(a: Point3, b: Point3, t: float) -> Point3:
    return (
        float(a[0] + ((b[0] - a[0]) * t)),
        float(a[1] + ((b[1] - a[1]) * t)),
        float(a[2] + ((b[2] - a[2]) * t)),
    )


def _cubic_hermite_point(
    start_point: Point3,
    end_point: Point3,
    start_tangent: Point3,
    end_tangent: Point3,
    t: float,
) -> Point3:
    h00 = (2.0 * t * t * t) - (3.0 * t * t) + 1.0
    h10 = (t * t * t) - (2.0 * t * t) + t
    h01 = (-2.0 * t * t * t) + (3.0 * t * t)
    h11 = (t * t * t) - (t * t)
    return (
        (h00 * float(start_point[0]))
        + (h10 * float(start_tangent[0]))
        + (h01 * float(end_point[0]))
        + (h11 * float(end_tangent[0])),
        (h00 * float(start_point[1]))
        + (h10 * float(start_tangent[1]))
        + (h01 * float(end_point[1]))
        + (h11 * float(end_tangent[1])),
        (h00 * float(start_point[2]))
        + (h10 * float(start_tangent[2]))
        + (h01 * float(end_point[2]))
        + (h11 * float(end_tangent[2])),
    )


def trace_lattice_sample_path(
    path: TraceLatticePath,
    *,
    progress: float,
    turn_phase_ratio: float = 0.35,
) -> TraceLatticePose:
    if not path.steps:
        node_point = trace_lattice_node_point(path.start_state.node, spec=path.spec)
        forward = tuple(float(v) for v in path.start_state.orientation.forward)
        up = tuple(float(v) for v in path.start_state.orientation.up)
        return TraceLatticePose(
            position=node_point,
            forward=forward,
            up=up,
            active_step_index=0,
            effective_action=TraceLatticeAction.STRAIGHT,
            rotated=False,
        )

    t = max(0.0, min(1.0, float(progress)))
    step_count = len(path.steps)
    if t >= 1.0:
        step = path.steps[-1]
        node_point = trace_lattice_node_point(step.end_state.node, spec=path.spec)
        return TraceLatticePose(
            position=node_point,
            forward=tuple(float(v) for v in step.end_state.orientation.forward),
            up=tuple(float(v) for v in step.end_state.orientation.up),
            active_step_index=step_count - 1,
            effective_action=step.effective_action,
            rotated=True,
        )

    scaled = t * float(step_count)
    index = min(step_count - 1, int(math.floor(scaled)))
    local_t = scaled - float(index)
    step = path.steps[index]
    start_point = trace_lattice_node_point(step.start_state.node, spec=path.spec)
    end_point = trace_lattice_node_point(step.end_state.node, spec=path.spec)
    if step.effective_action is TraceLatticeAction.STRAIGHT:
        position = _point_lerp(start_point, end_point, local_t)
        forward = tuple(float(v) for v in step.rotated_orientation.forward)
        up = tuple(float(v) for v in step.rotated_orientation.up)
        rotated = True
    else:
        _ = turn_phase_ratio
        start_forward = tuple(float(v) for v in step.start_state.orientation.forward)
        end_forward = tuple(float(v) for v in step.rotated_orientation.forward)
        position = _cubic_hermite_point(
            start_point,
            end_point,
            tuple(
                float(component) * float(_TRACE_LATTICE_TURN_CURVE_TANGENT_SCALE)
                for component in start_forward
            ),
            tuple(
                float(component) * float(_TRACE_LATTICE_TURN_CURVE_TANGENT_SCALE)
                for component in end_forward
            ),
            local_t,
        )
        forward, up = _interpolated_orientation(step, turn_progress=local_t)
        rotated = local_t > 0.0

    return TraceLatticePose(
        position=position,
        forward=_normalize(forward),
        up=_normalize(up),
        active_step_index=index,
        effective_action=step.effective_action,
        rotated=rotated,
    )
