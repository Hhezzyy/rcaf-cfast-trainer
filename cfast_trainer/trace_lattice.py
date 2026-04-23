from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum


Int3 = tuple[int, int, int]
Point3 = tuple[float, float, float]
DEFAULT_TRACE_LATTICE_NODE_HOLD_PHASE_RATIO = 0.08


class TraceLatticeAction(StrEnum):
    STRAIGHT = "straight"
    LEFT = "left"
    RIGHT = "right"
    PUSH = "push"
    PULL = "pull"


class TraceLatticeMotionPhase(StrEnum):
    IDLE = "idle"
    TRANSLATING = "translating"
    HOLDING = "holding"
    ROTATING = "rotating"
    COMPLETE = "complete"


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
    travel_orientation: TraceLatticeOrientation
    rotated_orientation: TraceLatticeOrientation
    end_state: TraceLatticeState
    turns_after_translation: bool = True
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


@dataclass(frozen=True, slots=True)
class TraceLatticeMotionConfig:
    command_duration_s: float = 1.0
    turn_phase_ratio: float = 0.35
    node_hold_phase_ratio: float = DEFAULT_TRACE_LATTICE_NODE_HOLD_PHASE_RATIO
    max_update_dt_s: float = 0.25


@dataclass(frozen=True, slots=True)
class TraceLatticeMotionSnapshot:
    pose: TraceLatticePose
    phase: TraceLatticeMotionPhase
    completed_commands: int
    command_count: int
    command_progress: float
    phase_progress: float


DEFAULT_TRACE_LATTICE_SPEC = TraceLatticeSpec(cols=7, rows=7, levels=5)
_TRACE_LATTICE_EPSILON = 1e-9


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


def _dot_int(a: Int3, b: Int3) -> int:
    return (int(a[0]) * int(b[0])) + (int(a[1]) * int(b[1])) + (int(a[2]) * int(b[2]))


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


def _orientation_with_forward(
    orientation: TraceLatticeOrientation,
    *,
    forward: Int3,
) -> TraceLatticeOrientation:
    desired_forward = (
        int(forward[0]),
        int(forward[1]),
        int(forward[2]),
    )
    up_candidates = (
        orientation.up,
        trace_lattice_right(orientation),
        orientation.forward,
        (0, 0, 1),
        (0, 1, 0),
        (1, 0, 0),
    )
    for candidate in up_candidates:
        candidate_up = (
            int(candidate[0]),
            int(candidate[1]),
            int(candidate[2]),
        )
        if candidate_up == (0, 0, 0):
            continue
        if _dot_int(desired_forward, candidate_up) == 0:
            return TraceLatticeOrientation(
                forward=desired_forward,
                up=candidate_up,
            )
    return TraceLatticeOrientation(forward=desired_forward, up=(0, 0, 1))


def _boundary_orientation_toward(
    orientation: TraceLatticeOrientation,
    *,
    desired_forward: Int3,
) -> tuple[TraceLatticeAction, TraceLatticeOrientation]:
    candidates = (
        TraceLatticeAction.STRAIGHT,
        TraceLatticeAction.LEFT,
        TraceLatticeAction.RIGHT,
        TraceLatticeAction.PULL,
        TraceLatticeAction.PUSH,
    )
    desired = (
        int(desired_forward[0]),
        int(desired_forward[1]),
        int(desired_forward[2]),
    )
    scored: list[tuple[int, int, TraceLatticeAction, TraceLatticeOrientation]] = []
    for idx, action in enumerate(candidates):
        rotated = trace_lattice_rotate(orientation, action)
        score = _dot_int(rotated.forward, desired)
        if rotated.forward == desired:
            return action, rotated
        scored.append((score, -idx, action, rotated))
    scored.sort(reverse=True, key=lambda item: (item[0], item[1]))
    best_action = scored[0][2]
    return best_action, _orientation_with_forward(orientation, forward=desired)


def trace_lattice_boundary_override(
    state: TraceLatticeState,
    *,
    spec: TraceLatticeSpec = DEFAULT_TRACE_LATTICE_SPEC,
) -> tuple[TraceLatticeAction, str, TraceLatticeOrientation] | None:
    node = state.node
    if int(node.level) <= 0:
        action, orientation = _boundary_orientation_toward(
            state.orientation,
            desired_forward=(0, 0, 1),
        )
        return action, "bottom", orientation
    if int(node.level) >= int(spec.levels - 1):
        action, orientation = _boundary_orientation_toward(
            state.orientation,
            desired_forward=(0, 0, -1),
        )
        return action, "top", orientation
    if int(node.col) <= 0:
        action, orientation = _boundary_orientation_toward(
            state.orientation,
            desired_forward=(1, 0, 0),
        )
        return action, "left", orientation
    if int(node.col) >= int(spec.cols - 1):
        action, orientation = _boundary_orientation_toward(
            state.orientation,
            desired_forward=(-1, 0, 0),
        )
        return action, "right", orientation
    if int(node.row) <= 0:
        action, orientation = _boundary_orientation_toward(
            state.orientation,
            desired_forward=(0, 1, 0),
        )
        return action, "near", orientation
    if int(node.row) >= int(spec.rows - 1):
        action, orientation = _boundary_orientation_toward(
            state.orientation,
            desired_forward=(0, -1, 0),
        )
        return action, "far", orientation
    return None


def _node_in_bounds(
    node: TraceLatticeNode,
    *,
    spec: TraceLatticeSpec,
) -> bool:
    return (
        0 <= int(node.col) < int(spec.cols)
        and 0 <= int(node.row) < int(spec.rows)
        and 0 <= int(node.level) < int(spec.levels)
    )


def trace_lattice_execute_step(
    state: TraceLatticeState,
    *,
    action: TraceLatticeAction,
    spec: TraceLatticeSpec = DEFAULT_TRACE_LATTICE_SPEC,
) -> TraceLatticeStep:
    override = trace_lattice_boundary_override(state, spec=spec)
    effective = action if override is None else override[0]
    if override is None:
        travel_orientation = state.orientation
        rotated = trace_lattice_rotate(travel_orientation, effective)
        turns_after_translation = effective is not TraceLatticeAction.STRAIGHT
        override_reason = None
    else:
        travel_orientation = override[2]
        rotated = travel_orientation
        turns_after_translation = False
        override_reason = override[1]
    end_node = _add_node(state.node, travel_orientation.forward)
    if not _node_in_bounds(end_node, spec=spec):
        # The boundary override should already have chosen an inward travel vector.
        # This final guard keeps malformed specs or orientations from escaping the grid.
        safe_forward = (0, 1, 0)
        center_delta = (
            int(spec.cols // 2) - int(state.node.col),
            int(spec.rows // 2) - int(state.node.row),
            int(spec.levels // 2) - int(state.node.level),
        )
        for axis, delta in enumerate(center_delta):
            if delta == 0:
                continue
            candidate = [0, 0, 0]
            candidate[axis] = 1 if delta > 0 else -1
            candidate_node = _add_node(state.node, (candidate[0], candidate[1], candidate[2]))
            if _node_in_bounds(candidate_node, spec=spec):
                safe_forward = (candidate[0], candidate[1], candidate[2])
                break
        safe_orientation = _orientation_with_forward(
            travel_orientation,
            forward=safe_forward,
        )
        travel_orientation = safe_orientation
        rotated = safe_orientation if override is not None else trace_lattice_rotate(safe_orientation, effective)
        end_node = _add_node(state.node, travel_orientation.forward)
    end_state = TraceLatticeState(
        node=end_node,
        orientation=rotated,
    )
    return TraceLatticeStep(
        start_state=state,
        requested_action=action,
        effective_action=effective,
        travel_orientation=travel_orientation,
        rotated_orientation=rotated,
        end_state=end_state,
        turns_after_translation=turns_after_translation,
        overridden=override is not None,
        override_reason=override_reason,
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
    start_forward = tuple(float(v) for v in step.travel_orientation.forward)
    start_up = tuple(float(v) for v in step.travel_orientation.up)
    if step.effective_action is TraceLatticeAction.STRAIGHT:
        return start_forward, start_up

    if step.effective_action is TraceLatticeAction.LEFT:
        axis = tuple(float(v) for v in step.travel_orientation.up)
        angle_deg = 90.0 * float(turn_progress)
        return (
            _rotate_vector_about_axis(start_forward, axis, degrees=angle_deg),
            start_up,
        )
    if step.effective_action is TraceLatticeAction.RIGHT:
        axis = tuple(float(v) for v in step.travel_orientation.up)
        angle_deg = -90.0 * float(turn_progress)
        return (
            _rotate_vector_about_axis(start_forward, axis, degrees=angle_deg),
            start_up,
        )

    axis = tuple(float(v) for v in trace_lattice_right(step.travel_orientation))
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


def _clamped_turn_phase_ratio(value: float) -> float:
    return max(0.05, min(0.95, float(value)))


def _clamped_node_hold_phase_ratio(value: float) -> float:
    return max(0.0, min(0.40, float(value)))


def _non_straight_phase_ratios(
    *,
    turn_phase_ratio: float,
    node_hold_phase_ratio: float,
) -> tuple[float, float, float]:
    turn_ratio = _clamped_turn_phase_ratio(turn_phase_ratio)
    requested_hold = _clamped_node_hold_phase_ratio(node_hold_phase_ratio)
    hold_budget = max(0.0, min(requested_hold * 2.0, 0.95 - turn_ratio))
    hold_ratio = hold_budget * 0.5
    translate_ratio = max(_TRACE_LATTICE_EPSILON, 1.0 - turn_ratio - (2.0 * hold_ratio))
    return translate_ratio, hold_ratio, turn_ratio


def _clamped_command_duration_s(value: float) -> float:
    return max(_TRACE_LATTICE_EPSILON, float(value))


def _motion_config(config: TraceLatticeMotionConfig | None) -> TraceLatticeMotionConfig:
    if config is None:
        return TraceLatticeMotionConfig()
    return TraceLatticeMotionConfig(
        command_duration_s=_clamped_command_duration_s(config.command_duration_s),
        turn_phase_ratio=_clamped_turn_phase_ratio(config.turn_phase_ratio),
        node_hold_phase_ratio=_clamped_node_hold_phase_ratio(config.node_hold_phase_ratio),
        max_update_dt_s=max(0.0, float(config.max_update_dt_s)),
    )


def _step_command_progress_from_phase(
    *,
    step: TraceLatticeStep,
    phase: TraceLatticeMotionPhase,
    phase_progress: float,
    turn_phase_ratio: float,
    node_hold_phase_ratio: float,
    hold_after_rotation: bool,
) -> float:
    phase_t = max(0.0, min(1.0, float(phase_progress)))
    if phase is TraceLatticeMotionPhase.COMPLETE:
        return 1.0
    if step.effective_action is TraceLatticeAction.STRAIGHT or not step.turns_after_translation:
        return phase_t if phase is TraceLatticeMotionPhase.TRANSLATING else 0.0
    translate_ratio, hold_ratio, turn_ratio = _non_straight_phase_ratios(
        turn_phase_ratio=turn_phase_ratio,
        node_hold_phase_ratio=node_hold_phase_ratio,
    )
    if phase is TraceLatticeMotionPhase.TRANSLATING:
        return translate_ratio * phase_t
    if phase is TraceLatticeMotionPhase.HOLDING:
        if hold_after_rotation:
            return translate_ratio + hold_ratio + turn_ratio + (hold_ratio * phase_t)
        return translate_ratio + (hold_ratio * phase_t)
    if phase is TraceLatticeMotionPhase.ROTATING:
        return translate_ratio + hold_ratio + (turn_ratio * phase_t)
    return 0.0


def _trace_lattice_step_pose(
    *,
    step: TraceLatticeStep,
    spec: TraceLatticeSpec,
    local_t: float,
    step_index: int,
    turn_phase_ratio: float,
    node_hold_phase_ratio: float = DEFAULT_TRACE_LATTICE_NODE_HOLD_PHASE_RATIO,
) -> TraceLatticePose:
    start_point = trace_lattice_node_point(step.start_state.node, spec=spec)
    end_point = trace_lattice_node_point(step.end_state.node, spec=spec)
    progress = max(0.0, min(1.0, float(local_t)))
    if step.effective_action is TraceLatticeAction.STRAIGHT or not step.turns_after_translation:
        position = _point_lerp(start_point, end_point, progress)
        forward = tuple(float(v) for v in step.travel_orientation.forward)
        up = tuple(float(v) for v in step.travel_orientation.up)
        rotated = step.travel_orientation != step.start_state.orientation
    else:
        translate_ratio, hold_ratio, turn_ratio = _non_straight_phase_ratios(
            turn_phase_ratio=turn_phase_ratio,
            node_hold_phase_ratio=node_hold_phase_ratio,
        )
        turn_start = translate_ratio + hold_ratio
        turn_end = turn_start + turn_ratio
        if progress <= translate_ratio:
            move_t = progress / max(_TRACE_LATTICE_EPSILON, translate_ratio)
            position = _point_lerp(start_point, end_point, move_t)
            forward = tuple(float(v) for v in step.travel_orientation.forward)
            up = tuple(float(v) for v in step.travel_orientation.up)
            rotated = False
        elif progress <= turn_start:
            position = end_point
            forward = tuple(float(v) for v in step.travel_orientation.forward)
            up = tuple(float(v) for v in step.travel_orientation.up)
            rotated = False
        elif progress <= turn_end:
            position = end_point
            pivot_t = (progress - turn_start) / max(_TRACE_LATTICE_EPSILON, turn_ratio)
            forward, up = _interpolated_orientation(step, turn_progress=pivot_t)
            rotated = progress > 0.0
        else:
            position = end_point
            forward = tuple(float(v) for v in step.rotated_orientation.forward)
            up = tuple(float(v) for v in step.rotated_orientation.up)
            rotated = True

    return TraceLatticePose(
        position=position,
        forward=_normalize(forward),
        up=_normalize(up),
        active_step_index=int(step_index),
        effective_action=step.effective_action,
        rotated=rotated,
    )


def _final_path_pose(path: TraceLatticePath) -> TraceLatticePose:
    if not path.steps:
        node_point = trace_lattice_node_point(path.start_state.node, spec=path.spec)
        return TraceLatticePose(
            position=node_point,
            forward=tuple(float(v) for v in path.start_state.orientation.forward),
            up=tuple(float(v) for v in path.start_state.orientation.up),
            active_step_index=0,
            effective_action=TraceLatticeAction.STRAIGHT,
            rotated=False,
        )
    step = path.steps[-1]
    return TraceLatticePose(
        position=trace_lattice_node_point(step.end_state.node, spec=path.spec),
        forward=tuple(float(v) for v in step.end_state.orientation.forward),
        up=tuple(float(v) for v in step.end_state.orientation.up),
        active_step_index=len(path.steps) - 1,
        effective_action=step.effective_action,
        rotated=True,
    )


class TraceLatticeMotionPlayer:
    def __init__(
        self,
        *,
        start_state: TraceLatticeState | None = None,
        actions: tuple[TraceLatticeAction, ...] = (),
        spec: TraceLatticeSpec = DEFAULT_TRACE_LATTICE_SPEC,
        path: TraceLatticePath | None = None,
        config: TraceLatticeMotionConfig | None = None,
    ) -> None:
        if path is None:
            if start_state is None:
                raise ValueError("start_state is required when path is not provided")
            path = trace_lattice_build_path(
                start_state=start_state,
                actions=tuple(actions),
                spec=spec,
            )
        self._path = path
        self._config = _motion_config(config)
        self._step_index = 0
        self._phase_elapsed_s = 0.0
        self._phase = self._initial_phase_for_step(0)
        self._hold_after_rotation = False

    @property
    def path(self) -> TraceLatticePath:
        return self._path

    @property
    def completed(self) -> bool:
        return self._phase is TraceLatticeMotionPhase.COMPLETE

    def snapshot(self) -> TraceLatticeMotionSnapshot:
        command_count = len(self._path.steps)
        if command_count == 0:
            return TraceLatticeMotionSnapshot(
                pose=_final_path_pose(self._path),
                phase=TraceLatticeMotionPhase.IDLE,
                completed_commands=0,
                command_count=0,
                command_progress=1.0,
                phase_progress=1.0,
            )
        if self._phase is TraceLatticeMotionPhase.COMPLETE:
            return TraceLatticeMotionSnapshot(
                pose=_final_path_pose(self._path),
                phase=TraceLatticeMotionPhase.COMPLETE,
                completed_commands=command_count,
                command_count=command_count,
                command_progress=1.0,
                phase_progress=1.0,
            )

        step = self._path.steps[self._step_index]
        duration = self._phase_duration_s(step=step, phase=self._phase)
        phase_progress = 1.0 if duration <= _TRACE_LATTICE_EPSILON else self._phase_elapsed_s / duration
        command_progress = _step_command_progress_from_phase(
            step=step,
            phase=self._phase,
            phase_progress=phase_progress,
            turn_phase_ratio=self._config.turn_phase_ratio,
            node_hold_phase_ratio=self._config.node_hold_phase_ratio,
            hold_after_rotation=self._hold_after_rotation,
        )
        return TraceLatticeMotionSnapshot(
            pose=_trace_lattice_step_pose(
                step=step,
                spec=self._path.spec,
                local_t=command_progress,
                step_index=self._step_index,
                turn_phase_ratio=self._config.turn_phase_ratio,
                node_hold_phase_ratio=self._config.node_hold_phase_ratio,
            ),
            phase=self._phase,
            completed_commands=self._step_index,
            command_count=command_count,
            command_progress=max(0.0, min(1.0, float(command_progress))),
            phase_progress=max(0.0, min(1.0, float(phase_progress))),
        )

    def update(self, dt_s: float) -> TraceLatticeMotionSnapshot:
        if self.completed or not self._path.steps:
            return self.snapshot()
        remaining = max(0.0, float(dt_s))
        max_dt = float(self._config.max_update_dt_s)
        if max_dt > 0.0:
            remaining = min(remaining, max_dt)
        while remaining > _TRACE_LATTICE_EPSILON and not self.completed:
            step = self._path.steps[self._step_index]
            duration = self._phase_duration_s(step=step, phase=self._phase)
            if duration <= _TRACE_LATTICE_EPSILON:
                self._advance_phase()
                continue
            phase_remaining = max(0.0, duration - self._phase_elapsed_s)
            consumed = min(remaining, phase_remaining)
            self._phase_elapsed_s += consumed
            remaining -= consumed
            if self._phase_elapsed_s + _TRACE_LATTICE_EPSILON >= duration:
                self._phase_elapsed_s = duration
                self._advance_phase()
        return self.snapshot()

    def _initial_phase_for_step(self, step_index: int) -> TraceLatticeMotionPhase:
        if step_index >= len(self._path.steps):
            return TraceLatticeMotionPhase.COMPLETE
        return TraceLatticeMotionPhase.TRANSLATING

    def _phase_duration_s(
        self,
        *,
        step: TraceLatticeStep,
        phase: TraceLatticeMotionPhase,
    ) -> float:
        command_duration = _clamped_command_duration_s(self._config.command_duration_s)
        if step.effective_action is TraceLatticeAction.STRAIGHT or not step.turns_after_translation:
            return command_duration if phase is TraceLatticeMotionPhase.TRANSLATING else 0.0
        translate_ratio, hold_ratio, turn_ratio = _non_straight_phase_ratios(
            turn_phase_ratio=self._config.turn_phase_ratio,
            node_hold_phase_ratio=self._config.node_hold_phase_ratio,
        )
        if phase is TraceLatticeMotionPhase.TRANSLATING:
            return command_duration * translate_ratio
        if phase is TraceLatticeMotionPhase.HOLDING:
            return command_duration * hold_ratio
        if phase is TraceLatticeMotionPhase.ROTATING:
            return command_duration * turn_ratio
        return 0.0

    def _advance_phase(self) -> None:
        if self._phase is TraceLatticeMotionPhase.TRANSLATING:
            step = self._path.steps[self._step_index]
            if step.effective_action is not TraceLatticeAction.STRAIGHT and step.turns_after_translation:
                self._phase = TraceLatticeMotionPhase.HOLDING
                self._hold_after_rotation = False
                self._phase_elapsed_s = 0.0
                return
            self._step_index += 1
            self._phase = self._initial_phase_for_step(self._step_index)
            self._hold_after_rotation = False
            self._phase_elapsed_s = 0.0
            return
        if self._phase is TraceLatticeMotionPhase.HOLDING:
            if self._hold_after_rotation:
                self._step_index += 1
                self._phase = self._initial_phase_for_step(self._step_index)
                self._hold_after_rotation = False
            else:
                self._phase = TraceLatticeMotionPhase.ROTATING
            self._phase_elapsed_s = 0.0
            return
        if self._phase is TraceLatticeMotionPhase.ROTATING:
            self._phase = TraceLatticeMotionPhase.HOLDING
            self._hold_after_rotation = True
            self._phase_elapsed_s = 0.0
            return
        if self._phase is TraceLatticeMotionPhase.COMPLETE:
            self._phase_elapsed_s = 0.0
            return
        if self._phase is TraceLatticeMotionPhase.IDLE:
            self._step_index += 1
            self._phase = self._initial_phase_for_step(self._step_index)
            self._phase_elapsed_s = 0.0
            return
        self._phase = TraceLatticeMotionPhase.COMPLETE
        self._phase_elapsed_s = 0.0


def trace_lattice_sample_path(
    path: TraceLatticePath,
    *,
    progress: float,
    turn_phase_ratio: float = 0.35,
    node_hold_phase_ratio: float = DEFAULT_TRACE_LATTICE_NODE_HOLD_PHASE_RATIO,
) -> TraceLatticePose:
    if not path.steps:
        return _final_path_pose(path)

    t = max(0.0, min(1.0, float(progress)))
    step_count = len(path.steps)
    if t >= 1.0:
        return _final_path_pose(path)

    scaled = t * float(step_count)
    index = min(step_count - 1, int(math.floor(scaled)))
    local_t = scaled - float(index)
    return _trace_lattice_step_pose(
        step=path.steps[index],
        spec=path.spec,
        local_t=local_t,
        step_index=index,
        turn_phase_ratio=turn_phase_ratio,
        node_hold_phase_ratio=node_hold_phase_ratio,
    )
