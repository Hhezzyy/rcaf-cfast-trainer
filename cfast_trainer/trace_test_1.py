from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

from .clock import Clock
from .content_variants import content_metadata_from_payload
from .cognitive_core import (
    AttemptSummary,
    Phase,
    Problem,
    QuestionEvent,
    SeededRng,
    TestSnapshot,
    clamp01,
)
from .trace_lattice import (
    DEFAULT_TRACE_LATTICE_SPEC,
    TraceLatticeAction,
    TraceLatticePath,
    TraceLatticeSpec,
    TraceLatticeState,
    trace_lattice_build_path,
    trace_lattice_center_state,
    trace_lattice_node_point,
    trace_lattice_right,
    trace_lattice_sample_path,
    trace_lattice_state,
)

_STAGE_EPSILON_S = 1e-6
_VIEWPOINT_BEARING_DEG = 180
_RED_SAFE_BOX_MIN = 0.02
_RED_SAFE_BOX_MAX = 0.98
_BLUE_MARGIN = 0.20
_WORLD_X_NORMALIZER = 100.0
_WORLD_Z_SCREEN_CENTER = 12.5
_WORLD_Z_SCREEN_NORMALIZER = 44.0
_WORLD_DEPTH_SCREEN_CENTER = 34.0
_WORLD_DEPTH_X_MIX = 0.0
_WORLD_DEPTH_Y_MIX = 0.24
_RED_ALTITUDE_BAND = (4.0, 28.0)
_RED_DEPTH_BAND = (14.0, 146.0)
_BLUE_DEPTH_BAND = (0.0, 156.0)
_RED_RECOVERY_DISTANCE = 3.5
_TURN_HOLD_FRACTION = 0.14
_BLUE_OUTLINE_COLORS: tuple[tuple[int, int, int], ...] = (
    (220, 232, 246),
    (202, 226, 250),
    (188, 214, 244),
    (176, 206, 242),
)
_TT1_LATTICE_COL_SPACING = 16.0
_TT1_LATTICE_ROW_SPACING = 18.0
_TT1_LATTICE_LEVEL_SPACING = 7.5
_TT1_LATTICE_BASE_DEPTH = 16.0
_TT1_TURN_PHASE_RATIO = 0.35
_TT1_MAX_RED_RECOVERY_STEPS = 4


@dataclass(frozen=True, slots=True)
class TraceTest1Config:
    # Candidate guide indicates about 9 minutes total including instructions.
    scored_duration_s: float = 7.5 * 60.0
    practice_questions: int = 3
    practice_observe_s: float = 5.0
    scored_observe_s: float = 4.3
    allowed_commands: tuple["TraceTest1Command", ...] | None = None
    allowed_visible_commands: tuple["TraceTest1Command", ...] | None = None


class TraceTest1Command(StrEnum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    PUSH = "PUSH"
    PULL = "PULL"


@dataclass(frozen=True, slots=True)
class TraceTest1Attitude:
    roll_deg: float
    pitch_deg: float
    yaw_deg: float


@dataclass(frozen=True, slots=True)
class TraceTest1Option:
    code: int
    label: str
    command: TraceTest1Command


class TraceTest1TrialStage(StrEnum):
    OBSERVE = "observe"
    ANSWER_OPEN = "answer_open"


@dataclass(frozen=True, slots=True)
class TraceTest1SceneFrame:
    position: tuple[float, float, float]
    attitude: TraceTest1Attitude
    travel_heading_deg: float
    world_tangent: tuple[float, float, float]
    world_forward: tuple[float, float, float] = (0.0, 1.0, 0.0)
    world_up: tuple[float, float, float] = (0.0, 0.0, 1.0)


@dataclass(frozen=True, slots=True)
class TraceTest1SceneSnapshot:
    red_frame: TraceTest1SceneFrame
    blue_frames: tuple[TraceTest1SceneFrame, ...]


@dataclass(frozen=True, slots=True)
class TraceTest1DifficultyTier:
    blue_count: int
    speed_multiplier: float
    answer_open_progress: float
    immediate_turn_chance: float


@dataclass(frozen=True, slots=True)
class TraceTest1AircraftState:
    position: tuple[float, float, float]
    heading_deg: float


@dataclass(frozen=True, slots=True)
class TraceTest1AircraftPlan:
    start_state: TraceTest1AircraftState
    command: TraceTest1Command
    lead_distance: float
    maneuver_distance: float
    altitude_delta: float
    lattice_start: TraceLatticeState | None = None
    lattice_actions: tuple[TraceLatticeAction, ...] = ()
    lattice_spec: TraceLatticeSpec = field(default_factory=lambda: DEFAULT_TRACE_LATTICE_SPEC)


@dataclass(frozen=True, slots=True)
class TraceTest1PromptPlan:
    prompt_index: int
    answer_open_progress: float
    speed_multiplier: float
    red_plan: TraceTest1AircraftPlan
    blue_plans: tuple[TraceTest1AircraftPlan, ...]


@dataclass(frozen=True, slots=True)
class TraceTest1Payload:
    trial_stage: TraceTest1TrialStage
    stage_time_remaining_s: float | None
    observe_progress: float
    prompt_index: int
    active_command: TraceTest1Command
    blue_commands: tuple[TraceTest1Command, ...]
    scene: TraceTest1SceneSnapshot
    options: tuple[TraceTest1Option, ...]
    correct_code: int
    prompt_window_s: float
    answer_open_progress: float
    speed_multiplier: float
    viewpoint_bearing_deg: int


_COMMAND_TO_CODE = {
    TraceTest1Command.LEFT: 1,
    TraceTest1Command.RIGHT: 2,
    TraceTest1Command.PUSH: 3,
    TraceTest1Command.PULL: 4,
}
_CODE_TO_COMMAND = {code: command for command, code in _COMMAND_TO_CODE.items()}


def _wrap_heading(deg: float) -> float:
    value = float(deg) % 360.0
    return value + 360.0 if value < 0.0 else value


def _clamp(v: float, lo: float, hi: float) -> float:
    if v < lo:
        return float(lo)
    if v > hi:
        return float(hi)
    return float(v)


def _smoothstep01(value: float) -> float:
    t = _clamp(value, 0.0, 1.0)
    return t * t * (3.0 - (2.0 * t))


def _angle_delta_deg(a: float, b: float) -> float:
    return ((float(a) - float(b) + 180.0) % 360.0) - 180.0


def _heading_vector_xy(heading_deg: float) -> tuple[float, float]:
    radians = math.radians(float(heading_deg))
    return (math.sin(radians), math.cos(radians))


def _tt1_action_for_command(command: TraceTest1Command) -> TraceLatticeAction:
    return {
        TraceTest1Command.LEFT: TraceLatticeAction.LEFT,
        TraceTest1Command.RIGHT: TraceLatticeAction.RIGHT,
        TraceTest1Command.PUSH: TraceLatticeAction.PUSH,
        TraceTest1Command.PULL: TraceLatticeAction.PULL,
    }[command]


def _tt1_command_step_index(plan: TraceTest1AircraftPlan) -> int:
    command_action = _tt1_action_for_command(plan.command)
    for idx, action in enumerate(plan.lattice_actions):
        if action is command_action:
            return idx
    return 0


def _tt1_lattice_actions_for_command(
    command: TraceTest1Command,
    *,
    lead_steps: int,
) -> tuple[TraceLatticeAction, ...]:
    lead = max(0, int(lead_steps))
    return (
        (TraceLatticeAction.STRAIGHT,) * lead
        + (_tt1_action_for_command(command),)
        + (TraceLatticeAction.STRAIGHT,)
    )


def _tt1_world_position_from_lattice(
    point: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        float(point[0] * _TT1_LATTICE_COL_SPACING),
        float(_TT1_LATTICE_BASE_DEPTH + (point[1] * _TT1_LATTICE_ROW_SPACING)),
        float(12.0 + (point[2] * _TT1_LATTICE_LEVEL_SPACING)),
    )


def _tt1_build_lattice_path(plan: TraceTest1AircraftPlan) -> TraceLatticePath | None:
    if plan.lattice_start is None or not plan.lattice_actions:
        return None
    return trace_lattice_build_path(
        start_state=plan.lattice_start,
        actions=tuple(plan.lattice_actions),
        spec=plan.lattice_spec,
    )


def _tt1_heading_pitch_from_forward(
    forward: tuple[float, float, float],
) -> tuple[float, float]:
    dx, dy, dz = (float(forward[0]), float(forward[1]), float(forward[2]))
    horiz = max(1e-6, math.hypot(dx, dy))
    return (
        _wrap_heading(math.degrees(math.atan2(dx, dy))),
        float(math.degrees(math.atan2(dz, horiz))),
    )


def _tt1_scene_frame_from_lattice_plan(
    *,
    plan: TraceTest1AircraftPlan,
    progress: float,
) -> TraceTest1SceneFrame | None:
    path = _tt1_build_lattice_path(plan)
    if path is None:
        return None
    pose = trace_lattice_sample_path(
        path,
        progress=progress,
        turn_phase_ratio=_TT1_TURN_PHASE_RATIO,
    )
    world_position = _tt1_world_position_from_lattice(pose.position)
    world_forward = (
        float(pose.forward[0] * _TT1_LATTICE_COL_SPACING),
        float(pose.forward[1] * _TT1_LATTICE_ROW_SPACING),
        float(pose.forward[2] * _TT1_LATTICE_LEVEL_SPACING),
    )
    heading_deg, pitch_deg = _tt1_heading_pitch_from_forward(world_forward)
    return TraceTest1SceneFrame(
        position=world_position,
        attitude=TraceTest1Attitude(
            roll_deg=0.0,
            pitch_deg=float(pitch_deg),
            yaw_deg=float(heading_deg),
        ),
        travel_heading_deg=float(heading_deg),
        world_tangent=world_forward,
        world_forward=world_forward,
        world_up=(
            float(pose.up[0]),
            float(pose.up[1]),
            float(pose.up[2]),
        ),
    )


def _tt1_aircraft_state_from_lattice_state(
    state: TraceLatticeState,
) -> TraceTest1AircraftState:
    world_position = _tt1_world_position_from_lattice(
        trace_lattice_node_point(state.node, spec=DEFAULT_TRACE_LATTICE_SPEC)
    )
    world_forward = (
        float(state.orientation.forward[0] * _TT1_LATTICE_COL_SPACING),
        float(state.orientation.forward[1] * _TT1_LATTICE_ROW_SPACING),
        float(state.orientation.forward[2] * _TT1_LATTICE_LEVEL_SPACING),
    )
    heading_deg, _pitch_deg = _tt1_heading_pitch_from_forward(world_forward)
    return TraceTest1AircraftState(
        position=world_position,
        heading_deg=float(heading_deg),
    )


def _cardinal_heading(heading_deg: float) -> float:
    wrapped = _wrap_heading(heading_deg)
    return float((round(wrapped / 90.0) * 90) % 360)


def _move_point(
    point: tuple[float, float, float],
    *,
    heading_deg: float,
    distance: float,
    climb: float = 0.0,
) -> tuple[float, float, float]:
    dx, dy = _heading_vector_xy(heading_deg)
    return (
        float(point[0] + (dx * distance)),
        float(point[1] + (dy * distance)),
        float(point[2] + climb),
    )


def _turn_displacement(
    *,
    heading_deg: float,
    arc_length: float,
    progress: float,
    turn_sign: float,
) -> tuple[float, float]:
    radius = max(1e-6, float(arc_length) / (math.pi * 0.5))
    angle = _clamp(float(progress), 0.0, 1.0) * (math.pi * 0.5)
    forward_x, forward_y = _heading_vector_xy(heading_deg)
    right_x, right_y = _heading_vector_xy(heading_deg + 90.0)
    side_x = right_x * float(turn_sign)
    side_y = right_y * float(turn_sign)
    return (
        (forward_x * radius * math.sin(angle)) + (side_x * radius * (1.0 - math.cos(angle))),
        (forward_y * radius * math.sin(angle)) + (side_y * radius * (1.0 - math.cos(angle))),
    )


def _scene_position_for_plan(
    *,
    plan: TraceTest1AircraftPlan,
    answer_open_progress: float,
    progress: float,
) -> tuple[float, float, float]:
    t = _clamp(progress, 0.0, 1.0)
    split = _clamp(answer_open_progress, 0.12, 0.88)
    lead_t = 1.0 if t >= split else _clamp(t / max(0.001, split), 0.0, 1.0)
    answer_t = 0.0 if t <= split else _clamp((t - split) / max(0.001, 1.0 - split), 0.0, 1.0)

    start = plan.start_state.position
    start_heading = _cardinal_heading(plan.start_state.heading_deg)
    corner = _move_point(start, heading_deg=start_heading, distance=plan.lead_distance)
    if answer_t <= 0.0:
        return _move_point(start, heading_deg=start_heading, distance=plan.lead_distance * lead_t)

    if plan.command is TraceTest1Command.LEFT:
        dx, dy = _turn_displacement(
            heading_deg=start_heading,
            arc_length=plan.maneuver_distance,
            progress=answer_t,
            turn_sign=-1.0,
        )
        return (float(corner[0] + dx), float(corner[1] + dy), float(corner[2]))
    if plan.command is TraceTest1Command.RIGHT:
        dx, dy = _turn_displacement(
            heading_deg=start_heading,
            arc_length=plan.maneuver_distance,
            progress=answer_t,
            turn_sign=1.0,
        )
        return (float(corner[0] + dx), float(corner[1] + dy), float(corner[2]))

    climb_t = _smoothstep01(answer_t)
    forward_x, forward_y = _heading_vector_xy(start_heading)
    forward_distance = plan.maneuver_distance * answer_t
    return (
        float(corner[0] + (forward_x * forward_distance)),
        float(corner[1] + (forward_y * forward_distance)),
        float(corner[2] + (plan.altitude_delta * climb_t)),
    )


def _sample_frame_tangent(
    *,
    plan: TraceTest1AircraftPlan,
    answer_open_progress: float,
    progress: float,
    delta: float = 0.012,
) -> tuple[float, float, float]:
    center = _scene_position_for_plan(
        plan=plan,
        answer_open_progress=answer_open_progress,
        progress=progress,
    )
    prev = _scene_position_for_plan(
        plan=plan,
        answer_open_progress=answer_open_progress,
        progress=max(0.0, float(progress) - float(delta)),
    )
    nxt = _scene_position_for_plan(
        plan=plan,
        answer_open_progress=answer_open_progress,
        progress=min(1.0, float(progress) + float(delta)),
    )
    dx = float(nxt[0] - prev[0])
    dy = float(nxt[1] - prev[1])
    dz = float(nxt[2] - prev[2])
    if (dx * dx) + (dy * dy) + (dz * dz) > 1e-8:
        return (dx, dy, dz)
    dx = float(nxt[0] - center[0])
    dy = float(nxt[1] - center[1])
    dz = float(nxt[2] - center[2])
    if (dx * dx) + (dy * dy) + (dz * dz) > 1e-8:
        return (dx, dy, dz)
    return (
        float(center[0] - prev[0]),
        float(center[1] - prev[1]),
        float(center[2] - prev[2]),
    )


def _frame_attitude_from_tangent(
    *,
    plan: TraceTest1AircraftPlan,
    answer_open_progress: float,
    progress: float,
    tangent: tuple[float, float, float],
) -> tuple[float, TraceTest1Attitude]:
    dx, dy, dz = tangent
    horiz = math.hypot(dx, dy)
    start_heading = _cardinal_heading(plan.start_state.heading_deg)
    if horiz <= 1e-6 and abs(dz) <= 1e-6:
        travel_heading = start_heading
        pitch_deg = 0.0
    else:
        travel_heading = _wrap_heading(math.degrees(math.atan2(dx, dy)))
        pitch_deg = math.degrees(math.atan2(dz, max(1e-6, horiz)))

    prev_tangent = _sample_frame_tangent(
        plan=plan,
        answer_open_progress=answer_open_progress,
        progress=max(0.0, float(progress) - 0.018),
    )
    next_tangent = _sample_frame_tangent(
        plan=plan,
        answer_open_progress=answer_open_progress,
        progress=min(1.0, float(progress) + 0.018),
    )
    prev_heading = travel_heading if math.hypot(prev_tangent[0], prev_tangent[1]) <= 1e-6 else _wrap_heading(
        math.degrees(math.atan2(prev_tangent[0], prev_tangent[1]))
    )
    next_heading = travel_heading if math.hypot(next_tangent[0], next_tangent[1]) <= 1e-6 else _wrap_heading(
        math.degrees(math.atan2(next_tangent[0], next_tangent[1]))
    )
    turn_rate = _angle_delta_deg(next_heading, prev_heading) / 0.036
    roll_deg = _clamp(turn_rate * 0.18, -32.0, 32.0)
    return (
        float(travel_heading),
        TraceTest1Attitude(
            roll_deg=float(roll_deg),
            pitch_deg=float(pitch_deg),
            yaw_deg=float(travel_heading),
        ),
    )


def trace_test_1_command_from_code(code: int) -> TraceTest1Command:
    return _CODE_TO_COMMAND.get(int(code), TraceTest1Command.LEFT)


def _normalize_allowed_commands(
    commands: tuple[TraceTest1Command, ...] | None,
) -> tuple[TraceTest1Command, ...]:
    if commands is None:
        return tuple(TraceTest1Command)
    normalized: list[TraceTest1Command] = []
    seen: set[TraceTest1Command] = set()
    for raw in commands:
        try:
            command = raw if isinstance(raw, TraceTest1Command) else TraceTest1Command(str(raw))
        except ValueError as exc:
            raise ValueError(f"Unknown Trace Test 1 command: {raw}") from exc
        if command in seen:
            continue
        seen.add(command)
        normalized.append(command)
    if not normalized:
        raise ValueError("allowed_commands must not be empty")
    return tuple(normalized)


def trace_test_1_answer_code(raw: object) -> int | None:
    value = str(raw).strip().upper()
    if value == "":
        return None
    if value.isdigit():
        code = int(value)
        return code if 1 <= code <= 4 else None
    return {
        "LEFT": 1,
        "RIGHT": 2,
        "UP": 3,
        "PUSH": 3,
        "FORWARD": 3,
        "DOWN": 4,
        "PULL": 4,
        "BACK": 4,
        "BACKWARD": 4,
    }.get(value)


def trace_test_1_difficulty_tier(*, difficulty: float) -> TraceTest1DifficultyTier:
    d = clamp01(difficulty)
    if d < 0.34:
        return TraceTest1DifficultyTier(
            blue_count=1,
            speed_multiplier=0.95,
            answer_open_progress=0.44,
            immediate_turn_chance=0.0,
        )
    if d < 0.67:
        return TraceTest1DifficultyTier(
            blue_count=2,
            speed_multiplier=1.12,
            answer_open_progress=0.37,
            immediate_turn_chance=0.0,
        )
    if d < 0.89:
        return TraceTest1DifficultyTier(
            blue_count=4,
            speed_multiplier=1.38,
            answer_open_progress=0.30,
            immediate_turn_chance=0.78,
        )
    return TraceTest1DifficultyTier(
        blue_count=5,
        speed_multiplier=1.65,
        answer_open_progress=0.24,
        immediate_turn_chance=0.94,
    )


def trace_test_1_normalized_position(
    position: tuple[float, float, float],
) -> tuple[float, float]:
    world_x, world_y, world_z = position
    depth_offset = float(world_y) - _WORLD_DEPTH_SCREEN_CENTER
    return (
        float(
            0.5
            + (
                (
                    float(world_x)
                    + (depth_offset * _WORLD_DEPTH_X_MIX)
                )
                / _WORLD_X_NORMALIZER
            )
        ),
        float(
            0.64
            - (
                (
                    (float(world_z) - _WORLD_Z_SCREEN_CENTER)
                    + (depth_offset * _WORLD_DEPTH_Y_MIX)
                )
                / _WORLD_Z_SCREEN_NORMALIZER
            )
        ),
    )


def _scene_frame_for_plan(
    *,
    plan: TraceTest1AircraftPlan,
    answer_open_progress: float,
    progress: float,
) -> TraceTest1SceneFrame:
    lattice_frame = _tt1_scene_frame_from_lattice_plan(
        plan=plan,
        progress=progress,
    )
    if lattice_frame is not None:
        return lattice_frame
    position = _scene_position_for_plan(
        plan=plan,
        answer_open_progress=answer_open_progress,
        progress=progress,
    )
    tangent = _sample_frame_tangent(
        plan=plan,
        answer_open_progress=answer_open_progress,
        progress=progress,
    )
    travel_heading, attitude = _frame_attitude_from_tangent(
        plan=plan,
        answer_open_progress=answer_open_progress,
        progress=progress,
        tangent=tangent,
    )
    return TraceTest1SceneFrame(
        position=position,
        attitude=attitude,
        travel_heading_deg=float(travel_heading),
        world_tangent=tangent,
        world_forward=tangent,
        world_up=(0.0, 0.0, 1.0),
    )


def trace_test_1_scene_frames(
    *,
    prompt: TraceTest1PromptPlan,
    progress: float,
) -> TraceTest1SceneSnapshot:
    return TraceTest1SceneSnapshot(
        red_frame=_scene_frame_for_plan(
            plan=prompt.red_plan,
            answer_open_progress=prompt.answer_open_progress,
            progress=progress,
        ),
        blue_frames=tuple(
            _scene_frame_for_plan(
                plan=blue_plan,
                answer_open_progress=prompt.answer_open_progress,
                progress=progress,
            )
            for blue_plan in prompt.blue_plans
        ),
    )


def trace_test_1_aircraft_hpr(frame: TraceTest1SceneFrame) -> tuple[float, float, float]:
    return (
        float(frame.travel_heading_deg),
        float(frame.attitude.pitch_deg),
        float(frame.attitude.roll_deg),
    )


def _aircraft_state_from_frame(frame: TraceTest1SceneFrame) -> TraceTest1AircraftState:
    return TraceTest1AircraftState(
        position=frame.position,
        heading_deg=_wrap_heading(frame.travel_heading_deg),
    )


def _prompt_end_heading(plan: TraceTest1AircraftPlan) -> float:
    start_heading = _cardinal_heading(plan.start_state.heading_deg)
    if plan.command is TraceTest1Command.LEFT:
        return _cardinal_heading(start_heading - 90.0)
    if plan.command is TraceTest1Command.RIGHT:
        return _cardinal_heading(start_heading + 90.0)
    return start_heading


def _projected_safe(
    position: tuple[float, float, float],
) -> bool:
    nx, ny = trace_test_1_normalized_position(position)
    return (
        _RED_SAFE_BOX_MIN <= nx <= _RED_SAFE_BOX_MAX
        and _RED_SAFE_BOX_MIN <= ny <= _RED_SAFE_BOX_MAX
        and _RED_DEPTH_BAND[0] <= position[1] <= _RED_DEPTH_BAND[1]
        and _RED_ALTITUDE_BAND[0] <= position[2] <= _RED_ALTITUDE_BAND[1]
    )


def _offscreen_blue(
    position: tuple[float, float, float],
) -> bool:
    nx, ny = trace_test_1_normalized_position(position)
    return (
        nx < -_BLUE_MARGIN
        or nx > 1.0 + _BLUE_MARGIN
        or ny < -_BLUE_MARGIN
        or ny > 1.0 + _BLUE_MARGIN
        or position[1] < _BLUE_DEPTH_BAND[0]
        or position[1] > _BLUE_DEPTH_BAND[1]
    )


def _tt1_initial_red_state() -> TraceLatticeState:
    return trace_lattice_center_state(spec=DEFAULT_TRACE_LATTICE_SPEC)


def _tt1_red_lattice_actions(
    command: TraceTest1Command,
    *,
    lead_steps: int,
    recovery_steps: int = 0,
) -> tuple[TraceLatticeAction, ...]:
    return (
        (TraceLatticeAction.STRAIGHT,) * max(0, int(recovery_steps))
        + _tt1_lattice_actions_for_command(
            command,
            lead_steps=lead_steps,
        )
    )


class TraceTest1Generator:
    """Deterministic continuous TT1 world stream."""

    _OPTIONS: tuple[TraceTest1Option, ...] = (
        TraceTest1Option(code=1, label="Left", command=TraceTest1Command.LEFT),
        TraceTest1Option(code=2, label="Right", command=TraceTest1Command.RIGHT),
        TraceTest1Option(code=3, label="Push", command=TraceTest1Command.PUSH),
        TraceTest1Option(code=4, label="Pull", command=TraceTest1Command.PULL),
    )

    def __init__(
        self,
        *,
        seed: int,
        allowed_commands: tuple[TraceTest1Command, ...] | None = None,
        allowed_visible_commands: tuple[TraceTest1Command, ...] | None = None,
    ) -> None:
        self._rng = SeededRng(seed)
        self._allowed_commands = _normalize_allowed_commands(allowed_commands)
        self._allowed_visible_commands = (
            None
            if allowed_visible_commands is None
            else _normalize_allowed_commands(allowed_visible_commands)
        )
        self._prompt_index = 0
        self._last_red_command: TraceTest1Command | None = None
        self._tier: TraceTest1DifficultyTier | None = None
        self._red_lattice_state = _tt1_initial_red_state()
        self._blue_lattice_states: tuple[TraceLatticeState, ...] = ()
        self._red_state = _tt1_aircraft_state_from_lattice_state(self._red_lattice_state)
        self._blue_states: tuple[TraceTest1AircraftState, ...] = ()
        self._pending_prompt: TraceTest1PromptPlan | None = None

    def next_problem(self, *, difficulty: float) -> Problem:
        from .trace_scene_3d import classify_trace_test_1_view_maneuver

        tier = trace_test_1_difficulty_tier(difficulty=difficulty)
        self._ensure_tier_initialized(tier=tier)
        if self._pending_prompt is not None:
            self.commit_prompt(prompt=self._pending_prompt, progress=1.0)
        for _attempt in range(64):
            red_plan = self._build_red_plan(tier=tier)
            blue_plans = tuple(
                self._build_blue_plan(state=state, tier=tier) for state in self._blue_lattice_states
            )
            prompt = TraceTest1PromptPlan(
                prompt_index=int(self._prompt_index),
                answer_open_progress=float(tier.answer_open_progress),
                speed_multiplier=float(tier.speed_multiplier),
                red_plan=red_plan,
                blue_plans=blue_plans,
            )
            classified_command = classify_trace_test_1_view_maneuver(
                prompt=prompt,
                viewpoint_bearing_deg=_VIEWPOINT_BEARING_DEG,
            )
            if (
                self._allowed_visible_commands is not None
                and classified_command not in self._allowed_visible_commands
            ):
                self.commit_prompt(prompt=prompt, progress=1.0)
                continue
            self._pending_prompt = prompt
            self._prompt_index += 1
            correct_code = int(_COMMAND_TO_CODE[classified_command])
            return Problem(
                prompt=f"Trace Test 1 {classified_command.value}",
                answer=correct_code,
                payload=prompt,
            )
        raise RuntimeError("Unable to build a visible Trace Test 1 prompt from the current stream state")

    def commit_prompt(
        self,
        *,
        prompt: TraceTest1PromptPlan,
        progress: float,
    ) -> None:
        red_path = _tt1_build_lattice_path(prompt.red_plan)
        if red_path is not None:
            self._red_lattice_state = red_path.end_state
            self._red_state = _tt1_aircraft_state_from_lattice_state(self._red_lattice_state)
        else:
            scene = trace_test_1_scene_frames(prompt=prompt, progress=progress)
            self._red_state = _aircraft_state_from_frame(scene.red_frame)
        next_blue_lattice_states: list[TraceLatticeState] = []
        next_blue_states: list[TraceTest1AircraftState] = []
        for blue_plan in prompt.blue_plans:
            path = _tt1_build_lattice_path(blue_plan)
            if path is not None:
                next_blue_lattice_states.append(path.end_state)
                next_blue_states.append(_tt1_aircraft_state_from_lattice_state(path.end_state))
            else:
                frame = trace_test_1_scene_frames(
                    prompt=TraceTest1PromptPlan(
                        prompt_index=prompt.prompt_index,
                        answer_open_progress=prompt.answer_open_progress,
                        speed_multiplier=prompt.speed_multiplier,
                        red_plan=blue_plan,
                        blue_plans=(),
                    ),
                    progress=progress,
                ).red_frame
                next_blue_states.append(_aircraft_state_from_frame(frame))
        self._blue_lattice_states = tuple(next_blue_lattice_states)
        self._blue_states = tuple(next_blue_states)
        self._pending_prompt = None

    def _ensure_tier_initialized(self, *, tier: TraceTest1DifficultyTier) -> None:
        if self._tier == tier and len(self._blue_lattice_states) == tier.blue_count:
            return
        self._tier = tier
        if not self._blue_lattice_states:
            self._blue_lattice_states = tuple(
                self._spawn_initial_blue(index=idx) for idx in range(tier.blue_count)
            )
            self._blue_states = tuple(
                _tt1_aircraft_state_from_lattice_state(state) for state in self._blue_lattice_states
            )
            return
        current = list(self._blue_lattice_states[: tier.blue_count])
        while len(current) < tier.blue_count:
            current.append(self._spawn_initial_blue(index=len(current)))
        self._blue_lattice_states = tuple(current)
        self._blue_states = tuple(
            _tt1_aircraft_state_from_lattice_state(state) for state in self._blue_lattice_states
        )

    def _spawn_initial_blue(self, *, index: int) -> TraceLatticeState:
        presets = (
            trace_lattice_state(col=1, row=1, level=1, forward=(0, 1, 0), up=(0, 0, 1)),
            trace_lattice_state(col=5, row=1, level=3, forward=(0, 1, 0), up=(0, 0, 1)),
            trace_lattice_state(col=1, row=3, level=2, forward=(1, 0, 0), up=(0, 0, 1)),
            trace_lattice_state(col=5, row=4, level=2, forward=(-1, 0, 0), up=(0, 0, 1)),
            trace_lattice_state(col=3, row=1, level=3, forward=(0, 1, 0), up=(0, 0, 1)),
        )
        return presets[index % len(presets)]

    def _prompt_distances(self, *, tier: TraceTest1DifficultyTier) -> tuple[float, float, float]:
        speed = float(tier.speed_multiplier)
        return (4.8 * speed, 4.1 * speed, 2.2 * speed)

    def _build_red_plan(self, *, tier: TraceTest1DifficultyTier) -> TraceTest1AircraftPlan:
        lead_distance, maneuver_distance, altitude_delta = self._prompt_distances(tier=tier)
        candidates = [
            command for command in self._allowed_commands if command is not self._last_red_command
        ]
        if not candidates:
            candidates = list(self._allowed_commands)
        start_state = self._red_lattice_state
        ordered_candidates = self._rng.sample(candidates, k=len(candidates))
        prefer_immediate = self._rng.random() < float(tier.immediate_turn_chance)
        lead_step_order = (0, 1) if prefer_immediate else (1, 0)
        plan = self._find_red_plan(
            tier=tier,
            start_state=start_state,
            ordered_candidates=ordered_candidates,
            lead_step_order=lead_step_order,
            lead_distance=lead_distance,
            maneuver_distance=maneuver_distance,
            altitude_delta=altitude_delta,
            require_continuation=True,
        )
        if plan is None:
            plan = self._find_red_plan(
                tier=tier,
                start_state=start_state,
                ordered_candidates=ordered_candidates,
                lead_step_order=lead_step_order,
                lead_distance=lead_distance,
                maneuver_distance=maneuver_distance,
                altitude_delta=altitude_delta,
                require_continuation=False,
            )
        if plan is None:
            raise RuntimeError("Unable to build a safe Trace Test 1 red prompt from current state")
        self._last_red_command = plan.command
        return plan

    def _find_red_plan(
        self,
        *,
        tier: TraceTest1DifficultyTier,
        start_state: TraceLatticeState,
        ordered_candidates: list[TraceTest1Command],
        lead_step_order: tuple[int, int],
        lead_distance: float,
        maneuver_distance: float,
        altitude_delta: float,
        require_continuation: bool,
    ) -> TraceTest1AircraftPlan | None:
        for lead_steps in lead_step_order:
            for recovery_steps in range(_TT1_MAX_RED_RECOVERY_STEPS + 1):
                for command in ordered_candidates:
                    plan = TraceTest1AircraftPlan(
                        start_state=_tt1_aircraft_state_from_lattice_state(start_state),
                        command=command,
                        lead_distance=lead_distance,
                        maneuver_distance=maneuver_distance,
                        altitude_delta=altitude_delta if command is TraceTest1Command.PULL else -altitude_delta,
                        lattice_start=start_state,
                        lattice_actions=_tt1_red_lattice_actions(
                            command,
                            lead_steps=lead_steps,
                            recovery_steps=recovery_steps,
                        ),
                    )
                    if self._red_plan_is_safe(
                        plan=plan,
                        tier=tier,
                        require_continuation=require_continuation,
                    ):
                        return plan
        return None

    def _red_plan_is_safe(
        self,
        *,
        plan: TraceTest1AircraftPlan,
        tier: TraceTest1DifficultyTier,
        require_continuation: bool = True,
    ) -> bool:
        path = _tt1_build_lattice_path(plan)
        command_step_index = _tt1_command_step_index(plan)
        if path is None or len(path.steps) <= command_step_index:
            return False
        if path.steps[command_step_index].effective_action is not _tt1_action_for_command(plan.command):
            return False
        split = float(tier.answer_open_progress)
        checkpoints = (
            0.0,
            max(0.0, split * 0.5),
            split,
            min(1.0, split + ((1.0 - split) * 0.5)),
            1.0,
        )
        scene_points = [
            trace_test_1_scene_frames(
                prompt=TraceTest1PromptPlan(
                    prompt_index=self._prompt_index,
                    answer_open_progress=split,
                    speed_multiplier=tier.speed_multiplier,
                    red_plan=plan,
                    blue_plans=(),
                ),
                progress=checkpoint,
            ).red_frame.position
            for checkpoint in checkpoints
        ]
        if not all(_projected_safe(position) for position in scene_points):
            return False
        if not require_continuation:
            return True
        return self._red_state_has_safe_continuation(
            start_state=path.end_state,
            tier=tier,
            blocked_command=plan.command,
        )

    def _red_state_has_safe_continuation(
        self,
        *,
        start_state: TraceLatticeState,
        tier: TraceTest1DifficultyTier,
        blocked_command: TraceTest1Command,
    ) -> bool:
        lead_distance, maneuver_distance, altitude_delta = self._prompt_distances(tier=tier)
        candidates = [command for command in self._allowed_commands if command is not blocked_command]
        if not candidates:
            candidates = list(self._allowed_commands)
        lead_step_order = (0, 1) if float(tier.immediate_turn_chance) >= 0.5 else (1, 0)
        return (
            self._find_red_plan(
                tier=tier,
                start_state=start_state,
                ordered_candidates=candidates,
                lead_step_order=lead_step_order,
                lead_distance=lead_distance,
                maneuver_distance=maneuver_distance,
                altitude_delta=altitude_delta,
                require_continuation=False,
            )
            is not None
        )

    def _build_blue_plan(
        self,
        *,
        state: TraceLatticeState,
        tier: TraceTest1DifficultyTier,
    ) -> TraceTest1AircraftPlan:
        lead_distance, maneuver_distance, altitude_delta = self._prompt_distances(tier=tier)
        command = self._rng.choice(tuple(TraceTest1Command))
        lead_steps = 0 if self._rng.random() < float(tier.immediate_turn_chance) else 1
        return TraceTest1AircraftPlan(
            start_state=_tt1_aircraft_state_from_lattice_state(state),
            command=command,
            lead_distance=lead_distance,
            maneuver_distance=maneuver_distance,
            altitude_delta=altitude_delta if command is TraceTest1Command.PULL else -altitude_delta,
            lattice_start=state,
            lattice_actions=_tt1_lattice_actions_for_command(
                command,
                lead_steps=lead_steps,
            ),
        )


def build_trace_test_1_test(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    config: TraceTest1Config | None = None,
) -> "TraceTest1Engine":
    return TraceTest1Engine(
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        config=config,
    )


class TraceTest1Engine:
    def __init__(
        self,
        *,
        clock: Clock,
        seed: int,
        difficulty: float = 0.5,
        config: TraceTest1Config | None = None,
    ) -> None:
        cfg = config or TraceTest1Config()
        if cfg.scored_duration_s <= 0.0:
            raise ValueError("scored_duration_s must be > 0")
        if cfg.practice_questions < 0:
            raise ValueError("practice_questions must be >= 0")
        if cfg.practice_observe_s <= 0.0 or cfg.scored_observe_s <= 0.0:
            raise ValueError("observe durations must be > 0")

        self._clock = clock
        self._seed = int(seed)
        self._difficulty = clamp01(difficulty)
        self._cfg = cfg
        allowed_commands = _normalize_allowed_commands(self._cfg.allowed_commands)
        allowed_visible_commands = (
            None
            if self._cfg.allowed_visible_commands is None
            else _normalize_allowed_commands(self._cfg.allowed_visible_commands)
        )
        self._generator = TraceTest1Generator(
            seed=self._seed,
            allowed_commands=allowed_commands,
            allowed_visible_commands=allowed_visible_commands,
        )

        self._phase = Phase.INSTRUCTIONS
        self._current_problem: Problem | None = None
        self._current_prompt: TraceTest1PromptPlan | None = None
        self._trial_started_at_s: float | None = None
        self._current_prompt_answered = False
        self._locked_answer_raw = ""
        self._locked_answer_code: int | None = None
        self._locked_answered_at_s: float | None = None
        self._practice_answered = 0
        self._scored_started_at_s: float | None = None
        self._events: list[QuestionEvent] = []
        self._scored_attempted = 0
        self._scored_correct = 0
        self._scored_total_score = 0.0
        self._scored_max_score = 0.0

    @property
    def phase(self) -> Phase:
        return self._phase

    def can_exit(self) -> bool:
        return self._phase in (
            Phase.INSTRUCTIONS,
            Phase.PRACTICE,
            Phase.PRACTICE_DONE,
            Phase.RESULTS,
        )

    def start_practice(self) -> None:
        if self._phase is not Phase.INSTRUCTIONS:
            return
        if self._cfg.practice_questions <= 0:
            self._phase = Phase.PRACTICE_DONE
            return
        self._phase = Phase.PRACTICE
        self._deal_new_problem()

    def start(self) -> None:
        self.start_practice()

    def start_scored(self) -> None:
        if self._phase in (Phase.SCORED, Phase.RESULTS):
            return
        self._phase = Phase.SCORED
        self._scored_started_at_s = self._clock.now()
        self._deal_new_problem()

    def update(self) -> None:
        if self._phase not in (Phase.PRACTICE, Phase.SCORED):
            return
        if self._phase is Phase.SCORED and self.time_remaining_s() == 0.0:
            if self._current_prompt_answered:
                self._record_locked_answer()
            self._finish()
            return
        if self._current_prompt is None:
            return
        if self._elapsed_in_prompt_s() + _STAGE_EPSILON_S < self._prompt_duration_s():
            return
        if self._current_prompt_answered:
            self._record_locked_answer()
        else:
            self._record_auto_miss()
        self._advance_to_next_prompt(progress=1.0)

    def time_remaining_s(self) -> float | None:
        if self._phase is not Phase.SCORED or self._scored_started_at_s is None:
            return None
        remaining = self._cfg.scored_duration_s - (self._clock.now() - self._scored_started_at_s)
        return max(0.0, remaining)

    def current_prompt(self) -> str:
        if self._phase is Phase.INSTRUCTIONS:
            return (
                "Watch the continuous red stream.\n\n"
                "Arrow keys lock an answer at any time during the clip:\n"
                "Left, Right, Up for Push, and Down for Pull.\n"
                "Early or wrong presses still get scored against that clip when it ends."
            )
        if self._phase is Phase.PRACTICE_DONE:
            return "Practice complete. Press Enter to start the timed test."
        if self._phase is Phase.RESULTS:
            summary = self.scored_summary()
            acc_pct = int(round(summary.accuracy * 100))
            rt = "n/a" if summary.mean_response_time_s is None else f"{summary.mean_response_time_s:.2f}s"
            return (
                f"Results\nAttempted: {summary.attempted}\nCorrect: {summary.correct}\n"
                f"Accuracy: {acc_pct}%\nMean RT: {rt}\n"
                f"Throughput: {summary.throughput_per_min:.1f}/min"
            )
        return ""

    def submit_answer(self, raw: object) -> bool:
        if self._phase not in (Phase.PRACTICE, Phase.SCORED):
            return False
        if self._current_prompt_answered:
            return False
        user_answer = trace_test_1_answer_code(raw)
        if user_answer is None or self._current_problem is None or self._current_prompt is None:
            return False
        self._locked_answer_raw = str(raw)
        self._locked_answer_code = int(user_answer)
        self._locked_answered_at_s = float(self._clock.now())
        self._current_prompt_answered = True
        return True

    def scored_summary(self) -> AttemptSummary:
        duration_s = float(self._cfg.scored_duration_s)
        attempted = int(self._scored_attempted)
        correct = int(self._scored_correct)
        accuracy = 0.0 if attempted == 0 else correct / attempted
        throughput = (attempted / duration_s) * 60.0
        rts = [event.response_time_s for event in self._events if event.phase is Phase.SCORED]
        mean_rt = None if not rts else sum(rts) / len(rts)
        total_score = float(self._scored_total_score)
        max_score = float(self._scored_max_score)
        score_ratio = 0.0 if max_score == 0.0 else total_score / max_score
        return AttemptSummary(
            attempted=attempted,
            correct=correct,
            accuracy=float(accuracy),
            duration_s=duration_s,
            throughput_per_min=float(throughput),
            mean_response_time_s=None if mean_rt is None else float(mean_rt),
            total_score=total_score,
            max_score=max_score,
            score_ratio=float(score_ratio),
        )

    def events(self) -> list[QuestionEvent]:
        return list(self._events)

    def snapshot(self) -> TestSnapshot:
        return TestSnapshot(
            title="Trace Test 1",
            phase=self._phase,
            prompt=self.current_prompt(),
            input_hint="Arrow keys lock the clip at any time. Early or wrong presses are scored when that clip ends.",
            time_remaining_s=self.time_remaining_s(),
            attempted_scored=self._scored_attempted,
            correct_scored=self._scored_correct,
            payload=self._snapshot_payload(),
        )

    def _prompt_duration_s(self) -> float:
        return (
            float(self._cfg.practice_observe_s)
            if self._phase is Phase.PRACTICE
            else float(self._cfg.scored_observe_s)
        )

    def _elapsed_in_prompt_s(self) -> float:
        if self._trial_started_at_s is None:
            return 0.0
        return max(0.0, self._clock.now() - self._trial_started_at_s)

    def _current_progress(self) -> float:
        duration = max(0.001, self._prompt_duration_s())
        return _clamp(self._elapsed_in_prompt_s() / duration, 0.0, 1.0)

    def _answer_open_at_s(self) -> float:
        if self._current_prompt is None:
            return 0.0
        return float(self._prompt_duration_s() * self._current_prompt.answer_open_progress)

    def _snapshot_payload(self) -> TraceTest1Payload | None:
        if self._current_problem is None or self._current_prompt is None:
            return None
        progress = self._current_progress()
        elapsed = self._elapsed_in_prompt_s()
        duration = self._prompt_duration_s()
        answer_open_at_s = self._answer_open_at_s()
        if progress + _STAGE_EPSILON_S < self._current_prompt.answer_open_progress:
            stage = TraceTest1TrialStage.OBSERVE
            stage_time_remaining_s = max(0.0, answer_open_at_s - elapsed)
        else:
            stage = TraceTest1TrialStage.ANSWER_OPEN
            stage_time_remaining_s = max(0.0, duration - elapsed)
        scene = trace_test_1_scene_frames(prompt=self._current_prompt, progress=progress)
        return TraceTest1Payload(
            trial_stage=stage,
            stage_time_remaining_s=float(stage_time_remaining_s),
            observe_progress=float(progress),
            prompt_index=int(self._current_prompt.prompt_index),
            active_command=self._current_prompt.red_plan.command,
            blue_commands=tuple(blue_plan.command for blue_plan in self._current_prompt.blue_plans),
            scene=scene,
            options=TraceTest1Generator._OPTIONS,
            correct_code=int(self._current_problem.answer),
            prompt_window_s=float(duration),
            answer_open_progress=float(self._current_prompt.answer_open_progress),
            speed_multiplier=float(self._current_prompt.speed_multiplier),
            viewpoint_bearing_deg=_VIEWPOINT_BEARING_DEG,
        )

    def _deal_new_problem(self) -> None:
        self._current_problem = self._generator.next_problem(difficulty=self._difficulty)
        prompt = self._current_problem.payload
        assert isinstance(prompt, TraceTest1PromptPlan)
        self._current_prompt = prompt
        self._trial_started_at_s = self._clock.now()
        self._current_prompt_answered = False
        self._locked_answer_raw = ""
        self._locked_answer_code = None
        self._locked_answered_at_s = None

    def _advance_to_next_prompt(self, *, progress: float) -> None:
        if self._current_prompt is not None:
            self._generator.commit_prompt(prompt=self._current_prompt, progress=progress)
        if self._phase is Phase.PRACTICE and self._practice_answered >= self._cfg.practice_questions:
            self._phase = Phase.PRACTICE_DONE
            self._clear_current()
            return
        if self._phase is Phase.SCORED and self.time_remaining_s() == 0.0:
            self._finish()
            return
        self._deal_new_problem()

    def _record_auto_miss(self) -> None:
        if self._current_problem is None or self._current_prompt is None:
            return
        self._locked_answer_raw = ""
        self._locked_answer_code = None
        self._locked_answered_at_s = None
        presented_at_s = self._trial_started_at_s + self._answer_open_at_s()  # type: ignore[operator]
        answered_at_s = self._trial_started_at_s + self._prompt_duration_s()  # type: ignore[operator]
        response_time_s = max(0.0, answered_at_s - presented_at_s)
        self._events.append(
            QuestionEvent(
                index=len(self._events),
                phase=self._phase,
                prompt=self._current_problem.prompt,
                correct_answer=int(self._current_problem.answer),
                user_answer=0,
                is_correct=False,
                presented_at_s=float(presented_at_s),
                answered_at_s=float(answered_at_s),
                response_time_s=float(response_time_s),
                raw="",
                score=0.0,
                max_score=1.0,
                is_timeout=True,
                content_metadata=content_metadata_from_payload(self._current_problem.payload),
            )
        )
        if self._phase is Phase.SCORED:
            self._scored_attempted += 1
            self._scored_max_score += 1.0
        else:
            self._practice_answered += 1

    def _record_locked_answer(self) -> None:
        if (
            self._current_problem is None
            or self._current_prompt is None
            or self._locked_answer_code is None
            or self._locked_answered_at_s is None
        ):
            return
        payload = self._snapshot_payload()
        presented_at_s = self._trial_started_at_s + self._answer_open_at_s()  # type: ignore[operator]
        answered_at_s = float(self._locked_answered_at_s)
        response_time_s = max(0.0, answered_at_s - presented_at_s)
        is_correct = int(self._locked_answer_code) == int(self._current_problem.answer)
        self._events.append(
            QuestionEvent(
                index=len(self._events),
                phase=self._phase,
                prompt=self._current_problem.prompt,
                correct_answer=int(self._current_problem.answer),
                user_answer=int(self._locked_answer_code),
                is_correct=bool(is_correct),
                presented_at_s=float(presented_at_s),
                answered_at_s=float(answered_at_s),
                response_time_s=float(response_time_s),
                raw=str(self._locked_answer_raw),
                score=1.0 if is_correct else 0.0,
                max_score=1.0,
                content_metadata=content_metadata_from_payload(payload),
            )
        )

        if self._phase is Phase.SCORED:
            self._scored_attempted += 1
            self._scored_max_score += 1.0
            self._scored_total_score += 1.0 if is_correct else 0.0
            if is_correct:
                self._scored_correct += 1
        else:
            self._practice_answered += 1

        self._locked_answer_raw = ""
        self._locked_answer_code = None
        self._locked_answered_at_s = None

    def _clear_current(self) -> None:
        self._current_problem = None
        self._current_prompt = None
        self._trial_started_at_s = None
        self._current_prompt_answered = False
        self._locked_answer_raw = ""
        self._locked_answer_code = None
        self._locked_answered_at_s = None

    def _finish(self) -> None:
        self._phase = Phase.RESULTS
        self._clear_current()


def build_trace_test_1_final_attitude(*, prompt: TraceTest1PromptPlan) -> TraceTest1Attitude:
    return trace_test_1_scene_frames(prompt=prompt, progress=1.0).red_frame.attitude
