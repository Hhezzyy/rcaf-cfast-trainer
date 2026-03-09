from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import StrEnum

from .clock import Clock
from .cognitive_core import (
    AttemptSummary,
    Phase,
    Problem,
    QuestionEvent,
    SeededRng,
    TestSnapshot,
    clamp01,
    lerp_int,
)

_STAGE_EPSILON_S = 1e-6


@dataclass(frozen=True, slots=True)
class TraceTest1Config:
    # Candidate guide indicates ~9 minutes total including instructions.
    scored_duration_s: float = 7.5 * 60.0
    practice_questions: int = 3
    practice_observe_s: float = 5.0
    scored_observe_s: float = 4.3


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
    QUESTION = "question"


@dataclass(frozen=True, slots=True)
class TraceTest1Payload:
    trial_stage: TraceTest1TrialStage
    stage_time_remaining_s: float | None
    observe_progress: float
    scene_turn_index: int
    reference: TraceTest1Attitude
    candidate: TraceTest1Attitude
    viewpoint_bearing_deg: int
    options: tuple[TraceTest1Option, ...]
    correct_code: int


@dataclass(frozen=True, slots=True)
class TraceTest1SceneFrame:
    position: tuple[float, float, float]
    attitude: TraceTest1Attitude
    travel_heading_deg: float


_COMMAND_TO_CODE = {
    TraceTest1Command.LEFT: 1,
    TraceTest1Command.RIGHT: 2,
    TraceTest1Command.PUSH: 3,
    TraceTest1Command.PULL: 4,
}
_CODE_TO_COMMAND = {code: command for command, code in _COMMAND_TO_CODE.items()}
_LANE_HEADING_DEG = 0.0
_TARGET_COMMAND_SCRIPT: tuple[TraceTest1Command, ...] = (
    TraceTest1Command.LEFT,
    TraceTest1Command.PUSH,
    TraceTest1Command.RIGHT,
    TraceTest1Command.PULL,
    TraceTest1Command.LEFT,
    TraceTest1Command.RIGHT,
    TraceTest1Command.PUSH,
    TraceTest1Command.PULL,
    TraceTest1Command.RIGHT,
    TraceTest1Command.LEFT,
    TraceTest1Command.PULL,
    TraceTest1Command.PUSH,
)
_DISTRACTOR_COMMAND_SCRIPTS: tuple[tuple[TraceTest1Command, ...], ...] = (
    (
        TraceTest1Command.PULL,
        TraceTest1Command.RIGHT,
        TraceTest1Command.LEFT,
        TraceTest1Command.PUSH,
        TraceTest1Command.RIGHT,
        TraceTest1Command.PULL,
        TraceTest1Command.LEFT,
        TraceTest1Command.RIGHT,
        TraceTest1Command.PUSH,
        TraceTest1Command.PULL,
        TraceTest1Command.LEFT,
        TraceTest1Command.RIGHT,
    ),
    (
        TraceTest1Command.PUSH,
        TraceTest1Command.LEFT,
        TraceTest1Command.PULL,
        TraceTest1Command.RIGHT,
        TraceTest1Command.LEFT,
        TraceTest1Command.PUSH,
        TraceTest1Command.RIGHT,
        TraceTest1Command.LEFT,
        TraceTest1Command.PULL,
        TraceTest1Command.RIGHT,
        TraceTest1Command.PUSH,
        TraceTest1Command.LEFT,
    ),
    (
        TraceTest1Command.RIGHT,
        TraceTest1Command.PULL,
        TraceTest1Command.PUSH,
        TraceTest1Command.LEFT,
        TraceTest1Command.PULL,
        TraceTest1Command.LEFT,
        TraceTest1Command.RIGHT,
        TraceTest1Command.PUSH,
        TraceTest1Command.LEFT,
        TraceTest1Command.RIGHT,
        TraceTest1Command.PULL,
        TraceTest1Command.PUSH,
    ),
)
_SLOT_STAGE_SPLITS: tuple[tuple[float, float], ...] = (
    (0.42, 0.66),
    (0.35, 0.58),
    (0.46, 0.70),
    (0.39, 0.62),
)
_PITCH_COMMAND_DISPLAY_DEG = 90.0


def _wrap_heading(deg: float) -> float:
    value = float(deg) % 360.0
    return value + 360.0 if value < 0.0 else value


def _clamp(v: float, lo: float, hi: float) -> float:
    if v < lo:
        return float(lo)
    if v > hi:
        return float(hi)
    return float(v)


def _lerp(a: float, b: float, t: float) -> float:
    return float(a + ((b - a) * t))


def _smoothstep(t: float) -> float:
    u = _clamp(float(t), 0.0, 1.0)
    return float(u * u * (3.0 - (2.0 * u)))


def _heading_vector_xy(heading_deg: float) -> tuple[float, float]:
    radians = math.radians(float(heading_deg))
    return (math.sin(radians), math.cos(radians))


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


def _move_forward(
    point: tuple[float, float, float],
    *,
    heading_deg: float,
    pitch_deg: float,
    distance: float,
) -> tuple[float, float, float]:
    heading_rad = math.radians(float(heading_deg))
    pitch_rad = math.radians(float(pitch_deg))
    horiz = float(math.cos(pitch_rad) * distance)
    return (
        float(point[0] + (math.sin(heading_rad) * horiz)),
        float(point[1] + (math.cos(heading_rad) * horiz)),
        float(point[2] + (math.sin(pitch_rad) * distance)),
    )


def _yawing_forward_point(
    *,
    start: tuple[float, float, float],
    heading_deg: float,
    turn_deg: float,
    distance: float,
    progress: float,
) -> tuple[tuple[float, float, float], float]:
    u = _clamp(progress, 0.0, 1.0)
    if u <= 0.0 or abs(distance) <= 1e-6:
        return start, _wrap_heading(heading_deg)
    total_turn = float(turn_deg) * u
    total_dist = float(distance) * u
    steps = max(1, int(round(18.0 * u)))
    step_turn = total_turn / steps
    step_dist = total_dist / steps
    pos = start
    heading = float(heading_deg)
    for _ in range(steps):
        mid_heading = heading + (step_turn * 0.5)
        pos = _move_point(pos, heading_deg=mid_heading, distance=step_dist)
        heading += step_turn
    return pos, _wrap_heading(heading)


def _polyline_point(
    waypoints: tuple[tuple[float, float, float], ...],
    *,
    progress: float,
) -> tuple[float, float, float]:
    if len(waypoints) <= 1:
        return waypoints[0]
    t = _clamp(progress, 0.0, 1.0)
    lengths: list[float] = []
    total = 0.0
    for start, end in zip(waypoints, waypoints[1:], strict=False):
        seg = math.dist(start, end)
        lengths.append(seg)
        total += seg
    if total <= 1e-6:
        return waypoints[-1]
    remaining = total * t
    for idx, seg in enumerate(lengths):
        if remaining <= seg or idx == len(lengths) - 1:
            local = 0.0 if seg <= 1e-6 else remaining / seg
            start = waypoints[idx]
            end = waypoints[idx + 1]
            return tuple(
                _lerp(a, b, local) for a, b in zip(start, end, strict=False)
            )
        remaining -= seg
    return waypoints[-1]


def _trace_test_1_path(
    *,
    slot: int,
    command: TraceTest1Command,
) -> tuple[tuple[float, float, float], ...]:
    lane_specs = (
        ((-24.0, 72.0, 8.0), 0.0, 18.0, 16.0, 10.0),
        ((34.0, 78.0, 16.0), 0.0, 18.0, 20.0, 8.0),
        ((-36.0, 62.0, 0.0), 0.0, 20.0, 24.0, 9.0),
        ((30.0, 56.0, 12.0), 0.0, 16.0, 18.0, 7.0),
    )
    start, base_heading, leg1, leg2, climb_mag = lane_specs[slot]
    inward = _move_point(start, heading_deg=base_heading, distance=leg1 * 0.64)
    corner = _move_point(start, heading_deg=base_heading, distance=leg1)

    if command is TraceTest1Command.LEFT:
        exit_heading = base_heading - 90.0
        arc = (
            corner[0] + ((_heading_vector_xy(base_heading)[0] + _heading_vector_xy(exit_heading)[0]) * 3.6),
            corner[1] + ((_heading_vector_xy(base_heading)[1] + _heading_vector_xy(exit_heading)[1]) * 3.6),
            corner[2],
        )
        end = _move_point(corner, heading_deg=exit_heading, distance=leg2)
        return (start, inward, arc, end)

    if command is TraceTest1Command.RIGHT:
        exit_heading = base_heading + 90.0
        arc = (
            corner[0] + ((_heading_vector_xy(base_heading)[0] + _heading_vector_xy(exit_heading)[0]) * 3.6),
            corner[1] + ((_heading_vector_xy(base_heading)[1] + _heading_vector_xy(exit_heading)[1]) * 3.6),
            corner[2],
        )
        end = _move_point(corner, heading_deg=exit_heading, distance=leg2)
        return (start, inward, arc, end)

    if command is TraceTest1Command.PUSH:
        dip = _move_point(corner, heading_deg=base_heading, distance=leg2 * 0.35, climb=-(climb_mag * 0.72))
        end = _move_point(dip, heading_deg=base_heading, distance=leg2 * 0.65, climb=-(climb_mag * 0.28))
        return (start, inward, dip, end)

    lift = _move_point(corner, heading_deg=base_heading, distance=leg2 * 0.35, climb=(climb_mag * 0.72))
    end = _move_point(lift, heading_deg=base_heading, distance=leg2 * 0.65, climb=(climb_mag * 0.28))
    return (start, inward, lift, end)


def _lerp_attitude(
    start: TraceTest1Attitude,
    end: TraceTest1Attitude,
    *,
    t: float,
) -> TraceTest1Attitude:
    u = _smoothstep(t)
    yaw_delta = _wrap_heading(end.yaw_deg - start.yaw_deg)
    if yaw_delta > 180.0:
        yaw_delta -= 360.0
    return TraceTest1Attitude(
        roll_deg=_lerp(start.roll_deg, end.roll_deg, u),
        pitch_deg=_lerp(start.pitch_deg, end.pitch_deg, u),
        yaw_deg=_wrap_heading(start.yaw_deg + (yaw_delta * u)),
    )


def trace_test_1_command_from_code(code: int) -> TraceTest1Command:
    return _CODE_TO_COMMAND.get(int(code), TraceTest1Command.LEFT)


def trace_test_1_apply_command(
    *,
    start: TraceTest1Attitude,
    command: TraceTest1Command,
) -> TraceTest1Attitude:
    if command is TraceTest1Command.LEFT:
        return TraceTest1Attitude(
            roll_deg=-58.0,
            pitch_deg=start.pitch_deg,
            yaw_deg=start.yaw_deg,
        )
    if command is TraceTest1Command.RIGHT:
        return TraceTest1Attitude(
            roll_deg=58.0,
            pitch_deg=start.pitch_deg,
            yaw_deg=start.yaw_deg,
        )
    if command is TraceTest1Command.PUSH:
        return TraceTest1Attitude(
            roll_deg=start.roll_deg,
            pitch_deg=-_PITCH_COMMAND_DISPLAY_DEG,
            yaw_deg=start.yaw_deg,
        )
    return TraceTest1Attitude(
        roll_deg=start.roll_deg,
        pitch_deg=_PITCH_COMMAND_DISPLAY_DEG,
        yaw_deg=start.yaw_deg,
    )


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


def trace_test_1_distractor_attitudes(
    *,
    reference: TraceTest1Attitude,
    target: TraceTest1Attitude,
) -> tuple[TraceTest1Attitude, ...]:
    return (
        reference,
        TraceTest1Attitude(
            roll_deg=_clamp((target.roll_deg * -0.55) + 18.0, -80.0, 80.0),
            pitch_deg=_clamp((reference.pitch_deg * 0.65) - 8.0, -40.0, 40.0),
            yaw_deg=_wrap_heading(reference.yaw_deg + 68.0),
        ),
        TraceTest1Attitude(
            roll_deg=_clamp((reference.roll_deg * 0.42) - 20.0, -80.0, 80.0),
            pitch_deg=_clamp((target.pitch_deg * -0.35) + 12.0, -40.0, 40.0),
            yaw_deg=_wrap_heading(target.yaw_deg - 84.0),
        ),
    )


def trace_test_1_scene_frames(
    *,
    reference: TraceTest1Attitude,
    candidate: TraceTest1Attitude,
    correct_code: int,
    progress: float,
    scene_turn_index: int = 0,
) -> tuple[TraceTest1SceneFrame, tuple[TraceTest1SceneFrame, ...]]:
    target_command = trace_test_1_command_from_code(correct_code)
    target = _trace_test_1_motion_frame(
        start=reference,
        end=candidate,
        command=target_command,
        progress=progress,
        slot=0,
        scene_turn_index=scene_turn_index,
    )

    distractors: list[TraceTest1SceneFrame] = []
    for idx in range(1, 4):
        command = _slot_command_for_turn(slot=idx, turn_index=scene_turn_index)
        start_attitude = _trace_test_1_distractor_start(reference=reference, index=idx)
        end_attitude = trace_test_1_apply_command(start=start_attitude, command=command)
        distractors.append(
            _trace_test_1_motion_frame(
                start=start_attitude,
                end=end_attitude,
                command=command,
                progress=progress,
                slot=idx,
                scene_turn_index=scene_turn_index,
            )
        )
    return target, tuple(distractors)


def _slot_command_for_turn(*, slot: int, turn_index: int) -> TraceTest1Command:
    script_idx = int(turn_index % len(_TARGET_COMMAND_SCRIPT))
    if slot == 0:
        return _TARGET_COMMAND_SCRIPT[script_idx]
    distractor_script = _DISTRACTOR_COMMAND_SCRIPTS[slot - 1]
    return distractor_script[int(turn_index % len(distractor_script))]


def _slot_specs(
    *,
    slot: int,
) -> tuple[tuple[float, float, float], float, float, float, float]:
    lane_specs = (
        ((-18.0, 66.0, 8.0), 0.0, 14.0, 12.0, 6.0),
        ((36.0, 71.0, 16.0), 0.0, 16.0, 18.0, 7.0),
        ((-36.0, 58.0, 2.0), 0.0, 17.0, 20.0, 8.0),
        ((32.0, 54.0, 12.0), 0.0, 15.0, 19.0, 6.0),
    )
    return lane_specs[slot]


def _advance_slot_state(
    *,
    position: tuple[float, float, float],
    heading_deg: float,
    command: TraceTest1Command,
    translate_leg: float,
    exit_leg: float,
    climb_mag: float,
) -> tuple[tuple[float, float, float], float]:
    corner = _move_point(position, heading_deg=heading_deg, distance=translate_leg)
    rotate_forward_leg = min(6.0, max(3.0, translate_leg * 0.28))
    if command is TraceTest1Command.LEFT:
        rotate_end, new_heading = _yawing_forward_point(
            start=corner,
            heading_deg=heading_deg,
            turn_deg=-90.0,
            distance=rotate_forward_leg,
            progress=1.0,
        )
        end_pos = _move_point(rotate_end, heading_deg=new_heading, distance=exit_leg)
        return end_pos, new_heading
    if command is TraceTest1Command.RIGHT:
        rotate_end, new_heading = _yawing_forward_point(
            start=corner,
            heading_deg=heading_deg,
            turn_deg=90.0,
            distance=rotate_forward_leg,
            progress=1.0,
        )
        end_pos = _move_point(rotate_end, heading_deg=new_heading, distance=exit_leg)
        return end_pos, new_heading
    vertical_sign = -1.0 if command is TraceTest1Command.PUSH else 1.0
    end_pos = (
        float(corner[0]),
        float(corner[1]),
        float(corner[2] + (vertical_sign * exit_leg)),
    )
    return end_pos, _wrap_heading(heading_deg)


def _slot_state_at_turn(
    *,
    slot: int,
    turn_index: int,
) -> tuple[tuple[float, float, float], float]:
    position, heading_deg, translate_leg, exit_leg, climb_mag = _slot_specs(slot=slot)
    current_pos = position
    current_heading = float(heading_deg)
    for idx in range(max(0, int(turn_index))):
        command = _slot_command_for_turn(slot=slot, turn_index=idx)
        current_pos, current_heading = _advance_slot_state(
            position=current_pos,
            heading_deg=current_heading,
            command=command,
            translate_leg=translate_leg,
            exit_leg=exit_leg,
            climb_mag=climb_mag,
        )
    return current_pos, current_heading


def _trace_test_1_distractor_start(
    *,
    reference: TraceTest1Attitude,
    index: int,
) -> TraceTest1Attitude:
    starts = (
        TraceTest1Attitude(roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0),
        TraceTest1Attitude(roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0),
        TraceTest1Attitude(roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0),
    )
    return starts[index - 1]


def _trace_test_1_motion_frame(
    *,
    start: TraceTest1Attitude,
    end: TraceTest1Attitude,
    command: TraceTest1Command,
    progress: float,
    slot: int,
    scene_turn_index: int = 0,
) -> TraceTest1SceneFrame:
    _ = start, end
    t = _clamp(progress, 0.0, 1.0)
    start_pos, base_heading = _slot_state_at_turn(
        slot=slot,
        turn_index=scene_turn_index,
    )
    _, _, translate_leg, exit_leg, climb_mag = _slot_specs(slot=slot)
    _ = climb_mag
    corner = _move_point(start_pos, heading_deg=base_heading, distance=translate_leg)
    rotate_forward_leg = min(6.0, max(3.0, translate_leg * 0.28))
    push_pull_display_pitch = (
        -_PITCH_COMMAND_DISPLAY_DEG
        if command is TraceTest1Command.PUSH
        else _PITCH_COMMAND_DISPLAY_DEG
    )
    push_pull_vertical_sign = -1.0 if command is TraceTest1Command.PUSH else 1.0
    push_pull_rotate_vertical = min(
        exit_leg * 0.36,
        max(1.4, rotate_forward_leg * 0.55),
    )
    turn_end_heading = _wrap_heading(base_heading)
    if command is TraceTest1Command.LEFT:
        rotate_end, turn_end_heading = _yawing_forward_point(
            start=corner,
            heading_deg=base_heading,
            turn_deg=-90.0,
            distance=rotate_forward_leg,
            progress=1.0,
        )
    elif command is TraceTest1Command.RIGHT:
        rotate_end, turn_end_heading = _yawing_forward_point(
            start=corner,
            heading_deg=base_heading,
            turn_deg=90.0,
            distance=rotate_forward_leg,
            progress=1.0,
        )
    elif command in (TraceTest1Command.PUSH, TraceTest1Command.PULL):
        # Pitch commands rotate while moving vertically so motion stays continuous.
        rotate_end = (
            float(corner[0]),
            float(corner[1]),
            float(corner[2] + (push_pull_vertical_sign * push_pull_rotate_vertical)),
        )
    else:
        rotate_end = _move_point(corner, heading_deg=base_heading, distance=rotate_forward_leg)
    stage_translate_end, stage_rotate_end = _SLOT_STAGE_SPLITS[slot]

    if t <= stage_translate_end:
        u = _clamp(t / stage_translate_end, 0.0, 1.0)
        pos = _move_point(start_pos, heading_deg=base_heading, distance=translate_leg * u)
        travel_heading_deg = _wrap_heading(base_heading)
        roll_deg = 0.0
        pitch_deg = 0.0
    elif t <= stage_rotate_end:
        u = _clamp((t - stage_translate_end) / (stage_rotate_end - stage_translate_end), 0.0, 1.0)
        if command is TraceTest1Command.LEFT:
            pos, travel_heading_deg = _yawing_forward_point(
                start=corner,
                heading_deg=base_heading,
                turn_deg=-90.0,
                distance=rotate_forward_leg,
                progress=u,
            )
            roll_deg = 0.0
            pitch_deg = 0.0
        elif command is TraceTest1Command.RIGHT:
            pos, travel_heading_deg = _yawing_forward_point(
                start=corner,
                heading_deg=base_heading,
                turn_deg=90.0,
                distance=rotate_forward_leg,
                progress=u,
            )
            roll_deg = 0.0
            pitch_deg = 0.0
        elif command is TraceTest1Command.PUSH:
            travel_heading_deg = _wrap_heading(base_heading)
            roll_deg = 0.0
            pitch_deg = push_pull_display_pitch * u
            pos = (
                float(corner[0]),
                float(corner[1]),
                float(corner[2] + (push_pull_vertical_sign * push_pull_rotate_vertical * u)),
            )
        else:
            travel_heading_deg = _wrap_heading(base_heading)
            roll_deg = 0.0
            pitch_deg = push_pull_display_pitch * u
            pos = (
                float(corner[0]),
                float(corner[1]),
                float(corner[2] + (push_pull_vertical_sign * push_pull_rotate_vertical * u)),
            )
    else:
        u = _clamp((t - stage_rotate_end) / (1.0 - stage_rotate_end), 0.0, 1.0)
        if command is TraceTest1Command.LEFT:
            travel_heading_deg = turn_end_heading
            roll_deg = 0.0
            pitch_deg = 0.0
            pos = _move_point(rotate_end, heading_deg=travel_heading_deg, distance=exit_leg * u)
        elif command is TraceTest1Command.RIGHT:
            travel_heading_deg = turn_end_heading
            roll_deg = 0.0
            pitch_deg = 0.0
            pos = _move_point(rotate_end, heading_deg=travel_heading_deg, distance=exit_leg * u)
        elif command is TraceTest1Command.PUSH:
            travel_heading_deg = _wrap_heading(base_heading)
            roll_deg = 0.0
            pitch_deg = push_pull_display_pitch
            vertical_exit_leg = max(0.0, exit_leg - push_pull_rotate_vertical)
            pos = (
                float(rotate_end[0]),
                float(rotate_end[1]),
                float(rotate_end[2] + (push_pull_vertical_sign * vertical_exit_leg * u)),
            )
        else:
            travel_heading_deg = _wrap_heading(base_heading)
            roll_deg = 0.0
            pitch_deg = push_pull_display_pitch
            vertical_exit_leg = max(0.0, exit_leg - push_pull_rotate_vertical)
            pos = (
                float(rotate_end[0]),
                float(rotate_end[1]),
                float(rotate_end[2] + (push_pull_vertical_sign * vertical_exit_leg * u)),
            )

    attitude = TraceTest1Attitude(
        roll_deg=roll_deg,
        pitch_deg=pitch_deg,
        yaw_deg=float(travel_heading_deg),
    )
    return TraceTest1SceneFrame(
        position=pos,
        attitude=attitude,
        travel_heading_deg=float(travel_heading_deg),
    )


class TraceTest1Generator:
    """Deterministic aircraft trace task with right-angle motion changes."""

    _OPTIONS: tuple[TraceTest1Option, ...] = (
        TraceTest1Option(code=1, label="Left", command=TraceTest1Command.LEFT),
        TraceTest1Option(code=2, label="Right", command=TraceTest1Command.RIGHT),
        TraceTest1Option(code=3, label="Push", command=TraceTest1Command.PUSH),
        TraceTest1Option(code=4, label="Pull", command=TraceTest1Command.PULL),
    )

    def __init__(self, *, seed: int) -> None:
        self._rng = SeededRng(seed)
        self._scene_turn_index = 0

    def next_problem(self, *, difficulty: float) -> Problem:
        _ = difficulty
        viewpoint_bearing_deg = 180
        reference = TraceTest1Attitude(
            roll_deg=0.0,
            pitch_deg=0.0,
            yaw_deg=_LANE_HEADING_DEG,
        )

        command = _slot_command_for_turn(slot=0, turn_index=self._scene_turn_index)

        candidate = trace_test_1_apply_command(start=reference, command=command)

        correct_code = int(_COMMAND_TO_CODE[command])

        payload = TraceTest1Payload(
            trial_stage=TraceTest1TrialStage.QUESTION,
            stage_time_remaining_s=None,
            observe_progress=1.0,
            scene_turn_index=int(self._scene_turn_index),
            reference=reference,
            candidate=candidate,
            viewpoint_bearing_deg=viewpoint_bearing_deg,
            options=self._OPTIONS,
            correct_code=correct_code,
        )
        self._scene_turn_index += 1

        return Problem(
            prompt=(
                "Which way did the stick move for the red aircraft?\n\n"
                "Use the arrow keys: Left, Right, Up (Push), or Down (Pull)."
            ),
            answer=correct_code,
            payload=payload,
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
        self._generator = TraceTest1Generator(seed=self._seed)

        self._phase = Phase.INSTRUCTIONS
        self._current: Problem | None = None
        self._current_payload: TraceTest1Payload | None = None
        self._observe_started_at_s: float | None = None
        self._question_started_at_s: float | None = None
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
            self._finish()
            return
        # Keep the current question active until the user answers.
        # The maneuver animation may finish (observe_progress reaches 1.0),
        # but we no longer auto-advance to the next trial on timeout.
        _ = self._observe_time_remaining_s()

    def time_remaining_s(self) -> float | None:
        if self._phase is not Phase.SCORED or self._scored_started_at_s is None:
            return None
        remaining = self._cfg.scored_duration_s - (self._clock.now() - self._scored_started_at_s)
        return max(0.0, remaining)

    def current_prompt(self) -> str:
        if self._phase is Phase.INSTRUCTIONS:
            return "Watch the red aircraft, ignore the distractors, then answer with the arrow keys."
        if self._phase is Phase.PRACTICE_DONE:
            return "Practice complete. Press Enter to start the timed test."
        if self._phase is Phase.RESULTS:
            s = self.scored_summary()
            acc_pct = int(round(s.accuracy * 100))
            rt = "n/a" if s.mean_response_time_s is None else f"{s.mean_response_time_s:.2f}s"
            return (
                f"Results\nAttempted: {s.attempted}\nCorrect: {s.correct}\n"
                f"Accuracy: {acc_pct}%\nMean RT: {rt}\nThroughput: {s.throughput_per_min:.1f}/min"
            )
        payload = self._current_payload
        if payload is None or self._current is None:
            return ""
        return self._current.prompt

    def submit_answer(self, raw: object) -> bool:
        if self._phase not in (Phase.PRACTICE, Phase.SCORED):
            return False
        payload = self._current_payload
        if payload is None:
            return False
        remaining = self._observe_time_remaining_s()
        if remaining is not None:
            duration = max(0.001, self._observe_duration_s())
            progress = 1.0 - _clamp(remaining / duration, 0.0, 1.0)
            turn_starts_at = float(_SLOT_STAGE_SPLITS[0][0])
            if progress + _STAGE_EPSILON_S < turn_starts_at:
                return False
        if remaining is not None and remaining < 0.0:
            return False
        user_answer = trace_test_1_answer_code(raw)
        if user_answer is None:
            return False

        assert self._current is not None
        assert self._question_started_at_s is not None
        answered_at_s = self._clock.now()
        response_time_s = max(0.0, answered_at_s - self._question_started_at_s)
        is_correct = int(user_answer) == int(self._current.answer)
        self._events.append(
            QuestionEvent(
                index=len(self._events),
                phase=self._phase,
                prompt=self._current.prompt,
                correct_answer=int(self._current.answer),
                user_answer=int(user_answer),
                is_correct=bool(is_correct),
                presented_at_s=float(self._question_started_at_s),
                answered_at_s=float(answered_at_s),
                response_time_s=float(response_time_s),
                raw=str(raw),
                score=1.0 if is_correct else 0.0,
                max_score=1.0,
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

        if self._phase is Phase.PRACTICE and self._practice_answered >= self._cfg.practice_questions:
            self._phase = Phase.PRACTICE_DONE
            self._clear_current()
            return True

        if self._phase is Phase.SCORED and self.time_remaining_s() == 0.0:
            self._finish()
            return True

        self._deal_new_problem()
        return True

    def scored_summary(self) -> AttemptSummary:
        duration_s = float(self._cfg.scored_duration_s)
        attempted = int(self._scored_attempted)
        correct = int(self._scored_correct)
        accuracy = 0.0 if attempted == 0 else correct / attempted
        throughput = (attempted / duration_s) * 60.0
        rts = [e.response_time_s for e in self._events if e.phase is Phase.SCORED]
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

    def snapshot(self) -> TestSnapshot:
        payload = self._snapshot_payload()
        return TestSnapshot(
            title="Trace Test 1",
            phase=self._phase,
            prompt=self.current_prompt(),
            input_hint="Answer with arrow keys. Up = Push, Down = Pull.",
            time_remaining_s=self.time_remaining_s(),
            attempted_scored=self._scored_attempted,
            correct_scored=self._scored_correct,
            payload=payload,
        )

    def _observe_duration_s(self) -> float:
        return (
            float(self._cfg.practice_observe_s)
            if self._phase is Phase.PRACTICE
            else float(self._cfg.scored_observe_s)
        )

    def _observe_time_remaining_s(self) -> float | None:
        if self._observe_started_at_s is None:
            return None
        duration = max(0.001, self._observe_duration_s())
        elapsed = max(0.0, self._clock.now() - self._observe_started_at_s)
        remaining = duration - elapsed
        if remaining <= _STAGE_EPSILON_S:
            return 0.0
        return float(remaining)

    def _snapshot_payload(self) -> TraceTest1Payload | None:
        payload = self._current_payload
        if payload is None:
            return None
        duration = max(0.001, self._observe_duration_s())
        remaining = self._observe_time_remaining_s()
        if remaining is None:
            remaining = duration
        progress = 1.0 - _clamp(remaining / duration, 0.0, 1.0)
        return replace(
            payload,
            trial_stage=TraceTest1TrialStage.QUESTION,
            stage_time_remaining_s=float(remaining),
            observe_progress=float(progress),
        )

    def _deal_new_problem(self) -> None:
        self._current = self._generator.next_problem(difficulty=self._difficulty)
        base_payload = self._current.payload
        assert isinstance(base_payload, TraceTest1Payload)
        now_s = self._clock.now()
        self._current_payload = replace(
            base_payload,
            trial_stage=TraceTest1TrialStage.QUESTION,
            stage_time_remaining_s=float(self._observe_duration_s()),
            observe_progress=0.0,
        )
        self._observe_started_at_s = now_s
        self._question_started_at_s = now_s

    def _expire_current_problem(self) -> None:
        if self._current is None or self._question_started_at_s is None:
            return
        answered_at_s = self._clock.now()
        response_time_s = max(0.0, answered_at_s - self._question_started_at_s)
        self._events.append(
            QuestionEvent(
                index=len(self._events),
                phase=self._phase,
                prompt=self._current.prompt,
                correct_answer=int(self._current.answer),
                user_answer=0,
                is_correct=False,
                presented_at_s=float(self._question_started_at_s),
                answered_at_s=float(answered_at_s),
                response_time_s=float(response_time_s),
                raw="TIMEOUT",
                score=0.0,
                max_score=1.0,
            )
        )

        if self._phase is Phase.SCORED:
            self._scored_attempted += 1
            self._scored_max_score += 1.0
        else:
            self._practice_answered += 1

        if self._phase is Phase.PRACTICE and self._practice_answered >= self._cfg.practice_questions:
            self._phase = Phase.PRACTICE_DONE
            self._clear_current()
            return

        if self._phase is Phase.SCORED and self.time_remaining_s() == 0.0:
            self._finish()
            return

        self._deal_new_problem()

    def _set_trial_stage(self, stage: TraceTest1TrialStage) -> None:
        if self._current_payload is None:
            return
        self._current_payload = replace(
            self._current_payload,
            trial_stage=stage,
            stage_time_remaining_s=None if stage is TraceTest1TrialStage.QUESTION else self._observe_duration_s(),
            observe_progress=1.0 if stage is TraceTest1TrialStage.QUESTION else 0.0,
        )
        if stage is TraceTest1TrialStage.QUESTION:
            self._question_started_at_s = self._clock.now()

    def _clear_current(self) -> None:
        self._current = None
        self._current_payload = None
        self._observe_started_at_s = None
        self._question_started_at_s = None

    def _finish(self) -> None:
        self._phase = Phase.RESULTS
        self._clear_current()


def trace_test_1_aircraft_marker_offsets(
    *,
    attitude: TraceTest1Attitude,
    viewpoint_bearing_deg: int,
    scale: float = 1.0,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Returns projected offsets for (nose, left_wing, right_wing, tail)."""

    roll = math.radians(float(attitude.roll_deg))
    pitch = _clamp(float(attitude.pitch_deg), -45.0, 45.0)
    yaw_relative = math.radians(
        _wrap_heading(float(attitude.yaw_deg) - float(viewpoint_bearing_deg))
    )

    pitch_squash = 1.0 - (abs(pitch) / 150.0)
    yaw_shear = math.sin(yaw_relative) * 0.25
    yaw_vertical = math.cos(yaw_relative) * 0.14

    points = (
        (0.0, -26.0),  # nose
        (-24.0, 0.0),  # left wing
        (24.0, 0.0),  # right wing
        (0.0, 19.0),  # tail
    )

    projected: list[tuple[float, float]] = []
    for x, y in points:
        xr = (x * math.cos(roll)) - (y * math.sin(roll))
        yr = (x * math.sin(roll)) + (y * math.cos(roll))
        yr *= pitch_squash

        xr += y * yaw_shear
        yr += x * yaw_vertical

        if y < 0.0:
            yr -= pitch * 0.40
        elif y > 0.0:
            yr += pitch * 0.40

        projected.append((xr * scale, yr * scale))

    return (
        projected[0],
        projected[1],
        projected[2],
        projected[3],
    )
