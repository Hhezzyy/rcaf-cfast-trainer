from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from .clock import Clock
from .cognitive_core import (
    Problem,
    SeededRng,
    TimedTextInputTest,
    clamp01,
    lerp_int,
)


@dataclass(frozen=True, slots=True)
class TraceTest1Config:
    # Candidate guide indicates ~9 minutes total including instructions.
    scored_duration_s: float = 7.5 * 60.0
    practice_questions: int = 3


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


@dataclass(frozen=True, slots=True)
class TraceTest1Payload:
    reference: TraceTest1Attitude
    candidate: TraceTest1Attitude
    viewpoint_bearing_deg: int
    options: tuple[TraceTest1Option, ...]
    correct_code: int


def _wrap_heading(deg: float) -> float:
    value = float(deg) % 360.0
    return value + 360.0 if value < 0.0 else value


def _clamp(v: float, lo: float, hi: float) -> float:
    if v < lo:
        return float(lo)
    if v > hi:
        return float(hi)
    return float(v)


class TraceTest1Generator:
    """Deterministic 3-D orientation discrimination task."""

    _OPTIONS: tuple[TraceTest1Option, ...] = (
        TraceTest1Option(code=1, label="Left", command=TraceTest1Command.LEFT),
        TraceTest1Option(code=2, label="Right", command=TraceTest1Command.RIGHT),
        TraceTest1Option(code=3, label="Push", command=TraceTest1Command.PUSH),
        TraceTest1Option(code=4, label="Pull", command=TraceTest1Command.PULL),
    )

    def __init__(self, *, seed: int) -> None:
        self._rng = SeededRng(seed)

    def next_problem(self, *, difficulty: float) -> Problem:
        d = clamp01(difficulty)

        reference = TraceTest1Attitude(
            roll_deg=float(self._rng.randint(-35, 35)),
            pitch_deg=float(self._rng.randint(-18, 18)),
            yaw_deg=float(self._rng.randint(0, 359)),
        )

        command = self._rng.choice(
            (
                TraceTest1Command.LEFT,
                TraceTest1Command.RIGHT,
                TraceTest1Command.PUSH,
                TraceTest1Command.PULL,
            )
        )

        roll_step = float(lerp_int(34, 16, d))
        pitch_step = float(lerp_int(26, 10, d))
        yaw_couple = float(lerp_int(18, 8, d))
        pitch_yaw_couple = float(lerp_int(12, 5, d))

        if command is TraceTest1Command.LEFT:
            candidate = TraceTest1Attitude(
                roll_deg=_clamp(reference.roll_deg - roll_step, -85.0, 85.0),
                pitch_deg=reference.pitch_deg,
                yaw_deg=_wrap_heading(reference.yaw_deg - yaw_couple),
            )
        elif command is TraceTest1Command.RIGHT:
            candidate = TraceTest1Attitude(
                roll_deg=_clamp(reference.roll_deg + roll_step, -85.0, 85.0),
                pitch_deg=reference.pitch_deg,
                yaw_deg=_wrap_heading(reference.yaw_deg + yaw_couple),
            )
        elif command is TraceTest1Command.PUSH:
            candidate = TraceTest1Attitude(
                roll_deg=reference.roll_deg,
                pitch_deg=_clamp(reference.pitch_deg - pitch_step, -45.0, 45.0),
                yaw_deg=_wrap_heading(reference.yaw_deg + pitch_yaw_couple),
            )
        else:
            candidate = TraceTest1Attitude(
                roll_deg=reference.roll_deg,
                pitch_deg=_clamp(reference.pitch_deg + pitch_step, -45.0, 45.0),
                yaw_deg=_wrap_heading(reference.yaw_deg - pitch_yaw_couple),
            )

        command_to_code = {option.command: int(option.code) for option in self._OPTIONS}
        correct_code = int(command_to_code[command])

        payload = TraceTest1Payload(
            reference=reference,
            candidate=candidate,
            viewpoint_bearing_deg=int(self._rng.randint(0, 359)),
            options=self._OPTIONS,
            correct_code=correct_code,
        )

        prompt_lines = [
            "Which control input best explains the red aircraft orientation?",
            "",
        ]
        prompt_lines.extend(f"{option.code}) {option.label}" for option in self._OPTIONS)

        return Problem(
            prompt="\n".join(prompt_lines),
            answer=correct_code,
            payload=payload,
        )


def build_trace_test_1_test(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    config: TraceTest1Config | None = None,
) -> TimedTextInputTest:
    cfg = config or TraceTest1Config()

    instructions = [
        "Trace Test 1",
        "",
        "Assess orientation changes in 3-D space.",
        "The blue aircraft is the reference; the red aircraft shows a changed orientation.",
        "Choose which command best matches that change.",
        "",
        "Commands:",
        "- Left: bank/turn left",
        "- Right: bank/turn right",
        "- Push: nose moves down",
        "- Pull: nose moves up",
        "",
        "Controls:",
        "- Press 1-4 (or type 1-4)",
        "- Press Enter to submit",
        "",
        "You will get a short practice, then a timed scored block.",
    ]

    return TimedTextInputTest(
        title="Trace Test 1",
        instructions=instructions,
        generator=TraceTest1Generator(seed=seed),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        practice_questions=cfg.practice_questions,
        scored_duration_s=cfg.scored_duration_s,
    )


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
