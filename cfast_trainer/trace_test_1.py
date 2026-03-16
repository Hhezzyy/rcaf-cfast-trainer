from __future__ import annotations

import math
from dataclasses import dataclass
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

_STAGE_EPSILON_S = 1e-6
_VIEWPOINT_BEARING_DEG = 180
_RED_SAFE_BOX_MIN = 0.15
_RED_SAFE_BOX_MAX = 0.85
_BLUE_MARGIN = 0.20
_WORLD_X_NORMALIZER = 48.0
_WORLD_Z_SCREEN_CENTER = 12.0
_WORLD_Z_SCREEN_NORMALIZER = 36.0
_RED_ALTITUDE_BAND = (6.0, 22.0)
_RED_DEPTH_BAND = (4.0, 48.0)
_BLUE_DEPTH_BAND = (-18.0, 70.0)
_RED_RECOVERY_DISTANCE = 3.5
_TURN_HOLD_FRACTION = 0.14
_BLUE_OUTLINE_COLORS: tuple[tuple[int, int, int], ...] = (
    (220, 232, 246),
    (202, 226, 250),
    (188, 214, 244),
    (176, 206, 242),
)


@dataclass(frozen=True, slots=True)
class TraceTest1Config:
    # Candidate guide indicates about 9 minutes total including instructions.
    scored_duration_s: float = 7.5 * 60.0
    practice_questions: int = 3
    practice_observe_s: float = 5.0
    scored_observe_s: float = 4.3
    allowed_commands: tuple["TraceTest1Command", ...] | None = None


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


@dataclass(frozen=True, slots=True)
class TraceTest1SceneSnapshot:
    red_frame: TraceTest1SceneFrame
    blue_frames: tuple[TraceTest1SceneFrame, ...]


@dataclass(frozen=True, slots=True)
class TraceTest1DifficultyTier:
    blue_count: int
    speed_multiplier: float
    answer_open_progress: float


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


def _heading_vector_xy(heading_deg: float) -> tuple[float, float]:
    radians = math.radians(float(heading_deg))
    return (math.sin(radians), math.cos(radians))


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
        return TraceTest1DifficultyTier(blue_count=1, speed_multiplier=1.00, answer_open_progress=0.42)
    if d < 0.67:
        return TraceTest1DifficultyTier(blue_count=2, speed_multiplier=1.15, answer_open_progress=0.36)
    if d < 0.89:
        return TraceTest1DifficultyTier(blue_count=3, speed_multiplier=1.30, answer_open_progress=0.31)
    return TraceTest1DifficultyTier(blue_count=4, speed_multiplier=1.45, answer_open_progress=0.27)


def trace_test_1_normalized_position(
    position: tuple[float, float, float],
) -> tuple[float, float]:
    world_x, _world_y, world_z = position
    return (
        float(0.5 + (world_x / _WORLD_X_NORMALIZER)),
        float(0.64 - ((world_z - _WORLD_Z_SCREEN_CENTER) / _WORLD_Z_SCREEN_NORMALIZER)),
    )


def _scene_frame_for_plan(
    *,
    plan: TraceTest1AircraftPlan,
    answer_open_progress: float,
    progress: float,
) -> TraceTest1SceneFrame:
    t = _clamp(progress, 0.0, 1.0)
    split = _clamp(answer_open_progress, 0.12, 0.88)
    lead_t = 1.0 if t >= split else _clamp(t / max(0.001, split), 0.0, 1.0)
    answer_t = 0.0 if t <= split else _clamp((t - split) / max(0.001, 1.0 - split), 0.0, 1.0)

    start = plan.start_state.position
    start_heading = _cardinal_heading(plan.start_state.heading_deg)
    corner = _move_point(start, heading_deg=start_heading, distance=plan.lead_distance)
    lead_position = _move_point(start, heading_deg=start_heading, distance=plan.lead_distance * lead_t)
    position = lead_position
    travel_heading = start_heading
    roll_deg = 0.0
    pitch_deg = 0.0

    if answer_t > 0.0:
        if plan.command is TraceTest1Command.LEFT:
            travel_heading = _cardinal_heading(start_heading - 90.0)
            move_t = 0.0 if answer_t <= _TURN_HOLD_FRACTION else _clamp(
                (answer_t - _TURN_HOLD_FRACTION) / max(0.001, 1.0 - _TURN_HOLD_FRACTION),
                0.0,
                1.0,
            )
            position = _move_point(corner, heading_deg=travel_heading, distance=plan.maneuver_distance * move_t)
        elif plan.command is TraceTest1Command.RIGHT:
            travel_heading = _cardinal_heading(start_heading + 90.0)
            move_t = 0.0 if answer_t <= _TURN_HOLD_FRACTION else _clamp(
                (answer_t - _TURN_HOLD_FRACTION) / max(0.001, 1.0 - _TURN_HOLD_FRACTION),
                0.0,
                1.0,
            )
            position = _move_point(corner, heading_deg=travel_heading, distance=plan.maneuver_distance * move_t)
        else:
            position = _move_point(
                corner,
                heading_deg=start_heading,
                distance=plan.maneuver_distance * answer_t,
                climb=plan.altitude_delta * answer_t,
            )
            travel_heading = start_heading
            pitch_deg = -34.0 if plan.command is TraceTest1Command.PUSH else 34.0

    return TraceTest1SceneFrame(
        position=position,
        attitude=TraceTest1Attitude(
            roll_deg=roll_deg,
            pitch_deg=pitch_deg,
            yaw_deg=float(travel_heading),
        ),
        travel_heading_deg=float(travel_heading),
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
    ) -> None:
        self._rng = SeededRng(seed)
        self._allowed_commands = _normalize_allowed_commands(allowed_commands)
        self._prompt_index = 0
        self._last_red_command: TraceTest1Command | None = None
        self._tier: TraceTest1DifficultyTier | None = None
        self._red_state = TraceTest1AircraftState(position=(0.0, 8.0, 12.0), heading_deg=0.0)
        self._blue_states: tuple[TraceTest1AircraftState, ...] = ()
        self._pending_prompt: TraceTest1PromptPlan | None = None

    def next_problem(self, *, difficulty: float) -> Problem:
        tier = trace_test_1_difficulty_tier(difficulty=difficulty)
        self._ensure_tier_initialized(tier=tier)
        if self._pending_prompt is not None:
            self.commit_prompt(prompt=self._pending_prompt, progress=1.0)

        red_plan = self._build_red_plan(tier=tier)
        blue_plans = tuple(self._build_blue_plan(state=state, tier=tier) for state in self._blue_states)
        prompt = TraceTest1PromptPlan(
            prompt_index=int(self._prompt_index),
            answer_open_progress=float(tier.answer_open_progress),
            speed_multiplier=float(tier.speed_multiplier),
            red_plan=red_plan,
            blue_plans=blue_plans,
        )
        self._pending_prompt = prompt
        self._prompt_index += 1
        correct_code = int(_COMMAND_TO_CODE[red_plan.command])
        return Problem(
            prompt=f"Trace Test 1 {red_plan.command.value}",
            answer=correct_code,
            payload=prompt,
        )

    def commit_prompt(
        self,
        *,
        prompt: TraceTest1PromptPlan,
        progress: float,
    ) -> None:
        scene = trace_test_1_scene_frames(prompt=prompt, progress=progress)
        self._red_state = _aircraft_state_from_frame(scene.red_frame)
        self._blue_states = tuple(
            self._advance_blue_state(frame=frame, tier=self._tier)
            for frame in scene.blue_frames
        )
        self._pending_prompt = None

    def _ensure_tier_initialized(self, *, tier: TraceTest1DifficultyTier) -> None:
        if self._tier == tier and len(self._blue_states) == tier.blue_count:
            return
        self._tier = tier
        if not self._blue_states:
            self._blue_states = tuple(self._spawn_initial_blue(index=idx) for idx in range(tier.blue_count))
            return
        current = list(self._blue_states[: tier.blue_count])
        while len(current) < tier.blue_count:
            current.append(self._spawn_initial_blue(index=len(current)))
        self._blue_states = tuple(current)

    def _spawn_initial_blue(self, *, index: int) -> TraceTest1AircraftState:
        heading = (0.0, 90.0, 270.0, 180.0)[index % 4]
        if heading == 0.0:
            position = (
                float(-18.0 + self._rng.uniform(-4.0, 4.0) + (index * 7.0)),
                float(-10.0 + self._rng.uniform(-6.0, 4.0)),
                float(9.0 + self._rng.uniform(-2.0, 4.0)),
            )
        elif heading == 90.0:
            position = (
                float(-34.0 + self._rng.uniform(-4.0, 0.0)),
                float(16.0 + self._rng.uniform(0.0, 28.0)),
                float(10.0 + self._rng.uniform(-2.0, 5.0)),
            )
        elif heading == 270.0:
            position = (
                float(34.0 + self._rng.uniform(0.0, 4.0)),
                float(14.0 + self._rng.uniform(0.0, 28.0)),
                float(10.0 + self._rng.uniform(-2.0, 5.0)),
            )
        else:
            position = (
                float(-16.0 + self._rng.uniform(-8.0, 8.0)),
                float(58.0 + self._rng.uniform(-6.0, 10.0)),
                float(10.0 + self._rng.uniform(-2.0, 5.0)),
            )
        return TraceTest1AircraftState(position=position, heading_deg=heading)

    def _prompt_distances(self, *, tier: TraceTest1DifficultyTier) -> tuple[float, float, float]:
        speed = float(tier.speed_multiplier)
        return (4.8 * speed, 4.1 * speed, 2.2 * speed)

    def _build_red_plan(self, *, tier: TraceTest1DifficultyTier) -> TraceTest1AircraftPlan:
        lead_distance, maneuver_distance, altitude_delta = self._prompt_distances(tier=tier)
        for _ in range(5):
            candidates = [
                command for command in self._allowed_commands if command is not self._last_red_command
            ]
            if not candidates:
                candidates = list(self._allowed_commands)
            for command in self._rng.sample(candidates, k=len(candidates)):
                plan = TraceTest1AircraftPlan(
                    start_state=self._red_state,
                    command=command,
                    lead_distance=lead_distance,
                    maneuver_distance=maneuver_distance,
                    altitude_delta=altitude_delta if command is TraceTest1Command.PULL else -altitude_delta,
                )
                if self._red_plan_is_safe(plan=plan, tier=tier):
                    self._last_red_command = command
                    return plan
            self._red_state = self._advance_red_recovery()

        fallback_command = next(
            (
                command
                for command in self._allowed_commands
                if command is not self._last_red_command
            ),
            self._allowed_commands[0],
        )
        fallback = TraceTest1AircraftPlan(
            start_state=self._red_state,
            command=fallback_command,
            lead_distance=lead_distance,
            maneuver_distance=maneuver_distance,
            altitude_delta=altitude_delta if fallback_command is TraceTest1Command.PULL else -altitude_delta,
        )
        self._last_red_command = fallback.command
        return fallback

    def _red_plan_is_safe(self, *, plan: TraceTest1AircraftPlan, tier: TraceTest1DifficultyTier) -> bool:
        end_heading = _prompt_end_heading(plan)
        if int(round(end_heading)) % 360 == 180:
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
        return all(_projected_safe(position) for position in scene_points)

    def _advance_red_recovery(self) -> TraceTest1AircraftState:
        x, y, z = self._red_state.position
        if y > (_RED_DEPTH_BAND[1] - 3.0):
            if x > 0.5:
                heading = 270.0
            elif x < -0.5:
                heading = 90.0
            else:
                heading = self._rng.choice((90.0, 270.0))
        elif x > 8.0:
            heading = 270.0
        elif x < -8.0:
            heading = 90.0
        else:
            heading = 0.0
        position = _move_point(
            (x, y, z),
            heading_deg=heading,
            distance=_RED_RECOVERY_DISTANCE,
            climb=_clamp(12.0 - z, -1.4, 1.4),
        )
        return TraceTest1AircraftState(
            position=(
                float(_clamp(position[0], -20.0, 20.0)),
                float(_clamp(position[1], _RED_DEPTH_BAND[0], _RED_DEPTH_BAND[1])),
                float(_clamp(position[2], _RED_ALTITUDE_BAND[0], _RED_ALTITUDE_BAND[1])),
            ),
            heading_deg=heading,
        )

    def _build_blue_plan(
        self,
        *,
        state: TraceTest1AircraftState,
        tier: TraceTest1DifficultyTier,
    ) -> TraceTest1AircraftPlan:
        lead_distance, maneuver_distance, altitude_delta = self._prompt_distances(tier=tier)
        command = self._rng.choice(tuple(TraceTest1Command))
        return TraceTest1AircraftPlan(
            start_state=state,
            command=command,
            lead_distance=lead_distance,
            maneuver_distance=maneuver_distance,
            altitude_delta=altitude_delta if command is TraceTest1Command.PULL else -altitude_delta,
        )

    def _advance_blue_state(
        self,
        *,
        frame: TraceTest1SceneFrame,
        tier: TraceTest1DifficultyTier | None,
    ) -> TraceTest1AircraftState:
        state = _aircraft_state_from_frame(frame)
        if not _offscreen_blue(frame.position):
            return state
        heading = _wrap_heading(state.heading_deg)
        altitude = float(_clamp(frame.position[2], 7.0, 24.0))
        if heading == 0.0:
            position = (
                float(self._rng.uniform(-22.0, 22.0)),
                float(-14.0 + self._rng.uniform(-4.0, 3.0)),
                altitude,
            )
        elif heading == 180.0:
            position = (
                float(self._rng.uniform(-22.0, 22.0)),
                float(66.0 + self._rng.uniform(0.0, 8.0)),
                altitude,
            )
        elif heading == 90.0:
            position = (
                float(-34.0 + self._rng.uniform(-5.0, -1.0)),
                float(self._rng.uniform(6.0, 54.0)),
                altitude,
            )
        else:
            position = (
                float(34.0 + self._rng.uniform(1.0, 5.0)),
                float(self._rng.uniform(6.0, 54.0)),
                altitude,
            )
        return TraceTest1AircraftState(position=position, heading_deg=heading)


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
        self._generator = TraceTest1Generator(seed=self._seed, allowed_commands=allowed_commands)

        self._phase = Phase.INSTRUCTIONS
        self._current_problem: Problem | None = None
        self._current_prompt: TraceTest1PromptPlan | None = None
        self._trial_started_at_s: float | None = None
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
        if self._current_prompt is None:
            return
        if self._elapsed_in_prompt_s() + _STAGE_EPSILON_S < self._prompt_duration_s():
            return
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
                "Answer with the arrow keys when the red aircraft changes:\n"
                "Left, Right, Up for Push, and Down for Pull."
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
        payload = self._snapshot_payload()
        if payload is None or payload.trial_stage is not TraceTest1TrialStage.ANSWER_OPEN:
            return False
        user_answer = trace_test_1_answer_code(raw)
        if user_answer is None or self._current_problem is None or self._current_prompt is None:
            return False
        answered_at_s = self._clock.now()
        presented_at_s = self._trial_started_at_s + self._answer_open_at_s()  # type: ignore[operator]
        response_time_s = max(0.0, answered_at_s - presented_at_s)
        is_correct = int(user_answer) == int(self._current_problem.answer)
        self._events.append(
            QuestionEvent(
                index=len(self._events),
                phase=self._phase,
                prompt=self._current_problem.prompt,
                correct_answer=int(self._current_problem.answer),
                user_answer=int(user_answer),
                is_correct=bool(is_correct),
                presented_at_s=float(presented_at_s),
                answered_at_s=float(answered_at_s),
                response_time_s=float(response_time_s),
                raw=str(raw),
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

        self._advance_to_next_prompt(progress=self._current_progress())
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
            input_hint="Arrow keys answer immediately once the red maneuver begins.",
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
                content_metadata=content_metadata_from_payload(self._current_problem.payload),
            )
        )
        if self._phase is Phase.SCORED:
            self._scored_attempted += 1
            self._scored_max_score += 1.0
        else:
            self._practice_answered += 1

    def _clear_current(self) -> None:
        self._current_problem = None
        self._current_prompt = None
        self._trial_started_at_s = None

    def _finish(self) -> None:
        self._phase = Phase.RESULTS
        self._clear_current()


def build_trace_test_1_final_attitude(*, prompt: TraceTest1PromptPlan) -> TraceTest1Attitude:
    return trace_test_1_scene_frames(prompt=prompt, progress=1.0).red_frame.attitude
