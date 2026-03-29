from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import StrEnum

from .clock import Clock
from .content_variants import content_metadata_from_payload, stable_variant_id
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
    TraceLatticeState,
    trace_lattice_build_path,
    trace_lattice_node_point,
    trace_lattice_sample_path,
    trace_lattice_state,
)

_STAGE_EPSILON_S = 1e-6
_TT2_TURN_PHASE_RATIO = 0.35
_TT2_WORLD_COL_SPACING = 22.0
_TT2_WORLD_ROW_SPACING = 24.0
_TT2_WORLD_LEVEL_SPACING = 8.0
_TT2_WORLD_BASE_DEPTH = 42.0
_TT2_WORLD_BASE_ALTITUDE = 9.0


@dataclass(frozen=True, slots=True)
class TraceTest2Config:
    # Candidate guide indicates about 9 minutes total including instructions.
    scored_duration_s: float = 7.5 * 60.0
    practice_questions: int = 2
    practice_observe_s: float = 5.5
    scored_observe_s: float = 4.8
    allowed_question_kinds: tuple["TraceTest2QuestionKind", ...] | None = None


class TraceTest2TrialStage(StrEnum):
    OBSERVE = "observe"
    QUESTION = "question"


class TraceTest2QuestionKind(StrEnum):
    NO_DIRECTION_CHANGE = "no_direction_change"
    TURNED_LEFT = "turned_left"
    TURNED_RIGHT = "turned_right"
    ENDED_LEFTMOST = "ended_leftmost"
    ENDED_RIGHTMOST = "ended_rightmost"
    ENDED_HIGHEST = "ended_highest"
    ENDED_LOWEST = "ended_lowest"


class TraceTest2MotionKind(StrEnum):
    STRAIGHT = "straight"
    LEFT = "left"
    RIGHT = "right"
    CLIMB = "climb"
    DESCEND = "descend"


@dataclass(frozen=True, slots=True)
class TraceTest2Point3:
    x: float
    y: float
    z: float


@dataclass(frozen=True, slots=True)
class TraceTest2AircraftTrack:
    code: int
    color_name: str
    color_rgb: tuple[int, int, int]
    waypoints: tuple[TraceTest2Point3, ...]
    motion_kind: TraceTest2MotionKind
    direction_changed: bool
    ended_screen_x: float
    ended_altitude_z: float
    lattice_path: TraceLatticePath | None = None
    lateral_scale: float = 1.0
    vertical_scale: float = 1.0


@dataclass(frozen=True, slots=True)
class TraceTest2Option:
    code: int
    label: str
    color_name: str | None = None
    color_rgb: tuple[int, int, int] | None = None


@dataclass(frozen=True, slots=True)
class TraceTest2Payload:
    trial_stage: TraceTest2TrialStage
    stage_time_remaining_s: float | None
    observe_progress: float
    block_kind: str
    trial_index_in_block: int
    trials_in_block: int
    question_kind: TraceTest2QuestionKind
    stem: str
    viewpoint_bearing_deg: int
    aircraft: tuple[TraceTest2AircraftTrack, ...]
    options: tuple[TraceTest2Option, ...]
    correct_code: int
    content_family: str = ""
    variant_id: str = ""
    content_pack: str = ""


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(lo if v < lo else hi if v > hi else v)


def _lerp(a: float, b: float, t: float) -> float:
    return float(a + ((b - a) * t))


def _point_lerp(a: TraceTest2Point3, b: TraceTest2Point3, t: float) -> TraceTest2Point3:
    return TraceTest2Point3(
        x=_lerp(a.x, b.x, t),
        y=_lerp(a.y, b.y, t),
        z=_lerp(a.z, b.z, t),
    )


def _quadratic_point(
    a: TraceTest2Point3,
    b: TraceTest2Point3,
    c: TraceTest2Point3,
    t: float,
) -> TraceTest2Point3:
    u = _clamp(float(t), 0.0, 1.0)
    one_minus = 1.0 - u
    return TraceTest2Point3(
        x=(one_minus * one_minus * a.x) + (2.0 * one_minus * u * b.x) + (u * u * c.x),
        y=(one_minus * one_minus * a.y) + (2.0 * one_minus * u * b.y) + (u * u * c.y),
        z=(one_minus * one_minus * a.z) + (2.0 * one_minus * u * b.z) + (u * u * c.z),
    )


def _direction_changed(waypoints: tuple[TraceTest2Point3, ...]) -> bool:
    if len(waypoints) < 3:
        return False
    prev = waypoints[1]
    start = waypoints[0]
    end = waypoints[2]
    first = (prev.x - start.x, prev.y - start.y, prev.z - start.z)
    second = (end.x - prev.x, end.y - prev.y, end.z - prev.z)
    delta = math.dist(first, second)
    return delta > 0.5


def _screen_metric(point: TraceTest2Point3) -> tuple[float, float]:
    screen_x = float(point.x * 1.1)
    screen_y = float(((128.0 - point.y) * 1.26) - ((point.z - 8.0) * 2.1))
    return screen_x, screen_y


def _tt2_world_point_from_lattice(
    point: tuple[float, float, float],
    *,
    lateral_scale: float,
    vertical_scale: float,
) -> TraceTest2Point3:
    return TraceTest2Point3(
        x=float(point[0] * _TT2_WORLD_COL_SPACING * float(lateral_scale)),
        y=float(_TT2_WORLD_BASE_DEPTH + (point[1] * _TT2_WORLD_ROW_SPACING)),
        z=float(_TT2_WORLD_BASE_ALTITUDE + (point[2] * _TT2_WORLD_LEVEL_SPACING * float(vertical_scale))),
    )


def _tt2_direction_changed_for_path(path: TraceLatticePath) -> bool:
    return any(step.effective_action is not TraceLatticeAction.STRAIGHT for step in path.steps)


def _tt2_motion_kind_for_path(path: TraceLatticePath) -> TraceTest2MotionKind:
    for step in path.steps:
        if step.effective_action is TraceLatticeAction.LEFT:
            return TraceTest2MotionKind.LEFT
        if step.effective_action is TraceLatticeAction.RIGHT:
            return TraceTest2MotionKind.RIGHT
        if step.effective_action is TraceLatticeAction.PULL:
            return TraceTest2MotionKind.CLIMB
        if step.effective_action is TraceLatticeAction.PUSH:
            return TraceTest2MotionKind.DESCEND
    return TraceTest2MotionKind.STRAIGHT


def _tt2_is_high_difficulty(difficulty: float) -> bool:
    return clamp01(difficulty) >= 0.72


def _tt2_actions_for_role(
    *,
    role_key: str,
    difficulty: float,
) -> tuple[TraceLatticeAction, ...]:
    high_difficulty = _tt2_is_high_difficulty(difficulty)
    cruise_steps = 5 if high_difficulty else 4
    if role_key == "steady":
        return (TraceLatticeAction.STRAIGHT,) * cruise_steps

    turn_action = {
        "left": TraceLatticeAction.LEFT,
        "right": TraceLatticeAction.RIGHT,
        "climb": TraceLatticeAction.PULL,
    }[role_key]
    lead = () if high_difficulty else (TraceLatticeAction.STRAIGHT,)
    settle_steps = max(1, cruise_steps - len(lead) - 1)
    return lead + (turn_action,) + ((TraceLatticeAction.STRAIGHT,) * settle_steps)


def _normalize_allowed_question_kinds(
    question_kinds: tuple[TraceTest2QuestionKind, ...] | None,
) -> tuple[TraceTest2QuestionKind, ...]:
    if question_kinds is None:
        return TraceTest2Generator._QUESTION_KINDS
    normalized: list[TraceTest2QuestionKind] = []
    seen: set[TraceTest2QuestionKind] = set()
    for raw in question_kinds:
        try:
            kind = (
                raw
                if isinstance(raw, TraceTest2QuestionKind)
                else TraceTest2QuestionKind(str(raw))
            )
        except ValueError as exc:
            raise ValueError(f"Unknown Trace Test 2 question kind: {raw}") from exc
        if kind in seen:
            continue
        seen.add(kind)
        normalized.append(kind)
    if not normalized:
        raise ValueError("allowed_question_kinds must not be empty")
    return tuple(normalized)


def trace_test_2_track_position(
    *,
    track: TraceTest2AircraftTrack,
    progress: float,
) -> TraceTest2Point3:
    if track.lattice_path is not None:
        pose = trace_lattice_sample_path(
            track.lattice_path,
            progress=progress,
            turn_phase_ratio=_TT2_TURN_PHASE_RATIO,
        )
        return _tt2_world_point_from_lattice(
            pose.position,
            lateral_scale=float(track.lateral_scale),
            vertical_scale=float(track.vertical_scale),
        )
    if len(track.waypoints) == 1:
        return track.waypoints[0]
    if len(track.waypoints) == 2:
        return _point_lerp(track.waypoints[0], track.waypoints[1], _clamp(progress, 0.0, 1.0))
    if len(track.waypoints) == 3:
        return _quadratic_point(
            track.waypoints[0],
            track.waypoints[1],
            track.waypoints[2],
            _clamp(progress, 0.0, 1.0),
        )
    t = _clamp(progress, 0.0, 1.0)
    seg_lengths: list[float] = []
    total = 0.0
    for start, end in zip(track.waypoints, track.waypoints[1:], strict=False):
        seg = math.dist((start.x, start.y, start.z), (end.x, end.y, end.z))
        seg_lengths.append(seg)
        total += seg
    if total <= 1e-6:
        return track.waypoints[-1]
    remaining = total * t
    for idx, seg in enumerate(seg_lengths):
        if remaining <= seg or idx == len(seg_lengths) - 1:
            local = 0.0 if seg <= 1e-6 else remaining / seg
            return _point_lerp(track.waypoints[idx], track.waypoints[idx + 1], local)
        remaining -= seg
    return track.waypoints[-1]


def trace_test_2_track_tangent(
    *,
    track: TraceTest2AircraftTrack,
    progress: float,
) -> tuple[float, float, float]:
    if track.lattice_path is not None:
        pose = trace_lattice_sample_path(
            track.lattice_path,
            progress=progress,
            turn_phase_ratio=_TT2_TURN_PHASE_RATIO,
        )
        return (
            float(pose.forward[0] * _TT2_WORLD_COL_SPACING * float(track.lateral_scale)),
            float(pose.forward[1] * _TT2_WORLD_ROW_SPACING),
            float(pose.forward[2] * _TT2_WORLD_LEVEL_SPACING * float(track.vertical_scale)),
        )
    if len(track.waypoints) == 1:
        return (0.0, 0.0, 0.0)
    if len(track.waypoints) == 2:
        start, end = track.waypoints
        return (
            float(end.x - start.x),
            float(end.y - start.y),
            float(end.z - start.z),
        )
    if len(track.waypoints) == 3:
        start, middle, end = track.waypoints
        u = _clamp(progress, 0.0, 1.0)
        return (
            float((2.0 * (1.0 - u) * (middle.x - start.x)) + (2.0 * u * (end.x - middle.x))),
            float((2.0 * (1.0 - u) * (middle.y - start.y)) + (2.0 * u * (end.y - middle.y))),
            float((2.0 * (1.0 - u) * (middle.z - start.z)) + (2.0 * u * (end.z - middle.z))),
        )
    step = 0.03
    pos = trace_test_2_track_position(track=track, progress=max(0.0, float(progress) - step))
    future = trace_test_2_track_position(track=track, progress=min(1.0, float(progress) + step))
    return (
        float(future.x - pos.x),
        float(future.y - pos.y),
        float(future.z - pos.z),
    )


class TraceTest2Generator:
    """Deterministic guide-style movement-memory clips."""

    _COLOR_SPECS: tuple[tuple[str, tuple[int, int, int]], ...] = (
        ("Red", (228, 54, 56)),
        ("Blue", (76, 136, 244)),
        ("Silver", (202, 208, 222)),
        ("Yellow", (236, 210, 92)),
    )
    _ROLE_KEYS: tuple[str, ...] = ("steady", "left", "right", "climb")
    _QUESTION_KINDS: tuple[TraceTest2QuestionKind, ...] = (
        TraceTest2QuestionKind.NO_DIRECTION_CHANGE,
        TraceTest2QuestionKind.TURNED_LEFT,
        TraceTest2QuestionKind.TURNED_RIGHT,
        TraceTest2QuestionKind.ENDED_LEFTMOST,
        TraceTest2QuestionKind.ENDED_HIGHEST,
    )

    def __init__(
        self,
        *,
        seed: int,
        allowed_question_kinds: tuple[TraceTest2QuestionKind, ...] | None = None,
    ) -> None:
        self._rng = SeededRng(seed)
        self._allowed_question_kinds = _normalize_allowed_question_kinds(
            allowed_question_kinds
        )

    def next_problem(self, *, difficulty: float) -> Problem:
        d = clamp01(difficulty)
        question_kind = self._rng.choice(self._allowed_question_kinds)
        role_order = self._rng.sample(self._ROLE_KEYS, k=len(self._ROLE_KEYS))
        tracks = tuple(
            self._build_track(
                code=idx + 1,
                color_name=color_name,
                color_rgb=color_rgb,
                role_key=role_order[idx],
                difficulty=d,
            )
            for idx, (color_name, color_rgb) in enumerate(self._COLOR_SPECS)
        )
        options, correct_code = self._options_and_answer(
            question_kind=question_kind,
            tracks=tracks,
        )
        stem = self._stem_for(question_kind)
        payload = TraceTest2Payload(
            trial_stage=TraceTest2TrialStage.QUESTION,
            stage_time_remaining_s=None,
            observe_progress=1.0,
            block_kind="scored",
            trial_index_in_block=1,
            trials_in_block=1,
            question_kind=question_kind,
            stem=stem,
            viewpoint_bearing_deg=0,
            aircraft=tracks,
            options=options,
            correct_code=correct_code,
            content_family="motion_memory",
            variant_id=stable_variant_id(question_kind.value, role_order),
            content_pack="trace_tt2",
        )
        prompt_lines = [stem, ""]
        prompt_lines.extend(f"{option.code}) {option.label}" for option in options)
        return Problem(
            prompt="\n".join(prompt_lines),
            answer=int(correct_code),
            payload=payload,
        )

    def _build_track(
        self,
        *,
        code: int,
        color_name: str,
        color_rgb: tuple[int, int, int],
        role_key: str,
        difficulty: float,
    ) -> TraceTest2AircraftTrack:
        d = clamp01(difficulty)
        lateral_scale = 1.02 + (0.10 * d)
        altitude_scale = 1.00 + (0.08 * d)
        templates: dict[str, tuple[TraceTest2MotionKind, TraceLatticeState]] = {
            "steady": (
                TraceTest2MotionKind.STRAIGHT,
                trace_lattice_state(col=1, row=1, level=2, forward=(0, 1, 0), up=(0, 0, 1)),
            ),
            "left": (
                TraceTest2MotionKind.LEFT,
                trace_lattice_state(col=5, row=1, level=3, forward=(0, 1, 0), up=(0, 0, 1)),
            ),
            "right": (
                TraceTest2MotionKind.RIGHT,
                trace_lattice_state(col=1, row=2, level=1, forward=(0, 1, 0), up=(0, 0, 1)),
            ),
            "climb": (
                TraceTest2MotionKind.CLIMB,
                trace_lattice_state(col=4, row=1, level=1, forward=(0, 1, 0), up=(0, 0, 1)),
            ),
        }
        declared_motion_kind, start_state = templates[role_key]
        actions = _tt2_actions_for_role(role_key=role_key, difficulty=d)
        path = trace_lattice_build_path(
            start_state=start_state,
            actions=actions,
            spec=DEFAULT_TRACE_LATTICE_SPEC,
        )
        waypoints = tuple(
            _tt2_world_point_from_lattice(
                trace_lattice_node_point(node, spec=DEFAULT_TRACE_LATTICE_SPEC),
                lateral_scale=lateral_scale,
                vertical_scale=altitude_scale,
            )
            for node in (
                path.start_state.node,
                *(step.end_state.node for step in path.steps),
            )
        )
        motion_kind = _tt2_motion_kind_for_path(path)
        if declared_motion_kind in {TraceTest2MotionKind.LEFT, TraceTest2MotionKind.RIGHT}:
            motion_kind = declared_motion_kind
        ended_screen_x, _ = _screen_metric(waypoints[-1])
        return TraceTest2AircraftTrack(
            code=code,
            color_name=color_name,
            color_rgb=color_rgb,
            waypoints=waypoints,
            motion_kind=motion_kind,
            direction_changed=_tt2_direction_changed_for_path(path),
            ended_screen_x=float(ended_screen_x),
            ended_altitude_z=float(waypoints[-1].z),
            lattice_path=path,
            lateral_scale=float(lateral_scale),
            vertical_scale=float(altitude_scale),
        )

    @staticmethod
    def _stem_for(question_kind: TraceTest2QuestionKind) -> str:
        return {
            TraceTest2QuestionKind.NO_DIRECTION_CHANGE: "Which aircraft did not change direction?",
            TraceTest2QuestionKind.TURNED_LEFT: "Which aircraft turned left?",
            TraceTest2QuestionKind.TURNED_RIGHT: "Which aircraft turned right?",
            TraceTest2QuestionKind.ENDED_LEFTMOST: "Which aircraft ended furthest left?",
            TraceTest2QuestionKind.ENDED_RIGHTMOST: "Which aircraft ended furthest right?",
            TraceTest2QuestionKind.ENDED_HIGHEST: "Which aircraft ended highest?",
            TraceTest2QuestionKind.ENDED_LOWEST: "Which aircraft ended lowest?",
        }[question_kind]

    def _options_and_answer(
        self,
        *,
        question_kind: TraceTest2QuestionKind,
        tracks: tuple[TraceTest2AircraftTrack, ...],
    ) -> tuple[tuple[TraceTest2Option, ...], int]:
        options = tuple(
            TraceTest2Option(
                code=track.code,
                label=track.color_name,
                color_name=track.color_name,
                color_rgb=track.color_rgb,
            )
            for track in tracks
        )
        if question_kind is TraceTest2QuestionKind.NO_DIRECTION_CHANGE:
            answer = next(track.code for track in tracks if not track.direction_changed)
            return options, int(answer)
        if question_kind is TraceTest2QuestionKind.TURNED_LEFT:
            answer = next(track.code for track in tracks if track.motion_kind is TraceTest2MotionKind.LEFT)
            return options, int(answer)
        if question_kind is TraceTest2QuestionKind.TURNED_RIGHT:
            answer = next(track.code for track in tracks if track.motion_kind is TraceTest2MotionKind.RIGHT)
            return options, int(answer)
        if question_kind is TraceTest2QuestionKind.ENDED_LEFTMOST:
            answer = min(tracks, key=lambda track: track.ended_screen_x).code
            return options, int(answer)
        if question_kind is TraceTest2QuestionKind.ENDED_RIGHTMOST:
            answer = max(tracks, key=lambda track: track.ended_screen_x).code
            return options, int(answer)
        if question_kind is TraceTest2QuestionKind.ENDED_HIGHEST:
            answer = max(tracks, key=lambda track: track.ended_altitude_z).code
            return options, int(answer)
        answer = min(tracks, key=lambda track: track.ended_altitude_z).code
        return options, int(answer)


class TraceTest2Engine:
    def __init__(
        self,
        *,
        clock: Clock,
        seed: int,
        difficulty: float = 0.5,
        config: TraceTest2Config | None = None,
    ) -> None:
        cfg = config or TraceTest2Config()
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
        allowed_question_kinds = _normalize_allowed_question_kinds(
            self._cfg.allowed_question_kinds
        )
        self._generator = TraceTest2Generator(
            seed=self._seed,
            allowed_question_kinds=allowed_question_kinds,
        )

        self._phase = Phase.INSTRUCTIONS
        self._current: Problem | None = None
        self._current_payload: TraceTest2Payload | None = None
        self._trial_started_at_s: float | None = None
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
        self._ensure_question_open_state()

    def time_remaining_s(self) -> float | None:
        if self._phase is not Phase.SCORED or self._scored_started_at_s is None:
            return None
        remaining = self._cfg.scored_duration_s - (self._clock.now() - self._scored_started_at_s)
        return max(0.0, remaining)

    def current_prompt(self) -> str:
        if self._phase is Phase.INSTRUCTIONS:
            return "Watch the aircraft clip first. When it ends, answer with A/S/D/F."
        if self._phase is Phase.PRACTICE_DONE:
            return "Practice complete. Press Enter to start the timed test."
        if self._phase is Phase.RESULTS:
            summary = self.scored_summary()
            acc_pct = int(round(summary.accuracy * 100))
            rt = (
                "n/a"
                if summary.mean_response_time_s is None
                else f"{summary.mean_response_time_s:.2f}s"
            )
            return (
                f"Results\nAttempted: {summary.attempted}\nCorrect: {summary.correct}\n"
                f"Accuracy: {acc_pct}%\nMean RT: {rt}\n"
                f"Throughput: {summary.throughput_per_min:.1f}/min"
            )
        payload = self._snapshot_payload()
        if payload is None or self._current is None:
            return ""
        if payload.trial_stage is TraceTest2TrialStage.OBSERVE:
            return "Watch the aircraft scene."
        return self._current.prompt

    def submit_answer(self, raw: object) -> bool:
        if self._phase not in (Phase.PRACTICE, Phase.SCORED):
            return False
        payload = self._snapshot_payload()
        if payload is None or payload.trial_stage is not TraceTest2TrialStage.QUESTION:
            return False
        raw_in = raw if isinstance(raw, str) else str(raw)
        value = raw_in.strip()
        if value == "":
            return False
        try:
            user_answer = int(value)
        except ValueError:
            return False
        self._ensure_question_open_state()
        assert self._current is not None
        assert self._question_started_at_s is not None

        answered_at_s = self._clock.now()
        response_time_s = max(0.0, answered_at_s - self._question_started_at_s)
        is_correct = int(user_answer) == int(self._current.answer)
        event = QuestionEvent(
            index=len(self._events),
            phase=self._phase,
            prompt=self._current.prompt,
            correct_answer=int(self._current.answer),
            user_answer=int(user_answer),
            is_correct=bool(is_correct),
            presented_at_s=float(self._question_started_at_s),
            answered_at_s=float(answered_at_s),
            response_time_s=float(response_time_s),
            raw=raw_in,
            score=1.0 if is_correct else 0.0,
            max_score=1.0,
            content_metadata=content_metadata_from_payload(payload),
        )
        self._events.append(event)

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
            title="Trace Test 2",
            phase=self._phase,
            prompt=self.current_prompt(),
            input_hint="Watch first. Then answer with A/S/D/F, or use 1-4 and Enter.",
            time_remaining_s=self.time_remaining_s(),
            attempted_scored=self._scored_attempted,
            correct_scored=self._scored_correct,
            payload=self._snapshot_payload(),
        )

    def _observe_duration_s(self) -> float:
        return (
            float(self._cfg.practice_observe_s)
            if self._phase is Phase.PRACTICE
            else float(self._cfg.scored_observe_s)
        )

    def _elapsed_in_trial_s(self) -> float | None:
        if self._trial_started_at_s is None:
            return None
        return max(0.0, self._clock.now() - self._trial_started_at_s)

    def _ensure_question_open_state(self) -> None:
        if self._question_started_at_s is not None or self._trial_started_at_s is None:
            return
        elapsed = self._elapsed_in_trial_s()
        if elapsed is None:
            return
        observe_duration = self._observe_duration_s()
        if elapsed + _STAGE_EPSILON_S < observe_duration:
            return
        self._question_started_at_s = self._trial_started_at_s + observe_duration

    def _snapshot_payload(self) -> TraceTest2Payload | None:
        payload = self._current_payload
        if payload is None:
            return None
        duration = max(0.001, self._observe_duration_s())
        elapsed = self._elapsed_in_trial_s()
        if elapsed is None:
            elapsed = 0.0
        if elapsed + _STAGE_EPSILON_S < duration:
            stage = TraceTest2TrialStage.OBSERVE
            stage_time_remaining_s = max(0.0, duration - elapsed)
            progress = _clamp(elapsed / duration, 0.0, 1.0)
        else:
            stage = TraceTest2TrialStage.QUESTION
            stage_time_remaining_s = None
            progress = 1.0
        return replace(
            payload,
            trial_stage=stage,
            stage_time_remaining_s=stage_time_remaining_s,
            observe_progress=float(progress),
        )

    def _deal_new_problem(self) -> None:
        self._current = self._generator.next_problem(difficulty=self._difficulty)
        base_payload = self._current.payload
        assert isinstance(base_payload, TraceTest2Payload)
        block_kind = "practice" if self._phase is Phase.PRACTICE else "scored"
        trial_index = (
            self._practice_answered + 1
            if self._phase is Phase.PRACTICE
            else self._scored_attempted + 1
        )
        trial_total = (
            int(self._cfg.practice_questions)
            if self._phase is Phase.PRACTICE
            else max(1, int(round(self._cfg.scored_duration_s / max(1.0, self._cfg.scored_observe_s + 2.0))))
        )
        now_s = self._clock.now()
        self._current_payload = replace(
            base_payload,
            trial_stage=TraceTest2TrialStage.OBSERVE,
            stage_time_remaining_s=float(self._observe_duration_s()),
            observe_progress=0.0,
            block_kind=block_kind,
            trial_index_in_block=int(trial_index),
            trials_in_block=int(trial_total),
        )
        self._trial_started_at_s = now_s
        self._question_started_at_s = None

    def _clear_current(self) -> None:
        self._current = None
        self._current_payload = None
        self._trial_started_at_s = None
        self._question_started_at_s = None

    def _finish(self) -> None:
        self._phase = Phase.RESULTS
        self._clear_current()


def build_trace_test_2_test(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    config: TraceTest2Config | None = None,
) -> TraceTest2Engine:
    return TraceTest2Engine(
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        config=config,
    )
