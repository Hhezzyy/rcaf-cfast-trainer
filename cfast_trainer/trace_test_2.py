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
)

_STAGE_EPSILON_S = 1e-6


@dataclass(frozen=True, slots=True)
class TraceTest2Config:
    # Candidate guide indicates ~9 minutes total including instructions.
    scored_duration_s: float = 7.5 * 60.0
    practice_questions: int = 2
    practice_observe_s: float = 5.5
    scored_observe_s: float = 4.8


class TraceTest2TrialStage(StrEnum):
    OBSERVE = "observe"
    QUESTION = "question"


class TraceTest2QuestionKind(StrEnum):
    STILL_VISIBLE_AT_END = "still_visible_at_end"
    ENDED_MOST_LEFT = "ended_most_left"
    STARTED_LOWEST = "started_lowest"
    LEAST_TIME_ON_SCREEN = "least_time_on_screen"
    RED_LEFT_TURNS = "red_left_turns"


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
    left_turns: int
    visible_fraction: float
    visible_at_end: bool
    started_screen_y: float
    ended_screen_x: float


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


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(lo if v < lo else hi if v > hi else v)


def _smoothstep(t: float) -> float:
    x = _clamp(float(t), 0.0, 1.0)
    return float(x * x * (3.0 - (2.0 * x)))


def _lerp(a: float, b: float, t: float) -> float:
    return float(a + ((b - a) * t))


def _point_lerp(a: TraceTest2Point3, b: TraceTest2Point3, t: float) -> TraceTest2Point3:
    return TraceTest2Point3(
        x=_lerp(a.x, b.x, t),
        y=_lerp(a.y, b.y, t),
        z=_lerp(a.z, b.z, t),
    )


def _turn_heading_deg(a: TraceTest2Point3, b: TraceTest2Point3) -> float:
    dx = float(b.x - a.x)
    dy = float(b.y - a.y)
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return 0.0
    return float(math.degrees(math.atan2(dx, dy)) % 360.0)


def _screen_metric(point: TraceTest2Point3) -> tuple[float, float]:
    screen_x = float(point.x)
    screen_y = float(((116.0 - point.y) * 1.18) - ((point.z - 8.0) * 2.05))
    return screen_x, screen_y


def _point_visible(point: TraceTest2Point3) -> bool:
    screen_x, screen_y = _screen_metric(point)
    return -28.0 <= screen_x <= 28.0 and -28.0 <= screen_y <= 28.0


def _count_left_turns(waypoints: tuple[TraceTest2Point3, ...]) -> int:
    headings: list[float] = []
    for start, end in zip(waypoints, waypoints[1:], strict=False):
        heading = _turn_heading_deg(start, end)
        if math.dist((start.x, start.y), (end.x, end.y)) > 1e-6:
            headings.append(heading)
    turns = 0
    for prev, nxt in zip(headings, headings[1:], strict=False):
        delta = ((nxt - prev) + 540.0) % 360.0 - 180.0
        if delta < -35.0:
            turns += 1
    return int(turns)


def trace_test_2_track_position(
    *,
    track: TraceTest2AircraftTrack,
    progress: float,
) -> TraceTest2Point3:
    if len(track.waypoints) == 1:
        return track.waypoints[0]
    t = _smoothstep(progress)
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


def _trace_test_2_track_stats(
    waypoints: tuple[TraceTest2Point3, ...],
) -> tuple[int, float, bool, float, float]:
    visible_samples = 0
    total_samples = 33
    for sample_idx in range(total_samples):
        progress = sample_idx / max(1, total_samples - 1)
        point = trace_test_2_track_position(
            track=TraceTest2AircraftTrack(
                code=0,
                color_name="",
                color_rgb=(0, 0, 0),
                waypoints=waypoints,
                left_turns=0,
                visible_fraction=0.0,
                visible_at_end=False,
                started_screen_y=0.0,
                ended_screen_x=0.0,
            ),
            progress=progress,
        )
        if _point_visible(point):
            visible_samples += 1
    left_turns = _count_left_turns(waypoints)
    visible_fraction = visible_samples / total_samples
    visible_at_end = _point_visible(waypoints[-1])
    _, started_screen_y = _screen_metric(waypoints[0])
    ended_screen_x, _ = _screen_metric(waypoints[-1])
    return (
        left_turns,
        float(visible_fraction),
        bool(visible_at_end),
        float(started_screen_y),
        float(ended_screen_x),
    )


class TraceTest2Generator:
    """Deterministic scene-memory clips built from 3-D aircraft paths."""

    _COLOR_SPECS: tuple[tuple[str, tuple[int, int, int]], ...] = (
        ("Red", (228, 54, 56)),
        ("Blue", (76, 136, 244)),
        ("Silver", (202, 208, 222)),
        ("Yellow", (236, 210, 92)),
    )
    _QUESTION_KINDS: tuple[TraceTest2QuestionKind, ...] = (
        TraceTest2QuestionKind.STILL_VISIBLE_AT_END,
        TraceTest2QuestionKind.ENDED_MOST_LEFT,
        TraceTest2QuestionKind.STARTED_LOWEST,
        TraceTest2QuestionKind.LEAST_TIME_ON_SCREEN,
        TraceTest2QuestionKind.RED_LEFT_TURNS,
    )

    def __init__(self, *, seed: int) -> None:
        self._rng = SeededRng(seed)

    def next_problem(self, *, difficulty: float) -> Problem:
        d = clamp01(difficulty)
        question_kind = self._rng.choice(self._QUESTION_KINDS)
        role_order = self._rng.sample(("alpha", "bravo", "charlie", "delta"), k=4)
        variant = int(self._rng.randint(0, 2))

        aircraft = tuple(
            self._build_track(
                code=idx + 1,
                color_name=color_name,
                color_rgb=color_rgb,
                role_key=role_order[idx],
                difficulty=d,
                variant=variant,
            )
            for idx, (color_name, color_rgb) in enumerate(self._COLOR_SPECS)
        )

        options, correct_code = self._options_and_answer(
            question_kind=question_kind,
            aircraft=aircraft,
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
            aircraft=aircraft,
            options=options,
            correct_code=correct_code,
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
        variant: int,
    ) -> TraceTest2AircraftTrack:
        lateral_bias = (-1.4, 0.0, 1.4)[variant]
        altitude_bias = (-0.7, 0.0, 0.9)[variant]
        speed_bias = 2.0 + (difficulty * 3.0)

        role_waypoints = {
            "alpha": (
                TraceTest2Point3(-34.0 + lateral_bias, 84.0, 14.0 + altitude_bias),
                TraceTest2Point3(-10.0 + lateral_bias, 84.0, 14.0 + altitude_bias),
                TraceTest2Point3(-10.0 + lateral_bias, 102.0 + speed_bias, 14.0 + altitude_bias),
                TraceTest2Point3(-32.0 + lateral_bias, 102.0 + speed_bias, 13.0 + altitude_bias),
            ),
            "bravo": (
                TraceTest2Point3(-10.0 + lateral_bias, 58.0, 1.5 + altitude_bias),
                TraceTest2Point3(12.0 + lateral_bias, 58.0, 1.5 + altitude_bias),
                TraceTest2Point3(34.0 + lateral_bias, 58.0, 4.0 + altitude_bias),
            ),
            "charlie": (
                TraceTest2Point3(-38.0 + lateral_bias, 98.0, 20.0 + altitude_bias),
                TraceTest2Point3(-18.0 + lateral_bias, 98.0, 20.0 + altitude_bias),
                TraceTest2Point3(-18.0 + lateral_bias, 114.0 + (speed_bias * 0.7), 18.0 + altitude_bias),
            ),
            "delta": (
                TraceTest2Point3(32.0 + lateral_bias, 92.0, 10.0 + altitude_bias),
                TraceTest2Point3(14.0 + lateral_bias, 92.0, 10.0 + altitude_bias),
                TraceTest2Point3(34.0 + lateral_bias, 106.0 + (speed_bias * 0.35), 12.0 + altitude_bias),
            ),
        }
        waypoints = role_waypoints[role_key]
        left_turns, visible_fraction, visible_at_end, started_screen_y, ended_screen_x = (
            _trace_test_2_track_stats(waypoints)
        )
        return TraceTest2AircraftTrack(
            code=code,
            color_name=color_name,
            color_rgb=color_rgb,
            waypoints=waypoints,
            left_turns=left_turns,
            visible_fraction=visible_fraction,
            visible_at_end=visible_at_end,
            started_screen_y=started_screen_y,
            ended_screen_x=ended_screen_x,
        )

    @staticmethod
    def _stem_for(question_kind: TraceTest2QuestionKind) -> str:
        if question_kind is TraceTest2QuestionKind.STILL_VISIBLE_AT_END:
            return "Which aircraft was still on screen at the end?"
        if question_kind is TraceTest2QuestionKind.ENDED_MOST_LEFT:
            return "Which aircraft ended furthest left?"
        if question_kind is TraceTest2QuestionKind.STARTED_LOWEST:
            return "Which aircraft started lowest on the screen?"
        if question_kind is TraceTest2QuestionKind.LEAST_TIME_ON_SCREEN:
            return "Which aircraft spent the least time on screen?"
        return "How many left turns did the red aircraft make?"

    def _options_and_answer(
        self,
        *,
        question_kind: TraceTest2QuestionKind,
        aircraft: tuple[TraceTest2AircraftTrack, ...],
    ) -> tuple[tuple[TraceTest2Option, ...], int]:
        if question_kind is TraceTest2QuestionKind.RED_LEFT_TURNS:
            red = next(track for track in aircraft if track.color_name == "Red")
            options = (
                TraceTest2Option(code=1, label="0"),
                TraceTest2Option(code=2, label="1"),
                TraceTest2Option(code=3, label="2"),
                TraceTest2Option(code=4, label="3"),
            )
            return options, int(_clamp(red.left_turns, 0, 3) + 1)

        options = tuple(
            TraceTest2Option(
                code=track.code,
                label=track.color_name,
                color_name=track.color_name,
                color_rgb=track.color_rgb,
            )
            for track in aircraft
        )
        if question_kind is TraceTest2QuestionKind.STILL_VISIBLE_AT_END:
            answer = next(track.code for track in aircraft if track.visible_at_end)
            return options, int(answer)
        if question_kind is TraceTest2QuestionKind.ENDED_MOST_LEFT:
            answer = min(aircraft, key=lambda track: track.ended_screen_x).code
            return options, int(answer)
        if question_kind is TraceTest2QuestionKind.STARTED_LOWEST:
            answer = max(aircraft, key=lambda track: track.started_screen_y).code
            return options, int(answer)
        answer = min(aircraft, key=lambda track: track.visible_fraction).code
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
        self._generator = TraceTest2Generator(seed=self._seed)

        self._phase = Phase.INSTRUCTIONS
        self._current: Problem | None = None
        self._current_payload: TraceTest2Payload | None = None
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
        # The movement animation may finish (observe_progress reaches 1.0),
        # but the trial no longer auto-advances on a stage transition.
        _ = self._observe_time_remaining_s()

    def time_remaining_s(self) -> float | None:
        if self._phase is not Phase.SCORED or self._scored_started_at_s is None:
            return None
        remaining = self._cfg.scored_duration_s - (self._clock.now() - self._scored_started_at_s)
        return max(0.0, remaining)

    def current_prompt(self) -> str:
        if self._phase is Phase.INSTRUCTIONS:
            return "Watch the aircraft movement and answer the recall question with A/S/D/F."
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
        if self._current_payload is None or self._current is None:
            return ""
        return self._current.prompt

    def submit_answer(self, raw: object) -> bool:
        if self._phase not in (Phase.PRACTICE, Phase.SCORED):
            return False
        payload = self._current_payload
        if payload is None:
            return False
        raw_in = raw if isinstance(raw, str) else str(raw)
        value = raw_in.strip()
        if value == "":
            return False
        try:
            user_answer = int(value)
        except ValueError:
            return False

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
            title="Trace Test 2",
            phase=self._phase,
            prompt=self.current_prompt(),
            input_hint="Answer with A/S/D/F or Up/Down and Enter.",
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

    def _snapshot_payload(self) -> TraceTest2Payload | None:
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
            trial_stage=TraceTest2TrialStage.QUESTION,
            stage_time_remaining_s=float(remaining),
            observe_progress=float(progress),
        )

    def _deal_new_problem(self) -> None:
        self._current = self._generator.next_problem(difficulty=self._difficulty)
        base_payload = self._current.payload
        assert isinstance(base_payload, TraceTest2Payload)
        block_kind = "practice" if self._phase is Phase.PRACTICE else "scored"
        trial_index = (
            self._practice_answered + 1 if self._phase is Phase.PRACTICE else self._scored_attempted + 1
        )
        trial_total = (
            int(self._cfg.practice_questions)
            if self._phase is Phase.PRACTICE
            else max(1, int(round(self._cfg.scored_duration_s / max(1.0, self._cfg.scored_observe_s + 2.0))))
        )
        now_s = self._clock.now()
        self._current_payload = replace(
            base_payload,
            trial_stage=TraceTest2TrialStage.QUESTION,
            stage_time_remaining_s=float(self._observe_duration_s()),
            observe_progress=0.0,
            block_kind=block_kind,
            trial_index_in_block=int(trial_index),
            trials_in_block=int(trial_total),
        )
        self._observe_started_at_s = now_s
        self._question_started_at_s = now_s

    def _clear_current(self) -> None:
        self._current = None
        self._current_payload = None
        self._observe_started_at_s = None
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
