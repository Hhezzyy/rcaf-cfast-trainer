from __future__ import annotations

import os
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import cast

from .clock import Clock
from .cognitive_core import (
    AnswerScorer,
    AttemptSummary,
    Phase,
    Problem,
    QuestionEvent,
    SeededRng,
    TestSnapshot,
    clamp01,
    lerp_int,
)


@dataclass(frozen=True, slots=True)
class SpatialIntegrationConfig:
    # Guide indicates ~28 minutes total including instructions.
    # We model this as three scored sections with a short practice before each.
    practice_questions: int = 2
    scored_questions_per_section: int = 8
    practice_memorize_s: float = 5.5
    scored_memorize_s: float = 4.5
    skip_practice_for_testing: bool = False
    start_section: str = "A"  # "A" | "B" | "C"


class SpatialIntegrationSection(StrEnum):
    PART_A = "A"
    PART_B = "B"
    PART_C = "C"


class SpatialIntegrationSceneView(StrEnum):
    TOPDOWN = "topdown"
    OBLIQUE = "oblique"


class SpatialIntegrationTrialStage(StrEnum):
    MEMORIZE = "memorize"
    QUESTION = "question"


class SpatialIntegrationQuestionKind(StrEnum):
    LANDMARK_LOCATION = "landmark_location"
    AIRCRAFT_EXTRAPOLATION = "aircraft_extrapolation"


@dataclass(frozen=True, slots=True)
class SpatialIntegrationPoint:
    x: int
    y: int
    z: int


@dataclass(frozen=True, slots=True)
class SpatialIntegrationVector:
    dx: int
    dy: int
    dz: int


@dataclass(frozen=True, slots=True)
class SpatialIntegrationLandmark:
    label: str
    x: int
    y: int


@dataclass(frozen=True, slots=True)
class SpatialIntegrationOption:
    code: int
    label: str
    point: SpatialIntegrationPoint
    error: int


@dataclass(frozen=True, slots=True)
class SpatialIntegrationPayload:
    section: SpatialIntegrationSection
    trial_stage: SpatialIntegrationTrialStage
    scene_view: SpatialIntegrationSceneView
    stage_time_remaining_s: float | None
    block_kind: str  # "practice" | "scored"
    trial_index_in_block: int
    trials_in_block: int
    kind: SpatialIntegrationQuestionKind
    stem: str
    query_label: str
    grid_cols: int
    grid_rows: int
    alt_levels: int
    landmarks: tuple[SpatialIntegrationLandmark, ...]
    aircraft_prev: SpatialIntegrationPoint
    aircraft_now: SpatialIntegrationPoint
    velocity: SpatialIntegrationVector
    show_aircraft_motion: bool
    options: tuple[SpatialIntegrationOption, ...]
    correct_code: int
    correct_point: SpatialIntegrationPoint
    full_credit_error: int
    zero_credit_error: int


def _clamp(v: int, lo: int, hi: int) -> int:
    return int(lo if v < lo else hi if v > hi else v)


def _point_error(a: SpatialIntegrationPoint, b: SpatialIntegrationPoint) -> int:
    return int(abs(a.x - b.x) + abs(a.y - b.y) + abs(a.z - b.z))


def _cell_label(point: SpatialIntegrationPoint) -> str:
    col = chr(ord("A") + int(point.x))
    row = int(point.y) + 1
    level = int(point.z) + 1
    return f"{col}{row}-L{level}"


def _section_name(section: SpatialIntegrationSection) -> str:
    if section is SpatialIntegrationSection.PART_A:
        return "Section A"
    if section is SpatialIntegrationSection.PART_B:
        return "Section B"
    return "Section C"


class SpatialIntegrationScorer(AnswerScorer):
    """Exact cell gets full credit; near 3-D picks receive partial credit."""

    def score(self, *, problem: Problem, user_answer: int, raw: str) -> float:
        _ = raw
        payload = problem.payload
        if not isinstance(payload, SpatialIntegrationPayload):
            return 1.0 if int(user_answer) == int(problem.answer) else 0.0

        by_code = {opt.code: opt for opt in payload.options}
        selected = by_code.get(int(user_answer))
        if selected is None:
            return 0.0

        err = max(0, int(selected.error))
        full = max(0, int(payload.full_credit_error))
        zero = max(full + 1, int(payload.zero_credit_error))
        if err <= full:
            return 1.0
        if err >= zero:
            return 0.0
        return clamp01((zero - err) / float(zero - full))


class SpatialIntegrationGenerator:
    """Deterministic scene/question generator for sections A/B/C."""

    _LANDMARK_LABELS = ("TWR", "HGR", "WDM", "RDG", "LKE", "VLG")

    def __init__(self, *, seed: int) -> None:
        self._rng = SeededRng(seed)

    def next_problem(
        self,
        *,
        section: SpatialIntegrationSection,
        difficulty: float,
    ) -> Problem:
        d = clamp01(difficulty)
        grid_cols = lerp_int(4, 6, d)
        grid_rows = lerp_int(4, 6, d)
        alt_levels = lerp_int(3, 5, d)
        landmarks = self._sample_landmarks(grid_cols=grid_cols, grid_rows=grid_rows, count=4)

        if section is SpatialIntegrationSection.PART_C:
            kind = SpatialIntegrationQuestionKind.AIRCRAFT_EXTRAPOLATION
            scene_view = SpatialIntegrationSceneView.OBLIQUE
            prev_point, now_point, velocity = self._sample_motion_state(
                grid_cols=grid_cols,
                grid_rows=grid_rows,
                alt_levels=alt_levels,
                difficulty=d,
            )
            correct_point = SpatialIntegrationPoint(
                x=now_point.x + velocity.dx,
                y=now_point.y + velocity.dy,
                z=now_point.z + velocity.dz,
            )
            query_label = "AIRCRAFT"
            stem = "Where will the aircraft be after one more equal movement step?"
        else:
            kind = SpatialIntegrationQuestionKind.LANDMARK_LOCATION
            scene_view = (
                SpatialIntegrationSceneView.TOPDOWN
                if section is SpatialIntegrationSection.PART_A
                else SpatialIntegrationSceneView.OBLIQUE
            )
            target = cast(SpatialIntegrationLandmark, self._rng.choice(landmarks))
            query_label = str(target.label)
            stem = f"Where was {target.label} located in the memorized scene?"
            correct_point = SpatialIntegrationPoint(
                x=int(target.x),
                y=int(target.y),
                z=0,
            )
            now_point = self._sample_point(
                grid_cols=grid_cols,
                grid_rows=grid_rows,
                alt_levels=alt_levels,
            )
            prev_point = now_point
            velocity = SpatialIntegrationVector(dx=0, dy=0, dz=0)

        option_points = self._build_option_points(
            kind=kind,
            correct=correct_point,
            now=now_point,
            velocity=velocity,
            grid_cols=grid_cols,
            grid_rows=grid_rows,
            alt_levels=alt_levels,
        )

        order = self._rng.sample((0, 1, 2, 3, 4), k=5)
        options: list[SpatialIntegrationOption] = []
        correct_code = 1
        for code, idx in enumerate(order, start=1):
            point = option_points[idx]
            option = SpatialIntegrationOption(
                code=code,
                label=_cell_label(point),
                point=point,
                error=_point_error(correct_point, point),
            )
            options.append(option)
            if point == correct_point:
                correct_code = code

        payload = SpatialIntegrationPayload(
            section=section,
            trial_stage=SpatialIntegrationTrialStage.QUESTION,
            scene_view=scene_view,
            stage_time_remaining_s=None,
            block_kind="scored",
            trial_index_in_block=1,
            trials_in_block=1,
            kind=kind,
            stem=stem,
            query_label=query_label,
            grid_cols=grid_cols,
            grid_rows=grid_rows,
            alt_levels=alt_levels,
            landmarks=landmarks,
            aircraft_prev=prev_point,
            aircraft_now=now_point,
            velocity=velocity,
            show_aircraft_motion=(section is SpatialIntegrationSection.PART_C),
            options=tuple(options),
            correct_code=int(correct_code),
            correct_point=correct_point,
            full_credit_error=0,
            zero_credit_error=lerp_int(5, 3, d),
        )

        prompt_lines = [stem, ""]
        for option in options:
            prompt_lines.append(f"{option.code}) {option.label}")

        return Problem(
            prompt="\n".join(prompt_lines),
            answer=int(correct_code),
            payload=payload,
        )

    def _sample_landmarks(
        self,
        *,
        grid_cols: int,
        grid_rows: int,
        count: int,
    ) -> tuple[SpatialIntegrationLandmark, ...]:
        labels = self._rng.sample(self._LANDMARK_LABELS, k=min(count, len(self._LANDMARK_LABELS)))
        points: list[tuple[int, int]] = []
        while len(points) < len(labels):
            candidate = (
                int(self._rng.randint(0, grid_cols - 1)),
                int(self._rng.randint(0, grid_rows - 1)),
            )
            if candidate in points:
                continue
            points.append(candidate)
        return tuple(
            SpatialIntegrationLandmark(label=str(lbl), x=int(pt[0]), y=int(pt[1]))
            for lbl, pt in zip(labels, points, strict=True)
        )

    def _sample_point(
        self,
        *,
        grid_cols: int,
        grid_rows: int,
        alt_levels: int,
    ) -> SpatialIntegrationPoint:
        return SpatialIntegrationPoint(
            x=int(self._rng.randint(0, grid_cols - 1)),
            y=int(self._rng.randint(0, grid_rows - 1)),
            z=int(self._rng.randint(0, alt_levels - 1)),
        )

    def _sample_motion_state(
        self,
        *,
        grid_cols: int,
        grid_rows: int,
        alt_levels: int,
        difficulty: float,
    ) -> tuple[SpatialIntegrationPoint, SpatialIntegrationPoint, SpatialIntegrationVector]:
        max_lat = 1 if difficulty < 0.72 else 2
        max_vert = 1 if difficulty < 0.86 else 2
        while True:
            dx = int(self._rng.randint(-max_lat, max_lat))
            dy = int(self._rng.randint(-max_lat, max_lat))
            dz = int(self._rng.randint(-max_vert, max_vert))
            if dx == 0 and dy == 0 and dz == 0:
                continue

            valid_x = [
                x for x in range(grid_cols) if 0 <= x - dx < grid_cols and 0 <= x + dx < grid_cols
            ]
            valid_y = [
                y for y in range(grid_rows) if 0 <= y - dy < grid_rows and 0 <= y + dy < grid_rows
            ]
            valid_z = [
                z
                for z in range(alt_levels)
                if 0 <= z - dz < alt_levels and 0 <= z + dz < alt_levels
            ]
            if not valid_x or not valid_y or not valid_z:
                continue

            now = SpatialIntegrationPoint(
                x=int(self._rng.choice(valid_x)),
                y=int(self._rng.choice(valid_y)),
                z=int(self._rng.choice(valid_z)),
            )
            prev = SpatialIntegrationPoint(
                x=now.x - dx,
                y=now.y - dy,
                z=now.z - dz,
            )
            vec = SpatialIntegrationVector(dx=dx, dy=dy, dz=dz)
            return prev, now, vec

    def _build_option_points(
        self,
        *,
        kind: SpatialIntegrationQuestionKind,
        correct: SpatialIntegrationPoint,
        now: SpatialIntegrationPoint,
        velocity: SpatialIntegrationVector,
        grid_cols: int,
        grid_rows: int,
        alt_levels: int,
    ) -> tuple[
        SpatialIntegrationPoint,
        SpatialIntegrationPoint,
        SpatialIntegrationPoint,
        SpatialIntegrationPoint,
        SpatialIntegrationPoint,
    ]:
        points: list[SpatialIntegrationPoint] = [correct]

        def add(point: SpatialIntegrationPoint) -> None:
            if point in points:
                return
            points.append(point)

        if kind is SpatialIntegrationQuestionKind.AIRCRAFT_EXTRAPOLATION:
            add(
                SpatialIntegrationPoint(
                    x=_clamp(now.x + velocity.dx, 0, grid_cols - 1),
                    y=_clamp(now.y + velocity.dy, 0, grid_rows - 1),
                    z=now.z,
                )
            )
            add(
                SpatialIntegrationPoint(
                    x=now.x,
                    y=now.y,
                    z=_clamp(now.z + velocity.dz, 0, alt_levels - 1),
                )
            )
            add(
                SpatialIntegrationPoint(
                    x=_clamp(now.x - velocity.dx, 0, grid_cols - 1),
                    y=_clamp(now.y - velocity.dy, 0, grid_rows - 1),
                    z=_clamp(now.z + velocity.dz, 0, alt_levels - 1),
                )
            )
        else:
            add(
                SpatialIntegrationPoint(
                    x=correct.y % grid_cols, y=correct.x % grid_rows, z=correct.z
                )
            )
            add(
                SpatialIntegrationPoint(
                    x=correct.x,
                    y=correct.y,
                    z=_clamp(correct.z + 1, 0, alt_levels - 1),
                )
            )
            add(
                SpatialIntegrationPoint(
                    x=_clamp(correct.x + 1, 0, grid_cols - 1),
                    y=_clamp(correct.y - 1, 0, grid_rows - 1),
                    z=correct.z,
                )
            )

        while len(points) < 5:
            candidate = SpatialIntegrationPoint(
                x=_clamp(correct.x + int(self._rng.randint(-2, 2)), 0, grid_cols - 1),
                y=_clamp(correct.y + int(self._rng.randint(-2, 2)), 0, grid_rows - 1),
                z=_clamp(correct.z + int(self._rng.randint(-1, 1)), 0, alt_levels - 1),
            )
            if candidate == correct:
                continue
            add(candidate)

        return points[0], points[1], points[2], points[3], points[4]


class SpatialIntegrationEngine:
    """Three-section memory test: A(top-down), B(oblique), C(oblique+aircraft)."""

    _SECTION_ORDER = (
        SpatialIntegrationSection.PART_A,
        SpatialIntegrationSection.PART_B,
        SpatialIntegrationSection.PART_C,
    )

    def __init__(
        self,
        *,
        clock: Clock,
        seed: int,
        difficulty: float = 0.5,
        config: SpatialIntegrationConfig | None = None,
    ) -> None:
        cfg = config or SpatialIntegrationConfig()
        if cfg.practice_questions < 0:
            raise ValueError("practice_questions must be >= 0")
        if cfg.scored_questions_per_section < 0:
            raise ValueError("scored_questions_per_section must be >= 0")
        if cfg.practice_memorize_s < 0.1:
            raise ValueError("practice_memorize_s must be >= 0.1")
        if cfg.scored_memorize_s < 0.1:
            raise ValueError("scored_memorize_s must be >= 0.1")

        self._clock = clock
        self._seed = int(seed)
        self._difficulty = clamp01(difficulty)
        self._cfg = cfg

        start = str(cfg.start_section).strip().upper()
        self._start_section = (
            SpatialIntegrationSection.PART_B
            if start == "B"
            else SpatialIntegrationSection.PART_C
            if start == "C"
            else SpatialIntegrationSection.PART_A
        )
        self._section_idx = self._SECTION_ORDER.index(self._start_section)

        self._generator = SpatialIntegrationGenerator(seed=self._seed)
        self._scorer = SpatialIntegrationScorer()

        self._phase = Phase.INSTRUCTIONS
        self._last_update_at_s = self._clock.now()

        self._current_problem: Problem | None = None
        self._stage_ends_at_s: float | None = None
        self._question_presented_at_s: float | None = None

        self._block_is_practice = True
        self._block_trial_index = 0
        self._block_trial_target = 0

        self._pending_done_action: str | None = None  # "start_scored" | "start_next_practice"
        self._practice_done_prompt = "Practice complete."
        self._practice_feedback: str | None = None

        self._events: list[QuestionEvent] = []
        self._scored_attempted = 0
        self._scored_correct = 0
        self._scored_total_score = 0.0
        self._scored_max_score = 0.0
        self._scored_elapsed_s = 0.0

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def difficulty(self) -> float:
        return float(self._difficulty)

    def can_exit(self) -> bool:
        return self._phase in (
            Phase.INSTRUCTIONS,
            Phase.PRACTICE,
            Phase.PRACTICE_DONE,
            Phase.RESULTS,
        )

    def events(self) -> list[QuestionEvent]:
        return list(self._events)

    def start_practice(self) -> None:
        if self._phase is not Phase.INSTRUCTIONS:
            return
        self._section_idx = self._SECTION_ORDER.index(self._start_section)
        if self._cfg.skip_practice_for_testing:
            section = self._active_section()
            self._phase = Phase.PRACTICE_DONE
            self._pending_done_action = "start_scored"
            self._practice_done_prompt = (
                f"{_section_name(section)} practice skipped (testing mode). "
                "Press Enter to start scored."
            )
            self._current_problem = None
            return
        self._begin_block(is_practice=True)

    def start_scored(self) -> None:
        if self._phase is Phase.PRACTICE_DONE:
            if self._pending_done_action == "start_scored":
                self._begin_block(is_practice=False)
                return
            if self._pending_done_action == "start_next_practice":
                self._section_idx += 1
                if self._section_idx >= len(self._SECTION_ORDER):
                    self._to_results()
                    return
                if self._cfg.skip_practice_for_testing:
                    section = self._active_section()
                    self._phase = Phase.PRACTICE_DONE
                    self._pending_done_action = "start_scored"
                    self._practice_done_prompt = (
                        f"{_section_name(section)} practice skipped (testing mode). "
                        "Press Enter to start scored."
                    )
                    self._current_problem = None
                    return
                self._begin_block(is_practice=True)
                return
            return

        # Testing convenience: allow direct start into scored from instructions.
        if self._phase is Phase.INSTRUCTIONS:
            self._section_idx = self._SECTION_ORDER.index(self._start_section)
            self._begin_block(is_practice=False)

    def submit_answer(self, raw: str) -> bool:
        raw_in = str(raw).strip()
        if self._handle_control_command(raw_in):
            return True

        if self._phase not in (Phase.PRACTICE, Phase.SCORED):
            return False
        if self._current_problem is None:
            return False

        payload = self._payload_or_none(self._current_problem)
        if payload is None or payload.trial_stage is not SpatialIntegrationTrialStage.QUESTION:
            return False
        if raw_in == "":
            return False

        try:
            user_answer = int(raw_in)
        except ValueError:
            return False

        now = self._clock.now()
        presented = (
            self._question_presented_at_s if self._question_presented_at_s is not None else now
        )
        response_time_s = max(0.0, now - presented)

        score = float(
            self._scorer.score(problem=self._current_problem, user_answer=user_answer, raw=raw_in)
        )
        if score < 0.0:
            score = 0.0
        elif score > 1.0:
            score = 1.0
        is_full_correct = score >= 1.0 - 1e-9

        event = QuestionEvent(
            index=len(self._events),
            phase=self._phase,
            prompt=self._current_problem.prompt,
            correct_answer=int(self._current_problem.answer),
            user_answer=int(user_answer),
            is_correct=bool(is_full_correct),
            presented_at_s=float(presented),
            answered_at_s=float(now),
            response_time_s=float(response_time_s),
            raw=str(raw_in),
            score=float(score),
            max_score=1.0,
        )
        self._events.append(event)

        if self._phase is Phase.SCORED:
            self._scored_attempted += 1
            self._scored_max_score += 1.0
            self._scored_total_score += score
            if is_full_correct:
                self._scored_correct += 1
        else:
            if is_full_correct:
                self._practice_feedback = "Practice: correct"
            elif score > 0.0:
                self._practice_feedback = "Practice: close"
            else:
                self._practice_feedback = "Practice: incorrect"

        self._block_trial_index += 1
        if self._block_trial_index >= self._block_trial_target:
            self._complete_active_block()
        else:
            self._deal_next_trial()
        return True

    def update(self) -> None:
        now = self._clock.now()
        dt = max(0.0, now - self._last_update_at_s)
        self._last_update_at_s = now

        if self._phase is Phase.SCORED:
            self._scored_elapsed_s += dt

        if self._phase not in (Phase.PRACTICE, Phase.SCORED):
            return
        if self._current_problem is None:
            return
        payload = self._payload_or_none(self._current_problem)
        if payload is None:
            return
        if payload.trial_stage is not SpatialIntegrationTrialStage.MEMORIZE:
            return
        if self._stage_ends_at_s is None:
            return
        if now < self._stage_ends_at_s:
            return
        self._set_trial_stage(SpatialIntegrationTrialStage.QUESTION)
        self._question_presented_at_s = now

    def time_remaining_s(self) -> float | None:
        if self._phase not in (Phase.PRACTICE, Phase.SCORED):
            return None
        if self._current_problem is None:
            return None
        payload = self._payload_or_none(self._current_problem)
        if payload is None or payload.trial_stage is not SpatialIntegrationTrialStage.MEMORIZE:
            return None
        if self._stage_ends_at_s is None:
            return None
        return max(0.0, self._stage_ends_at_s - self._clock.now())

    def current_prompt(self) -> str:
        if self._phase is Phase.INSTRUCTIONS:
            return (
                "Spatial Integration\n"
                "Section A: top-down memory scene.\n"
                "Section B: lower/oblique memory scene.\n"
                "Section C: oblique scene with moving aircraft.\n\n"
                "Each section has a short practice, then scored trials.\n"
                "Controls: type A, S, D, F, or G then Enter.\n"
                "Testing shortcuts: F10 skip practice, F11 skip section, F8 skip all."
            )
        if self._phase is Phase.PRACTICE_DONE:
            return self._practice_done_prompt
        if self._phase is Phase.RESULTS:
            s = self.scored_summary()
            acc = int(round(s.accuracy * 100.0))
            rt = "n/a" if s.mean_response_time_s is None else f"{s.mean_response_time_s:.2f}s"
            return (
                "Results\n"
                f"Attempted: {s.attempted}\n"
                f"Correct: {s.correct}\n"
                f"Accuracy: {acc}%\n"
                f"Mean RT: {rt}\n"
                f"Throughput: {s.throughput_per_min:.1f}/min"
            )
        if self._current_problem is None:
            return ""
        payload = self._payload_or_none(self._current_problem)
        if payload is None:
            return self._current_problem.prompt
        if payload.trial_stage is SpatialIntegrationTrialStage.MEMORIZE:
            return (
                f"{_section_name(payload.section)} {payload.block_kind.title()} "
                f"({payload.trial_index_in_block}/{payload.trials_in_block})\n"
                "Memorize the scene."
            )
        return self._current_problem.prompt

    def scored_summary(self) -> AttemptSummary:
        attempted = int(self._scored_attempted)
        correct = int(self._scored_correct)
        duration_s = max(0.001, float(self._scored_elapsed_s))
        accuracy = 0.0 if attempted == 0 else correct / attempted
        throughput = (attempted / duration_s) * 60.0
        rts = [e.response_time_s for e in self._events if e.phase is Phase.SCORED]
        mean_rt = None if not rts else sum(rts) / len(rts)

        total_score = float(self._scored_total_score)
        max_score = float(self._scored_max_score)
        score_ratio = 0.0 if max_score <= 0.0 else total_score / max_score

        return AttemptSummary(
            attempted=attempted,
            correct=correct,
            accuracy=float(accuracy),
            duration_s=float(duration_s),
            throughput_per_min=float(throughput),
            mean_response_time_s=mean_rt,
            total_score=total_score,
            max_score=max_score,
            score_ratio=score_ratio,
        )

    def snapshot(self) -> TestSnapshot:
        payload = self._snapshot_payload()
        hint = "Type A, S, D, F, or G then Enter"
        if (
            isinstance(payload, SpatialIntegrationPayload)
            and payload.trial_stage is SpatialIntegrationTrialStage.MEMORIZE
        ):
            hint = "Memorize scene (wait for question)"
        return TestSnapshot(
            title="Spatial Integration",
            phase=self._phase,
            prompt=self.current_prompt(),
            input_hint=hint,
            time_remaining_s=self.time_remaining_s(),
            attempted_scored=int(self._scored_attempted),
            correct_scored=int(self._scored_correct),
            payload=payload,
            practice_feedback=self._practice_feedback,
        )

    def _begin_block(self, *, is_practice: bool) -> None:
        self._block_is_practice = bool(is_practice)
        self._block_trial_index = 0
        self._block_trial_target = (
            int(self._cfg.practice_questions)
            if is_practice
            else int(self._cfg.scored_questions_per_section)
        )
        self._practice_feedback = None

        if self._block_trial_target <= 0:
            self._complete_active_block()
            return

        self._phase = Phase.PRACTICE if is_practice else Phase.SCORED
        self._pending_done_action = None
        self._practice_done_prompt = ""
        self._deal_next_trial()

    def _deal_next_trial(self) -> None:
        section = self._active_section()
        problem = self._generator.next_problem(section=section, difficulty=self._difficulty)
        payload = self._payload_or_none(problem)
        if payload is None:
            self._current_problem = problem
            return
        staged = replace(
            payload,
            block_kind="practice" if self._block_is_practice else "scored",
            trial_index_in_block=int(self._block_trial_index + 1),
            trials_in_block=int(self._block_trial_target),
            trial_stage=SpatialIntegrationTrialStage.MEMORIZE,
            stage_time_remaining_s=self._memorize_duration_s(),
        )
        self._current_problem = Problem(
            prompt=problem.prompt,
            answer=problem.answer,
            tolerance=problem.tolerance,
            payload=staged,
        )
        self._question_presented_at_s = None
        self._stage_ends_at_s = self._clock.now() + self._memorize_duration_s()

    def _set_trial_stage(self, stage: SpatialIntegrationTrialStage) -> None:
        if self._current_problem is None:
            return
        payload = self._payload_or_none(self._current_problem)
        if payload is None:
            return
        updated = replace(
            payload,
            trial_stage=stage,
            stage_time_remaining_s=None
            if stage is SpatialIntegrationTrialStage.QUESTION
            else self._memorize_duration_s(),
        )
        self._current_problem = Problem(
            prompt=self._current_problem.prompt,
            answer=self._current_problem.answer,
            tolerance=self._current_problem.tolerance,
            payload=updated,
        )

    def _complete_active_block(self) -> None:
        section = self._active_section()
        self._current_problem = None
        self._stage_ends_at_s = None
        self._question_presented_at_s = None

        if self._block_is_practice:
            self._phase = Phase.PRACTICE_DONE
            self._pending_done_action = "start_scored"
            self._practice_done_prompt = (
                f"{_section_name(section)} practice complete. Press Enter to start scored."
            )
            return

        if self._section_idx + 1 < len(self._SECTION_ORDER):
            next_section = self._SECTION_ORDER[self._section_idx + 1]
            self._phase = Phase.PRACTICE_DONE
            self._pending_done_action = "start_next_practice"
            self._practice_done_prompt = (
                f"{_section_name(section)} scored complete. "
                f"Press Enter to begin {_section_name(next_section)} practice."
            )
            return

        self._to_results()

    def _to_results(self) -> None:
        self._phase = Phase.RESULTS
        self._pending_done_action = None
        self._current_problem = None
        self._stage_ends_at_s = None
        self._question_presented_at_s = None

    def _handle_control_command(self, raw: str) -> bool:
        token = str(raw).strip().lower()
        if token in {"__skip_all__", "skip_all"}:
            self._to_results()
            return True

        if token in {"__skip_practice__", "skip_practice"}:
            if self._phase is not Phase.PRACTICE:
                return False
            section = self._active_section()
            self._phase = Phase.PRACTICE_DONE
            self._pending_done_action = "start_scored"
            self._practice_done_prompt = (
                f"{_section_name(section)} practice skipped. Press Enter to start scored."
            )
            self._current_problem = None
            self._stage_ends_at_s = None
            self._question_presented_at_s = None
            return True

        if token in {"__skip_section__", "skip_section"}:
            section = self._active_section()
            if self._section_idx + 1 < len(self._SECTION_ORDER):
                next_section = self._SECTION_ORDER[self._section_idx + 1]
                self._phase = Phase.PRACTICE_DONE
                self._pending_done_action = "start_next_practice"
                self._practice_done_prompt = (
                    f"{_section_name(section)} skipped. "
                    f"Press Enter to begin {_section_name(next_section)} practice."
                )
                self._current_problem = None
                self._stage_ends_at_s = None
                self._question_presented_at_s = None
                return True
            self._to_results()
            return True

        return False

    def _snapshot_payload(self) -> object | None:
        if self._current_problem is None:
            return None
        payload = self._payload_or_none(self._current_problem)
        if payload is None:
            return self._current_problem.payload
        if payload.trial_stage is not SpatialIntegrationTrialStage.MEMORIZE:
            return payload
        if self._stage_ends_at_s is None:
            return payload
        rem = max(0.0, self._stage_ends_at_s - self._clock.now())
        return replace(payload, stage_time_remaining_s=float(rem))

    def _memorize_duration_s(self) -> float:
        return float(
            self._cfg.practice_memorize_s
            if self._block_is_practice
            else self._cfg.scored_memorize_s
        )

    def _active_section(self) -> SpatialIntegrationSection:
        idx = max(0, min(len(self._SECTION_ORDER) - 1, int(self._section_idx)))
        return self._SECTION_ORDER[idx]

    @staticmethod
    def _payload_or_none(problem: Problem) -> SpatialIntegrationPayload | None:
        payload = problem.payload
        return payload if isinstance(payload, SpatialIntegrationPayload) else None


def build_spatial_integration_test(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    config: SpatialIntegrationConfig | None = None,
) -> SpatialIntegrationEngine:
    cfg = config or SpatialIntegrationConfig()

    skip_env = os.environ.get("CFAST_SPATIAL_SKIP_PRACTICE", "").strip().lower()
    if skip_env in {"1", "true", "yes", "on"}:
        cfg = replace(cfg, skip_practice_for_testing=True)

    start_env = os.environ.get("CFAST_SPATIAL_START_SECTION", "").strip().upper()
    if start_env in {"A", "B", "C"}:
        cfg = replace(cfg, start_section=start_env)

    return SpatialIntegrationEngine(
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        config=cfg,
    )
