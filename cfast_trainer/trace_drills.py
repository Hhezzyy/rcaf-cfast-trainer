from __future__ import annotations

from dataclasses import dataclass

from .ant_drills import ANT_DRILL_MODE_PROFILES, AntDrillAttemptSummary, AntDrillMode
from .clock import Clock
from .cognitive_core import AttemptSummary, Phase, QuestionEvent, TestSnapshot, clamp01
from .trace_test_1 import (
    TraceTest1Command,
    TraceTest1Config,
    TraceTest1Payload,
    build_trace_test_1_test,
)
from .trace_test_2 import (
    TraceTest2Config,
    TraceTest2Payload,
    TraceTest2QuestionKind,
    build_trace_test_2_test,
)

_DEFAULT_TT1_CONFIG = TraceTest1Config()
_DEFAULT_TT2_CONFIG = TraceTest2Config()
_SKIP_ALL_TOKENS = {"__skip_all__", "skip_all"}
_SKIP_SECTION_TOKENS = {"__skip_section__", "skip_section"}


def _normalize_mode(mode: AntDrillMode | str) -> AntDrillMode:
    if isinstance(mode, AntDrillMode):
        return mode
    return AntDrillMode(str(mode).strip().lower())


def _difficulty_to_level(difficulty: float) -> int:
    return max(1, min(10, int(round(clamp01(difficulty) * 9.0)) + 1))


def _summary_from_attempts(
    *,
    attempt_summaries: tuple[AttemptSummary, ...],
    events: tuple[QuestionEvent, ...],
    mode: AntDrillMode,
    difficulty: float,
) -> AntDrillAttemptSummary:
    attempted = sum(summary.attempted for summary in attempt_summaries)
    correct = sum(summary.correct for summary in attempt_summaries)
    duration_s = float(sum(summary.duration_s for summary in attempt_summaries))
    total_score = float(sum(summary.total_score for summary in attempt_summaries))
    max_score = float(sum(summary.max_score for summary in attempt_summaries))
    accuracy = 0.0 if attempted == 0 else correct / attempted
    throughput = (attempted / duration_s) * 60.0 if duration_s > 0.0 else 0.0
    correct_per_min = (correct / duration_s) * 60.0 if duration_s > 0.0 else 0.0
    scored_events = tuple(event for event in events if event.phase is Phase.SCORED)
    mean_rt = None
    if scored_events:
        mean_rt = sum(event.response_time_s for event in scored_events) / len(scored_events)
    timeouts = sum(1 for event in scored_events if str(event.raw).strip().upper() in {"", "TIMEOUT"})
    fixation_rate = 0.0 if attempted == 0 else timeouts / attempted
    level = _difficulty_to_level(difficulty)
    return AntDrillAttemptSummary(
        attempted=attempted,
        correct=correct,
        accuracy=float(accuracy),
        duration_s=duration_s,
        throughput_per_min=float(throughput),
        mean_response_time_s=None if mean_rt is None else float(mean_rt),
        total_score=total_score,
        max_score=max_score,
        score_ratio=0.0 if max_score == 0.0 else total_score / max_score,
        correct_per_min=float(correct_per_min),
        timeouts=int(timeouts),
        fixation_rate=float(fixation_rate),
        max_timeout_streak=_max_timeout_streak(scored_events),
        mode=mode.value,
        difficulty_level=level,
        difficulty_level_start=level,
        difficulty_level_end=level,
        difficulty_change_count=0,
        adaptive_enabled=False,
        adaptive_window_size=0,
    )


def _max_timeout_streak(events: tuple[QuestionEvent, ...]) -> int:
    max_streak = 0
    current = 0
    for event in events:
        is_timeout = str(event.raw).strip().upper() in {"", "TIMEOUT"}
        if is_timeout:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak


def _force_engine_result(engine: object) -> None:
    if not hasattr(engine, "_phase"):
        return
    engine._phase = Phase.RESULTS
    for attr in (
        "_current_problem",
        "_current_prompt",
        "_trial_started_at_s",
        "_current",
        "_current_payload",
        "_question_started_at_s",
    ):
        if hasattr(engine, attr):
            setattr(engine, attr, None)


@dataclass(frozen=True, slots=True)
class TraceDrillConfig:
    practice_questions_per_segment: int | None = None
    scored_duration_s: float | None = None


class TraceSingleDrill:
    def __init__(
        self,
        *,
        title: str,
        instructions: tuple[str, ...],
        engine: object,
        seed: int,
        difficulty: float,
        mode: AntDrillMode,
        scored_duration_s: float,
        practice_questions_per_segment: int,
    ) -> None:
        self._title = str(title)
        self._instructions = tuple(str(line) for line in instructions)
        self._engine = engine
        self._seed = int(seed)
        self._difficulty = float(difficulty)
        self._mode = mode
        self._scored_duration_s = float(scored_duration_s)
        self._practice_questions_per_segment = max(0, int(practice_questions_per_segment))
        self._phase = Phase.INSTRUCTIONS

    def __getattr__(self, name: str):
        return getattr(self._engine, name)

    @property
    def phase(self) -> Phase:
        return self._engine.phase

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def difficulty(self) -> float:
        return self._difficulty

    @property
    def practice_questions(self) -> int:
        return self._practice_questions_per_segment

    @property
    def scored_duration_s(self) -> float:
        return self._scored_duration_s

    def can_exit(self) -> bool:
        return self._engine.can_exit()

    def start_practice(self) -> None:
        if self._engine.phase is Phase.INSTRUCTIONS:
            self._engine.start_practice()
        self._phase = self._engine.phase

    def start_scored(self) -> None:
        if self._engine.phase is Phase.INSTRUCTIONS:
            self._engine.start_practice()
        if self._engine.phase is Phase.PRACTICE_DONE:
            self._engine.start_scored()
        self._phase = self._engine.phase

    def submit_answer(self, raw: str) -> bool:
        token = str(raw).strip().lower()
        if token in _SKIP_ALL_TOKENS or token in _SKIP_SECTION_TOKENS:
            _force_engine_result(self._engine)
            self._phase = Phase.RESULTS
            return True
        accepted = bool(self._engine.submit_answer(raw))
        self._phase = self._engine.phase
        return accepted

    def update(self) -> None:
        self._engine.update()
        self._phase = self._engine.phase

    def snapshot(self) -> TestSnapshot:
        snap = self._engine.snapshot()
        prompt = str(snap.prompt)
        input_hint = str(snap.input_hint)
        if snap.phase is Phase.INSTRUCTIONS:
            prompt = "\n".join(self._instructions)
            input_hint = "Press Enter to begin."
        elif snap.phase is Phase.PRACTICE_DONE:
            input_hint = "Press Enter to continue."
        return TestSnapshot(
            title=self._title,
            phase=snap.phase,
            prompt=prompt,
            input_hint=input_hint,
            time_remaining_s=snap.time_remaining_s,
            attempted_scored=snap.attempted_scored,
            correct_scored=snap.correct_scored,
            payload=snap.payload,
            practice_feedback=snap.practice_feedback,
        )

    def events(self) -> list[QuestionEvent]:
        return self._engine.events()

    def scored_summary(self) -> AntDrillAttemptSummary:
        return _summary_from_attempts(
            attempt_summaries=(self._engine.scored_summary(),),
            events=tuple(self.events()),
            mode=self._mode,
            difficulty=self._difficulty,
        )


@dataclass(frozen=True, slots=True)
class _TraceSegment:
    title: str
    engine: object


class TraceMixedDrill:
    def __init__(
        self,
        *,
        title: str,
        instructions: tuple[str, ...],
        segments: tuple[_TraceSegment, ...],
        seed: int,
        difficulty: float,
        mode: AntDrillMode,
        scored_duration_s: float,
        practice_questions_per_segment: int,
    ) -> None:
        if not segments:
            raise ValueError("TraceMixedDrill requires at least one segment")
        self._title = str(title)
        self._instructions = tuple(str(line) for line in instructions)
        self._segments = segments
        self._seed = int(seed)
        self._difficulty = float(difficulty)
        self._mode = mode
        self._scored_duration_s = float(scored_duration_s)
        self._practice_questions_per_segment = max(0, int(practice_questions_per_segment))
        self._phase = Phase.INSTRUCTIONS
        self._run_cycle = "instructions"
        self._active_index = 0

    def __getattr__(self, name: str):
        engine = self._current_engine()
        if engine is None:
            raise AttributeError(name)
        return getattr(engine, name)

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def difficulty(self) -> float:
        return self._difficulty

    @property
    def practice_questions(self) -> int:
        return self._practice_questions_per_segment * len(self._segments)

    @property
    def scored_duration_s(self) -> float:
        return self._scored_duration_s

    def can_exit(self) -> bool:
        engine = self._current_engine()
        if engine is None:
            return self._phase in (Phase.INSTRUCTIONS, Phase.PRACTICE_DONE, Phase.RESULTS)
        return engine.can_exit()

    def start_practice(self) -> None:
        if self._phase is not Phase.INSTRUCTIONS:
            return
        self._run_cycle = "practice"
        self._active_index = 0
        self._current_engine().start_practice()
        self._sync_phase()

    def start_scored(self) -> None:
        if self._phase is Phase.INSTRUCTIONS:
            self.start_practice()
        if self._phase is not Phase.PRACTICE_DONE:
            return
        self._run_cycle = "scored"
        self._active_index = 0
        self._current_engine().start_scored()
        self._sync_phase()

    def submit_answer(self, raw: str) -> bool:
        token = str(raw).strip().lower()
        if token in _SKIP_ALL_TOKENS:
            self._finish_all()
            return True
        if token in _SKIP_SECTION_TOKENS:
            self._skip_current_section()
            return True
        engine = self._current_engine()
        if engine is None:
            return False
        accepted = bool(engine.submit_answer(raw))
        self._sync_phase()
        return accepted

    def update(self) -> None:
        engine = self._current_engine()
        if engine is not None:
            engine.update()
        self._sync_phase()

    def snapshot(self) -> TestSnapshot:
        if self._phase is Phase.INSTRUCTIONS:
            return TestSnapshot(
                title=self._title,
                phase=Phase.INSTRUCTIONS,
                prompt="\n".join(self._instructions),
                input_hint="Press Enter to begin.",
                time_remaining_s=None,
                attempted_scored=0,
                correct_scored=0,
            )
        if self._phase is Phase.PRACTICE_DONE:
            return TestSnapshot(
                title=self._title,
                phase=Phase.PRACTICE_DONE,
                prompt="Practice complete. Press Enter to start the timed test.",
                input_hint="Press Enter to continue.",
                time_remaining_s=None,
                attempted_scored=0,
                correct_scored=0,
            )
        if self._phase is Phase.RESULTS:
            summary = self.scored_summary()
            acc_pct = int(round(summary.accuracy * 100))
            rt = "n/a" if summary.mean_response_time_s is None else f"{summary.mean_response_time_s:.2f}s"
            return TestSnapshot(
                title=self._title,
                phase=Phase.RESULTS,
                prompt=(
                    f"Results\nAttempted: {summary.attempted}\nCorrect: {summary.correct}\n"
                    f"Accuracy: {acc_pct}%\nMean RT: {rt}\n"
                    f"Throughput: {summary.throughput_per_min:.1f}/min"
                ),
                input_hint="",
                time_remaining_s=None,
                attempted_scored=summary.attempted,
                correct_scored=summary.correct,
            )

        engine = self._current_engine()
        assert engine is not None
        snap = engine.snapshot()
        return TestSnapshot(
            title=self._segments[self._active_index].title,
            phase=snap.phase,
            prompt=str(snap.prompt),
            input_hint=str(snap.input_hint),
            time_remaining_s=snap.time_remaining_s,
            attempted_scored=snap.attempted_scored + self._completed_attempts_before_current(),
            correct_scored=snap.correct_scored + self._completed_correct_before_current(),
            payload=snap.payload,
            practice_feedback=snap.practice_feedback,
        )

    def events(self) -> list[QuestionEvent]:
        events: list[QuestionEvent] = []
        for segment in self._segments:
            events.extend(segment.engine.events())
        return events

    def scored_summary(self) -> AntDrillAttemptSummary:
        return _summary_from_attempts(
            attempt_summaries=tuple(segment.engine.scored_summary() for segment in self._segments),
            events=tuple(self.events()),
            mode=self._mode,
            difficulty=self._difficulty,
        )

    def _current_engine(self):
        if self._run_cycle not in {"practice", "scored"}:
            return None
        return self._segments[self._active_index].engine

    def _sync_phase(self) -> None:
        while True:
            if self._run_cycle == "practice":
                engine = self._segments[self._active_index].engine
                if engine.phase is Phase.PRACTICE_DONE:
                    if self._active_index + 1 < len(self._segments):
                        self._active_index += 1
                        next_engine = self._segments[self._active_index].engine
                        if next_engine.phase is Phase.INSTRUCTIONS:
                            next_engine.start_practice()
                        continue
                    self._run_cycle = "practice_done"
                    self._phase = Phase.PRACTICE_DONE
                    return
                self._phase = engine.phase
                return
            if self._run_cycle == "scored":
                engine = self._segments[self._active_index].engine
                if engine.phase is Phase.RESULTS:
                    if self._active_index + 1 < len(self._segments):
                        self._active_index += 1
                        next_engine = self._segments[self._active_index].engine
                        if next_engine.phase in (Phase.INSTRUCTIONS, Phase.PRACTICE_DONE):
                            next_engine.start_scored()
                        continue
                    self._run_cycle = "results"
                    self._phase = Phase.RESULTS
                    return
                self._phase = engine.phase
                return
            if self._run_cycle == "practice_done":
                self._phase = Phase.PRACTICE_DONE
                return
            if self._run_cycle == "results":
                self._phase = Phase.RESULTS
                return
            self._phase = Phase.INSTRUCTIONS
            return

    def _skip_current_section(self) -> None:
        if self._phase not in (Phase.PRACTICE, Phase.SCORED):
            return
        engine = self._segments[self._active_index].engine
        _force_engine_result(engine if self._run_cycle == "scored" else engine)
        if self._run_cycle == "practice":
            engine._phase = Phase.PRACTICE_DONE
        self._sync_phase()

    def _finish_all(self) -> None:
        for idx, segment in enumerate(self._segments):
            engine = segment.engine
            if idx < self._active_index:
                continue
            _force_engine_result(engine)
        self._run_cycle = "results"
        self._phase = Phase.RESULTS

    def _completed_attempts_before_current(self) -> int:
        return sum(
            segment.engine.scored_summary().attempted
            for segment in self._segments[: self._active_index]
        )

    def _completed_correct_before_current(self) -> int:
        return sum(
            segment.engine.scored_summary().correct
            for segment in self._segments[: self._active_index]
        )


def _build_tt1_engine(
    *,
    clock: Clock,
    seed: int,
    difficulty: float,
    mode: AntDrillMode,
    config: TraceDrillConfig | None,
    allowed_commands: tuple[TraceTest1Command, ...] | None,
) -> tuple[object, float, int]:
    mode_profile = ANT_DRILL_MODE_PROFILES[mode]
    cfg = config or TraceDrillConfig()
    practice_questions = (
        int(cfg.practice_questions_per_segment)
        if cfg.practice_questions_per_segment is not None
        else (2 if mode in (AntDrillMode.FRESH, AntDrillMode.BUILD) else 0)
    )
    scored_duration_s = (
        float(cfg.scored_duration_s)
        if cfg.scored_duration_s is not None
        else float(mode_profile.scored_duration_s)
    )
    engine = build_trace_test_1_test(
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        config=TraceTest1Config(
            scored_duration_s=scored_duration_s,
            practice_questions=practice_questions,
            practice_observe_s=float(_DEFAULT_TT1_CONFIG.practice_observe_s * mode_profile.cap_scale),
            scored_observe_s=float(_DEFAULT_TT1_CONFIG.scored_observe_s * mode_profile.cap_scale),
            allowed_commands=allowed_commands,
        ),
    )
    return engine, scored_duration_s, practice_questions


def _build_tt2_engine(
    *,
    clock: Clock,
    seed: int,
    difficulty: float,
    mode: AntDrillMode,
    config: TraceDrillConfig | None,
    allowed_question_kinds: tuple[TraceTest2QuestionKind, ...] | None,
) -> tuple[object, float, int]:
    mode_profile = ANT_DRILL_MODE_PROFILES[mode]
    cfg = config or TraceDrillConfig()
    practice_questions = (
        int(cfg.practice_questions_per_segment)
        if cfg.practice_questions_per_segment is not None
        else (2 if mode in (AntDrillMode.FRESH, AntDrillMode.BUILD) else 0)
    )
    scored_duration_s = (
        float(cfg.scored_duration_s)
        if cfg.scored_duration_s is not None
        else float(mode_profile.scored_duration_s)
    )
    engine = build_trace_test_2_test(
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        config=TraceTest2Config(
            scored_duration_s=scored_duration_s,
            practice_questions=practice_questions,
            practice_observe_s=float(_DEFAULT_TT2_CONFIG.practice_observe_s * mode_profile.cap_scale),
            scored_observe_s=float(_DEFAULT_TT2_CONFIG.scored_observe_s * mode_profile.cap_scale),
            allowed_question_kinds=allowed_question_kinds,
        ),
    )
    return engine, scored_duration_s, practice_questions


def _build_single_trace_drill(
    *,
    title: str,
    instructions: tuple[str, ...],
    engine: object,
    seed: int,
    difficulty: float,
    mode: AntDrillMode,
    scored_duration_s: float,
    practice_questions_per_segment: int,
) -> TraceSingleDrill:
    return TraceSingleDrill(
        title=title,
        instructions=instructions,
        engine=engine,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        scored_duration_s=scored_duration_s,
        practice_questions_per_segment=practice_questions_per_segment,
    )


def build_tt1_lateral_anchor_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: TraceDrillConfig | None = None,
) -> TraceSingleDrill:
    normalized_mode = _normalize_mode(mode)
    engine, scored_duration_s, practice_questions = _build_tt1_engine(
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=normalized_mode,
        config=config,
        allowed_commands=(TraceTest1Command.LEFT, TraceTest1Command.RIGHT),
    )
    return _build_single_trace_drill(
        title=f"Trace Test 1: Lateral Anchor ({ANT_DRILL_MODE_PROFILES[normalized_mode].label})",
        instructions=(
            "Trace Test 1 lateral block.",
            "Watch the red aircraft and answer only Left or Right with the arrow keys.",
            "Ignore vertical changes; this block filters to 90-degree lateral turns only.",
        ),
        engine=engine,
        seed=seed,
        difficulty=difficulty,
        mode=normalized_mode,
        scored_duration_s=scored_duration_s,
        practice_questions_per_segment=practice_questions,
    )


def build_tt1_vertical_anchor_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: TraceDrillConfig | None = None,
) -> TraceSingleDrill:
    normalized_mode = _normalize_mode(mode)
    engine, scored_duration_s, practice_questions = _build_tt1_engine(
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=normalized_mode,
        config=config,
        allowed_commands=(TraceTest1Command.PUSH, TraceTest1Command.PULL),
    )
    return _build_single_trace_drill(
        title=f"Trace Test 1: Vertical Anchor ({ANT_DRILL_MODE_PROFILES[normalized_mode].label})",
        instructions=(
            "Trace Test 1 vertical block.",
            "Watch the red aircraft and answer only Push or Pull with Up or Down.",
            "Ignore lateral changes; this block filters to vertical attitude changes only.",
        ),
        engine=engine,
        seed=seed,
        difficulty=difficulty,
        mode=normalized_mode,
        scored_duration_s=scored_duration_s,
        practice_questions_per_segment=practice_questions,
    )


def build_tt1_command_switch_run_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.TEMPO,
    config: TraceDrillConfig | None = None,
) -> TraceSingleDrill:
    normalized_mode = _normalize_mode(mode)
    engine, scored_duration_s, practice_questions = _build_tt1_engine(
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=normalized_mode,
        config=config,
        allowed_commands=None,
    )
    return _build_single_trace_drill(
        title=f"Trace Test 1: Command Switch Run ({ANT_DRILL_MODE_PROFILES[normalized_mode].label})",
        instructions=(
            "Trace Test 1 mixed command block.",
            "The live stream stays intact and all four commands can appear.",
            "Answer with arrow keys as soon as the maneuver opens and reset immediately after misses.",
        ),
        engine=engine,
        seed=seed,
        difficulty=difficulty,
        mode=normalized_mode,
        scored_duration_s=scored_duration_s,
        practice_questions_per_segment=practice_questions,
    )


def build_tt2_steady_anchor_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: TraceDrillConfig | None = None,
) -> TraceSingleDrill:
    normalized_mode = _normalize_mode(mode)
    engine, scored_duration_s, practice_questions = _build_tt2_engine(
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=normalized_mode,
        config=config,
        allowed_question_kinds=(TraceTest2QuestionKind.NO_DIRECTION_CHANGE,),
    )
    return _build_single_trace_drill(
        title=f"Trace Test 2: Steady Anchor ({ANT_DRILL_MODE_PROFILES[normalized_mode].label})",
        instructions=(
            "Trace Test 2 steady-recall block.",
            "Watch the clip first, then answer only the no-direction-change question family.",
            "Use A/S/D/F immediately after the observe stage opens into the question screen.",
        ),
        engine=engine,
        seed=seed,
        difficulty=difficulty,
        mode=normalized_mode,
        scored_duration_s=scored_duration_s,
        practice_questions_per_segment=practice_questions,
    )


def build_tt2_turn_trace_run_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.BUILD,
    config: TraceDrillConfig | None = None,
) -> TraceSingleDrill:
    normalized_mode = _normalize_mode(mode)
    engine, scored_duration_s, practice_questions = _build_tt2_engine(
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=normalized_mode,
        config=config,
        allowed_question_kinds=(
            TraceTest2QuestionKind.TURNED_LEFT,
            TraceTest2QuestionKind.TURNED_RIGHT,
        ),
    )
    return _build_single_trace_drill(
        title=f"Trace Test 2: Turn Trace Run ({ANT_DRILL_MODE_PROFILES[normalized_mode].label})",
        instructions=(
            "Trace Test 2 turn-recall block.",
            "Watch the clip first, then answer only left-turn and right-turn questions.",
            "Stay on the guide-style recall flow: observe first, then answer on the question screen.",
        ),
        engine=engine,
        seed=seed,
        difficulty=difficulty,
        mode=normalized_mode,
        scored_duration_s=scored_duration_s,
        practice_questions_per_segment=practice_questions,
    )


def build_tt2_position_recall_run_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.TEMPO,
    config: TraceDrillConfig | None = None,
) -> TraceSingleDrill:
    normalized_mode = _normalize_mode(mode)
    engine, scored_duration_s, practice_questions = _build_tt2_engine(
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=normalized_mode,
        config=config,
        allowed_question_kinds=(
            TraceTest2QuestionKind.ENDED_LEFTMOST,
            TraceTest2QuestionKind.ENDED_HIGHEST,
        ),
    )
    return _build_single_trace_drill(
        title=f"Trace Test 2: Position Recall Run ({ANT_DRILL_MODE_PROFILES[normalized_mode].label})",
        instructions=(
            "Trace Test 2 position-recall block.",
            "Watch the clip first, then answer only leftmost and highest end-state questions.",
            "Use the normal TT2 recall controls after the observe stage ends.",
        ),
        engine=engine,
        seed=seed,
        difficulty=difficulty,
        mode=normalized_mode,
        scored_duration_s=scored_duration_s,
        practice_questions_per_segment=practice_questions,
    )


def _build_mixed_trace_drill(
    *,
    title_base: str,
    instructions: tuple[str, ...],
    clock: Clock,
    seed: int,
    difficulty: float,
    mode: AntDrillMode | str,
    config: TraceDrillConfig | None,
) -> TraceMixedDrill:
    normalized_mode = _normalize_mode(mode)
    mode_profile = ANT_DRILL_MODE_PROFILES[normalized_mode]
    cfg = config or TraceDrillConfig()
    practice_questions_per_segment = (
        int(cfg.practice_questions_per_segment)
        if cfg.practice_questions_per_segment is not None
        else (1 if normalized_mode in (AntDrillMode.FRESH, AntDrillMode.BUILD) else 0)
    )
    scored_duration_s = (
        float(cfg.scored_duration_s)
        if cfg.scored_duration_s is not None
        else float(mode_profile.scored_duration_s)
    )
    per_segment_duration = scored_duration_s / 2.0
    tt1_engine = build_trace_test_1_test(
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        config=TraceTest1Config(
            scored_duration_s=per_segment_duration,
            practice_questions=practice_questions_per_segment,
            practice_observe_s=float(_DEFAULT_TT1_CONFIG.practice_observe_s * mode_profile.cap_scale),
            scored_observe_s=float(_DEFAULT_TT1_CONFIG.scored_observe_s * mode_profile.cap_scale),
            allowed_commands=None,
        ),
    )
    tt2_engine = build_trace_test_2_test(
        clock=clock,
        seed=seed + 1,
        difficulty=difficulty,
        config=TraceTest2Config(
            scored_duration_s=per_segment_duration,
            practice_questions=practice_questions_per_segment,
            practice_observe_s=float(_DEFAULT_TT2_CONFIG.practice_observe_s * mode_profile.cap_scale),
            scored_observe_s=float(_DEFAULT_TT2_CONFIG.scored_observe_s * mode_profile.cap_scale),
            allowed_question_kinds=None,
        ),
    )
    return TraceMixedDrill(
        title=f"Trace Tests: {title_base} ({mode_profile.label})",
        instructions=instructions,
        segments=(
            _TraceSegment(
                title=f"Trace Test 1: {title_base} ({mode_profile.label})",
                engine=tt1_engine,
            ),
            _TraceSegment(
                title=f"Trace Test 2: {title_base} ({mode_profile.label})",
                engine=tt2_engine,
            ),
        ),
        seed=seed,
        difficulty=difficulty,
        mode=normalized_mode,
        scored_duration_s=scored_duration_s,
        practice_questions_per_segment=practice_questions_per_segment,
    )


def build_trace_mixed_tempo_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.TEMPO,
    config: TraceDrillConfig | None = None,
) -> TraceMixedDrill:
    return _build_mixed_trace_drill(
        title_base="Mixed Tempo",
        instructions=(
            "Mixed trace block: Trace Test 1 first, then Trace Test 2.",
            "The TT1 stream runs first with arrow-key answers, then TT2 runs with observe-then-recall.",
            "The scored block splits time evenly across both tests.",
        ),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=config,
    )


def build_trace_pressure_run_drill(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    mode: AntDrillMode | str = AntDrillMode.STRESS,
    config: TraceDrillConfig | None = None,
) -> TraceMixedDrill:
    return _build_mixed_trace_drill(
        title_base="Pressure Run",
        instructions=(
            "Pressure trace block: Trace Test 1 first, then Trace Test 2.",
            "Both tests stay live, but the ANT stress profile tightens the observe windows.",
            "Recover immediately after misses and keep the block moving.",
        ),
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        mode=mode,
        config=config,
    )
