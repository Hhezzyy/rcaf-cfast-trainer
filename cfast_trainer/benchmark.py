from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum

from .adaptive_difficulty import build_resolved_difficulty_context, difficulty_ratio_for_level
from .airborne_numerical import build_airborne_numerical_test
from .angles_bearings_degrees import (
    AnglesBearingsDegreesConfig,
    build_angles_bearings_degrees_test,
)
from .auditory_capacity import AuditoryCapacityConfig, build_auditory_capacity_test
from .clock import Clock
from .cognitive_core import Phase, TestSnapshot
from .cognitive_updating import CognitiveUpdatingConfig, build_cognitive_updating_test
from .colours_letters_numbers import (
    ColoursLettersNumbersConfig,
    build_colours_letters_numbers_test,
)
from .digit_recognition import build_digit_recognition_test
from .guide_skill_catalog import OFFICIAL_GUIDE_TESTS
from .instrument_comprehension import (
    InstrumentComprehensionConfig,
    build_instrument_comprehension_test,
)
from .math_reasoning import MathReasoningConfig, build_math_reasoning_test
from .numerical_operations import NumericalOperationsConfig, build_numerical_operations_test
from .rapid_tracking import RapidTrackingConfig, build_rapid_tracking_test
from .results import AttemptResult, attempt_result_from_engine
from .sensory_motor_apparatus import (
    SensoryMotorApparatusConfig,
    build_sensory_motor_apparatus_test,
)
from .situational_awareness import SituationalAwarenessConfig, build_situational_awareness_test
from .spatial_integration import SpatialIntegrationConfig, build_spatial_integration_test
from .system_logic import SystemLogicConfig, build_system_logic_test
from .table_reading import TableReadingConfig, build_table_reading_test
from .target_recognition import TargetRecognitionConfig, build_target_recognition_test
from .telemetry import TelemetryAnalytics, TelemetryEvent, telemetry_analytics_from_events
from .trace_test_1 import TraceTest1Config, build_trace_test_1_test
from .trace_test_2 import TraceTest2Config, build_trace_test_2_test
from .training_modes import split_half_note_fragment
from .vigilance import VigilanceConfig, build_vigilance_test
from .visual_search import VisualSearchConfig, build_visual_search_test


@dataclass(frozen=True, slots=True)
class BenchmarkProbePlan:
    probe_code: str
    label: str
    builder: Callable[[int], object]
    seed: int
    difficulty_level: int
    duration_s: float


@dataclass(frozen=True, slots=True)
class BenchmarkPlan:
    code: str
    title: str
    version: int
    description: str
    notes: tuple[str, ...]
    probes: tuple[BenchmarkProbePlan, ...]

    @property
    def scored_duration_s(self) -> float:
        return float(sum(probe.duration_s for probe in self.probes))


class BenchmarkStage(StrEnum):
    INTRO = "intro"
    PROBE = "probe"
    PROBE_RESULTS = "probe_results"
    RESULTS = "results"


@dataclass(frozen=True, slots=True)
class BenchmarkProbeResult:
    probe_code: str
    label: str
    seed: int
    difficulty_level: int
    duration_s: float
    attempted: int
    correct: int
    accuracy: float
    throughput_per_min: float
    mean_rt_ms: float | None
    total_score: float | None
    max_score: float | None
    score_ratio: float | None
    difficulty_level_start: int | None
    difficulty_level_end: int | None
    completed: bool


@dataclass(frozen=True, slots=True)
class BenchmarkSnapshot:
    stage: BenchmarkStage
    title: str
    subtitle: str
    prompt: str
    note_lines: tuple[str, ...]
    probe_index: int
    probe_total: int
    current_probe_code: str | None
    current_probe_label: str | None
    current_probe_seed: int | None
    current_probe_difficulty_level: int | None
    probe_time_remaining_s: float | None
    battery_time_remaining_s: float | None
    attempted_total: int
    correct_total: int
    completed_probe_results: tuple[BenchmarkProbeResult, ...]
    latest_probe_result: BenchmarkProbeResult | None = None


@dataclass(frozen=True, slots=True)
class BenchmarkSummary:
    attempted: int
    correct: int
    accuracy: float
    duration_s: float
    throughput_per_min: float
    mean_response_time_s: float | None
    total_score: float = 0.0
    max_score: float = 0.0
    score_ratio: float = 0.0
    difficulty_level: int = 5
    difficulty_level_start: int | None = None
    difficulty_level_end: int | None = None
    difficulty_change_count: int = 0
    probe_count: int = 0
    completed_probes: int = 0
    benchmark_code: str = "benchmark_battery"
    benchmark_version: int = 3
    difficulty_policy: str = "fixed_per_probe"


class _PreparedBenchmarkProbeRuntime:
    _SKIP_TOKENS = frozenset({"__skip_section__", "skip_section", "__skip_all__", "skip_all"})

    def __init__(self, *, engine: object, probe_label: str) -> None:
        self._engine = engine
        self._probe_label = str(probe_label)
        self._launched = False
        self._forced_results = False

    def __getattr__(self, name: str) -> object:
        return getattr(self._engine, name)

    @property
    def _clock(self) -> object:
        return getattr(self._engine, "_clock")

    @_clock.setter
    def _clock(self, value: object) -> None:
        setattr(self._engine, "_clock", value)

    @property
    def phase(self) -> Phase:
        engine_phase = getattr(self._engine, "phase", None)
        if self._forced_results or engine_phase is Phase.RESULTS:
            return Phase.RESULTS
        if not self._launched:
            return Phase.INSTRUCTIONS
        if isinstance(engine_phase, Phase):
            return engine_phase
        return Phase.SCORED

    def can_exit(self) -> bool:
        if not self._launched:
            return True
        can_exit = getattr(self._engine, "can_exit", None)
        if callable(can_exit):
            return bool(can_exit())
        return self.phase is not Phase.SCORED

    def start_practice(self) -> None:
        if self._forced_results:
            return
        if self._launched:
            starter = getattr(self._engine, "start_practice", None)
            if callable(starter):
                starter()
            return
        self._launch_into_scored()

    def start_scored(self) -> None:
        if self._forced_results:
            return
        if self._launched:
            starter = getattr(self._engine, "start_scored", None)
            if callable(starter):
                starter()
            return
        self._launch_into_scored()

    def submit_answer(self, raw: str) -> bool:
        token = str(raw).strip().lower()
        if not self._launched:
            if token in self._SKIP_TOKENS:
                self._forced_results = True
                return True
            return False
        submit = getattr(self._engine, "submit_answer", None)
        if not callable(submit):
            return False
        return bool(submit(raw))

    def update(self) -> None:
        if not self._launched or self._forced_results:
            return
        update = getattr(self._engine, "update", None)
        if callable(update):
            update()

    def finish(self) -> None:
        if not self._launched:
            self._forced_results = True
            return
        finish = getattr(self._engine, "finish", None)
        if callable(finish):
            finish()
            return
        if hasattr(self._engine, "phase"):
            try:
                self._engine.phase = Phase.RESULTS
            except Exception:
                if hasattr(self._engine, "_phase"):
                    self._engine._phase = Phase.RESULTS

    def snapshot(self) -> TestSnapshot:
        if self._launched:
            return self._engine.snapshot()
        base = self._engine.snapshot()
        return TestSnapshot(
            title=str(base.title),
            phase=Phase.INSTRUCTIONS,
            prompt=("Benchmark probe guide ready.\nPress Enter to begin the timed segment."),
            input_hint="Press Enter to begin timed block",
            time_remaining_s=None,
            attempted_scored=int(base.attempted_scored),
            correct_scored=int(base.correct_scored),
            payload=None,
        )

    def _launch_into_scored(self) -> None:
        if self._launched or self._forced_results:
            return
        starter = getattr(self._engine, "start_scored", None)
        if callable(starter):
            starter()
        if getattr(self._engine, "phase", None) is not Phase.SCORED:
            practice_starter = getattr(self._engine, "start_practice", None)
            if callable(practice_starter):
                practice_starter()
            if callable(starter):
                starter()
        self._launched = True


class BenchmarkSession:
    _NOMINAL_DIFFICULTY = 0.5

    def __init__(self, *, plan: BenchmarkPlan) -> None:
        self._plan = plan
        self._stage = BenchmarkStage.INTRO
        self._run_seed = _random_seed()
        self._current_probe_index = 0
        self._current_engine: object | None = None
        self._completed_attempts: list[AttemptResult] = []
        self._probe_seeds = [int(probe.seed) for probe in self._plan.probes]
        self._latest_probe_result: BenchmarkProbeResult | None = None

    @property
    def seed(self) -> int:
        return int(self._run_seed)

    @property
    def difficulty(self) -> float:
        return float(self._NOMINAL_DIFFICULTY)

    @property
    def practice_questions(self) -> int:
        return 0

    @property
    def scored_duration_s(self) -> float:
        return self._plan.scored_duration_s

    @property
    def phase(self) -> Phase:
        if self._stage is BenchmarkStage.RESULTS:
            return Phase.RESULTS
        if self._stage is BenchmarkStage.PROBE:
            current_phase = getattr(self._current_engine, "phase", None)
            if isinstance(current_phase, Phase):
                return current_phase
            return Phase.SCORED
        if self._stage is BenchmarkStage.PROBE_RESULTS:
            return Phase.PRACTICE_DONE
        return Phase.INSTRUCTIONS

    @property
    def stage(self) -> BenchmarkStage:
        return self._stage

    def can_exit(self) -> bool:
        if self._stage in (BenchmarkStage.INTRO, BenchmarkStage.PROBE_RESULTS, BenchmarkStage.RESULTS):
            return True
        current_engine = self._current_engine
        if current_engine is None:
            return True
        can_exit = getattr(current_engine, "can_exit", None)
        if callable(can_exit):
            return bool(can_exit())
        return getattr(current_engine, "phase", None) is not Phase.SCORED

    def current_engine(self) -> object | None:
        return self._current_engine

    def current_probe_plan(self) -> BenchmarkProbePlan | None:
        if self._stage not in (BenchmarkStage.PROBE, BenchmarkStage.PROBE_RESULTS):
            return None
        if not (0 <= self._current_probe_index < len(self._plan.probes)):
            return None
        return replace(
            self._plan.probes[self._current_probe_index],
            seed=int(self._probe_seeds[self._current_probe_index]),
        )

    def latest_completed_attempt(self) -> AttemptResult | None:
        if not self._completed_attempts:
            return None
        return self._completed_attempts[-1]

    def latest_completed_probe(self) -> BenchmarkProbePlan | None:
        if not self._completed_attempts:
            return None
        index = len(self._completed_attempts) - 1
        if not (0 <= index < len(self._plan.probes)):
            return None
        return replace(self._plan.probes[index], seed=int(self._probe_seeds[index]))

    def latest_completed_probe_index(self) -> int | None:
        if not self._completed_attempts:
            return None
        return len(self._completed_attempts) - 1

    def activate(self) -> None:
        if self._stage is not BenchmarkStage.INTRO:
            return
        if not self._plan.probes:
            self._stage = BenchmarkStage.RESULTS
            return
        self._start_probe(0)

    def continue_after_probe_results(self) -> None:
        if self._stage is not BenchmarkStage.PROBE_RESULTS:
            return
        next_index = self._current_probe_index + 1
        self._latest_probe_result = None
        if next_index >= len(self._plan.probes):
            self._stage = BenchmarkStage.RESULTS
            self._current_probe_index = len(self._plan.probes)
            return
        self._start_probe(next_index)

    def start_practice(self) -> None:
        self.activate()

    def start_scored(self) -> None:
        self.activate()

    def restart_current_probe(self, *, seed_override: int | None = None) -> None:
        if self._stage is not BenchmarkStage.PROBE:
            return
        if not (0 <= self._current_probe_index < len(self._probe_seeds)):
            return
        self._probe_seeds[self._current_probe_index] = (
            _random_seed() if seed_override is None else _clamp_seed(seed_override)
        )
        self._current_engine = None
        self._latest_probe_result = None
        self._start_probe(self._current_probe_index)

    def submit_answer(self, raw: str) -> bool:
        engine = self._current_engine
        if self._stage is not BenchmarkStage.PROBE or engine is None:
            return False
        submit = getattr(engine, "submit_answer", None)
        if not callable(submit):
            return False
        return bool(submit(raw))

    def _submit_current_probe_skip(self, tokens: tuple[str, ...]) -> bool:
        engine = self._current_engine
        if self._stage is not BenchmarkStage.PROBE or engine is None:
            return False
        submit = getattr(engine, "submit_answer", None)
        if not callable(submit):
            return False
        for token in tokens:
            if submit(token):
                return True
        return False

    def debug_skip_probe(self) -> None:
        engine = self._current_engine
        if self._stage is not BenchmarkStage.PROBE or engine is None:
            return
        if not self._submit_current_probe_skip(
            ("__skip_section__", "skip_section", "__skip_all__", "skip_all")
        ):
            finish = getattr(engine, "finish", None)
            if callable(finish):
                finish()
            elif hasattr(engine, "phase"):
                try:
                    engine.phase = Phase.RESULTS
                except Exception:
                    if hasattr(engine, "_phase"):
                        engine._phase = Phase.RESULTS
        self.sync_runtime()

    def update(self) -> None:
        if self._stage is not BenchmarkStage.PROBE or self._current_engine is None:
            return
        update = getattr(self._current_engine, "update", None)
        if callable(update):
            update()
        self.sync_runtime()

    def sync_runtime(self) -> None:
        if self._stage is not BenchmarkStage.PROBE or self._current_engine is None:
            return
        if getattr(self._current_engine, "phase", None) is not Phase.RESULTS:
            return
        self._complete_current_probe()

    def snapshot(self) -> BenchmarkSnapshot:
        completed_probe_results = tuple(self._completed_probe_results())
        summary = self.scored_summary()
        probe_total = len(self._plan.probes)

        if self._stage is BenchmarkStage.INTRO:
            return BenchmarkSnapshot(
                stage=self._stage,
                title=self._plan.title,
                subtitle="Benchmark Battery",
                prompt=self._plan.description,
                note_lines=self._intro_note_lines(),
                probe_index=0,
                probe_total=probe_total,
                current_probe_code=None,
                current_probe_label=None,
                current_probe_seed=None,
                current_probe_difficulty_level=None,
                probe_time_remaining_s=None,
                battery_time_remaining_s=self._plan.scored_duration_s,
                attempted_total=summary.attempted,
                correct_total=summary.correct,
                completed_probe_results=completed_probe_results,
            )

        if self._stage is BenchmarkStage.PROBE_RESULTS:
            probe = self.current_probe_plan()
            result = self._latest_probe_result
            label = "Current Probe" if probe is None else probe.label
            accuracy = 0.0 if result is None else result.accuracy * 100.0
            score_text = "n/a"
            if result is not None and result.score_ratio is not None:
                score_text = f"{result.score_ratio * 100.0:.1f}%"
            note_lines = (
                ()
                if result is None
                else (
                    f"Probe {self._current_probe_index + 1}/{probe_total}",
                    f"Seed: {result.seed}",
                    f"Attempted: {result.attempted}",
                    f"Correct: {result.correct}",
                    f"Accuracy: {accuracy:.1f}%",
                    f"Score ratio: {score_text}",
                )
            )
            return BenchmarkSnapshot(
                stage=self._stage,
                title=label,
                subtitle="Probe Results",
                prompt=(
                    "This probe is complete.\n"
                    "Press Enter to continue to the next probe."
                ),
                note_lines=note_lines,
                probe_index=self._current_probe_index + 1,
                probe_total=probe_total,
                current_probe_code=None if probe is None else probe.probe_code,
                current_probe_label=None if probe is None else probe.label,
                current_probe_seed=None if probe is None else probe.seed,
                current_probe_difficulty_level=(
                    None if probe is None else probe.difficulty_level
                ),
                probe_time_remaining_s=0.0,
                battery_time_remaining_s=self._battery_time_remaining_s(),
                attempted_total=summary.attempted,
                correct_total=summary.correct,
                completed_probe_results=completed_probe_results,
                latest_probe_result=result,
            )

        if self._stage is BenchmarkStage.RESULTS:
            return BenchmarkSnapshot(
                stage=self._stage,
                title=self._plan.title,
                subtitle="Battery Results",
                prompt=(
                    "Benchmark complete.\n"
                    f"Attempted: {summary.attempted}\n"
                    f"Correct: {summary.correct}\n"
                    f"Accuracy: {summary.accuracy * 100.0:.1f}%\n"
                    f"Score: {summary.total_score:.1f}/{summary.max_score:.1f}"
                ),
                note_lines=self._results_note_lines(summary),
                probe_index=probe_total,
                probe_total=probe_total,
                current_probe_code=None,
                current_probe_label=None,
                current_probe_seed=None,
                current_probe_difficulty_level=None,
                probe_time_remaining_s=0.0,
                battery_time_remaining_s=0.0,
                attempted_total=summary.attempted,
                correct_total=summary.correct,
                completed_probe_results=completed_probe_results,
            )

        current_probe = self.current_probe_plan()
        return BenchmarkSnapshot(
            stage=self._stage,
            title=self._plan.title,
            subtitle=(
                f"Probe {self._current_probe_index + 1}/{probe_total}: "
                f"{current_probe.label if current_probe is not None else ''}"
            ),
            prompt="",
            note_lines=(
                "Locked order and fixed level-5 difficulty.",
                "Probe seeds are randomized each benchmark run.",
                "Pause menu restart resets only the active probe.",
            ),
            probe_index=self._current_probe_index + 1,
            probe_total=probe_total,
            current_probe_code=None if current_probe is None else current_probe.probe_code,
            current_probe_label=None if current_probe is None else current_probe.label,
            current_probe_seed=None if current_probe is None else current_probe.seed,
            current_probe_difficulty_level=(
                None if current_probe is None else current_probe.difficulty_level
            ),
            probe_time_remaining_s=self._current_probe_time_remaining_s(),
            battery_time_remaining_s=self._battery_time_remaining_s(),
            attempted_total=summary.attempted,
            correct_total=summary.correct,
            completed_probe_results=completed_probe_results,
            latest_probe_result=self._latest_probe_result,
        )

    def scored_summary(self) -> BenchmarkSummary:
        probe_attempts = self._probe_attempts(include_partial=True)
        attempted = sum(result.attempted for _plan, result, _completed in probe_attempts)
        correct = sum(result.correct for _plan, result, _completed in probe_attempts)
        accuracy = 0.0 if attempted <= 0 else float(correct) / float(attempted)
        duration_s = sum(result.duration_s for _plan, result, _completed in probe_attempts)
        throughput_per_min = (
            0.0 if duration_s <= 0.0 else (float(attempted) / float(duration_s)) * 60.0
        )
        total_score = sum(
            float(result.total_score or 0.0) for _plan, result, _completed in probe_attempts
        )
        max_score = sum(
            float(result.max_score or 0.0) for _plan, result, _completed in probe_attempts
        )
        score_ratio = 0.0 if max_score <= 0.0 else total_score / max_score

        analytics = self._aggregate_analytics()
        available_plans = [plan for plan, _result, _completed in probe_attempts]
        difficulty_level_start = (
            None if not available_plans else available_plans[0].difficulty_level
        )
        difficulty_level_end = None if not available_plans else available_plans[-1].difficulty_level
        difficulty_change_count = 0
        if available_plans:
            difficulty_change_count = sum(
                1
                for previous, current in zip(available_plans, available_plans[1:], strict=False)
                if previous.difficulty_level != current.difficulty_level
            )

        return BenchmarkSummary(
            attempted=attempted,
            correct=correct,
            accuracy=float(accuracy),
            duration_s=float(duration_s),
            throughput_per_min=float(throughput_per_min),
            mean_response_time_s=(
                None if analytics.mean_rt_ms is None else float(analytics.mean_rt_ms) / 1000.0
            ),
            total_score=float(total_score),
            max_score=float(max_score),
            score_ratio=float(score_ratio),
            difficulty_level=5,
            difficulty_level_start=difficulty_level_start,
            difficulty_level_end=difficulty_level_end,
            difficulty_change_count=difficulty_change_count,
            probe_count=len(self._plan.probes),
            completed_probes=len(self._completed_attempts),
            benchmark_code=self._plan.code,
            benchmark_version=self._plan.version,
        )

    def result_metrics(self) -> dict[str, str]:
        metrics = {
            "benchmark.version": str(int(self._plan.version)),
            "benchmark.run_seed": str(int(self._run_seed)),
            "benchmark.probe_count": str(len(self._plan.probes)),
            "benchmark.completed_probes": str(len(self._completed_attempts)),
            "benchmark.total_duration_s": f"{self._plan.scored_duration_s:.6f}",
        }

        attempt_lookup: dict[str, tuple[AttemptResult, bool]] = {
            plan.probe_code: (result, completed)
            for plan, result, completed in self._probe_attempts(include_partial=True)
        }
        for index, probe in enumerate(self._plan.probes):
            prefix = f"probe.{probe.probe_code}."
            metrics[f"{prefix}label"] = probe.label
            metrics[f"{prefix}index"] = str(index)
            metrics[f"{prefix}seed"] = str(int(probe.seed))
            metrics[f"{prefix}difficulty_level"] = str(int(probe.difficulty_level))
            metrics[f"{prefix}duration_s"] = f"{float(probe.duration_s):.6f}"
            saved = attempt_lookup.get(probe.probe_code)
            metrics[f"{prefix}completed"] = "1" if saved and saved[1] else "0"
            if not saved:
                continue
            result, _completed = saved
            for key, value in result.metrics.items():
                metrics[f"{prefix}{key}"] = str(value)
        return metrics

    def events(self) -> list[TelemetryEvent]:
        events: list[TelemetryEvent] = []
        item_offset = 0

        for index, (probe, result, completed) in enumerate(
            self._probe_attempts(include_partial=True)
        ):
            offset_ms = self._probe_offset_ms(index)
            probe_extra = {
                "probe_code": probe.probe_code,
                "probe_index": index,
                "probe_seed": int(probe.seed),
                "difficulty_level": int(probe.difficulty_level),
            }
            events.append(
                TelemetryEvent(
                    family="benchmark",
                    kind="probe_started",
                    phase=Phase.SCORED.value,
                    seq=len(events),
                    item_index=None,
                    is_scored=False,
                    is_correct=None,
                    is_timeout=False,
                    response_time_ms=None,
                    score=None,
                    max_score=None,
                    difficulty_level=int(probe.difficulty_level),
                    occurred_at_ms=offset_ms,
                    prompt=probe.label,
                    extra=probe_extra,
                )
            )

            local_item_map = self._local_item_index_map(result=result, item_offset=item_offset)
            item_offset += len(local_item_map)
            last_occurred_ms = offset_ms
            for inner_event in result.events:
                local_extra = dict(inner_event.extra or {})
                local_extra.update(probe_extra)
                if inner_event.item_index is not None:
                    local_extra["inner_item_index"] = int(inner_event.item_index)
                occurred_at_ms = None
                if inner_event.occurred_at_ms is not None:
                    occurred_at_ms = offset_ms + int(inner_event.occurred_at_ms)
                    last_occurred_ms = max(last_occurred_ms, occurred_at_ms)
                mapped_item_index = None
                if inner_event.item_index is not None:
                    mapped_item_index = local_item_map.get(int(inner_event.item_index))
                events.append(
                    TelemetryEvent(
                        family=inner_event.family,
                        kind=inner_event.kind,
                        phase=inner_event.phase,
                        seq=len(events),
                        item_index=mapped_item_index,
                        is_scored=inner_event.is_scored,
                        is_correct=inner_event.is_correct,
                        is_timeout=inner_event.is_timeout,
                        response_time_ms=inner_event.response_time_ms,
                        score=inner_event.score,
                        max_score=inner_event.max_score,
                        difficulty_level=(
                            inner_event.difficulty_level
                            if inner_event.difficulty_level is not None
                            else int(probe.difficulty_level)
                        ),
                        occurred_at_ms=occurred_at_ms,
                        prompt=inner_event.prompt,
                        expected=inner_event.expected,
                        response=inner_event.response,
                        extra=local_extra or None,
                    )
                )

            if completed:
                completed_at_ms = max(
                    last_occurred_ms,
                    offset_ms + int(round(float(probe.duration_s) * 1000.0)),
                )
                extra = dict(probe_extra)
                extra["attempted"] = int(result.attempted)
                extra["accuracy"] = float(result.accuracy)
                events.append(
                    TelemetryEvent(
                        family="benchmark",
                        kind="probe_completed",
                        phase=Phase.SCORED.value,
                        seq=len(events),
                        item_index=None,
                        is_scored=False,
                        is_correct=None,
                        is_timeout=False,
                        response_time_ms=None,
                        score=result.total_score,
                        max_score=result.max_score,
                        difficulty_level=int(probe.difficulty_level),
                        occurred_at_ms=completed_at_ms,
                        prompt=probe.label,
                        extra=extra,
                    )
                )

        return events

    def _start_probe(self, index: int) -> None:
        self._current_probe_index = int(index)
        probe = self._plan.probes[self._current_probe_index]
        seed = int(self._probe_seeds[self._current_probe_index])
        engine = probe.builder(seed)
        engine._difficulty_code = str(probe.probe_code)
        engine._resolved_difficulty_context = build_resolved_difficulty_context(
            probe.probe_code,
            mode="fixed",
            launch_level=int(probe.difficulty_level),
            fixed_level=int(probe.difficulty_level),
            adaptive_enabled=False,
        )
        self._current_engine = _PreparedBenchmarkProbeRuntime(
            engine=engine,
            probe_label=probe.label,
        )
        self._latest_probe_result = None
        self._stage = BenchmarkStage.PROBE

    def _complete_current_probe(self) -> None:
        probe = self.current_probe_plan()
        engine = self._current_engine
        if probe is None or engine is None:
            return
        attempt = attempt_result_from_engine(engine, test_code=probe.probe_code)
        self._completed_attempts.append(attempt)
        self._latest_probe_result = BenchmarkProbeResult(
            probe_code=probe.probe_code,
            label=probe.label,
            seed=probe.seed,
            difficulty_level=probe.difficulty_level,
            duration_s=probe.duration_s,
            attempted=attempt.attempted,
            correct=attempt.correct,
            accuracy=attempt.accuracy,
            throughput_per_min=attempt.throughput_per_min,
            mean_rt_ms=attempt.mean_rt_ms,
            total_score=attempt.total_score,
            max_score=attempt.max_score,
            score_ratio=attempt.score_ratio,
            difficulty_level_start=attempt.difficulty_level_start,
            difficulty_level_end=attempt.difficulty_level_end,
            completed=True,
        )
        self._current_engine = None
        self._stage = BenchmarkStage.PROBE_RESULTS

    def _current_probe_snapshot(self) -> object | None:
        engine = self._current_engine
        getter = getattr(engine, "snapshot", None)
        if not callable(getter):
            return None
        try:
            return getter()
        except Exception:
            return None

    def _current_probe_time_remaining_s(self) -> float | None:
        if self._stage is not BenchmarkStage.PROBE:
            return 0.0 if self._stage is BenchmarkStage.RESULTS else None
        snap = self._current_probe_snapshot()
        value = getattr(snap, "time_remaining_s", None)
        try:
            if value is not None:
                return float(value)
        except Exception:
            pass
        probe = self.current_probe_plan()
        return None if probe is None else float(probe.duration_s)

    def _battery_time_remaining_s(self) -> float | None:
        if self._stage is BenchmarkStage.INTRO:
            return self._plan.scored_duration_s
        if self._stage is BenchmarkStage.RESULTS:
            return 0.0
        if self._stage is BenchmarkStage.PROBE_RESULTS:
            return float(
                sum(probe.duration_s for probe in self._plan.probes[self._current_probe_index + 1 :])
            )
        current_remaining = self._current_probe_time_remaining_s() or 0.0
        future = sum(
            probe.duration_s for probe in self._plan.probes[self._current_probe_index + 1 :]
        )
        return float(current_remaining + future)

    def _probe_attempts(
        self,
        *,
        include_partial: bool,
    ) -> list[tuple[BenchmarkProbePlan, AttemptResult, bool]]:
        attempts: list[tuple[BenchmarkProbePlan, AttemptResult, bool]] = [
            (probe, result, True)
            for probe, result in zip(self._plan.probes, self._completed_attempts, strict=False)
        ]
        if include_partial:
            partial = self._partial_attempt_result()
            probe = self.current_probe_plan()
            if partial is not None and probe is not None:
                attempts.append((probe, partial, False))
        return attempts

    def _partial_attempt_result(self) -> AttemptResult | None:
        probe = self.current_probe_plan()
        engine = self._current_engine
        if probe is None or engine is None:
            return None
        try:
            return attempt_result_from_engine(engine, test_code=probe.probe_code)
        except Exception:
            return None

    def _completed_probe_results(self) -> list[BenchmarkProbeResult]:
        results: list[BenchmarkProbeResult] = []
        for probe, attempt in zip(self._plan.probes, self._completed_attempts, strict=False):
            results.append(
                BenchmarkProbeResult(
                    probe_code=probe.probe_code,
                    label=probe.label,
                    seed=probe.seed,
                    difficulty_level=probe.difficulty_level,
                    duration_s=probe.duration_s,
                    attempted=attempt.attempted,
                    correct=attempt.correct,
                    accuracy=attempt.accuracy,
                    throughput_per_min=attempt.throughput_per_min,
                    mean_rt_ms=attempt.mean_rt_ms,
                    total_score=attempt.total_score,
                    max_score=attempt.max_score,
                    score_ratio=attempt.score_ratio,
                    difficulty_level_start=attempt.difficulty_level_start,
                    difficulty_level_end=attempt.difficulty_level_end,
                    completed=True,
                )
            )
        return results

    def _aggregate_analytics(self) -> TelemetryAnalytics:
        summary = self._base_summary_fields()
        return telemetry_analytics_from_events(
            self.events(),
            duration_s=summary["duration_s"],
            is_complete=self.phase is Phase.RESULTS,
            difficulty_level_start=summary["difficulty_level_start"],
            difficulty_level_end=summary["difficulty_level_end"],
            difficulty_change_count=summary["difficulty_change_count"],
        )

    def _base_summary_fields(self) -> dict[str, float | int | None]:
        probe_attempts = self._probe_attempts(include_partial=True)
        plans = [plan for plan, _result, _completed in probe_attempts]
        difficulty_level_start = None if not plans else plans[0].difficulty_level
        difficulty_level_end = None if not plans else plans[-1].difficulty_level
        difficulty_change_count = sum(
            1
            for previous, current in zip(plans, plans[1:], strict=False)
            if previous.difficulty_level != current.difficulty_level
        )
        return {
            "duration_s": sum(result.duration_s for _plan, result, _completed in probe_attempts),
            "difficulty_level_start": difficulty_level_start,
            "difficulty_level_end": difficulty_level_end,
            "difficulty_change_count": difficulty_change_count,
        }

    def _local_item_index_map(
        self,
        *,
        result: AttemptResult,
        item_offset: int,
    ) -> dict[int, int]:
        local_indices = sorted(
            {
                int(event.item_index)
                for event in result.events
                if event.is_scored and event.item_index is not None
            }
        )
        return {raw: item_offset + idx for idx, raw in enumerate(local_indices)}

    def _probe_offset_ms(self, index: int) -> int:
        total = sum(probe.duration_s for probe in self._plan.probes[:index])
        return int(round(float(total) * 1000.0))

    def _intro_note_lines(self) -> tuple[str, ...]:
        lines = [
            "Approximate scored time: 28 minutes.",
            f"Benchmark run seed: {self._run_seed}",
            "Probe order is fixed and per-probe difficulty stays locked at level 5.",
            "Probe seeds are randomized each run. Practice and adaptive difficulty stay disabled.",
        ]
        for idx, probe in enumerate(self._plan.probes, start=1):
            lines.append(
                f"{idx}. {probe.label} | seed {probe.seed} | "
                f"level {probe.difficulty_level} | {int(round(probe.duration_s))}s"
            )
        return tuple(lines)

    def _results_note_lines(self, summary: BenchmarkSummary) -> tuple[str, ...]:
        lines = [
            f"Completed probes: {len(self._completed_attempts)}/{len(self._plan.probes)}",
            f"Throughput: {summary.throughput_per_min:.1f}/min",
        ]
        if summary.mean_response_time_s is not None:
            lines.append(f"Mean RT: {summary.mean_response_time_s * 1000.0:.0f} ms")
        for probe in self._completed_probe_results():
            score = "n/a"
            if probe.score_ratio is not None:
                score = f"{probe.score_ratio * 100.0:.0f}%"
            detail = f"{probe.label}: {probe.correct}/{probe.attempted} correct, score {score}"
            saved = next(
                (
                    result
                    for plan, result, completed in self._probe_attempts(include_partial=False)
                    if completed and plan.probe_code == probe.probe_code
                ),
                None,
            )
            split_fragment = None if saved is None else split_half_note_fragment(saved.metrics)
            lines.append(detail if split_fragment is None else f"{detail} | {split_fragment}")
        return tuple(lines)


def _clamp_seed(value: int) -> int:
    return max(1, min((2**31) - 1, int(value)))


def _random_seed() -> int:
    return _clamp_seed(random.SystemRandom().randint(1, (2**31) - 1))


def _build_benchmark_probe_engine(
    *,
    clock: Clock,
    probe_code: str,
    seed: int,
    duration_s: float,
) -> object:
    difficulty = difficulty_ratio_for_level(probe_code, 5)
    if probe_code == "numerical_operations":
        return build_numerical_operations_test(
            clock=clock,
            seed=seed,
            difficulty=difficulty,
            config=NumericalOperationsConfig(
                scored_duration_s=duration_s,
                practice_questions=0,
            ),
        )
    if probe_code == "visual_search":
        return build_visual_search_test(
            clock=clock,
            seed=seed,
            difficulty=difficulty,
            config=VisualSearchConfig(
                scored_duration_s=duration_s,
                practice_questions=0,
            ),
        )
    if probe_code == "digit_recognition":
        return build_digit_recognition_test(
            clock=clock,
            seed=seed,
            difficulty=difficulty,
            practice=False,
            scored_duration_s=duration_s,
        )
    if probe_code == "angles_bearings_degrees":
        return build_angles_bearings_degrees_test(
            clock=clock,
            seed=seed,
            difficulty=difficulty,
            config=AnglesBearingsDegreesConfig(
                scored_duration_s=duration_s,
                practice_questions=0,
            ),
        )
    if probe_code == "sensory_motor_apparatus":
        return build_sensory_motor_apparatus_test(
            clock=clock,
            seed=seed,
            difficulty=difficulty,
            config=SensoryMotorApparatusConfig(
                practice_duration_s=0.0,
                scored_duration_s=duration_s,
            ),
        )
    if probe_code == "math_reasoning":
        return build_math_reasoning_test(
            clock=clock,
            seed=seed,
            difficulty=difficulty,
            config=MathReasoningConfig(
                scored_duration_s=duration_s,
                practice_questions=0,
            ),
        )
    if probe_code == "target_recognition":
        return build_target_recognition_test(
            clock=clock,
            seed=seed,
            difficulty=difficulty,
            config=TargetRecognitionConfig(
                scored_duration_s=duration_s,
                practice_questions=0,
            ),
        )
    if probe_code == "colours_letters_numbers":
        return build_colours_letters_numbers_test(
            clock=clock,
            seed=seed,
            difficulty=difficulty,
            practice=False,
            scored_duration_s=duration_s,
            config=ColoursLettersNumbersConfig(
                scored_duration_s=duration_s,
                practice_rounds=0,
            ),
        )
    if probe_code == "instrument_comprehension":
        return build_instrument_comprehension_test(
            clock=clock,
            seed=seed,
            difficulty=difficulty,
            config=InstrumentComprehensionConfig(
                scored_duration_s=duration_s,
                practice_questions=0,
            ),
        )
    if probe_code == "rapid_tracking":
        return build_rapid_tracking_test(
            clock=clock,
            seed=seed,
            difficulty=difficulty,
            config=RapidTrackingConfig(
                practice_duration_s=0.0,
                scored_duration_s=duration_s,
            ),
        )
    if probe_code == "airborne_numerical":
        return build_airborne_numerical_test(
            clock=clock,
            seed=seed,
            practice=False,
            difficulty=difficulty,
            scored_duration_s=duration_s,
        )
    if probe_code == "vigilance":
        return build_vigilance_test(
            clock=clock,
            seed=seed,
            difficulty=difficulty,
            config=VigilanceConfig(
                practice_duration_s=0.0,
                scored_duration_s=duration_s,
            ),
        )
    if probe_code == "auditory_capacity":
        return build_auditory_capacity_test(
            clock=clock,
            seed=seed,
            difficulty=difficulty,
            config=AuditoryCapacityConfig(
                seed=seed,
                practice_duration_s=0.0,
                scored_duration_s=duration_s,
                practice_enabled=False,
            ),
        )
    if probe_code == "spatial_integration":
        half = duration_s / 2.0
        return build_spatial_integration_test(
            clock=clock,
            seed=seed,
            difficulty=difficulty,
            config=SpatialIntegrationConfig(
                practice_scenes_per_part=0,
                static_scored_duration_s=half,
                aircraft_scored_duration_s=half,
                skip_practice_for_testing=True,
            ),
        )
    if probe_code == "table_reading":
        return build_table_reading_test(
            clock=clock,
            seed=seed,
            difficulty=difficulty,
            config=TableReadingConfig(
                scored_duration_s=duration_s,
                practice_questions=0,
            ),
        )
    if probe_code == "cognitive_updating":
        return build_cognitive_updating_test(
            clock=clock,
            seed=seed,
            difficulty=difficulty,
            config=CognitiveUpdatingConfig(
                scored_duration_s=duration_s,
                practice_questions=0,
            ),
        )
    if probe_code == "trace_test_1":
        return build_trace_test_1_test(
            clock=clock,
            seed=seed,
            difficulty=difficulty,
            config=TraceTest1Config(
                scored_duration_s=duration_s,
                practice_questions=0,
            ),
        )
    if probe_code == "system_logic":
        return build_system_logic_test(
            clock=clock,
            seed=seed,
            difficulty=difficulty,
            config=SystemLogicConfig(
                scored_duration_s=duration_s,
                practice_questions=0,
            ),
        )
    if probe_code == "situational_awareness":
        return build_situational_awareness_test(
            clock=clock,
            seed=seed,
            difficulty=difficulty,
            config=SituationalAwarenessConfig(
                scored_duration_s=duration_s,
                practice_scenarios=0,
                practice_scenario_duration_s=0.0,
                scored_scenario_duration_s=40.0,
            ),
        )
    if probe_code == "trace_test_2":
        return build_trace_test_2_test(
            clock=clock,
            seed=seed,
            difficulty=difficulty,
            config=TraceTest2Config(
                scored_duration_s=duration_s,
                practice_questions=0,
            ),
        )
    raise KeyError(f"unsupported benchmark probe: {probe_code}")


def build_benchmark_plan(*, clock: Clock) -> BenchmarkPlan:
    official_codes = tuple(test.test_code for test in OFFICIAL_GUIDE_TESTS)
    probe_order = (
        "numerical_operations",
        "visual_search",
        "digit_recognition",
        "angles_bearings_degrees",
        "sensory_motor_apparatus",
        "math_reasoning",
        "target_recognition",
        "colours_letters_numbers",
        "instrument_comprehension",
        "rapid_tracking",
        "airborne_numerical",
        "vigilance",
        "auditory_capacity",
        "spatial_integration",
        "table_reading",
        "cognitive_updating",
        "trace_test_1",
        "system_logic",
        "situational_awareness",
        "trace_test_2",
    )
    assert tuple(sorted(probe_order)) == tuple(sorted(official_codes))

    def _probe(
        *,
        probe_code: str,
        label: str,
        duration_s: float,
    ) -> BenchmarkProbePlan:
        seed = _random_seed()
        return BenchmarkProbePlan(
            probe_code=probe_code,
            label=label,
            builder=lambda resolved_seed, probe_code=probe_code, duration_s=duration_s: _build_benchmark_probe_engine(
                clock=clock,
                probe_code=probe_code,
                seed=int(resolved_seed),
                duration_s=float(duration_s),
            ),
            seed=seed,
            difficulty_level=5,
            duration_s=float(duration_s),
        )

    return BenchmarkPlan(
        code="benchmark_battery",
        title="Benchmark Battery (~28m)",
        version=3,
        description=(
            "A 20-probe benchmark battery covering every official guide test once with "
            "balanced domain time and locked level-5 runtime settings."
        ),
        notes=(
            "The benchmark keeps a locked probe order and level-5 difficulty.",
            "Fresh randomized seeds are generated for every new benchmark run.",
            "Practice is disabled on every probe.",
        ),
        probes=(
            _probe(
                probe_code="numerical_operations",
                label="Numerical Operations",
                duration_s=45.0,
            ),
            _probe(
                probe_code="visual_search",
                label="Visual Search",
                duration_s=60.0,
            ),
            _probe(
                probe_code="digit_recognition",
                label="Digit Recognition",
                duration_s=45.0,
            ),
            _probe(
                probe_code="angles_bearings_degrees",
                label="Angles, Bearings and Degrees",
                duration_s=45.0,
            ),
            _probe(
                probe_code="sensory_motor_apparatus",
                label="Sensory Motor Apparatus",
                duration_s=195.0,
            ),
            _probe(
                probe_code="math_reasoning",
                label="Mathematics Reasoning",
                duration_s=75.0,
            ),
            _probe(
                probe_code="target_recognition",
                label="Target Recognition",
                duration_s=120.0,
            ),
            _probe(
                probe_code="colours_letters_numbers",
                label="Colours, Letters and Numbers",
                duration_s=60.0,
            ),
            _probe(
                probe_code="instrument_comprehension",
                label="Instrument Comprehension",
                duration_s=90.0,
            ),
            _probe(
                probe_code="rapid_tracking",
                label="Rapid Tracking",
                duration_s=135.0,
            ),
            _probe(
                probe_code="airborne_numerical",
                label="Airborne Numerical",
                duration_s=135.0,
            ),
            _probe(
                probe_code="vigilance",
                label="Vigilance",
                duration_s=150.0,
            ),
            _probe(
                probe_code="auditory_capacity",
                label="Auditory Capacity",
                duration_s=60.0,
            ),
            _probe(
                probe_code="spatial_integration",
                label="Spatial Integration",
                duration_s=90.0,
            ),
            _probe(
                probe_code="table_reading",
                label="Table Reading",
                duration_s=45.0,
            ),
            _probe(
                probe_code="cognitive_updating",
                label="Cognitive Updating",
                duration_s=60.0,
            ),
            _probe(
                probe_code="trace_test_1",
                label="Trace Test 1",
                duration_s=45.0,
            ),
            _probe(
                probe_code="system_logic",
                label="System Logic",
                duration_s=45.0,
            ),
            _probe(
                probe_code="situational_awareness",
                label="Situational Awareness",
                duration_s=120.0,
            ),
            _probe(
                probe_code="trace_test_2",
                label="Trace Test 2",
                duration_s=45.0,
            ),
        ),
    )
