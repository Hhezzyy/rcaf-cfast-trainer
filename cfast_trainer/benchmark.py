from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from .adaptive_difficulty import build_resolved_difficulty_context, difficulty_ratio_for_level
from .ant_drills import AntDrillMode
from .clock import Clock
from .cognitive_core import Phase
from .cln_drills import ClnDrillConfig, build_cln_sequence_math_recall_drill
from .numerical_operations import NumericalOperationsConfig, build_numerical_operations_test
from .results import AttemptResult, attempt_result_from_engine
from .rt_drills import RtDrillConfig, build_rt_lock_anchor_drill
from .sl_drills import SlDrillConfig, build_sl_graph_rule_anchor_drill
from .table_reading import TableReadingConfig, build_table_reading_test
from .telemetry import TelemetryAnalytics, TelemetryEvent, telemetry_analytics_from_events
from .training_modes import split_half_note_fragment
from .visual_search import VisualSearchConfig, build_visual_search_test
@dataclass(frozen=True, slots=True)
class BenchmarkProbePlan:
    probe_code: str
    label: str
    builder: Callable[[], object]
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


class BenchmarkStage(str, Enum):
    INTRO = "intro"
    PROBE = "probe"
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
    benchmark_version: int = 1
    difficulty_policy: str = "fixed_per_probe"


class BenchmarkSession:
    _NOMINAL_DIFFICULTY = 0.5

    def __init__(self, *, plan: BenchmarkPlan) -> None:
        self._plan = plan
        self._stage = BenchmarkStage.INTRO
        self._current_probe_index = 0
        self._current_engine: object | None = None
        self._completed_attempts: list[AttemptResult] = []

    @property
    def seed(self) -> int:
        return int(self._plan.version)

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
            return Phase.SCORED
        return Phase.INSTRUCTIONS

    @property
    def stage(self) -> BenchmarkStage:
        return self._stage

    def can_exit(self) -> bool:
        return self._stage in (BenchmarkStage.INTRO, BenchmarkStage.RESULTS)

    def current_engine(self) -> object | None:
        return self._current_engine

    def current_probe_plan(self) -> BenchmarkProbePlan | None:
        if self._stage is not BenchmarkStage.PROBE:
            return None
        if not (0 <= self._current_probe_index < len(self._plan.probes)):
            return None
        return self._plan.probes[self._current_probe_index]

    def activate(self) -> None:
        if self._stage is not BenchmarkStage.INTRO:
            return
        if not self._plan.probes:
            self._stage = BenchmarkStage.RESULTS
            return
        self._start_probe(0)

    def start_practice(self) -> None:
        self.activate()

    def start_scored(self) -> None:
        self.activate()

    def submit_answer(self, raw: str) -> bool:
        engine = self._current_engine
        if self._stage is not BenchmarkStage.PROBE or engine is None:
            return False
        submit = getattr(engine, "submit_answer", None)
        if not callable(submit):
            return False
        return bool(submit(raw))

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
                subtitle="Fixed Benchmark Battery",
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
        probe_snap = self._current_probe_snapshot()
        return BenchmarkSnapshot(
            stage=self._stage,
            title=self._plan.title,
            subtitle=(
                f"Probe {self._current_probe_index + 1}/{probe_total}: "
                f"{current_probe.label if current_probe is not None else ''}"
            ),
            prompt="",
            note_lines=(
                "Fixed order, fixed seeds, fixed difficulty.",
                "Pause menu controls the whole battery, not the current probe.",
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
        )

    def scored_summary(self) -> BenchmarkSummary:
        probe_attempts = self._probe_attempts(include_partial=True)
        attempted = sum(result.attempted for _plan, result, _completed in probe_attempts)
        correct = sum(result.correct for _plan, result, _completed in probe_attempts)
        accuracy = 0.0 if attempted <= 0 else float(correct) / float(attempted)
        duration_s = sum(result.duration_s for _plan, result, _completed in probe_attempts)
        throughput_per_min = 0.0 if duration_s <= 0.0 else (float(attempted) / float(duration_s)) * 60.0
        total_score = sum(float(result.total_score or 0.0) for _plan, result, _completed in probe_attempts)
        max_score = sum(float(result.max_score or 0.0) for _plan, result, _completed in probe_attempts)
        score_ratio = 0.0 if max_score <= 0.0 else total_score / max_score

        analytics = self._aggregate_analytics()
        available_plans = [plan for plan, _result, _completed in probe_attempts]
        difficulty_level_start = None if not available_plans else available_plans[0].difficulty_level
        difficulty_level_end = None if not available_plans else available_plans[-1].difficulty_level
        difficulty_change_count = 0
        if available_plans:
            difficulty_change_count = sum(
                1
                for previous, current in zip(available_plans, available_plans[1:])
                if previous.difficulty_level != current.difficulty_level
            )

        return BenchmarkSummary(
            attempted=attempted,
            correct=correct,
            accuracy=float(accuracy),
            duration_s=float(duration_s),
            throughput_per_min=float(throughput_per_min),
            mean_response_time_s=(
                None
                if analytics.mean_rt_ms is None
                else float(analytics.mean_rt_ms) / 1000.0
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

        for index, (probe, result, completed) in enumerate(self._probe_attempts(include_partial=True)):
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
        engine = probe.builder()
        setattr(engine, "_difficulty_code", str(probe.probe_code))
        setattr(
            engine,
            "_resolved_difficulty_context",
            build_resolved_difficulty_context(
                probe.probe_code,
                mode="fixed",
                launch_level=int(probe.difficulty_level),
                fixed_level=int(probe.difficulty_level),
                adaptive_enabled=False,
            ),
        )
        self._current_engine = engine
        self._stage = BenchmarkStage.PROBE
        starter = getattr(engine, "start_scored", None)
        if callable(starter):
            starter()

    def _complete_current_probe(self) -> None:
        probe = self.current_probe_plan()
        engine = self._current_engine
        if probe is None or engine is None:
            return
        self._completed_attempts.append(
            attempt_result_from_engine(engine, test_code=probe.probe_code)
        )
        next_index = self._current_probe_index + 1
        self._current_engine = None
        if next_index >= len(self._plan.probes):
            self._stage = BenchmarkStage.RESULTS
            self._current_probe_index = len(self._plan.probes)
            return
        self._start_probe(next_index)

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
            for previous, current in zip(plans, plans[1:])
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
            "Approximate scored time: 13 minutes.",
            "Fixed order, fixed seeds, and fixed per-probe difficulty.",
            "No practice blocks, no adaptive difficulty, and no benchmark-time overrides.",
        ]
        for idx, probe in enumerate(self._plan.probes, start=1):
            lines.append(
                f"{idx}. {probe.label} | seed {probe.seed} | level {probe.difficulty_level} | {int(round(probe.duration_s))}s"
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


def build_benchmark_plan(*, clock: Clock) -> BenchmarkPlan:
    return BenchmarkPlan(
        code="benchmark_battery",
        title="Benchmark Battery (13m)",
        version=1,
        description=(
            "A short fixed benchmark battery covering arithmetic speed, cross-reference lookup, "
            "visual scanning, symbol/rule extraction, low-load tracking, and mild dual-task recall."
        ),
        notes=(
            "The benchmark uses a locked probe order and fixed seeds.",
            "Difficulty and test-seed overrides do not apply here.",
        ),
        probes=(
            BenchmarkProbePlan(
                probe_code="numerical_operations",
                label="Mental Arithmetic Speed",
                builder=lambda: build_numerical_operations_test(
                    clock=clock,
                    seed=1101,
                    difficulty=difficulty_ratio_for_level("numerical_operations", 5),
                    config=NumericalOperationsConfig(
                        scored_duration_s=120.0,
                        practice_questions=0,
                    ),
                ),
                seed=1101,
                difficulty_level=5,
                duration_s=120.0,
            ),
            BenchmarkProbePlan(
                probe_code="table_reading",
                label="Table Lookup / Cross-Reference Speed",
                builder=lambda: build_table_reading_test(
                    clock=clock,
                    seed=1201,
                    difficulty=difficulty_ratio_for_level("table_reading", 5),
                    config=TableReadingConfig(
                        scored_duration_s=120.0,
                        practice_questions=0,
                    ),
                ),
                seed=1201,
                difficulty_level=5,
                duration_s=120.0,
            ),
            BenchmarkProbePlan(
                probe_code="visual_search",
                label="Visual Target Scan Speed",
                builder=lambda: build_visual_search_test(
                    clock=clock,
                    seed=1301,
                    difficulty=difficulty_ratio_for_level("visual_search", 5),
                    config=VisualSearchConfig(
                        scored_duration_s=120.0,
                        practice_questions=0,
                    ),
                ),
                seed=1301,
                difficulty_level=5,
                duration_s=120.0,
            ),
            BenchmarkProbePlan(
                probe_code="sl_graph_rule_anchor",
                label="Symbol / Rule Extraction",
                builder=lambda: build_sl_graph_rule_anchor_drill(
                    clock=clock,
                    seed=1401,
                    difficulty=difficulty_ratio_for_level("sl_graph_rule_anchor", 4),
                    mode=AntDrillMode.BUILD,
                    config=SlDrillConfig(
                        practice_questions=0,
                        scored_duration_s=150.0,
                    ),
                ),
                seed=1401,
                difficulty_level=4,
                duration_s=150.0,
            ),
            BenchmarkProbePlan(
                probe_code="rt_lock_anchor",
                label="Tracking Stability Under Low Load",
                builder=lambda: build_rt_lock_anchor_drill(
                    clock=clock,
                    seed=1501,
                    difficulty=difficulty_ratio_for_level("rt_lock_anchor", 4),
                    mode=AntDrillMode.BUILD,
                    config=RtDrillConfig(scored_duration_s=120.0),
                ),
                seed=1501,
                difficulty_level=4,
                duration_s=120.0,
            ),
            BenchmarkProbePlan(
                probe_code="cln_sequence_math_recall",
                label="Dual-Task Recall Under Mild Load",
                builder=lambda: build_cln_sequence_math_recall_drill(
                    clock=clock,
                    seed=1601,
                    difficulty=difficulty_ratio_for_level("cln_sequence_math_recall", 4),
                    mode=AntDrillMode.BUILD,
                    config=ClnDrillConfig(
                        practice_rounds=0,
                        scored_duration_s=150.0,
                    ),
                ),
                seed=1601,
                difficulty_level=4,
                duration_s=150.0,
            ),
        ),
    )
