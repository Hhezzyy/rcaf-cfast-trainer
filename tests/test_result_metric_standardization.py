from __future__ import annotations

from dataclasses import dataclass

import pytest

from cfast_trainer.abd_drills import AbdDrillConfig, build_abd_angle_anchor_drill
from cfast_trainer.ac_drills import AcDrillConfig, build_ac_callsign_filter_run_drill
from cfast_trainer.adaptive_difficulty import (
    build_resolved_difficulty_context,
    difficulty_profile_for_code,
    difficulty_ratio_for_level,
)
from cfast_trainer.ant_drills import AntDrillMode
from cfast_trainer.auditory_capacity import AuditoryCapacityPayload
from cfast_trainer.benchmark import BenchmarkPlan, BenchmarkProbePlan, BenchmarkSession
from cfast_trainer.cognitive_core import Phase
from cfast_trainer.cognitive_core import TestSnapshot as SnapshotModel
from cfast_trainer.cognitive_updating import (
    CognitiveUpdatingConfig,
    CognitiveUpdatingPayload,
    CognitiveUpdatingRuntime,
    build_cognitive_updating_test,
)
from cfast_trainer.dr_drills import DigitRecognitionDrillConfig, build_dr_visible_copy_drill
from cfast_trainer.dtb_drills import (
    DualTaskBridgeDrillConfig,
    build_dtb_tracking_filter_digit_report_drill,
)
from cfast_trainer.angles_bearings_degrees import AnglesBearingsTrainingPayload
from cfast_trainer.numerical_operations import NumericalOperationsConfig, build_numerical_operations_test
from cfast_trainer.rapid_tracking import RapidTrackingPayload
from cfast_trainer.results import attempt_result_from_engine
from cfast_trainer.rt_drills import RtDrillConfig, build_rt_lock_anchor_drill
from cfast_trainer.telemetry import TelemetryEvent
from cfast_trainer.vs_drills import VsDrillConfig, build_vs_target_preview_drill

_COMMON_METRIC_KEYS = (
    "attempted",
    "correct",
    "score_ratio",
    "duration_s",
    "completed",
    "aborted",
    "mean_rt_ms",
    "median_rt_ms",
    "rt_variance_ms2",
    "timeout_count",
    "timeout_rate",
    "longest_lapse_streak",
    "first_half_accuracy",
    "second_half_accuracy",
    "first_half_mean_rt_ms",
    "second_half_mean_rt_ms",
    "first_3m_accuracy",
    "last_3m_accuracy",
    "first_3m_timeout_rate",
    "last_3m_timeout_rate",
    "post_error_next_item_rt_inflation_ms",
    "post_error_next_item_accuracy_drop",
    "difficulty_level_start",
    "difficulty_level_end",
    "difficulty_change_count",
)


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return float(self.t)

    def advance(self, dt: float) -> None:
        self.t += float(dt)


@dataclass(frozen=True, slots=True)
class _FakeProbeSummary:
    attempted: int
    correct: int
    accuracy: float
    duration_s: float
    throughput_per_min: float
    mean_response_time_s: float | None
    total_score: float
    max_score: float
    score_ratio: float
    difficulty_level: int
    difficulty_level_start: int
    difficulty_level_end: int
    difficulty_change_count: int = 0


class _FakeProbeEngine:
    def __init__(
        self,
        *,
        clock: FakeClock,
        title: str,
        seed: int,
        difficulty_level: int,
        scored_duration_s: float,
        attempted: int,
        correct: int,
    ) -> None:
        self._clock = clock
        self._title = str(title)
        self.seed = int(seed)
        self.difficulty = float(difficulty_level - 1) / 9.0
        self.practice_questions = 0
        self.scored_duration_s = float(scored_duration_s)
        self.phase = Phase.INSTRUCTIONS
        self._started_at_s: float | None = None
        self._attempted = int(attempted)
        self._correct = int(correct)
        self._difficulty_level = int(difficulty_level)
        self._events = self._build_events()

    def _build_events(self) -> list[TelemetryEvent]:
        events: list[TelemetryEvent] = []
        for index in range(self._attempted):
            is_correct = index < self._correct
            events.append(
                TelemetryEvent(
                    family="question",
                    kind="question",
                    phase=Phase.SCORED.value,
                    seq=index,
                    item_index=index + 1,
                    is_scored=True,
                    is_correct=is_correct,
                    is_timeout=False,
                    response_time_ms=550 + (index * 100),
                    score=1.0 if is_correct else 0.0,
                    max_score=1.0,
                    difficulty_level=self._difficulty_level,
                    occurred_at_ms=(index + 1) * 1000,
                    prompt=f"Q{index + 1}",
                    expected=str(index + 1),
                    response=str(index + 1 if is_correct else 0),
                )
            )
        return events

    def start_scored(self) -> None:
        self.phase = Phase.SCORED
        self._started_at_s = self._clock.now()

    def update(self) -> None:
        return

    def submit_answer(self, raw: str) -> bool:
        _ = raw
        return False

    def finish(self) -> None:
        self.phase = Phase.RESULTS

    def snapshot(self) -> SnapshotModel:
        remaining = None
        if self.phase is Phase.SCORED:
            started_at_s = 0.0 if self._started_at_s is None else self._started_at_s
            remaining = max(0.0, self.scored_duration_s - (self._clock.now() - started_at_s))
        attempted = self._attempted if self.phase is Phase.RESULTS else 0
        correct = self._correct if self.phase is Phase.RESULTS else 0
        return SnapshotModel(
            title=self._title,
            phase=self.phase,
            prompt=self._title,
            input_hint="",
            time_remaining_s=remaining,
            attempted_scored=attempted,
            correct_scored=correct,
            payload=None,
        )

    def scored_summary(self) -> _FakeProbeSummary:
        accuracy = 0.0 if self._attempted <= 0 else float(self._correct) / float(self._attempted)
        throughput = 0.0 if self.scored_duration_s <= 0.0 else (self._attempted / self.scored_duration_s) * 60.0
        return _FakeProbeSummary(
            attempted=self._attempted,
            correct=self._correct,
            accuracy=accuracy,
            duration_s=self.scored_duration_s,
            throughput_per_min=throughput,
            mean_response_time_s=0.65,
            total_score=float(self._correct),
            max_score=float(self._attempted),
            score_ratio=accuracy,
            difficulty_level=self._difficulty_level,
            difficulty_level_start=self._difficulty_level,
            difficulty_level_end=self._difficulty_level,
        )

    def events(self) -> list[TelemetryEvent]:
        return list(self._events)


def _metric_float(metrics: dict[str, str], key: str) -> float:
    return float(metrics[key])


def _set_fixed_context(engine: object, *, test_code: str, level: int) -> None:
    setattr(
        engine,
        "_resolved_difficulty_context",
        build_resolved_difficulty_context(
            test_code,
            mode="fixed",
            launch_level=level,
            fixed_level=level,
            adaptive_enabled=False,
        ),
    )


def _run_numerical_wrong_attempt(*, level: int = 6):
    clock = FakeClock()
    engine = build_numerical_operations_test(
        clock=clock,
        seed=101,
        difficulty=difficulty_ratio_for_level("numerical_operations", level),
        config=NumericalOperationsConfig(scored_duration_s=4.0, practice_questions=0),
    )
    _set_fixed_context(engine, test_code="numerical_operations", level=level)
    engine.start_scored()
    answer = int(engine._current.answer)
    wrong = answer + 7 if answer != 0 else 9
    assert engine.submit_answer(str(wrong)) is True
    clock.advance(4.0)
    engine.update()
    assert engine.phase is Phase.RESULTS
    return attempt_result_from_engine(engine, test_code="numerical_operations")


def _run_visible_copy_order_error_attempt(*, level: int = 5):
    clock = FakeClock()
    drill = build_dr_visible_copy_drill(
        clock=clock,
        seed=211,
        difficulty=difficulty_ratio_for_level("dr_visible_copy", level),
        mode=AntDrillMode.BUILD,
        config=DigitRecognitionDrillConfig(practice_questions=0, scored_duration_s=4.0),
    )
    _set_fixed_context(drill, test_code="dr_visible_copy", level=level)
    drill.start_practice()
    drill.start_scored()
    expected = str(drill._current.answer)
    wrong = expected[::-1]
    if wrong == expected:
        wrong = expected[1:] + expected[:1]
    assert drill.submit_answer(wrong) is True
    assert drill.submit_answer("__skip_section__") is True
    return attempt_result_from_engine(drill, test_code="dr_visible_copy")


def _run_cognitive_updating_revision_attempt(*, level: int = 5):
    clock = FakeClock()
    engine = build_cognitive_updating_test(
        clock=clock,
        seed=257,
        difficulty=difficulty_ratio_for_level("cognitive_updating", level),
        config=CognitiveUpdatingConfig(scored_duration_s=4.0, practice_questions=0),
    )
    _set_fixed_context(engine, test_code="cognitive_updating", level=level)
    engine.start_scored()
    payload = engine._current.payload
    assert isinstance(payload, CognitiveUpdatingPayload)
    runtime = CognitiveUpdatingRuntime(payload=payload, clock=clock)
    runtime.append_comms_digit("1")
    runtime.append_comms_digit("2")
    runtime.clear_comms()
    setattr(engine, "_telemetry_runtime_events", runtime.events())
    clock.advance(0.5)
    assert engine.submit_answer(str(engine._current.answer)) is True
    clock.advance(4.0)
    engine.update()
    assert engine.phase is Phase.RESULTS
    return attempt_result_from_engine(engine, test_code="cognitive_updating")


def _run_rt_attempt(*, level: int):
    clock = FakeClock()
    drill = build_rt_lock_anchor_drill(
        clock=clock,
        seed=307,
        difficulty=difficulty_ratio_for_level("rt_lock_anchor", level),
        mode=AntDrillMode.BUILD,
        config=RtDrillConfig(scored_duration_s=8.0),
    )
    _set_fixed_context(drill, test_code="rt_lock_anchor", level=level)
    drill.start_practice()
    controls = ((0.22, -0.12), (0.08, 0.0), (-0.16, 0.18), (0.0, 0.0))
    for idx in range(20):
        if drill.phase is not Phase.SCORED:
            break
        cx, cy = controls[idx % len(controls)]
        drill.set_control(horizontal=cx, vertical=cy)
        payload = drill.snapshot().payload
        assert isinstance(payload, RapidTrackingPayload)
        if payload.target_in_capture_box and idx % 3 == 0:
            assert drill.submit_answer("CAPTURE") is True
        clock.advance(0.5)
        drill.update()
    if drill.phase is Phase.SCORED:
        assert drill.submit_answer("__skip_section__") is True
    return attempt_result_from_engine(drill, test_code="rt_lock_anchor")


def _run_ac_attempt(*, level: int):
    clock = FakeClock()
    drill = build_ac_callsign_filter_run_drill(
        clock=clock,
        seed=409,
        difficulty=difficulty_ratio_for_level("ac_callsign_filter_run", level),
        mode=AntDrillMode.BUILD,
        config=AcDrillConfig(scored_duration_s=6.0),
    )
    _set_fixed_context(drill, test_code="ac_callsign_filter_run", level=level)
    drill.start_practice()

    remembered: dict[int, str] = {}
    handled_commands: set[int] = set()
    recalled: set[int] = set()
    beep_answered: set[int] = set()
    for _ in range(32):
        if drill.phase is not Phase.SCORED:
            break
        payload = drill.snapshot().payload
        assert isinstance(payload, AuditoryCapacityPayload)

        instruction_uid = payload.instruction_uid
        if instruction_uid is not None and instruction_uid not in handled_commands:
            if payload.color_command is not None:
                drill.set_colour(payload.color_command)
                handled_commands.add(instruction_uid)
            elif payload.number_command is not None:
                drill.set_number(payload.number_command)
                handled_commands.add(instruction_uid)
        if instruction_uid is not None and payload.sequence_display is not None:
            remembered[instruction_uid] = payload.sequence_display
        if (
            instruction_uid is not None
            and payload.sequence_response_open
            and instruction_uid not in recalled
            and instruction_uid in remembered
        ):
            assert drill.submit_answer(remembered[instruction_uid]) is True
            recalled.add(instruction_uid)
        if payload.beep_active and instruction_uid is not None and instruction_uid not in beep_answered:
            assert drill.submit_answer("SPACE") is True
            beep_answered.add(instruction_uid)

        target_y = payload.gates[0].y_norm if payload.gates else 0.0
        drill.set_control(
            horizontal=max(-1.0, min(1.0, -payload.ball_x * 2.0)),
            vertical=max(-1.0, min(1.0, (target_y - payload.ball_y) * 5.0)),
        )
        clock.advance(0.25)
        drill.update()
    if drill.phase is Phase.SCORED:
        assert drill.submit_answer("__skip_section__") is True
    return attempt_result_from_engine(drill, test_code="ac_callsign_filter_run")


def _run_dtb_attempt():
    clock = FakeClock()
    drill = build_dtb_tracking_filter_digit_report_drill(
        clock=clock,
        seed=503,
        difficulty=0.55,
        mode=AntDrillMode.TEMPO,
        config=DualTaskBridgeDrillConfig(scored_duration_s=42.0),
    )
    _set_fixed_context(drill, test_code="dtb_tracking_filter_digit_report", level=6)
    drill.start_scored()
    controls = ((0.18, -0.10), (0.06, 0.0), (-0.14, 0.16), (0.0, 0.0))
    for idx in range(96):
        if drill.phase is not Phase.SCORED:
            break
        cx, cy = controls[idx % len(controls)]
        drill.set_control(horizontal=cx, vertical=cy)
        payload = drill.snapshot().payload
        assert isinstance(payload, RapidTrackingPayload)
        status = drill.bridge_status()
        if status.channel == "command" and status.expected_response is not None:
            assert drill.submit_answer(f"CMD:{status.expected_response}") is True
        elif status.channel == "recall" and status.recall_input_active and status.expected_response:
            assert drill.submit_answer(f"DIGITS:{status.expected_response}") is True
        elif payload.target_in_capture_box and idx % 4 == 0:
            assert drill.submit_answer("CAPTURE") is True
        clock.advance(0.5)
        drill.update()
    if drill.phase is Phase.SCORED:
        drill.submit_answer("CAPTURE")
        clock.advance(42.0)
        drill.update()
    return attempt_result_from_engine(drill, test_code="dtb_tracking_filter_digit_report")


def _run_vs_attempt(*, level: int):
    clock = FakeClock()
    drill = build_vs_target_preview_drill(
        clock=clock,
        seed=607,
        difficulty=difficulty_ratio_for_level("vs_target_preview", level),
        mode=AntDrillMode.BUILD,
        config=VsDrillConfig(practice_questions=0, scored_duration_s=5.0),
    )
    _set_fixed_context(drill, test_code="vs_target_preview", level=level)
    drill.start_practice()
    drill.start_scored()
    assert drill.submit_answer(str(int(drill._current.answer))) is True
    assert drill.submit_answer("__skip_section__") is True
    return attempt_result_from_engine(drill, test_code="vs_target_preview")


def _run_abd_attempt(*, level: int):
    clock = FakeClock()
    drill = build_abd_angle_anchor_drill(
        clock=clock,
        seed=613,
        difficulty=difficulty_ratio_for_level("abd_angle_anchor", level),
        mode=AntDrillMode.BUILD,
        config=AbdDrillConfig(practice_questions=0, scored_duration_s=5.0),
    )
    _set_fixed_context(drill, test_code="abd_angle_anchor", level=level)
    drill.start_scored()
    payload = drill.snapshot().payload
    assert isinstance(payload, AnglesBearingsTrainingPayload)
    assert drill.submit_answer(str(int(drill._current.answer))) is True
    assert drill.submit_answer("__skip_section__") is True
    return attempt_result_from_engine(drill, test_code="abd_angle_anchor")


def _build_small_benchmark_plan(*, clock: FakeClock) -> BenchmarkPlan:
    def _probe(
        *,
        probe_code: str,
        label: str,
        seed: int,
        difficulty_level: int,
        duration_s: float,
        attempted: int,
        correct: int,
    ) -> BenchmarkProbePlan:
        return BenchmarkProbePlan(
            probe_code=probe_code,
            label=label,
            builder=lambda: _FakeProbeEngine(
                clock=clock,
                title=label,
                seed=seed,
                difficulty_level=difficulty_level,
                scored_duration_s=duration_s,
                attempted=attempted,
                correct=correct,
            ),
            seed=seed,
            difficulty_level=difficulty_level,
            duration_s=duration_s,
        )

    return BenchmarkPlan(
        code="benchmark_battery",
        title="Benchmark Battery",
        version=1,
        description="Small benchmark plan for result standardization tests.",
        notes=(),
        probes=(
            _probe(
                probe_code="visual_search",
                label="Visual Search Probe",
                seed=701,
                difficulty_level=3,
                duration_s=30.0,
                attempted=2,
                correct=1,
            ),
            _probe(
                probe_code="rt_lock_anchor",
                label="Rapid Tracking Probe",
                seed=702,
                difficulty_level=6,
                duration_s=30.0,
                attempted=3,
                correct=2,
            ),
        ),
    )


def test_arithmetic_text_input_result_emits_common_metrics_and_arithmetic_error_type() -> None:
    result = _run_numerical_wrong_attempt()

    for key in _COMMON_METRIC_KEYS:
        assert key in result.metrics
    assert result.metrics["completed"] == "1"
    assert result.metrics["aborted"] == "0"
    assert result.metrics["arithmetic_error_type"] != ""


def test_sequence_memory_result_emits_intrusion_omission_and_order_error_metrics() -> None:
    result = _run_visible_copy_order_error_attempt()

    assert result.metrics["intrusion_count"] != ""
    assert result.metrics["omission_count"] != ""
    assert result.metrics["order_error_count"] != ""


def test_cognitive_updating_result_emits_revision_count() -> None:
    result = _run_cognitive_updating_revision_attempt()

    assert result.metrics["difficulty_family_id"] == "visual_memory_updating"
    assert result.metrics["revision_count"] == "1"


def test_tracking_result_emits_rms_tracking_overshoot_and_reversal_metrics() -> None:
    result = _run_rt_attempt(level=6)

    assert result.metrics["rms_tracking_error"] != ""
    assert result.metrics["overshoot_count"] != ""
    assert result.metrics["reversal_count"] != ""


def test_command_multitask_result_emits_switch_cost_and_false_command_rate() -> None:
    result = _run_dtb_attempt()

    assert result.metrics["switch_cost_ms"] != ""
    assert result.metrics["false_command_rate"] != ""


def test_distractor_heavy_result_emits_distractor_capture_count() -> None:
    result = _run_ac_attempt(level=6)

    assert result.metrics["distractor_capture_count"] != ""


def test_benchmark_result_emits_composite_realized_difficulty_and_prefixed_probe_metrics() -> None:
    clock = FakeClock()
    session = BenchmarkSession(plan=_build_small_benchmark_plan(clock=clock))
    session.activate()
    current = session.current_engine()
    assert isinstance(current, _FakeProbeEngine)
    current.finish()
    session.sync_runtime()
    current = session.current_engine()
    assert isinstance(current, _FakeProbeEngine)
    current.finish()
    session.sync_runtime()

    result = attempt_result_from_engine(session, test_code="benchmark_battery")

    assert result.metrics["difficulty_family_id"] == "benchmark_battery"
    assert result.metrics["difficulty_profile_level_start"] == "3"
    assert result.metrics["difficulty_profile_level_end"] == "6"
    assert result.metrics["probe.visual_search.difficulty_profile_level"] != ""
    assert result.metrics["probe.rt_lock_anchor.difficulty_profile_level"] != ""
    assert result.metrics["probe.visual_search.difficulty_axis_distractor_density"] != ""
    assert result.metrics["probe.rt_lock_anchor.difficulty_axis_control_sensitivity"] != ""


@pytest.mark.parametrize(
    "runner",
    (
        lambda: _run_numerical_wrong_attempt(level=5),
        lambda: _run_vs_attempt(level=5),
        lambda: _run_visible_copy_order_error_attempt(level=5),
        lambda: _run_rt_attempt(level=5),
        lambda: _run_abd_attempt(level=5),
    ),
)
def test_representative_families_emit_all_required_difficulty_axis_metrics(runner) -> None:
    result = runner()

    assert result.metrics["difficulty_family_id"] != ""
    assert result.metrics["difficulty_profile_level"] != ""
    assert result.metrics["difficulty_profile_mode"] != ""
    for axis_name in (
        "content_complexity",
        "time_pressure",
        "distractor_density",
        "multitask_concurrency",
        "memory_span_delay",
        "switch_frequency",
        "control_sensitivity",
        "spatial_ambiguity",
        "source_integration_depth",
    ):
        assert result.metrics[f"difficulty_axis_{axis_name}"] != ""


@pytest.mark.parametrize(
    ("test_code", "axis_name", "runner"),
    (
        ("vs_target_preview", "distractor_density", _run_vs_attempt),
        ("rt_lock_anchor", "control_sensitivity", _run_rt_attempt),
        ("ac_callsign_filter_run", "multitask_concurrency", _run_ac_attempt),
        ("abd_angle_anchor", "spatial_ambiguity", _run_abd_attempt),
    ),
)
def test_changing_level_changes_realized_difficulty_metrics_for_representative_families(
    test_code: str,
    axis_name: str,
    runner,
) -> None:
    low_result = runner(level=2)
    high_result = runner(level=8)
    low_profile = difficulty_profile_for_code(test_code, 2, "build")
    high_profile = difficulty_profile_for_code(test_code, 8, "build")

    assert low_result.metrics["difficulty_profile_level"] == "2"
    assert high_result.metrics["difficulty_profile_level"] == "8"
    assert _metric_float(low_result.metrics, f"difficulty_axis_{axis_name}") == pytest.approx(
        getattr(low_profile.axes, axis_name),
        abs=1e-6,
    )
    assert _metric_float(high_result.metrics, f"difficulty_axis_{axis_name}") == pytest.approx(
        getattr(high_profile.axes, axis_name),
        abs=1e-6,
    )
    assert getattr(high_profile.axes, axis_name) > getattr(low_profile.axes, axis_name)
