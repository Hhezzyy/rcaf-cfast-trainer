from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import TYPE_CHECKING

from .cognitive_core import Phase, TestSnapshot
from .results import AttemptResult, attempt_result_from_engine
from .telemetry import TelemetryEvent

if TYPE_CHECKING:
    from .clock import Clock


_MODE_BUILD = "build"
_MODE_TEMPO = "tempo"
_MODE_STRESS = "stress"
_MODE_FRESH = "fresh"
_MODE_PRESSURE = "pressure"
_MODE_FATIGUE_PROBE = "fatigue_probe"
_MODE_RECOVERY = "recovery"
_MANUAL_MODE_ORDER = (
    _MODE_FRESH,
    _MODE_BUILD,
    _MODE_TEMPO,
    _MODE_PRESSURE,
    _MODE_FATIGUE_PROBE,
    _MODE_RECOVERY,
    _MODE_STRESS,
)
_FRESH_EXCLUDE_TOKENS = (
    "mixed",
    "pressure",
    "stress",
    "family_run",
    "grouped",
    "full_",
    "domain_run",
    "switch_run",
    "air_speed_run",
    "family_switch",
    "tempo_sweep",
    "disturbance",
)
_FRESH_INCLUDE_TOKENS = (
    "_anchor",
    "_prime",
    "_preview",
    "_copy",
    "_calibration",
    "_probe",
    "_match",
)
_FRESH_EXPLICIT_CODES = {
    "numerical_operations",
    "visual_search",
    "table_reading",
    "cln_colour_lane",
    "cln_math_prime",
    "cln_sequence_math_recall",
    "ma_one_step_fluency",
    "ma_written_numerical_extraction",
    "ma_percentage_snap",
    "ma_rate_time_distance",
    "ma_fuel_endurance",
    "ma_mixed_conversion_caps",
    "sl_one_rule_identify",
    "sl_missing_step_complete",
    "sl_two_source_reconcile",
    "sl_rule_match",
    "sl_fast_reject",
    "tbl_single_lookup_anchor",
    "tbl_two_table_xref",
    "tbl_distractor_grid",
    "tbl_lookup_compute",
    "tbl_shrinking_cap_run",
    "dtb_tracking_recall",
    "dtb_tracking_command_filter",
    "dtb_tracking_filter_digit_report",
    "dtb_tracking_interference_recovery",
    "vs_multi_target_class_search",
    "sma_split_axis_control",
}


def mode_token(mode: object | None) -> str:
    if mode is None:
        return ""
    raw = getattr(mode, "value", mode)
    return str(raw).strip().lower()


def is_fatigue_probe_mode(mode: object | None) -> bool:
    return mode_token(mode) == _MODE_FATIGUE_PROBE


def is_pressure_mode(mode: object | None) -> bool:
    return mode_token(mode) == _MODE_PRESSURE


def supports_training_mode(test_code: str, mode: object | None) -> bool:
    token = mode_token(mode)
    if token in {
        _MODE_BUILD,
        _MODE_TEMPO,
        _MODE_STRESS,
        _MODE_PRESSURE,
        _MODE_FATIGUE_PROBE,
        _MODE_RECOVERY,
    }:
        return True
    if token != _MODE_FRESH:
        return False
    code = str(test_code).strip().lower()
    if code in _FRESH_EXPLICIT_CODES:
        return True
    if code.startswith(("ma_", "tbl_", "sl_", "dtb_")):
        return True
    if any(token in code for token in _FRESH_EXCLUDE_TOKENS):
        return False
    return any(token in code for token in _FRESH_INCLUDE_TOKENS)


def supported_manual_modes(test_code: str) -> tuple[object, ...]:
    from .ant_drills import AntDrillMode

    modes: list[object] = []
    for token in _MANUAL_MODE_ORDER:
        if not supports_training_mode(test_code, token):
            continue
        modes.append(AntDrillMode(token))
    return tuple(modes)


def fatigue_probe_segment_config(config: object, *, scored_duration_s: float) -> object:
    if not is_dataclass(config):
        return config
    updates: dict[str, object] = {}
    field_names = {field.name for field in fields(config)}
    if "scored_duration_s" in field_names:
        updates["scored_duration_s"] = float(scored_duration_s)
    for name in ("practice_questions", "practice_rounds", "practice_duration_s"):
        if name in field_names:
            updates[name] = 0
    if not updates:
        return config
    return replace(config, **updates)


@dataclass(frozen=True, slots=True)
class FatigueProbeConfig:
    baseline_duration_s: float = 180.0
    loader_duration_s: float = 240.0
    late_duration_s: float = 180.0
    loader_seed_offset: int = 7001

    @property
    def total_duration_s(self) -> float:
        return (
            float(self.baseline_duration_s)
            + float(self.loader_duration_s)
            + float(self.late_duration_s)
        )


def maybe_build_fatigue_probe_drill(
    *,
    mode: object,
    title_base: str,
    clock: Clock,
    seed: int,
    difficulty: float,
    build_segment: Callable[[str, int, float], object],
    config: FatigueProbeConfig | None = None,
) -> FatigueProbeDrill | None:
    if not is_fatigue_probe_mode(mode):
        return None
    resolved = config or FatigueProbeConfig()
    return FatigueProbeDrill(
        title=f"{title_base} (Fatigue Probe)",
        instructions=(
            title_base,
            "Mode: Fatigue Probe",
            "Run a matched baseline, a pressure loader, and a late repeat on the same drill.",
            "The baseline and late probe use the same seed and difficulty so degradation is comparable.",
            "Press Enter to stage the composite block.",
        ),
        seed=int(seed),
        difficulty=float(difficulty),
        clock=clock,
        build_segment=build_segment,
        config=resolved,
    )


class FatigueProbeDrill:
    def __init__(
        self,
        *,
        title: str,
        instructions: tuple[str, ...],
        seed: int,
        difficulty: float,
        clock: Clock,
        build_segment: Callable[[str, int, float], object],
        config: FatigueProbeConfig,
    ) -> None:
        self._title = str(title)
        self._instructions = tuple(str(line) for line in instructions)
        self._seed = int(seed)
        self._difficulty = float(difficulty)
        self._clock = clock
        self._build_segment = build_segment
        self._config = config
        self._phase = Phase.INSTRUCTIONS
        self._segment_names = ("baseline", "loader", "late")
        self._current_segment_index = -1
        self._current_engine: object | None = None
        self._completed_segments: list[tuple[str, AttemptResult]] = []

    def __getattr__(self, name: str):
        if self._current_engine is None:
            raise AttributeError(name)
        return getattr(self._current_engine, name)

    @property
    def phase(self) -> Phase:
        if self._phase is Phase.SCORED and self._current_engine is not None:
            return Phase.SCORED
        return self._phase

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def difficulty(self) -> float:
        return self._difficulty

    @property
    def practice_questions(self) -> int:
        return 0

    @property
    def scored_duration_s(self) -> float:
        return float(self._config.total_duration_s)

    def can_exit(self) -> bool:
        return self.phase in (Phase.INSTRUCTIONS, Phase.PRACTICE_DONE, Phase.RESULTS)

    def start_practice(self) -> None:
        if self._phase is Phase.INSTRUCTIONS:
            self._phase = Phase.PRACTICE_DONE

    def start_scored(self) -> None:
        if self._phase in (Phase.SCORED, Phase.RESULTS):
            return
        self._phase = Phase.SCORED
        self._completed_segments.clear()
        self._start_segment(0)

    def update(self) -> None:
        if self._phase is not Phase.SCORED or self._current_engine is None:
            return
        updater = getattr(self._current_engine, "update", None)
        if callable(updater):
            updater()
        self._maybe_advance_segment()

    def submit_answer(self, raw: str) -> bool:
        if self._phase is not Phase.SCORED or self._current_engine is None:
            return False
        submit = getattr(self._current_engine, "submit_answer", None)
        if not callable(submit):
            return False
        accepted = bool(submit(raw))
        self._maybe_advance_segment()
        return accepted

    def snapshot(self) -> TestSnapshot:
        if self._phase is Phase.INSTRUCTIONS:
            return TestSnapshot(
                title=self._title,
                phase=Phase.INSTRUCTIONS,
                prompt="\n".join(self._instructions),
                input_hint="Press Enter to continue",
                time_remaining_s=None,
                attempted_scored=0,
                correct_scored=0,
            )
        if self._phase is Phase.PRACTICE_DONE:
            return TestSnapshot(
                title=self._title,
                phase=Phase.PRACTICE_DONE,
                prompt=(
                    "Fatigue probe staged.\n"
                    "Baseline 3m -> loader 4m -> late repeat 3m.\n"
                    "Press Enter to start the scored composite block."
                ),
                input_hint="Press Enter to continue",
                time_remaining_s=None,
                attempted_scored=0,
                correct_scored=0,
            )
        if self._phase is Phase.RESULTS:
            summary = self.scored_summary()
            return TestSnapshot(
                title=self._title,
                phase=Phase.RESULTS,
                prompt=(
                    "Results\n"
                    f"Attempted: {summary.attempted}\n"
                    f"Correct: {summary.correct}\n"
                    f"Accuracy: {summary.accuracy * 100.0:.1f}%\n"
                    f"Score ratio: {summary.score_ratio * 100.0:.1f}%"
                ),
                input_hint="Press Enter to continue",
                time_remaining_s=0.0,
                attempted_scored=int(summary.attempted),
                correct_scored=int(summary.correct),
            )
        assert self._current_engine is not None
        inner = self._current_engine.snapshot()
        segment_label = self._segment_display_name(self._segment_name_for_index(self._current_segment_index))
        return TestSnapshot(
            title=self._title,
            phase=Phase.SCORED,
            prompt=str(inner.prompt),
            input_hint=str(inner.input_hint),
            time_remaining_s=self._total_time_remaining_s(),
            attempted_scored=int(inner.attempted_scored),
            correct_scored=int(inner.correct_scored),
            payload=inner.payload,
            practice_feedback=(
                f"{segment_label} | {inner.practice_feedback}"
                if inner.practice_feedback
                else segment_label
            ),
        )

    def events(self) -> list[TelemetryEvent]:
        events: list[TelemetryEvent] = []
        item_offset = 0
        for segment_name, result, completed in self._segment_attempts(include_partial=True):
            segment_start_ms = self._segment_offset_ms(segment_name)
            segment_extra = {"segment": segment_name}
            events.append(
                TelemetryEvent(
                    family="fatigue_probe",
                    kind=self._segment_started_kind(segment_name),
                    phase=Phase.SCORED.value,
                    seq=len(events),
                    item_index=None,
                    is_scored=False,
                    is_correct=None,
                    is_timeout=False,
                    response_time_ms=None,
                    score=None,
                    max_score=None,
                    difficulty_level=result.difficulty_level_start,
                    occurred_at_ms=segment_start_ms,
                    prompt=self._segment_display_name(segment_name),
                    extra=segment_extra,
                )
            )
            local_item_map = _local_item_index_map(result=result, item_offset=item_offset)
            item_offset += len(local_item_map)
            last_occurred_ms = segment_start_ms
            for inner_event in result.events:
                extra = dict(inner_event.extra or {})
                extra.update(segment_extra)
                if inner_event.item_index is not None:
                    extra["inner_item_index"] = int(inner_event.item_index)
                occurred_at_ms = None
                if inner_event.occurred_at_ms is not None:
                    occurred_at_ms = segment_start_ms + int(inner_event.occurred_at_ms)
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
                        difficulty_level=inner_event.difficulty_level,
                        occurred_at_ms=occurred_at_ms,
                        prompt=inner_event.prompt,
                        expected=inner_event.expected,
                        response=inner_event.response,
                        extra=extra or None,
                    )
                )
            if completed:
                completed_at_ms = max(
                    last_occurred_ms,
                    segment_start_ms + int(round(self._segment_duration_s(segment_name) * 1000.0)),
                )
                events.append(
                    TelemetryEvent(
                        family="fatigue_probe",
                        kind=self._segment_completed_kind(segment_name),
                        phase=Phase.SCORED.value,
                        seq=len(events),
                        item_index=None,
                        is_scored=False,
                        is_correct=None,
                        is_timeout=False,
                        response_time_ms=None,
                        score=result.total_score,
                        max_score=result.max_score,
                        difficulty_level=result.difficulty_level_end,
                        occurred_at_ms=completed_at_ms,
                        prompt=self._segment_display_name(segment_name),
                        extra=segment_extra,
                    )
                )
        return events

    def scored_summary(self):
        from .ant_drills import AntDrillAttemptSummary

        attempt_results = [result for _name, result, _completed in self._segment_attempts(include_partial=True)]
        attempted = sum(result.attempted for result in attempt_results)
        correct = sum(result.correct for result in attempt_results)
        total_score = sum(float(result.total_score or 0.0) for result in attempt_results)
        max_score = sum(float(result.max_score or 0.0) for result in attempt_results)
        duration_s = self._observed_duration_s()
        accuracy = 0.0 if attempted <= 0 else float(correct) / float(attempted)
        throughput_per_min = 0.0 if duration_s <= 0.0 else (float(attempted) / float(duration_s)) * 60.0
        correct_per_min = 0.0 if duration_s <= 0.0 else (float(correct) / float(duration_s)) * 60.0
        scored_events = [
            event
            for result in attempt_results
            for event in result.events
            if event.is_scored and event.response_time_ms is not None
        ]
        mean_rt_s = (
            None
            if not scored_events
            else sum(int(event.response_time_ms) for event in scored_events) / float(len(scored_events)) / 1000.0
        )
        timeouts = sum(
            1
            for result in attempt_results
            for event in result.events
            if event.is_scored and event.is_timeout
        )
        fixation_rate = 0.0 if attempted <= 0 else float(timeouts) / float(attempted)
        start_level = next((result.difficulty_level_start for result in attempt_results if result.difficulty_level_start is not None), None)
        end_level = next(
            (
                result.difficulty_level_end
                for result in reversed(attempt_results)
                if result.difficulty_level_end is not None
            ),
            start_level,
        )
        return AntDrillAttemptSummary(
            attempted=attempted,
            correct=correct,
            accuracy=accuracy,
            duration_s=duration_s,
            throughput_per_min=throughput_per_min,
            mean_response_time_s=mean_rt_s,
            total_score=float(total_score),
            max_score=float(max_score),
            score_ratio=0.0 if max_score <= 0.0 else float(total_score) / float(max_score),
            correct_per_min=correct_per_min,
            timeouts=timeouts,
            fixation_rate=fixation_rate,
            max_timeout_streak=max(
                (
                    int(result.metrics.get("longest_lapse_streak", "0") or 0)
                    for result in attempt_results
                ),
                default=0,
            ),
            mode=_MODE_FATIGUE_PROBE,
            difficulty_level=int(end_level or 1),
            difficulty_level_start=start_level,
            difficulty_level_end=end_level,
            difficulty_change_count=sum(
                int(result.metrics.get("difficulty_change_count", "0") or 0)
                for result in attempt_results
            ),
            adaptive_enabled=False,
            adaptive_window_size=0,
        )

    def result_metrics(self) -> dict[str, str]:
        metrics = {"training_mode": _MODE_FATIGUE_PROBE}
        segment_lookup = {
            segment_name: result
            for segment_name, result, _completed in self._segment_attempts(include_partial=True)
        }
        for segment_name in self._segment_names:
            result = segment_lookup.get(segment_name)
            if result is None:
                continue
            prefix = f"fatigue_probe.{segment_name}."
            for key, value in result.metrics.items():
                metrics[f"{prefix}{key}"] = str(value)
        baseline = segment_lookup.get("baseline")
        late = segment_lookup.get("late")
        if baseline is not None and late is not None:
            baseline_accuracy = _metric_float(baseline.metrics, "accuracy")
            late_accuracy = _metric_float(late.metrics, "accuracy")
            baseline_rt = _metric_float(baseline.metrics, "mean_rt_ms")
            late_rt = _metric_float(late.metrics, "mean_rt_ms")
            baseline_timeout = _metric_float(baseline.metrics, "timeout_rate")
            late_timeout = _metric_float(late.metrics, "timeout_rate")
            metrics["fatigue_probe.delta_accuracy"] = _fmt_metric_delta(
                None
                if baseline_accuracy is None or late_accuracy is None
                else late_accuracy - baseline_accuracy
            )
            metrics["fatigue_probe.delta_mean_rt_ms"] = _fmt_metric_delta(
                None if baseline_rt is None or late_rt is None else late_rt - baseline_rt
            )
            metrics["fatigue_probe.delta_timeout_rate"] = _fmt_metric_delta(
                None
                if baseline_timeout is None or late_timeout is None
                else late_timeout - baseline_timeout
            )
        return metrics

    def _start_segment(self, index: int) -> None:
        self._current_segment_index = int(index)
        segment_name = self._segment_name_for_index(index)
        self._current_engine = self._build_segment(
            self._segment_mode(segment_name),
            self._segment_seed(segment_name),
            self._segment_duration_s(segment_name),
        )
        _force_start_scored(self._current_engine)

    def _maybe_advance_segment(self) -> None:
        if self._phase is not Phase.SCORED or self._current_engine is None:
            return
        if _engine_phase(self._current_engine) is not Phase.RESULTS:
            return
        segment_name = self._segment_name_for_index(self._current_segment_index)
        self._completed_segments.append(
            (segment_name, attempt_result_from_engine(self._current_engine, test_code=segment_name))
        )
        if self._current_segment_index + 1 >= len(self._segment_names):
            self._current_engine = None
            self._phase = Phase.RESULTS
            return
        self._start_segment(self._current_segment_index + 1)

    def _segment_attempts(self, *, include_partial: bool) -> list[tuple[str, AttemptResult, bool]]:
        attempts: list[tuple[str, AttemptResult, bool]] = [
            (segment_name, result, True) for segment_name, result in self._completed_segments
        ]
        if include_partial and self._phase is Phase.SCORED and self._current_engine is not None:
            segment_name = self._segment_name_for_index(self._current_segment_index)
            attempts.append(
                (
                    segment_name,
                    attempt_result_from_engine(self._current_engine, test_code=segment_name),
                    False,
                )
            )
        return attempts

    def _observed_duration_s(self) -> float:
        events = self.events()
        if not events:
            return 0.0
        observed_ms = max(
            (int(event.occurred_at_ms) for event in events if event.occurred_at_ms is not None),
            default=0,
        )
        if self._phase is Phase.RESULTS:
            return float(self.scored_duration_s)
        return float(observed_ms) / 1000.0

    def _total_time_remaining_s(self) -> float | None:
        if self._phase is not Phase.SCORED:
            return None
        consumed = 0.0
        for segment_name, _result in self._completed_segments:
            consumed += self._segment_duration_s(segment_name)
        if self._current_engine is not None:
            consumed += max(
                0.0,
                self._segment_duration_s(self._segment_name_for_index(self._current_segment_index))
                - float(_engine_time_remaining_s(self._current_engine) or 0.0),
            )
        return max(0.0, float(self.scored_duration_s) - consumed)

    def _segment_duration_s(self, segment_name: str) -> float:
        if segment_name == "baseline":
            return float(self._config.baseline_duration_s)
        if segment_name == "loader":
            return float(self._config.loader_duration_s)
        return float(self._config.late_duration_s)

    def _segment_offset_ms(self, segment_name: str) -> int:
        if segment_name == "baseline":
            return 0
        if segment_name == "loader":
            return int(round(self._config.baseline_duration_s * 1000.0))
        return int(round((self._config.baseline_duration_s + self._config.loader_duration_s) * 1000.0))

    def _segment_seed(self, segment_name: str) -> int:
        if segment_name == "loader":
            return int(self._seed + self._config.loader_seed_offset)
        return int(self._seed)

    def _segment_mode(self, segment_name: str) -> str:
        if segment_name == "loader":
            return _MODE_PRESSURE
        return _MODE_BUILD

    def _segment_name_for_index(self, index: int) -> str:
        return self._segment_names[max(0, min(len(self._segment_names) - 1, int(index)))]

    @staticmethod
    def _segment_display_name(segment_name: str) -> str:
        if segment_name == "baseline":
            return "Baseline"
        if segment_name == "loader":
            return "Pressure Loader"
        return "Late Repeat"

    @staticmethod
    def _segment_started_kind(segment_name: str) -> str:
        if segment_name == "baseline":
            return "baseline_started"
        if segment_name == "loader":
            return "loader_started"
        return "late_probe_started"

    @staticmethod
    def _segment_completed_kind(segment_name: str) -> str:
        if segment_name == "baseline":
            return "baseline_completed"
        if segment_name == "loader":
            return "loader_completed"
        return "late_probe_completed"


def split_half_note_fragment(metrics: Mapping[str, str], *, prefix: str = "") -> str | None:
    first_accuracy = _metric_float(metrics, f"{prefix}first_half_accuracy")
    second_accuracy = _metric_float(metrics, f"{prefix}second_half_accuracy")
    first_rt = _metric_float(metrics, f"{prefix}first_half_mean_rt_ms")
    second_rt = _metric_float(metrics, f"{prefix}second_half_mean_rt_ms")
    if (
        first_accuracy is None
        and second_accuracy is None
        and first_rt is None
        and second_rt is None
    ):
        return None
    first_acc_text = "--" if first_accuracy is None else f"{first_accuracy * 100.0:.0f}%"
    second_acc_text = "--" if second_accuracy is None else f"{second_accuracy * 100.0:.0f}%"
    first_rt_text = "--" if first_rt is None else f"{first_rt:.0f} ms"
    second_rt_text = "--" if second_rt is None else f"{second_rt:.0f} ms"
    return f"1H {first_acc_text} / {first_rt_text} | 2H {second_acc_text} / {second_rt_text}"


def _engine_phase(engine: object) -> Phase | None:
    phase = getattr(engine, "phase", None)
    return phase if isinstance(phase, Phase) else None


def _engine_time_remaining_s(engine: object) -> float | None:
    getter = getattr(engine, "time_remaining_s", None)
    if callable(getter):
        try:
            value = getter()
        except Exception:
            return None
    else:
        value = getattr(engine, "time_remaining_s", None)
    try:
        return None if value is None else float(value)
    except Exception:
        return None


def _force_start_scored(engine: object) -> None:
    starter = getattr(engine, "start_scored", None)
    if not callable(starter):
        return
    starter()
    if _engine_phase(engine) is Phase.SCORED:
        return
    practice_starter = getattr(engine, "start_practice", None)
    if callable(practice_starter):
        practice_starter()
    starter()


def _local_item_index_map(*, result: AttemptResult, item_offset: int) -> dict[int, int]:
    mapping: dict[int, int] = {}
    next_item = int(item_offset)
    for event in result.events:
        if event.item_index is None:
            continue
        inner_index = int(event.item_index)
        if inner_index in mapping:
            continue
        next_item += 1
        mapping[inner_index] = next_item
    return mapping


def _metric_float(metrics: Mapping[str, str], key: str) -> float | None:
    raw = metrics.get(key)
    if raw is None:
        return None
    token = str(raw).strip()
    if token == "":
        return None
    try:
        return float(token)
    except Exception:
        return None


def _fmt_metric_delta(value: float | None) -> str:
    return "" if value is None else f"{float(value):.6f}"
