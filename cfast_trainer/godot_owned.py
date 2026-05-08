from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

from .clock import Clock
from .cognitive_core import Phase, TestSnapshot
from .telemetry import TelemetryEvent


KIND_AUDITORY_CAPACITY = "auditory_capacity"
KIND_RAPID_TRACKING = "rapid_tracking"
KIND_SPATIAL_INTEGRATION = "spatial_integration"
KIND_TRACE_TEST_1 = "trace_test_1"
KIND_TRACE_TEST_2 = "trace_test_2"

GODOT_OWNED_KINDS = {
    KIND_AUDITORY_CAPACITY,
    KIND_RAPID_TRACKING,
    KIND_SPATIAL_INTEGRATION,
    KIND_TRACE_TEST_1,
    KIND_TRACE_TEST_2,
}

_DIRECT_KIND_BY_CODE = {
    "auditory_capacity": KIND_AUDITORY_CAPACITY,
    "rapid_tracking": KIND_RAPID_TRACKING,
    "spatial_integration": KIND_SPATIAL_INTEGRATION,
    "trace_test_1": KIND_TRACE_TEST_1,
    "trace_test_2": KIND_TRACE_TEST_2,
    "auditory_capacity_workout": KIND_AUDITORY_CAPACITY,
    "rapid_tracking_workout": KIND_RAPID_TRACKING,
    "spatial_integration_workout": KIND_SPATIAL_INTEGRATION,
    "trace_test_1_workout": KIND_TRACE_TEST_1,
    "trace_test_2_workout": KIND_TRACE_TEST_2,
}

_SI_STATIC_KINDS = (
    "landmark_grid",
    "scene_reconstruction",
)
_SI_AIRCRAFT_KINDS = (
    "aircraft_route_selection",
    "aircraft_continuation_selection",
    "aircraft_location_grid",
)
_SI_ALL_KINDS = _SI_STATIC_KINDS + _SI_AIRCRAFT_KINDS


def spatial_integration_godot_config(
    *,
    test_code: str,
    mode: str = "standard",
    duration_s: float | None = None,
    practice_scenes_per_part: int = 0,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Declarative Godot run config for Spatial Integration variants."""
    token = str(test_code or "").strip().lower()
    parts: tuple[str, ...]
    allowed: tuple[str, ...]
    if token in {"si_landmark_anchor"}:
        parts = ("static",)
        allowed = ("landmark_grid",)
    elif token in {"si_reconstruction_run"}:
        parts = ("static",)
        allowed = ("scene_reconstruction",)
    elif token in {"si_static_mixed_run", "si_static_multiview_integration"}:
        parts = ("static",)
        allowed = _SI_STATIC_KINDS
    elif token in {"si_route_anchor"}:
        parts = ("aircraft",)
        allowed = ("aircraft_route_selection",)
    elif token in {"si_continuation_prime"}:
        parts = ("aircraft",)
        allowed = ("aircraft_continuation_selection",)
    elif token in {"si_aircraft_grid_run"}:
        parts = ("aircraft",)
        allowed = ("aircraft_location_grid",)
    elif token in {"si_moving_aircraft_multiview_integration", "si_aircraft_multiview_integration"}:
        parts = ("aircraft",)
        allowed = _SI_AIRCRAFT_KINDS
    else:
        parts = ("static", "aircraft")
        allowed = _SI_ALL_KINDS

    config: dict[str, object] = {
        "spatial_integration": True,
        "parts": list(parts),
        "allowed_question_kinds": list(allowed),
        "static_study_s": 12.0,
        "aircraft_study_s": 15.0,
        "question_time_limit_s": 8.0,
        "practice_scenes_per_part": max(0, int(practice_scenes_per_part)),
        "grid_cols": 8,
        "grid_rows": 8,
        "alt_levels": 5,
        "mode": str(mode or "standard"),
    }
    if duration_s is not None:
        config["duration_s"] = max(1.0, float(duration_s))
    if extra:
        config.update(_as_dict(extra))
    return config


def godot_kind_for_test_code(test_code: str | None) -> str | None:
    token = str(test_code or "").strip().lower()
    if token in _DIRECT_KIND_BY_CODE:
        return _DIRECT_KIND_BY_CODE[token]
    if token.startswith("ac_"):
        return KIND_AUDITORY_CAPACITY
    if token.startswith("rt_"):
        return KIND_RAPID_TRACKING
    if token.startswith("si_"):
        return KIND_SPATIAL_INTEGRATION
    if token.startswith("tt1_"):
        return KIND_TRACE_TEST_1
    if token.startswith("tt2_"):
        return KIND_TRACE_TEST_2
    if token.startswith("trace_"):
        return KIND_TRACE_TEST_1
    return None


def default_title_for_godot_kind(kind: str, test_code: str) -> str:
    normalized = str(kind)
    if normalized == KIND_AUDITORY_CAPACITY:
        return "Auditory Capacity"
    if normalized == KIND_RAPID_TRACKING:
        return "Rapid Tracking"
    if normalized == KIND_SPATIAL_INTEGRATION:
        return "Spatial Integration"
    if normalized == KIND_TRACE_TEST_1:
        return "Trace Test 1"
    if normalized == KIND_TRACE_TEST_2:
        return "Trace Test 2"
    return str(test_code).replace("_", " ").title()


@dataclass(frozen=True, slots=True)
class GodotOwnedTestSpec:
    kind: str
    test_code: str
    title: str
    seed: int
    difficulty: float
    launch_id: str
    run_key: str
    phase: str
    duration_s: float
    practice_enabled: bool = False
    practice_duration_s: float = 0.0
    mode: str = "standard"
    config: dict[str, object] = field(default_factory=dict)
    audio: dict[str, object] = field(default_factory=dict)
    assets: dict[str, object] = field(default_factory=dict)
    result_metadata: dict[str, object] = field(default_factory=dict)

    def with_phase(self, phase: Phase | str) -> "GodotOwnedTestSpec":
        phase_token = str(getattr(phase, "value", phase)).strip().lower() or "unknown"
        return replace(
            self,
            phase=phase_token,
            run_key=f"{self.test_code}:{self.seed}:{self.mode}:{self.launch_id}:{phase_token}",
        )


@dataclass(frozen=True, slots=True)
class GodotOwnedPayload:
    spec: GodotOwnedTestSpec
    progress: dict[str, object] = field(default_factory=dict)
    error: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class GodotOwnedSummary:
    attempted: int
    correct: int
    accuracy: float
    duration_s: float
    throughput_per_min: float
    mean_response_time_s: float | None
    total_score: float = 0.0
    max_score: float = 0.0
    score_ratio: float = 0.0
    difficulty_level_start: int | None = None
    difficulty_level_end: int | None = None
    difficulty_change_count: int = 0


def _finite_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(number):
        return float(default)
    return float(number)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    number = _finite_float(value, default=float("nan"))
    if not math.isfinite(number):
        return None
    return float(number)


def _coerce_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _as_dict(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    return {}


def _as_list(value: object) -> list[object]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def _difficulty_level_from_ratio(value: object) -> int:
    ratio = max(0.0, min(1.0, _finite_float(value, 0.5)))
    return max(1, min(10, int(round(ratio * 9.0)) + 1))


def _summary_from_mapping(
    data: Mapping[str, object],
    *,
    fallback_duration_s: float,
    fallback_difficulty: float,
) -> GodotOwnedSummary:
    attempted = max(0, _coerce_int(data.get("attempted", data.get("attempts", 0))))
    correct = max(0, _coerce_int(data.get("correct", data.get("hits", 0))))
    accuracy = _finite_float(data.get("accuracy", 0.0 if attempted <= 0 else correct / attempted))
    duration_s = max(0.0, _finite_float(data.get("duration_s", fallback_duration_s)))
    throughput = _finite_float(
        data.get(
            "throughput_per_min",
            0.0 if duration_s <= 0.0 else (float(attempted) / duration_s) * 60.0,
        )
    )
    total_score = _finite_float(data.get("total_score", data.get("score", float(correct))))
    max_score = _finite_float(data.get("max_score", max(float(attempted), total_score, 0.0)))
    score_ratio = _finite_float(
        data.get("score_ratio", 0.0 if max_score <= 0.0 else total_score / max_score)
    )
    level = _difficulty_level_from_ratio(fallback_difficulty)
    return GodotOwnedSummary(
        attempted=attempted,
        correct=correct,
        accuracy=max(0.0, min(1.0, accuracy)),
        duration_s=duration_s,
        throughput_per_min=throughput,
        mean_response_time_s=_optional_float(data.get("mean_response_time_s")),
        total_score=total_score,
        max_score=max_score,
        score_ratio=max(0.0, min(1.0, score_ratio)),
        difficulty_level_start=_coerce_int(data.get("difficulty_level_start", level), level),
        difficulty_level_end=_coerce_int(data.get("difficulty_level_end", level), level),
        difficulty_change_count=max(0, _coerce_int(data.get("difficulty_change_count", 0))),
    )


def _event_from_mapping(raw: Mapping[str, object], *, seq: int, fallback_kind: str) -> TelemetryEvent:
    extra = raw.get("extra")
    if not isinstance(extra, Mapping):
        extra = {
            str(key): value
            for key, value in raw.items()
            if key
            not in {
                "family",
                "kind",
                "phase",
                "seq",
                "item_index",
                "is_scored",
                "is_correct",
                "is_timeout",
                "response_time_ms",
                "score",
                "max_score",
                "difficulty_level",
                "occurred_at_ms",
                "prompt",
                "expected",
                "response",
                "extra",
            }
        } or None
    response_time_ms = raw.get("response_time_ms")
    if response_time_ms is None and raw.get("response_time_s") is not None:
        response_time_ms = int(round(_finite_float(raw.get("response_time_s")) * 1000.0))
    occurred_at_ms = raw.get("occurred_at_ms")
    if occurred_at_ms is None and raw.get("occurred_at_s") is not None:
        occurred_at_ms = int(round(_finite_float(raw.get("occurred_at_s")) * 1000.0))
    return TelemetryEvent(
        family=str(raw.get("family", fallback_kind)),
        kind=str(raw.get("kind", "event")),
        phase=str(raw.get("phase", Phase.SCORED.value)),
        seq=_coerce_int(raw.get("seq", seq), seq),
        item_index=None if raw.get("item_index") is None else _coerce_int(raw.get("item_index")),
        is_scored=bool(
            raw.get("is_scored", str(raw.get("phase", Phase.SCORED.value)) == Phase.SCORED.value)
        ),
        is_correct=None if raw.get("is_correct") is None else bool(raw.get("is_correct")),
        is_timeout=bool(raw.get("is_timeout", False)),
        response_time_ms=None if response_time_ms is None else _coerce_int(response_time_ms),
        score=None if raw.get("score") is None else _finite_float(raw.get("score")),
        max_score=None if raw.get("max_score") is None else _finite_float(raw.get("max_score")),
        difficulty_level=(
            None if raw.get("difficulty_level") is None else _coerce_int(raw.get("difficulty_level"))
        ),
        occurred_at_ms=None if occurred_at_ms is None else _coerce_int(occurred_at_ms),
        prompt=None if raw.get("prompt") is None else str(raw.get("prompt")),
        expected=None if raw.get("expected") is None else str(raw.get("expected")),
        response=None if raw.get("response") is None else str(raw.get("response")),
        extra=dict(extra) if isinstance(extra, Mapping) and extra else None,
    )


class GodotOwnedTestEngine:
    """Thin Python shell for tests whose live runtime is authoritative in Godot."""

    def __init__(
        self,
        *,
        clock: Clock,
        kind: str,
        test_code: str,
        title: str,
        seed: int,
        difficulty: float,
        duration_s: float = 180.0,
        mode: str = "standard",
        practice_enabled: bool = False,
        practice_duration_s: float = 0.0,
        config: Mapping[str, object] | None = None,
        audio: Mapping[str, object] | None = None,
        assets: Mapping[str, object] | None = None,
        result_metadata: Mapping[str, object] | None = None,
    ) -> None:
        if kind not in GODOT_OWNED_KINDS:
            raise ValueError(f"unsupported Godot-owned test kind: {kind}")
        self._clock = clock
        self._kind = str(kind)
        self._test_code = str(test_code)
        self._title = str(title)
        self._seed = int(seed)
        self._difficulty = max(0.0, min(1.0, _finite_float(difficulty, 0.5)))
        self._scored_duration_s = max(1.0, _finite_float(duration_s, 180.0))
        self._practice_enabled = bool(practice_enabled)
        self._practice_duration_s = max(0.0, _finite_float(practice_duration_s, 0.0))
        self._mode = str(mode or "standard")
        self._launch_id = f"{id(self):x}"
        self._config = _as_dict(config or {})
        self._audio = _as_dict(audio or {})
        self._assets = _as_dict(assets or {})
        self._result_metadata = _as_dict(result_metadata or {})
        self._phase = Phase.INSTRUCTIONS
        self._started_at_s: float | None = None
        self._progress: dict[str, object] = {}
        self._events: list[TelemetryEvent] = []
        self._metrics: dict[str, object] = {
            "renderer_backend": "godot_4",
            "godot_authority": "1",
            "godot_kind": self._kind,
            "godot_test_code": self._test_code,
            "godot_mode": self._mode,
        }
        self._summary = _summary_from_mapping(
            {},
            fallback_duration_s=self._scored_duration_s,
            fallback_difficulty=self._difficulty,
        )
        self._error: dict[str, object] | None = None
        self._result_metrics_overrides: dict[str, str] = {}

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def difficulty(self) -> float:
        return self._difficulty

    @property
    def practice_questions(self) -> int:
        return 1 if self._practice_enabled else 0

    @property
    def scored_duration_s(self) -> float:
        return self._scored_duration_s

    @property
    def phase(self) -> Phase:
        return self._phase

    def can_exit(self) -> bool:
        return self._phase in (Phase.INSTRUCTIONS, Phase.PRACTICE_DONE, Phase.RESULTS)

    def start_practice(self) -> None:
        if self._phase is Phase.INSTRUCTIONS:
            self._phase = Phase.PRACTICE if self._practice_enabled else Phase.PRACTICE_DONE
            if self._phase is Phase.PRACTICE:
                self._started_at_s = float(self._clock.now())

    def start_scored(self) -> None:
        if self._phase is Phase.RESULTS:
            return
        self._phase = Phase.SCORED
        self._started_at_s = float(self._clock.now())
        self._progress.clear()
        self._error = None

    def submit_answer(self, raw: str) -> bool:
        token = str(raw).strip().lower()
        if token in {"__skip_section__", "__timeout__", "__force_complete__"}:
            self._complete_without_godot()
            return True
        if self._phase is Phase.INSTRUCTIONS:
            self.start_practice()
            return True
        if self._phase is Phase.PRACTICE:
            self._phase = Phase.PRACTICE_DONE
            return True
        if self._phase is Phase.PRACTICE_DONE:
            self.start_scored()
            return True
        return self._phase is Phase.RESULTS

    def update(self) -> None:
        return

    def snapshot(self) -> TestSnapshot:
        payload = self._payload()
        if self._phase is Phase.INSTRUCTIONS:
            return TestSnapshot(
                title=self._title,
                phase=Phase.INSTRUCTIONS,
                prompt=(
                    "This 3D test runs in the Godot window.\n"
                    "Press Enter to stage the Godot-owned runtime."
                ),
                input_hint="Press Enter to continue",
                time_remaining_s=None,
                attempted_scored=0,
                correct_scored=0,
                payload=payload,
            )
        if self._phase is Phase.PRACTICE:
            return TestSnapshot(
                title=self._title,
                phase=Phase.PRACTICE,
                prompt="Godot practice runtime active.",
                input_hint="Use the Godot window controls. Press Enter here to continue.",
                time_remaining_s=None,
                attempted_scored=0,
                correct_scored=0,
                payload=payload,
            )
        if self._phase is Phase.PRACTICE_DONE:
            return TestSnapshot(
                title=self._title,
                phase=Phase.PRACTICE_DONE,
                prompt="Godot runtime staged. Press Enter to start the scored block.",
                input_hint="Press Enter to continue",
                time_remaining_s=None,
                attempted_scored=0,
                correct_scored=0,
                payload=payload,
            )
        if self._phase is Phase.RESULTS:
            summary = self._summary
            detail = ""
            if self._error is not None:
                detail = "\nGodot error: " + str(
                    self._error.get("detail", self._error.get("reason", "unknown"))
                )
            return TestSnapshot(
                title=self._title,
                phase=Phase.RESULTS,
                prompt=(
                    "Results\n"
                    f"Attempted: {summary.attempted}\n"
                    f"Correct: {summary.correct}\n"
                    f"Accuracy: {summary.accuracy * 100.0:.1f}%\n"
                    f"Score ratio: {summary.score_ratio * 100.0:.1f}%"
                    f"{detail}"
                ),
                input_hint="Press Enter to continue",
                time_remaining_s=0.0,
                attempted_scored=summary.attempted,
                correct_scored=summary.correct,
                payload=payload,
            )
        elapsed = self._elapsed_s()
        remaining = max(0.0, self._scored_duration_s - elapsed)
        return TestSnapshot(
            title=self._title,
            phase=Phase.SCORED,
            prompt="Godot runtime active. Keep focus in the Godot window.",
            input_hint="Godot owns live controls. Esc opens pause.",
            time_remaining_s=remaining,
            attempted_scored=int(self._progress.get("attempted", self._summary.attempted)),
            correct_scored=int(self._progress.get("correct", self._summary.correct)),
            payload=payload,
        )

    def events(self) -> list[TelemetryEvent]:
        return list(self._events)

    def result_metrics(self) -> dict[str, object]:
        return dict(self._metrics)

    def scored_summary(self) -> GodotOwnedSummary:
        return self._summary

    def apply_godot_authoritative_message(self, message: Mapping[str, object]) -> None:
        command = str(message.get("command", "")).strip().lower()
        if command.startswith("auditory_"):
            command = "godot_" + command.removeprefix("auditory_")
        if command in {"ready", "progress", "event", "complete", "error"}:
            command = "godot_" + command
        if command == "godot_ready":
            self._metrics["godot_ready"] = "1"
            self._metrics["godot_run_key"] = str(message.get("run_key", ""))
            return
        if command == "godot_progress":
            self._progress.update(_as_dict(message.get("progress", message)))
            return
        if command == "godot_event":
            event_data = _as_dict(message.get("event", message))
            self._events.append(
                _event_from_mapping(event_data, seq=len(self._events), fallback_kind=self._kind)
            )
            return
        if command == "godot_complete":
            self._apply_completion(message)
            return
        if command == "godot_error":
            self._apply_error(message)

    def _payload(self) -> GodotOwnedPayload:
        return GodotOwnedPayload(
            spec=self._spec().with_phase(self._phase),
            progress=dict(self._progress),
            error=None if self._error is None else dict(self._error),
        )

    def _spec(self) -> GodotOwnedTestSpec:
        return GodotOwnedTestSpec(
            kind=self._kind,
            test_code=self._test_code,
            title=self._title,
            seed=self._seed,
            difficulty=self._difficulty,
            launch_id=self._launch_id,
            run_key=f"{self._test_code}:{self._seed}:{self._mode}:{self._launch_id}:{self._phase.value}",
            phase=self._phase.value,
            duration_s=self._scored_duration_s,
            practice_enabled=self._practice_enabled,
            practice_duration_s=self._practice_duration_s,
            mode=self._mode,
            config=dict(self._config),
            audio=dict(self._audio),
            assets=dict(self._assets),
            result_metadata=dict(self._result_metadata),
        )

    def _elapsed_s(self) -> float:
        if self._started_at_s is None:
            return 0.0
        return max(0.0, float(self._clock.now()) - float(self._started_at_s))

    def _apply_completion(self, message: Mapping[str, object]) -> None:
        result = _as_dict(message.get("result", message))
        summary_data = _as_dict(result.get("summary", message.get("summary", {})))
        metrics = _as_dict(result.get("metrics", message.get("metrics", {})))
        event_values = _as_list(result.get("events", message.get("events", [])))
        self._summary = _summary_from_mapping(
            summary_data,
            fallback_duration_s=self._scored_duration_s,
            fallback_difficulty=self._difficulty,
        )
        for key, value in metrics.items():
            self._metrics[str(key)] = value
        self._metrics["godot_complete"] = "1"
        self._metrics["godot_authority"] = "1"
        self._events = []
        for raw_event in event_values:
            if isinstance(raw_event, Mapping):
                self._events.append(
                    _event_from_mapping(raw_event, seq=len(self._events), fallback_kind=self._kind)
                )
        self._phase = Phase.RESULTS

    def _apply_error(self, message: Mapping[str, object]) -> None:
        self._error = {
            "reason": str(message.get("reason", "godot_error")),
            "detail": str(message.get("detail", message.get("error", "Godot runtime error"))),
        }
        self._metrics["godot_error"] = str(self._error["reason"])
        self._metrics["godot_error_detail"] = str(self._error["detail"])
        self._summary = _summary_from_mapping(
            {},
            fallback_duration_s=self._elapsed_s(),
            fallback_difficulty=self._difficulty,
        )
        self._phase = Phase.RESULTS

    def _complete_without_godot(self) -> None:
        self._summary = _summary_from_mapping(
            {
                "attempted": int(self._progress.get("attempted", 0) or 0),
                "correct": int(self._progress.get("correct", 0) or 0),
                "duration_s": self._elapsed_s() or self._scored_duration_s,
            },
            fallback_duration_s=self._elapsed_s() or self._scored_duration_s,
            fallback_difficulty=self._difficulty,
        )
        self._metrics["godot_complete"] = "0"
        self._metrics["manual_completion"] = "1"
        self._phase = Phase.RESULTS


def build_godot_owned_test(
    *,
    clock: Clock,
    seed: int,
    difficulty: float,
    kind: str | None = None,
    test_code: str,
    title: str | None = None,
    duration_s: float | None = None,
    mode: str = "standard",
    practice_enabled: bool = False,
    practice_duration_s: float = 0.0,
    config: Mapping[str, object] | None = None,
    audio: Mapping[str, object] | None = None,
    assets: Mapping[str, object] | None = None,
    result_metadata: Mapping[str, object] | None = None,
) -> GodotOwnedTestEngine:
    resolved_kind = kind or godot_kind_for_test_code(test_code)
    if resolved_kind is None:
        raise ValueError(f"test code is not Godot-owned: {test_code}")
    resolved_title = title or default_title_for_godot_kind(resolved_kind, test_code)
    return GodotOwnedTestEngine(
        clock=clock,
        kind=resolved_kind,
        test_code=test_code,
        title=resolved_title,
        seed=seed,
        difficulty=difficulty,
        duration_s=180.0 if duration_s is None else float(duration_s),
        mode=mode,
        practice_enabled=practice_enabled,
        practice_duration_s=practice_duration_s,
        config=config,
        audio=audio,
        assets=assets,
        result_metadata=result_metadata,
    )
