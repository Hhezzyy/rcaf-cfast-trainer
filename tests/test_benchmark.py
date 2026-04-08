from __future__ import annotations

import os
import re
import sqlite3
from dataclasses import dataclass

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
import pytest

from cfast_trainer.adaptive_difficulty import difficulty_ratio_for_level
from cfast_trainer.app import (
    INTRO_LOADING_MIN_FRAMES,
    App,
    BenchmarkScreen,
    CognitiveTestScreen,
    DifficultySettingsStore,
    MenuItem,
    MenuScreen,
    TestSeedSettingsStore,
    run,
)
from cfast_trainer.benchmark import (
    BenchmarkPlan,
    BenchmarkProbePlan,
    BenchmarkSession,
    BenchmarkStage,
    build_benchmark_plan,
)
from cfast_trainer.cognitive_core import Phase
from cfast_trainer.cognitive_core import TestSnapshot as SnapshotModel
from cfast_trainer.persistence import ResultsStore
from cfast_trainer.results import attempt_result_from_engine
from cfast_trainer.sensory_motor_apparatus import (
    SensoryMotorApparatusConfig,
    build_sensory_motor_apparatus_test,
)
from cfast_trainer.target_recognition import (
    TargetRecognitionPayload,
    TargetRecognitionSceneEntity,
    TargetRecognitionSystemCycle,
)
from cfast_trainer.telemetry import TelemetryEvent


@dataclass
class _FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return float(self.t)

    def advance(self, dt: float) -> None:
        self.t += float(dt)


class _RecordingFont:
    def __init__(self, base: pygame.font.Font, sink: list[str]) -> None:
        self._base = base
        self._sink = sink

    def render(self, text: str, antialias: bool, color: object) -> pygame.Surface:
        self._sink.append(str(text))
        return self._base.render(text, antialias, color)

    def __getattr__(self, name: str) -> object:
        return getattr(self._base, name)


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
        clock: _FakeClock,
        title: str,
        seed: int,
        difficulty_level: int,
        scored_duration_s: float,
        attempted: int,
        correct: int,
        payload: object | None = None,
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
        self._payload = payload
        self._difficulty_level = int(difficulty_level)
        self._events = self._build_events()
        self.start_practice_calls = 0
        self.start_scored_calls = 0
        self.submit_calls: list[str] = []

    def _build_events(self) -> list[TelemetryEvent]:
        events: list[TelemetryEvent] = []
        for index in range(self._attempted):
            is_correct = index < self._correct
            rt_ms = 550 + (index * 125)
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
                    response_time_ms=rt_ms,
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

    def start_practice(self) -> None:
        self.start_practice_calls += 1
        if self.phase is Phase.INSTRUCTIONS:
            self.phase = Phase.PRACTICE_DONE

    def start_scored(self) -> None:
        self.start_scored_calls += 1
        if self.phase is Phase.INSTRUCTIONS:
            return
        self.phase = Phase.SCORED
        self._started_at_s = self._clock.now()

    def update(self) -> None:
        return

    def submit_answer(self, raw: str) -> bool:
        token = str(raw).strip().lower()
        self.submit_calls.append(str(raw))
        if token in {"__skip_section__", "skip_section", "__skip_all__", "skip_all"}:
            self.phase = Phase.RESULTS
            return True
        return True

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
            payload=self._payload,
        )

    def scored_summary(self) -> _FakeProbeSummary:
        attempted = self._attempted
        correct = self._correct
        accuracy = 0.0 if attempted <= 0 else float(correct) / float(attempted)
        mean_rt_s = None if attempted <= 0 else (550.0 + ((attempted - 1) * 62.5)) / 1000.0
        throughput = (
            0.0 if self.scored_duration_s <= 0.0 else (attempted / self.scored_duration_s) * 60.0
        )
        return _FakeProbeSummary(
            attempted=attempted,
            correct=correct,
            accuracy=accuracy,
            duration_s=self.scored_duration_s,
            throughput_per_min=throughput,
            mean_response_time_s=mean_rt_s,
            total_score=float(correct),
            max_score=float(attempted),
            score_ratio=accuracy,
            difficulty_level=self._difficulty_level,
            difficulty_level_start=self._difficulty_level,
            difficulty_level_end=self._difficulty_level,
        )

    def events(self) -> list[TelemetryEvent]:
        return list(self._events)


def _build_small_plan(
    *,
    clock: _FakeClock,
    created: list[_FakeProbeEngine] | None = None,
) -> BenchmarkPlan:
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
        def _builder(resolved_seed: int) -> _FakeProbeEngine:
            engine = _FakeProbeEngine(
                clock=clock,
                title=label,
                seed=resolved_seed,
                difficulty_level=difficulty_level,
                scored_duration_s=duration_s,
                attempted=attempted,
                correct=correct,
            )
            if created is not None:
                created.append(engine)
            return engine

        return BenchmarkProbePlan(
            probe_code=probe_code,
            label=label,
            builder=_builder,
            seed=seed,
            difficulty_level=difficulty_level,
            duration_s=duration_s,
        )

    return BenchmarkPlan(
        code="benchmark_battery",
        title="Benchmark Battery (13m)",
        version=1,
        description="Small fixed benchmark plan for tests.",
        notes=(),
        probes=(
            _probe(
                probe_code="alpha",
                label="Alpha Probe",
                seed=101,
                difficulty_level=5,
                duration_s=30.0,
                attempted=2,
                correct=1,
            ),
            _probe(
                probe_code="beta",
                label="Beta Probe",
                seed=202,
                difficulty_level=4,
                duration_s=30.0,
                attempted=2,
                correct=2,
            ),
        ),
    )


def _build_sma_benchmark_plan(*, clock: _FakeClock) -> BenchmarkPlan:
    return BenchmarkPlan(
        code="benchmark_battery",
        title="Benchmark Battery",
        version=1,
        description="Small SMA benchmark plan for runtime handoff tests.",
        notes=(),
        probes=(
            BenchmarkProbePlan(
                probe_code="sensory_motor_apparatus",
                label="Sensory Motor Apparatus",
                builder=lambda resolved_seed: build_sensory_motor_apparatus_test(
                    clock=clock,
                    seed=resolved_seed,
                    difficulty=0.5,
                    config=SensoryMotorApparatusConfig(
                        practice_duration_s=0.0,
                        scored_duration_s=2.0,
                        tick_hz=120.0,
                    ),
                ),
                seed=1501,
                difficulty_level=5,
                duration_s=2.0,
            ),
        ),
    )


def _build_target_recognition_payload(*, active_panels: tuple[str, ...]) -> TargetRecognitionPayload:
    return TargetRecognitionPayload(
        scene_rows=2,
        scene_cols=2,
        scene_cells=("TRK:F", "BLD:N", "TRK:H", "TNK:F"),
        scene_entities=(
            TargetRecognitionSceneEntity("truck", "friendly", False, False),
            TargetRecognitionSceneEntity("building", "neutral", False, False),
            TargetRecognitionSceneEntity("truck", "hostile", False, False),
            TargetRecognitionSceneEntity("tank", "friendly", False, False),
        ),
        scene_target="Friendly Truck",
        scene_has_target=True,
        scene_target_options=("Friendly Truck", "Hostile Truck", "Neutral Building", "Friendly Tank"),
        light_pattern=("G", "B", "R"),
        light_target_pattern=("G", "B", "R"),
        light_has_target=True,
        scan_tokens=("<>", "[]", "/\\", "()"),
        scan_target="<>",
        scan_has_target=True,
        system_rows=("A1B2", "C3D4", "E5F6"),
        system_target="A1B2",
        system_has_target=True,
        system_cycles=(
            TargetRecognitionSystemCycle(
                target="A1B2",
                columns=(
                    ("C3D4", "E5F6", "G7H8"),
                    ("A1B2", "J9K1", "L2M3"),
                    ("N4P5", "Q6R7", "S8T9"),
                ),
            ),
        ),
        system_step_interval_s=1.4,
        full_credit_error=0,
        zero_credit_error=3,
        active_panels=active_panels,
        light_interval_range_s=(4.5, 6.5),
        scan_interval_range_s=(4.2, 6.2),
        scan_repeat_range=(2, 3),
    )


def _build_target_recognition_benchmark_plan(
    *,
    clock: _FakeClock,
    payload: TargetRecognitionPayload,
    created: list[_FakeProbeEngine] | None = None,
) -> BenchmarkPlan:
    def _builder(resolved_seed: int) -> _FakeProbeEngine:
        engine = _FakeProbeEngine(
            clock=clock,
            title="Target Recognition",
            seed=resolved_seed,
            difficulty_level=5,
            scored_duration_s=30.0,
            attempted=0,
            correct=0,
            payload=payload,
        )
        if created is not None:
            created.append(engine)
        return engine

    return BenchmarkPlan(
        code="benchmark_battery",
        title="Benchmark Battery",
        version=1,
        description="Target Recognition benchmark plan for UI regressions.",
        notes=(),
        probes=(
            BenchmarkProbePlan(
                probe_code="target_recognition",
                label="Target Recognition",
                builder=_builder,
                seed=1701,
                difficulty_level=5,
                duration_s=30.0,
            ),
        ),
    )


def _install_recording_fonts(*fonts: object) -> list[str]:
    captured: list[str] = []
    for obj in fonts:
        for attr in ("_small_font", "_tiny_font", "_mid_font", "_big_font", "_num_header_font"):
            font = getattr(obj, attr, None)
            if isinstance(font, _RecordingFont) or font is None:
                continue
            setattr(obj, attr, _RecordingFont(font, captured))
    return captured


def _start_benchmark_probe(screen: BenchmarkScreen, surface: pygame.Surface) -> None:
    screen.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))
    screen.render(surface)
    for _ in range(INTRO_LOADING_MIN_FRAMES):
        screen.render(surface)
    screen.handle_event(
        pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": 0, "unicode": ""})
    )


def test_build_benchmark_plan_is_randomized_and_totals_1665_seconds() -> None:
    plan = build_benchmark_plan(clock=_FakeClock())
    second_plan = build_benchmark_plan(clock=_FakeClock())

    assert plan.code == "benchmark_battery"
    assert plan.title == "Benchmark Battery (~28m)"
    assert plan.version == 3
    assert tuple(probe.probe_code for probe in plan.probes) == (
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
    assert tuple(probe.seed for probe in plan.probes) != tuple(probe.seed for probe in second_plan.probes)
    assert all(1 <= int(probe.seed) <= (2**31) - 1 for probe in plan.probes)
    assert tuple(probe.difficulty_level for probe in plan.probes) == (5,) * 20
    assert tuple(int(probe.duration_s) for probe in plan.probes) == (
        45,
        60,
        45,
        45,
        195,
        75,
        120,
        60,
        90,
        135,
        135,
        150,
        60,
        90,
        45,
        60,
        45,
        45,
        120,
        45,
    )
    assert plan.scored_duration_s == pytest.approx(1665.0)


def test_benchmark_session_advances_probe_by_probe_and_accumulates_results() -> None:
    clock = _FakeClock()
    created: list[_FakeProbeEngine] = []
    session = BenchmarkSession(plan=_build_small_plan(clock=clock, created=created))

    intro = session.snapshot()
    assert intro.stage is BenchmarkStage.INTRO
    assert intro.battery_time_remaining_s == pytest.approx(60.0)

    session.activate()
    assert session.stage is BenchmarkStage.PROBE
    assert len(created) == 1
    assert session.current_engine() is not None
    assert getattr(session.current_engine(), "phase", None) is Phase.INSTRUCTIONS
    assert created[0].phase is Phase.INSTRUCTIONS
    assert created[0].start_practice_calls == 0
    assert created[0].start_scored_calls == 0

    created[0].finish()
    session.sync_runtime()
    assert session.stage is BenchmarkStage.PROBE_RESULTS
    assert len(created) == 1

    mid = session.snapshot()
    assert mid.probe_index == 1
    assert len(mid.completed_probe_results) == 1
    assert mid.completed_probe_results[0].probe_code == "alpha"

    session.continue_after_probe_results()
    assert session.stage is BenchmarkStage.PROBE
    assert len(created) == 2
    assert session.current_engine() is not None
    assert getattr(session.current_engine(), "phase", None) is Phase.INSTRUCTIONS
    assert created[1].phase is Phase.INSTRUCTIONS
    assert created[1].start_practice_calls == 0
    assert created[1].start_scored_calls == 0

    created[1].finish()
    session.sync_runtime()
    assert session.stage is BenchmarkStage.PROBE_RESULTS
    session.continue_after_probe_results()
    assert session.stage is BenchmarkStage.RESULTS
    assert session.current_engine() is None

    summary = session.scored_summary()
    assert summary.attempted == 4
    assert summary.correct == 3
    assert summary.completed_probes == 2
    assert summary.difficulty_level_start == 5
    assert summary.difficulty_level_end == 4
    assert summary.difficulty_change_count == 1

    results_snapshot = session.snapshot()
    assert results_snapshot.stage is BenchmarkStage.RESULTS
    assert any("1H " in line and "2H " in line for line in results_snapshot.note_lines)


def test_benchmark_attempt_persists_as_one_activity_with_prefixed_probe_metrics(tmp_path) -> None:
    clock = _FakeClock()
    created: list[_FakeProbeEngine] = []
    session = BenchmarkSession(plan=_build_small_plan(clock=clock, created=created))
    session.activate()
    created[0].finish()
    session.sync_runtime()
    session.continue_after_probe_results()
    created[1].finish()
    session.sync_runtime()
    session.continue_after_probe_results()

    result = attempt_result_from_engine(session, test_code="benchmark_battery")
    store = ResultsStore(tmp_path / "results.sqlite3")
    saved = store.record_attempt(result=result, app_version="test")

    with sqlite3.connect(store.path) as conn:
        attempt_rows = conn.execute(
            "SELECT COUNT(*), MIN(test_code), MAX(test_code) FROM attempt"
        ).fetchone()
        activity_count = conn.execute("SELECT COUNT(*) FROM activity_session").fetchone()
        metric_rows = dict(
            conn.execute(
                "SELECT key, value FROM attempt_metric WHERE attempt_id=?",
                (saved.attempt_id,),
            ).fetchall()
        )
        telemetry_kinds = [
            row[0]
            for row in conn.execute(
                "SELECT kind FROM telemetry_event WHERE activity_session_id=? ORDER BY seq",
                (saved.activity_session_id,),
            ).fetchall()
        ]

    assert attempt_rows == (1, "benchmark_battery", "benchmark_battery")
    assert activity_count == (1,)
    assert metric_rows["benchmark.version"] == "1"
    assert metric_rows["benchmark.probe_count"] == "2"
    assert metric_rows["probe.alpha.attempted"] == "2"
    assert metric_rows["probe.alpha.completed"] == "1"
    assert metric_rows["probe.beta.attempted"] == "2"
    assert metric_rows["probe.beta.completed"] == "1"
    assert telemetry_kinds.count("probe_started") == 2
    assert telemetry_kinds.count("probe_completed") == 2
    assert "activity_completed" in telemetry_kinds

    alpha_state = store.difficulty_state(scope_kind="code", scope_key="alpha")
    beta_state = store.difficulty_state(scope_kind="code", scope_key="beta")
    assert alpha_state is not None
    assert beta_state is not None
    assert alpha_state.last_end_level == 5
    assert beta_state.last_end_level == 4


def test_benchmark_launch_ignores_difficulty_and_seed_overrides(tmp_path) -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        difficulty_store = DifficultySettingsStore(tmp_path / "difficulty-settings.json")
        difficulty_store.set_global_override_enabled(True)
        difficulty_store.set_global_level(10)
        difficulty_store.set_test_level(test_code="numerical_operations", level=9)
        seed_store = TestSeedSettingsStore(tmp_path / "test-seeds.json")
        seed_store.set_rapid_tracking_seed_override_enabled(True)
        seed_store.set_rapid_tracking_seed_value(9999)
        app = App(
            surface=surface,
            font=font,
            difficulty_settings_store=difficulty_store,
            test_seed_settings_store=seed_store,
        )
        app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))
        clock = _FakeClock()
        session = BenchmarkSession(plan=build_benchmark_plan(clock=clock))
        screen = BenchmarkScreen(
            app,
            session=session,
            session_factory=lambda: BenchmarkSession(plan=build_benchmark_plan(clock=clock)),
        )
        app.push(screen)

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
        )
        screen.render(surface)

        engine = session.current_engine()
        assert engine is not None
        assert getattr(engine, "seed", None) != 9999
        assert 1 <= int(getattr(engine, "seed", 0) or 0) <= (2**31) - 1
        assert getattr(engine, "difficulty", None) == pytest.approx(
            difficulty_ratio_for_level("numerical_operations", 5)
        )
        assert getattr(engine, "phase", None) is Phase.INSTRUCTIONS
        assert session.phase is Phase.INSTRUCTIONS
    finally:
        pygame.quit()


def test_benchmark_screen_uses_nested_runtime_and_restart_resets_only_current_probe(tmp_path) -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        store = ResultsStore(tmp_path / "results.sqlite3")
        app = App(surface=surface, font=font, results_store=store)
        app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))

        clock = _FakeClock()

        def build_session() -> BenchmarkSession:
            return BenchmarkSession(plan=_build_small_plan(clock=clock))

        session = build_session()
        screen = BenchmarkScreen(app, session=session, session_factory=build_session)
        app.push(screen)

        _start_benchmark_probe(screen, surface)

        assert isinstance(screen._runtime_screen, CognitiveTestScreen)
        assert len(app._screens) == 2

        original_seed = session.current_probe_plan().seed

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "unicode": ""})
        )
        assert screen._pause_menu_active is True
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""})
        )
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""})
        )
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
        )

        assert app._screens[-1] is screen
        assert screen._session.stage is BenchmarkStage.PROBE
        assert screen._session.snapshot().current_probe_code == "alpha"
        assert screen._session.current_probe_plan().seed != original_seed

        session = store.session_summary()
        assert session is not None
        assert session.activity_count == 1
        assert session.aborted_activity_count == 0
        assert session.attempt_count == 0
    finally:
        pygame.quit()


def test_benchmark_pause_menu_shows_unified_actions_and_settings(tmp_path) -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        store = ResultsStore(tmp_path / "benchmark-settings.sqlite3")
        app = App(surface=surface, font=font, results_store=store)
        app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))

        clock = _FakeClock()
        screen = BenchmarkScreen(
            app,
            session=BenchmarkSession(plan=_build_small_plan(clock=clock)),
            session_factory=lambda: BenchmarkSession(plan=_build_small_plan(clock=clock)),
        )
        app.push(screen)
        _start_benchmark_probe(screen, surface)

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "unicode": ""})
        )

        assert screen._pause_menu_options() == (
            "Resume",
            "Skip Current Segment",
            "Restart Current",
            "Settings",
            "Main Menu",
        )

        settings_index = screen._pause_menu_options().index("Settings")
        for _ in range(settings_index):
            screen.handle_event(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""})
            )
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
        )

        assert [key for key, _label, _value in screen._pause_settings_rows()] == [
            "seed_mode",
            "seed_value",
            "review_mode",
            "joystick_bindings",
            "apply_restart",
            "back",
        ]
    finally:
        pygame.quit()


def test_benchmark_pause_menu_backspace_matches_escape() -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        app = App(surface=surface, font=font)
        app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))
        clock = _FakeClock()
        session = BenchmarkSession(plan=_build_small_plan(clock=clock))
        screen = BenchmarkScreen(
            app,
            session=session,
            session_factory=lambda: BenchmarkSession(plan=_build_small_plan(clock=clock)),
        )
        app.push(screen)
        _start_benchmark_probe(screen, surface)

        runtime = screen._runtime_screen
        assert isinstance(runtime, CognitiveTestScreen)

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_BACKSPACE, "unicode": ""})
        )

        assert screen._pause_menu_active is True
        assert runtime._pause_menu_active is False
    finally:
        pygame.quit()


def test_benchmark_pause_menu_skip_current_segment_advances_to_next_probe() -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        app = App(surface=surface, font=font)
        app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))
        clock = _FakeClock()
        session = BenchmarkSession(plan=_build_small_plan(clock=clock))
        screen = BenchmarkScreen(
            app,
            session=session,
            session_factory=lambda: BenchmarkSession(plan=_build_small_plan(clock=clock)),
        )
        app.push(screen)
        _start_benchmark_probe(screen, surface)

        assert session.snapshot().current_probe_code == "alpha"
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "unicode": ""})
        )
        skip_index = screen._pause_menu_options().index("Skip Current Segment")
        for _ in range(skip_index):
            screen.handle_event(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""})
            )
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
        )

        assert session.stage is BenchmarkStage.PROBE_RESULTS
        assert session.snapshot().current_probe_code == "alpha"
    finally:
        pygame.quit()


def test_benchmark_keypad_enter_starts_intro_and_continues_probe_results() -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        app = App(surface=surface, font=font)
        app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))
        clock = _FakeClock()
        session = BenchmarkSession(plan=_build_small_plan(clock=clock))
        screen = BenchmarkScreen(app, session=session)
        app.push(screen)

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_KP_ENTER, "unicode": ""})
        )

        assert session.stage is BenchmarkStage.PROBE

        engine = session.current_engine()
        assert engine is not None
        engine.finish()
        screen.render(surface)
        assert session.stage is BenchmarkStage.PROBE_RESULTS

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_KP_ENTER, "unicode": ""})
        )

        assert session.stage is BenchmarkStage.PROBE
        assert session.snapshot().probe_index == 2
    finally:
        pygame.quit()


def test_benchmark_escape_opens_pause_on_probe_results_and_final_results() -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        app = App(surface=surface, font=font)
        app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))
        clock = _FakeClock()
        session = BenchmarkSession(plan=_build_small_plan(clock=clock))
        screen = BenchmarkScreen(app, session=session)
        app.push(screen)

        _start_benchmark_probe(screen, surface)
        engine = session.current_engine()
        assert engine is not None
        engine.finish()
        screen.render(surface)
        assert session.stage is BenchmarkStage.PROBE_RESULTS

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "unicode": ""})
        )

        assert screen._pause_menu_active is True
        screen._pause_menu_hitboxes = {}
        screen.render(surface)
        assert screen._pause_menu_hitboxes

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "unicode": ""})
        )
        assert screen._pause_menu_active is False

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
        )
        engine = session.current_engine()
        assert engine is not None
        engine.finish()
        screen.render(surface)
        assert session.stage is BenchmarkStage.PROBE_RESULTS
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
        )
        screen.render(surface)
        assert session.stage is BenchmarkStage.RESULTS

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "unicode": ""})
        )

        assert screen._pause_menu_active is True
        screen._pause_menu_hitboxes = {}
        screen.render(surface)
        assert screen._pause_menu_hitboxes
    finally:
        pygame.quit()


def test_benchmark_pause_freezes_wrapped_probe_timer() -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        app = App(surface=surface, font=font)
        app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))
        clock = _FakeClock()
        created: list[_FakeProbeEngine] = []
        session = BenchmarkSession(plan=_build_small_plan(clock=clock, created=created))
        screen = BenchmarkScreen(app, session=session)
        app.push(screen)

        _start_benchmark_probe(screen, surface)
        screen.render(surface)

        clock.advance(5.0)
        screen.render(surface)
        before_pause = session.current_engine().snapshot().time_remaining_s
        assert before_pause is not None

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "unicode": ""})
        )
        paused_before = session.current_engine().snapshot().time_remaining_s

        clock.advance(15.0)
        screen.render(surface)
        paused_after = session.current_engine().snapshot().time_remaining_s

        assert screen._pause_menu_active is True
        assert paused_before == pytest.approx(before_pause)
        assert paused_after == pytest.approx(paused_before)
    finally:
        pygame.quit()


def test_benchmark_screen_hides_probe_overlay_during_normal_use() -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        app = App(surface=surface, font=font)
        app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))
        clock = _FakeClock()
        session = BenchmarkSession(plan=_build_small_plan(clock=clock))
        screen = BenchmarkScreen(
            app,
            session=session,
            session_factory=lambda: BenchmarkSession(plan=_build_small_plan(clock=clock)),
        )
        app.push(screen)
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
        )

        calls: list[str] = []
        screen._render_probe_overlay = lambda _surface, snap: calls.append(
            str(getattr(snap, "current_probe_code", ""))
        )

        screen.render(surface)

        assert session.stage is BenchmarkStage.PROBE
        assert calls == []
    finally:
        pygame.quit()


def test_benchmark_screen_shows_probe_overlay_in_dev_mode() -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        app = App(surface=surface, font=font)
        app._dev_tools_enabled = True
        app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))
        clock = _FakeClock()
        session = BenchmarkSession(plan=_build_small_plan(clock=clock))
        screen = BenchmarkScreen(
            app,
            session=session,
            session_factory=lambda: BenchmarkSession(plan=_build_small_plan(clock=clock)),
        )
        app.push(screen)
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
        )

        calls: list[str] = []
        screen._render_probe_overlay = lambda _surface, snap: calls.append(
            str(getattr(snap, "current_probe_code", ""))
        )

        screen.render(surface)

        assert session.stage is BenchmarkStage.PROBE
        assert calls == ["alpha"]
    finally:
        pygame.quit()


def test_benchmark_probe_overlay_hides_timer_text() -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        app = App(surface=surface, font=font)
        app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))
        clock = _FakeClock()
        session = BenchmarkSession(plan=_build_small_plan(clock=clock))
        screen = BenchmarkScreen(app, session=session)
        captured = _install_recording_fonts(screen)

        screen._render_probe_overlay(surface, session.snapshot())

        assert not any("Probe time" in text or "Battery remaining" in text for text in captured)
        assert not any(re.search(r"\b\d{2}:\d{2}\b", text) for text in captured)
    finally:
        pygame.quit()


def test_standard_runtime_screen_hides_timer_text() -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        app = App(surface=surface, font=font)
        app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))
        clock = _FakeClock()
        engine = _FakeProbeEngine(
            clock=clock,
            title="Mathematics Reasoning",
            seed=1601,
            difficulty_level=5,
            scored_duration_s=75.0,
            attempted=0,
            correct=0,
        )
        engine.phase = Phase.SCORED
        engine._started_at_s = clock.now()
        screen = CognitiveTestScreen(app, engine_factory=lambda: engine)
        captured = _install_recording_fonts(screen)

        screen.render(surface)

        assert not any("Time remaining" in text for text in captured)
        assert not any(re.search(r"\b\d{2}:\d{2}\b", text) for text in captured)
    finally:
        pygame.quit()


def test_benchmark_probe_waits_on_buffer_before_starting_timed_block(tmp_path) -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        app = App(surface=surface, font=font)
        app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))
        clock = _FakeClock()
        created: list[_FakeProbeEngine] = []
        session = BenchmarkSession(plan=_build_small_plan(clock=clock, created=created))
        screen = BenchmarkScreen(
            app,
            session=session,
            session_factory=lambda: BenchmarkSession(plan=_build_small_plan(clock=clock)),
        )
        app.push(screen)

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
        )
        screen.render(surface)

        assert getattr(session.current_engine(), "phase", None) is Phase.INSTRUCTIONS
        assert created[0].phase is Phase.INSTRUCTIONS
        assert created[0].start_scored_calls == 0
        runtime = screen._runtime_screen
        assert isinstance(runtime, CognitiveTestScreen)
        assert runtime._intro_loading_complete(Phase.INSTRUCTIONS) is False

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": 0, "unicode": ""})
        )
        assert getattr(session.current_engine(), "phase", None) is Phase.INSTRUCTIONS
        assert created[0].phase is Phase.INSTRUCTIONS
        assert created[0].start_scored_calls == 0

        for _ in range(INTRO_LOADING_MIN_FRAMES):
            screen.render(surface)

        assert runtime._intro_loading_complete(Phase.INSTRUCTIONS) is True
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": 0, "unicode": ""})
        )

        assert created[0].phase is Phase.SCORED
        assert getattr(session.current_engine(), "phase", None) is Phase.SCORED
        assert created[0].start_scored_calls >= 1
    finally:
        pygame.quit()


def test_benchmark_sma_probe_can_resume_second_timed_segment() -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        app = App(surface=surface, font=font)
        app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))
        clock = _FakeClock()
        session = BenchmarkSession(plan=_build_sma_benchmark_plan(clock=clock))
        screen = BenchmarkScreen(
            app,
            session=session,
            session_factory=lambda: BenchmarkSession(plan=_build_sma_benchmark_plan(clock=clock)),
        )
        app.push(screen)

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
        )
        screen.render(surface)
        for _ in range(INTRO_LOADING_MIN_FRAMES):
            screen.render(surface)
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": 0, "unicode": ""})
        )
        assert getattr(session.current_engine(), "phase", None) is Phase.SCORED

        for _ in range(11):
            clock.advance(0.1)
            screen.render(surface)

        runtime = screen._runtime_screen
        assert isinstance(runtime, CognitiveTestScreen)
        assert getattr(session.current_engine(), "phase", None) is Phase.PRACTICE_DONE
        for _ in range(INTRO_LOADING_MIN_FRAMES):
            screen.render(surface)
        assert runtime._intro_loading_complete(Phase.PRACTICE_DONE) is True

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": 0, "unicode": ""})
        )

        engine = session.current_engine()
        assert engine is not None
        assert getattr(engine, "phase", None) is Phase.SCORED
        snap = engine.snapshot()
        assert snap.payload is not None
        assert snap.payload.block_index == 2
        assert snap.payload.block_kind == "scored"
    finally:
        pygame.quit()


def test_benchmark_target_recognition_mouse_clicks_reach_runtime() -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        app = App(surface=surface, font=font)
        app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))
        clock = _FakeClock()
        payload = _build_target_recognition_payload(active_panels=("scene",))
        created: list[_FakeProbeEngine] = []
        session = BenchmarkSession(
            plan=_build_target_recognition_benchmark_plan(
                clock=clock,
                payload=payload,
                created=created,
            )
        )
        screen = BenchmarkScreen(app, session=session)
        app.push(screen)

        _start_benchmark_probe(screen, surface)
        screen.render(surface)
        clock.advance(2.0)
        screen.render(surface)

        runtime = screen._runtime_screen
        assert isinstance(runtime, CognitiveTestScreen)
        assert "Friendly Truck" in runtime._tr_scene_active_targets
        target_center = None
        for hit_rect, glyph_id in runtime._tr_scene_symbol_hitboxes:
            glyph = runtime._tr_scene_glyphs[glyph_id]
            if "Friendly Truck" in glyph.matching_labels:
                target_center = hit_rect.center
                break
        assert target_center is not None

        screen.handle_event(
            pygame.event.Event(
                pygame.MOUSEBUTTONDOWN,
                {"button": 1, "pos": target_center},
            )
        )

        assert created[0].submit_calls == ["1"]
    finally:
        pygame.quit()


def test_benchmark_target_recognition_streams_freeze_while_paused() -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        app = App(surface=surface, font=font)
        app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))
        clock = _FakeClock()
        payload = _build_target_recognition_payload(active_panels=("scene", "light", "scan", "system"))
        session = BenchmarkSession(
            plan=_build_target_recognition_benchmark_plan(
                clock=clock,
                payload=payload,
            )
        )
        screen = BenchmarkScreen(app, session=session)
        app.push(screen)

        _start_benchmark_probe(screen, surface)
        clock.advance(4.0)
        screen.render(surface)

        runtime = screen._runtime_screen
        assert isinstance(runtime, CognitiveTestScreen)
        before = {
            "scene_anim": float(runtime._tr_scene_anim_frame),
            "light_pattern": runtime._tr_light_current_pattern,
            "scan_pattern": runtime._tr_scan_current_pattern,
            "scan_reveal": int(runtime._tr_scan_reveal_index),
            "system_offset": int(runtime._tr_system_row_offset),
            "system_frac": float(runtime._tr_system_row_frac),
        }

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "unicode": ""})
        )
        clock.advance(12.0)
        screen.render(surface)

        after = {
            "scene_anim": float(runtime._tr_scene_anim_frame),
            "light_pattern": runtime._tr_light_current_pattern,
            "scan_pattern": runtime._tr_scan_current_pattern,
            "scan_reveal": int(runtime._tr_scan_reveal_index),
            "system_offset": int(runtime._tr_system_row_offset),
            "system_frac": float(runtime._tr_system_row_frac),
        }

        assert screen._pause_menu_active is True
        assert after["scene_anim"] == pytest.approx(before["scene_anim"], abs=1e-6)
        assert after["system_frac"] == pytest.approx(before["system_frac"], abs=1e-6)
        assert after["light_pattern"] == before["light_pattern"]
        assert after["scan_pattern"] == before["scan_pattern"]
        assert after["scan_reveal"] == before["scan_reveal"]
        assert after["system_offset"] == before["system_offset"]
    finally:
        pygame.quit()


def test_benchmark_segment_handoff_swaps_to_next_probe_buffer_in_same_frame() -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        app = App(surface=surface, font=font)
        app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))
        clock = _FakeClock()
        created: list[_FakeProbeEngine] = []
        session = BenchmarkSession(plan=_build_small_plan(clock=clock, created=created))
        screen = BenchmarkScreen(
            app,
            session=session,
            session_factory=lambda: BenchmarkSession(plan=_build_small_plan(clock=clock)),
        )
        app.push(screen)

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
        )
        screen.render(surface)
        for _ in range(INTRO_LOADING_MIN_FRAMES):
            screen.render(surface)
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "mod": 0, "unicode": ""})
        )
        assert created[0].phase is Phase.SCORED

        created[0].finish()
        screen.render(surface)

        assert session.stage is BenchmarkStage.PROBE_RESULTS
        assert session.snapshot().current_probe_code == "alpha"
        assert screen._runtime_engine_id is None

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
        )
        screen.render(surface)

        assert session.stage is BenchmarkStage.PROBE
        assert session.snapshot().current_probe_code == "beta"
        assert session.current_engine() is not None
        assert getattr(session.current_engine(), "phase", None) is Phase.INSTRUCTIONS
        assert created[1].phase is Phase.INSTRUCTIONS
        assert screen._runtime_engine_id == id(session.current_engine())
    finally:
        pygame.quit()


def test_benchmark_pause_menu_skip_does_not_persist_attempt(tmp_path) -> None:
    pygame.init()
    try:
        surface = pygame.display.set_mode((960, 540))
        font = pygame.font.Font(None, 36)
        store = ResultsStore(tmp_path / "benchmark-skip.sqlite3")
        app = App(surface=surface, font=font, results_store=store)
        app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))
        clock = _FakeClock()
        session = BenchmarkSession(plan=_build_small_plan(clock=clock))
        screen = BenchmarkScreen(
            app,
            session=session,
            session_factory=lambda: BenchmarkSession(plan=_build_small_plan(clock=clock)),
        )
        app.push(screen)
        _start_benchmark_probe(screen, surface)

        for _ in range(2):
            screen.handle_event(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "unicode": ""})
            )
            skip_index = screen._pause_menu_options().index("Skip Current Segment")
            for _ in range(skip_index):
                screen.handle_event(
                    pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""})
                )
            screen.handle_event(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )
            if session.stage is BenchmarkStage.PROBE_RESULTS:
                screen.handle_event(
                    pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
                )

        screen.render(surface)

        session_summary = store.session_summary()
        assert session_summary is not None
        assert session_summary.activity_count == 1
        assert session_summary.completed_activity_count == 0
        assert session_summary.aborted_activity_count == 1
        assert session_summary.attempt_count == 0
        assert screen._results_persistence_lines == ["Local save skipped in dev mode."]
    finally:
        pygame.quit()


def test_main_menu_places_benchmark_second_after_adaptive() -> None:
    def inject(frame: int) -> None:
        if frame == 1:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""})
            )
        elif frame == 2:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )

    assert run(max_frames=24, event_injector=inject) == 0


def test_tests_menu_places_benchmark_first() -> None:
    def inject(frame: int) -> None:
        if frame == 1:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""})
            )
        elif frame == 2:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""})
            )
        elif frame == 3:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )
        elif frame == 4:
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""})
            )

    assert run(max_frames=28, event_injector=inject) == 0
