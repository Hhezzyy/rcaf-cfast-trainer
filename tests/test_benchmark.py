from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
import pytest

from cfast_trainer.adaptive_difficulty import difficulty_ratio_for_level
from cfast_trainer.app import (
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
from cfast_trainer.telemetry import TelemetryEvent


@dataclass
class _FakeClock:
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
        clock: _FakeClock,
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

    def start_scored(self) -> None:
        self.phase = Phase.SCORED
        self._started_at_s = self._clock.now()

    def update(self) -> None:
        return

    def submit_answer(self, raw: str) -> bool:
        token = str(raw).strip().lower()
        if token in {"__skip_section__", "skip_section", "__skip_all__", "skip_all"}:
            self.phase = Phase.RESULTS
            return True
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
        attempted = self._attempted
        correct = self._correct
        accuracy = 0.0 if attempted <= 0 else float(correct) / float(attempted)
        mean_rt_s = None if attempted <= 0 else (550.0 + ((attempted - 1) * 62.5)) / 1000.0
        throughput = 0.0 if self.scored_duration_s <= 0.0 else (attempted / self.scored_duration_s) * 60.0
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
        def _builder() -> _FakeProbeEngine:
            engine = _FakeProbeEngine(
                clock=clock,
                title=label,
                seed=seed,
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


def test_build_benchmark_plan_is_fixed_and_totals_3600_seconds() -> None:
    plan = build_benchmark_plan(clock=_FakeClock())

    assert plan.code == "benchmark_battery"
    assert plan.title == "Benchmark Battery (~60m)"
    assert plan.version == 2
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
    assert tuple(probe.seed for probe in plan.probes) == (
        1101,
        1201,
        1301,
        1401,
        1501,
        1601,
        1701,
        1801,
        1901,
        2001,
        2101,
        2201,
        2301,
        2401,
        2501,
        2601,
        2701,
        2801,
        2901,
        3001,
    )
    assert tuple(probe.difficulty_level for probe in plan.probes) == (5,) * 20
    assert tuple(int(probe.duration_s) for probe in plan.probes) == (
        90,
        135,
        90,
        105,
        420,
        150,
        255,
        120,
        210,
        300,
        300,
        330,
        120,
        210,
        90,
        120,
        105,
        90,
        270,
        90,
    )
    assert plan.scored_duration_s == pytest.approx(3600.0)


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
    assert session.current_engine() is created[0]

    created[0].finish()
    session.sync_runtime()
    assert session.stage is BenchmarkStage.PROBE
    assert len(created) == 2
    assert session.current_engine() is created[1]

    mid = session.snapshot()
    assert mid.probe_index == 2
    assert len(mid.completed_probe_results) == 1
    assert mid.completed_probe_results[0].probe_code == "alpha"

    created[1].finish()
    session.sync_runtime()
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
    created[1].finish()
    session.sync_runtime()

    result = attempt_result_from_engine(session, test_code="benchmark_battery")
    store = ResultsStore(tmp_path / "results.sqlite3")
    saved = store.record_attempt(result=result, app_version="test")

    with sqlite3.connect(store.path) as conn:
        attempt_rows = conn.execute("SELECT COUNT(*), MIN(test_code), MAX(test_code) FROM attempt").fetchone()
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

        screen.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))
        screen.render(surface)

        engine = session.current_engine()
        assert engine is not None
        assert getattr(engine, "seed", None) == 1101
        assert getattr(engine, "difficulty", None) == pytest.approx(
            difficulty_ratio_for_level("numerical_operations", 5)
        )
    finally:
        pygame.quit()


def test_benchmark_screen_uses_nested_runtime_and_restart_is_battery_wide(tmp_path) -> None:
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

        screen = BenchmarkScreen(app, session=build_session(), session_factory=build_session)
        app.push(screen)

        screen.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))
        screen.render(surface)

        assert isinstance(screen._runtime_screen, CognitiveTestScreen)
        assert len(app._screens) == 2

        screen.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "unicode": ""}))
        assert screen._pause_menu_active is True
        screen.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        screen.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        screen.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))

        new_screen = app._screens[-1]
        assert isinstance(new_screen, BenchmarkScreen)
        assert new_screen is not screen
        assert new_screen._session.stage is BenchmarkStage.INTRO

        session = store.session_summary()
        assert session is not None
        assert session.activity_count == 2
        assert session.aborted_activity_count == 1
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
        screen.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))
        screen.render(surface)

        screen.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "unicode": ""}))

        assert screen._pause_menu_options() == (
            "Resume",
            "Skip Current Segment",
            "Restart Current",
            "Settings",
            "Main Menu",
        )

        settings_index = screen._pause_menu_options().index("Settings")
        for _ in range(settings_index):
            screen.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        screen.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))

        assert [key for key, _label, _value in screen._pause_settings_rows()] == [
            "review_mode",
            "joystick_bindings",
            "back",
        ]
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
        screen = BenchmarkScreen(app, session=session, session_factory=lambda: BenchmarkSession(plan=_build_small_plan(clock=clock)))
        app.push(screen)
        screen.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))
        screen.render(surface)

        assert session.snapshot().current_probe_code == "alpha"
        screen.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "unicode": ""}))
        skip_index = screen._pause_menu_options().index("Skip Current Segment")
        for _ in range(skip_index):
            screen.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""}))
        screen.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))

        assert session.stage is BenchmarkStage.PROBE
        assert session.snapshot().current_probe_code == "beta"
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
        screen = BenchmarkScreen(app, session=session, session_factory=lambda: BenchmarkSession(plan=_build_small_plan(clock=clock)))
        app.push(screen)
        screen.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN, "unicode": ""}))
        screen.render(surface)

        for _ in range(2):
            screen.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "unicode": ""}))
            skip_index = screen._pause_menu_options().index("Skip Current Segment")
            for _ in range(skip_index):
                screen.handle_event(
                    pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN, "unicode": ""})
                )
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
