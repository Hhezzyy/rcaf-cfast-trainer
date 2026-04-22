from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import pygame
import pytest

from cfast_trainer.adaptive_difficulty import difficulty_ratio_for_level
from cfast_trainer.app import App, AntWorkoutScreen, MenuItem, MenuScreen
from cfast_trainer.ant_drills import AntDrillMode
from cfast_trainer.ant_workouts import (
    AntWorkoutBlockPlan,
    AntWorkoutPlan,
    AntWorkoutSession,
    AntWorkoutStage,
    build_ant_workout_plan,
    build_workout_block_engine,
)
from cfast_trainer.persistence import ResultsStore
from cfast_trainer.results import attempt_result_from_engine


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _build_small_workout_plan() -> AntWorkoutPlan:
    return AntWorkoutPlan(
        code="airborne_numerical_workout",
        title="Airborne Numerical Workout Smoke",
        description="Short deterministic workout for tests.",
        notes=("Block setup is untimed.",),
        blocks=(
            AntWorkoutBlockPlan(
                block_id="snap",
                label="Snap Facts Block",
                description="Arithmetic under hard caps.",
                focus_skills=("Arithmetic retrieval",),
                drill_code="ant_snap_facts_sprint",
                mode=AntDrillMode.BUILD,
                duration_min=0.25,
            ),
            AntWorkoutBlockPlan(
                block_id="scenario",
                label="Scenario Pressure Block",
                description="Grouped scenario work under pressure.",
                focus_skills=("Full-question solving",),
                drill_code="airborne_scenario_pressure",
                mode=AntDrillMode.BUILD,
                duration_min=0.25,
            ),
        ),
    )


def _finish_current_block_with_one_correct_answer(
    session: AntWorkoutSession,
    clock: FakeClock,
) -> None:
    engine = session.current_engine()
    assert engine is not None
    answer = engine._current.answer
    assert session.submit_answer(str(answer)) is True
    remaining = engine.time_remaining_s()
    assert remaining is not None
    clock.advance(remaining + 0.1)
    session.update()


def _complete_small_workout(clock: FakeClock) -> AntWorkoutSession:
    session = AntWorkoutSession(
        clock=clock,
        seed=123,
        plan=_build_small_workout_plan(),
        starting_level=5,
    )

    assert session.stage is AntWorkoutStage.INTRO
    assert session.can_exit() is True
    session.adjust_starting_level(1)
    assert session.difficulty == pytest.approx(
        difficulty_ratio_for_level("airborne_numerical_workout", 6)
    )

    session.activate()
    assert session.stage is AntWorkoutStage.BLOCK_SETUP
    session.adjust_block_level(-1)
    first_setup = session.snapshot()
    assert first_setup.block_default_level == 6
    assert first_setup.block_override_level == 5
    session.activate()

    assert session.stage is AntWorkoutStage.BLOCK
    _finish_current_block_with_one_correct_answer(session, clock)
    assert session.stage is AntWorkoutStage.BLOCK_RESULTS
    session.activate()

    assert session.stage is AntWorkoutStage.BLOCK_SETUP
    second_setup = session.snapshot()
    assert second_setup.block_default_level == 6
    assert second_setup.block_override_level == 6
    session.activate()

    assert session.stage is AntWorkoutStage.BLOCK
    _finish_current_block_with_one_correct_answer(session, clock)
    assert session.stage is AntWorkoutStage.BLOCK_RESULTS
    session.activate()

    assert session.stage is AntWorkoutStage.RESULTS
    assert session.can_exit() is True
    return session


def _build_app_and_workout_screen(
    *,
    clock: FakeClock,
    session: AntWorkoutSession,
) -> tuple[App, AntWorkoutScreen]:
    pygame.init()
    surface = pygame.display.set_mode((960, 540))
    font = pygame.font.Font(None, 36)
    app = App(surface=surface, font=font)
    root = MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True)
    app.push(root)
    screen = AntWorkoutScreen(app, session=session, test_code="airborne_numerical_workout")
    app.push(screen)
    return app, screen


def test_airborne_numerical_workout_session_runs_setups_blocks_and_results() -> None:
    clock = FakeClock()
    session = _complete_small_workout(clock)

    summary = session.scored_summary()
    snapshot = session.snapshot()

    assert summary.block_count == 2
    assert summary.completed_blocks == 2
    assert summary.attempted == 2
    assert summary.correct == 2
    assert summary.workout_code == "airborne_numerical_workout"
    assert summary.difficulty_level_start == 6
    assert summary.difficulty_level_end == 6
    assert len(session.events()) == 2
    assert session.stage is AntWorkoutStage.RESULTS
    assert any("Block splits:" == line for line in snapshot.note_lines)
    assert any("1H " in line and "2H " in line for line in snapshot.note_lines)


def test_workout_intro_and_completion_have_no_extra_questions() -> None:
    clock = FakeClock()
    session = AntWorkoutSession(
        clock=clock,
        seed=123,
        plan=_build_small_workout_plan(),
        starting_level=5,
    )

    intro = session.snapshot()
    forbidden = "re" + "flection"
    assert intro.stage is AntWorkoutStage.INTRO
    assert forbidden not in intro.prompt.lower()
    assert all(forbidden not in line.lower() for line in intro.note_lines)

    session.activate()
    assert session.stage is AntWorkoutStage.BLOCK_SETUP
    assert forbidden not in session.snapshot().prompt.lower()

    session.activate()
    _finish_current_block_with_one_correct_answer(session, clock)
    session.activate()
    session.activate()
    _finish_current_block_with_one_correct_answer(session, clock)
    session.activate()

    results = session.snapshot()
    assert results.stage is AntWorkoutStage.RESULTS
    assert forbidden not in results.prompt.lower()
    assert all(forbidden not in line.lower() for line in results.note_lines)


def test_airborne_numerical_workout_attempt_result_omits_setup_metrics(tmp_path) -> None:
    clock = FakeClock()
    session = _complete_small_workout(clock)

    result = attempt_result_from_engine(session, test_code="airborne_numerical_workout")
    store = ResultsStore(tmp_path / "results.sqlite3")
    store.record_attempt(result=result, app_version="test", input_profile_id="default")

    assert result.attempted == 2
    assert result.correct == 2
    assert result.metrics["workout_code"] == "airborne_numerical_workout"
    assert "pre_focus_one" not in result.metrics
    assert "post_next_rule" not in result.metrics

    session_summary = store.session_summary()
    workout_summary = store.test_session_summary("airborne_numerical_workout")

    assert session_summary is not None
    assert session_summary.attempt_count == 1
    assert workout_summary is not None
    assert workout_summary.attempt_count == 1
    assert workout_summary.latest_accuracy == 1.0


def test_real_airborne_numerical_workout_matches_standard_90_minute_structure() -> None:
    plan = build_ant_workout_plan("airborne_numerical_workout")
    drill_codes = tuple(block.drill_code for block in plan.blocks)

    assert plan.scored_duration_s == pytest.approx(90.0 * 60.0)
    assert drill_codes == (
        "ant_info_grabber",
        "ant_snap_facts_sprint",
        "ant_time_flip",
        "ant_distance_scan",
        "ant_route_time_solve",
        "ant_endurance_solve",
        "ant_fuel_burn_solve",
        "airborne_scenario_steady",
        "airborne_scenario_pressure",
    )
    assert {
        "Information search",
        "Retention",
        "Arithmetic retrieval",
        "Time conversion",
        "Distance scanning",
        "Route-time solving",
        "Endurance solving",
        "Fuel-burn solving",
        "Full-question solving",
    }.issubset(set(plan.focus_skills))


@pytest.mark.parametrize(
    ("drill_code", "expected_title_prefix"),
    (
        ("ma_one_step_fluency", "Mental Arithmetic: One-Step Fluency"),
        ("ma_written_numerical_extraction", "Mental Arithmetic: Written Numerical Extraction"),
        ("vs_multi_target_class_search", "Visual Search: Multi-Target Class Search"),
        ("dr_visual_digit_query", "Digit Recognition: Visual Digit Query"),
        ("dr_difference_count", "Digit Recognition: Difference Count"),
        ("sma_split_axis_control", "Sensory Motor Apparatus: Split Axis Control"),
        ("rt_obscured_target_prediction", "Rapid Tracking: Obscured Target Prediction"),
        ("tbl_single_lookup_anchor", "Table Reading: Single Lookup Anchor"),
        ("sl_one_rule_identify", "System Logic: One-Rule Identify"),
        ("dtb_tracking_recall", "Dual-Task Bridge: Tracking + Recall"),
    ),
)
def test_build_workout_block_engine_supports_new_primitive_drill_codes(
    drill_code: str,
    expected_title_prefix: str,
) -> None:
    clock = FakeClock()
    block = AntWorkoutBlockPlan(
        block_id=f"test_{drill_code}",
        label=drill_code,
        description="Primitive drill smoke block.",
        focus_skills=("Primitive",),
        drill_code=drill_code,
        mode=AntDrillMode.BUILD,
        duration_min=0.25,
    )

    engine = build_workout_block_engine(
        clock=clock,
        block_seed=1234,
        difficulty_level=5,
        block=block,
    )
    snap = engine.snapshot()

    assert str(snap.title).startswith(expected_title_prefix)


@pytest.mark.parametrize(
    ("legacy_code", "canonical_code"),
    (
        ("abd_angle_calibration", "abd_angle_tempo"),
        ("abd_bearing_calibration", "abd_bearing_tempo"),
        ("ic_attitude_frame", "ic_instrument_attitude_matching"),
        ("si_static_mixed_run", "si_static_multiview_integration"),
        ("si_aircraft_multiview_integration", "si_moving_aircraft_multiview_integration"),
        ("tt1_command_switch_run", "trace_orientation_decode"),
        ("tt2_position_recall_run", "trace_movement_recall"),
    ),
)
def test_build_workout_block_engine_routes_replacement_aliases_through_canonical_builders(
    legacy_code: str,
    canonical_code: str,
) -> None:
    clock = FakeClock()
    legacy_block = AntWorkoutBlockPlan(
        block_id=f"legacy_{legacy_code}",
        label=legacy_code,
        description="Legacy alias block.",
        focus_skills=("Primitive",),
        drill_code=legacy_code,
        mode=AntDrillMode.BUILD,
        duration_min=0.25,
    )
    canonical_block = AntWorkoutBlockPlan(
        block_id=f"canonical_{canonical_code}",
        label=canonical_code,
        description="Canonical block.",
        focus_skills=("Primitive",),
        drill_code=canonical_code,
        mode=AntDrillMode.BUILD,
        duration_min=0.25,
    )

    legacy_engine = build_workout_block_engine(
        clock=clock,
        block_seed=4321,
        difficulty_level=5,
        block=legacy_block,
    )
    canonical_engine = build_workout_block_engine(
        clock=clock,
        block_seed=4321,
        difficulty_level=5,
        block=canonical_block,
    )
    legacy_snapshot = legacy_engine.snapshot()
    canonical_snapshot = canonical_engine.snapshot()
    legacy_result = attempt_result_from_engine(legacy_engine, test_code=legacy_block.drill_code)

    assert legacy_block.drill_code == legacy_code
    assert legacy_snapshot.title == canonical_snapshot.title
    assert legacy_snapshot.prompt == canonical_snapshot.prompt
    assert legacy_engine.difficulty == pytest.approx(canonical_engine.difficulty)
    assert getattr(legacy_engine, "_difficulty_code") == legacy_code
    assert legacy_result.test_code == legacy_code
    assert getattr(legacy_engine, "_resolved_difficulty_context").code_scope_key == canonical_code


def test_workout_dev_skip_hotkeys_advance_shell_skip_block_and_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CFAST_ENABLE_DEV_TOOLS", "1")
    clock = FakeClock()
    session = AntWorkoutSession(
        clock=clock,
        seed=123,
        plan=_build_small_workout_plan(),
        starting_level=5,
    )
    _app, screen = _build_app_and_workout_screen(clock=clock, session=session)
    try:
        for _ in range(4):
            screen.handle_event(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_F10, "unicode": ""})
            )

        assert session.stage is AntWorkoutStage.BLOCK

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_F11, "unicode": ""})
        )
        assert session.stage is AntWorkoutStage.BLOCK_RESULTS

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_F10, "unicode": ""})
        )
        assert session.stage is AntWorkoutStage.BLOCK_SETUP

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_F8, "unicode": ""})
        )
        assert session.stage is AntWorkoutStage.RESULTS
        assert session.scored_summary().completed_blocks == 1
    finally:
        pygame.quit()


def test_workout_pause_menu_shows_unified_actions() -> None:
    clock = FakeClock()
    session = AntWorkoutSession(
        clock=clock,
        seed=123,
        plan=_build_small_workout_plan(),
        starting_level=5,
    )
    _app, screen = _build_app_and_workout_screen(clock=clock, session=session)
    try:
        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "unicode": ""})
        )

        assert screen._pause_menu_options() == (
            "Resume",
            "Skip Current Segment",
            "Settings",
            "Main Menu",
        )
    finally:
        pygame.quit()


def test_workout_pause_menu_skip_current_segment_advances_current_block() -> None:
    clock = FakeClock()
    session = AntWorkoutSession(
        clock=clock,
        seed=123,
        plan=_build_small_workout_plan(),
        starting_level=5,
    )
    _app, screen = _build_app_and_workout_screen(clock=clock, session=session)
    try:
        session.activate()
        session.activate()
        session.activate()
        session.activate()
        session.activate()
        assert session.stage is AntWorkoutStage.BLOCK

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

        assert session.stage is AntWorkoutStage.BLOCK_RESULTS
        assert screen._pause_menu_active is False
    finally:
        pygame.quit()


def test_workout_pause_freezes_block_timer() -> None:
    clock = FakeClock()
    session = AntWorkoutSession(
        clock=clock,
        seed=123,
        plan=_build_small_workout_plan(),
        starting_level=5,
    )
    _app, screen = _build_app_and_workout_screen(clock=clock, session=session)
    surface = screen._app.surface
    try:
        session.activate()
        session.activate()
        session.activate()
        session.activate()
        session.activate()

        assert session.stage is AntWorkoutStage.BLOCK
        screen.render(surface)
        before = session.snapshot().block_time_remaining_s
        assert before is not None

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "unicode": ""})
        )
        clock.advance(5.0)
        screen.render(surface)
        paused = session.snapshot().block_time_remaining_s

        assert paused == pytest.approx(before)

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "unicode": ""})
        )
        clock.advance(2.0)
        screen.render(surface)
        resumed = session.snapshot().block_time_remaining_s

        assert resumed == pytest.approx(before - 2.0)
    finally:
        pygame.quit()


def test_workout_keypad_enter_advances_from_block_results() -> None:
    clock = FakeClock()
    session = AntWorkoutSession(
        clock=clock,
        seed=123,
        plan=_build_small_workout_plan(),
        starting_level=5,
    )
    _app, screen = _build_app_and_workout_screen(clock=clock, session=session)
    try:
        session.activate()
        session.activate()
        session.activate()
        session.activate()
        session.activate()
        assert session.stage is AntWorkoutStage.BLOCK

        _finish_current_block_with_one_correct_answer(session, clock)
        assert session.stage is AntWorkoutStage.BLOCK_RESULTS

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_KP_ENTER, "unicode": ""})
        )

        assert session.stage is AntWorkoutStage.BLOCK_SETUP
        assert session.snapshot().block_index == 2
    finally:
        pygame.quit()


def test_workout_escape_opens_pause_during_setup_and_results() -> None:
    clock = FakeClock()
    session = AntWorkoutSession(
        clock=clock,
        seed=123,
        plan=_build_small_workout_plan(),
        starting_level=5,
    )
    app, screen = _build_app_and_workout_screen(clock=clock, session=session)
    try:
        session.activate()
        assert session.stage is AntWorkoutStage.BLOCK_SETUP

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "unicode": ""})
        )

        assert screen._pause_menu_active is True
        screen._pause_menu_hitboxes = {}
        screen.render(app.surface)
        assert screen._pause_menu_hitboxes

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "unicode": ""})
        )
        assert screen._pause_menu_active is False
    finally:
        pygame.quit()

    clock = FakeClock()
    session = _complete_small_workout(clock)
    app, screen = _build_app_and_workout_screen(clock=clock, session=session)
    try:
        assert session.stage is AntWorkoutStage.RESULTS

        screen.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE, "unicode": ""})
        )

        assert screen._pause_menu_active is True
        screen._pause_menu_hitboxes = {}
        screen.render(app.surface)
        assert screen._pause_menu_hitboxes
    finally:
        pygame.quit()


def test_workout_pause_menu_skip_does_not_persist_attempt(
    tmp_path,
) -> None:
    clock = FakeClock()
    session = AntWorkoutSession(
        clock=clock,
        seed=123,
        plan=_build_small_workout_plan(),
        starting_level=5,
    )
    pygame.init()
    surface = pygame.display.set_mode((960, 540))
    font = pygame.font.Font(None, 36)
    store = ResultsStore(tmp_path / "results.sqlite3")
    app = App(surface=surface, font=font, results_store=store)
    app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))
    screen = AntWorkoutScreen(app, session=session, test_code="airborne_numerical_workout")
    app.push(screen)
    try:
        session.activate()
        session.activate()
        session.activate()
        session.activate()
        session.activate()
        assert session.stage is AntWorkoutStage.BLOCK

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
        session.debug_finish()
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


def test_workout_screen_uses_single_activity_session_while_running_block(tmp_path) -> None:
    clock = FakeClock()
    session = AntWorkoutSession(
        clock=clock,
        seed=123,
        plan=_build_small_workout_plan(),
        starting_level=5,
    )
    pygame.init()
    surface = pygame.display.set_mode((960, 540))
    font = pygame.font.Font(None, 36)
    store = ResultsStore(tmp_path / "results.sqlite3")
    app = App(surface=surface, font=font, results_store=store)
    app.push(MenuScreen(app, "Main Menu", [MenuItem("Quit", app.quit)], is_root=True))
    screen = AntWorkoutScreen(app, session=session, test_code="airborne_numerical_workout")
    app.push(screen)
    try:
        session.activate()
        session.activate()
        session.activate()
        session.activate()
        session.activate()
        assert session.stage is AntWorkoutStage.BLOCK

        screen.render(surface)

        with sqlite3.connect(store.path) as conn:
            activity_count = conn.execute("SELECT COUNT(*) FROM activity_session").fetchone()
            attempt_count = conn.execute("SELECT COUNT(*) FROM attempt").fetchone()

        assert activity_count == (1,)
        assert attempt_count == (0,)
    finally:
        pygame.quit()
