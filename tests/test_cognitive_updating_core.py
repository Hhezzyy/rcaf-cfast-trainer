from __future__ import annotations

from dataclasses import dataclass, replace
from typing import cast

import pytest

from cfast_trainer.cognitive_updating import (
    CognitiveUpdatingConfig,
    CognitiveUpdatingGenerator,
    CognitiveUpdatingPayload,
    CognitiveUpdatingRuntime,
    CognitiveUpdatingScorer,
    CognitiveUpdatingTrainingProfile,
    build_cognitive_updating_test,
    decode_cognitive_updating_submission_raw,
)


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _signature(payload: CognitiveUpdatingPayload) -> tuple[str, tuple[tuple[str, int], ...]]:
    fields = (
        ("warnings", len(payload.warning_lines)),
        ("messages", len(payload.message_lines)),
        ("parcel_fields", len(payload.parcel_target)),
        ("tanks", len(payload.tank_levels_l)),
    )
    return payload.question, fields


def _payload_for(seed: int) -> CognitiveUpdatingPayload:
    return cast(
        CognitiveUpdatingPayload,
        CognitiveUpdatingGenerator(seed=seed).next_problem(difficulty=0.5).payload,
    )


def _task_message_lines(snap) -> tuple[str, ...]:
    return tuple(line for line in snap.message_lines[4:] if line != "")


def test_generator_determinism_same_seed_same_sequence() -> None:
    seed = 447
    gen_a = CognitiveUpdatingGenerator(seed=seed)
    gen_b = CognitiveUpdatingGenerator(seed=seed)

    seq_a = [gen_a.next_problem(difficulty=0.6) for _ in range(20)]
    seq_b = [gen_b.next_problem(difficulty=0.6) for _ in range(20)]

    view_a = [
        (
            problem.prompt,
            problem.answer,
            _signature(cast(CognitiveUpdatingPayload, problem.payload)),
        )
        for problem in seq_a
    ]
    view_b = [
        (
            problem.prompt,
            problem.answer,
            _signature(cast(CognitiveUpdatingPayload, problem.payload)),
        )
        for problem in seq_b
    ]
    assert view_a == view_b


def test_generated_payload_has_multiple_components_and_submenus() -> None:
    payload = cast(
        CognitiveUpdatingPayload,
        CognitiveUpdatingGenerator(seed=91).next_problem(difficulty=0.5).payload,
    )

    assert len(payload.warning_lines) >= 2
    assert len(payload.message_lines) >= 2
    assert len(payload.parcel_target) == 3
    assert len(payload.tank_levels_l) == 3
    assert payload.active_tank in (1, 2, 3)
    assert payload.pressure_low < payload.pressure_high
    assert 0 <= payload.dispenser_lit <= 4
    assert len(payload.comms_code) == 4
    assert payload.comms_code.isdigit()


def test_scorer_exact_and_estimation_behaviour() -> None:
    problem = CognitiveUpdatingGenerator(seed=203).next_problem(difficulty=0.75)
    payload = cast(CognitiveUpdatingPayload, problem.payload)
    scorer = CognitiveUpdatingScorer()

    exact = scorer.score(problem=problem, user_answer=problem.answer, raw=str(problem.answer))
    tolerance = max(1, payload.estimate_tolerance)
    near = scorer.score(
        problem=problem,
        user_answer=int(problem.answer) + tolerance,
        raw=str(int(problem.answer) + tolerance),
    )
    far = scorer.score(
        problem=problem,
        user_answer=int(problem.answer) + (tolerance * 3),
        raw=str(int(problem.answer) + (tolerance * 3)),
    )

    assert exact == 1.0
    assert near == pytest.approx(0.5)
    assert far == 0.0


def test_runtime_state_machine_changes_code_from_actions() -> None:
    clock = FakeClock()
    payload = cast(
        CognitiveUpdatingPayload,
        CognitiveUpdatingGenerator(seed=511).next_problem(difficulty=0.5).payload,
    )
    runtime = CognitiveUpdatingRuntime(payload=payload, clock=clock)

    initial = runtime.snapshot()
    assert len(initial.state_code) == 4
    assert initial.event_count == 0

    runtime.set_pump(True)
    runtime.adjust_knots(7)
    runtime.toggle_camera("alpha")
    runtime.toggle_sensor("air")
    runtime.append_comms_digit("1")
    runtime.append_comms_digit("2")
    runtime.append_comms_digit("3")
    runtime.append_comms_digit("4")
    clock.advance(5.0)

    after = runtime.snapshot()
    assert after.event_count >= 8
    assert after.elapsed_s == 5
    assert len(after.state_code) == 4
    assert after.comms_input == "1234"
    raw = runtime.build_submission_raw()
    assert raw.startswith("1234")
    assert len(raw) == 32
    parsed = decode_cognitive_updating_submission_raw(raw)
    assert parsed is not None
    assert parsed.entered_code == "1234"
    assert len(parsed.state_code) == 4


def test_runtime_clock_tracks_elapsed_seconds() -> None:
    clock = FakeClock()
    payload = _payload_for(512)
    runtime = CognitiveUpdatingRuntime(payload=payload, clock=clock)

    start = runtime.snapshot()
    clock.advance(7.0)
    after = runtime.snapshot()

    assert start.clock_hms != ""
    assert after.clock_hms != start.clock_hms


def test_runtime_starts_with_no_warnings() -> None:
    clock = FakeClock()
    payload = _payload_for(516)
    runtime = CognitiveUpdatingRuntime(payload=payload, clock=clock)

    snap = runtime.snapshot()
    assert snap.warning_lines == ()


def test_sensor_activation_is_button_action_not_toggle() -> None:
    clock = FakeClock()
    payload = _payload_for(513)
    runtime = CognitiveUpdatingRuntime(payload=payload, clock=clock)

    clock.advance(3.0)
    before = runtime.snapshot()
    runtime.toggle_sensor("air")
    first = runtime.snapshot()
    assert first.air_sensor_armed is True
    assert first.event_count == before.event_count + 1
    assert first.air_time_left_s > before.air_time_left_s

    clock.advance(1.0)
    runtime.toggle_sensor("air")
    second = runtime.snapshot()
    assert second.air_sensor_armed is True
    assert second.event_count == first.event_count + 1
    assert second.air_time_left_s >= first.air_time_left_s - 1

    clock.advance(1.0)
    third = runtime.snapshot()
    assert third.air_sensor_armed is False


def test_camera_button_flashes_and_resets_due_message() -> None:
    clock = FakeClock()
    payload = _payload_for(514)
    runtime = CognitiveUpdatingRuntime(payload=payload, clock=clock)

    clock.advance(2.0)
    before = runtime.snapshot()
    runtime.toggle_camera("alpha")
    first = runtime.snapshot()

    assert first.alpha_armed is True
    assert first.event_count == before.event_count + 1
    assert any("Alpha Camera" in line for line in _task_message_lines(before))
    assert all("Alpha Camera" not in line for line in _task_message_lines(first))

    clock.advance(1.0)
    second = runtime.snapshot()
    assert second.alpha_armed is False


def test_message_lines_include_objective_and_comms_status_rows() -> None:
    clock = FakeClock()
    payload = _payload_for(515)
    runtime = CognitiveUpdatingRuntime(payload=payload, clock=clock)

    start = runtime.snapshot()
    assert start.message_lines[0] == ""
    assert start.message_lines[1] == ""
    assert start.message_lines[2] == ""
    assert start.message_lines[3] == ""
    assert len(_task_message_lines(start)) == 2

    clock.advance(3.0)
    lat = runtime.snapshot()
    assert lat.message_lines[0].startswith("Latitude: ")
    assert lat.message_lines[1] == ""

    clock.advance(8.0)
    lon = runtime.snapshot()
    assert lon.message_lines[1].startswith("Longitude: ")

    clock.advance(4.0)
    comms = runtime.snapshot()
    assert comms.message_lines[2] == ""
    assert comms.message_lines[3].startswith("Comms Code: ")

    clock.advance(8.0)
    timed = runtime.snapshot()
    assert timed.message_lines[2].startswith("Time: ")


def test_objective_message_fragments_clear_after_successful_drop() -> None:
    clock = FakeClock()
    payload = replace(
        _payload_for(521),
        parcel_target=(725880, 292172, 180742),
        objective_deadline_s=40,
    )
    runtime = CognitiveUpdatingRuntime(payload=payload, clock=clock)

    clock.advance(23.0)
    before = runtime.snapshot()
    assert before.message_lines[:3] == (
        "Latitude: 725880",
        "Longitude: 292172",
        "Time: 180742",
    )

    for ch in "725880":
        runtime.append_parcel_digit(ch)
    for ch in "292172":
        runtime.append_parcel_digit(ch)
    for ch in "180742":
        runtime.append_parcel_digit(ch)
    runtime.activate_dispenser()

    after = runtime.snapshot()
    assert after.message_lines[:3] == ("", "", "")


def test_objective_deadline_rollover_clears_entry_and_restarts_message_cycle() -> None:
    clock = FakeClock()
    payload = replace(
        _payload_for(522),
        parcel_target=(725880, 292172, 180742),
        objective_deadline_s=8,
        message_reveal_lat_s=1.0,
        message_reveal_lon_s=2.0,
        message_reveal_time_s=3.0,
    )
    runtime = CognitiveUpdatingRuntime(payload=payload, clock=clock)

    clock.advance(4.0)
    visible = runtime.snapshot()
    assert visible.message_lines[:3] == (
        "Latitude: 725880",
        "Longitude: 292172",
        "Time: 180742",
    )

    for ch in "725":
        runtime.append_parcel_digit(ch)
    partially_entered = runtime.snapshot()
    assert partially_entered.parcel_values[0] == "725"

    clock.advance(4.1)
    rolled = runtime.snapshot()
    assert rolled.parcel_values == ("", "", "")
    assert rolled.active_parcel_field == 0
    assert rolled.message_lines[:3] == ("", "", "")
    assert 7 <= rolled.objective_deadline_left_s <= 8

    clock.advance(1.1)
    restarted = runtime.snapshot()
    assert restarted.message_lines[0] == "Latitude: 725880"


def test_sensor_message_panel_prioritizes_urgent_items_and_clears_completed_sensor() -> None:
    clock = FakeClock()
    payload = replace(
        _payload_for(523),
        alpha_camera_due_s=45,
        bravo_camera_due_s=48,
        air_sensor_due_s=8,
        ground_sensor_due_s=10,
    )
    runtime = CognitiveUpdatingRuntime(payload=payload, clock=clock)

    start = runtime.snapshot()
    assert _task_message_lines(start) == (
        "Air Sensor due in: 08s",
        "Ground Sensor due in: 10s",
    )

    runtime.toggle_sensor("air")
    after = runtime.snapshot()
    assert all("Air Sensor" not in line for line in _task_message_lines(after))
    assert any("Ground Sensor" in line for line in _task_message_lines(after))


def test_message_panel_orders_overdue_task_before_nearest_due_task() -> None:
    clock = FakeClock()
    payload = replace(
        _payload_for(524),
        alpha_camera_due_s=9,
        bravo_camera_due_s=18,
        air_sensor_due_s=14,
        ground_sensor_due_s=26,
    )
    runtime = CognitiveUpdatingRuntime(payload=payload, clock=clock)

    clock.advance(12.0)
    snap = runtime.snapshot()

    task_lines = _task_message_lines(snap)
    assert len(task_lines) == 2
    assert task_lines[0].startswith("Activate Alpha Camera at: ")
    assert task_lines[1] == "Air Sensor due in: 02s"


def test_objective_dispenser_lights_match_reference_progression() -> None:
    clock = FakeClock()
    payload = _payload_for(517)
    runtime = CognitiveUpdatingRuntime(payload=payload, clock=clock)

    start = runtime.snapshot()
    assert start.dispenser_lit == 1

    lat, lon, due = payload.parcel_target
    for ch in f"{int(lat):06d}":
        runtime.append_parcel_digit(ch)
    after_lat = runtime.snapshot()
    assert after_lat.dispenser_lit == 2

    runtime.set_parcel_field(1)
    for ch in f"{int(lon):06d}":
        runtime.append_parcel_digit(ch)
    runtime.set_parcel_field(2)
    for ch in f"{int(due):06d}":
        runtime.append_parcel_digit(ch)
    filled = runtime.snapshot()
    assert filled.dispenser_lit == 5


def test_objective_time_value_is_independent_from_drop_deadline() -> None:
    clock = FakeClock()
    base = _payload_for(519)
    payload = replace(
        base,
        parcel_target=(725880, 292172, 180742),
        objective_deadline_s=40,
    )
    runtime = CognitiveUpdatingRuntime(payload=payload, clock=clock)

    clock.advance(23.0)
    snap = runtime.snapshot()

    assert snap.message_lines[2] == "Time: 180742"
    assert snap.objective_deadline_left_s == 17


def test_objective_cycle_resets_after_successful_drop() -> None:
    clock = FakeClock()
    base = _payload_for(520)
    payload = replace(
        base,
        parcel_target=(725880, 292172, 180742),
        objective_deadline_s=40,
    )
    runtime = CognitiveUpdatingRuntime(payload=payload, clock=clock)

    for ch in "725880":
        runtime.append_parcel_digit(ch)
    for ch in "292172":
        runtime.append_parcel_digit(ch)
    for ch in "180742":
        runtime.append_parcel_digit(ch)

    ready = runtime.snapshot()
    assert ready.objective_drop_ready is True
    assert ready.dispenser_lit == 5

    runtime.activate_dispenser()
    after = runtime.snapshot()
    assert after.parcel_values == ("", "", "")
    assert after.active_parcel_field == 0
    assert after.objective_drop_ready is False
    assert after.dispenser_lit == 1
    assert 39 <= after.objective_deadline_left_s <= 40


def test_navigation_and_engine_drift_rates_match_reference_video() -> None:
    clock = FakeClock()
    base = _payload_for(518)
    payload = replace(
        base,
        required_knots=100,
        current_knots=100,
        tank_levels_l=(458, 449, 448),
        active_tank=1,
    )
    runtime = CognitiveUpdatingRuntime(payload=payload, clock=clock)

    clock.advance(30.0)
    snap = runtime.snapshot()

    assert 114 <= snap.current_knots <= 116
    assert 427 <= snap.tank_levels_l[0] <= 429
    assert snap.tank_levels_l[1] == 449
    assert snap.tank_levels_l[2] == 448


def test_scorer_encoded_submission_weights_operations() -> None:
    problem = CognitiveUpdatingGenerator(seed=777).next_problem(difficulty=0.5)
    scorer = CognitiveUpdatingScorer()

    def _enc(
        *,
        entered: int,
        state: int,
        controls: int,
        navigation: int,
        engine: int,
        sensors: int,
        objectives: int,
        warning_penalty: int,
        overall: int,
        events: int,
    ) -> str:
        return (
            f"{entered:04d}{state:04d}"
            f"{controls:03d}{navigation:03d}{engine:03d}{sensors:03d}{objectives:03d}"
            f"{warning_penalty:03d}{overall:03d}{events:03d}"
        )

    # raw format:
    # ENTERED(4) + STATE(4) + CONTROLS(3) + NAV(3) + ENG(3) + SENS(3)
    # + OBJ(3) + WARN(3) + OVERALL(3) + EVENTS(3)
    strong = scorer.score(
        problem=problem,
        user_answer=0,
        raw=_enc(
            entered=1221,
            state=1221,
            controls=100,
            navigation=100,
            engine=100,
            sensors=100,
            objectives=100,
            warning_penalty=0,
            overall=100,
            events=10,
        ),
    )
    weak_ops = scorer.score(
        problem=problem,
        user_answer=0,
        raw=_enc(
            entered=3333,
            state=3333,
            controls=20,
            navigation=20,
            engine=20,
            sensors=20,
            objectives=20,
            warning_penalty=80,
            overall=20,
            events=10,
        ),
    )
    wrong_code = scorer.score(
        problem=problem,
        user_answer=0,
        raw=_enc(
            entered=9999,
            state=1221,
            controls=100,
            navigation=100,
            engine=100,
            sensors=100,
            objectives=100,
            warning_penalty=0,
            overall=100,
            events=10,
        ),
    )

    assert strong == pytest.approx(1.0)
    assert weak_ops < strong
    assert wrong_code < strong


def test_warning_penalty_samples_every_five_seconds_and_reduces_scores() -> None:
    clock = FakeClock()
    base = _payload_for(901)
    payload = replace(
        base,
        pressure_low=100,
        pressure_high=110,
        pressure_value=80,
        required_knots=120,
        current_knots=170,
        tank_levels_l=(460, 320, 300),
        active_tank=1,
    )
    runtime = CognitiveUpdatingRuntime(payload=payload, clock=clock)

    assert runtime.snapshot().warnings_penalty_points == 0
    clock.advance(4.9)
    assert runtime.snapshot().warnings_penalty_points == 0
    clock.advance(0.2)
    first_tick = runtime.snapshot()
    assert first_tick.warnings_penalty_points > 0
    clock.advance(10.0)
    later = runtime.snapshot()
    assert later.warnings_penalty_points > first_tick.warnings_penalty_points
    assert later.controls_score < 100
    assert later.navigation_score < 100
    assert later.engine_score < 100


def test_objective_mistype_blocks_drop_and_yields_zero_objective_score() -> None:
    clock = FakeClock()
    base = _payload_for(902)
    payload = replace(base, parcel_target=(321, 654, 40))
    runtime = CognitiveUpdatingRuntime(payload=payload, clock=clock)

    runtime.append_parcel_digit("9")
    runtime.activate_dispenser()
    snap = runtime.snapshot()

    assert snap.objective_drop_ready is False
    assert snap.objective_drop_complete is False
    assert snap.objectives_score == 0
    assert "Objective Warning" in snap.warning_lines


def test_pump_pressure_rise_and_fall_are_fixed_rate() -> None:
    clock = FakeClock()
    base = _payload_for(903)
    payload = replace(
        base,
        pressure_low=80,
        pressure_high=120,
        pressure_value=80,
        pump_on=False,
    )
    runtime = CognitiveUpdatingRuntime(payload=payload, clock=clock)

    runtime.set_pump(True)
    p0 = runtime.snapshot().pressure_value
    clock.advance(5.0)
    p1 = runtime.snapshot().pressure_value
    clock.advance(5.0)
    p2 = runtime.snapshot().pressure_value

    rise_1 = p1 - p0
    rise_2 = p2 - p1
    assert rise_1 > 0
    assert abs(rise_1 - rise_2) <= 1

    runtime.set_pump(False)
    clock.advance(5.0)
    p3 = runtime.snapshot().pressure_value
    clock.advance(5.0)
    p4 = runtime.snapshot().pressure_value

    drop_1 = p2 - p3
    drop_2 = p3 - p4
    assert abs(drop_1 - drop_2) <= 1


def test_timer_boundary_transitions_to_results_and_rejects_late_submit() -> None:
    clock = FakeClock()
    engine = build_cognitive_updating_test(
        clock=clock,
        seed=123,
        difficulty=0.5,
        config=CognitiveUpdatingConfig(scored_duration_s=2.0, practice_questions=0),
    )
    engine.start_scored()

    clock.advance(2.0)
    engine.update()

    assert engine.phase.value == "results"
    assert engine.submit_answer("222") is False

    summary = engine.scored_summary()
    assert summary.attempted == 0
    assert summary.correct == 0


def test_training_profile_payload_exposes_active_domains_family_and_focus() -> None:
    profile = CognitiveUpdatingTrainingProfile(
        active_domains=("navigation", "state_code"),
        scenario_family="compressed",
        focus_label="Navigation",
        starting_upper_tab_index=3,
        starting_lower_tab_index=1,
    )
    problem = CognitiveUpdatingGenerator(seed=604).next_problem_for_selection(
        difficulty=0.5,
        training_profile=profile,
        scenario_family=profile.scenario_family,
    )
    payload = cast(CognitiveUpdatingPayload, problem.payload)

    assert payload.active_domains == ("navigation", "state_code")
    assert payload.scenario_family == "compressed"
    assert payload.focus_label == "Navigation"
    assert payload.starting_upper_tab_index == 3
    assert payload.starting_lower_tab_index == 1


def test_inactive_domains_are_neutralized_for_focused_training_profiles() -> None:
    clock = FakeClock()
    payload = cast(
        CognitiveUpdatingPayload,
        CognitiveUpdatingGenerator(seed=605).next_problem_for_selection(
            difficulty=0.5,
            training_profile=CognitiveUpdatingTrainingProfile(
                active_domains=("controls", "state_code"),
                focus_label="Controls",
            ),
            scenario_family="baseline",
        ).payload,
    )
    runtime = CognitiveUpdatingRuntime(payload=payload, clock=clock)

    clock.advance(30.0)
    snap = runtime.snapshot()

    assert snap.navigation_score == 100
    assert snap.engine_score == 100
    assert snap.sensors_score == 100
    assert snap.objectives_score == 100
    assert "Air Speed Warning" not in snap.warning_lines
    assert "Engine Panel" not in snap.warning_lines
    assert "Sensor Panel" not in snap.warning_lines
    assert "Objective Warning" not in snap.warning_lines
