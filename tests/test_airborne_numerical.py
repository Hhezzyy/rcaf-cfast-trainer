from __future__ import annotations

from cfast_trainer.airborne_numerical import (
    TEMPLATES_BY_NAME,
    AirborneNumericalGenerator,
    AirborneScenario,
    _max_route_directness_ratio,
    _route_directness_ratio,
    _template_edge_geometry_lengths,
    build_airborne_numerical_test,
)
from cfast_trainer.clock import RealClock
from cfast_trainer.cognitive_core import Phase, SeededRng, round_half_up


def _hhmm_to_minutes(hhmm: int) -> int:
    hh = int(f"{hhmm:04d}"[:2])
    mm = int(f"{hhmm:04d}"[2:])
    return hh * 60 + mm


def _minutes_to_hhmm_int(minutes: int) -> int:
    minutes %= 24 * 60
    hh = minutes // 60
    mm = minutes % 60
    return int(f"{hh:02d}{mm:02d}")


def _derived_values(scenario: AirborneScenario) -> dict[str, int]:
    route_distance = sum(leg.distance for leg in scenario.legs)
    route_minutes = int(
        round_half_up((route_distance / scenario.speed_value) * float(scenario.speed_minutes))
    )
    start_minutes = _hhmm_to_minutes(int(scenario.start_time_hhmm))
    arrival = _minutes_to_hhmm_int(start_minutes + route_minutes)
    empty_minutes = int(
        round_half_up(
            (scenario.start_fuel_liters / scenario.fuel_burn_per_hr) * float(scenario.fuel_minutes)
        )
    )
    empty = _minutes_to_hhmm_int(start_minutes + empty_minutes)
    fuel_used = int(
        round_half_up((route_minutes / float(scenario.fuel_minutes)) * scenario.fuel_burn_per_hr)
    )
    return {
        "route_distance": route_distance,
        "route_minutes": route_minutes,
        "arrival": arrival,
        "empty": empty,
        "fuel_endurance": empty_minutes,
        "fuel_used": fuel_used,
        "parcel_effect": int(max(0, scenario.weight_speed_table[0][1] - scenario.speed_value)),
        "takeoff": int(scenario.start_time_hhmm),
    }


def _expected_answer(scenario: AirborneScenario) -> int:
    derived = _derived_values(scenario)
    if scenario.question_kind == "arrival_time":
        return derived["arrival"]
    if scenario.question_kind == "takeoff_time":
        return derived["takeoff"]
    if scenario.question_kind == "empty_time":
        return derived["empty"]
    if scenario.question_kind == "fuel_endurance":
        return derived["fuel_endurance"]
    if scenario.question_kind == "fuel_burned":
        return derived["fuel_used"]
    if scenario.question_kind == "distance_travelled":
        return derived["route_distance"]
    if scenario.question_kind == "parcel_weight":
        return int(scenario.parcel_weight_kg)
    if scenario.question_kind == "parcel_effect":
        return derived["parcel_effect"]
    raise AssertionError(f"unexpected question kind: {scenario.question_kind}")


def test_airborne_numerical_practice_flow_accepts_generated_answers() -> None:
    clock = RealClock()
    test = build_airborne_numerical_test(clock, seed=12345, practice=True)
    assert test.snapshot().phase == Phase.INSTRUCTIONS

    test.start()
    snap = test.snapshot()
    assert snap.phase == Phase.PRACTICE

    scenario = snap.payload
    assert isinstance(scenario, AirborneScenario)

    expected = _expected_answer(scenario)
    test.submit_answer(expected)
    assert test.snapshot().phase in (Phase.PRACTICE, Phase.PRACTICE_DONE)


def test_airborne_numerical_generation_logic_matches_answers_and_input_limits() -> None:
    for seed in range(1, 80):
        gen = AirborneNumericalGenerator(SeededRng(seed))
        for _ in range(5):
            problem = gen.generate()
            scenario = problem.payload
            assert isinstance(scenario, AirborneScenario)

            derived = _derived_values(scenario)
            assert scenario.route_distance_total == derived["route_distance"]
            assert scenario.route_travel_minutes == derived["route_minutes"]
            assert int(scenario.arrival_time_hhmm) == derived["arrival"]
            assert int(scenario.empty_time_hhmm) == derived["empty"]
            assert scenario.fuel_used_on_route == derived["fuel_used"]
            assert problem.answer == _expected_answer(scenario)

            assert scenario.answer_digits == 4
            rendered = f"{problem.answer:04d}"
            assert len(rendered) == 4
            assert 0 <= problem.answer <= 9999
            if scenario.answer_format == "hhmm":
                hh = int(rendered[:2])
                mm = int(rendered[2:])
                assert 0 <= hh <= 23
                assert 0 <= mm <= 59


def test_airborne_numerical_varies_units_formats_and_question_types() -> None:
    speed_units: set[str] = set()
    fuel_units: set[str] = set()
    question_kinds: set[str] = set()
    parcel_formats: set[str] = set()
    fuel_formats: set[str] = set()

    for seed in range(200, 320):
        problem = AirborneNumericalGenerator(SeededRng(seed)).generate()
        scenario = problem.payload
        assert isinstance(scenario, AirborneScenario)
        speed_units.add(scenario.speed_unit)
        fuel_units.add(scenario.fuel_unit)
        question_kinds.add(scenario.question_kind)
        parcel_formats.add(scenario.parcel_reference_format)
        fuel_formats.add(scenario.fuel_reference_format)

    assert any("/min" in unit for unit in speed_units)
    assert {"km/h", "knots"}.issubset(speed_units)
    assert {"L/hr", "L/min"}.issubset(fuel_units)
    assert {
        "arrival_time",
        "takeoff_time",
        "empty_time",
        "fuel_endurance",
        "fuel_burned",
        "distance_travelled",
        "parcel_weight",
        "parcel_effect",
    }.issubset(question_kinds)
    assert {"table", "chart"}.issubset(parcel_formats)
    assert {"table", "chart"}.issubset(fuel_formats)


def test_airborne_numerical_practice_questions_cover_full_question_bank() -> None:
    clock = RealClock()
    test = build_airborne_numerical_test(clock, seed=24680, practice=True)
    test.start()

    seen_kinds: set[str] = set()
    seen_speed_units: set[str] = set()
    seen_parcel_formats: set[str] = set()
    seen_fuel_formats: set[str] = set()

    for _ in range(8):
        snap = test.snapshot()
        assert snap.phase == Phase.PRACTICE
        scenario = snap.payload
        assert isinstance(scenario, AirborneScenario)
        seen_kinds.add(scenario.question_kind)
        seen_speed_units.add(scenario.speed_unit)
        seen_parcel_formats.add(scenario.parcel_reference_format)
        seen_fuel_formats.add(scenario.fuel_reference_format)
        test.submit_answer(f"{_expected_answer(scenario):04d}")

    assert test.snapshot().phase == Phase.PRACTICE_DONE
    assert seen_kinds == {
        "arrival_time",
        "takeoff_time",
        "empty_time",
        "fuel_endurance",
        "fuel_burned",
        "distance_travelled",
        "parcel_weight",
        "parcel_effect",
    }
    assert len(seen_speed_units) >= 4
    assert {"table", "chart"}.issubset(seen_parcel_formats)
    assert {"table", "chart"}.issubset(seen_fuel_formats)


def test_airborne_numerical_scored_generation_is_deterministic_and_difficulty_bounded() -> None:
    low_a = AirborneNumericalGenerator(SeededRng(9090))
    low_b = AirborneNumericalGenerator(SeededRng(9090))

    low_seq_a = [low_a.next_problem(difficulty=0.0) for _ in range(12)]
    low_seq_b = [low_b.next_problem(difficulty=0.0) for _ in range(12)]

    assert [(problem.prompt, problem.answer) for problem in low_seq_a] == [
        (problem.prompt, problem.answer) for problem in low_seq_b
    ]
    for problem in low_seq_a:
        scenario = problem.payload
        assert isinstance(scenario, AirborneScenario)
        assert scenario.speed_minutes == 60
        assert scenario.fuel_minutes == 60
        assert scenario.parcel_reference_format == "table"
        assert scenario.fuel_reference_format == "table"
        assert len(scenario.legs) <= 2

    high = AirborneNumericalGenerator(SeededRng(9090))
    high_seq = [high.next_problem(difficulty=1.0) for _ in range(18)]
    assert [(problem.prompt, problem.answer) for problem in high_seq[:12]] != [
        (problem.prompt, problem.answer) for problem in low_seq_a
    ]
    assert any(
        isinstance(problem.payload, AirborneScenario)
        and (
            problem.payload.speed_minutes != 60
            or problem.payload.fuel_minutes != 60
            or problem.payload.parcel_reference_format == "chart"
            or problem.payload.fuel_reference_format == "chart"
        )
        for problem in high_seq
    )


def test_airborne_numerical_routes_keep_unique_stops_and_support_direct_legs() -> None:
    seen_leg_counts: set[int] = set()

    for seed in range(1, 160):
        gen = AirborneNumericalGenerator(SeededRng(seed))
        for _ in range(3):
            problem = gen.generate()
            scenario = problem.payload
            assert isinstance(scenario, AirborneScenario)

            route = tuple(int(node) for node in scenario.route)
            assert len(route) == len(set(route))
            assert len(scenario.legs) == len(route) - 1
            seen_leg_counts.add(len(scenario.legs))

    assert 1 in seen_leg_counts
    assert max(seen_leg_counts) >= 4


def test_airborne_numerical_routes_stay_within_plausible_geometry_bounds() -> None:
    for seed in range(1, 120):
        gen = AirborneNumericalGenerator(SeededRng(seed))
        for _ in range(3):
            problem = gen.generate()
            scenario = problem.payload
            assert isinstance(scenario, AirborneScenario)

            template = TEMPLATES_BY_NAME[scenario.template_name]
            ratio = _route_directness_ratio(scenario.route, template)
            assert ratio <= _max_route_directness_ratio(len(scenario.legs))


def test_airborne_numerical_edge_distances_follow_map_geometry_order() -> None:
    for seed in range(1, 120):
        gen = AirborneNumericalGenerator(SeededRng(seed))
        for _ in range(3):
            problem = gen.generate()
            scenario = problem.payload
            assert isinstance(scenario, AirborneScenario)

            template = TEMPLATES_BY_NAME[scenario.template_name]
            geometry = _template_edge_geometry_lengths(template)
            order = sorted(range(len(geometry)), key=lambda idx: (geometry[idx], idx))
            ordered_distances = [scenario.edge_distances[idx] for idx in order]
            assert ordered_distances == sorted(ordered_distances)
