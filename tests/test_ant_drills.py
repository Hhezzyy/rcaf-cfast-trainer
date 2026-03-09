from __future__ import annotations

from dataclasses import dataclass

from cfast_trainer.ant_drills import (
    AntAdaptiveDifficultyConfig,
    AntDistanceScanConfig,
    AntDistanceScanGenerator,
    AntDrillMode,
    AntEnduranceSolveConfig,
    AntEnduranceSolveGenerator,
    AntFuelBurnSolveConfig,
    AntFuelBurnSolveGenerator,
    AntInfoGrabberConfig,
    AntInfoGrabberGenerator,
    AntMixedTempoSetConfig,
    AntMixedTempoSetGenerator,
    AntPayloadReferenceConfig,
    AntPayloadReferenceGenerator,
    AntProblemRuntimeMeta,
    AntRouteTimeSolveConfig,
    AntRouteTimeSolveGenerator,
    AntSnapFactsSprintConfig,
    AntSnapFactsSprintGenerator,
    AntTimeFlipConfig,
    AntTimeFlipGenerator,
    build_ant_distance_scan_drill,
    build_ant_endurance_solve_drill,
    build_ant_fuel_burn_solve_drill,
    build_ant_info_grabber_drill,
    build_ant_mixed_tempo_set_drill,
    build_ant_payload_reference_drill,
    build_ant_route_time_solve_drill,
    build_ant_snap_facts_sprint_drill,
    build_ant_time_flip_drill,
)
from cfast_trainer.airborne_numerical import AirborneScenario
from cfast_trainer.cognitive_core import Phase


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _mixed_family(problem: object) -> str:
    payload = getattr(problem, "payload", None)
    if isinstance(payload, AntProblemRuntimeMeta):
        return payload.family
    assert isinstance(payload, AirborneScenario)
    if payload.question_kind in {"arrival_time", "takeoff_time"}:
        return "route_time"
    if payload.question_kind in {"empty_time", "fuel_endurance"}:
        return "endurance"
    if payload.question_kind == "fuel_burned":
        return "fuel_burn"
    if payload.question_kind == "distance_travelled":
        return "distance"
    assert payload.question_kind in {"parcel_weight", "parcel_effect"}
    return "payload"


def test_ant_snap_facts_generator_is_deterministic_for_same_seed() -> None:
    gen1 = AntSnapFactsSprintGenerator(seed=123, skin="mixed")
    gen2 = AntSnapFactsSprintGenerator(seed=123, skin="mixed")

    seq1 = [gen1.next_problem(difficulty=0.65) for _ in range(25)]
    seq2 = [gen2.next_problem(difficulty=0.65) for _ in range(25)]

    assert [(problem.prompt, problem.answer) for problem in seq1] == [
        (problem.prompt, problem.answer) for problem in seq2
    ]


def test_ant_time_flip_generator_is_deterministic_and_varied() -> None:
    gen1 = AntTimeFlipGenerator(seed=321, skin="mixed")
    gen2 = AntTimeFlipGenerator(seed=321, skin="mixed")

    seq1 = [gen1.next_problem(difficulty=0.75) for _ in range(30)]
    seq2 = [gen2.next_problem(difficulty=0.75) for _ in range(30)]

    assert [(problem.prompt, problem.answer) for problem in seq1] == [
        (problem.prompt, problem.answer) for problem in seq2
    ]
    prompts = [problem.prompt for problem in seq1]
    assert any("HHMM" in prompt for prompt in prompts)
    assert any("/hr" in prompt or "/min" in prompt for prompt in prompts)
    assert any("aviation" not in prompt.lower() for prompt in prompts)


def test_ant_mixed_tempo_set_generator_is_deterministic_and_switches_families() -> None:
    gen1 = AntMixedTempoSetGenerator(seed=909, skin="mixed")
    gen2 = AntMixedTempoSetGenerator(seed=909, skin="mixed")

    seq1 = [gen1.next_problem(difficulty=1.0) for _ in range(80)]
    seq2 = [gen2.next_problem(difficulty=1.0) for _ in range(80)]

    assert [(problem.prompt, problem.answer) for problem in seq1] == [
        (problem.prompt, problem.answer) for problem in seq2
    ]
    families = {_mixed_family(problem) for problem in seq1}
    assert families == {
        "snap",
        "time",
        "route_time",
        "endurance",
        "fuel_burn",
        "distance",
        "payload",
    }
    assert any(isinstance(problem.payload, AirborneScenario) for problem in seq1)
    caps = {
        problem.payload.family: problem.payload.base_cap_s
        for problem in seq1
        if isinstance(problem.payload, AntProblemRuntimeMeta)
    }
    assert caps["snap"] < caps["time"]


def test_ant_route_time_solve_generator_is_deterministic_and_low_level_stays_clean() -> None:
    gen1 = AntRouteTimeSolveGenerator(seed=111)
    gen2 = AntRouteTimeSolveGenerator(seed=111)

    seq1 = [gen1.next_problem(difficulty=0.0) for _ in range(12)]
    seq2 = [gen2.next_problem(difficulty=0.0) for _ in range(12)]

    assert [(problem.prompt, problem.answer) for problem in seq1] == [
        (problem.prompt, problem.answer) for problem in seq2
    ]

    for problem in seq1:
        scenario = problem.payload
        assert isinstance(scenario, AirborneScenario)
        assert scenario.question_kind == "arrival_time"
        assert scenario.parcel_reference_format == "table"
        assert scenario.speed_minutes == 60
        assert len(scenario.legs) <= 2


def test_ant_fuel_burn_solve_generator_is_deterministic_and_uses_fuel_burn_family() -> None:
    gen1 = AntFuelBurnSolveGenerator(seed=222)
    gen2 = AntFuelBurnSolveGenerator(seed=222)

    seq1 = [gen1.next_problem(difficulty=0.7) for _ in range(18)]
    seq2 = [gen2.next_problem(difficulty=0.7) for _ in range(18)]

    assert [(problem.prompt, problem.answer) for problem in seq1] == [
        (problem.prompt, problem.answer) for problem in seq2
    ]
    assert all(isinstance(problem.payload, AirborneScenario) for problem in seq1)
    assert all(problem.payload.question_kind == "fuel_burned" for problem in seq1)


def test_ant_endurance_solve_generator_is_deterministic_and_low_level_stays_simple() -> None:
    gen1 = AntEnduranceSolveGenerator(seed=333)
    gen2 = AntEnduranceSolveGenerator(seed=333)

    seq1 = [gen1.next_problem(difficulty=0.0) for _ in range(12)]
    seq2 = [gen2.next_problem(difficulty=0.0) for _ in range(12)]

    assert [(problem.prompt, problem.answer) for problem in seq1] == [
        (problem.prompt, problem.answer) for problem in seq2
    ]
    for problem in seq1:
        scenario = problem.payload
        assert isinstance(scenario, AirborneScenario)
        assert scenario.question_kind == "fuel_endurance"
        assert scenario.fuel_reference_format == "table"
        assert scenario.fuel_minutes == 60


def test_ant_distance_scan_generator_is_deterministic_and_scans_distance_family() -> None:
    gen1 = AntDistanceScanGenerator(seed=444)
    gen2 = AntDistanceScanGenerator(seed=444)

    seq1 = [gen1.next_problem(difficulty=0.6) for _ in range(12)]
    seq2 = [gen2.next_problem(difficulty=0.6) for _ in range(12)]

    assert [(problem.prompt, problem.answer) for problem in seq1] == [
        (problem.prompt, problem.answer) for problem in seq2
    ]
    assert all(isinstance(problem.payload, AirborneScenario) for problem in seq1)
    assert all(problem.payload.question_kind == "distance_travelled" for problem in seq1)


def test_ant_payload_reference_generator_is_deterministic_and_uses_payload_families() -> None:
    gen1 = AntPayloadReferenceGenerator(seed=555)
    gen2 = AntPayloadReferenceGenerator(seed=555)

    seq1 = [gen1.next_problem(difficulty=1.0) for _ in range(30)]
    seq2 = [gen2.next_problem(difficulty=1.0) for _ in range(30)]

    assert [(problem.prompt, problem.answer) for problem in seq1] == [
        (problem.prompt, problem.answer) for problem in seq2
    ]
    assert all(isinstance(problem.payload, AirborneScenario) for problem in seq1)
    assert {"parcel_weight", "parcel_effect"} == {problem.payload.question_kind for problem in seq1}


def test_ant_snap_facts_timeout_auto_advances_and_tracks_fixation_rate() -> None:
    clock = FakeClock()
    engine = build_ant_snap_facts_sprint_drill(
        clock=clock,
        seed=77,
        difficulty=0.5,
        mode=AntDrillMode.BUILD,
        config=AntSnapFactsSprintConfig(practice_questions=0, scored_duration_s=30.0),
    )
    engine.start_scored()

    first_prompt = engine.snapshot().prompt
    cap_s = float(engine._current_cap_s)
    clock.advance(cap_s + 0.05)
    engine.update()

    snap = engine.snapshot()
    summary = engine.scored_summary()

    assert snap.phase is Phase.SCORED
    assert snap.attempted_scored == 1
    assert first_prompt != snap.prompt
    assert summary.attempted == 1
    assert summary.correct == 0
    assert summary.timeouts == 1
    assert summary.fixation_rate == 1.0
    assert summary.max_timeout_streak == 1
    assert engine.events()[0].raw == "__timeout__"


def test_ant_snap_facts_build_feedback_is_immediate_and_tempo_is_not() -> None:
    clock = FakeClock()
    build_engine = build_ant_snap_facts_sprint_drill(
        clock=clock,
        seed=17,
        difficulty=0.5,
        mode=AntDrillMode.BUILD,
        config=AntSnapFactsSprintConfig(practice_questions=0, scored_duration_s=30.0, skin="abstract"),
    )
    tempo_engine = build_ant_snap_facts_sprint_drill(
        clock=clock,
        seed=17,
        difficulty=0.5,
        mode=AntDrillMode.TEMPO,
        config=AntSnapFactsSprintConfig(practice_questions=0, scored_duration_s=30.0, skin="abstract"),
    )
    mirror = AntSnapFactsSprintGenerator(seed=17, skin="abstract")
    first_problem = mirror.next_problem(difficulty=0.5)

    build_engine.start_scored()
    tempo_engine.start_scored()
    assert float(build_engine._current_cap_s) > float(tempo_engine._current_cap_s)

    assert build_engine.submit_answer(str(first_problem.answer)) is True
    assert tempo_engine.submit_answer(str(first_problem.answer)) is True

    assert build_engine.snapshot().practice_feedback == "Correct. Commit and move."
    assert tempo_engine.snapshot().practice_feedback is None


def test_ant_tempo_adaptive_difficulty_moves_up_on_clean_window() -> None:
    clock = FakeClock()
    engine = build_ant_snap_facts_sprint_drill(
        clock=clock,
        seed=91,
        difficulty=0.5,
        mode=AntDrillMode.TEMPO,
        config=AntSnapFactsSprintConfig(
            practice_questions=0,
            scored_duration_s=30.0,
            skin="abstract",
            adaptive=AntAdaptiveDifficultyConfig(enabled=True, window_size=3),
        ),
    )
    engine.start_scored()

    start_level = engine.scored_summary().difficulty_level_start
    for _ in range(3):
        answer = engine._current.answer
        assert engine.submit_answer(str(answer)) is True

    summary = engine.scored_summary()
    changes = engine.difficulty_changes()

    assert start_level == 5
    assert summary.difficulty_level_end == 6
    assert summary.difficulty_change_count == 1
    assert len(changes) == 1
    assert changes[0].old_level == 5
    assert changes[0].new_level == 6


def test_ant_tempo_adaptive_difficulty_moves_down_on_fixation_window() -> None:
    clock = FakeClock()
    engine = build_ant_snap_facts_sprint_drill(
        clock=clock,
        seed=55,
        difficulty=(8 - 1) / 9.0,
        mode=AntDrillMode.TEMPO,
        config=AntSnapFactsSprintConfig(
            practice_questions=0,
            scored_duration_s=45.0,
            adaptive=AntAdaptiveDifficultyConfig(enabled=True, window_size=3),
        ),
    )
    engine.start_scored()

    for _ in range(3):
        clock.advance(float(engine._current_cap_s) + 0.05)
        engine.update()

    summary = engine.scored_summary()
    changes = engine.difficulty_changes()

    assert summary.timeouts == 3
    assert summary.difficulty_level_start == 8
    assert summary.difficulty_level_end == 7
    assert summary.difficulty_change_count == 1
    assert len(changes) == 1
    assert changes[0].old_level == 8
    assert changes[0].new_level == 7


def test_ant_time_flip_build_uses_longer_caps_and_supports_hhmm_answers() -> None:
    clock = FakeClock()
    build_engine = build_ant_time_flip_drill(
        clock=clock,
        seed=202,
        difficulty=0.6,
        mode=AntDrillMode.BUILD,
        config=AntTimeFlipConfig(practice_questions=0, scored_duration_s=30.0, skin="abstract"),
    )
    stress_engine = build_ant_time_flip_drill(
        clock=clock,
        seed=202,
        difficulty=0.6,
        mode=AntDrillMode.STRESS,
        config=AntTimeFlipConfig(practice_questions=0, scored_duration_s=30.0, skin="abstract"),
    )

    build_engine.start_scored()
    stress_engine.start_scored()

    assert float(build_engine._current_cap_s) > float(stress_engine._current_cap_s)
    prompt = build_engine.snapshot().prompt
    answer = build_engine._current.answer
    assert build_engine.submit_answer(str(answer)) is True
    assert build_engine.scored_summary().attempted == 1
    assert ("HHMM" in prompt) or ("min" in prompt)


def test_ant_mixed_tempo_set_uses_family_specific_caps() -> None:
    clock = FakeClock()
    engine = build_ant_mixed_tempo_set_drill(
        clock=clock,
        seed=404,
        difficulty=1.0,
        mode=AntDrillMode.TEMPO,
        config=AntMixedTempoSetConfig(practice_questions=0, scored_duration_s=45.0, skin="abstract"),
    )
    engine.start_scored()

    seen: dict[str, float] = {}
    for _ in range(120):
        problem = engine._current
        assert problem is not None
        seen[_mixed_family(problem)] = float(engine._current_cap_s)
        if len(seen) == 7:
            break
        assert engine.submit_answer(str(problem.answer)) is True

    assert set(seen) == {
        "snap",
        "time",
        "route_time",
        "endurance",
        "fuel_burn",
        "distance",
        "payload",
    }
    assert seen["snap"] < seen["time"] < seen["route_time"]
    assert seen["time"] < seen["fuel_burn"]
    assert seen["distance"] >= seen["snap"]


def test_ant_mixed_tempo_set_reuses_live_airborne_payloads_and_tolerance() -> None:
    clock = FakeClock()
    engine = build_ant_mixed_tempo_set_drill(
        clock=clock,
        seed=707,
        difficulty=1.0,
        mode=AntDrillMode.TEMPO,
        config=AntMixedTempoSetConfig(practice_questions=0, scored_duration_s=90.0),
    )
    engine.start_scored()

    found_airborne = False
    found_tolerant = False
    for _ in range(120):
        problem = engine._current
        assert problem is not None
        if isinstance(problem.payload, AirborneScenario):
            found_airborne = True
            assert isinstance(engine.snapshot().payload, AirborneScenario)
            if problem.tolerance > 0:
                found_tolerant = True
                attempted_before = engine.scored_summary().attempted
                tolerant_answer = int(problem.answer) + int(problem.tolerance)
                assert engine.submit_answer(str(tolerant_answer)) is True
                summary = engine.scored_summary()
                assert summary.attempted == attempted_before + 1
                assert summary.correct >= 1
                break
        assert engine.submit_answer(str(problem.answer)) is True

    assert found_airborne is True
    assert found_tolerant is True


def test_ant_route_time_solve_engine_exposes_airborne_payload_and_caps() -> None:
    clock = FakeClock()
    engine = build_ant_route_time_solve_drill(
        clock=clock,
        seed=515,
        difficulty=0.2,
        mode=AntDrillMode.BUILD,
        config=AntRouteTimeSolveConfig(practice_questions=0, scored_duration_s=45.0),
    )

    engine.start_scored()
    snap = engine.snapshot()
    assert isinstance(snap.payload, AirborneScenario)
    assert snap.payload.question_kind in {"arrival_time", "takeoff_time"}
    assert float(engine._current_cap_s) >= 12.0


def test_ant_fuel_burn_solve_accepts_chart_tolerance_from_live_ant_scorer() -> None:
    clock = FakeClock()
    engine = build_ant_fuel_burn_solve_drill(
        clock=clock,
        seed=909,
        difficulty=1.0,
        mode=AntDrillMode.TEMPO,
        config=AntFuelBurnSolveConfig(practice_questions=0, scored_duration_s=90.0),
    )
    engine.start_scored()

    found_chart = False
    attempted_before = 0
    for _ in range(80):
        problem = engine._current
        assert problem is not None
        scenario = problem.payload
        assert isinstance(scenario, AirborneScenario)
        if problem.tolerance > 0:
            found_chart = True
            attempted_before = engine.scored_summary().attempted
            tolerant_answer = int(problem.answer) + int(problem.tolerance)
            assert engine.submit_answer(str(tolerant_answer)) is True
            summary = engine.scored_summary()
            assert summary.attempted == attempted_before + 1
            assert summary.correct >= 1
            break
        assert engine.submit_answer(str(problem.answer)) is True

    assert found_chart is True


def test_ant_fuel_burn_solve_advanced_items_get_more_cap_than_exact_items() -> None:
    gen = AntFuelBurnSolveGenerator(seed=707)
    seen: dict[str, float] = {}
    for _ in range(200):
        problem = gen.next_problem(difficulty=0.8)
        scenario = problem.payload
        assert isinstance(scenario, AirborneScenario)
        label = (
            "advanced"
            if (
                scenario.fuel_reference_format == "chart"
                or scenario.parcel_reference_format == "chart"
                or scenario.speed_minutes != 60
                or scenario.fuel_minutes != 60
            )
            else "exact"
        )
        seen[label] = float(gen.cap_for_problem(problem=problem, level=8))
        if len(seen) == 2:
            break

    assert set(seen) == {"exact", "advanced"}
    assert seen["advanced"] > seen["exact"]


def test_ant_info_grabber_generator_is_deterministic_and_uses_airborne_payloads() -> None:
    gen1 = AntInfoGrabberGenerator(seed=818)
    gen2 = AntInfoGrabberGenerator(seed=818)

    seq1 = [gen1.next_problem(difficulty=0.8) for _ in range(10)]
    seq2 = [gen2.next_problem(difficulty=0.8) for _ in range(10)]

    assert [(problem.prompt, problem.answer) for problem in seq1] == [
        (problem.prompt, problem.answer) for problem in seq2
    ]
    assert all(isinstance(problem.payload, AirborneScenario) for problem in seq1)
    assert all("Target:" in problem.prompt for problem in seq1)


def test_ant_info_grabber_partial_digit_credit_counts_without_exact_hit() -> None:
    clock = FakeClock()
    engine = build_ant_info_grabber_drill(
        clock=clock,
        seed=919,
        difficulty=0.9,
        mode=AntDrillMode.BUILD,
        config=AntInfoGrabberConfig(practice_questions=0, scored_duration_s=45.0),
    )
    engine.start_scored()

    problem = engine._current
    assert problem is not None
    expected = f"{problem.answer:04d}"
    partial = expected[:-1] + ("0" if expected[-1] != "0" else "1")

    assert engine.submit_answer(partial) is True
    summary = engine.scored_summary()

    assert summary.attempted == 1
    assert summary.correct == 0
    assert 0.0 < summary.total_score < 1.0
    assert 0.0 < summary.score_ratio < 1.0


def test_ant_endurance_distance_payload_engines_expose_airborne_payloads() -> None:
    clock = FakeClock()
    engines = (
        build_ant_endurance_solve_drill(
            clock=clock,
            seed=1111,
            difficulty=0.2,
            mode=AntDrillMode.BUILD,
            config=AntEnduranceSolveConfig(practice_questions=0, scored_duration_s=30.0),
        ),
        build_ant_distance_scan_drill(
            clock=clock,
            seed=2222,
            difficulty=0.4,
            mode=AntDrillMode.TEMPO,
            config=AntDistanceScanConfig(practice_questions=0, scored_duration_s=30.0),
        ),
        build_ant_payload_reference_drill(
            clock=clock,
            seed=3333,
            difficulty=0.8,
            mode=AntDrillMode.TEMPO,
            config=AntPayloadReferenceConfig(practice_questions=0, scored_duration_s=30.0),
        ),
    )

    for engine in engines:
        engine.start_scored()
        snap = engine.snapshot()
        assert isinstance(snap.payload, AirborneScenario)
