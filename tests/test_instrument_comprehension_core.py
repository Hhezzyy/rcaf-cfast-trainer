from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from cfast_trainer.cognitive_core import Problem
from cfast_trainer.instrument_comprehension import (
    InstrumentAircraftViewPreset,
    InstrumentComprehensionConfig,
    InstrumentComprehensionGenerator,
    InstrumentHeadingDisplayMode,
    InstrumentComprehensionPayload,
    InstrumentOptionRenderMode,
    InstrumentComprehensionScorer,
    InstrumentComprehensionTrialKind,
    InstrumentOption,
    InstrumentState,
    airspeed_turn,
    altimeter_hand_turns,
    build_instrument_comprehension_test,
    instrument_aircraft_view_preset_for_code,
    instrument_aircraft_reverse_prompt_view_preset,
)
from cfast_trainer.instrument_orientation_solver import (
    INSTRUMENT_COMMON_MISREAD_TAGS,
    apply_distractor_profile,
    display_match_error,
    display_observation_from_state,
    lower_band_profile_pool,
    interpreted_heading_from_display,
    north_up_heading_from_display,
)
from cfast_trainer.instrument_aircraft_cards import (
    aircraft_card_semantic_drift_tags,
    aircraft_card_pose_distance,
    aircraft_card_pose_signature,
)


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


def test_generator_determinism_same_seed_same_sequence() -> None:
    seed = 901
    g1 = InstrumentComprehensionGenerator(seed=seed)
    g2 = InstrumentComprehensionGenerator(seed=seed)

    seq1 = [g1.next_problem(difficulty=0.6) for _ in range(30)]
    seq2 = [g2.next_problem(difficulty=0.6) for _ in range(30)]

    assert [(p.prompt, p.answer, p.payload) for p in seq1] == [
        (p.prompt, p.answer, p.payload) for p in seq2
    ]


def test_generator_emits_all_required_trial_kinds() -> None:
    gen = InstrumentComprehensionGenerator(seed=1234)
    seen: set[InstrumentComprehensionTrialKind] = set()
    for _ in range(120):
        p = gen.next_problem(difficulty=0.6)
        payload = p.payload
        assert isinstance(payload, InstrumentComprehensionPayload)
        seen.add(payload.kind)

    assert seen == {
        InstrumentComprehensionTrialKind.AIRCRAFT_TO_INSTRUMENTS,
        InstrumentComprehensionTrialKind.INSTRUMENTS_TO_DESCRIPTION,
        InstrumentComprehensionTrialKind.INSTRUMENTS_TO_AIRCRAFT,
    }


def test_generator_emits_both_heading_display_modes() -> None:
    gen = InstrumentComprehensionGenerator(seed=1441)
    seen: set[InstrumentHeadingDisplayMode] = set()
    for _ in range(24):
        p = gen.next_problem(difficulty=0.6)
        payload = p.payload
        assert isinstance(payload, InstrumentComprehensionPayload)
        seen.add(payload.heading_display_mode)

    assert seen == {
        InstrumentHeadingDisplayMode.ROTATING_ROSE,
        InstrumentHeadingDisplayMode.MOVING_ARROW,
    }


def test_part1_option_generation_uses_fixed_view_family_by_code() -> None:
    gen = InstrumentComprehensionGenerator(seed=31415)
    problem = gen.next_problem_for_kind(
        kind=InstrumentComprehensionTrialKind.INSTRUMENTS_TO_AIRCRAFT,
        difficulty=0.6,
    )
    payload = problem.payload
    assert isinstance(payload, InstrumentComprehensionPayload)

    assert tuple(option.view_preset for option in payload.options) == (
        InstrumentAircraftViewPreset.FRONT_LEFT,
        InstrumentAircraftViewPreset.FRONT_RIGHT,
        InstrumentAircraftViewPreset.PROFILE_LEFT,
        InstrumentAircraftViewPreset.PROFILE_RIGHT,
        InstrumentAircraftViewPreset.TOP_OBLIQUE,
    )
    assert payload.option_render_mode is InstrumentOptionRenderMode.AIRCRAFT
    assert payload.prompt_view_preset is None
    assert tuple(
        instrument_aircraft_view_preset_for_code(option.code) for option in payload.options
    ) == tuple(option.view_preset for option in payload.options)
    assert payload.options[problem.answer - 1].distractor_tag == "correct"


def test_part2_reverse_generation_uses_aircraft_prompt_and_panel_answers() -> None:
    gen = InstrumentComprehensionGenerator(seed=27182)
    problem = gen.next_problem_for_kind(
        kind=InstrumentComprehensionTrialKind.AIRCRAFT_TO_INSTRUMENTS,
        difficulty=0.6,
    )
    payload = problem.payload
    assert isinstance(payload, InstrumentComprehensionPayload)

    assert payload.option_render_mode is InstrumentOptionRenderMode.INSTRUMENT_PANEL
    assert payload.prompt_view_preset is instrument_aircraft_reverse_prompt_view_preset()
    assert all(option.view_preset is None for option in payload.options)
    assert payload.options[problem.answer - 1].distractor_tag == "correct"


def test_generator_correct_answer_is_unique_minimum_under_display_matching() -> None:
    gen = InstrumentComprehensionGenerator(seed=8282)
    for kind in InstrumentComprehensionTrialKind:
        problem = gen.next_problem_for_kind(kind=kind, difficulty=0.65)
        payload = problem.payload
        assert isinstance(payload, InstrumentComprehensionPayload)
        prompt_observation = display_observation_from_state(
            payload.prompt_state,
            payload.heading_display_mode,
        )
        recomputed_errors = tuple(
            display_match_error(
                prompt_observation,
                option.state,
                kind=payload.kind,
                heading_display_mode=payload.heading_display_mode,
            )
            for option in payload.options
        )
        correct_option = payload.options[problem.answer - 1]

        assert recomputed_errors == payload.option_errors
        assert correct_option.distractor_tag == "correct"
        assert correct_option.distractor_profile_tag == "correct"
        assert payload.option_errors[problem.answer - 1] == min(payload.option_errors)
        assert payload.option_errors.count(min(payload.option_errors)) == 1


def test_common_misread_tags_appear_when_distinct() -> None:
    gen = InstrumentComprehensionGenerator(seed=6161)
    for kind in InstrumentComprehensionTrialKind:
        problem = gen.next_problem_for_kind(kind=kind, difficulty=0.6)
        payload = problem.payload
        assert isinstance(payload, InstrumentComprehensionPayload)
        prompt_observation = display_observation_from_state(
            payload.prompt_state,
            payload.heading_display_mode,
        )
        expected: list[str] = []
        seen_states = {payload.prompt_state}
        for tag in INSTRUMENT_COMMON_MISREAD_TAGS:
            candidate = apply_distractor_profile(
                payload.prompt_state,
                profile_tag=tag,
                kind=payload.kind,
                heading_display_mode=payload.heading_display_mode,
                difficulty=0.6,
            )
            if candidate != payload.prompt_state and candidate not in seen_states:
                expected.append(tag)
                seen_states.add(candidate)

        option_tags = {option.distractor_tag for option in payload.options}
        assert set(expected).issubset(option_tags)
        heading_observation = prompt_observation.heading
        if "misread_other_heading_mode" in expected:
            if payload.heading_display_mode is InstrumentHeadingDisplayMode.ROTATING_ROSE:
                expected_heading = interpreted_heading_from_display(
                    heading_observation,
                    assumed_mode=InstrumentHeadingDisplayMode.MOVING_ARROW,
                )
            else:
                expected_heading = interpreted_heading_from_display(
                    heading_observation,
                    assumed_mode=InstrumentHeadingDisplayMode.ROTATING_ROSE,
                )
            option = next(
                option for option in payload.options if option.distractor_tag == "misread_other_heading_mode"
            )
            assert option.state.heading_deg == expected_heading
        if "misread_north_up" in expected:
            option = next(
                option for option in payload.options if option.distractor_tag == "misread_north_up"
            )
            assert option.state.heading_deg == north_up_heading_from_display(heading_observation)


def test_easier_band_random_profile_comes_from_lower_band_pool() -> None:
    gen = InstrumentComprehensionGenerator(seed=9797)
    problem = gen.next_problem_for_kind(
        kind=InstrumentComprehensionTrialKind.INSTRUMENTS_TO_DESCRIPTION,
        difficulty=0.82,
    )
    payload = problem.payload
    assert isinstance(payload, InstrumentComprehensionPayload)

    easier_option = next(option for option in payload.options if option.distractor_tag == "easier_band_random")
    assert easier_option.distractor_profile_tag in set(
        lower_band_profile_pool(kind=payload.kind, difficulty=0.82)
    )


def test_part1_and_part2_option_states_do_not_collapse_into_near_duplicate_aircraft_cards() -> None:
    gen = InstrumentComprehensionGenerator(seed=4401)
    threshold = 26.0
    for kind in (
        InstrumentComprehensionTrialKind.INSTRUMENTS_TO_AIRCRAFT,
        InstrumentComprehensionTrialKind.AIRCRAFT_TO_INSTRUMENTS,
    ):
        problem = gen.next_problem_for_kind(kind=kind, difficulty=0.72)
        payload = problem.payload
        assert isinstance(payload, InstrumentComprehensionPayload)
        signatures = []
        for option in payload.options:
            preset = (
                option.view_preset
                if kind is InstrumentComprehensionTrialKind.INSTRUMENTS_TO_AIRCRAFT
                else (payload.prompt_view_preset or InstrumentAircraftViewPreset.FRONT_LEFT)
            )
            signatures.append(aircraft_card_pose_signature(option.state, view_preset=preset))
        for idx, left in enumerate(signatures):
            for right in signatures[idx + 1 :]:
                assert aircraft_card_pose_distance(left, right) >= threshold


def test_part1_and_part2_aircraft_cards_stay_semantically_consistent() -> None:
    gen = InstrumentComprehensionGenerator(seed=5151)
    for kind in (
        InstrumentComprehensionTrialKind.INSTRUMENTS_TO_AIRCRAFT,
        InstrumentComprehensionTrialKind.AIRCRAFT_TO_INSTRUMENTS,
    ):
        problem = gen.next_problem_for_kind(kind=kind, difficulty=0.72)
        payload = problem.payload
        assert isinstance(payload, InstrumentComprehensionPayload)
        if kind is InstrumentComprehensionTrialKind.AIRCRAFT_TO_INSTRUMENTS:
            assert aircraft_card_semantic_drift_tags(
                payload.prompt_state,
                view_preset=payload.prompt_view_preset or InstrumentAircraftViewPreset.FRONT_LEFT,
            ) == ()
        for option in payload.options:
            preset = (
                option.view_preset
                if kind is InstrumentComprehensionTrialKind.INSTRUMENTS_TO_AIRCRAFT
                else (payload.prompt_view_preset or InstrumentAircraftViewPreset.FRONT_LEFT)
            )
            assert aircraft_card_semantic_drift_tags(
                option.state,
                view_preset=preset or InstrumentAircraftViewPreset.FRONT_LEFT,
            ) == ()


def test_fixed_canonical_states_stay_unique_minimum_across_all_ic_parts() -> None:
    class FixedStateGenerator(InstrumentComprehensionGenerator):
        def __init__(
            self,
            *,
            seed: int,
            fixed_state: InstrumentState,
            fixed_mode: InstrumentHeadingDisplayMode,
        ) -> None:
            super().__init__(seed=seed)
            self._fixed_state = fixed_state
            self._fixed_mode = fixed_mode

        def _sample_state(self, *, difficulty: float) -> InstrumentState:
            _ = difficulty
            return self._fixed_state

        def _sample_heading_display_mode(self) -> InstrumentHeadingDisplayMode:
            return self._fixed_mode

    canonical_states = (
        InstrumentState(
            speed_kts=220,
            altitude_ft=4800,
            vertical_rate_fpm=900,
            bank_deg=16,
            pitch_deg=6,
            heading_deg=90,
            slip=0,
        ),
        InstrumentState(
            speed_kts=205,
            altitude_ft=6200,
            vertical_rate_fpm=-700,
            bank_deg=-20,
            pitch_deg=-5,
            heading_deg=225,
            slip=0,
        ),
    )
    for fixed_state, mode in zip(
        canonical_states,
        (
            InstrumentHeadingDisplayMode.ROTATING_ROSE,
            InstrumentHeadingDisplayMode.MOVING_ARROW,
        ),
        strict=False,
    ):
        gen = FixedStateGenerator(seed=700 + fixed_state.heading_deg, fixed_state=fixed_state, fixed_mode=mode)
        for kind in InstrumentComprehensionTrialKind:
            problem = gen.next_problem_for_kind(kind=kind, difficulty=0.7)
            payload = problem.payload
            assert isinstance(payload, InstrumentComprehensionPayload)
            assert payload.options[problem.answer - 1].distractor_tag == "correct"
            assert payload.option_errors[problem.answer - 1] == min(payload.option_errors)
            assert payload.option_errors.count(min(payload.option_errors)) == 1


def test_scoring_exact_and_estimation_behavior() -> None:
    scorer = InstrumentComprehensionScorer()

    base = InstrumentState(
        speed_kts=240,
        altitude_ft=1500,
        vertical_rate_fpm=-400,
        bank_deg=-10,
        pitch_deg=-4,
        heading_deg=315,
        slip=0,
    )
    payload = InstrumentComprehensionPayload(
        kind=InstrumentComprehensionTrialKind.INSTRUMENTS_TO_DESCRIPTION,
        prompt_state=base,
        prompt_description="Speed 240 knots ...",
        options=(
            InstrumentOption(code=1, state=base, description="A"),
            InstrumentOption(code=2, state=replace(base, heading_deg=330), description="B"),
            InstrumentOption(code=3, state=replace(base, altitude_ft=2200), description="C"),
            InstrumentOption(
                code=4, state=replace(base, bank_deg=25, pitch_deg=8), description="D"
            ),
            InstrumentOption(code=5, state=replace(base, speed_kts=280), description="E"),
        ),
        option_errors=(0, 15, 42, 98, 60),
        full_credit_error=0,
        zero_credit_error=90,
    )
    problem = Problem(prompt="Pick best", answer=1, payload=payload)

    assert scorer.score(problem=problem, user_answer=1, raw="1") == pytest.approx(1.0)
    assert scorer.score(problem=problem, user_answer=2, raw="2") == pytest.approx(
        (90 - 15) / 90, abs=1e-9
    )
    assert scorer.score(problem=problem, user_answer=3, raw="3") == pytest.approx(
        (90 - 42) / 90, abs=1e-9
    )
    assert scorer.score(problem=problem, user_answer=4, raw="4") == pytest.approx(0.0)
    assert scorer.score(problem=problem, user_answer=9, raw="9") == pytest.approx(0.0)


def test_altimeter_hands_use_thousands_and_hundreds_turns() -> None:
    thousands_turn, hundreds_turn = altimeter_hand_turns(1500)
    assert thousands_turn == pytest.approx(0.15)
    assert hundreds_turn == pytest.approx(0.5)

    thousands_turn, hundreds_turn = altimeter_hand_turns(9876)
    assert thousands_turn == pytest.approx(0.9876)
    assert hundreds_turn == pytest.approx(0.876)


def test_airspeed_turn_uses_full_zero_to_360_circle() -> None:
    assert airspeed_turn(0) == pytest.approx(0.0)
    assert airspeed_turn(180) == pytest.approx(0.5)
    assert airspeed_turn(360) == pytest.approx(1.0)
    assert airspeed_turn(420) == pytest.approx(1.0)


def test_timer_boundary_transitions_to_results_and_rejects_late_submit() -> None:
    clock = FakeClock()
    engine = build_instrument_comprehension_test(
        clock=clock,
        seed=7,
        difficulty=0.5,
        config=InstrumentComprehensionConfig(scored_duration_s=2.0, practice_questions=0),
    )
    engine.start_scored()

    assert engine.time_remaining_s() == pytest.approx(2.0 / 3.0)
    clock.advance(2.0 / 3.0)
    engine.update()

    assert engine.phase.value == "practice_done"
    engine.start_scored()
    assert engine.phase.value == "scored"
    clock.advance(2.0 / 3.0)
    engine.update()
    assert engine.phase.value == "practice_done"
    engine.start_scored()
    assert engine.phase.value == "scored"
    clock.advance(2.0 / 3.0)
    engine.update()
    assert engine.phase.value == "results"
    assert engine.submit_answer("1") is False


def test_skip_commands_advance_across_practice_parts_and_results() -> None:
    clock = FakeClock()
    engine = build_instrument_comprehension_test(
        clock=clock,
        seed=9,
        difficulty=0.5,
        config=InstrumentComprehensionConfig(scored_duration_s=20.0, practice_questions=1),
    )

    engine.start_practice()
    assert engine.phase.value == "practice"

    assert engine.submit_answer("__skip_practice__") is True
    assert engine.phase.value == "practice_done"

    engine.start_scored()
    assert engine.phase.value == "scored"

    assert engine.submit_answer("__skip_section__") is True
    assert engine.phase.value == "practice_done"

    engine.start_scored()
    assert engine.phase.value == "practice"

    assert engine.submit_answer("__skip_practice__") is True
    assert engine.phase.value == "practice_done"

    engine.start_scored()
    assert engine.phase.value == "scored"

    assert engine.submit_answer("__skip_section__") is True
    assert engine.phase.value == "practice_done"

    engine.start_scored()
    assert engine.phase.value == "practice"

    assert engine.submit_answer("__skip_practice__") is True
    assert engine.phase.value == "practice_done"

    engine.start_scored()
    assert engine.phase.value == "scored"

    assert engine.submit_answer("__skip_all__") is True
    assert engine.phase.value == "results"
