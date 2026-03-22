from __future__ import annotations

from dataclasses import dataclass

from cfast_trainer.ic_drills import (
    build_ic_attitude_frame_drill,
    build_ic_description_prime_drill,
    build_ic_description_run_drill,
    build_ic_heading_anchor_drill,
    build_ic_mixed_part_run_drill,
    build_ic_part1_orientation_run_drill,
    build_ic_pressure_run_drill,
    build_ic_reverse_panel_prime_drill,
    build_ic_reverse_panel_run_drill,
)
from cfast_trainer.instrument_comprehension import (
    InstrumentHeadingDisplayMode,
    InstrumentComprehensionPayload,
    InstrumentComprehensionTrialKind,
)
from cfast_trainer.instrument_orientation_solver import (
    display_match_error,
    display_observation_from_state,
)


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _difficulty_for_level(level: int) -> float:
    return float(level - 1) / 9.0


def _problem_signature(engine) -> tuple[object, ...]:
    current = engine._current
    assert current is not None
    payload = current.payload
    assert isinstance(payload, InstrumentComprehensionPayload)
    return (
        current.prompt,
        current.answer,
        payload.kind,
        payload.heading_display_mode,
        payload.prompt_state,
        tuple(
            (
                option.code,
                option.state,
                option.view_preset,
                option.description,
                option.distractor_tag,
                option.distractor_profile_tag,
            )
            for option in payload.options
        ),
        payload.option_errors,
    )


def _capture_signatures(builder, *, seed: int, count: int = 6) -> tuple[tuple[object, ...], ...]:
    clock = FakeClock()
    engine = builder(clock=clock, seed=seed, difficulty=0.5)
    engine.start_scored()
    captured: list[tuple[object, ...]] = []
    for _ in range(count):
        captured.append(_problem_signature(engine))
        answer = engine._current.answer
        assert engine.submit_answer(str(answer)) is True
    return tuple(captured)


def test_ic_drill_families_are_deterministic_for_same_seed() -> None:
    builders = (
        build_ic_heading_anchor_drill,
        build_ic_attitude_frame_drill,
        build_ic_part1_orientation_run_drill,
        build_ic_reverse_panel_prime_drill,
        build_ic_reverse_panel_run_drill,
        build_ic_description_prime_drill,
        build_ic_description_run_drill,
        build_ic_mixed_part_run_drill,
        build_ic_pressure_run_drill,
    )

    for builder in builders:
        assert _capture_signatures(builder, seed=1414) == _capture_signatures(builder, seed=1414)


def test_ic_single_part_drills_emit_only_the_intended_part() -> None:
    single_part_builders = (
        (build_ic_heading_anchor_drill, InstrumentComprehensionTrialKind.INSTRUMENTS_TO_AIRCRAFT),
        (build_ic_attitude_frame_drill, InstrumentComprehensionTrialKind.INSTRUMENTS_TO_AIRCRAFT),
        (build_ic_part1_orientation_run_drill, InstrumentComprehensionTrialKind.INSTRUMENTS_TO_AIRCRAFT),
        (build_ic_reverse_panel_prime_drill, InstrumentComprehensionTrialKind.AIRCRAFT_TO_INSTRUMENTS),
        (build_ic_reverse_panel_run_drill, InstrumentComprehensionTrialKind.AIRCRAFT_TO_INSTRUMENTS),
        (build_ic_description_prime_drill, InstrumentComprehensionTrialKind.INSTRUMENTS_TO_DESCRIPTION),
        (build_ic_description_run_drill, InstrumentComprehensionTrialKind.INSTRUMENTS_TO_DESCRIPTION),
    )

    for builder, expected_kind in single_part_builders:
        clock = FakeClock()
        engine = builder(clock=clock, seed=77, difficulty=0.5)
        engine.start_scored()
        for _ in range(5):
            payload = engine._current.payload
            assert isinstance(payload, InstrumentComprehensionPayload)
            assert payload.kind is expected_kind
            assert engine.submit_answer(str(engine._current.answer)) is True


def test_ic_mixed_part_run_uses_fixed_three_and_three_rhythm() -> None:
    clock = FakeClock()
    engine = build_ic_mixed_part_run_drill(clock=clock, seed=909, difficulty=0.5)
    engine.start_scored()

    kinds: list[InstrumentComprehensionTrialKind] = []
    for _ in range(8):
        payload = engine._current.payload
        assert isinstance(payload, InstrumentComprehensionPayload)
        kinds.append(payload.kind)
        assert engine.submit_answer(str(engine._current.answer)) is True

    assert kinds[:6] == [
        InstrumentComprehensionTrialKind.INSTRUMENTS_TO_AIRCRAFT,
        InstrumentComprehensionTrialKind.INSTRUMENTS_TO_AIRCRAFT,
        InstrumentComprehensionTrialKind.AIRCRAFT_TO_INSTRUMENTS,
        InstrumentComprehensionTrialKind.AIRCRAFT_TO_INSTRUMENTS,
        InstrumentComprehensionTrialKind.INSTRUMENTS_TO_DESCRIPTION,
        InstrumentComprehensionTrialKind.INSTRUMENTS_TO_DESCRIPTION,
    ]
    assert kinds[6:] == [
        InstrumentComprehensionTrialKind.INSTRUMENTS_TO_AIRCRAFT,
        InstrumentComprehensionTrialKind.INSTRUMENTS_TO_AIRCRAFT,
    ]


def test_ic_pressure_run_alternates_parts_item_by_item() -> None:
    clock = FakeClock()
    engine = build_ic_pressure_run_drill(clock=clock, seed=505, difficulty=0.5)
    engine.start_scored()

    kinds: list[InstrumentComprehensionTrialKind] = []
    for _ in range(6):
        payload = engine._current.payload
        assert isinstance(payload, InstrumentComprehensionPayload)
        kinds.append(payload.kind)
        assert engine.submit_answer(str(engine._current.answer)) is True

    assert kinds == [
        InstrumentComprehensionTrialKind.INSTRUMENTS_TO_AIRCRAFT,
        InstrumentComprehensionTrialKind.AIRCRAFT_TO_INSTRUMENTS,
        InstrumentComprehensionTrialKind.INSTRUMENTS_TO_DESCRIPTION,
        InstrumentComprehensionTrialKind.INSTRUMENTS_TO_AIRCRAFT,
        InstrumentComprehensionTrialKind.AIRCRAFT_TO_INSTRUMENTS,
        InstrumentComprehensionTrialKind.INSTRUMENTS_TO_DESCRIPTION,
    ]


def test_ic_attitude_frame_levels_l2_l5_l8_are_materially_different() -> None:
    def summarize(level: int) -> tuple[set[int], float]:
        clock = FakeClock()
        engine = build_ic_attitude_frame_drill(
            clock=clock,
            seed=611,
            difficulty=_difficulty_for_level(level),
        )
        engine.start_scored()
        headings: set[int] = set()
        nearest_distractor_errors: list[int] = []
        for _ in range(16):
            payload = engine._current.payload
            assert isinstance(payload, InstrumentComprehensionPayload)
            headings.add(payload.prompt_state.heading_deg % 360)
            nearest_distractor_errors.append(min(error for error in payload.option_errors if error > 0))
            assert engine.submit_answer(str(engine._current.answer)) is True
        return headings, sum(nearest_distractor_errors) / len(nearest_distractor_errors)

    low_headings, low_nearest = summarize(2)
    mid_headings, mid_nearest = summarize(5)
    high_headings, high_nearest = summarize(8)

    assert all(heading % 90 == 0 for heading in low_headings)
    assert all(heading % 90 == 0 for heading in mid_headings)
    assert any(heading % 90 != 0 for heading in high_headings)
    assert low_nearest > mid_nearest > high_nearest


def test_ic_drills_emit_both_heading_display_modes() -> None:
    builders = (
        build_ic_heading_anchor_drill,
        build_ic_reverse_panel_run_drill,
        build_ic_description_run_drill,
    )
    for builder in builders:
        clock = FakeClock()
        engine = builder(clock=clock, seed=919, difficulty=0.6)
        engine.start_scored()
        seen: set[InstrumentHeadingDisplayMode] = set()
        for _ in range(10):
            payload = engine._current.payload
            assert isinstance(payload, InstrumentComprehensionPayload)
            seen.add(payload.heading_display_mode)
            assert engine.submit_answer(str(engine._current.answer)) is True
        assert seen == {
            InstrumentHeadingDisplayMode.ROTATING_ROSE,
            InstrumentHeadingDisplayMode.MOVING_ARROW,
        }


def test_ic_drills_keep_display_model_grounded_correct_answer() -> None:
    builders = (
        build_ic_heading_anchor_drill,
        build_ic_reverse_panel_run_drill,
        build_ic_description_run_drill,
    )
    for builder in builders:
        clock = FakeClock()
        engine = builder(clock=clock, seed=939, difficulty=0.65)
        engine.start_scored()
        for _ in range(4):
            payload = engine._current.payload
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
            assert recomputed_errors == payload.option_errors
            assert payload.options[engine._current.answer - 1].distractor_tag == "correct"
            assert payload.option_errors[engine._current.answer - 1] == min(payload.option_errors)
            assert payload.option_errors.count(min(payload.option_errors)) == 1
            assert engine.submit_answer(str(engine._current.answer)) is True
