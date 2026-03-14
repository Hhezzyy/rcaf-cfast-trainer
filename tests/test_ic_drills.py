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
    InstrumentComprehensionPayload,
    InstrumentComprehensionTrialKind,
)


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _problem_signature(engine) -> tuple[object, ...]:
    current = engine._current
    assert current is not None
    payload = current.payload
    assert isinstance(payload, InstrumentComprehensionPayload)
    return (
        current.prompt,
        current.answer,
        payload.kind,
        payload.prompt_state,
        tuple((option.code, option.state, option.view_preset, option.description) for option in payload.options),
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
