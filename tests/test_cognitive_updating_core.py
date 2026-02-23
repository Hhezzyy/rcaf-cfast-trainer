from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest

from cfast_trainer.cognitive_updating import (
    CognitiveUpdatingConfig,
    CognitiveUpdatingGenerator,
    CognitiveUpdatingPayload,
    CognitiveUpdatingRuntime,
    CognitiveUpdatingScorer,
    build_cognitive_updating_test,
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


def test_generator_determinism_same_seed_same_sequence() -> None:
    seed = 447
    gen_a = CognitiveUpdatingGenerator(seed=seed)
    gen_b = CognitiveUpdatingGenerator(seed=seed)

    seq_a = [gen_a.next_problem(difficulty=0.6) for _ in range(20)]
    seq_b = [gen_b.next_problem(difficulty=0.6) for _ in range(20)]

    view_a = [
        (problem.prompt, problem.answer, _signature(cast(CognitiveUpdatingPayload, problem.payload)))
        for problem in seq_a
    ]
    view_b = [
        (problem.prompt, problem.answer, _signature(cast(CognitiveUpdatingPayload, problem.payload)))
        for problem in seq_b
    ]
    assert view_a == view_b


def test_generated_payload_has_multiple_components_and_submenus() -> None:
    payload = cast(CognitiveUpdatingPayload, CognitiveUpdatingGenerator(seed=91).next_problem(difficulty=0.5).payload)

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
    payload = cast(CognitiveUpdatingPayload, CognitiveUpdatingGenerator(seed=511).next_problem(difficulty=0.5).payload)
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
    assert runtime.build_submission_raw().startswith("1234")


def test_scorer_encoded_submission_weights_operations() -> None:
    problem = CognitiveUpdatingGenerator(seed=777).next_problem(difficulty=0.5)
    scorer = CognitiveUpdatingScorer()

    # raw format: ENTERED(4) + STATE(4) + EVENT_COUNT(2)
    strong = scorer.score(problem=problem, user_answer=0, raw="1221122107")
    weak_ops = scorer.score(problem=problem, user_answer=0, raw="3333333307")
    wrong_code = scorer.score(problem=problem, user_answer=0, raw="9999122107")

    assert strong == pytest.approx(1.0)
    assert weak_ops < strong
    assert wrong_code < strong


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
