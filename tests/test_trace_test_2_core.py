from __future__ import annotations

from dataclasses import dataclass

import pytest

from cfast_trainer.cognitive_core import Phase
from cfast_trainer.trace_test_2 import (
    TraceTest2Config,
    TraceTest2Generator,
    TraceTest2Payload,
    TraceTest2QuestionKind,
    TraceTest2TrialStage,
    build_trace_test_2_test,
)


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


def test_generator_determinism_same_seed_same_sequence() -> None:
    seed = 211
    g1 = TraceTest2Generator(seed=seed)
    g2 = TraceTest2Generator(seed=seed)

    seq1 = [g1.next_problem(difficulty=0.57) for _ in range(20)]
    seq2 = [g2.next_problem(difficulty=0.57) for _ in range(20)]

    assert [(p.prompt, p.answer, p.payload) for p in seq1] == [
        (p.prompt, p.answer, p.payload) for p in seq2
    ]


def test_generated_problem_has_four_color_options_and_valid_answer() -> None:
    payload = TraceTest2Generator(seed=17).next_problem(difficulty=0.5).payload
    assert isinstance(payload, TraceTest2Payload)
    assert len(payload.aircraft) == 4
    assert tuple(track.color_name for track in payload.aircraft) == (
        "Red",
        "Blue",
        "Silver",
        "Yellow",
    )
    if payload.question_kind is TraceTest2QuestionKind.RED_LEFT_TURNS:
        assert tuple(option.label for option in payload.options) == ("0", "1", "2", "3")
    else:
        assert tuple(option.label for option in payload.options) == (
            "Red",
            "Blue",
            "Silver",
            "Yellow",
        )
    assert tuple(option.code for option in payload.options) == (1, 2, 3, 4)
    assert 1 <= payload.correct_code <= 4


def test_generated_scene_contains_unique_recall_facts() -> None:
    payload = TraceTest2Generator(seed=33).next_problem(difficulty=0.6).payload
    assert isinstance(payload, TraceTest2Payload)

    visible = [track.code for track in payload.aircraft if track.visible_at_end]
    lowest = max(payload.aircraft, key=lambda track: track.started_screen_y).code
    leftmost = min(payload.aircraft, key=lambda track: track.ended_screen_x).code
    shortest = min(payload.aircraft, key=lambda track: track.visible_fraction).code

    assert len(visible) == 1
    assert len({lowest, leftmost, shortest, visible[0]}) >= 3


def test_trial_stays_on_current_problem_after_animation_completes() -> None:
    clock = FakeClock()
    engine = build_trace_test_2_test(
        clock=clock,
        seed=99,
        difficulty=0.5,
        config=TraceTest2Config(
            scored_duration_s=6.0,
            practice_questions=1,
            practice_observe_s=1.0,
            scored_observe_s=1.0,
        ),
    )
    engine.start_practice()

    snap = engine.snapshot()
    payload = snap.payload
    assert isinstance(payload, TraceTest2Payload)
    assert payload.trial_stage is TraceTest2TrialStage.QUESTION
    assert payload.observe_progress == pytest.approx(0.0)
    first_payload = payload

    clock.advance(0.5)
    payload = engine.snapshot().payload
    assert isinstance(payload, TraceTest2Payload)
    assert payload.trial_stage is TraceTest2TrialStage.QUESTION
    assert 0.45 <= payload.observe_progress <= 0.55

    clock.advance(0.51)
    engine.update()
    snap = engine.snapshot()
    payload = snap.payload
    assert isinstance(payload, TraceTest2Payload)
    assert payload.trial_stage is TraceTest2TrialStage.QUESTION
    assert payload.observe_progress == pytest.approx(1.0)
    assert payload.stem == first_payload.stem
    assert payload.correct_code == first_payload.correct_code


def test_timer_boundary_transitions_to_results_and_rejects_late_submit() -> None:
    clock = FakeClock()
    engine = build_trace_test_2_test(
        clock=clock,
        seed=7,
        difficulty=0.5,
        config=TraceTest2Config(
            scored_duration_s=2.0,
            practice_questions=0,
            practice_observe_s=1.0,
            scored_observe_s=1.0,
        ),
    )
    engine.start_scored()

    assert engine.phase is Phase.SCORED
    clock.advance(2.0)
    engine.update()

    assert engine.phase is Phase.RESULTS
    assert engine.submit_answer("1") is False
