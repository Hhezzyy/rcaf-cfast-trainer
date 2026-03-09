from __future__ import annotations

from dataclasses import dataclass

import pytest

from cfast_trainer.cognitive_core import Phase
from cfast_trainer.trace_test_1 import (
    TraceTest1Attitude,
    TraceTest1Command,
    TraceTest1Config,
    TraceTest1Generator,
    TraceTest1Payload,
    TraceTest1TrialStage,
    build_trace_test_1_test,
    trace_test_1_answer_code,
    trace_test_1_scene_frames,
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
    g1 = TraceTest1Generator(seed=seed)
    g2 = TraceTest1Generator(seed=seed)

    seq1 = [g1.next_problem(difficulty=0.58) for _ in range(30)]
    seq2 = [g2.next_problem(difficulty=0.58) for _ in range(30)]

    assert [(p.prompt, p.answer, p.payload) for p in seq1] == [
        (p.prompt, p.answer, p.payload) for p in seq2
    ]


def test_generated_answer_maps_to_payload_correct_code_and_options() -> None:
    gen = TraceTest1Generator(seed=51)
    problem = gen.next_problem(difficulty=0.5)

    payload = problem.payload
    assert isinstance(payload, TraceTest1Payload)
    assert len(payload.options) == 4
    assert tuple(option.label for option in payload.options) == (
        "Left",
        "Right",
        "Push",
        "Pull",
    )
    assert tuple(option.code for option in payload.options) == (1, 2, 3, 4)
    assert payload.correct_code == problem.answer
    assert 1 <= payload.correct_code <= 4
    assert payload.reference.yaw_deg == pytest.approx(0.0)


def test_generator_emits_all_command_types() -> None:
    gen = TraceTest1Generator(seed=1234)
    seen: set[TraceTest1Command] = set()

    for _ in range(140):
        problem = gen.next_problem(difficulty=0.6)
        payload = problem.payload
        assert isinstance(payload, TraceTest1Payload)
        option_map = {int(option.code): option.command for option in payload.options}
        seen.add(option_map[int(payload.correct_code)])

    assert seen == {
        TraceTest1Command.LEFT,
        TraceTest1Command.RIGHT,
        TraceTest1Command.PUSH,
        TraceTest1Command.PULL,
    }


def test_generator_does_not_repeat_same_command_back_to_back() -> None:
    gen = TraceTest1Generator(seed=222)
    previous: int | None = None

    for _ in range(24):
        problem = gen.next_problem(difficulty=0.6)
        if previous is not None:
            assert problem.answer != previous
        previous = int(problem.answer)


def test_timer_boundary_transitions_to_results_and_rejects_late_submit() -> None:
    clock = FakeClock()
    engine = build_trace_test_1_test(
        clock=clock,
        seed=7,
        difficulty=0.5,
        config=TraceTest1Config(scored_duration_s=2.0, practice_questions=0),
    )
    engine.start_scored()

    assert engine.time_remaining_s() == pytest.approx(2.0)
    clock.advance(2.0)
    engine.update()

    assert engine.phase.value == "results"
    assert engine.submit_answer("1") is False


def test_trial_stays_on_current_problem_after_animation_completes() -> None:
    clock = FakeClock()
    engine = build_trace_test_1_test(
        clock=clock,
        seed=19,
        difficulty=0.5,
        config=TraceTest1Config(
            scored_duration_s=6.0,
            practice_questions=2,
            practice_observe_s=1.0,
            scored_observe_s=1.0,
        ),
    )
    engine.start_practice()

    payload = engine.snapshot().payload
    assert isinstance(payload, TraceTest1Payload)
    assert payload.trial_stage is TraceTest1TrialStage.QUESTION
    assert payload.observe_progress == pytest.approx(0.0)
    first_payload = payload

    clock.advance(0.5)
    payload = engine.snapshot().payload
    assert isinstance(payload, TraceTest1Payload)
    assert payload.trial_stage is TraceTest1TrialStage.QUESTION
    assert 0.45 <= payload.observe_progress <= 0.55

    clock.advance(0.51)
    engine.update()
    payload = engine.snapshot().payload
    assert isinstance(payload, TraceTest1Payload)
    assert payload.trial_stage is TraceTest1TrialStage.QUESTION
    assert payload.observe_progress == pytest.approx(1.0)
    assert payload.scene_turn_index == first_payload.scene_turn_index
    assert payload.correct_code == first_payload.correct_code


def test_submit_answer_blocked_until_turn_begins() -> None:
    clock = FakeClock()
    engine = build_trace_test_1_test(
        clock=clock,
        seed=77,
        difficulty=0.5,
        config=TraceTest1Config(
            scored_duration_s=6.0,
            practice_questions=1,
            practice_observe_s=1.0,
            scored_observe_s=1.0,
        ),
    )
    engine.start_practice()

    payload = engine.snapshot().payload
    assert isinstance(payload, TraceTest1Payload)

    # Input is ignored while the animation is still running.
    assert engine.submit_answer(str(payload.correct_code)) is False
    assert engine.phase is Phase.PRACTICE

    clock.advance(0.40)
    engine.update()
    assert engine.submit_answer(str(payload.correct_code)) is False

    # Once turn motion starts, input is accepted.
    clock.advance(0.03)
    engine.update()
    assert engine.submit_answer(str(payload.correct_code)) is True
    assert engine.phase is Phase.PRACTICE_DONE


def test_answer_parser_accepts_arrow_aliases() -> None:
    assert trace_test_1_answer_code("LEFT") == 1
    assert trace_test_1_answer_code("RIGHT") == 2
    assert trace_test_1_answer_code("UP") == 3
    assert trace_test_1_answer_code("PUSH") == 3
    assert trace_test_1_answer_code("DOWN") == 4
    assert trace_test_1_answer_code("PULL") == 4
    assert trace_test_1_answer_code("bogus") is None


def test_scene_frames_follow_path_heading_and_command_motion() -> None:
    reference = TraceTest1Attitude(roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0)
    left_end = TraceTest1Attitude(roll_deg=-58.0, pitch_deg=0.0, yaw_deg=0.0)
    left_early, _ = trace_test_1_scene_frames(
        reference=reference,
        candidate=left_end,
        correct_code=1,
        progress=0.15,
    )
    left_mid, _ = trace_test_1_scene_frames(
        reference=reference,
        candidate=left_end,
        correct_code=1,
        progress=0.55,
    )
    left_late, _ = trace_test_1_scene_frames(
        reference=reference,
        candidate=left_end,
        correct_code=1,
        progress=0.90,
    )

    assert left_early.position[1] < left_mid.position[1]
    assert left_mid.position[0] < left_early.position[0]
    assert left_late.position[0] < left_mid.position[0]
    assert left_early.travel_heading_deg == pytest.approx(0.0, abs=0.01)
    assert 240.0 <= left_mid.travel_heading_deg <= 350.0
    assert left_late.travel_heading_deg == pytest.approx(270.0, abs=2.0)
    assert left_early.attitude.yaw_deg == pytest.approx(0.0, abs=2.0)
    assert left_mid.attitude.yaw_deg == pytest.approx(left_mid.travel_heading_deg, abs=2.0)
    assert left_early.attitude.roll_deg == pytest.approx(0.0, abs=2.0)
    assert left_mid.attitude.roll_deg == pytest.approx(0.0, abs=2.0)

    push_end = TraceTest1Attitude(roll_deg=0.0, pitch_deg=-90.0, yaw_deg=0.0)
    push_early, _ = trace_test_1_scene_frames(
        reference=reference,
        candidate=push_end,
        correct_code=3,
        progress=0.15,
    )
    push_mid, _ = trace_test_1_scene_frames(
        reference=reference,
        candidate=push_end,
        correct_code=3,
        progress=0.55,
    )
    push_late, _ = trace_test_1_scene_frames(
        reference=reference,
        candidate=push_end,
        correct_code=3,
        progress=0.90,
    )

    assert push_early.position[1] < push_mid.position[1]
    assert push_mid.position[2] < push_early.position[2]
    assert push_late.position[1] == pytest.approx(push_mid.position[1], abs=0.01)
    assert push_late.position[2] < push_mid.position[2]
    assert push_early.travel_heading_deg == pytest.approx(0.0, abs=0.01)
    assert push_late.travel_heading_deg == pytest.approx(0.0, abs=0.01)
    assert push_early.attitude.yaw_deg == pytest.approx(0.0, abs=2.0)
    assert push_late.attitude.yaw_deg == pytest.approx(0.0, abs=2.0)
    assert push_early.attitude.pitch_deg == pytest.approx(0.0, abs=2.0)
    assert push_mid.attitude.pitch_deg < -15.0
    assert push_late.attitude.pitch_deg == pytest.approx(-90.0, abs=2.0)


def test_pull_turns_in_place_then_climbs_without_horizontal_drift() -> None:
    reference = TraceTest1Attitude(roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0)
    pull_end = TraceTest1Attitude(roll_deg=0.0, pitch_deg=90.0, yaw_deg=0.0)
    pull_early, _ = trace_test_1_scene_frames(
        reference=reference,
        candidate=pull_end,
        correct_code=4,
        progress=0.15,
    )
    pull_mid, _ = trace_test_1_scene_frames(
        reference=reference,
        candidate=pull_end,
        correct_code=4,
        progress=0.55,
    )
    pull_late, _ = trace_test_1_scene_frames(
        reference=reference,
        candidate=pull_end,
        correct_code=4,
        progress=0.90,
    )
    assert pull_early.position[1] < pull_mid.position[1]
    assert pull_late.position[1] == pytest.approx(pull_mid.position[1], abs=0.01)
    assert pull_late.position[0] == pytest.approx(pull_mid.position[0], abs=0.01)
    assert pull_late.position[2] > pull_mid.position[2]
    assert pull_late.attitude.pitch_deg == pytest.approx(90.0, abs=2.0)


def test_scene_turn_index_keeps_continuous_path_between_questions() -> None:
    reference = TraceTest1Attitude(roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0)
    left_end = TraceTest1Attitude(roll_deg=-58.0, pitch_deg=0.0, yaw_deg=0.0)
    push_end = TraceTest1Attitude(roll_deg=0.0, pitch_deg=-90.0, yaw_deg=0.0)

    end_turn_0, _ = trace_test_1_scene_frames(
        reference=reference,
        candidate=left_end,
        correct_code=1,
        progress=1.0,
        scene_turn_index=0,
    )
    start_turn_1, _ = trace_test_1_scene_frames(
        reference=reference,
        candidate=push_end,
        correct_code=3,
        progress=0.0,
        scene_turn_index=1,
    )

    assert start_turn_1.position[0] == pytest.approx(end_turn_0.position[0], abs=0.01)
    assert start_turn_1.position[1] == pytest.approx(end_turn_0.position[1], abs=0.01)
    assert start_turn_1.position[2] == pytest.approx(end_turn_0.position[2], abs=0.01)
