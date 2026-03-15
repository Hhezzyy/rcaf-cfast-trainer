from __future__ import annotations

from dataclasses import dataclass

import pytest

from cfast_trainer.cognitive_core import Phase
from cfast_trainer.trace_test_1 import (
    TraceTest1AircraftPlan,
    TraceTest1AircraftState,
    TraceTest1Command,
    TraceTest1Config,
    TraceTest1Generator,
    TraceTest1Payload,
    TraceTest1PromptPlan,
    TraceTest1TrialStage,
    build_trace_test_1_test,
    trace_test_1_answer_code,
    trace_test_1_difficulty_tier,
    trace_test_1_normalized_position,
    trace_test_1_scene_frames,
)


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


def _next_prompt_for_command(command: TraceTest1Command) -> TraceTest1PromptPlan:
    gen = TraceTest1Generator(seed=321)
    for _ in range(24):
        problem = gen.next_problem(difficulty=0.55)
        payload = problem.payload
        assert isinstance(payload, TraceTest1PromptPlan)
        if payload.red_plan.command is command:
            return payload
    raise AssertionError(f"did not generate {command}")


def _manual_prompt(
    *,
    command: TraceTest1Command,
    heading_deg: float = 0.0,
) -> TraceTest1PromptPlan:
    return TraceTest1PromptPlan(
        prompt_index=0,
        answer_open_progress=0.42,
        speed_multiplier=1.0,
        red_plan=TraceTest1AircraftPlan(
            start_state=TraceTest1AircraftState(position=(0.0, 8.0, 12.0), heading_deg=heading_deg),
            command=command,
            lead_distance=4.0,
            maneuver_distance=3.0,
            altitude_delta=2.0 if command is TraceTest1Command.PULL else -2.0,
        ),
        blue_plans=(),
    )


def test_generator_determinism_same_seed_same_sequence() -> None:
    seed = 901
    g1 = TraceTest1Generator(seed=seed)
    g2 = TraceTest1Generator(seed=seed)

    seq1 = [g1.next_problem(difficulty=0.58) for _ in range(20)]
    seq2 = [g2.next_problem(difficulty=0.58) for _ in range(20)]

    assert [(p.prompt, p.answer, p.payload) for p in seq1] == [
        (p.prompt, p.answer, p.payload) for p in seq2
    ]


def test_difficulty_tiers_match_blue_counts_speed_and_answer_open_points() -> None:
    assert trace_test_1_difficulty_tier(difficulty=0.10).blue_count == 1
    assert trace_test_1_difficulty_tier(difficulty=0.10).speed_multiplier == pytest.approx(1.00)
    assert trace_test_1_difficulty_tier(difficulty=0.10).answer_open_progress == pytest.approx(0.42)

    assert trace_test_1_difficulty_tier(difficulty=0.50).blue_count == 2
    assert trace_test_1_difficulty_tier(difficulty=0.50).speed_multiplier == pytest.approx(1.15)
    assert trace_test_1_difficulty_tier(difficulty=0.50).answer_open_progress == pytest.approx(0.36)

    assert trace_test_1_difficulty_tier(difficulty=0.80).blue_count == 3
    assert trace_test_1_difficulty_tier(difficulty=0.80).speed_multiplier == pytest.approx(1.30)
    assert trace_test_1_difficulty_tier(difficulty=0.80).answer_open_progress == pytest.approx(0.31)

    assert trace_test_1_difficulty_tier(difficulty=0.95).blue_count == 4
    assert trace_test_1_difficulty_tier(difficulty=0.95).speed_multiplier == pytest.approx(1.45)
    assert trace_test_1_difficulty_tier(difficulty=0.95).answer_open_progress == pytest.approx(0.27)


def test_generator_emits_all_commands_without_back_to_back_repeats() -> None:
    gen = TraceTest1Generator(seed=1234)
    seen: set[TraceTest1Command] = set()
    previous: int | None = None

    for _ in range(60):
        problem = gen.next_problem(difficulty=0.6)
        payload = problem.payload
        assert isinstance(payload, TraceTest1PromptPlan)
        seen.add(payload.red_plan.command)
        if previous is not None:
            assert int(problem.answer) != previous
        previous = int(problem.answer)

    assert seen == {
        TraceTest1Command.LEFT,
        TraceTest1Command.RIGHT,
        TraceTest1Command.PUSH,
        TraceTest1Command.PULL,
    }


def test_allowed_commands_none_matches_default_generator_sequence() -> None:
    default = TraceTest1Generator(seed=777)
    explicit = TraceTest1Generator(seed=777, allowed_commands=None)

    default_answers = [default.next_problem(difficulty=0.6).answer for _ in range(12)]
    explicit_answers = [explicit.next_problem(difficulty=0.6).answer for _ in range(12)]

    assert explicit_answers == default_answers


def test_allowed_commands_filter_limits_emitted_commands() -> None:
    gen = TraceTest1Generator(
        seed=888,
        allowed_commands=(TraceTest1Command.LEFT, TraceTest1Command.RIGHT),
    )

    for _ in range(24):
        payload = gen.next_problem(difficulty=0.6).payload
        assert isinstance(payload, TraceTest1PromptPlan)
        assert payload.red_plan.command in {
            TraceTest1Command.LEFT,
            TraceTest1Command.RIGHT,
        }


def test_allowed_commands_validation_rejects_empty_and_invalid_values() -> None:
    with pytest.raises(ValueError, match="allowed_commands must not be empty"):
        build_trace_test_1_test(
            clock=FakeClock(),
            seed=91,
            difficulty=0.5,
            config=TraceTest1Config(allowed_commands=()),
        )

    with pytest.raises(ValueError, match="Unknown Trace Test 1 command"):
        build_trace_test_1_test(
            clock=FakeClock(),
            seed=92,
            difficulty=0.5,
            config=TraceTest1Config(allowed_commands=("bogus",)),  # type: ignore[arg-type]
        )


def test_generator_uses_tier_blue_count_in_prompt_payload() -> None:
    low = TraceTest1Generator(seed=51).next_problem(difficulty=0.10).payload
    high = TraceTest1Generator(seed=51).next_problem(difficulty=0.95).payload

    assert isinstance(low, TraceTest1PromptPlan)
    assert isinstance(high, TraceTest1PromptPlan)
    assert len(low.blue_plans) == 1
    assert len(high.blue_plans) == 4


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

    assert engine.phase is Phase.RESULTS
    assert engine.submit_answer("LEFT") is False


def test_trial_transitions_from_observe_to_answer_open() -> None:
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
    assert payload.trial_stage is TraceTest1TrialStage.OBSERVE
    assert payload.answer_open_progress == pytest.approx(0.36)

    clock.advance(0.35)
    engine.update()
    payload = engine.snapshot().payload
    assert isinstance(payload, TraceTest1Payload)
    assert payload.trial_stage is TraceTest1TrialStage.OBSERVE
    assert 0.34 <= payload.observe_progress <= 0.36

    clock.advance(0.02)
    engine.update()
    payload = engine.snapshot().payload
    assert isinstance(payload, TraceTest1Payload)
    assert payload.trial_stage is TraceTest1TrialStage.ANSWER_OPEN


def test_submit_answer_blocked_until_answer_open() -> None:
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
    assert engine.submit_answer(str(payload.correct_code)) is False

    clock.advance(0.35)
    engine.update()
    assert engine.submit_answer(str(payload.correct_code)) is False

    clock.advance(0.02)
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


def test_left_and_right_use_exact_90_degree_turns() -> None:
    left_prompt = _manual_prompt(command=TraceTest1Command.LEFT)
    right_prompt = _manual_prompt(command=TraceTest1Command.RIGHT)

    left_scene = trace_test_1_scene_frames(prompt=left_prompt, progress=0.80)
    right_scene = trace_test_1_scene_frames(prompt=right_prompt, progress=0.80)

    assert left_scene.red_frame.travel_heading_deg == pytest.approx(270.0, abs=0.01)
    assert left_scene.red_frame.position[1] == pytest.approx(12.0, abs=0.01)
    assert left_scene.red_frame.position[0] < 0.0
    assert left_scene.red_frame.attitude.roll_deg == pytest.approx(0.0, abs=0.01)

    assert right_scene.red_frame.travel_heading_deg == pytest.approx(90.0, abs=0.01)
    assert right_scene.red_frame.position[1] == pytest.approx(12.0, abs=0.01)
    assert right_scene.red_frame.position[0] > 0.0
    assert right_scene.red_frame.attitude.roll_deg == pytest.approx(0.0, abs=0.01)


def test_left_and_right_turns_hold_the_corner_before_moving_sideways() -> None:
    left_prompt = _manual_prompt(command=TraceTest1Command.LEFT)
    right_prompt = _manual_prompt(command=TraceTest1Command.RIGHT)

    left_corner = trace_test_1_scene_frames(prompt=left_prompt, progress=0.44).red_frame
    left_hold = trace_test_1_scene_frames(prompt=left_prompt, progress=0.48).red_frame
    left_move = trace_test_1_scene_frames(prompt=left_prompt, progress=0.72).red_frame

    assert left_corner.position == pytest.approx(left_hold.position, abs=0.01)
    assert left_move.position[0] < left_hold.position[0]
    assert left_move.position[1] == pytest.approx(left_hold.position[1], abs=0.01)

    right_corner = trace_test_1_scene_frames(prompt=right_prompt, progress=0.44).red_frame
    right_hold = trace_test_1_scene_frames(prompt=right_prompt, progress=0.48).red_frame
    right_move = trace_test_1_scene_frames(prompt=right_prompt, progress=0.72).red_frame

    assert right_corner.position == pytest.approx(right_hold.position, abs=0.01)
    assert right_move.position[0] > right_hold.position[0]
    assert right_move.position[1] == pytest.approx(right_hold.position[1], abs=0.01)


def test_push_and_pull_keep_forward_motion_while_changing_altitude() -> None:
    push_prompt = _manual_prompt(command=TraceTest1Command.PUSH)
    pull_prompt = _manual_prompt(command=TraceTest1Command.PULL)

    push_start = trace_test_1_scene_frames(prompt=push_prompt, progress=0.30).red_frame
    push_mid = trace_test_1_scene_frames(prompt=push_prompt, progress=0.82).red_frame
    pull_start = trace_test_1_scene_frames(prompt=pull_prompt, progress=0.30).red_frame
    pull_mid = trace_test_1_scene_frames(prompt=pull_prompt, progress=0.82).red_frame

    assert push_mid.position[1] > push_start.position[1]
    assert push_mid.position[2] < push_start.position[2]
    assert push_mid.travel_heading_deg == pytest.approx(0.0, abs=0.01)

    assert pull_mid.position[1] > pull_start.position[1]
    assert pull_mid.position[2] > pull_start.position[2]
    assert pull_mid.travel_heading_deg == pytest.approx(0.0, abs=0.01)


def test_stream_continues_from_current_world_state_after_answer() -> None:
    clock = FakeClock()
    engine = build_trace_test_1_test(
        clock=clock,
        seed=93,
        difficulty=0.5,
        config=TraceTest1Config(
            scored_duration_s=6.0,
            practice_questions=2,
            practice_observe_s=1.0,
            scored_observe_s=1.0,
        ),
    )
    engine.start_practice()

    clock.advance(0.40)
    engine.update()
    before = engine.snapshot().payload
    assert isinstance(before, TraceTest1Payload)
    answered_position = before.scene.red_frame.position
    correct_code = before.correct_code

    assert engine.submit_answer(str(correct_code)) is True
    after = engine.snapshot().payload
    assert isinstance(after, TraceTest1Payload)
    assert after.prompt_index == before.prompt_index + 1
    assert after.scene.red_frame.position[0] == pytest.approx(answered_position[0], abs=0.01)
    assert after.scene.red_frame.position[1] == pytest.approx(answered_position[1], abs=0.01)
    assert after.scene.red_frame.position[2] == pytest.approx(answered_position[2], abs=0.01)


def test_auto_miss_records_event_and_keeps_stream_continuous() -> None:
    clock = FakeClock()
    engine = build_trace_test_1_test(
        clock=clock,
        seed=111,
        difficulty=0.5,
        config=TraceTest1Config(
            scored_duration_s=6.0,
            practice_questions=2,
            practice_observe_s=0.5,
            scored_observe_s=0.5,
        ),
    )
    engine.start_practice()

    current_prompt = engine._current_prompt
    assert isinstance(current_prompt, TraceTest1PromptPlan)
    expected_end = trace_test_1_scene_frames(prompt=current_prompt, progress=1.0).red_frame.position

    clock.advance(0.5)
    engine.update()

    assert engine._events[-1].user_answer == 0
    assert engine._events[-1].raw == ""
    payload = engine.snapshot().payload
    assert isinstance(payload, TraceTest1Payload)
    assert payload.prompt_index == current_prompt.prompt_index + 1
    assert payload.scene.red_frame.position[0] == pytest.approx(expected_end[0], abs=0.01)
    assert payload.scene.red_frame.position[1] == pytest.approx(expected_end[1], abs=0.01)
    assert payload.scene.red_frame.position[2] == pytest.approx(expected_end[2], abs=0.01)


def test_no_reset_between_practice_and_scored_start() -> None:
    clock = FakeClock()
    engine = build_trace_test_1_test(
        clock=clock,
        seed=139,
        difficulty=0.5,
        config=TraceTest1Config(
            scored_duration_s=6.0,
            practice_questions=1,
            practice_observe_s=1.0,
            scored_observe_s=1.0,
        ),
    )
    engine.start_practice()
    clock.advance(0.40)
    engine.update()
    payload = engine.snapshot().payload
    assert isinstance(payload, TraceTest1Payload)
    practice_position = payload.scene.red_frame.position

    assert engine.submit_answer(str(payload.correct_code)) is True
    assert engine.phase is Phase.PRACTICE_DONE

    engine.start_scored()
    scored_payload = engine.snapshot().payload
    assert isinstance(scored_payload, TraceTest1Payload)
    assert scored_payload.scene.red_frame.position[0] == pytest.approx(practice_position[0], abs=0.01)
    assert scored_payload.scene.red_frame.position[1] == pytest.approx(practice_position[1], abs=0.01)
    assert scored_payload.scene.red_frame.position[2] == pytest.approx(practice_position[2], abs=0.01)


def test_red_stays_inside_safe_box_across_tiers() -> None:
    for difficulty in (0.10, 0.50, 0.80, 0.95):
        gen = TraceTest1Generator(seed=int(1000 + (difficulty * 100.0)))
        for _ in range(16):
            prompt = gen.next_problem(difficulty=difficulty).payload
            assert isinstance(prompt, TraceTest1PromptPlan)
            for progress in (0.0, prompt.answer_open_progress, 1.0):
                red = trace_test_1_scene_frames(prompt=prompt, progress=progress).red_frame
                nx, ny = trace_test_1_normalized_position(red.position)
                assert 0.15 <= nx <= 0.85
                assert 0.15 <= ny <= 0.85
                assert 4.0 <= red.position[1] <= 48.0
            gen.commit_prompt(prompt=prompt, progress=1.0)


def test_all_trace_test_1_headings_stay_on_cardinal_90_degree_axes() -> None:
    gen = TraceTest1Generator(seed=1601)
    for _ in range(20):
        prompt = gen.next_problem(difficulty=0.95).payload
        assert isinstance(prompt, TraceTest1PromptPlan)
        for progress in (0.0, 0.2, prompt.answer_open_progress, 0.6, 1.0):
            scene = trace_test_1_scene_frames(prompt=prompt, progress=progress)
            frames = (scene.red_frame, *scene.blue_frames)
            for frame in frames:
                assert frame.travel_heading_deg % 90.0 == pytest.approx(0.0, abs=0.01)
        gen.commit_prompt(prompt=prompt, progress=1.0)
