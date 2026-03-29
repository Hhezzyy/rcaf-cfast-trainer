from __future__ import annotations

from dataclasses import dataclass
import math

import pytest

from cfast_trainer.cognitive_core import Phase
from cfast_trainer.trace_lattice import TraceLatticeAction, trace_lattice_state
from cfast_trainer.trace_test_1 import (
    TraceTest1AircraftPlan,
    TraceTest1Command,
    TraceTest1Config,
    TraceTest1Generator,
    TraceTest1Payload,
    TraceTest1PromptPlan,
    TraceTest1TrialStage,
    _tt1_action_for_command,
    _tt1_aircraft_state_from_lattice_state,
    _tt1_build_lattice_path,
    _tt1_command_step_index,
    build_trace_test_1_test,
    trace_test_1_answer_code,
    trace_test_1_difficulty_tier,
    trace_test_1_normalized_position,
    trace_test_1_scene_frames,
)
from cfast_trainer.trace_test_1_gl import project_scene_position


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
    del heading_deg
    state = {
        TraceTest1Command.LEFT: trace_lattice_state(col=4, row=1, level=2, forward=(0, 1, 0), up=(0, 0, 1)),
        TraceTest1Command.RIGHT: trace_lattice_state(col=2, row=1, level=2, forward=(0, 1, 0), up=(0, 0, 1)),
        TraceTest1Command.PUSH: trace_lattice_state(col=3, row=1, level=3, forward=(0, 1, 0), up=(0, 0, 1)),
        TraceTest1Command.PULL: trace_lattice_state(col=3, row=1, level=1, forward=(0, 1, 0), up=(0, 0, 1)),
    }[command]
    return TraceTest1PromptPlan(
        prompt_index=0,
        answer_open_progress=0.36,
        speed_multiplier=1.15,
        red_plan=TraceTest1AircraftPlan(
            start_state=_tt1_aircraft_state_from_lattice_state(state),
            command=command,
            lead_distance=5.0,
            maneuver_distance=4.0,
            altitude_delta=2.0 if command is TraceTest1Command.PULL else -2.0,
            lattice_start=state,
            lattice_actions=(
                TraceLatticeAction.STRAIGHT,
                _tt1_action_for_command(command),
                TraceLatticeAction.STRAIGHT,
            ),
        ),
        blue_plans=(),
    )


def _angle_delta_deg(a: float, b: float) -> float:
    return ((float(a) - float(b) + 180.0) % 360.0) - 180.0


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
    assert trace_test_1_difficulty_tier(difficulty=0.10).speed_multiplier == pytest.approx(0.95)
    assert trace_test_1_difficulty_tier(difficulty=0.10).answer_open_progress == pytest.approx(0.44)
    assert trace_test_1_difficulty_tier(difficulty=0.10).immediate_turn_chance == pytest.approx(0.0)

    assert trace_test_1_difficulty_tier(difficulty=0.50).blue_count == 2
    assert trace_test_1_difficulty_tier(difficulty=0.50).speed_multiplier == pytest.approx(1.12)
    assert trace_test_1_difficulty_tier(difficulty=0.50).answer_open_progress == pytest.approx(0.37)
    assert trace_test_1_difficulty_tier(difficulty=0.50).immediate_turn_chance == pytest.approx(0.0)

    assert trace_test_1_difficulty_tier(difficulty=0.80).blue_count == 4
    assert trace_test_1_difficulty_tier(difficulty=0.80).speed_multiplier == pytest.approx(1.38)
    assert trace_test_1_difficulty_tier(difficulty=0.80).answer_open_progress == pytest.approx(0.30)
    assert trace_test_1_difficulty_tier(difficulty=0.80).immediate_turn_chance == pytest.approx(0.78)

    assert trace_test_1_difficulty_tier(difficulty=0.95).blue_count == 5
    assert trace_test_1_difficulty_tier(difficulty=0.95).speed_multiplier == pytest.approx(1.65)
    assert trace_test_1_difficulty_tier(difficulty=0.95).answer_open_progress == pytest.approx(0.24)
    assert trace_test_1_difficulty_tier(difficulty=0.95).immediate_turn_chance == pytest.approx(0.94)


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
    assert len(high.blue_plans) == 5


def test_easy_tier_red_prompt_always_moves_one_step_before_turning() -> None:
    gen = TraceTest1Generator(seed=141)

    for _ in range(20):
        payload = gen.next_problem(difficulty=0.10).payload
        assert isinstance(payload, TraceTest1PromptPlan)
        assert payload.red_plan.lattice_actions[0] is TraceLatticeAction.STRAIGHT
        assert payload.red_plan.lattice_actions[1] is _tt1_action_for_command(payload.red_plan.command)


def test_hard_tier_red_prompt_usually_turns_immediately() -> None:
    gen = TraceTest1Generator(seed=142)
    immediate_turns = 0

    for _ in range(24):
        payload = gen.next_problem(difficulty=0.95).payload
        assert isinstance(payload, TraceTest1PromptPlan)
        if payload.red_plan.lattice_actions[0] is _tt1_action_for_command(payload.red_plan.command):
            immediate_turns += 1

    assert immediate_turns >= 18


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
    assert payload.answer_open_progress == pytest.approx(0.37)

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


def test_left_and_right_lattice_paths_finish_in_expected_heading_family() -> None:
    left_prompt = _manual_prompt(command=TraceTest1Command.LEFT)
    right_prompt = _manual_prompt(command=TraceTest1Command.RIGHT)

    left_mid = trace_test_1_scene_frames(prompt=left_prompt, progress=0.80).red_frame
    left_end = trace_test_1_scene_frames(prompt=left_prompt, progress=1.0).red_frame
    right_mid = trace_test_1_scene_frames(prompt=right_prompt, progress=0.80).red_frame
    right_end = trace_test_1_scene_frames(prompt=right_prompt, progress=1.0).red_frame

    assert left_mid.position[0] < 0.0
    assert left_mid.position[1] >= left_prompt.red_plan.start_state.position[1]
    assert left_mid.attitude.roll_deg == pytest.approx(0.0)
    assert abs(_angle_delta_deg(left_mid.travel_heading_deg, 270.0)) <= 2.0
    assert abs(_angle_delta_deg(left_end.travel_heading_deg, 270.0)) <= 2.0

    assert right_mid.position[0] > 0.0
    assert right_mid.position[1] >= right_prompt.red_plan.start_state.position[1]
    assert right_mid.attitude.roll_deg == pytest.approx(0.0)
    assert _angle_delta_deg(right_mid.travel_heading_deg, 0.0) > 0.0
    assert abs(_angle_delta_deg(right_end.travel_heading_deg, 90.0)) <= 2.0


def test_left_and_right_turn_paths_rotate_in_place_before_translation() -> None:
    left_prompt = _manual_prompt(command=TraceTest1Command.LEFT)
    right_prompt = _manual_prompt(command=TraceTest1Command.RIGHT)

    left_positions = [
        trace_test_1_scene_frames(prompt=left_prompt, progress=progress).red_frame.position
        for progress in (0.36, 0.40, 0.68, 1.0)
    ]
    right_positions = [
        trace_test_1_scene_frames(prompt=right_prompt, progress=progress).red_frame.position
        for progress in (0.36, 0.40, 0.68, 1.0)
    ]

    assert left_positions[0] == pytest.approx(left_positions[1])
    assert left_positions[2][0] < left_positions[1][0]
    assert left_positions[3][0] < left_positions[2][0]
    assert left_positions[2][1] == pytest.approx(left_positions[1][1], abs=0.01)

    assert right_positions[0] == pytest.approx(right_positions[1])
    assert right_positions[2][0] > right_positions[1][0]
    assert right_positions[3][0] > right_positions[2][0]
    assert right_positions[2][1] == pytest.approx(right_positions[1][1], abs=0.01)


def test_push_and_pull_pitch_in_place_before_vertical_translation() -> None:
    push_prompt = _manual_prompt(command=TraceTest1Command.PUSH)
    pull_prompt = _manual_prompt(command=TraceTest1Command.PULL)

    push_start = trace_test_1_scene_frames(prompt=push_prompt, progress=0.36).red_frame
    push_turn = trace_test_1_scene_frames(prompt=push_prompt, progress=0.40).red_frame
    push_mid = trace_test_1_scene_frames(prompt=push_prompt, progress=0.82).red_frame
    pull_start = trace_test_1_scene_frames(prompt=pull_prompt, progress=0.36).red_frame
    pull_turn = trace_test_1_scene_frames(prompt=pull_prompt, progress=0.40).red_frame
    pull_mid = trace_test_1_scene_frames(prompt=pull_prompt, progress=0.82).red_frame

    assert push_turn.position == pytest.approx(push_start.position)
    assert push_mid.position[1] == pytest.approx(push_start.position[1], abs=0.01)
    assert push_mid.position[2] < push_start.position[2]
    assert push_mid.travel_heading_deg == pytest.approx(0.0, abs=0.01)
    assert push_mid.attitude.pitch_deg < 0.0

    assert pull_turn.position == pytest.approx(pull_start.position)
    assert pull_mid.position[1] == pytest.approx(pull_start.position[1], abs=0.01)
    assert pull_mid.position[2] > pull_start.position[2]
    assert pull_mid.travel_heading_deg == pytest.approx(0.0, abs=0.01)
    assert pull_mid.attitude.pitch_deg > 0.0


def test_submit_answer_deals_next_lattice_prompt_without_repeating_command() -> None:
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
    correct_code = before.correct_code

    assert engine.submit_answer(str(correct_code)) is True
    after = engine.snapshot().payload
    assert isinstance(after, TraceTest1Payload)
    assert after.prompt_index == before.prompt_index + 1
    assert engine._current_prompt is not None
    assert engine._current_prompt.red_plan.lattice_start is not None
    assert engine._current_prompt.red_plan.lattice_actions == (
        TraceLatticeAction.STRAIGHT,
        _tt1_action_for_command(engine._current_prompt.red_plan.command),
        TraceLatticeAction.STRAIGHT,
    )
    assert after.active_command is not before.active_command


def test_auto_miss_records_event_and_deals_next_lattice_prompt() -> None:
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
    assert engine._current_prompt is not None
    assert engine._current_prompt.red_plan.lattice_start is not None
    assert engine._current_prompt.red_plan.lattice_actions[1] is _tt1_action_for_command(
        engine._current_prompt.red_plan.command
    )
    assert payload.scene.red_frame.position != pytest.approx(expected_end, abs=0.01)


def test_scored_start_deals_valid_lattice_prompt_after_practice() -> None:
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

    assert engine.submit_answer(str(payload.correct_code)) is True
    assert engine.phase is Phase.PRACTICE_DONE

    engine.start_scored()
    scored_payload = engine.snapshot().payload
    assert isinstance(scored_payload, TraceTest1Payload)
    assert engine._current_prompt is not None
    assert engine._current_prompt.red_plan.lattice_start is not None
    assert scored_payload.correct_code in {1, 2, 3, 4}


def test_red_lattice_paths_stay_on_screen_across_tiers() -> None:
    for difficulty in (0.10, 0.50, 0.80, 0.95):
        gen = TraceTest1Generator(seed=int(1000 + (difficulty * 100.0)))
        for _ in range(16):
            prompt = gen.next_problem(difficulty=difficulty).payload
            assert isinstance(prompt, TraceTest1PromptPlan)
            path = _tt1_build_lattice_path(prompt.red_plan)
            assert path is not None
            assert path.steps[_tt1_command_step_index(prompt.red_plan)].effective_action is _tt1_action_for_command(
                prompt.red_plan.command
            )
            for progress in (0.0, prompt.answer_open_progress, 1.0):
                red = trace_test_1_scene_frames(prompt=prompt, progress=progress).red_frame
                center, _scale = project_scene_position(red.position, size=(960, 540))
                assert 0.0 <= center[0] <= 960.0
                assert 0.0 <= center[1] <= 540.0
                assert 0.0 <= red.position[2] <= 28.0
            gen.commit_prompt(prompt=prompt, progress=1.0)


def test_all_trace_test_1_frames_keep_finite_motion_state() -> None:
    gen = TraceTest1Generator(seed=1601)
    for _ in range(20):
        prompt = gen.next_problem(difficulty=0.95).payload
        assert isinstance(prompt, TraceTest1PromptPlan)
        for progress in (0.0, 0.2, prompt.answer_open_progress, 0.6, 1.0):
            scene = trace_test_1_scene_frames(prompt=prompt, progress=progress)
            frames = (scene.red_frame, *scene.blue_frames)
            for frame in frames:
                assert all(math.isfinite(value) for value in frame.position)
                assert math.isfinite(frame.travel_heading_deg)
                assert math.isfinite(frame.attitude.pitch_deg)
                assert math.isfinite(frame.attitude.roll_deg)
                assert math.isfinite(frame.attitude.yaw_deg)
        gen.commit_prompt(prompt=prompt, progress=1.0)
