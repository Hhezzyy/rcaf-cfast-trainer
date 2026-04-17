from __future__ import annotations

from dataclasses import dataclass
import math

import pytest

from cfast_trainer.cognitive_core import Phase
from cfast_trainer.trace_scene_3d import (
    build_trace_test_2_scene3d,
    trace_test_2_track_sample_points,
)
from cfast_trainer.trace_lattice import DEFAULT_TRACE_LATTICE_SPEC, TraceLatticeAction
from cfast_trainer.trace_test_2 import (
    TraceTest2AircraftTrack,
    TraceTest2Config,
    TraceTest2Generator,
    TraceTest2MotionKind,
    TraceTest2Point3,
    TraceTest2Payload,
    TraceTest2QuestionKind,
    TraceTest2TrialStage,
    build_trace_test_2_test,
    trace_test_2_track_position,
    trace_test_2_track_tangent,
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


def test_trace_test_2_scene3d_builder_is_deterministic_and_practice_adds_ghosts() -> None:
    payload = TraceTest2Generator(seed=72).next_problem(difficulty=0.58).payload

    assert isinstance(payload, TraceTest2Payload)

    scored_snapshot = build_trace_test_2_scene3d(payload=payload, practice_mode=False)
    repeat_snapshot = build_trace_test_2_scene3d(payload=payload, practice_mode=False)
    practice_snapshot = build_trace_test_2_scene3d(payload=payload, practice_mode=True)

    assert scored_snapshot == repeat_snapshot
    assert len(scored_snapshot.aircraft) == len(payload.aircraft)
    assert practice_snapshot.aircraft == scored_snapshot.aircraft
    assert len(practice_snapshot.ghosts) > 0


def test_generated_problem_has_four_color_options_and_guide_question_kinds() -> None:
    payload = TraceTest2Generator(seed=17).next_problem(difficulty=0.5).payload

    assert isinstance(payload, TraceTest2Payload)
    assert len(payload.aircraft) == 4
    assert tuple(track.color_name for track in payload.aircraft) == ("Red", "Blue", "Silver", "Yellow")
    assert tuple(option.label for option in payload.options) == ("Red", "Blue", "Silver", "Yellow")
    assert tuple(option.code for option in payload.options) == (1, 2, 3, 4)
    assert payload.question_kind in {
        TraceTest2QuestionKind.NO_DIRECTION_CHANGE,
        TraceTest2QuestionKind.TURNED_LEFT,
        TraceTest2QuestionKind.TURNED_RIGHT,
        TraceTest2QuestionKind.ENDED_LEFTMOST,
        TraceTest2QuestionKind.ENDED_HIGHEST,
    }
    assert 1 <= payload.correct_code <= 4


def test_generator_emits_only_guide_recall_kinds() -> None:
    gen = TraceTest2Generator(seed=71)
    seen: set[TraceTest2QuestionKind] = set()

    for _ in range(40):
        payload = gen.next_problem(difficulty=0.61).payload
        assert isinstance(payload, TraceTest2Payload)
        seen.add(payload.question_kind)

    assert seen == {
        TraceTest2QuestionKind.NO_DIRECTION_CHANGE,
        TraceTest2QuestionKind.TURNED_LEFT,
        TraceTest2QuestionKind.TURNED_RIGHT,
        TraceTest2QuestionKind.ENDED_LEFTMOST,
        TraceTest2QuestionKind.ENDED_HIGHEST,
    }


def test_allowed_question_kinds_none_matches_default_generator_sequence() -> None:
    default = TraceTest2Generator(seed=611)
    explicit = TraceTest2Generator(seed=611, allowed_question_kinds=None)

    default_kinds = [default.next_problem(difficulty=0.58).payload.question_kind for _ in range(12)]
    explicit_kinds = [explicit.next_problem(difficulty=0.58).payload.question_kind for _ in range(12)]

    assert explicit_kinds == default_kinds


def test_allowed_question_kinds_filter_limits_emitted_questions() -> None:
    gen = TraceTest2Generator(
        seed=612,
        allowed_question_kinds=(
            TraceTest2QuestionKind.TURNED_LEFT,
            TraceTest2QuestionKind.TURNED_RIGHT,
        ),
    )

    for _ in range(24):
        payload = gen.next_problem(difficulty=0.6).payload
        assert isinstance(payload, TraceTest2Payload)
        assert payload.question_kind in {
            TraceTest2QuestionKind.TURNED_LEFT,
            TraceTest2QuestionKind.TURNED_RIGHT,
        }


def test_allowed_question_kind_validation_rejects_empty_and_invalid_values() -> None:
    with pytest.raises(ValueError, match="allowed_question_kinds must not be empty"):
        build_trace_test_2_test(
            clock=FakeClock(),
            seed=613,
            difficulty=0.5,
            config=TraceTest2Config(allowed_question_kinds=()),
        )

    with pytest.raises(ValueError, match="Unknown Trace Test 2 question kind"):
        build_trace_test_2_test(
            clock=FakeClock(),
            seed=614,
            difficulty=0.5,
            config=TraceTest2Config(
                allowed_question_kinds=("bogus",),  # type: ignore[arg-type]
            ),
        )


def test_generated_scene_contains_unique_answer_facts() -> None:
    payload = TraceTest2Generator(seed=33).next_problem(difficulty=0.6).payload
    assert isinstance(payload, TraceTest2Payload)

    straight = [track.code for track in payload.aircraft if not track.direction_changed]
    left = [track.code for track in payload.aircraft if track.motion_kind is TraceTest2MotionKind.LEFT]
    right = [track.code for track in payload.aircraft if track.motion_kind is TraceTest2MotionKind.RIGHT]
    leftmost = min(payload.aircraft, key=lambda track: track.ended_screen_x).code
    highest = max(payload.aircraft, key=lambda track: track.ended_altitude_z).code

    assert len(straight) == 1
    assert len(left) == 1
    assert len(right) == 1
    assert len({straight[0], left[0], right[0], leftmost, highest}) >= 4


def test_track_sample_points_cover_start_mid_and_end_positions() -> None:
    payload = TraceTest2Generator(seed=33).next_problem(difficulty=0.6).payload
    assert isinstance(payload, TraceTest2Payload)

    for track in payload.aircraft:
        samples = trace_test_2_track_sample_points(track=track)
        assert len(samples) >= 3
        assert samples[0] == pytest.approx(
            (track.waypoints[0].x, track.waypoints[0].y, track.waypoints[0].z)
        )
        end = track.waypoints[-1]
        assert samples[-1] == pytest.approx((end.x, end.y, end.z))


def test_easy_difficulty_non_straight_tracks_take_one_forward_step_before_turning() -> None:
    payload = TraceTest2Generator(seed=34).next_problem(difficulty=0.10).payload
    assert isinstance(payload, TraceTest2Payload)

    for track in payload.aircraft:
        assert track.lattice_path is not None
        if track.motion_kind is TraceTest2MotionKind.STRAIGHT:
            continue
        assert track.lattice_path.steps[0].effective_action is TraceLatticeAction.STRAIGHT


def test_hard_difficulty_non_straight_tracks_turn_earlier_and_cover_more_steps() -> None:
    low = TraceTest2Generator(seed=35).next_problem(difficulty=0.10).payload
    high = TraceTest2Generator(seed=35).next_problem(difficulty=0.95).payload

    assert isinstance(low, TraceTest2Payload)
    assert isinstance(high, TraceTest2Payload)

    low_by_motion = {track.motion_kind: track for track in low.aircraft}
    high_by_motion = {track.motion_kind: track for track in high.aircraft}
    expected_first_actions = {
        TraceTest2MotionKind.LEFT: TraceLatticeAction.LEFT,
        TraceTest2MotionKind.RIGHT: TraceLatticeAction.RIGHT,
        TraceTest2MotionKind.CLIMB: TraceLatticeAction.PULL,
    }

    for motion_kind, expected_action in expected_first_actions.items():
        assert low_by_motion[motion_kind].lattice_path is not None
        assert high_by_motion[motion_kind].lattice_path is not None
        assert low_by_motion[motion_kind].lattice_path.steps[0].effective_action is TraceLatticeAction.STRAIGHT
        assert high_by_motion[motion_kind].lattice_path.steps[0].effective_action is expected_action
        assert len(high_by_motion[motion_kind].lattice_path.steps) > len(low_by_motion[motion_kind].lattice_path.steps)

    assert low_by_motion[TraceTest2MotionKind.STRAIGHT].lattice_path is not None
    assert high_by_motion[TraceTest2MotionKind.STRAIGHT].lattice_path is not None
    assert len(high_by_motion[TraceTest2MotionKind.STRAIGHT].lattice_path.steps) > len(
        low_by_motion[TraceTest2MotionKind.STRAIGHT].lattice_path.steps
    )


def test_generated_tracks_use_lattice_paths_that_stay_in_bounds() -> None:
    payload = TraceTest2Generator(seed=33).next_problem(difficulty=0.6).payload
    assert isinstance(payload, TraceTest2Payload)

    for track in payload.aircraft:
        assert track.lattice_path is not None
        assert track.lattice_path.start_state.node.col in range(DEFAULT_TRACE_LATTICE_SPEC.cols)
        assert track.lattice_path.start_state.node.row in range(DEFAULT_TRACE_LATTICE_SPEC.rows)
        assert track.lattice_path.start_state.node.level in range(DEFAULT_TRACE_LATTICE_SPEC.levels)
        for step in track.lattice_path.steps:
            assert step.end_state.node.col in range(DEFAULT_TRACE_LATTICE_SPEC.cols)
            assert step.end_state.node.row in range(DEFAULT_TRACE_LATTICE_SPEC.rows)
            assert step.end_state.node.level in range(DEFAULT_TRACE_LATTICE_SPEC.levels)


def test_generated_tracks_preserve_role_specific_smooth_tangent_behavior() -> None:
    payload = TraceTest2Generator(seed=29).next_problem(difficulty=0.55).payload
    assert isinstance(payload, TraceTest2Payload)

    for track in payload.aircraft:
        start_tangent = trace_test_2_track_tangent(track=track, progress=0.0)
        mid_tangent = trace_test_2_track_tangent(track=track, progress=0.5)
        end_tangent = trace_test_2_track_tangent(track=track, progress=1.0)

        assert all(math.isfinite(component) for component in (*start_tangent, *mid_tangent, *end_tangent))
        assert start_tangent[1] > 0.0
        if track.motion_kind is TraceTest2MotionKind.STRAIGHT:
            assert track.direction_changed is False
            assert start_tangent[0] == pytest.approx(0.0, abs=1e-6)
            assert mid_tangent[0] == pytest.approx(0.0, abs=1e-6)
            assert end_tangent[0] == pytest.approx(0.0, abs=1e-6)
            assert start_tangent[2] == pytest.approx(0.0, abs=1e-6)
            assert mid_tangent[2] == pytest.approx(0.0, abs=1e-6)
            assert end_tangent[2] == pytest.approx(0.0, abs=1e-6)
        elif track.motion_kind in (TraceTest2MotionKind.LEFT, TraceTest2MotionKind.RIGHT):
            assert track.direction_changed is True
            assert mid_tangent[2] == pytest.approx(0.0, abs=1e-6)
            assert end_tangent[2] == pytest.approx(0.0, abs=1e-6)
            expected_sign = -1.0 if track.motion_kind is TraceTest2MotionKind.LEFT else 1.0
            assert start_tangent[0] == pytest.approx(0.0, abs=1e-6)
            assert mid_tangent[0] * expected_sign > 0.0
            assert end_tangent[0] * expected_sign > 0.0
        else:
            assert track.direction_changed is True
            assert start_tangent[0] == pytest.approx(0.0, abs=1e-6)
            assert mid_tangent[0] == pytest.approx(0.0, abs=1e-6)
            assert end_tangent[0] == pytest.approx(0.0, abs=1e-6)
            assert mid_tangent[2] > 0.0
            assert end_tangent[2] > 0.0


def test_generated_turning_tracks_pivot_before_translating_on_turn_steps() -> None:
    payload = TraceTest2Generator(seed=29).next_problem(difficulty=0.55).payload
    assert isinstance(payload, TraceTest2Payload)

    for track in payload.aircraft:
        if track.motion_kind is TraceTest2MotionKind.STRAIGHT:
            continue
        assert track.lattice_path is not None
        turn_index = next(
            idx
            for idx, step in enumerate(track.lattice_path.steps)
            if step.effective_action is not TraceLatticeAction.STRAIGHT
        )
        step_count = len(track.lattice_path.steps)
        early = trace_test_2_track_position(
            track=track,
            progress=(float(turn_index) + 0.08) / float(step_count),
        )
        mid = trace_test_2_track_position(
            track=track,
            progress=(float(turn_index) + 0.20) / float(step_count),
        )
        late = trace_test_2_track_position(
            track=track,
            progress=(float(turn_index) + 0.60) / float(step_count),
        )

        assert mid == pytest.approx(early)
        if track.motion_kind in (TraceTest2MotionKind.LEFT, TraceTest2MotionKind.RIGHT):
            expected_sign = -1.0 if track.motion_kind is TraceTest2MotionKind.LEFT else 1.0
            assert (late.x - mid.x) * expected_sign > 0.0
            assert late.y == pytest.approx(mid.y)
        else:
            assert late.z > mid.z
            assert late.y == pytest.approx(mid.y)


def test_generated_turning_tracks_keep_tangent_direction_consistent_with_motion_family() -> None:
    payload = TraceTest2Generator(seed=29).next_problem(difficulty=0.55).payload
    assert isinstance(payload, TraceTest2Payload)

    for track in payload.aircraft:
        if track.motion_kind is TraceTest2MotionKind.STRAIGHT:
            continue
        assert track.lattice_path is not None
        turn_index = next(
            idx
            for idx, step in enumerate(track.lattice_path.steps)
            if step.effective_action is not TraceLatticeAction.STRAIGHT
        )
        step_count = len(track.lattice_path.steps)
        mid_tangent = trace_test_2_track_tangent(
            track=track,
            progress=(float(turn_index) + 0.20) / float(step_count),
        )

        assert math.isfinite(mid_tangent[0])
        assert math.isfinite(mid_tangent[1])
        assert math.isfinite(mid_tangent[2])
        if track.motion_kind is TraceTest2MotionKind.LEFT:
            assert mid_tangent[0] < 0.0
        elif track.motion_kind is TraceTest2MotionKind.RIGHT:
            assert mid_tangent[0] > 0.0
        elif track.motion_kind is TraceTest2MotionKind.CLIMB:
            assert mid_tangent[2] > 0.0


def test_trace_test_2_start_positions_project_inside_the_visible_window() -> None:
    payload = TraceTest2Generator(seed=71).next_problem(difficulty=0.58).payload
    assert isinstance(payload, TraceTest2Payload)

    for track in payload.aircraft:
        start = trace_test_2_track_position(track=track, progress=0.0)
        assert -70.0 <= start.x <= 70.0
        assert start.y >= 40.0
        assert 0.0 <= start.z <= 40.0


def test_track_position_uses_quadratic_control_polygon_for_three_point_tracks() -> None:
    track = TraceTest2AircraftTrack(
        code=1,
        color_name="Red",
        color_rgb=(255, 0, 0),
        waypoints=(
            TraceTest2Point3(0.0, 0.0, 0.0),
            TraceTest2Point3(0.0, 10.0, 0.0),
            TraceTest2Point3(10.0, 10.0, 0.0),
        ),
        motion_kind=TraceTest2MotionKind.RIGHT,
        direction_changed=True,
        ended_screen_x=10.0,
        ended_altitude_z=0.0,
    )

    quarter = trace_test_2_track_position(track=track, progress=0.25)
    half = trace_test_2_track_position(track=track, progress=0.5)
    three_quarters = trace_test_2_track_position(track=track, progress=0.75)

    assert trace_test_2_track_position(track=track, progress=0.0) == track.waypoints[0]
    assert trace_test_2_track_position(track=track, progress=1.0) == track.waypoints[-1]
    assert quarter.x == pytest.approx(0.625, abs=0.01)
    assert quarter.y == pytest.approx(4.375, abs=0.01)
    assert half.x == pytest.approx(2.5, abs=0.01)
    assert half.y == pytest.approx(7.5, abs=0.01)
    assert three_quarters.x == pytest.approx(5.625, abs=0.01)
    assert three_quarters.y == pytest.approx(9.375, abs=0.01)
    assert trace_test_2_track_tangent(track=track, progress=0.5) == pytest.approx((10.0, 10.0, 0.0))


def test_generated_answer_facts_match_smooth_track_end_states() -> None:
    payload = TraceTest2Generator(seed=33).next_problem(difficulty=0.6).payload
    assert isinstance(payload, TraceTest2Payload)

    final_positions = {
        track.code: trace_test_2_track_position(track=track, progress=1.0)
        for track in payload.aircraft
    }

    assert min(payload.aircraft, key=lambda track: track.ended_screen_x).code == min(
        final_positions,
        key=lambda code: final_positions[code].x,
    )
    assert max(payload.aircraft, key=lambda track: track.ended_altitude_z).code == max(
        final_positions,
        key=lambda code: final_positions[code].z,
    )


def test_trial_transitions_from_observe_to_question_and_rejects_input_during_observe() -> None:
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

    payload = engine.snapshot().payload
    assert isinstance(payload, TraceTest2Payload)
    assert payload.trial_stage is TraceTest2TrialStage.OBSERVE
    assert engine.submit_answer("1") is False

    clock.advance(0.50)
    engine.update()
    payload = engine.snapshot().payload
    assert isinstance(payload, TraceTest2Payload)
    assert payload.trial_stage is TraceTest2TrialStage.OBSERVE
    assert 0.49 <= payload.observe_progress <= 0.51

    clock.advance(0.55)
    engine.update()
    payload = engine.snapshot().payload
    assert isinstance(payload, TraceTest2Payload)
    assert payload.trial_stage is TraceTest2TrialStage.QUESTION
    assert payload.observe_progress == pytest.approx(1.0)
    assert engine.submit_answer("1") is True
    assert engine.phase is Phase.PRACTICE_DONE


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
