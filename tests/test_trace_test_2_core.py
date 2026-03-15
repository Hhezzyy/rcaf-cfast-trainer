from __future__ import annotations

from dataclasses import dataclass

import pytest

from cfast_trainer.cognitive_core import Phase
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


def test_generated_tracks_use_only_axis_aligned_segments() -> None:
    payload = TraceTest2Generator(seed=29).next_problem(difficulty=0.55).payload
    assert isinstance(payload, TraceTest2Payload)

    for track in payload.aircraft:
        start, middle, end = track.waypoints
        first_axes = {
            axis
            for axis, delta in (
                ("x", middle.x - start.x),
                ("y", middle.y - start.y),
                ("z", middle.z - start.z),
            )
            if abs(delta) > 1e-6
        }
        second_axes = {
            axis
            for axis, delta in (
                ("x", end.x - middle.x),
                ("y", end.y - middle.y),
                ("z", end.z - middle.z),
            )
            if abs(delta) > 1e-6
        }

        assert len(first_axes) == 1
        assert len(second_axes) == 1
        assert first_axes == {"y"}
        if track.motion_kind is TraceTest2MotionKind.STRAIGHT:
            assert second_axes == {"y"}
        elif track.motion_kind in (TraceTest2MotionKind.LEFT, TraceTest2MotionKind.RIGHT):
            assert second_axes == {"x"}
        else:
            assert second_axes == {"z"}


def test_track_position_is_linear_across_the_two_straight_segments() -> None:
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

    assert quarter.x == pytest.approx(0.0, abs=0.01)
    assert quarter.y == pytest.approx(5.0, abs=0.01)
    assert half.x == pytest.approx(0.0, abs=0.01)
    assert half.y == pytest.approx(10.0, abs=0.01)
    assert three_quarters.x == pytest.approx(5.0, abs=0.01)
    assert three_quarters.y == pytest.approx(10.0, abs=0.01)


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
