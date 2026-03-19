from __future__ import annotations

from dataclasses import dataclass

import pytest

from cfast_trainer.ant_drills import AntDrillMode
from cfast_trainer.cognitive_core import Phase
from cfast_trainer.si_drills import (
    SiDrillConfig,
    build_si_aircraft_multiview_integration_drill,
    build_si_aircraft_grid_run_drill,
    build_si_continuation_prime_drill,
    build_si_landmark_anchor_drill,
    build_si_mixed_tempo_drill,
    build_si_pressure_run_drill,
    build_si_reconstruction_run_drill,
    build_si_route_anchor_drill,
    build_si_static_mixed_run_drill,
)
from cfast_trainer.spatial_integration import (
    SpatialIntegrationAnswerMode,
    SpatialIntegrationPart,
    SpatialIntegrationPayload,
    SpatialIntegrationQuestionKind,
    SpatialIntegrationTrialStage,
)


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


def _difficulty_for_level(level: int) -> float:
    return float(level - 1) / 9.0


def _wait_for_question(
    *,
    drill: object,
    clock: FakeClock,
    max_steps: int = 400,
    dt: float = 0.1,
) -> SpatialIntegrationPayload | None:
    for _ in range(max_steps):
        snap = drill.snapshot()
        if snap.phase is Phase.RESULTS:
            return None
        if snap.phase is Phase.PRACTICE_DONE:
            drill.start_scored()
            continue
        payload = snap.payload if isinstance(snap.payload, SpatialIntegrationPayload) else None
        if payload is not None and payload.trial_stage is SpatialIntegrationTrialStage.QUESTION:
            return payload
        clock.advance(dt)
        drill.update()
    raise AssertionError("Timed out waiting for Spatial Integration question stage.")


def _answer_current(payload: SpatialIntegrationPayload) -> str:
    if payload.answer_mode is SpatialIntegrationAnswerMode.GRID_CLICK:
        return payload.correct_answer_token
    return str(payload.correct_code)


@pytest.mark.parametrize(
    ("builder", "expected_part", "expected_kinds"),
    (
        (
            build_si_landmark_anchor_drill,
            SpatialIntegrationPart.STATIC,
            (SpatialIntegrationQuestionKind.LANDMARK_GRID,),
        ),
        (
            build_si_reconstruction_run_drill,
            SpatialIntegrationPart.STATIC,
            (SpatialIntegrationQuestionKind.SCENE_RECONSTRUCTION,),
        ),
        (
            build_si_static_mixed_run_drill,
            SpatialIntegrationPart.STATIC,
            (
                SpatialIntegrationQuestionKind.LANDMARK_GRID,
                SpatialIntegrationQuestionKind.SCENE_RECONSTRUCTION,
            ),
        ),
        (
            build_si_route_anchor_drill,
            SpatialIntegrationPart.AIRCRAFT,
            (SpatialIntegrationQuestionKind.AIRCRAFT_ROUTE_SELECTION,),
        ),
        (
            build_si_continuation_prime_drill,
            SpatialIntegrationPart.AIRCRAFT,
            (SpatialIntegrationQuestionKind.AIRCRAFT_CONTINUATION_SELECTION,),
        ),
        (
            build_si_aircraft_multiview_integration_drill,
            SpatialIntegrationPart.AIRCRAFT,
            (
                SpatialIntegrationQuestionKind.AIRCRAFT_ROUTE_SELECTION,
                SpatialIntegrationQuestionKind.AIRCRAFT_CONTINUATION_SELECTION,
                SpatialIntegrationQuestionKind.AIRCRAFT_LOCATION_GRID,
            ),
        ),
        (
            build_si_aircraft_grid_run_drill,
            SpatialIntegrationPart.AIRCRAFT,
            (SpatialIntegrationQuestionKind.AIRCRAFT_LOCATION_GRID,),
        ),
    ),
)
def test_si_focused_drills_emit_expected_part_and_question_families(
    builder,
    expected_part,
    expected_kinds,
) -> None:
    clock = FakeClock()
    drill = builder(
        clock=clock,
        seed=501,
        difficulty=0.55,
        mode=AntDrillMode.BUILD,
        config=SiDrillConfig(scored_duration_s=24.0),
    )

    drill.start_practice()
    payload = _wait_for_question(drill=drill, clock=clock)
    assert payload is not None

    assert drill.practice_questions == 1
    assert drill.scored_duration_s == pytest.approx(24.0)
    assert payload.part is expected_part
    assert payload.kind in expected_kinds


def test_si_mode_defaults_set_practice_scene_count_and_duration() -> None:
    build_clock = FakeClock()
    build_drill = build_si_landmark_anchor_drill(
        clock=build_clock,
        seed=17,
        difficulty=0.5,
        mode=AntDrillMode.BUILD,
    )
    tempo_clock = FakeClock()
    tempo_drill = build_si_mixed_tempo_drill(
        clock=tempo_clock,
        seed=17,
        difficulty=0.5,
        mode=AntDrillMode.TEMPO,
    )
    stress_clock = FakeClock()
    stress_drill = build_si_pressure_run_drill(
        clock=stress_clock,
        seed=17,
        difficulty=0.5,
        mode=AntDrillMode.STRESS,
    )

    assert build_drill.practice_questions == 1
    assert build_drill.scored_duration_s == pytest.approx(180.0)
    assert tempo_drill.practice_questions == 0
    assert tempo_drill.scored_duration_s == pytest.approx(150.0)
    assert stress_drill.practice_questions == 0
    assert stress_drill.scored_duration_s == pytest.approx(180.0)


def test_si_mixed_tempo_runs_static_then_aircraft() -> None:
    clock = FakeClock()
    drill = build_si_mixed_tempo_drill(
        clock=clock,
        seed=44,
        difficulty=0.5,
        mode=AntDrillMode.TEMPO,
        config=SiDrillConfig(scored_duration_s=40.0),
    )

    drill.start_practice()
    seen_parts: list[SpatialIntegrationPart] = []
    for _ in range(12):
        if drill.phase is Phase.RESULTS:
            break
        payload = _wait_for_question(drill=drill, clock=clock)
        if payload is None:
            break
        if not seen_parts or seen_parts[-1] is not payload.part:
            seen_parts.append(payload.part)
        assert drill.submit_answer(_answer_current(payload)) is True
        if drill.phase is Phase.RESULTS:
            break

    assert seen_parts[:2] == [
        SpatialIntegrationPart.STATIC,
        SpatialIntegrationPart.AIRCRAFT,
    ]


@pytest.mark.parametrize(
    "builder",
    (
        build_si_static_mixed_run_drill,
        build_si_aircraft_multiview_integration_drill,
    ),
)
def test_wave2_spatial_drills_levels_l2_l5_l8_are_materially_different(builder) -> None:
    def summarize(level: int) -> tuple[int, int]:
        clock = FakeClock()
        drill = builder(
            clock=clock,
            seed=733,
            difficulty=_difficulty_for_level(level),
            mode=AntDrillMode.BUILD,
            config=SiDrillConfig(scored_duration_s=24.0),
        )
        drill.start_practice()
        payload = _wait_for_question(drill=drill, clock=clock)
        assert payload is not None
        scene = drill._engine._current_scene
        assert scene is not None
        if builder is build_si_static_mixed_run_drill:
            question = next(
                item
                for item in scene.questions
                if item.kind is SpatialIntegrationQuestionKind.LANDMARK_GRID
            )
            return (len(question.answer_map_landmarks), len(scene.landmarks))
        question = next(
            item
            for item in scene.questions
            if item.kind is SpatialIntegrationQuestionKind.AIRCRAFT_LOCATION_GRID
        )
        return (len(question.answer_map_route_points), len(question.answer_map_landmarks))

    low_primary, low_secondary = summarize(2)
    mid_primary, mid_secondary = summarize(5)
    high_primary, high_secondary = summarize(8)

    assert low_primary >= mid_primary >= high_primary
    assert low_secondary >= mid_secondary >= high_secondary
    assert (low_primary, low_secondary) != (mid_primary, mid_secondary)
    assert (mid_primary, mid_secondary) != (high_primary, high_secondary)
