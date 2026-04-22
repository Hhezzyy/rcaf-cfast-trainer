from __future__ import annotations

from dataclasses import dataclass

import pytest

from cfast_trainer.cognitive_core import Problem
from cfast_trainer.target_recognition import (
    TargetRecognitionConfig,
    TargetRecognitionGenerator,
    TargetRecognitionPayload,
    TargetRecognitionSceneEntity,
    TargetRecognitionScorer,
    TargetRecognitionSystemCycle,
    build_target_recognition_test,
)


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


def test_generator_determinism_same_seed_same_sequence() -> None:
    seed = 31337
    g1 = TargetRecognitionGenerator(seed=seed)
    g2 = TargetRecognitionGenerator(seed=seed)

    seq1 = [g1.next_problem(difficulty=0.6) for _ in range(25)]
    seq2 = [g2.next_problem(difficulty=0.6) for _ in range(25)]

    assert [(p.prompt, p.answer, p.payload) for p in seq1] == [
        (p.prompt, p.answer, p.payload) for p in seq2
    ]


def test_scene_spawn_pressure_scales_with_difficulty() -> None:
    low = TargetRecognitionGenerator(seed=202604).next_problem(difficulty=0.0).payload
    high = TargetRecognitionGenerator(seed=202604).next_problem(difficulty=1.0).payload
    assert isinstance(low, TargetRecognitionPayload)
    assert isinstance(high, TargetRecognitionPayload)

    assert high.scene_spawn_interval_range_s[0] < low.scene_spawn_interval_range_s[0]
    assert high.scene_spawn_interval_range_s[1] < low.scene_spawn_interval_range_s[1]
    assert high.scene_spawn_burst_chance > low.scene_spawn_burst_chance
    assert high.scene_spawn_burst_range == (1, 2)


def test_generated_answer_matches_panel_presence_flags() -> None:
    gen = TargetRecognitionGenerator(seed=123)
    p = gen.next_problem(difficulty=0.55)
    payload = p.payload
    assert isinstance(payload, TargetRecognitionPayload)

    expected = (
        int(payload.scene_has_target)
        + int(payload.light_has_target)
        + int(payload.scan_has_target)
        + int(payload.system_has_target)
    )
    assert p.answer == expected


def test_scene_target_is_specific_compound_instruction() -> None:
    gen = TargetRecognitionGenerator(seed=2027)
    p = gen.next_problem(difficulty=0.7)
    payload = p.payload
    assert isinstance(payload, TargetRecognitionPayload)
    assert payload.scene_target_options
    assert len(payload.scene_target_options) >= 4
    assert payload.scene_target_options[0] == payload.scene_target

    for option in payload.scene_target_options:
        words = option.replace("(HP)", "").split()
        assert any(w in words for w in ("Hostile", "Friendly", "Neutral"))
        assert any(w in words for w in ("Truck", "Tank", "Building"))


def test_system_panel_generates_three_columns_and_top_target_per_cycle() -> None:
    gen = TargetRecognitionGenerator(seed=2026)
    p = gen.next_problem(difficulty=0.6)
    payload = p.payload
    assert isinstance(payload, TargetRecognitionPayload)
    assert payload.system_cycles

    for cycle in payload.system_cycles:
        assert len(cycle.columns) == 3
        assert cycle.columns[1]
        assert cycle.columns[1][0] == cycle.target
        for idx, col in enumerate(cycle.columns):
            if idx == 1:
                assert cycle.target in col
            else:
                assert cycle.target not in col


def test_scoring_exact_and_estimation_behavior() -> None:
    scorer = TargetRecognitionScorer()
    payload = TargetRecognitionPayload(
        scene_rows=2,
        scene_cols=2,
        scene_cells=("TRK", "BLD", "ICE", "TNK"),
        scene_entities=(
            TargetRecognitionSceneEntity("truck", "friendly", True, False),
            TargetRecognitionSceneEntity("building", "neutral", False, False),
            TargetRecognitionSceneEntity("truck", "hostile", False, True),
            TargetRecognitionSceneEntity("tank", "friendly", False, False),
        ),
        scene_target="TRK",
        scene_has_target=True,
        scene_target_options=("Damaged Friendly Truck", "Hostile Tank", "Neutral Building"),
        light_pattern=("G", "B", "R"),
        light_target_pattern=("G", "B", "R"),
        light_has_target=True,
        scan_tokens=("<>", "[]", "/\\"),
        scan_target="<>",
        scan_has_target=True,
        system_rows=("A1B2", "C3D4", "E5F6"),
        system_target="A1B2",
        system_has_target=True,
        system_cycles=(
            TargetRecognitionSystemCycle(
                target="A1B2",
                columns=(
                    ("C3D4", "E5F6", "G7H8"),
                    ("A1B2", "J9K1", "L2M3"),
                    ("N4P5", "Q6R7", "S8T9"),
                ),
            ),
        ),
        system_step_interval_s=0.5,
        full_credit_error=0,
        zero_credit_error=3,
    )
    problem = Problem(prompt="Count matched panels", answer=2, payload=payload)

    assert scorer.score(problem=problem, user_answer=2, raw="2") == 1.0
    assert scorer.score(problem=problem, user_answer=3, raw="3") == pytest.approx(2 / 3, abs=1e-9)
    assert scorer.score(problem=problem, user_answer=4, raw="4") == pytest.approx(1 / 3, abs=1e-9)
    assert scorer.score(problem=problem, user_answer=5, raw="5") == 0.0


def test_timer_boundary_transitions_to_results_and_rejects_late_submit() -> None:
    clock = FakeClock()
    engine = build_target_recognition_test(
        clock=clock,
        seed=7,
        difficulty=0.5,
        config=TargetRecognitionConfig(scored_duration_s=2.0, practice_questions=0),
    )
    engine.start_scored()

    assert engine.time_remaining_s() == pytest.approx(2.0)
    clock.advance(2.0)
    engine.update()

    assert engine.phase.value == "results"
    assert engine.submit_answer("2") is False
