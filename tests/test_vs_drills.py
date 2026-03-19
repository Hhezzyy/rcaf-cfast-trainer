from __future__ import annotations

from dataclasses import dataclass

from cfast_trainer.ant_drills import AntDrillMode
from cfast_trainer.visual_search import VisualSearchPayload, VisualSearchTaskKind
from cfast_trainer.vs_drills import (
    VsMixedTempoGenerator,
    build_vs_family_run_drill,
    build_vs_matrix_routine_priority_switch_drill,
    build_vs_multi_target_class_search_drill,
    build_vs_priority_switch_search_drill,
    build_vs_target_preview_drill,
)


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


def test_target_preview_drill_uses_visual_search_payload_and_feedback_banner() -> None:
    clock = FakeClock()
    engine = build_vs_target_preview_drill(clock=clock, seed=11, difficulty=0.4)

    engine.start_practice()
    payload = engine.snapshot().payload

    assert isinstance(payload, VisualSearchPayload)
    wrong_answer = payload.answer if hasattr(payload, "answer") else None
    correct_answer = int(engine._current.answer)
    assert engine.submit_answer(str(correct_answer + 1)) is True
    feedback = engine.snapshot().practice_feedback or ""
    assert str(correct_answer) in feedback


def test_family_run_letters_emits_alphanumeric_only() -> None:
    clock = FakeClock()
    engine = build_vs_family_run_drill(
        clock=clock,
        seed=23,
        kind=VisualSearchTaskKind.ALPHANUMERIC,
        difficulty=0.6,
        mode=AntDrillMode.TEMPO,
    )

    engine.start_practice()
    kinds = []
    for _ in range(6):
        payload = engine.snapshot().payload
        assert isinstance(payload, VisualSearchPayload)
        kinds.append(payload.kind)
        assert engine.submit_answer(str(engine._current.answer)) is True
        if engine.phase.value == "practice_done":
            break

    assert set(kinds) == {VisualSearchTaskKind.ALPHANUMERIC}


def test_family_run_symbols_emits_symbol_code_only() -> None:
    clock = FakeClock()
    engine = build_vs_family_run_drill(
        clock=clock,
        seed=29,
        kind=VisualSearchTaskKind.SYMBOL_CODE,
        difficulty=0.6,
        mode=AntDrillMode.TEMPO,
    )

    engine.start_practice()
    kinds = []
    for _ in range(6):
        payload = engine.snapshot().payload
        assert isinstance(payload, VisualSearchPayload)
        kinds.append(payload.kind)
        assert engine.submit_answer(str(engine._current.answer)) is True
        if engine.phase.value == "practice_done":
            break

    assert set(kinds) == {VisualSearchTaskKind.SYMBOL_CODE}


def test_mixed_tempo_emits_both_live_visual_search_families() -> None:
    generator = VsMixedTempoGenerator(seed=41)

    kinds = {
        engine_payload.kind
        for engine_payload in (
            generator.next_problem(difficulty=0.8).payload for _ in range(36)
        )
        if isinstance(engine_payload, VisualSearchPayload)
    }

    assert kinds == {
        VisualSearchTaskKind.ALPHANUMERIC,
        VisualSearchTaskKind.SYMBOL_CODE,
    }


def _difficulty_for_level(level: int) -> float:
    return float(level - 1) / 9.0


def test_multi_target_class_search_emits_mixed_class_payload_metadata() -> None:
    clock = FakeClock()
    engine = build_vs_multi_target_class_search_drill(clock=clock, seed=101, difficulty=0.6)
    engine.start_practice()
    payload = engine.snapshot().payload
    assert isinstance(payload, VisualSearchPayload)
    assert payload.class_count >= 2
    assert len(payload.active_classes) == payload.class_count
    assert payload.switch_mode == "class_cycle"


def test_priority_switch_search_marks_priority_mode() -> None:
    clock = FakeClock()
    engine = build_vs_priority_switch_search_drill(
        clock=clock,
        seed=131,
        difficulty=0.7,
        mode=AntDrillMode.TEMPO,
    )
    engine.start_practice()
    payload = engine.snapshot().payload
    assert isinstance(payload, VisualSearchPayload)
    assert payload.priority_label != ""
    assert payload.switch_mode in {"priority_hold", "priority_switch"}


def test_matrix_routine_priority_switch_uses_routine_and_priority_prompts() -> None:
    clock = FakeClock()
    engine = build_vs_matrix_routine_priority_switch_drill(
        clock=clock,
        seed=151,
        difficulty=0.6,
        mode=AntDrillMode.STRESS,
    )
    engine.start_scored()
    prompts: list[str] = []
    modes: set[str] = set()
    for _ in range(4):
        snap = engine.snapshot()
        payload = snap.payload
        assert isinstance(payload, VisualSearchPayload)
        prompts.append(snap.prompt)
        modes.add(payload.switch_mode)
        assert engine.submit_answer(str(engine._current.answer)) is True
    assert any("Routine" in prompt for prompt in prompts)
    assert any("Priority" in prompt for prompt in prompts)
    assert modes >= {"routine_sweep", "priority_interrupt"}


def test_wave1_scan_drills_l2_l5_l8_scale_materially() -> None:
    builders = (
        build_vs_multi_target_class_search_drill,
        build_vs_priority_switch_search_drill,
        build_vs_matrix_routine_priority_switch_drill,
    )
    for builder in builders:
        signatures: list[tuple[int, int, float, str, float | None]] = []
        for level in (2, 5, 8):
            clock = FakeClock()
            engine = builder(clock=clock, seed=181, difficulty=_difficulty_for_level(level))
            engine.start_scored()
            payload = engine._current.payload
            assert isinstance(payload, VisualSearchPayload)
            signatures.append(
                (
                    payload.class_count,
                    payload.rows * payload.cols,
                    round(float(payload.salience_level), 3),
                    payload.switch_mode,
                    None if engine._current_cap_s is None else round(float(engine._current_cap_s), 3),
                )
            )
        low, mid, high = signatures
        assert low[0] <= mid[0] <= high[0]
        assert low[1] <= mid[1] <= high[1]
        assert low[4] is not None and mid[4] is not None and high[4] is not None
        assert low[4] > mid[4] > high[4]
        assert low != mid != high
