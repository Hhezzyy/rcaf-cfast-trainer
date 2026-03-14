from __future__ import annotations

from dataclasses import dataclass

from cfast_trainer.ant_drills import AntDrillMode
from cfast_trainer.visual_search import VisualSearchPayload, VisualSearchTaskKind
from cfast_trainer.vs_drills import (
    VsMixedTempoGenerator,
    build_vs_family_run_drill,
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
