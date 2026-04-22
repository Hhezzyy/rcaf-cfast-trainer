from __future__ import annotations

from dataclasses import dataclass

from cfast_trainer.ant_drills import AntDrillMode
from cfast_trainer.cognitive_core import Phase
from cfast_trainer.target_recognition import TargetRecognitionPayload
from cfast_trainer.tr_drills import (
    build_tr_category_sweep_drill,
    build_tr_damaged_sweep_drill,
    build_tr_light_anchor_drill,
    build_tr_mixed_tempo_drill,
    build_tr_panel_switch_run_drill,
    build_tr_pressure_run_drill,
    build_tr_priority_sweep_drill,
    build_tr_scan_anchor_drill,
    build_tr_scene_anchor_drill,
    build_tr_scene_modifier_run_drill,
    build_tr_system_anchor_drill,
)


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


def _builders():
    return (
        build_tr_scene_anchor_drill,
        build_tr_scene_modifier_run_drill,
        build_tr_priority_sweep_drill,
        build_tr_damaged_sweep_drill,
        build_tr_category_sweep_drill,
        build_tr_light_anchor_drill,
        build_tr_scan_anchor_drill,
        build_tr_system_anchor_drill,
        build_tr_panel_switch_run_drill,
        build_tr_mixed_tempo_drill,
        build_tr_pressure_run_drill,
    )


def _current_payload(engine) -> TargetRecognitionPayload:
    payload = engine._current.payload
    assert isinstance(payload, TargetRecognitionPayload)
    return payload


def test_same_seed_gives_same_problem_stream_for_each_tr_drill() -> None:
    for builder in _builders():
        clock_a = FakeClock()
        clock_b = FakeClock()
        engine_a = builder(clock=clock_a, seed=717, difficulty=0.58, mode=AntDrillMode.BUILD)
        engine_b = builder(clock=clock_b, seed=717, difficulty=0.58, mode=AntDrillMode.BUILD)
        engine_a.start_scored()
        engine_b.start_scored()

        for _ in range(6):
            payload_a = _current_payload(engine_a)
            payload_b = _current_payload(engine_b)
            assert engine_a._current.prompt == engine_b._current.prompt
            assert engine_a._current.answer == engine_b._current.answer
            assert payload_a == payload_b
            assert engine_a.submit_answer(str(engine_a._current.answer)) is True
            assert engine_b.submit_answer(str(engine_b._current.answer)) is True


def test_tr_drill_scene_spawn_pressure_is_deterministic_and_mode_scaled() -> None:
    build_engine = build_tr_pressure_run_drill(
        clock=FakeClock(),
        seed=717,
        difficulty=0.82,
        mode=AntDrillMode.BUILD,
    )
    stress_engine = build_tr_pressure_run_drill(
        clock=FakeClock(),
        seed=717,
        difficulty=0.82,
        mode=AntDrillMode.STRESS,
    )
    build_engine.start_scored()
    stress_engine.start_scored()
    build_payload = _current_payload(build_engine)
    stress_payload = _current_payload(stress_engine)

    assert stress_payload.scene_spawn_interval_range_s[0] < build_payload.scene_spawn_interval_range_s[0]
    assert stress_payload.scene_spawn_interval_range_s[1] < build_payload.scene_spawn_interval_range_s[1]
    assert stress_payload.scene_spawn_burst_chance > build_payload.scene_spawn_burst_chance

    repeat_engine = build_tr_pressure_run_drill(
        clock=FakeClock(),
        seed=717,
        difficulty=0.82,
        mode=AntDrillMode.STRESS,
    )
    repeat_engine.start_scored()
    assert _current_payload(repeat_engine) == stress_payload


def test_focused_tr_drills_keep_scene_active_for_anchor_variants() -> None:
    cases = (
        (build_tr_scene_anchor_drill, ("scene",)),
        (build_tr_scene_modifier_run_drill, ("scene",)),
        (build_tr_priority_sweep_drill, ("scene",)),
        (build_tr_damaged_sweep_drill, ("scene",)),
        (build_tr_category_sweep_drill, ("scene",)),
        (build_tr_light_anchor_drill, ("scene", "light")),
        (build_tr_scan_anchor_drill, ("scene", "scan")),
        (build_tr_system_anchor_drill, ("scene", "system")),
    )
    for builder, expected in cases:
        engine = builder(clock=FakeClock(), seed=818, difficulty=0.5, mode=AntDrillMode.BUILD)
        engine.start_scored()
        payload = _current_payload(engine)
        assert payload.active_panels == expected


def test_tr_panel_switch_run_repeats_fixed_scene_backed_cycle() -> None:
    engine = build_tr_panel_switch_run_drill(
        clock=FakeClock(),
        seed=919,
        difficulty=0.5,
        mode=AntDrillMode.BUILD,
    )
    engine.start_scored()

    seen: list[tuple[str, ...]] = []
    for _ in range(8):
        seen.append(_current_payload(engine).active_panels)
        assert engine.submit_answer(str(engine._current.answer)) is True

    assert seen == [
        ("scene",),
        ("scene", "light"),
        ("scene", "scan"),
        ("scene", "system"),
        ("scene",),
        ("scene", "light"),
        ("scene", "scan"),
        ("scene", "system"),
    ]


def test_tr_mixed_tempo_repeats_fixed_scene_backed_six_step_cycle() -> None:
    engine = build_tr_mixed_tempo_drill(
        clock=FakeClock(),
        seed=1212,
        difficulty=0.5,
        mode=AntDrillMode.TEMPO,
    )
    engine.start_scored()

    seen: list[tuple[str, ...]] = []
    for _ in range(6):
        seen.append(_current_payload(engine).active_panels)
        assert engine.submit_answer(str(engine._current.answer)) is True

    assert seen == [
        ("scene", "light"),
        ("scene", "scan", "system"),
        ("scene", "scan"),
        ("scene", "light", "system"),
        ("scene", "light", "scan"),
        ("scene", "light", "scan", "system"),
    ]


def test_tr_pressure_run_always_keeps_all_four_panels_active() -> None:
    engine = build_tr_pressure_run_drill(
        clock=FakeClock(),
        seed=4141,
        difficulty=0.5,
        mode=AntDrillMode.STRESS,
    )
    engine.start_scored()

    for _ in range(6):
        payload = _current_payload(engine)
        assert payload.active_panels == ("scene", "light", "scan", "system")
        assert engine.submit_answer(str(engine._current.answer)) is True


def test_tr_drill_instructions_keep_mouse_only_flow() -> None:
    engine = build_tr_light_anchor_drill(
        clock=FakeClock(),
        seed=313,
        difficulty=0.5,
        mode=AntDrillMode.BUILD,
    )
    assert engine.phase is Phase.INSTRUCTIONS
    assert "Mouse only" in engine.current_prompt()


def test_tr_scene_clear_all_variants_emit_expected_objectives() -> None:
    cases = (
        (
            build_tr_priority_sweep_drill,
            lambda payload: payload.scene_objective_label == "All Priority Targets"
            and all("(HP)" in label for label in payload.scene_target_options),
        ),
        (
            build_tr_damaged_sweep_drill,
            lambda payload: payload.scene_objective_label == "All Damaged Targets"
            and all("DAMAGED" in label.upper() for label in payload.scene_target_options),
        ),
        (
            build_tr_category_sweep_drill,
            lambda payload: payload.scene_objective_label in {"All Trucks", "All Tanks", "All Buildings"}
            and all(
                {
                    "All Trucks": "Truck",
                    "All Tanks": "Tank",
                    "All Buildings": "Building",
                }[payload.scene_objective_label]
                in label
                for label in payload.scene_target_options
            ),
        ),
    )
    for builder, predicate in cases:
        engine = builder(clock=FakeClock(), seed=909, difficulty=0.62, mode=AntDrillMode.BUILD)
        engine.start_scored()
        saw_present = False
        for _ in range(8):
            payload = _current_payload(engine)
            assert payload.active_panels == ("scene",)
            assert payload.scene_clear_all_targets is True
            assert payload.scene_target == payload.scene_objective_label
            if payload.scene_has_target:
                saw_present = True
                assert predicate(payload)
                assert payload.scene_target_options
            else:
                assert payload.scene_target_options == ()
            assert "clear" in engine._current.prompt.lower()
            assert engine.submit_answer(str(engine._current.answer)) is True
        assert saw_present is True
