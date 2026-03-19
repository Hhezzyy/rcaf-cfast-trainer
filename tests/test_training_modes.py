from __future__ import annotations

from dataclasses import dataclass

from cfast_trainer.ant_drills import AntDrillMode
from cfast_trainer.cognitive_core import Phase
from cfast_trainer.no_drills import NoDrillConfig, build_no_fact_prime_drill
from cfast_trainer.training_modes import (
    FatigueProbeConfig,
    maybe_build_fatigue_probe_drill,
    supported_manual_modes,
)


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return float(self.t)

    def advance(self, dt: float) -> None:
        self.t += float(dt)


def test_supported_manual_modes_limits_fresh_to_foundational_drills() -> None:
    arithmetic_modes = tuple(mode.value for mode in supported_manual_modes("ma_one_step_fluency"))
    mixed_visual_modes = tuple(mode.value for mode in supported_manual_modes("vs_mixed_tempo"))
    cln_benchmark_modes = tuple(mode.value for mode in supported_manual_modes("cln_sequence_math_recall"))
    cln_warmup_modes = tuple(mode.value for mode in supported_manual_modes("cln_colour_lane"))

    assert arithmetic_modes == (
        "fresh",
        "build",
        "tempo",
        "pressure",
        "fatigue_probe",
        "recovery",
        "stress",
    )
    assert mixed_visual_modes == (
        "build",
        "tempo",
        "pressure",
        "fatigue_probe",
        "recovery",
        "stress",
    )
    assert cln_benchmark_modes[0] == "fresh"
    assert cln_warmup_modes[0] == "fresh"


def test_supported_manual_modes_keeps_canonical_anchor_drills_in_fresh_mode() -> None:
    quantitative_modes = tuple(mode.value for mode in supported_manual_modes("ma_written_numerical_extraction"))
    visual_modes = tuple(mode.value for mode in supported_manual_modes("vs_multi_target_class_search"))
    psychomotor_modes = tuple(mode.value for mode in supported_manual_modes("sma_split_axis_control"))

    assert quantitative_modes[0] == "fresh"
    assert visual_modes[0] == "fresh"
    assert psychomotor_modes[0] == "fresh"


def test_fatigue_probe_drill_reuses_seed_for_baseline_and_late_repeat() -> None:
    clock = FakeClock()
    calls: list[tuple[str, int, float]] = []
    probe = maybe_build_fatigue_probe_drill(
        mode=AntDrillMode.FATIGUE_PROBE,
        title_base="Numerical Operations: Fact Prime",
        clock=clock,
        seed=321,
        difficulty=0.5,
        config=FatigueProbeConfig(
            baseline_duration_s=1.0,
            loader_duration_s=1.0,
            late_duration_s=1.0,
            loader_seed_offset=17,
        ),
        build_segment=lambda segment_mode, segment_seed, segment_duration_s: (
            calls.append((str(segment_mode), int(segment_seed), float(segment_duration_s))),
            build_no_fact_prime_drill(
                clock=clock,
                seed=int(segment_seed),
                difficulty=0.5,
                mode=AntDrillMode(str(segment_mode)),
                config=NoDrillConfig(
                    practice_questions=0,
                    scored_duration_s=float(segment_duration_s),
                ),
            ),
        )[1],
    )

    assert probe is not None

    probe.start_scored()
    while probe.phase is Phase.SCORED:
        current_engine = probe._current_engine
        assert current_engine is not None
        remaining = current_engine.time_remaining_s()
        assert remaining is not None
        clock.advance(float(remaining) + 0.05)
        probe.update()

    metrics = probe.result_metrics()
    event_kinds = [event.kind for event in probe.events() if event.family == "fatigue_probe"]

    assert calls == [
        ("build", 321, 1.0),
        ("pressure", 338, 1.0),
        ("build", 321, 1.0),
    ]
    assert probe.phase is Phase.RESULTS
    assert metrics["training_mode"] == "fatigue_probe"
    assert "fatigue_probe.baseline.accuracy" in metrics
    assert "fatigue_probe.loader.accuracy" in metrics
    assert "fatigue_probe.late.accuracy" in metrics
    assert "fatigue_probe.delta_accuracy" in metrics
    assert event_kinds == [
        "baseline_started",
        "baseline_completed",
        "loader_started",
        "loader_completed",
        "late_probe_started",
        "late_probe_completed",
    ]
