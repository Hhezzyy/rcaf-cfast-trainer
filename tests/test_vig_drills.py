from __future__ import annotations

from dataclasses import dataclass

import pytest

from cfast_trainer.ant_drills import AntDrillMode
from cfast_trainer.cognitive_core import Phase
from cfast_trainer.vig_drills import (
    VigilanceDrillConfig,
    build_vig_clean_scan_drill,
    build_vig_density_ladder_drill,
    build_vig_entry_anchor_drill,
    build_vig_pressure_run_drill,
    build_vig_steady_capture_run_drill,
    build_vig_tempo_sweep_drill,
)
from cfast_trainer.vigilance import VigilancePayload


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


@pytest.mark.parametrize(
    ("builder", "drill_code", "spawn_interval_s", "max_active_symbols"),
    (
        (build_vig_entry_anchor_drill, "vig_entry_anchor", 1.15, 4),
        (build_vig_clean_scan_drill, "vig_clean_scan", 0.95, 5),
        (build_vig_steady_capture_run_drill, "vig_steady_capture_run", 0.85, 6),
        (build_vig_density_ladder_drill, "vig_density_ladder", 0.78, 7),
        (build_vig_tempo_sweep_drill, "vig_tempo_sweep", 0.68, 7),
        (build_vig_pressure_run_drill, "vig_pressure_run", 0.60, 8),
    ),
)
def test_vigilance_drill_builders_apply_expected_stream_profiles(
    builder,
    drill_code: str,
    spawn_interval_s: float,
    max_active_symbols: int,
) -> None:
    clock = FakeClock()
    drill = builder(clock=clock, seed=31, difficulty=0.5)

    assert drill.spawn_interval_s == pytest.approx(spawn_interval_s)
    assert drill.max_active_symbols == max_active_symbols
    assert drill.snapshot().title.startswith("Vigilance:")
    assert drill_code in {
        "vig_entry_anchor",
        "vig_clean_scan",
        "vig_steady_capture_run",
        "vig_density_ladder",
        "vig_tempo_sweep",
        "vig_pressure_run",
    }


@pytest.mark.parametrize(
    ("mode", "practice_duration_s", "scored_duration_s"),
    (
        (AntDrillMode.BUILD, 45.0, 180.0),
        (AntDrillMode.TEMPO, 30.0, 150.0),
        (AntDrillMode.STRESS, 20.0, 180.0),
    ),
)
def test_vigilance_drill_modes_use_expected_practice_and_scored_defaults(
    mode: AntDrillMode,
    practice_duration_s: float,
    scored_duration_s: float,
) -> None:
    clock = FakeClock()
    drill = build_vig_entry_anchor_drill(clock=clock, seed=41, difficulty=0.5, mode=mode)

    assert drill.practice_duration_s == pytest.approx(practice_duration_s)
    assert drill.scored_duration_s == pytest.approx(scored_duration_s)


def test_vigilance_drill_summary_maps_captures_and_expiries_to_workout_metrics() -> None:
    clock = FakeClock()
    drill = build_vig_entry_anchor_drill(
        clock=clock,
        seed=53,
        difficulty=0.5,
        mode=AntDrillMode.BUILD,
        config=VigilanceDrillConfig(
            practice_duration_s=0.0,
            scored_duration_s=8.0,
            spawn_interval_s=0.6,
            max_active_symbols=4,
        ),
    )

    drill.start_scored()
    captured_once = False

    while drill.phase is Phase.SCORED:
        clock.advance(0.1)
        drill.update()
        payload = drill.snapshot().payload
        assert isinstance(payload, VigilancePayload)
        if payload.symbols and not captured_once:
            symbol = payload.symbols[0]
            assert drill.submit_answer(f"{symbol.row},{symbol.col}") is True
            captured_once = True

    scored_events = tuple(event for event in drill.events() if event.phase is Phase.SCORED)
    captures = [event for event in scored_events if event.is_correct]
    expiries = [event for event in scored_events if not event.is_correct]
    summary = drill.scored_summary()

    assert captures
    assert expiries
    assert summary.attempted == len(scored_events)
    assert summary.correct == len(captures)
    assert summary.timeouts == len(expiries)
    assert summary.total_score == pytest.approx(sum(event.score for event in scored_events))
    assert summary.max_score == pytest.approx(sum(event.max_score for event in scored_events))
    assert summary.fixation_rate == pytest.approx(len(expiries) / len(scored_events))
