from __future__ import annotations

from cfast_trainer.airborne_numerical import build_airborne_numerical_test
from cfast_trainer.clock import RealClock
from cfast_trainer.cognitive_core import Phase, round_half_up


def _hhmm_to_minutes(hhmm: int) -> int:
    hh = int(f"{hhmm:04d}"[:2])
    mm = int(f"{hhmm:04d}"[2:])
    return hh * 60 + mm


def _minutes_to_hhmm_int(minutes: int) -> int:
    minutes %= 24 * 60
    hh = minutes // 60
    mm = minutes % 60
    return int(f"{hh:02d}{mm:02d}")


def test_airborne_numerical_practice_flow() -> None:
    clock = RealClock()
    test = build_airborne_numerical_test(clock, seed=12345, practice=True)
    assert test.snapshot().phase == Phase.INSTRUCTIONS

    test.start()
    snap = test.snapshot()
    assert snap.phase == Phase.PRACTICE

    scenario = snap.payload
    assert scenario is not None

    start_minutes = _hhmm_to_minutes(int(scenario.start_time_hhmm))

    if str(snap.prompt).startswith("ARRIVAL TIME"):
        total_dist = sum(l.distance for l in scenario.legs)
        route_minutes = int(round_half_up((total_dist / scenario.speed_value) * 60.0))
        expected = _minutes_to_hhmm_int(start_minutes + route_minutes)
    else:
        endurance_min = int(round_half_up((scenario.start_fuel_liters / scenario.fuel_burn_per_hr) * 60.0))
        expected = _minutes_to_hhmm_int(start_minutes + endurance_min)

    test.submit_answer(expected)
    assert test.snapshot().phase in (Phase.PRACTICE, Phase.PRACTICE_DONE)
