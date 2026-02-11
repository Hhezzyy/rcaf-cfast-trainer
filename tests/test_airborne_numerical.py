from __future__ import annotations

from cfast_trainer.airborne_numerical import (
    AirborneArrivalTimeScorer,
    AirborneNumericalGenerator,
    AirborneScenario,
    _hhmm_str_to_minutes,
    _minutes_to_hhmm_str,
)
from cfast_trainer.cognitive_core import Problem


def test_generator_determinism() -> None:
    g1 = AirborneNumericalGenerator(seed=123)
    g2 = AirborneNumericalGenerator(seed=123)

    out1: list[tuple[str, int, object]] = []
    out2: list[tuple[str, int, object]] = []

    for _ in range(20):
        p1 = g1.next_problem(difficulty=0.6)
        p2 = g2.next_problem(difficulty=0.6)
        out1.append((p1.prompt, p1.answer, p1.payload))
        out2.append((p2.prompt, p2.answer, p2.payload))

    assert out1 == out2


def test_scorer_requires_exactly_4_digits() -> None:
    g = AirborneNumericalGenerator(seed=7)
    p = g.next_problem(difficulty=0.5)
    assert isinstance(p.payload, AirborneScenario)

    scorer = AirborneArrivalTimeScorer(zero_at_minutes=30)

    # Wrong length => 0
    assert scorer.score(problem=p, user_answer=0, raw="930") == 0.0
    assert scorer.score(problem=p, user_answer=0, raw="09300") == 0.0
    assert scorer.score(problem=p, user_answer=0, raw="09:30") == 0.0


def test_scorer_linear_decay_to_zero_at_30_minutes() -> None:
    # Build a simple known problem payload: start 1000, route 30 minutes -> correct 1030.
    scenario = AirborneScenario(
        template_name="X",
        node_names=("AAA", "BBB"),
        edge_distances=(60,),
        route=(0, 1),
        legs=(),
        start_time_hhmm="1000",
        speed_value=240,
        speed_unit="kt",
        distance_unit="NM",
        fuel_burn_per_hr=1000,
        parcel_weight=100,
    )
    # Hack: encode "route minutes" via legs by borrowing formula in scorer (legs are empty -> 0).
    # So set answer to start time here; we will instead create a payload where legs sum to 30 min by faking legs.
    # Easiest: reuse generator for a real payload and then manually compare via computed correct from payload.
    g = AirborneNumericalGenerator(seed=1)
    p = g.next_problem(difficulty=0.5)
    assert isinstance(p.payload, AirborneScenario)
    scorer = AirborneArrivalTimeScorer(zero_at_minutes=30)

    # Determine correct HHMM from payload (same computation scorer uses)
    start = _hhmm_str_to_minutes(p.payload.start_time_hhmm)
    # compute route minutes exactly as scorer does: sum round(distance/speed*60)
    route_min = 0
    for leg in p.payload.legs:
        route_min += int(round((leg.distance / p.payload.speed_value) * 60.0))
    correct = _minutes_to_hhmm_str(start + route_min)

    # Exact => 1.0
    assert scorer.score(problem=p, user_answer=int(correct), raw=correct) == 1.0

    # 15 min error => 0.5
    off15 = _minutes_to_hhmm_str(_hhmm_str_to_minutes(correct) + 15)
    s15 = scorer.score(problem=p, user_answer=int(off15), raw=off15)
    assert abs(s15 - 0.5) < 1e-9

    # 30 min error => 0.0
    off30 = _minutes_to_hhmm_str(_hhmm_str_to_minutes(correct) + 30)
    s30 = scorer.score(problem=p, user_answer=int(off30), raw=off30)
    assert abs(s30 - 0.0) < 1e-9
