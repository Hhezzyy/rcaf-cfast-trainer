from __future__ import annotations

from dataclasses import dataclass

import pytest

from cfast_trainer.ant_drills import AntDrillMode
from cfast_trainer.cognitive_core import Phase
from cfast_trainer.sa_drills import (
    SA_FAMILY_CYCLE,
    SaDrillConfig,
    build_sa_action_selection_run_drill,
    build_sa_contact_identification_prime_drill,
    build_sa_family_switch_run_drill,
    build_sa_future_projection_run_drill,
    build_sa_mixed_tempo_drill,
    build_sa_picture_anchor_drill,
    build_sa_pressure_run_drill,
    build_sa_status_recall_prime_drill,
)
from cfast_trainer.situational_awareness import SA_CHANNEL_ORDER, SA_QUERY_KIND_ORDER, SituationalAwarenessPayload


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


def _run_drill(
    drill,
    clock: FakeClock,
    *,
    duration_s: float,
) -> tuple[list[tuple], object]:
    drill.start_practice()
    answered: set[int] = set()
    frames: list[tuple] = []
    steps = max(1, int(round(duration_s))) + 3
    for _ in range(steps):
        if drill.phase is not Phase.SCORED:
            break
        snap = drill.snapshot()
        payload = snap.payload
        assert isinstance(payload, SituationalAwarenessPayload)
        if payload.active_query is not None and payload.active_query.query_id not in answered:
            assert drill.submit_answer(payload.active_query.correct_answer_token) is True
            answered.add(payload.active_query.query_id)
        frames.append(
            (
                snap.phase.value,
                payload.active_channels,
                payload.active_query_kinds,
                payload.segment_label,
                payload.focus_label,
                str(payload.scenario_family),
                None if payload.active_query is None else str(payload.active_query.kind),
                None if payload.active_query is None else payload.active_query.correct_answer_token,
            )
        )
        clock.advance(1.0)
        drill.update()
    return frames, drill.scored_summary()


@pytest.mark.parametrize(
    "builder",
    (
        build_sa_picture_anchor_drill,
        build_sa_contact_identification_prime_drill,
        build_sa_status_recall_prime_drill,
        build_sa_future_projection_run_drill,
        build_sa_action_selection_run_drill,
        build_sa_family_switch_run_drill,
        build_sa_mixed_tempo_drill,
        build_sa_pressure_run_drill,
    ),
)
def test_sa_drills_are_deterministic_for_same_seed_and_answers(builder) -> None:
    c1 = FakeClock()
    c2 = FakeClock()
    d1 = builder(
        clock=c1,
        seed=515,
        difficulty=0.55,
        mode=AntDrillMode.BUILD,
        config=SaDrillConfig(scored_duration_s=20.0),
    )
    d2 = builder(
        clock=c2,
        seed=515,
        difficulty=0.55,
        mode=AntDrillMode.BUILD,
        config=SaDrillConfig(scored_duration_s=20.0),
    )

    frames1, summary1 = _run_drill(d1, c1, duration_s=20.0)
    frames2, summary2 = _run_drill(d2, c2, duration_s=20.0)

    assert frames1 == frames2
    assert summary1 == summary2


@pytest.mark.parametrize(
    ("builder", "expected_channels", "expected_query_kinds"),
    (
        (
            build_sa_picture_anchor_drill,
            ("pictorial", "coded", "numerical"),
            ("current_location", "current_allegiance", "vehicle_type"),
        ),
        (
            build_sa_contact_identification_prime_drill,
            ("pictorial", "coded"),
            ("current_allegiance", "vehicle_type"),
        ),
        (
            build_sa_status_recall_prime_drill,
            ("coded", "numerical", "aural"),
            ("sighting_grid", "report_variation", "rule_action"),
        ),
        (
            build_sa_future_projection_run_drill,
            ("pictorial", "numerical", "aural"),
            ("instructed_destination", "actual_destination"),
        ),
        (
            build_sa_action_selection_run_drill,
            SA_CHANNEL_ORDER,
            ("rule_action",),
        ),
    ),
)
def test_sa_focused_drills_emit_expected_channels_and_query_kinds(
    builder,
    expected_channels,
    expected_query_kinds,
) -> None:
    clock = FakeClock()
    drill = builder(
        clock=clock,
        seed=91,
        difficulty=0.5,
        mode=AntDrillMode.BUILD,
        config=SaDrillConfig(scored_duration_s=20.0),
    )
    drill.start_practice()
    payload = drill.snapshot().payload
    assert isinstance(payload, SituationalAwarenessPayload)
    assert payload.active_channels == expected_channels
    assert payload.active_query_kinds == expected_query_kinds


def test_sa_family_switch_run_repeats_fixed_family_cycle() -> None:
    clock = FakeClock()
    drill = build_sa_family_switch_run_drill(
        clock=clock,
        seed=17,
        difficulty=0.5,
        mode=AntDrillMode.TEMPO,
        config=SaDrillConfig(scored_duration_s=260.0),
    )
    drill.start_practice()

    observed: list[str] = []
    for _ in range(360):
        payload = drill.snapshot().payload
        assert isinstance(payload, SituationalAwarenessPayload)
        family = str(payload.scenario_family)
        if not observed or observed[-1] != family:
            observed.append(family)
        if len(observed) >= 5:
            break
        clock.advance(1.0)
        drill.update()

    assert observed[:5] == [family.value for family in (*SA_FAMILY_CYCLE, SA_FAMILY_CYCLE[0])]


def test_sa_mixed_tempo_repeats_fixed_query_kind_cycle() -> None:
    clock = FakeClock()
    drill = build_sa_mixed_tempo_drill(
        clock=clock,
        seed=27,
        difficulty=0.5,
        mode=AntDrillMode.TEMPO,
        config=SaDrillConfig(scored_duration_s=480.0),
    )
    drill.start_practice()

    observed: list[str] = []
    expected = list(SA_QUERY_KIND_ORDER[:6])
    for _ in range(540):
        payload = drill.snapshot().payload
        if not isinstance(payload, SituationalAwarenessPayload):
            break
        query_kind = payload.active_query_kinds[0]
        if not observed or observed[-1] != query_kind:
            observed.append(query_kind)
        if len(observed) >= len(expected):
            break
        clock.advance(1.0)
        drill.update()

    assert observed[: len(expected)] == expected


def test_sa_pressure_run_keeps_all_channels_and_query_kinds_active() -> None:
    clock = FakeClock()
    drill = build_sa_pressure_run_drill(
        clock=clock,
        seed=37,
        difficulty=0.5,
        mode=AntDrillMode.STRESS,
        config=SaDrillConfig(scored_duration_s=12.0),
    )
    drill.start_practice()

    payload = drill.snapshot().payload
    assert isinstance(payload, SituationalAwarenessPayload)
    assert payload.active_channels == SA_CHANNEL_ORDER
    assert payload.active_query_kinds == SA_QUERY_KIND_ORDER
