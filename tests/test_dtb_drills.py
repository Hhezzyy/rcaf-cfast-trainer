from __future__ import annotations

from dataclasses import dataclass

import pytest

from cfast_trainer.ant_drills import AntDrillMode
from cfast_trainer.cognitive_core import Phase
from cfast_trainer.cognitive_core import TestSnapshot as SnapshotModel
from cfast_trainer.dtb_drills import (
    DualTaskBridgeDrillConfig,
    DualTaskBridgeStatus,
    build_dtb_tracking_command_filter_drill,
    build_dtb_tracking_filter_digit_report_drill,
    build_dtb_tracking_interference_recovery_drill,
    build_dtb_tracking_recall_drill,
)
from cfast_trainer.rapid_tracking import RapidTrackingPayload
from cfast_trainer.results import attempt_result_from_engine


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


def _run_bridge(builder, *, seed: int, duration_s: float = 24.0):
    clock = FakeClock()
    drill = builder(
        clock=clock,
        seed=seed,
        difficulty=0.55,
        mode=AntDrillMode.TEMPO,
        config=DualTaskBridgeDrillConfig(scored_duration_s=duration_s),
    )
    drill.start_scored()
    controls = ((0.18, -0.10), (0.06, 0.0), (-0.14, 0.16), (0.0, 0.0))
    frames: list[tuple[object, ...]] = []
    steps = max(1, int(round(duration_s / 0.5))) + 4
    for idx in range(steps):
        if drill.phase is not Phase.SCORED:
            break
        cx, cy = controls[idx % len(controls)]
        drill.set_control(horizontal=cx, vertical=cy)
        snap = drill.snapshot()
        payload = snap.payload
        assert isinstance(payload, RapidTrackingPayload)
        status = drill.bridge_status()
        if payload.target_in_capture_box and idx % 4 == 0:
            assert drill.submit_answer("CAPTURE") is True
        frames.append(
            (
                snap.phase.value,
                payload.segment_label,
                payload.focus_label,
                snap.prompt,
                status.channel,
                status.filter_label,
                status.visible_digits,
                status.interference_active,
                status.recovery_active,
                payload.capture_points,
            )
        )
        clock.advance(0.5)
        drill.update()
    return tuple(frames), drill, clock


def _advance_until(
    drill,
    clock: FakeClock,
    predicate,
    *,
    max_steps: int = 200,
) -> tuple[DualTaskBridgeStatus, SnapshotModel]:
    for _ in range(max_steps):
        snap = drill.snapshot()
        status = drill.bridge_status()
        if predicate(status, snap):
            return status, snap
        clock.advance(0.5)
        drill.update()
    raise AssertionError("bridge condition was not reached")


def _difficulty_for_level(level: int) -> float:
    return float(level - 1) / 9.0


@pytest.mark.parametrize(
    "builder",
    (
        build_dtb_tracking_recall_drill,
        build_dtb_tracking_command_filter_drill,
        build_dtb_tracking_filter_digit_report_drill,
        build_dtb_tracking_interference_recovery_drill,
    ),
)
def test_dtb_drills_are_deterministic_for_same_seed_and_controls(builder) -> None:
    frames_a, drill_a, _clock_a = _run_bridge(builder, seed=515)
    frames_b, drill_b, _clock_b = _run_bridge(builder, seed=515)

    assert frames_a == frames_b
    assert drill_a.scored_summary() == drill_b.scored_summary()


def test_dtb_tracking_recall_opens_delayed_digit_report_and_records_bridge_events() -> None:
    clock = FakeClock()
    drill = build_dtb_tracking_recall_drill(
        clock=clock,
        seed=91,
        difficulty=0.5,
        config=DualTaskBridgeDrillConfig(scored_duration_s=30.0),
    )
    drill.start_scored()

    status, snap = _advance_until(
        drill,
        clock,
        lambda status, _snap: status.channel == "recall" and status.visible_digits != "",
    )
    assert status.expected_response == status.visible_digits
    assert status.visible_digits in snap.prompt

    status, _snap = _advance_until(
        drill,
        clock,
        lambda status, _snap: status.channel == "recall" and status.recall_input_active,
    )
    assert drill.submit_answer(f"DIGITS:{status.expected_response}") is True

    result = attempt_result_from_engine(drill, test_code="dtb_tracking_recall")
    assert int(result.metrics["bridge.recall_attempted"]) >= 1
    assert any(event.kind == "recall_resolved" for event in result.events)


def test_dtb_tracking_command_filter_accepts_qwer_style_command_responses() -> None:
    clock = FakeClock()
    drill = build_dtb_tracking_command_filter_drill(
        clock=clock,
        seed=92,
        difficulty=0.5,
        config=DualTaskBridgeDrillConfig(scored_duration_s=30.0),
    )
    drill.start_scored()

    status, snap = _advance_until(
        drill,
        clock,
        lambda status, _snap: status.channel == "command" and status.expected_response is not None,
    )
    assert status.filter_label != ""
    assert "Cue:" in snap.prompt
    assert drill.submit_answer(f"CMD:{status.expected_response}") is True

    result = attempt_result_from_engine(drill, test_code="dtb_tracking_command_filter")
    assert int(result.metrics["bridge.command_attempted"]) >= 1
    assert any(event.kind == "command_resolved" for event in result.events)


def test_dtb_filter_digit_report_cycles_both_command_and_recall_channels() -> None:
    frames, drill, _clock = _run_bridge(
        build_dtb_tracking_filter_digit_report_drill,
        seed=211,
        duration_s=42.0,
    )
    seen_channels = {frame[4] for frame in frames}

    assert "command" in seen_channels
    assert "recall" in seen_channels
    result = attempt_result_from_engine(drill, test_code="dtb_tracking_filter_digit_report")
    assert any(event.family == "dual_task_bridge" for event in result.events)


def test_dtb_interference_recovery_exposes_bursts_and_recovery_windows() -> None:
    frames, drill, _clock = _run_bridge(
        build_dtb_tracking_interference_recovery_drill,
        seed=377,
        duration_s=42.0,
    )

    assert any(bool(frame[7]) for frame in frames)
    assert any(bool(frame[8]) for frame in frames)
    result = attempt_result_from_engine(drill, test_code="dtb_tracking_interference_recovery")
    assert any(event.family == "dual_task_bridge" for event in result.events)


@pytest.mark.parametrize(
    ("builder", "expect_recall", "expect_interference"),
    (
        (build_dtb_tracking_recall_drill, True, False),
        (build_dtb_tracking_command_filter_drill, False, False),
        (build_dtb_tracking_filter_digit_report_drill, True, False),
        (build_dtb_tracking_interference_recovery_drill, False, True),
    ),
)
def test_wave1_bridge_drills_l2_l5_l8_scale_materially(
    builder,
    expect_recall: bool,
    expect_interference: bool,
) -> None:
    signatures: list[tuple[int, int, int, float | None, float]] = []
    for level in (2, 5, 8):
        clock = FakeClock()
        drill = builder(
            clock=clock,
            seed=271,
            difficulty=_difficulty_for_level(level),
            config=DualTaskBridgeDrillConfig(scored_duration_s=36.0),
        )
        schedule = tuple(drill._cue_schedule)
        recall_lengths = [len(cue.visible_digits) for cue in schedule if cue.cue_kind == "recall"]
        interference_count = sum(1 for cue in schedule if cue.interference_active)
        shortest_command_deadline = min(
            (cue.deadline_s - cue.start_s) for cue in schedule if cue.cue_kind == "command"
        ) if any(cue.cue_kind == "command" for cue in schedule) else None
        signatures.append(
            (
                len(schedule),
                max(recall_lengths, default=0),
                interference_count,
                None if shortest_command_deadline is None else round(float(shortest_command_deadline), 3),
                round(float(drill.difficulty), 3),
            )
        )

    low, mid, high = signatures
    assert low[0] <= high[0]
    if expect_recall:
        assert low[1] <= mid[1] <= high[1]
    if expect_interference:
        assert low[2] <= mid[2] <= high[2]
    if low[3] is not None and mid[3] is not None and high[3] is not None:
        assert low[3] > mid[3] > high[3]
    assert low[4] < mid[4] < high[4]
    assert low != mid != high
