from __future__ import annotations

from .ant_drills import AntDrillMode
from .ant_workouts import AntWorkoutBlockPlan, AntWorkoutPlan


def _block(
    block_id: str,
    label: str,
    description: str,
    focus_skills: tuple[str, ...],
    drill_code: str,
    mode: AntDrillMode,
    minutes: float,
) -> AntWorkoutBlockPlan:
    return AntWorkoutBlockPlan(
        block_id=block_id,
        label=label,
        description=description,
        focus_skills=focus_skills,
        drill_code=drill_code,
        mode=mode,
        duration_min=minutes,
    )


def build_trace_test_1_workout_plan(*, duration_scale: float = 1.0) -> AntWorkoutPlan:
    scale = max(0.05, float(duration_scale))
    blocks = (
        _block(
            "tt1_lateral_anchor_build",
            "Lateral Anchor Warm-Up",
            "Start with isolated TT1 left-versus-right work so the lateral command mapping is settled before the mixed stream speeds up.",
            ("TT1 left-right discrimination", "Continuous stream timing"),
            "tt1_lateral_anchor",
            AntDrillMode.BUILD,
            10 * scale,
        ),
        _block(
            "tt1_vertical_anchor_build",
            "Push/Pull Anchor",
            "Isolate TT1 push-versus-pull recognition before the full TT1 command family comes back in.",
            ("TT1 push-pull discrimination", "Forward-travel vertical cue reading"),
            "tt1_vertical_anchor",
            AntDrillMode.BUILD,
            10 * scale,
        ),
        _block(
            "tt1_command_switch_tempo",
            "Command Switch Tempo",
            "Bring all four TT1 commands back under tempo observe windows while the continuous stream stays live.",
            ("TT1 command switching", "Answer-window discipline"),
            "tt1_command_switch_run",
            AntDrillMode.TEMPO,
            10 * scale,
        ),
        _block(
            "tt1_lateral_anchor_tempo",
            "Lateral Tempo Lock",
            "Reinforce left and right discrimination at a faster pace before the later pressure blocks.",
            ("TT1 left-right discrimination", "Tempo pacing"),
            "tt1_lateral_anchor",
            AntDrillMode.TEMPO,
            10 * scale,
        ),
        _block(
            "tt1_vertical_anchor_tempo",
            "Vertical Tempo Lock",
            "Repeat the vertical cue family at tempo so push and pull stay readable when the stream speeds up.",
            ("TT1 push-pull discrimination", "Tempo pacing"),
            "tt1_vertical_anchor",
            AntDrillMode.TEMPO,
            10 * scale,
        ),
        _block(
            "tt1_command_switch_stress_prime",
            "Command Stress Prime",
            "Return to all four TT1 commands under the tighter stress observe windows.",
            ("TT1 command switching", "Pressure tolerance"),
            "tt1_command_switch_run",
            AntDrillMode.STRESS,
            10 * scale,
        ),
        _block(
            "tt1_command_switch_extended_tempo",
            "Extended Command Run",
            "Hold the full TT1 command family for a longer tempo block without breaking the continuous stream rhythm.",
            ("TT1 command switching", "Sustained tempo control"),
            "tt1_command_switch_run",
            AntDrillMode.TEMPO,
            15 * scale,
        ),
        _block(
            "tt1_command_switch_pressure_finish",
            "Pressure Finish",
            "Finish with the full TT1 command family under stress timing for the longest block in the workout.",
            ("TT1 command switching", "Pressure tolerance"),
            "tt1_command_switch_run",
            AntDrillMode.STRESS,
            15 * scale,
        ),
    )
    return AntWorkoutPlan(
        code="trace_test_1_workout",
        title="Trace Test 1 Workout (90m)",
        description=(
            "Standard 90-minute Trace Test 1 workout with lateral and vertical anchors, "
            "then longer full-command TT1 tempo and stress blocks."
        ),
        notes=(
            "Block setup screens do not count toward the 90-minute drill clock.",
            "This workout stays inside Trace Test 1 only; it does not chain into Trace Test 2.",
            "Later blocks reuse the full TT1 command family under tighter observe windows instead of introducing a mixed-test segment.",
        ),
        blocks=blocks,
    )


def build_trace_test_2_workout_plan(*, duration_scale: float = 1.0) -> AntWorkoutPlan:
    scale = max(0.05, float(duration_scale))
    blocks = (
        _block(
            "tt2_steady_anchor_build",
            "Steady Recall Warm-Up",
            "Start with TT2 no-direction-change recall so the observe-then-answer rhythm settles cleanly.",
            ("TT2 steady-track recall", "Observe-then-answer rhythm"),
            "tt2_steady_anchor",
            AntDrillMode.BUILD,
            10 * scale,
        ),
        _block(
            "tt2_turn_trace_build",
            "Turn Recall Prime",
            "Bring in TT2 left-turn and right-turn recall before end-state questions start sharing the block time.",
            ("TT2 turn recall", "Guide-style question reading"),
            "tt2_turn_trace_run",
            AntDrillMode.BUILD,
            10 * scale,
        ),
        _block(
            "tt2_position_recall_tempo",
            "Position Recall Tempo",
            "Shift to TT2 end-state recall with leftmost and highest questions under tempo timing.",
            ("TT2 end-state recall", "Observe-then-answer rhythm"),
            "tt2_position_recall_run",
            AntDrillMode.TEMPO,
            10 * scale,
        ),
        _block(
            "tt2_steady_anchor_tempo",
            "Steady Tempo Lock",
            "Repeat the steady-track family at a faster pace to keep the simplest recall judgment automatic.",
            ("TT2 steady-track recall", "Tempo pacing"),
            "tt2_steady_anchor",
            AntDrillMode.TEMPO,
            10 * scale,
        ),
        _block(
            "tt2_turn_trace_tempo",
            "Turn Tempo Lock",
            "Revisit left-turn and right-turn questions at tempo after the position-recall block.",
            ("TT2 turn recall", "Tempo pacing"),
            "tt2_turn_trace_run",
            AntDrillMode.TEMPO,
            10 * scale,
        ),
        _block(
            "tt2_position_recall_stress_prime",
            "Position Stress Prime",
            "Bring the leftmost and highest recall families back under the tighter stress observe windows.",
            ("TT2 end-state recall", "Pressure tolerance"),
            "tt2_position_recall_run",
            AntDrillMode.STRESS,
            10 * scale,
        ),
        _block(
            "tt2_turn_trace_extended_tempo",
            "Extended Turn Recall",
            "Hold the TT2 turn families for a longer tempo block without leaving the observe-question layout.",
            ("TT2 turn recall", "Sustained tempo control"),
            "tt2_turn_trace_run",
            AntDrillMode.TEMPO,
            15 * scale,
        ),
        _block(
            "tt2_position_recall_pressure_finish",
            "Pressure Finish",
            "Finish with TT2 end-state recall under stress timing for the longest block in the workout.",
            ("TT2 end-state recall", "Pressure tolerance"),
            "tt2_position_recall_run",
            AntDrillMode.STRESS,
            15 * scale,
        ),
    )
    return AntWorkoutPlan(
        code="trace_test_2_workout",
        title="Trace Test 2 Workout (90m)",
        description=(
            "Standard 90-minute Trace Test 2 workout with steady, turn, and end-state recall blocks, "
            "then longer tempo and stress TT2 finishers."
        ),
        notes=(
            "Block setup screens do not count toward the 90-minute drill clock.",
            "This workout stays inside Trace Test 2 only; it does not chain into Trace Test 1.",
            "Later blocks keep the guide-style observe screen and separate question screen while tightening the TT2 timing profile.",
        ),
        blocks=blocks,
    )


def trace_workout_menu_entries() -> tuple[tuple[str, str], ...]:
    return (
        ("trace_test_1_workout", "Trace Test 1 Workout (90m)"),
        ("trace_test_2_workout", "Trace Test 2 Workout (90m)"),
    )
