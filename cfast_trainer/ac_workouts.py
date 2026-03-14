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


def build_ac_workout_plan(*, duration_scale: float = 1.0) -> AntWorkoutPlan:
    scale = max(0.05, float(duration_scale))
    blocks = (
        _block(
            "gate_anchor",
            "Gate Anchor Warm-Up",
            "Start with pure gate flight so the ball-control rhythm is active before commands and recall channels come online.",
            ("Psychomotor gate flight", "Ball settling"),
            "ac_gate_anchor",
            AntDrillMode.BUILD,
            10 * scale,
        ),
        _block(
            "state_commands",
            "State Command Prime",
            "Keep gate pressure low and clean up colour and number responses before the callsign filtering blocks start.",
            ("State-command filtering", "Colour-number response"),
            "ac_state_command_prime",
            AntDrillMode.BUILD,
            10 * scale,
        ),
        _block(
            "gate_directives",
            "Gate Directive Run",
            "Bring next-matching-gate directives online and keep moving through the gate stream without freezing.",
            ("Next-gate directives", "Gate-rule execution"),
            "ac_gate_directive_run",
            AntDrillMode.BUILD,
            10 * scale,
        ),
        _block(
            "digit_prime",
            "Digit Sequence Prime",
            "Stay on low gate pressure and sharpen the 5-6 digit memory cycle before mixed tempo blocks return.",
            ("Digit recall", "Auditory memory control"),
            "ac_digit_sequence_prime",
            AntDrillMode.BUILD,
            10 * scale,
        ),
        _block(
            "trigger_anchor",
            "Trigger Cue Anchor",
            "Hold the tunnel cleanly and answer the trigger cue on time before it gets folded into a mixed callsign block.",
            ("Trigger response", "Cue timing"),
            "ac_trigger_cue_anchor",
            AntDrillMode.BUILD,
            10 * scale,
        ),
        _block(
            "callsign_filter",
            "Callsign Filter Run",
            "Combine gates, state commands, directives, and distractors so the callsign filter starts to feel automatic.",
            ("Callsign filtering", "Distractor rejection"),
            "ac_callsign_filter_run",
            AntDrillMode.TEMPO,
            10 * scale,
        ),
        _block(
            "mixed_tempo",
            "Mixed Tempo Run",
            "Repeat the fixed six-segment auditory cycle so flying, filtering, recall, and trigger work all stay warm together.",
            ("Mixed auditory tempo", "Channel switching", "Full-task rhythm"),
            "ac_mixed_tempo",
            AntDrillMode.TEMPO,
            15 * scale,
        ),
        _block(
            "pressure_run",
            "Pressure Run",
            "Finish with all channels active, faster cadence, heavier disturbance, and the highest distractor pressure in the family.",
            ("Pressure tolerance", "Full auditory integration"),
            "ac_pressure_run",
            AntDrillMode.STRESS,
            15 * scale,
        ),
    )
    return AntWorkoutPlan(
        code="auditory_capacity_workout",
        title="Auditory Capacity Workout (90m)",
        description=(
            "Standard 90-minute Auditory Capacity workout with typed reflection, focused channel warm-ups, "
            "a mixed tempo block, and a final full-pressure run on the live auditory task."
        ),
        notes=(
            "Typed reflections and block setup screens do not count toward the 90-minute drill clock.",
            "Every block reuses the live Auditory Capacity screen, audio path, and trigger/recall control model.",
            "Controls stay the normal live model throughout: fly the ball, change colour or number, submit digit recall, and answer the beep with the configured trigger binding or Space.",
        ),
        blocks=blocks,
    )


def ac_workout_menu_entries() -> tuple[tuple[str, str], ...]:
    return (("auditory_capacity_workout", "Auditory Capacity Workout (90m)"),)
