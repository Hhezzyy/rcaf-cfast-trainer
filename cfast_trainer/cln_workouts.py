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


def build_cln_workout_plan(*, duration_scale: float = 1.0) -> AntWorkoutPlan:
    scale = max(0.05, float(duration_scale))
    blocks = (
        _block(
            "sequence_copy",
            "Sequence Copy Warm-Up",
            "Type visible letter strings directly so the encoding rhythm and chunking pattern turn on before delayed recall starts.",
            ("Encoding rhythm", "Chunking", "Visible-supported memory"),
            "cln_sequence_copy",
            AntDrillMode.BUILD,
            10 * scale,
        ),
        _block(
            "sequence_match",
            "Sequence Match Warm-Up",
            "Bridge from scaffolded visible work into the real CLN memory-choice format with short delays and the live corner controls.",
            ("Delayed recall", "Memory response format"),
            "cln_sequence_match",
            AntDrillMode.BUILD,
            8 * scale,
        ),
        _block(
            "math_prime",
            "Math Prime",
            "Prime clean arithmetic inside the CLN shell before the other channels start competing for the same attention.",
            ("Arithmetic priming", "Typed calculation"),
            "cln_math_prime",
            AntDrillMode.BUILD,
            10 * scale,
        ),
        _block(
            "colour_lane",
            "Colour Lane Warm-Up",
            "Build scan rhythm and key-to-lane mapping with the live CLN colour stream only.",
            ("Lane mapping", "Hit timing", "Colour scanning"),
            "cln_colour_lane",
            AntDrillMode.BUILD,
            10 * scale,
        ),
        _block(
            "memory_math",
            "Memory + Math Build",
            "Hold the sequence while solving typed arithmetic without the lane-clearing stream on top yet.",
            ("Memory under interference", "Arithmetic under interference"),
            "cln_memory_math",
            AntDrillMode.TEMPO,
            12 * scale,
        ),
        _block(
            "memory_colour",
            "Memory + Colour Build",
            "Hold the sequence while clearing the lanes so the first real CLN multitask pairing feels controlled.",
            ("Memory under interference", "Colour-lane multitask"),
            "cln_memory_colour",
            AntDrillMode.TEMPO,
            12 * scale,
        ),
        _block(
            "full_steady",
            "Full CLN Steady",
            "Run all three CLN channels together at moderate timings before the hardest pressure block.",
            ("Full multitask integration", "Steady overlap control"),
            "cln_full_steady",
            AntDrillMode.TEMPO,
            13 * scale,
        ),
        _block(
            "full_pressure",
            "Full CLN Pressure",
            "Finish with the full three-channel CLN structure under denser overlap and harder timing.",
            ("Full multitask integration", "Pressure tolerance"),
            "cln_full_pressure",
            AntDrillMode.STRESS,
            15 * scale,
        ),
    )

    return AntWorkoutPlan(
        code="colours_letters_numbers_workout",
        title="Colours, Letters and Numbers Workout (90m)",
        description=(
            "Standard 90-minute Colours, Letters and Numbers workout with typed reflection, "
            "channel-split warm-ups, paired interference blocks, and full multitask pressure."
        ),
        notes=(
            "Typed reflections and block setup screens do not count toward the 90-minute drill clock.",
            "Early blocks split memory, math, and colour so each channel is warm before the paired and full CLN blocks.",
            "The late blocks keep the real CLN control scheme: A/S/D/F/G for memory, digits plus Enter for math, and Q/W/E/R for colour lanes.",
        ),
        blocks=blocks,
    )


def cln_workout_menu_entries() -> tuple[tuple[str, str], ...]:
    return (("colours_letters_numbers_workout", "Colours, Letters and Numbers Workout (90m)"),)
