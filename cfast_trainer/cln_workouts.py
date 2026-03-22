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
            8 * scale,
        ),
        _block(
            "sequence_match",
            "Sequence Match Warm-Up",
            "Bridge from scaffolded visible work into the real CLN memory-choice format with short delays and the live corner controls.",
            ("Delayed recall", "Memory response format"),
            "cln_sequence_match",
            AntDrillMode.BUILD,
            7 * scale,
        ),
        _block(
            "math_prime",
            "Math Prime",
            "Prime clean arithmetic inside the CLN shell before the other channels start competing for the same attention.",
            ("Arithmetic priming", "Typed calculation"),
            "cln_math_prime",
            AntDrillMode.BUILD,
            8 * scale,
        ),
        _block(
            "colour_lane",
            "Colour Lane Warm-Up",
            "Build scan rhythm and key-to-lane mapping with the live CLN colour stream only.",
            ("Lane mapping", "Hit timing", "Colour scanning"),
            "cln_colour_lane",
            AntDrillMode.BUILD,
            8 * scale,
        ),
        _block(
            "memory_math",
            "Memory + Math Build",
            "Hold the sequence while solving typed arithmetic without the lane-clearing stream on top yet.",
            ("Memory under interference", "Arithmetic under interference"),
            "cln_memory_math",
            AntDrillMode.TEMPO,
            10 * scale,
        ),
        _block(
            "memory_colour",
            "Memory + Colour Build",
            "Hold the sequence while clearing the lanes so the first real CLN multitask pairing feels controlled.",
            ("Memory under interference", "Colour-lane multitask"),
            "cln_memory_colour",
            AntDrillMode.TEMPO,
            10 * scale,
        ),
        _block(
            "full_steady",
            "Full CLN Steady",
            "Run all three CLN channels together at moderate timings before the hardest pressure block.",
            ("Full multitask integration", "Steady overlap control"),
            "cln_full_steady",
            AntDrillMode.TEMPO,
            10 * scale,
        ),
        _block(
            "overdrive_blue_return",
            "Overdrive Blue Return",
            "Reintroduce the blue lane so the full CLN structure expands back to four colour lanes under live load.",
            ("Full multitask integration", "Blue-lane recovery"),
            "cln_overdrive_blue_return",
            AntDrillMode.TEMPO,
            9 * scale,
        ),
        _block(
            "overdrive_six_choice_memory",
            "Overdrive Six-Choice Memory",
            "Keep the full CLN shell running while the delayed memory grid expands from five choices to six.",
            ("Memory discrimination", "Pressure tolerance"),
            "cln_overdrive_six_choice_memory",
            AntDrillMode.PRESSURE,
            10 * scale,
        ),
        _block(
            "overdrive_dual_math",
            "Overdrive Dual Math",
            "Finish with the standard typed math lane plus a second multiple-choice math panel while memory and colour stay live.",
            ("Dual-math switching", "Full multitask integration", "Pressure tolerance"),
            "cln_overdrive_dual_math",
            AntDrillMode.STRESS,
            10 * scale,
        ),
    )

    return AntWorkoutPlan(
        code="colours_letters_numbers_workout",
        title="Colours, Letters and Numbers Workout (90m)",
        description=(
            "Standard 90-minute Colours, Letters and Numbers workout with typed reflection, "
            "three-lane baseline warm-ups, paired interference blocks, and late overdrive finishers."
        ),
        notes=(
            "Typed reflections and block setup screens do not count toward the 90-minute drill clock.",
            "Early blocks split memory, math, and colour so each channel is warm before the paired and full CLN blocks.",
            "The standard CLN baseline uses A/S/D/F/G for memory, digits plus Enter for math, and Q/W/E for colour lanes.",
            "The late overdrive blocks reintroduce blue, expand memory choice count, and add a bonus multiple-choice math panel.",
        ),
        blocks=blocks,
    )


def cln_workout_menu_entries() -> tuple[tuple[str, str], ...]:
    return (("colours_letters_numbers_workout", "Colours, Letters and Numbers Workout (90m)"),)
