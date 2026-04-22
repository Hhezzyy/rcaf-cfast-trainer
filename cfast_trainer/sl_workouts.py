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


def build_sl_workout_plan(*, duration_scale: float = 1.0) -> AntWorkoutPlan:
    scale = max(0.05, float(duration_scale))
    blocks = (
        _block(
            "quantitative_anchor",
            "Quantitative Anchor Warm-Up",
            "Warm up duration, capacity, and resource questions before faster mixed System Logic switching starts.",
            ("Quantitative reasoning", "Table-equation collation"),
            "sl_quantitative_anchor",
            AntDrillMode.BUILD,
            10 * scale,
        ),
        _block(
            "flow_trace_anchor",
            "Flow Trace Anchor Warm-Up",
            "Stay on dependency and feed-path tracing until the diagram-to-fact handoff feels automatic.",
            ("Flow tracing", "Dependency reading"),
            "sl_flow_trace_anchor",
            AntDrillMode.BUILD,
            10 * scale,
        ),
        _block(
            "graph_rule_anchor",
            "Graph + Rule Anchor Warm-Up",
            "Use graph bands and rule panes together before the fault and family switching blocks return.",
            ("Graph interpretation", "Rule application"),
            "sl_graph_rule_anchor",
            AntDrillMode.BUILD,
            10 * scale,
        ),
        _block(
            "fault_prime",
            "Fault Diagnosis Prime",
            "Rehearse the advisory and state-diagnosis items before the index and family switching blocks tighten up.",
            ("Fault diagnosis", "Priority selection"),
            "sl_fault_diagnosis_prime",
            AntDrillMode.BUILD,
            10 * scale,
        ),
        _block(
            "index_switch",
            "Index Switch Run",
            "Keep the guide layout live and deliberately move the right-side index before answering.",
            ("Index switching", "Multi-pane collation"),
            "sl_index_switch_run",
            AntDrillMode.TEMPO,
            10 * scale,
        ),
        _block(
            "family_run",
            "Family Run",
            "Cycle oil, fuel, electrical, hydraulic, and thermal items in a fixed order so family switches stop costing time.",
            ("System-family switching", "Context reset control"),
            "sl_family_run",
            AntDrillMode.TEMPO,
            10 * scale,
        ),
        _block(
            "mixed_tempo",
            "Mixed Tempo Run",
            "Run the fixed quantitative, trace, graph/rule, diagnosis rhythm while system families continue rotating.",
            ("Mixed reasoning tempo", "Family + reasoning switching"),
            "sl_mixed_tempo",
            AntDrillMode.TEMPO,
            15 * scale,
        ),
        _block(
            "pressure_run",
            "Pressure Run",
            "Finish with the hardest mixed System Logic block using all families and all reasoning modes on the live two-pane UI.",
            ("Pressure tolerance", "Full System Logic integration"),
            "sl_pressure_run",
            AntDrillMode.STRESS,
            15 * scale,
        ),
    )
    return AntWorkoutPlan(
        code="system_logic_workout",
        title="System Logic Workout (90m)",
        description=(
            "Standard 90-minute System Logic workout with guide-style two-pane live blocks, "
            "focused reasoning warm-ups, and a final pressure run."
        ),
        notes=(
            "Block setup screens do not count toward the 90-minute drill clock.",
            "Every block reuses the live System Logic two-pane guide layout with the right-side index and A-E answer row.",
            "Controls stay keyboard-only throughout: Up/Down to move the index, A-E or 1-5 to choose, then Enter to submit.",
        ),
        blocks=blocks,
    )


def sl_workout_menu_entries() -> tuple[tuple[str, str], ...]:
    return (("system_logic_workout", "System Logic Workout (90m)"),)
