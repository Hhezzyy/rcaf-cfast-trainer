from __future__ import annotations

from dataclasses import dataclass

from .clock import Clock
from .cognitive_core import AnswerScorer, Problem, SeededRng, TimedTextInputTest, clamp01, lerp_int


@dataclass(frozen=True, slots=True)
class SystemLogicDocument:
    title: str
    kind: str
    lines: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SystemLogicFolder:
    name: str
    documents: tuple[SystemLogicDocument, ...]


@dataclass(frozen=True, slots=True)
class SystemLogicPayload:
    scenario_code: str
    folders: tuple[SystemLogicFolder, ...]
    question: str
    answer_unit: str
    correct_value: int
    estimate_tolerance: int


@dataclass(frozen=True, slots=True)
class SystemLogicConfig:
    scored_duration_s: float = 1200.0
    practice_questions: int = 3


class SystemLogicScorer(AnswerScorer):
    """Exact answers receive full credit; near estimates receive partial credit."""

    def score(self, *, problem: Problem, user_answer: int, raw: str) -> float:
        payload = problem.payload
        if not isinstance(payload, SystemLogicPayload):
            return 1.0 if int(user_answer) == int(problem.answer) else 0.0

        target = int(payload.correct_value)
        delta = abs(int(user_answer) - target)
        if delta == 0:
            return 1.0

        tolerance = max(1, int(payload.estimate_tolerance))
        max_delta = tolerance * 2
        if delta >= max_delta:
            return 0.0
        return float(max_delta - delta) / float(max_delta)


class SystemLogicGenerator:
    """Deterministic System Logic generator with folder/submenu sources."""

    def __init__(self, *, seed: int) -> None:
        self._rng = SeededRng(seed)
        self._scenario_index = 0

    def next_problem(self, *, difficulty: float) -> Problem:
        difficulty = clamp01(difficulty)
        scenario_code = self._next_scenario_code()
        variant = self._rng.randint(0, 2)

        if variant == 0:
            payload = self._build_power_distribution_case(
                scenario_code=scenario_code,
                difficulty=difficulty,
            )
        elif variant == 1:
            payload = self._build_fuel_balance_case(
                scenario_code=scenario_code,
                difficulty=difficulty,
            )
        else:
            payload = self._build_sortie_capacity_case(
                scenario_code=scenario_code,
                difficulty=difficulty,
            )

        unit_hint = payload.answer_unit if payload.answer_unit else "value"
        prompt = (
            f"{payload.scenario_code}\n{payload.question}\n"
            f"Enter whole-number answer ({unit_hint})."
        )
        return Problem(prompt=prompt, answer=payload.correct_value, payload=payload)

    def _next_scenario_code(self) -> str:
        self._scenario_index += 1
        return f"SYS-{self._scenario_index:03d}"

    def _build_power_distribution_case(
        self,
        *,
        scenario_code: str,
        difficulty: float,
    ) -> SystemLogicPayload:
        modules = ("PWR-A", "PWR-B", "PWR-C", "PWR-D")
        out_lo = lerp_int(20, 40, difficulty)
        out_hi = lerp_int(50, 90, difficulty)
        outputs = tuple(self._rng.randint(out_lo // 10, out_hi // 10) * 10 for _ in modules)

        active_count = 2 if difficulty < 0.55 else 3
        active_idx = sorted(self._rng.sample([0, 1, 2, 3], k=active_count))
        active_modules = tuple(modules[i] for i in active_idx)
        base_total = sum(outputs[i] for i in active_idx)

        temp_curve = (
            ("0-5 C", 0, 5, 80),
            ("6-20 C", 6, 20, 100),
            ("21-30 C", 21, 30, 110),
            ("31-40 C", 31, 40, 90),
        )
        curve_index = self._rng.randint(0, len(temp_curve) - 1)
        band_label, temp_lo, temp_hi, efficiency_pct = temp_curve[curve_index]
        ambient_temp = self._rng.randint(temp_lo, temp_hi)

        duration_min = self._rng.randint(lerp_int(8, 12, difficulty), lerp_int(20, 34, difficulty))
        fixed_loss = self._rng.randint(1, 5) * 5
        gross_per_min = (base_total * efficiency_pct) // 100
        if gross_per_min <= fixed_loss:
            fixed_loss = max(5, gross_per_min // 3)
        net_per_min = max(1, gross_per_min - fixed_loss)
        correct = net_per_min * duration_min
        tolerance = max(2, correct // 25)

        folders = (
            SystemLogicFolder(
                name="Tables",
                documents=(
                    self._doc(
                        title="Subsystem Output Matrix",
                        kind="table",
                        lines=(
                            "Module | Output (u/min)",
                            *(
                                f"{name:>6} | {value:>4}"
                                for name, value in zip(modules, outputs, strict=True)
                            ),
                        ),
                    ),
                    self._doc(
                        title="Bus Layout",
                        kind="diagram",
                        lines=(
                            "PWR-A ----+",
                            "PWR-B ----+--> Mission Bus",
                            "PWR-C ----+",
                            "PWR-D ----+",
                        ),
                    ),
                ),
            ),
            SystemLogicFolder(
                name="Graphs",
                documents=(
                    self._doc(
                        title="Temperature Efficiency Curve",
                        kind="graph",
                        lines=tuple(f"{label:>7} -> {pct:>3}%" for label, _, _, pct in temp_curve),
                    ),
                    self._doc(
                        title="Sensor Readout",
                        kind="facts",
                        lines=(
                            f"Ambient temperature: {ambient_temp} C",
                            f"Matched curve band: {band_label}",
                        ),
                    ),
                ),
            ),
            SystemLogicFolder(
                name="Rules",
                documents=(
                    self._doc(
                        title="Computation Rule",
                        kind="equation",
                        lines=(
                            "gross_per_min = active_output * efficiency_pct / 100",
                            "net_per_min = gross_per_min - fixed_system_loss",
                            "mission_total = net_per_min * mission_minutes",
                        ),
                    ),
                    self._doc(
                        title="Operational Statement",
                        kind="facts",
                        lines=(
                            "Only active modules feed the Mission Bus.",
                            "Use whole-number arithmetic throughout.",
                        ),
                    ),
                ),
            ),
            SystemLogicFolder(
                name="Task",
                documents=(
                    self._doc(
                        title="Mission Card",
                        kind="facts",
                        lines=(
                            f"Active modules: {', '.join(active_modules)}",
                            f"Mission duration: {duration_min} minutes",
                            f"Fixed system loss: {fixed_loss} u/min",
                        ),
                    ),
                    self._doc(
                        title="Question",
                        kind="facts",
                        lines=("Compute total mission output in whole units.",),
                    ),
                ),
            ),
        )

        return SystemLogicPayload(
            scenario_code=scenario_code,
            folders=folders,
            question="Using folder data, what is the total mission output?",
            answer_unit="units",
            correct_value=correct,
            estimate_tolerance=tolerance,
        )

    def _build_fuel_balance_case(
        self,
        *,
        scenario_code: str,
        difficulty: float,
    ) -> SystemLogicPayload:
        engine_draw = self._rng.randint(12, 22) * 5
        avionics_draw = self._rng.randint(6, 14) * 5
        cooling_draw = self._rng.randint(2, 8) * 5
        duration_min = self._rng.randint(lerp_int(12, 16, difficulty), lerp_int(26, 38, difficulty))

        threat_curve = (("Blue", 90), ("Amber", 110), ("Red", 130))
        threat_label, threat_pct = threat_curve[self._rng.randint(0, len(threat_curve) - 1)]

        weighted_primary = ((engine_draw + avionics_draw) * threat_pct) // 100
        total_draw = weighted_primary + cooling_draw
        reserve = self._rng.randint(14, 44) * 5
        start_fuel = total_draw * duration_min + reserve
        tolerance = max(3, reserve // 20)

        folders = (
            SystemLogicFolder(
                name="Tables",
                documents=(
                    self._doc(
                        title="Fuel Draw Table",
                        kind="table",
                        lines=(
                            "Subsystem | Draw (L/min)",
                            f"Engine    | {engine_draw}",
                            f"Avionics  | {avionics_draw}",
                            f"Cooling   | {cooling_draw}",
                        ),
                    ),
                    self._doc(
                        title="Flow Diagram",
                        kind="diagram",
                        lines=(
                            "Tank -> Pump -> Engine",
                            "Tank -> Pump -> Avionics",
                            "Tank -> Pump -> Cooling",
                        ),
                    ),
                ),
            ),
            SystemLogicFolder(
                name="Graphs",
                documents=(
                    self._doc(
                        title="Threat Multiplier Graph",
                        kind="graph",
                        lines=tuple(f"{level:>5} -> {pct:>3}%" for level, pct in threat_curve),
                    ),
                    self._doc(
                        title="Threat Status",
                        kind="facts",
                        lines=(f"Current threat level: {threat_label}",),
                    ),
                ),
            ),
            SystemLogicFolder(
                name="Rules",
                documents=(
                    self._doc(
                        title="Fuel Equation",
                        kind="equation",
                        lines=(
                            "primary = (engine + avionics) * threat_pct / 100",
                            "total_draw = primary + cooling",
                            "remaining = start_fuel - (total_draw * minutes)",
                        ),
                    ),
                    self._doc(
                        title="Statement",
                        kind="facts",
                        lines=("Threat multiplier applies only to engine+avionics draw.",),
                    ),
                ),
            ),
            SystemLogicFolder(
                name="Task",
                documents=(
                    self._doc(
                        title="Segment Brief",
                        kind="facts",
                        lines=(
                            f"Start fuel: {start_fuel} L",
                            f"Segment duration: {duration_min} minutes",
                            "Compute end-of-segment remaining fuel.",
                        ),
                    ),
                    self._doc(
                        title="Question",
                        kind="facts",
                        lines=("How many liters remain?",),
                    ),
                ),
            ),
        )

        return SystemLogicPayload(
            scenario_code=scenario_code,
            folders=folders,
            question="Using folder data, what is the remaining fuel after the segment?",
            answer_unit="L",
            correct_value=reserve,
            estimate_tolerance=tolerance,
        )

    def _build_sortie_capacity_case(
        self,
        *,
        scenario_code: str,
        difficulty: float,
    ) -> SystemLogicPayload:
        base_per_sortie = self._rng.randint(10, 20) * 10
        weather_curve = (("Clear", 100), ("Rain", 110), ("Crosswind", 120), ("Icing", 130))
        weather_label, weather_pct = weather_curve[self._rng.randint(0, len(weather_curve) - 1)]
        effective_per_sortie = (base_per_sortie * weather_pct) // 100

        reserve_stock = self._rng.randint(12, 26) * 10
        baseline_sorties = self._rng.randint(3, lerp_int(6, 9, difficulty))
        stock_units = reserve_stock + (effective_per_sortie * baseline_sorties)
        stock_units += self._rng.randint(0, max(1, effective_per_sortie - 1))
        available = max(0, stock_units - reserve_stock)
        max_sorties = available // max(1, effective_per_sortie)
        tolerance = max(1, max_sorties // 5)

        folders = (
            SystemLogicFolder(
                name="Tables",
                documents=(
                    self._doc(
                        title="Resource Table",
                        kind="table",
                        lines=(
                            "Metric | Value",
                            f"Base units per sortie | {base_per_sortie}",
                            f"Mandatory reserve     | {reserve_stock}",
                        ),
                    ),
                    self._doc(
                        title="Supply Path",
                        kind="diagram",
                        lines=(
                            "Main stock -> Sortie allocation",
                            "Main stock -> Mandatory reserve",
                        ),
                    ),
                ),
            ),
            SystemLogicFolder(
                name="Graphs",
                documents=(
                    self._doc(
                        title="Weather Consumption Multiplier",
                        kind="graph",
                        lines=tuple(f"{label:>9} -> {pct:>3}%" for label, pct in weather_curve),
                    ),
                    self._doc(
                        title="Daily Weather",
                        kind="facts",
                        lines=(f"Current weather: {weather_label}",),
                    ),
                ),
            ),
            SystemLogicFolder(
                name="Rules",
                documents=(
                    self._doc(
                        title="Capacity Equation",
                        kind="equation",
                        lines=(
                            "effective_per_sortie = base_per_sortie * weather_pct / 100",
                            "usable_stock = stock_units - reserve_stock",
                            "max_full_sorties = floor(usable_stock / effective_per_sortie)",
                        ),
                    ),
                    self._doc(
                        title="Statement",
                        kind="facts",
                        lines=("Only complete sorties count.",),
                    ),
                ),
            ),
            SystemLogicFolder(
                name="Task",
                documents=(
                    self._doc(
                        title="Operations Card",
                        kind="facts",
                        lines=(
                            f"Stock units on hand: {stock_units}",
                            f"Reserve to keep: {reserve_stock}",
                        ),
                    ),
                    self._doc(
                        title="Question",
                        kind="facts",
                        lines=("What is the maximum number of full sorties?",),
                    ),
                ),
            ),
        )

        return SystemLogicPayload(
            scenario_code=scenario_code,
            folders=folders,
            question="Using folder data, what is the maximum number of full sorties?",
            answer_unit="sorties",
            correct_value=max_sorties,
            estimate_tolerance=tolerance,
        )

    @staticmethod
    def _doc(*, title: str, kind: str, lines: tuple[str, ...]) -> SystemLogicDocument:
        return SystemLogicDocument(title=title, kind=kind, lines=lines)


def build_system_logic_test(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    config: SystemLogicConfig | None = None,
) -> TimedTextInputTest:
    """Factory for the System Logic test session."""

    cfg = config or SystemLogicConfig()
    instructions = [
        "System Logic Test",
        "",
        "Use folder tabs and submenu documents to collate information.",
        "Each item combines tables, graphs, equations, diagrams, and factual statements.",
        "",
        "Controls during questions:",
        "- Left / Right or Tab: switch folder tabs",
        "- Up / Down: switch submenu document",
        "- Type whole-number answer and press Enter",
        "",
        "Scoring: exact answers score full credit; close estimates earn partial credit.",
        "Once the timed block starts, continue until completion.",
    ]

    generator = SystemLogicGenerator(seed=seed)
    return TimedTextInputTest(
        title="System Logic",
        instructions=instructions,
        generator=generator,
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        practice_questions=cfg.practice_questions,
        scored_duration_s=cfg.scored_duration_s,
        scorer=SystemLogicScorer(),
    )
