from __future__ import annotations

from dataclasses import dataclass

from .clock import Clock
from .cognitive_core import AnswerScorer, Problem, SeededRng, TimedTextInputTest, clamp01


def canonical_system_logic_reasoning_family(reasoning_mode: str) -> str:
    token = str(reasoning_mode).strip().lower()
    mapping = {
        "quantitative_duration": "quantitative",
        "dependency_trace": "trace",
        "rule_application": "graph_rule",
        "fault_diagnosis": "diagnosis",
        "state_diagnosis": "diagnosis",
    }
    return mapping.get(token, token)


@dataclass(frozen=True, slots=True)
class SystemLogicDocument:
    title: str
    kind: str
    lines: tuple[str, ...] = ()
    table_headers: tuple[str, ...] = ()
    table_rows: tuple[tuple[str, ...], ...] = ()
    graph_points: tuple[tuple[str, int], ...] = ()
    graph_unit: str = ""
    diagram_paths: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True, slots=True)
class SystemLogicIndexEntry:
    code: int
    label: str
    top_document: SystemLogicDocument
    bottom_document: SystemLogicDocument


@dataclass(frozen=True, slots=True)
class SystemLogicAnswerChoice:
    code: int
    text: str


@dataclass(frozen=True, slots=True)
class SystemLogicPayload:
    scenario_code: str
    system_family: str
    index_entries: tuple[SystemLogicIndexEntry, ...]
    question: str
    answer_choices: tuple[SystemLogicAnswerChoice, ...]
    correct_choice_code: int
    reasoning_mode: str
    required_index_codes: tuple[int, ...]
    required_document_kinds: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SystemLogicConfig:
    scored_duration_s: float = 34.0 * 60.0
    practice_questions: int = 3


@dataclass(frozen=True, slots=True)
class _SystemLogicTemplateSpec:
    family: str
    reasoning_mode: str
    builder_name: str

    @property
    def reasoning_family(self) -> str:
        return canonical_system_logic_reasoning_family(self.reasoning_mode)


class SystemLogicScorer(AnswerScorer):
    """System Logic now uses exact multiple-choice scoring."""

    def score(self, *, problem: Problem, user_answer: int, raw: str) -> float:
        payload = problem.payload
        if not isinstance(payload, SystemLogicPayload):
            return 1.0 if int(user_answer) == int(problem.answer) else 0.0
        return 1.0 if int(user_answer) == int(payload.correct_choice_code) else 0.0


class SystemLogicGenerator:
    """Deterministic guide-style multi-source System Logic generator."""

    _SYSTEM_FAMILIES = (
        "oil",
        "fuel",
        "electrical",
        "hydraulic",
        "thermal",
    )
    _TEMPLATE_SPECS = (
        _SystemLogicTemplateSpec("oil", "fault_diagnosis", "_build_oil_advisory_case"),
        _SystemLogicTemplateSpec("oil", "quantitative_duration", "_build_oil_reserve_case"),
        _SystemLogicTemplateSpec("fuel", "quantitative_duration", "_build_fuel_endurance_case"),
        _SystemLogicTemplateSpec("fuel", "dependency_trace", "_build_fuel_feed_path_case"),
        _SystemLogicTemplateSpec("electrical", "dependency_trace", "_build_electrical_transfer_case"),
        _SystemLogicTemplateSpec("electrical", "quantitative_duration", "_build_electrical_battery_case"),
        _SystemLogicTemplateSpec("hydraulic", "rule_application", "_build_hydraulic_transfer_case"),
        _SystemLogicTemplateSpec("hydraulic", "state_diagnosis", "_build_hydraulic_warning_case"),
        _SystemLogicTemplateSpec("thermal", "state_diagnosis", "_build_thermal_fan_case"),
        _SystemLogicTemplateSpec("thermal", "rule_application", "_build_thermal_derate_case"),
    )

    def __init__(self, *, seed: int) -> None:
        self._rng = SeededRng(seed)
        self._scenario_index = 0

    @classmethod
    def supported_families_for_reasoning_family(cls, reasoning_family: str) -> tuple[str, ...]:
        target = canonical_system_logic_reasoning_family(reasoning_family)
        ordered: list[str] = []
        seen: set[str] = set()
        for family in cls._SYSTEM_FAMILIES:
            if any(
                spec.family == family and canonical_system_logic_reasoning_family(spec.reasoning_mode) == target
                for spec in cls._TEMPLATE_SPECS
            ):
                if family not in seen:
                    seen.add(family)
                    ordered.append(family)
        return tuple(ordered)

    @classmethod
    def reasoning_families(cls) -> tuple[str, ...]:
        ordered: list[str] = []
        seen: set[str] = set()
        for spec in cls._TEMPLATE_SPECS:
            family = canonical_system_logic_reasoning_family(spec.reasoning_mode)
            if family not in seen:
                seen.add(family)
                ordered.append(family)
        return tuple(ordered)

    def next_problem(self, *, difficulty: float) -> Problem:
        family = self._SYSTEM_FAMILIES[self._scenario_index % len(self._SYSTEM_FAMILIES)]
        return self.next_problem_for_selection(difficulty=difficulty, family=family)

    def next_problem_for_selection(
        self,
        *,
        difficulty: float,
        family: str | None = None,
        reasoning_family: str | None = None,
    ) -> Problem:
        difficulty = clamp01(difficulty)
        self._scenario_index += 1
        scenario_code = f"SYS-{self._scenario_index:03d}"
        spec = self._pick_template_spec(family=family, reasoning_family=reasoning_family)
        payload = self._payload_from_spec(spec=spec, scenario_code=scenario_code, difficulty=difficulty)
        prompt = (
            f"{payload.scenario_code}\n"
            f"{payload.question}\n"
            "Use Up/Down to inspect the right-side index, select A-E or 1-5, then press Enter."
        )
        return Problem(prompt=prompt, answer=payload.correct_choice_code, payload=payload)

    def _pick_template_spec(
        self,
        *,
        family: str | None,
        reasoning_family: str | None,
    ) -> _SystemLogicTemplateSpec:
        requested_family = None if family is None else str(family).strip().lower()
        requested_reasoning = (
            None if reasoning_family is None else canonical_system_logic_reasoning_family(reasoning_family)
        )
        matches = tuple(
            spec
            for spec in self._TEMPLATE_SPECS
            if (requested_family is None or spec.family == requested_family)
            and (
                requested_reasoning is None
                or canonical_system_logic_reasoning_family(spec.reasoning_mode) == requested_reasoning
            )
        )
        if not matches:
            raise ValueError(
                f"No System Logic template for family={requested_family!r}, reasoning_family={requested_reasoning!r}"
            )
        if len(matches) == 1:
            return matches[0]
        return matches[int(self._rng.randint(0, len(matches) - 1))]

    def _payload_from_spec(
        self,
        *,
        spec: _SystemLogicTemplateSpec,
        scenario_code: str,
        difficulty: float,
    ) -> SystemLogicPayload:
        builder = getattr(self, spec.builder_name)
        return builder(scenario_code=scenario_code, difficulty=difficulty)

    def _build_fuel_endurance_case(
        self,
        *,
        scenario_code: str,
        difficulty: float,
    ) -> SystemLogicPayload:
        forward_l = self._rng.randint(22, 34) * 100
        aft_l = self._rng.randint(18, 30) * 100
        reserve_l = self._rng.randint(4, 8) * 100
        burn_lpm = self._rng.randint(7, 11) * 10
        crossfeed_open = self._rng.random() < (0.35 + difficulty * 0.35)
        selected_feed = "FORWARD" if self._rng.random() < 0.5 else "AFT"

        if crossfeed_open:
            accessible_l = max(0, forward_l + aft_l - reserve_l)
        elif selected_feed == "FORWARD":
            accessible_l = max(0, forward_l - reserve_l)
        else:
            accessible_l = max(0, aft_l - reserve_l)
        correct_minutes = max(1, accessible_l // burn_lpm)

        distractors = {
            max(1, forward_l // burn_lpm),
            max(1, aft_l // burn_lpm),
            max(1, (forward_l + aft_l) // burn_lpm),
            max(1, (forward_l + aft_l - reserve_l // 2) // burn_lpm),
            max(1, accessible_l // max(1, burn_lpm + 10)),
        }
        choices = self._numeric_choices(correct_minutes, distractors, suffix=" min")

        entries = (
            SystemLogicIndexEntry(
                code=0,
                label="Tank Status",
                top_document=self._table_doc(
                    title="Fuel Table",
                    headers=("Tank", "Litres"),
                    rows=(
                        ("Forward", str(forward_l)),
                        ("Aft", str(aft_l)),
                        ("Locked reserve", str(reserve_l)),
                    ),
                ),
                bottom_document=self._facts_doc(
                    title="Feed Selector",
                    lines=(
                        f"Selected feed tank: {selected_feed}",
                        f"Crossfeed valve: {'OPEN' if crossfeed_open else 'CLOSED'}",
                        "Locked reserve cannot be used for endurance.",
                    ),
                ),
            ),
            SystemLogicIndexEntry(
                code=1,
                label="Flow Rule",
                top_document=self._equation_doc(
                    title="Endurance Rule",
                    lines=(
                        "accessible fuel = chosen tanks - locked reserve",
                        "minutes available = floor(accessible fuel / burn rate)",
                    ),
                ),
                bottom_document=self._diagram_doc(
                    title="Feed Path",
                    paths=(
                        ("Forward", "Crossfeed", "Feed manifold", "Engine"),
                        ("Aft", "Crossfeed", "Feed manifold", "Engine"),
                    ),
                ),
            ),
            SystemLogicIndexEntry(
                code=2,
                label="Mission Card",
                top_document=self._facts_doc(
                    title="Mission Load",
                    lines=(
                        f"Current burn rate: {burn_lpm} L/min",
                        "Assume steady consumption throughout the segment.",
                    ),
                ),
                bottom_document=self._graph_doc(
                    title="Burn Reference",
                    points=(
                        ("Idle", max(10, burn_lpm - 30)),
                        ("Cruise", burn_lpm),
                        ("Climb", burn_lpm + 20),
                    ),
                    unit="L/min",
                ),
            ),
            SystemLogicIndexEntry(
                code=3,
                label="Answer Check",
                top_document=self._facts_doc(
                    title="Question Focus",
                    lines=(
                        "Compute usable endurance with current feed routing.",
                        "Round down to a whole minute.",
                    ),
                ),
                bottom_document=self._facts_doc(
                    title="Common Errors",
                    lines=(
                        "Do not count locked reserve.",
                        "Do not use both tanks unless crossfeed is open.",
                    ),
                ),
            ),
        )

        question = "How many whole minutes of fuel are available before the locked reserve is reached?"
        return SystemLogicPayload(
            scenario_code=scenario_code,
            system_family="fuel",
            index_entries=entries,
            question=question,
            answer_choices=choices,
            correct_choice_code=self._find_correct_choice_code(choices, f"{correct_minutes} min"),
            reasoning_mode="quantitative_duration",
            required_index_codes=(0, 1, 2),
            required_document_kinds=("table", "equation"),
        )

    def _build_fuel_feed_path_case(
        self,
        *,
        scenario_code: str,
        difficulty: float,
    ) -> SystemLogicPayload:
        _ = difficulty
        forward_pump_ok = self._rng.random() < 0.8
        aft_pump_ok = self._rng.random() < 0.8
        crossfeed_open = self._rng.random() < 0.55
        selected_feed = str(self._rng.choice(("FORWARD", "AFT", "BALANCE")))

        if selected_feed == "BALANCE" and crossfeed_open and forward_pump_ok and aft_pump_ok:
            correct_text = "Both tanks through crossfeed"
        elif selected_feed in ("FORWARD", "BALANCE") and forward_pump_ok:
            correct_text = "Forward tank only"
        elif selected_feed in ("AFT", "BALANCE") and aft_pump_ok:
            correct_text = "Aft tank only"
        elif crossfeed_open and forward_pump_ok:
            correct_text = "Forward tank only"
        elif crossfeed_open and aft_pump_ok:
            correct_text = "Aft tank only"
        else:
            correct_text = "No tank reaches engine"

        choices = self._text_choices(
            correct_text=correct_text,
            distractors=(
                "Forward tank only",
                "Aft tank only",
                "Both tanks through crossfeed",
                "Center collector only",
                "No tank reaches engine",
            ),
        )

        entries = (
            SystemLogicIndexEntry(
                code=0,
                label="Pump Status",
                top_document=self._table_doc(
                    title="Feed Pumps",
                    headers=("Source", "Status"),
                    rows=(
                        ("Forward boost pump", "OK" if forward_pump_ok else "FAILED"),
                        ("Aft boost pump", "OK" if aft_pump_ok else "FAILED"),
                        ("Crossfeed valve", "OPEN" if crossfeed_open else "CLOSED"),
                    ),
                ),
                bottom_document=self._facts_doc(
                    title="Selector State",
                    lines=(
                        f"Feed selector: {selected_feed}",
                        "Only the active feed path reaches the engine manifold.",
                    ),
                ),
            ),
            SystemLogicIndexEntry(
                code=1,
                label="Feed Diagram",
                top_document=self._diagram_doc(
                    title="Routing Map",
                    paths=(
                        ("Forward tank", "Forward pump", "Selector", "Engine"),
                        ("Aft tank", "Aft pump", "Selector", "Engine"),
                        ("Forward tank", "Crossfeed", "Aft branch", "Selector"),
                    ),
                ),
                bottom_document=self._facts_doc(
                    title="Crossfeed Rule",
                    lines=(
                        "Crossfeed only helps when the valve is open.",
                        "BALANCE uses both branches only when both boost pumps are available.",
                    ),
                ),
            ),
            SystemLogicIndexEntry(
                code=2,
                label="Fault Notes",
                top_document=self._facts_doc(
                    title="Failure Handling",
                    lines=(
                        "A failed pump cannot deliver from its own tank.",
                        "There is no gravity-feed bypass for this item.",
                    ),
                ),
                bottom_document=self._graph_doc(
                    title="Delivery Margin",
                    points=(("Blocked", 0), ("Single", 1), ("Balanced", 2)),
                    unit="paths",
                ),
            ),
            SystemLogicIndexEntry(
                code=3,
                label="Question Focus",
                top_document=self._facts_doc(
                    title="Operator Task",
                    lines=("Identify the fuel path that still reaches the engine right now.",),
                ),
                bottom_document=self._facts_doc(
                    title="Common Error",
                    lines=("Do not assume the failed branch can contribute through a closed selector.",),
                ),
            ),
        )

        question = "Which feed path still delivers fuel to the engine under the current routing and pump states?"
        return SystemLogicPayload(
            scenario_code=scenario_code,
            system_family="fuel",
            index_entries=entries,
            question=question,
            answer_choices=choices,
            correct_choice_code=self._find_correct_choice_code(choices, correct_text),
            reasoning_mode="dependency_trace",
            required_index_codes=(0, 1, 2),
            required_document_kinds=("table", "diagram"),
        )

    def _build_electrical_transfer_case(
        self,
        *,
        scenario_code: str,
        difficulty: float,
    ) -> SystemLogicPayload:
        transfer_ceiling = 1 + int(difficulty >= 0.4) + int(difficulty >= 0.75)
        failure_bus = "MISSION"
        loads = (
            ("EO turret", "MISSION", 3),
            ("Data link", "MISSION", 2),
            ("IFF transponder", "MISSION", 1),
            ("Cabin fan", "HOTEL", 4),
        )
        survivors = [
            name for name, bus, priority in loads if bus == failure_bus and priority <= transfer_ceiling
        ]
        correct_name = survivors[-1] if survivors else "No transfer load"

        choices = self._text_choices(
            correct_text=correct_name,
            distractors=tuple(name for name, _, _ in loads) + ("No transfer load",),
        )

        entries = (
            SystemLogicIndexEntry(
                code=0,
                label="Bus Map",
                top_document=self._diagram_doc(
                    title="Electrical Routing",
                    paths=(
                        ("Generator A", "ESSENTIAL", "Battery"),
                        ("Generator B", "MISSION", "Transfer relay", "ESSENTIAL"),
                    ),
                ),
                bottom_document=self._facts_doc(
                    title="Fault Log",
                    lines=(
                        f"Failed bus: {failure_bus}",
                        "Battery remains healthy.",
                    ),
                ),
            ),
            SystemLogicIndexEntry(
                code=1,
                label="Load Table",
                top_document=self._table_doc(
                    title="Mission Loads",
                    headers=("Load", "Bus", "Priority"),
                    rows=tuple((name, bus, str(priority)) for name, bus, priority in loads),
                ),
                bottom_document=self._facts_doc(
                    title="Priority Meaning",
                    lines=(
                        "Priority 1 = retain first",
                        "Higher number = shed sooner",
                    ),
                ),
            ),
            SystemLogicIndexEntry(
                code=2,
                label="Transfer Rule",
                top_document=self._equation_doc(
                    title="Relay Rule",
                    lines=(
                        "Only failed-bus loads can transfer.",
                        "Transfer allowed when priority <= ceiling.",
                        "Loads above ceiling are shed.",
                    ),
                ),
                bottom_document=self._facts_doc(
                    title="Current Ceiling",
                    lines=(f"Transfer ceiling for this fault: priority {transfer_ceiling}",),
                ),
            ),
            SystemLogicIndexEntry(
                code=3,
                label="Operator Task",
                top_document=self._facts_doc(
                    title="Question Focus",
                    lines=("Identify the mission load that remains powered after the fault.",),
                ),
                bottom_document=self._graph_doc(
                    title="Transfer Margin",
                    points=(("P1", 1), ("P2", 2), ("P3", 3), ("P4", 4)),
                    unit="priority",
                ),
            ),
        )

        question = "Which mission-bus load remains powered after the transfer relay applies the current ceiling?"
        return SystemLogicPayload(
            scenario_code=scenario_code,
            system_family="electrical",
            index_entries=entries,
            question=question,
            answer_choices=choices,
            correct_choice_code=self._find_correct_choice_code(choices, correct_name),
            reasoning_mode="dependency_trace",
            required_index_codes=(0, 1, 2),
            required_document_kinds=("diagram", "table"),
        )

    def _build_electrical_battery_case(
        self,
        *,
        scenario_code: str,
        difficulty: float,
    ) -> SystemLogicPayload:
        battery_wh = self._rng.randint(15, 22) * 100
        reserve_wh = self._rng.randint(3, 5) * 100
        essential_w = self._rng.randint(7, 11) * 10
        mission_w = self._rng.randint(6, 10) * 10
        hotel_w = self._rng.randint(4, 8) * 10
        active_profile = str(self._rng.choice(("ESSENTIAL ONLY", "ESSENTIAL + MISSION", "ALL BUSES")))
        load_w = essential_w
        if active_profile in ("ESSENTIAL + MISSION", "ALL BUSES"):
            load_w += mission_w
        if active_profile == "ALL BUSES":
            load_w += hotel_w
        usable_wh = max(0, battery_wh - reserve_wh)
        correct_minutes = max(1, (usable_wh * 60) // max(1, load_w))

        distractors = {
            max(1, (battery_wh * 60) // max(1, load_w)),
            max(1, (usable_wh * 60) // max(1, essential_w)),
            max(1, (usable_wh * 60) // max(1, mission_w + essential_w)),
            max(1, (reserve_wh * 60) // max(1, load_w)),
            max(1, (usable_wh * 60) // max(1, load_w + 20)),
        }
        choices = self._numeric_choices(correct_minutes, distractors, suffix=" min")

        entries = (
            SystemLogicIndexEntry(
                code=0,
                label="Battery State",
                top_document=self._table_doc(
                    title="Battery Snapshot",
                    headers=("Metric", "Value"),
                    rows=(
                        ("Battery energy", f"{battery_wh} Wh"),
                        ("Locked reserve", f"{reserve_wh} Wh"),
                    ),
                ),
                bottom_document=self._facts_doc(
                    title="Bus Profile",
                    lines=(
                        f"Active bus profile: {active_profile}",
                        "Locked reserve cannot be counted for endurance.",
                    ),
                ),
            ),
            SystemLogicIndexEntry(
                code=1,
                label="Load Table",
                top_document=self._table_doc(
                    title="Live Electrical Loads",
                    headers=("Load group", "Draw"),
                    rows=(
                        ("Essential", f"{essential_w} W"),
                        ("Mission", f"{mission_w} W"),
                        ("Hotel", f"{hotel_w} W"),
                    ),
                ),
                bottom_document=self._facts_doc(
                    title="Load Rule",
                    lines=("Only the load groups in the active profile count for battery time.",),
                ),
            ),
            SystemLogicIndexEntry(
                code=2,
                label="Endurance Rule",
                top_document=self._equation_doc(
                    title="Battery Time Rule",
                    lines=(
                        "usable energy = battery energy - locked reserve",
                        "minutes = floor((usable energy / live load) * 60)",
                    ),
                ),
                bottom_document=self._graph_doc(
                    title="Reference Load Bands",
                    points=(("Essential", essential_w), ("Profile", load_w), ("All buses", essential_w + mission_w + hotel_w)),
                    unit="W",
                ),
            ),
            SystemLogicIndexEntry(
                code=3,
                label="Question Focus",
                top_document=self._facts_doc(
                    title="Operator Task",
                    lines=("Compute whole-minute battery endurance for the current bus profile.",),
                ),
                bottom_document=self._facts_doc(
                    title="Common Error",
                    lines=("Do not include locked reserve or offline load groups.",),
                ),
            ),
        )

        question = "How many whole minutes of battery time remain before the locked reserve is reached?"
        return SystemLogicPayload(
            scenario_code=scenario_code,
            system_family="electrical",
            index_entries=entries,
            question=question,
            answer_choices=choices,
            correct_choice_code=self._find_correct_choice_code(choices, f"{correct_minutes} min"),
            reasoning_mode="quantitative_duration",
            required_index_codes=(0, 1, 2),
            required_document_kinds=("table", "equation"),
        )

    def _build_hydraulic_transfer_case(
        self,
        *,
        scenario_code: str,
        difficulty: float,
    ) -> SystemLogicPayload:
        pump_a_psi = self._rng.randint(11, 18) * 100
        pump_b_psi = self._rng.randint(23, 31) * 100
        transfer_threshold = 1900 + int(difficulty * 300)
        actuators = (
            ("Wheel brakes", "SECONDARY", "YES", 1200),
            ("Landing gear", "PRIMARY", "YES", 1600),
            ("Flaps", "PRIMARY", "NO", 1500),
            ("Nose steering", "SECONDARY", "NO", 1100),
        )

        valid = [
            name
            for name, primary_source, dual_feed, demand in actuators
            if demand <= pump_b_psi
            and (
                primary_source == "SECONDARY"
                or (pump_a_psi < transfer_threshold and dual_feed == "YES")
            )
        ]
        correct_name = "Landing gear" if "Landing gear" in valid else valid[0]

        choices = self._text_choices(
            correct_text=correct_name,
            distractors=tuple(name for name, _, _, _ in actuators),
        )

        entries = (
            SystemLogicIndexEntry(
                code=0,
                label="Pressure Table",
                top_document=self._table_doc(
                    title="Hydraulic Sources",
                    headers=("Source", "Pressure psi"),
                    rows=(
                        ("Pump A", str(pump_a_psi)),
                        ("Pump B", str(pump_b_psi)),
                        ("Transfer threshold", str(transfer_threshold)),
                    ),
                ),
                bottom_document=self._facts_doc(
                    title="Fault State",
                    lines=("Pump A pressure is degraded during the current segment.",),
                ),
            ),
            SystemLogicIndexEntry(
                code=1,
                label="Actuator Map",
                top_document=self._table_doc(
                    title="Actuator Demands",
                    headers=("Actuator", "Primary", "Dual", "Demand"),
                    rows=tuple(
                        (name, primary_source, dual_feed, str(demand))
                        for name, primary_source, dual_feed, demand in actuators
                    ),
                ),
                bottom_document=self._diagram_doc(
                    title="Manifold Layout",
                    paths=(
                        ("Pump A", "Primary manifold", "Landing gear"),
                        ("Pump B", "Secondary manifold", "Wheel brakes"),
                        ("Secondary manifold", "Transfer valve", "Landing gear"),
                    ),
                ),
            ),
            SystemLogicIndexEntry(
                code=2,
                label="Transfer Rule",
                top_document=self._equation_doc(
                    title="Valve Logic",
                    lines=(
                        "Transfer valve opens when Pump A < threshold.",
                        "Only dual-feed actuators can use transferred pressure.",
                        "An actuator must still meet its demand pressure.",
                    ),
                ),
                bottom_document=self._facts_doc(
                    title="Decision Aid",
                    lines=("Pick the actuator that remains operable right now.",),
                ),
            ),
            SystemLogicIndexEntry(
                code=3,
                label="Operator Note",
                top_document=self._facts_doc(
                    title="Current Task",
                    lines=("Assume no accumulator assist and no partial-credit operation.",),
                ),
                bottom_document=self._graph_doc(
                    title="Pump Margin",
                    points=(("Pump A", pump_a_psi), ("Pump B", pump_b_psi)),
                    unit="psi",
                ),
            ),
        )

        question = "Which actuator can still operate after the transfer valve opens under the current pressures?"
        return SystemLogicPayload(
            scenario_code=scenario_code,
            system_family="hydraulic",
            index_entries=entries,
            question=question,
            answer_choices=choices,
            correct_choice_code=self._find_correct_choice_code(choices, correct_name),
            reasoning_mode="rule_application",
            required_index_codes=(0, 1, 2),
            required_document_kinds=("table", "diagram"),
        )

    def _build_hydraulic_warning_case(
        self,
        *,
        scenario_code: str,
        difficulty: float,
    ) -> SystemLogicPayload:
        primary_psi = self._rng.randint(12, 18) * 100
        secondary_psi = self._rng.randint(10, 16) * 100
        fluid_temp_c = self._rng.randint(84, 114)
        hot_limit = 104 - int(difficulty * 4)

        if primary_psi < 1400 and secondary_psi < 1200:
            correct_text = "DUAL PRESSURE"
        elif fluid_temp_c >= hot_limit and primary_psi >= 1400 and secondary_psi >= 1200:
            correct_text = "OVERHEAT"
        elif primary_psi < 1400:
            correct_text = "PRIMARY LOW"
        elif secondary_psi < 1200:
            correct_text = "SECONDARY LOW"
        else:
            correct_text = "NORMAL"

        choices = self._text_choices(
            correct_text=correct_text,
            distractors=("NORMAL", "PRIMARY LOW", "SECONDARY LOW", "DUAL PRESSURE", "OVERHEAT"),
        )

        entries = (
            SystemLogicIndexEntry(
                code=0,
                label="Sensor Snapshot",
                top_document=self._table_doc(
                    title="Hydraulic Sensors",
                    headers=("Metric", "Value"),
                    rows=(
                        ("Primary pressure", f"{primary_psi} psi"),
                        ("Secondary pressure", f"{secondary_psi} psi"),
                        ("Fluid temperature", f"{fluid_temp_c} C"),
                    ),
                ),
                bottom_document=self._facts_doc(
                    title="Sampling Note",
                    lines=("Readings are steady and free of transient spikes.",),
                ),
            ),
            SystemLogicIndexEntry(
                code=1,
                label="Pressure Limits",
                top_document=self._graph_doc(
                    title="Reference Limits",
                    points=(("Primary", 1400), ("Secondary", 1200), ("Hot", hot_limit)),
                    unit="limit",
                ),
                bottom_document=self._facts_doc(
                    title="Limit Use",
                    lines=(f"Overheat requires fluid temperature of {hot_limit} C or above.",),
                ),
            ),
            SystemLogicIndexEntry(
                code=2,
                label="Advisory Rule",
                top_document=self._equation_doc(
                    title="Priority Logic",
                    lines=(
                        "Dual pressure outranks single-source warnings.",
                        "Overheat applies only when both pressure channels remain nominal.",
                        "Otherwise show the highest-priority low-pressure warning.",
                    ),
                ),
                bottom_document=self._diagram_doc(
                    title="Channel Map",
                    paths=(
                        ("Primary pump", "Primary manifold", "Primary sensor"),
                        ("Secondary pump", "Secondary manifold", "Secondary sensor"),
                    ),
                ),
            ),
            SystemLogicIndexEntry(
                code=3,
                label="Question Focus",
                top_document=self._facts_doc(
                    title="Operator Task",
                    lines=("Choose the single highest-priority hydraulic advisory.",),
                ),
                bottom_document=self._facts_doc(
                    title="Common Error",
                    lines=("Do not show OVERHEAT if either pressure channel is already below limit.",),
                ),
            ),
        )

        question = "Which hydraulic advisory should be shown right now?"
        return SystemLogicPayload(
            scenario_code=scenario_code,
            system_family="hydraulic",
            index_entries=entries,
            question=question,
            answer_choices=choices,
            correct_choice_code=self._find_correct_choice_code(choices, correct_text),
            reasoning_mode="state_diagnosis",
            required_index_codes=(0, 1, 2),
            required_document_kinds=("table", "graph"),
        )

    def _build_oil_advisory_case(
        self,
        *,
        scenario_code: str,
        difficulty: float,
    ) -> SystemLogicPayload:
        temp_c = self._rng.randint(75, 118)
        pressure_psi = self._rng.randint(34, 62)
        min_curve = (
            ("80C", 55),
            ("90C", 52),
            ("100C", 48),
            ("110C", 44),
            ("120C", 40),
        )
        if temp_c <= 85:
            required_min = 55
        elif temp_c <= 95:
            required_min = 52
        elif temp_c <= 105:
            required_min = 48
        elif temp_c <= 115:
            required_min = 44
        else:
            required_min = 40

        pressure_delta = required_min - pressure_psi
        if temp_c >= 112 and pressure_delta <= 0:
            correct_text = "OVERHEAT"
        elif pressure_delta >= 6:
            correct_text = "LOW PRESSURE"
        elif pressure_delta >= 1:
            correct_text = "CAUTION"
        else:
            correct_text = "NORMAL"

        choices = self._text_choices(
            correct_text=correct_text,
            distractors=("NORMAL", "CAUTION", "LOW PRESSURE", "FILTER BYPASS", "OVERHEAT"),
        )

        entries = (
            SystemLogicIndexEntry(
                code=0,
                label="Sensor Snapshot",
                top_document=self._table_doc(
                    title="Gearbox B Sensors",
                    headers=("Metric", "Value"),
                    rows=(
                        ("Oil temperature", f"{temp_c} C"),
                        ("Oil pressure", f"{pressure_psi} psi"),
                    ),
                ),
                bottom_document=self._facts_doc(
                    title="Context",
                    lines=("Readings are stabilized and already corrected for altitude.",),
                ),
            ),
            SystemLogicIndexEntry(
                code=1,
                label="Pressure Curve",
                top_document=self._graph_doc(
                    title="Minimum Pressure by Temperature",
                    points=min_curve,
                    unit="psi",
                ),
                bottom_document=self._facts_doc(
                    title="Graph Use",
                    lines=("Use the next highest temperature band when between marks.",),
                ),
            ),
            SystemLogicIndexEntry(
                code=2,
                label="Advisory Rule",
                top_document=self._equation_doc(
                    title="Advisory Logic",
                    lines=(
                        "pressure >= minimum => NORMAL unless temperature >= 112C",
                        "1-5 psi below minimum => CAUTION",
                        "6+ psi below minimum => LOW PRESSURE",
                        "temperature >= 112C with pressure normal => OVERHEAT",
                    ),
                ),
                bottom_document=self._facts_doc(
                    title="Selection Rule",
                    lines=("Choose the single highest-priority advisory that applies.",),
                ),
            ),
            SystemLogicIndexEntry(
                code=3,
                label="Flow Map",
                top_document=self._diagram_doc(
                    title="Lubrication Path",
                    paths=(("Reservoir", "Pump", "Gearbox B", "Cooler"),),
                ),
                bottom_document=self._facts_doc(
                    title="Question Focus",
                    lines=("Determine the advisory for Gearbox B only.",),
                ),
            ),
        )

        question = "Which advisory should be shown for Gearbox B?"
        return SystemLogicPayload(
            scenario_code=scenario_code,
            system_family="oil",
            index_entries=entries,
            question=question,
            answer_choices=choices,
            correct_choice_code=self._find_correct_choice_code(choices, correct_text),
            reasoning_mode="fault_diagnosis",
            required_index_codes=(0, 1, 2),
            required_document_kinds=("table", "graph"),
        )

    def _build_oil_reserve_case(
        self,
        *,
        scenario_code: str,
        difficulty: float,
    ) -> SystemLogicPayload:
        sump_l = self._rng.randint(42, 58)
        reserve_l = self._rng.randint(9, 14)
        leak_rate_lpm = self._rng.randint(2, 5)
        operating_mode = str(self._rng.choice(("CRUISE", "CLIMB", "DASH")))
        multiplier = {"CRUISE": 1, "CLIMB": 2, "DASH": 3}[operating_mode]
        effective_rate = leak_rate_lpm * multiplier
        correct_minutes = max(1, (max(0, sump_l - reserve_l)) // max(1, effective_rate))

        distractors = {
            max(1, (sump_l - reserve_l) // max(1, leak_rate_lpm)),
            max(1, sump_l // max(1, effective_rate)),
            max(1, (sump_l - reserve_l) // max(1, effective_rate + 1)),
            max(1, (sump_l - reserve_l) // max(1, effective_rate - 1)),
            max(1, reserve_l // max(1, leak_rate_lpm)),
        }
        choices = self._numeric_choices(correct_minutes, distractors, suffix=" min")

        entries = (
            SystemLogicIndexEntry(
                code=0,
                label="Sump Status",
                top_document=self._table_doc(
                    title="Oil Quantity",
                    headers=("Metric", "Value"),
                    rows=(("Current sump", f"{sump_l} L"), ("Caution reserve", f"{reserve_l} L")),
                ),
                bottom_document=self._facts_doc(
                    title="Leak Source",
                    lines=(f"Base leak rate: {leak_rate_lpm} L/min",),
                ),
            ),
            SystemLogicIndexEntry(
                code=1,
                label="Mode Curve",
                top_document=self._graph_doc(
                    title="Leak Multipliers",
                    points=(("CRUISE", 1), ("CLIMB", 2), ("DASH", 3)),
                    unit="x rate",
                ),
                bottom_document=self._facts_doc(
                    title="Current Mode",
                    lines=(f"Operating mode: {operating_mode}",),
                ),
            ),
            SystemLogicIndexEntry(
                code=2,
                label="Endurance Rule",
                top_document=self._equation_doc(
                    title="Leak Endurance",
                    lines=(
                        "effective rate = base leak rate x mode multiplier",
                        "minutes = floor((current sump - caution reserve) / effective rate)",
                    ),
                ),
                bottom_document=self._facts_doc(
                    title="Selection Rule",
                    lines=("Use whole minutes only and do not count the caution reserve.",),
                ),
            ),
            SystemLogicIndexEntry(
                code=3,
                label="Question Focus",
                top_document=self._diagram_doc(
                    title="Lubrication Branch",
                    paths=(("Reservoir", "Scavenge pump", "Sump", "Leak point"),),
                ),
                bottom_document=self._facts_doc(
                    title="Operator Task",
                    lines=("Compute time remaining until the sump reaches the caution reserve.",),
                ),
            ),
        )

        question = "How many whole minutes remain before the oil sump reaches the caution reserve?"
        return SystemLogicPayload(
            scenario_code=scenario_code,
            system_family="oil",
            index_entries=entries,
            question=question,
            answer_choices=choices,
            correct_choice_code=self._find_correct_choice_code(choices, f"{correct_minutes} min"),
            reasoning_mode="quantitative_duration",
            required_index_codes=(0, 1, 2),
            required_document_kinds=("table", "equation"),
        )

    def _build_thermal_fan_case(
        self,
        *,
        scenario_code: str,
        difficulty: float,
    ) -> SystemLogicPayload:
        loop1_kw = self._rng.randint(5, 8) * 10
        loop2_kw = self._rng.randint(6, 10) * 10
        fan_curve = (
            ("LOW", 70),
            ("MEDIUM", 95),
            ("HIGH", 120),
            ("BOOST", 145),
        )
        ambient_c = self._rng.randint(24, 38)
        ambient_penalty = 10 if ambient_c >= 34 else 0
        required_capacity = loop2_kw + ambient_penalty

        if required_capacity <= 70:
            correct_text = "LOW"
        elif required_capacity <= 95:
            correct_text = "MEDIUM"
        elif required_capacity <= 120:
            correct_text = "HIGH"
        else:
            correct_text = "BOOST"

        choices = self._text_choices(
            correct_text=correct_text,
            distractors=("LOW", "MEDIUM", "HIGH", "BOOST", "SHUTDOWN"),
        )

        entries = (
            SystemLogicIndexEntry(
                code=0,
                label="Loop Loads",
                top_document=self._table_doc(
                    title="Cooling Loads",
                    headers=("Loop", "Heat load kW"),
                    rows=(("Loop 1", str(loop1_kw)), ("Loop 2", str(loop2_kw))),
                ),
                bottom_document=self._facts_doc(
                    title="Environment",
                    lines=(f"Ambient temperature: {ambient_c} C",),
                ),
            ),
            SystemLogicIndexEntry(
                code=1,
                label="Fan Curve",
                top_document=self._graph_doc(
                    title="Heat Exchanger Capacity",
                    points=fan_curve,
                    unit="kW",
                ),
                bottom_document=self._facts_doc(
                    title="Hot-Day Penalty",
                    lines=("Add 10 kW required capacity when ambient is 34 C or above.",),
                ),
            ),
            SystemLogicIndexEntry(
                code=2,
                label="Loop Diagram",
                top_document=self._diagram_doc(
                    title="Cooling Paths",
                    paths=(
                        ("Loop 1", "Exchanger", "Outlet"),
                        ("Loop 2", "Exchanger", "Outlet"),
                    ),
                ),
                bottom_document=self._equation_doc(
                    title="Selection Rule",
                    lines=(
                        "Pick the lowest fan mode whose capacity >= required load.",
                        "For this item use Loop 2 load only.",
                    ),
                ),
            ),
            SystemLogicIndexEntry(
                code=3,
                label="Question Focus",
                top_document=self._facts_doc(
                    title="Operator Task",
                    lines=("Determine the minimum fan mode that keeps Loop 2 stable.",),
                ),
                bottom_document=self._facts_doc(
                    title="Stability Rule",
                    lines=("Any capacity shortfall means the loop is unstable.",),
                ),
            ),
        )

        question = "What is the lowest fan mode that keeps Loop 2 within exchanger capacity?"
        return SystemLogicPayload(
            scenario_code=scenario_code,
            system_family="thermal/cooling",
            index_entries=entries,
            question=question,
            answer_choices=choices,
            correct_choice_code=self._find_correct_choice_code(choices, correct_text),
            reasoning_mode="state_diagnosis",
            required_index_codes=(0, 1, 2),
            required_document_kinds=("table", "graph"),
        )

    def _build_thermal_derate_case(
        self,
        *,
        scenario_code: str,
        difficulty: float,
    ) -> SystemLogicPayload:
        ambient_c = self._rng.randint(26, 41)
        sink_temp_c = self._rng.randint(48, 69)
        thermal_margin = self._rng.randint(6, 14)
        corrected_temp = ambient_c + thermal_margin + max(0, sink_temp_c - 55) // 2

        if corrected_temp <= 42:
            correct_text = "FULL"
        elif corrected_temp <= 50:
            correct_text = "DERATE 1"
        elif corrected_temp <= 58:
            correct_text = "DERATE 2"
        elif corrected_temp <= 66:
            correct_text = "DERATE 3"
        else:
            correct_text = "SHUTDOWN"

        choices = self._text_choices(
            correct_text=correct_text,
            distractors=("FULL", "DERATE 1", "DERATE 2", "DERATE 3", "SHUTDOWN"),
        )

        entries = (
            SystemLogicIndexEntry(
                code=0,
                label="Thermal Inputs",
                top_document=self._table_doc(
                    title="Heat Inputs",
                    headers=("Metric", "Value"),
                    rows=(
                        ("Ambient", f"{ambient_c} C"),
                        ("Sink temperature", f"{sink_temp_c} C"),
                        ("Thermal margin", f"{thermal_margin} C"),
                    ),
                ),
                bottom_document=self._facts_doc(
                    title="Correction Note",
                    lines=("Sink temperature only adds half of the amount above 55 C.",),
                ),
            ),
            SystemLogicIndexEntry(
                code=1,
                label="Derate Bands",
                top_document=self._graph_doc(
                    title="Corrected Temperature Bands",
                    points=(("FULL", 42), ("D1", 50), ("D2", 58), ("D3", 66), ("STOP", 74)),
                    unit="C",
                ),
                bottom_document=self._facts_doc(
                    title="Band Rule",
                    lines=("Choose the first derate band that still contains the corrected temperature.",),
                ),
            ),
            SystemLogicIndexEntry(
                code=2,
                label="Computation Rule",
                top_document=self._equation_doc(
                    title="Corrected Temperature",
                    lines=(
                        "corrected temperature = ambient + thermal margin + half of sink excess above 55 C",
                        "use floor division for the half-sink term",
                    ),
                ),
                bottom_document=self._diagram_doc(
                    title="Cooling Chain",
                    paths=(("Ambient", "Sink", "Controller", "Power limiter"),),
                ),
            ),
            SystemLogicIndexEntry(
                code=3,
                label="Question Focus",
                top_document=self._facts_doc(
                    title="Operator Task",
                    lines=("Determine the required power derate band.",),
                ),
                bottom_document=self._facts_doc(
                    title="Common Error",
                    lines=("Do not add the full sink temperature above 55 C; only half of the excess counts.",),
                ),
            ),
        )

        question = "Which thermal derate band applies after the corrected temperature is calculated?"
        return SystemLogicPayload(
            scenario_code=scenario_code,
            system_family="thermal/cooling",
            index_entries=entries,
            question=question,
            answer_choices=choices,
            correct_choice_code=self._find_correct_choice_code(choices, correct_text),
            reasoning_mode="rule_application",
            required_index_codes=(0, 1, 2),
            required_document_kinds=("graph", "equation"),
        )

    @staticmethod
    def _facts_doc(*, title: str, lines: tuple[str, ...]) -> SystemLogicDocument:
        return SystemLogicDocument(title=title, kind="facts", lines=lines)

    @staticmethod
    def _table_doc(
        *,
        title: str,
        headers: tuple[str, ...],
        rows: tuple[tuple[str, ...], ...],
    ) -> SystemLogicDocument:
        return SystemLogicDocument(
            title=title,
            kind="table",
            table_headers=headers,
            table_rows=rows,
        )

    @staticmethod
    def _graph_doc(
        *,
        title: str,
        points: tuple[tuple[str, int], ...],
        unit: str,
    ) -> SystemLogicDocument:
        return SystemLogicDocument(
            title=title,
            kind="graph",
            graph_points=points,
            graph_unit=unit,
        )

    @staticmethod
    def _equation_doc(*, title: str, lines: tuple[str, ...]) -> SystemLogicDocument:
        return SystemLogicDocument(title=title, kind="equation", lines=lines)

    @staticmethod
    def _diagram_doc(
        *,
        title: str,
        paths: tuple[tuple[str, ...], ...],
    ) -> SystemLogicDocument:
        return SystemLogicDocument(title=title, kind="diagram", diagram_paths=paths)

    def _numeric_choices(
        self,
        correct_value: int,
        distractors: set[int],
        *,
        suffix: str,
    ) -> tuple[SystemLogicAnswerChoice, ...]:
        values = [correct_value]
        for candidate in sorted(distractors):
            if candidate > 0 and candidate not in values:
                values.append(candidate)
            if len(values) == 5:
                break
        offset = 5
        while len(values) < 5:
            candidate = max(1, correct_value + offset)
            if candidate not in values:
                values.append(candidate)
            offset += 5
        ordered = self._rng.sample(values, k=len(values))
        return tuple(
            SystemLogicAnswerChoice(code=index + 1, text=f"{value}{suffix}")
            for index, value in enumerate(ordered)
        )

    def _text_choices(
        self,
        *,
        correct_text: str,
        distractors: tuple[str, ...],
    ) -> tuple[SystemLogicAnswerChoice, ...]:
        texts = [correct_text]
        for candidate in distractors:
            if candidate != correct_text and candidate not in texts:
                texts.append(candidate)
            if len(texts) == 5:
                break
        ordered = self._rng.sample(texts, k=len(texts))
        return tuple(
            SystemLogicAnswerChoice(code=index + 1, text=text)
            for index, text in enumerate(ordered)
        )

    @staticmethod
    def _find_correct_choice_code(
        choices: tuple[SystemLogicAnswerChoice, ...],
        target_text: str,
    ) -> int:
        for choice in choices:
            if choice.text == target_text:
                return choice.code
        raise AssertionError(f"missing correct System Logic choice for {target_text!r}")


def build_system_logic_test(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    config: SystemLogicConfig | None = None,
) -> TimedTextInputTest:
    """Factory for the guide-style System Logic test session."""

    cfg = config or SystemLogicConfig()
    instructions = [
        "System Logic Test",
        "",
        "Use the right-side index to inspect subsystem documents.",
        "Each item uses two visible panes and mixes tables, graphs, equations, diagrams, and fact sheets.",
        "",
        "Controls during questions:",
        "- Up / Down: move through index entries 0-3",
        "- A / B / C / D / E or 1-5: choose an answer",
        "- Enter: submit the selected answer",
        "",
        "Scoring is exact correct / incorrect.",
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
