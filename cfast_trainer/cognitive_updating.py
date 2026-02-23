from __future__ import annotations

from dataclasses import dataclass

from .clock import Clock
from .cognitive_core import AnswerScorer, Problem, SeededRng, TimedTextInputTest, clamp01, lerp_int


@dataclass(frozen=True, slots=True)
class CognitiveUpdatingDocument:
    title: str
    kind: str
    lines: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CognitiveUpdatingPanel:
    name: str
    documents: tuple[CognitiveUpdatingDocument, ...]


@dataclass(frozen=True, slots=True)
class CognitiveUpdatingPayload:
    scenario_code: str
    warning_lines: tuple[str, ...]
    clock_hms: str
    panels: tuple[CognitiveUpdatingPanel, ...]
    question: str
    answer_unit: str
    correct_value: int
    estimate_tolerance: int


@dataclass(frozen=True, slots=True)
class CognitiveUpdatingConfig:
    # Candidate guide indicates ~35 minutes including instructions.
    scored_duration_s: float = 30.0 * 60.0
    practice_questions: int = 3


class CognitiveUpdatingScorer(AnswerScorer):
    """Exact answers receive full credit; near estimates receive partial credit."""

    def score(self, *, problem: Problem, user_answer: int, raw: str) -> float:
        _ = raw
        payload = problem.payload
        if not isinstance(payload, CognitiveUpdatingPayload):
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


class CognitiveUpdatingGenerator:
    """Deterministic multitask snapshot generator for the Cognitive Updating test."""

    def __init__(self, *, seed: int) -> None:
        self._rng = SeededRng(seed)
        self._scenario_index = 0

    def next_problem(self, *, difficulty: float) -> Problem:
        difficulty = clamp01(difficulty)
        scenario_code = self._next_scenario_code()

        clock_h = self._rng.randint(6, 9)
        clock_m = self._rng.randint(0, 59)
        clock_s = self._rng.randint(0, 59)
        now_total_s = (clock_h * 3600) + (clock_m * 60) + clock_s

        tight_due = lerp_int(18, 10, difficulty)
        wide_due = lerp_int(58, 34, difficulty)
        air_due_delta_s = self._rng.randint(tight_due, wide_due)
        ground_due_delta_s = self._rng.randint(tight_due, wide_due)
        earliest_sensor = "AIR" if air_due_delta_s <= ground_due_delta_s else "GROUND"
        sensor_digit = 1 if earliest_sensor == "AIR" else 2

        pressure_low = self._rng.randint(86, 94)
        pressure_high = pressure_low + self._rng.randint(14, 20)
        pressure_state = self._rng.randint(0, 2)
        pressure_span = lerp_int(16, 8, difficulty)
        if pressure_state == 0:
            pressure_value = pressure_low - self._rng.randint(1, pressure_span)
            pressure_digit = 1
            pressure_warning = "LOW PRESSURE"
        elif pressure_state == 1:
            pressure_value = self._rng.randint(pressure_low, pressure_high)
            pressure_digit = 2
            pressure_warning = "PRESSURE NOMINAL"
        else:
            pressure_value = pressure_high + self._rng.randint(1, pressure_span)
            pressure_digit = 3
            pressure_warning = "HIGH PRESSURE"

        required_knots = self._rng.randint(90, 160)
        speed_state = self._rng.randint(0, 2)
        speed_span = lerp_int(24, 12, difficulty)
        if speed_state == 0:
            current_knots = required_knots - (10 + self._rng.randint(1, speed_span))
            speed_digit = 1
            speed_warning = "CHECK AIRSPEED"
        elif speed_state == 1:
            current_knots = required_knots + self._rng.randint(-10, 10)
            speed_digit = 2
            speed_warning = "AIRSPEED NOMINAL"
        else:
            current_knots = required_knots + (10 + self._rng.randint(1, speed_span))
            speed_digit = 3
            speed_warning = "AIRSPEED WARNING"
        current_knots = max(40, current_knots)

        tank_values = tuple(self._rng.randint(360, 460) for _ in range(3))
        active_tank = self._rng.randint(1, 3)

        warning_lines = [pressure_warning, speed_warning]
        if min(air_due_delta_s, ground_due_delta_s) <= 20:
            warning_lines.append("SENSOR PANEL")

        answer = (sensor_digit * 100) + (pressure_digit * 10) + speed_digit
        tolerance = max(1, lerp_int(4, 1, difficulty))

        panels = (
            CognitiveUpdatingPanel(
                name="Messages",
                documents=(
                    self._doc(
                        title="Warning",
                        kind="status",
                        lines=tuple(warning_lines),
                    ),
                    self._doc(
                        title="Clock",
                        kind="status",
                        lines=(
                            f"Current: {self._fmt_hms(now_total_s)}",
                            f"Air due: {self._fmt_hms(now_total_s + air_due_delta_s)}",
                            f"Ground due: {self._fmt_hms(now_total_s + ground_due_delta_s)}",
                        ),
                    ),
                ),
            ),
            CognitiveUpdatingPanel(
                name="Objectives",
                documents=(
                    self._doc(
                        title="Mission Card",
                        kind="facts",
                        lines=(
                            "Compute the 3-digit stabilization code.",
                            "Hundreds: earliest sensor due (AIR=1, GROUND=2).",
                            "Tens: pressure state (LOW=1, RANGE=2, HIGH=3).",
                            "Ones: speed state (LOW=1, RANGE=2, HIGH=3).",
                        ),
                    ),
                    self._doc(
                        title="Comms",
                        kind="status",
                        lines=(
                            "Route alpha active.",
                            "Log code after each system update.",
                        ),
                    ),
                ),
            ),
            CognitiveUpdatingPanel(
                name="Controls",
                documents=(
                    self._doc(
                        title="Hydraulics",
                        kind="table",
                        lines=(
                            f"Current pressure: {pressure_value}",
                            f"Safe range: {pressure_low} to {pressure_high}",
                        ),
                    ),
                    self._doc(
                        title="Hydraulic Pump",
                        kind="status",
                        lines=(
                            f"Panel callout: {pressure_warning}",
                            "Pump state updates pressure channel.",
                        ),
                    ),
                ),
            ),
            CognitiveUpdatingPanel(
                name="Sensors",
                documents=(
                    self._doc(
                        title="Sensor Timers",
                        kind="table",
                        lines=(
                            f"Air Sensor: T-{air_due_delta_s:02d}s",
                            f"Ground Sensor: T-{ground_due_delta_s:02d}s",
                            f"Earliest due: {earliest_sensor}",
                        ),
                    ),
                    self._doc(
                        title="Video Recording",
                        kind="status",
                        lines=(
                            "Alpha camera ready.",
                            "Bravo camera ready.",
                        ),
                    ),
                ),
            ),
            CognitiveUpdatingPanel(
                name="Engine",
                documents=(
                    self._doc(
                        title="Airspeed",
                        kind="table",
                        lines=(
                            f"Required knots: {required_knots}",
                            f"Current knots: {current_knots}",
                            "In-range band: +/-10 knots",
                        ),
                    ),
                    self._doc(
                        title="Fuel Tanks",
                        kind="status",
                        lines=(
                            f"Tank 1: {tank_values[0]} L",
                            f"Tank 2: {tank_values[1]} L",
                            f"Tank 3: {tank_values[2]} L",
                            f"Active tank: {active_tank}",
                        ),
                    ),
                ),
            ),
        )

        payload = CognitiveUpdatingPayload(
            scenario_code=scenario_code,
            warning_lines=tuple(warning_lines),
            clock_hms=self._fmt_hms(now_total_s),
            panels=panels,
            question="Using panel data, enter the 3-digit stabilization code.",
            answer_unit="code",
            correct_value=answer,
            estimate_tolerance=tolerance,
        )

        prompt = (
            f"{payload.scenario_code}\n{payload.question}\n"
            f"Enter whole-number answer ({payload.answer_unit})."
        )
        return Problem(prompt=prompt, answer=payload.correct_value, payload=payload)

    def _next_scenario_code(self) -> str:
        self._scenario_index += 1
        return f"CUP-{self._scenario_index:03d}"

    @staticmethod
    def _doc(*, title: str, kind: str, lines: tuple[str, ...]) -> CognitiveUpdatingDocument:
        return CognitiveUpdatingDocument(title=title, kind=kind, lines=lines)

    @staticmethod
    def _fmt_hms(total_seconds: int) -> str:
        value = int(total_seconds) % (24 * 3600)
        hh = value // 3600
        mm = (value % 3600) // 60
        ss = value % 60
        return f"{hh:02d}:{mm:02d}:{ss:02d}"


def build_cognitive_updating_test(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    config: CognitiveUpdatingConfig | None = None,
) -> TimedTextInputTest:
    """Factory for the Cognitive Updating test session."""

    cfg = config or CognitiveUpdatingConfig()
    instructions = [
        "Cognitive Updating Test",
        "",
        "Manage and coordinate simultaneous system tasks using menu pages.",
        "You must update priorities while monitoring warnings and clock cues.",
        "",
        "Goal:",
        "Compute the 3-digit stabilization code from the active panel state.",
        "",
        "Controls during questions:",
        "- Left / Right or Tab: switch panel",
        "- Up / Down: switch panel page",
        "- Type whole-number answer and press Enter",
        "",
        "Scoring: exact answers score full credit; close estimates earn partial credit.",
        "Once the timed block starts, continue until completion.",
    ]

    generator = CognitiveUpdatingGenerator(seed=seed)
    return TimedTextInputTest(
        title="Cognitive Updating",
        instructions=instructions,
        generator=generator,
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        practice_questions=cfg.practice_questions,
        scored_duration_s=cfg.scored_duration_s,
        scorer=CognitiveUpdatingScorer(),
    )
