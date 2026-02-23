from __future__ import annotations

from dataclasses import dataclass

from .clock import Clock
from .cognitive_core import AnswerScorer, Problem, SeededRng, TimedTextInputTest, clamp01, lerp_int


@dataclass(frozen=True, slots=True)
class CognitiveUpdatingPayload:
    scenario_code: str
    clock_hms: str
    warning_lines: tuple[str, ...]
    message_lines: tuple[str, ...]
    pressure_low: int
    pressure_high: int
    pressure_value: int
    pump_on: bool
    comms_code: str
    comms_time_limit_s: int
    required_knots: int
    current_knots: int
    tank_levels_l: tuple[int, int, int]
    active_tank: int
    alpha_camera_due_hms: str
    bravo_camera_due_hms: str
    alpha_camera_due_s: int
    bravo_camera_due_s: int
    air_sensor_due_s: int
    ground_sensor_due_s: int
    parcel_target: tuple[int, int, int]
    dispenser_lit: int
    question: str
    answer_unit: str
    correct_value: int
    estimate_tolerance: int


@dataclass(frozen=True, slots=True)
class CognitiveUpdatingConfig:
    # Candidate guide indicates ~35 minutes including instructions.
    scored_duration_s: float = 30.0 * 60.0
    practice_questions: int = 3


@dataclass(frozen=True, slots=True)
class CognitiveUpdatingActionEvent:
    at_s: float
    action: str
    value: str


@dataclass(frozen=True, slots=True)
class CognitiveUpdatingRuntimeSnapshot:
    elapsed_s: int
    current_knots: int
    pump_on: bool
    active_tank: int
    alpha_armed: bool
    bravo_armed: bool
    air_sensor_armed: bool
    ground_sensor_armed: bool
    parcel_values: tuple[str, str, str]
    active_parcel_field: int
    comms_input: str
    dispenser_lit: int
    air_time_left_s: int
    ground_time_left_s: int
    comms_time_left_s: int
    state_code: str
    operation_score_hint: float
    event_count: int


def _distance_score(*, user_answer: int, target: int, tolerance: int) -> float:
    delta = abs(int(user_answer) - int(target))
    if delta == 0:
        return 1.0
    tol = max(1, int(tolerance))
    max_delta = tol * 2
    if delta >= max_delta:
        return 0.0
    return float(max_delta - delta) / float(max_delta)


def _pressure_digit(*, payload: CognitiveUpdatingPayload, pump_on: bool) -> int:
    adjusted_pressure = int(payload.pressure_value) + (4 if pump_on else -4)
    if adjusted_pressure < int(payload.pressure_low):
        return 1
    if adjusted_pressure > int(payload.pressure_high):
        return 3
    return 2


def _speed_digit(*, payload: CognitiveUpdatingPayload, current_knots: int) -> int:
    low = int(payload.required_knots) - 10
    high = int(payload.required_knots) + 10
    if int(current_knots) < low:
        return 1
    if int(current_knots) > high:
        return 3
    return 2


def _camera_digit(
    *,
    payload: CognitiveUpdatingPayload,
    alpha_armed_at_s: float | None,
    bravo_armed_at_s: float | None,
) -> int:
    alpha_due = int(payload.alpha_camera_due_s)
    bravo_due = int(payload.bravo_camera_due_s)

    if alpha_due <= bravo_due:
        primary_armed = alpha_armed_at_s
        primary_due = alpha_due
        alternate_armed = bravo_armed_at_s
        alternate_due = bravo_due
    else:
        primary_armed = bravo_armed_at_s
        primary_due = bravo_due
        alternate_armed = alpha_armed_at_s
        alternate_due = alpha_due

    if primary_armed is not None and primary_armed <= primary_due:
        return 1
    if alternate_armed is not None and alternate_armed <= alternate_due:
        return 2
    return 3


def _sensor_digit(
    *,
    payload: CognitiveUpdatingPayload,
    air_armed_at_s: float | None,
    ground_armed_at_s: float | None,
) -> int:
    air_due = int(payload.air_sensor_due_s)
    ground_due = int(payload.ground_sensor_due_s)

    if air_due <= ground_due:
        primary_armed = air_armed_at_s
        primary_due = air_due
        alternate_armed = ground_armed_at_s
        alternate_due = ground_due
    else:
        primary_armed = ground_armed_at_s
        primary_due = ground_due
        alternate_armed = air_armed_at_s
        alternate_due = air_due

    if primary_armed is not None and primary_armed <= primary_due:
        return 1
    if alternate_armed is not None and alternate_armed <= alternate_due:
        return 2
    return 3


def _operation_score_from_state_code(state_code: str) -> float:
    if len(state_code) != 4 or not state_code.isdigit():
        return 0.0
    digits = tuple(int(ch) for ch in state_code)
    checks = (
        digits[0] == 1,  # camera timing task
        digits[1] == 2,  # pressure in nominal range
        digits[2] == 2,  # speed in nominal range
        digits[3] == 1,  # sensor timing task
    )
    return float(sum(1 for ok in checks if ok)) / 4.0


class CognitiveUpdatingRuntime:
    """Deterministic state machine for panel operations and countdowns."""

    def __init__(self, *, payload: CognitiveUpdatingPayload, clock: Clock) -> None:
        self._payload = payload
        self._clock = clock
        self._started_at_s = float(clock.now())

        self._current_knots = int(payload.current_knots)
        self._pump_on = bool(payload.pump_on)
        self._active_tank = int(payload.active_tank)
        self._alpha_armed = False
        self._bravo_armed = False
        self._air_sensor_armed = False
        self._ground_sensor_armed = False
        self._alpha_armed_at_s: float | None = None
        self._bravo_armed_at_s: float | None = None
        self._air_sensor_armed_at_s: float | None = None
        self._ground_sensor_armed_at_s: float | None = None
        self._parcel_values = ["", "", ""]
        self._active_parcel_field = 0
        self._comms_input = ""
        self._dispenser_lit = int(payload.dispenser_lit)
        self._events: list[CognitiveUpdatingActionEvent] = []

    def _now_elapsed_s(self) -> float:
        return max(0.0, float(self._clock.now()) - self._started_at_s)

    def _record(self, action: str, value: str) -> None:
        self._events.append(
            CognitiveUpdatingActionEvent(
                at_s=self._now_elapsed_s(),
                action=action,
                value=value,
            )
        )

    def toggle_camera(self, camera: str) -> None:
        now_s = self._now_elapsed_s()
        token = camera.strip().lower()
        if token == "alpha":
            self._alpha_armed = not self._alpha_armed
            if self._alpha_armed and self._alpha_armed_at_s is None:
                self._alpha_armed_at_s = now_s
            self._record("camera_alpha", "1" if self._alpha_armed else "0")
            return
        if token == "bravo":
            self._bravo_armed = not self._bravo_armed
            if self._bravo_armed and self._bravo_armed_at_s is None:
                self._bravo_armed_at_s = now_s
            self._record("camera_bravo", "1" if self._bravo_armed else "0")

    def toggle_sensor(self, sensor: str) -> None:
        now_s = self._now_elapsed_s()
        token = sensor.strip().lower()
        if token == "air":
            self._air_sensor_armed = not self._air_sensor_armed
            if self._air_sensor_armed and self._air_sensor_armed_at_s is None:
                self._air_sensor_armed_at_s = now_s
            self._record("sensor_air", "1" if self._air_sensor_armed else "0")
            return
        if token == "ground":
            self._ground_sensor_armed = not self._ground_sensor_armed
            if self._ground_sensor_armed and self._ground_sensor_armed_at_s is None:
                self._ground_sensor_armed_at_s = now_s
            self._record("sensor_ground", "1" if self._ground_sensor_armed else "0")

    def set_pump(self, on: bool) -> None:
        self._pump_on = bool(on)
        self._record("pump", "1" if self._pump_on else "0")

    def adjust_knots(self, delta: int) -> None:
        change = int(delta)
        if change == 0:
            return
        self._current_knots = max(40, min(260, self._current_knots + change))
        self._record("knots", str(self._current_knots))

    def set_active_tank(self, tank: int) -> None:
        idx = max(1, min(3, int(tank)))
        self._active_tank = idx
        self._record("tank", str(idx))

    def activate_dispenser(self) -> None:
        self._dispenser_lit = min(5, self._dispenser_lit + 1)
        self._record("dispenser", str(self._dispenser_lit))

    def set_parcel_field(self, index: int) -> None:
        idx = max(0, min(2, int(index)))
        self._active_parcel_field = idx
        self._record("parcel_field", str(idx))

    def append_parcel_digit(self, digit: str) -> None:
        if len(digit) != 1 or not digit.isdigit():
            return
        idx = max(0, min(2, self._active_parcel_field))
        if len(self._parcel_values[idx]) >= 4:
            return
        self._parcel_values[idx] += digit
        self._record("parcel_digit", digit)
        if len(self._parcel_values[idx]) >= 3 and idx < 2:
            self._active_parcel_field = idx + 1

    def append_comms_digit(self, digit: str) -> None:
        if len(digit) != 1 or not digit.isdigit():
            return
        if len(self._comms_input) >= 4:
            return
        self._comms_input += digit
        self._record("comms_digit", digit)

    def clear_comms(self) -> None:
        self._comms_input = ""
        self._record("comms_clear", "")

    def state_code(self) -> str:
        cam_digit = _camera_digit(
            payload=self._payload,
            alpha_armed_at_s=self._alpha_armed_at_s,
            bravo_armed_at_s=self._bravo_armed_at_s,
        )
        pressure_digit = _pressure_digit(payload=self._payload, pump_on=self._pump_on)
        speed_digit = _speed_digit(payload=self._payload, current_knots=self._current_knots)
        sensor_digit = _sensor_digit(
            payload=self._payload,
            air_armed_at_s=self._air_sensor_armed_at_s,
            ground_armed_at_s=self._ground_sensor_armed_at_s,
        )
        return f"{cam_digit}{pressure_digit}{speed_digit}{sensor_digit}"

    def build_submission_raw(self) -> str:
        if len(self._comms_input) != 4:
            return ""
        code = self.state_code()
        event_count = min(99, len(self._events))
        return f"{self._comms_input}{code}{event_count:02d}"

    def snapshot(self) -> CognitiveUpdatingRuntimeSnapshot:
        elapsed = int(self._now_elapsed_s())
        code = self.state_code()
        return CognitiveUpdatingRuntimeSnapshot(
            elapsed_s=elapsed,
            current_knots=self._current_knots,
            pump_on=self._pump_on,
            active_tank=self._active_tank,
            alpha_armed=self._alpha_armed,
            bravo_armed=self._bravo_armed,
            air_sensor_armed=self._air_sensor_armed,
            ground_sensor_armed=self._ground_sensor_armed,
            parcel_values=(self._parcel_values[0], self._parcel_values[1], self._parcel_values[2]),
            active_parcel_field=self._active_parcel_field,
            comms_input=self._comms_input,
            dispenser_lit=self._dispenser_lit,
            air_time_left_s=max(0, int(self._payload.air_sensor_due_s) - elapsed),
            ground_time_left_s=max(0, int(self._payload.ground_sensor_due_s) - elapsed),
            comms_time_left_s=max(0, int(self._payload.comms_time_limit_s) - elapsed),
            state_code=code,
            operation_score_hint=_operation_score_from_state_code(code),
            event_count=len(self._events),
        )

    def events(self) -> tuple[CognitiveUpdatingActionEvent, ...]:
        return tuple(self._events)


class CognitiveUpdatingScorer(AnswerScorer):
    """Score code entry plus operation quality from deterministic runtime state."""

    def score(self, *, problem: Problem, user_answer: int, raw: str) -> float:
        payload = problem.payload
        if not isinstance(payload, CognitiveUpdatingPayload):
            return 1.0 if int(user_answer) == int(problem.answer) else 0.0

        digits = "".join(ch for ch in str(raw).strip() if ch.isdigit())
        if len(digits) >= 8:
            entered_text = digits[0:4]
            state_text = digits[4:8]
            entered = int(entered_text)
            target = int(state_text)
            code_score = _distance_score(
                user_answer=entered,
                target=target,
                tolerance=max(1, int(payload.estimate_tolerance)),
            )
            ops_score = _operation_score_from_state_code(state_text)
            combined = (0.7 * code_score) + (0.3 * ops_score)
            return max(0.0, min(1.0, combined))

        return _distance_score(
            user_answer=int(user_answer),
            target=int(payload.correct_value),
            tolerance=max(1, int(payload.estimate_tolerance)),
        )


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

        camera_due_min = lerp_int(22, 10, difficulty)
        camera_due_max = lerp_int(70, 36, difficulty)
        alpha_camera_due_s = self._rng.randint(camera_due_min, camera_due_max)
        bravo_camera_due_s = self._rng.randint(camera_due_min, camera_due_max)
        earliest_camera = "ALPHA" if alpha_camera_due_s <= bravo_camera_due_s else "BRAVO"
        camera_digit = 1 if earliest_camera == "ALPHA" else 2

        sensor_due_min = lerp_int(18, 8, difficulty)
        sensor_due_max = lerp_int(62, 30, difficulty)
        air_sensor_due_s = self._rng.randint(sensor_due_min, sensor_due_max)
        ground_sensor_due_s = self._rng.randint(sensor_due_min, sensor_due_max)
        earliest_sensor = "AIR" if air_sensor_due_s <= ground_sensor_due_s else "GROUND"
        sensor_digit = 1 if earliest_sensor == "AIR" else 2

        pressure_low = self._rng.randint(86, 94)
        pressure_high = pressure_low + self._rng.randint(14, 20)
        pressure_state = self._rng.randint(0, 2)
        pressure_span = lerp_int(16, 8, difficulty)
        if pressure_state == 0:
            pressure_value = pressure_low - self._rng.randint(1, pressure_span)
            pressure_digit = 1
            pressure_warning = "Check Pressure"
            pump_on = True
        elif pressure_state == 1:
            pressure_value = self._rng.randint(pressure_low, pressure_high)
            pressure_digit = 2
            pressure_warning = "Pressure Nominal"
            pump_on = bool(self._rng.randint(0, 1))
        else:
            pressure_value = pressure_high + self._rng.randint(1, pressure_span)
            pressure_digit = 3
            pressure_warning = "Pressure High"
            pump_on = False

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

        tank_levels = tuple(self._rng.randint(360, 460) for _ in range(3))
        active_tank = self._rng.randint(1, 3)

        warning_lines = [pressure_warning, speed_warning]
        if min(air_sensor_due_s, ground_sensor_due_s) <= 20:
            warning_lines.append("SENSOR PANEL")

        parcel_lat = self._rng.randint(120, 950)
        parcel_lon = self._rng.randint(120, 950)
        parcel_time = self._rng.randint(1, 59)
        dispenser_lit = self._rng.randint(0, 4)

        answer_digits = (camera_digit, pressure_digit, speed_digit, sensor_digit)
        answer_text = "".join(str(d) for d in answer_digits)
        answer = int(answer_text)
        tolerance = max(1, lerp_int(6, 2, difficulty))

        message_lines = (
            f"Activate Alpha Camera at: {self._fmt_hms(now_total_s + alpha_camera_due_s)}",
            f"Activate Bravo Camera at: {self._fmt_hms(now_total_s + bravo_camera_due_s)}",
            f"Air Sensor due in: {air_sensor_due_s:02d}s",
            f"Ground Sensor due in: {ground_sensor_due_s:02d}s",
        )

        payload = CognitiveUpdatingPayload(
            scenario_code=scenario_code,
            clock_hms=self._fmt_hms(now_total_s),
            warning_lines=tuple(warning_lines),
            message_lines=message_lines,
            pressure_low=pressure_low,
            pressure_high=pressure_high,
            pressure_value=pressure_value,
            pump_on=pump_on,
            comms_code=answer_text,
            comms_time_limit_s=lerp_int(48, 26, difficulty),
            required_knots=required_knots,
            current_knots=current_knots,
            tank_levels_l=tank_levels,
            active_tank=active_tank,
            alpha_camera_due_hms=self._fmt_hms(now_total_s + alpha_camera_due_s),
            bravo_camera_due_hms=self._fmt_hms(now_total_s + bravo_camera_due_s),
            alpha_camera_due_s=alpha_camera_due_s,
            bravo_camera_due_s=bravo_camera_due_s,
            air_sensor_due_s=air_sensor_due_s,
            ground_sensor_due_s=ground_sensor_due_s,
            parcel_target=(parcel_lat, parcel_lon, parcel_time),
            dispenser_lit=dispenser_lit,
            question="Enter the 4-digit comms code from the live panel state.",
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
        "Manage and coordinate simultaneous system tasks using dual multifunction displays.",
        "You must update priorities while monitoring warnings and clock cues.",
        "",
        "Goal:",
        "Compute and enter the 4-digit comms code from the live panel state.",
        "",
        "Controls during questions: mouse tabs/buttons or keyboard shortcuts.",
        "- Q/E: switch upper display (Messages, Objectives, Controls)",
        "- A/D: switch lower display (Navigation, Sensors, Engine)",
        "- Type digits and press Enter, or use panel keypads",
        "",
        "No backspace editing is available during this test.",
        "Score combines code accuracy with operation quality from your panel actions.",
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
