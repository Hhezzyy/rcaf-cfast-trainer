from __future__ import annotations

from dataclasses import dataclass
import math

from .clock import Clock
from .cognitive_core import AnswerScorer, Problem, SeededRng, TimedTextInputTest, clamp01, lerp_int

GRACE_WINDOW_S = 5
EVAL_INTERVAL_S = 5.0
BUTTON_FLASH_S = 0.35
PUMP_RISE_PER_S = 0.60
PUMP_FALL_PER_S = 0.32
KNOTS_DRIFT_PER_S = 0.50
ACTIVE_TANK_DRAIN_PER_S = 1.00
IDLE_TANK_DRAIN_PER_S = 0.00
MESSAGE_REVEAL_LAT_S = 3.0
MESSAGE_REVEAL_LON_S = 11.0
MESSAGE_REVEAL_COMMS_S = 20.0
MESSAGE_REVEAL_TIME_S = 23.0
COGNITIVE_UPDATING_DOMAIN_ORDER = (
    "controls",
    "navigation",
    "engine",
    "sensors",
    "objectives",
    "state_code",
)
COGNITIVE_UPDATING_SCENARIO_FAMILIES = (
    "baseline",
    "compressed",
    "staggered",
    "crosscheck",
    "recovery_window",
)


@dataclass(frozen=True, slots=True)
class _CognitiveUpdatingScenarioFamilyConfig:
    label: str
    camera_due_scale: float
    sensor_due_scale: float
    objective_deadline_scale: float
    comms_time_limit_scale: float
    message_reveal_scale: float
    pressure_drift_scale: float
    speed_drift_scale: float
    tank_drain_scale: float
    starting_upper_tab_index: int
    starting_lower_tab_index: int


_COGNITIVE_UPDATING_SCENARIO_CONFIGS: dict[str, _CognitiveUpdatingScenarioFamilyConfig] = {
    "baseline": _CognitiveUpdatingScenarioFamilyConfig(
        label="Baseline",
        camera_due_scale=1.0,
        sensor_due_scale=1.0,
        objective_deadline_scale=1.0,
        comms_time_limit_scale=1.0,
        message_reveal_scale=1.0,
        pressure_drift_scale=1.0,
        speed_drift_scale=1.0,
        tank_drain_scale=1.0,
        starting_upper_tab_index=2,
        starting_lower_tab_index=1,
    ),
    "compressed": _CognitiveUpdatingScenarioFamilyConfig(
        label="Compressed",
        camera_due_scale=0.82,
        sensor_due_scale=0.84,
        objective_deadline_scale=0.82,
        comms_time_limit_scale=0.88,
        message_reveal_scale=0.72,
        pressure_drift_scale=1.14,
        speed_drift_scale=1.16,
        tank_drain_scale=1.12,
        starting_upper_tab_index=0,
        starting_lower_tab_index=4,
    ),
    "staggered": _CognitiveUpdatingScenarioFamilyConfig(
        label="Staggered",
        camera_due_scale=1.18,
        sensor_due_scale=1.10,
        objective_deadline_scale=1.16,
        comms_time_limit_scale=1.08,
        message_reveal_scale=1.18,
        pressure_drift_scale=0.92,
        speed_drift_scale=0.94,
        tank_drain_scale=1.04,
        starting_upper_tab_index=5,
        starting_lower_tab_index=3,
    ),
    "crosscheck": _CognitiveUpdatingScenarioFamilyConfig(
        label="Crosscheck",
        camera_due_scale=0.92,
        sensor_due_scale=0.88,
        objective_deadline_scale=1.02,
        comms_time_limit_scale=0.94,
        message_reveal_scale=0.84,
        pressure_drift_scale=1.08,
        speed_drift_scale=1.10,
        tank_drain_scale=1.08,
        starting_upper_tab_index=1,
        starting_lower_tab_index=2,
    ),
    "recovery_window": _CognitiveUpdatingScenarioFamilyConfig(
        label="Recovery Window",
        camera_due_scale=1.12,
        sensor_due_scale=1.08,
        objective_deadline_scale=1.10,
        comms_time_limit_scale=1.14,
        message_reveal_scale=1.08,
        pressure_drift_scale=0.86,
        speed_drift_scale=0.88,
        tank_drain_scale=0.94,
        starting_upper_tab_index=4,
        starting_lower_tab_index=0,
    ),
}


def _clamp_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, int(value)))


def _clamp_float(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def _lerp_float(low: float, high: float, amount: float) -> float:
    t = clamp01(amount)
    return float(low) + ((float(high) - float(low)) * t)


@dataclass(frozen=True, slots=True)
class _CognitiveUpdatingDifficultyParams:
    active_domains: tuple[str, ...]
    camera_due_min_s: int
    camera_due_max_s: int
    sensor_due_min_s: int
    sensor_due_max_s: int
    objective_deadline_min_s: int
    objective_deadline_max_s: int
    comms_time_limit_s: int
    response_grace_window_s: int
    pressure_band_width: int
    speed_tolerance_knots: int
    tank_spread_tolerance_l: int
    estimate_tolerance: int
    drift_multiplier: float
    message_reveal_scale: float


def _full_mixed_domains_for_difficulty(difficulty: float) -> tuple[str, ...]:
    d = clamp01(difficulty)
    if d < 0.25:
        return ("controls", "navigation", "state_code")
    if d < 0.50:
        return ("controls", "navigation", "sensors", "state_code")
    if d < 0.75:
        return ("controls", "navigation", "engine", "sensors", "state_code")
    return COGNITIVE_UPDATING_DOMAIN_ORDER


def _difficulty_params(difficulty: float) -> _CognitiveUpdatingDifficultyParams:
    d = clamp01(difficulty)
    return _CognitiveUpdatingDifficultyParams(
        active_domains=_full_mixed_domains_for_difficulty(d),
        camera_due_min_s=lerp_int(36, 10, d),
        camera_due_max_s=lerp_int(90, 34, d),
        sensor_due_min_s=lerp_int(30, 8, d),
        sensor_due_max_s=lerp_int(82, 28, d),
        objective_deadline_min_s=lerp_int(70, 22, d),
        objective_deadline_max_s=lerp_int(130, 48, d),
        comms_time_limit_s=lerp_int(70, 24, d),
        response_grace_window_s=lerp_int(8, 3, d),
        pressure_band_width=lerp_int(28, 12, d),
        speed_tolerance_knots=lerp_int(18, 6, d),
        tank_spread_tolerance_l=lerp_int(80, 35, d),
        estimate_tolerance=max(1, lerp_int(8, 2, d)),
        drift_multiplier=0.50 + d,
        message_reveal_scale=_lerp_float(0.70, 1.20, d),
    )


def canonical_cognitive_updating_domain(name: str) -> str:
    token = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    if token in ("control", "controls_page"):
        return "controls"
    if token in ("nav", "navigation_page"):
        return "navigation"
    if token in ("eng", "engine_page"):
        return "engine"
    if token in ("sensor", "sensors_page", "video"):
        return "sensors"
    if token in ("objective", "objectives_page"):
        return "objectives"
    if token in ("code", "state", "statecode", "comms"):
        return "state_code"
    return token


def normalize_cognitive_updating_active_domains(*domains: str) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    raw_domains = domains or COGNITIVE_UPDATING_DOMAIN_ORDER
    for name in raw_domains:
        token = canonical_cognitive_updating_domain(name)
        if token in COGNITIVE_UPDATING_DOMAIN_ORDER and token not in seen:
            seen.add(token)
            ordered.append(token)
    if not ordered:
        return COGNITIVE_UPDATING_DOMAIN_ORDER
    return tuple(ordered)


def canonical_cognitive_updating_scenario_family(name: str) -> str:
    token = str(name).strip().lower()
    if token not in _COGNITIVE_UPDATING_SCENARIO_CONFIGS:
        return "baseline"
    return token


def supported_cognitive_updating_scenario_families() -> tuple[str, ...]:
    return COGNITIVE_UPDATING_SCENARIO_FAMILIES


def _distance_score(*, user_answer: int, target: int, tolerance: int) -> float:
    delta = abs(int(user_answer) - int(target))
    if delta == 0:
        return 1.0
    tol = max(1, int(tolerance))
    max_delta = tol * 2
    if delta >= max_delta:
        return 0.0
    return float(max_delta - delta) / float(max_delta)


def _domain_score(*, bad_ticks: int, eval_ticks: int, manual_hits: int) -> int:
    if eval_ticks <= 0:
        base = 100
    else:
        ratio = float(max(0, bad_ticks)) / float(max(1, eval_ticks))
        base = int(round(100.0 * (1.0 - ratio)))
    bonus = min(20, max(0, int(manual_hits)) * 2)
    return _clamp_int(base + bonus, 0, 100)


def _task_points(*, due_s: int, first_action_s: float | None) -> int:
    if first_action_s is None:
        return 0
    return 25 if abs(float(first_action_s) - float(due_s)) <= float(GRACE_WINDOW_S) else 0


def _parse_hms(hms: str) -> int:
    token = str(hms).strip()
    parts = token.split(":")
    if len(parts) != 3:
        return 0
    try:
        hh = int(parts[0])
        mm = int(parts[1])
        ss = int(parts[2])
    except ValueError:
        return 0
    hh = max(0, min(23, hh))
    mm = max(0, min(59, mm))
    ss = max(0, min(59, ss))
    return (hh * 3600) + (mm * 60) + ss


def _fmt_hms(total_seconds: int) -> str:
    value = int(total_seconds) % (24 * 3600)
    hh = value // 3600
    mm = (value % 3600) // 60
    ss = value % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


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
    objective_deadline_s: int
    dispenser_lit: int
    question: str
    answer_unit: str
    correct_value: int
    estimate_tolerance: int
    response_grace_window_s: int = GRACE_WINDOW_S
    speed_tolerance_knots: int = 10
    tank_spread_tolerance_l: int = 50
    active_domains: tuple[str, ...] = COGNITIVE_UPDATING_DOMAIN_ORDER
    scenario_family: str = "baseline"
    focus_label: str = "Full Mixed"
    starting_upper_tab_index: int = 2
    starting_lower_tab_index: int = 1
    pressure_drift_scale: float = 1.0
    speed_drift_scale: float = 1.0
    tank_drain_scale: float = 1.0
    warning_penalty_scale: float = 1.0
    message_reveal_lat_s: float = MESSAGE_REVEAL_LAT_S
    message_reveal_lon_s: float = MESSAGE_REVEAL_LON_S
    message_reveal_comms_s: float = MESSAGE_REVEAL_COMMS_S
    message_reveal_time_s: float = MESSAGE_REVEAL_TIME_S


@dataclass(frozen=True, slots=True)
class CognitiveUpdatingConfig:
    # Candidate guide indicates ~35 minutes including instructions.
    scored_duration_s: float = 30.0 * 60.0
    practice_questions: int = 3


@dataclass(frozen=True, slots=True)
class CognitiveUpdatingTrainingProfile:
    active_domains: tuple[str, ...] = COGNITIVE_UPDATING_DOMAIN_ORDER
    scenario_family: str | None = None
    focus_label: str = "Full Mixed"
    camera_due_scale: float = 1.0
    sensor_due_scale: float = 1.0
    objective_deadline_scale: float = 1.0
    comms_time_limit_scale: float = 1.0
    message_reveal_scale: float = 1.0
    pressure_drift_scale: float = 1.0
    speed_drift_scale: float = 1.0
    tank_drain_scale: float = 1.0
    warning_penalty_scale: float = 1.0
    starting_upper_tab_index: int | None = None
    starting_lower_tab_index: int | None = None


@dataclass(frozen=True, slots=True)
class CognitiveUpdatingActionEvent:
    at_s: float
    action: str
    value: str


@dataclass(frozen=True, slots=True)
class CognitiveUpdatingSubmission:
    entered_code: str
    state_code: str
    controls_score: int
    navigation_score: int
    engine_score: int
    sensors_score: int
    objectives_score: int
    warnings_penalty_points: int
    overall_score: int
    event_count: int


def decode_cognitive_updating_submission_raw(raw: str) -> CognitiveUpdatingSubmission | None:
    digits = "".join(ch for ch in str(raw).strip() if ch.isdigit())
    if len(digits) < 32:
        return None
    value = digits[:32]
    return CognitiveUpdatingSubmission(
        entered_code=value[0:4],
        state_code=value[4:8],
        controls_score=_clamp_int(int(value[8:11]), 0, 100),
        navigation_score=_clamp_int(int(value[11:14]), 0, 100),
        engine_score=_clamp_int(int(value[14:17]), 0, 100),
        sensors_score=_clamp_int(int(value[17:20]), 0, 100),
        objectives_score=_clamp_int(int(value[20:23]), 0, 100),
        warnings_penalty_points=_clamp_int(int(value[23:26]), 0, 999),
        overall_score=_clamp_int(int(value[26:29]), 0, 100),
        event_count=_clamp_int(int(value[29:32]), 0, 999),
    )


@dataclass(frozen=True, slots=True)
class CognitiveUpdatingRuntimeSnapshot:
    elapsed_s: int
    clock_hms: str
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
    comms_swap_in_s: int
    current_comms_code: str
    next_comms_code: str | None
    objective_deadline_left_s: int
    state_code: str
    operation_score_hint: float
    event_count: int
    pressure_value: int
    tank_levels_l: tuple[int, int, int]
    warning_lines: tuple[str, ...]
    message_lines: tuple[str, ...]
    controls_score: int
    navigation_score: int
    engine_score: int
    sensors_score: int
    objectives_score: int
    warnings_penalty_points: int
    overall_score: int
    objective_drop_ready: bool
    objective_drop_complete: bool


class CognitiveUpdatingRuntime:
    """Deterministic state machine for panel operations, warnings, and scoring."""

    def __init__(self, *, payload: CognitiveUpdatingPayload, clock: Clock) -> None:
        self._payload = payload
        self._clock = clock
        self._active_domains = set(normalize_cognitive_updating_active_domains(*payload.active_domains))
        self._started_at_s = float(clock.now())
        self._clock_base_s = _parse_hms(payload.clock_hms)
        self._last_domain_advance_s = {
            domain: 0.0 for domain in COGNITIVE_UPDATING_DOMAIN_ORDER
        }
        self._next_eval_s = EVAL_INTERVAL_S

        self._pressure_value = float(payload.pressure_value)
        self._current_knots = float(payload.current_knots)
        self._tank_levels = [
            float(payload.tank_levels_l[0]),
            float(payload.tank_levels_l[1]),
            float(payload.tank_levels_l[2]),
        ]

        self._pump_on = bool(payload.pump_on)
        self._active_tank = int(payload.active_tank)
        self._alpha_armed_at_s: float | None = None
        self._bravo_armed_at_s: float | None = None
        self._air_sensor_armed_at_s: float | None = None
        self._ground_sensor_armed_at_s: float | None = None
        self._alpha_last_press_s: float | None = None
        self._bravo_last_press_s: float | None = None
        self._air_last_press_s: float | None = None
        self._ground_last_press_s: float | None = None
        self._alpha_hits = 0
        self._bravo_hits = 0
        self._air_hits = 0
        self._ground_hits = 0
        self._alpha_interval_s = max(8, int(payload.alpha_camera_due_s))
        self._bravo_interval_s = max(8, int(payload.bravo_camera_due_s))
        self._air_interval_s = max(8, int(payload.air_sensor_due_s))
        self._ground_interval_s = max(8, int(payload.ground_sensor_due_s))
        self._alpha_next_due_s = float(payload.alpha_camera_due_s)
        self._bravo_next_due_s = float(payload.bravo_camera_due_s)
        self._air_next_due_s = float(payload.air_sensor_due_s)
        self._ground_next_due_s = float(payload.ground_sensor_due_s)
        self._alpha_message_visible_from_s = 0.0
        self._bravo_message_visible_from_s = 0.0
        self._air_message_visible_from_s = 0.0
        self._ground_message_visible_from_s = 0.0

        self._parcel_values = ["", "", ""]
        self._active_parcel_field = 0
        self._comms_input = ""
        self._comms_cycle_duration_s = max(1, int(payload.comms_time_limit_s))
        self._comms_cycle_index = 0
        self._comms_seed_base = self._build_comms_seed_base()
        self._comms_code_cache: dict[int, str] = {0: str(payload.comms_code)}

        self._objective_cycle_duration_s = max(1, int(payload.objective_deadline_s))
        self._objective_cycle_started_at_s = 0.0
        self._objective_deadline_s = float(self._objective_cycle_duration_s)
        self._objective_successes = 0
        self._objective_last_drop_late = False
        self._objective_last_drop_at_s: float | None = None

        self._eval_ticks = 0
        self._controls_bad_ticks = 0
        self._navigation_bad_ticks = 0
        self._engine_bad_ticks = 0
        self._warning_penalty_points = 0
        self._controls_input_hits = 0
        self._navigation_input_hits = 0
        self._engine_input_hits = 0

        self._events: list[CognitiveUpdatingActionEvent] = []

    def _build_comms_seed_base(self) -> int:
        seed = 17
        seed_text = (
            f"{self._payload.scenario_code}|{self._payload.clock_hms}|"
            f"{self._payload.parcel_target}|{self._payload.required_knots}|"
            f"{self._payload.pressure_low}:{self._payload.pressure_high}"
        )
        for ch in seed_text:
            seed = ((seed * 131) + ord(ch)) % 9_000
        return seed

    def _comms_code_for_cycle(self, cycle_index: int) -> str:
        idx = max(0, int(cycle_index))
        cached = self._comms_code_cache.get(idx)
        if cached is not None:
            return cached
        previous = self._comms_code_for_cycle(idx - 1)
        code_value = ((self._comms_seed_base + (idx * 1379)) % 9_000) + 1_000
        code = f"{code_value:04d}"
        if code == previous:
            code = f"{(((code_value + 173) % 9_000) + 1_000):04d}"
        self._comms_code_cache[idx] = code
        return code

    def _advance_comms_cycle(self, at_s: float) -> None:
        cycle_index = max(0, int(float(at_s) // float(self._comms_cycle_duration_s)))
        if cycle_index == self._comms_cycle_index:
            return
        self._comms_cycle_index = cycle_index
        self._comms_input = ""

    def _current_comms_code(self) -> str:
        return self._comms_code_for_cycle(self._comms_cycle_index)

    def _next_comms_code(self) -> str:
        return self._comms_code_for_cycle(self._comms_cycle_index + 1)

    def _comms_cycle_end_s(self) -> float:
        return float((self._comms_cycle_index + 1) * self._comms_cycle_duration_s)

    def _comms_remaining_s(self, now_s: float) -> float:
        return max(0.0, self._comms_cycle_end_s() - float(now_s))

    def _comms_swap_in_s(self, now_s: float) -> int:
        remaining = self._comms_remaining_s(now_s)
        whole = int(math.floor(remaining))
        if (remaining - float(whole)) <= 1e-9:
            return max(0, whole)
        return max(0, whole + 1)

    def _comms_reveal_threshold_s(self) -> float:
        return min(
            float(self._comms_cycle_duration_s),
            max(0.0, float(self._payload.message_reveal_comms_s)),
        )

    def _now_elapsed_s(self) -> float:
        return max(0.0, float(self._clock.now()) - self._started_at_s)

    def _record(self, action: str, value: str) -> None:
        self._events.append(
            CognitiveUpdatingActionEvent(at_s=self._now_elapsed_s(), action=action, value=value)
        )

    def _domain_active(self, domain: str) -> bool:
        return canonical_cognitive_updating_domain(domain) in self._active_domains

    def _grace_window_s(self) -> float:
        return max(0.0, float(self._payload.response_grace_window_s))

    def _controls_in_range(self) -> bool:
        if not self._domain_active("controls"):
            return True
        return (
            int(self._payload.pressure_low)
            <= int(round(self._pressure_value))
            <= int(self._payload.pressure_high)
        )

    def _navigation_in_range(self) -> bool:
        if not self._domain_active("navigation"):
            return True
        tolerance = max(1, int(self._payload.speed_tolerance_knots))
        return abs(int(round(self._current_knots)) - int(self._payload.required_knots)) <= tolerance

    def _tank_spread(self) -> int:
        values = [int(round(v)) for v in self._tank_levels]
        return max(values) - min(values)

    def _engine_in_range(self) -> bool:
        if not self._domain_active("engine"):
            return True
        return self._tank_spread() < max(1, int(self._payload.tank_spread_tolerance_l))

    def _camera_overdue(self, at_s: float) -> bool:
        if not self._domain_active("sensors"):
            return False
        grace = self._grace_window_s()
        return (at_s > float(self._alpha_next_due_s + grace)) or (
            at_s > float(self._bravo_next_due_s + grace)
        )

    def _sensor_overdue(self, at_s: float) -> bool:
        if not self._domain_active("sensors"):
            return False
        grace = self._grace_window_s()
        return (at_s > float(self._air_next_due_s + grace)) or (
            at_s > float(self._ground_next_due_s + grace)
        )

    def _task_message_reveal_lead_s(self, interval_s: int) -> float:
        return _clamp_float(float(interval_s) * 0.45, 6.0, 16.0)

    def _task_visible(self, *, now_s: float, due_s: float, visible_from_s: float) -> bool:
        return now_s >= float(visible_from_s) and now_s <= float(due_s + self._grace_window_s())

    def _advance_sensor_cycles(self, at_s: float) -> None:
        task_specs = (
            ("_alpha_next_due_s", self._alpha_interval_s, "_alpha_message_visible_from_s"),
            ("_bravo_next_due_s", self._bravo_interval_s, "_bravo_message_visible_from_s"),
            ("_air_next_due_s", self._air_interval_s, "_air_message_visible_from_s"),
            ("_ground_next_due_s", self._ground_interval_s, "_ground_message_visible_from_s"),
        )
        for due_attr, interval_s, visible_attr in task_specs:
            due_s = float(getattr(self, due_attr))
            grace = self._grace_window_s()
            if at_s <= float(due_s + grace):
                continue
            while at_s > float(due_s + grace):
                due_s += float(interval_s)
            setattr(self, due_attr, due_s)
            setattr(
                self,
                visible_attr,
                max(0.0, due_s - self._task_message_reveal_lead_s(interval_s)),
            )

    def _advance_objective_cycle(self, at_s: float) -> None:
        if not self._domain_active("objectives"):
            return
        if at_s < float(self._objective_deadline_s):
            return
        while at_s >= float(self._objective_deadline_s):
            self._objective_cycle_started_at_s = float(self._objective_deadline_s)
            self._objective_deadline_s = (
                self._objective_cycle_started_at_s + float(self._objective_cycle_duration_s)
            )
        self._parcel_values = ["", "", ""]
        self._active_parcel_field = 0

    def _objective_mistyped(self) -> bool:
        if not self._domain_active("objectives"):
            return False
        for typed, token in zip(self._parcel_values, self._parcel_target_tokens(), strict=True):
            if typed == "":
                continue
            if len(typed) > len(token):
                return True
            if not token.startswith(typed):
                return True
        return False

    def _parcel_exact(self) -> bool:
        for typed, token in zip(self._parcel_values, self._parcel_target_tokens(), strict=True):
            if typed != token:
                return False
        return True

    def _parcel_target_tokens(self) -> tuple[str, str, str]:
        lat, lon, time_code = self._payload.parcel_target
        return (
            f"{int(lat):06d}",
            f"{int(lon):06d}",
            f"{int(time_code):06d}",
        )

    def _objective_drop_ready(self) -> bool:
        if not self._domain_active("objectives"):
            return False
        return (not self._objective_mistyped()) and self._parcel_exact()

    def _collect_warnings(self, at_s: float) -> tuple[str, ...]:
        warnings: list[str] = []
        if self._domain_active("controls") and not self._controls_in_range():
            warnings.append("Check Pressure")
        if self._domain_active("navigation") and not self._navigation_in_range():
            warnings.append("Air Speed Warning")
        if self._domain_active("engine") and not self._engine_in_range():
            warnings.append("Engine Panel")
        if self._domain_active("sensors") and (self._camera_overdue(at_s) or self._sensor_overdue(at_s)):
            warnings.append("Sensor Panel")
        if self._domain_active("objectives") and (
            self._objective_mistyped() or (at_s > float(self._objective_deadline_s))
        ):
            warnings.append("Objective Warning")
        return tuple(warnings)

    def _evaluate_interval(self, at_s: float) -> None:
        self._eval_ticks += 1
        if self._domain_active("controls") and not self._controls_in_range():
            self._controls_bad_ticks += 1
        if self._domain_active("navigation") and not self._navigation_in_range():
            self._navigation_bad_ticks += 1
        if self._domain_active("engine") and not self._engine_in_range():
            self._engine_bad_ticks += 1
        penalty = float(len(self._collect_warnings(at_s))) * float(self._payload.warning_penalty_scale)
        self._warning_penalty_points += int(round(penalty))

    def _advance_domain(self, domain: str, at_s: float | None = None) -> float:
        now_s = self._now_elapsed_s() if at_s is None else max(0.0, float(at_s))
        token = canonical_cognitive_updating_domain(domain)
        if token not in self._last_domain_advance_s:
            return now_s
        dt = now_s - float(self._last_domain_advance_s[token])
        if dt <= 0.0:
            return now_s

        if token == "controls" and self._domain_active("controls"):
            pressure_scale = max(0.0, float(self._payload.pressure_drift_scale))
            if self._pump_on:
                self._pressure_value += (PUMP_RISE_PER_S * pressure_scale) * dt
            else:
                self._pressure_value -= (PUMP_FALL_PER_S * pressure_scale) * dt
            self._pressure_value = _clamp_float(self._pressure_value, 65.0, 138.0)

        if token == "navigation" and self._domain_active("navigation"):
            speed_scale = max(0.0, float(self._payload.speed_drift_scale))
            if self._current_knots >= float(self._payload.required_knots):
                self._current_knots += (KNOTS_DRIFT_PER_S * speed_scale) * dt
            else:
                self._current_knots -= (KNOTS_DRIFT_PER_S * speed_scale) * dt
            self._current_knots = _clamp_float(self._current_knots, 40.0, 220.0)

        if token == "engine" and self._domain_active("engine"):
            drain_scale = max(0.0, float(self._payload.tank_drain_scale))
            for idx in range(3):
                drain = (
                    ACTIVE_TANK_DRAIN_PER_S if (idx + 1) == self._active_tank else IDLE_TANK_DRAIN_PER_S
                ) * drain_scale
                self._tank_levels[idx] = max(260.0, self._tank_levels[idx] - (drain * dt))

        if token == "sensors" and self._domain_active("sensors"):
            self._advance_sensor_cycles(now_s)

        if token == "objectives" and self._domain_active("objectives"):
            self._advance_objective_cycle(now_s)

        if token == "state_code":
            self._advance_comms_cycle(now_s)

        self._last_domain_advance_s[token] = now_s
        return now_s

    def _advance_all(self) -> float:
        now_s = self._now_elapsed_s()
        for domain in COGNITIVE_UPDATING_DOMAIN_ORDER:
            self._advance_domain(domain, now_s)

        while self._next_eval_s <= now_s + 1e-9:
            self._evaluate_interval(self._next_eval_s)
            self._next_eval_s += EVAL_INTERVAL_S

        return now_s

    def toggle_camera(self, camera: str) -> None:
        if not self._domain_active("sensors"):
            return
        now_s = self._advance_domain("sensors")
        token = camera.strip().lower()
        if token == "alpha":
            if abs(float(now_s) - float(self._alpha_next_due_s)) <= self._grace_window_s():
                self._alpha_hits += 1
            self._alpha_last_press_s = now_s
            self._alpha_next_due_s = now_s + float(self._alpha_interval_s)
            self._alpha_message_visible_from_s = max(
                0.0,
                self._alpha_next_due_s - self._task_message_reveal_lead_s(self._alpha_interval_s),
            )
            if self._alpha_armed_at_s is None:
                self._alpha_armed_at_s = now_s
            self._record("camera_alpha", "1")
            return
        if token == "bravo":
            if abs(float(now_s) - float(self._bravo_next_due_s)) <= self._grace_window_s():
                self._bravo_hits += 1
            self._bravo_last_press_s = now_s
            self._bravo_next_due_s = now_s + float(self._bravo_interval_s)
            self._bravo_message_visible_from_s = max(
                0.0,
                self._bravo_next_due_s - self._task_message_reveal_lead_s(self._bravo_interval_s),
            )
            if self._bravo_armed_at_s is None:
                self._bravo_armed_at_s = now_s
            self._record("camera_bravo", "1")

    def toggle_sensor(self, sensor: str) -> None:
        if not self._domain_active("sensors"):
            return
        now_s = self._advance_domain("sensors")
        token = sensor.strip().lower()
        if token == "air":
            if abs(float(now_s) - float(self._air_next_due_s)) <= self._grace_window_s():
                self._air_hits += 1
            self._air_last_press_s = now_s
            self._air_next_due_s = now_s + float(self._air_interval_s)
            self._air_message_visible_from_s = max(
                0.0,
                self._air_next_due_s - self._task_message_reveal_lead_s(self._air_interval_s),
            )
            if self._air_sensor_armed_at_s is None:
                self._air_sensor_armed_at_s = now_s
            self._record("sensor_air", "1")
            return
        if token == "ground":
            if abs(float(now_s) - float(self._ground_next_due_s)) <= self._grace_window_s():
                self._ground_hits += 1
            self._ground_last_press_s = now_s
            self._ground_next_due_s = now_s + float(self._ground_interval_s)
            self._ground_message_visible_from_s = max(
                0.0,
                self._ground_next_due_s - self._task_message_reveal_lead_s(self._ground_interval_s),
            )
            if self._ground_sensor_armed_at_s is None:
                self._ground_sensor_armed_at_s = now_s
            self._record("sensor_ground", "1")

    def set_pump(self, on: bool) -> None:
        if not self._domain_active("controls"):
            return
        self._advance_domain("controls")
        self._pump_on = bool(on)
        if self._controls_in_range():
            self._controls_input_hits += 1
        self._record("pump", "1" if self._pump_on else "0")

    def adjust_knots(self, delta: int) -> None:
        if not self._domain_active("navigation"):
            return
        self._advance_domain("navigation")
        change = int(delta)
        if change == 0:
            return
        self._current_knots = _clamp_float(self._current_knots + float(change), 40.0, 220.0)
        if self._navigation_in_range():
            self._navigation_input_hits += 1
        self._record("knots", str(int(round(self._current_knots))))

    def set_active_tank(self, tank: int) -> None:
        if not self._domain_active("engine"):
            return
        self._advance_domain("engine")
        idx = _clamp_int(tank, 1, 3)
        self._active_tank = idx
        if self._engine_in_range():
            self._engine_input_hits += 1
        self._record("tank", str(idx))

    def activate_dispenser(self) -> None:
        if not self._domain_active("objectives"):
            return
        now_s = self._advance_domain("objectives")
        ready = self._objective_drop_ready()
        if ready:
            self._objective_successes += 1
            self._objective_last_drop_late = now_s > float(self._objective_deadline_s)
            self._objective_last_drop_at_s = now_s
            self._parcel_values = ["", "", ""]
            self._active_parcel_field = 0
            self._objective_cycle_started_at_s = now_s
            self._objective_deadline_s = now_s + float(self._objective_cycle_duration_s)
        self._record("objective_drop", "1" if ready else "0")

    def set_parcel_field(self, index: int) -> None:
        if not self._domain_active("objectives"):
            return
        self._advance_domain("objectives")
        idx = _clamp_int(index, 0, 2)
        self._active_parcel_field = idx
        self._record("parcel_field", str(idx))

    def append_parcel_digit(self, digit: str) -> None:
        if not self._domain_active("objectives"):
            return
        self._advance_domain("objectives")
        if len(digit) != 1 or not digit.isdigit():
            return
        idx = _clamp_int(self._active_parcel_field, 0, 2)
        target_len = len(self._parcel_target_tokens()[idx])
        if len(self._parcel_values[idx]) >= target_len:
            return
        self._parcel_values[idx] += digit
        self._record("parcel_digit", digit)
        if len(self._parcel_values[idx]) >= target_len and idx < 2:
            self._active_parcel_field = idx + 1

    def append_comms_digit(self, digit: str) -> None:
        self._advance_domain("state_code")
        if len(digit) != 1 or not digit.isdigit():
            return
        if len(self._comms_input) >= 4:
            return
        self._comms_input += digit
        self._record("comms_digit", digit)

    def clear_comms(self) -> None:
        self._advance_domain("state_code")
        self._comms_input = ""
        self._record("comms_clear", "")

    def _camera_score(self) -> int:
        return _clamp_int((self._alpha_hits + self._bravo_hits) * 25, 0, 50)

    def _sensor_score(self) -> int:
        return _clamp_int((self._air_hits + self._ground_hits) * 25, 0, 50)

    def _objectives_score(self) -> int:
        if not self._domain_active("objectives"):
            return 100
        if self._objective_mistyped():
            return 0
        if self._objective_successes > 0:
            return 40 if self._objective_last_drop_late else 100
        if self._objective_drop_ready():
            return 80
        filled = sum(1 for v in self._parcel_values if v != "")
        return _clamp_int(filled * 20, 0, 60)

    def _objective_lights(self) -> int:
        if not self._domain_active("objectives"):
            return 1
        completed_fields = sum(
            1
            for typed, token in zip(self._parcel_values, self._parcel_target_tokens(), strict=True)
            if typed == token
        )
        lights = 1 + completed_fields
        if self._objective_drop_ready():
            return 5
        return _clamp_int(lights, 1, 4)

    def _camera_digit(self) -> int:
        if not self._domain_active("sensors"):
            return 2
        if self._camera_score() >= 50:
            return 1
        if (self._alpha_hits + self._bravo_hits) > 0:
            return 2
        return 3

    def _sensor_digit(self) -> int:
        if not self._domain_active("sensors"):
            return 2
        if self._sensor_score() >= 50:
            return 1
        if (self._air_hits + self._ground_hits) > 0:
            return 2
        return 3

    def _pressure_digit(self) -> int:
        if not self._domain_active("controls"):
            return 2
        pressure = int(round(self._pressure_value))
        if pressure < int(self._payload.pressure_low):
            return 1
        if pressure > int(self._payload.pressure_high):
            return 3
        return 2

    def _speed_digit(self) -> int:
        if not self._domain_active("navigation"):
            return 2
        speed = int(round(self._current_knots))
        tolerance = max(1, int(self._payload.speed_tolerance_knots))
        low = int(self._payload.required_knots) - tolerance
        high = int(self._payload.required_knots) + tolerance
        if speed < low:
            return 1
        if speed > high:
            return 3
        return 2

    def _state_code_from_current_values(self) -> str:
        return (
            f"{self._camera_digit()}"
            f"{self._pressure_digit()}"
            f"{self._speed_digit()}"
            f"{self._sensor_digit()}"
        )

    def state_code(self) -> str:
        self._advance_all()
        return self._state_code_from_current_values()

    def _score_components(self) -> tuple[int, int, int, int, int, int]:
        controls_score = (
            _domain_score(
                bad_ticks=self._controls_bad_ticks,
                eval_ticks=self._eval_ticks,
                manual_hits=self._controls_input_hits,
            )
            if self._domain_active("controls")
            else 100
        )
        navigation_score = (
            _domain_score(
                bad_ticks=self._navigation_bad_ticks,
                eval_ticks=self._eval_ticks,
                manual_hits=self._navigation_input_hits,
            )
            if self._domain_active("navigation")
            else 100
        )
        engine_score = (
            _domain_score(
                bad_ticks=self._engine_bad_ticks,
                eval_ticks=self._eval_ticks,
                manual_hits=self._engine_input_hits,
            )
            if self._domain_active("engine")
            else 100
        )
        sensors_score = (
            _clamp_int(self._camera_score() + self._sensor_score(), 0, 100)
            if self._domain_active("sensors")
            else 100
        )
        objectives_score = self._objectives_score()
        overall_score = _clamp_int(
            int(
                round(
                    (
                        controls_score
                        + navigation_score
                        + engine_score
                        + sensors_score
                        + objectives_score
                    )
                    / 5
                )
            )
            - int(self._warning_penalty_points),
            0,
            100,
        )
        return (
            controls_score,
            navigation_score,
            engine_score,
            sensors_score,
            objectives_score,
            overall_score,
        )

    def build_submission_raw(self) -> str:
        self._advance_domain("state_code")
        if len(self._comms_input) != 4:
            return ""
        (
            controls_score,
            navigation_score,
            engine_score,
            sensors_score,
            objectives_score,
            overall_score,
        ) = self._score_components()
        return (
            f"{self._comms_input}"
            f"{self._current_comms_code()}"
            f"{controls_score:03d}"
            f"{navigation_score:03d}"
            f"{engine_score:03d}"
            f"{sensors_score:03d}"
            f"{objectives_score:03d}"
            f"{int(self._warning_penalty_points):03d}"
            f"{overall_score:03d}"
            f"{min(999, len(self._events)):03d}"
        )

    def snapshot(self) -> CognitiveUpdatingRuntimeSnapshot:
        now_s = float(self._advance_all())
        elapsed = int(now_s)
        (
            controls_score,
            navigation_score,
            engine_score,
            sensors_score,
            objectives_score,
            overall_score,
        ) = self._score_components()

        warning_lines = self._collect_warnings(float(elapsed))
        air_left = max(0, int(round(self._air_next_due_s - now_s)))
        ground_left = max(0, int(round(self._ground_next_due_s - now_s)))
        comms_remaining_s = self._comms_remaining_s(now_s)
        comms_swap_in_s = self._comms_swap_in_s(now_s)
        current_comms_code = self._current_comms_code()
        next_comms_code = (
            self._next_comms_code()
            if comms_remaining_s <= self._comms_reveal_threshold_s()
            else None
        )
        target_lat, target_lon, target_time = self._parcel_target_tokens()
        objective_elapsed_s = max(0.0, now_s - float(self._objective_cycle_started_at_s))
        latitude_line = (
            f"Latitude: {target_lat}"
            if self._domain_active("objectives")
            and objective_elapsed_s >= float(self._payload.message_reveal_lat_s)
            else ""
        )
        longitude_line = (
            f"Longitude: {target_lon}"
            if self._domain_active("objectives")
            and objective_elapsed_s >= float(self._payload.message_reveal_lon_s)
            else ""
        )
        time_line = (
            f"Time: {target_time}"
            if self._domain_active("objectives")
            and objective_elapsed_s >= float(self._payload.message_reveal_time_s)
            else ""
        )
        comms_line = (
            f"New Comms Code: {next_comms_code} in {comms_swap_in_s:02d}s"
            if next_comms_code is not None
            else ""
        )
        sensor_domain_lines: list[str] = []
        if self._domain_active("sensors"):
            task_lines: list[tuple[int, float, int, str]] = []
            task_specs = (
                (
                    "alpha",
                    self._alpha_next_due_s,
                    self._alpha_message_visible_from_s,
                    0,
                    lambda: f"Activate Alpha Camera at: {_fmt_hms(self._clock_base_s + int(round(self._alpha_next_due_s)))}",
                ),
                (
                    "bravo",
                    self._bravo_next_due_s,
                    self._bravo_message_visible_from_s,
                    1,
                    lambda: f"Activate Bravo Camera at: {_fmt_hms(self._clock_base_s + int(round(self._bravo_next_due_s)))}",
                ),
                (
                    "air",
                    self._air_next_due_s,
                    self._air_message_visible_from_s,
                    2,
                    lambda: f"Air Sensor due in: {max(0, int(round(self._air_next_due_s - now_s))):02d}s",
                ),
                (
                    "ground",
                    self._ground_next_due_s,
                    self._ground_message_visible_from_s,
                    3,
                    lambda: f"Ground Sensor due in: {max(0, int(round(self._ground_next_due_s - now_s))):02d}s",
                ),
            )
            for _name, due_s, visible_from_s, order, builder in task_specs:
                if not self._task_visible(now_s=now_s, due_s=due_s, visible_from_s=visible_from_s):
                    continue
                overdue_rank = 0 if now_s > float(due_s) else 1
                task_lines.append((overdue_rank, float(due_s), order, builder()))
            task_lines.sort(key=lambda item: (item[0], item[1], item[2]))
            sensor_domain_lines = [line for *_prefix, line in task_lines[:2]]
        message_lines = (
            latitude_line,
            longitude_line,
            time_line,
            comms_line,
            sensor_domain_lines[0] if len(sensor_domain_lines) > 0 else "",
            sensor_domain_lines[1] if len(sensor_domain_lines) > 1 else "",
        )
        deadline_left = max(0, int(round(self._objective_deadline_s - now_s)))

        alpha_flash = (
            self._alpha_last_press_s is not None
            and (now_s - float(self._alpha_last_press_s)) <= BUTTON_FLASH_S
        )
        bravo_flash = (
            self._bravo_last_press_s is not None
            and (now_s - float(self._bravo_last_press_s)) <= BUTTON_FLASH_S
        )
        air_flash = (
            self._air_last_press_s is not None
            and (now_s - float(self._air_last_press_s)) <= BUTTON_FLASH_S
        )
        ground_flash = (
            self._ground_last_press_s is not None
            and (now_s - float(self._ground_last_press_s)) <= BUTTON_FLASH_S
        )

        drop_flash = (
            self._objective_last_drop_at_s is not None
            and (now_s - float(self._objective_last_drop_at_s)) <= BUTTON_FLASH_S
        )

        return CognitiveUpdatingRuntimeSnapshot(
            elapsed_s=elapsed,
            clock_hms=_fmt_hms(self._clock_base_s + elapsed),
            current_knots=int(round(self._current_knots)),
            pump_on=self._pump_on,
            active_tank=self._active_tank,
            alpha_armed=alpha_flash,
            bravo_armed=bravo_flash,
            air_sensor_armed=air_flash,
            ground_sensor_armed=ground_flash,
            parcel_values=(self._parcel_values[0], self._parcel_values[1], self._parcel_values[2]),
            active_parcel_field=self._active_parcel_field,
            comms_input=self._comms_input,
            dispenser_lit=self._objective_lights(),
            air_time_left_s=air_left,
            ground_time_left_s=ground_left,
            comms_time_left_s=comms_swap_in_s,
            comms_swap_in_s=comms_swap_in_s,
            current_comms_code=current_comms_code,
            next_comms_code=next_comms_code,
            objective_deadline_left_s=deadline_left,
            state_code=self._state_code_from_current_values(),
            operation_score_hint=float(overall_score) / 100.0,
            event_count=len(self._events),
            pressure_value=int(round(self._pressure_value)),
            tank_levels_l=(
                int(round(self._tank_levels[0])),
                int(round(self._tank_levels[1])),
                int(round(self._tank_levels[2])),
            ),
            warning_lines=warning_lines,
            message_lines=message_lines,
            controls_score=controls_score,
            navigation_score=navigation_score,
            engine_score=engine_score,
            sensors_score=sensors_score,
            objectives_score=objectives_score,
            warnings_penalty_points=int(self._warning_penalty_points),
            overall_score=overall_score,
            objective_drop_ready=self._objective_drop_ready(),
            objective_drop_complete=drop_flash,
        )

    def events(self) -> tuple[CognitiveUpdatingActionEvent, ...]:
        return tuple(self._events)


class CognitiveUpdatingScorer(AnswerScorer):
    """Score code entry plus per-domain performance and warning penalties."""

    def score(self, *, problem: Problem, user_answer: int, raw: str) -> float:
        payload = problem.payload
        if not isinstance(payload, CognitiveUpdatingPayload):
            return 1.0 if int(user_answer) == int(problem.answer) else 0.0

        decoded = decode_cognitive_updating_submission_raw(raw)
        if decoded is not None:
            code_score = _distance_score(
                user_answer=int(decoded.entered_code),
                target=int(decoded.state_code),
                tolerance=max(1, int(payload.estimate_tolerance)),
            )
            sub_avg = (
                float(decoded.controls_score)
                + float(decoded.navigation_score)
                + float(decoded.engine_score)
                + float(decoded.sensors_score)
                + float(decoded.objectives_score)
            ) / 500.0
            overall_component = float(decoded.overall_score) / 100.0
            warning_factor = min(0.80, float(decoded.warnings_penalty_points) / 100.0)
            combined = (
                (0.45 * code_score)
                + (0.35 * overall_component)
                + (0.20 * sub_avg)
                - (0.20 * warning_factor)
            )
            return _clamp_float(combined, 0.0, 1.0)

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
        return self.next_problem_for_selection(
            difficulty=difficulty,
            scenario_family="baseline",
        )

    @classmethod
    def supported_scenario_families(cls) -> tuple[str, ...]:
        return supported_cognitive_updating_scenario_families()

    def next_problem_for_selection(
        self,
        *,
        difficulty: float,
        training_profile: CognitiveUpdatingTrainingProfile | None = None,
        scenario_family: str | None = None,
    ) -> Problem:
        difficulty = clamp01(difficulty)
        difficulty_profile = _difficulty_params(difficulty)
        use_default_difficulty_domains = training_profile is None
        profile = training_profile or CognitiveUpdatingTrainingProfile()
        active_domains = (
            difficulty_profile.active_domains
            if use_default_difficulty_domains
            else normalize_cognitive_updating_active_domains(*profile.active_domains)
        )
        family_token = canonical_cognitive_updating_scenario_family(
            scenario_family or profile.scenario_family or "baseline"
        )
        family_cfg = _COGNITIVE_UPDATING_SCENARIO_CONFIGS[family_token]
        scenario_code = self._next_scenario_code()

        clock_h_low, clock_h_high = (6, 9)
        if family_token == "compressed":
            clock_h_low, clock_h_high = (7, 10)
        elif family_token == "staggered":
            clock_h_low, clock_h_high = (5, 8)
        clock_h = self._rng.randint(clock_h_low, clock_h_high)
        clock_m = self._rng.randint(0, 59)
        clock_s = self._rng.randint(0, 59)
        now_total_s = (clock_h * 3600) + (clock_m * 60) + clock_s

        camera_scale = max(0.35, family_cfg.camera_due_scale * float(profile.camera_due_scale))
        sensor_scale = max(0.35, family_cfg.sensor_due_scale * float(profile.sensor_due_scale))
        objective_scale = max(
            0.35, family_cfg.objective_deadline_scale * float(profile.objective_deadline_scale)
        )
        comms_scale = max(0.35, family_cfg.comms_time_limit_scale * float(profile.comms_time_limit_scale))
        reveal_scale = max(0.35, family_cfg.message_reveal_scale * float(profile.message_reveal_scale))
        pressure_drift_scale = max(
            0.0,
            family_cfg.pressure_drift_scale
            * float(profile.pressure_drift_scale)
            * float(difficulty_profile.drift_multiplier),
        )
        speed_drift_scale = max(
            0.0,
            family_cfg.speed_drift_scale
            * float(profile.speed_drift_scale)
            * float(difficulty_profile.drift_multiplier),
        )
        tank_drain_scale = max(
            0.0,
            family_cfg.tank_drain_scale
            * float(profile.tank_drain_scale)
            * float(difficulty_profile.drift_multiplier),
        )

        camera_due_min = max(8, int(round(difficulty_profile.camera_due_min_s * camera_scale)))
        camera_due_max = max(
            camera_due_min,
            int(round(difficulty_profile.camera_due_max_s * camera_scale)),
        )
        alpha_camera_due_s = self._rng.randint(camera_due_min, camera_due_max)
        bravo_camera_due_s = self._rng.randint(camera_due_min, camera_due_max)
        earliest_camera = "ALPHA" if alpha_camera_due_s <= bravo_camera_due_s else "BRAVO"
        camera_digit = 1 if earliest_camera == "ALPHA" else 2

        sensor_due_min = max(8, int(round(difficulty_profile.sensor_due_min_s * sensor_scale)))
        sensor_due_max = max(
            sensor_due_min,
            int(round(difficulty_profile.sensor_due_max_s * sensor_scale)),
        )
        air_sensor_due_s = self._rng.randint(sensor_due_min, sensor_due_max)
        ground_sensor_due_s = self._rng.randint(sensor_due_min, sensor_due_max)
        earliest_sensor = "AIR" if air_sensor_due_s <= ground_sensor_due_s else "GROUND"
        sensor_digit = 1 if earliest_sensor == "AIR" else 2

        pressure_low = self._rng.randint(84, 94)
        pressure_high = pressure_low + int(difficulty_profile.pressure_band_width)
        pressure_value = self._rng.randint(pressure_low, pressure_high)
        pressure_digit = 2
        pressure_warning = "Pressure Nominal"
        pump_on = bool(self._rng.randint(0, 1))

        knots_low, knots_high = (90, 160)
        if family_token == "compressed":
            knots_low, knots_high = (108, 176)
        elif family_token == "staggered":
            knots_low, knots_high = (80, 150)
        required_knots = self._rng.randint(knots_low, knots_high)
        current_knots = required_knots + self._rng.randint(-4, 4)
        speed_digit = 2
        speed_warning = "AIRSPEED NOMINAL"
        current_knots = max(40, current_knots)

        tank_center_low, tank_center_high = (390, 450)
        if family_token == "compressed":
            tank_center_low, tank_center_high = (372, 430)
        elif family_token == "staggered":
            tank_center_low, tank_center_high = (405, 468)
        tank_center = self._rng.randint(tank_center_low, tank_center_high)
        tank_levels = tuple(
            _clamp_int(tank_center + self._rng.randint(-12, 12), 320, 520) for _ in range(3)
        )
        active_tank = self._rng.randint(1, 3)

        warning_lines: list[str] = []
        if "controls" in active_domains:
            warning_lines.append(pressure_warning)
        if "navigation" in active_domains:
            warning_lines.append(speed_warning)
        if "sensors" in active_domains and min(air_sensor_due_s, ground_sensor_due_s) <= 20:
            warning_lines.append("SENSOR PANEL")

        parcel_lat = self._rng.randint(100000, 999999)
        parcel_lon = self._rng.randint(100000, 999999)
        parcel_time = int(self._fmt_hms(self._rng.randint(0, (24 * 3600) - 1)).replace(":", ""))
        objective_deadline_s = _clamp_int(
            round(
                self._rng.randint(
                    difficulty_profile.objective_deadline_min_s,
                    difficulty_profile.objective_deadline_max_s,
                )
                * objective_scale
            ),
            12,
            140,
        )
        dispenser_lit = 0

        answer_digits = (
            camera_digit if "sensors" in active_domains else 2,
            pressure_digit if "controls" in active_domains else 2,
            speed_digit if "navigation" in active_domains else 2,
            sensor_digit if "sensors" in active_domains else 2,
        )
        answer_text = "".join(str(d) for d in answer_digits)
        answer = int(answer_text)
        tolerance = int(difficulty_profile.estimate_tolerance)

        reveal_multiplier = reveal_scale * float(difficulty_profile.message_reveal_scale)
        message_reveal_lat_s = max(1.0, MESSAGE_REVEAL_LAT_S * reveal_multiplier)
        message_reveal_lon_s = max(1.0, MESSAGE_REVEAL_LON_S * reveal_multiplier)
        message_reveal_time_s = max(1.0, MESSAGE_REVEAL_TIME_S * reveal_multiplier)
        message_reveal_comms_s = max(1.0, MESSAGE_REVEAL_COMMS_S * reveal_multiplier)
        message_lines = (
            f"Activate Alpha Camera at: {self._fmt_hms(now_total_s + alpha_camera_due_s)}"
            if "sensors" in active_domains
            else "",
            f"Activate Bravo Camera at: {self._fmt_hms(now_total_s + bravo_camera_due_s)}"
            if "sensors" in active_domains
            else "",
            f"Air Sensor due in: {air_sensor_due_s:02d}s" if "sensors" in active_domains else "",
            f"Ground Sensor due in: {ground_sensor_due_s:02d}s" if "sensors" in active_domains else "",
        )

        focus_label = str(profile.focus_label).strip() or "Full Mixed"
        starting_upper_tab_index = (
            family_cfg.starting_upper_tab_index
            if profile.starting_upper_tab_index is None
            else _clamp_int(profile.starting_upper_tab_index, 0, 5)
        )
        starting_lower_tab_index = (
            family_cfg.starting_lower_tab_index
            if profile.starting_lower_tab_index is None
            else _clamp_int(profile.starting_lower_tab_index, 0, 5)
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
            comms_time_limit_s=max(
                12,
                int(round(float(difficulty_profile.comms_time_limit_s) * comms_scale)),
            ),
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
            objective_deadline_s=objective_deadline_s,
            dispenser_lit=dispenser_lit,
            question="Enter the active 4-digit comms code from the live communications panel.",
            answer_unit="code",
            correct_value=answer,
            estimate_tolerance=tolerance,
            response_grace_window_s=int(difficulty_profile.response_grace_window_s),
            speed_tolerance_knots=int(difficulty_profile.speed_tolerance_knots),
            tank_spread_tolerance_l=int(difficulty_profile.tank_spread_tolerance_l),
            active_domains=active_domains,
            scenario_family=family_token,
            focus_label=focus_label,
            starting_upper_tab_index=starting_upper_tab_index,
            starting_lower_tab_index=starting_lower_tab_index,
            pressure_drift_scale=pressure_drift_scale,
            speed_drift_scale=speed_drift_scale,
            tank_drain_scale=tank_drain_scale,
            warning_penalty_scale=max(0.0, float(profile.warning_penalty_scale)),
            message_reveal_lat_s=message_reveal_lat_s,
            message_reveal_lon_s=message_reveal_lon_s,
            message_reveal_comms_s=message_reveal_comms_s,
            message_reveal_time_s=message_reveal_time_s,
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
        return _fmt_hms(total_seconds)


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
        "Scoring model:",
        "- Domain subscores: Controls, Navigation, Engine, Sensors, Objectives",
        "- Warnings are sampled every 5 seconds and subtract points",
        "- Late sensor/objective timing earns no points and increases warning penalty",
        "",
        "Controls:",
        "- Q/E: switch upper display (Messages, Objectives, Controls)",
        "- A/D: switch lower display (Navigation, Sensors, Engine)",
        "- No backspace editing",
        "- Enter or Comms Submit sends your coded response",
        "",
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
