from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from .clock import Clock
from .cognitive_core import AttemptSummary, Phase, SeededRng, TestSnapshot, clamp01

AUDITORY_GATE_SPAWN_X_NORM = 1.65
AUDITORY_GATE_PLAYER_X_NORM = 0.0
AUDITORY_GATE_RETIRE_X_NORM = -1.25
AUDITORY_TRIANGLE_GATE_POINTS: tuple[tuple[float, float], ...] = (
    (0.0, 1.22),
    (-1.056, -0.61),
    (1.056, -0.61),
)


@dataclass(frozen=True, slots=True)
class AuditoryCapacityConfig:
    seed: int | None = None
    practice_duration_s: float = 60.0
    scored_duration_s: float = 13.0 * 60.0
    run_duration_seconds: float | None = None
    practice_enabled: bool = True
    tick_hz: float = 120.0

    control_gain: float = 1.24
    disturbance_gain: float = 0.42
    tube_half_width: float = 0.82
    tube_half_height: float = 0.60
    tunnel_curvature_intensity: float = 0.56

    gate_speed_norm_per_s: float = 0.30
    gate_spawn_rate: float = 0.24
    gate_interval_s: float = 4.2

    command_rate: float = 0.26
    distractor_rate: float = 0.34
    callsign_count: int = 3
    digit_sequence_max_len: int = 6
    recall_prompt_rate: float = 0.045
    response_window_seconds: float = 2.40

    # Optional fidelity knobs kept as config flags instead of hardcoded behavior.
    enable_overlapping_voices: bool = True
    enable_distortion: bool = False
    callsign_interval_s: float = 22.0
    beep_interval_s: float = 36.0
    color_command_interval_s: float = 72.0
    sequence_interval_s: float = 240.0
    sequence_display_s: float = 0.90
    sequence_response_s: float = 7.00

    drift_relax_start_ratio: float = 0.14
    min_drift_scale: float = 0.24


class AuditoryCapacityCommandType(StrEnum):
    CHANGE_COLOUR = "change_colour"
    CHANGE_NUMBER = "change_number"
    GATE_DIRECTIVE = "gate_directive"
    DIGIT_SEQUENCE = "digit_sequence"
    PRESS_TRIGGER = "press_trigger"
    SET_FORBIDDEN_GATE_COLOUR = "set_forbidden_gate_colour"
    SET_FORBIDDEN_GATE_SHAPE = "set_forbidden_gate_shape"
    DIGIT_APPEND = "digit_append"
    RECALL_DIGITS = "recall_digits"
    CHANGE_CALLSIGN = "change_callsign"
    NO_OP_DISTRACTOR = "no_op_distractor"


class AuditoryCapacityEventKind(StrEnum):
    COMMAND = "command"
    GATE = "gate"
    DIGIT_RECALL = "digit_recall"
    TRIGGER = "trigger"
    COLLISION = "collision"
    FALSE_RESPONSE = "false_response"


@dataclass(frozen=True, slots=True)
class AuditoryCapacityColorRule:
    color: str
    required_shape: str


@dataclass(frozen=True, slots=True)
class AuditoryCapacityGateDirective:
    action: str
    match_kind: str
    match_value: str


@dataclass(frozen=True, slots=True)
class AuditoryCapacityInstructionEvent:
    event_id: int
    timestamp_s: float
    addressed_call_sign: str
    speaker_id: str
    command_type: AuditoryCapacityCommandType
    payload: str | int | AuditoryCapacityGateDirective | None
    expires_at_s: float
    is_distractor: bool


@dataclass(frozen=True, slots=True)
class AuditoryCapacityGate:
    gate_id: int
    x_norm: float
    y_norm: float
    color: str
    shape: str
    aperture_norm: float


@dataclass(frozen=True, slots=True)
class AuditoryCapacityMetrics:
    boundary_violations: int
    gate_hits: int
    gate_misses: int
    forbidden_gate_hits: int
    correct_command_executions: int
    false_responses_to_distractors: int
    missed_valid_commands: int
    digit_recall_attempts: int
    digit_recall_accuracy: float


@dataclass(frozen=True, slots=True)
class AuditoryCapacityPayload:
    session_seed: int
    phase_elapsed_s: float
    active_channels: tuple[str, ...]
    segment_label: str
    segment_index: int
    segment_total: int
    segment_time_remaining_s: float

    ball_x: float
    ball_y: float
    ball_contact_ratio: float
    control_x: float
    control_y: float
    disturbance_x: float
    disturbance_y: float

    tube_half_width: float
    tube_half_height: float

    ball_color: str
    ball_color_strength: float
    ball_visual_color: str
    ball_visual_strength: float
    ball_number: int
    active_callsign: str

    assigned_callsigns: tuple[str, ...]
    active_instruction: AuditoryCapacityInstructionEvent | None
    instruction_text: str | None
    instruction_uid: int | None
    instruction_command_type: str | None
    briefing_active: bool

    callsign_cue: str | None
    callsign_blocks_gate: bool
    beep_active: bool
    color_command: str | None
    number_command: int | None

    sequence_display: str | None
    sequence_response_open: bool
    recall_target_length: int | None
    digit_buffer_length: int

    forbidden_gate_color: str | None
    forbidden_gate_shape: str | None
    target_gate_id: int | None
    target_gate_action: str | None
    color_rules: tuple[AuditoryCapacityColorRule, ...]
    gates: tuple[AuditoryCapacityGate, ...]
    metrics: AuditoryCapacityMetrics

    gate_hits: int
    gate_misses: int
    forbidden_gate_hits: int
    collisions: int
    false_alarms: int
    correct_command_executions: int
    false_responses_to_distractors: int
    missed_valid_commands: int
    digit_recall_attempts: int
    digit_recall_accuracy: float
    points: float

    background_noise_level: float
    distortion_level: float
    background_noise_source: str | None

    command_time_left_s: float | None
    sequence_show_time_left_s: float | None
    sequence_response_time_left_s: float | None

    next_command_in_s: float | None
    next_gate_in_s: float | None
    next_callsign_in_s: float | None
    next_beep_in_s: float | None
    next_color_in_s: float | None
    next_sequence_in_s: float | None


@dataclass(frozen=True, slots=True)
class AuditoryCapacityEvent:
    phase: Phase
    kind: AuditoryCapacityEventKind
    expected: str
    response: str
    is_correct: bool
    score: float
    response_time_s: float | None
    occurred_at_s: float | None = None
    command_type: str | None = None
    addressed_call_sign: str | None = None


@dataclass(frozen=True, slots=True)
class AuditoryCapacityGatePlan:
    y_norm: float
    color: str
    shape: str
    aperture_norm: float


@dataclass(frozen=True, slots=True)
class AuditoryCapacityDisturbance:
    vx: float
    vy: float
    duration_s: float


@dataclass(frozen=True, slots=True)
class AuditoryCapacityTrainingProfile:
    enable_gates: bool = True
    enable_state_commands: bool = True
    enable_gate_directives: bool = True
    enable_digit_sequences: bool = True
    enable_trigger_cues: bool = True
    enable_distractors: bool = True
    gate_rate_scale: float = 1.0
    command_rate_scale: float = 1.0
    directive_rate_scale: float = 1.0
    sequence_rate_scale: float = 1.0
    beep_rate_scale: float = 1.0
    response_window_scale: float = 1.0
    disturbance_scale: float = 1.0
    tube_width_scale: float = 1.0
    tube_height_scale: float = 1.0
    noise_level_scale: float = 1.0
    distortion_level_scale: float = 1.0
    digit_sequence_min_len: int = 5
    digit_sequence_max_len: int = 6


@dataclass(frozen=True, slots=True)
class AuditoryCapacityTrainingSegment:
    label: str
    duration_s: float
    active_channels: tuple[str, ...]
    profile: AuditoryCapacityTrainingProfile = AuditoryCapacityTrainingProfile()


@dataclass(slots=True)
class _LiveGate:
    gate_id: int
    x_norm: float
    y_norm: float
    color: str
    shape: str
    aperture_norm: float
    scored: bool = False


@dataclass(slots=True)
class _PendingStateCommand:
    event: AuditoryCapacityInstructionEvent
    expected_color: str | None
    expected_number: int | None


@dataclass(slots=True)
class _PendingRecall:
    event: AuditoryCapacityInstructionEvent
    target_digits: str
    show_until_s: float


@dataclass(slots=True)
class _ActiveGateDirective:
    event: AuditoryCapacityInstructionEvent
    directive: AuditoryCapacityGateDirective
    target_gate_id: int | None = None


@dataclass(slots=True)
class _PendingBeep:
    event: AuditoryCapacityInstructionEvent
    active_until_s: float
    responded: bool = False


class AuditoryCapacityScenarioGenerator:
    CALLSIGNS: tuple[str, ...] = (
        "EAGLE",
        "RAVEN",
        "FALCON",
        "VIPER",
        "COBRA",
        "TALON",
        "MOOSE",
        "LANCER",
        "SABER",
        "NOVA",
    )
    COLORS: tuple[str, ...] = ("RED", "GREEN", "BLUE", "YELLOW")
    SHAPES: tuple[str, ...] = ("CIRCLE", "TRIANGLE", "SQUARE")
    SPEAKERS: tuple[str, ...] = ("lead", "wing", "ops", "aux")
    _GATE_LANES: tuple[float, ...] = (-0.54, -0.36, -0.18, 0.0, 0.18, 0.36, 0.54)
    _APERTURE_BANDS: tuple[tuple[float, float], ...] = (
        (0.10, 0.13),
        (0.13, 0.16),
        (0.16, 0.19),
        (0.19, 0.23),
        (0.23, 0.27),
    )
    _DIRECTIVE_CADENCE_MULTIPLIERS: tuple[float, ...] = (0.84, 1.18, 0.94, 1.06, 0.78, 1.12)

    def __init__(self, *, seed: int) -> None:
        self._rng = SeededRng(seed)
        self._last_gate_lane_index: int | None = None
        self._last_gate_color: str | None = None
        self._last_gate_shape: str | None = None
        self._directive_cadence_index = 0

    def _weighted_pick_index(self, weights: tuple[float, ...]) -> int:
        total = sum(max(0.0, float(weight)) for weight in weights)
        if total <= 1e-9:
            return 0
        pick = self._rng.uniform(0.0, total)
        running = 0.0
        for idx, weight in enumerate(weights):
            running += max(0.0, float(weight))
            if running >= pick:
                return idx
        return max(0, len(weights) - 1)

    def assign_callsigns(self, *, count: int) -> tuple[str, ...]:
        picked = self._rng.sample(self.CALLSIGNS, k=max(1, min(int(count), len(self.CALLSIGNS))))
        return tuple(str(v) for v in picked)

    def next_gate(self, *, difficulty: float, curve_bias: float = 0.0) -> AuditoryCapacityGatePlan:
        d = clamp01(difficulty)
        lane_bias = max(-0.30, min(0.30, float(curve_bias) * 0.42))
        lane_weights = []
        for idx, lane in enumerate(self._GATE_LANES):
            distance = abs(lane - lane_bias)
            weight = max(0.12, 1.25 - (distance * 1.7))
            if self._last_gate_lane_index == idx:
                weight *= 0.58
            lane_weights.append(weight)
        lane_idx = self._weighted_pick_index(tuple(lane_weights))
        self._last_gate_lane_index = lane_idx
        y = self._GATE_LANES[lane_idx] + self._rng.uniform(-0.045, 0.045) + (lane_bias * 0.35)
        y = max(-0.58, min(0.58, y))

        color_choices = [color for color in self.COLORS if color != self._last_gate_color]
        shape_choices = [shape for shape in self.SHAPES if shape != self._last_gate_shape]
        if not color_choices:
            color_choices = list(self.COLORS)
        if not shape_choices:
            shape_choices = list(self.SHAPES)
        color = str(self._rng.choice(color_choices))
        shape = str(self._rng.choice(shape_choices))
        self._last_gate_color = color
        self._last_gate_shape = shape

        band_target = (1.0 - d) * float(len(self._APERTURE_BANDS) - 1)
        band_weights = []
        for idx, _band in enumerate(self._APERTURE_BANDS):
            band_weights.append(max(0.18, 1.40 - abs(float(idx) - band_target)))
        band_idx = self._weighted_pick_index(tuple(band_weights))
        band_lo, band_hi = self._APERTURE_BANDS[band_idx]
        aperture = self._rng.uniform(band_lo, band_hi)
        return AuditoryCapacityGatePlan(
            y_norm=float(y),
            color=color,
            shape=shape,
            aperture_norm=float(aperture),
        )

    def next_disturbance(self, *, difficulty: float) -> AuditoryCapacityDisturbance:
        d = clamp01(difficulty)
        mag_lo = 0.15 + (0.10 * d)
        mag_hi = 0.34 + (0.12 * d)
        magnitude = self._rng.uniform(mag_lo, mag_hi)
        motif = self._rng.choice(
            ("crosswind", "updraft", "downdraft", "diagonal_pos", "diagonal_neg", "eddy")
        )
        if motif == "crosswind":
            vx = self._rng.choice((-1.0, 1.0)) * magnitude
            vy = self._rng.uniform(-magnitude * 0.24, magnitude * 0.24)
        elif motif == "updraft":
            vx = self._rng.uniform(-magnitude * 0.30, magnitude * 0.30)
            vy = magnitude
        elif motif == "downdraft":
            vx = self._rng.uniform(-magnitude * 0.30, magnitude * 0.30)
            vy = -magnitude
        elif motif == "diagonal_pos":
            vx = magnitude * self._rng.uniform(0.45, 1.0)
            vy = magnitude * self._rng.uniform(0.45, 1.0)
        elif motif == "diagonal_neg":
            vx = -magnitude * self._rng.uniform(0.45, 1.0)
            vy = magnitude * self._rng.uniform(0.45, 1.0)
        else:
            vx = self._rng.uniform(-magnitude, magnitude)
            vy = self._rng.uniform(-magnitude, magnitude)
        dur_lo = max(0.28, 0.85 - (0.34 * d))
        dur_hi = max(dur_lo + 0.05, 1.40 - (0.42 * d))
        duration_s = self._rng.uniform(dur_lo, dur_hi)
        return AuditoryCapacityDisturbance(vx=float(vx), vy=float(vy), duration_s=float(duration_s))

    def jittered_interval(self, *, base_s: float, difficulty: float) -> float:
        d = clamp01(difficulty)
        tightness = 0.22 - (0.12 * d)
        lo = max(0.12, base_s * (1.0 - tightness))
        hi = max(lo + 0.02, base_s * (1.0 + tightness))
        return float(self._rng.uniform(lo, hi))

    def directive_interval(self, *, base_s: float, difficulty: float) -> float:
        cadence = self._DIRECTIVE_CADENCE_MULTIPLIERS[
            self._directive_cadence_index % len(self._DIRECTIVE_CADENCE_MULTIPLIERS)
        ]
        self._directive_cadence_index += 1
        return self.jittered_interval(
            base_s=max(0.12, float(base_s) * cadence),
            difficulty=max(0.0, min(1.0, float(difficulty) * 0.94)),
        )

    def choose_addressed_callsign(
        self,
        *,
        assigned_callsigns: tuple[str, ...],
        distractor_rate: float,
    ) -> tuple[str, bool]:
        is_distractor = (
            len(assigned_callsigns) > 0
            and len(assigned_callsigns) < len(self.CALLSIGNS)
            and self._rng.random() < max(0.0, min(1.0, float(distractor_rate)))
        )
        if is_distractor:
            others = [c for c in self.CALLSIGNS if c not in assigned_callsigns]
            if others:
                return str(self._rng.choice(others)), True
        return str(self._rng.choice(assigned_callsigns)), False

    def digit_sequence(self, *, minimum_len: int = 5, maximum_len: int = 6) -> str:
        lo = max(1, int(minimum_len))
        hi = max(lo, int(maximum_len))
        count = self._rng.randint(lo, hi)
        return "".join(str(self._rng.randint(0, 9)) for _ in range(count))

    def next_gate_directive(self) -> AuditoryCapacityGateDirective:
        if self._rng.random() < 0.57:
            return AuditoryCapacityGateDirective(
                action="AVOID" if self._rng.random() < 0.58 else "PASS",
                match_kind="COLOR",
                match_value=str(self._rng.choice(self.COLORS)),
            )
        return AuditoryCapacityGateDirective(
            action="AVOID" if self._rng.random() < 0.58 else "PASS",
            match_kind="SHAPE",
            match_value=str(self._rng.choice(self.SHAPES)),
        )

    def build_instruction_script(
        self,
        *,
        duration_s: float,
        difficulty: float,
        config: AuditoryCapacityConfig,
        callsigns: tuple[str, ...],
        starting_callsign: str,
    ) -> tuple[AuditoryCapacityInstructionEvent, ...]:
        _ = starting_callsign
        if duration_s <= 0.0:
            return ()

        d = clamp01(difficulty)
        command_rate = max(0.05, float(config.command_rate))
        recall_rate = max(0.01, float(config.recall_prompt_rate))
        base_command_interval = 1.0 / command_rate
        base_recall_interval = 1.0 / recall_rate
        response_window = max(0.4, float(config.response_window_seconds))
        active_memory_len = 0
        event_id = 1
        t = max(0.35, min(5.0, base_command_interval * 0.72))
        next_recall_after = t + max(1.0, min(duration_s * 0.45, base_recall_interval * 0.82))
        events: list[AuditoryCapacityInstructionEvent] = []

        while t < (duration_s - 0.12):
            is_distractor = (
                len(callsigns) > 1 and self._rng.random() < max(0.0, min(0.95, config.distractor_rate))
            )
            addressed = str(self._rng.choice(callsigns))
            if is_distractor:
                others = [c for c in self.CALLSIGNS if c not in callsigns]
                if others:
                    addressed = str(self._rng.choice(others))
                else:
                    is_distractor = False

            can_recall = active_memory_len > 0 and t >= next_recall_after
            kind = self._pick_command_type(
                can_recall=can_recall,
                allow_callsign_change=False,
                difficulty=d,
            )
            payload: str | int | None
            if kind is AuditoryCapacityCommandType.CHANGE_COLOUR:
                payload = str(self._rng.choice(self.COLORS))
            elif kind is AuditoryCapacityCommandType.CHANGE_NUMBER:
                payload = int(self._rng.randint(1, 9))
            elif kind is AuditoryCapacityCommandType.SET_FORBIDDEN_GATE_COLOUR:
                payload = str(self._rng.choice(self.COLORS))
            elif kind is AuditoryCapacityCommandType.SET_FORBIDDEN_GATE_SHAPE:
                payload = str(self._rng.choice(self.SHAPES))
            elif kind is AuditoryCapacityCommandType.DIGIT_APPEND:
                payload = str(self._rng.randint(0, 9))
                if not is_distractor:
                    active_memory_len = min(int(config.digit_sequence_max_len), active_memory_len + 1)
            elif kind is AuditoryCapacityCommandType.RECALL_DIGITS:
                payload = None
                if not is_distractor:
                    active_memory_len = 0
                next_recall_after = t + self.jittered_interval(
                    base_s=base_recall_interval,
                    difficulty=d,
                )
            elif kind is AuditoryCapacityCommandType.CHANGE_CALLSIGN:
                payload = str(self._rng.choice(callsigns))
            else:
                payload = "stand by"

            if kind is AuditoryCapacityCommandType.DIGIT_APPEND:
                expires_at = t + max(0.75, min(response_window, config.sequence_display_s))
            elif kind is AuditoryCapacityCommandType.RECALL_DIGITS:
                expires_at = t + max(response_window, config.sequence_response_s)
            elif kind in (
                AuditoryCapacityCommandType.CHANGE_COLOUR,
                AuditoryCapacityCommandType.CHANGE_NUMBER,
            ):
                expires_at = t + response_window
            else:
                expires_at = t + max(1.1, response_window * 0.9)

            events.append(
                AuditoryCapacityInstructionEvent(
                    event_id=event_id,
                    timestamp_s=float(t),
                    addressed_call_sign=str(addressed),
                    speaker_id=str(self._rng.choice(self.SPEAKERS)),
                    command_type=kind,
                    payload=payload,
                    expires_at_s=float(expires_at),
                    is_distractor=bool(is_distractor),
                )
            )
            event_id += 1

            overlap_cut = 0.0
            if config.enable_overlapping_voices and self._rng.random() < (0.10 + (0.18 * d)):
                overlap_cut = self._rng.uniform(0.12, 0.42) * base_command_interval
            t += max(
                0.30,
                self.jittered_interval(base_s=base_command_interval, difficulty=d) - overlap_cut,
            )

        return tuple(events)

    def _pick_command_type(
        self,
        *,
        can_recall: bool,
        allow_callsign_change: bool,
        difficulty: float,
    ) -> AuditoryCapacityCommandType:
        roll = self._rng.random()
        if can_recall and roll < (0.20 + (0.12 * difficulty)):
            return AuditoryCapacityCommandType.RECALL_DIGITS
        if roll < 0.20:
            return AuditoryCapacityCommandType.DIGIT_APPEND
        if roll < 0.38:
            return AuditoryCapacityCommandType.CHANGE_COLOUR
        if roll < 0.54:
            return AuditoryCapacityCommandType.CHANGE_NUMBER
        if roll < 0.68:
            return AuditoryCapacityCommandType.SET_FORBIDDEN_GATE_COLOUR
        if roll < 0.80:
            return AuditoryCapacityCommandType.SET_FORBIDDEN_GATE_SHAPE
        if allow_callsign_change and roll < 0.90:
            return AuditoryCapacityCommandType.CHANGE_CALLSIGN
        return AuditoryCapacityCommandType.NO_OP_DISTRACTOR


def score_sequence_answer(expected: str, response: str) -> float:
    expected_digits = "".join(ch for ch in str(expected) if ch.isdigit())
    response_digits = "".join(ch for ch in str(response) if ch.isdigit())

    if expected_digits == "":
        return 0.0
    if response_digits == expected_digits:
        return 1.0

    target_len = len(expected_digits)
    match_positions = 0
    for idx, digit in enumerate(expected_digits):
        if idx < len(response_digits) and response_digits[idx] == digit:
            match_positions += 1

    position_ratio = match_positions / float(target_len)
    len_delta = abs(len(response_digits) - target_len)
    len_penalty = min(0.45, (len_delta / float(target_len)) * 0.45)
    score = position_ratio - len_penalty
    return max(0.0, min(1.0, score))


def tube_contact_ratio(
    *,
    x: float,
    y: float,
    tube_half_width: float,
    tube_half_height: float,
) -> float:
    half_w = max(1e-6, float(tube_half_width))
    half_h = max(1e-6, float(tube_half_height))
    return math.sqrt(((float(x) / half_w) ** 2) + ((float(y) / half_h) ** 2))


def project_inside_tube(
    *,
    x: float,
    y: float,
    tube_half_width: float,
    tube_half_height: float,
    inset_ratio: float = 0.995,
) -> tuple[float, float, float]:
    ratio = tube_contact_ratio(
        x=x,
        y=y,
        tube_half_width=tube_half_width,
        tube_half_height=tube_half_height,
    )
    if ratio <= 1.0:
        return float(x), float(y), float(ratio)

    inset = max(0.80, min(0.999, float(inset_ratio)))
    scale = inset / ratio
    return float(x) * scale, float(y) * scale, float(ratio)


class AuditoryCapacityEngine:
    _MAX_UPDATE_DT_S = 0.50
    _CIRCLE_GATE_CONTROL_SCALE = 0.78
    _BALL_VISUAL_SUCCESS_COLOR = "GREEN"
    _BALL_VISUAL_ERROR_COLOR = "RED"
    _BALL_VISUAL_IDLE_COLOR = "WHITE"
    _BALL_VISUAL_SUCCESS_DURATION_S = 0.35
    _BALL_VISUAL_COLOR_CHANGE_DURATION_S = 0.30
    _BALL_VISUAL_ERROR_DURATION_S = 0.40
    _ALL_ACTIVE_CHANNELS = (
        "gates",
        "state_commands",
        "gate_directives",
        "digit_recall",
        "trigger",
        "distractors",
    )

    def __init__(
        self,
        *,
        clock: Clock,
        seed: int,
        difficulty: float = 0.5,
        config: AuditoryCapacityConfig | None = None,
        practice_segments: tuple[AuditoryCapacityTrainingSegment, ...] = (),
        scored_segments: tuple[AuditoryCapacityTrainingSegment, ...] = (),
    ) -> None:
        cfg = config or AuditoryCapacityConfig()
        d = clamp01(difficulty)
        run_duration_s = (
            float(cfg.run_duration_seconds)
            if cfg.run_duration_seconds is not None
            else float(cfg.scored_duration_s)
        )
        practice_duration_s = float(cfg.practice_duration_s) if cfg.practice_enabled else 0.0

        if cfg.tick_hz <= 0.0:
            raise ValueError("tick_hz must be > 0")
        if practice_duration_s < 0.0:
            raise ValueError("practice_duration_s must be >= 0")
        if run_duration_s <= 0.0:
            raise ValueError("run_duration_seconds must be > 0")
        if cfg.callsign_count < 1:
            raise ValueError("callsign_count must be >= 1")
        if cfg.command_rate <= 0.0:
            raise ValueError("command_rate must be > 0")
        if cfg.gate_spawn_rate <= 0.0:
            raise ValueError("gate_spawn_rate must be > 0")
        if cfg.recall_prompt_rate <= 0.0:
            raise ValueError("recall_prompt_rate must be > 0")
        if cfg.response_window_seconds <= 0.0:
            raise ValueError("response_window_seconds must be > 0")
        if cfg.digit_sequence_max_len < 1:
            raise ValueError("digit_sequence_max_len must be >= 1")
        if not (0.0 <= cfg.distractor_rate <= 1.0):
            raise ValueError("distractor_rate must be in [0.0, 1.0]")
        if not (0.0 <= cfg.drift_relax_start_ratio <= 0.95):
            raise ValueError("drift_relax_start_ratio must be in [0.0, 0.95]")
        if not (0.0 < cfg.min_drift_scale <= 1.0):
            raise ValueError("min_drift_scale must be in (0.0, 1.0]")

        self._clock = clock
        self._seed = int(seed)
        self._difficulty = d
        self._cfg = cfg
        self._practice_duration_s = practice_duration_s
        self._run_duration_s = run_duration_s
        self._tick_dt = 1.0 / float(cfg.tick_hz)
        self._practice_segments = self._normalize_segments(practice_segments)
        self._scored_segments = self._normalize_segments(scored_segments)

        self._gen = AuditoryCapacityScenarioGenerator(seed=self._seed)

        self._phase = Phase.INSTRUCTIONS
        self._phase_started_at_s = self._clock.now()
        self._last_update_at_s = self._phase_started_at_s
        self._accumulator_s = 0.0
        self._sim_elapsed_s = 0.0
        self._phase_duration_s = 0.0
        self._briefing_duration_s = 0.0
        self._active_segments: tuple[AuditoryCapacityTrainingSegment, ...] = ()
        self._segment_index = 0
        self._segment_started_at_s = 0.0
        self._segment_duration_s = 0.0
        self._segment_label = ""
        self._segment_active_channels: tuple[str, ...] = self._ALL_ACTIVE_CHANNELS
        self._segment_profile = AuditoryCapacityTrainingProfile()

        self._ball_x = 0.0
        self._ball_y = 0.0
        self._control_x = 0.0
        self._control_y = 0.0
        self._ball_contact_ratio = 0.0

        self._dist_x = 0.0
        self._dist_y = 0.0
        self._dist_until_s = 0.0

        self._gates: list[_LiveGate] = []
        self._next_gate_id = 1
        self._next_gate_at_s = 0.0

        self._assigned_callsigns = self._gen.assign_callsigns(count=self._cfg.callsign_count)
        self._active_callsign = "ALL ASSIGNED"
        self._ball_color = "RED"
        self._ball_color_strength = 1.0
        self._ball_visual_flash_color = self._BALL_VISUAL_IDLE_COLOR
        self._ball_visual_flash_started_at_s = 0.0
        self._ball_visual_flash_until_s = 0.0
        self._ball_visual_flash_duration_s = 0.0
        self._ball_number = 1
        self._forbidden_gate_color: str | None = None
        self._forbidden_gate_shape: str | None = None
        self._memory_digits = ""

        self._instruction_script: tuple[AuditoryCapacityInstructionEvent, ...] = ()
        self._next_instruction_index = 0
        self._runtime_event_id = 1
        self._next_state_command_at_s = 0.0
        self._next_gate_directive_at_s = 0.0
        self._next_digit_sequence_at_s = 0.0
        self._next_beep_at_s = 0.0
        self._recent_instruction: AuditoryCapacityInstructionEvent | None = None
        self._recent_instruction_text: str | None = None
        self._pending_state_command: _PendingStateCommand | None = None
        self._pending_distractor_state_command: _PendingStateCommand | None = None
        self._active_gate_directive: _ActiveGateDirective | None = None
        self._active_recall: _PendingRecall | None = None
        self._distractor_recall: _PendingRecall | None = None
        self._active_beep: _PendingBeep | None = None

        self._outside_tube = False
        self._noise_level_override: float | None = None
        self._distortion_level_override: float | None = None
        self._noise_source_override: str | None = None

        self._gate_hits = 0
        self._gate_misses = 0
        self._forbidden_gate_hits = 0
        self._collisions = 0
        self._false_alarms = 0
        self._correct_command_executions = 0
        self._false_responses_to_distractors = 0
        self._missed_valid_commands = 0
        self._digit_recall_attempts = 0
        self._digit_recall_score_total = 0.0

        self._scored_attempted = 0
        self._scored_correct = 0
        self._scored_total_score = 0.0
        self._scored_max_score = 0.0
        self._events: list[AuditoryCapacityEvent] = []

    @property
    def phase(self) -> Phase:
        return self._phase

    def can_exit(self) -> bool:
        return self._phase is not Phase.SCORED

    def set_control(self, *, horizontal: float, vertical: float) -> None:
        self._control_x = max(-1.0, min(1.0, float(horizontal)))
        self._control_y = max(-1.0, min(1.0, float(vertical)))

    def set_colour(self, value: str) -> bool:
        color = self._canonical_color(value)
        if color is None or self._phase not in (Phase.PRACTICE, Phase.SCORED):
            return False
        return self._submit_state_response(color=color, number=None)

    def set_number(self, value: int | str) -> bool:
        number = self._canonical_number(value)
        if number is None or self._phase not in (Phase.PRACTICE, Phase.SCORED):
            return False
        return self._submit_state_response(color=None, number=number)

    def submit_digit_recall(self, raw: str) -> bool:
        if self._phase not in (Phase.PRACTICE, Phase.SCORED):
            return False

        response = "".join(ch for ch in str(raw) if ch.isdigit())
        if response == "":
            return False

        recall = self._active_recall
        if recall is not None and self._sim_elapsed_s >= recall.show_until_s:
            score = score_sequence_answer(recall.target_digits, response)
            rt = max(0.0, self._sim_elapsed_s - recall.show_until_s)
            self._digit_recall_attempts += 1
            self._digit_recall_score_total += score
            self._record_event(
                kind=AuditoryCapacityEventKind.DIGIT_RECALL,
                expected=recall.target_digits,
                response=response,
                is_correct=score >= 1.0 - 1e-9,
                score=score,
                response_time_s=rt,
                command_type=recall.event.command_type.value,
                addressed_call_sign=recall.event.addressed_call_sign,
            )
            self._active_recall = None
            return True

        if self._distractor_recall is not None:
            self._record_false_response(
                response=f"RECALL:{response}",
                command_type=AuditoryCapacityCommandType.RECALL_DIGITS.value,
                distractor=True,
                addressed_call_sign=self._distractor_recall.event.addressed_call_sign,
            )
            self._distractor_recall = None
            return True

        self._record_false_response(
            response=f"RECALL:{response}",
            command_type=AuditoryCapacityCommandType.RECALL_DIGITS.value,
            distractor=False,
            addressed_call_sign=None,
        )
        return True

    def submit_trigger_press(self, raw: str = "TRIGGER") -> bool:
        if self._phase not in (Phase.PRACTICE, Phase.SCORED):
            return False

        pending = self._active_beep
        response = str(raw).strip().upper() or "TRIGGER"
        if pending is not None and self._sim_elapsed_s <= pending.active_until_s and not pending.responded:
            pending.responded = True
            rt = max(0.0, self._sim_elapsed_s - pending.event.timestamp_s)
            self._record_event(
                kind=AuditoryCapacityEventKind.TRIGGER,
                expected="PRESS_TRIGGER",
                response=response,
                is_correct=True,
                score=1.0,
                response_time_s=rt,
                command_type=pending.event.command_type.value,
                addressed_call_sign=pending.event.addressed_call_sign,
            )
            self._active_beep = None
            return True

        self._record_false_response(
            response=response,
            command_type=AuditoryCapacityCommandType.PRESS_TRIGGER.value,
            distractor=False,
            addressed_call_sign=None,
        )
        return True

    def set_audio_overrides(
        self,
        *,
        noise_level: float | None = None,
        distortion_level: float | None = None,
        noise_source: str | None = None,
    ) -> None:
        self._noise_level_override = (
            None if noise_level is None else clamp01(float(noise_level))
        )
        self._distortion_level_override = (
            None if distortion_level is None else clamp01(float(distortion_level))
        )
        source = None if noise_source is None else str(noise_source).strip()
        self._noise_source_override = source or None

    def start_practice(self) -> None:
        if self._phase is not Phase.INSTRUCTIONS:
            return
        if self._practice_duration_s <= 0.0:
            self._phase = Phase.PRACTICE_DONE
            self._phase_started_at_s = self._clock.now()
            return
        self._phase = Phase.PRACTICE
        self._begin_runtime_phase(duration_s=self._practice_duration_s, reset_scores=False)

    def start_scored(self) -> None:
        if self._phase is not Phase.PRACTICE_DONE:
            return
        self._phase = Phase.SCORED
        self._begin_runtime_phase(duration_s=self._run_duration_s, reset_scores=True)

    def submit_answer(self, raw: str) -> bool:
        if self._phase not in (Phase.PRACTICE, Phase.SCORED):
            return False

        command = str(raw).strip().upper()
        if command == "":
            return False

        direct_color = self._canonical_color(command)
        if direct_color is not None:
            return self.set_colour(direct_color)
        if command.startswith("COL:") or command.startswith("COLOR:"):
            _, _, token = command.partition(":")
            return self.set_colour(token)
        if command.startswith("NUM:") or command.startswith("SETNUM:") or command.startswith("BALL:"):
            _, _, token = command.partition(":")
            return self.set_number(token)
        if command.startswith("SEQ:") or command.startswith("DIGITS:"):
            _, _, token = command.partition(":")
            return self.submit_digit_recall(token)
        if any(ch.isdigit() for ch in command):
            return self.submit_digit_recall(command)
        if command in {"BEEP", "SPACE", "TRIGGER"}:
            return self.submit_trigger_press(command)
        if command in {"CALL", "CALLSIGN", "CS"}:
            self._record_false_response(
                response=command,
                command_type="legacy_input",
                distractor=False,
                addressed_call_sign=None,
            )
            return True
        return False

    def update(self) -> None:
        now = self._clock.now()
        dt = now - self._last_update_at_s
        self._last_update_at_s = now

        if dt <= 0.0:
            self._refresh_phase_boundaries(now)
            return

        dt = min(float(dt), self._MAX_UPDATE_DT_S)
        self._accumulator_s += dt

        while self._accumulator_s >= self._tick_dt:
            self._accumulator_s -= self._tick_dt
            if self._phase in (Phase.PRACTICE, Phase.SCORED):
                self._step(self._tick_dt)

        self._refresh_phase_boundaries(now)

    def time_remaining_s(self) -> float | None:
        now = self._clock.now()
        if self._phase is Phase.PRACTICE:
            rem = self._practice_duration_s - (now - self._phase_started_at_s)
            return max(0.0, rem)
        if self._phase is Phase.SCORED:
            rem = self._run_duration_s - (now - self._phase_started_at_s)
            return max(0.0, rem)
        return None

    def snapshot(self) -> TestSnapshot:
        payload = self._build_payload() if self._phase in (Phase.PRACTICE, Phase.SCORED) else None
        return TestSnapshot(
            title="Auditory Capacity",
            phase=self._phase,
            prompt=self.current_prompt(),
            input_hint=(
                "Q/W/E/R change colour  keypad 0-9 sets number  "
                "digits+Enter submits recall  Space or the configured trigger binding answers the beep  "
                "WASD/arrows or HOTAS fly the ball"
            ),
            time_remaining_s=self.time_remaining_s(),
            attempted_scored=self._scored_attempted,
            correct_scored=self._scored_correct,
            payload=payload,
        )

    def current_prompt(self) -> str:
        if self._phase is Phase.INSTRUCTIONS:
            callsign_roster = ", ".join(self._assigned_callsigns) if self._assigned_callsigns else "—"
            return "\n".join(
                [
                    "Auditory Capacity Trainer",
                    "",
                    "Track the ball through the tunnel while following call-sign filtered instructions.",
                    f"- Your call signs for this run: {callsign_roster}",
                    "- Follow instructions addressed to any of your assigned call signs",
                    "- Q/W/E/R changes colour and keypad 0-9 changes the ball number",
                    "- When you hear a 5-6 digit sequence, remember it and type it with Enter",
                    "- When you hear the beep, press the configured trigger binding or Space once",
                    "- Gate instructions apply to the next matching gate only",
                    "- Ignore instructions addressed to call signs that are not yours",
                    "",
                    "Once the scored run starts, exit stays locked until completion.",
                    "Press Enter to start practice.",
                ]
            )
        if self._phase is Phase.PRACTICE_DONE:
            callsign_roster = ", ".join(self._assigned_callsigns) if self._assigned_callsigns else "—"
            return (
                f"Practice complete. Your call signs remain {callsign_roster}. "
                "Review the rules, then press Enter to begin the timed scored block."
            )
        if self._phase is Phase.RESULTS:
            s = self.scored_summary()
            mean_rt = (
                "—"
                if s.mean_response_time_s is None
                else f"{s.mean_response_time_s * 1000.0:.0f} ms"
            )
            recall_accuracy = self._digit_recall_accuracy()
            return "\n".join(
                [
                    "Results",
                    "",
                    f"Attempted: {s.attempted}",
                    f"Correct:   {s.correct}",
                    f"Accuracy:  {s.accuracy * 100.0:.1f}%",
                    f"Rate:      {s.throughput_per_min:.1f} / min",
                    f"Mean RT:   {mean_rt}",
                    f"Gates hit/miss: {self._gate_hits}/{self._gate_misses}",
                    f"Forbidden gate hits: {self._forbidden_gate_hits}",
                    f"Boundary violations: {self._collisions}",
                    f"Cmd ok/missed: {self._correct_command_executions}/{self._missed_valid_commands}",
                    f"Recall accuracy: {recall_accuracy * 100.0:.1f}%",
                    "",
                    "Press Enter to return.",
                ]
            )
        return "Track, filter by call sign, follow next-gate directives, recall digit groups, and answer the beep cue."

    def scored_summary(self) -> AttemptSummary:
        duration = float(self._run_duration_s)
        attempted = int(self._scored_attempted)
        correct = int(self._scored_correct)
        accuracy = 0.0 if attempted == 0 else correct / attempted
        throughput = (attempted / duration) * 60.0
        rts = [
            float(evt.response_time_s)
            for evt in self._events
            if evt.phase is Phase.SCORED and evt.response_time_s is not None
        ]
        mean_rt = None if not rts else (sum(rts) / len(rts))
        total_score = float(self._scored_total_score)
        max_score = float(self._scored_max_score)
        ratio = 0.0 if max_score <= 0.0 else total_score / max_score
        return AttemptSummary(
            attempted=attempted,
            correct=correct,
            accuracy=float(accuracy),
            duration_s=duration,
            throughput_per_min=float(throughput),
            mean_response_time_s=mean_rt,
            total_score=total_score,
            max_score=max_score,
            score_ratio=float(ratio),
        )

    def events(self) -> list[AuditoryCapacityEvent]:
        return list(self._events)

    def result_metrics(self) -> dict[str, str]:
        return {
            "auditory.boundary_violations": str(int(self._collisions)),
            "auditory.gate_hits": str(int(self._gate_hits)),
            "auditory.gate_misses": str(int(self._gate_misses)),
            "auditory.forbidden_gate_hits": str(int(self._forbidden_gate_hits)),
            "auditory.false_alarms": str(int(self._false_alarms)),
            "auditory.correct_command_executions": str(int(self._correct_command_executions)),
            "auditory.false_responses_to_distractors": str(
                int(self._false_responses_to_distractors)
            ),
            "auditory.missed_valid_commands": str(int(self._missed_valid_commands)),
            "auditory.digit_recall_attempts": str(int(self._digit_recall_attempts)),
            "auditory.digit_recall_accuracy": f"{self._digit_recall_accuracy():.6f}",
        }

    @classmethod
    def _normalize_segments(
        cls, segments: tuple[AuditoryCapacityTrainingSegment, ...]
    ) -> tuple[AuditoryCapacityTrainingSegment, ...]:
        normalized: list[AuditoryCapacityTrainingSegment] = []
        for segment in segments:
            duration_s = max(0.0, float(segment.duration_s))
            if duration_s <= 1e-9:
                continue
            channels = tuple(
                str(channel).strip()
                for channel in segment.active_channels
                if str(channel).strip() != ""
            )
            normalized.append(
                AuditoryCapacityTrainingSegment(
                    label=str(segment.label).strip() or "Segment",
                    duration_s=duration_s,
                    active_channels=channels or cls._ALL_ACTIVE_CHANNELS,
                    profile=segment.profile,
                )
            )
        return tuple(normalized)

    def _resolve_phase_segments(
        self,
        *,
        duration_s: float,
        phase_label: str,
        custom_segments: tuple[AuditoryCapacityTrainingSegment, ...],
    ) -> tuple[AuditoryCapacityTrainingSegment, ...]:
        total_duration = max(0.0, float(duration_s))
        if total_duration <= 1e-9:
            return ()
        if not custom_segments:
            return (
                AuditoryCapacityTrainingSegment(
                    label=phase_label,
                    duration_s=total_duration,
                    active_channels=self._ALL_ACTIVE_CHANNELS,
                ),
            )

        remaining = total_duration
        resolved: list[AuditoryCapacityTrainingSegment] = []
        for segment in custom_segments:
            if remaining <= 1e-9:
                break
            seg_duration = min(max(0.0, float(segment.duration_s)), remaining)
            if seg_duration <= 1e-9:
                continue
            resolved.append(
                AuditoryCapacityTrainingSegment(
                    label=segment.label,
                    duration_s=seg_duration,
                    active_channels=segment.active_channels,
                    profile=segment.profile,
                )
            )
            remaining -= seg_duration
        if not resolved:
            return (
                AuditoryCapacityTrainingSegment(
                    label=phase_label,
                    duration_s=total_duration,
                    active_channels=self._ALL_ACTIVE_CHANNELS,
                ),
            )
        if remaining > 1e-9:
            last = resolved[-1]
            resolved[-1] = AuditoryCapacityTrainingSegment(
                label=last.label,
                duration_s=float(last.duration_s + remaining),
                active_channels=last.active_channels,
                profile=last.profile,
            )
        return tuple(resolved)

    def _channel_active(self, channel: str) -> bool:
        return str(channel).strip() in self._segment_active_channels

    def _effective_tube_half_width(self) -> float:
        scale = max(0.40, float(self._segment_profile.tube_width_scale))
        return max(0.20, float(self._cfg.tube_half_width) * scale)

    def _effective_tube_half_height(self) -> float:
        scale = max(0.40, float(self._segment_profile.tube_height_scale))
        return max(0.16, float(self._cfg.tube_half_height) * scale)

    def _effective_response_window_s(self) -> float:
        return max(
            0.40,
            float(self._cfg.response_window_seconds)
            * max(0.35, float(self._segment_profile.response_window_scale)),
        )

    def _effective_gate_interval_s(self) -> float:
        scale = max(0.05, float(self._segment_profile.gate_rate_scale))
        return self._gate_interval_s() / scale

    def _effective_command_interval_s(self) -> float:
        base = 1.0 / max(0.05, float(self._cfg.command_rate))
        scale = max(0.05, float(self._segment_profile.command_rate_scale))
        return base / scale

    def _effective_directive_interval_s(self) -> float:
        base = max(5.2, self._effective_gate_interval_s() * 1.45)
        scale = max(0.05, float(self._segment_profile.directive_rate_scale))
        return base / scale

    def _effective_sequence_interval_s(self) -> float:
        base = max(24.0, float(self._cfg.sequence_interval_s))
        scale = max(0.05, float(self._segment_profile.sequence_rate_scale))
        return base / scale

    def _effective_beep_interval_s(self) -> float:
        base = max(10.0, float(self._cfg.beep_interval_s))
        scale = max(0.05, float(self._segment_profile.beep_rate_scale))
        return base / scale

    def _reset_segment_runtime_state(self) -> None:
        self._gates.clear()
        self._recent_instruction = None
        self._recent_instruction_text = None
        self._pending_state_command = None
        self._pending_distractor_state_command = None
        self._active_gate_directive = None
        self._active_recall = None
        self._distractor_recall = None
        self._active_beep = None
        self._memory_digits = ""

    def _schedule_runtime_channels(self, *, start_at_s: float) -> None:
        if self._channel_active("gates"):
            gate_interval = self._effective_gate_interval_s()
            self._next_gate_at_s = start_at_s + max(0.60, min(18.0, gate_interval * 0.78))
        else:
            self._next_gate_at_s = math.inf

        if self._channel_active("state_commands"):
            self._next_state_command_at_s = start_at_s + max(
                0.85,
                self._gen.jittered_interval(
                    base_s=self._effective_command_interval_s(),
                    difficulty=self._difficulty,
                )
                * 0.72,
            )
        else:
            self._next_state_command_at_s = math.inf

        if self._channel_active("gate_directives") and self._channel_active("gates"):
            self._next_gate_directive_at_s = start_at_s + max(
                4.0,
                self._gen.directive_interval(
                    base_s=self._effective_directive_interval_s(),
                    difficulty=self._difficulty,
                ),
            )
        else:
            self._next_gate_directive_at_s = math.inf

        if self._channel_active("digit_recall"):
            self._next_digit_sequence_at_s = start_at_s + max(
                7.0,
                min(
                    max(28.0, self._phase_duration_s * 0.24),
                    self._effective_sequence_interval_s(),
                ),
            )
        else:
            self._next_digit_sequence_at_s = math.inf

        if self._channel_active("trigger"):
            self._next_beep_at_s = start_at_s + max(
                6.0,
                self._gen.jittered_interval(
                    base_s=self._effective_beep_interval_s(),
                    difficulty=self._difficulty,
                )
                * 0.80,
            )
        else:
            self._next_beep_at_s = math.inf

    def _activate_segment(self, index: int) -> None:
        if index < 0 or index >= len(self._active_segments):
            return
        segment = self._active_segments[index]
        self._segment_index = int(index)
        self._segment_started_at_s = float(self._sim_elapsed_s)
        self._segment_duration_s = max(0.0, float(segment.duration_s))
        self._segment_label = str(segment.label)
        channels = tuple(
            channel for channel in segment.active_channels if str(channel).strip() != ""
        )
        self._segment_active_channels = channels or self._ALL_ACTIVE_CHANNELS
        self._segment_profile = segment.profile
        self._reset_segment_runtime_state()
        self._schedule_runtime_channels(start_at_s=self._sim_elapsed_s)

    def _sync_active_segment(self) -> None:
        if not self._active_segments:
            return
        while (
            self._segment_index + 1 < len(self._active_segments)
            and self._sim_elapsed_s >= (self._segment_started_at_s + self._segment_duration_s)
        ):
            self._activate_segment(self._segment_index + 1)

    def _begin_runtime_phase(self, *, duration_s: float, reset_scores: bool) -> None:
        now = self._clock.now()
        self._phase_started_at_s = now
        self._last_update_at_s = now
        self._accumulator_s = 0.0
        self._sim_elapsed_s = 0.0
        self._phase_duration_s = max(0.0, float(duration_s))
        phase_segments = (
            self._scored_segments if self._phase is Phase.SCORED else self._practice_segments
        )
        phase_label = "Scored" if self._phase is Phase.SCORED else "Practice"
        self._active_segments = self._resolve_phase_segments(
            duration_s=self._phase_duration_s,
            phase_label=phase_label,
            custom_segments=phase_segments,
        )
        if phase_segments:
            self._briefing_duration_s = 0.0
        else:
            self._briefing_duration_s = min(8.0, max(3.0, self._phase_duration_s * 0.15))

        if not self._assigned_callsigns:
            self._assigned_callsigns = self._gen.assign_callsigns(count=self._cfg.callsign_count)
        self._active_callsign = "ALL ASSIGNED"
        self._ball_x = 0.0
        self._ball_y = 0.0
        self._ball_contact_ratio = 0.0
        self._control_x = 0.0
        self._control_y = 0.0
        self._dist_x = 0.0
        self._dist_y = 0.0
        self._dist_until_s = 0.0
        self._ball_color = "RED"
        self._ball_color_strength = 1.0
        self._ball_visual_flash_color = self._BALL_VISUAL_IDLE_COLOR
        self._ball_visual_flash_started_at_s = 0.0
        self._ball_visual_flash_until_s = 0.0
        self._ball_visual_flash_duration_s = 0.0
        self._ball_number = 1
        self._forbidden_gate_color = None
        self._forbidden_gate_shape = None
        self._memory_digits = ""
        self._outside_tube = False

        self._next_gate_id = 1

        self._instruction_script = self._build_briefing_script(duration_s=self._phase_duration_s)
        self._next_instruction_index = 0
        self._runtime_event_id = max(
            1,
            len(self._instruction_script) + 1,
        )
        self._activate_segment(0)
        if self._briefing_duration_s > 0.0:
            self._next_gate_at_s = max(self._next_gate_at_s, self._briefing_duration_s)
            self._next_state_command_at_s = max(self._next_state_command_at_s, self._briefing_duration_s)
            self._next_gate_directive_at_s = max(
                self._next_gate_directive_at_s, self._briefing_duration_s
            )
            self._next_digit_sequence_at_s = max(self._next_digit_sequence_at_s, self._briefing_duration_s)
            self._next_beep_at_s = max(self._next_beep_at_s, self._briefing_duration_s)

        self._gate_hits = 0
        self._gate_misses = 0
        self._forbidden_gate_hits = 0
        self._collisions = 0
        self._false_alarms = 0
        self._correct_command_executions = 0
        self._false_responses_to_distractors = 0
        self._missed_valid_commands = 0
        self._digit_recall_attempts = 0
        self._digit_recall_score_total = 0.0

        if reset_scores:
            self._scored_attempted = 0
            self._scored_correct = 0
            self._scored_total_score = 0.0
            self._scored_max_score = 0.0

    def _step(self, dt: float) -> None:
        self._sim_elapsed_s += dt
        self._sync_active_segment()

        self._update_disturbance()
        control_scale = self._ball_control_scale()
        next_x = self._ball_x + (
            ((self._control_x * self._cfg.control_gain * control_scale))
            + (self._dist_x * self._cfg.disturbance_gain)
        ) * dt
        next_y = self._ball_y + (
            ((self._control_y * self._cfg.control_gain * control_scale))
            + (self._dist_y * self._cfg.disturbance_gain)
        ) * dt

        next_x = max(-1.15, min(1.15, next_x))
        next_y = max(-0.92, min(0.92, next_y))

        clamped_x, clamped_y, contact_ratio = project_inside_tube(
            x=next_x,
            y=next_y,
            tube_half_width=self._effective_tube_half_width(),
            tube_half_height=self._effective_tube_half_height(),
        )
        self._ball_x = clamped_x
        self._ball_y = clamped_y
        self._ball_contact_ratio = contact_ratio

        outside_now = contact_ratio >= 1.0
        if outside_now and not self._outside_tube:
            self._outside_tube = True
            self._collisions += 1
            self._record_event(
                kind=AuditoryCapacityEventKind.COLLISION,
                expected="inside tunnel",
                response="boundary_violation",
                is_correct=False,
                score=0.0,
                response_time_s=None,
                command_type=None,
                addressed_call_sign=None,
            )
        elif not outside_now and self._outside_tube:
            self._outside_tube = False

        self._update_instruction_channel()
        self._update_gates(dt)
        self._ball_color_strength = 1.0 if self._ball_color else 0.0

    def _ball_control_scale(self) -> float:
        gate = self._nearest_upcoming_gate()
        if gate is None:
            return 1.0
        if str(gate.shape).upper() == "CIRCLE":
            return float(self._CIRCLE_GATE_CONTROL_SCALE)
        return 1.0

    def _nearest_upcoming_gate(self) -> _LiveGate | None:
        closest: _LiveGate | None = None
        closest_x = float("inf")
        for gate in self._gates:
            x_norm = float(gate.x_norm)
            if x_norm < AUDITORY_GATE_PLAYER_X_NORM:
                continue
            if x_norm < closest_x:
                closest = gate
                closest_x = x_norm
        return closest

    def _update_disturbance(self) -> None:
        if self._sim_elapsed_s < self._briefing_duration_s:
            self._dist_x = 0.0
            self._dist_y = 0.0
            self._dist_until_s = self._briefing_duration_s
            return
        if self._sim_elapsed_s < self._dist_until_s:
            return
        disturbance = self._gen.next_disturbance(difficulty=self._difficulty)
        drift_scale = self._drift_scale()
        curve_scale = 1.0 + (max(0.0, float(self._cfg.tunnel_curvature_intensity)) * 0.20)
        disturbance_scale = max(0.0, float(self._segment_profile.disturbance_scale))
        self._dist_x = disturbance.vx * drift_scale * curve_scale * disturbance_scale
        self._dist_y = disturbance.vy * drift_scale * curve_scale * disturbance_scale
        linger = 1.0 + ((1.0 - drift_scale) * 0.45)
        self._dist_until_s = self._sim_elapsed_s + (disturbance.duration_s * linger)

    def _drift_scale(self) -> float:
        start = float(self._cfg.drift_relax_start_ratio)
        minimum = float(self._cfg.min_drift_scale)
        progress = self._phase_progress_ratio()
        if progress <= start:
            return 1.0
        remaining = max(1e-6, 1.0 - start)
        t = clamp01((progress - start) / remaining)
        return 1.0 - ((1.0 - minimum) * t)

    def _phase_progress_ratio(self) -> float:
        duration = 0.0
        if self._phase is Phase.SCORED:
            duration = self._run_duration_s
        elif self._phase is Phase.PRACTICE:
            duration = self._practice_duration_s
        if duration <= 0.0:
            return 0.0
        elapsed = max(0.0, self._clock.now() - self._phase_started_at_s)
        return clamp01(elapsed / duration)

    def _update_instruction_channel(self) -> None:
        while (
            self._next_instruction_index < len(self._instruction_script)
            and self._instruction_script[self._next_instruction_index].timestamp_s <= self._sim_elapsed_s
        ):
            event = self._instruction_script[self._next_instruction_index]
            self._next_instruction_index += 1
            self._activate_instruction(event)

        self._advance_runtime_scheduler()

        if self._recent_instruction is not None and self._sim_elapsed_s > self._recent_instruction.expires_at_s:
            self._recent_instruction = None
            self._recent_instruction_text = None

        pending = self._pending_state_command
        if pending is not None and self._sim_elapsed_s > pending.event.expires_at_s:
            self._missed_valid_commands += 1
            self._record_event(
                kind=AuditoryCapacityEventKind.COMMAND,
                expected=self._pending_command_expected(pending),
                response="MISS",
                is_correct=False,
                score=0.0,
                response_time_s=None,
                command_type=pending.event.command_type.value,
                addressed_call_sign=pending.event.addressed_call_sign,
            )
            self._pending_state_command = None

        if (
            self._pending_distractor_state_command is not None
            and self._sim_elapsed_s > self._pending_distractor_state_command.event.expires_at_s
        ):
            self._pending_distractor_state_command = None

        recall = self._active_recall
        if recall is not None and self._sim_elapsed_s > recall.event.expires_at_s:
            self._digit_recall_attempts += 1
            self._record_event(
                kind=AuditoryCapacityEventKind.DIGIT_RECALL,
                expected=recall.target_digits,
                response="MISS",
                is_correct=False,
                score=0.0,
                response_time_s=None,
                command_type=recall.event.command_type.value,
                addressed_call_sign=recall.event.addressed_call_sign,
            )
            self._active_recall = None

        if self._distractor_recall is not None and self._sim_elapsed_s > self._distractor_recall.event.expires_at_s:
            self._distractor_recall = None

        beep = self._active_beep
        if beep is not None and self._sim_elapsed_s > beep.active_until_s:
            if not beep.responded:
                self._record_event(
                    kind=AuditoryCapacityEventKind.TRIGGER,
                    expected="PRESS_TRIGGER",
                    response="MISS",
                    is_correct=False,
                    score=0.0,
                    response_time_s=None,
                    command_type=beep.event.command_type.value,
                    addressed_call_sign=beep.event.addressed_call_sign,
                )
            self._active_beep = None

    def _advance_runtime_scheduler(self) -> None:
        if self._sim_elapsed_s < self._briefing_duration_s:
            return

        while True:
            due_at = min(
                self._next_state_command_at_s,
                self._next_gate_directive_at_s,
                self._next_digit_sequence_at_s,
                self._next_beep_at_s,
            )
            if due_at <= 0.0 or due_at > self._sim_elapsed_s:
                return

            if due_at == self._next_state_command_at_s:
                event = self._build_runtime_state_command_event(timestamp_s=due_at)
                self._next_state_command_at_s = due_at + self._gen.jittered_interval(
                    base_s=self._effective_command_interval_s(),
                    difficulty=self._difficulty,
                )
            elif due_at == self._next_gate_directive_at_s:
                event = self._build_runtime_gate_directive_event(timestamp_s=due_at)
                self._next_gate_directive_at_s = due_at + self._gen.directive_interval(
                    base_s=self._effective_directive_interval_s(),
                    difficulty=self._difficulty,
                )
            elif due_at == self._next_digit_sequence_at_s:
                event = self._build_runtime_digit_sequence_event(timestamp_s=due_at)
                self._next_digit_sequence_at_s = due_at + self._gen.jittered_interval(
                    base_s=max(
                        24.0,
                        min(
                            max(28.0, self._phase_duration_s * 0.24),
                            self._effective_sequence_interval_s(),
                        ),
                    ),
                    difficulty=self._difficulty,
                )
            else:
                event = self._build_runtime_beep_event(timestamp_s=due_at)
                self._next_beep_at_s = due_at + self._gen.jittered_interval(
                    base_s=self._effective_beep_interval_s(),
                    difficulty=self._difficulty,
                )
            self._activate_instruction(event)

    def _activate_instruction(self, event: AuditoryCapacityInstructionEvent) -> None:
        self._recent_instruction = event
        self._recent_instruction_text = self._instruction_text(event)
        should_follow = (
            (not event.is_distractor)
            and (
                event.addressed_call_sign in self._assigned_callsigns
                or event.addressed_call_sign == "ALL ASSIGNED"
            )
        )

        if event.command_type is AuditoryCapacityCommandType.CHANGE_COLOUR:
            pending = _PendingStateCommand(
                event=event,
                expected_color=self._canonical_color(event.payload),
                expected_number=None,
            )
            if should_follow:
                self._replace_pending_state_command(pending)
            else:
                self._pending_distractor_state_command = pending
            return

        if event.command_type is AuditoryCapacityCommandType.CHANGE_NUMBER:
            pending = _PendingStateCommand(
                event=event,
                expected_color=None,
                expected_number=self._canonical_number(event.payload),
            )
            if should_follow:
                self._replace_pending_state_command(pending)
            else:
                self._pending_distractor_state_command = pending
            return

        if event.command_type is AuditoryCapacityCommandType.GATE_DIRECTIVE:
            directive = (
                event.payload
                if isinstance(event.payload, AuditoryCapacityGateDirective)
                else None
            )
            if directive is None:
                return
            if should_follow:
                self._active_gate_directive = _ActiveGateDirective(
                    event=event,
                    directive=directive,
                    target_gate_id=None,
                )
                self._bind_gate_directive_to_next_match()
            return

        if event.command_type is AuditoryCapacityCommandType.DIGIT_SEQUENCE:
            digits = "".join(ch for ch in str(event.payload) if ch.isdigit())
            if digits == "":
                return
            show_until = event.timestamp_s + max(0.60, float(self._cfg.sequence_display_s))
            pending = _PendingRecall(
                event=event,
                target_digits=digits,
                show_until_s=show_until,
            )
            if should_follow:
                self._active_recall = pending
                self._memory_digits = digits
            else:
                self._distractor_recall = pending
            return

        if event.command_type is AuditoryCapacityCommandType.PRESS_TRIGGER:
            if should_follow:
                window_s = min(
                    float(event.expires_at_s),
                    event.timestamp_s + 0.85,
                )
                self._active_beep = _PendingBeep(
                    event=event,
                    active_until_s=window_s,
                )
            return

        if event.command_type is AuditoryCapacityCommandType.RECALL_DIGITS:
            show_until = event.timestamp_s + max(0.35, float(self._cfg.sequence_display_s))
            pending = _PendingRecall(
                event=event,
                target_digits=self._memory_digits,
                show_until_s=show_until,
            )
            if should_follow:
                self._active_recall = pending
                self._memory_digits = ""
            else:
                self._distractor_recall = pending
            return

        if not should_follow:
            return

        if event.command_type is AuditoryCapacityCommandType.SET_FORBIDDEN_GATE_COLOUR:
            self._forbidden_gate_color = self._canonical_color(event.payload)
            return
        if event.command_type is AuditoryCapacityCommandType.SET_FORBIDDEN_GATE_SHAPE:
            self._forbidden_gate_shape = self._canonical_shape(event.payload)
            return
        if event.command_type is AuditoryCapacityCommandType.DIGIT_APPEND:
            digit = "".join(ch for ch in str(event.payload) if ch.isdigit())[:1]
            if digit and len(self._memory_digits) < self._cfg.digit_sequence_max_len:
                self._memory_digits += digit
            return
        if event.command_type is AuditoryCapacityCommandType.CHANGE_CALLSIGN:
            return

    def _choose_runtime_addressed_callsign(self, *, distractor_rate: float) -> tuple[str, bool]:
        if not self._segment_profile.enable_distractors:
            if self._assigned_callsigns:
                return (str(self._gen._rng.choice(self._assigned_callsigns)), False)
            return ("ALL ASSIGNED", False)
        return self._gen.choose_addressed_callsign(
            assigned_callsigns=self._assigned_callsigns,
            distractor_rate=distractor_rate,
        )

    def _build_runtime_state_command_event(self, *, timestamp_s: float) -> AuditoryCapacityInstructionEvent:
        addressed, is_distractor = self._choose_runtime_addressed_callsign(
            distractor_rate=self._cfg.distractor_rate,
        )
        if self._gen._rng.random() < 0.54:
            command_type = AuditoryCapacityCommandType.CHANGE_COLOUR
            payload: str | int | AuditoryCapacityGateDirective | None = str(
                self._gen._rng.choice(self._gen.COLORS)
            )
        else:
            command_type = AuditoryCapacityCommandType.CHANGE_NUMBER
            payload = int(self._gen._rng.randint(0, 9))
        event = AuditoryCapacityInstructionEvent(
            event_id=self._runtime_event_id,
            timestamp_s=float(timestamp_s),
            addressed_call_sign=addressed,
            speaker_id="instructor",
            command_type=command_type,
            payload=payload,
            expires_at_s=float(timestamp_s + max(0.85, self._effective_response_window_s())),
            is_distractor=is_distractor,
        )
        self._runtime_event_id += 1
        return event

    def _build_runtime_gate_directive_event(self, *, timestamp_s: float) -> AuditoryCapacityInstructionEvent:
        addressed, is_distractor = self._choose_runtime_addressed_callsign(
            distractor_rate=self._cfg.distractor_rate * 0.85,
        )
        event = AuditoryCapacityInstructionEvent(
            event_id=self._runtime_event_id,
            timestamp_s=float(timestamp_s),
            addressed_call_sign=addressed,
            speaker_id="instructor",
            command_type=AuditoryCapacityCommandType.GATE_DIRECTIVE,
            payload=self._gen.next_gate_directive(),
            expires_at_s=float(timestamp_s + max(1.8, self._effective_response_window_s() * 1.25)),
            is_distractor=is_distractor,
        )
        self._runtime_event_id += 1
        return event

    def _build_runtime_digit_sequence_event(self, *, timestamp_s: float) -> AuditoryCapacityInstructionEvent:
        max_len = max(
            1,
            min(
                int(self._cfg.digit_sequence_max_len),
                int(self._segment_profile.digit_sequence_max_len),
            ),
        )
        min_len = max(
            1,
            min(max_len, int(self._segment_profile.digit_sequence_min_len)),
        )
        digits = self._gen.digit_sequence(
            minimum_len=min_len,
            maximum_len=max_len,
        )
        event = AuditoryCapacityInstructionEvent(
            event_id=self._runtime_event_id,
            timestamp_s=float(timestamp_s),
            addressed_call_sign=str(self._gen._rng.choice(self._assigned_callsigns)),
            speaker_id="instructor",
            command_type=AuditoryCapacityCommandType.DIGIT_SEQUENCE,
            payload=digits,
            expires_at_s=float(
                timestamp_s
                + max(0.60, float(self._cfg.sequence_display_s))
                + max(
                    1.50,
                    float(self._cfg.sequence_response_s)
                    * max(0.35, float(self._segment_profile.response_window_scale)),
                )
            ),
            is_distractor=False,
        )
        self._runtime_event_id += 1
        return event

    def _build_runtime_beep_event(self, *, timestamp_s: float) -> AuditoryCapacityInstructionEvent:
        event = AuditoryCapacityInstructionEvent(
            event_id=self._runtime_event_id,
            timestamp_s=float(timestamp_s),
            addressed_call_sign="ALL ASSIGNED",
            speaker_id="instructor",
            command_type=AuditoryCapacityCommandType.PRESS_TRIGGER,
            payload="beep",
            expires_at_s=float(timestamp_s + min(0.85, self._effective_response_window_s())),
            is_distractor=False,
        )
        self._runtime_event_id += 1
        return event

    def _build_briefing_script(self, *, duration_s: float) -> tuple[AuditoryCapacityInstructionEvent, ...]:
        if duration_s <= 0.0:
            return ()
        callsigns = ", ".join(self._assigned_callsigns) if self._assigned_callsigns else "—"
        lines = [
            "Stay calm. The run starts quiet while the instructor briefs you.",
            f"Your call signs for this block are {callsigns}. Follow only those.",
            "Use Q W E R for colour, keypad numbers for the ball, and type digit sequences with Enter.",
            "When you hear the beep, press trigger or Space once. Next-gate instructions apply only to the next matching gate.",
        ]
        slots = max(1, len(lines))
        spacing = max(0.9, min(2.2, self._briefing_duration_s / float(slots + 1)))
        events: list[AuditoryCapacityInstructionEvent] = []
        event_id = 1
        for idx, line in enumerate(lines):
            timestamp_s = min(
                max(0.18, spacing * (idx + 0.45)),
                max(0.22, self._briefing_duration_s - 0.30),
            )
            expires_at_s = min(self._briefing_duration_s, timestamp_s + 2.6)
            events.append(
                AuditoryCapacityInstructionEvent(
                    event_id=event_id,
                    timestamp_s=float(timestamp_s),
                    addressed_call_sign="INSTRUCTOR",
                    speaker_id="instructor",
                    command_type=AuditoryCapacityCommandType.NO_OP_DISTRACTOR,
                    payload=line,
                    expires_at_s=float(expires_at_s),
                    is_distractor=False,
                )
            )
            event_id += 1
        return tuple(events)

    @staticmethod
    def _stable_variant_index(*, seed: int, salt: str, event_id: int, count: int) -> int:
        total = (int(seed) & 0xFFFFFFFF) ^ ((int(event_id) * 0x45D9F3B) & 0xFFFFFFFF)
        for ch in salt:
            total = ((total * 33) + ord(ch)) & 0xFFFFFFFF
        return total % max(1, int(count))

    def _trigger_ball_visual_feedback(self, *, color: str, duration_s: float) -> None:
        self._ball_visual_flash_color = (
            str(color).strip().upper() or self._BALL_VISUAL_IDLE_COLOR
        )
        self._ball_visual_flash_started_at_s = float(self._sim_elapsed_s)
        self._ball_visual_flash_duration_s = max(0.05, float(duration_s))
        self._ball_visual_flash_until_s = (
            self._ball_visual_flash_started_at_s + self._ball_visual_flash_duration_s
        )

    def _ball_visual_state(self) -> tuple[str, float]:
        if self._ball_contact_ratio >= 1.0:
            return (self._BALL_VISUAL_ERROR_COLOR, 1.0)
        if self._sim_elapsed_s >= self._ball_visual_flash_until_s:
            return (self._BALL_VISUAL_IDLE_COLOR, 0.0)
        duration = max(0.05, float(self._ball_visual_flash_duration_s))
        remaining = max(0.0, float(self._ball_visual_flash_until_s - self._sim_elapsed_s))
        strength = clamp01(remaining / duration)
        return (self._ball_visual_flash_color, strength**0.85)

    def _bind_gate_directive_to_next_match(self) -> None:
        directive_state = self._active_gate_directive
        if directive_state is None:
            return

        active_gate = None
        if directive_state.target_gate_id is not None:
            active_gate = next(
                (gate for gate in self._gates if gate.gate_id == directive_state.target_gate_id and not gate.scored),
                None,
            )
            if active_gate is None:
                directive_state.target_gate_id = None

        if directive_state.target_gate_id is not None:
            return

        matches = [
            gate
            for gate in self._gates
            if (not gate.scored)
            and gate.x_norm >= AUDITORY_GATE_PLAYER_X_NORM
            and self._gate_matches_directive(gate, directive_state.directive)
        ]
        if not matches:
            return
        target = min(matches, key=lambda gate: float(gate.x_norm))
        directive_state.target_gate_id = int(target.gate_id)

    @staticmethod
    def _gate_matches_directive(gate: _LiveGate, directive: AuditoryCapacityGateDirective) -> bool:
        if directive.match_kind == "COLOR":
            return str(gate.color).upper() == directive.match_value
        if directive.match_kind == "SHAPE":
            return str(gate.shape).upper() == directive.match_value
        return False

    def _replace_pending_state_command(self, pending: _PendingStateCommand) -> None:
        current = self._pending_state_command
        if current is not None:
            self._missed_valid_commands += 1
            self._record_event(
                kind=AuditoryCapacityEventKind.COMMAND,
                expected=self._pending_command_expected(current),
                response="MISS",
                is_correct=False,
                score=0.0,
                response_time_s=None,
                command_type=current.event.command_type.value,
                addressed_call_sign=current.event.addressed_call_sign,
            )
        self._pending_state_command = pending

    def _update_gates(self, dt: float) -> None:
        if not self._channel_active("gates"):
            self._gates.clear()
            self._active_gate_directive = None
            self._next_gate_at_s = math.inf
            return
        if self._sim_elapsed_s >= self._next_gate_at_s:
            curve_bias = self._curve_gate_bias()
            plan = self._gen.next_gate(difficulty=self._difficulty, curve_bias=curve_bias)
            self._gates.append(
                _LiveGate(
                    gate_id=self._next_gate_id,
                    x_norm=AUDITORY_GATE_SPAWN_X_NORM,
                    y_norm=plan.y_norm,
                    color=plan.color,
                    shape=plan.shape,
                    aperture_norm=plan.aperture_norm,
                    scored=False,
                )
            )
            self._next_gate_id += 1
            self._next_gate_at_s = self._sim_elapsed_s + self._gen.jittered_interval(
                base_s=self._effective_gate_interval_s(),
                difficulty=self._difficulty,
            )
            self._bind_gate_directive_to_next_match()

        speed = max(0.15, float(self._cfg.gate_speed_norm_per_s))
        active: list[_LiveGate] = []
        for gate in self._gates:
            gate.x_norm -= speed * dt
            if not gate.scored and gate.x_norm <= AUDITORY_GATE_PLAYER_X_NORM:
                inside_aperture = abs(self._ball_y - gate.y_norm) <= gate.aperture_norm
                should_pass, expected = self._gate_action(gate)
                if should_pass:
                    is_correct = inside_aperture and (not self._outside_tube)
                    if is_correct:
                        self._gate_hits += 1
                    else:
                        self._gate_misses += 1
                else:
                    is_correct = (not inside_aperture) and (not self._outside_tube)
                    if is_correct:
                        self._gate_hits += 1
                    else:
                        self._gate_misses += 1
                        if inside_aperture:
                            self._forbidden_gate_hits += 1

                gate_action = "PASS" if should_pass else "AVOID"
                pilot_action = "PASS" if inside_aperture else "SKIP"
                self._record_event(
                    kind=AuditoryCapacityEventKind.GATE,
                    expected=expected,
                    response=f"{gate.color}/{gate.shape}/{pilot_action}/{gate_action}",
                    is_correct=is_correct,
                    score=1.0 if is_correct else 0.0,
                    response_time_s=None,
                    command_type="gate",
                    addressed_call_sign="ALL ASSIGNED",
                )
                gate.scored = True
                if (
                    self._active_gate_directive is not None
                    and self._active_gate_directive.target_gate_id == gate.gate_id
                ):
                    self._active_gate_directive = None

            if (not gate.scored) and gate.x_norm >= AUDITORY_GATE_RETIRE_X_NORM:
                active.append(gate)
        self._gates = active
        self._bind_gate_directive_to_next_match()

    def _gate_action(self, gate: _LiveGate) -> tuple[bool, str]:
        gate_color = str(gate.color).strip().upper()
        gate_shape = str(gate.shape).strip().upper()
        directive_state = self._active_gate_directive
        if directive_state is not None and directive_state.target_gate_id == gate.gate_id:
            action = str(directive_state.directive.action).upper()
            should_pass = action != "AVOID"
            expected = f"{action}:{gate_color}/{gate_shape}"
            return should_pass, expected
        forbidden_color = None if self._forbidden_gate_color is None else str(self._forbidden_gate_color)
        forbidden_shape = None if self._forbidden_gate_shape is None else str(self._forbidden_gate_shape)
        forbidden = (
            (forbidden_color is not None and gate_color == forbidden_color)
            or (forbidden_shape is not None and gate_shape == forbidden_shape)
        )
        if forbidden:
            expected = f"AVOID:{gate_color}/{gate_shape}"
            return False, expected
        expected = f"PASS:{gate_color}/{gate_shape}"
        return True, expected

    def _submit_state_response(self, *, color: str | None, number: int | None) -> bool:
        pending = self._pending_state_command
        if pending is not None:
            expected = self._pending_command_expected(pending)
            response_parts: list[str] = []
            if color is not None:
                response_parts.append(f"COLOUR:{color}")
            if number is not None:
                response_parts.append(f"NUMBER:{number}")
            response = "|".join(response_parts) if response_parts else "NO_INPUT"
            rt = max(0.0, self._sim_elapsed_s - pending.event.timestamp_s)
            is_correct = (
                (pending.expected_color is not None and color == pending.expected_color)
                or (pending.expected_number is not None and number == pending.expected_number)
            )
            self._record_event(
                kind=AuditoryCapacityEventKind.COMMAND,
                expected=expected,
                response=response,
                is_correct=is_correct,
                score=1.0 if is_correct else 0.0,
                response_time_s=rt,
                command_type=pending.event.command_type.value,
                addressed_call_sign=pending.event.addressed_call_sign,
            )
            if is_correct:
                self._correct_command_executions += 1
                if color is not None:
                    self._ball_color = color
                    self._ball_color_strength = 1.0
                if number is not None:
                    self._ball_number = number
            self._pending_state_command = None
            return True

        distractor = self._pending_distractor_state_command
        if distractor is not None:
            response_parts = []
            if color is not None:
                response_parts.append(f"COLOUR:{color}")
            if number is not None:
                response_parts.append(f"NUMBER:{number}")
            self._record_false_response(
                response="|".join(response_parts) if response_parts else "NO_INPUT",
                command_type=distractor.event.command_type.value,
                distractor=True,
                addressed_call_sign=distractor.event.addressed_call_sign,
            )
            self._pending_distractor_state_command = None
            return True

        response_parts = []
        if color is not None:
            response_parts.append(f"COLOUR:{color}")
        if number is not None:
            response_parts.append(f"NUMBER:{number}")
        self._record_false_response(
            response="|".join(response_parts) if response_parts else "NO_INPUT",
            command_type="state_input",
            distractor=False,
            addressed_call_sign=None,
        )
        return True

    def _record_false_response(
        self,
        *,
        response: str,
        command_type: str,
        distractor: bool,
        addressed_call_sign: str | None,
    ) -> None:
        self._false_alarms += 1
        if distractor:
            self._false_responses_to_distractors += 1
        expected = "IGNORE_DISTRACTOR" if distractor else "NO_ACTIVE_COMMAND"
        self._record_event(
            kind=AuditoryCapacityEventKind.FALSE_RESPONSE,
            expected=expected,
            response=response,
            is_correct=False,
            score=0.0,
            response_time_s=None,
            command_type=command_type,
            addressed_call_sign=addressed_call_sign,
        )

    def _record_event(
        self,
        *,
        kind: AuditoryCapacityEventKind,
        expected: str,
        response: str,
        is_correct: bool,
        score: float,
        response_time_s: float | None,
        command_type: str | None,
        addressed_call_sign: str | None,
    ) -> None:
        score_clamped = max(0.0, min(1.0, float(score)))
        evt = AuditoryCapacityEvent(
            phase=self._phase,
            kind=kind,
            expected=str(expected),
            response=str(response),
            is_correct=bool(is_correct),
            score=score_clamped,
            response_time_s=response_time_s,
            occurred_at_s=float(self._sim_elapsed_s),
            command_type=command_type,
            addressed_call_sign=addressed_call_sign,
        )
        self._events.append(evt)
        if is_correct:
            if (
                kind is AuditoryCapacityEventKind.COMMAND
                and command_type == AuditoryCapacityCommandType.CHANGE_COLOUR.value
            ):
                chosen_color = None
                for part in str(response).split("|"):
                    if part.startswith("COLOUR:"):
                        chosen_color = part.partition(":")[2]
                        break
                self._trigger_ball_visual_feedback(
                    color=chosen_color or self._BALL_VISUAL_SUCCESS_COLOR,
                    duration_s=self._BALL_VISUAL_COLOR_CHANGE_DURATION_S,
                )
            elif kind in (
                AuditoryCapacityEventKind.COMMAND,
                AuditoryCapacityEventKind.GATE,
                AuditoryCapacityEventKind.DIGIT_RECALL,
                AuditoryCapacityEventKind.TRIGGER,
            ):
                self._trigger_ball_visual_feedback(
                    color=self._BALL_VISUAL_SUCCESS_COLOR,
                    duration_s=self._BALL_VISUAL_SUCCESS_DURATION_S,
                )
        else:
            self._trigger_ball_visual_feedback(
                color=self._BALL_VISUAL_ERROR_COLOR,
                duration_s=self._BALL_VISUAL_ERROR_DURATION_S,
            )
        if self._phase is not Phase.SCORED:
            return
        self._scored_attempted += 1
        self._scored_max_score += 1.0
        self._scored_total_score += score_clamped
        if is_correct:
            self._scored_correct += 1

    def _refresh_phase_boundaries(self, now: float) -> None:
        if self._phase is Phase.PRACTICE:
            if now - self._phase_started_at_s >= self._practice_duration_s:
                self._phase = Phase.PRACTICE_DONE
                self._phase_started_at_s = now
                self._last_update_at_s = now
                self._accumulator_s = 0.0
            return
        if self._phase is Phase.SCORED:
            if now - self._phase_started_at_s >= self._run_duration_s:
                self._phase = Phase.RESULTS
                self._phase_started_at_s = now
                self._last_update_at_s = now
                self._accumulator_s = 0.0

    def _time_until(self, due_at: float) -> float | None:
        if not math.isfinite(float(due_at)):
            return None
        return max(0.0, float(due_at - self._sim_elapsed_s))

    def _segment_time_remaining_s(self) -> float:
        if not self._active_segments:
            return 0.0
        end_at = self._segment_started_at_s + self._segment_duration_s
        return max(0.0, float(end_at - self._sim_elapsed_s))

    def _build_payload(self) -> AuditoryCapacityPayload:
        sequence_display: str | None = None
        if self._active_recall is not None and self._sim_elapsed_s <= self._active_recall.show_until_s:
            sequence_display = self._active_recall.target_digits
        elif (
            self._recent_instruction is not None
            and self._recent_instruction.command_type is AuditoryCapacityCommandType.DIGIT_APPEND
            and self._sim_elapsed_s <= self._recent_instruction.expires_at_s
        ):
            sequence_display = "".join(ch for ch in str(self._recent_instruction.payload) if ch.isdigit())[:1]

        recall_target_length = None
        sequence_show_left = None
        sequence_resp_left = None
        recall_entry_open = False
        if self._active_recall is not None:
            recall_target_length = len(self._active_recall.target_digits)
            if self._sim_elapsed_s < self._active_recall.show_until_s:
                sequence_show_left = max(0.0, self._active_recall.show_until_s - self._sim_elapsed_s)
            else:
                recall_entry_open = self._sim_elapsed_s <= self._active_recall.event.expires_at_s
                sequence_resp_left = max(
                    0.0,
                    self._active_recall.event.expires_at_s - self._sim_elapsed_s,
                )

        briefing_active = self._sim_elapsed_s < self._briefing_duration_s
        envelope = self._distractor_envelope()
        noise_peak = 0.30 + (0.50 * self._difficulty)
        if self._segment_profile.enable_distractors:
            noise_level = clamp01(
                noise_peak * envelope * max(0.0, float(self._segment_profile.noise_level_scale))
            )
        else:
            noise_level = 0.0
        distortion_level = 0.0
        if self._cfg.enable_distortion:
            distortion_level = clamp01(
                (0.04 + (0.20 * self._difficulty))
                * envelope
                * max(0.0, float(self._segment_profile.distortion_level_scale))
            )
        if self._noise_level_override is not None:
            noise_level = float(self._noise_level_override)
        if self._distortion_level_override is not None:
            distortion_level = float(self._distortion_level_override)

        gates = tuple(
            AuditoryCapacityGate(
                gate_id=g.gate_id,
                x_norm=float(g.x_norm),
                y_norm=float(g.y_norm),
                color=g.color,
                shape=g.shape,
                aperture_norm=float(g.aperture_norm),
            )
            for g in self._gates
        )
        pending_color = (
            self._pending_state_command.expected_color
            if self._pending_state_command is not None
            else None
        )
        pending_number = (
            self._pending_state_command.expected_number
            if self._pending_state_command is not None
            else None
        )
        directive_state = self._active_gate_directive
        forbidden_gate_color = self._forbidden_gate_color
        forbidden_gate_shape = self._forbidden_gate_shape
        target_gate_id = None
        target_gate_action = None
        if directive_state is not None:
            target_gate_id = directive_state.target_gate_id
            target_gate_action = directive_state.directive.action
            if directive_state.directive.action == "AVOID":
                if directive_state.directive.match_kind == "COLOR":
                    forbidden_gate_color = directive_state.directive.match_value
                elif directive_state.directive.match_kind == "SHAPE":
                    forbidden_gate_shape = directive_state.directive.match_value
        active_instruction = self._recent_instruction
        command_time_left_s = (
            None
            if active_instruction is None
            else max(0.0, float(active_instruction.expires_at_s - self._sim_elapsed_s))
        )
        next_command_candidates: list[float] = []
        if self._next_instruction_index < len(self._instruction_script):
            next_command_candidates.append(
                max(
                    0.0,
                    float(self._instruction_script[self._next_instruction_index].timestamp_s - self._sim_elapsed_s),
                )
            )
        for due_at in (
            self._next_state_command_at_s,
            self._next_gate_directive_at_s,
            self._next_digit_sequence_at_s,
            self._next_beep_at_s,
        ):
            if due_at > self._sim_elapsed_s:
                remaining = self._time_until(due_at)
                if remaining is not None:
                    next_command_candidates.append(remaining)
        next_command_in_s = min(next_command_candidates) if next_command_candidates else None
        beep_active = (
            self._active_beep is not None
            and not self._active_beep.responded
            and self._sim_elapsed_s <= self._active_beep.active_until_s
        )
        next_beep_in_s = 0.0 if beep_active else self._time_until(self._next_beep_at_s)

        metrics = AuditoryCapacityMetrics(
            boundary_violations=int(self._collisions),
            gate_hits=int(self._gate_hits),
            gate_misses=int(self._gate_misses),
            forbidden_gate_hits=int(self._forbidden_gate_hits),
            correct_command_executions=int(self._correct_command_executions),
            false_responses_to_distractors=int(self._false_responses_to_distractors),
            missed_valid_commands=int(self._missed_valid_commands),
            digit_recall_attempts=int(self._digit_recall_attempts),
            digit_recall_accuracy=float(self._digit_recall_accuracy()),
        )
        ball_visual_color, ball_visual_strength = self._ball_visual_state()

        return AuditoryCapacityPayload(
            session_seed=int(self._seed),
            phase_elapsed_s=float(self._sim_elapsed_s),
            active_channels=tuple(self._segment_active_channels),
            segment_label=self._segment_label,
            segment_index=int(self._segment_index + 1),
            segment_total=max(1, len(self._active_segments)),
            segment_time_remaining_s=self._segment_time_remaining_s(),
            ball_x=float(self._ball_x),
            ball_y=float(self._ball_y),
            ball_contact_ratio=float(self._ball_contact_ratio),
            control_x=float(self._control_x),
            control_y=float(self._control_y),
            disturbance_x=float(self._dist_x),
            disturbance_y=float(self._dist_y),
            tube_half_width=float(self._effective_tube_half_width()),
            tube_half_height=float(self._effective_tube_half_height()),
            ball_color=self._ball_color,
            ball_color_strength=float(self._ball_color_strength),
            ball_visual_color=ball_visual_color,
            ball_visual_strength=float(ball_visual_strength),
            ball_number=int(self._ball_number),
            active_callsign="ALL ASSIGNED",
            assigned_callsigns=self._assigned_callsigns,
            active_instruction=active_instruction,
            instruction_text=self._recent_instruction_text,
            instruction_uid=None if active_instruction is None else int(active_instruction.event_id),
            instruction_command_type=(
                None if active_instruction is None else active_instruction.command_type.value
            ),
            briefing_active=briefing_active,
            callsign_cue=(
                None if active_instruction is None else active_instruction.addressed_call_sign
            ),
            callsign_blocks_gate=bool(target_gate_action == "AVOID"),
            beep_active=beep_active,
            color_command=pending_color,
            number_command=pending_number,
            sequence_display=sequence_display,
            sequence_response_open=recall_entry_open,
            recall_target_length=recall_target_length,
            digit_buffer_length=len(self._memory_digits),
            forbidden_gate_color=forbidden_gate_color,
            forbidden_gate_shape=forbidden_gate_shape,
            target_gate_id=target_gate_id,
            target_gate_action=target_gate_action,
            color_rules=(),
            gates=gates,
            metrics=metrics,
            gate_hits=int(self._gate_hits),
            gate_misses=int(self._gate_misses),
            forbidden_gate_hits=int(self._forbidden_gate_hits),
            collisions=int(self._collisions),
            false_alarms=int(self._false_alarms),
            correct_command_executions=int(self._correct_command_executions),
            false_responses_to_distractors=int(self._false_responses_to_distractors),
            missed_valid_commands=int(self._missed_valid_commands),
            digit_recall_attempts=int(self._digit_recall_attempts),
            digit_recall_accuracy=float(self._digit_recall_accuracy()),
            points=float(self._scored_total_score),
            background_noise_level=float(noise_level),
            distortion_level=float(distortion_level),
            background_noise_source=self._noise_source_label(),
            command_time_left_s=command_time_left_s,
            sequence_show_time_left_s=sequence_show_left,
            sequence_response_time_left_s=sequence_resp_left,
            next_command_in_s=next_command_in_s,
            next_gate_in_s=self._time_until(self._next_gate_at_s),
            next_callsign_in_s=next_command_in_s,
            next_beep_in_s=next_beep_in_s,
            next_color_in_s=self._time_until(self._next_state_command_at_s),
            next_sequence_in_s=self._time_until(self._next_digit_sequence_at_s),
        )

    def _gate_interval_s(self) -> float:
        if self._cfg.gate_spawn_rate > 0.0:
            return 1.0 / float(self._cfg.gate_spawn_rate)
        return max(0.25, float(self._cfg.gate_interval_s))

    def _curve_gate_bias(self) -> float:
        strength = max(0.0, float(self._cfg.tunnel_curvature_intensity))
        return 0.10 * strength * math.sin((self._sim_elapsed_s * 0.48) + 0.70)

    def _distractor_envelope(self) -> float:
        if self._phase_duration_s <= 0.0:
            return 0.0
        if self._sim_elapsed_s < self._briefing_duration_s:
            return 0.0
        active_window = max(1e-6, self._phase_duration_s - self._briefing_duration_s)
        t = clamp01((self._sim_elapsed_s - self._briefing_duration_s) / active_window)
        if t <= 0.70:
            return clamp01((t / 0.70) ** 1.85)
        fade_t = clamp01((t - 0.70) / 0.30)
        return clamp01(1.0 - (fade_t**0.55))

    def _digit_recall_accuracy(self) -> float:
        if self._digit_recall_attempts <= 0:
            return 0.0
        return self._digit_recall_score_total / float(self._digit_recall_attempts)

    @staticmethod
    def _canonical_color(raw: str | int | None) -> str | None:
        token = str(raw).strip().upper()
        if token in ("R", "RED"):
            return "RED"
        if token in ("W", "G", "GREEN"):
            return "GREEN"
        if token in ("Q", "B", "BLUE"):
            return "BLUE"
        if token in ("E", "Y", "YELLOW"):
            return "YELLOW"
        return None

    @staticmethod
    def _canonical_shape(raw: str | int | None) -> str | None:
        token = str(raw).strip().upper()
        if token in {"C", "CIRCLE"}:
            return "CIRCLE"
        if token in {"T", "TRIANGLE"}:
            return "TRIANGLE"
        if token in {"S", "SQUARE"}:
            return "SQUARE"
        return None

    @staticmethod
    def _canonical_number(raw: int | str | None) -> int | None:
        token = "".join(ch for ch in str(raw) if ch.isdigit())
        if token == "":
            return None
        value = int(token)
        if value < 0 or value > 9:
            return None
        return value

    @staticmethod
    def _pending_command_expected(pending: _PendingStateCommand) -> str:
        if pending.expected_color is not None:
            return f"COLOUR:{pending.expected_color}"
        if pending.expected_number is not None:
            return f"NUMBER:{pending.expected_number}"
        return "STATE"

    def _instruction_text(self, event: AuditoryCapacityInstructionEvent) -> str:
        cs = str(event.addressed_call_sign)
        token = event.command_type
        payload = event.payload
        if token is AuditoryCapacityCommandType.CHANGE_COLOUR:
            variants = (
                f"{cs}. Change colour to {payload}.",
                f"{cs}. Set colour {payload}.",
                f"{cs}. Switch to {payload}.",
            )
            return variants[
                self._stable_variant_index(
                    seed=self._seed,
                    salt="change_colour",
                    event_id=event.event_id,
                    count=len(variants),
                )
            ]
        if token is AuditoryCapacityCommandType.CHANGE_NUMBER:
            variants = (
                f"{cs}. Set number {payload}.",
                f"{cs}. Change number to {payload}.",
                f"{cs}. Number {payload}.",
            )
            return variants[
                self._stable_variant_index(
                    seed=self._seed,
                    salt="change_number",
                    event_id=event.event_id,
                    count=len(variants),
                )
            ]
        if token is AuditoryCapacityCommandType.GATE_DIRECTIVE:
            directive = payload if isinstance(payload, AuditoryCapacityGateDirective) else None
            if directive is None:
                return f"{cs}. Stand by."
            label = str(directive.match_value).lower()
            if directive.match_kind == "COLOR":
                label = f"{label} gate"
            else:
                label = f"the {label}"
            if directive.action == "AVOID":
                variants = (
                    f"{cs}. Next gate. Avoid {label}. Do not go through it.",
                    f"{cs}. Avoid the next {label}. Do not go through it.",
                    f"{cs}. Next matching gate. Avoid {label}.",
                )
                return variants[
                    self._stable_variant_index(
                        seed=self._seed,
                        salt="gate_avoid",
                        event_id=event.event_id,
                        count=len(variants),
                    )
                ]
            variants = (
                f"{cs}. Go through the next {label}.",
                f"{cs}. Next matching gate. Go through {label}.",
                f"{cs}. Take the next {label}.",
            )
            return variants[
                self._stable_variant_index(
                    seed=self._seed,
                    salt="gate_pass",
                    event_id=event.event_id,
                    count=len(variants),
                )
            ]
        if token is AuditoryCapacityCommandType.DIGIT_SEQUENCE:
            digits = " ".join(ch for ch in str(payload) if ch.isdigit())
            variants = (
                f"{cs}. Remember digits. {digits}.",
                f"{cs}. Hold these digits. {digits}.",
                f"{cs}. Digits to remember. {digits}.",
            )
            return variants[
                self._stable_variant_index(
                    seed=self._seed,
                    salt="digit_sequence",
                    event_id=event.event_id,
                    count=len(variants),
                )
            ]
        if token is AuditoryCapacityCommandType.PRESS_TRIGGER:
            variants = (
                f"{cs}. Beep cue. Press trigger or Space now.",
                f"{cs}. Beep. Press trigger or Space now.",
                f"{cs}. When the beep sounds, press trigger or Space.",
            )
            return variants[
                self._stable_variant_index(
                    seed=self._seed,
                    salt="press_trigger",
                    event_id=event.event_id,
                    count=len(variants),
                )
            ]
        if token is AuditoryCapacityCommandType.SET_FORBIDDEN_GATE_COLOUR:
            return f"{cs}. Avoid {payload} coloured gates."
        if token is AuditoryCapacityCommandType.SET_FORBIDDEN_GATE_SHAPE:
            return f"{cs}. Avoid {payload} gates."
        if token is AuditoryCapacityCommandType.DIGIT_APPEND:
            return f"{cs}. Digit {payload}."
        if token is AuditoryCapacityCommandType.RECALL_DIGITS:
            return f"{cs}. Recall digits."
        if token is AuditoryCapacityCommandType.CHANGE_CALLSIGN:
            return f"{cs}. Change call sign to {payload}."
        if token is AuditoryCapacityCommandType.NO_OP_DISTRACTOR and payload is not None:
            return str(payload)
        return f"{cs}. Stand by."

    def _noise_source_label(self) -> str | None:
        if self._noise_source_override is not None:
            return self._noise_source_override
        return None


def build_auditory_capacity_test(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    config: AuditoryCapacityConfig | None = None,
    practice_segments: tuple[AuditoryCapacityTrainingSegment, ...] = (),
    scored_segments: tuple[AuditoryCapacityTrainingSegment, ...] = (),
) -> AuditoryCapacityEngine:
    actual_seed = seed
    if config is not None and config.seed is not None:
        actual_seed = int(config.seed)
    return AuditoryCapacityEngine(
        clock=clock,
        seed=actual_seed,
        difficulty=difficulty,
        config=config,
        practice_segments=practice_segments,
        scored_segments=scored_segments,
    )
