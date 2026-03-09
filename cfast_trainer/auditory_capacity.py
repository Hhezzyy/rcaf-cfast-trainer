from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from .clock import Clock
from .cognitive_core import AttemptSummary, Phase, SeededRng, TestSnapshot, clamp01


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
    tube_half_width: float = 0.84
    tube_half_height: float = 0.50
    tunnel_curvature_intensity: float = 0.56

    gate_speed_norm_per_s: float = 0.30
    gate_spawn_rate: float = 0.095
    gate_interval_s: float = 11.5

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
    COLLISION = "collision"
    FALSE_RESPONSE = "false_response"


@dataclass(frozen=True, slots=True)
class AuditoryCapacityColorRule:
    color: str
    required_shape: str


@dataclass(frozen=True, slots=True)
class AuditoryCapacityInstructionEvent:
    event_id: int
    timestamp_s: float
    addressed_call_sign: str
    speaker_id: str
    command_type: AuditoryCapacityCommandType
    payload: str | int | None
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
    ball_number: int
    active_callsign: str

    assigned_callsigns: tuple[str, ...]
    active_instruction: AuditoryCapacityInstructionEvent | None
    instruction_text: str | None
    instruction_uid: int | None
    instruction_command_type: str | None

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

    def __init__(self, *, seed: int) -> None:
        self._rng = SeededRng(seed)

    def assign_callsigns(self, *, count: int) -> tuple[str, ...]:
        picked = self._rng.sample(self.CALLSIGNS, k=max(1, min(int(count), len(self.CALLSIGNS))))
        return tuple(str(v) for v in picked)

    def next_gate(self, *, difficulty: float, curve_bias: float = 0.0) -> AuditoryCapacityGatePlan:
        d = clamp01(difficulty)
        y_span = 0.50 - (0.11 * d)
        y = self._rng.uniform(-y_span, y_span) + float(curve_bias)
        y = max(-0.58, min(0.58, y))
        color = str(self._rng.choice(self.COLORS))
        shape = str(self._rng.choice(self.SHAPES))
        aperture = max(0.08, min(0.26, 0.23 - (0.10 * d) + self._rng.uniform(-0.02, 0.02)))
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

    def __init__(
        self,
        *,
        clock: Clock,
        seed: int,
        difficulty: float = 0.5,
        config: AuditoryCapacityConfig | None = None,
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

        self._gen = AuditoryCapacityScenarioGenerator(seed=self._seed)

        self._phase = Phase.INSTRUCTIONS
        self._phase_started_at_s = self._clock.now()
        self._last_update_at_s = self._phase_started_at_s
        self._accumulator_s = 0.0
        self._sim_elapsed_s = 0.0

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
        self._ball_number = 1
        self._forbidden_gate_color: str | None = None
        self._forbidden_gate_shape: str | None = None
        self._memory_digits = ""

        self._instruction_script: tuple[AuditoryCapacityInstructionEvent, ...] = ()
        self._next_instruction_index = 0
        self._recent_instruction: AuditoryCapacityInstructionEvent | None = None
        self._recent_instruction_text: str | None = None
        self._pending_state_command: _PendingStateCommand | None = None
        self._pending_distractor_state_command: _PendingStateCommand | None = None
        self._active_recall: _PendingRecall | None = None
        self._distractor_recall: _PendingRecall | None = None

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
        if command in {"CALL", "CALLSIGN", "CS", "BEEP", "SPACE", "TRIGGER"}:
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
                "Q/W/E/R change colour  keypad 1-9 sets number  "
                "digits+Enter submits recall  WASD/arrows or HOTAS fly the ball"
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
                    "Track the ball in the tunnel while following call-sign filtered instructions.",
                    f"- Your call signs for this run: {callsign_roster}",
                    "- Follow commands addressed to any of your assigned call signs",
                    "- Q/W/E/R changes colour",
                    "- Keypad digits change the ball number",
                    "- Digits plus Enter submits delayed recall",
                    "- Gate rules can forbid colours or shapes",
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
                "Press Enter to begin the timed scored block."
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
        return "Track, filter by call sign, maintain ball state, and recall digits under load."

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

    def _begin_runtime_phase(self, *, duration_s: float, reset_scores: bool) -> None:
        now = self._clock.now()
        self._phase_started_at_s = now
        self._last_update_at_s = now
        self._accumulator_s = 0.0
        self._sim_elapsed_s = 0.0

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
        self._ball_number = 1
        self._forbidden_gate_color = None
        self._forbidden_gate_shape = None
        self._memory_digits = ""
        self._outside_tube = False

        self._gates.clear()
        self._next_gate_at_s = max(
            0.80,
            min(18.0, self._gate_interval_s() * 0.70),
        )
        self._next_gate_id = 1

        self._instruction_script = self._gen.build_instruction_script(
            duration_s=duration_s,
            difficulty=self._difficulty,
            config=self._cfg,
            callsigns=self._assigned_callsigns,
            starting_callsign=self._assigned_callsigns[0],
        )
        self._next_instruction_index = 0
        self._recent_instruction = None
        self._recent_instruction_text = None
        self._pending_state_command = None
        self._pending_distractor_state_command = None
        self._active_recall = None
        self._distractor_recall = None

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

        self._update_disturbance()
        next_x = self._ball_x + (
            (self._control_x * self._cfg.control_gain) + (self._dist_x * self._cfg.disturbance_gain)
        ) * dt
        next_y = self._ball_y + (
            (self._control_y * self._cfg.control_gain) + (self._dist_y * self._cfg.disturbance_gain)
        ) * dt

        next_x = max(-1.15, min(1.15, next_x))
        next_y = max(-0.92, min(0.92, next_y))

        clamped_x, clamped_y, contact_ratio = project_inside_tube(
            x=next_x,
            y=next_y,
            tube_half_width=self._cfg.tube_half_width,
            tube_half_height=self._cfg.tube_half_height,
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
        self._ball_color_strength = max(0.0, self._ball_color_strength - (0.19 * dt))

    def _update_disturbance(self) -> None:
        if self._sim_elapsed_s < self._dist_until_s:
            return
        disturbance = self._gen.next_disturbance(difficulty=self._difficulty)
        drift_scale = self._drift_scale()
        curve_scale = 1.0 + (max(0.0, float(self._cfg.tunnel_curvature_intensity)) * 0.20)
        self._dist_x = disturbance.vx * drift_scale * curve_scale
        self._dist_y = disturbance.vy * drift_scale * curve_scale
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

    def _activate_instruction(self, event: AuditoryCapacityInstructionEvent) -> None:
        self._recent_instruction = event
        self._recent_instruction_text = self._instruction_text(event)
        should_follow = (
            not event.is_distractor and event.addressed_call_sign in self._assigned_callsigns
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
        if self._sim_elapsed_s >= self._next_gate_at_s:
            curve_bias = self._curve_gate_bias()
            plan = self._gen.next_gate(difficulty=self._difficulty, curve_bias=curve_bias)
            self._gates.append(
                _LiveGate(
                    gate_id=self._next_gate_id,
                    x_norm=1.20,
                    y_norm=plan.y_norm,
                    color=plan.color,
                    shape=plan.shape,
                    aperture_norm=plan.aperture_norm,
                    scored=False,
                )
            )
            self._next_gate_id += 1
            self._next_gate_at_s = self._sim_elapsed_s + self._gen.jittered_interval(
                base_s=self._gate_interval_s(),
                difficulty=self._difficulty,
            )

        speed = max(0.15, float(self._cfg.gate_speed_norm_per_s))
        active: list[_LiveGate] = []
        for gate in self._gates:
            gate.x_norm -= speed * dt
            if not gate.scored and gate.x_norm <= self._ball_x:
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

            if (not gate.scored) and gate.x_norm >= -1.25:
                active.append(gate)
        self._gates = active

    def _gate_action(self, gate: _LiveGate) -> tuple[bool, str]:
        gate_color = str(gate.color).strip().upper()
        gate_shape = str(gate.shape).strip().upper()
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
            command_type=command_type,
            addressed_call_sign=addressed_call_sign,
        )
        self._events.append(evt)
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

    def _build_payload(self) -> AuditoryCapacityPayload:
        sequence_display: str | None = None
        if (
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

        progress = self._phase_progress_ratio()
        ramp = progress * progress
        noise_floor = 0.08 + (0.12 * self._difficulty)
        noise_peak = 0.35 + (0.45 * self._difficulty)
        noise_level = clamp01(noise_floor + ((noise_peak - noise_floor) * ramp))
        distortion_level = 0.0
        if self._cfg.enable_distortion:
            distortion_level = clamp01((0.04 + (0.20 * self._difficulty)) * ramp)
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
        active_instruction = self._recent_instruction
        command_time_left_s = (
            None
            if active_instruction is None
            else max(0.0, float(active_instruction.expires_at_s - self._sim_elapsed_s))
        )
        next_command_in_s = None
        if self._next_instruction_index < len(self._instruction_script):
            next_command_in_s = max(
                0.0,
                float(self._instruction_script[self._next_instruction_index].timestamp_s - self._sim_elapsed_s),
            )

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

        return AuditoryCapacityPayload(
            ball_x=float(self._ball_x),
            ball_y=float(self._ball_y),
            ball_contact_ratio=float(self._ball_contact_ratio),
            control_x=float(self._control_x),
            control_y=float(self._control_y),
            disturbance_x=float(self._dist_x),
            disturbance_y=float(self._dist_y),
            tube_half_width=float(self._cfg.tube_half_width),
            tube_half_height=float(self._cfg.tube_half_height),
            ball_color=self._ball_color,
            ball_color_strength=float(self._ball_color_strength),
            ball_number=int(self._ball_number),
            active_callsign="ALL ASSIGNED",
            assigned_callsigns=self._assigned_callsigns,
            active_instruction=active_instruction,
            instruction_text=self._recent_instruction_text,
            instruction_uid=None if active_instruction is None else int(active_instruction.event_id),
            instruction_command_type=(
                None if active_instruction is None else active_instruction.command_type.value
            ),
            callsign_cue=(
                None if active_instruction is None else active_instruction.addressed_call_sign
            ),
            callsign_blocks_gate=False,
            beep_active=False,
            color_command=pending_color,
            number_command=pending_number,
            sequence_display=sequence_display,
            sequence_response_open=recall_entry_open,
            recall_target_length=recall_target_length,
            digit_buffer_length=len(self._memory_digits),
            forbidden_gate_color=self._forbidden_gate_color,
            forbidden_gate_shape=self._forbidden_gate_shape,
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
            next_gate_in_s=max(0.0, float(self._next_gate_at_s - self._sim_elapsed_s)),
            next_callsign_in_s=next_command_in_s,
            next_beep_in_s=None,
            next_color_in_s=next_command_in_s,
            next_sequence_in_s=next_command_in_s,
        )

    def _gate_interval_s(self) -> float:
        if self._cfg.gate_spawn_rate > 0.0:
            return 1.0 / float(self._cfg.gate_spawn_rate)
        return max(0.25, float(self._cfg.gate_interval_s))

    def _curve_gate_bias(self) -> float:
        strength = max(0.0, float(self._cfg.tunnel_curvature_intensity))
        return 0.10 * strength * math.sin((self._sim_elapsed_s * 0.48) + 0.70)

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
            return f"{cs}. Change colour to {payload}."
        if token is AuditoryCapacityCommandType.CHANGE_NUMBER:
            return f"{cs}. Set number {payload}."
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
        return f"{cs}. Stand by."

    def _noise_source_label(self) -> str | None:
        if self._noise_source_override is not None:
            return self._noise_source_override
        if self._cfg.enable_overlapping_voices:
            return "simulated_voice_chatter"
        return None


def build_auditory_capacity_test(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    config: AuditoryCapacityConfig | None = None,
) -> AuditoryCapacityEngine:
    actual_seed = seed
    if config is not None and config.seed is not None:
        actual_seed = int(config.seed)
    return AuditoryCapacityEngine(
        clock=clock,
        seed=actual_seed,
        difficulty=difficulty,
        config=config,
    )
