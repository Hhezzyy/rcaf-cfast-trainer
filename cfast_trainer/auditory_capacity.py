from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .clock import Clock
from .cognitive_core import AttemptSummary, Phase, SeededRng, TestSnapshot, clamp01, lerp_int


@dataclass(frozen=True, slots=True)
class AuditoryCapacityConfig:
    # Public guide wording describes this as a long, sustained multi-task block.
    practice_duration_s: float = 60.0
    scored_duration_s: float = 13.0 * 60.0
    tick_hz: float = 120.0

    control_gain: float = 1.24
    disturbance_gain: float = 0.64
    tube_half_width: float = 0.92
    tube_half_height: float = 0.56

    gate_speed_norm_per_s: float = 0.72
    gate_interval_s: float = 2.00

    callsign_interval_s: float = 1.95
    beep_interval_s: float = 1.60
    color_command_interval_s: float = 4.40
    sequence_interval_s: float = 8.20

    cue_window_s: float = 1.25
    sequence_display_s: float = 1.75
    sequence_response_s: float = 3.50


class AuditoryCapacityEventKind(StrEnum):
    CALLSIGN = "callsign"
    BEEP = "beep"
    COLOR = "color"
    SEQUENCE = "sequence"
    GATE = "gate"
    COLLISION = "collision"
    FALSE_ALARM = "false_alarm"


@dataclass(frozen=True, slots=True)
class AuditoryCapacityColorRule:
    color: str
    required_shape: str


@dataclass(frozen=True, slots=True)
class AuditoryCapacityGate:
    gate_id: int
    x_norm: float
    y_norm: float
    color: str
    shape: str
    aperture_norm: float


@dataclass(frozen=True, slots=True)
class AuditoryCapacityPayload:
    ball_x: float
    ball_y: float
    control_x: float
    control_y: float
    disturbance_x: float
    disturbance_y: float

    tube_half_width: float
    tube_half_height: float

    ball_color: str
    ball_color_strength: float

    assigned_callsigns: tuple[str, ...]
    callsign_cue: str | None
    beep_active: bool
    color_command: str | None

    sequence_display: str | None
    sequence_response_open: bool

    color_rules: tuple[AuditoryCapacityColorRule, ...]
    gates: tuple[AuditoryCapacityGate, ...]

    gate_hits: int
    gate_misses: int
    collisions: int
    false_alarms: int
    points: float

    background_noise_level: float
    distortion_level: float


@dataclass(frozen=True, slots=True)
class AuditoryCapacityEvent:
    phase: Phase
    kind: AuditoryCapacityEventKind
    expected: str
    response: str
    is_correct: bool
    score: float
    response_time_s: float | None


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
class _CueState:
    target: str
    issued_at_s: float
    expires_at_s: float
    expects_response: bool


@dataclass(slots=True)
class _SequenceState:
    target: str
    show_until_s: float
    expire_at_s: float


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

    def __init__(self, *, seed: int) -> None:
        self._rng = SeededRng(seed)

    def assign_callsigns(self) -> tuple[str, ...]:
        picked = self._rng.sample(self.CALLSIGNS, k=3)
        return tuple(str(v) for v in picked)

    def next_callsign_cue(self, *, difficulty: float) -> str:
        _ = difficulty
        return str(self._rng.choice(self.CALLSIGNS))

    def next_sequence(self, *, difficulty: float) -> str:
        d = clamp01(difficulty)
        n = lerp_int(4, 7, d)
        digits = [str(self._rng.randint(0, 9)) for _ in range(n)]
        return "".join(digits)

    def next_color_command(self, *, current_color: str, last_command: str | None) -> str:
        options = [c for c in self.COLORS if c != current_color and c != last_command]
        if not options:
            options = [c for c in self.COLORS if c != current_color]
        if not options:
            return str(current_color)
        return str(self._rng.choice(options))

    def next_gate(self, *, difficulty: float) -> AuditoryCapacityGatePlan:
        d = clamp01(difficulty)
        y_span = 0.50 - (0.11 * d)
        y = self._rng.uniform(-y_span, y_span)
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

        # Use sin/cos-free cardinal vectors for deterministic simplicity and lower CPU load.
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

    def build_color_rules(self) -> tuple[AuditoryCapacityColorRule, ...]:
        colors = list(self.COLORS)
        shapes = [str(v) for v in self.SHAPES]
        rules: list[AuditoryCapacityColorRule] = []
        for color in colors:
            shape = str(self._rng.choice(shapes))
            rules.append(AuditoryCapacityColorRule(color=color, required_shape=shape))
        return tuple(rules)


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


class AuditoryCapacityEngine:
    """Continuous multi-task engine for the Auditory Capacity Test.

    Channels run concurrently:
    - Psychomotor tracking: keep the ball inside the tube and pass gates.
    - Callsign monitoring: respond only to assigned callsigns.
    - Beep response: trigger when the beep cue appears.
    - Color command: change ball color when commanded.
    - Sequence memory: recall numeric sequences after brief display.
    """

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

        if cfg.tick_hz <= 0.0:
            raise ValueError("tick_hz must be > 0")
        if cfg.practice_duration_s < 0.0:
            raise ValueError("practice_duration_s must be >= 0")
        if cfg.scored_duration_s <= 0.0:
            raise ValueError("scored_duration_s must be > 0")
        if cfg.cue_window_s <= 0.0:
            raise ValueError("cue_window_s must be > 0")
        if cfg.sequence_display_s <= 0.0:
            raise ValueError("sequence_display_s must be > 0")
        if cfg.sequence_response_s <= 0.0:
            raise ValueError("sequence_response_s must be > 0")

        self._clock = clock
        self._seed = int(seed)
        self._difficulty = d
        self._cfg = cfg
        self._tick_dt = 1.0 / float(cfg.tick_hz)

        self._gen = AuditoryCapacityScenarioGenerator(seed=self._seed)

        self._phase = Phase.INSTRUCTIONS
        self._phase_started_at_s = self._clock.now()
        self._last_update_at_s = self._phase_started_at_s
        self._accumulator_s = 0.0
        self._sim_elapsed_s = 0.0

        self._ball_x = 0.00
        self._ball_y = 0.00
        self._control_x = 0.0
        self._control_y = 0.0

        self._dist_x = 0.0
        self._dist_y = 0.0
        self._dist_until_s = 0.0

        self._gates: list[_LiveGate] = []
        self._next_gate_id = 1
        self._next_gate_at_s = 0.0

        self._assigned_callsigns: tuple[str, ...] = ()
        self._rules: tuple[AuditoryCapacityColorRule, ...] = ()

        self._ball_color = "RED"
        self._ball_color_strength = 1.0
        self._last_color_command: str | None = None

        self._next_callsign_at_s = 0.0
        self._active_callsign: _CueState | None = None

        self._next_beep_at_s = 0.0
        self._active_beep: _CueState | None = None

        self._next_color_at_s = 0.0
        self._active_color: _CueState | None = None

        self._next_sequence_at_s = 0.0
        self._active_sequence: _SequenceState | None = None

        self._outside_tube = False

        self._gate_hits = 0
        self._gate_misses = 0
        self._collisions = 0
        self._false_alarms = 0

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

    def start_practice(self) -> None:
        if self._phase is not Phase.INSTRUCTIONS:
            return
        if self._cfg.practice_duration_s <= 0.0:
            self._phase = Phase.PRACTICE_DONE
            self._phase_started_at_s = self._clock.now()
            return
        self._phase = Phase.PRACTICE
        self._begin_runtime_phase(reset_scores=False)

    def start_scored(self) -> None:
        if self._phase is not Phase.PRACTICE_DONE:
            return
        self._phase = Phase.SCORED
        self._begin_runtime_phase(reset_scores=True)

    def submit_answer(self, raw: str) -> bool:
        if self._phase not in (Phase.PRACTICE, Phase.SCORED):
            return False

        command = str(raw).strip().upper()
        if command == "":
            return False

        if command in ("CALL", "CALLSIGN", "CS"):
            return self._submit_callsign_response()
        if command in ("BEEP", "TRIGGER", "SPACE"):
            return self._submit_beep_response()
        if command.startswith("COL:"):
            color = self._canonical_color(command[4:])
            if color is None:
                return False
            return self._submit_color_response(color)
        if command.startswith("SEQ:"):
            return self._submit_sequence_response(command[4:])
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
            rem = self._cfg.practice_duration_s - (now - self._phase_started_at_s)
            return max(0.0, rem)
        if self._phase is Phase.SCORED:
            rem = self._cfg.scored_duration_s - (now - self._phase_started_at_s)
            return max(0.0, rem)
        return None

    def snapshot(self) -> TestSnapshot:
        payload = self._build_payload() if self._phase in (Phase.PRACTICE, Phase.SCORED) else None
        return TestSnapshot(
            title="Auditory Capacity",
            phase=self._phase,
            prompt=self.current_prompt(),
            input_hint="C=callsign  SPACE=beep  Q/W/E/R=color  digits+Enter=sequence",
            time_remaining_s=self.time_remaining_s(),
            attempted_scored=self._scored_attempted,
            correct_scored=self._scored_correct,
            payload=payload,
        )

    def current_prompt(self) -> str:
        if self._phase is Phase.INSTRUCTIONS:
            return "\n".join(
                [
                    "Auditory Capacity Test",
                    "",
                    "Track the ball in the tube while handling concurrent audio tasks:",
                    "- Respond only to your assigned call signs (C)",
                    "- Trigger when a beep appears (Space)",
                    "- Change ball color when commanded (Q/W/E/R)",
                    "- Memorize and enter digit sequences",
                    "- Fly through gates matching your current color and shape rule",
                    "",
                    "Once scored begins, exit is locked until completion.",
                    "Press Enter to start practice.",
                ]
            )

        if self._phase is Phase.PRACTICE_DONE:
            return "Practice complete. Press Enter to begin the timed scored block."

        if self._phase is Phase.RESULTS:
            s = self.scored_summary()
            mean_rt = (
                "—"
                if s.mean_response_time_s is None
                else f"{s.mean_response_time_s*1000.0:.0f} ms"
            )
            return "\n".join(
                [
                    "Results",
                    "",
                    f"Attempted: {s.attempted}",
                    f"Correct:   {s.correct}",
                    f"Accuracy:  {s.accuracy*100.0:.1f}%",
                    f"Rate:      {s.throughput_per_min:.1f} / min",
                    f"Mean RT:   {mean_rt}",
                    f"Gate hits: {self._gate_hits}  misses: {self._gate_misses}",
                    f"Collisions:{self._collisions}  False alarms: {self._false_alarms}",
                    f"Score:     {s.total_score:.1f}/{s.max_score:.1f}",
                    "",
                    "Press Enter to return.",
                ]
            )

        return "Manage all channels continuously until time expires."

    def scored_summary(self) -> AttemptSummary:
        duration = float(self._cfg.scored_duration_s)
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

    def _begin_runtime_phase(self, *, reset_scores: bool) -> None:
        now = self._clock.now()
        self._phase_started_at_s = now
        self._last_update_at_s = now
        self._accumulator_s = 0.0
        self._sim_elapsed_s = 0.0

        self._assigned_callsigns = self._gen.assign_callsigns()
        self._rules = self._gen.build_color_rules()

        self._ball_x = 0.0
        self._ball_y = 0.0
        self._control_x = 0.0
        self._control_y = 0.0

        self._dist_x = 0.0
        self._dist_y = 0.0
        self._dist_until_s = 0.0

        self._gates.clear()
        self._next_gate_at_s = 0.85

        self._active_callsign = None
        self._active_beep = None
        self._active_color = None
        self._active_sequence = None

        self._next_callsign_at_s = 1.10
        self._next_beep_at_s = 0.95
        self._next_color_at_s = 2.15
        self._next_sequence_at_s = 3.10

        self._ball_color = "RED"
        self._ball_color_strength = 1.0
        self._last_color_command = None

        self._outside_tube = False

        self._gate_hits = 0
        self._gate_misses = 0
        self._collisions = 0
        self._false_alarms = 0

        if reset_scores:
            self._scored_attempted = 0
            self._scored_correct = 0
            self._scored_total_score = 0.0
            self._scored_max_score = 0.0

    def _step(self, dt: float) -> None:
        self._sim_elapsed_s += dt

        self._update_disturbance()

        self._ball_x += (
            (self._control_x * self._cfg.control_gain) + (self._dist_x * self._cfg.disturbance_gain)
        ) * dt
        self._ball_y += (
            (self._control_y * self._cfg.control_gain) + (self._dist_y * self._cfg.disturbance_gain)
        ) * dt

        self._ball_x = max(-1.15, min(1.15, self._ball_x))
        self._ball_y = max(-0.92, min(0.92, self._ball_y))

        outside_now = (
            abs(self._ball_x) > float(self._cfg.tube_half_width)
            or abs(self._ball_y) > float(self._cfg.tube_half_height)
        )
        if outside_now and not self._outside_tube:
            self._outside_tube = True
            self._collisions += 1
            self._record_event(
                kind=AuditoryCapacityEventKind.COLLISION,
                expected="inside tube",
                response="collision",
                is_correct=False,
                score=0.0,
                response_time_s=None,
            )
        elif not outside_now and self._outside_tube:
            self._outside_tube = False

        self._update_gates(dt)
        self._update_callsign_channel()
        self._update_beep_channel()
        self._update_color_channel()
        self._update_sequence_channel()

        self._ball_color_strength = max(0.0, self._ball_color_strength - (0.19 * dt))

    def _update_disturbance(self) -> None:
        if self._sim_elapsed_s < self._dist_until_s:
            return
        disturbance = self._gen.next_disturbance(difficulty=self._difficulty)
        self._dist_x = disturbance.vx
        self._dist_y = disturbance.vy
        self._dist_until_s = self._sim_elapsed_s + disturbance.duration_s

    def _update_gates(self, dt: float) -> None:
        if self._sim_elapsed_s >= self._next_gate_at_s:
            plan = self._gen.next_gate(difficulty=self._difficulty)
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
                base_s=self._cfg.gate_interval_s,
                difficulty=self._difficulty,
            )

        speed = max(0.15, float(self._cfg.gate_speed_norm_per_s))
        active: list[_LiveGate] = []
        for gate in self._gates:
            gate.x_norm -= speed * dt

            if not gate.scored and gate.x_norm <= self._ball_x:
                expected_shape = self._shape_for_color(self._ball_color)
                inside_aperture = abs(self._ball_y - gate.y_norm) <= gate.aperture_norm
                color_ok = gate.color == self._ball_color
                shape_ok = gate.shape == expected_shape
                is_correct = inside_aperture and color_ok and shape_ok and (not self._outside_tube)

                if is_correct:
                    self._gate_hits += 1
                else:
                    self._gate_misses += 1

                self._record_event(
                    kind=AuditoryCapacityEventKind.GATE,
                    expected=f"{self._ball_color}/{expected_shape}",
                    response=f"{gate.color}/{gate.shape}/{('PASS' if inside_aperture else 'MISS')}",
                    is_correct=is_correct,
                    score=1.0 if is_correct else 0.0,
                    response_time_s=None,
                )
                gate.scored = True

            if gate.x_norm >= -1.25:
                active.append(gate)

        self._gates = active

    def _update_callsign_channel(self) -> None:
        if self._sim_elapsed_s >= self._next_callsign_at_s and self._active_callsign is None:
            cue = self._gen.next_callsign_cue(difficulty=self._difficulty)
            expects = cue in self._assigned_callsigns
            issued = self._sim_elapsed_s
            self._active_callsign = _CueState(
                target=cue,
                issued_at_s=issued,
                expires_at_s=issued + self._cfg.cue_window_s,
                expects_response=expects,
            )
            self._next_callsign_at_s = self._sim_elapsed_s + self._gen.jittered_interval(
                base_s=self._cfg.callsign_interval_s,
                difficulty=self._difficulty,
            )

        active = self._active_callsign
        if active is None:
            return
        if self._sim_elapsed_s < active.expires_at_s:
            return

        # Expired with no response.
        if active.expects_response:
            self._record_event(
                kind=AuditoryCapacityEventKind.CALLSIGN,
                expected=active.target,
                response="MISS",
                is_correct=False,
                score=0.0,
                response_time_s=None,
            )
        else:
            self._record_event(
                kind=AuditoryCapacityEventKind.CALLSIGN,
                expected=f"IGNORE:{active.target}",
                response="NO_RESPONSE",
                is_correct=True,
                score=1.0,
                response_time_s=None,
            )

        self._active_callsign = None

    def _update_beep_channel(self) -> None:
        if self._sim_elapsed_s >= self._next_beep_at_s and self._active_beep is None:
            issued = self._sim_elapsed_s
            self._active_beep = _CueState(
                target="BEEP",
                issued_at_s=issued,
                expires_at_s=issued + self._cfg.cue_window_s,
                expects_response=True,
            )
            self._next_beep_at_s = self._sim_elapsed_s + self._gen.jittered_interval(
                base_s=self._cfg.beep_interval_s,
                difficulty=self._difficulty,
            )

        active = self._active_beep
        if active is None:
            return
        if self._sim_elapsed_s < active.expires_at_s:
            return

        self._record_event(
            kind=AuditoryCapacityEventKind.BEEP,
            expected="BEEP",
            response="MISS",
            is_correct=False,
            score=0.0,
            response_time_s=None,
        )
        self._active_beep = None

    def _update_color_channel(self) -> None:
        if self._sim_elapsed_s >= self._next_color_at_s and self._active_color is None:
            target = self._gen.next_color_command(
                current_color=self._ball_color,
                last_command=self._last_color_command,
            )
            issued = self._sim_elapsed_s
            self._active_color = _CueState(
                target=target,
                issued_at_s=issued,
                expires_at_s=issued + self._cfg.cue_window_s,
                expects_response=True,
            )
            self._last_color_command = target
            self._next_color_at_s = self._sim_elapsed_s + self._gen.jittered_interval(
                base_s=self._cfg.color_command_interval_s,
                difficulty=self._difficulty,
            )

        active = self._active_color
        if active is None:
            return
        if self._sim_elapsed_s < active.expires_at_s:
            return

        self._record_event(
            kind=AuditoryCapacityEventKind.COLOR,
            expected=active.target,
            response="MISS",
            is_correct=False,
            score=0.0,
            response_time_s=None,
        )
        self._active_color = None

    def _update_sequence_channel(self) -> None:
        if self._sim_elapsed_s >= self._next_sequence_at_s and self._active_sequence is None:
            target = self._gen.next_sequence(difficulty=self._difficulty)
            shown_until = self._sim_elapsed_s + self._cfg.sequence_display_s
            expire = shown_until + self._cfg.sequence_response_s
            self._active_sequence = _SequenceState(
                target=target,
                show_until_s=shown_until,
                expire_at_s=expire,
            )
            self._next_sequence_at_s = self._sim_elapsed_s + self._gen.jittered_interval(
                base_s=self._cfg.sequence_interval_s,
                difficulty=self._difficulty,
            )

        active = self._active_sequence
        if active is None:
            return
        if self._sim_elapsed_s < active.expire_at_s:
            return

        self._record_event(
            kind=AuditoryCapacityEventKind.SEQUENCE,
            expected=active.target,
            response="MISS",
            is_correct=False,
            score=0.0,
            response_time_s=None,
        )
        self._active_sequence = None

    def _submit_callsign_response(self) -> bool:
        active = self._active_callsign
        if active is None:
            self._record_false_alarm(channel="CALL")
            return True

        rt = max(0.0, self._sim_elapsed_s - active.issued_at_s)
        if active.expects_response:
            self._record_event(
                kind=AuditoryCapacityEventKind.CALLSIGN,
                expected=active.target,
                response=active.target,
                is_correct=True,
                score=1.0,
                response_time_s=rt,
            )
        else:
            self._record_event(
                kind=AuditoryCapacityEventKind.CALLSIGN,
                expected=f"IGNORE:{active.target}",
                response="FALSE_TRIGGER",
                is_correct=False,
                score=0.0,
                response_time_s=rt,
            )
        self._active_callsign = None
        return True

    def _submit_beep_response(self) -> bool:
        active = self._active_beep
        if active is None:
            self._record_false_alarm(channel="BEEP")
            return True

        rt = max(0.0, self._sim_elapsed_s - active.issued_at_s)
        self._record_event(
            kind=AuditoryCapacityEventKind.BEEP,
            expected="BEEP",
            response="TRIGGER",
            is_correct=True,
            score=1.0,
            response_time_s=rt,
        )
        self._active_beep = None
        return True

    def _submit_color_response(self, color: str) -> bool:
        active = self._active_color
        if active is None:
            self._record_false_alarm(channel=f"COL:{color}")
            return True

        rt = max(0.0, self._sim_elapsed_s - active.issued_at_s)
        is_correct = color == active.target and color != self._ball_color
        self._record_event(
            kind=AuditoryCapacityEventKind.COLOR,
            expected=active.target,
            response=color,
            is_correct=is_correct,
            score=1.0 if is_correct else 0.0,
            response_time_s=rt,
        )
        if is_correct:
            self._ball_color = color
            self._ball_color_strength = 1.0
        self._active_color = None
        return True

    def _submit_sequence_response(self, response_raw: str) -> bool:
        active = self._active_sequence
        if active is None:
            self._record_false_alarm(channel="SEQ")
            return True

        if self._sim_elapsed_s < active.show_until_s:
            return False

        response = "".join(ch for ch in str(response_raw) if ch.isdigit())
        if response == "":
            return False

        score = score_sequence_answer(active.target, response)
        is_correct = score >= 1.0 - 1e-9
        rt = max(0.0, self._sim_elapsed_s - active.show_until_s)
        self._record_event(
            kind=AuditoryCapacityEventKind.SEQUENCE,
            expected=active.target,
            response=response,
            is_correct=is_correct,
            score=score,
            response_time_s=rt,
        )
        self._active_sequence = None
        return True

    def _record_false_alarm(self, *, channel: str) -> None:
        self._false_alarms += 1
        self._record_event(
            kind=AuditoryCapacityEventKind.FALSE_ALARM,
            expected="NO_CUE",
            response=channel,
            is_correct=False,
            score=0.0,
            response_time_s=None,
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
            if now - self._phase_started_at_s >= self._cfg.practice_duration_s:
                self._phase = Phase.PRACTICE_DONE
                self._phase_started_at_s = now
                self._last_update_at_s = now
                self._accumulator_s = 0.0
            return

        if self._phase is Phase.SCORED:
            if now - self._phase_started_at_s >= self._cfg.scored_duration_s:
                self._phase = Phase.RESULTS
                self._phase_started_at_s = now
                self._last_update_at_s = now
                self._accumulator_s = 0.0

    def _shape_for_color(self, color: str) -> str:
        for rule in self._rules:
            if rule.color == color:
                return rule.required_shape
        return "CIRCLE"

    def _build_payload(self) -> AuditoryCapacityPayload:
        sequence_display: str | None = None
        sequence_open = False
        if self._active_sequence is not None:
            if self._sim_elapsed_s < self._active_sequence.show_until_s:
                sequence_display = self._active_sequence.target
            else:
                sequence_open = self._sim_elapsed_s <= self._active_sequence.expire_at_s

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

        return AuditoryCapacityPayload(
            ball_x=float(self._ball_x),
            ball_y=float(self._ball_y),
            control_x=float(self._control_x),
            control_y=float(self._control_y),
            disturbance_x=float(self._dist_x),
            disturbance_y=float(self._dist_y),
            tube_half_width=float(self._cfg.tube_half_width),
            tube_half_height=float(self._cfg.tube_half_height),
            ball_color=self._ball_color,
            ball_color_strength=float(self._ball_color_strength),
            assigned_callsigns=self._assigned_callsigns,
            callsign_cue=None if self._active_callsign is None else self._active_callsign.target,
            beep_active=self._active_beep is not None,
            color_command=None if self._active_color is None else self._active_color.target,
            sequence_display=sequence_display,
            sequence_response_open=sequence_open,
            color_rules=self._rules,
            gates=gates,
            gate_hits=int(self._gate_hits),
            gate_misses=int(self._gate_misses),
            collisions=int(self._collisions),
            false_alarms=int(self._false_alarms),
            points=float(self._scored_total_score),
            background_noise_level=float(0.35 + (0.45 * self._difficulty)),
            distortion_level=float(0.20 + (0.60 * self._difficulty)),
        )

    @staticmethod
    def _canonical_color(raw: str) -> str | None:
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


def build_auditory_capacity_test(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    config: AuditoryCapacityConfig | None = None,
) -> AuditoryCapacityEngine:
    return AuditoryCapacityEngine(
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        config=config,
    )
