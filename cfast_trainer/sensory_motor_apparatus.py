from __future__ import annotations

import math
from dataclasses import dataclass

from .clock import Clock
from .cognitive_core import Phase, SeededRng, TestSnapshot, clamp01, lerp_int


@dataclass(frozen=True, slots=True)
class SensoryMotorApparatusConfig:
    # Candidate guide indicates ~9 minutes including instructions.
    practice_duration_s: float = 45.0
    scored_duration_s: float = 7.0 * 60.0
    tick_hz: float = 120.0
    control_gain: float = 1.25
    on_target_radius: float = 0.10
    good_window_error: float = 0.22
    field_limit: float = 1.0


@dataclass(frozen=True, slots=True)
class DisturbanceVector:
    vx: float
    vy: float
    duration_s: float


@dataclass(frozen=True, slots=True)
class SensoryMotorApparatusPayload:
    dot_x: float
    dot_y: float
    control_x: float
    control_y: float
    disturbance_x: float
    disturbance_y: float
    phase_elapsed_s: float
    mean_error: float
    rms_error: float
    on_target_s: float
    on_target_ratio: float


@dataclass(frozen=True, slots=True)
class SensoryMotorApparatusSummary:
    attempted: int
    correct: int
    accuracy: float
    duration_s: float
    throughput_per_min: float
    mean_error: float
    rms_error: float
    on_target_s: float
    on_target_ratio: float
    total_score: float
    max_score: float
    score_ratio: float


class SensoryMotorDisturbanceGenerator:
    """Deterministic disturbance stream for continuous psychomotor tracking."""

    def __init__(self, *, seed: int) -> None:
        self._rng = SeededRng(seed)

    def next_vector(self, *, difficulty: float) -> DisturbanceVector:
        d = clamp01(difficulty)
        mag_lo = lerp_int(28, 40, d) / 100.0
        mag_hi = lerp_int(46, 62, d) / 100.0
        interval_lo = max(0.32, 0.72 - (0.28 * d))
        interval_hi = max(interval_lo + 0.04, 1.28 - (0.30 * d))

        magnitude = self._rng.uniform(mag_lo, mag_hi)
        angle = self._rng.uniform(0.0, math.tau)
        duration_s = self._rng.uniform(interval_lo, interval_hi)

        return DisturbanceVector(
            vx=math.cos(angle) * magnitude,
            vy=math.sin(angle) * magnitude,
            duration_s=duration_s,
        )


def score_window(*, mean_error: float, good_window_error: float) -> float:
    """Score a one-second tracking window from average radial error.

    - mean_error <= threshold => 1.0
    - mean_error >= 2 * threshold => 0.0
    - linear in between
    """

    threshold = max(0.001, float(good_window_error))
    value = max(0.0, float(mean_error))
    if value <= threshold:
        return 1.0
    fail_at = threshold * 2.0
    if value >= fail_at:
        return 0.0
    return (fail_at - value) / threshold


class SensoryMotorApparatusEngine:
    """Continuous joystick/pedal tracking test with fixed-rate deterministic update."""

    def __init__(
        self,
        *,
        clock: Clock,
        seed: int,
        difficulty: float = 0.5,
        config: SensoryMotorApparatusConfig | None = None,
    ) -> None:
        self._clock = clock
        self._seed = int(seed)
        self._difficulty = clamp01(difficulty)
        self._config = config or SensoryMotorApparatusConfig()
        if self._config.tick_hz <= 0.0:
            raise ValueError("tick_hz must be > 0")
        if self._config.practice_duration_s < 0.0:
            raise ValueError("practice_duration_s must be >= 0")
        if self._config.scored_duration_s <= 0.0:
            raise ValueError("scored_duration_s must be > 0")

        self._phase = Phase.INSTRUCTIONS
        self._phase_started_at_s = self._clock.now()
        self._last_update_at_s = self._phase_started_at_s
        self._accumulator_s = 0.0
        self._sim_elapsed_s = 0.0

        self._disturbance_generator = SensoryMotorDisturbanceGenerator(seed=self._seed)
        self._disturbance_until_s = 0.0
        self._disturbance_x = 0.0
        self._disturbance_y = 0.0

        self._dot_x = 0.35
        self._dot_y = -0.24

        self._control_x = 0.0
        self._control_y = 0.0

        self._practice_samples = 0
        self._practice_sum_error = 0.0

        self._scored_samples = 0
        self._scored_sum_error = 0.0
        self._scored_sum_error2 = 0.0
        self._scored_on_target_s = 0.0

        self._window_elapsed_s = 0.0
        self._window_sum_error = 0.0
        self._window_samples = 0

        self._scored_attempted = 0
        self._scored_correct = 0
        self._scored_total_score = 0.0
        self._scored_max_score = 0.0

        self._tick_dt = 1.0 / float(self._config.tick_hz)

    @property
    def phase(self) -> Phase:
        return self._phase

    def can_exit(self) -> bool:
        return self._phase in (
            Phase.INSTRUCTIONS,
            Phase.PRACTICE,
            Phase.PRACTICE_DONE,
            Phase.RESULTS,
        )

    def start_practice(self) -> None:
        if self._phase is not Phase.INSTRUCTIONS:
            return
        if self._config.practice_duration_s <= 0.0:
            self._phase = Phase.PRACTICE_DONE
            self._phase_started_at_s = self._clock.now()
            return
        self._phase = Phase.PRACTICE
        self._reset_tracking_state()

    def start_scored(self) -> None:
        if self._phase in (Phase.SCORED, Phase.RESULTS):
            return
        self._phase = Phase.SCORED
        self._reset_tracking_state()

    def submit_answer(self, raw: str) -> bool:
        _ = raw
        # Continuous psychomotor task; no per-question submission.
        return False

    def set_control(self, *, horizontal: float, vertical: float) -> None:
        self._control_x = max(-1.0, min(1.0, float(horizontal)))
        self._control_y = max(-1.0, min(1.0, float(vertical)))

    def update(self) -> None:
        now = self._clock.now()
        dt = now - self._last_update_at_s
        self._last_update_at_s = now
        if dt <= 0.0:
            self._refresh_phase_boundaries(now)
            return
        dt = min(float(dt), 0.5)
        self._accumulator_s += dt

        while self._accumulator_s >= self._tick_dt:
            self._accumulator_s -= self._tick_dt
            if self._phase in (Phase.PRACTICE, Phase.SCORED):
                self._step(self._tick_dt)

        self._refresh_phase_boundaries(now)

    def time_remaining_s(self) -> float | None:
        now = self._clock.now()
        if self._phase is Phase.PRACTICE:
            rem = self._config.practice_duration_s - (now - self._phase_started_at_s)
            return max(0.0, rem)
        if self._phase is Phase.SCORED:
            rem = self._config.scored_duration_s - (now - self._phase_started_at_s)
            return max(0.0, rem)
        return None

    def snapshot(self) -> TestSnapshot:
        payload = SensoryMotorApparatusPayload(
            dot_x=float(self._dot_x),
            dot_y=float(self._dot_y),
            control_x=float(self._control_x),
            control_y=float(self._control_y),
            disturbance_x=float(self._disturbance_x),
            disturbance_y=float(self._disturbance_y),
            phase_elapsed_s=max(0.0, self._clock.now() - self._phase_started_at_s),
            mean_error=self._current_mean_error(),
            rms_error=self._current_rms_error(),
            on_target_s=float(self._scored_on_target_s),
            on_target_ratio=self._current_on_target_ratio(),
        )

        return TestSnapshot(
            title="Sensory Motor Apparatus",
            phase=self._phase,
            prompt=self.current_prompt(),
            input_hint=(
                "Use rudder for left/right and joystick axis 1 "
                "for up/down (WASD/arrow fallback)"
            ),
            time_remaining_s=self.time_remaining_s(),
            attempted_scored=self._scored_attempted,
            correct_scored=self._scored_correct,
            payload=payload,
            practice_feedback=None,
        )

    def current_prompt(self) -> str:
        if self._phase is Phase.INSTRUCTIONS:
            return (
                "Sensory Motor Apparatus\n"
                "Use joystick + foot pedal to keep the red dot as close as possible "
                "to the crosshair center.\n"
                "Press Enter to begin practice."
            )
        if self._phase is Phase.PRACTICE_DONE:
            return "Practice complete. Press Enter to start the timed test."
        if self._phase is Phase.RESULTS:
            summary = self.scored_summary()
            acc = int(round(summary.accuracy * 100))
            return (
                "Results\n"
                f"Windows: {summary.attempted}\n"
                f"Accurate: {summary.correct} ({acc}%)\n"
                f"Mean Error: {summary.mean_error:.3f}\n"
                f"On-Target Time: {summary.on_target_s:.1f}s "
                f"({summary.on_target_ratio * 100.0:.1f}%)"
            )
        return "Track the dot and keep it centered."

    def scored_summary(self) -> SensoryMotorApparatusSummary:
        attempted = int(self._scored_attempted)
        correct = int(self._scored_correct)
        duration_s = float(self._config.scored_duration_s)
        accuracy = 0.0 if attempted == 0 else correct / attempted
        throughput_per_min = (attempted / duration_s) * 60.0

        mean_error = (
            0.0 if self._scored_samples == 0 else self._scored_sum_error / self._scored_samples
        )
        rms_error = 0.0
        if self._scored_samples > 0:
            rms_error = math.sqrt(self._scored_sum_error2 / self._scored_samples)

        on_target_ratio = (
            0.0 if duration_s <= 0.0 else min(1.0, self._scored_on_target_s / duration_s)
        )

        max_score = float(self._scored_max_score)
        total_score = float(self._scored_total_score)
        score_ratio = 0.0 if max_score <= 0.0 else total_score / max_score

        return SensoryMotorApparatusSummary(
            attempted=attempted,
            correct=correct,
            accuracy=accuracy,
            duration_s=duration_s,
            throughput_per_min=throughput_per_min,
            mean_error=float(mean_error),
            rms_error=float(rms_error),
            on_target_s=float(self._scored_on_target_s),
            on_target_ratio=float(on_target_ratio),
            total_score=total_score,
            max_score=max_score,
            score_ratio=score_ratio,
        )

    def _reset_tracking_state(self) -> None:
        self._phase_started_at_s = self._clock.now()
        self._last_update_at_s = self._phase_started_at_s
        self._accumulator_s = 0.0
        self._sim_elapsed_s = 0.0

        # Deterministic re-seeding of initial dot offset per phase start.
        initial = self._disturbance_generator.next_vector(difficulty=self._difficulty)
        self._dot_x = max(-0.75, min(0.75, initial.vx))
        self._dot_y = max(-0.75, min(0.75, initial.vy))

        self._disturbance_x = 0.0
        self._disturbance_y = 0.0
        self._disturbance_until_s = 0.0

        if self._phase is Phase.PRACTICE:
            self._practice_samples = 0
            self._practice_sum_error = 0.0

        if self._phase is Phase.SCORED:
            self._scored_samples = 0
            self._scored_sum_error = 0.0
            self._scored_sum_error2 = 0.0
            self._scored_on_target_s = 0.0
            self._window_elapsed_s = 0.0
            self._window_sum_error = 0.0
            self._window_samples = 0
            self._scored_attempted = 0
            self._scored_correct = 0
            self._scored_total_score = 0.0
            self._scored_max_score = 0.0

    def _step(self, dt: float) -> None:
        self._sim_elapsed_s += dt

        if self._sim_elapsed_s >= self._disturbance_until_s:
            disturbance = self._disturbance_generator.next_vector(difficulty=self._difficulty)
            self._disturbance_x = disturbance.vx
            self._disturbance_y = disturbance.vy
            self._disturbance_until_s = self._sim_elapsed_s + disturbance.duration_s

        self._dot_x += ((self._control_x * self._config.control_gain) + self._disturbance_x) * dt
        self._dot_y += ((self._control_y * self._config.control_gain) + self._disturbance_y) * dt

        lim = float(self._config.field_limit)
        self._dot_x = max(-lim, min(lim, self._dot_x))
        self._dot_y = max(-lim, min(lim, self._dot_y))

        radial_error = math.sqrt((self._dot_x * self._dot_x) + (self._dot_y * self._dot_y))

        if self._phase is Phase.PRACTICE:
            self._practice_samples += 1
            self._practice_sum_error += radial_error
            return

        if self._phase is not Phase.SCORED:
            return

        self._scored_samples += 1
        self._scored_sum_error += radial_error
        self._scored_sum_error2 += radial_error * radial_error

        if radial_error <= self._config.on_target_radius:
            self._scored_on_target_s += dt

        self._window_elapsed_s += dt
        self._window_sum_error += radial_error
        self._window_samples += 1

        if self._window_elapsed_s >= 1.0:
            self._score_active_window()
            self._window_elapsed_s = max(0.0, self._window_elapsed_s - 1.0)

    def _refresh_phase_boundaries(self, now: float) -> None:
        if self._phase is Phase.PRACTICE:
            if now - self._phase_started_at_s >= self._config.practice_duration_s:
                self._phase = Phase.PRACTICE_DONE
                self._phase_started_at_s = now
                self._last_update_at_s = now
                self._accumulator_s = 0.0
            return

        if self._phase is Phase.SCORED:
            if now - self._phase_started_at_s >= self._config.scored_duration_s:
                if self._window_samples > 0:
                    self._score_active_window()
                    self._window_elapsed_s = 0.0
                self._phase = Phase.RESULTS
                self._phase_started_at_s = now
                self._last_update_at_s = now
                self._accumulator_s = 0.0

    def _score_active_window(self) -> None:
        mean_error = (
            0.0 if self._window_samples == 0 else self._window_sum_error / self._window_samples
        )
        window_score = score_window(
            mean_error=mean_error,
            good_window_error=self._config.good_window_error,
        )
        self._scored_attempted += 1
        self._scored_max_score += 1.0
        self._scored_total_score += window_score
        if window_score >= 1.0 - 1e-9:
            self._scored_correct += 1

        self._window_sum_error = 0.0
        self._window_samples = 0

    def _current_mean_error(self) -> float:
        if self._phase is Phase.PRACTICE:
            if self._practice_samples == 0:
                return 0.0
            return self._practice_sum_error / self._practice_samples
        if self._phase in (Phase.SCORED, Phase.RESULTS):
            if self._scored_samples == 0:
                return 0.0
            return self._scored_sum_error / self._scored_samples
        return 0.0

    def _current_rms_error(self) -> float:
        if self._phase in (Phase.SCORED, Phase.RESULTS):
            if self._scored_samples == 0:
                return 0.0
            return math.sqrt(self._scored_sum_error2 / self._scored_samples)
        return 0.0

    def _current_on_target_ratio(self) -> float:
        if self._phase in (Phase.SCORED, Phase.RESULTS):
            elapsed = self._clock.now() - self._phase_started_at_s
            if self._phase is Phase.RESULTS:
                elapsed = self._config.scored_duration_s
            if elapsed <= 0.0:
                return 0.0
            return max(0.0, min(1.0, self._scored_on_target_s / elapsed))
        return 0.0


def build_sensory_motor_apparatus_test(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    config: SensoryMotorApparatusConfig | None = None,
) -> SensoryMotorApparatusEngine:
    return SensoryMotorApparatusEngine(
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        config=config,
    )
