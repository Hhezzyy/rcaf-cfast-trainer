from __future__ import annotations

import math
from dataclasses import dataclass

from .clock import Clock
from .cognitive_core import Phase, SeededRng, TestSnapshot, clamp01, lerp_int


@dataclass(frozen=True, slots=True)
class RapidTrackingConfig:
    # Candidate guide (page 12) describes ~16 minutes including instructions.
    practice_duration_s: float = 60.0
    scored_duration_s: float = 14.0 * 60.0
    tick_hz: float = 120.0

    # FPV camera/look behavior.
    camera_limit: float = 4.5
    camera_control_speed: float = 1.18
    camera_control_response_hz: float = 6.8

    # Disturbance drift while tracking.
    view_limit: float = 1.0

    # HUD behavior: crosshair and target box appear after sustained on-target tracking.
    hud_acquire_s: float = 1.35
    hud_persist_s: float = 2.20

    # Guide-style scoring.
    on_target_radius: float = 0.10
    good_window_error: float = 0.19

    # Corner preview card when targets switch.
    target_preview_s: float = 2.25


@dataclass(frozen=True, slots=True)
class RapidTrackingDriftVector:
    vx: float
    vy: float
    duration_s: float


@dataclass(frozen=True, slots=True)
class RapidTrackingSceneSegment:
    kind: str  # "tank" | "plane"
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    duration_s: float
    arc_x: float = 0.0
    arc_y: float = 0.0


@dataclass(frozen=True, slots=True)
class RapidTrackingPayload:
    target_rel_x: float
    target_rel_y: float
    reticle_x: float
    reticle_y: float
    camera_x: float
    camera_y: float
    target_vx: float
    target_vy: float
    target_visible: bool
    target_is_moving: bool
    target_kind: str
    target_time_to_switch_s: float
    target_switch_preview_s: float
    obscured_time_left_s: float
    phase_elapsed_s: float
    mean_error: float
    rms_error: float
    on_target_s: float
    on_target_ratio: float
    obscured_tracking_ratio: float
    hud_visible: bool
    lock_progress: float
    terrain_ridge_y: float


@dataclass(frozen=True, slots=True)
class RapidTrackingSummary:
    attempted: int
    correct: int
    accuracy: float
    duration_s: float
    throughput_per_min: float
    mean_error: float
    rms_error: float
    on_target_s: float
    on_target_ratio: float
    obscured_time_s: float
    obscured_tracking_ratio: float
    moving_target_ratio: float
    total_score: float
    max_score: float
    score_ratio: float


class RapidTrackingDriftGenerator:
    """Deterministic drift profile for continuous moving-view tracking."""

    def __init__(self, *, seed: int) -> None:
        self._rng = SeededRng(seed)

    def next_vector(self, *, difficulty: float) -> RapidTrackingDriftVector:
        d = clamp01(difficulty)
        mag_lo = lerp_int(8, 14, d) / 100.0
        mag_hi = lerp_int(18, 30, d) / 100.0
        interval_lo = max(0.28, 0.86 - (0.32 * d))
        interval_hi = max(interval_lo + 0.06, 1.52 - (0.54 * d))

        magnitude = self._rng.uniform(mag_lo, mag_hi)
        angle = self._rng.uniform(0.0, math.tau)
        duration_s = self._rng.uniform(interval_lo, interval_hi)

        return RapidTrackingDriftVector(
            vx=math.cos(angle) * magnitude,
            vy=math.sin(angle) * magnitude,
            duration_s=duration_s,
        )


def score_window(*, mean_error: float, good_window_error: float) -> float:
    threshold = max(0.001, float(good_window_error))
    value = max(0.0, float(mean_error))
    if value <= threshold:
        return 1.0
    fail_at = threshold * 2.0
    if value >= fail_at:
        return 0.0
    return (fail_at - value) / threshold


class RapidTrackingEngine:
    """Continuous eye-hand tracking with scripted helicopter-style pursuit scene."""

    _MAX_UPDATE_DT_S = 0.50

    _SCENE_SCRIPT: tuple[RapidTrackingSceneSegment, ...] = (
        RapidTrackingSceneSegment(
            kind="tank",
            start_x=-2.65,
            start_y=0.46,
            end_x=1.95,
            end_y=0.62,
            duration_s=9.4,
            arc_x=0.18,
            arc_y=0.20,
        ),
        RapidTrackingSceneSegment(
            kind="plane",
            start_x=1.60,
            start_y=-0.32,
            end_x=-2.10,
            end_y=-0.52,
            duration_s=7.8,
            arc_x=-0.26,
            arc_y=-0.12,
        ),
        RapidTrackingSceneSegment(
            kind="tank",
            start_x=-1.95,
            start_y=0.52,
            end_x=2.35,
            end_y=0.82,
            duration_s=9.0,
            arc_x=0.14,
            arc_y=0.26,
        ),
        RapidTrackingSceneSegment(
            kind="plane",
            start_x=2.25,
            start_y=-0.66,
            end_x=-1.85,
            end_y=-0.30,
            duration_s=8.4,
            arc_x=0.08,
            arc_y=-0.30,
        ),
        RapidTrackingSceneSegment(
            kind="tank",
            start_x=-2.25,
            start_y=0.35,
            end_x=1.70,
            end_y=0.74,
            duration_s=9.2,
            arc_x=0.22,
            arc_y=0.30,
        ),
        RapidTrackingSceneSegment(
            kind="plane",
            start_x=1.85,
            start_y=-0.18,
            end_x=-2.45,
            end_y=-0.50,
            duration_s=8.0,
            arc_x=-0.22,
            arc_y=-0.18,
        ),
    )

    def __init__(
        self,
        *,
        clock: Clock,
        seed: int,
        difficulty: float = 0.5,
        config: RapidTrackingConfig | None = None,
    ) -> None:
        cfg = config or RapidTrackingConfig()
        if cfg.tick_hz <= 0.0:
            raise ValueError("tick_hz must be > 0")
        if cfg.practice_duration_s < 0.0:
            raise ValueError("practice_duration_s must be >= 0")
        if cfg.scored_duration_s <= 0.0:
            raise ValueError("scored_duration_s must be > 0")
        if cfg.camera_control_response_hz <= 0.0:
            raise ValueError("camera_control_response_hz must be > 0")
        if cfg.hud_acquire_s <= 0.0:
            raise ValueError("hud_acquire_s must be > 0")
        if cfg.hud_persist_s <= 0.0:
            raise ValueError("hud_persist_s must be > 0")
        if len(self._SCENE_SCRIPT) == 0:
            raise ValueError("scene script must not be empty")

        self._clock = clock
        self._seed = int(seed)
        self._difficulty = clamp01(difficulty)
        self._cfg = cfg
        self._tick_dt = 1.0 / float(cfg.tick_hz)
        self._rng = SeededRng(self._seed)
        self._drift_gen = RapidTrackingDriftGenerator(seed=self._seed + 41)

        self._phase = Phase.INSTRUCTIONS
        self._phase_started_at_s = self._clock.now()
        self._last_update_at_s = self._phase_started_at_s
        self._accumulator_s = 0.0
        self._sim_elapsed_s = 0.0

        self._control_x = 0.0
        self._control_y = 0.0
        self._look_x = 0.0
        self._look_y = 0.0

        self._reticle_x = 0.0
        self._reticle_y = 0.0

        self._camera_x = 0.0
        self._camera_y = 0.0
        self._drift_x = 0.0
        self._drift_y = 0.0
        self._drift_until_s = 0.0

        self._target_x = 0.0
        self._target_y = 0.0
        self._target_vx = 0.0
        self._target_vy = 0.0
        self._target_is_moving = True
        self._target_kind = "tank"

        self._script_index = 0
        self._segment_started_s = 0.0
        self._segment_duration_s = 1.0
        self._segment_start_x = 0.0
        self._segment_start_y = 0.0
        self._segment_control_x = 0.0
        self._segment_control_y = 0.0
        self._segment_end_x = 0.0
        self._segment_end_y = 0.0
        self._target_switch_at_s = 0.0
        self._target_preview_until_s = 0.0

        self._terrain_ridge_y = 0.0
        self._target_terrain_occluded = False

        self._lock_hold_s = 0.0
        self._hud_visible_until_s = 0.0

        self._practice_samples = 0
        self._practice_sum_error = 0.0

        self._scored_samples = 0
        self._scored_sum_error = 0.0
        self._scored_sum_error2 = 0.0
        self._scored_on_target_s = 0.0
        self._scored_obscured_s = 0.0
        self._scored_obscured_on_target_s = 0.0
        self._scored_moving_target_s = 0.0

        self._window_elapsed_s = 0.0
        self._window_sum_error = 0.0
        self._window_samples = 0

        self._scored_attempted = 0
        self._scored_correct = 0
        self._scored_total_score = 0.0
        self._scored_max_score = 0.0

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
        if self._cfg.practice_duration_s <= 0.0:
            self._phase = Phase.PRACTICE_DONE
            self._phase_started_at_s = self._clock.now()
            return
        self._phase = Phase.PRACTICE
        self._reset_runtime_state(reset_scores=False)

    def start_scored(self) -> None:
        if self._phase in (Phase.SCORED, Phase.RESULTS):
            return
        self._phase = Phase.SCORED
        self._reset_runtime_state(reset_scores=True)

    def submit_answer(self, raw: str) -> bool:
        _ = raw
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
        target_rel_x = self._target_x - self._camera_x
        target_rel_y = self._target_y - self._camera_y

        preview_left = max(0.0, self._target_preview_until_s - self._sim_elapsed_s)
        hud_visible = self._sim_elapsed_s <= self._hud_visible_until_s
        lock_progress = clamp01(self._lock_hold_s / self._cfg.hud_acquire_s)

        payload = RapidTrackingPayload(
            target_rel_x=float(target_rel_x),
            target_rel_y=float(target_rel_y),
            reticle_x=float(self._reticle_x),
            reticle_y=float(self._reticle_y),
            camera_x=float(self._camera_x),
            camera_y=float(self._camera_y),
            target_vx=float(self._target_vx),
            target_vy=float(self._target_vy),
            target_visible=not self._target_terrain_occluded,
            target_is_moving=bool(self._target_is_moving),
            target_kind=str(self._target_kind),
            target_time_to_switch_s=max(0.0, self._target_switch_at_s - self._sim_elapsed_s),
            target_switch_preview_s=float(preview_left),
            obscured_time_left_s=0.0,
            phase_elapsed_s=max(0.0, self._clock.now() - self._phase_started_at_s),
            mean_error=self._current_mean_error(),
            rms_error=self._current_rms_error(),
            on_target_s=float(self._scored_on_target_s),
            on_target_ratio=self._current_on_target_ratio(),
            obscured_tracking_ratio=self._current_obscured_tracking_ratio(),
            hud_visible=bool(hud_visible),
            lock_progress=float(lock_progress),
            terrain_ridge_y=float(self._terrain_ridge_y),
        )

        return TestSnapshot(
            title="Rapid Tracking",
            phase=self._phase,
            prompt=self.current_prompt(),
            input_hint=(
                "Move camera with rudder (left/right) and joystick axis 1 (up/down). "
                "Keep target centered to acquire lock."
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
                "Rapid Tracking\n"
                "Track tank and aircraft targets from a helicopter-style moving viewpoint.\n"
                "Move the camera, keep the target centered, "
                "and maintain lock as terrain occludes it.\n"
                "HUD box/crosshair appear only after sustained on-target tracking.\n"
                "Press Enter to begin practice."
            )
        if self._phase is Phase.PRACTICE_DONE:
            return "Practice complete. Press Enter to start the timed test."
        if self._phase is Phase.RESULTS:
            summary = self.scored_summary()
            acc = int(round(summary.accuracy * 100))
            on_target_pct = summary.on_target_ratio * 100.0
            obscured_pct = summary.obscured_tracking_ratio * 100.0
            return (
                "Results\n"
                f"Windows: {summary.attempted}\n"
                f"Accurate: {summary.correct} ({acc}%)\n"
                f"Mean Error: {summary.mean_error:.3f}\n"
                "On-Target Time: "
                f"{summary.on_target_s:.1f}s ({on_target_pct:.1f}%)\n"
                f"Obscured Tracking: {obscured_pct:.1f}%"
            )
        return "Track and keep the target centered to hold lock."

    def scored_summary(self) -> RapidTrackingSummary:
        attempted = int(self._scored_attempted)
        correct = int(self._scored_correct)
        duration_s = float(self._cfg.scored_duration_s)
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

        obscured_tracking_ratio = 0.0
        if self._scored_obscured_s > 0.0:
            obscured_tracking_ratio = self._scored_obscured_on_target_s / self._scored_obscured_s

        moving_target_ratio = (
            0.0 if duration_s <= 0.0 else min(1.0, self._scored_moving_target_s / duration_s)
        )

        max_score = float(self._scored_max_score)
        total_score = float(self._scored_total_score)
        score_ratio = 0.0 if max_score <= 0.0 else total_score / max_score

        return RapidTrackingSummary(
            attempted=attempted,
            correct=correct,
            accuracy=accuracy,
            duration_s=duration_s,
            throughput_per_min=throughput_per_min,
            mean_error=float(mean_error),
            rms_error=float(rms_error),
            on_target_s=float(self._scored_on_target_s),
            on_target_ratio=float(on_target_ratio),
            obscured_time_s=float(self._scored_obscured_s),
            obscured_tracking_ratio=float(obscured_tracking_ratio),
            moving_target_ratio=float(moving_target_ratio),
            total_score=total_score,
            max_score=max_score,
            score_ratio=score_ratio,
        )

    def _reset_runtime_state(self, *, reset_scores: bool) -> None:
        now = self._clock.now()
        self._phase_started_at_s = now
        self._last_update_at_s = now
        self._accumulator_s = 0.0
        self._sim_elapsed_s = 0.0

        self._look_x = 0.0
        self._look_y = 0.0
        self._reticle_x = 0.0
        self._reticle_y = 0.0

        self._camera_x = 0.0
        self._camera_y = 0.0
        self._drift_x = 0.0
        self._drift_y = 0.0
        self._drift_until_s = 0.0
        self._refresh_drift_vector()

        self._target_x = 0.0
        self._target_y = 0.0
        self._target_vx = 0.0
        self._target_vy = 0.0
        self._target_is_moving = True
        self._target_kind = "tank"

        self._script_index = 0
        self._segment_started_s = 0.0
        self._segment_duration_s = 1.0
        self._segment_start_x = 0.0
        self._segment_start_y = 0.0
        self._segment_control_x = 0.0
        self._segment_control_y = 0.0
        self._segment_end_x = 0.0
        self._segment_end_y = 0.0
        self._target_switch_at_s = 0.0
        self._target_preview_until_s = 0.0
        self._start_scene_segment(initial=True)

        self._terrain_ridge_y = 0.0
        self._target_terrain_occluded = False

        self._lock_hold_s = 0.0
        self._hud_visible_until_s = 0.0

        if not reset_scores:
            self._practice_samples = 0
            self._practice_sum_error = 0.0

        if reset_scores:
            self._scored_samples = 0
            self._scored_sum_error = 0.0
            self._scored_sum_error2 = 0.0
            self._scored_on_target_s = 0.0
            self._scored_obscured_s = 0.0
            self._scored_obscured_on_target_s = 0.0
            self._scored_moving_target_s = 0.0

            self._window_elapsed_s = 0.0
            self._window_sum_error = 0.0
            self._window_samples = 0

            self._scored_attempted = 0
            self._scored_correct = 0
            self._scored_total_score = 0.0
            self._scored_max_score = 0.0

    def _step(self, dt: float) -> None:
        self._sim_elapsed_s += dt

        if self._sim_elapsed_s >= self._drift_until_s:
            self._refresh_drift_vector()

        self._advance_camera(dt)
        self._advance_target()

        target_rel_x = self._target_x - self._camera_x
        target_rel_y = self._target_y - self._camera_y
        self._terrain_ridge_y = self._mountain_ridge_for(target_rel_x)
        self._target_terrain_occluded = self._is_occluded_by_terrain(
            target_rel_x=target_rel_x,
            target_rel_y=target_rel_y,
            target_kind=self._target_kind,
        )

        err_x = target_rel_x - self._reticle_x
        err_y = target_rel_y - self._reticle_y
        tracking_error = math.sqrt((err_x * err_x) + (err_y * err_y))

        on_target = tracking_error <= self._cfg.on_target_radius
        self._update_hud_lock_state(dt=dt, on_target=on_target)

        if self._phase is Phase.PRACTICE:
            self._practice_samples += 1
            self._practice_sum_error += tracking_error
            return

        if self._phase is not Phase.SCORED:
            return

        self._scored_samples += 1
        self._scored_sum_error += tracking_error
        self._scored_sum_error2 += tracking_error * tracking_error

        if on_target:
            self._scored_on_target_s += dt

        if self._target_terrain_occluded:
            self._scored_obscured_s += dt
            if on_target:
                self._scored_obscured_on_target_s += dt

        if self._target_is_moving:
            self._scored_moving_target_s += dt

        self._window_elapsed_s += dt
        self._window_sum_error += tracking_error
        self._window_samples += 1

        if self._window_elapsed_s >= 1.0:
            self._score_active_window()
            self._window_elapsed_s = max(0.0, self._window_elapsed_s - 1.0)

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
                if self._window_samples > 0:
                    self._score_active_window()
                    self._window_elapsed_s = 0.0
                self._phase = Phase.RESULTS
                self._phase_started_at_s = now
                self._last_update_at_s = now
                self._accumulator_s = 0.0

    def _advance_camera(self, dt: float) -> None:
        response = min(1.0, dt * self._cfg.camera_control_response_hz)
        self._look_x += (self._control_x - self._look_x) * response
        self._look_y += (self._control_y - self._look_y) * response

        self._camera_x += (self._drift_x + (self._look_x * self._cfg.camera_control_speed)) * dt
        self._camera_y += (self._drift_y + (self._look_y * self._cfg.camera_control_speed)) * dt

        lim = float(self._cfg.camera_limit)
        self._camera_x = max(-lim, min(lim, self._camera_x))
        self._camera_y = max(-lim, min(lim, self._camera_y))

        # FPV center-fixed aiming reference.
        self._reticle_x = 0.0
        self._reticle_y = 0.0

    def _advance_target(self) -> None:
        while self._sim_elapsed_s >= self._target_switch_at_s:
            self._start_scene_segment(initial=False)

        seg_elapsed = self._sim_elapsed_s - self._segment_started_s
        if self._segment_duration_s <= 0.0:
            u = 1.0
        else:
            u = max(0.0, min(1.0, seg_elapsed / self._segment_duration_s))

        one_minus = 1.0 - u
        x = (
            (one_minus * one_minus * self._segment_start_x)
            + (2.0 * one_minus * u * self._segment_control_x)
            + (u * u * self._segment_end_x)
        )
        y = (
            (one_minus * one_minus * self._segment_start_y)
            + (2.0 * one_minus * u * self._segment_control_y)
            + (u * u * self._segment_end_y)
        )

        dx_du = (2.0 * one_minus * (self._segment_control_x - self._segment_start_x)) + (
            2.0 * u * (self._segment_end_x - self._segment_control_x)
        )
        dy_du = (2.0 * one_minus * (self._segment_control_y - self._segment_start_y)) + (
            2.0 * u * (self._segment_end_y - self._segment_control_y)
        )

        if self._segment_duration_s <= 0.0:
            vx = 0.0
            vy = 0.0
        else:
            inv = 1.0 / self._segment_duration_s
            vx = dx_du * inv
            vy = dy_du * inv

        bob = 0.0
        if self._target_kind == "plane":
            bob = math.sin((self._sim_elapsed_s * 2.1) + (self._script_index * 0.7)) * 0.022
        elif self._target_kind == "tank":
            bob = math.sin((self._sim_elapsed_s * 3.8) + (self._script_index * 1.1)) * 0.008

        self._target_x = float(x)
        self._target_y = float(y + bob)
        self._target_vx = float(vx)
        self._target_vy = float(vy)
        self._target_is_moving = math.hypot(vx, vy) > 0.018

    def _start_scene_segment(self, *, initial: bool) -> None:
        if initial:
            self._script_index = 0
        else:
            self._script_index = (self._script_index + 1) % len(self._SCENE_SCRIPT)

        segment = self._SCENE_SCRIPT[self._script_index]

        prev_kind = self._target_kind
        next_kind = str(segment.kind)

        if initial:
            start_x = float(segment.start_x)
            start_y = float(segment.start_y)
        else:
            # Blend current target with scripted lane so switches stay smooth.
            lane_blend = 0.24
            start_x = (self._target_x * (1.0 - lane_blend)) + (segment.start_x * lane_blend)
            start_y = (self._target_y * (1.0 - lane_blend)) + (segment.start_y * lane_blend)

        duration_scale = 1.0 - (0.15 * self._difficulty)
        duration = max(5.0, float(segment.duration_s) * duration_scale)

        self._segment_started_s = self._sim_elapsed_s
        self._segment_duration_s = duration
        self._segment_start_x = start_x
        self._segment_start_y = start_y
        self._segment_end_x = float(segment.end_x)
        self._segment_end_y = float(segment.end_y)
        self._segment_control_x = ((start_x + self._segment_end_x) * 0.5) + float(segment.arc_x)
        self._segment_control_y = ((start_y + self._segment_end_y) * 0.5) + float(segment.arc_y)
        self._target_switch_at_s = self._sim_elapsed_s + duration
        self._target_kind = next_kind

        if initial or next_kind != prev_kind:
            self._target_preview_until_s = self._sim_elapsed_s + self._cfg.target_preview_s

    def _refresh_drift_vector(self) -> None:
        drift = self._drift_gen.next_vector(difficulty=self._difficulty)
        self._drift_x = float(drift.vx)
        self._drift_y = float(drift.vy)
        self._drift_until_s = self._sim_elapsed_s + float(drift.duration_s)

    def _mountain_ridge_for(self, rel_x: float) -> float:
        x = float(rel_x)
        major = 0.14 * math.sin((x * 2.5) + (self._camera_x * 0.62) + 0.35)
        mid = 0.09 * math.sin((x * 4.7) - (self._camera_x * 0.44) + 1.15)
        fine = 0.05 * math.sin((x * 8.4) + (self._camera_x * 0.18) - 0.55)
        return 0.12 + major + mid + fine

    def _is_occluded_by_terrain(
        self, *, target_rel_x: float, target_rel_y: float, target_kind: str
    ) -> bool:
        if abs(target_rel_x) > (self._cfg.view_limit * 1.20):
            return False

        ridge_y = self._mountain_ridge_for(target_rel_x)
        if target_kind == "tank":
            return target_rel_y >= (ridge_y - 0.02)
        return target_rel_y >= (ridge_y + 0.09)

    def _update_hud_lock_state(self, *, dt: float, on_target: bool) -> None:
        previous = self._lock_hold_s

        if on_target:
            self._lock_hold_s += dt
            if previous < self._cfg.hud_acquire_s <= self._lock_hold_s:
                self._hud_visible_until_s = self._sim_elapsed_s + self._cfg.hud_persist_s
            if self._lock_hold_s >= self._cfg.hud_acquire_s:
                self._hud_visible_until_s = max(
                    self._hud_visible_until_s, self._sim_elapsed_s + 0.25
                )
            return

        self._lock_hold_s = max(0.0, self._lock_hold_s - (dt * 1.9))

    def _score_active_window(self) -> None:
        mean_error = (
            0.0 if self._window_samples == 0 else self._window_sum_error / self._window_samples
        )
        window_score = score_window(
            mean_error=mean_error,
            good_window_error=self._cfg.good_window_error,
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
                elapsed = self._cfg.scored_duration_s
            if elapsed <= 0.0:
                return 0.0
            return max(0.0, min(1.0, self._scored_on_target_s / elapsed))
        return 0.0

    def _current_obscured_tracking_ratio(self) -> float:
        if self._scored_obscured_s <= 0.0:
            return 0.0
        return max(0.0, min(1.0, self._scored_obscured_on_target_s / self._scored_obscured_s))


def build_rapid_tracking_test(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    config: RapidTrackingConfig | None = None,
) -> RapidTrackingEngine:
    return RapidTrackingEngine(
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        config=config,
    )
