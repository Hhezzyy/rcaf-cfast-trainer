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
    camera_limit: float = 5.2
    camera_control_speed: float = 1.52
    camera_control_response_hz: float = 8.6

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

    # Camera-box trigger capture.
    capture_box_half_width: float = 0.085
    capture_box_half_height: float = 0.075
    capture_zoom_s: float = 0.48
    capture_cooldown_s: float = 0.42
    capture_flash_s: float = 0.34
    capture_zoom_strength: float = 0.85


@dataclass(frozen=True, slots=True)
class RapidTrackingDriftVector:
    vx: float
    vy: float
    duration_s: float


@dataclass(frozen=True, slots=True)
class RapidTrackingSceneSegment:
    kind: str  # "soldier" | "building" | "truck" | "helicopter" | "jet"
    variant: str
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    duration_s: float
    arc_x: float = 0.0
    arc_y: float = 0.0
    handoff: str = "smooth"  # "smooth" | "jump"


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
    target_variant: str
    target_handoff_mode: str
    difficulty: float
    difficulty_tier: str
    camera_assist_strength: float
    turbulence_strength: float
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
    scene_progress: float
    capture_box_half_width: float
    capture_box_half_height: float
    target_in_capture_box: bool
    capture_zoom: float
    capture_points: int
    capture_hits: int
    capture_attempts: int
    capture_accuracy: float
    capture_feedback: str
    capture_flash_s: float


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
    capture_points: int
    capture_hits: int
    capture_attempts: int
    capture_accuracy: float


@dataclass(frozen=True, slots=True)
class RapidTrackingDifficultyProfile:
    tier: str
    scene_script: tuple[RapidTrackingSceneSegment, ...]
    loop_limit: int
    duration_scale: float
    turbulence_strength: float
    drift_duration_scale: float
    camera_assist_strength: float


def rapid_tracking_target_label(*, kind: str, variant: str = "") -> str:
    target_kind = str(kind).strip().lower()
    target_variant = str(variant).strip().lower()
    if target_kind == "building":
        if target_variant == "garage":
            return "GARAGE"
        if target_variant == "tower":
            return "TOWER"
        return "HANGAR"
    if target_kind == "soldier":
        return "SOLDIER"
    if target_kind == "truck":
        return "TRUCK"
    if target_kind == "helicopter":
        return "HELICOPTER"
    if target_kind == "jet":
        return "JET"
    return target_kind.upper()


def rapid_tracking_target_description(*, kind: str, variant: str = "") -> str:
    target_kind = str(kind).strip().lower()
    target_variant = str(variant).strip().lower()
    if target_kind == "building":
        if target_variant == "garage":
            return "TRACK GARAGE UNTIL THE NEXT TARGET EMERGES"
        if target_variant == "tower":
            return "TRACK THE BUILDING DURING THE HANDOFF"
        return "TRACK THE HANGAR DURING THE HANDOFF"
    if target_kind == "soldier":
        return "DIFFERENTLY DRESSED SOLDIER AMONG A PATROL GROUP"
    if target_kind == "truck":
        return "GROUND VEHICLE"
    if target_kind == "helicopter":
        return "ROTARY-WING AIRCRAFT"
    if target_kind == "jet":
        return "FAST FIXED-WING AIRCRAFT"
    return "TARGET"


def rapid_tracking_target_cue(*, kind: str, variant: str = "", handoff_mode: str = "") -> str:
    target_kind = str(kind).strip().lower()
    mode = str(handoff_mode).strip().lower()
    if target_kind == "soldier":
        cue = "FOOT PATROL / SLOW"
    elif target_kind == "building":
        cue = "SHORT STATIC HANDOFF"
    elif target_kind == "truck":
        cue = "ROAD MOVEMENT / MEDIUM"
    elif target_kind == "helicopter":
        cue = "AIR HANDOFF / FAST"
    elif target_kind == "jet":
        cue = "FAST AIR PASS / FASTEST"
    else:
        cue = "TRACK TARGET"
    if mode == "jump":
        return f"{cue}  JUMP SWITCH"
    return f"{cue}  SMOOTH SWITCH"


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
            kind="soldier",
            variant="target",
            start_x=-2.30,
            start_y=0.88,
            end_x=-1.34,
            end_y=0.66,
            duration_s=7.0,
            arc_x=0.16,
            arc_y=0.08,
            handoff="smooth",
        ),
        RapidTrackingSceneSegment(
            kind="building",
            variant="hangar",
            start_x=-1.28,
            start_y=0.58,
            end_x=-1.28,
            end_y=0.58,
            duration_s=1.6,
            handoff="smooth",
        ),
        RapidTrackingSceneSegment(
            kind="truck",
            variant="olive",
            start_x=-1.18,
            start_y=0.62,
            end_x=0.54,
            end_y=0.78,
            duration_s=5.2,
            arc_x=0.24,
            arc_y=0.10,
            handoff="smooth",
        ),
        RapidTrackingSceneSegment(
            kind="building",
            variant="garage",
            start_x=0.62,
            start_y=0.66,
            end_x=0.62,
            end_y=0.66,
            duration_s=1.4,
            handoff="smooth",
        ),
        RapidTrackingSceneSegment(
            kind="helicopter",
            variant="green",
            start_x=0.58,
            start_y=0.16,
            end_x=1.86,
            end_y=-0.06,
            duration_s=4.6,
            arc_x=0.22,
            arc_y=-0.18,
            handoff="smooth",
        ),
        RapidTrackingSceneSegment(
            kind="building",
            variant="tower",
            start_x=1.54,
            start_y=0.18,
            end_x=1.54,
            end_y=0.18,
            duration_s=1.5,
            handoff="jump",
        ),
        RapidTrackingSceneSegment(
            kind="soldier",
            variant="target",
            start_x=1.46,
            start_y=0.32,
            end_x=0.30,
            end_y=0.56,
            duration_s=6.6,
            arc_x=-0.18,
            arc_y=0.12,
            handoff="smooth",
        ),
        RapidTrackingSceneSegment(
            kind="building",
            variant="hangar",
            start_x=0.20,
            start_y=0.52,
            end_x=0.20,
            end_y=0.52,
            duration_s=1.3,
            handoff="smooth",
        ),
        RapidTrackingSceneSegment(
            kind="jet",
            variant="red",
            start_x=0.18,
            start_y=-0.34,
            end_x=-2.20,
            end_y=-0.74,
            duration_s=3.8,
            arc_x=-0.32,
            arc_y=-0.24,
            handoff="smooth",
        ),
        RapidTrackingSceneSegment(
            kind="building",
            variant="tower",
            start_x=-1.64,
            start_y=0.12,
            end_x=-1.64,
            end_y=0.12,
            duration_s=1.4,
            handoff="jump",
        ),
        RapidTrackingSceneSegment(
            kind="truck",
            variant="olive",
            start_x=-1.58,
            start_y=0.28,
            end_x=1.70,
            end_y=0.72,
            duration_s=5.0,
            arc_x=0.28,
            arc_y=0.12,
            handoff="smooth",
        ),
        RapidTrackingSceneSegment(
            kind="building",
            variant="garage",
            start_x=1.68,
            start_y=0.64,
            end_x=1.68,
            end_y=0.64,
            duration_s=1.4,
            handoff="smooth",
        ),
        RapidTrackingSceneSegment(
            kind="jet",
            variant="yellow",
            start_x=1.58,
            start_y=-0.30,
            end_x=-1.74,
            end_y=-0.90,
            duration_s=3.4,
            arc_x=-0.22,
            arc_y=-0.18,
            handoff="jump",
        ),
        RapidTrackingSceneSegment(
            kind="building",
            variant="hangar",
            start_x=-1.18,
            start_y=0.50,
            end_x=-1.18,
            end_y=0.50,
            duration_s=1.4,
            handoff="smooth",
        ),
        RapidTrackingSceneSegment(
            kind="soldier",
            variant="target",
            start_x=-1.34,
            start_y=0.80,
            end_x=-0.12,
            end_y=0.60,
            duration_s=6.4,
            arc_x=0.18,
            arc_y=0.10,
            handoff="smooth",
        ),
    )

    _LOW_SCENE_SCRIPT: tuple[RapidTrackingSceneSegment, ...] = (
        RapidTrackingSceneSegment(
            kind="soldier",
            variant="target",
            start_x=-2.10,
            start_y=0.86,
            end_x=-1.18,
            end_y=0.66,
            duration_s=7.4,
            arc_x=0.12,
            arc_y=0.08,
            handoff="smooth",
        ),
        RapidTrackingSceneSegment(
            kind="building",
            variant="hangar",
            start_x=-1.14,
            start_y=0.58,
            end_x=-1.14,
            end_y=0.58,
            duration_s=1.8,
            handoff="smooth",
        ),
        RapidTrackingSceneSegment(
            kind="truck",
            variant="olive",
            start_x=-1.06,
            start_y=0.60,
            end_x=0.42,
            end_y=0.74,
            duration_s=5.8,
            arc_x=0.16,
            arc_y=0.08,
            handoff="smooth",
        ),
        RapidTrackingSceneSegment(
            kind="soldier",
            variant="target",
            start_x=0.40,
            start_y=0.72,
            end_x=-0.34,
            end_y=0.62,
            duration_s=6.8,
            arc_x=-0.10,
            arc_y=0.06,
            handoff="smooth",
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
        self._target_kind = "soldier"
        self._target_variant = "target"
        self._target_handoff_mode = "smooth"
        self._difficulty_tier = self._difficulty_profile().tier
        self._loop_count = 0

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
        self._capture_points = 0
        self._capture_hits = 0
        self._capture_attempts = 0
        self._capture_zoom_until_s = 0.0
        self._capture_last_at_s = -999.0
        self._capture_feedback_until_s = 0.0
        self._capture_feedback = ""

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
        self._difficulty_tier = ""
        self._loop_count = 0

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

    def _difficulty_profile(self) -> RapidTrackingDifficultyProfile:
        d = clamp01(self._difficulty)
        if d <= 0.25:
            return RapidTrackingDifficultyProfile(
                tier="low",
                scene_script=self._LOW_SCENE_SCRIPT,
                loop_limit=1,
                duration_scale=4.2,
                turbulence_strength=0.0,
                drift_duration_scale=3.0,
                camera_assist_strength=0.90,
            )
        if d >= 0.75:
            return RapidTrackingDifficultyProfile(
                tier="high",
                scene_script=self._SCENE_SCRIPT,
                loop_limit=3,
                duration_scale=0.72,
                turbulence_strength=1.30,
                drift_duration_scale=0.58,
                camera_assist_strength=0.0,
            )
        return RapidTrackingDifficultyProfile(
            tier="mid",
            scene_script=self._SCENE_SCRIPT,
            loop_limit=1,
            duration_scale=1.55,
            turbulence_strength=0.58,
            drift_duration_scale=1.18,
            camera_assist_strength=0.0,
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
        token = str(raw).strip().upper()
        if token in {"CAPTURE", "TRIGGER", "SHOT"}:
            return self._capture_trigger()
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
        profile = self._difficulty_profile()
        target_rel_x = self._target_x - self._camera_x
        target_rel_y = self._target_y - self._camera_y

        preview_left = max(0.0, self._target_preview_until_s - self._sim_elapsed_s)
        hud_visible = self._sim_elapsed_s <= self._hud_visible_until_s
        lock_progress = clamp01(self._lock_hold_s / self._cfg.hud_acquire_s)
        target_in_capture_box = self._target_in_capture_box(require_visible=True)
        capture_flash_s = max(0.0, self._capture_feedback_until_s - self._sim_elapsed_s)
        capture_accuracy = (
            0.0 if self._capture_attempts <= 0 else self._capture_hits / self._capture_attempts
        )

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
            target_variant=str(self._target_variant),
            target_handoff_mode=str(self._target_handoff_mode),
            difficulty=float(self._difficulty),
            difficulty_tier=str(profile.tier),
            camera_assist_strength=float(profile.camera_assist_strength),
            turbulence_strength=float(profile.turbulence_strength),
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
            scene_progress=float(self._scene_progress()),
            capture_box_half_width=float(self._cfg.capture_box_half_width),
            capture_box_half_height=float(self._cfg.capture_box_half_height),
            target_in_capture_box=bool(target_in_capture_box),
            capture_zoom=float(self._current_capture_zoom()),
            capture_points=int(self._capture_points),
            capture_hits=int(self._capture_hits),
            capture_attempts=int(self._capture_attempts),
            capture_accuracy=float(capture_accuracy),
            capture_feedback=str(self._capture_feedback),
            capture_flash_s=float(capture_flash_s),
        )

        return TestSnapshot(
            title="Rapid Tracking",
            phase=self._phase,
            prompt=self.current_prompt(),
            input_hint=(
                "Move camera with rudder (left/right) and joystick axis 1 (up/down). "
                "Keep target centered, then press trigger/Space when the target is inside the "
                "center camera box."
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
                "Track a marked soldier, then switch to buildings, vehicles, helicopters, "
                "and jets as the handoff changes.\n"
                "Use the red camera cue line and the live target feed to confirm the active "
                "target; if it leaves the screen, follow the edge pointer back onto it.\n"
                "The helicopter starts high, descends into the valley, then leaves orbit and "
                "runs a forward path late in the test.\n"
                "When the target enters a building or garage, switch to the structure until "
                "the next target emerges.\n"
                "Move the camera, keep the target centered, and maintain lock through "
                "smooth and jump handoffs.\n"
                "Press trigger or Space when the target is inside the center camera box to zoom "
                "and score camera points.\n"
                "Buildings are brief static targets; jets are the fastest targets.\n"
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
                f"Obscured Tracking: {obscured_pct:.1f}%\n"
                "Camera Hits: "
                f"{summary.capture_hits}/{summary.capture_attempts}  "
                f"Points: {summary.capture_points}"
            )
        return "Track, hold lock, and trigger a camera capture when the target enters the center box."

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
        capture_attempts = int(self._capture_attempts)
        capture_hits = int(self._capture_hits)
        capture_accuracy = 0.0 if capture_attempts == 0 else capture_hits / capture_attempts

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
            capture_points=int(self._capture_points),
            capture_hits=capture_hits,
            capture_attempts=capture_attempts,
            capture_accuracy=float(capture_accuracy),
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
        self._target_kind = "soldier"
        self._target_variant = "target"
        self._target_handoff_mode = "smooth"

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
        self._capture_zoom_until_s = 0.0
        self._capture_last_at_s = -999.0
        self._capture_feedback_until_s = 0.0
        self._capture_feedback = ""
        self._capture_points = 0
        self._capture_hits = 0
        self._capture_attempts = 0

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
        profile = self._difficulty_profile()
        response = min(1.0, dt * self._cfg.camera_control_response_hz)
        self._look_x += (self._control_x - self._look_x) * response
        self._look_y += (self._control_y - self._look_y) * response

        self._camera_x += (self._drift_x + (self._look_x * self._cfg.camera_control_speed)) * dt
        self._camera_y += (self._drift_y + (self._look_y * self._cfg.camera_control_speed)) * dt

        if profile.camera_assist_strength > 0.0:
            assist_response = min(1.0, dt * (2.6 + (profile.camera_assist_strength * 5.2)))
            self._camera_x += (
                (self._target_x - self._camera_x)
                * assist_response
                * profile.camera_assist_strength
                * 0.92
            )
            self._camera_y += (
                (self._target_y - self._camera_y)
                * assist_response
                * profile.camera_assist_strength
                * 0.84
            )

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
        kind = self._target_kind
        dominant_ground_axis_x = abs(self._segment_end_x - self._segment_start_x) >= abs(
            self._segment_end_y - self._segment_start_y
        )

        if kind == "truck":
            if dominant_ground_axis_x:
                x = self._segment_start_x + ((self._segment_end_x - self._segment_start_x) * u)
                y = self._segment_start_y
                dx_du = self._segment_end_x - self._segment_start_x
                dy_du = 0.0
            else:
                x = self._segment_start_x
                y = self._segment_start_y + ((self._segment_end_y - self._segment_start_y) * u)
                dx_du = 0.0
                dy_du = self._segment_end_y - self._segment_start_y
        else:
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

        extra_x = 0.0
        extra_y = 0.0
        extra_vx = 0.0
        extra_vy = 0.0
        phase = self._sim_elapsed_s + (self._script_index * 0.73)
        if kind == "building":
            self._target_x = float(self._segment_end_x)
            self._target_y = float(self._segment_end_y)
            self._target_vx = 0.0
            self._target_vy = 0.0
            self._target_is_moving = False
            return
        if kind == "soldier":
            weave_amp = 0.018 + (0.010 * self._difficulty)
            stride_amp = 0.010 + (0.006 * self._difficulty)
            weave_freq = 2.4 + (0.7 * self._difficulty)
            stride_freq = 5.8 + (1.0 * self._difficulty)
            extra_x = math.sin(phase * weave_freq) * weave_amp
            extra_y = math.sin((phase * stride_freq) + 0.45) * stride_amp
            extra_vx = math.cos(phase * weave_freq) * weave_amp * weave_freq
            extra_vy = math.cos((phase * stride_freq) + 0.45) * stride_amp * stride_freq
        elif kind == "truck":
            surge_amp = 0.012 + (0.005 * self._difficulty)
            surge_freq = 1.9 + (0.5 * self._difficulty)
            surge = math.sin((phase * surge_freq) + 0.8) * surge_amp
            surge_v = math.cos((phase * surge_freq) + 0.8) * surge_amp * surge_freq
            if dominant_ground_axis_x:
                extra_x = surge
                extra_vx = surge_v
            else:
                extra_y = surge
                extra_vy = surge_v
        elif kind == "helicopter":
            hover_amp = 0.024 + (0.012 * self._difficulty)
            bob_amp = 0.016 + (0.010 * self._difficulty)
            hover_freq = 1.3 + (0.4 * self._difficulty)
            bob_freq = 2.3 + (0.6 * self._difficulty)
            extra_x = math.sin(phase * hover_freq) * hover_amp
            extra_y = math.sin((phase * bob_freq) + 0.65) * bob_amp
            extra_vx = math.cos(phase * hover_freq) * hover_amp * hover_freq
            extra_vy = math.cos((phase * bob_freq) + 0.65) * bob_amp * bob_freq
        elif kind == "jet":
            wx_amp = 0.018 + (0.028 * self._difficulty)
            wy_amp = 0.018 + (0.030 * self._difficulty)
            wx_freq = 2.0 + (1.0 * self._difficulty)
            wy_freq = 2.6 + (1.4 * self._difficulty)
            extra_x = math.sin(phase * wx_freq) * wx_amp
            extra_y = math.sin((phase * wy_freq) + 0.4) * wy_amp
            extra_vx = math.cos(phase * wx_freq) * wx_amp * wx_freq
            extra_vy = math.cos((phase * wy_freq) + 0.4) * wy_amp * wy_freq

        self._target_x = float(x + extra_x)
        self._target_y = float(y + extra_y)
        self._target_vx = float(vx + extra_vx)
        self._target_vy = float(vy + extra_vy)
        speed_threshold = 0.012 if kind == "soldier" else 0.020
        self._target_is_moving = math.hypot(self._target_vx, self._target_vy) > speed_threshold

    def _start_scene_segment(self, *, initial: bool) -> None:
        profile = self._difficulty_profile()
        script = profile.scene_script
        if len(script) == 0:
            return

        tier_changed = self._difficulty_tier != profile.tier
        self._difficulty_tier = profile.tier

        if initial or tier_changed:
            self._script_index = 0
            self._loop_count = 0
        else:
            next_index = self._script_index + 1
            if next_index >= len(script):
                self._loop_count += 1
                if profile.loop_limit > 0 and self._loop_count >= profile.loop_limit:
                    self._target_switch_at_s = math.inf
                    self._target_preview_until_s = 0.0
                    return
                next_index = 0
            self._script_index = next_index

        segment = script[self._script_index]

        prev_kind = self._target_kind
        prev_variant = self._target_variant
        next_kind = str(segment.kind)
        next_variant = str(segment.variant)
        next_handoff = "smooth" if initial else str(segment.handoff).strip().lower()

        if initial:
            start_x = float(segment.start_x)
            start_y = float(segment.start_y)
        else:
            if next_handoff == "jump":
                start_x = float(segment.start_x)
                start_y = float(segment.start_y)
            else:
                lane_blend = 0.20 if (next_kind == "building" or prev_kind == "building") else 0.26
                start_x = (self._target_x * (1.0 - lane_blend)) + (segment.start_x * lane_blend)
                start_y = (self._target_y * (1.0 - lane_blend)) + (segment.start_y * lane_blend)

        scene_progress = self._scene_progress()
        pressure = clamp01((self._difficulty * 0.44) + (scene_progress * 0.60))
        if next_kind == "soldier":
            duration_scale = 1.04 - (0.16 * pressure)
            duration = max(4.4, float(segment.duration_s) * duration_scale)
            arc_scale = 0.90 + (0.12 * pressure)
        elif next_kind == "building":
            duration_scale = 0.98 - (0.10 * pressure)
            duration = max(1.0, min(2.2, float(segment.duration_s) * duration_scale))
            arc_scale = 0.0
        elif next_kind == "truck":
            duration_scale = 0.92 - (0.18 * pressure)
            duration = max(3.2, float(segment.duration_s) * duration_scale)
            arc_scale = 0.98 + (0.14 * pressure)
        elif next_kind == "helicopter":
            duration_scale = 0.84 - (0.20 * pressure)
            duration = max(2.7, float(segment.duration_s) * duration_scale)
            arc_scale = 1.06 + (0.20 * pressure)
        else:
            duration_scale = 0.72 - (0.24 * pressure)
            duration = max(1.9, float(segment.duration_s) * duration_scale)
            arc_scale = 1.18 + (0.24 * pressure)

        duration *= profile.duration_scale

        self._segment_started_s = self._sim_elapsed_s
        self._segment_duration_s = duration
        self._segment_start_x = start_x
        self._segment_start_y = start_y
        self._segment_end_x = float(segment.end_x)
        self._segment_end_y = float(segment.end_y)
        self._segment_control_x = ((start_x + self._segment_end_x) * 0.5) + (
            float(segment.arc_x) * arc_scale
        )
        self._segment_control_y = ((start_y + self._segment_end_y) * 0.5) + (
            float(segment.arc_y) * arc_scale
        )
        self._target_switch_at_s = self._sim_elapsed_s + duration
        self._target_kind = next_kind
        self._target_variant = next_variant
        self._target_handoff_mode = next_handoff

        if initial or next_kind != prev_kind or next_variant != prev_variant or next_handoff == "jump":
            self._target_preview_until_s = self._sim_elapsed_s + self._cfg.target_preview_s

    def _refresh_drift_vector(self) -> None:
        profile = self._difficulty_profile()
        if profile.turbulence_strength <= 0.0:
            self._drift_x = 0.0
            self._drift_y = 0.0
            self._drift_until_s = self._sim_elapsed_s + 60.0
            return
        drift = self._drift_gen.next_vector(difficulty=self._difficulty)
        self._drift_x = float(drift.vx * profile.turbulence_strength)
        self._drift_y = float(drift.vy * profile.turbulence_strength)
        self._drift_until_s = self._sim_elapsed_s + float(
            drift.duration_s * profile.drift_duration_scale
        )

    def _mountain_ridge_for(self, rel_x: float) -> float:
        x = float(rel_x)
        major = 0.14 * math.sin((x * 2.5) + (self._camera_x * 0.62) + 0.35)
        mid = 0.09 * math.sin((x * 4.7) - (self._camera_x * 0.44) + 1.15)
        fine = 0.05 * math.sin((x * 8.4) + (self._camera_x * 0.18) - 0.55)
        return 0.12 + major + mid + fine

    def _is_occluded_by_terrain(
        self, *, target_rel_x: float, target_rel_y: float, target_kind: str
    ) -> bool:
        kind = str(target_kind).strip().lower()
        if kind == "building":
            return False
        if abs(target_rel_x) > (self._cfg.view_limit * 1.20):
            return False

        ridge_y = self._mountain_ridge_for(target_rel_x)
        if kind in {"soldier", "truck"}:
            return target_rel_y >= (ridge_y + 0.12)
        if kind == "helicopter":
            return target_rel_y >= (ridge_y + 0.22)
        return target_rel_y >= (ridge_y + 0.34)

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

    def _capture_trigger(self) -> bool:
        if self._phase not in (Phase.PRACTICE, Phase.SCORED):
            return False
        if (self._sim_elapsed_s - self._capture_last_at_s) < self._cfg.capture_cooldown_s:
            return False

        self._capture_last_at_s = self._sim_elapsed_s
        self._capture_attempts += 1

        hit = self._target_in_capture_box(require_visible=True)
        if hit:
            if self._target_kind in {"helicopter", "jet"}:
                points = 2
            else:
                points = 1
            self._capture_hits += 1
            self._capture_points += points
            self._capture_zoom_until_s = self._sim_elapsed_s + self._cfg.capture_zoom_s
            self._capture_feedback = f"+{points}"
            self._capture_feedback_until_s = self._sim_elapsed_s + self._cfg.capture_flash_s
            return True

        self._capture_feedback = "MISS"
        self._capture_feedback_until_s = self._sim_elapsed_s + self._cfg.capture_flash_s
        return True

    def _target_in_capture_box(self, *, require_visible: bool) -> bool:
        if require_visible and self._target_terrain_occluded:
            return False
        target_rel_x = self._target_x - self._camera_x
        target_rel_y = self._target_y - self._camera_y
        return (
            abs(target_rel_x - self._reticle_x) <= self._cfg.capture_box_half_width
            and abs(target_rel_y - self._reticle_y) <= self._cfg.capture_box_half_height
        )

    def _current_capture_zoom(self) -> float:
        remaining = self._capture_zoom_until_s - self._sim_elapsed_s
        if remaining <= 0.0 or self._cfg.capture_zoom_s <= 0.0:
            return 0.0
        t = 1.0 - max(0.0, min(1.0, remaining / self._cfg.capture_zoom_s))
        pulse = math.sin(t * math.pi)
        kick = 1.0 - t
        return max(0.0, ((pulse * 0.72) + (kick * 0.28)) * self._cfg.capture_zoom_strength)

    def _scene_progress(self) -> float:
        if self._phase is Phase.PRACTICE:
            if self._cfg.practice_duration_s <= 0.0:
                return 1.0
            return clamp01((self._clock.now() - self._phase_started_at_s) / self._cfg.practice_duration_s)
        if self._phase in (Phase.SCORED, Phase.RESULTS):
            if self._cfg.scored_duration_s <= 0.0:
                return 1.0
            elapsed = self._clock.now() - self._phase_started_at_s
            if self._phase is Phase.RESULTS:
                elapsed = self._cfg.scored_duration_s
            return clamp01(elapsed / self._cfg.scored_duration_s)
        return 0.0

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
