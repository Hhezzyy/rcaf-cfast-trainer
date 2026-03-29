from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

from .clock import Clock
from .content_variants import content_metadata_from_payload
from .cognitive_core import (
    Phase,
    QuestionEvent,
    SeededRng,
    TestSnapshot,
    clamp01,
    lerp_int,
)


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
    guide_band_half_width: float = 0.12


@dataclass(frozen=True, slots=True)
class DisturbanceVector:
    vx: float
    vy: float
    duration_s: float


@dataclass(frozen=True, slots=True)
class SensoryMotorSegment:
    control_mode: str = "split"
    axis_focus: str = "both"
    disturbance_profile: str = "balanced"
    duration_s: float = 0.0
    label: str = ""
    kind: str = "scored"
    index: int = 0
    total: int = 0
    pause_after: bool = False


@dataclass(frozen=True, slots=True)
class SensoryMotorApparatusPayload:
    dot_x: float
    dot_y: float
    control_x: float
    control_y: float
    disturbance_x: float
    disturbance_y: float
    control_mode: str
    block_kind: str
    block_index: int
    block_total: int
    axis_focus: str
    guide_band_half_width: float
    segment_label: str
    segment_index: int
    segment_total: int
    segment_time_remaining_s: float
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
    overshoot_count: int
    reversal_count: int


@dataclass(frozen=True, slots=True)
class SensoryMotorWindowResult:
    index: int
    segment_label: str
    control_mode: str
    axis_focus: str
    disturbance_profile: str
    mean_error: float
    score: float
    duration_s: float
    presented_at_s: float
    answered_at_s: float


def _normalize_control_mode(value: str) -> str:
    token = str(value).strip().lower()
    if token == "joystick_only":
        return "joystick_only"
    return "split"


def _normalize_axis_focus(value: str) -> str:
    token = str(value).strip().lower()
    if token == "horizontal":
        return "horizontal"
    if token == "vertical":
        return "vertical"
    return "both"


def _normalize_disturbance_profile(value: str) -> str:
    token = str(value).strip().lower()
    if token in {
        "steady",
        "balanced",
        "pulse",
        "axis_bias_horizontal",
        "axis_bias_vertical",
        "pressure",
    }:
        return token
    return "balanced"


class SensoryMotorDisturbanceGenerator:
    """Deterministic disturbance stream for continuous psychomotor tracking."""

    def __init__(self, *, seed: int) -> None:
        self._rng = SeededRng(seed)

    def next_vector(
        self,
        *,
        difficulty: float,
        profile: str = "balanced",
        axis_focus: str = "both",
    ) -> DisturbanceVector:
        d = clamp01(difficulty)
        profile_key = _normalize_disturbance_profile(profile)
        focus_key = _normalize_axis_focus(axis_focus)

        mag_lo = lerp_int(28, 40, d) / 100.0
        mag_hi = lerp_int(46, 62, d) / 100.0
        interval_lo = max(0.32, 0.72 - (0.28 * d))
        interval_hi = max(interval_lo + 0.04, 1.28 - (0.30 * d))

        mag_scale = 1.0
        interval_scale = 1.0
        duration_scale = 1.0
        if profile_key == "steady":
            mag_scale = 0.72
            interval_scale = 1.35
            duration_scale = 1.45
        elif profile_key == "pulse":
            mag_scale = 1.14
            interval_scale = 0.72
            duration_scale = 0.62
        elif profile_key == "pressure":
            mag_scale = 1.28
            interval_scale = 0.66
            duration_scale = 0.78
        elif profile_key in {"axis_bias_horizontal", "axis_bias_vertical"}:
            mag_scale = 0.96
            interval_scale = 0.90
            duration_scale = 1.00

        magnitude = self._rng.uniform(mag_lo * mag_scale, mag_hi * mag_scale)
        duration_s = self._rng.uniform(
            max(0.18, interval_lo * interval_scale),
            max(max(0.22, interval_lo * interval_scale) + 0.04, interval_hi * duration_scale),
        )

        if profile_key == "axis_bias_horizontal":
            direction = -1.0 if self._rng.random() < 0.5 else 1.0
            vx = direction * magnitude
            vy = self._rng.uniform(-magnitude * 0.18, magnitude * 0.18)
        elif profile_key == "axis_bias_vertical":
            direction = -1.0 if self._rng.random() < 0.5 else 1.0
            vy = direction * magnitude
            vx = self._rng.uniform(-magnitude * 0.18, magnitude * 0.18)
        else:
            angle = self._rng.uniform(0.0, math.tau)
            vx = math.cos(angle) * magnitude
            vy = math.sin(angle) * magnitude

        if focus_key == "horizontal":
            vy = 0.0
        elif focus_key == "vertical":
            vx = 0.0

        return DisturbanceVector(vx=vx, vy=vy, duration_s=duration_s)


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
        title: str = "Sensory Motor Apparatus",
        practice_segments: Sequence[SensoryMotorSegment] | None = None,
        scored_segments: Sequence[SensoryMotorSegment] | None = None,
    ) -> None:
        self._clock = clock
        self._seed = int(seed)
        self._difficulty = clamp01(difficulty)
        self._config = config or SensoryMotorApparatusConfig()
        self._title = str(title)
        self._default_four_block_layout = practice_segments is None and scored_segments is None
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
        self._segment_elapsed_s = 0.0
        self._total_scored_elapsed_s = 0.0

        self._disturbance_generator = SensoryMotorDisturbanceGenerator(seed=self._seed)
        self._disturbance_until_s = 0.0
        self._disturbance_x = 0.0
        self._disturbance_y = 0.0

        self._dot_x = 0.35
        self._dot_y = -0.24

        self._control_x = 0.0
        self._control_y = 0.0

        self._scored_samples = 0
        self._scored_sum_error = 0.0
        self._scored_sum_error2 = 0.0
        self._scored_on_target_s = 0.0
        self._scored_overshoot_count = 0
        self._scored_reversal_count = 0
        self._prev_dot_sign_x: int | None = None
        self._prev_dot_sign_y: int | None = None

        self._window_elapsed_s = 0.0
        self._window_sum_error = 0.0
        self._window_samples = 0

        self._scored_attempted = 0
        self._scored_correct = 0
        self._scored_total_score = 0.0
        self._scored_max_score = 0.0

        self._practice_block_sum_error = 0.0
        self._practice_block_samples = 0
        self._scored_block_sum_error = 0.0
        self._scored_block_sum_error2 = 0.0
        self._scored_block_samples = 0
        self._scored_block_on_target_s = 0.0
        self._window_results: list[SensoryMotorWindowResult] = []

        practice_duration = max(0.0, float(self._config.practice_duration_s) * 0.5)
        scored_duration = max(0.0, float(self._config.scored_duration_s) * 0.5)
        self._practice_blocks = self._build_blocks(
            kind="practice",
            duration_s=practice_duration,
            segments=practice_segments,
            pause_after_default=True,
        )
        self._scored_blocks = self._build_blocks(
            kind="scored",
            duration_s=scored_duration,
            segments=scored_segments,
            pause_after_default=True,
        )
        self._practice_total_duration_s = sum(block.duration_s for block in self._practice_blocks)
        self._scored_total_duration_s = sum(block.duration_s for block in self._scored_blocks)
        if self._scored_total_duration_s <= 0.0:
            raise ValueError("scored segments must provide positive total duration")

        self._current_block: SensoryMotorSegment | None = None
        self._pending_block: SensoryMotorSegment | None = (
            self._practice_blocks[0] if self._practice_blocks else self._scored_blocks[0]
        )
        self._pending_done_action: str | None = None
        self._tick_dt = 1.0 / float(self._config.tick_hz)

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def difficulty(self) -> float:
        return self._difficulty

    @property
    def practice_questions(self) -> int:
        return 0

    @property
    def scored_duration_s(self) -> float:
        return float(self._scored_total_duration_s)

    def can_exit(self) -> bool:
        return self._phase in (
            Phase.INSTRUCTIONS,
            Phase.PRACTICE,
            Phase.PRACTICE_DONE,
            Phase.RESULTS,
        )

    def events(self) -> list[QuestionEvent]:
        return [
            QuestionEvent(
                index=result.index,
                phase=Phase.SCORED,
                prompt=f"{result.segment_label}: keep the dot centered",
                correct_answer=1,
                user_answer=1 if result.score >= 1.0 - 1e-9 else 0,
                is_correct=result.score >= 1.0 - 1e-9,
                presented_at_s=result.presented_at_s,
                answered_at_s=result.answered_at_s,
                response_time_s=result.duration_s,
                raw=f"{result.control_mode}/{result.axis_focus}/{result.disturbance_profile}",
                score=result.score,
                max_score=1.0,
                content_metadata=content_metadata_from_payload(
                    None,
                    extras={
                        "content_family": "sensory_motor",
                        "variant_id": f"{result.control_mode}:{result.axis_focus}:{result.disturbance_profile}",
                        "content_pack": "sensory_motor_apparatus",
                    },
                ),
            )
            for result in self._window_results
        ]

    def window_results(self) -> tuple[SensoryMotorWindowResult, ...]:
        return tuple(self._window_results)

    def start_practice(self) -> None:
        if self._phase is not Phase.INSTRUCTIONS:
            return
        if not self._practice_blocks:
            self._pending_block = self._scored_blocks[0] if self._scored_blocks else None
            self._pending_done_action = "start_scored"
            self._phase = Phase.PRACTICE_DONE
            self._phase_started_at_s = self._clock.now()
            return
        self._start_block(
            self._practice_blocks[0],
            reset_scored_totals=False,
            reset_scoring_window=True,
        )

    def start_scored(self) -> None:
        if self._phase is Phase.RESULTS:
            return
        if self._phase is Phase.INSTRUCTIONS:
            if self._practice_blocks:
                return
            if self._scored_blocks:
                self._start_block(
                    self._scored_blocks[0],
                    reset_scored_totals=True,
                    reset_scoring_window=True,
                )
            return
        if self._phase is not Phase.PRACTICE_DONE or self._pending_block is None:
            return
        reset_scored_totals = (
            self._pending_block.kind == "scored" and self._pending_block.index == 1
        )
        reset_scoring_window = not (
            self._pending_block.kind == "scored" and self._pending_block.index > 1
        )
        self._start_block(
            self._pending_block,
            reset_scored_totals=reset_scored_totals,
            reset_scoring_window=reset_scoring_window,
        )

    def submit_answer(self, raw: str) -> bool:
        token = str(raw).strip().lower()
        if token in {"__skip_all__", "skip_all"}:
            if self._phase in (Phase.PRACTICE, Phase.SCORED, Phase.PRACTICE_DONE):
                self._finish_to_results(now=self._clock.now())
                return True
            return False
        if token in {"__skip_practice__", "skip_practice"} and self._phase is Phase.PRACTICE:
            self._queue_transition_to_scored(now=self._clock.now())
            return True
        if token in {"__skip_section__", "skip_section"} and self._phase is Phase.SCORED:
            if self._current_block is not None and self._current_block.index < len(self._scored_blocks):
                self._finalize_block_window()
                self._queue_transition(now=self._clock.now(), next_block=self._next_scored_block())
            else:
                self._finish_to_results(now=self._clock.now())
            return True
        # Continuous psychomotor task; no per-question submission.
        return False

    def set_control(self, *, horizontal: float, vertical: float) -> None:
        block = self._active_or_pending_block()
        axis_focus = "both" if block is None else str(block.axis_focus)
        next_horizontal = max(-1.0, min(1.0, float(horizontal)))
        next_vertical = max(-1.0, min(1.0, float(vertical)))
        if axis_focus == "horizontal":
            next_vertical = 0.0
        elif axis_focus == "vertical":
            next_horizontal = 0.0
        self._control_x = next_horizontal
        self._control_y = next_vertical

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
        if self._phase not in (Phase.PRACTICE, Phase.SCORED):
            return None
        if self._current_block is None:
            return None
        rem = float(self._current_block.duration_s) - (self._clock.now() - self._phase_started_at_s)
        return max(0.0, rem)

    def snapshot(self) -> TestSnapshot:
        block = self._active_or_pending_block()
        control_mode = "split" if block is None else str(block.control_mode)
        block_kind = "" if block is None else str(block.kind)
        block_index = 0 if block is None else int(block.index)
        block_total = 0 if block is None else int(block.total)
        axis_focus = "both" if block is None else str(block.axis_focus)
        segment_label = "" if block is None else str(block.label)
        segment_index = 0 if block is None else int(block.index)
        segment_total = 0 if block is None else int(block.total)
        segment_time_remaining = 0.0 if block is None else max(0.0, self.time_remaining_s() or 0.0)
        payload = SensoryMotorApparatusPayload(
            dot_x=float(self._dot_x),
            dot_y=float(self._dot_y),
            control_x=float(self._control_x),
            control_y=float(self._control_y),
            disturbance_x=float(self._disturbance_x),
            disturbance_y=float(self._disturbance_y),
            control_mode=control_mode,
            block_kind=block_kind,
            block_index=block_index,
            block_total=block_total,
            axis_focus=axis_focus,
            guide_band_half_width=(
                float(self._config.guide_band_half_width) if axis_focus != "both" else 0.0
            ),
            segment_label=segment_label,
            segment_index=segment_index,
            segment_total=segment_total,
            segment_time_remaining_s=segment_time_remaining,
            phase_elapsed_s=max(0.0, self._clock.now() - self._phase_started_at_s),
            mean_error=self._current_mean_error(),
            rms_error=self._current_rms_error(),
            on_target_s=float(self._current_on_target_s_value()),
            on_target_ratio=self._current_on_target_ratio(),
        )

        return TestSnapshot(
            title=self._title,
            phase=self._phase,
            prompt=self.current_prompt(),
            input_hint=self._input_hint_for_block(block),
            time_remaining_s=self.time_remaining_s(),
            attempted_scored=self._scored_attempted,
            correct_scored=self._scored_correct,
            payload=payload,
            practice_feedback=None,
        )

    def current_prompt(self) -> str:
        block = self._active_or_pending_block()
        if self._phase is Phase.INSTRUCTIONS:
            if self._default_four_block_layout:
                return (
                    "Sensory Motor Apparatus\n"
                    "You will complete four blocks: joystick-only practice, split-controls practice, "
                    "joystick-only timed tracking, then split-controls timed tracking.\n"
                    f"First block: {self._block_heading(block)}.\n"
                    "Press Enter to begin practice."
                )
            return (
                "Sensory Motor Apparatus\n"
                f"First block: {self._block_heading(block)}.\n"
                "Press Enter to begin."
            )
        if self._phase is Phase.PRACTICE_DONE:
            if block is None:
                return "Block complete. Press Enter to continue."
            return (
                "Block complete.\n"
                f"Next: {self._block_heading(block)}.\n"
                "Press Enter to continue."
            )
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
        if block is None:
            return "Track the dot and keep it centered."
        if block.axis_focus == "horizontal":
            return f"{self._block_heading(block)}. Keep the dot inside the horizontal guide band."
        if block.axis_focus == "vertical":
            return f"{self._block_heading(block)}. Keep the dot inside the vertical guide band."
        return f"{self._block_heading(block)}. Track the dot and keep it centered."

    def scored_summary(self) -> SensoryMotorApparatusSummary:
        attempted = int(self._scored_attempted)
        correct = int(self._scored_correct)
        duration_s = float(self._scored_total_duration_s)
        accuracy = 0.0 if attempted == 0 else correct / attempted
        throughput_per_min = 0.0 if duration_s <= 0.0 else (attempted / duration_s) * 60.0

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
            overshoot_count=int(self._scored_overshoot_count),
            reversal_count=int(self._scored_reversal_count),
        )

    def _build_blocks(
        self,
        *,
        kind: str,
        duration_s: float,
        segments: Sequence[SensoryMotorSegment] | None,
        pause_after_default: bool,
    ) -> tuple[SensoryMotorSegment, ...]:
        if segments is None:
            if duration_s <= 0.0:
                return ()
            return (
                SensoryMotorSegment(
                    kind=kind,
                    index=1,
                    total=2,
                    control_mode="joystick_only",
                    axis_focus="both",
                    disturbance_profile="balanced",
                    duration_s=float(duration_s),
                    label="Joystick Only",
                    pause_after=pause_after_default,
                ),
                SensoryMotorSegment(
                    kind=kind,
                    index=2,
                    total=2,
                    control_mode="split",
                    axis_focus="both",
                    disturbance_profile="balanced",
                    duration_s=float(duration_s),
                    label="Split Controls",
                    pause_after=pause_after_default,
                ),
            )

        cleaned: list[SensoryMotorSegment] = []
        provided = tuple(segment for segment in segments if float(segment.duration_s) > 0.0)
        total = len(provided)
        for idx, segment in enumerate(provided, start=1):
            cleaned.append(
                replace(
                    segment,
                    kind=kind,
                    index=idx,
                    total=total,
                    control_mode=_normalize_control_mode(segment.control_mode),
                    axis_focus=_normalize_axis_focus(segment.axis_focus),
                    disturbance_profile=_normalize_disturbance_profile(segment.disturbance_profile),
                    label=str(segment.label).strip() or self._segment_label_for_mode(segment.control_mode),
                )
            )
        return tuple(cleaned)

    def _segment_label_for_mode(self, control_mode: str) -> str:
        return "Joystick Only" if _normalize_control_mode(control_mode) == "joystick_only" else "Split Controls"

    def _block_heading(self, block: SensoryMotorSegment | None) -> str:
        if block is None:
            return "Tracking Block"
        block_label = "Practice" if block.kind == "practice" else "Timed Test"
        label = str(block.label).strip() or self._segment_label_for_mode(block.control_mode)
        return f"{block_label} {block.index} of {block.total} - {label}"

    def _input_hint_for_block(self, block: SensoryMotorSegment | None) -> str:
        if block is None:
            return "Use joystick tracking controls to keep the dot centered."
        if block.control_mode == "joystick_only":
            base = "Use joystick X for left/right and joystick axis 1 for up/down."
        else:
            base = "Use rudder for left/right and joystick axis 1 for up/down."
        if block.axis_focus == "horizontal":
            return f"{base} Vertical input is ignored during this segment."
        if block.axis_focus == "vertical":
            return f"{base} Horizontal input is ignored during this segment."
        return base

    def _active_or_pending_block(self) -> SensoryMotorSegment | None:
        if self._phase in (Phase.INSTRUCTIONS, Phase.PRACTICE_DONE) and self._pending_block is not None:
            return self._pending_block
        return self._current_block

    def _start_block(
        self,
        block: SensoryMotorSegment,
        *,
        reset_scored_totals: bool,
        reset_scoring_window: bool,
    ) -> None:
        self._current_block = block
        self._pending_block = None
        self._pending_done_action = None
        self._phase = Phase.PRACTICE if block.kind == "practice" else Phase.SCORED
        if reset_scored_totals:
            self._reset_scored_totals()
        self._reset_tracking_state(
            reset_scoring_window=reset_scoring_window,
            reset_position=True,
            reset_disturbance=True,
        )

    def _next_practice_block(self) -> SensoryMotorSegment | None:
        if self._current_block is None or self._current_block.kind != "practice":
            return self._practice_blocks[0] if self._practice_blocks else None
        next_index = self._current_block.index
        return self._practice_blocks[next_index] if next_index < len(self._practice_blocks) else None

    def _next_scored_block(self) -> SensoryMotorSegment | None:
        if self._current_block is None or self._current_block.kind != "scored":
            return self._scored_blocks[0] if self._scored_blocks else None
        next_index = self._current_block.index
        return self._scored_blocks[next_index] if next_index < len(self._scored_blocks) else None

    def _queue_transition(self, *, now: float, next_block: SensoryMotorSegment | None) -> None:
        if next_block is None:
            self._finish_to_results(now=now)
            return
        self._phase = Phase.PRACTICE_DONE
        self._phase_started_at_s = float(now)
        self._last_update_at_s = float(now)
        self._accumulator_s = 0.0
        self._pending_block = next_block
        if next_block.kind == "practice":
            self._pending_done_action = "start_next_practice"
        elif next_block.index == 1:
            self._pending_done_action = "start_scored"
        else:
            self._pending_done_action = "start_next_scored"

    def _advance_inline_segment(self, *, now: float, next_block: SensoryMotorSegment) -> None:
        self._current_block = next_block
        self._pending_block = None
        self._pending_done_action = None
        self._phase_started_at_s = float(now)
        self._last_update_at_s = float(now)
        self._accumulator_s = 0.0
        self._reset_tracking_state(
            reset_scoring_window=False,
            reset_position=False,
            reset_disturbance=True,
        )

    def _queue_transition_to_scored(self, *, now: float) -> None:
        self._queue_transition(
            now=now,
            next_block=self._scored_blocks[0] if self._scored_blocks else None,
        )

    def _finish_to_results(self, *, now: float) -> None:
        self._phase = Phase.RESULTS
        self._phase_started_at_s = float(now)
        self._last_update_at_s = float(now)
        self._accumulator_s = 0.0
        self._pending_block = None
        self._pending_done_action = None

    def _reset_tracking_state(
        self,
        *,
        reset_scoring_window: bool,
        reset_position: bool,
        reset_disturbance: bool,
    ) -> None:
        self._phase_started_at_s = self._clock.now()
        self._last_update_at_s = self._phase_started_at_s
        self._accumulator_s = 0.0
        self._segment_elapsed_s = 0.0

        block = self._current_block
        axis_focus = "both" if block is None else block.axis_focus
        profile = "balanced" if block is None else block.disturbance_profile

        if reset_position:
            initial = self._disturbance_generator.next_vector(
                difficulty=self._difficulty,
                profile=profile,
                axis_focus=axis_focus,
            )
            self._dot_x = max(-0.75, min(0.75, initial.vx))
            self._dot_y = max(-0.75, min(0.75, initial.vy))
            if axis_focus == "horizontal":
                self._dot_y = 0.0
            elif axis_focus == "vertical":
                self._dot_x = 0.0
        else:
            if axis_focus == "horizontal":
                self._dot_y = 0.0
            elif axis_focus == "vertical":
                self._dot_x = 0.0

        if reset_disturbance:
            self._disturbance_x = 0.0
            self._disturbance_y = 0.0
            self._disturbance_until_s = 0.0

        self._practice_block_samples = 0
        self._practice_block_sum_error = 0.0
        self._scored_block_samples = 0
        self._scored_block_sum_error = 0.0
        self._scored_block_sum_error2 = 0.0
        self._scored_block_on_target_s = 0.0
        if reset_scoring_window:
            self._window_elapsed_s = 0.0
            self._window_sum_error = 0.0
            self._window_samples = 0

    def _reset_scored_totals(self) -> None:
        self._scored_samples = 0
        self._scored_sum_error = 0.0
        self._scored_sum_error2 = 0.0
        self._scored_on_target_s = 0.0
        self._scored_overshoot_count = 0
        self._scored_reversal_count = 0
        self._prev_dot_sign_x = None
        self._prev_dot_sign_y = None
        self._scored_attempted = 0
        self._scored_correct = 0
        self._scored_total_score = 0.0
        self._scored_max_score = 0.0
        self._total_scored_elapsed_s = 0.0
        self._window_results = []

    def _error_sign(self, value: float) -> int:
        deadband = max(float(self._config.on_target_radius) * 0.5, 1e-6)
        if value > deadband:
            return 1
        if value < -deadband:
            return -1
        return 0

    def _record_direction_change(self, *, axis: str, error_value: float) -> None:
        current = self._error_sign(error_value)
        previous = self._prev_dot_sign_x if axis == "x" else self._prev_dot_sign_y
        if current != 0:
            if previous is not None and previous != 0 and current != previous:
                self._scored_reversal_count += 1
                if abs(float(error_value)) > float(self._config.on_target_radius):
                    self._scored_overshoot_count += 1
            if axis == "x":
                self._prev_dot_sign_x = current
            else:
                self._prev_dot_sign_y = current

    def _finalize_block_window(self) -> None:
        if self._window_samples > 0:
            self._score_active_window()
            self._window_elapsed_s = 0.0

    def _step(self, dt: float) -> None:
        block = self._current_block
        if block is None:
            return

        self._segment_elapsed_s += dt

        if self._segment_elapsed_s >= self._disturbance_until_s:
            disturbance = self._disturbance_generator.next_vector(
                difficulty=self._difficulty,
                profile=block.disturbance_profile,
                axis_focus=block.axis_focus,
            )
            self._disturbance_x = disturbance.vx
            self._disturbance_y = disturbance.vy
            self._disturbance_until_s = self._segment_elapsed_s + disturbance.duration_s

        control_x = self._control_x
        control_y = self._control_y
        disturbance_x = self._disturbance_x
        disturbance_y = self._disturbance_y
        if block.axis_focus == "horizontal":
            control_y = 0.0
            disturbance_y = 0.0
            self._dot_y = 0.0
        elif block.axis_focus == "vertical":
            control_x = 0.0
            disturbance_x = 0.0
            self._dot_x = 0.0

        self._dot_x += ((control_x * self._config.control_gain) + disturbance_x) * dt
        self._dot_y += ((control_y * self._config.control_gain) + disturbance_y) * dt

        lim = float(self._config.field_limit)
        self._dot_x = max(-lim, min(lim, self._dot_x))
        self._dot_y = max(-lim, min(lim, self._dot_y))
        if block.axis_focus == "horizontal":
            self._dot_y = 0.0
        elif block.axis_focus == "vertical":
            self._dot_x = 0.0

        radial_error = math.sqrt((self._dot_x * self._dot_x) + (self._dot_y * self._dot_y))

        if self._phase is Phase.PRACTICE:
            self._practice_block_samples += 1
            self._practice_block_sum_error += radial_error
            return

        if self._phase is not Phase.SCORED:
            return

        self._total_scored_elapsed_s += dt
        self._scored_block_samples += 1
        self._scored_block_sum_error += radial_error
        self._scored_block_sum_error2 += radial_error * radial_error
        self._scored_samples += 1
        self._scored_sum_error += radial_error
        self._scored_sum_error2 += radial_error * radial_error
        if block.axis_focus != "vertical":
            self._record_direction_change(axis="x", error_value=self._dot_x)
        if block.axis_focus != "horizontal":
            self._record_direction_change(axis="y", error_value=self._dot_y)

        if radial_error <= self._config.on_target_radius:
            self._scored_block_on_target_s += dt
            self._scored_on_target_s += dt

        self._window_elapsed_s += dt
        self._window_sum_error += radial_error
        self._window_samples += 1

        if self._window_elapsed_s >= 1.0:
            self._score_active_window()
            self._window_elapsed_s = max(0.0, self._window_elapsed_s - 1.0)

    def _refresh_phase_boundaries(self, now: float) -> None:
        if self._phase is Phase.PRACTICE:
            active_duration = 0.0 if self._current_block is None else float(self._current_block.duration_s)
            if now - self._phase_started_at_s >= active_duration:
                next_block = self._next_practice_block()
                if next_block is None:
                    self._queue_transition_to_scored(now=now)
                elif self._current_block is not None and self._current_block.pause_after:
                    self._queue_transition(now=now, next_block=next_block)
                else:
                    self._advance_inline_segment(now=now, next_block=next_block)
            return

        if self._phase is Phase.SCORED:
            active_duration = 0.0 if self._current_block is None else float(self._current_block.duration_s)
            if now - self._phase_started_at_s >= active_duration:
                next_block = self._next_scored_block()
                if next_block is None:
                    self._finalize_block_window()
                    self._queue_transition(now=now, next_block=None)
                elif self._current_block is not None and self._current_block.pause_after:
                    self._queue_transition(now=now, next_block=next_block)
                else:
                    self._advance_inline_segment(now=now, next_block=next_block)

    def _score_active_window(self) -> None:
        if self._window_elapsed_s <= 0.0:
            self._window_sum_error = 0.0
            self._window_samples = 0
            return
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

        block = self._current_block
        label = "Tracking" if block is None else (str(block.label).strip() or self._block_heading(block))
        control_mode = "split" if block is None else str(block.control_mode)
        axis_focus = "both" if block is None else str(block.axis_focus)
        disturbance_profile = "balanced" if block is None else str(block.disturbance_profile)
        answered_at_s = self._phase_started_at_s + self._segment_elapsed_s
        presented_at_s = answered_at_s - self._window_elapsed_s
        self._window_results.append(
            SensoryMotorWindowResult(
                index=len(self._window_results),
                segment_label=label,
                control_mode=control_mode,
                axis_focus=axis_focus,
                disturbance_profile=disturbance_profile,
                mean_error=float(mean_error),
                score=float(window_score),
                duration_s=float(self._window_elapsed_s),
                presented_at_s=float(presented_at_s),
                answered_at_s=float(answered_at_s),
            )
        )

        self._window_sum_error = 0.0
        self._window_samples = 0

    def _current_mean_error(self) -> float:
        if self._phase is Phase.PRACTICE:
            if self._practice_block_samples == 0:
                return 0.0
            return self._practice_block_sum_error / self._practice_block_samples
        if self._phase is Phase.PRACTICE_DONE:
            if self._current_block is not None and self._current_block.kind == "practice":
                if self._practice_block_samples == 0:
                    return 0.0
                return self._practice_block_sum_error / self._practice_block_samples
            if self._scored_block_samples == 0:
                return 0.0
            return self._scored_block_sum_error / self._scored_block_samples
        if self._phase in (Phase.SCORED, Phase.RESULTS):
            if self._phase is Phase.SCORED:
                if self._scored_block_samples == 0:
                    return 0.0
                return self._scored_block_sum_error / self._scored_block_samples
            if self._scored_samples == 0:
                return 0.0
            return self._scored_sum_error / self._scored_samples
        return 0.0

    def _current_rms_error(self) -> float:
        if self._phase is Phase.SCORED:
            if self._scored_block_samples == 0:
                return 0.0
            return math.sqrt(self._scored_block_sum_error2 / self._scored_block_samples)
        if self._phase is Phase.PRACTICE_DONE:
            if self._current_block is not None and self._current_block.kind == "scored":
                if self._scored_block_samples == 0:
                    return 0.0
                return math.sqrt(self._scored_block_sum_error2 / self._scored_block_samples)
            return 0.0
        if self._phase is Phase.RESULTS:
            if self._scored_samples == 0:
                return 0.0
            return math.sqrt(self._scored_sum_error2 / self._scored_samples)
        return 0.0

    def _current_on_target_s_value(self) -> float:
        if self._phase is Phase.SCORED:
            return float(self._scored_block_on_target_s)
        if self._phase is Phase.PRACTICE_DONE:
            return float(self._scored_block_on_target_s) if (
                self._current_block is not None and self._current_block.kind == "scored"
            ) else 0.0
        if self._phase is Phase.RESULTS:
            return float(self._scored_on_target_s)
        return 0.0

    def _current_on_target_ratio(self) -> float:
        if self._phase is Phase.SCORED:
            elapsed = self._clock.now() - self._phase_started_at_s
            if elapsed <= 0.0:
                return 0.0
            return max(0.0, min(1.0, self._scored_block_on_target_s / elapsed))
        if self._phase is Phase.RESULTS:
            elapsed = self._scored_total_duration_s
            if elapsed <= 0.0:
                return 0.0
            return max(0.0, min(1.0, self._scored_on_target_s / elapsed))
        if self._phase is Phase.PRACTICE_DONE:
            if self._current_block is not None and self._current_block.kind == "scored":
                elapsed = 0.0 if self._current_block is None else float(self._current_block.duration_s)
                if elapsed <= 0.0:
                    return 0.0
                return max(0.0, min(1.0, self._scored_block_on_target_s / elapsed))
        return 0.0


def build_sensory_motor_apparatus_test(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    config: SensoryMotorApparatusConfig | None = None,
    title: str = "Sensory Motor Apparatus",
    practice_segments: Sequence[SensoryMotorSegment] | None = None,
    scored_segments: Sequence[SensoryMotorSegment] | None = None,
) -> SensoryMotorApparatusEngine:
    return SensoryMotorApparatusEngine(
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        config=config,
        title=title,
        practice_segments=practice_segments,
        scored_segments=scored_segments,
    )
