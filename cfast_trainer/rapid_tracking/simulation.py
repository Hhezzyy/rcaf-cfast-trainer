from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..cognitive_core import Phase
from .config import RapidTrackingConfig, RapidTrackingLayoutPolicy, RapidTrackingTrainingSegment
from .debug import RapidTrackingDebugState
from .legacy import RapidTrackingEngine as _LegacyRapidTrackingEngine


def _mode_for_phase(phase: Phase) -> str:
    if phase is Phase.PRACTICE_DONE:
        return "practice"
    if phase is Phase.RESULTS:
        return "scored"
    return str(phase.value)


class RapidTrackingSimulation(_LegacyRapidTrackingEngine):
    """Compatibility-friendly RT simulation wrapper with deterministic reset hooks."""

    def __init__(
        self,
        *,
        clock: Any,
        seed: int,
        difficulty: float = 0.5,
        config: RapidTrackingConfig | None = None,
        title: str = "Rapid Tracking",
        practice_segments: Sequence[RapidTrackingTrainingSegment] | None = None,
        scored_segments: Sequence[RapidTrackingTrainingSegment] | None = None,
        control_scheme: str = "joystick_only",
        layout_policy: RapidTrackingLayoutPolicy | str = RapidTrackingLayoutPolicy.DEFAULT,
    ) -> None:
        self._simulation_seed = int(seed)
        self._simulation_difficulty = float(difficulty)
        self._simulation_config = RapidTrackingConfig() if config is None else config
        self._simulation_title = str(title)
        self._simulation_control_scheme = str(control_scheme)
        self._simulation_practice_segments = (
            None if practice_segments is None else tuple(practice_segments)
        )
        self._simulation_scored_segments = (
            None if scored_segments is None else tuple(scored_segments)
        )
        self._simulation_layout_policy = layout_policy
        self._entered = False
        self._viewport_size = (0, 0)
        self._debug_state = RapidTrackingDebugState(
            last_reset_seed=int(seed),
            active_mode="instructions",
        )
        super().__init__(
            clock=clock,
            seed=int(seed),
            difficulty=float(difficulty),
            config=self._simulation_config,
            title=str(title),
            practice_segments=practice_segments,
            scored_segments=scored_segments,
            control_scheme=control_scheme,
            layout_policy=layout_policy,
        )

    @property
    def debug_state(self) -> RapidTrackingDebugState:
        return self._debug_state

    @property
    def dev_tools_enabled(self) -> bool:
        return bool(getattr(self, "_dev_tools_enabled", False))

    def set_dev_tools_enabled(self, enabled: bool) -> None:
        self._dev_tools_enabled = bool(enabled)

    def enter(self) -> None:
        self._entered = True

    def exit(self) -> None:
        self._entered = False

    def resize(self, width: int, height: int) -> None:
        self._viewport_size = (max(0, int(width)), max(0, int(height)))
        self._debug_state.viewport_size = self._viewport_size

    def start_practice(self) -> None:
        super().start_practice()
        self._debug_state.active_mode = _mode_for_phase(self.phase)

    def start_scored(self) -> None:
        super().start_scored()
        self._debug_state.active_mode = _mode_for_phase(self.phase)

    def update(self, dt: float | None = None) -> None:
        _ = dt
        now = float(self._clock.now())
        last = float(getattr(self, "_last_update_at_s", now))
        self._debug_state.last_update_dt_s = max(0.0, now - last)
        super().update()
        self._debug_state.active_mode = _mode_for_phase(self.phase)

    def snapshot_state(self) -> dict[str, object]:
        return {
            "session_seed": int(self.seed),
            "scene_seed": int(self.scene_seed),
            "layout_policy": str(self.layout_policy.value),
            "control_scheme": str(self.control_scheme),
            "entered": bool(self._entered),
            "viewport_size": tuple(self._viewport_size),
            "active_mode": str(self._debug_state.active_mode),
            "debug_overlay": bool(self._debug_state.overlay_enabled),
            "diagnostics": bool(self._debug_state.diagnostics_enabled),
        }

    def toggle_debug_overlay(self) -> bool:
        self._debug_state.overlay_enabled = not self._debug_state.overlay_enabled
        return bool(self._debug_state.overlay_enabled)

    def toggle_camera_diagnostics(self) -> bool:
        self._debug_state.diagnostics_enabled = not self._debug_state.diagnostics_enabled
        return bool(self._debug_state.diagnostics_enabled)

    def restart_practice(self) -> None:
        self.reset(mode="practice")

    def restart_scored(self) -> None:
        self.reset(mode="scored")

    def return_to_instructions(self) -> None:
        self.reset(mode="instructions")

    def reseed(self) -> int:
        next_seed = int(self.seed) + 1
        self.reset(seed=next_seed)
        return int(self.seed)

    def handle_event(self, event: object) -> bool:
        _ = event
        return False

    def _preserve_reset_state(self) -> dict[str, object]:
        return {
            "_entered": bool(self._entered),
            "_viewport_size": tuple(self._viewport_size),
            "_debug_state": RapidTrackingDebugState(
                overlay_enabled=bool(self._debug_state.overlay_enabled),
                diagnostics_enabled=bool(self._debug_state.diagnostics_enabled),
                last_reset_seed=self._debug_state.last_reset_seed,
                last_reset_reason=str(self._debug_state.last_reset_reason),
                reset_count=int(self._debug_state.reset_count),
                viewport_size=tuple(self._debug_state.viewport_size),
                last_update_dt_s=float(self._debug_state.last_update_dt_s),
                active_mode=str(self._debug_state.active_mode),
            ),
            "_dev_tools_enabled": bool(self.dev_tools_enabled),
        }

    def _restore_reset_state(
        self,
        preserved: dict[str, object],
        *,
        seed: int,
        mode: str,
        reason: str,
    ) -> None:
        self._entered = bool(preserved["_entered"])
        self._viewport_size = tuple(preserved["_viewport_size"])  # type: ignore[assignment]
        self._debug_state = preserved["_debug_state"]  # type: ignore[assignment]
        self._dev_tools_enabled = bool(preserved["_dev_tools_enabled"])
        self._debug_state.reset_count += 1
        self._debug_state.last_reset_seed = int(seed)
        self._debug_state.last_reset_reason = str(reason)
        self._debug_state.active_mode = str(mode)
        self._debug_state.viewport_size = self._viewport_size

    def reset(self, seed: int | None = None, mode: str | None = None) -> None:
        preserved = self._preserve_reset_state()
        current_seed = int(self.seed)
        next_seed = int(current_seed if seed is None else seed)
        next_mode = str(mode or _mode_for_phase(self.phase))
        replacement = type(self)(
            clock=self._clock,
            seed=next_seed,
            difficulty=self._simulation_difficulty,
            config=self._simulation_config,
            title=self._simulation_title,
            practice_segments=self._simulation_practice_segments,
            scored_segments=self._simulation_scored_segments,
            control_scheme=self._simulation_control_scheme,
            layout_policy=self._simulation_layout_policy,
        )
        if next_mode == "practice":
            replacement.start_practice()
        elif next_mode == "scored":
            replacement.start_scored()

        replacement_state = replacement.__dict__.copy()
        self.__dict__.clear()
        self.__dict__.update(replacement_state)
        self._restore_reset_state(
            preserved,
            seed=next_seed,
            mode=next_mode,
            reason="reseed" if seed is not None and next_seed != current_seed else "reset",
        )


__all__ = [
    "RapidTrackingSimulation",
]
