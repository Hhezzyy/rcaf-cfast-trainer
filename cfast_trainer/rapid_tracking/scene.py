from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pygame

from ..cognitive_core import TestSnapshot
from .entities import RapidTrackingPayload
from .renderer import RapidTrackingUiContext, render_rapid_tracking_screen
from .simulation import RapidTrackingSimulation


@dataclass(slots=True)
class _RapidTrackingScreenBinding:
    app: Any
    small_font: pygame.font.Font
    tiny_font: pygame.font.Font


class RapidTrackingExercise(RapidTrackingSimulation):
    """Exercise lifecycle wrapper around the deterministic RT simulation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._screen_binding: _RapidTrackingScreenBinding | None = None
        self._dev_button_hitboxes: dict[str, pygame.Rect] = {}

    def bind_screen_context(
        self,
        *,
        app: Any,
        small_font: pygame.font.Font,
        tiny_font: pygame.font.Font,
    ) -> None:
        self._screen_binding = _RapidTrackingScreenBinding(
            app=app,
            small_font=small_font,
            tiny_font=tiny_font,
        )
        self.set_dev_tools_enabled(bool(app.dev_tools_enabled()))

    def set_dev_button_hitboxes(self, hitboxes: dict[str, pygame.Rect]) -> None:
        self._dev_button_hitboxes = {str(key): pygame.Rect(value) for key, value in hitboxes.items()}

    def _preserve_reset_state(self) -> dict[str, object]:
        preserved = super()._preserve_reset_state()
        preserved["_screen_binding"] = self._screen_binding
        preserved["_dev_button_hitboxes"] = {
            str(key): pygame.Rect(value) for key, value in self._dev_button_hitboxes.items()
        }
        return preserved

    def _restore_reset_state(
        self,
        preserved: dict[str, object],
        *,
        seed: int,
        mode: str,
        reason: str,
    ) -> None:
        super()._restore_reset_state(
            preserved,
            seed=seed,
            mode=mode,
            reason=reason,
        )
        self._screen_binding = preserved["_screen_binding"]  # type: ignore[assignment]
        self._dev_button_hitboxes = {
            str(key): pygame.Rect(value)
            for key, value in dict(preserved["_dev_button_hitboxes"]).items()
        }

    def handle_event(self, event: object) -> bool:
        if not self.dev_tools_enabled or not hasattr(event, "type"):
            return False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_F2:
                self.toggle_debug_overlay()
                return True
            if event.key == pygame.K_F3:
                self.toggle_camera_diagnostics()
                return True
            if event.key == pygame.K_F5 and bool(getattr(event, "mod", 0) & pygame.KMOD_SHIFT):
                self.return_to_instructions()
                return True
            if event.key == pygame.K_F5:
                self.reset()
                return True
            if event.key == pygame.K_F6:
                self.reseed()
                return True
        if event.type == pygame.MOUSEBUTTONDOWN and getattr(event, "button", 0) == 1:
            pos = getattr(event, "pos", None)
            if pos is None:
                return False
            for action, rect in self._dev_button_hitboxes.items():
                if rect.collidepoint(pos):
                    self._run_dev_action(action)
                    return True
        return False

    def _run_dev_action(self, action: str) -> None:
        if action == "reset":
            self.reset()
            return
        if action == "reseed":
            self.reseed()
            return
        if action == "instructions":
            self.return_to_instructions()
            return
        if action == "practice":
            self.restart_practice()
            return
        if action == "scored":
            self.restart_scored()
            return
        if action == "debug":
            self.toggle_debug_overlay()
            return
        if action == "camera":
            self.toggle_camera_diagnostics()

    def render(self, surface: pygame.Surface) -> None:
        if self._screen_binding is None:
            return
        self.resize(*surface.get_size())
        snap = self.snapshot()
        payload = snap.payload if isinstance(snap.payload, RapidTrackingPayload) else None
        render_rapid_tracking_screen(
            surface=surface,
            snap=snap,
            payload=payload,
            engine=self,
            context=RapidTrackingUiContext(
                app=self._screen_binding.app,
                small_font=self._screen_binding.small_font,
                tiny_font=self._screen_binding.tiny_font,
            ),
        )

    def snapshot(self) -> TestSnapshot:
        snap = super().snapshot()
        self.debug_state.active_mode = snap.phase.value
        return snap


class RapidTrackingEngine(RapidTrackingExercise):
    """Historical engine name retained for app/test compatibility."""


__all__ = [
    "RapidTrackingExercise",
    "RapidTrackingEngine",
]
