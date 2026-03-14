from __future__ import annotations

import time
from typing import Protocol


class Clock(Protocol):
    """Monotonic clock abstraction.

    Core logic depends on this interface rather than calling real time directly.
    """

    def now(self) -> float:
        """Return monotonic seconds."""


class RealClock:
    """Production clock backed by time.monotonic()."""

    def now(self) -> float:
        return time.monotonic()


class PausableClock:
    """Clock wrapper that can freeze elapsed time while the UI is paused."""

    def __init__(self, base: Clock) -> None:
        self._base = base
        self._paused = False
        self._pause_started_at_s = 0.0
        self._total_paused_s = 0.0

    def now(self) -> float:
        base_now = self._pause_started_at_s if self._paused else self._base.now()
        return max(0.0, float(base_now) - float(self._total_paused_s))

    def pause(self) -> None:
        if self._paused:
            return
        self._pause_started_at_s = float(self._base.now())
        self._paused = True

    def resume(self) -> None:
        if not self._paused:
            return
        resumed_at_s = float(self._base.now())
        self._total_paused_s += max(0.0, resumed_at_s - self._pause_started_at_s)
        self._paused = False

    def is_paused(self) -> bool:
        return bool(self._paused)
