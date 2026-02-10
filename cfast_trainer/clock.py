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