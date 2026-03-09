"""RCAF CFAST Trainer (offline‑first).

Core deterministic logic and pygame adapters will be added in later tasks.
"""

from __future__ import annotations

import enum


if not hasattr(enum, "StrEnum"):
    class _CompatStrEnum(str, enum.Enum):
        pass


    enum.StrEnum = _CompatStrEnum
