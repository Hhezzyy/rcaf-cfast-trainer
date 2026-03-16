from __future__ import annotations

from collections.abc import Callable, MutableSequence, Sequence
from enum import Enum
from typing import TypeVar

T = TypeVar("T")


_DEFAULT_METADATA_ATTRS: tuple[str, ...] = (
    "content_family",
    "variant_id",
    "content_pack",
    "question_kind",
    "template_name",
    "scenario_family",
    "system_family",
    "reasoning_mode",
    "domain_key",
    "kind",
    "family",
    "part",
    "focus_label",
)


def content_metadata_from_payload(
    payload: object | None,
    *,
    extras: dict[str, object] | None = None,
) -> dict[str, object] | None:
    if payload is None and not extras:
        return None

    metadata: dict[str, object] = {}
    if payload is not None:
        for attr in _DEFAULT_METADATA_ATTRS:
            if not hasattr(payload, attr):
                continue
            normalized = _normalize_metadata_value(getattr(payload, attr))
            if normalized is None:
                continue
            metadata[attr] = normalized
    if extras:
        for key, value in extras.items():
            normalized = _normalize_metadata_value(value)
            if normalized is None:
                continue
            metadata[str(key)] = normalized
    return metadata or None


def pick_with_recent_history(
    rng: object,
    items: Sequence[T],
    *,
    recent_keys: MutableSequence[str],
    horizon: int = 2,
    key: Callable[[T], str] | None = None,
) -> T:
    if not items:
        raise ValueError("items must not be empty")
    if len(items) == 1:
        choice = items[0]
        _remember_recent_key(
            recent_keys,
            _normalize_recent_key(choice if key is None else key(choice)),
            horizon=horizon,
        )
        return choice

    getter = key or (lambda item: str(item))
    recent = set(recent_keys[-max(0, int(horizon)) :]) if horizon > 0 else set()
    filtered = [item for item in items if _normalize_recent_key(getter(item)) not in recent]
    pool = filtered or list(items)
    randint = getattr(rng, "randint")
    choice = pool[int(randint(0, len(pool) - 1))]
    _remember_recent_key(recent_keys, _normalize_recent_key(getter(choice)), horizon=horizon)
    return choice


def stable_variant_id(*parts: object) -> str:
    tokens: list[str] = []
    for part in parts:
        token = str(_normalize_metadata_value(part) or "").strip().lower()
        if token:
            tokens.append(token.replace(" ", "_"))
    return ":".join(tokens)


def _normalize_metadata_value(value: object) -> object | None:
    if value is None:
        return None
    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, str):
        token = str(value).strip()
        return token or None
    if isinstance(value, (tuple, list)):
        normalized_items = tuple(
            item
            for item in (_normalize_metadata_value(item) for item in value)
            if item is not None
        )
        return normalized_items or None
    return str(value)


def _normalize_recent_key(value: object) -> str:
    return str(_normalize_metadata_value(value) or "").strip().lower()


def _remember_recent_key(recent_keys: MutableSequence[str], key: str, *, horizon: int) -> None:
    if not key or horizon <= 0:
        return
    recent_keys.append(key)
    overflow = len(recent_keys) - int(horizon)
    if overflow <= 0:
        return
    del recent_keys[:overflow]
