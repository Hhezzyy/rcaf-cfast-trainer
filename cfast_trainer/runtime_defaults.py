from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

RUNTIME_DEFAULTS_STORE_ENV = "CFAST_RUNTIME_DEFAULTS_PATH"
AUDITORY_NOISE_LEVEL_ENV = "CFAST_AUDITORY_NOISE_LEVEL"
AUDITORY_DISTORTION_LEVEL_ENV = "CFAST_AUDITORY_DISTORTION_LEVEL"
AUDITORY_NOISE_SOURCE_ENV = "CFAST_AUDITORY_NOISE_SOURCE"


def _clamp_ratio(value: object) -> float | None:
    if value is None:
        return None
    try:
        ratio = float(value)
    except Exception:
        return None
    if ratio <= 0.0:
        return 0.0
    if ratio >= 1.0:
        return 1.0
    return float(ratio)


def _normalize_window_mode(value: object) -> str | None:
    token = str(value).strip().lower()
    if token in {"windowed", "resizable"}:
        return "windowed"
    if token in {"fullscreen", "exclusive"}:
        return "fullscreen"
    if token in {"borderless", "windowed_fullscreen", "desktop"}:
        return "borderless"
    return None


@dataclass(slots=True)
class RuntimeDefaultsState:
    window_mode: str | None = None
    use_opengl: bool | None = None
    auditory_noise_level: float | None = None
    auditory_distortion_level: float | None = None
    auditory_noise_source: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "window_mode": self.window_mode,
            "use_opengl": self.use_opengl,
            "auditory_noise_level": self.auditory_noise_level,
            "auditory_distortion_level": self.auditory_distortion_level,
            "auditory_noise_source": self.auditory_noise_source,
        }


class RuntimeDefaultsStore:
    _version = 1

    def __init__(self, path: Path) -> None:
        self._path = path
        self._state = RuntimeDefaultsState()
        self._load()

    @classmethod
    def default_path(cls) -> Path:
        explicit = os.environ.get(RUNTIME_DEFAULTS_STORE_ENV)
        if explicit:
            return Path(explicit).expanduser()
        return Path.home() / ".rcaf_cfast_runtime_defaults.json"

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(payload, dict):
            return
        self._state.window_mode = _normalize_window_mode(payload.get("window_mode"))
        use_opengl = payload.get("use_opengl")
        self._state.use_opengl = None if use_opengl is None else bool(use_opengl)
        self._state.auditory_noise_level = _clamp_ratio(payload.get("auditory_noise_level"))
        self._state.auditory_distortion_level = _clamp_ratio(
            payload.get("auditory_distortion_level")
        )
        source = payload.get("auditory_noise_source")
        if source is None:
            self._state.auditory_noise_source = None
        else:
            token = str(source).strip()
            self._state.auditory_noise_source = token or None

    def save(self) -> None:
        payload = {
            "version": self._version,
            **self._state.to_dict(),
        }
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._path.with_suffix(f"{self._path.suffix}.tmp")
            tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp_path.replace(self._path)
        except Exception:
            return

    def stored_window_mode(self) -> str | None:
        return self._state.window_mode

    def set_window_mode(self, value: str | None) -> None:
        self._state.window_mode = _normalize_window_mode(value)
        self.save()

    def stored_use_opengl(self) -> bool | None:
        return self._state.use_opengl

    def set_use_opengl(self, value: bool | None) -> None:
        self._state.use_opengl = None if value is None else bool(value)
        self.save()

    def stored_auditory_noise_level(self) -> float | None:
        return self._state.auditory_noise_level

    def set_auditory_noise_level(self, value: float | None) -> None:
        self._state.auditory_noise_level = _clamp_ratio(value)
        self.save()

    def stored_auditory_distortion_level(self) -> float | None:
        return self._state.auditory_distortion_level

    def set_auditory_distortion_level(self, value: float | None) -> None:
        self._state.auditory_distortion_level = _clamp_ratio(value)
        self.save()

    def stored_auditory_noise_source(self) -> str | None:
        return self._state.auditory_noise_source

    def set_auditory_noise_source(self, value: str | None) -> None:
        token = None if value is None else str(value).strip() or None
        self._state.auditory_noise_source = token
        self.save()

    def resolved_auditory_noise_level(self) -> float | None:
        env_value = os.environ.get(AUDITORY_NOISE_LEVEL_ENV)
        if env_value is not None:
            return _clamp_ratio(env_value)
        return self._state.auditory_noise_level

    def resolved_auditory_distortion_level(self) -> float | None:
        env_value = os.environ.get(AUDITORY_DISTORTION_LEVEL_ENV)
        if env_value is not None:
            return _clamp_ratio(env_value)
        return self._state.auditory_distortion_level

    def resolved_auditory_noise_source(self) -> str | None:
        env_value = os.environ.get(AUDITORY_NOISE_SOURCE_ENV)
        if env_value is not None:
            token = str(env_value).strip()
            return token or None
        return self._state.auditory_noise_source
