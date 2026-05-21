from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .activity_runtime_catalog import OFFICIAL_TEST_MENU_ORDER, build_official_test_runtime
from .adaptive_difficulty import difficulty_profile_for_code, difficulty_ratio_for_level
from .adaptive_scheduler import build_adaptive_session_plan
from .benchmark import build_benchmark_plan
from .canonical_drill_registry import CANONICAL_DRILL_REGISTRY, CanonicalDrillSpec
from .guide_skill_catalog import TEST_DIFFICULTY_OPTIONS


_VOLATILE_KEYS = frozenset(
    {
        "launch_id",
        "run_key",
        "godot_run_key",
    }
)


@dataclass(frozen=True, slots=True)
class AuditClock:
    t: float = 0.0

    def now(self) -> float:
        return float(self.t)


@dataclass(frozen=True, slots=True)
class ChallengeSignature:
    code: str
    level: int
    ratio: float
    axes: tuple[tuple[str, float], ...]


@dataclass(frozen=True, slots=True)
class ReplaySignature:
    code: str
    family: str
    level: int
    seed: int
    signature: str


def seed_settings_activity_codes() -> tuple[str, ...]:
    """All activities with user-addressable manual seed slots."""

    ordered: list[str] = []
    seen: set[str] = set()
    for code, _label in (
        *TEST_DIFFICULTY_OPTIONS,
        ("adaptive_session", "Adaptive Session"),
        ("benchmark_battery", "Benchmark Battery"),
    ):
        token = str(code).strip().lower()
        if token == "" or token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return tuple(ordered)


def challenge_signature_for_code(code: str, level: int) -> ChallengeSignature:
    profile = difficulty_profile_for_code(code, level)
    axes = profile.axes
    axis_rows = tuple(
        (name, round(float(getattr(axes, name)), 6))
        for name in (
            "content_complexity",
            "time_pressure",
            "distractor_density",
            "multitask_concurrency",
            "memory_span_delay",
            "switch_frequency",
            "control_sensitivity",
            "spatial_ambiguity",
            "source_integration_depth",
        )
    )
    return ChallengeSignature(
        code=str(code),
        level=int(profile.level),
        ratio=round(float(profile.legacy_ratio), 6),
        axes=axis_rows,
    )


def difficulty_ladder_signatures(code: str) -> tuple[ChallengeSignature, ...]:
    return tuple(challenge_signature_for_code(code, level) for level in range(1, 11))


def official_runtime_replay_signature(
    code: str,
    *,
    seed: int,
    level: int,
) -> ReplaySignature:
    engine = build_official_test_runtime(
        clock=AuditClock(),
        test_code=code,
        seed=int(seed),
        difficulty=difficulty_ratio_for_level(code, level),
        duration_s=30.0,
        practice_enabled=False,
    )
    return ReplaySignature(
        code=str(code),
        family="official_test",
        level=int(level),
        seed=int(seed),
        signature=_engine_signature(engine),
    )


def canonical_drill_replay_signature(
    spec: CanonicalDrillSpec,
    *,
    seed: int,
    level: int,
) -> ReplaySignature:
    builder = spec.resolve_builder()
    kwargs: dict[str, Any] = {
        "clock": AuditClock(),
        "seed": int(seed),
        "difficulty": difficulty_ratio_for_level(spec.drill_code, level),
    }
    parameters = inspect.signature(builder).parameters
    if "mode" in parameters:
        kwargs["mode"] = "build"
    engine = builder(**kwargs)
    return ReplaySignature(
        code=spec.drill_code,
        family="canonical_drill",
        level=int(level),
        seed=int(seed),
        signature=_engine_signature(engine),
    )


def benchmark_plan_signature(*, seed: int) -> str:
    plan = build_benchmark_plan(clock=AuditClock(), seed=int(seed))
    normalized = {
        "code": plan.code,
        "run_seed": plan.run_seed,
        "probes": [
            {
                "probe_code": probe.probe_code,
                "seed": probe.seed,
                "difficulty_level": probe.difficulty_level,
                "duration_s": probe.duration_s,
            }
            for probe in plan.probes
        ],
    }
    return stable_signature(normalized)


def adaptive_plan_signature(*, seed: int, history: Iterable[object] = ()) -> str:
    plan = build_adaptive_session_plan(history=list(history), seed=int(seed), variant="adaptive")
    normalized = {
        "code": plan.code,
        "variant": plan.variant,
        "blocks": [
            {
                "code": block.drill_code,
                "seed": block.seed,
                "level": block.difficulty_level,
                "primitive": block.primitive_id,
                "mode": block.mode,
            }
            for block in plan.blocks
        ],
    }
    return stable_signature(normalized)


def builder_backed_canonical_drills() -> tuple[CanonicalDrillSpec, ...]:
    return tuple(
        spec
        for spec in CANONICAL_DRILL_REGISTRY
        if spec.builder_module and spec.builder_name
    )


def stable_signature(value: object) -> str:
    payload = json.dumps(_normalize(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def randomness_source_violations(root: Path) -> tuple[str, ...]:
    allowed_functions = {"_new_seed", "_random_seed"}
    violations: list[str] = []
    entropy_call = "System" + "Random("
    for path in sorted(Path(root).glob("cfast_trainer/**/*.py")):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        active_function = ""
        for idx, line in enumerate(lines, start=1):
            stripped = line.strip()
            if stripped.startswith("def ") and "(" in stripped:
                active_function = stripped.split("def ", 1)[1].split("(", 1)[0]
            if entropy_call not in line:
                continue
            if active_function in allowed_functions:
                continue
            violations.append(f"{path}:{idx}:{stripped}")
    return tuple(violations)


def _engine_signature(engine: object) -> str:
    starter = getattr(engine, "start_scored", None)
    if callable(starter):
        try:
            starter()
        except Exception:
            pass
    updater = getattr(engine, "update", None)
    if callable(updater):
        try:
            updater()
        except Exception:
            pass
    snapshotter = getattr(engine, "snapshot", None)
    snapshot = snapshotter() if callable(snapshotter) else None
    metrics = getattr(engine, "_result_metrics_overrides", None)
    return stable_signature(
        {
            "snapshot": snapshot,
            "metrics": metrics if isinstance(metrics, Mapping) else {},
        }
    )


def _normalize(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        normalized: dict[str, object] = {}
        for field in fields(value):
            if field.name in _VOLATILE_KEYS:
                continue
            normalized[field.name] = _normalize(getattr(value, field.name))
        return normalized
    if isinstance(value, Mapping):
        return {
            str(key): _normalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in _VOLATILE_KEYS
        }
    if isinstance(value, (tuple, list)):
        return [_normalize(item) for item in value]
    if isinstance(value, set | frozenset):
        return sorted(_normalize(item) for item in value)
    return repr(value)
