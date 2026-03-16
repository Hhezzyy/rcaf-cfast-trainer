from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .clock import Clock
from .content_variants import content_metadata_from_payload
from .cognitive_core import AttemptSummary, Phase, QuestionEvent, SeededRng, TestSnapshot, clamp01

_GRID_SIZE = 10
_ROW_LABELS = tuple("ABCDEFGHIJ")
_DEFAULT_CLOCK_BASE_S = 11 * 60 * 60
_CALLSIGN_POOL = (
    "LEEDS",
    "RAVEN",
    "SABRE",
    "VIKING",
    "EMBER",
    "ARROW",
    "NOMAD",
    "FALCON",
    "TALON",
    "MERLIN",
    "ORBIT",
    "EAGLE",
)
_AFFILIATIONS = ("friendly", "hostile", "unknown", "friendly", "unknown")
_HEADING_VECTORS: dict[str, tuple[int, int]] = {
    "N": (0, -1),
    "NE": (1, -1),
    "E": (1, 0),
    "SE": (1, 1),
    "S": (0, 1),
    "SW": (-1, 1),
    "W": (-1, 0),
    "NW": (-1, -1),
}
SA_CHANNEL_ORDER = ("pictorial", "coded", "numerical", "aural")


@dataclass(frozen=True, slots=True)
class SituationalAwarenessConfig:
    scored_duration_s: float = 25.0 * 60.0
    practice_scenarios: int = 3
    practice_scenario_duration_s: float = 45.0
    scored_scenario_duration_s: float = 60.0
    query_interval_min_s: int = 12
    query_interval_max_s: int = 18
    update_interval_s: float = 1.0
    min_track_count: int = 4
    max_track_count: int = 5


class SituationalAwarenessScenarioFamily(StrEnum):
    MERGE_CONFLICT = "merge_conflict"
    FUEL_PRIORITY = "fuel_priority"
    ROUTE_HANDOFF = "route_handoff"
    CHANNEL_WAYPOINT_CHANGE = "channel_waypoint_change"


class SituationalAwarenessQueryKind(StrEnum):
    CURRENT_LOCATION = "current_location"
    ORIGIN_LOCATION = "origin_location"
    FUTURE_LOCATION = "future_location"
    SAFE_TO_MOVE = "safe_to_move"
    STATUS_RECALL = "status_recall"


class SituationalAwarenessAnswerMode(StrEnum):
    GRID_CELL = "grid_cell"
    CHOICE = "choice"


SA_QUERY_KIND_ORDER = tuple(kind.value for kind in SituationalAwarenessQueryKind)


@dataclass(frozen=True, slots=True)
class SituationalAwarenessTrainingProfile:
    min_track_count: int | None = None
    max_track_count: int | None = None
    query_interval_min_s: int | None = None
    query_interval_max_s: int | None = None
    response_window_s: int | None = None
    update_time_scale: float = 1.0
    pressure_scale: float = 1.0
    contact_ttl_s: float | None = None
    cue_card_ttl_s: float | None = None
    top_strip_ttl_s: float | None = None
    visual_density_scale: float = 1.0
    audio_density_scale: float = 1.0
    allow_visible_answers: bool = False


@dataclass(frozen=True, slots=True)
class SituationalAwarenessTrainingSegment:
    label: str
    duration_s: float
    active_channels: tuple[str, ...] = SA_CHANNEL_ORDER
    active_query_kinds: tuple[str, ...] = SA_QUERY_KIND_ORDER
    scenario_families: tuple[SituationalAwarenessScenarioFamily, ...] = tuple(
        SituationalAwarenessScenarioFamily
    )
    focus_label: str = ""
    profile: SituationalAwarenessTrainingProfile = SituationalAwarenessTrainingProfile()


@dataclass(frozen=True, slots=True)
class SituationalAwarenessWaypoint:
    name: str
    x: int
    y: int


@dataclass(frozen=True, slots=True)
class SituationalAwarenessVisibleContact:
    callsign: str
    affiliation: str
    x: float
    y: float
    cell_label: str
    heading: str
    fade: float


@dataclass(frozen=True, slots=True)
class SituationalAwarenessCueCard:
    callsign: str
    affiliation: str
    next_waypoint: str
    eta_clock_text: str
    altitude_text: str
    channel_text: str
    fade: float


@dataclass(frozen=True, slots=True)
class SituationalAwarenessAnswerChoice:
    code: int
    text: str


@dataclass(frozen=True, slots=True)
class SituationalAwarenessActiveQuery:
    query_id: int
    kind: SituationalAwarenessQueryKind
    answer_mode: SituationalAwarenessAnswerMode
    prompt: str
    correct_answer_token: str
    expires_in_s: float
    subject_callsign: str | None = None
    future_offset_s: int | None = None
    answer_choices: tuple[SituationalAwarenessAnswerChoice, ...] = ()


@dataclass(frozen=True, slots=True)
class SituationalAwarenessPayload:
    scenario_family: SituationalAwarenessScenarioFamily
    scenario_label: str
    scenario_index: int
    scenario_total: int
    active_channels: tuple[str, ...]
    active_query_kinds: tuple[str, ...]
    focus_label: str
    segment_label: str
    segment_index: int
    segment_total: int
    segment_time_remaining_s: float
    scenario_elapsed_s: float
    scenario_time_remaining_s: float
    next_query_in_s: float | None
    visible_contacts: tuple[SituationalAwarenessVisibleContact, ...]
    cue_card: SituationalAwarenessCueCard | None
    waypoints: tuple[SituationalAwarenessWaypoint, ...]
    top_strip_text: str
    top_strip_fade: float
    display_clock_text: str
    active_query: SituationalAwarenessActiveQuery | None
    answer_mode: SituationalAwarenessAnswerMode | None
    correct_answer_token: str | None
    announcement_token: tuple[object, ...] | None
    announcement_lines: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _AssetSeed:
    index: int
    callsign: str
    affiliation: str
    x: float
    y: float
    heading: str
    speed_cells_per_min: int
    channel: int
    altitude_fl: int
    waypoint: str


@dataclass(frozen=True, slots=True)
class _UpdateEvent:
    at_s: int
    callsign: str
    modality: str
    line: str
    heading: str | None = None
    speed_cells_per_min: int | None = None
    channel: int | None = None
    altitude_fl: int | None = None
    waypoint: str | None = None


@dataclass(frozen=True, slots=True)
class _ContactSweepEvent:
    at_s: int
    callsigns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _QuerySlot:
    at_s: int
    kind: SituationalAwarenessQueryKind
    future_offset_s: int | None = None


@dataclass(frozen=True, slots=True)
class _ScenarioPlan:
    family: SituationalAwarenessScenarioFamily
    label: str
    duration_s: int
    waypoints: tuple[SituationalAwarenessWaypoint, ...]
    asset_seeds: tuple[_AssetSeed, ...]
    updates: tuple[_UpdateEvent, ...]
    contact_sweeps: tuple[_ContactSweepEvent, ...]
    query_slots: tuple[_QuerySlot, ...]
    active_query_kinds: tuple[str, ...]
    response_window_s: int | None
    display_clock_base_s: int
    context: dict[str, Any]


@dataclass(slots=True)
class _LiveAssetState:
    index: int
    callsign: str
    affiliation: str
    x: float
    y: float
    origin_cell: str
    heading: str
    speed_cells_per_min: int
    channel: int
    altitude_fl: int
    waypoint: str

    def copy(self) -> _LiveAssetState:
        return _LiveAssetState(
            index=self.index,
            callsign=self.callsign,
            affiliation=self.affiliation,
            x=float(self.x),
            y=float(self.y),
            origin_cell=str(self.origin_cell),
            heading=str(self.heading),
            speed_cells_per_min=int(self.speed_cells_per_min),
            channel=int(self.channel),
            altitude_fl=int(self.altitude_fl),
            waypoint=str(self.waypoint),
        )


@dataclass(frozen=True, slots=True)
class _AssetMemorySample:
    x: float
    y: float
    heading: str
    channel: int
    altitude_fl: int
    waypoint: str


@dataclass(slots=True)
class _VisibleContactState:
    callsign: str
    shown_at_s: int
    visible_until_s: int


@dataclass(slots=True)
class _CueCardState:
    callsign: str
    shown_at_s: int
    visible_until_s: int


@dataclass(slots=True)
class _TopStripState:
    text: str
    shown_at_s: int
    visible_until_s: int


@dataclass(frozen=True, slots=True)
class _ActiveQueryState:
    query_id: int
    asked_at_s: int
    expires_at_s: int
    kind: SituationalAwarenessQueryKind
    answer_mode: SituationalAwarenessAnswerMode
    prompt: str
    correct_answer_token: str
    subject_callsign: str | None = None
    future_offset_s: int | None = None
    answer_choices: tuple[SituationalAwarenessAnswerChoice, ...] = ()


def _clamp_grid(value: float) -> int:
    if value <= 0:
        return 0
    if value >= (_GRID_SIZE - 1):
        return _GRID_SIZE - 1
    return int(round(float(value)))


def cell_label_from_xy(x: float, y: float) -> str:
    ix = _clamp_grid(x)
    iy = _clamp_grid(y)
    return f"{_ROW_LABELS[iy]}{ix}"


def cell_xy_from_label(label: str) -> tuple[int, int] | None:
    token = str(label).strip().upper().replace(",", "")
    if len(token) < 2:
        return None
    row_token = token[0]
    col_token = token[1:]
    if row_token not in _ROW_LABELS or not col_token.isdigit():
        return None
    x = int(col_token)
    y = _ROW_LABELS.index(row_token)
    if x < 0 or x >= _GRID_SIZE:
        return None
    return x, y


def normalize_grid_cell_token(raw: str) -> str | None:
    token = str(raw).strip().upper().replace(" ", "").replace(",", "")
    if len(token) < 2:
        return None
    row = token[0]
    col = token[1:]
    if row not in _ROW_LABELS or not col.isdigit():
        return None
    x = int(col)
    if x < 0 or x >= _GRID_SIZE:
        return None
    return f"{row}{x}"


def _normalize_channels(channels: Sequence[str] | None) -> tuple[str, ...]:
    if not channels:
        return SA_CHANNEL_ORDER
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in channels:
        token = str(raw).strip().lower()
        if token not in SA_CHANNEL_ORDER or token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return tuple(ordered) if ordered else SA_CHANNEL_ORDER


def _normalize_query_kind_names(kinds: Sequence[str] | None) -> tuple[str, ...]:
    if not kinds:
        return SA_QUERY_KIND_ORDER
    valid = set(SA_QUERY_KIND_ORDER)
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in kinds:
        token = str(raw).strip().lower()
        if token not in valid or token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return tuple(ordered) if ordered else SA_QUERY_KIND_ORDER


def _normalize_heading(raw: str) -> str:
    token = str(raw).strip().upper()
    return token if token in _HEADING_VECTORS else "E"


def _sign(value: float) -> int:
    if value > 0.05:
        return 1
    if value < -0.05:
        return -1
    return 0


def _heading_from_delta(dx: float, dy: float) -> str:
    sx = _sign(dx)
    sy = _sign(dy)
    for heading, vector in _HEADING_VECTORS.items():
        if vector == (sx, sy):
            return heading
    return "E"


def _clock_text(total_s: int) -> str:
    seconds = int(total_s) % (24 * 60 * 60)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _family_label(family: SituationalAwarenessScenarioFamily) -> str:
    if family is SituationalAwarenessScenarioFamily.MERGE_CONFLICT:
        return "Conflict / Safety"
    if family is SituationalAwarenessScenarioFamily.FUEL_PRIORITY:
        return "Status Update"
    if family is SituationalAwarenessScenarioFamily.ROUTE_HANDOFF:
        return "Route Handoff"
    return "Channel / Waypoint"


class SituationalAwarenessScenarioGenerator:
    def __init__(self, *, seed: int, config: SituationalAwarenessConfig) -> None:
        self._rng = SeededRng(seed)
        self._config = config
        self._last_family: SituationalAwarenessScenarioFamily | None = None

    def next_scenario(
        self,
        *,
        difficulty: float,
        scenario_index: int,
        total_scenarios: int,
        duration_s: int,
        practice_focus: SituationalAwarenessQueryKind | None = None,
        allowed_families: Sequence[SituationalAwarenessScenarioFamily] | None = None,
        active_query_kinds: Sequence[str] | None = None,
        training_profile: SituationalAwarenessTrainingProfile | None = None,
    ) -> _ScenarioPlan:
        family = self._pick_family(
            practice_focus=practice_focus,
            allowed_families=allowed_families,
        )
        active_kind_names = _normalize_query_kind_names(active_query_kinds)
        track_count = self._pick_track_count(difficulty=difficulty, training_profile=training_profile)
        if family is SituationalAwarenessScenarioFamily.MERGE_CONFLICT:
            return self._build_merge_conflict(
                track_count=track_count,
                scenario_index=scenario_index,
                total_scenarios=total_scenarios,
                duration_s=duration_s,
                difficulty=difficulty,
                practice_focus=practice_focus,
                active_query_kinds=active_kind_names,
                training_profile=training_profile,
            )
        if family is SituationalAwarenessScenarioFamily.FUEL_PRIORITY:
            return self._build_status_update(
                track_count=track_count,
                scenario_index=scenario_index,
                total_scenarios=total_scenarios,
                duration_s=duration_s,
                difficulty=difficulty,
                practice_focus=practice_focus,
                active_query_kinds=active_kind_names,
                training_profile=training_profile,
            )
        if family is SituationalAwarenessScenarioFamily.ROUTE_HANDOFF:
            return self._build_route_handoff(
                track_count=track_count,
                scenario_index=scenario_index,
                total_scenarios=total_scenarios,
                duration_s=duration_s,
                difficulty=difficulty,
                practice_focus=practice_focus,
                active_query_kinds=active_kind_names,
                training_profile=training_profile,
            )
        return self._build_channel_waypoint_change(
            track_count=track_count,
            scenario_index=scenario_index,
            total_scenarios=total_scenarios,
            duration_s=duration_s,
            difficulty=difficulty,
            practice_focus=practice_focus,
            active_query_kinds=active_kind_names,
            training_profile=training_profile,
        )

    def _pick_family(
        self,
        *,
        practice_focus: SituationalAwarenessQueryKind | None,
        allowed_families: Sequence[SituationalAwarenessScenarioFamily] | None = None,
    ) -> SituationalAwarenessScenarioFamily:
        allowed = tuple(allowed_families or tuple(SituationalAwarenessScenarioFamily))
        if not allowed:
            allowed = tuple(SituationalAwarenessScenarioFamily)
        if practice_focus is SituationalAwarenessQueryKind.SAFE_TO_MOVE:
            family = SituationalAwarenessScenarioFamily.MERGE_CONFLICT
        elif practice_focus is SituationalAwarenessQueryKind.FUTURE_LOCATION:
            family = SituationalAwarenessScenarioFamily.ROUTE_HANDOFF
        elif practice_focus is SituationalAwarenessQueryKind.STATUS_RECALL:
            family = SituationalAwarenessScenarioFamily.FUEL_PRIORITY
        elif practice_focus in (
            SituationalAwarenessQueryKind.CURRENT_LOCATION,
            SituationalAwarenessQueryKind.ORIGIN_LOCATION,
        ):
            family = SituationalAwarenessScenarioFamily.CHANNEL_WAYPOINT_CHANGE
        else:
            family = self._rng.choice(allowed)
            if self._last_family is family and len(allowed) > 1:
                family = self._rng.choice(tuple(item for item in allowed if item is not family))
        if family not in allowed:
            family = allowed[0]
        self._last_family = family
        return family

    def _pick_track_count(
        self,
        *,
        difficulty: float,
        training_profile: SituationalAwarenessTrainingProfile | None = None,
    ) -> int:
        if training_profile is not None:
            low = (
                max(3, int(training_profile.min_track_count))
                if training_profile.min_track_count is not None
                else max(3, int(self._config.min_track_count))
            )
            high = (
                max(low, int(training_profile.max_track_count))
                if training_profile.max_track_count is not None
                else max(low, int(self._config.max_track_count))
            )
        else:
            low = max(3, int(self._config.min_track_count))
            high = max(low, int(self._config.max_track_count))
        if high == low:
            return low
        return high if clamp01(difficulty) >= 0.66 else low

    def _sample_callsigns(self, count: int) -> list[str]:
        return self._rng.sample(_CALLSIGN_POOL, k=count)

    def _make_asset_seed(
        self,
        *,
        index: int,
        callsign: str,
        affiliation: str,
        x: float,
        y: float,
        speed: int,
        channel: int,
        altitude_fl: int,
        waypoint: str,
        heading: str | None = None,
    ) -> _AssetSeed:
        target = cell_xy_from_label(waypoint)
        if heading is None and target is not None:
            heading = _heading_from_delta(float(target[0]) - float(x), float(target[1]) - float(y))
        return _AssetSeed(
            index=index,
            callsign=callsign,
            affiliation=affiliation,
            x=float(x),
            y=float(y),
            heading=_normalize_heading(heading or "E"),
            speed_cells_per_min=max(1, int(speed)),
            channel=max(1, min(9, int(channel))),
            altitude_fl=max(120, int(altitude_fl)),
            waypoint=normalize_grid_cell_token(waypoint) or "A0",
        )

    def _make_query_slots(
        self,
        *,
        duration_s: int,
        family: SituationalAwarenessScenarioFamily,
        difficulty: float,
        practice_focus: SituationalAwarenessQueryKind | None,
        active_query_kinds: Sequence[str],
        training_profile: SituationalAwarenessTrainingProfile | None,
    ) -> tuple[_QuerySlot, ...]:
        slots: list[_QuerySlot] = []
        query_interval_min = (
            int(training_profile.query_interval_min_s)
            if training_profile is not None and training_profile.query_interval_min_s is not None
            else int(self._config.query_interval_min_s)
        )
        query_interval_max = (
            int(training_profile.query_interval_max_s)
            if training_profile is not None and training_profile.query_interval_max_s is not None
            else int(self._config.query_interval_max_s)
        )
        query_interval_min = max(4, query_interval_min)
        query_interval_max = max(query_interval_min, query_interval_max)
        at_s = self._rng.randint(query_interval_min, query_interval_max)
        sequence = self._query_sequence_for_family(
            family=family,
            difficulty=difficulty,
            practice_focus=practice_focus,
            active_query_kinds=active_query_kinds,
            training_profile=training_profile,
        )
        seq_index = 0
        while at_s < max(18, duration_s - 6):
            kind = sequence[seq_index % len(sequence)]
            seq_index += 1
            future_offset_s = None
            if kind is SituationalAwarenessQueryKind.FUTURE_LOCATION:
                max_offset = max(12, min(24, duration_s - at_s - 4))
                future_offset_s = max(8, min(max_offset, 12 + self._rng.randint(0, 8)))
            slots.append(_QuerySlot(at_s=at_s, kind=kind, future_offset_s=future_offset_s))
            at_s += self._rng.randint(query_interval_min, query_interval_max)
        return tuple(slots)

    def _query_sequence_for_family(
        self,
        *,
        family: SituationalAwarenessScenarioFamily,
        difficulty: float,
        practice_focus: SituationalAwarenessQueryKind | None,
        active_query_kinds: Sequence[str],
        training_profile: SituationalAwarenessTrainingProfile | None,
    ) -> tuple[SituationalAwarenessQueryKind, ...]:
        active_kind_names = _normalize_query_kind_names(active_query_kinds)
        if tuple(active_kind_names) != SA_QUERY_KIND_ORDER:
            return tuple(SituationalAwarenessQueryKind(name) for name in active_kind_names)
        if practice_focus is not None:
            return (practice_focus, practice_focus)
        pressure_scale = 1.0 if training_profile is None else max(0.5, float(training_profile.pressure_scale))
        high_pressure = clamp01(difficulty) >= 0.7 or pressure_scale > 1.05
        mapping = {
            SituationalAwarenessScenarioFamily.MERGE_CONFLICT: (
                SituationalAwarenessQueryKind.CURRENT_LOCATION,
                SituationalAwarenessQueryKind.SAFE_TO_MOVE,
                SituationalAwarenessQueryKind.FUTURE_LOCATION,
            ),
            SituationalAwarenessScenarioFamily.FUEL_PRIORITY: (
                SituationalAwarenessQueryKind.STATUS_RECALL,
                SituationalAwarenessQueryKind.CURRENT_LOCATION,
                SituationalAwarenessQueryKind.SAFE_TO_MOVE,
            ),
            SituationalAwarenessScenarioFamily.ROUTE_HANDOFF: (
                SituationalAwarenessQueryKind.FUTURE_LOCATION,
                SituationalAwarenessQueryKind.STATUS_RECALL,
                SituationalAwarenessQueryKind.ORIGIN_LOCATION,
            ),
            SituationalAwarenessScenarioFamily.CHANNEL_WAYPOINT_CHANGE: (
                SituationalAwarenessQueryKind.CURRENT_LOCATION,
                SituationalAwarenessQueryKind.STATUS_RECALL,
                SituationalAwarenessQueryKind.ORIGIN_LOCATION,
            ),
        }
        base = mapping[family]
        if not high_pressure:
            return base
        return base + (
            SituationalAwarenessQueryKind.FUTURE_LOCATION,
            SituationalAwarenessQueryKind.CURRENT_LOCATION,
        )

    def _schedule_updates(
        self,
        updates: tuple[_UpdateEvent, ...],
        *,
        duration_s: int,
        training_profile: SituationalAwarenessTrainingProfile | None,
    ) -> tuple[_UpdateEvent, ...]:
        if training_profile is None:
            return updates
        scale = max(0.45, float(training_profile.update_time_scale))
        if abs(scale - 1.0) < 1e-6:
            return updates
        scheduled: list[_UpdateEvent] = []
        seen: set[tuple[int, str, str]] = set()
        for update in updates:
            at_s = max(4, min(duration_s - 4, int(round(float(update.at_s) * scale))))
            key = (at_s, update.callsign, update.line)
            if key in seen:
                at_s = min(duration_s - 4, at_s + 1)
                key = (at_s, update.callsign, update.line)
            seen.add(key)
            scheduled.append(
                _UpdateEvent(
                    at_s=at_s,
                    callsign=update.callsign,
                    modality=update.modality,
                    line=update.line,
                    heading=update.heading,
                    speed_cells_per_min=update.speed_cells_per_min,
                    channel=update.channel,
                    altitude_fl=update.altitude_fl,
                    waypoint=update.waypoint,
                )
            )
        scheduled.sort(key=lambda item: (item.at_s, item.callsign))
        return tuple(scheduled)

    def _make_contact_sweeps(
        self,
        *,
        callsigns: Sequence[str],
        duration_s: int,
        training_profile: SituationalAwarenessTrainingProfile | None,
    ) -> tuple[_ContactSweepEvent, ...]:
        scale = 1.0 if training_profile is None else max(0.5, float(training_profile.update_time_scale))
        visual_density = (
            1.0 if training_profile is None else max(0.5, float(training_profile.visual_density_scale))
        )
        interval = max(3, min(8, int(round((5.0 * scale) / visual_density))))
        events: list[_ContactSweepEvent] = []
        idx = 0
        at_s = 2
        while at_s < max(6, duration_s - 2):
            count = 1
            if len(callsigns) >= 4 and ((idx + 1) % 3 == 0):
                count = 2
            picked = tuple(callsigns[(idx + offset) % len(callsigns)] for offset in range(count))
            events.append(_ContactSweepEvent(at_s=at_s, callsigns=picked))
            at_s += interval
            idx += 1
        return tuple(events)

    def _scenario_plan(
        self,
        *,
        family: SituationalAwarenessScenarioFamily,
        scenario_index: int,
        total_scenarios: int,
        duration_s: int,
        difficulty: float,
        practice_focus: SituationalAwarenessQueryKind | None,
        active_query_kinds: tuple[str, ...],
        training_profile: SituationalAwarenessTrainingProfile | None,
        asset_seeds: tuple[_AssetSeed, ...],
        updates: tuple[_UpdateEvent, ...],
        context: dict[str, Any],
    ) -> _ScenarioPlan:
        display_clock_base_s = _DEFAULT_CLOCK_BASE_S + (scenario_index - 1) * 95
        return _ScenarioPlan(
            family=family,
            label=f"{_family_label(family)} {scenario_index}/{total_scenarios}",
            duration_s=duration_s,
            waypoints=(),
            asset_seeds=asset_seeds,
            updates=self._schedule_updates(
                updates,
                duration_s=duration_s,
                training_profile=training_profile,
            ),
            contact_sweeps=self._make_contact_sweeps(
                callsigns=tuple(seed.callsign for seed in asset_seeds),
                duration_s=duration_s,
                training_profile=training_profile,
            ),
            query_slots=self._make_query_slots(
                duration_s=duration_s,
                family=family,
                difficulty=difficulty,
                practice_focus=practice_focus,
                active_query_kinds=active_query_kinds,
                training_profile=training_profile,
            ),
            active_query_kinds=active_query_kinds,
            response_window_s=(
                None
                if training_profile is None or training_profile.response_window_s is None
                else int(training_profile.response_window_s)
            ),
            display_clock_base_s=display_clock_base_s,
            context=context,
        )

    def _build_merge_conflict(
        self,
        *,
        track_count: int,
        scenario_index: int,
        total_scenarios: int,
        duration_s: int,
        difficulty: float,
        practice_focus: SituationalAwarenessQueryKind | None,
        active_query_kinds: tuple[str, ...],
        training_profile: SituationalAwarenessTrainingProfile | None,
    ) -> _ScenarioPlan:
        callsigns = self._sample_callsigns(max(track_count, 4))
        seeds = [
            self._make_asset_seed(
                index=1,
                callsign=callsigns[0],
                affiliation="friendly",
                x=1.0,
                y=4.0,
                speed=5,
                channel=2,
                altitude_fl=220,
                waypoint="F4",
            ),
            self._make_asset_seed(
                index=2,
                callsign=callsigns[1],
                affiliation="hostile",
                x=8.0,
                y=4.0,
                speed=5,
                channel=4,
                altitude_fl=240,
                waypoint="D4",
            ),
            self._make_asset_seed(
                index=3,
                callsign=callsigns[2],
                affiliation="unknown",
                x=2.0,
                y=1.0,
                speed=3,
                channel=1,
                altitude_fl=300,
                waypoint="E4",
            ),
            self._make_asset_seed(
                index=4,
                callsign=callsigns[3],
                affiliation="friendly",
                x=7.0,
                y=8.0,
                speed=3,
                channel=3,
                altitude_fl=180,
                waypoint="F6",
            ),
        ]
        if track_count >= 5:
            seeds.append(
                self._make_asset_seed(
                    index=5,
                    callsign=callsigns[4],
                    affiliation="unknown",
                    x=0.0,
                    y=8.0,
                    speed=2,
                    channel=5,
                    altitude_fl=260,
                    waypoint="C8",
                )
            )
        updates = [
            _UpdateEvent(
                8,
                callsigns[0],
                "audio_plus_visual",
                f"{callsigns[0]} channel 3, direct F5.",
                heading="NE",
                channel=3,
                waypoint="F5",
            ),
            _UpdateEvent(
                16,
                callsigns[1],
                "audio_only",
                f"{callsigns[1]} maintain D4, hold FL230.",
                altitude_fl=230,
                waypoint="D4",
                heading="W",
            ),
            _UpdateEvent(
                24,
                callsigns[2],
                "audio_plus_visual",
                f"{callsigns[2]} turn south, direct E6.",
                heading="S",
                waypoint="E6",
            ),
        ]
        if track_count >= 4:
            updates.append(
                _UpdateEvent(
                    32,
                    callsigns[3],
                    "visual_only",
                    f"{callsigns[3]} continue F6.",
                )
            )
        return self._scenario_plan(
            family=SituationalAwarenessScenarioFamily.MERGE_CONFLICT,
            scenario_index=scenario_index,
            total_scenarios=total_scenarios,
            duration_s=duration_s,
            difficulty=difficulty,
            practice_focus=practice_focus,
            active_query_kinds=active_query_kinds,
            training_profile=training_profile,
            asset_seeds=tuple(seeds[:track_count]),
            updates=tuple(updates),
            context={
                "priority_callsign": callsigns[0],
                "conflict_pair": (callsigns[0], callsigns[1]),
                "merge_cell": "E4",
            },
        )

    def _build_status_update(
        self,
        *,
        track_count: int,
        scenario_index: int,
        total_scenarios: int,
        duration_s: int,
        difficulty: float,
        practice_focus: SituationalAwarenessQueryKind | None,
        active_query_kinds: tuple[str, ...],
        training_profile: SituationalAwarenessTrainingProfile | None,
    ) -> _ScenarioPlan:
        callsigns = self._sample_callsigns(max(track_count, 4))
        seeds = [
            self._make_asset_seed(
                index=1,
                callsign=callsigns[0],
                affiliation="friendly",
                x=1.0,
                y=2.0,
                speed=4,
                channel=1,
                altitude_fl=210,
                waypoint="E2",
            ),
            self._make_asset_seed(
                index=2,
                callsign=callsigns[1],
                affiliation="unknown",
                x=8.0,
                y=2.0,
                speed=4,
                channel=3,
                altitude_fl=250,
                waypoint="D2",
            ),
            self._make_asset_seed(
                index=3,
                callsign=callsigns[2],
                affiliation="hostile",
                x=2.0,
                y=7.0,
                speed=3,
                channel=4,
                altitude_fl=300,
                waypoint="E7",
            ),
            self._make_asset_seed(
                index=4,
                callsign=callsigns[3],
                affiliation="friendly",
                x=7.0,
                y=8.0,
                speed=2,
                channel=2,
                altitude_fl=190,
                waypoint="G6",
            ),
        ]
        if track_count >= 5:
            seeds.append(
                self._make_asset_seed(
                    index=5,
                    callsign=callsigns[4],
                    affiliation="unknown",
                    x=5.0,
                    y=8.0,
                    speed=2,
                    channel=5,
                    altitude_fl=280,
                    waypoint="C8",
                )
            )
        updates = [
            _UpdateEvent(
                6,
                callsigns[0],
                "audio_plus_visual",
                f"{callsigns[0]} direct F3, channel 2.",
                heading="E",
                channel=2,
                waypoint="F3",
            ),
            _UpdateEvent(
                14,
                callsigns[1],
                "audio_plus_visual",
                f"{callsigns[1]} hold FL260, next waypoint C2.",
                altitude_fl=260,
                waypoint="C2",
                heading="W",
            ),
            _UpdateEvent(
                22,
                callsigns[2],
                "audio_only",
                f"{callsigns[2]} channel 1, direct E5.",
                channel=1,
                heading="N",
                waypoint="E5",
            ),
        ]
        if track_count >= 4:
            updates.append(
                _UpdateEvent(
                    30,
                    callsigns[3],
                    "visual_only",
                    f"{callsigns[3]} continue G6.",
                )
            )
        return self._scenario_plan(
            family=SituationalAwarenessScenarioFamily.FUEL_PRIORITY,
            scenario_index=scenario_index,
            total_scenarios=total_scenarios,
            duration_s=duration_s,
            difficulty=difficulty,
            practice_focus=practice_focus,
            active_query_kinds=active_query_kinds,
            training_profile=training_profile,
            asset_seeds=tuple(seeds[:track_count]),
            updates=tuple(updates),
            context={"priority_callsign": callsigns[0], "merge_cell": "F3"},
        )

    def _build_route_handoff(
        self,
        *,
        track_count: int,
        scenario_index: int,
        total_scenarios: int,
        duration_s: int,
        difficulty: float,
        practice_focus: SituationalAwarenessQueryKind | None,
        active_query_kinds: tuple[str, ...],
        training_profile: SituationalAwarenessTrainingProfile | None,
    ) -> _ScenarioPlan:
        callsigns = self._sample_callsigns(max(track_count, 4))
        seeds = [
            self._make_asset_seed(
                index=1,
                callsign=callsigns[0],
                affiliation="friendly",
                x=1.0,
                y=7.0,
                speed=4,
                channel=1,
                altitude_fl=250,
                waypoint="F4",
            ),
            self._make_asset_seed(
                index=2,
                callsign=callsigns[1],
                affiliation="unknown",
                x=7.0,
                y=1.0,
                speed=3,
                channel=2,
                altitude_fl=240,
                waypoint="D4",
            ),
            self._make_asset_seed(
                index=3,
                callsign=callsigns[2],
                affiliation="friendly",
                x=2.0,
                y=2.0,
                speed=2,
                channel=4,
                altitude_fl=190,
                waypoint="F2",
            ),
            self._make_asset_seed(
                index=4,
                callsign=callsigns[3],
                affiliation="hostile",
                x=8.0,
                y=8.0,
                speed=2,
                channel=3,
                altitude_fl=320,
                waypoint="D8",
            ),
        ]
        if track_count >= 5:
            seeds.append(
                self._make_asset_seed(
                    index=5,
                    callsign=callsigns[4],
                    affiliation="unknown",
                    x=4.0,
                    y=8.0,
                    speed=2,
                    channel=5,
                    altitude_fl=280,
                    waypoint="B8",
                )
            )
        updates = [
            _UpdateEvent(
                10,
                callsigns[0],
                "audio_plus_visual",
                f"{callsigns[0]} handoff channel 3, direct G5.",
                channel=3,
                heading="NE",
                waypoint="G5",
            ),
            _UpdateEvent(
                18,
                callsigns[0],
                "audio_plus_visual",
                f"{callsigns[0]} continue G7.",
                heading="S",
                waypoint="G7",
            ),
            _UpdateEvent(
                26,
                callsigns[1],
                "audio_only",
                f"{callsigns[1]} hold FL250, next waypoint D5.",
                altitude_fl=250,
                waypoint="D5",
                heading="S",
            ),
            _UpdateEvent(
                34,
                callsigns[2],
                "visual_only",
                f"{callsigns[2]} continue F2.",
            ),
        ]
        return self._scenario_plan(
            family=SituationalAwarenessScenarioFamily.ROUTE_HANDOFF,
            scenario_index=scenario_index,
            total_scenarios=total_scenarios,
            duration_s=duration_s,
            difficulty=difficulty,
            practice_focus=practice_focus,
            active_query_kinds=active_query_kinds,
            training_profile=training_profile,
            asset_seeds=tuple(seeds[:track_count]),
            updates=tuple(updates),
            context={"priority_callsign": callsigns[0], "merge_cell": "G5"},
        )

    def _build_channel_waypoint_change(
        self,
        *,
        track_count: int,
        scenario_index: int,
        total_scenarios: int,
        duration_s: int,
        difficulty: float,
        practice_focus: SituationalAwarenessQueryKind | None,
        active_query_kinds: tuple[str, ...],
        training_profile: SituationalAwarenessTrainingProfile | None,
    ) -> _ScenarioPlan:
        callsigns = self._sample_callsigns(max(track_count, 4))
        seeds = [
            self._make_asset_seed(
                index=1,
                callsign=callsigns[0],
                affiliation="friendly",
                x=2.0,
                y=2.0,
                speed=3,
                channel=1,
                altitude_fl=210,
                waypoint="G2",
            ),
            self._make_asset_seed(
                index=2,
                callsign=callsigns[1],
                affiliation="unknown",
                x=7.0,
                y=6.0,
                speed=4,
                channel=4,
                altitude_fl=250,
                waypoint="C6",
            ),
            self._make_asset_seed(
                index=3,
                callsign=callsigns[2],
                affiliation="hostile",
                x=1.0,
                y=8.0,
                speed=3,
                channel=3,
                altitude_fl=300,
                waypoint="E5",
            ),
            self._make_asset_seed(
                index=4,
                callsign=callsigns[3],
                affiliation="friendly",
                x=8.0,
                y=1.0,
                speed=2,
                channel=2,
                altitude_fl=270,
                waypoint="D4",
            ),
        ]
        if track_count >= 5:
            seeds.append(
                self._make_asset_seed(
                    index=5,
                    callsign=callsigns[4],
                    affiliation="unknown",
                    x=4.0,
                    y=5.0,
                    speed=2,
                    channel=5,
                    altitude_fl=230,
                    waypoint="B5",
                )
            )
        updates = [
            _UpdateEvent(
                8,
                callsigns[1],
                "audio_plus_visual",
                f"{callsigns[1]} channel 2, direct E6.",
                channel=2,
                waypoint="E6",
                heading="W",
            ),
            _UpdateEvent(
                16,
                callsigns[0],
                "audio_plus_visual",
                f"{callsigns[0]} hold FL230, continue G2.",
                altitude_fl=230,
                waypoint="G2",
                heading="E",
            ),
            _UpdateEvent(
                24,
                callsigns[2],
                "audio_only",
                f"{callsigns[2]} direct C6, channel 1.",
                channel=1,
                waypoint="C6",
                heading="NE",
            ),
        ]
        if track_count >= 4:
            updates.append(
                _UpdateEvent(
                    32,
                    callsigns[3],
                    "visual_only",
                    f"{callsigns[3]} continue D4.",
                )
            )
        return self._scenario_plan(
            family=SituationalAwarenessScenarioFamily.CHANNEL_WAYPOINT_CHANGE,
            scenario_index=scenario_index,
            total_scenarios=total_scenarios,
            duration_s=duration_s,
            difficulty=difficulty,
            practice_focus=practice_focus,
            active_query_kinds=active_query_kinds,
            training_profile=training_profile,
            asset_seeds=tuple(seeds[:track_count]),
            updates=tuple(updates),
            context={"priority_callsign": callsigns[1], "merge_cell": "E6"},
        )


class SituationalAwarenessTest:
    def __init__(
        self,
        *,
        clock: Clock,
        seed: int,
        difficulty: float = 0.5,
        config: SituationalAwarenessConfig | None = None,
        title: str = "Situational Awareness",
        practice_segments: Sequence[SituationalAwarenessTrainingSegment] | None = None,
        scored_segments: Sequence[SituationalAwarenessTrainingSegment] | None = None,
    ) -> None:
        self._clock = clock
        self._seed = int(seed)
        self._difficulty = clamp01(difficulty)
        self._config = config or SituationalAwarenessConfig()
        self._generator = SituationalAwarenessScenarioGenerator(seed=self._seed, config=self._config)
        self._title = str(title)
        self._instructions = [
            f"{self._title} Test" if "Test" not in self._title else self._title,
            "",
            "Build the picture from short radio calls, brief cue-card flashes, and fading grid sweeps.",
            "Most of the state is hidden: keep track of where assets are now, where they came from, where they will be later, and whether a move is safe.",
            "The map keeps updating while questions are open. Misses do not pause the world.",
            "",
            "Answer modes:",
            "- Grid cell: click a cell or type row+column, then Enter",
            "- Choice: click an option or press 1-4, then Enter",
            "",
            "Practice runs three guided scenarios before the 25-minute timed block.",
        ]

        self._custom_segment_layout = practice_segments is not None or scored_segments is not None
        self._practice_segments = self._normalize_segments(practice_segments)
        self._scored_segments = self._normalize_segments(scored_segments)

        self._phase = Phase.INSTRUCTIONS
        self._events: list[QuestionEvent] = []
        self._practice_feedback: str | None = None

        self._practice_scenario_index = 0
        self._practice_total = max(0, int(self._config.practice_scenarios))
        self._practice_focus_order = (
            SituationalAwarenessQueryKind.CURRENT_LOCATION,
            SituationalAwarenessQueryKind.STATUS_RECALL,
            SituationalAwarenessQueryKind.SAFE_TO_MOVE,
        )

        self._scored_started_at_s: float | None = None
        self._scored_attempted = 0
        self._scored_correct = 0
        self._scored_total_score = 0.0
        self._scored_max_score = 0.0

        self._scenario_started_at_s: float | None = None
        self._scenario_plan: _ScenarioPlan | None = None
        self._scenario_index = 0
        self._scenario_total = 0
        self._processed_ticks = 0
        self._update_cursor = 0
        self._contact_sweep_cursor = 0
        self._query_slot_cursor = 0
        self._current_query: _ActiveQueryState | None = None
        self._live_assets: dict[str, _LiveAssetState] = {}
        self._asset_history: dict[str, list[_AssetMemorySample]] = {}
        self._visible_contacts: dict[str, _VisibleContactState] = {}
        self._cue_card_state: _CueCardState | None = None
        self._top_strip_state: _TopStripState | None = None
        self._announcement_serial = 0
        self._announcement_token: tuple[object, ...] | None = None
        self._announcement_lines: tuple[str, ...] = ()
        self._active_segments: tuple[SituationalAwarenessTrainingSegment, ...] = ()
        self._active_segment_index = 0
        self._current_segment: SituationalAwarenessTrainingSegment | None = None

    @property
    def phase(self) -> Phase:
        return self._phase

    @classmethod
    def _normalize_segments(
        cls,
        segments: Sequence[SituationalAwarenessTrainingSegment] | None,
    ) -> tuple[SituationalAwarenessTrainingSegment, ...]:
        if not segments:
            return ()
        normalized: list[SituationalAwarenessTrainingSegment] = []
        for raw in segments:
            duration_s = max(18.0, float(raw.duration_s))
            label = str(raw.label).strip() or "Situational Awareness Segment"
            focus_label = str(raw.focus_label).strip() or label
            channels = _normalize_channels(raw.active_channels)
            query_kinds = _normalize_query_kind_names(raw.active_query_kinds)
            families = tuple(raw.scenario_families) or tuple(SituationalAwarenessScenarioFamily)
            profile = raw.profile
            normalized.append(
                SituationalAwarenessTrainingSegment(
                    label=label,
                    duration_s=duration_s,
                    active_channels=channels,
                    active_query_kinds=query_kinds,
                    scenario_families=families,
                    focus_label=focus_label,
                    profile=SituationalAwarenessTrainingProfile(
                        min_track_count=profile.min_track_count,
                        max_track_count=profile.max_track_count,
                        query_interval_min_s=profile.query_interval_min_s,
                        query_interval_max_s=profile.query_interval_max_s,
                        response_window_s=profile.response_window_s,
                        update_time_scale=max(0.45, float(profile.update_time_scale)),
                        pressure_scale=max(0.5, float(profile.pressure_scale)),
                        contact_ttl_s=profile.contact_ttl_s,
                        cue_card_ttl_s=profile.cue_card_ttl_s,
                        top_strip_ttl_s=profile.top_strip_ttl_s,
                        visual_density_scale=max(0.5, float(profile.visual_density_scale)),
                        audio_density_scale=max(0.5, float(profile.audio_density_scale)),
                        allow_visible_answers=bool(profile.allow_visible_answers),
                    ),
                )
            )
        return tuple(normalized)

    def instructions(self) -> list[str]:
        return list(self._instructions)

    def events(self) -> list[QuestionEvent]:
        return list(self._events)

    def start(self) -> None:
        self.start_practice()

    def start_practice(self) -> None:
        if self._phase is not Phase.INSTRUCTIONS:
            return
        if self._custom_segment_layout:
            self._active_segments = self._practice_segments
            self._active_segment_index = 0
            if len(self._active_segments) <= 0:
                self._phase = Phase.PRACTICE_DONE
                return
            self._phase = Phase.PRACTICE
            self._scenario_total = len(self._active_segments)
            self._begin_segment(is_practice=True)
            return
        if self._practice_total <= 0:
            self._phase = Phase.PRACTICE_DONE
            return
        self._phase = Phase.PRACTICE
        self._practice_scenario_index = 0
        self._begin_practice_scenario()

    def start_scored(self) -> None:
        if self._phase in (Phase.SCORED, Phase.RESULTS):
            return
        self._phase = Phase.SCORED
        self._scored_started_at_s = self._clock.now()
        if self._custom_segment_layout:
            self._active_segments = self._scored_segments
            self._active_segment_index = 0
            self._scenario_index = 0
            self._scenario_total = max(1, len(self._active_segments))
            if len(self._active_segments) <= 0:
                self._phase = Phase.RESULTS
                return
            self._begin_segment(is_practice=False)
            return
        self._scenario_index = 0
        total = int(self._config.scored_duration_s // max(1.0, self._config.scored_scenario_duration_s))
        if total * float(self._config.scored_scenario_duration_s) < float(self._config.scored_duration_s):
            total += 1
        self._scenario_total = max(1, total)
        self._begin_scored_scenario()

    def can_exit(self) -> bool:
        return self._phase in (Phase.INSTRUCTIONS, Phase.PRACTICE, Phase.PRACTICE_DONE, Phase.RESULTS)

    def update(self) -> None:
        if self._phase not in (Phase.PRACTICE, Phase.SCORED):
            return
        if self._scenario_plan is None or self._scenario_started_at_s is None:
            return

        target_elapsed = int(max(0.0, self._clock.now() - self._scenario_started_at_s))
        capped_elapsed = min(target_elapsed, int(self._scenario_plan.duration_s))
        while self._processed_ticks < capped_elapsed:
            self._processed_ticks += 1
            self._advance_one_tick(self._processed_ticks)

        if capped_elapsed >= int(self._scenario_plan.duration_s):
            self._end_current_scenario()

        if self._phase is Phase.SCORED and self.time_remaining_s() == 0.0:
            self._phase = Phase.RESULTS
            self._scenario_plan = None
            self._current_segment = None
            self._current_query = None

    def submit_answer(self, raw: str) -> bool:
        if self._phase not in (Phase.PRACTICE, Phase.SCORED):
            return False
        if self._current_query is None or self._scenario_started_at_s is None:
            return False
        normalized = self._normalize_submission(raw, answer_mode=self._current_query.answer_mode)
        if normalized is None:
            return False
        event = self._record_query_result(raw=normalized, is_timeout=False)
        if self._phase is Phase.PRACTICE:
            expected = self._current_query.correct_answer_token
            self._practice_feedback = (
                f"Correct. {expected} was the right answer."
                if event.is_correct
                else f"Incorrect. Correct answer: {expected}."
            )
        self._current_query = None
        return True

    def time_remaining_s(self) -> float | None:
        if self._phase is not Phase.SCORED or self._scored_started_at_s is None:
            return None
        remaining = float(self._config.scored_duration_s) - (self._clock.now() - self._scored_started_at_s)
        return max(0.0, remaining)

    def snapshot(self) -> TestSnapshot:
        if self._phase is Phase.INSTRUCTIONS:
            return TestSnapshot(
                title=self._title,
                phase=self._phase,
                prompt="Press Enter to begin the guided practice scenarios.",
                input_hint="Press Enter to continue",
                time_remaining_s=None,
                attempted_scored=self._scored_attempted,
                correct_scored=self._scored_correct,
                payload=None,
                practice_feedback=None,
            )

        if self._phase is Phase.PRACTICE_DONE:
            return TestSnapshot(
                title=self._title,
                phase=self._phase,
                prompt="Practice complete. Press Enter to start the 25-minute timed block.",
                input_hint="Press Enter to continue",
                time_remaining_s=None,
                attempted_scored=self._scored_attempted,
                correct_scored=self._scored_correct,
                payload=None,
                practice_feedback=self._practice_feedback,
            )

        if self._phase is Phase.RESULTS:
            summary = self.scored_summary()
            prompt = (
                f"Attempted {summary.attempted} | Correct {summary.correct} | "
                f"Accuracy {summary.accuracy * 100.0:.1f}%"
            )
            return TestSnapshot(
                title=self._title,
                phase=self._phase,
                prompt=prompt,
                input_hint="Press Escape to exit",
                time_remaining_s=0.0,
                attempted_scored=self._scored_attempted,
                correct_scored=self._scored_correct,
                payload=None,
                practice_feedback=None,
            )

        payload = self._payload()
        prompt = (
            payload.active_query.prompt
            if payload.active_query is not None
            else f"Monitor the fading picture. Next query in {int(round(payload.next_query_in_s or 0.0)):02d}s."
        )
        input_hint = self._input_hint(payload)
        time_remaining = self.time_remaining_s() if self._phase is Phase.SCORED else None
        return TestSnapshot(
            title=self._title,
            phase=self._phase,
            prompt=prompt,
            input_hint=input_hint,
            time_remaining_s=time_remaining,
            attempted_scored=self._scored_attempted,
            correct_scored=self._scored_correct,
            payload=payload,
            practice_feedback=self._practice_feedback if self._phase is Phase.PRACTICE else None,
        )

    def scored_summary(self) -> AttemptSummary:
        duration_s = float(self._config.scored_duration_s)
        attempted = int(self._scored_attempted)
        correct = int(self._scored_correct)
        accuracy = 0.0 if attempted == 0 else float(correct) / float(attempted)
        throughput = (float(attempted) / duration_s) * 60.0 if duration_s > 0.0 else 0.0
        scored_events = [event for event in self._events if event.phase is Phase.SCORED]
        rts = [event.response_time_s for event in scored_events]
        mean_rt = None if not rts else sum(rts) / len(rts)
        return AttemptSummary(
            attempted=attempted,
            correct=correct,
            accuracy=accuracy,
            duration_s=duration_s,
            throughput_per_min=throughput,
            mean_response_time_s=mean_rt,
            total_score=float(self._scored_total_score),
            max_score=float(self._scored_max_score),
            score_ratio=(
                0.0
                if self._scored_max_score <= 0.0
                else float(self._scored_total_score) / float(self._scored_max_score)
            ),
        )

    def _begin_practice_scenario(self) -> None:
        self._practice_feedback = None
        self._scenario_index = self._practice_scenario_index + 1
        focus = self._practice_focus_order[min(self._practice_scenario_index, len(self._practice_focus_order) - 1)]
        self._scenario_total = max(1, self._practice_total)
        self._start_scenario(
            duration_s=int(self._config.practice_scenario_duration_s),
            practice_focus=focus,
        )

    def _begin_segment(self, *, is_practice: bool) -> None:
        if len(self._active_segments) <= 0 or self._active_segment_index >= len(self._active_segments):
            if is_practice:
                self._phase = Phase.PRACTICE_DONE
            else:
                self._phase = Phase.RESULTS
            self._scenario_plan = None
            self._current_segment = None
            self._current_query = None
            return
        self._practice_feedback = None
        self._scenario_index = self._active_segment_index + 1
        self._scenario_total = len(self._active_segments)
        segment = self._active_segments[self._active_segment_index]
        self._current_segment = segment
        self._start_scenario(
            duration_s=int(round(float(segment.duration_s))),
            practice_focus=None,
            segment=segment,
        )

    def _begin_scored_scenario(self) -> None:
        if self.time_remaining_s() is not None and self.time_remaining_s() <= 0.0:
            self._phase = Phase.RESULTS
            self._scenario_plan = None
            self._current_segment = None
            self._current_query = None
            return
        self._practice_feedback = None
        self._scenario_index += 1
        remaining = int(self.time_remaining_s() or 0.0)
        duration = min(int(self._config.scored_scenario_duration_s), max(1, remaining))
        self._start_scenario(duration_s=duration, practice_focus=None)

    def _start_scenario(
        self,
        *,
        duration_s: int,
        practice_focus: SituationalAwarenessQueryKind | None,
        segment: SituationalAwarenessTrainingSegment | None = None,
    ) -> None:
        self._current_segment = segment
        self._scenario_plan = self._generator.next_scenario(
            difficulty=self._difficulty,
            scenario_index=self._scenario_index,
            total_scenarios=self._scenario_total,
            duration_s=max(18, int(duration_s)),
            practice_focus=practice_focus,
            allowed_families=None if segment is None else segment.scenario_families,
            active_query_kinds=None if segment is None else segment.active_query_kinds,
            training_profile=None if segment is None else segment.profile,
        )
        self._scenario_started_at_s = self._clock.now()
        self._processed_ticks = 0
        self._update_cursor = 0
        self._contact_sweep_cursor = 0
        self._query_slot_cursor = 0
        self._current_query = None
        self._live_assets = {
            seed.callsign: _LiveAssetState(
                index=seed.index,
                callsign=seed.callsign,
                affiliation=seed.affiliation,
                x=float(seed.x),
                y=float(seed.y),
                origin_cell=cell_label_from_xy(seed.x, seed.y),
                heading=seed.heading,
                speed_cells_per_min=seed.speed_cells_per_min,
                channel=seed.channel,
                altitude_fl=seed.altitude_fl,
                waypoint=seed.waypoint,
            )
            for seed in self._scenario_plan.asset_seeds
        }
        self._asset_history = {
            seed.callsign: [
                _AssetMemorySample(
                    x=float(seed.x),
                    y=float(seed.y),
                    heading=seed.heading,
                    channel=seed.channel,
                    altitude_fl=seed.altitude_fl,
                    waypoint=seed.waypoint,
                )
            ]
            for seed in self._scenario_plan.asset_seeds
        }
        self._visible_contacts = {}
        self._cue_card_state = None
        self._top_strip_state = None
        intro_line = self._scenario_intro_line(self._scenario_plan)
        self._set_announcement(
            lines=(
                f"{self._scenario_plan.label}.",
                intro_line,
            ),
            reason=("scenario", self._phase.value, self._scenario_index),
        )
        seed_order = sorted(self._live_assets.values(), key=lambda item: item.index)
        if seed_order and "pictorial" in self._active_channels():
            self._reveal_contact(seed_order[0].callsign, tick=0)
        if seed_order and ("coded" in self._active_channels() or "numerical" in self._active_channels()):
            self._show_cue_card(seed_order[0].callsign, tick=0)
        if "coded" in self._active_channels() or "numerical" in self._active_channels():
            self._show_top_strip(intro_line, tick=0)

    @staticmethod
    def _scenario_intro_line(plan: _ScenarioPlan) -> str:
        if plan.family is SituationalAwarenessScenarioFamily.MERGE_CONFLICT:
            return "Watch the grid sweeps and radio calls for a possible conflict."
        if plan.family is SituationalAwarenessScenarioFamily.FUEL_PRIORITY:
            return "Hold the coded status picture while the map only reveals short contact flashes."
        if plan.family is SituationalAwarenessScenarioFamily.ROUTE_HANDOFF:
            return "Track the handoff and keep projecting the route after the update fades."
        return "Track channel and waypoint changes from short cue-card flashes and radio chatter."

    def _end_current_scenario(self) -> None:
        if self._current_query is not None:
            self._record_query_result(raw="", is_timeout=True)
            self._current_query = None

        if self._custom_segment_layout:
            self._active_segment_index += 1
            if self._phase is Phase.PRACTICE:
                self._begin_segment(is_practice=True)
                return
            if self._phase is Phase.SCORED:
                if self.time_remaining_s() is None or self.time_remaining_s() <= 0.0:
                    self._phase = Phase.RESULTS
                    self._scenario_plan = None
                    self._current_segment = None
                    return
                self._begin_segment(is_practice=False)
                return

        if self._phase is Phase.PRACTICE:
            self._practice_scenario_index += 1
            if self._practice_scenario_index >= self._practice_total:
                self._phase = Phase.PRACTICE_DONE
                self._scenario_plan = None
                self._current_segment = None
                return
            self._begin_practice_scenario()
            return

        if self._phase is Phase.SCORED:
            if self.time_remaining_s() is None or self.time_remaining_s() <= 0.0:
                self._phase = Phase.RESULTS
                self._scenario_plan = None
                self._current_segment = None
                return
            self._begin_scored_scenario()

    def _advance_one_tick(self, tick: int) -> None:
        if self._scenario_plan is None:
            return

        if self._current_query is not None and tick >= int(self._current_query.expires_at_s):
            self._record_query_result(raw="", is_timeout=True)
            self._current_query = None

        for asset in self._live_assets.values():
            dx, dy = _HEADING_VECTORS.get(asset.heading, (0, 0))
            asset.x = max(0.0, min(float(_GRID_SIZE - 1), asset.x + (dx * asset.speed_cells_per_min / 60.0)))
            asset.y = max(0.0, min(float(_GRID_SIZE - 1), asset.y + (dy * asset.speed_cells_per_min / 60.0)))

        while (
            self._update_cursor < len(self._scenario_plan.updates)
            and int(self._scenario_plan.updates[self._update_cursor].at_s) == int(tick)
        ):
            self._apply_update(self._scenario_plan.updates[self._update_cursor], tick=tick)
            self._update_cursor += 1

        while (
            self._contact_sweep_cursor < len(self._scenario_plan.contact_sweeps)
            and int(self._scenario_plan.contact_sweeps[self._contact_sweep_cursor].at_s) == int(tick)
        ):
            self._apply_contact_sweep(self._scenario_plan.contact_sweeps[self._contact_sweep_cursor], tick=tick)
            self._contact_sweep_cursor += 1

        self._record_asset_history()

        if self._current_query is None and self._query_slot_cursor < len(self._scenario_plan.query_slots):
            slot = self._scenario_plan.query_slots[self._query_slot_cursor]
            if int(slot.at_s) <= int(tick):
                self._query_slot_cursor += 1
                self._spawn_query(slot=slot, tick=tick)

    def _apply_update_to_state(self, asset: _LiveAssetState, update: _UpdateEvent) -> None:
        if update.heading is not None:
            asset.heading = _normalize_heading(update.heading)
        if update.speed_cells_per_min is not None:
            asset.speed_cells_per_min = max(1, int(update.speed_cells_per_min))
        if update.channel is not None:
            asset.channel = int(update.channel)
        if update.altitude_fl is not None:
            asset.altitude_fl = int(update.altitude_fl)
        if update.waypoint is not None:
            asset.waypoint = normalize_grid_cell_token(update.waypoint) or asset.waypoint

    def _apply_update(self, update: _UpdateEvent, *, tick: int) -> None:
        asset = self._live_assets.get(update.callsign)
        if asset is None:
            return
        self._apply_update_to_state(asset, update)
        if update.modality in {"audio_plus_visual", "visual_only"}:
            if "pictorial" in self._active_channels():
                self._reveal_contact(asset.callsign, tick=tick)
            if "coded" in self._active_channels() or "numerical" in self._active_channels():
                self._show_cue_card(asset.callsign, tick=tick)
                self._show_top_strip(update.line, tick=tick)
        elif "coded" in self._active_channels() or "numerical" in self._active_channels():
            self._show_top_strip(update.line, tick=tick)
        self._set_announcement(lines=(update.line,), reason=("sa_update", self._scenario_index, tick, asset.callsign))

    def _apply_contact_sweep(self, sweep: _ContactSweepEvent, *, tick: int) -> None:
        if "pictorial" not in self._active_channels():
            return
        for callsign in sweep.callsigns:
            self._reveal_contact(callsign, tick=tick)

    def _record_asset_history(self) -> None:
        for asset in self._live_assets.values():
            history = self._asset_history.setdefault(asset.callsign, [])
            history.append(
                _AssetMemorySample(
                    x=float(asset.x),
                    y=float(asset.y),
                    heading=asset.heading,
                    channel=asset.channel,
                    altitude_fl=asset.altitude_fl,
                    waypoint=asset.waypoint,
                )
            )

    def _contact_ttl_s(self) -> int:
        profile = self._current_profile()
        if profile.contact_ttl_s is not None:
            return max(2, int(round(float(profile.contact_ttl_s))))
        return max(2, int(round(6.0 / max(0.65, float(profile.pressure_scale)))))

    def _cue_card_ttl_s(self) -> int:
        profile = self._current_profile()
        if profile.cue_card_ttl_s is not None:
            return max(2, int(round(float(profile.cue_card_ttl_s))))
        return max(2, int(round(7.0 / max(0.65, float(profile.pressure_scale)))))

    def _top_strip_ttl_s(self) -> int:
        profile = self._current_profile()
        if profile.top_strip_ttl_s is not None:
            return max(2, int(round(float(profile.top_strip_ttl_s))))
        return max(2, int(round(5.0 / max(0.65, float(profile.pressure_scale)))))

    def _reveal_contact(self, callsign: str, *, tick: int) -> None:
        self._visible_contacts[callsign] = _VisibleContactState(
            callsign=callsign,
            shown_at_s=int(tick),
            visible_until_s=int(tick) + self._contact_ttl_s(),
        )

    def _show_cue_card(self, callsign: str, *, tick: int) -> None:
        self._cue_card_state = _CueCardState(
            callsign=callsign,
            shown_at_s=int(tick),
            visible_until_s=int(tick) + self._cue_card_ttl_s(),
        )

    def _show_top_strip(self, text: str, *, tick: int) -> None:
        if str(text).strip() == "":
            return
        self._top_strip_state = _TopStripState(
            text=str(text),
            shown_at_s=int(tick),
            visible_until_s=int(tick) + self._top_strip_ttl_s(),
        )

    def _spawn_query(self, *, slot: _QuerySlot, tick: int) -> None:
        if self._scenario_plan is None:
            return
        expires_at_s = (
            self._scenario_plan.query_slots[self._query_slot_cursor].at_s
            if self._query_slot_cursor < len(self._scenario_plan.query_slots)
            else int(self._scenario_plan.duration_s)
        )
        if self._scenario_plan.response_window_s is not None:
            expires_at_s = min(
                expires_at_s,
                tick + max(4, int(self._scenario_plan.response_window_s)),
                int(self._scenario_plan.duration_s),
            )
        query = self._build_query(slot=slot, tick=tick, expires_at_s=expires_at_s)
        if not self._current_profile().allow_visible_answers and query.subject_callsign is not None:
            self._visible_contacts.pop(query.subject_callsign, None)
            if (
                self._cue_card_state is not None
                and self._cue_card_state.callsign == query.subject_callsign
            ):
                self._cue_card_state = None
        self._current_query = query
        self._set_announcement(
            lines=(self._spoken_family_tag(self._scenario_plan.family), query.prompt),
            reason=("sa_query", self._scenario_index, query.query_id),
        )

    @staticmethod
    def _spoken_family_tag(family: SituationalAwarenessScenarioFamily) -> str:
        if family is SituationalAwarenessScenarioFamily.MERGE_CONFLICT:
            return "Conflict update."
        if family is SituationalAwarenessScenarioFamily.FUEL_PRIORITY:
            return "Status update."
        if family is SituationalAwarenessScenarioFamily.ROUTE_HANDOFF:
            return "Route handoff update."
        return "Channel and waypoint update."

    def _build_query(
        self,
        *,
        slot: _QuerySlot,
        tick: int,
        expires_at_s: int,
    ) -> _ActiveQueryState:
        query_id = (self._scenario_index * 100) + tick
        builders = {
            SituationalAwarenessQueryKind.CURRENT_LOCATION: self._build_current_location_query,
            SituationalAwarenessQueryKind.ORIGIN_LOCATION: self._build_origin_location_query,
            SituationalAwarenessQueryKind.FUTURE_LOCATION: self._build_future_location_query,
            SituationalAwarenessQueryKind.SAFE_TO_MOVE: self._build_safe_to_move_query,
            SituationalAwarenessQueryKind.STATUS_RECALL: self._build_status_recall_query,
        }
        order = [
            slot.kind,
            SituationalAwarenessQueryKind.CURRENT_LOCATION,
            SituationalAwarenessQueryKind.STATUS_RECALL,
            SituationalAwarenessQueryKind.FUTURE_LOCATION,
            SituationalAwarenessQueryKind.ORIGIN_LOCATION,
            SituationalAwarenessQueryKind.SAFE_TO_MOVE,
        ]
        for kind in order:
            builder = builders[kind]
            query = builder(
                query_id=query_id,
                tick=tick,
                expires_at_s=expires_at_s,
                future_offset_s=max(8, int(slot.future_offset_s or 12)),
            )
            if query is not None:
                return query
        return _ActiveQueryState(
            query_id=query_id,
            asked_at_s=tick,
            expires_at_s=expires_at_s,
            kind=SituationalAwarenessQueryKind.CURRENT_LOCATION,
            answer_mode=SituationalAwarenessAnswerMode.GRID_CELL,
            prompt="Where is the priority contact now?",
            correct_answer_token="E5",
        )

    def _memory_candidates(self) -> list[_LiveAssetState]:
        assets = sorted(self._live_assets.values(), key=lambda item: item.index)
        if self._current_profile().allow_visible_answers:
            return assets
        hidden = [asset for asset in assets if not self._is_contact_visible(asset.callsign)]
        return hidden or assets

    def _is_contact_visible(self, callsign: str) -> bool:
        state = self._visible_contacts.get(callsign)
        return state is not None and int(state.visible_until_s) > int(self._processed_ticks)

    def _pick_asset_for_query(
        self,
        candidates: Sequence[_LiveAssetState],
        *,
        preferred_callsign: str | None = None,
    ) -> _LiveAssetState:
        ordered = list(candidates)
        if preferred_callsign is not None:
            for asset in ordered:
                if asset.callsign == preferred_callsign:
                    return asset
        if not ordered:
            raise ValueError("Expected at least one candidate asset.")
        index = (self._scenario_index + self._query_slot_cursor - 1) % len(ordered)
        return ordered[index]

    def _build_current_location_query(
        self,
        *,
        query_id: int,
        tick: int,
        expires_at_s: int,
        future_offset_s: int,
    ) -> _ActiveQueryState | None:
        candidates = self._memory_candidates()
        if not candidates:
            return None
        target = self._pick_asset_for_query(candidates)
        return _ActiveQueryState(
            query_id=query_id,
            asked_at_s=tick,
            expires_at_s=expires_at_s,
            kind=SituationalAwarenessQueryKind.CURRENT_LOCATION,
            answer_mode=SituationalAwarenessAnswerMode.GRID_CELL,
            prompt=f"Where is {target.callsign} now?",
            correct_answer_token=cell_label_from_xy(target.x, target.y),
            subject_callsign=target.callsign,
        )

    def _build_origin_location_query(
        self,
        *,
        query_id: int,
        tick: int,
        expires_at_s: int,
        future_offset_s: int,
    ) -> _ActiveQueryState | None:
        candidates = sorted(self._live_assets.values(), key=lambda item: item.index)
        if not candidates:
            return None
        target = self._pick_asset_for_query(candidates)
        return _ActiveQueryState(
            query_id=query_id,
            asked_at_s=tick,
            expires_at_s=expires_at_s,
            kind=SituationalAwarenessQueryKind.ORIGIN_LOCATION,
            answer_mode=SituationalAwarenessAnswerMode.GRID_CELL,
            prompt=f"Where did {target.callsign} enter the picture?",
            correct_answer_token=target.origin_cell,
            subject_callsign=target.callsign,
        )

    def _build_future_location_query(
        self,
        *,
        query_id: int,
        tick: int,
        expires_at_s: int,
        future_offset_s: int,
    ) -> _ActiveQueryState | None:
        candidates = self._memory_candidates()
        if not candidates:
            return None
        target = self._pick_asset_for_query(candidates)
        projected = self._project_asset_cell(
            callsign=target.callsign,
            start_tick=tick,
            offset_s=future_offset_s,
        )
        return _ActiveQueryState(
            query_id=query_id,
            asked_at_s=tick,
            expires_at_s=expires_at_s,
            kind=SituationalAwarenessQueryKind.FUTURE_LOCATION,
            answer_mode=SituationalAwarenessAnswerMode.GRID_CELL,
            prompt=f"Where will {target.callsign} be in {future_offset_s}s if the current routing continues?",
            correct_answer_token=projected,
            subject_callsign=target.callsign,
            future_offset_s=future_offset_s,
        )

    def _build_safe_to_move_query(
        self,
        *,
        query_id: int,
        tick: int,
        expires_at_s: int,
        future_offset_s: int,
    ) -> _ActiveQueryState | None:
        if self._scenario_plan is None:
            return None
        preferred = self._scenario_plan.context.get("priority_callsign")
        assets = sorted(self._live_assets.values(), key=lambda item: item.index)
        if not assets:
            return None
        target = self._pick_asset_for_query(assets, preferred_callsign=str(preferred) if preferred else None)
        destination = normalize_grid_cell_token(str(self._scenario_plan.context.get("merge_cell", target.waypoint)))
        if destination is None:
            destination = target.waypoint
        decision = self._safe_move_decision(target.callsign, destination, start_tick=tick)
        choices = (
            SituationalAwarenessAnswerChoice(1, "Safe now."),
            SituationalAwarenessAnswerChoice(2, "Wait one sweep, then go."),
            SituationalAwarenessAnswerChoice(3, "Unsafe. Hold clear of traffic."),
            SituationalAwarenessAnswerChoice(4, "Request a fresh picture first."),
        )
        correct = {"safe_now": "1", "wait": "2", "unsafe": "3", "refresh": "4"}[decision]
        return _ActiveQueryState(
            query_id=query_id,
            asked_at_s=tick,
            expires_at_s=expires_at_s,
            kind=SituationalAwarenessQueryKind.SAFE_TO_MOVE,
            answer_mode=SituationalAwarenessAnswerMode.CHOICE,
            prompt=f"Is it safe for {target.callsign} to proceed to {destination} now?",
            correct_answer_token=correct,
            subject_callsign=target.callsign,
            answer_choices=choices,
        )

    def _build_status_recall_query(
        self,
        *,
        query_id: int,
        tick: int,
        expires_at_s: int,
        future_offset_s: int,
    ) -> _ActiveQueryState | None:
        assets = sorted(self._live_assets.values(), key=lambda item: item.index)
        if not assets:
            return None
        target = self._pick_asset_for_query(assets)
        subkind = ("channel", "altitude", "waypoint", "eta")[(query_id + target.index) % 4]
        if subkind == "channel":
            prompt = f"What communication channel is {target.callsign} using?"
            correct_text = f"CH {target.channel}"
            distractors = tuple(
                f"CH {value}"
                for value in range(1, 6)
                if value != target.channel
            )
        elif subkind == "altitude":
            prompt = f"What altitude is {target.callsign} holding?"
            correct_text = f"FL{target.altitude_fl}"
            distractors = tuple(
                f"FL{value}"
                for value in (
                    target.altitude_fl - 20,
                    target.altitude_fl - 10,
                    target.altitude_fl + 10,
                    target.altitude_fl + 20,
                )
                if value != target.altitude_fl and value >= 120
            )
        elif subkind == "waypoint":
            prompt = f"What is {target.callsign}'s next waypoint?"
            correct_text = target.waypoint
            other_waypoints = [asset.waypoint for asset in assets if asset.callsign != target.callsign]
            distractors = tuple(dict.fromkeys(other_waypoints + ["C5", "F5", "H4"]))
        else:
            prompt = f"When is {target.callsign} due at its next waypoint?"
            eta_s = self._estimate_time_to_waypoint_s(target)
            correct_text = _clock_text(
                int(self._scenario_clock_s()) + eta_s
            )
            distractors = tuple(
                _clock_text(int(self._scenario_clock_s()) + max(0, eta_s + delta))
                for delta in (-20, -10, 10, 20)
            )
        choices, correct_token = self._choice_card(
            query_id=query_id,
            correct_text=correct_text,
            distractors=distractors,
        )
        return _ActiveQueryState(
            query_id=query_id,
            asked_at_s=tick,
            expires_at_s=expires_at_s,
            kind=SituationalAwarenessQueryKind.STATUS_RECALL,
            answer_mode=SituationalAwarenessAnswerMode.CHOICE,
            prompt=prompt,
            correct_answer_token=correct_token,
            subject_callsign=target.callsign,
            answer_choices=choices,
        )

    def _choice_card(
        self,
        *,
        query_id: int,
        correct_text: str,
        distractors: Sequence[str],
    ) -> tuple[tuple[SituationalAwarenessAnswerChoice, ...], str]:
        unique_distractors: list[str] = []
        for item in distractors:
            text = str(item).strip()
            if text == "" or text == str(correct_text) or text in unique_distractors:
                continue
            unique_distractors.append(text)
            if len(unique_distractors) >= 3:
                break
        while len(unique_distractors) < 3:
            unique_distractors.append(f"Hold {len(unique_distractors) + 1}")
        correct_index = int((query_id % 4) + 1)
        ordered: list[str] = []
        distractor_index = 0
        for code in range(1, 5):
            if code == correct_index:
                ordered.append(str(correct_text))
            else:
                ordered.append(unique_distractors[distractor_index])
                distractor_index += 1
        choices = tuple(
            SituationalAwarenessAnswerChoice(code=index + 1, text=text)
            for index, text in enumerate(ordered)
        )
        return choices, str(correct_index)

    def _safe_move_decision(self, callsign: str, destination: str, *, start_tick: int) -> str:
        dest_xy = cell_xy_from_label(destination)
        if dest_xy is None:
            return "refresh"
        states = {name: asset.copy() for name, asset in self._live_assets.items()}
        breaches_now = self._count_destination_conflicts(
            states=states,
            destination=dest_xy,
            subject_callsign=callsign,
        )
        if breaches_now["hostile"] or breaches_now["unknown"]:
            return "unsafe"
        if breaches_now["friendly"]:
            return "wait"

        pending_updates = [
            update
            for update in self._scenario_plan.updates
            if int(update.at_s) > int(start_tick)
        ]
        update_cursor = 0
        for step in range(1, 13):
            sim_tick = int(start_tick) + step
            for asset in states.values():
                dx, dy = _HEADING_VECTORS.get(asset.heading, (0, 0))
                asset.x = max(0.0, min(float(_GRID_SIZE - 1), asset.x + (dx * asset.speed_cells_per_min / 60.0)))
                asset.y = max(0.0, min(float(_GRID_SIZE - 1), asset.y + (dy * asset.speed_cells_per_min / 60.0)))
            while update_cursor < len(pending_updates) and int(pending_updates[update_cursor].at_s) == sim_tick:
                update = pending_updates[update_cursor]
                asset = states.get(update.callsign)
                if asset is not None:
                    self._apply_update_to_state(asset, update)
                update_cursor += 1
            breaches = self._count_destination_conflicts(
                states=states,
                destination=dest_xy,
                subject_callsign=callsign,
            )
            if breaches["hostile"] or breaches["unknown"]:
                return "unsafe"
            if breaches["friendly"]:
                return "wait"
        return "safe_now"

    def _count_destination_conflicts(
        self,
        *,
        states: dict[str, _LiveAssetState],
        destination: tuple[int, int],
        subject_callsign: str,
    ) -> dict[str, int]:
        counts = {"friendly": 0, "hostile": 0, "unknown": 0}
        for asset in states.values():
            if asset.callsign == subject_callsign:
                continue
            dx = float(destination[0]) - float(asset.x)
            dy = float(destination[1]) - float(asset.y)
            if ((dx * dx) + (dy * dy)) ** 0.5 <= 0.85:
                counts[asset.affiliation] += 1
        return counts

    def _project_asset_cell(self, *, callsign: str, start_tick: int, offset_s: int) -> str:
        states = {name: asset.copy() for name, asset in self._live_assets.items()}
        pending_updates = [
            update
            for update in self._scenario_plan.updates
            if int(update.at_s) > int(start_tick)
        ]
        update_cursor = 0
        end_tick = min(int(self._scenario_plan.duration_s), int(start_tick) + max(1, int(offset_s)))
        for sim_tick in range(int(start_tick) + 1, end_tick + 1):
            for asset in states.values():
                dx, dy = _HEADING_VECTORS.get(asset.heading, (0, 0))
                asset.x = max(0.0, min(float(_GRID_SIZE - 1), asset.x + (dx * asset.speed_cells_per_min / 60.0)))
                asset.y = max(0.0, min(float(_GRID_SIZE - 1), asset.y + (dy * asset.speed_cells_per_min / 60.0)))
            while update_cursor < len(pending_updates) and int(pending_updates[update_cursor].at_s) == sim_tick:
                update = pending_updates[update_cursor]
                asset = states.get(update.callsign)
                if asset is not None:
                    self._apply_update_to_state(asset, update)
                update_cursor += 1
        projected = states.get(callsign)
        if projected is None:
            return "A0"
        return cell_label_from_xy(projected.x, projected.y)

    def _estimate_time_to_waypoint_s(self, asset: _LiveAssetState) -> int:
        target = cell_xy_from_label(asset.waypoint)
        if target is None or asset.speed_cells_per_min <= 0:
            return 0
        dx = float(target[0]) - float(asset.x)
        dy = float(target[1]) - float(asset.y)
        distance = ((dx * dx) + (dy * dy)) ** 0.5
        if distance <= 0.05:
            return 0
        return max(0, int(round((distance / float(asset.speed_cells_per_min)) * 60.0)))

    def _record_query_result(self, *, raw: str, is_timeout: bool) -> QuestionEvent:
        assert self._current_query is not None
        assert self._scenario_started_at_s is not None
        prompt = self._current_query.prompt
        correct = self._current_query.correct_answer_token
        presented_at = self._scenario_started_at_s + float(self._current_query.asked_at_s)
        if is_timeout:
            answered_at = self._scenario_started_at_s + float(self._current_query.expires_at_s)
            response_time = max(0.0, answered_at - presented_at)
            user_answer = 0
            submitted = ""
            is_correct = False
        else:
            answered_at = self._clock.now()
            response_time = max(0.0, answered_at - presented_at)
            submitted = str(raw)
            user_answer = self._user_answer_value(raw=submitted)
            is_correct = submitted == correct

        score = 1.0 if is_correct else 0.0
        payload = self.snapshot().payload
        event = QuestionEvent(
            index=len(self._events),
            phase=self._phase,
            prompt=prompt,
            correct_answer=self._user_answer_value(raw=correct),
            user_answer=user_answer,
            is_correct=is_correct,
            presented_at_s=presented_at,
            answered_at_s=answered_at,
            response_time_s=response_time,
            raw=submitted,
            score=score,
            max_score=1.0,
            is_timeout=is_timeout,
            content_metadata=content_metadata_from_payload(payload),
        )
        self._events.append(event)

        if self._phase is Phase.SCORED:
            self._scored_attempted += 1
            self._scored_correct += 1 if is_correct else 0
            self._scored_total_score += score
            self._scored_max_score += 1.0
        return event

    @staticmethod
    def _user_answer_value(*, raw: str) -> int:
        token = str(raw).strip().upper()
        if token.isdigit():
            return int(token)
        cell = cell_xy_from_label(token)
        if cell is None:
            return 0
        return (cell[1] * 10) + cell[0]

    def _normalize_submission(
        self,
        raw: str,
        *,
        answer_mode: SituationalAwarenessAnswerMode,
    ) -> str | None:
        token = str(raw).strip().upper()
        if answer_mode is SituationalAwarenessAnswerMode.GRID_CELL:
            return normalize_grid_cell_token(token)
        if not token.isdigit():
            return None
        value = int(token)
        return str(value) if 1 <= value <= 4 else None

    def _active_channels(self) -> tuple[str, ...]:
        if self._current_segment is None:
            return SA_CHANNEL_ORDER
        return self._current_segment.active_channels

    def _active_query_kind_names(self) -> tuple[str, ...]:
        if self._current_segment is None:
            return SA_QUERY_KIND_ORDER
        return self._current_segment.active_query_kinds

    def _current_profile(self) -> SituationalAwarenessTrainingProfile:
        if self._current_segment is None:
            return SituationalAwarenessTrainingProfile()
        return self._current_segment.profile

    def _focus_label(self) -> str:
        if self._current_segment is None:
            return "Full mixed picture"
        return self._current_segment.focus_label

    def _segment_label(self) -> str:
        if self._current_segment is None:
            return self._scenario_plan.label if self._scenario_plan is not None else self._title
        return self._current_segment.label

    def _scenario_clock_s(self) -> float:
        assert self._scenario_plan is not None
        assert self._scenario_started_at_s is not None
        return float(self._scenario_plan.display_clock_base_s) + max(
            0.0, self._clock.now() - self._scenario_started_at_s
        )

    def _payload(self) -> SituationalAwarenessPayload:
        assert self._scenario_plan is not None
        assert self._scenario_started_at_s is not None
        scenario_elapsed_s = max(0.0, self._clock.now() - self._scenario_started_at_s)
        scenario_remaining_s = max(0.0, float(self._scenario_plan.duration_s) - scenario_elapsed_s)

        contacts: list[SituationalAwarenessVisibleContact] = []
        ttl = max(1.0, float(self._contact_ttl_s()))
        for state in list(self._visible_contacts.values()):
            if int(state.visible_until_s) <= int(scenario_elapsed_s):
                continue
            asset = self._live_assets.get(state.callsign)
            if asset is None:
                continue
            fade = max(0.0, min(1.0, (float(state.visible_until_s) - scenario_elapsed_s) / ttl))
            contacts.append(
                SituationalAwarenessVisibleContact(
                    callsign=asset.callsign,
                    affiliation=asset.affiliation,
                    x=float(asset.x),
                    y=float(asset.y),
                    cell_label=cell_label_from_xy(asset.x, asset.y),
                    heading=asset.heading,
                    fade=fade,
                )
            )
        contacts.sort(key=lambda item: item.callsign)

        cue_card = None
        if self._cue_card_state is not None and int(self._cue_card_state.visible_until_s) > int(scenario_elapsed_s):
            asset = self._live_assets.get(self._cue_card_state.callsign)
            if asset is not None:
                ttl_card = max(1.0, float(self._cue_card_ttl_s()))
                cue_card = SituationalAwarenessCueCard(
                    callsign=asset.callsign,
                    affiliation=asset.affiliation,
                    next_waypoint=asset.waypoint,
                    eta_clock_text=_clock_text(int(self._scenario_clock_s()) + self._estimate_time_to_waypoint_s(asset)),
                    altitude_text=f"FL{asset.altitude_fl}",
                    channel_text=f"CH {asset.channel}",
                    fade=max(
                        0.0,
                        min(1.0, (float(self._cue_card_state.visible_until_s) - scenario_elapsed_s) / ttl_card),
                    ),
                )

        top_strip_text = ""
        top_strip_fade = 0.0
        if self._top_strip_state is not None and int(self._top_strip_state.visible_until_s) > int(scenario_elapsed_s):
            ttl_strip = max(1.0, float(self._top_strip_ttl_s()))
            top_strip_text = self._top_strip_state.text
            top_strip_fade = max(
                0.0,
                min(1.0, (float(self._top_strip_state.visible_until_s) - scenario_elapsed_s) / ttl_strip),
            )

        next_query_in_s = None
        if self._current_query is None and self._query_slot_cursor < len(self._scenario_plan.query_slots):
            next_query_in_s = max(
                0.0,
                float(self._scenario_plan.query_slots[self._query_slot_cursor].at_s) - float(self._processed_ticks),
            )

        active_query = None
        answer_mode = None
        correct_answer_token = None
        if self._current_query is not None:
            active_query = SituationalAwarenessActiveQuery(
                query_id=self._current_query.query_id,
                kind=self._current_query.kind,
                answer_mode=self._current_query.answer_mode,
                prompt=self._current_query.prompt,
                correct_answer_token=self._current_query.correct_answer_token,
                expires_in_s=max(0.0, float(self._current_query.expires_at_s) - float(scenario_elapsed_s)),
                subject_callsign=self._current_query.subject_callsign,
                future_offset_s=self._current_query.future_offset_s,
                answer_choices=self._current_query.answer_choices,
            )
            answer_mode = self._current_query.answer_mode
            correct_answer_token = self._current_query.correct_answer_token

        return SituationalAwarenessPayload(
            scenario_family=self._scenario_plan.family,
            scenario_label=self._scenario_plan.label,
            scenario_index=self._scenario_index,
            scenario_total=self._scenario_total,
            active_channels=self._active_channels(),
            active_query_kinds=self._active_query_kind_names(),
            focus_label=self._focus_label(),
            segment_label=self._segment_label(),
            segment_index=self._scenario_index,
            segment_total=self._scenario_total,
            segment_time_remaining_s=scenario_remaining_s,
            scenario_elapsed_s=scenario_elapsed_s,
            scenario_time_remaining_s=scenario_remaining_s,
            next_query_in_s=next_query_in_s,
            visible_contacts=tuple(contacts),
            cue_card=cue_card,
            waypoints=self._scenario_plan.waypoints,
            top_strip_text=top_strip_text,
            top_strip_fade=top_strip_fade,
            display_clock_text=_clock_text(int(self._scenario_clock_s())),
            active_query=active_query,
            answer_mode=answer_mode,
            correct_answer_token=correct_answer_token,
            announcement_token=self._announcement_token,
            announcement_lines=self._announcement_lines,
        )

    def _input_hint(self, payload: SituationalAwarenessPayload) -> str:
        if payload.active_query is None:
            return "Monitor the fading grid sweeps, cue card, and aural updates."
        if payload.answer_mode is SituationalAwarenessAnswerMode.GRID_CELL:
            return "Click a grid cell or type row+column, then press Enter."
        return "Click a choice or press 1-4, then press Enter."

    def _set_announcement(self, *, lines: tuple[str, ...], reason: tuple[object, ...]) -> None:
        self._announcement_serial += 1
        self._announcement_token = (*reason, self._announcement_serial)
        if "aural" not in self._active_channels():
            self._announcement_lines = ()
            return
        self._announcement_lines = tuple(line for line in lines if str(line).strip() != "")


def build_situational_awareness_test(
    *,
    clock: Clock,
    seed: int,
    difficulty: float = 0.5,
    config: SituationalAwarenessConfig | None = None,
    title: str = "Situational Awareness",
    practice_segments: Sequence[SituationalAwarenessTrainingSegment] | None = None,
    scored_segments: Sequence[SituationalAwarenessTrainingSegment] | None = None,
) -> SituationalAwarenessTest:
    return SituationalAwarenessTest(
        clock=clock,
        seed=seed,
        difficulty=difficulty,
        config=config,
        title=title,
        practice_segments=practice_segments,
        scored_segments=scored_segments,
    )
