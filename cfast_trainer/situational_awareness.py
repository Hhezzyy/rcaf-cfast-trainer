from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .clock import Clock
from .cognitive_core import AttemptSummary, Phase, QuestionEvent, SeededRng, TestSnapshot, clamp01

_GRID_SIZE = 10
_ROW_LABELS = tuple("ABCDEFGHIJ")
_CALLSIGN_POOL = (
    "R1",
    "R2",
    "R3",
    "T4",
    "K5",
    "N6",
    "F7",
    "M8",
    "H9",
    "P1",
    "L2",
    "S3",
)
_WAYPOINT_MAP = {
    "ALFA": (1, 1),
    "BRAVO": (8, 1),
    "CHARLIE": (8, 8),
    "DELTA": (1, 8),
    "ECHO": (5, 2),
    "FOXTROT": (5, 7),
}
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
_FUEL_STATES = ("NORMAL", "LOW", "MIN")
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
    FUTURE_POSITION = "future_position"
    CONTACT_IDENTIFICATION = "contact_identification"
    CODE_OR_STATUS_RECALL = "code_or_status_recall"
    ACTION_SELECTION = "action_selection"


class SituationalAwarenessAnswerMode(StrEnum):
    GRID_CELL = "grid_cell"
    TRACK_INDEX = "track_index"
    ACTION = "action"


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
class SituationalAwarenessTrack:
    index: int
    callsign: str
    x: float
    y: float
    cell_label: str
    heading: str
    speed_cells_per_min: int
    squawk: int
    channel: int
    altitude_fl: int
    fuel_state: str
    waypoint: str


@dataclass(frozen=True, slots=True)
class SituationalAwarenessStatusEntry:
    track_index: int
    callsign: str
    cell_label: str
    heading: str
    speed_cells_per_min: int
    squawk: int
    channel: int
    altitude_fl: int
    fuel_state: str
    waypoint: str


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
    tracks: tuple[SituationalAwarenessTrack, ...]
    status_entries: tuple[SituationalAwarenessStatusEntry, ...]
    waypoints: tuple[SituationalAwarenessWaypoint, ...]
    recent_feed_lines: tuple[str, ...]
    active_query: SituationalAwarenessActiveQuery | None
    answer_mode: SituationalAwarenessAnswerMode | None
    correct_answer_token: str | None
    announcement_token: tuple[object, ...] | None
    announcement_lines: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _TrackSeed:
    index: int
    callsign: str
    x: float
    y: float
    heading: str
    speed_cells_per_min: int
    squawk: int
    channel: int
    altitude_fl: int
    fuel_state: str
    waypoint: str


@dataclass(frozen=True, slots=True)
class _TrackUpdateEvent:
    at_s: int
    callsign: str
    line: str
    heading: str | None = None
    speed_cells_per_min: int | None = None
    squawk: int | None = None
    channel: int | None = None
    altitude_fl: int | None = None
    fuel_state: str | None = None
    waypoint: str | None = None


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
    track_seeds: tuple[_TrackSeed, ...]
    updates: tuple[_TrackUpdateEvent, ...]
    query_slots: tuple[_QuerySlot, ...]
    active_query_kinds: tuple[str, ...]
    response_window_s: int | None
    context: dict[str, Any]


@dataclass(slots=True)
class _LiveTrackState:
    index: int
    callsign: str
    x: float
    y: float
    heading: str
    speed_cells_per_min: int
    squawk: int
    channel: int
    altitude_fl: int
    fuel_state: str
    waypoint: str

    def copy(self) -> _LiveTrackState:
        return _LiveTrackState(
            index=self.index,
            callsign=self.callsign,
            x=float(self.x),
            y=float(self.y),
            heading=str(self.heading),
            speed_cells_per_min=int(self.speed_cells_per_min),
            squawk=int(self.squawk),
            channel=int(self.channel),
            altitude_fl=int(self.altitude_fl),
            fuel_state=str(self.fuel_state),
            waypoint=str(self.waypoint),
        )


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


@dataclass(frozen=True, slots=True)
class _CurrentProblem:
    prompt: str
    answer: str


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
            return self._build_fuel_priority(
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
        if practice_focus is SituationalAwarenessQueryKind.FUTURE_POSITION:
            family = SituationalAwarenessScenarioFamily.ROUTE_HANDOFF
        elif practice_focus in (
            SituationalAwarenessQueryKind.CONTACT_IDENTIFICATION,
            SituationalAwarenessQueryKind.CODE_OR_STATUS_RECALL,
        ):
            family = SituationalAwarenessScenarioFamily.CHANNEL_WAYPOINT_CHANGE
        elif practice_focus is SituationalAwarenessQueryKind.ACTION_SELECTION:
            family = SituationalAwarenessScenarioFamily.FUEL_PRIORITY
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
        if clamp01(difficulty) < 0.65:
            return low
        return high

    def _base_waypoints(self) -> tuple[SituationalAwarenessWaypoint, ...]:
        return tuple(
            SituationalAwarenessWaypoint(name=name, x=coords[0], y=coords[1])
            for name, coords in _WAYPOINT_MAP.items()
        )

    def _sample_callsigns(self, count: int) -> list[str]:
        return self._rng.sample(_CALLSIGN_POOL, k=count)

    def _make_track_seed(
        self,
        *,
        index: int,
        callsign: str,
        x: float,
        y: float,
        heading: str,
        speed: int,
        channel: int,
        altitude_fl: int,
        waypoint: str,
        fuel_state: str = "NORMAL",
        squawk: int | None = None,
    ) -> _TrackSeed:
        return _TrackSeed(
            index=index,
            callsign=callsign,
            x=float(x),
            y=float(y),
            heading=str(heading).upper(),
            speed_cells_per_min=max(1, int(speed)),
            squawk=int(squawk if squawk is not None else 2000 + self._rng.randint(0, 5777)),
            channel=max(1, min(9, int(channel))),
            altitude_fl=max(120, int(altitude_fl)),
            fuel_state=str(fuel_state).upper(),
            waypoint=str(waypoint).upper(),
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
            if kind is SituationalAwarenessQueryKind.FUTURE_POSITION:
                max_offset = max(15, min(35, duration_s - at_s - 4))
                future_offset_s = max(12, min(max_offset, 18 + self._rng.randint(0, 12)))
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
        if practice_focus is SituationalAwarenessQueryKind.FUTURE_POSITION:
            return (
                SituationalAwarenessQueryKind.FUTURE_POSITION,
                SituationalAwarenessQueryKind.FUTURE_POSITION,
            )
        if practice_focus in (
            SituationalAwarenessQueryKind.CONTACT_IDENTIFICATION,
            SituationalAwarenessQueryKind.CODE_OR_STATUS_RECALL,
        ):
            return (
                SituationalAwarenessQueryKind.CONTACT_IDENTIFICATION,
                SituationalAwarenessQueryKind.CODE_OR_STATUS_RECALL,
            )
        if practice_focus is SituationalAwarenessQueryKind.ACTION_SELECTION:
            return (
                SituationalAwarenessQueryKind.ACTION_SELECTION,
                SituationalAwarenessQueryKind.CODE_OR_STATUS_RECALL,
            )

        pressure_scale = 1.0 if training_profile is None else max(0.5, float(training_profile.pressure_scale))
        high_pressure = clamp01(difficulty) >= 0.7 or pressure_scale > 1.05
        mapping = {
            SituationalAwarenessScenarioFamily.MERGE_CONFLICT: (
                SituationalAwarenessQueryKind.FUTURE_POSITION,
                SituationalAwarenessQueryKind.CONTACT_IDENTIFICATION,
                SituationalAwarenessQueryKind.ACTION_SELECTION,
            ),
            SituationalAwarenessScenarioFamily.FUEL_PRIORITY: (
                SituationalAwarenessQueryKind.CODE_OR_STATUS_RECALL,
                SituationalAwarenessQueryKind.ACTION_SELECTION,
                SituationalAwarenessQueryKind.CONTACT_IDENTIFICATION,
            ),
            SituationalAwarenessScenarioFamily.ROUTE_HANDOFF: (
                SituationalAwarenessQueryKind.FUTURE_POSITION,
                SituationalAwarenessQueryKind.CONTACT_IDENTIFICATION,
                SituationalAwarenessQueryKind.CODE_OR_STATUS_RECALL,
            ),
            SituationalAwarenessScenarioFamily.CHANNEL_WAYPOINT_CHANGE: (
                SituationalAwarenessQueryKind.CODE_OR_STATUS_RECALL,
                SituationalAwarenessQueryKind.CONTACT_IDENTIFICATION,
                SituationalAwarenessQueryKind.FUTURE_POSITION,
            ),
        }
        base = mapping[family]
        if not high_pressure:
            return base
        return base + (
            SituationalAwarenessQueryKind.ACTION_SELECTION,
            SituationalAwarenessQueryKind.CODE_OR_STATUS_RECALL,
        )

    def _schedule_updates(
        self,
        updates: tuple[_TrackUpdateEvent, ...],
        *,
        duration_s: int,
        training_profile: SituationalAwarenessTrainingProfile | None,
    ) -> tuple[_TrackUpdateEvent, ...]:
        if training_profile is None:
            return updates
        scale = max(0.45, float(training_profile.update_time_scale))
        if abs(scale - 1.0) < 1e-6:
            return updates
        scheduled: list[_TrackUpdateEvent] = []
        seen: set[tuple[int, str, str]] = set()
        for update in updates:
            at_s = max(4, min(duration_s - 4, int(round(float(update.at_s) * scale))))
            key = (at_s, update.callsign, update.line)
            if key in seen:
                at_s = min(duration_s - 4, at_s + 1)
                key = (at_s, update.callsign, update.line)
            seen.add(key)
            scheduled.append(
                _TrackUpdateEvent(
                    at_s=at_s,
                    callsign=update.callsign,
                    line=update.line,
                    heading=update.heading,
                    speed_cells_per_min=update.speed_cells_per_min,
                    squawk=update.squawk,
                    channel=update.channel,
                    altitude_fl=update.altitude_fl,
                    fuel_state=update.fuel_state,
                    waypoint=update.waypoint,
                )
            )
        scheduled.sort(key=lambda item: (item.at_s, item.callsign))
        return tuple(scheduled)

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
        base_tracks = [
            self._make_track_seed(
                index=1,
                callsign=callsigns[0],
                x=1.0,
                y=4.0,
                heading="E",
                speed=5,
                channel=1,
                altitude_fl=240,
                waypoint="ECHO",
            ),
            self._make_track_seed(
                index=2,
                callsign=callsigns[1],
                x=8.0,
                y=4.0,
                heading="W",
                speed=5,
                channel=2,
                altitude_fl=250,
                waypoint="DELTA",
            ),
            self._make_track_seed(
                index=3,
                callsign=callsigns[2],
                x=2.0,
                y=1.0,
                heading="SE",
                speed=3,
                channel=3,
                altitude_fl=310,
                waypoint="CHARLIE",
            ),
            self._make_track_seed(
                index=4,
                callsign=callsigns[3],
                x=7.0,
                y=8.0,
                heading="NW",
                speed=3,
                channel=4,
                altitude_fl=210,
                waypoint="BRAVO",
            ),
        ]
        tracks = list(base_tracks[:track_count])
        if track_count >= 5:
            tracks.append(
                self._make_track_seed(
                    index=5,
                    callsign=callsigns[4],
                    x=0.0,
                    y=8.0,
                    heading="E",
                    speed=2,
                    channel=5,
                    altitude_fl=180,
                    waypoint="FOXTROT",
                )
            )

        update_events = [
            _TrackUpdateEvent(8, callsigns[0], f"{callsigns[0]} check in on channel 3.", channel=3),
            _TrackUpdateEvent(14, callsigns[1], f"{callsigns[1]} descend and hold FL230.", altitude_fl=230),
            _TrackUpdateEvent(22, callsigns[2], f"{callsigns[2]} turn south direct DELTA.", heading="S", waypoint="DELTA"),
            _TrackUpdateEvent(34, callsigns[1], f"{callsigns[1]} confirm visual and maintain course."),
        ]
        updates = self._schedule_updates(
            tuple(update_events),
            duration_s=duration_s,
            training_profile=training_profile,
        )
        return _ScenarioPlan(
            family=SituationalAwarenessScenarioFamily.MERGE_CONFLICT,
            label=f"Merge Conflict {scenario_index}/{total_scenarios}",
            duration_s=duration_s,
            waypoints=self._base_waypoints(),
            track_seeds=tuple(tracks),
            updates=updates,
            query_slots=self._make_query_slots(
                duration_s=duration_s,
                family=SituationalAwarenessScenarioFamily.MERGE_CONFLICT,
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
            context={
                "conflict_pair": (callsigns[0], callsigns[1]),
                "priority_callsign": callsigns[0],
            },
        )

    def _build_fuel_priority(
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
        base_tracks = [
            self._make_track_seed(
                index=1,
                callsign=callsigns[0],
                x=1.0,
                y=3.0,
                heading="E",
                speed=4,
                channel=1,
                altitude_fl=220,
                waypoint="BRAVO",
                fuel_state="LOW",
            ),
            self._make_track_seed(
                index=2,
                callsign=callsigns[1],
                x=8.0,
                y=3.0,
                heading="W",
                speed=4,
                channel=2,
                altitude_fl=240,
                waypoint="ALFA",
            ),
            self._make_track_seed(
                index=3,
                callsign=callsigns[2],
                x=2.0,
                y=7.0,
                heading="N",
                speed=3,
                channel=3,
                altitude_fl=300,
                waypoint="ECHO",
            ),
            self._make_track_seed(
                index=4,
                callsign=callsigns[3],
                x=7.0,
                y=1.0,
                heading="S",
                speed=3,
                channel=4,
                altitude_fl=200,
                waypoint="FOXTROT",
            ),
        ]
        tracks = list(base_tracks[:track_count])
        if track_count >= 5:
            tracks.append(
                self._make_track_seed(
                    index=5,
                    callsign=callsigns[4],
                    x=5.0,
                    y=8.0,
                    heading="NW",
                    speed=2,
                    channel=5,
                    altitude_fl=260,
                    waypoint="CHARLIE",
                )
            )

        update_events = [
            _TrackUpdateEvent(6, callsigns[0], f"{callsigns[0]} reports minimum fuel.", fuel_state="MIN"),
            _TrackUpdateEvent(12, callsigns[0], f"{callsigns[0]} switch channel 2 and continue BRAVO.", channel=2),
            _TrackUpdateEvent(18, callsigns[1], f"{callsigns[1]} adjust to heading NW.", heading="NW"),
        ]
        if track_count >= 4:
            update_events.append(
                _TrackUpdateEvent(24, callsigns[3], f"{callsigns[3]} climb and hold FL220.", altitude_fl=220)
            )
        updates = self._schedule_updates(
            tuple(update_events),
            duration_s=duration_s,
            training_profile=training_profile,
        )
        return _ScenarioPlan(
            family=SituationalAwarenessScenarioFamily.FUEL_PRIORITY,
            label=f"Fuel Priority {scenario_index}/{total_scenarios}",
            duration_s=duration_s,
            waypoints=self._base_waypoints(),
            track_seeds=tuple(tracks),
            updates=updates,
            query_slots=self._make_query_slots(
                duration_s=duration_s,
                family=SituationalAwarenessScenarioFamily.FUEL_PRIORITY,
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
            context={
                "conflict_pair": (callsigns[0], callsigns[1]),
                "priority_callsign": callsigns[0],
            },
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
        base_tracks = [
            self._make_track_seed(
                index=1,
                callsign=callsigns[0],
                x=1.0,
                y=7.0,
                heading="NE",
                speed=4,
                channel=1,
                altitude_fl=260,
                waypoint="BRAVO",
            ),
            self._make_track_seed(
                index=2,
                callsign=callsigns[1],
                x=7.0,
                y=1.0,
                heading="SW",
                speed=3,
                channel=2,
                altitude_fl=240,
                waypoint="DELTA",
            ),
            self._make_track_seed(
                index=3,
                callsign=callsigns[2],
                x=2.0,
                y=2.0,
                heading="E",
                speed=2,
                channel=3,
                altitude_fl=190,
                waypoint="CHARLIE",
            ),
            self._make_track_seed(
                index=4,
                callsign=callsigns[3],
                x=8.0,
                y=8.0,
                heading="W",
                speed=2,
                channel=4,
                altitude_fl=320,
                waypoint="ALFA",
            ),
        ]
        tracks = list(base_tracks[:track_count])
        if track_count >= 5:
            tracks.append(
                self._make_track_seed(
                    index=5,
                    callsign=callsigns[4],
                    x=4.0,
                    y=8.0,
                    heading="N",
                    speed=2,
                    channel=5,
                    altitude_fl=280,
                    waypoint="FOXTROT",
                )
            )

        update_events = [
            _TrackUpdateEvent(10, callsigns[0], f"{callsigns[0]} hand off to channel 3, direct DELTA.", channel=3, waypoint="DELTA"),
            _TrackUpdateEvent(18, callsigns[0], f"{callsigns[0]} turn east and continue DELTA.", heading="E"),
            _TrackUpdateEvent(26, callsigns[1], f"{callsigns[1]} change squawk and hold FL250.", squawk=4100 + self._rng.randint(0, 499), altitude_fl=250),
            _TrackUpdateEvent(34, callsigns[2], f"{callsigns[2]} proceed direct ECHO.", waypoint="ECHO"),
        ]
        updates = self._schedule_updates(
            tuple(update_events),
            duration_s=duration_s,
            training_profile=training_profile,
        )
        return _ScenarioPlan(
            family=SituationalAwarenessScenarioFamily.ROUTE_HANDOFF,
            label=f"Route Handoff {scenario_index}/{total_scenarios}",
            duration_s=duration_s,
            waypoints=self._base_waypoints(),
            track_seeds=tuple(tracks),
            updates=updates,
            query_slots=self._make_query_slots(
                duration_s=duration_s,
                family=SituationalAwarenessScenarioFamily.ROUTE_HANDOFF,
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
            context={"handoff_callsign": callsigns[0]},
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
        base_tracks = [
            self._make_track_seed(
                index=1,
                callsign=callsigns[0],
                x=2.0,
                y=2.0,
                heading="E",
                speed=3,
                channel=1,
                altitude_fl=210,
                waypoint="BRAVO",
            ),
            self._make_track_seed(
                index=2,
                callsign=callsigns[1],
                x=7.0,
                y=6.0,
                heading="W",
                speed=4,
                channel=4,
                altitude_fl=250,
                waypoint="ALFA",
            ),
            self._make_track_seed(
                index=3,
                callsign=callsigns[2],
                x=1.0,
                y=8.0,
                heading="NE",
                speed=3,
                channel=3,
                altitude_fl=300,
                waypoint="ECHO",
            ),
            self._make_track_seed(
                index=4,
                callsign=callsigns[3],
                x=8.0,
                y=1.0,
                heading="SW",
                speed=2,
                channel=2,
                altitude_fl=270,
                waypoint="DELTA",
            ),
        ]
        tracks = list(base_tracks[:track_count])
        if track_count >= 5:
            tracks.append(
                self._make_track_seed(
                    index=5,
                    callsign=callsigns[4],
                    x=4.0,
                    y=5.0,
                    heading="S",
                    speed=2,
                    channel=5,
                    altitude_fl=230,
                    waypoint="CHARLIE",
                )
            )

        update_events = [
            _TrackUpdateEvent(8, callsigns[1], f"{callsigns[1]} change to channel 2 and proceed ECHO.", channel=2, waypoint="ECHO"),
            _TrackUpdateEvent(16, callsigns[0], f"{callsigns[0]} update squawk code and hold FL230.", squawk=4300 + self._rng.randint(0, 499), altitude_fl=230),
            _TrackUpdateEvent(24, callsigns[2], f"{callsigns[2]} report low fuel and continue ALFA.", fuel_state="LOW", waypoint="ALFA"),
        ]
        if track_count >= 4:
            update_events.append(
                _TrackUpdateEvent(32, callsigns[3], f"{callsigns[3]} turn west and hold channel 1.", heading="W", channel=1)
            )
        updates = self._schedule_updates(
            tuple(update_events),
            duration_s=duration_s,
            training_profile=training_profile,
        )
        return _ScenarioPlan(
            family=SituationalAwarenessScenarioFamily.CHANNEL_WAYPOINT_CHANGE,
            label=f"Channel Shift {scenario_index}/{total_scenarios}",
            duration_s=duration_s,
            waypoints=self._base_waypoints(),
            track_seeds=tuple(tracks),
            updates=updates,
            query_slots=self._make_query_slots(
                duration_s=duration_s,
                family=SituationalAwarenessScenarioFamily.CHANNEL_WAYPOINT_CHANGE,
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
            context={"handoff_callsign": callsigns[1]},
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
            "Monitor verbal, numerical, pictorial, and coded information on a live tactical display.",
            "Build and update a mental picture of moving tracks, channel changes, fuel priorities, and future conflicts.",
            "Answer direct-response queries about future position, track identity, coded status, and the best immediate action.",
            "",
            "Answer modes:",
            "- Grid cell: click a cell or type row+column, then Enter",
            "- Track index: press 1-5 or click a track row, then Enter",
            "- Action: press 1-4 or click an action card, then Enter",
            "",
            "Practice runs three guided 45-second scenarios before the 25-minute timed block.",
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
            SituationalAwarenessQueryKind.FUTURE_POSITION,
            SituationalAwarenessQueryKind.CONTACT_IDENTIFICATION,
            SituationalAwarenessQueryKind.ACTION_SELECTION,
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
        self._live_tracks: dict[str, _LiveTrackState] = {}
        self._recent_feed_lines: deque[str] = deque(maxlen=4)
        self._feed_cursor = 0
        self._query_slot_cursor = 0
        self._current_query: _ActiveQueryState | None = None
        self._current_problem: _CurrentProblem | None = None
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
            self._current_problem = None

    def submit_answer(self, raw: str) -> bool:
        if self._phase not in (Phase.PRACTICE, Phase.SCORED):
            return False
        if self._current_query is None or self._scenario_started_at_s is None:
            return False
        normalized = self._normalize_submission(raw, answer_mode=self._current_query.answer_mode)
        if normalized is None:
            return False
        event = self._record_query_result(raw=normalized, is_timeout=False)
        is_correct = event.is_correct
        if self._phase is Phase.PRACTICE:
            expected = self._current_query.correct_answer_token
            self._practice_feedback = (
                f"Correct. {expected} was the right answer."
                if is_correct
                else f"Incorrect. Correct answer: {expected}."
            )
        self._current_query = None
        self._current_problem = None
        return True

    def time_remaining_s(self) -> float | None:
        if self._phase is not Phase.SCORED or self._scored_started_at_s is None:
            return None
        remaining = float(self._config.scored_duration_s) - (self._clock.now() - self._scored_started_at_s)
        return max(0.0, remaining)

    def snapshot(self) -> TestSnapshot:
        if self._phase is Phase.INSTRUCTIONS:
            prompt = "Press Enter to begin the guided practice scenarios."
            return TestSnapshot(
                title=self._title,
                phase=self._phase,
                prompt=prompt,
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
            else f"Monitor the picture. Next query in {int(round(payload.next_query_in_s or 0.0)):02d}s."
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
            self._current_problem = None
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
            self._current_problem = None
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
        self._feed_cursor = 0
        self._query_slot_cursor = 0
        self._current_query = None
        self._current_problem = None
        self._live_tracks = {
            seed.callsign: _LiveTrackState(
                index=seed.index,
                callsign=seed.callsign,
                x=float(seed.x),
                y=float(seed.y),
                heading=seed.heading,
                speed_cells_per_min=seed.speed_cells_per_min,
                squawk=seed.squawk,
                channel=seed.channel,
                altitude_fl=seed.altitude_fl,
                fuel_state=seed.fuel_state,
                waypoint=seed.waypoint,
            )
            for seed in self._scenario_plan.track_seeds
        }
        self._recent_feed_lines.clear()
        self._set_announcement(
            lines=(
                f"{self._scenario_plan.label}.",
                self._scenario_intro_line(self._scenario_plan),
            ),
            reason=("scenario", self._phase.value, self._scenario_index),
        )
        self._recent_feed_lines.append(self._scenario_intro_line(self._scenario_plan))

    @staticmethod
    def _scenario_intro_line(plan: _ScenarioPlan) -> str:
        if plan.family is SituationalAwarenessScenarioFamily.MERGE_CONFLICT:
            return "Monitor converging traffic and identify the correct deconfliction action."
        if plan.family is SituationalAwarenessScenarioFamily.FUEL_PRIORITY:
            return "Watch fuel priority, route overlap, and the direct track for the minimum-fuel flight."
        if plan.family is SituationalAwarenessScenarioFamily.ROUTE_HANDOFF:
            return "Track route handoff updates, channel changes, and future position after the handoff."
        return "Monitor channel, squawk, altitude, and waypoint changes while the picture evolves."

    def _end_current_scenario(self) -> None:
        if self._current_query is not None:
            self._record_query_result(raw="", is_timeout=True)
            self._current_query = None
            self._current_problem = None

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
            self._current_problem = None

        for track in self._live_tracks.values():
            dx, dy = _HEADING_VECTORS.get(track.heading, (0, 0))
            track.x = max(0.0, min(float(_GRID_SIZE - 1), track.x + (dx * track.speed_cells_per_min / 60.0)))
            track.y = max(0.0, min(float(_GRID_SIZE - 1), track.y + (dy * track.speed_cells_per_min / 60.0)))

        while (
            self._feed_cursor < len(self._scenario_plan.updates)
            and int(self._scenario_plan.updates[self._feed_cursor].at_s) == int(tick)
        ):
            self._apply_update(self._scenario_plan.updates[self._feed_cursor], tick=tick)
            self._feed_cursor += 1

        if self._current_query is None and self._query_slot_cursor < len(self._scenario_plan.query_slots):
            slot = self._scenario_plan.query_slots[self._query_slot_cursor]
            if int(slot.at_s) <= int(tick):
                self._query_slot_cursor += 1
                self._spawn_query(slot=slot, tick=tick)

    def _apply_update(self, update: _TrackUpdateEvent, *, tick: int) -> None:
        track = self._live_tracks.get(update.callsign)
        if track is None:
            return
        if update.heading is not None:
            track.heading = str(update.heading).upper()
        if update.speed_cells_per_min is not None:
            track.speed_cells_per_min = max(1, int(update.speed_cells_per_min))
        if update.squawk is not None:
            track.squawk = int(update.squawk)
        if update.channel is not None:
            track.channel = int(update.channel)
        if update.altitude_fl is not None:
            track.altitude_fl = int(update.altitude_fl)
        if update.fuel_state is not None:
            track.fuel_state = str(update.fuel_state).upper()
        if update.waypoint is not None:
            track.waypoint = str(update.waypoint).upper()
        line = f"T+{tick:02d} {update.line}"
        self._recent_feed_lines.append(line)
        self._set_announcement(lines=(update.line,), reason=("feed", self._scenario_index, tick, update.callsign))

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
        self._current_query = query
        self._current_problem = _CurrentProblem(prompt=query.prompt, answer=query.correct_answer_token)
        self._set_announcement(
            lines=(
                self._spoken_family_tag(self._scenario_plan.family),
                query.prompt,
            ),
            reason=("query", self._scenario_index, query.query_id),
        )

    @staticmethod
    def _spoken_family_tag(family: SituationalAwarenessScenarioFamily) -> str:
        if family is SituationalAwarenessScenarioFamily.MERGE_CONFLICT:
            return "Traffic merge update."
        if family is SituationalAwarenessScenarioFamily.FUEL_PRIORITY:
            return "Fuel priority update."
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
        kind = slot.kind
        if kind is SituationalAwarenessQueryKind.FUTURE_POSITION:
            query = self._build_future_position_query(
                query_id=query_id,
                tick=tick,
                expires_at_s=expires_at_s,
                future_offset_s=max(12, int(slot.future_offset_s or 20)),
            )
            if query is not None:
                return query
        if kind is SituationalAwarenessQueryKind.CONTACT_IDENTIFICATION:
            query = self._build_contact_identification_query(
                query_id=query_id,
                tick=tick,
                expires_at_s=expires_at_s,
            )
            if query is not None:
                return query
        if kind is SituationalAwarenessQueryKind.CODE_OR_STATUS_RECALL:
            query = self._build_code_status_query(
                query_id=query_id,
                tick=tick,
                expires_at_s=expires_at_s,
            )
            if query is not None:
                return query
        query = self._build_action_query(query_id=query_id, tick=tick, expires_at_s=expires_at_s)
        if query is not None:
            return query
        return self._build_contact_identification_query(
            query_id=query_id,
            tick=tick,
            expires_at_s=expires_at_s,
        ) or self._build_code_status_query(
            query_id=query_id,
            tick=tick,
            expires_at_s=expires_at_s,
        ) or self._build_future_position_query(
            query_id=query_id,
            tick=tick,
            expires_at_s=expires_at_s,
            future_offset_s=18,
        ) or _ActiveQueryState(
            query_id=query_id,
            asked_at_s=tick,
            expires_at_s=expires_at_s,
            kind=SituationalAwarenessQueryKind.CODE_OR_STATUS_RECALL,
            answer_mode=SituationalAwarenessAnswerMode.TRACK_INDEX,
            prompt="Which indexed track is currently active?",
            correct_answer_token="1",
        )

    def _build_future_position_query(
        self,
        *,
        query_id: int,
        tick: int,
        expires_at_s: int,
        future_offset_s: int,
    ) -> _ActiveQueryState | None:
        if self._scenario_plan is None:
            return None
        valid_tracks = [track for track in self._live_tracks.values() if track.speed_cells_per_min > 0]
        if not valid_tracks:
            return None
        target = self._pick_track_for_query(valid_tracks)
        projected = self._project_track_cell(
            callsign=target.callsign,
            start_tick=tick,
            offset_s=min(future_offset_s, max(6, int(self._scenario_plan.duration_s) - tick)),
        )
        prompt = (
            f"Future position: where will track {target.index} ({target.callsign}) be in "
            f"{future_offset_s}s if current routing and updates continue?"
        )
        return _ActiveQueryState(
            query_id=query_id,
            asked_at_s=tick,
            expires_at_s=expires_at_s,
            kind=SituationalAwarenessQueryKind.FUTURE_POSITION,
            answer_mode=SituationalAwarenessAnswerMode.GRID_CELL,
            prompt=prompt,
            correct_answer_token=projected,
            subject_callsign=target.callsign,
            future_offset_s=future_offset_s,
        )

    def _build_contact_identification_query(
        self,
        *,
        query_id: int,
        tick: int,
        expires_at_s: int,
    ) -> _ActiveQueryState | None:
        targets = sorted(self._live_tracks.values(), key=lambda item: item.index)
        for target in targets:
            cell = cell_label_from_xy(target.x, target.y)
            prompt = (
                f"Contact identification: which indexed track is at {cell}, "
                f"heading {target.heading}, on CH {target.channel}?"
            )
            if self._track_index_query_is_unique(
                target=target,
                predicate=lambda track: (
                    cell_label_from_xy(track.x, track.y) == cell
                    and track.heading == target.heading
                    and int(track.channel) == int(target.channel)
                ),
            ):
                return _ActiveQueryState(
                    query_id=query_id,
                    asked_at_s=tick,
                    expires_at_s=expires_at_s,
                    kind=SituationalAwarenessQueryKind.CONTACT_IDENTIFICATION,
                    answer_mode=SituationalAwarenessAnswerMode.TRACK_INDEX,
                    prompt=prompt,
                    correct_answer_token=str(target.index),
                    subject_callsign=target.callsign,
                )
        return None

    def _build_code_status_query(
        self,
        *,
        query_id: int,
        tick: int,
        expires_at_s: int,
    ) -> _ActiveQueryState | None:
        targets = sorted(self._live_tracks.values(), key=lambda item: item.index)
        for target in targets:
            prompt = (
                f"Code/status recall: which indexed track is squawking {target.squawk:04d} "
                f"at FL{target.altitude_fl} on CH {target.channel}?"
            )
            if self._track_index_query_is_unique(
                target=target,
                predicate=lambda track: (
                    int(track.squawk) == int(target.squawk)
                    and int(track.altitude_fl) == int(target.altitude_fl)
                    and int(track.channel) == int(target.channel)
                ),
            ):
                return _ActiveQueryState(
                    query_id=query_id,
                    asked_at_s=tick,
                    expires_at_s=expires_at_s,
                    kind=SituationalAwarenessQueryKind.CODE_OR_STATUS_RECALL,
                    answer_mode=SituationalAwarenessAnswerMode.TRACK_INDEX,
                    prompt=prompt,
                    correct_answer_token=str(target.index),
                    subject_callsign=target.callsign,
                )
        return None

    def _build_action_query(
        self,
        *,
        query_id: int,
        tick: int,
        expires_at_s: int,
    ) -> _ActiveQueryState | None:
        if self._scenario_plan is None:
            return None
        pair = self._scenario_plan.context.get("conflict_pair")
        priority = str(self._scenario_plan.context.get("priority_callsign", ""))
        if not isinstance(pair, tuple) or len(pair) != 2 or priority == "":
            return None
        other = pair[1] if pair[0] == priority else pair[0]
        priority_track = self._live_tracks.get(priority)
        other_track = self._live_tracks.get(other)
        if priority_track is None or other_track is None:
            return None
        merge_hint = cell_label_from_xy(
            (priority_track.x + other_track.x) / 2.0,
            (priority_track.y + other_track.y) / 2.0,
        )
        if self._scenario_plan.family is SituationalAwarenessScenarioFamily.FUEL_PRIORITY:
            choices = (
                SituationalAwarenessAnswerChoice(
                    1,
                    f"Keep {priority} direct and vector track {other_track.index} off the merge path.",
                ),
                SituationalAwarenessAnswerChoice(
                    2,
                    f"Hold {priority} and keep track {other_track.index} direct.",
                ),
                SituationalAwarenessAnswerChoice(
                    3,
                    "Delay both tracks and change both to standby channel.",
                ),
                SituationalAwarenessAnswerChoice(
                    4,
                    "Maintain current routing and monitor one more sweep.",
                ),
            )
            prompt = (
                f"Action selection: {priority} is {priority_track.fuel_state}. "
                f"Best immediate action near {merge_hint}?"
            )
            correct = "1"
        else:
            choices = (
                SituationalAwarenessAnswerChoice(
                    1,
                    f"Vector track {other_track.index} away and keep {priority} direct.",
                ),
                SituationalAwarenessAnswerChoice(
                    2,
                    f"Hold {priority} and keep track {other_track.index} direct.",
                ),
                SituationalAwarenessAnswerChoice(
                    3,
                    "Delay both tracks and reconfirm both squawks before acting.",
                ),
                SituationalAwarenessAnswerChoice(
                    4,
                    "Maintain current routes and wait one more query cycle.",
                ),
            )
            prompt = (
                f"Action selection: tracks {priority_track.index} and {other_track.index} "
                f"converge near {merge_hint}. Best immediate action?"
            )
            correct = "1"
        return _ActiveQueryState(
            query_id=query_id,
            asked_at_s=tick,
            expires_at_s=expires_at_s,
            kind=SituationalAwarenessQueryKind.ACTION_SELECTION,
            answer_mode=SituationalAwarenessAnswerMode.ACTION,
            prompt=prompt,
            correct_answer_token=correct,
            subject_callsign=priority,
            answer_choices=choices,
        )

    def _track_index_query_is_unique(
        self,
        *,
        target: _LiveTrackState,
        predicate,
    ) -> bool:
        matches = [track for track in self._live_tracks.values() if predicate(track)]
        return len(matches) == 1 and matches[0].callsign == target.callsign

    def _pick_track_for_query(self, tracks: list[_LiveTrackState]) -> _LiveTrackState:
        preferred = sorted(tracks, key=lambda item: (item.index, item.callsign))
        return preferred[(self._scenario_index + self._query_slot_cursor - 1) % len(preferred)]

    def _project_track_cell(self, *, callsign: str, start_tick: int, offset_s: int) -> str:
        if self._scenario_plan is None:
            return "A0"
        tracks = {name: track.copy() for name, track in self._live_tracks.items()}
        end_tick = min(int(self._scenario_plan.duration_s), int(start_tick) + max(1, int(offset_s)))
        pending_updates = [event for event in self._scenario_plan.updates if int(event.at_s) > int(start_tick)]
        update_cursor = 0
        for sim_tick in range(int(start_tick) + 1, end_tick + 1):
            for track in tracks.values():
                dx, dy = _HEADING_VECTORS.get(track.heading, (0, 0))
                track.x = max(0.0, min(float(_GRID_SIZE - 1), track.x + (dx * track.speed_cells_per_min / 60.0)))
                track.y = max(0.0, min(float(_GRID_SIZE - 1), track.y + (dy * track.speed_cells_per_min / 60.0)))
            while update_cursor < len(pending_updates) and int(pending_updates[update_cursor].at_s) == sim_tick:
                update = pending_updates[update_cursor]
                track = tracks.get(update.callsign)
                if track is not None:
                    if update.heading is not None:
                        track.heading = str(update.heading).upper()
                    if update.speed_cells_per_min is not None:
                        track.speed_cells_per_min = int(update.speed_cells_per_min)
                    if update.squawk is not None:
                        track.squawk = int(update.squawk)
                    if update.channel is not None:
                        track.channel = int(update.channel)
                    if update.altitude_fl is not None:
                        track.altitude_fl = int(update.altitude_fl)
                    if update.fuel_state is not None:
                        track.fuel_state = str(update.fuel_state).upper()
                    if update.waypoint is not None:
                        track.waypoint = str(update.waypoint).upper()
                update_cursor += 1
        projected = tracks.get(callsign)
        if projected is None:
            return "A0"
        return cell_label_from_xy(projected.x, projected.y)

    def _record_query_result(self, *, raw: str, is_timeout: bool) -> QuestionEvent:
        assert self._current_query is not None
        assert self._scenario_started_at_s is not None
        prompt = self._current_query.prompt
        correct = self._current_query.correct_answer_token
        now = self._clock.now()
        presented_at = self._scenario_started_at_s + float(self._current_query.asked_at_s)
        if is_timeout:
            answered_at = self._scenario_started_at_s + float(self._current_query.expires_at_s)
            response_time = max(0.0, answered_at - presented_at)
            user_answer = -1
            submitted = ""
            is_correct = False
        else:
            answered_at = now
            response_time = max(0.0, answered_at - presented_at)
            submitted = str(raw)
            user_answer = self._user_answer_value(raw=submitted)
            is_correct = submitted == correct

        score = 1.0 if is_correct else 0.0
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
            return -1
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
        if answer_mode is SituationalAwarenessAnswerMode.ACTION:
            return str(value) if 1 <= value <= 4 else None
        return str(value) if 1 <= value <= max(5, len(self._live_tracks)) else None

    def _active_channels(self) -> tuple[str, ...]:
        if self._current_segment is None:
            return SA_CHANNEL_ORDER
        return self._current_segment.active_channels

    def _active_query_kind_names(self) -> tuple[str, ...]:
        if self._current_segment is None:
            return SA_QUERY_KIND_ORDER
        return self._current_segment.active_query_kinds

    def _focus_label(self) -> str:
        if self._current_segment is None:
            return "Full mixed picture"
        return self._current_segment.focus_label

    def _segment_label(self) -> str:
        if self._current_segment is None:
            return self._scenario_plan.label if self._scenario_plan is not None else self._title
        return self._current_segment.label

    def _payload(self) -> SituationalAwarenessPayload:
        assert self._scenario_plan is not None
        assert self._scenario_started_at_s is not None
        scenario_elapsed_s = max(0.0, self._clock.now() - self._scenario_started_at_s)
        scenario_remaining_s = max(0.0, float(self._scenario_plan.duration_s) - scenario_elapsed_s)

        tracks = tuple(
            SituationalAwarenessTrack(
                index=track.index,
                callsign=track.callsign,
                x=float(track.x),
                y=float(track.y),
                cell_label=cell_label_from_xy(track.x, track.y),
                heading=track.heading,
                speed_cells_per_min=int(track.speed_cells_per_min),
                squawk=int(track.squawk),
                channel=int(track.channel),
                altitude_fl=int(track.altitude_fl),
                fuel_state=str(track.fuel_state),
                waypoint=str(track.waypoint),
            )
            for track in sorted(self._live_tracks.values(), key=lambda item: item.index)
        )
        status_entries = tuple(
            SituationalAwarenessStatusEntry(
                track_index=track.index,
                callsign=track.callsign,
                cell_label=track.cell_label,
                heading=track.heading,
                speed_cells_per_min=track.speed_cells_per_min,
                squawk=track.squawk,
                channel=track.channel,
                altitude_fl=track.altitude_fl,
                fuel_state=track.fuel_state,
                waypoint=track.waypoint,
            )
            for track in tracks
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
            tracks=tracks,
            status_entries=status_entries,
            waypoints=self._scenario_plan.waypoints,
            recent_feed_lines=tuple(self._recent_feed_lines),
            active_query=active_query,
            answer_mode=answer_mode,
            correct_answer_token=correct_answer_token,
            announcement_token=self._announcement_token,
            announcement_lines=self._announcement_lines,
        )

    def _input_hint(self, payload: SituationalAwarenessPayload) -> str:
        if payload.active_query is None:
            return "Monitor the grid, coded panel, and aural feed for the next query."
        if payload.answer_mode is SituationalAwarenessAnswerMode.GRID_CELL:
            return "Click a grid cell or type row+column, then press Enter."
        if payload.answer_mode is SituationalAwarenessAnswerMode.TRACK_INDEX:
            return "Click a track row or press 1-5, then press Enter."
        return "Click an action card or press 1-4, then press Enter."

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
