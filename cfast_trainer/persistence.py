from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from .adaptive_difficulty import (
    AdaptiveDifficultyState,
    family_id_for_code,
    merge_adaptive_state,
    metric_float,
    scope_keys_for_code,
)
from .primitive_ranking import rank_primitives
from .results import AttemptResult
from .telemetry import TelemetryEvent, lifecycle_event

RESULTS_DB_ENV = "CFAST_RESULTS_DB_PATH"
SCHEMA_VERSION = 5


def open_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    _migrate(conn)
    return conn


def _utc_now_epoch() -> float:
    return time.time()


def _utc_now_iso() -> str:
    return _utc_iso_from_epoch(_utc_now_epoch())


def _utc_iso_from_epoch(epoch_s: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch_s))


@dataclass(frozen=True, slots=True)
class SavedAttempt:
    attempt_id: int
    session_id: int
    activity_session_id: int


@dataclass(frozen=True, slots=True)
class AttemptHistoryEntry:
    attempt_id: int
    session_id: int
    activity_session_id: int | None
    activity_code: str | None
    activity_kind: str | None
    test_code: str
    test_version: int
    rng_seed: int
    difficulty: float
    started_at_utc: str
    completed_at_utc: str
    difficulty_level_start: int | None
    difficulty_level_end: int | None
    metrics: dict[str, str]
    parent_activity_session_id: int | None = None
    origin_activity_code: str | None = None
    origin_activity_kind: str | None = None
    origin_item_index: int | None = None


@dataclass(frozen=True, slots=True)
class SessionSummary:
    session_id: int
    started_at_utc: str
    completed_at_utc: str
    exit_reason: str | None
    activity_count: int
    completed_activity_count: int
    aborted_activity_count: int
    attempt_count: int
    unique_tests: int
    mean_accuracy: float | None
    mean_score_ratio: float | None


@dataclass(frozen=True, slots=True)
class TestSessionSummary:
    session_id: int
    test_code: str
    attempt_count: int
    latest_accuracy: float | None
    best_accuracy: float | None
    latest_score_ratio: float | None
    best_score_ratio: float | None


@dataclass(frozen=True, slots=True)
class DifficultyStateSummary:
    scope_kind: str
    scope_key: str
    recommended_level: int
    last_start_level: int | None
    last_end_level: int | None
    ewma_accuracy: float | None
    ewma_score_ratio: float | None
    ewma_timeout_rate: float | None
    ewma_mean_rt_ms: float | None
    sample_count: int
    updated_at_utc: str
    mastery: float | None = None
    speed: float | None = None
    fatigue_penalty: float | None = None
    post_error_penalty: float | None = None
    instability_penalty: float | None = None
    retention_need: float | None = None
    confidence: float | None = None
    leverage: float | None = None
    level_confidence: float | None = None
    last_successful_level: int | None = None
    last_meltdown_level: int | None = None


class ResultsStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._session_id: int | None = None

    @classmethod
    def default_path(cls) -> Path:
        explicit = os.environ.get(RESULTS_DB_ENV)
        if explicit:
            return Path(explicit).expanduser()
        return Path.home() / ".rcaf_cfast_results.sqlite3"

    @property
    def path(self) -> Path:
        return self._path

    @property
    def session_id(self) -> int | None:
        return self._session_id

    def start_app_session(self, *, app_version: str) -> int:
        conn = open_db(self._path)
        try:
            self._session_id = _ensure_session(
                conn=conn,
                session_id=self._session_id,
                app_version=app_version,
            )
            _refresh_session_materializations(conn=conn, session_id=self._session_id)
            return int(self._session_id)
        finally:
            conn.close()

    def close_app_session(self, *, exit_reason: str = "app_quit") -> None:
        if self._session_id is None:
            return
        conn = open_db(self._path)
        try:
            _finalize_session(
                conn=conn,
                session_id=self._session_id,
                exit_reason=exit_reason,
            )
        finally:
            conn.close()

    def start_activity_session(
        self,
        *,
        activity_code: str,
        activity_kind: str,
        app_version: str,
        test_version: int,
        engine: object,
        input_profile_id: str | None = None,
        parent_activity_session_id: int | None = None,
        origin_activity_code: str | None = None,
        origin_activity_kind: str | None = None,
        origin_item_index: int | None = None,
    ) -> int:
        conn = open_db(self._path)
        try:
            self._session_id = _ensure_session(
                conn=conn,
                session_id=self._session_id,
                app_version=app_version,
            )
            activity_session_id = _insert_activity_session(
                conn=conn,
                session_id=self._session_id,
                activity_code=activity_code,
                activity_kind=activity_kind,
                app_version=app_version,
                test_version=test_version,
                engine=engine,
                input_profile_id=input_profile_id,
                parent_activity_session_id=parent_activity_session_id,
                origin_activity_code=origin_activity_code,
                origin_activity_kind=origin_activity_kind,
                origin_item_index=origin_item_index,
            )
            _refresh_session_materializations(conn=conn, session_id=self._session_id)
            return activity_session_id
        finally:
            conn.close()

    def complete_activity_session(
        self,
        *,
        activity_session_id: int,
        result: AttemptResult,
        app_version: str,
        input_profile_id: str | None = None,
        completion_reason: str = "completed",
    ) -> SavedAttempt | None:
        conn = open_db(self._path)
        try:
            row = _fetch_activity_session_row(conn=conn, activity_session_id=activity_session_id)
            if row is None:
                return None
            self._session_id = int(row["session_id"])
            if row["ended_at_utc"] is not None:
                attempt_id = _existing_attempt_id_for_activity(
                    conn=conn,
                    activity_session_id=activity_session_id,
                )
                if attempt_id is None:
                    return None
                return SavedAttempt(
                    attempt_id=attempt_id,
                    session_id=int(row["session_id"]),
                    activity_session_id=activity_session_id,
                )

            attempt_id = _insert_attempt(
                conn=conn,
                session_id=int(row["session_id"]),
                activity_session_id=activity_session_id,
                result=result,
                app_version=app_version,
                input_profile_id=input_profile_id,
                started_at_utc=str(row["started_at_utc"]),
            )
            _write_activity_metrics(
                conn=conn,
                activity_session_id=activity_session_id,
                result=result,
            )
            _update_difficulty_states(conn=conn, result=result)
            _write_telemetry_events(
                conn=conn,
                session_id=int(row["session_id"]),
                activity_session_id=activity_session_id,
                attempt_id=attempt_id,
                events=result.events,
            )
            _finalize_activity_session(
                conn=conn,
                session_id=int(row["session_id"]),
                activity_session_id=activity_session_id,
                completion_reason=completion_reason,
            )
            _refresh_session_materializations(conn=conn, session_id=int(row["session_id"]))
            return SavedAttempt(
                attempt_id=attempt_id,
                session_id=int(row["session_id"]),
                activity_session_id=activity_session_id,
            )
        finally:
            conn.close()

    def abort_activity_session(
        self,
        *,
        activity_session_id: int,
        app_version: str,
        completion_reason: str,
        result: AttemptResult | None = None,
        input_profile_id: str | None = None,
    ) -> None:
        conn = open_db(self._path)
        try:
            row = _fetch_activity_session_row(conn=conn, activity_session_id=activity_session_id)
            if row is None:
                return
            self._session_id = int(row["session_id"])
            if row["ended_at_utc"] is not None:
                return

            if result is not None:
                _write_activity_metrics(
                    conn=conn,
                    activity_session_id=activity_session_id,
                    result=result,
                )
                _update_difficulty_states(conn=conn, result=result)
                _write_telemetry_events(
                    conn=conn,
                    session_id=int(row["session_id"]),
                    activity_session_id=activity_session_id,
                    attempt_id=None,
                    events=result.events,
                )
            _update_activity_session_runtime_metadata(
                conn=conn,
                activity_session_id=activity_session_id,
                app_version=app_version,
                input_profile_id=input_profile_id,
                result=result,
            )
            _finalize_activity_session(
                conn=conn,
                session_id=int(row["session_id"]),
                activity_session_id=activity_session_id,
                completion_reason=completion_reason,
            )
            _refresh_session_materializations(conn=conn, session_id=int(row["session_id"]))
        finally:
            conn.close()

    def record_attempt(
        self,
        *,
        result: AttemptResult,
        app_version: str,
        input_profile_id: str | None = None,
        activity_code: str | None = None,
        activity_kind: str = "legacy_attempt",
        test_version: int | None = None,
        parent_activity_session_id: int | None = None,
        origin_activity_code: str | None = None,
        origin_activity_kind: str | None = None,
        origin_item_index: int | None = None,
    ) -> SavedAttempt:
        activity_session_id = self.start_activity_session(
            activity_code=str(activity_code or result.test_code),
            activity_kind=str(activity_kind),
            app_version=app_version,
            test_version=result.test_version if test_version is None else int(test_version),
            engine=_AttemptResultEngineShim(result),
            input_profile_id=input_profile_id,
            parent_activity_session_id=parent_activity_session_id,
            origin_activity_code=origin_activity_code,
            origin_activity_kind=origin_activity_kind,
            origin_item_index=origin_item_index,
        )
        saved = self.complete_activity_session(
            activity_session_id=activity_session_id,
            result=result,
            app_version=app_version,
            input_profile_id=input_profile_id,
            completion_reason="completed",
        )
        if saved is None:
            raise RuntimeError("failed to persist activity session")
        return saved

    def session_summary(self) -> SessionSummary | None:
        if self._session_id is None:
            return None
        return load_session_summary(db_path=self._path, session_id=self._session_id)

    def test_session_summary(self, test_code: str) -> TestSessionSummary | None:
        if self._session_id is None:
            return None
        return load_test_session_summary(
            db_path=self._path,
            session_id=self._session_id,
            test_code=test_code,
        )

    def recent_attempt_history(
        self,
        *,
        since_days: int | None = 28,
        limit: int | None = None,
        child_item_preferred: bool = False,
    ) -> list[AttemptHistoryEntry]:
        return load_recent_attempt_history(
            db_path=self._path,
            since_days=since_days,
            limit=limit,
            child_item_preferred=child_item_preferred,
        )

    def difficulty_state(
        self,
        *,
        scope_kind: str,
        scope_key: str,
    ) -> DifficultyStateSummary | None:
        conn = open_db(self._path)
        try:
            return _load_difficulty_state(
                conn=conn,
                scope_kind=str(scope_kind),
                scope_key=str(scope_key),
            )
        finally:
            conn.close()

    def reset_difficulty_state(
        self,
        *,
        test_code: str | None = None,
        scope_key: str | None = None,
        clear_all: bool = False,
    ) -> None:
        conn = open_db(self._path)
        try:
            with conn:
                if clear_all:
                    conn.execute("DELETE FROM difficulty_state")
                    return
                if test_code:
                    code_scope, primitive_scope = scope_keys_for_code(test_code)
                    family_scope = family_id_for_code(test_code)
                    conn.execute(
                        "DELETE FROM difficulty_state "
                        "WHERE (scope_kind=? AND scope_key=?) "
                        "OR (scope_kind=? AND scope_key=?) "
                        "OR (scope_kind=? AND scope_key=?)",
                        ("code", code_scope, "family", family_scope, "primitive", primitive_scope),
                    )
                if scope_key:
                    conn.execute(
                        "DELETE FROM difficulty_state WHERE scope_key=?",
                        (str(scope_key),),
                    )
        finally:
            conn.close()


class _AttemptResultEngineShim:
    def __init__(self, result: AttemptResult) -> None:
        self._result = result
        self.seed = result.seed
        self.difficulty = result.difficulty
        self.practice_questions = result.practice_questions
        self.scored_duration_s = result.scored_duration_s


def record_attempt(
    *,
    db_path: Path,
    result: AttemptResult,
    app_version: str,
    session_id: int | None = None,
    input_profile_id: str | None = None,
    activity_code: str | None = None,
    activity_kind: str = "legacy_attempt",
    test_version: int | None = None,
    parent_activity_session_id: int | None = None,
    origin_activity_code: str | None = None,
    origin_activity_kind: str | None = None,
    origin_item_index: int | None = None,
) -> SavedAttempt:
    store = ResultsStore(db_path)
    if session_id is not None:
        store._session_id = int(session_id)
    return store.record_attempt(
        result=result,
        app_version=app_version,
        input_profile_id=input_profile_id,
        activity_code=activity_code,
        activity_kind=activity_kind,
        test_version=test_version,
        parent_activity_session_id=parent_activity_session_id,
        origin_activity_code=origin_activity_code,
        origin_activity_kind=origin_activity_kind,
        origin_item_index=origin_item_index,
    )


def record_math_reasoning_attempt(*, db_path: Path, result: AttemptResult, app_version: str) -> int:
    return record_attempt(
        db_path=db_path,
        result=result,
        app_version=app_version,
    ).attempt_id


def load_session_summary(*, db_path: Path, session_id: int) -> SessionSummary | None:
    conn = open_db(db_path)
    try:
        _refresh_session_materializations(conn=conn, session_id=session_id)
        return _fetch_session_summary(conn=conn, session_id=session_id)
    finally:
        conn.close()


def load_test_session_summary(
    *,
    db_path: Path,
    session_id: int,
    test_code: str,
) -> TestSessionSummary | None:
    conn = open_db(db_path)
    try:
        return _fetch_test_session_summary(
            conn=conn,
            session_id=session_id,
            test_code=test_code,
        )
    finally:
        conn.close()


def load_recent_attempt_history(
    *,
    db_path: Path,
    since_days: int | None = 28,
    limit: int | None = None,
    child_item_preferred: bool = False,
) -> list[AttemptHistoryEntry]:
    conn = open_db(db_path)
    try:
        return _load_recent_attempt_history(
            conn=conn,
            since_days=since_days,
            limit=limit,
            child_item_preferred=child_item_preferred,
        )
    finally:
        conn.close()


def _load_recent_attempt_history(
    *,
    conn: sqlite3.Connection,
    since_days: int | None,
    limit: int | None,
    child_item_preferred: bool,
) -> list[AttemptHistoryEntry]:
    query = """
        SELECT
            attempt.id,
            attempt.session_id,
            attempt.activity_session_id,
            activity_session.activity_code,
            activity_session.activity_kind,
            attempt.test_code,
            attempt.test_version,
            attempt.rng_seed,
            attempt.difficulty,
            attempt.started_at_utc,
            attempt.completed_at_utc,
            attempt.difficulty_level_start,
            attempt.difficulty_level_end,
            activity_session.parent_activity_session_id,
            activity_session.origin_activity_code,
            activity_session.origin_activity_kind,
            activity_session.origin_item_index
        FROM attempt
        LEFT JOIN activity_session ON activity_session.id = attempt.activity_session_id
    """
    params: list[object] = []
    if since_days is not None:
        cutoff_utc = _utc_iso_from_epoch(_utc_now_epoch() - (max(0, int(since_days)) * 86400.0))
        query += " WHERE attempt.completed_at_utc >= ?"
        params.append(cutoff_utc)
    query += " ORDER BY attempt.completed_at_utc DESC, attempt.id DESC"
    if limit is not None:
        query += " LIMIT ?"
        params.append(max(0, int(limit)))

    rows = conn.execute(query, tuple(params)).fetchall()
    if not rows:
        return []

    attempt_ids = [int(row[0]) for row in rows]
    metrics: dict[int, dict[str, str]] = {attempt_id: {} for attempt_id in attempt_ids}
    placeholders = ", ".join("?" for _ in attempt_ids)
    metric_rows = conn.execute(
        f"""
        SELECT attempt_id, key, value
        FROM attempt_metric
        WHERE attempt_id IN ({placeholders})
        ORDER BY attempt_id, key
        """,
        tuple(attempt_ids),
    ).fetchall()
    for attempt_id, key, value in metric_rows:
        metrics[int(attempt_id)][str(key)] = str(value)

    out: list[AttemptHistoryEntry] = []
    for row in rows:
        out.append(
            AttemptHistoryEntry(
                attempt_id=int(row[0]),
                session_id=int(row[1]),
                activity_session_id=None if row[2] is None else int(row[2]),
                activity_code=None if row[3] is None else str(row[3]),
                activity_kind=None if row[4] is None else str(row[4]),
                test_code=str(row[5]),
                test_version=int(row[6]),
                rng_seed=int(row[7]),
                difficulty=float(row[8]),
                started_at_utc=str(row[9]),
                completed_at_utc=str(row[10]),
                difficulty_level_start=None if row[11] is None else int(row[11]),
                difficulty_level_end=None if row[12] is None else int(row[12]),
                metrics=dict(metrics.get(int(row[0]), {})),
                parent_activity_session_id=None if row[13] is None else int(row[13]),
                origin_activity_code=None if row[14] is None else str(row[14]),
                origin_activity_kind=None if row[15] is None else str(row[15]),
                origin_item_index=None if row[16] is None else int(row[16]),
            )
        )
    return _prefer_child_items(out) if child_item_preferred else out


def _load_difficulty_state(
    *,
    conn: sqlite3.Connection,
    scope_kind: str,
    scope_key: str,
) -> DifficultyStateSummary | None:
    row = conn.execute(
        """
        SELECT
            scope_kind,
            scope_key,
            recommended_level,
            last_start_level,
            last_end_level,
            ewma_accuracy,
            ewma_score_ratio,
            ewma_timeout_rate,
            ewma_mean_rt_ms,
            sample_count,
            updated_at_utc,
            mastery,
            speed,
            fatigue_penalty,
            post_error_penalty,
            instability_penalty,
            retention_need,
            confidence,
            leverage,
            level_confidence,
            last_successful_level,
            last_meltdown_level
        FROM difficulty_state
        WHERE scope_kind=? AND scope_key=?
        """,
        (str(scope_kind), str(scope_key)),
    ).fetchone()
    if row is None:
        return None
    return DifficultyStateSummary(
        scope_kind=str(row[0]),
        scope_key=str(row[1]),
        recommended_level=int(row[2]),
        last_start_level=None if row[3] is None else int(row[3]),
        last_end_level=None if row[4] is None else int(row[4]),
        ewma_accuracy=None if row[5] is None else float(row[5]),
        ewma_score_ratio=None if row[6] is None else float(row[6]),
        ewma_timeout_rate=None if row[7] is None else float(row[7]),
        ewma_mean_rt_ms=None if row[8] is None else float(row[8]),
        sample_count=int(row[9]),
        updated_at_utc=str(row[10]),
        mastery=None if row[11] is None else float(row[11]),
        speed=None if row[12] is None else float(row[12]),
        fatigue_penalty=None if row[13] is None else float(row[13]),
        post_error_penalty=None if row[14] is None else float(row[14]),
        instability_penalty=None if row[15] is None else float(row[15]),
        retention_need=None if row[16] is None else float(row[16]),
        confidence=None if row[17] is None else float(row[17]),
        leverage=None if row[18] is None else float(row[18]),
        level_confidence=None if row[19] is None else float(row[19]),
        last_successful_level=None if row[20] is None else int(row[20]),
        last_meltdown_level=None if row[21] is None else int(row[21]),
    )


def _prefer_child_items(entries: list[AttemptHistoryEntry]) -> list[AttemptHistoryEntry]:
    parent_ids_with_children = {
        int(entry.parent_activity_session_id)
        for entry in entries
        if entry.parent_activity_session_id is not None
    }
    if not parent_ids_with_children:
        return list(entries)
    return [
        entry
        for entry in entries
        if entry.parent_activity_session_id is not None
        or entry.activity_session_id is None
        or int(entry.activity_session_id) not in parent_ids_with_children
    ]


def _update_difficulty_states(
    *,
    conn: sqlite3.Connection,
    result: AttemptResult,
) -> None:
    updated_at_utc = _utc_now_iso()
    recent_history = _load_recent_attempt_history(
        conn=conn,
        since_days=None,
        limit=None,
        child_item_preferred=True,
    )
    ranking = rank_primitives(recent_history, now_utc=updated_at_utc)
    ranked_states = {state.primitive_id: state for state in ranking.primitive_states}
    for item in _difficulty_state_inputs_from_result(result):
        code_scope, primitive_scope = scope_keys_for_code(item["test_code"])
        family_scope = family_id_for_code(item["test_code"])
        for scope_kind, scope_key in (
            ("code", code_scope),
            ("family", family_scope),
            ("primitive", primitive_scope),
        ):
            if scope_kind == "primitive" and scope_key in ranked_states:
                ranked = ranked_states[str(scope_key)]
                merged = AdaptiveDifficultyState(
                    scope_kind="primitive",
                    scope_key=str(scope_key),
                    recommended_level=int(ranked.recommended_level),
                    last_start_level=ranked.last_start_level,
                    last_end_level=ranked.last_end_level,
                    ewma_accuracy=ranked.ewma_accuracy,
                    ewma_score_ratio=ranked.ewma_score_ratio,
                    ewma_timeout_rate=ranked.ewma_timeout_rate,
                    ewma_mean_rt_ms=ranked.ewma_mean_rt_ms,
                    sample_count=int(ranked.evidence_count),
                    updated_at_utc=str(updated_at_utc),
                    mastery=ranked.mastery,
                    speed=ranked.speed,
                    fatigue_penalty=ranked.fatigue_penalty,
                    post_error_penalty=ranked.post_error_penalty,
                    instability_penalty=ranked.instability_penalty,
                    retention_need=ranked.retention_need,
                    confidence=ranked.confidence,
                    leverage=ranked.leverage,
                    level_confidence=ranked.level_confidence,
                    last_successful_level=ranked.last_successful_level,
                    last_meltdown_level=ranked.last_meltdown_level,
                )
            else:
                previous = _load_difficulty_state(
                    conn=conn,
                    scope_kind=str(scope_kind),
                    scope_key=str(scope_key),
                )
                merged = merge_adaptive_state(
                    previous=None
                    if previous is None
                    else AdaptiveDifficultyState(
                        scope_kind=previous.scope_kind,  # type: ignore[arg-type]
                        scope_key=previous.scope_key,
                        recommended_level=previous.recommended_level,
                        last_start_level=previous.last_start_level,
                        last_end_level=previous.last_end_level,
                        ewma_accuracy=previous.ewma_accuracy,
                        ewma_score_ratio=previous.ewma_score_ratio,
                        ewma_timeout_rate=previous.ewma_timeout_rate,
                        ewma_mean_rt_ms=previous.ewma_mean_rt_ms,
                        sample_count=previous.sample_count,
                        updated_at_utc=previous.updated_at_utc,
                        mastery=previous.mastery,
                        speed=previous.speed,
                        fatigue_penalty=previous.fatigue_penalty,
                        post_error_penalty=previous.post_error_penalty,
                        instability_penalty=previous.instability_penalty,
                        retention_need=previous.retention_need,
                        confidence=previous.confidence,
                        leverage=previous.leverage,
                        level_confidence=previous.level_confidence,
                        last_successful_level=previous.last_successful_level,
                        last_meltdown_level=previous.last_meltdown_level,
                    ),
                    scope_kind=scope_kind,  # type: ignore[arg-type]
                    scope_key=scope_key,
                    start_level=item["start_level"],
                    end_level=item["end_level"],
                    accuracy=item["accuracy"],
                    score_ratio=item["score_ratio"],
                    timeout_rate=item["timeout_rate"],
                    mean_rt_ms=item["mean_rt_ms"],
                    updated_at_utc=updated_at_utc,
                    training_mode=item.get("training_mode"),
                )
            with conn:
                conn.execute(
                    """
                    INSERT INTO difficulty_state(
                        scope_kind,
                        scope_key,
                        recommended_level,
                        last_start_level,
                        last_end_level,
                        ewma_accuracy,
                        ewma_score_ratio,
                        ewma_timeout_rate,
                        ewma_mean_rt_ms,
                        sample_count,
                        updated_at_utc,
                        mastery,
                        speed,
                        fatigue_penalty,
                        post_error_penalty,
                        instability_penalty,
                        retention_need,
                        confidence,
                        leverage,
                        level_confidence,
                        last_successful_level,
                        last_meltdown_level
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(scope_kind, scope_key) DO UPDATE SET
                        recommended_level=excluded.recommended_level,
                        last_start_level=excluded.last_start_level,
                        last_end_level=excluded.last_end_level,
                        ewma_accuracy=excluded.ewma_accuracy,
                        ewma_score_ratio=excluded.ewma_score_ratio,
                        ewma_timeout_rate=excluded.ewma_timeout_rate,
                        ewma_mean_rt_ms=excluded.ewma_mean_rt_ms,
                        sample_count=excluded.sample_count,
                        updated_at_utc=excluded.updated_at_utc,
                        mastery=excluded.mastery,
                        speed=excluded.speed,
                        fatigue_penalty=excluded.fatigue_penalty,
                        post_error_penalty=excluded.post_error_penalty,
                        instability_penalty=excluded.instability_penalty,
                        retention_need=excluded.retention_need,
                        confidence=excluded.confidence,
                        leverage=excluded.leverage,
                        level_confidence=excluded.level_confidence,
                        last_successful_level=excluded.last_successful_level,
                        last_meltdown_level=excluded.last_meltdown_level
                    """,
                    (
                        str(merged.scope_kind),
                        str(merged.scope_key),
                        int(merged.recommended_level),
                        None if merged.last_start_level is None else int(merged.last_start_level),
                        None if merged.last_end_level is None else int(merged.last_end_level),
                        None if merged.ewma_accuracy is None else float(merged.ewma_accuracy),
                        None if merged.ewma_score_ratio is None else float(merged.ewma_score_ratio),
                        None if merged.ewma_timeout_rate is None else float(merged.ewma_timeout_rate),
                        None if merged.ewma_mean_rt_ms is None else float(merged.ewma_mean_rt_ms),
                        int(merged.sample_count),
                        str(merged.updated_at_utc),
                        None if merged.mastery is None else float(merged.mastery),
                        None if merged.speed is None else float(merged.speed),
                        None if merged.fatigue_penalty is None else float(merged.fatigue_penalty),
                        None if merged.post_error_penalty is None else float(merged.post_error_penalty),
                        None if merged.instability_penalty is None else float(merged.instability_penalty),
                        None if merged.retention_need is None else float(merged.retention_need),
                        None if merged.confidence is None else float(merged.confidence),
                        None if merged.leverage is None else float(merged.leverage),
                        None if merged.level_confidence is None else float(merged.level_confidence),
                        None if merged.last_successful_level is None else int(merged.last_successful_level),
                        None if merged.last_meltdown_level is None else int(merged.last_meltdown_level),
                    ),
                )


def _difficulty_state_inputs_from_result(result: AttemptResult) -> list[dict[str, object]]:
    def _base_item(
        *,
        test_code: str,
        metrics: dict[str, str],
        start_level: int | None,
        end_level: int | None,
        default_accuracy: float | None = None,
        default_score_ratio: float | None = None,
    ) -> dict[str, object]:
        accuracy = metric_float(metrics, "accuracy")
        score_ratio = metric_float(metrics, "score_ratio")
        return {
            "test_code": str(test_code),
            "start_level": start_level,
            "end_level": end_level,
            "accuracy": default_accuracy if accuracy is None else accuracy,
            "score_ratio": default_score_ratio if score_ratio is None else score_ratio,
            "timeout_rate": metric_float(metrics, "timeout_rate"),
            "mean_rt_ms": metric_float(metrics, "mean_rt_ms"),
            "training_mode": str(metrics.get("training_mode", "")).strip().lower() or None,
        }

    items = [
        _base_item(
            test_code=result.test_code,
            metrics=result.metrics,
            start_level=result.difficulty_level_start,
            end_level=result.difficulty_level_end,
            default_accuracy=float(result.accuracy),
            default_score_ratio=result.score_ratio,
        )
    ]

    probe_groups: dict[str, dict[str, str]] = {}
    block_groups: dict[str, dict[str, str]] = {}
    for key, value in result.metrics.items():
        token = str(key)
        if token.startswith("probe."):
            rest = token[len("probe.") :]
            probe_code, sep, subkey = rest.partition(".")
            if sep and probe_code:
                probe_groups.setdefault(probe_code, {})[subkey] = str(value)
        elif token.startswith("block."):
            rest = token[len("block.") :]
            block_key, sep, subkey = rest.partition(".")
            if sep and block_key:
                block_groups.setdefault(block_key, {})[subkey] = str(value)

    for probe_code, metrics in probe_groups.items():
        if metrics.get("completed", "") not in {"1", "true", "True"}:
            continue
        difficulty_level = metrics.get("difficulty_level")
        start_level = metric_float(metrics, "difficulty_level_start")
        end_level = metric_float(metrics, "difficulty_level_end")
        if start_level is None and difficulty_level not in (None, ""):
            start_level = metric_float({"v": str(difficulty_level)}, "v")
        if end_level is None and difficulty_level not in (None, ""):
            end_level = metric_float({"v": str(difficulty_level)}, "v")
        items.append(
            _base_item(
                test_code=probe_code,
                metrics=metrics,
                start_level=None if start_level is None else int(start_level),
                end_level=None if end_level is None else int(end_level),
            )
        )

    for _block_key, metrics in block_groups.items():
        if metrics.get("completed", "") not in {"1", "true", "True"}:
            continue
        drill_code = str(metrics.get("drill_code", "")).strip()
        if drill_code == "":
            continue
        difficulty_level = metrics.get("difficulty_level")
        start_level = metric_float(metrics, "difficulty_level_start")
        end_level = metric_float(metrics, "difficulty_level_end")
        if start_level is None and difficulty_level not in (None, ""):
            start_level = metric_float({"v": str(difficulty_level)}, "v")
        if end_level is None and difficulty_level not in (None, ""):
            end_level = metric_float({"v": str(difficulty_level)}, "v")
        items.append(
            _base_item(
                test_code=drill_code,
                metrics=metrics,
                start_level=None if start_level is None else int(start_level),
                end_level=None if end_level is None else int(end_level),
            )
        )
    return items


def _migrate(conn: sqlite3.Connection) -> None:
    row = conn.execute("PRAGMA user_version;").fetchone()
    ver = int(row[0]) if row else 0
    if ver >= SCHEMA_VERSION:
        return

    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS session (
                id INTEGER PRIMARY KEY,
                created_at_utc TEXT NOT NULL,
                started_at_utc TEXT,
                ended_at_utc TEXT,
                exit_reason TEXT,
                app_version TEXT
            );
            """
        )
        _add_column_if_missing(conn, "session", "started_at_utc TEXT")
        _add_column_if_missing(conn, "session", "ended_at_utc TEXT")
        _add_column_if_missing(conn, "session", "exit_reason TEXT")
        _add_column_if_missing(conn, "session", "app_version TEXT")
        conn.execute(
            "UPDATE session SET started_at_utc = COALESCE(started_at_utc, created_at_utc) "
            "WHERE started_at_utc IS NULL"
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS attempt (
                id INTEGER PRIMARY KEY,
                session_id INTEGER NOT NULL REFERENCES session(id) ON DELETE CASCADE,
                test_code TEXT NOT NULL,
                test_version INTEGER NOT NULL,
                app_version TEXT NOT NULL,
                rng_seed INTEGER NOT NULL,
                difficulty REAL NOT NULL,
                input_profile_id TEXT,
                practice_questions INTEGER NOT NULL,
                scored_duration_s REAL NOT NULL,
                started_at_utc TEXT NOT NULL,
                completed_at_utc TEXT NOT NULL,
                activity_session_id INTEGER REFERENCES activity_session(id) ON DELETE SET NULL,
                difficulty_level_start INTEGER,
                difficulty_level_end INTEGER
            );
            """
        )
        _add_column_if_missing(
            conn,
            "attempt",
            "activity_session_id INTEGER REFERENCES activity_session(id) ON DELETE SET NULL",
        )
        _add_column_if_missing(conn, "attempt", "difficulty_level_start INTEGER")
        _add_column_if_missing(conn, "attempt", "difficulty_level_end INTEGER")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS metric (
                attempt_id INTEGER NOT NULL REFERENCES attempt(id) ON DELETE CASCADE,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                PRIMARY KEY (attempt_id, key)
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cognitive_event (
                id INTEGER PRIMARY KEY,
                attempt_id INTEGER NOT NULL REFERENCES attempt(id) ON DELETE CASCADE,
                seq INTEGER NOT NULL,
                phase TEXT NOT NULL,
                prompt TEXT NOT NULL,
                expected TEXT NOT NULL,
                response TEXT NOT NULL,
                is_correct INTEGER NOT NULL,
                presented_at_ms INTEGER NOT NULL,
                answered_at_ms INTEGER NOT NULL,
                rt_ms INTEGER NOT NULL
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_cognitive_event_attempt_seq "
            "ON cognitive_event(attempt_id, seq);"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS activity_session (
                id INTEGER PRIMARY KEY,
                session_id INTEGER NOT NULL REFERENCES session(id) ON DELETE CASCADE,
                parent_activity_session_id INTEGER REFERENCES activity_session(id) ON DELETE SET NULL,
                activity_code TEXT NOT NULL,
                activity_kind TEXT NOT NULL,
                origin_activity_code TEXT,
                origin_activity_kind TEXT,
                origin_item_index INTEGER,
                app_version TEXT NOT NULL,
                test_version INTEGER NOT NULL,
                input_profile_id TEXT,
                rng_seed INTEGER,
                difficulty REAL,
                practice_questions INTEGER,
                scored_duration_s REAL,
                started_at_utc TEXT NOT NULL,
                ended_at_utc TEXT,
                completion_reason TEXT
            );
            """
        )
        _add_column_if_missing(
            conn,
            "activity_session",
            "parent_activity_session_id INTEGER REFERENCES activity_session(id) ON DELETE SET NULL",
        )
        _add_column_if_missing(conn, "activity_session", "origin_activity_code TEXT")
        _add_column_if_missing(conn, "activity_session", "origin_activity_kind TEXT")
        _add_column_if_missing(conn, "activity_session", "origin_item_index INTEGER")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_activity_session_session "
            "ON activity_session(session_id, id);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_activity_session_parent "
            "ON activity_session(parent_activity_session_id, origin_item_index);"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS attempt_metric (
                attempt_id INTEGER NOT NULL REFERENCES attempt(id) ON DELETE CASCADE,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                PRIMARY KEY (attempt_id, key)
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS activity_metric (
                activity_session_id INTEGER NOT NULL REFERENCES activity_session(id) ON DELETE CASCADE,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                PRIMARY KEY (activity_session_id, key)
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS telemetry_event (
                id INTEGER PRIMARY KEY,
                session_id INTEGER NOT NULL REFERENCES session(id) ON DELETE CASCADE,
                activity_session_id INTEGER REFERENCES activity_session(id) ON DELETE CASCADE,
                attempt_id INTEGER REFERENCES attempt(id) ON DELETE CASCADE,
                seq INTEGER NOT NULL,
                family TEXT NOT NULL,
                kind TEXT NOT NULL,
                phase TEXT NOT NULL,
                item_index INTEGER,
                is_scored INTEGER NOT NULL,
                is_correct INTEGER,
                is_timeout INTEGER NOT NULL,
                response_time_ms INTEGER,
                score REAL,
                max_score REAL,
                difficulty_level INTEGER,
                occurred_at_ms INTEGER,
                prompt TEXT,
                expected TEXT,
                response TEXT,
                extra_json TEXT
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_telemetry_event_activity_seq "
            "ON telemetry_event(activity_session_id, seq);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_telemetry_event_attempt_seq "
            "ON telemetry_event(attempt_id, seq);"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS session_summary (
                session_id INTEGER PRIMARY KEY REFERENCES session(id) ON DELETE CASCADE,
                started_at_utc TEXT NOT NULL,
                completed_at_utc TEXT NOT NULL,
                exit_reason TEXT,
                activity_count INTEGER NOT NULL,
                completed_activity_count INTEGER NOT NULL,
                aborted_activity_count INTEGER NOT NULL,
                attempt_count INTEGER NOT NULL,
                unique_tests INTEGER NOT NULL,
                mean_accuracy REAL,
                mean_score_ratio REAL
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS session_metric (
                session_id INTEGER NOT NULL REFERENCES session(id) ON DELETE CASCADE,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                PRIMARY KEY (session_id, key)
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS difficulty_state (
                scope_kind TEXT NOT NULL,
                scope_key TEXT NOT NULL,
                recommended_level INTEGER NOT NULL,
                last_start_level INTEGER,
                last_end_level INTEGER,
                ewma_accuracy REAL,
                ewma_score_ratio REAL,
                ewma_timeout_rate REAL,
                ewma_mean_rt_ms REAL,
                sample_count INTEGER NOT NULL,
                updated_at_utc TEXT NOT NULL,
                mastery REAL,
                speed REAL,
                fatigue_penalty REAL,
                post_error_penalty REAL,
                instability_penalty REAL,
                retention_need REAL,
                confidence REAL,
                leverage REAL,
                level_confidence REAL,
                last_successful_level INTEGER,
                last_meltdown_level INTEGER,
                PRIMARY KEY (scope_kind, scope_key)
            );
            """
        )
        _add_column_if_missing(conn, "difficulty_state", "mastery REAL")
        _add_column_if_missing(conn, "difficulty_state", "speed REAL")
        _add_column_if_missing(conn, "difficulty_state", "fatigue_penalty REAL")
        _add_column_if_missing(conn, "difficulty_state", "post_error_penalty REAL")
        _add_column_if_missing(conn, "difficulty_state", "instability_penalty REAL")
        _add_column_if_missing(conn, "difficulty_state", "retention_need REAL")
        _add_column_if_missing(conn, "difficulty_state", "confidence REAL")
        _add_column_if_missing(conn, "difficulty_state", "leverage REAL")
        _add_column_if_missing(conn, "difficulty_state", "level_confidence REAL")
        _add_column_if_missing(conn, "difficulty_state", "last_successful_level INTEGER")
        _add_column_if_missing(conn, "difficulty_state", "last_meltdown_level INTEGER")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_difficulty_state_scope "
            "ON difficulty_state(scope_key, scope_kind);"
        )

        conn.execute(
            """
            INSERT OR IGNORE INTO attempt_metric(attempt_id, key, value)
            SELECT attempt_id, key, value
            FROM metric
            """
        )

        conn.execute(
            """
            UPDATE attempt
            SET difficulty_level_start = COALESCE(
                difficulty_level_start,
                MAX(1, MIN(10, CAST(ROUND((difficulty * 9.0) + 1.0) AS INTEGER)))
            ),
            difficulty_level_end = COALESCE(
                difficulty_level_end,
                MAX(1, MIN(10, CAST(ROUND((difficulty * 9.0) + 1.0) AS INTEGER)))
            )
            WHERE difficulty_level_start IS NULL OR difficulty_level_end IS NULL
            """
        )

        rows = conn.execute(
            """
            SELECT
                id,
                session_id,
                test_code,
                test_version,
                app_version,
                input_profile_id,
                rng_seed,
                difficulty,
                practice_questions,
                scored_duration_s,
                started_at_utc,
                completed_at_utc
            FROM attempt
            WHERE activity_session_id IS NULL
            ORDER BY id
            """
        ).fetchall()
        for row in rows:
            cur = conn.execute(
                """
                INSERT INTO activity_session(
                    session_id, parent_activity_session_id, activity_code, activity_kind,
                    origin_activity_code, origin_activity_kind, origin_item_index,
                    app_version, test_version,
                    input_profile_id, rng_seed, difficulty, practice_questions, scored_duration_s,
                    started_at_utc, ended_at_utc, completion_reason
                )
                VALUES (?, NULL, ?, ?, NULL, NULL, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(row[1]),
                    str(row[2]),
                    "legacy_attempt",
                    str(row[4]),
                    int(row[3]),
                    row[5],
                    int(row[6]),
                    float(row[7]),
                    int(row[8]),
                    float(row[9]),
                    str(row[10]),
                    str(row[11]),
                    "completed",
                ),
            )
            conn.execute(
                "UPDATE attempt SET activity_session_id=? WHERE id=?",
                (int(cur.lastrowid), int(row[0])),
            )

        session_rows = conn.execute("SELECT id FROM session").fetchall()
        for session_row in session_rows:
            _refresh_session_materializations(conn=conn, session_id=int(session_row[0]))

        conn.execute(f"PRAGMA user_version={SCHEMA_VERSION};")


def _column_names(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {str(row[1]) for row in rows}


def _add_column_if_missing(conn: sqlite3.Connection, table: str, column_def: str) -> None:
    name = column_def.split()[0]
    if name in _column_names(conn, table):
        return
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column_def}")


def _ensure_session(
    *,
    conn: sqlite3.Connection,
    session_id: int | None,
    app_version: str,
) -> int:
    if session_id is not None:
        row = conn.execute("SELECT id FROM session WHERE id=?", (int(session_id),)).fetchone()
        if row is not None:
            return int(row[0])
    now_utc = _utc_now_iso()
    with conn:
        cur = conn.execute(
            """
            INSERT INTO session(created_at_utc, started_at_utc, ended_at_utc, exit_reason, app_version)
            VALUES (?, ?, NULL, NULL, ?)
            """,
            (now_utc, now_utc, str(app_version)),
        )
    return int(cur.lastrowid)


def _insert_activity_session(
    *,
    conn: sqlite3.Connection,
    session_id: int,
    activity_code: str,
    activity_kind: str,
    app_version: str,
    test_version: int,
    engine: object,
    input_profile_id: str | None,
    parent_activity_session_id: int | None,
    origin_activity_code: str | None,
    origin_activity_kind: str | None,
    origin_item_index: int | None,
) -> int:
    now_utc = _utc_now_iso()
    seed = getattr(engine, "seed", getattr(engine, "_seed", None))
    difficulty = getattr(engine, "difficulty", getattr(engine, "_difficulty", None))
    practice_questions = getattr(
        engine,
        "practice_questions",
        getattr(engine, "_practice_questions", None),
    )
    scored_duration_s = getattr(
        engine,
        "scored_duration_s",
        getattr(engine, "_scored_duration_s", None),
    )

    with conn:
        cur = conn.execute(
            """
            INSERT INTO activity_session(
                session_id, parent_activity_session_id, activity_code, activity_kind,
                origin_activity_code, origin_activity_kind, origin_item_index,
                app_version, test_version,
                input_profile_id, rng_seed, difficulty, practice_questions, scored_duration_s,
                started_at_utc, ended_at_utc, completion_reason
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)
            """,
            (
                int(session_id),
                None if parent_activity_session_id is None else int(parent_activity_session_id),
                str(activity_code),
                str(activity_kind),
                None if origin_activity_code is None else str(origin_activity_code),
                None if origin_activity_kind is None else str(origin_activity_kind),
                None if origin_item_index is None else int(origin_item_index),
                str(app_version),
                int(test_version),
                input_profile_id,
                None if seed is None else int(seed),
                None if difficulty is None else float(difficulty),
                None if practice_questions is None else int(practice_questions),
                None if scored_duration_s is None else float(scored_duration_s),
                now_utc,
            ),
        )
        activity_session_id = int(cur.lastrowid)
        _write_telemetry_events(
            conn=conn,
            session_id=session_id,
            activity_session_id=activity_session_id,
            attempt_id=None,
            events=[
                lifecycle_event(
                    family="lifecycle",
                    kind="activity_started",
                    seq=0,
                    occurred_at_ms=0,
                    extra={
                        "activity_code": str(activity_code),
                        "activity_kind": str(activity_kind),
                    },
                )
            ],
        )
    return activity_session_id


def _insert_attempt(
    *,
    conn: sqlite3.Connection,
    session_id: int,
    activity_session_id: int,
    result: AttemptResult,
    app_version: str,
    input_profile_id: str | None,
    started_at_utc: str,
) -> int:
    completed_at_utc = _utc_now_iso()

    with conn:
        cur = conn.execute(
            """
            INSERT INTO attempt(
                session_id, activity_session_id, test_code, test_version, app_version,
                rng_seed, difficulty, input_profile_id, practice_questions, scored_duration_s,
                started_at_utc, completed_at_utc, difficulty_level_start, difficulty_level_end
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(session_id),
                int(activity_session_id),
                str(result.test_code),
                int(result.test_version),
                str(app_version),
                int(result.seed),
                float(result.difficulty),
                input_profile_id,
                int(result.practice_questions),
                float(result.scored_duration_s),
                str(started_at_utc),
                completed_at_utc,
                None if result.difficulty_level_start is None else int(result.difficulty_level_start),
                None if result.difficulty_level_end is None else int(result.difficulty_level_end),
            ),
        )
        attempt_id = int(cur.lastrowid)
        for key, value in result.metrics.items():
            conn.execute(
                "INSERT INTO attempt_metric(attempt_id, key, value) VALUES (?, ?, ?)",
                (attempt_id, str(key), str(value)),
            )
    return attempt_id


def _write_activity_metrics(
    *,
    conn: sqlite3.Connection,
    activity_session_id: int,
    result: AttemptResult,
) -> None:
    with conn:
        conn.execute(
            "DELETE FROM activity_metric WHERE activity_session_id=?",
            (int(activity_session_id),),
        )
        base_metrics = {
            "attempted": str(int(result.attempted)),
            "correct": str(int(result.correct)),
            "accuracy": f"{float(result.accuracy):.6f}",
            "throughput_per_min": f"{float(result.throughput_per_min):.6f}",
            "difficulty_level_start": ""
            if result.difficulty_level_start is None
            else str(int(result.difficulty_level_start)),
            "difficulty_level_end": ""
            if result.difficulty_level_end is None
            else str(int(result.difficulty_level_end)),
        }
        merged = {**result.metrics, **base_metrics}
        for key, value in merged.items():
            conn.execute(
                "INSERT INTO activity_metric(activity_session_id, key, value) VALUES (?, ?, ?)",
                (int(activity_session_id), str(key), str(value)),
            )


def _write_telemetry_events(
    *,
    conn: sqlite3.Connection,
    session_id: int,
    activity_session_id: int | None,
    attempt_id: int | None,
    events: list[TelemetryEvent],
) -> None:
    if not events:
        return
    if activity_session_id is not None:
        seq_offset = _next_activity_event_seq(conn=conn, activity_session_id=activity_session_id)
    else:
        seq_offset = _next_session_event_seq(conn=conn, session_id=session_id)

    with conn:
        for offset, event in enumerate(events):
            extra_json = None
            if event.extra:
                extra_json = json.dumps(event.extra, sort_keys=True)
            conn.execute(
                """
                INSERT INTO telemetry_event(
                    session_id, activity_session_id, attempt_id, seq, family, kind, phase,
                    item_index, is_scored, is_correct, is_timeout, response_time_ms,
                    score, max_score, difficulty_level, occurred_at_ms,
                    prompt, expected, response, extra_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(session_id),
                    None if activity_session_id is None else int(activity_session_id),
                    None if attempt_id is None else int(attempt_id),
                    int(seq_offset + offset),
                    str(event.family),
                    str(event.kind),
                    str(event.phase),
                    None if event.item_index is None else int(event.item_index),
                    1 if event.is_scored else 0,
                    None if event.is_correct is None else (1 if event.is_correct else 0),
                    1 if event.is_timeout else 0,
                    None if event.response_time_ms is None else int(event.response_time_ms),
                    None if event.score is None else float(event.score),
                    None if event.max_score is None else float(event.max_score),
                    None if event.difficulty_level is None else int(event.difficulty_level),
                    None if event.occurred_at_ms is None else int(event.occurred_at_ms),
                    event.prompt,
                    event.expected,
                    event.response,
                    extra_json,
                ),
            )


def _next_activity_event_seq(*, conn: sqlite3.Connection, activity_session_id: int) -> int:
    row = conn.execute(
        "SELECT COALESCE(MAX(seq), -1) FROM telemetry_event WHERE activity_session_id=?",
        (int(activity_session_id),),
    ).fetchone()
    return 0 if row is None else int(row[0]) + 1


def _next_session_event_seq(*, conn: sqlite3.Connection, session_id: int) -> int:
    row = conn.execute(
        "SELECT COALESCE(MAX(seq), -1) FROM telemetry_event "
        "WHERE session_id=? AND activity_session_id IS NULL",
        (int(session_id),),
    ).fetchone()
    return 0 if row is None else int(row[0]) + 1


def _fetch_activity_session_row(
    *,
    conn: sqlite3.Connection,
    activity_session_id: int,
) -> sqlite3.Row | None:
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM activity_session WHERE id=?",
        (int(activity_session_id),),
    ).fetchone()
    conn.row_factory = None
    return row


def _existing_attempt_id_for_activity(
    *,
    conn: sqlite3.Connection,
    activity_session_id: int,
) -> int | None:
    row = conn.execute(
        "SELECT id FROM attempt WHERE activity_session_id=? ORDER BY id DESC LIMIT 1",
        (int(activity_session_id),),
    ).fetchone()
    return None if row is None else int(row[0])


def _update_activity_session_runtime_metadata(
    *,
    conn: sqlite3.Connection,
    activity_session_id: int,
    app_version: str,
    input_profile_id: str | None,
    result: AttemptResult | None,
) -> None:
    if result is None:
        return
    with conn:
        conn.execute(
            """
            UPDATE activity_session
            SET app_version=?,
                input_profile_id=COALESCE(?, input_profile_id),
                rng_seed=?,
                difficulty=?,
                practice_questions=?,
                scored_duration_s=?,
                test_version=?
            WHERE id=?
            """,
            (
                str(app_version),
                input_profile_id,
                int(result.seed),
                float(result.difficulty),
                int(result.practice_questions),
                float(result.scored_duration_s),
                int(result.test_version),
                int(activity_session_id),
            ),
        )


def _finalize_activity_session(
    *,
    conn: sqlite3.Connection,
    session_id: int,
    activity_session_id: int,
    completion_reason: str,
) -> None:
    now_utc = _utc_now_iso()
    with conn:
        conn.execute(
            """
            UPDATE activity_session
            SET ended_at_utc=COALESCE(ended_at_utc, ?),
                completion_reason=COALESCE(completion_reason, ?)
            WHERE id=?
            """,
            (now_utc, str(completion_reason), int(activity_session_id)),
        )
        _write_telemetry_events(
            conn=conn,
            session_id=session_id,
            activity_session_id=activity_session_id,
            attempt_id=None,
            events=[
                lifecycle_event(
                    family="lifecycle",
                    kind="activity_completed"
                    if completion_reason == "completed"
                    else "activity_aborted",
                    seq=0,
                    occurred_at_ms=None,
                    extra={"completion_reason": str(completion_reason)},
                )
            ],
        )


def _finalize_session(*, conn: sqlite3.Connection, session_id: int, exit_reason: str) -> None:
    row = conn.execute(
        "SELECT ended_at_utc FROM session WHERE id=?",
        (int(session_id),),
    ).fetchone()
    if row is None or row[0] is not None:
        return
    activity_rows = conn.execute(
        "SELECT id FROM activity_session WHERE session_id=? AND ended_at_utc IS NULL",
        (int(session_id),),
    ).fetchall()
    for activity_row in activity_rows:
        _finalize_activity_session(
            conn=conn,
            session_id=session_id,
            activity_session_id=int(activity_row[0]),
            completion_reason=str(exit_reason),
        )
    now_utc = _utc_now_iso()
    with conn:
        conn.execute(
            "UPDATE session SET ended_at_utc=?, exit_reason=? WHERE id=?",
            (now_utc, str(exit_reason), int(session_id)),
        )
        _write_telemetry_events(
            conn=conn,
            session_id=session_id,
            activity_session_id=None,
            attempt_id=None,
            events=[
                lifecycle_event(
                    family="lifecycle",
                    kind="app_quit",
                    seq=0,
                    occurred_at_ms=None,
                    extra={"exit_reason": str(exit_reason)},
                )
            ],
        )
        _refresh_session_materializations(conn=conn, session_id=session_id)


def _refresh_session_materializations(*, conn: sqlite3.Connection, session_id: int) -> None:
    session_row = conn.execute(
        """
        SELECT
            started_at_utc,
            COALESCE(ended_at_utc, started_at_utc),
            exit_reason
        FROM session
        WHERE id=?
        """,
        (int(session_id),),
    ).fetchone()
    if session_row is None:
        return

    counts_row = conn.execute(
        """
        SELECT
            COUNT(*) AS activity_count,
            SUM(CASE WHEN completion_reason = 'completed' THEN 1 ELSE 0 END) AS completed_count,
            SUM(CASE WHEN completion_reason IS NOT NULL AND completion_reason <> 'completed' THEN 1 ELSE 0 END) AS aborted_count
        FROM activity_session
        WHERE session_id=?
        """,
        (int(session_id),),
    ).fetchone()
    attempt_row = conn.execute(
        """
        SELECT
            COUNT(*) AS attempt_count,
            COUNT(DISTINCT test_code) AS unique_tests
        FROM attempt
        WHERE session_id=?
        """,
        (int(session_id),),
    ).fetchone()
    avg_row = conn.execute(
        """
        SELECT
            AVG(CASE WHEN key = 'accuracy' THEN CAST(NULLIF(value, '') AS REAL) END) AS mean_accuracy,
            AVG(CASE WHEN key = 'score_ratio' THEN CAST(NULLIF(value, '') AS REAL) END) AS mean_score_ratio
        FROM attempt_metric
        WHERE attempt_id IN (SELECT id FROM attempt WHERE session_id=?)
        """,
        (int(session_id),),
    ).fetchone()

    with conn:
        conn.execute(
            """
            INSERT INTO session_summary(
                session_id, started_at_utc, completed_at_utc, exit_reason,
                activity_count, completed_activity_count, aborted_activity_count,
                attempt_count, unique_tests, mean_accuracy, mean_score_ratio
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                started_at_utc=excluded.started_at_utc,
                completed_at_utc=excluded.completed_at_utc,
                exit_reason=excluded.exit_reason,
                activity_count=excluded.activity_count,
                completed_activity_count=excluded.completed_activity_count,
                aborted_activity_count=excluded.aborted_activity_count,
                attempt_count=excluded.attempt_count,
                unique_tests=excluded.unique_tests,
                mean_accuracy=excluded.mean_accuracy,
                mean_score_ratio=excluded.mean_score_ratio
            """,
            (
                int(session_id),
                str(session_row[0]),
                str(session_row[1]),
                session_row[2],
                int(counts_row[0] or 0),
                int(counts_row[1] or 0),
                int(counts_row[2] or 0),
                int(attempt_row[0] or 0),
                int(attempt_row[1] or 0),
                None if avg_row[0] is None else float(avg_row[0]),
                None if avg_row[1] is None else float(avg_row[1]),
            ),
        )
        conn.execute("DELETE FROM session_metric WHERE session_id=?", (int(session_id),))

        for key, aggregate, alias in (
            ("mean_rt_ms", "AVG", "avg_value"),
            ("rt_variance_ms2", "AVG", "avg_value"),
            ("timeout_rate", "AVG", "avg_value"),
            ("post_error_next_item_rt_inflation_ms", "AVG", "avg_value"),
            ("longest_lapse_streak", "MAX", "max_value"),
        ):
            row = conn.execute(
                f"""
                SELECT {aggregate}(CAST(NULLIF(value, '') AS REAL))
                FROM attempt_metric
                WHERE key = ? AND attempt_id IN (SELECT id FROM attempt WHERE session_id=?)
                """,
                (key, int(session_id)),
            ).fetchone()
            if row is None or row[0] is None:
                continue
            conn.execute(
                "INSERT INTO session_metric(session_id, key, value) VALUES (?, ?, ?)",
                (int(session_id), key, f"{float(row[0]):.6f}"),
            )


def _fetch_session_summary(*, conn: sqlite3.Connection, session_id: int) -> SessionSummary | None:
    row = conn.execute(
        """
        SELECT
            session_id,
            started_at_utc,
            completed_at_utc,
            exit_reason,
            activity_count,
            completed_activity_count,
            aborted_activity_count,
            attempt_count,
            unique_tests,
            mean_accuracy,
            mean_score_ratio
        FROM session_summary
        WHERE session_id=?
        """,
        (int(session_id),),
    ).fetchone()
    if row is None:
        return None
    return SessionSummary(
        session_id=int(row[0]),
        started_at_utc=str(row[1]),
        completed_at_utc=str(row[2]),
        exit_reason=None if row[3] is None else str(row[3]),
        activity_count=int(row[4]),
        completed_activity_count=int(row[5]),
        aborted_activity_count=int(row[6]),
        attempt_count=int(row[7]),
        unique_tests=int(row[8]),
        mean_accuracy=None if row[9] is None else float(row[9]),
        mean_score_ratio=None if row[10] is None else float(row[10]),
    )


def _fetch_test_session_summary(
    *,
    conn: sqlite3.Connection,
    session_id: int,
    test_code: str,
) -> TestSessionSummary | None:
    row = conn.execute(
        """
        SELECT
            COUNT(*),
            MAX(
                (
                    SELECT CAST(NULLIF(m.value, '') AS REAL)
                    FROM attempt_metric AS m
                    WHERE m.attempt_id = a.id AND m.key = 'accuracy'
                )
            ),
            MAX(
                (
                    SELECT CAST(NULLIF(m.value, '') AS REAL)
                    FROM attempt_metric AS m
                    WHERE m.attempt_id = a.id AND m.key = 'score_ratio'
                )
            )
        FROM attempt AS a
        WHERE a.session_id = ? AND a.test_code = ?
        """,
        (int(session_id), str(test_code)),
    ).fetchone()
    if row is None or int(row[0]) <= 0:
        return None

    latest_row = conn.execute(
        """
        SELECT
            (
                SELECT CAST(NULLIF(m.value, '') AS REAL)
                FROM attempt_metric AS m
                WHERE m.attempt_id = a.id AND m.key = 'accuracy'
            ),
            (
                SELECT CAST(NULLIF(m.value, '') AS REAL)
                FROM attempt_metric AS m
                WHERE m.attempt_id = a.id AND m.key = 'score_ratio'
            )
        FROM attempt AS a
        WHERE a.session_id = ? AND a.test_code = ?
        ORDER BY a.id DESC
        LIMIT 1
        """,
        (int(session_id), str(test_code)),
    ).fetchone()

    latest_accuracy = None if latest_row is None or latest_row[0] is None else float(latest_row[0])
    latest_score_ratio = None if latest_row is None or latest_row[1] is None else float(latest_row[1])

    return TestSessionSummary(
        session_id=int(session_id),
        test_code=str(test_code),
        attempt_count=int(row[0]),
        latest_accuracy=latest_accuracy,
        best_accuracy=None if row[1] is None else float(row[1]),
        latest_score_ratio=latest_score_ratio,
        best_score_ratio=None if row[2] is None else float(row[2]),
    )
