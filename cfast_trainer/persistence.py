from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from .results import AttemptResult

RESULTS_DB_ENV = "CFAST_RESULTS_DB_PATH"
SCHEMA_VERSION = 1


def open_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    _migrate(conn)
    return conn


def _utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time()))


def _utc_iso_from_epoch(epoch_s: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch_s))


@dataclass(frozen=True, slots=True)
class SavedAttempt:
    attempt_id: int
    session_id: int


@dataclass(frozen=True, slots=True)
class SessionSummary:
    session_id: int
    started_at_utc: str
    completed_at_utc: str
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

    def record_attempt(
        self,
        *,
        result: AttemptResult,
        app_version: str,
        input_profile_id: str | None = None,
    ) -> SavedAttempt:
        saved = record_attempt(
            db_path=self._path,
            result=result,
            app_version=app_version,
            session_id=self._session_id,
            input_profile_id=input_profile_id,
        )
        self._session_id = saved.session_id
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
                created_at_utc TEXT NOT NULL
            );
            """
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
                completed_at_utc TEXT NOT NULL
            );
            """
        )
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
        conn.execute(f"PRAGMA user_version={SCHEMA_VERSION};")


def record_attempt(
    *,
    db_path: Path,
    result: AttemptResult,
    app_version: str,
    session_id: int | None = None,
    input_profile_id: str | None = None,
) -> SavedAttempt:
    conn = open_db(db_path)
    try:
        session_id = _ensure_session(conn=conn, session_id=session_id)
        attempt_id = _insert_attempt(
            conn=conn,
            session_id=session_id,
            result=result,
            app_version=app_version,
            input_profile_id=input_profile_id,
        )
        return SavedAttempt(attempt_id=attempt_id, session_id=session_id)
    finally:
        conn.close()


def record_math_reasoning_attempt(*, db_path: Path, result: AttemptResult, app_version: str) -> int:
    saved = record_attempt(
        db_path=db_path,
        result=result,
        app_version=app_version,
    )
    return saved.attempt_id


def load_session_summary(*, db_path: Path, session_id: int) -> SessionSummary | None:
    conn = open_db(db_path)
    try:
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


def _ensure_session(*, conn: sqlite3.Connection, session_id: int | None) -> int:
    if session_id is not None:
        row = conn.execute("SELECT id FROM session WHERE id=?", (int(session_id),)).fetchone()
        if row is not None:
            return int(row[0])
    with conn:
        cur = conn.execute("INSERT INTO session(created_at_utc) VALUES (?)", (_utc_now_iso(),))
    return int(cur.lastrowid)


def _insert_attempt(
    *,
    conn: sqlite3.Connection,
    session_id: int,
    result: AttemptResult,
    app_version: str,
    input_profile_id: str | None,
) -> int:
    completed_at_epoch = time.time()
    started_at_epoch = completed_at_epoch - max(0.0, float(result.duration_s))
    completed_at_utc = _utc_iso_from_epoch(completed_at_epoch)
    started_at_utc = _utc_iso_from_epoch(started_at_epoch)

    with conn:
        cur = conn.execute(
            """
            INSERT INTO attempt(
                session_id, test_code, test_version, app_version,
                rng_seed, difficulty, input_profile_id,
                practice_questions, scored_duration_s,
                started_at_utc, completed_at_utc
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                str(result.test_code),
                int(result.test_version),
                app_version,
                int(result.seed),
                float(result.difficulty),
                input_profile_id,
                int(result.practice_questions),
                float(result.scored_duration_s),
                started_at_utc,
                completed_at_utc,
            ),
        )
        attempt_id = int(cur.lastrowid)

        for k, v in result.metrics.items():
            conn.execute(
                "INSERT INTO metric(attempt_id, key, value) VALUES (?, ?, ?)", (attempt_id, k, v)
            )

        for e in result.events:
            # Map QuestionEvent -> persistence schema.
            response_text = (e.raw or "").strip() or str(e.user_answer)
            conn.execute(
                """
                INSERT INTO cognitive_event(
                    attempt_id, seq, phase, prompt, expected, response, is_correct,
                    presented_at_ms, answered_at_ms, rt_ms
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    attempt_id,
                    int(e.index),
                    str(e.phase.value),
                    str(e.prompt),
                    str(e.correct_answer),
                    response_text,
                    1 if e.is_correct else 0,
                    int(round(e.presented_at_s * 1000.0)),
                    int(round(e.answered_at_s * 1000.0)),
                    int(round(e.response_time_s * 1000.0)),
                ),
            )

    return attempt_id


def _fetch_session_summary(*, conn: sqlite3.Connection, session_id: int) -> SessionSummary | None:
    row = conn.execute(
        """
        SELECT
            s.id,
            s.created_at_utc,
            COALESCE(MAX(a.completed_at_utc), s.created_at_utc),
            COUNT(a.id),
            COUNT(DISTINCT a.test_code),
            AVG(
                (
                    SELECT CAST(NULLIF(m.value, '') AS REAL)
                    FROM metric AS m
                    WHERE m.attempt_id = a.id AND m.key = 'accuracy'
                )
            ),
            AVG(
                (
                    SELECT CAST(NULLIF(m.value, '') AS REAL)
                    FROM metric AS m
                    WHERE m.attempt_id = a.id AND m.key = 'score_ratio'
                )
            )
        FROM session AS s
        LEFT JOIN attempt AS a ON a.session_id = s.id
        WHERE s.id = ?
        GROUP BY s.id, s.created_at_utc
        """,
        (int(session_id),),
    ).fetchone()
    if row is None:
        return None
    return SessionSummary(
        session_id=int(row[0]),
        started_at_utc=str(row[1]),
        completed_at_utc=str(row[2]),
        attempt_count=int(row[3]),
        unique_tests=int(row[4]),
        mean_accuracy=None if row[5] is None else float(row[5]),
        mean_score_ratio=None if row[6] is None else float(row[6]),
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
                    FROM metric AS m
                    WHERE m.attempt_id = a.id AND m.key = 'accuracy'
                )
            ),
            MAX(
                (
                    SELECT CAST(NULLIF(m.value, '') AS REAL)
                    FROM metric AS m
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
                FROM metric AS m
                WHERE m.attempt_id = a.id AND m.key = 'accuracy'
            ),
            (
                SELECT CAST(NULLIF(m.value, '') AS REAL)
                FROM metric AS m
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
