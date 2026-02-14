from __future__ import annotations

from pathlib import Path
import sqlite3
import time

from .results import AttemptResult

SCHEMA_VERSION = 1


def open_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    _migrate(conn)
    return conn


def _utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time()))


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
            "CREATE INDEX IF NOT EXISTS idx_cognitive_event_attempt_seq ON cognitive_event(attempt_id, seq);"
        )
        conn.execute(f"PRAGMA user_version={SCHEMA_VERSION};")


def record_math_reasoning_attempt(*, db_path: Path, result: AttemptResult, app_version: str) -> int:
    """
    Minimal persistence for this single test:
      session -> attempt -> metric + cognitive_event
    """
    conn = open_db(db_path)
    try:
        return _insert_attempt(conn=conn, result=result, app_version=app_version)
    finally:
        conn.close()


def _insert_attempt(*, conn: sqlite3.Connection, result: AttemptResult, app_version: str) -> int:
    now = _utc_now_iso()

    with conn:
        cur = conn.execute("INSERT INTO session(created_at_utc) VALUES (?)", (now,))
        session_id = int(cur.lastrowid)

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
                None,
                int(result.practice_questions),
                float(result.scored_duration_s),
                now,
                _utc_now_iso(),
            ),
        )
        attempt_id = int(cur.lastrowid)

        mean_rt = "" if result.mean_rt_ms is None else f"{result.mean_rt_ms:.3f}"
        median_rt = "" if result.median_rt_ms is None else f"{result.median_rt_ms:.3f}"
        metrics = {
            "attempted": str(result.attempted),
            "correct": str(result.correct),
            "accuracy": f"{result.accuracy:.6f}",
            "throughput_per_min": f"{result.throughput_per_min:.6f}",
            "mean_rt_ms": mean_rt,
            "median_rt_ms": median_rt,
        }
        for k, v in metrics.items():
            conn.execute("INSERT INTO metric(attempt_id, key, value) VALUES (?, ?, ?)", (attempt_id, k, v))

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