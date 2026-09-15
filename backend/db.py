"""PostgreSQL persistence for participant answers (replaces the Supabase client).

Connection settings come from ``DATABASE_URL`` or, when that is not set, from the
``POSTGRES_HOST`` / ``POSTGRES_PORT`` / ``POSTGRES_DB`` / ``POSTGRES_USER`` /
``POSTGRES_PASSWORD`` variables used by docker-compose.

One row per participant and study lives in the ``answers`` table. The full chat
transcript is stored there as JSON, so nothing that matters is kept in the
browser cookie.
"""

import logging
import os
import threading
import time
from typing import Any

import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

log = logging.getLogger(__name__)

# Columns the app may write, mapped to whether they hold JSON.
ANSWER_FIELDS = {
    "consent": False,
    "name": False,
    "age": False,
    "gender": False,
    "community": False,
    "agent_language": False,
    "usecase": False,
    "scenario": False,
    "initial": False,
    "context": False,
    "final": False,
    "chatbot_summary": True,
    "transcript": True,
    "completed_at": False,
}

# Column order used by the admin exports.
EXPORT_COLUMNS = [
    "id", "study", "user_id", "created_at", "updated_at", "completed_at",
    "consent", "name", "age", "gender", "community", "agent_language",
    "usecase", "scenario", "initial", "context", "final",
    "chatbot_summary", "transcript",
]

# Idempotent schema set-up; new columns should be appended as
# "ALTER TABLE answers ADD COLUMN IF NOT EXISTS ..." statements.
_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS answers (
        id BIGSERIAL PRIMARY KEY,
        user_id TEXT NOT NULL,
        study TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        consent BOOLEAN NOT NULL DEFAULT FALSE,
        "name" TEXT,
        age INTEGER,
        gender TEXT,
        community TEXT,
        agent_language TEXT,
        usecase TEXT,
        scenario TEXT,
        "initial" TEXT,
        "context" TEXT,
        "final" TEXT,
        chatbot_summary JSONB,
        transcript JSONB NOT NULL DEFAULT '[]'::jsonb,
        completed_at TIMESTAMPTZ,
        UNIQUE (user_id, study)
    )
    """,
    "CREATE INDEX IF NOT EXISTS answers_study_idx ON answers (study)",
    "CREATE INDEX IF NOT EXISTS answers_user_id_idx ON answers (user_id)",
]
_SCHEMA_LOCK_ID = 7_462_019  # arbitrary key for pg_advisory_lock

_pool: ConnectionPool | None = None
_pool_lock = threading.Lock()


def _conninfo() -> str:
    url = os.environ.get("DATABASE_URL", "").strip()
    if url:
        return url
    return make_conninfo(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=os.environ.get("POSTGRES_PORT", "5432"),
        dbname=os.environ.get("POSTGRES_DB", "futures"),
        user=os.environ.get("POSTGRES_USER", "futures"),
        password=os.environ.get("POSTGRES_PASSWORD", ""),
    )


def get_pool() -> ConnectionPool:
    """Return the process-wide pool, creating it on first use (after fork)."""
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = ConnectionPool(
                    conninfo=_conninfo(),
                    min_size=1,
                    max_size=int(os.environ.get("DB_POOL_SIZE", "10")),
                    kwargs={"row_factory": dict_row, "autocommit": True},
                    name="answers",
                    open=True,
                )
    return _pool


def init_db(retries: int = 30, delay: float = 2.0) -> None:
    """Create the schema, waiting for the database to accept connections."""
    for attempt in range(1, retries + 1):
        try:
            with psycopg.connect(_conninfo(), autocommit=True) as conn:
                conn.execute("SELECT pg_advisory_lock(%s)", (_SCHEMA_LOCK_ID,))
                try:
                    for statement in _SCHEMA:
                        conn.execute(statement)
                finally:
                    conn.execute("SELECT pg_advisory_unlock(%s)", (_SCHEMA_LOCK_ID,))
            return
        except psycopg.OperationalError as exc:
            if attempt == retries:
                raise
            log.warning("Database not ready (%s); retrying in %ss", exc, delay)
            time.sleep(delay)


def _prepare(fields: dict[str, Any]) -> dict[str, Any]:
    unknown = set(fields) - set(ANSWER_FIELDS)
    if unknown:
        raise ValueError(f"Unknown answer fields: {sorted(unknown)}")
    return {
        key: Jsonb(value) if ANSWER_FIELDS[key] and value is not None else value
        for key, value in fields.items()
    }


def _fetchone(query, params=()) -> dict | None:
    with get_pool().connection() as conn:
        return conn.execute(query, params).fetchone()


def _fetchall(query, params=()) -> list[dict]:
    with get_pool().connection() as conn:
        return conn.execute(query, params).fetchall()


def ping() -> None:
    _fetchone("SELECT 1 AS ok")


def get_answer(user_id: str, study: str) -> dict | None:
    return _fetchone(
        "SELECT * FROM answers WHERE user_id = %s AND study = %s", (user_id, study)
    )


def user_id_exists(user_id: str) -> bool:
    """True if this participant ID was already used in any study."""
    return _fetchone(
        "SELECT 1 AS found FROM answers WHERE user_id = %s LIMIT 1", (user_id,)
    ) is not None


def create_answer(user_id: str, study: str, **fields) -> bool:
    """Insert a participant row. Returns False if (user_id, study) already exists."""
    fields = _prepare(fields)
    columns = ["user_id", "study", *fields]
    query = sql.SQL(
        "INSERT INTO answers ({columns}) VALUES ({values}) "
        "ON CONFLICT (user_id, study) DO NOTHING RETURNING id"
    ).format(
        columns=sql.SQL(", ").join(map(sql.Identifier, columns)),
        values=sql.SQL(", ").join([sql.Placeholder()] * len(columns)),
    )
    return _fetchone(query, [user_id, study, *fields.values()]) is not None


def update_answer(user_id: str, study: str, **fields) -> None:
    fields = _prepare(fields)
    if not fields:
        return
    query = sql.SQL(
        "UPDATE answers SET {assignments}, updated_at = now() "
        "WHERE user_id = %s AND study = %s"
    ).format(
        assignments=sql.SQL(", ").join(
            sql.SQL("{} = %s").format(sql.Identifier(key)) for key in fields
        )
    )
    with get_pool().connection() as conn:
        conn.execute(query, [*fields.values(), user_id, study])


def list_answers(study: str | None = None) -> list[dict]:
    if study:
        return _fetchall(
            "SELECT * FROM answers WHERE study = %s ORDER BY created_at DESC", (study,)
        )
    return _fetchall("SELECT * FROM answers ORDER BY created_at DESC")


def get_answer_by_id(answer_id: int) -> dict | None:
    return _fetchone("SELECT * FROM answers WHERE id = %s", (answer_id,))


def delete_answer(answer_id: int) -> None:
    with get_pool().connection() as conn:
        conn.execute("DELETE FROM answers WHERE id = %s", (answer_id,))


def study_counts() -> dict[str, dict[str, int]]:
    rows = _fetchall(
        "SELECT study, count(*) AS started, count(completed_at) AS completed "
        "FROM answers GROUP BY study"
    )
    return {row["study"]: {"started": row["started"], "completed": row["completed"]} for row in rows}
