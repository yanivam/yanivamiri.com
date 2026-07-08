"""SQLite storage for booth demo sessions and emails."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from demo.constants import DB_DIR

DB_PATH = DB_DIR / "booth.db"


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_email_columns(conn: sqlite3.Connection) -> None:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(emails)").fetchall()}
    if "name" not in cols:
        conn.execute("ALTER TABLE emails ADD COLUMN name TEXT")


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                selected_drugs TEXT NOT NULL,
                selected_factors TEXT NOT NULL,
                background TEXT,
                result_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS emails (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                name TEXT,
                email TEXT NOT NULL,
                consent INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );
            """
        )
        _ensure_email_columns(conn)


def save_session(
    session_id: str,
    selected_drugs: list[str],
    selected_factors: list[str],
    background: str | None,
    result: dict,
) -> str:
    init_db()
    created_at = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO sessions (session_id, selected_drugs, selected_factors, background, result_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                json.dumps(selected_drugs),
                json.dumps(selected_factors),
                background,
                json.dumps(result),
                created_at,
            ),
        )
    return session_id


def save_email(
    session_id: str,
    email: str,
    consent: bool,
    name: str | None = None,
) -> None:
    init_db()
    created_at = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO emails (session_id, name, email, consent, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                session_id,
                (name or "").strip() or None,
                email.strip().lower(),
                1 if consent else 0,
                created_at,
            ),
        )


def get_session(session_id: str) -> dict | None:
    init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    if row is None:
        return None
    return {
        "session_id": row["session_id"],
        "selected_drugs": json.loads(row["selected_drugs"]),
        "selected_factors": json.loads(row["selected_factors"]),
        "background": row["background"],
        "result": json.loads(row["result_json"]),
        "created_at": row["created_at"],
    }


def save_email_for_session(
    session_id: str,
    email: str,
    consent: bool,
    name: str | None = None,
) -> None:
    """Attach contact info to an existing session (e.g. from results page)."""
    if not get_session(session_id):
        raise ValueError("Unknown session")
    save_email(session_id, email, consent, name=name)


def export_all_sessions() -> list[dict]:
    """All booth runs for post-conference analysis."""
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT session_id, selected_drugs, selected_factors, background, created_at
            FROM sessions ORDER BY created_at ASC
            """
        ).fetchall()
    return [
        {
            "session_id": row["session_id"],
            "selected_drugs": json.loads(row["selected_drugs"]),
            "selected_factors": json.loads(row["selected_factors"]),
            "background": row["background"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def export_all_emails() -> list[dict]:
    """Contacts collected at launch or on the results page."""
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT session_id, name, email, consent, created_at
            FROM emails ORDER BY created_at ASC
            """
        ).fetchall()
    return [
        {
            "session_id": row["session_id"],
            "name": row["name"] or "",
            "email": row["email"],
            "consent": bool(row["consent"]),
            "created_at": row["created_at"],
        }
        for row in rows
    ]
