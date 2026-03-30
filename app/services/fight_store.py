from __future__ import annotations

import sqlite3
import logging

from app.config import DB_PATH

logger = logging.getLogger(__name__)


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fights (
                fight_id   TEXT PRIMARY KEY,
                model_a    TEXT NOT NULL,
                model_b    TEXT NOT NULL,
                prompt     TEXT NOT NULL,
                winner     TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )
            """
        )
    logger.info("Fight store initialised (db=%s)", DB_PATH)


def save_fight(fight_id: str, model_a: str, model_b: str, prompt: str) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO fights (fight_id, model_a, model_b, prompt) VALUES (?, ?, ?, ?)",
            (fight_id, model_a, model_b, prompt),
        )


def record_vote(fight_id: str, winner: str) -> dict[str, str] | None:
    """Record the winner ('A' or 'B') and return model names for reveal."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT model_a, model_b, winner FROM fights WHERE fight_id = ?",
            (fight_id,),
        ).fetchone()
        if row is None:
            return None
        if row["winner"] is not None:
            return {"model_a": row["model_a"], "model_b": row["model_b"]}
        conn.execute(
            "UPDATE fights SET winner = ? WHERE fight_id = ?",
            (winner, fight_id),
        )
    return {"model_a": row["model_a"], "model_b": row["model_b"]}
