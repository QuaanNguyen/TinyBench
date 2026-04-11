from __future__ import annotations

import math
import sqlite3
import logging
from typing import Any

from app.config import DB_PATH

logger = logging.getLogger(__name__)

_ELO_INITIAL = 1500
_ELO_K = 32


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS job_metrics (
                job_id         TEXT PRIMARY KEY,
                model_id       TEXT NOT NULL,
                throughput_tps REAL,
                n_tokens       INTEGER,
                generation_ms  REAL,
                created_at     TEXT DEFAULT (datetime('now'))
            )
            """
        )
    logger.info("Ranking store initialised (db=%s)", DB_PATH)


def save_job_metrics(
    job_id: str,
    model_id: str,
    throughput_tps: float | None,
    n_tokens: int | None,
    generation_ms: float | None,
) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO job_metrics "
            "(job_id, model_id, throughput_tps, n_tokens, generation_ms) "
            "VALUES (?, ?, ?, ?, ?)",
            (job_id, model_id, throughput_tps, n_tokens, generation_ms),
        )


def _compute_elo() -> dict[str, dict[str, Any]]:
    """Compute ELO ratings from all voted fights."""
    ratings: dict[str, float] = {}
    wins: dict[str, int] = {}
    losses: dict[str, int] = {}

    with _connect() as conn:
        rows = conn.execute(
            "SELECT model_a, model_b, winner "
            "FROM fights WHERE winner IS NOT NULL "
            "ORDER BY created_at"
        ).fetchall()

    for row in rows:
        model_a = row["model_a"]
        model_b = row["model_b"]
        winner_label = row["winner"]

        if winner_label == "A":
            winner_model, loser_model = model_a, model_b
        else:
            winner_model, loser_model = model_b, model_a

        ratings.setdefault(winner_model, float(_ELO_INITIAL))
        ratings.setdefault(loser_model, float(_ELO_INITIAL))
        wins.setdefault(winner_model, 0)
        wins.setdefault(loser_model, 0)
        losses.setdefault(winner_model, 0)
        losses.setdefault(loser_model, 0)

        r_w = ratings[winner_model]
        r_l = ratings[loser_model]
        expected_w = 1.0 / (1.0 + 10.0 ** ((r_l - r_w) / 400.0))
        expected_l = 1.0 - expected_w

        ratings[winner_model] = r_w + _ELO_K * (1.0 - expected_w)
        ratings[loser_model] = r_l + _ELO_K * (0.0 - expected_l)

        wins[winner_model] += 1
        losses[loser_model] += 1

    result: dict[str, dict[str, Any]] = {}
    for model_id in ratings:
        n_games = wins.get(model_id, 0) + losses.get(model_id, 0)
        ci = round(400.0 / math.sqrt(n_games)) if n_games > 0 else 0
        result[model_id] = {
            "score": round(ratings[model_id]),
            "ci": ci,
            "wins": wins.get(model_id, 0),
            "losses": losses.get(model_id, 0),
            "votes": wins.get(model_id, 0),
        }

    return result


def _get_avg_throughput() -> dict[str, float]:
    """Average throughput_tps per model from job_metrics."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT model_id, AVG(throughput_tps) as avg_tps "
            "FROM job_metrics "
            "WHERE throughput_tps IS NOT NULL AND throughput_tps > 0 "
            "GROUP BY model_id"
        ).fetchall()
    return {row["model_id"]: round(row["avg_tps"], 2) for row in rows}


def compute_rankings(
    context_lengths: dict[str, int | None] | None = None,
) -> dict[str, Any]:
    """Compute full ranking data for the API response."""
    elo_data = _compute_elo()
    throughput_data = _get_avg_throughput()
    context_lengths = context_lengths or {}

    models = []
    for rank, (model_id, data) in enumerate(
        sorted(elo_data.items(), key=lambda x: x[1]["score"], reverse=True),
        start=1,
    ):
        models.append(
            {
                "rank": rank,
                "model_id": model_id,
                "score": data["score"],
                "ci": data["ci"],
                "votes": data["votes"],
                "wins": data["wins"],
                "losses": data["losses"],
                "context_length": context_lengths.get(model_id),
                "throughput_tps": throughput_data.get(model_id),
                "power_w": None,
            }
        )

    with _connect() as conn:
        total_votes = conn.execute(
            "SELECT COUNT(*) FROM fights WHERE winner IS NOT NULL"
        ).fetchone()[0]
        last_row = conn.execute(
            "SELECT MAX(created_at) FROM fights WHERE winner IS NOT NULL"
        ).fetchone()
        last_updated = last_row[0] if last_row else None

    return {
        "updated_at": last_updated,
        "total_votes": total_votes,
        "total_models": len(models),
        "models": models,
    }
