"""Postgres-backed memory store for the BTC oracle.

This is the "never forget" layer. Every signal, every realized outcome,
every calibration model state, every lesson learned — all of it lives in
a single local Postgres database (Docker-managed).

Design principles:

  1. **Dual-write for safety**: every write here is mirrored to the existing
     JSONL files. If Postgres ever has a hiccup, JSONL is the fallback.
     We'll drop JSONL once we've trusted Postgres for ~2 weeks.

  2. **Connection pooling, not per-call connects**: a single global pool is
     opened lazily on first call. Postgres connections are cheap (~1ms) but
     opening one per signal would be wasteful at 1 signal/min × 525k/year.

  3. **Defensive writes**: a Postgres outage must NEVER kill the daemon
     cycle. All writes wrap their I/O in try/except, log on failure, and
     return False instead of raising. The signal still gets written to JSONL.

  4. **Schema migrations live HERE**, not in init.sql. init.sql runs exactly
     once at first DB creation. Anything we add later goes through
     ensure_schema() at startup, which is idempotent.

Connection config:
    Read from env vars with sensible defaults matching docker-compose.yml.
    Override any of:
        PG_HOST     (default 127.0.0.1)
        PG_PORT     (default 5433)
        PG_USER     (default trader)
        PG_PASSWORD (default trader_local_dev)
        PG_DATABASE (default trader_stack)
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Connection config + pool (lazy)
# ---------------------------------------------------------------------------


def _conn_kwargs() -> dict:
    """Connection parameters, env-overridable."""
    return {
        "host":     os.getenv("PG_HOST", "127.0.0.1"),
        "port":     int(os.getenv("PG_PORT", "5433")),
        "user":     os.getenv("PG_USER", "trader"),
        "password": os.getenv("PG_PASSWORD", "trader_local_dev"),
        "dbname":   os.getenv("PG_DATABASE", "trader_stack"),
        # 5s connect timeout — fail fast if Postgres is down rather than
        # hanging the daemon for a minute.
        "connect_timeout": 5,
    }


_pool = None  # lazily initialized SimpleConnectionPool


def _get_pool():
    """Get-or-create the global connection pool."""
    global _pool
    if _pool is not None:
        return _pool
    try:
        # psycopg2's pool is enough for our 1-writer + occasional-readers
        # workload. If we ever fan out to multiple writers, swap for asyncpg.
        from psycopg2.pool import SimpleConnectionPool
    except ImportError as e:
        raise RuntimeError(
            "psycopg2-binary not installed. Add it to requirements.txt and "
            "reinstall: pip install psycopg2-binary"
        ) from e

    _pool = SimpleConnectionPool(
        minconn=1,
        maxconn=5,    # daemon (1) + occasional ad-hoc queries
        **_conn_kwargs(),
    )
    log.info("Postgres pool opened to %s:%s/%s",
             _conn_kwargs()["host"], _conn_kwargs()["port"], _conn_kwargs()["dbname"])
    return _pool


@contextmanager
def get_connection():
    """Context manager yielding a pooled connection. Auto-returned on exit."""
    pool = _get_pool()
    conn = pool.getconn()
    try:
        yield conn
    finally:
        pool.putconn(conn)


# ---------------------------------------------------------------------------
# Sanity check — used by init_postgres.sh and the test helper
# ---------------------------------------------------------------------------


def test_connection() -> dict:
    """Verify the schema is present and report a quick summary.

    Returns a dict with row counts and the latest signal timestamp, suitable
    for printing to a human or returning via JSON.
    """
    out: dict[str, Any] = {"ok": False}
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
                # Verify the four core tables exist
                cur.execute("""
                    SELECT tablename FROM pg_tables
                    WHERE schemaname = 'public'
                      AND tablename IN ('signals', 'outcomes', 'calibration', 'lessons')
                    ORDER BY tablename
                """)
                tables = [r[0] for r in cur.fetchall()]
                cur.execute("SELECT COUNT(*) FROM signals")
                n_signals = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM outcomes")
                n_outcomes = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM calibration")
                n_calibration = cur.fetchone()[0]
                cur.execute("SELECT MAX(ts) FROM signals")
                latest = cur.fetchone()[0]

                out.update({
                    "ok": True,
                    "tables": tables,
                    "row_counts": {
                        "signals": n_signals,
                        "outcomes": n_outcomes,
                        "calibration": n_calibration,
                    },
                    "latest_signal_ts": latest.isoformat() if latest else None,
                })
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


# ---------------------------------------------------------------------------
# Idempotent schema migrations (runs at daemon startup)
# ---------------------------------------------------------------------------


def ensure_schema() -> None:
    """Apply any schema changes added AFTER the initial init.sql.

    Idempotent — safe to call on every daemon start. Each migration should
    be guarded with IF NOT EXISTS or pg_catalog lookups so re-runs are no-ops.

    This is where you add columns / indices later without manually editing
    init.sql or recreating the DB. Each migration goes in a numbered block.
    """
    migrations = [
        # M1: placeholder. Add real migrations below as the schema evolves.
        # Example shape:
        #   "ALTER TABLE signals ADD COLUMN IF NOT EXISTS new_field TEXT",
    ]
    if not migrations:
        return
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                for sql in migrations:
                    cur.execute(sql)
            conn.commit()
        log.info("Applied %d schema migrations", len(migrations))
    except Exception as e:
        log.error("Schema migration failed: %s", e)


# ---------------------------------------------------------------------------
# Signal write — the main hot-path call
# ---------------------------------------------------------------------------


def write_signal(signal_dict: dict, calibrated_confidence: Optional[float] = None) -> Optional[int]:
    """Persist a signal to Postgres. Returns the row ID, or None on failure.

    NEVER raises — Postgres outages must not kill the daemon. Callers should
    write to JSONL regardless of this function's return value.

    `signal_dict` is the dict produced by signal.Signal.to_dict() — same
    shape that already goes into signals.jsonl.
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                ts_iso = signal_dict["timestamp"]
                if ts_iso.endswith("Z"):
                    ts_iso = ts_iso[:-1] + "+00:00"

                # The "payload" column holds everything we don't normalize.
                # We strip out the fields we DO normalize so the payload
                # doesn't duplicate them (saves space, avoids drift).
                payload = {
                    "per_model":     signal_dict.get("per_model"),
                    "microstructure": signal_dict.get("microstructure"),
                    "volatility":    signal_dict.get("volatility"),
                    "filter_fired":  signal_dict.get("filter_fired"),
                    "notes":         signal_dict.get("notes"),
                }

                regime = signal_dict.get("regime") or {}
                cur.execute("""
                    INSERT INTO signals (
                        ts, last_close, horizon_minutes,
                        direction, confidence,
                        raw_direction, raw_confidence,
                        consensus_pct,
                        regime, regime_confidence,
                        calibrated_confidence,
                        payload
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    ts_iso,
                    float(signal_dict["last_close"]),
                    int(signal_dict["horizon_minutes"]),
                    signal_dict["direction"],
                    float(signal_dict["confidence"]),
                    signal_dict.get("raw_direction", signal_dict["direction"]),
                    float(signal_dict.get("raw_confidence", signal_dict["confidence"])),
                    float(signal_dict.get("consensus_pct_change", 0.0)),
                    regime.get("label"),
                    regime.get("confidence"),
                    calibrated_confidence,
                    json.dumps(payload, default=str),
                ))
                row_id = cur.fetchone()[0]
            conn.commit()
            return row_id
    except Exception as e:
        log.warning("Postgres write_signal failed (will rely on JSONL): %s", e)
        return None


# ---------------------------------------------------------------------------
# Outcome write — fills in the realized future after `horizon_minutes`
# ---------------------------------------------------------------------------


def write_outcome(
    signal_id: int,
    realized_at: datetime,
    realized_price: float,
    realized_pct: float,
    actual_direction: str,
) -> bool:
    """Persist a realized outcome. Returns True on success, False on failure.

    Idempotent — if the outcome already exists for this signal, we update it
    rather than failing (handy if the resolver job runs twice).
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO outcomes (
                        signal_id, realized_at, realized_price, realized_pct, actual_direction
                    ) VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (signal_id) DO UPDATE SET
                        realized_at = EXCLUDED.realized_at,
                        realized_price = EXCLUDED.realized_price,
                        realized_pct = EXCLUDED.realized_pct,
                        actual_direction = EXCLUDED.actual_direction,
                        resolved_at = NOW()
                """, (signal_id, realized_at, realized_price, realized_pct, actual_direction))
            conn.commit()
        return True
    except Exception as e:
        log.warning("Postgres write_outcome failed for signal_id=%s: %s", signal_id, e)
        return False


# ---------------------------------------------------------------------------
# Reads — used by the calibration layer, the backtest, the dashboard
# ---------------------------------------------------------------------------


def query_unresolved_signals(min_age_minutes: int) -> list[dict]:
    """Return signals whose outcome hasn't been resolved yet AND whose horizon
    has fully elapsed. Used by the outcome-resolver background job.

    `min_age_minutes` is how old the signal must be relative to NOW (so we
    don't try to resolve a signal whose 10-min horizon hasn't completed).
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT s.id, s.ts, s.last_close, s.horizon_minutes
                    FROM signals s
                    LEFT JOIN outcomes o ON o.signal_id = s.id
                    WHERE o.signal_id IS NULL
                      AND s.ts < NOW() - (%s::int * INTERVAL '1 minute')
                    ORDER BY s.ts ASC
                    LIMIT 500
                """, (min_age_minutes,))
                return [
                    {"id": r[0], "ts": r[1], "last_close": float(r[2]),
                     "horizon_minutes": int(r[3])}
                    for r in cur.fetchall()
                ]
    except Exception as e:
        log.warning("query_unresolved_signals failed: %s", e)
        return []


def query_recent_with_outcomes(n: int = 1000) -> list[dict]:
    """The bread-and-butter query for calibration + backtest.

    Returns the most recent N signals that DO have a resolved outcome,
    joined with that outcome. The order is newest-first; reverse on the
    caller side if you want chronological.
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT
                        id, ts, last_close, horizon_minutes,
                        direction, confidence,
                        raw_direction, raw_confidence,
                        consensus_pct, regime, regime_confidence,
                        calibrated_confidence,
                        payload,
                        realized_at, realized_price, realized_pct, actual_direction,
                        hit
                    FROM v_signal_outcomes
                    WHERE actual_direction IS NOT NULL
                    ORDER BY ts DESC
                    LIMIT %s
                """, (n,))
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, r)) for r in cur.fetchall()]
    except Exception as e:
        log.warning("query_recent_with_outcomes failed: %s", e)
        return []


# ---------------------------------------------------------------------------
# Calibration model state — written by the calibration layer (Step 3)
# ---------------------------------------------------------------------------


def write_calibration(
    model_json: dict,
    n_training_samples: int,
    brier_score: Optional[float] = None,
    ece: Optional[float] = None,
    notes: Optional[str] = None,
) -> Optional[int]:
    """Persist a freshly-fitted calibration model. Returns the row id."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO calibration (
                        n_training_samples, model_json, brier_score, ece, notes
                    ) VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                """, (n_training_samples, json.dumps(model_json), brier_score, ece, notes))
                row_id = cur.fetchone()[0]
            conn.commit()
            return row_id
    except Exception as e:
        log.warning("write_calibration failed: %s", e)
        return None


def read_latest_calibration() -> Optional[dict]:
    """Return the most recently fitted calibration model, or None if none exist."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, fit_at, n_training_samples, model_json,
                           brier_score, ece, notes
                    FROM calibration
                    ORDER BY fit_at DESC
                    LIMIT 1
                """)
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "id": row[0],
                    "fit_at": row[1],
                    "n_training_samples": row[2],
                    "model_json": row[3],   # Postgres returns JSONB as a dict already
                    "brier_score": row[4],
                    "ece": row[5],
                    "notes": row[6],
                }
    except Exception as e:
        log.warning("read_latest_calibration failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Lessons — the system's diary
# ---------------------------------------------------------------------------


def write_lesson(
    category: str,
    summary: str,
    metrics: Optional[dict] = None,
    period_start: Optional[datetime] = None,
    period_end: Optional[datetime] = None,
) -> Optional[int]:
    """Append a lesson. Categories are free-form; common ones:
    'daily_rollup', 'regime_shift', 'model_drift', 'init', 'anomaly'.
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO lessons (period_start, period_end, category, summary, metrics)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    period_start, period_end, category, summary,
                    json.dumps(metrics or {}, default=str),
                ))
                row_id = cur.fetchone()[0]
            conn.commit()
            return row_id
    except Exception as e:
        log.warning("write_lesson failed: %s", e)
        return None
