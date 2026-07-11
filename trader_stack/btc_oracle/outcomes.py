"""Outcome resolver — fills in the realized FX price for signals whose
horizon has elapsed, so the calibration layer has labeled training data.

Runs as a periodic background job inside the daemon (not its own process).
Each invocation:

  1. Queries Postgres for signals whose horizon has elapsed but have no
     outcome yet.
  2. Fetches the realized close at signal_ts + horizon for each, via
     feeds.fetch_ohlcv_range() (Dukascopy-backed — see feeds.py).
  3. Writes the outcome row, linking signal_id → realized_price.

We batch the fetches: one call covers many signals if they land in the same
time window.

Failure modes (all non-fatal — the daemon must never die from this):
  - Postgres unreachable → log and skip; next cycle retries.
  - Dukascopy fetch fails → log and skip; signal stays unresolved, next cycle retries.
  - Signal too old (>MAX_WINDOW_MINUTES ago) → mark as 'unresolvable' so we
    stop retrying it forever. (Rare; only happens after long outages.)

NOTE: Dukascopy has publish lag (see feeds.py module docstring) — a signal
whose horizon just elapsed a few minutes ago may not resolve until the
relevant hourly tick file is published. RESOLUTION_AGE_MULT gives it a head
start; genuinely fresh gaps just retry next cycle.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from . import store
from .feeds import fetch_ohlcv_range
from .oracle_config import OracleConfig

log = logging.getLogger(__name__)


# How old a signal must be (relative to its OWN horizon) before we resolve it.
# 1.5x the horizon = we wait until well after the bar closes AND Dukascopy has
# had a chance to publish the hour. 10-min horizon → resolve after 15 min.
RESOLUTION_AGE_MULT = 1.5

# Max window (minutes) we'll pull in one go before marking older signals
# 'too old' for this pass (they're retried next pass, just not batched with
# the rest of the window).
MAX_WINDOW_MINUTES = 1500


def _classify(pct: float, threshold: float) -> str:
    """Same classification rule as the live signal layer."""
    if pct >= threshold:
        return "UP"
    if pct <= -threshold:
        return "DOWN"
    return "FLAT"


def _fetch_price_window(cfg: OracleConfig, start: datetime, end: datetime):
    """Fetch M1 closes for [start, end] via the Dukascopy feed. Returns a
    DataFrame indexed by timestamps, or None on failure/empty result."""
    try:
        df = fetch_ohlcv_range(cfg, start, end)
    except Exception as e:
        log.warning("Dukascopy fetch for outcome resolution failed: %s", e)
        return None

    if df.empty:
        return None
    return df[["timestamps", "close"]].set_index("timestamps")


def resolve_pending(
    cfg: OracleConfig,
    flat_threshold: float = 0.0015,
    max_to_resolve: int = 200,
) -> dict:
    """One pass: find unresolved signals, fetch realized prices, write outcomes.

    Returns a dict with counts: {checked, resolved, failed, skipped_too_old}.
    Safe to call every cycle — it's a no-op if there's nothing to resolve.
    """
    out = {"checked": 0, "resolved": 0, "failed": 0, "skipped_too_old": 0}

    # min_age_minutes: the LARGEST horizon we use is the one to wait past.
    # Conservatively wait 15 min — covers our 10-min default with safety margin.
    pending = store.query_unresolved_signals(min_age_minutes=15)
    if not pending:
        return out

    pending = pending[:max_to_resolve]
    out["checked"] = len(pending)

    # Find the time window we need to fetch bars for.
    # Each signal's "future_ts" = ts + horizon_minutes.
    earliest_future = min(p["ts"] + timedelta(minutes=int(p["horizon_minutes"])) for p in pending)
    latest_future = max(p["ts"] + timedelta(minutes=int(p["horizon_minutes"])) for p in pending)

    # If the window spans more than MAX_WINDOW_MINUTES, cap it — older signals
    # are marked too-old for THIS pass and retried next time.
    now = datetime.now(timezone.utc)
    if (now - earliest_future) > timedelta(minutes=MAX_WINDOW_MINUTES):
        cutoff = now - timedelta(minutes=MAX_WINDOW_MINUTES - 10)
        pending_recent = [p for p in pending
                          if (p["ts"] + timedelta(minutes=int(p["horizon_minutes"]))) >= cutoff]
        out["skipped_too_old"] = len(pending) - len(pending_recent)
        if not pending_recent:
            log.info("Outcome resolver: all %d pending signals are >%dh old, skipping",
                     len(pending), MAX_WINDOW_MINUTES // 60)
            return out
        pending = pending_recent
        earliest_future = min(p["ts"] + timedelta(minutes=int(p["horizon_minutes"])) for p in pending)
        latest_future = max(p["ts"] + timedelta(minutes=int(p["horizon_minutes"])) for p in pending)

    # Pad the window by a few minutes on either side so searchsorted is safe.
    window_start = earliest_future - timedelta(minutes=2)
    window_end = latest_future + timedelta(minutes=2)
    bars = _fetch_price_window(cfg, window_start, window_end)
    if bars is None or bars.empty:
        out["failed"] = len(pending)
        return out

    # Resolve each one against the bar window.
    for sig in pending:
        future_ts = sig["ts"] + timedelta(minutes=int(sig["horizon_minutes"]))
        # Find the bar whose timestamp matches (or is just after) future_ts.
        try:
            idx = bars.index.searchsorted(future_ts)
            if idx >= len(bars):
                idx = len(bars) - 1
            realized_price = float(bars["close"].iloc[idx])
            realized_pct = realized_price / sig["last_close"] - 1.0
            actual_dir = _classify(realized_pct, flat_threshold)

            ok = store.write_outcome(
                signal_id=sig["id"],
                realized_at=future_ts,
                realized_price=realized_price,
                realized_pct=realized_pct,
                actual_direction=actual_dir,
            )
            if ok:
                out["resolved"] += 1
            else:
                out["failed"] += 1
        except Exception as e:
            log.warning("Failed to resolve signal id=%s: %s", sig["id"], e)
            out["failed"] += 1

    if out["resolved"] > 0:
        log.info("Outcome resolver: %d signals resolved (failed=%d, too_old=%d)",
                 out["resolved"], out["failed"], out["skipped_too_old"])

    return out
