"""Adaptive ensemble weighting from rolling per-model accuracy.

Reads recent signals from signals.jsonl + their realized outcomes, computes
each model's rolling directional hit-rate over `weight_lookback` signals, and
returns a fresh weight dict. Models that fall below the accuracy floor are
clamped (not zeroed — we don't want to fully exit a model on a bad day).

The runner calls `recompute_weights()` periodically (every K cycles); the
returned dict replaces ORACLE_CONFIG.weights in place.

The realized-outcome lookup uses the same OHLCV fetch used by backtest.py.
We keep it cheap by only looking at signals whose horizon has fully elapsed
AND that don't have an `actual` field already cached.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from .feeds import fetch_ohlcv
from .oracle_config import OracleConfig
from .state import read_recent_signals

log = logging.getLogger(__name__)


def _classify(pct: float, threshold: float) -> str:
    if pct >= threshold:
        return "UP"
    if pct <= -threshold:
        return "DOWN"
    return "FLAT"


def _outcomes_from_signals(
    signals: list[dict],
    prices: pd.DataFrame,
    flat_threshold: float,
) -> list[dict]:
    """For each signal whose horizon is in the past and inside the price window,
    append the realised direction. Returns a list of signal+actual pairs.
    """
    if prices.empty:
        return []

    paired: list[dict] = []
    px_ts = prices["timestamps"]
    if not pd.api.types.is_datetime64_any_dtype(px_ts):
        px_ts = pd.to_datetime(px_ts, utc=True)
    px_close = prices["close"].astype(float).to_numpy()
    px_start = px_ts.iloc[0]
    px_end = px_ts.iloc[-1]

    for sig in signals:
        try:
            ts_str = sig["timestamp"]
            if ts_str.endswith("Z"):
                ts_str = ts_str[:-1] + "+00:00"
            ts = datetime.fromisoformat(ts_str)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            horizon = int(sig.get("horizon_minutes", 10))
            sig_close = float(sig["last_close"])
            future_ts = ts + timedelta(minutes=horizon)
        except (KeyError, ValueError, TypeError):
            continue

        if not (px_start <= future_ts <= px_end):
            continue

        idx = px_ts.searchsorted(future_ts)
        idx = min(int(idx), len(px_close) - 1)
        actual_pct = px_close[idx] / sig_close - 1.0
        actual = _classify(actual_pct, flat_threshold)

        paired.append({"signal": sig, "actual": actual, "actual_pct": actual_pct})

    return paired


def recompute_weights(
    cfg: OracleConfig,
    enabled_models: list[str],
    min_weight: float = 0.3,
    max_weight: float = 1.5,
) -> Optional[dict[str, float]]:
    """Return updated weights based on rolling per-model accuracy.

    Returns None if there isn't enough data yet (fewer than `weight_lookback//2`
    paired outcomes). The runner should leave existing weights untouched in that
    case.
    """
    signals = read_recent_signals(n=cfg.weight_lookback, cfg=cfg)
    if len(signals) < 40:
        log.debug("Adaptive weighting: %d signals, need ≥40 — skipping", len(signals))
        return None

    # Fetch a price window covering the oldest signal's horizon up to now.
    try:
        prices = fetch_ohlcv(cfg, limit=min(1500, cfg.weight_lookback + 60))
    except Exception as e:
        log.warning("Adaptive weighting: could not fetch prices: %s", e)
        return None

    paired = _outcomes_from_signals(signals, prices, cfg.flat_threshold)
    if len(paired) < 30:
        log.debug("Adaptive weighting: only %d paired outcomes — skipping", len(paired))
        return None

    # Per-model hits / counts
    counts: dict[str, int] = {m: 0 for m in enabled_models}
    hits: dict[str, int] = {m: 0 for m in enabled_models}

    for pair in paired:
        actual = pair["actual"]
        per_model = pair["signal"].get("per_model", {})
        for m in enabled_models:
            info = per_model.get(m)
            if not info:
                continue
            vote = info.get("vote", "FLAT")
            counts[m] += 1
            if vote == actual:
                hits[m] += 1

    # Convert to accuracies and weights.
    new_weights: dict[str, float] = {}
    accuracies: dict[str, float] = {}
    for m in enabled_models:
        if counts[m] == 0:
            new_weights[m] = cfg.weights.get(m, 1.0)
            continue
        acc = hits[m] / counts[m]
        accuracies[m] = acc
        # Anchor: 0.5 accuracy → weight 1.0. Each 1% accuracy → ~0.1 weight,
        # clamped to [min_weight, max_weight].
        raw_w = 1.0 + (acc - 0.5) * 10.0
        new_weights[m] = max(min_weight, min(max_weight, raw_w))

    log.info(
        "Adaptive weighting (%d outcomes): %s → %s",
        len(paired),
        {m: f"{a:.3f}" for m, a in accuracies.items()},
        {m: f"{w:.2f}" for m, w in new_weights.items()},
    )

    return new_weights
