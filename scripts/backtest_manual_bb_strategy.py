#!/usr/bin/env python3
"""Manual Bollinger(20,2) mean-reversion strategy backtest — XAUUSD M1, cached 90d.

STRATEGY (as specified by the user; underspecified bits are called out below
as disclosed assumptions, not silent choices):
  - Touch: bar's low <= lower_band (candidate long) or high >= upper_band
    (candidate short). Bollinger(20,2) computed on M1 close.
  - Confirmation: the NEXT bar closes back inside the band, in the reversal
    direction, as a directional (not doji) candle:
      long:  close[i+1] > lower_band[i+1]  and  close[i+1] > open[i+1]
      short: close[i+1] < upper_band[i+1]  and  close[i+1] < open[i+1]
  - Entry: at the OPEN of the bar AFTER confirmation (i+2). The decision is
    fully determined by data through the close of bar i+1 — no lookahead.
  - Direction filter: EMA(8) on M5-resampled close, evaluated at the DECISION
    bar (i+1, same bar as confirmation — the freshest M5 data actually known
    at decision time). Long touches only taken when m5_close > m5_ema8
    (uptrend bias); short touches only when m5_close < m5_ema8. Equal ->
    no trade. M5 bars are merged onto the M1 grid via merge_asof(backward),
    so every M1 bar sees only the most recently CLOSED M5 candle — no
    lookahead (see add_indicators()).
  - Session filter ["London/NY only"]: DISCLOSED ASSUMPTION — reuses
    backtest.py's CostModel.is_asian_session() boundary (default 22:00-07:00
    UTC, itself an assumed/approximate cutover, see CostModel's own
    docstring) rather than inventing a second, possibly-inconsistent set of
    London/NY clock hours. "London/NY" == NOT Asian by this same definition.
    Checked at the ENTRY bar (i+2) — the moment the trade is actually live
    and paying session-dependent spread.
  - Hard stop ["just beyond the entry rail"]: DISCLOSED ASSUMPTION — no
    buffer size was specified, so this uses 0.25x ATR(14) (at the touch bar)
    beyond the touched band level, because it scales with recent
    volatility rather than being an arbitrary fixed-point number. This is a
    free parameter; sweep it before trusting the result.
      long stop  = lower_band[touch] - 0.25*atr14[touch]
      short stop = upper_band[touch] + 0.25*atr14[touch]
  - Exit: whichever comes first, checked intrabar (high/low) bar-by-bar
    starting at the entry bar itself:
      (a) stop hit (checked BEFORE target if both trigger in the same bar —
          conservative),
      (b) middle band (SMA20, recomputed each bar) touched in the trade's
          favor,
      (c) DISCLOSED ASSUMPTION — forced flat at the OPEN of the first bar
          where the session filter goes non-London/NY (a manual scalp is
          not held into the illiquid Asian session). Not specified by the
          user; added because otherwise a stuck mean-reversion trade could
          run for days.
  - One open position at a time; new touches are ignored while a trade is
    open. Scanning resumes at the bar AFTER an exit.

COSTS: reuses backtest.py's CostModel exactly — session-aware spread,
slippage, commission, swap — via cost_model.round_trip_cost_pips(), fed the
trade's ACTUAL entry/exit timestamps (variable, not a fixed horizon) and the
bar's own measured spread_bps as the floor, exactly like
backtest.py's pair_signals_with_outcomes() does for the ensemble signals.

REGIME BREAKDOWN: mirrors scripts/replay_backtest.py's exact walk-forward
periodic-refit HMM policy (initial fit at 1500 bars, refit every 1440 bars,
each fit using ONLY the trailing 1500 bars available at that point — see
MIN_HMM_BARS/REGIME_REFIT_BARS below). For efficiency this computes each
block's regime labels with ONE model.predict_proba() call over that block
(with a discarded 1500-bar burn-in prefix) instead of a separate per-bar
classify() call with its own rolling window. This is mathematically
equivalent for a GaussianHMM: predict_proba's posterior at the LAST row of
any window reduces to the pure forward-filtered probability (no
within-window future information is used for the last row), so batching
does not change any single bar's label — it only reduces call count from
~88k to ~60. Used for REPORTING ONLY; never a trading filter, so it has zero
effect on P&L.

WALK-FORWARD CAVEAT: Bollinger/EMA/ATR are fixed-formula indicators (nothing
fit in-sample), so the only walk-forward concern on the strategy side is the
regime HMM (handled above) — the entry/exit logic itself has no lookahead
by construction. Results are folded by ISO week (matching backtest.py's
fold="week") purely for reporting, to see whether any edge is stable across
weeks or concentrated in one lucky week.

NOT MODELED: position sizing / lot size (all P&L in "pips" == XAUUSD points,
$0.01, per feeds.py's pip_size() convention, matching every other report in
this project) and multi-trade concurrency (impossible by construction here).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from trader_stack.btc_oracle import regime as regime_mod  # noqa: E402
from trader_stack.btc_oracle.backtest import CostModel, _fold_key  # noqa: E402
from trader_stack.btc_oracle.feeds import pip_size  # noqa: E402
from trader_stack.btc_oracle.oracle_config import ORACLE_CONFIG, OracleConfig  # noqa: E402

import replay_backtest as rb  # noqa: E402  (reuse its cache-first history loader)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s INFO %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("backtest_manual_bb")

# Same constants as scripts/replay_backtest.py — kept identical so the
# regime breakdown reflects the same walk-forward fit cadence as the rest
# of the project's replay tooling.
MIN_HMM_BARS = 1500
REGIME_REFIT_BARS = 60 * 24  # daily

BB_PERIOD = 20
BB_STD = 2.0
ATR_PERIOD = 14
EMA_FILTER_SPAN = 8
STOP_ATR_MULT = 0.25  # disclosed assumption — see module docstring; overridable per variant

# EMA8 direction filter moves one level up from the bar timeframe, mirroring
# the original M1-strategy/M5-filter ratio: 1min->5min, 5min->15min,
# 15min->60min (H1), 60min->240min (H4).
FILTER_RULE_FOR_BAR_TIMEFRAME = {
    "1min": "5min", "5min": "15min", "15min": "60min", "60min": "240min",
}


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample an M1 OHLCV frame to a coarser bar (e.g. "5min") for the
    variant (c) M5-bar-timeframe strategy. label='right'/closed='left' keeps
    the same causal convention used throughout this module (a bin's own
    timestamp IS its close time)."""
    ts = pd.to_datetime(df["timestamps"], utc=True)
    idx = pd.DatetimeIndex(ts.to_numpy())
    idx.name = "timestamps"
    tmp = df.set_index(idx)
    agg = {"open": "first", "high": "max", "low": "min", "close": "last",
           "volume": "sum", "amount": "sum", "spread_bps": "mean"}
    if "taker_buy_ratio" in tmp.columns:
        agg["taker_buy_ratio"] = "mean"
    out = tmp.resample(rule, label="right", closed="left").agg(agg)
    out = out.dropna(subset=["open", "high", "low", "close"]).reset_index()
    return out


def add_indicators(df: pd.DataFrame, cost_model: CostModel, filter_rule: str = "5min") -> pd.DataFrame:
    """filter_rule is the higher-timeframe EMA8 direction filter's bar size —
    "5min" for the M1 strategy (variants baseline/a/b), "15min" for the M5
    bar-timeframe strategy (variant c), keeping the same 1:5 ratio."""
    df = df.copy()
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    open_ = df["open"].astype(float)

    sma20 = close.rolling(BB_PERIOD, min_periods=BB_PERIOD).mean()
    std20 = close.rolling(BB_PERIOD, min_periods=BB_PERIOD).std(ddof=0)
    df["bb_mid"] = sma20
    df["bb_upper"] = sma20 + BB_STD * std20
    df["bb_lower"] = sma20 - BB_STD * std20

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(ATR_PERIOD, min_periods=ATR_PERIOD).mean()

    # Higher-timeframe resample, EMA8, merged back onto the bar grid with an
    # as-of backward join — every bar sees only the most recently CLOSED
    # higher-timeframe candle (label='right' means the bin's own timestamp
    # IS its close time, and that close time coincides with one of our own
    # bars, so no lookahead: the higher-tf bar closing at t is fully
    # knowable using data through bar t).
    ts = pd.to_datetime(df["timestamps"], utc=True)
    df["timestamps"] = ts
    close_by_ts = pd.Series(close.to_numpy(), index=pd.DatetimeIndex(ts.to_numpy()))
    filter_close = close_by_ts.resample(filter_rule, label="right", closed="left").last().dropna()
    filter_ema = filter_close.ewm(span=EMA_FILTER_SPAN, adjust=False).mean()
    filter_frame = pd.DataFrame({"filter_ts": filter_close.index, "filter_close": filter_close.to_numpy(),
                                  "filter_ema": filter_ema.to_numpy()}).sort_values("filter_ts")
    # df is already globally sorted by timestamps (see _load_full_history),
    # so this merge preserves df's row order positionally.
    merged = pd.merge_asof(
        df[["timestamps"]], filter_frame, left_on="timestamps", right_on="filter_ts", direction="backward",
    )
    df["filter_close"] = merged["filter_close"].to_numpy()
    df["filter_ema"] = merged["filter_ema"].to_numpy()

    # Session tag — mirrors CostModel.is_asian_session exactly (same
    # boundary source as the cost model, see module docstring).
    hours = ts.dt.hour.to_numpy()
    start, end = cost_model.asian_session_start_hour_utc, cost_model.asian_session_end_hour_utc
    if start > end:
        is_asian = (hours >= start) | (hours < end)
    else:
        is_asian = (hours >= start) & (hours < end)
    df["is_asian"] = is_asian

    return df


# ---------------------------------------------------------------------------
# Regime labels (reporting-only, see module docstring)
# ---------------------------------------------------------------------------


def compute_regime_labels(df: pd.DataFrame, cfg: OracleConfig) -> np.ndarray:
    n = len(df)
    labels = np.full(n, "UNKNOWN", dtype=object)
    if n < MIN_HMM_BARS:
        return labels

    X_full = regime_mod._build_features(df)  # (n, 9), fully vectorized/causal

    # Build the same refit-boundary schedule as replay_backtest.py's
    # _maybe_refit_hmm, but batch-apply predict_proba per block instead of
    # per bar.
    refit_n_seen = [MIN_HMM_BARS]
    seen = MIN_HMM_BARS
    while True:
        block = seen // REGIME_REFIT_BARS
        next_seen = (block + 1) * REGIME_REFIT_BARS
        if next_seen > n:
            break
        refit_n_seen.append(next_seen)
        seen = next_seen

    for k, n_seen in enumerate(refit_n_seen):
        train_start = max(0, n_seen - MIN_HMM_BARS)
        window = df.iloc[train_start:n_seen]
        try:
            fitted = regime_mod.fit_hmm(window, symbol=cfg.dukascopy_symbol)
        except Exception as e:
            log.warning("Regime fit failed for block starting at n_seen=%d (%s) — leaving UNKNOWN", n_seen, e)
            continue

        block_start = n_seen - 1  # 0-indexed bar where this fit becomes active
        block_end = refit_n_seen[k + 1] - 1 if k + 1 < len(refit_n_seen) else n  # exclusive
        burn_in_start = max(0, block_start - MIN_HMM_BARS)

        X_std = fitted.standardize(X_full[burn_in_start:block_end])
        X_std_active = X_std[:, fitted.active_mask]
        try:
            posteriors = fitted.model.predict_proba(X_std_active)
        except Exception as e:
            log.warning("predict_proba failed for block starting at n_seen=%d (%s) — leaving UNKNOWN", n_seen, e)
            continue

        own_rows_start = block_start - burn_in_start
        own_posteriors = posteriors[own_rows_start:]
        best_states = own_posteriors.argmax(axis=1)
        block_labels = np.array(
            [fitted.state_label_map.get(int(s), f"STATE_{s}") for s in best_states], dtype=object
        )
        labels[block_start:block_end] = block_labels
        log.info("Regime block @ bar_index [%d:%d): labels=%s (n=%d)",
                  block_start, block_end, dict(zip(*np.unique(block_labels, return_counts=True))),
                  block_end - block_start)

    return labels


# ---------------------------------------------------------------------------
# Trade simulation
# ---------------------------------------------------------------------------


def simulate_trades(
    df: pd.DataFrame, cost_model: CostModel, cfg: OracleConfig,
    stop_atr_mult: float = STOP_ATR_MULT, require_confirmation: bool = True,
) -> tuple[list[dict], Optional[dict]]:
    """require_confirmation=True is the original spec: touch(i) ->
    confirm(i+1) -> entry at OPEN(i+2), rail anchored to the touch bar.
    require_confirmation=False is variant (a): entry at the CLOSE of the
    touch bar itself, direction filter and rail both read from that SAME
    bar (i) — no delay, no staleness between touch and entry."""
    open_ = df["open"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    bb_upper = df["bb_upper"].to_numpy(dtype=np.float64)
    bb_lower = df["bb_lower"].to_numpy(dtype=np.float64)
    bb_mid = df["bb_mid"].to_numpy(dtype=np.float64)
    # The target level watched DURING bar j must be the middle band as of the
    # last FULLY CLOSED bar (j-1) — bb_mid[j] itself is a rolling average
    # that includes bar j's own (not-yet-known) close, so comparing bar j's
    # intrabar high/low against bb_mid[j] is self-referential lookahead (a
    # resting order can't sit at a price level that depends on where the
    # same bar closes). Caught via a nonsensical negative-gross "target"
    # trade and 62% same-bar "instant" exits in an early smoke test.
    bb_mid_target = df["bb_mid"].shift(1).to_numpy(dtype=np.float64)
    atr14 = df["atr14"].to_numpy(dtype=np.float64)
    filter_close = df["filter_close"].to_numpy(dtype=np.float64)
    filter_ema = df["filter_ema"].to_numpy(dtype=np.float64)
    is_asian = df["is_asian"].to_numpy(dtype=bool)
    spread_bps = df["spread_bps"].to_numpy(dtype=np.float64)
    ts = df["timestamps"]  # pandas Series of Timestamps

    pip = pip_size(cfg.dukascopy_symbol)
    n = len(df)

    def valid(idx: int) -> bool:
        return (
            not np.isnan(bb_lower[idx]) and not np.isnan(bb_upper[idx])
            and not np.isnan(atr14[idx]) and not np.isnan(filter_ema[idx])
        )

    trades: list[dict] = []
    censored_open: Optional[dict] = None

    def scan_exit(start_idx: int, direction: str, stop_price: float):
        for j in range(start_idx, n):
            if is_asian[j]:
                return j, open_[j], "session_close"
            if direction == "UP":
                if low[j] <= stop_price:
                    return j, stop_price, "stop"
                if not np.isnan(bb_mid_target[j]) and high[j] >= bb_mid_target[j]:
                    return j, bb_mid_target[j], "target"
            else:
                if high[j] >= stop_price:
                    return j, stop_price, "stop"
                if not np.isnan(bb_mid_target[j]) and low[j] <= bb_mid_target[j]:
                    return j, bb_mid_target[j], "target"
        return None, None, None

    def record_trade(entry_ts, entry_price, direction, stop_price, entry_spread_pips,
                      exit_idx, exit_price, exit_reason, touch_idx):
        exit_ts = ts.iloc[exit_idx]
        gross_pips = ((exit_price - entry_price) if direction == "UP" else (entry_price - exit_price)) / pip
        cost_pips = cost_model.round_trip_cost_pips(
            direction, entry_ts.to_pydatetime(), exit_ts.to_pydatetime(), entry_spread_pips)
        net_pips = gross_pips - cost_pips
        trades.append({
            "entry_ts": entry_ts, "exit_ts": exit_ts,
            "direction": direction,
            "entry_price": entry_price, "exit_price": exit_price,
            "stop_price": stop_price, "exit_reason": exit_reason,
            "entry_spread_pips": entry_spread_pips,  # kept so cost can be re-scored under a different CostModel
            "gross_pips": gross_pips, "cost_pips": cost_pips, "net_pips": net_pips,
            "entry_idx": entry_idx if isinstance(entry_idx, int) else exit_idx, "touch_idx": touch_idx,
            "fold": _fold_key(entry_ts.to_pydatetime(), "week"),
        })

    i = BB_PERIOD
    while i < n - (2 if require_confirmation else 1):
        if not valid(i) or (require_confirmation and not valid(i + 1)):
            i += 1
            continue

        touch_long = low[i] <= bb_lower[i]
        touch_short = high[i] >= bb_upper[i]
        if not touch_long and not touch_short:
            i += 1
            continue

        if require_confirmation:
            confirm_long = touch_long and (close[i + 1] > bb_lower[i + 1]) and (close[i + 1] > open_[i + 1])
            confirm_short = touch_short and (close[i + 1] < bb_upper[i + 1]) and (close[i + 1] < open_[i + 1])

            direction = None
            if confirm_long and filter_close[i + 1] > filter_ema[i + 1]:
                direction = "UP"
            elif confirm_short and filter_close[i + 1] < filter_ema[i + 1]:
                direction = "DOWN"

            if direction is None:
                i += 1
                continue

            entry_idx = i + 2
            if entry_idx >= n or is_asian[entry_idx]:
                i += 1  # entry would land outside London/NY session — skip this setup
                continue

            entry_ts = ts.iloc[entry_idx]
            entry_price = open_[entry_idx]
            rail_idx = i  # rail (touch bar) — deliberately NOT recalculated at entry
        else:
            # Variant (a): no confirmation candle. Entry at the touch bar's
            # OWN close — direction filter and rail both read from bar i.
            direction = None
            if touch_long and filter_close[i] > filter_ema[i]:
                direction = "UP"
            elif touch_short and filter_close[i] < filter_ema[i]:
                direction = "DOWN"

            if direction is None or is_asian[i]:
                i += 1
                continue

            entry_idx = i
            entry_ts = ts.iloc[i]
            entry_price = close[i]
            rail_idx = i

        # Sanity guard: if the "target" has already been reached/passed by
        # entry time, the setup is invalidated (no room left, and a resting
        # target order behind current price is meaningless) — skip it.
        # Caught via a batch of negative-gross trades labeled "target" in an
        # earlier smoke test (entry already past the middle band at entry
        # time). bb_mid_target[entry_idx] is exactly "the middle band as of
        # the last bar whose close is already known at entry time" for BOTH
        # entry modes (entry_idx-1 for the confirm case's open-of-bar entry,
        # and bar i itself for the no-confirm case's close-of-bar entry,
        # since bb_mid_target[i] = bb_mid[i-1] either way — the no-confirm
        # case's own target validity is instead checked via bb_mid[i] itself
        # below, since AT the close of bar i we already know bar i's own
        # middle band).
        target_at_entry = bb_mid[i] if not require_confirmation else bb_mid_target[entry_idx]
        if np.isnan(target_at_entry):
            i += 1
            continue
        if direction == "UP" and not (entry_price < target_at_entry):
            i += 1
            continue
        if direction == "DOWN" and not (entry_price > target_at_entry):
            i += 1
            continue

        buffer = stop_atr_mult * atr14[rail_idx]
        if direction == "UP":
            stop_price = bb_lower[rail_idx] - buffer
        else:
            stop_price = bb_upper[rail_idx] + buffer

        # Sanity guard: for the confirm case the rail is anchored to the
        # TOUCH bar, 2 bars before entry — if price has already drifted past
        # it by entry time, the stop can end up on the WRONG side of
        # entry_price (a "stop" that's actually still in profit). Caught via
        # 4/120 trades in an earlier smoke test where exit_reason="stop" had
        # positive gross. For the no-confirm case this can't happen (rail
        # and entry share the same bar) but the guard is harmless either way.
        if direction == "UP" and not (stop_price < entry_price):
            i += 1
            continue
        if direction == "DOWN" and not (stop_price > entry_price):
            i += 1
            continue

        entry_spread_bps = spread_bps[entry_idx] if not np.isnan(spread_bps[entry_idx]) else 0.0
        entry_spread_pips = entry_spread_bps / 10000.0 * entry_price / pip

        scan_start = entry_idx if require_confirmation else entry_idx + 1
        if scan_start >= n:
            censored_open = {"entry_ts": str(entry_ts), "direction": direction, "entry_price": entry_price}
            i = n
            continue
        exit_idx, exit_price, exit_reason = scan_exit(scan_start, direction, stop_price)

        if exit_idx is None:
            # Ran off the end of the data still open — censored, excluded from stats.
            censored_open = {"entry_ts": str(entry_ts), "direction": direction, "entry_price": entry_price}
            i = n
            continue

        record_trade(entry_ts, entry_price, direction, stop_price, entry_spread_pips,
                     exit_idx, exit_price, exit_reason, i)
        i = exit_idx + 1

    return trades, censored_open


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def rescore_trades(trades: list[dict], cost_model: CostModel) -> list[dict]:
    """Re-cost an already-simulated trade list under a DIFFERENT CostModel,
    without resimulating. Entry/exit points and gross P&L don't depend on
    the cost model at all (cost only enters at reporting time), so this lets
    one expensive 365-day simulation be scored under Standard/raw-confirmed/
    raw-conservative side by side instead of rerunning the strategy 3x."""
    out = []
    for t in trades:
        cost_pips = cost_model.round_trip_cost_pips(
            t["direction"], t["entry_ts"].to_pydatetime(), t["exit_ts"].to_pydatetime(),
            t["entry_spread_pips"],
        )
        new_t = dict(t)
        new_t["cost_pips"] = cost_pips
        new_t["net_pips"] = t["gross_pips"] - cost_pips
        out.append(new_t)
    return out


def _max_drawdown_pips(trades: list[dict]) -> float:
    if not trades:
        return 0.0
    ordered = sorted(trades, key=lambda t: t["exit_ts"])
    cum = np.cumsum([t["net_pips"] for t in ordered])
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    return float(dd.max())


def build_report(trades: list[dict], regime_labels: np.ndarray, cost_model: CostModel) -> dict:
    if not trades:
        return {"n_trades": 0, "note": "No trades were generated by this strategy over the cached window."}

    n = len(trades)
    wins = [t for t in trades if t["net_pips"] > 0]
    win_rate = len(wins) / n
    gross = np.array([t["gross_pips"] for t in trades])
    net = np.array([t["net_pips"] for t in trades])
    spreads_used = np.array([cost_model.effective_spread_pips(t["entry_ts"].to_pydatetime()) for t in trades])
    mean_spread = float(spreads_used.mean())
    gross_exp = float(gross.mean())
    net_exp = float(net.mean())
    multiple = gross_exp / mean_spread if mean_spread > 0 else None

    exit_reasons: dict[str, int] = {}
    for t in trades:
        exit_reasons[t["exit_reason"]] = exit_reasons.get(t["exit_reason"], 0) + 1

    by_fold: dict[str, list[dict]] = {}
    for t in trades:
        by_fold.setdefault(t["fold"], []).append(t)
    fold_report = []
    for key in sorted(by_fold):
        ft = by_fold[key]
        fnet = np.array([t["net_pips"] for t in ft])
        fold_report.append({
            "fold": key, "n_trades": len(ft),
            "win_rate": float(np.mean(fnet > 0)),
            "expectancy_net_pips_per_trade": float(fnet.mean()),
            "total_net_pips": float(fnet.sum()),
        })

    by_regime: dict[str, dict] = {}
    for t in trades:
        r = regime_labels[t["touch_idx"]] if t["touch_idx"] < len(regime_labels) else "UNKNOWN"
        by_regime.setdefault(r, []).append(t)
    regime_report = {}
    for r, rt in by_regime.items():
        rnet = np.array([t["net_pips"] for t in rt])
        regime_report[r] = {
            "n_trades": len(rt),
            "win_rate": float(np.mean(rnet > 0)),
            "expectancy_net_pips_per_trade": float(rnet.mean()),
            "total_net_pips": float(rnet.sum()),
        }

    return {
        "n_trades": n,
        "win_rate": win_rate,
        "gross_expectancy_pips_per_trade": gross_exp,
        "net_expectancy_pips_per_trade": net_exp,
        "total_net_pips": float(net.sum()),
        "max_drawdown_pips": _max_drawdown_pips(trades),
        "mean_spread_pips_used": mean_spread,
        "gross_edge_spread_multiple": multiple,
        "passes_1_5x_spread": (multiple is not None and multiple >= 1.5),
        "passes_2x_spread": (multiple is not None and multiple >= 2.0),
        "net_edge_positive": bool(net_exp > 0),
        "exit_reasons": exit_reasons,
        "folds": fold_report,
        "by_regime": regime_report,
    }


def print_report(report: dict, censored: Optional[dict]) -> None:
    print("\n--- Manual Bollinger(20,2) strategy backtest — XAUUSD M1, 90d cached ---")
    if report["n_trades"] == 0:
        print(report["note"])
        return
    print(f"n_trades={report['n_trades']}  win_rate={report['win_rate']:.2%}")
    print(f"gross expectancy/trade: {report['gross_expectancy_pips_per_trade']:+.2f} pips")
    print(f"net expectancy/trade:   {report['net_expectancy_pips_per_trade']:+.2f} pips  "
          f"(mean session-aware spread used: {report['mean_spread_pips_used']:.1f} pips, "
          f"{report['gross_edge_spread_multiple']:.2f}x)" if report['gross_edge_spread_multiple'] else "")
    print(f"total net P&L: {report['total_net_pips']:+.1f} pips")
    print(f"max drawdown:  {report['max_drawdown_pips']:.1f} pips")
    print(f"passes >=1.5x spread: {report['passes_1_5x_spread']}   "
          f"passes >=2x spread: {report['passes_2x_spread']}   "
          f"net edge positive: {report['net_edge_positive']}")
    print(f"exit reasons: {report['exit_reasons']}")
    if censored:
        print(f"NOTE: 1 trade still open at end of cached data, excluded from stats: {censored}")

    print("\nBy ISO week:")
    for f in report["folds"]:
        print(f"  {f['fold']:10s} n={f['n_trades']:4d}  win_rate={f['win_rate']:.2%}  "
              f"expectancy_net={f['expectancy_net_pips_per_trade']:+.2f} pips  "
              f"total={f['total_net_pips']:+.1f} pips")

    print("\nBy regime (at touch bar):")
    for r, st in sorted(report["by_regime"].items()):
        print(f"  {r:10s} n={st['n_trades']:4d}  win_rate={st['win_rate']:.2%}  "
              f"expectancy_net={st['expectancy_net_pips_per_trade']:+.2f} pips  "
              f"total={st['total_net_pips']:+.1f} pips")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbol", default="XAUUSD")
    p.add_argument("--days", type=int, default=90)
    p.add_argument("--out", type=Path, default=None, help="Optional path to dump trades as JSONL.")
    p.add_argument("--report-json", type=Path, default=None, help="Optional path to dump the report dict as JSON.")
    p.add_argument("--stop-atr-mult", type=float, default=STOP_ATR_MULT,
                    help=f"Stop buffer as a multiple of ATR(14) beyond the touched rail (default {STOP_ATR_MULT}).")
    p.add_argument("--no-confirmation", action="store_true",
                    help="Variant (a): enter at the touch bar's own close, skipping the confirmation-candle delay.")
    p.add_argument("--bar-timeframe", default="1min", choices=list(FILTER_RULE_FOR_BAR_TIMEFRAME),
                    help="Run touch/confirm/exit on this bar size; the EMA8 direction filter automatically "
                         "moves one level up (1min->5min, 5min->15min, 15min->60min, 60min->240min).")
    p.add_argument("--variant-label", default=None, help="Label stamped into --report-json for the sweep table.")
    args = p.parse_args()

    cfg = replace(ORACLE_CONFIG, symbol=args.symbol)
    cost_model = CostModel()

    t0 = time.monotonic()
    full_df = rb._load_full_history(cfg, args.days)

    filter_rule = FILTER_RULE_FOR_BAR_TIMEFRAME[args.bar_timeframe]
    if args.bar_timeframe != "1min":
        log.info("Resampling to %s bars (filter on %s)...", args.bar_timeframe, filter_rule)
        full_df = resample_ohlcv(full_df, args.bar_timeframe)

    log.info("Computing indicators...")
    full_df = add_indicators(full_df, cost_model, filter_rule=filter_rule)

    log.info("Computing walk-forward regime labels (reporting only)...")
    regime_labels = compute_regime_labels(full_df, cfg)

    log.info("Simulating trades (stop_atr_mult=%.2f, require_confirmation=%s, bar_timeframe=%s)...",
              args.stop_atr_mult, not args.no_confirmation, args.bar_timeframe)
    trades, censored = simulate_trades(
        full_df, cost_model, cfg,
        stop_atr_mult=args.stop_atr_mult, require_confirmation=not args.no_confirmation,
    )
    log.info("Done in %.1fs. %d trades generated.", time.monotonic() - t0, len(trades))

    report = build_report(trades, regime_labels, cost_model)
    print_report(report, censored)

    if args.out:
        with open(args.out, "w") as f:
            for t in trades:
                row = dict(t)
                row["entry_ts"] = str(row["entry_ts"])
                row["exit_ts"] = str(row["exit_ts"])
                f.write(json.dumps(row, default=str) + "\n")
        log.info("Trades written to %s", args.out)

    if args.report_json:
        payload = dict(report)
        payload["variant_label"] = args.variant_label or (
            f"stop={args.stop_atr_mult}x confirm={not args.no_confirmation} tf={args.bar_timeframe}"
        )
        payload["censored_open"] = censored
        with open(args.report_json, "w") as f:
            json.dump(payload, f, default=str, indent=2)
        log.info("Report JSON written to %s", args.report_json)


if __name__ == "__main__":
    main()
