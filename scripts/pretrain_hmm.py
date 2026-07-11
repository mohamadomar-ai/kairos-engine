#!/usr/bin/env python3
"""Pre-train the regime HMM on historical FX minute data.

Why this exists
---------------
The daemon, on a cold start, has to fit the HMM from whatever buffer it can
pull right now (~1500 bars = ~25h of trading time). That works, but the
regime labels are noisy for the first several days because:

  - 4-state Gaussian HMM convergence on 1m data is genuinely slow.
  - The state-labelling (CHOP/TREND/BREAKOUT/CASCADE) is computed from
    relative quantiles inside the training window. With only ~25h of data,
    those quantiles don't represent the full range of regimes.

Running this script ONCE during install pulls N days of GBPJPY (or whatever
`cfg.symbol` is set to) M1 bars from the Dukascopy feed (see feeds.py — same
source and cache the daemon and backtest use, no API key required), trains a
stable HMM, and writes the same _FittedHMM pickle the daemon expects at:

    ~/.trader-stack/btc-oracle/regime_hmm_pretrained.pkl

On startup, the daemon's `get_or_fit` prefers a fresh-fit cached model. If
that doesn't exist (cold install), it falls back to this pre-trained one and
treats it as a "good enough" starting point, then refits periodically from
live data.

NOTE: this replaces an earlier BTC/Binance version of this script. Per
CLAUDE.md's hard constraints: the HMM was trained on BTC and MUST be
retrained on FX before regime labels mean anything for GBPJPY (or whatever
instrument you're validating) — that's the whole point of running this.

Timing: a fresh (uncached) pull is N_days * 24 hourly Dukascopy files. Cached
hours are instant on repeat runs (see feeds.py's on-disk tick cache) — only
the first run for a given date range pays the network cost, which can take
several minutes depending on connection quality; this script logs progress
daily so it doesn't look hung.

Usage
-----
    python scripts/pretrain_hmm.py                    # default: 90 days, cfg.symbol (GBPJPY)
    python scripts/pretrain_hmm.py --days 180
    python scripts/pretrain_hmm.py --symbol EURUSD.m  # different instrument
    python scripts/pretrain_hmm.py --csv path/to/histdata_export.csv  # local CSV instead of Dukascopy
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Import the project's regime module so we use exactly the same feature
# definitions and _FittedHMM shape as the live daemon.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trader_stack.btc_oracle import regime  # noqa: E402
from trader_stack.btc_oracle import feeds  # noqa: E402
from trader_stack.btc_oracle.oracle_config import ORACLE_CONFIG  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pretrain_hmm")

MIN_ROWS_FOR_STABLE_FIT = 5000


def fetch_fx_history(cfg, days: int) -> pd.DataFrame:
    """Pull `days` of M1 bars ending now from the Dukascopy feed, one day at
    a time (so progress can be logged and a bad day doesn't sink the whole
    pull — feeds.py already treats a failed/empty hour as non-fatal, this
    just gives visibility at the day level for a long historical range).

    Weekends and holidays legitimately return empty days (FX markets are
    closed) — that's expected, not an error.
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    log.info(
        "Fetching %d days of %s M1 bars from Dukascopy (%s → %s). "
        "First run is uncached — this can take a while; subsequent runs "
        "over the same range reuse the on-disk tick cache and are fast.",
        days, cfg.dukascopy_symbol, start.date(), end.date(),
    )

    frames: list[pd.DataFrame] = []
    day_start = start.replace(hour=0, minute=0, second=0, microsecond=0)
    n_days = 0
    n_empty_days = 0

    while day_start < end:
        day_end = min(day_start + timedelta(days=1), end)
        try:
            df = feeds.fetch_ohlcv_range(cfg, day_start, day_end)
        except Exception as e:
            log.warning("  %s: fetch failed (%s) — skipping this day", day_start.date(), e)
            df = pd.DataFrame()

        if df.empty:
            n_empty_days += 1
        else:
            frames.append(df)

        n_days += 1
        if n_days % 7 == 0:
            log.info("  ...%d/%d days checked (%d bars so far, %d empty days — weekends expected)",
                     n_days, days, sum(len(f) for f in frames), n_empty_days)

        day_start = day_end

    if not frames:
        raise RuntimeError(
            f"No Dukascopy bars returned for {cfg.dukascopy_symbol} over the last {days} days "
            "— check the symbol code and network connectivity."
        )

    out = pd.concat(frames, ignore_index=True).sort_values("timestamps").reset_index(drop=True)
    log.info("Dukascopy → %d bars (%d/%d days had no data — weekends/holidays)",
              len(out), n_empty_days, n_days)
    return out


def load_csv_history(path: Path, days: int) -> pd.DataFrame:
    """Load an OHLCV CSV (e.g. a HistData.com export) instead of Dukascopy.
    Expects at minimum: Timestamp/timestamp, Open/open, High/high, Low/low,
    Close/close, Volume/volume.
    """
    log.info("Loading %s", path)
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    rename_map = {
        "timestamp": "timestamps", "time": "timestamps", "date": "timestamps",
        "open_time": "timestamps",
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    required = {"timestamps", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}. Got: {list(df.columns)}")

    if pd.api.types.is_numeric_dtype(df["timestamps"]):
        sample = float(df["timestamps"].iloc[0])
        unit = "s" if sample < 10**12 else "ms"
        df["timestamps"] = pd.to_datetime(df["timestamps"], unit=unit, utc=True)
    else:
        df["timestamps"] = pd.to_datetime(df["timestamps"], utc=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["amount"] = df["close"] * df["volume"]  # FX volume is a tick-count/notional proxy, not real dollar volume

    df = df.dropna(subset=["open", "high", "low", "close", "volume"]).reset_index(drop=True)

    cutoff = df["timestamps"].iloc[-1] - pd.Timedelta(days=days)
    df = df[df["timestamps"] >= cutoff].reset_index(drop=True)
    log.info("CSV → %d rows in the last %d days", len(df), days)
    return df


def main() -> int:
    p = argparse.ArgumentParser(description="Pre-train the regime HMM offline on FX history.")
    p.add_argument("--days", type=int, default=90,
                   help="Days of recent history to use (default 90).")
    p.add_argument("--symbol", default=ORACLE_CONFIG.symbol,
                   help=f"Broker/instrument symbol (default {ORACLE_CONFIG.symbol!r} — "
                        "the configured default validation instrument).")
    p.add_argument("--csv", type=Path, default=None,
                   help="Load from a local CSV (e.g. a HistData.com export) instead of Dukascopy.")
    p.add_argument("--out", type=Path, default=None,
                   help="Output path. Default: ~/.trader-stack/btc-oracle/regime_hmm_pretrained.pkl")
    args = p.parse_args()

    cfg = ORACLE_CONFIG
    cfg.symbol = args.symbol
    cfg.ensure_state_dir()
    out_path = args.out or (cfg.state_dir / "regime_hmm_pretrained.pkl")

    # ---- Get data --------------------------------------------------------
    if args.csv:
        df = load_csv_history(args.csv, args.days)
    else:
        df = fetch_fx_history(cfg, args.days)

    if len(df) < MIN_ROWS_FOR_STABLE_FIT:
        log.error("Got only %d rows — too few for a stable HMM fit (need >=%d). "
                  "FX markets are closed on weekends, so --days needs to span enough "
                  "trading time, not just calendar time.", len(df), MIN_ROWS_FOR_STABLE_FIT)
        return 1

    # ---- Fit HMM ---------------------------------------------------------
    log.info("Fitting 4-state Gaussian HMM on %d bars of %s…", len(df), cfg.dukascopy_symbol)
    fitted = regime.fit_hmm(df, n_states=4, symbol=cfg.dukascopy_symbol)
    log.info("State labels: %s", fitted.state_label_map)

    # ---- Save ------------------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import pickle
    with open(out_path, "wb") as f:
        pickle.dump(fitted, f)
    log.info("Saved pretrained HMM → %s (%d training samples)", out_path, fitted.n_train_samples)

    # ---- Quick sanity check: classify the last bar -----------------------
    last = regime.classify(fitted, df.tail(500))
    log.info("Sanity check — last bar classifies as: %s @ %.2f confidence",
             last.label, last.confidence)
    log.info("Done. The daemon will pick this up on next start.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
