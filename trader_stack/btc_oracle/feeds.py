"""Data feed for the FX scalping oracle — Dukascopy historical tick data.

Why Dukascopy: free, no API key, no Wine/MT5 dependency (pure Python +
`requests` + stdlib `lzma`), and it gives real bid/ask ticks — which is
what makes a genuine spread/imbalance signal possible instead of faking one.

HONEST LIMITATION — read before trusting this for anything live:
    Dukascopy is a HISTORICAL data feed. Hourly tick files are published
    with a lag (typically minutes, sometimes longer) and the current/most
    recent hour is often not available yet. That's fine for the walk-forward
    backtest (step 3), which only ever asks for historical windows. It is
    NOT a low-latency live feed — at 1-cycle/minute cadence the daemon's
    "latest" bar can be stale by anywhere from a few minutes to over an hour.
    `_warn_if_stale()` below surfaces this in the log rather than hiding it.
    A real live source (the planned MT5 CSV bridge, see CLAUDE.md's ADD LATER
    section) is the eventual fix; this feed's whole job right now is to make
    walk-forward backtesting on real FX data possible.

Tick decode format: reverse-engineered from public open-source Dukascopy
downloaders (e.g. the `duka` and `dukascopy-node` projects), NOT from
official Dukascopy documentation — I'm confident (~high) in the byte layout
and month-is-zero-indexed URL quirk since it's a long-stable, widely
replicated format, less confident in the per-instrument price divisor for
anything outside majors/JPY-crosses. Sanity-check the first fetch for your
instrument against a broker chart before trusting it (see `sanity_check()`
at the bottom of this module).

Everything downstream (signal.py, regime.py, filters.py) already treats the
order-book and derivatives snapshots as Optional and degrades gracefully —
this feed leans on that: `derivs` is always None for FX (no free equivalent
of BTC futures funding/OI/long-short data at retail level; the broker's
overnight swap/rollover rate is a COST, handled in the backtest cost model,
not a forecasting signal).
"""

from __future__ import annotations

import logging
import lzma
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from .oracle_config import OracleConfig

log = logging.getLogger(__name__)

DUKASCOPY_BASE = "https://datafeed.dukascopy.com/datafeed"

# Price divisor ("point value") per instrument. Dukascopy ticks are encoded as
# integers; divide by this to get the actual price. JPY-quoted pairs are
# 3-decimal (divisor 1000); the rest of our tradable set (majors) is 5-decimal
# (divisor 100000). Verified against GBPJPY empirically (real ticks decode to
# plausible ~216 GBPJPY prices with a ~1.4 pip spread).
#
# XAUUSD verified empirically too (2026-07-08): fetched a real Dukascopy
# XAUUSD hour, decoded at divisor=1000 gives ~$4125 prices (divisor=100 or 10
# give absurd 5-6 digit "prices" — clearly wrong). Side finding worth knowing:
# ~99.98% of raw ticks in that sample end in the digit 5 when divided by
# 1000 — the underlying Dukascopy/interbank tick size is actually $0.005, one
# order of magnitude finer than the "point" CFI quotes retail spread in
# (see backtest.py's CostModel docstring for the point-size distinction).
_POINT_DIVISOR_OVERRIDES: dict[str, float] = {
    "XAUUSD": 1000.0,
}


def _point_divisor(dukascopy_symbol: str) -> float:
    if dukascopy_symbol in _POINT_DIVISOR_OVERRIDES:
        return _POINT_DIVISOR_OVERRIDES[dukascopy_symbol]
    return 1000.0 if dukascopy_symbol.endswith("JPY") else 100000.0


# CORRECTED 2026-07-09: the generic "point = 10 raw ticks" heuristic
# (10/_point_divisor) is right for FX pairs but was WRONG for XAUUSD.
# Live MT5 evidence: bid 4123.305 / ask 4123.435 = $0.130 spread == 130 raw
# ticks (at $0.001/tick, _point_divisor=1000) — i.e. for gold, 1 broker
# "point" == 1 raw tick, NOT 10. The previous 10/1000=0.01 pip_size made
# "150 points spread" cost $1.50, a 10x overstatement of the ~$0.13-0.15
# this codebase's own prose (and CFI's quoted 130-150 point figure) always
# meant. Confirmed this was a real bug, not just a rounding choice: gross
# P&L (computed as $diff / pip_size) and CostModel's spread_pips/
# commission_pips_round_trip constants (150, 15, 24, 35...) were on
# MISMATCHED scales — the constants were calibrated assuming point=$0.001,
# but pip_size() returned 0.01, so every cost figure was applied 10x too
# heavy relative to gross P&L. Every prior backtest/report in this project
# used the buggy 0.01 value.
_PIP_SIZE_OVERRIDES: dict[str, float] = {
    "XAUUSD": 0.001,
}


def pip_size(dukascopy_symbol: str) -> float:
    """Broker-quoted "point" size for an instrument — the unit backtest.py's
    CostModel counts spread/slippage/swap in. NOT the same thing as the raw
    Dukascopy tick granularity in general (see _POINT_DIVISOR_OVERRIDES's
    XAUUSD note), though for XAUUSD specifically they now coincide (both
    $0.001) — see _PIP_SIZE_OVERRIDES above for why gold needed an explicit
    override instead of the generic 10/_point_divisor heuristic.
        - FX majors:  divisor 100000 -> 0.0001  (standard pip)
        - JPY crosses: divisor 1000  -> 0.01    (standard JPY pip)
        - XAUUSD:      override -> 0.001 (== 1 raw tick, CONFIRMED against
          a live MT5 quote, see _PIP_SIZE_OVERRIDES docstring above).
    "pip" is FX jargon and gold traders would say "point", not "pip" — this
    function (and CostModel's *_pips_* field names) use "pip" as a generic
    stand-in for "the broker's smallest quoted cost unit" across instrument
    types, not literally an FX pip. Flagging the naming rather than silently
    leaving it ambiguous.
    """
    if dukascopy_symbol in _PIP_SIZE_OVERRIDES:
        return _PIP_SIZE_OVERRIDES[dukascopy_symbol]
    return 10.0 / _point_divisor(dukascopy_symbol)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OrderBookSnapshot:
    """Tick-derived quote proxy — FX retail has no free public L2 book.

    `spread_bps` is a REAL measured bid/ask spread from the latest tick.
    `imbalance` is an uptick/downtick ratio over the last
    `cfg.tick_imbalance_window` ticks, not real order-flow depth; it's the
    closest free proxy for the microstructure "which side is aggressing"
    read the old CCXT order-book imbalance gave for BTC.
    `bid_depth`/`ask_depth` are repurposed as up-tick/down-tick COUNTS (not
    size) — kept under the old field names because nothing downstream reads
    them by name, only `.imbalance` and `.spread_bps`.
    """

    timestamp: datetime
    mid_price: float
    bid_depth: float          # up-tick count over the window (proxy, not size)
    ask_depth: float          # down-tick count over the window (proxy, not size)
    imbalance: float          # (up - down) / (up + down), range [-1, +1]
    spread_bps: float         # (ask - bid) / mid * 10000, from the latest tick


@dataclass(frozen=True)
class DerivativesSnapshot:
    """Kept only so `Optional[DerivativesSnapshot]` type hints elsewhere
    still resolve at import time. `fetch_snapshot()` always returns
    `derivs=None` for FX — there is no free public equivalent of BTC
    perpetual-futures funding rate / open interest / long-short ratio at the
    retail level. The broker's overnight swap/rollover rate is a real
    analogue in spirit, but it's a COST, not a forecasting signal — it
    belongs in the Phase 3 backtest cost model, not here.
    """

    timestamp: datetime
    funding_rate: float
    open_interest_btc: float
    open_interest_delta_5m: float
    long_short_ratio: float
    top_trader_ls_ratio: float
    taker_buy_sell_ratio: float


@dataclass
class MarketSnapshot:
    """Everything captured in one cycle — passed to the forecasters."""

    timestamp: datetime
    ohlcv: pd.DataFrame                 # rolling buffer with timestamps + OHLCV + amount
    last_close: float
    book: Optional[OrderBookSnapshot]
    derivs: Optional[DerivativesSnapshot]


# ---------------------------------------------------------------------------
# Dukascopy tick download + decode
# ---------------------------------------------------------------------------

# Big-endian: time offset (ms into the hour), ask*divisor, bid*divisor,
# ask volume, bid volume. 20 bytes/tick.
_TICK_STRUCT = struct.Struct(">iiiff")

_TICK_COLUMNS = ["timestamps", "bid", "ask"]
_EMPTY_TICKS = pd.DataFrame(columns=_TICK_COLUMNS)

_BAR_COLUMNS = [
    "timestamps", "open", "high", "low", "close",
    "volume", "amount", "taker_buy_ratio", "spread_bps",
]
_EMPTY_BARS = pd.DataFrame(columns=_BAR_COLUMNS)


def _dukascopy_url(dukascopy_symbol: str, hour_start: datetime) -> str:
    # Dukascopy's month segment is ZERO-INDEXED (Jan="00" ... Dec="11").
    return (
        f"{DUKASCOPY_BASE}/{dukascopy_symbol}/{hour_start.year:04d}/"
        f"{hour_start.month - 1:02d}/{hour_start.day:02d}/{hour_start.hour:02d}h_ticks.bi5"
    )


def _decode_bi5(raw: bytes, hour_start: datetime, divisor: float) -> pd.DataFrame:
    """Decompress + unpack one hour's bi5 tick file into a bid/ask DataFrame."""
    if not raw:
        return _EMPTY_TICKS.copy()
    try:
        decompressed = lzma.decompress(raw)
    except lzma.LZMAError:
        # Some closed-market hours come back as a non-LZMA stub rather than 404.
        return _EMPTY_TICKS.copy()

    n = len(decompressed) // _TICK_STRUCT.size
    if n == 0:
        return _EMPTY_TICKS.copy()

    rows = [_TICK_STRUCT.unpack_from(decompressed, i * _TICK_STRUCT.size) for i in range(n)]
    arr = np.array(rows, dtype=np.float64)  # columns: t_ms, ask_int, bid_int, ask_vol, bid_vol

    ts = pd.to_datetime(hour_start, utc=True) + pd.to_timedelta(arr[:, 0], unit="ms")
    return pd.DataFrame({
        "timestamps": ts,
        "ask": arr[:, 1] / divisor,
        "bid": arr[:, 2] / divisor,
    })


def _cache_path(cfg: OracleConfig, dukascopy_symbol: str, hour_start: datetime) -> Path:
    return (
        cfg.dukascopy_cache_dir / dukascopy_symbol
        / f"{hour_start:%Y}" / f"{hour_start:%m}" / f"{hour_start:%d}"
        / f"{hour_start:%H}.csv.gz"
    )


def _fetch_hour_ticks(cfg: OracleConfig, hour_start: datetime) -> pd.DataFrame:
    """Fetch (or load from cache) one hour of ticks.

    Returns an empty frame — never raises — for hours with no data (market
    closed, not yet published, or a persistent network failure). Callers
    decide whether "nothing at all came back" across the whole request is
    fatal; a single bad hour inside a larger window is not.
    """
    symbol = cfg.dukascopy_symbol
    cache_file = _cache_path(cfg, symbol, hour_start)
    if cache_file.exists():
        try:
            cached = pd.read_csv(cache_file)
            # NOTE: read_csv's parse_dates is unreliable for tz-aware timestamps
            # on larger gzip'd files — it silently leaves the column as strings
            # instead of datetime64 for some file sizes (reproduced empirically;
            # not just a theoretical worry). Converting explicitly here avoids a
            # str-vs-Timestamp comparison crash later in _fetch_ticks_since's
            # sort_values when this frame gets concatenated with a freshly
            # decoded (properly-typed) one. format="ISO8601" (not the default
            # inferred format) because ticks landing exactly on a whole second
            # serialize without a fractional-second component (datetime.isoformat()
            # drops trailing .000000), giving genuinely mixed-precision ISO8601
            # strings within the same column — the default fixed-format inference
            # chokes on that mix.
            cached["timestamps"] = pd.to_datetime(cached["timestamps"], utc=True, format="ISO8601")
            return cached
        except Exception as e:
            log.warning("Corrupt tick cache %s (%s); refetching.", cache_file, e)

    url = _dukascopy_url(symbol, hour_start)
    raw = b""
    last_err: Optional[Exception] = None
    for attempt in range(cfg.dukascopy_max_retries + 1):
        try:
            r = requests.get(url, timeout=cfg.dukascopy_timeout_seconds)
            if r.status_code == 404:
                raw, last_err = b"", None
                break
            r.raise_for_status()
            raw, last_err = r.content, None
            break
        except Exception as e:
            last_err = e
            if attempt < cfg.dukascopy_max_retries:
                time.sleep(0.5 * (attempt + 1))

    if last_err is not None:
        log.warning("Dukascopy fetch failed for %s %s after retries: %s", symbol, hour_start, last_err)
        return _EMPTY_TICKS.copy()

    df = _decode_bi5(raw, hour_start, _point_divisor(symbol))

    # Cache complete hours only (never cache the still-forming current hour —
    # it would freeze a partial read as if it were final). Cache hits AND
    # misses (weekends, holidays) alike, so a long backtest doesn't re-request
    # the same known-empty weekend hours every run.
    now = datetime.now(timezone.utc)
    if hour_start + timedelta(hours=1) <= now:
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache_file, index=False)
        except Exception as e:
            log.debug("Tick cache write failed (%s); continuing uncached.", e)

    return df


def _fetch_ticks_since(cfg: OracleConfig, since: datetime, until: Optional[datetime] = None) -> pd.DataFrame:
    """Fetch all ticks in [since, until), one hourly file at a time, in parallel.

    Missing/holiday/weekend hours contribute zero rows and are not an error —
    only a totally empty result across the whole window is treated as fatal
    by the callers below.
    """
    until = until or datetime.now(timezone.utc)
    hour = since.replace(minute=0, second=0, microsecond=0)
    hours = []
    while hour < until:
        hours.append(hour)
        hour += timedelta(hours=1)
    if not hours:
        return _EMPTY_TICKS.copy()

    with ThreadPoolExecutor(max_workers=min(8, len(hours))) as pool:
        results = list(pool.map(lambda h: _fetch_hour_ticks(cfg, h), hours))

    frames = [df for df in results if not df.empty]
    if not frames:
        return _EMPTY_TICKS.copy()

    all_ticks = pd.concat(frames, ignore_index=True).sort_values("timestamps")
    mask = (all_ticks["timestamps"] >= since) & (all_ticks["timestamps"] < until)
    return all_ticks.loc[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Tick -> M1 OHLCV resample
# ---------------------------------------------------------------------------


def _ticks_to_m1(ticks: pd.DataFrame) -> pd.DataFrame:
    """Resample raw bid/ask ticks into 1-minute OHLC bars.

    OHLC is built from the tick MID price. Bar timestamp = bar OPEN time
    (matches the old Binance-kline convention the rest of the pipeline
    assumes — see signal.py's `_kronos_close_forecast`, which does
    `last_ts + 1min` to get the next bar's timestamp).

    `volume` = TICK COUNT per bar. FX retail feeds have no real traded
    volume; this is a liquidity/activity proxy only — per CLAUDE.md, handle
    it explicitly rather than pretending it's notional volume.
    `amount` = close * volume, which for FX is therefore a proxy-of-a-proxy:
    useful only as a relative liquidity signal, never as a real dollar-volume
    figure. Kept because Kronos's expected schema wants an `amount` column.
    `spread_bps` = the bar's mean REAL bid/ask spread — genuine cost data,
    kept for the walk-forward backtest's cost model (step 3) to consume.
    `taker_buy_ratio` = UPTICK RATIO (fraction of ticks where mid price rose
    vs. the previous tick), in [0, 1]. This is the closest free FX analogue
    of Binance's per-bar taker-buy-ratio and feeds the same downstream
    consumer in signal.py with the same semantics (0.5 = balanced).
    """
    if ticks.empty:
        return _EMPTY_BARS.copy()

    df = ticks.copy()
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df["spread_bps"] = (df["ask"] - df["bid"]) / df["mid"] * 10000.0
    df["uptick"] = df["mid"].diff().gt(0).astype(float)
    df.loc[df.index[:1], "uptick"] = np.nan  # first tick has no prior tick to compare

    g = df.set_index("timestamps").resample("1min", label="left", closed="left")
    bars = g["mid"].ohlc()
    bars["volume"] = g["mid"].count()
    bars["spread_bps"] = g["spread_bps"].mean()
    bars["taker_buy_ratio"] = g["uptick"].mean().fillna(0.5)

    bars = bars.dropna(subset=["open"])  # minutes with zero ticks (market closed)
    if bars.empty:
        return _EMPTY_BARS.copy()

    bars["amount"] = bars["close"] * bars["volume"]
    bars = bars.reset_index()
    return bars[_BAR_COLUMNS]


def _quote_from_ticks(ticks: pd.DataFrame, cfg: OracleConfig) -> Optional[OrderBookSnapshot]:
    """Derive a spread/imbalance snapshot from the tail of a tick set."""
    if ticks.empty or len(ticks) < 2:
        return None

    window = ticks.tail(cfg.tick_imbalance_window)
    last = ticks.iloc[-1]
    mid = float((last["bid"] + last["ask"]) / 2.0)
    spread_bps = float((last["ask"] - last["bid"]) / mid * 10000.0) if mid > 0 else 0.0

    window_mid = (window["bid"] + window["ask"]) / 2.0
    deltas = window_mid.diff().dropna()
    up = int((deltas > 0).sum())
    down = int((deltas < 0).sum())
    total = up + down
    imbalance = (up - down) / total if total > 0 else 0.0

    return OrderBookSnapshot(
        timestamp=datetime.now(timezone.utc),
        mid_price=mid,
        bid_depth=float(up),
        ask_depth=float(down),
        imbalance=float(imbalance),
        spread_bps=spread_bps,
    )


def _warn_if_stale(cfg: OracleConfig, bars: pd.DataFrame, now: datetime) -> None:
    if bars.empty:
        return
    last_ts = bars["timestamps"].iloc[-1]
    if last_ts.tzinfo is None:
        last_ts = last_ts.tz_localize("UTC")
    gap_minutes = (now - last_ts).total_seconds() / 60.0
    if gap_minutes > cfg.dukascopy_stale_warn_minutes:
        log.warning(
            "Dukascopy data for %s is %.0f min stale (last bar %s UTC). Expected "
            "for a historical feed (publish lag, or FX market closed) — this is "
            "NOT a live tick source. See feeds.py module docstring.",
            cfg.dukascopy_symbol, gap_minutes, last_ts,
        )


# ---------------------------------------------------------------------------
# Public API — same signatures the rest of the pipeline already calls
# ---------------------------------------------------------------------------


def fetch_ohlcv(cfg: OracleConfig, limit: int) -> pd.DataFrame:
    """Return the most recent `limit` 1-minute bars ending ~now.

    "~now" because Dukascopy publish lag means the freshest bars may be
    stale — see module docstring. If fewer than `limit` bars come back
    (pair just reopened after the weekend, thin history), you get whatever's
    available; callers already handle short buffers (this matched the old
    CCXT behavior for illiquid pairs).
    """
    now = datetime.now(timezone.utc)
    hours_needed = int(np.ceil(limit / 60.0)) + 2  # safety margin: gaps + publish lag
    since = now - timedelta(hours=hours_needed)

    ticks = _fetch_ticks_since(cfg, since, now)
    bars = _ticks_to_m1(ticks)
    if bars.empty:
        raise RuntimeError(
            f"No Dukascopy ticks for {cfg.dukascopy_symbol} in the last {hours_needed}h "
            "— check the symbol code and market hours."
        )
    _warn_if_stale(cfg, bars, now)
    return bars.tail(limit).reset_index(drop=True)


def fetch_ohlcv_range(cfg: OracleConfig, start: datetime, end: datetime) -> pd.DataFrame:
    """Return M1 bars covering an arbitrary historical [start, end) window.

    Distinct from fetch_ohlcv() (which is always "most recent N bars ending
    now"). Used by the outcome resolver, and will be the backbone of the
    Phase 3 walk-forward backtest harness.
    """
    ticks = _fetch_ticks_since(cfg, start - timedelta(minutes=2), end + timedelta(minutes=2))
    bars = _ticks_to_m1(ticks)
    mask = (bars["timestamps"] >= start) & (bars["timestamps"] < end)
    return bars.loc[mask].reset_index(drop=True)


def fetch_snapshot(cfg: OracleConfig, ohlcv_limit: Optional[int] = None) -> MarketSnapshot:
    """Fetch the rolling OHLCV buffer + a tick-derived quote snapshot.

    Both come from the SAME tick pull (one Dukascopy round-trip, not two) —
    the quote snapshot looks at the tail of the same tick set used to build
    the OHLCV bars. `derivs` is always None — see DerivativesSnapshot's
    docstring for why.
    """
    limit = ohlcv_limit if ohlcv_limit is not None else cfg.buffer_bars
    now = datetime.now(timezone.utc)
    hours_needed = int(np.ceil(limit / 60.0)) + 2
    since = now - timedelta(hours=hours_needed)

    ticks = _fetch_ticks_since(cfg, since, now)
    if ticks.empty:
        raise RuntimeError(
            f"No Dukascopy ticks for {cfg.dukascopy_symbol} in the last {hours_needed}h."
        )

    bars = _ticks_to_m1(ticks).tail(limit).reset_index(drop=True)
    if bars.empty:
        raise RuntimeError(
            f"Ticks came back for {cfg.dukascopy_symbol} but none resampled into a "
            "full M1 bar — check for a data gap."
        )
    _warn_if_stale(cfg, bars, now)

    book = _quote_from_ticks(ticks, cfg)

    return MarketSnapshot(
        timestamp=now,
        ohlcv=bars,
        last_close=float(bars["close"].iloc[-1]),
        book=book,
        derivs=None,
    )


# ---------------------------------------------------------------------------
# Manual sanity check — not called by the pipeline, run by hand once per
# instrument before trusting this feed for anything.
# ---------------------------------------------------------------------------


def sanity_check(cfg: OracleConfig, hours: int = 3) -> None:
    """Print recent bars + spread stats so you can eyeball them against a
    broker chart. Run with:

        python -m trader_stack.btc_oracle.feeds

    which fetches 3h of bars for the default configured symbol. Pass a
    different `cfg` (e.g. with `cfg.symbol` changed) to check another pair.
    """
    df = fetch_ohlcv(cfg, limit=hours * 60)
    print(f"Symbol: {cfg.symbol} -> dukascopy={cfg.dukascopy_symbol}")
    print(f"Bars fetched: {len(df)}")
    print(df.tail(10).to_string(index=False))
    print(f"Mean spread (bps): {df['spread_bps'].mean():.2f}")
    print(f"Mean tick count/bar (volume proxy): {df['volume'].mean():.1f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from .oracle_config import ORACLE_CONFIG
    sanity_check(ORACLE_CONFIG)
