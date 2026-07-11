"""Configuration for the FX scalping oracle (package still named `btc_oracle` —
this was originally the BTC minute-oracle daemon, repurposed for FX per
CLAUDE.md; env vars keep the BTC_ORACLE_ prefix for now rather than doing a
half-renamed migration — a pure rename is a follow-up, not a functional change).

All defaults are tuned for the documented hardware floor (16 GB RAM, no GPU).
Override anything via env var `BTC_ORACLE_*` or by mutating ORACLE_CONFIG before
calling daemon.run_forever().
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class OracleConfig:
    # ----- Market ------------------------------------------------------------
    # Broker-facing symbol. Broker is CFI (changed from Amana 2026-07-07).
    # CFI appends a trailing underscore (e.g. "XAUUSD_"), not Amana's ".m" —
    # dukascopy_symbol below strips ANY trailing non-alphabetic suffix, not
    # just a "."-delimited one, so both conventions (and plain codes with no
    # suffix) work without touching this property again.
    # Default validation instrument: XAUUSD (gold) — switched from GBPJPY
    # 2026-07-08 per direct instruction; see CLAUDE.md.
    symbol: str = os.getenv("BTC_ORACLE_SYMBOL", "XAUUSD_")

    # Data source for OHLCV + quote/spread data. Only "dukascopy" is
    # implemented today. Kept as its own field so a future live source (the
    # planned MT5 CSV bridge, per CLAUDE.md's ADD LATER section) can be
    # swapped in behind the same fetch_ohlcv()/fetch_snapshot() signatures
    # without touching any caller.
    data_source: str = os.getenv("BTC_ORACLE_DATA_SOURCE", "dukascopy")

    @property
    def dukascopy_symbol(self) -> str:
        """Broker symbol with any trailing broker-specific suffix stripped,
        upper-cased.

        Dukascopy's historical feed keys instruments by plain alphabetic
        pair/metal codes (EURUSD, GBPJPY, XAUUSD, ...). Brokers append their
        own suffix on top of that — Amana used ".m", CFI uses a trailing
        "_" — so this strips anything that isn't a leading run of letters,
        rather than assuming a specific separator character.
        """
        import re
        m = re.match(r"[A-Za-z]+", self.symbol)
        return m.group(0).upper() if m else self.symbol.upper()

    # How many recent ticks to look at when computing the live spread /
    # tick-imbalance snapshot (the FX analogue of the old CCXT order-book read).
    tick_imbalance_window: int = _env_int("BTC_ORACLE_TICK_WINDOW", 200)

    # Dukascopy HTTP behavior.
    dukascopy_timeout_seconds: int = _env_int("BTC_ORACLE_DUKASCOPY_TIMEOUT", 10)
    dukascopy_max_retries: int = _env_int("BTC_ORACLE_DUKASCOPY_RETRIES", 2)
    # Log a warning if the freshest bar we got back is older than this. Dukascopy
    # is a HISTORICAL feed with publish lag — not a live tick source. This
    # threshold makes that lag visible in the daemon log instead of silently
    # treating stale data as fresh. See feeds.py module docstring.
    dukascopy_stale_warn_minutes: float = _env_float("BTC_ORACLE_STALE_WARN_MIN", 15.0)

    # ----- Forecast horizon ------------------------------------------------
    horizon_minutes: int = _env_int("BTC_ORACLE_HORIZON", 10)
    # FLAT threshold: predicted move smaller than this (as a fraction) is "no signal".
    # 0.0015 = 0.15%; tuned for BTC's volatility — MUST be re-tuned for FX pip-scale
    # moves before this number means anything (a 0.15% GBPJPY move is ~29 pips,
    # a huge 10-minute move for a major pair). Revisit alongside volatility.py.
    flat_threshold: float = _env_float("BTC_ORACLE_FLAT_THRESHOLD", 0.0015)

    # ----- Data buffers ------------------------------------------------------
    # Number of 1m bars to maintain in the rolling buffer. 1024 covers all four
    # models' max context comfortably (~17h of FX trading time).
    buffer_bars: int = _env_int("BTC_ORACLE_BUFFER_BARS", 1024)
    # On each minute, how many of the most recent bars to refresh (handles missed minutes).
    refresh_bars: int = _env_int("BTC_ORACLE_REFRESH_BARS", 8)

    # ----- Model toggles -----------------------------------------------------
    # Disable any model to skip it. Useful if you don't want to download
    # Chronos-2 (120M) or TiRex (35M) up front.
    use_kronos: bool = _env_bool("BTC_ORACLE_USE_KRONOS", True)
    use_timesfm: bool = _env_bool("BTC_ORACLE_USE_TIMESFM", True)
    use_tirex: bool = _env_bool("BTC_ORACLE_USE_TIREX", True)
    use_chronos2: bool = _env_bool("BTC_ORACLE_USE_CHRONOS2", True)

    # ----- Model identifiers (HF Hub) ----------------------------------------
    tirex_model: str = "NX-AI/TiRex"
    chronos2_model: str = "amazon/chronos-2"

    # ----- Ensemble weights --------------------------------------------------
    # Initial uniform weights across enabled models. The signal layer will
    # also reweight by recent hit-rate if backtest history exists.
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "kronos": 1.0,
            "timesfm": 1.0,
            "tirex": 1.0,
            "chronos2": 1.0,
        }
    )
    # When true, reweight ensemble members by their realised hit-rate over the
    # last `weight_lookback` signals. Off by default until enough history exists.
    adaptive_weighting: bool = _env_bool("BTC_ORACLE_ADAPTIVE_WEIGHTING", False)
    weight_lookback: int = _env_int("BTC_ORACLE_WEIGHT_LOOKBACK", 200)

    # ----- Microstructure regime filter --------------------------------------
    # NOTE: this pair (funding_extreme_threshold / microstructure_weight) was
    # tuned for BTC perpetual-futures funding rate, which FX has no free
    # equivalent of. It's inert for FX right now (DerivativesSnapshot is
    # always None — see feeds.py) but left in place so a future carry/swap-rate
    # signal can reuse the same knob without re-plumbing signal.py.
    funding_extreme_threshold: float = _env_float("BTC_ORACLE_FUNDING_EXTREME", 0.0005)
    microstructure_weight: float = _env_float("BTC_ORACLE_MICRO_WEIGHT", 0.25)

    # ----- Phase 2: regime classification -------------------------------------
    use_regime: bool = _env_bool("BTC_ORACLE_USE_REGIME", True)
    # How often (in cycles) to refit the HMM from fresh history.
    regime_refit_interval: int = _env_int("BTC_ORACLE_REGIME_REFIT", 60 * 24)  # daily
    regime_max_age_hours: float = _env_float("BTC_ORACLE_REGIME_MAX_AGE_H", 24.0)

    # ----- Phase 2: trade filter gates ----------------------------------------
    use_trade_filter: bool = _env_bool("BTC_ORACLE_USE_FILTER", True)
    # Minimum ensemble confidence to fire a non-FLAT signal.
    min_confidence: float = _env_float("BTC_ORACLE_MIN_CONF", 0.65)
    # forecast |pct_change| must exceed noise_band × this multiplier to fire.
    noise_floor_multiplier: float = _env_float("BTC_ORACLE_NOISE_MULT", 0.75)
    # Max acceptable spread (bps) before the market is considered too thin/blown
    # out to trade. 5 bps was a BTC/USDT-spot default and hasn't been re-derived
    # for CFI yet (broker changed from Amana 2026-07-07; CFI's real spread isn't
    # confirmed — see CLAUDE.md Environment). 5 bps is a loose placeholder
    # ceiling for majors — TIGHTEN this once real CFI spread data comes back
    # from feeds.py, and once CFI's actual cost terms are known.
    spread_bps_max: float = _env_float("BTC_ORACLE_SPREAD_MAX_BPS", 5.0)
    # When True, regime=CHOP forces signal to FLAT regardless of ensemble.
    chop_blocks_signals: bool = _env_bool("BTC_ORACLE_CHOP_BLOCKS", True)

    # ----- Phase 2: adaptive weighting ----------------------------------------
    # Default refit interval in cycles. Overridden per-regime below.
    adaptive_refit_interval: int = _env_int("BTC_ORACLE_ADAPTIVE_REFIT", 60)
    # Regime-specific refit intervals — react faster in volatile regimes.
    # Each value is "every N cycles, recompute weights".
    adaptive_refit_by_regime: dict[str, int] = field(
        default_factory=lambda: {
            "CHOP":     120,   # 2h — slow drift; noise dominates short windows
            "TREND":     60,   # 1h — standard
            "BREAKOUT":  15,   # 15m — models can diverge fast
            "CASCADE":    5,   # 5m  — everything breaks; react NOW
        }
    )

    def reweight_interval_for_regime(self, regime_label: Optional[str]) -> int:
        """Return the per-cycle reweighting cadence for the current regime."""
        if not regime_label:
            return self.adaptive_refit_interval
        return self.adaptive_refit_by_regime.get(regime_label, self.adaptive_refit_interval)

    # ----- Daemon loop ---------------------------------------------------------
    # Seconds after the minute boundary before fetching. Bars finalize at ~T+1s.
    loop_offset_seconds: int = _env_int("BTC_ORACLE_LOOP_OFFSET", 3)
    # Max wall-clock budget for one forecast cycle (loop skips a tick if it exceeds).
    cycle_budget_seconds: int = _env_int("BTC_ORACLE_CYCLE_BUDGET", 50)

    # ----- Paths -----------------------------------------------------------
    state_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "BTC_ORACLE_STATE_DIR",
                str(Path.home() / ".trader-stack" / "btc-oracle"),
            )
        )
    )

    @property
    def state_file(self) -> Path:
        return self.state_dir / "state.json"

    @property
    def signals_log(self) -> Path:
        return self.state_dir / "signals.jsonl"

    @property
    def pid_file(self) -> Path:
        return self.state_dir / "oracle.pid"

    @property
    def daemon_log(self) -> Path:
        return self.state_dir / "daemon.log"

    @property
    def dukascopy_cache_dir(self) -> Path:
        """Cached hourly tick files. Complete hours never change once
        published, so they're cached forever once fetched — this matters a
        lot once the walk-forward backtest starts pulling months of history."""
        return self.state_dir / "dukascopy_cache"

    def ensure_state_dir(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)


ORACLE_CONFIG = OracleConfig()
