"""Walk-forward backtest of the oracle's logged signals, net of full FX costs.

WHAT THIS DOES:
    Reads ~/.trader-stack/btc-oracle/signals.jsonl (every signal the daemon —
    or a one-shot `oracle.py forecast` — has actually produced), fetches the
    realized price at signal-time + horizon from the Dukascopy feed (see
    feeds.py), and computes accuracy + P&L NET of a real cost model (spread +
    slippage + commission + swap), split into sequential time folds so you
    can see whether any edge is consistent over time or a one-fold fluke.

WHY THIS IS "WALK-FORWARD" AND WHAT IT ISN'T:
    Every signal in signals.jsonl was generated live (or one-shot) using only
    data available at that moment — there is no lookahead by construction.
    That makes this genuinely walk-forward.
    What it is NOT: a deep multi-year historical replay. Re-running all four
    foundation models (Kronos/TimesFM/TiRex/Chronos-2) bar-by-bar over months
    of M1 FX data is computationally infeasible on this hardware (no GPU,
    fp32-only per CLAUDE.md's hard constraint) and is not attempted here.
    Results are only as trustworthy as however much real signal history has
    accumulated — see `small_sample_warning` in the result.

COST MODEL — READ BEFORE TRUSTING THE GATE:
    "pip" here is used generically as "the broker's smallest quoted cost
    unit" across instrument types (see feeds.py's pip_size() docstring) —
    for FX that's a real pip, for XAUUSD it's CFI's "point" (feeds.py's
    pip_size("XAUUSD") = 0.01, matching a standard 2-decimal gold quote).

    spread_pips default (150) is CFI Standard's measured London/NY-session
    average (~130 points) with a small margin added. This is NOT the number
    used for every trade: CFI's own measurements put overnight/Asian-session
    spread at ~270-450 points, 2-3x the London/NY number (2026-07-08), so
    round_trip_cost_pips() automatically applies
    CostModel.asian_session_spread_multiplier (default 3.0x) to any trade
    whose entry hour falls in the Asian window (default 22:00-07:00 UTC,
    approximate) — see CostModel.SESSION_NOTE for the exact mechanics.
    Session boundaries and the multiplier are ASSUMED, not CFI-confirmed
    cutover times. commission defaults to 0.0, carried over from the earlier
    FX/CFI confirmation ("no commission") — NOT separately confirmed for
    gold; flag if that's wrong.
    slippage_pips_per_side and both swap rates are PLACEHOLDERS — not
    CFI-confirmed for ANY instrument, and probably worse placeholders for
    gold specifically (gold's tick-to-tick moves and session-dependent
    liquidity differ a lot from FX majors — the FX-derived 0.2 slippage
    guess has even less basis here than it did for GBPJPY).
    Swap defaulting to 0 UNDERSTATES cost for any position held across a
    rollover; fine for pure intraday scalps that close same-session (the
    common case at a 10-min horizon), NOT fine if you start holding
    overnight. Override CostModel's fields once you have real numbers.
    The measured historical spread from feeds.py's Dukascopy data is used as
    a floor alongside the quoted CFI spread — Dukascopy ticks tend to reflect
    near-interbank pricing, which can be tighter than a retail quote, so
    flooring at the broker's own number avoids an optimistic bias.

THE GATE (per CLAUDE.md: "edge must survive ~1.5-2x spread or discard"):
    Interpreted as a safety-margin check: gross (pre-cost) expectancy per
    trade should be at least 1.5-2x the round-trip spread, so there's real
    margin once full costs are paid — not just barely net-positive. Both the
    gross multiple AND the actual net-of-full-cost expectancy are reported;
    see GateResult.note for exactly how to read them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from .feeds import fetch_ohlcv_range, pip_size
from .oracle_config import ORACLE_CONFIG, OracleConfig
from .state import read_recent_signals

log = logging.getLogger(__name__)

SCOPE_NOTE = (
    "Replays signals ACTUALLY logged to signals.jsonl (live or one-shot) "
    "against realized Dukascopy prices — walk-forward by construction (no "
    "lookahead), but NOT a deep multi-year historical replay: re-running the "
    "full forecaster ensemble bar-by-bar over months of history is not "
    "attempted (infeasible on this no-GPU hardware). Trust scales with how "
    "much real signal history has accumulated — see small_sample_warning."
)

MIN_DIRECTIONAL_FOR_CONFIDENCE = 100  # below this, win-rate/expectancy is noise, not signal


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------


@dataclass
class CostModel:
    """Round-trip cost assumptions, in "pips" (see feeds.py's pip_size() —
    generically "the broker's smallest quoted cost unit", not literally an
    FX pip for non-FX instruments), for one directional trade.

    Two CFI account types are now in play, each with its own defaults —
    build the instance for the account you're modeling rather than relying
    solely on this class's own defaults (which remain the CFI Standard
    account, unchanged):

    CFI STANDARD (this class's defaults) — ASSUMED, not directly confirmed:
        spread_pips = 150, close to CFI Standard's measured London/NY
        average (~130 points, 2026-07-08). SESSION-DEPENDENT: overnight/
        Asian-session spread measured at ~270-450 points — 2-3x this
        default. commission_per_lot_round_trip = $0 ("no commission"),
        CONFIRMED (CFI, 2026-07-07, for the FX instrument then in use).

    CFI DYNAMIC TRADER / "raw" account — CONFIRMED (broker agent, 2026-07-09)
    for XAUUSD: spread ~15 points London/NY average, commission $9/lot
    round-turn = 9 points (already converted to points by the confirming
    source — this class does not do a $/lot -> points conversion itself).
    Round trip = 15 + 9 = 24 points. Build via:
        CostModel(spread_pips=15.0, commission_pips_round_trip=9.0, slippage_pips_per_side=0.0)
    A conservative padded variant (spread widening + slippage buffer, NOT
    itself CFI-confirmed) lands at 35 points total, e.g.:
        CostModel(spread_pips=26.0, commission_pips_round_trip=9.0, slippage_pips_per_side=0.0)
    The Asian-session multiplier (see below) still applies to spread_pips
    for this account type too — it was never separately reconfirmed for
    Dynamic Trader, so treat the resulting Asian-hour number (45 points for
    the confirmed variant) as an assumption layered on a confirmed base, not
    itself confirmed.

    round_trip_cost_pips() applies asian_session_spread_multiplier to
    spread_pips automatically for any trade whose entry falls in the Asian
    window (see session_note below) — this is NOT optional/off by default.
    commission_pips_round_trip is flat and NOT session-scaled (a broker
    commission doesn't widen overnight the way a market-maker's spread
    does).

    PLACEHOLDERS — NOT CFI-confirmed for any instrument, override before
    trusting the gate:
        slippage_pips_per_side, swap_long_pips_per_day, swap_short_pips_per_day,
        asian_session_start_hour_utc, asian_session_end_hour_utc (approximate
        session boundaries, not CFI-confirmed cutover times).
    """

    SESSION_NOTE = (
        "spread_pips is a London/NY-session base (~130-150 points measured). "
        "CFI's own measurements put overnight/Asian-session XAUUSD spread at "
        "~270-450 points — 2-3x the London/NY number. round_trip_cost_pips() "
        "applies asian_session_spread_multiplier to spread_pips for any trade "
        "whose entry_ts hour (assumed UTC) falls in "
        "[asian_session_start_hour_utc, asian_session_end_hour_utc) wrapping "
        "midnight (default 22:00-07:00 UTC, approximate — not a CFI-confirmed "
        "cutover). The 3.0x default lands at spread_pips * 3.0 = 450, the top "
        "of the measured Asian range, a deliberately conservative (higher-cost) "
        "choice for a live-money gate. Session boundaries and the multiplier "
        "are both ASSUMED, not exact — override with real fill data once you "
        "have it."
    )

    spread_pips: float = 150.0
    commission_per_lot_round_trip: float = 0.0
    # CONFIRMED for CFI Dynamic Trader (broker agent, 2026-07-09): $9/lot
    # round-turn on XAUUSD, pre-converted to 9 points by the confirming
    # source. 0.0 for CFI Standard (no commission). Flat, NOT session-scaled
    # — see class docstring.
    commission_pips_round_trip: float = 0.0
    # PLACEHOLDER: typical retail market-order slippage estimate for a liquid
    # instrument, NOT CFI-confirmed, and less-well-grounded for gold than it
    # was for an FX major (different liquidity/volatility profile). Applied
    # once per side (entry + exit).
    slippage_pips_per_side: float = 0.2
    # PLACEHOLDER: NOT CFI-confirmed. 0 understates cost for any position
    # held across the daily rollover (~22:00 UTC, approximated — actual NY
    # 5pm rollover shifts with DST, not modeled precisely here).
    swap_long_pips_per_day: float = 0.0
    swap_short_pips_per_day: float = 0.0
    rollover_hour_utc: int = 22
    # ASSUMED: CFI measured Asian-session XAUUSD spread at ~270-450 points vs
    # ~130 London/NY — a 2.1x-3.5x range. 3.0 is a deliberately conservative
    # pick near the top of that range (spread_pips * 3.0 = 450), so the gate
    # errs toward overstating cost for Asian-hour signals rather than
    # understating it. Session hours are UTC, wrap midnight, approximate.
    asian_session_spread_multiplier: float = 3.0
    asian_session_start_hour_utc: int = 22
    asian_session_end_hour_utc: int = 7
    # True (default, CFI Standard's existing behavior, unchanged): floor
    # spread_pips against the Dukascopy-measured historical spread, so an
    # optimistic ASSUMED number can't understate cost relative to real
    # wholesale-feed conditions. That floor was calibrated against the
    # Standard account's ASSUMED 150pt figure — for a CONFIRMED tighter
    # account (e.g. CFI Dynamic Trader's 15pt), Dukascopy's wholesale/
    # interbank tick spread is routinely WIDER than 15pts, so leaving this
    # True would silently erase the confirmed number's benefit almost every
    # trade. Set False when modeling a confirmed account-specific spread —
    # see scripts/cost_scenario_report.py.
    apply_measured_spread_floor: bool = True

    CONFIRMED_FIELDS: tuple[str, ...] = field(
        default=("commission_per_lot_round_trip",), repr=False, compare=False,
    )
    ASSUMED_FIELDS: tuple[str, ...] = field(
        default=(
            "spread_pips", "asian_session_spread_multiplier",
            "asian_session_start_hour_utc", "asian_session_end_hour_utc",
        ),
        repr=False, compare=False,
    )

    def crosses_rollover(self, entry_ts: datetime, exit_ts: datetime) -> bool:
        boundary = entry_ts.replace(
            hour=self.rollover_hour_utc, minute=0, second=0, microsecond=0
        )
        if boundary < entry_ts:
            boundary += timedelta(days=1)
        return entry_ts <= boundary < exit_ts

    def is_asian_session(self, ts: datetime) -> bool:
        """Whether `ts` (hour read as-is, codebase convention is UTC
        throughout — see feeds.py/regime.py) falls in the overnight/Asian
        spread-blowout window. Window wraps midnight (e.g. default
        22:00-07:00 UTC), so this can't be a simple start <= hour < end
        comparison.
        """
        start, end = self.asian_session_start_hour_utc, self.asian_session_end_hour_utc
        hour = ts.hour
        if start > end:  # wraps midnight, e.g. 22 -> 7
            return hour >= start or hour < end
        return start <= hour < end

    def effective_spread_pips(self, entry_ts: datetime) -> float:
        """spread_pips, bumped by asian_session_spread_multiplier if `entry_ts`
        falls in the Asian session window. This is the London/NY-vs-Asian
        session-aware base BEFORE flooring against the measured historical
        spread in round_trip_cost_pips.
        """
        if self.is_asian_session(entry_ts):
            return self.spread_pips * self.asian_session_spread_multiplier
        return self.spread_pips

    def round_trip_cost_pips(
        self,
        direction: str,
        entry_ts: datetime,
        exit_ts: datetime,
        measured_spread_pips: Optional[float],
    ) -> float:
        """Total round-trip cost in pips for one trade.

        commission_per_lot_round_trip ($/lot) isn't converted here — it's $0
        for CFI Standard so it's a no-op there; a real lot-size/notional
        model would be needed to convert a nonzero $ figure. Use
        commission_pips_round_trip instead for a broker fee already
        expressed in points (e.g. CFI Dynamic Trader's confirmed $9/lot ==
        9 points) — added flat, NOT session-scaled.

        Spread is session-aware: the Asian-session multiplier is applied to
        spread_pips BEFORE flooring against measured_spread_pips (the actual
        Dukascopy-observed spread at entry, when available) — the floor still
        wins if the real measured spread was even wider than our assumption,
        UNLESS apply_measured_spread_floor is False (see field docstring).
        """
        base_spread = self.effective_spread_pips(entry_ts)
        spread = max(base_spread, measured_spread_pips or 0.0) if self.apply_measured_spread_floor else base_spread
        cost = spread + 2.0 * self.slippage_pips_per_side + self.commission_pips_round_trip
        if self.crosses_rollover(entry_ts, exit_ts):
            cost += self.swap_long_pips_per_day if direction == "UP" else self.swap_short_pips_per_day
        return cost

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("CONFIRMED_FIELDS", None)
        d.pop("ASSUMED_FIELDS", None)
        d["confirmed_fields"] = list(self.CONFIRMED_FIELDS)
        d["assumed_fields"] = list(self.ASSUMED_FIELDS)  # real range given, not a single confirmed number
        d["placeholder_fields"] = [
            "slippage_pips_per_side", "swap_long_pips_per_day", "swap_short_pips_per_day",
        ]
        d["session_note"] = self.SESSION_NOTE
        return d


# ---------------------------------------------------------------------------
# Signal <-> realized-outcome pairing (shared with sweep.py)
# ---------------------------------------------------------------------------


@dataclass
class PairedSignal:
    ts: datetime
    horizon_minutes: int
    last_close: float
    pred_direction: str            # post-filter direction (what would actually have traded live)
    pred_confidence: float
    raw_direction: str             # pre-filter direction (for threshold re-sweeping)
    raw_confidence: float
    regime_label: str
    per_model: dict
    actual_direction: str
    actual_pct: float
    entry_spread_pips: Optional[float]   # measured historical spread at signal time


def _classify(pct: float, threshold: float) -> str:
    if pct >= threshold:
        return "UP"
    if pct <= -threshold:
        return "DOWN"
    return "FLAT"


def _parse_signal_ts(sig: dict) -> Optional[datetime]:
    try:
        ts_str = sig["timestamp"]
        if ts_str.endswith("Z"):
            ts_str = ts_str[:-1] + "+00:00"
        ts = datetime.fromisoformat(ts_str)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts
    except (KeyError, ValueError, TypeError):
        return None


def pair_signals_with_outcomes(
    cfg: OracleConfig,
    signals: list[dict],
    price_cache: Optional[pd.DataFrame] = None,
) -> tuple[list[PairedSignal], Optional[pd.DataFrame]]:
    """Attach the realized outcome + measured historical spread to each signal.

    Fetches ONE Dukascopy window covering the whole batch (via
    fetch_ohlcv_range) instead of one call per signal, and threads the cache
    through so repeated calls (e.g. sweep.py evaluating multiple thresholds
    over the same signal batch) don't re-fetch.
    """
    if not signals:
        return [], price_cache

    parsed = []
    for sig in signals:
        ts = _parse_signal_ts(sig)
        if ts is None:
            continue
        try:
            horizon = int(sig.get("horizon_minutes", cfg.horizon_minutes))
            last_close = float(sig["last_close"])
        except (KeyError, ValueError, TypeError):
            continue
        parsed.append((sig, ts, horizon, last_close))

    if not parsed:
        return [], price_cache

    earliest = min(ts for _, ts, _, _ in parsed)
    latest_future = max(ts + timedelta(minutes=h) for _, ts, h, _ in parsed)

    need_refresh = (
        price_cache is None or price_cache.empty
        or price_cache["timestamps"].iloc[0] > earliest
        or price_cache["timestamps"].iloc[-1] < latest_future
    )
    if need_refresh:
        try:
            price_cache = fetch_ohlcv_range(
                cfg, earliest - timedelta(minutes=2), latest_future + timedelta(minutes=2),
            )
        except Exception as e:
            log.warning("Could not fetch historical price window for backtest: %s", e)
            return [], price_cache

    if price_cache is None or price_cache.empty:
        return [], price_cache

    px_ts = price_cache["timestamps"]
    px_close = price_cache["close"].to_numpy()
    px_spread = price_cache["spread_bps"].to_numpy() if "spread_bps" in price_cache.columns else None
    px_start, px_end = px_ts.iloc[0], px_ts.iloc[-1]
    pip = pip_size(cfg.dukascopy_symbol)

    out: list[PairedSignal] = []
    for sig, ts, horizon, last_close in parsed:
        future_ts = ts + timedelta(minutes=horizon)
        if not (px_start <= future_ts <= px_end):
            continue
        idx = min(int(px_ts.searchsorted(future_ts)), len(price_cache) - 1)
        future_price = float(px_close[idx])
        actual_pct = future_price / last_close - 1.0
        actual = _classify(actual_pct, cfg.flat_threshold)

        entry_spread_pips = None
        if px_spread is not None and px_start <= ts <= px_end:
            entry_idx = min(int(px_ts.searchsorted(ts)), len(price_cache) - 1)
            spread_bps_at_entry = float(px_spread[entry_idx])
            entry_spread_pips = spread_bps_at_entry / 10000.0 * last_close / pip

        out.append(PairedSignal(
            ts=ts,
            horizon_minutes=horizon,
            last_close=last_close,
            pred_direction=sig.get("direction", "FLAT"),
            pred_confidence=float(sig.get("confidence", 0.0)),
            raw_direction=sig.get("raw_direction", sig.get("direction", "FLAT")),
            raw_confidence=float(sig.get("raw_confidence", sig.get("confidence", 0.0))),
            regime_label=(sig.get("regime") or {}).get("label", "UNKNOWN"),
            per_model=sig.get("per_model", {}),
            actual_direction=actual,
            actual_pct=actual_pct,
            entry_spread_pips=entry_spread_pips,
        ))

    return out, price_cache


def net_trade_pips(
    cfg: OracleConfig,
    cost_model: CostModel,
    p: PairedSignal,
    direction_override: Optional[str] = None,
) -> float:
    """Net P&L in pips for one trade, direction and costs included.

    Returns 0.0 for FLAT (no trade). `direction_override` lets sweep.py ask
    "what if I'd traded the RAW direction at a different confidence
    threshold" without re-pairing signals.
    """
    direction = direction_override if direction_override is not None else p.pred_direction
    if direction not in ("UP", "DOWN"):
        return 0.0
    pip = pip_size(cfg.dukascopy_symbol)
    raw_pips = (p.actual_pct if direction == "UP" else -p.actual_pct) * p.last_close / pip
    exit_ts = p.ts + timedelta(minutes=p.horizon_minutes)
    cost_pips = cost_model.round_trip_cost_pips(direction, p.ts, exit_ts, p.entry_spread_pips)
    return raw_pips - cost_pips


def gross_trade_pips(cfg: OracleConfig, p: PairedSignal, direction_override: Optional[str] = None) -> float:
    """Pre-cost P&L in pips — used for the gate's gross-edge-vs-spread check."""
    direction = direction_override if direction_override is not None else p.pred_direction
    if direction not in ("UP", "DOWN"):
        return 0.0
    pip = pip_size(cfg.dukascopy_symbol)
    return (p.actual_pct if direction == "UP" else -p.actual_pct) * p.last_close / pip


# ---------------------------------------------------------------------------
# Walk-forward folds
# ---------------------------------------------------------------------------


def _fold_key(ts: datetime, fold: str) -> str:
    if fold == "day":
        return ts.strftime("%Y-%m-%d")
    if fold == "week":
        iso = ts.isocalendar()
        return f"{iso[0]}-W{iso[1]:02d}"
    return "all"


@dataclass
class FoldResult:
    fold: str
    n_signals: int
    n_directional: int
    accuracy_directional: Optional[float]
    pnl_pips_with_costs: float
    expectancy_pips_per_trade_net: Optional[float]

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Random-direction baseline (overfitting / small-sample sanity check)
# ---------------------------------------------------------------------------


def _random_direction_baseline(
    cfg: OracleConfig,
    cost_model: CostModel,
    directional: list[PairedSignal],
    n_trials: int = 1000,
    seed: int = 1234,
) -> dict:
    """Re-assign a random UP/DOWN direction to the SAME signals (same realized
    outcomes, same costs) `n_trials` times, and see how often that beats the
    model's actual net P&L. Answers "is this distinguishable from a coin
    flip on this exact sample" — a sanity check, not a formal hypothesis
    test, and it only ever gets more meaningful as the sample grows.
    """
    if not directional:
        return {
            "n_trials": 0, "n_directional": 0,
            "model_pnl_pips": 0.0, "random_mean_pnl_pips": 0.0,
            "pct_random_trials_beating_or_matching_model": None,
        }

    model_pnl = sum(net_trade_pips(cfg, cost_model, p) for p in directional)

    rng = np.random.default_rng(seed)
    n = len(directional)
    random_pnls = np.empty(n_trials)
    for t in range(n_trials):
        dirs = rng.choice(("UP", "DOWN"), size=n)
        random_pnls[t] = sum(
            net_trade_pips(cfg, cost_model, p, direction_override=d)
            for p, d in zip(directional, dirs)
        )
    random_pnls.sort()

    return {
        "n_trials": n_trials,
        "n_directional": n,
        "model_pnl_pips": float(model_pnl),
        "random_mean_pnl_pips": float(random_pnls.mean()),
        "random_p95_pnl_pips": float(random_pnls[int(n_trials * 0.95)]),
        "pct_random_trials_beating_or_matching_model": float((random_pnls >= model_pnl).mean()),
    }


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


@dataclass
class GateResult:
    spread_pips_used: float
    gross_expectancy_pips_per_trade: Optional[float]
    net_expectancy_pips_per_trade: Optional[float]
    gross_edge_spread_multiple: Optional[float]
    passes_1_5x_spread: Optional[bool]
    passes_2x_spread: Optional[bool]
    net_edge_positive: Optional[bool]
    note: str

    def to_dict(self) -> dict:
        return asdict(self)


_GATE_NOTE = (
    "gross_expectancy = mean pre-cost pips per directional trade. "
    "gross_edge_spread_multiple = gross_expectancy / spread_pips_used — "
    "CLAUDE.md's '1.5-2x spread' rule is applied to THIS (a safety-margin "
    "check: is the gross edge comfortably bigger than the round-trip spread, "
    "before slippage/commission/swap eat into it). net_expectancy is the "
    "actual bottom line after the FULL cost model (spread+slippage+"
    "commission+swap) — net_edge_positive just checks it's > 0, a much "
    "lower bar than the 1.5-2x gross check. Both can matter: gross multiple "
    "high but net negative means costs alone kill it even though the raw "
    "signal has real edge. spread_pips_used is the MEAN session-aware spread "
    "actually applied across these trades (CostModel.effective_spread_pips "
    "per entry_ts, London/NY base vs the Asian-session multiplier — see "
    "CostModel.SESSION_NOTE) — NOT the flat London/NY base, so a batch "
    "weighted toward Asian-hour entries will show a correspondingly higher "
    "number here and a correspondingly harder gross multiple to clear."
)


def _build_gate(cfg: OracleConfig, cost_model: CostModel, directional: list[PairedSignal]) -> GateResult:
    if not directional:
        return GateResult(
            spread_pips_used=cost_model.spread_pips,
            gross_expectancy_pips_per_trade=None,
            net_expectancy_pips_per_trade=None,
            gross_edge_spread_multiple=None,
            passes_1_5x_spread=None,
            passes_2x_spread=None,
            net_edge_positive=None,
            note=_GATE_NOTE + " No directional trades to evaluate.",
        )

    gross = float(np.mean([gross_trade_pips(cfg, p) for p in directional]))
    net = float(np.mean([net_trade_pips(cfg, cost_model, p) for p in directional]))
    mean_spread = float(np.mean([cost_model.effective_spread_pips(p.ts) for p in directional]))
    multiple = (gross / mean_spread) if mean_spread > 0 else None

    return GateResult(
        spread_pips_used=mean_spread,
        gross_expectancy_pips_per_trade=gross,
        net_expectancy_pips_per_trade=net,
        gross_edge_spread_multiple=multiple,
        passes_1_5x_spread=(multiple is not None and multiple >= 1.5),
        passes_2x_spread=(multiple is not None and multiple >= 2.0),
        net_edge_positive=(net > 0.0),
        note=_GATE_NOTE,
    )


# ---------------------------------------------------------------------------
# Main result + entry point
# ---------------------------------------------------------------------------


@dataclass
class BacktestResult:
    n_signals: int
    n_directional: int
    coverage: float
    accuracy_directional: Optional[float]
    accuracy_all: Optional[float]
    confusion: dict
    by_confidence: list
    by_regime: dict
    per_model_accuracy: dict
    cost_model: dict
    pnl_pips_no_costs: float
    pnl_pips_with_costs: float
    fold_granularity: str
    folds: list
    gate: dict
    baseline_random_direction: dict
    small_sample_warning: Optional[str]
    scope_note: str = SCOPE_NOTE

    def to_dict(self) -> dict:
        return asdict(self)


def _empty_result(cost_model: CostModel, fold: str, reason: str) -> BacktestResult:
    return BacktestResult(
        n_signals=0, n_directional=0, coverage=0.0,
        accuracy_directional=None, accuracy_all=None,
        confusion={}, by_confidence=[], by_regime={}, per_model_accuracy={},
        cost_model=cost_model.to_dict(),
        pnl_pips_no_costs=0.0, pnl_pips_with_costs=0.0,
        fold_granularity=fold, folds=[],
        gate=_build_gate(ORACLE_CONFIG, cost_model, []).to_dict(),
        baseline_random_direction=_random_direction_baseline(ORACLE_CONFIG, cost_model, []),
        small_sample_warning=reason,
    )


def run_backtest(
    cfg: OracleConfig = ORACLE_CONFIG,
    last_n: int = 1000,
    cost_model: Optional[CostModel] = None,
    fold: str = "week",
    n_baseline_trials: int = 1000,
) -> BacktestResult:
    """Replay the last `last_n` logged signals (from cfg.signals_log — the
    live daemon's JSONL) and compute accuracy + P&L net of `cost_model`,
    folded by `fold` ("day" | "week" | "all").

    Thin wrapper around run_backtest_on_signals() — see that function for
    the actual scoring logic, factored out so callers with their OWN signal
    batch + price data (e.g. scripts/replay_backtest.py's offline
    walk-forward historical replay, which generates signals in-memory rather
    than reading a live JSONL) can reuse the exact same scoring/gate/fold
    logic without going through disk or re-fetching prices they already have.
    """
    cost_model = cost_model or CostModel()
    signals = read_recent_signals(n=last_n, cfg=cfg)
    if not signals:
        return _empty_result(
            cost_model, fold,
            "No signals found — run the daemon (or `oracle.py forecast`) to accumulate history first.",
        )
    return run_backtest_on_signals(
        cfg, signals, price_cache=None, cost_model=cost_model,
        fold=fold, n_baseline_trials=n_baseline_trials,
    )


def run_backtest_on_signals(
    cfg: OracleConfig,
    signals: list[dict],
    price_cache: Optional[pd.DataFrame] = None,
    cost_model: Optional[CostModel] = None,
    fold: str = "week",
    n_baseline_trials: int = 1000,
) -> BacktestResult:
    """Score an arbitrary batch of signal dicts (same schema Signal.to_dict()
    produces) against realized outcomes, net of `cost_model`.

    `price_cache`, if given, is used as-is by pair_signals_with_outcomes()
    instead of triggering a Dukascopy fetch — pass your own pre-loaded OHLCV
    DataFrame (e.g. the full walk-forward replay's in-memory 90-day frame)
    to avoid a redundant fetch/re-decode of data you already have.
    """
    cost_model = cost_model or CostModel()
    if not signals:
        return _empty_result(
            cost_model, fold,
            "No signals found — run the daemon (or `oracle.py forecast`) to accumulate history first.",
        )

    paired, _ = pair_signals_with_outcomes(cfg, signals, price_cache=price_cache)
    if not paired:
        return _empty_result(
            cost_model, fold,
            f"{len(signals)} signals were logged, but none could be paired with a realized "
            "price (Dukascopy fetch failed, or all signal timestamps fall outside available "
            "history). This is NOT the same as 'no signal history' — check the warnings above "
            "for the actual fetch error before assuming there's nothing to evaluate.",
        )

    confusion: dict[str, dict[str, int]] = {
        "UP": {"UP": 0, "DOWN": 0, "FLAT": 0},
        "DOWN": {"UP": 0, "DOWN": 0, "FLAT": 0},
        "FLAT": {"UP": 0, "DOWN": 0, "FLAT": 0},
    }
    by_conf: dict[int, dict[str, int]] = {i: {"n": 0, "hit": 0} for i in range(5)}
    by_regime: dict[str, dict[str, int]] = {}
    model_stats: dict[str, dict[str, int]] = {}
    hits_all = 0

    for p in paired:
        confusion[p.pred_direction][p.actual_direction] += 1
        if p.pred_direction == p.actual_direction:
            hits_all += 1

        b = min(int(p.pred_confidence * 5), 4)
        by_conf[b]["n"] += 1
        if p.pred_direction == p.actual_direction:
            by_conf[b]["hit"] += 1

        rstats = by_regime.setdefault(p.regime_label, {"n": 0, "n_directional": 0, "hits": 0})
        rstats["n"] += 1
        if p.pred_direction != "FLAT":
            rstats["n_directional"] += 1
            if p.pred_direction == p.actual_direction:
                rstats["hits"] += 1

        for m, info in p.per_model.items():
            stats = model_stats.setdefault(m, {"n": 0, "hit": 0})
            stats["n"] += 1
            if info.get("vote", "FLAT") == p.actual_direction:
                stats["hit"] += 1

    directional = [p for p in paired if p.pred_direction != "FLAT"]
    n_total = len(paired)
    n_directional = len(directional)
    hits_directional = sum(1 for p in directional if p.pred_direction == p.actual_direction)

    pip = pip_size(cfg.dukascopy_symbol)
    pnl_no_costs = sum(gross_trade_pips(cfg, p) for p in directional)
    pnl_with_costs = sum(net_trade_pips(cfg, cost_model, p) for p in directional)

    by_conf_list = [
        {"bucket": f"{b*20}-{(b+1)*20}%", "n": s["n"], "accuracy": s["hit"] / s["n"]}
        for b, s in by_conf.items() if s["n"] > 0
    ]

    by_regime_out = {
        r: (
            {
                "n": st["n"], "n_directional": st["n_directional"],
                "coverage": st["n_directional"] / st["n"],
                "accuracy_directional": st["hits"] / st["n_directional"],
            } if st["n_directional"] > 0 else
            {"n": st["n"], "n_directional": 0, "coverage": 0.0, "accuracy_directional": None}
        )
        for r, st in by_regime.items()
    }

    per_model_acc = {m: (s["hit"] / s["n"] if s["n"] else None) for m, s in model_stats.items()}

    # ---- Walk-forward folds -------------------------------------------------
    fold_map: dict[str, list[PairedSignal]] = {}
    for p in paired:
        fold_map.setdefault(_fold_key(p.ts, fold), []).append(p)

    folds = []
    for key in sorted(fold_map):
        fp = fold_map[key]
        fdir = [p for p in fp if p.pred_direction != "FLAT"]
        hits = sum(1 for p in fdir if p.pred_direction == p.actual_direction)
        net_pips_per_trade = [net_trade_pips(cfg, cost_model, p) for p in fdir]
        folds.append(FoldResult(
            fold=key,
            n_signals=len(fp),
            n_directional=len(fdir),
            accuracy_directional=(hits / len(fdir) if fdir else None),
            pnl_pips_with_costs=float(sum(net_pips_per_trade)),
            expectancy_pips_per_trade_net=(float(np.mean(net_pips_per_trade)) if fdir else None),
        ).to_dict())

    gate = _build_gate(cfg, cost_model, directional)
    baseline = _random_direction_baseline(cfg, cost_model, directional, n_trials=n_baseline_trials)

    small_sample_warning = None
    if n_directional < MIN_DIRECTIONAL_FOR_CONFIDENCE:
        small_sample_warning = (
            f"Only {n_directional} directional trades (< {MIN_DIRECTIONAL_FOR_CONFIDENCE}) — "
            "accuracy/expectancy/gate numbers are not statistically meaningful yet. "
            "Keep the daemon running to accumulate more signal history."
        )

    return BacktestResult(
        n_signals=n_total,
        n_directional=n_directional,
        coverage=(n_directional / n_total) if n_total else 0.0,
        accuracy_directional=(hits_directional / n_directional) if n_directional else None,
        accuracy_all=(hits_all / n_total) if n_total else None,
        confusion=confusion,
        by_confidence=by_conf_list,
        by_regime=by_regime_out,
        per_model_accuracy=per_model_acc,
        cost_model=cost_model.to_dict(),
        pnl_pips_no_costs=float(pnl_no_costs),
        pnl_pips_with_costs=float(pnl_with_costs),
        fold_granularity=fold,
        folds=folds,
        gate=gate.to_dict(),
        baseline_random_direction=baseline,
        small_sample_warning=small_sample_warning,
    )
