"""Signal layer: combines forecasts + microstructure into a single direction signal.

The output is a Signal dataclass with:
  - direction:  "UP" | "DOWN" | "FLAT"
  - confidence: 0.0 — 1.0
  - reasoning: structured breakdown of how it got there (for logging + UI)

The combiner is intentionally simple and inspectable. A heavier ML meta-learner
can replace `combine_signal()` later without changing the rest of the pipeline.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from .feeds import DerivativesSnapshot, MarketSnapshot, OrderBookSnapshot
from .forecasters_extra import (
    CloseForecast,
    chronos2_forecast,
    tirex_forecast,
)
from .oracle_config import OracleConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------


@dataclass
class Signal:
    timestamp: str                     # ISO 8601 UTC
    last_close: float
    horizon_minutes: int
    direction: str                     # "UP" | "DOWN" | "FLAT"  (post-filter)
    confidence: float                  # 0.0 — 1.0  (post-filter)
    consensus_pct_change: float        # weighted-mean predicted % change
    per_model: dict[str, dict]         # {model_name: {pct_change, vote, weight, ...}}
    microstructure: dict               # snapshot of book + derivs metrics that influenced the call
    # --- Phase 2 additions -----------------------------------------------
    raw_direction: str = "FLAT"        # ensemble direction BEFORE filter
    raw_confidence: float = 0.0        # ensemble confidence BEFORE filter
    regime: Optional[dict] = None      # {label, confidence, posteriors, features}
    volatility: Optional[dict] = None  # {realized, ewma, atr_14, atr_50, expansion, noise_band_pct}
    filter_fired: list = field(default_factory=list)   # which gates fired
    # ---------------------------------------------------------------------
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Forecast dispatcher — runs all enabled models in parallel
# ---------------------------------------------------------------------------


def _kronos_close_forecast(snap: MarketSnapshot, horizon: int) -> CloseForecast:
    """Adapter: run Kronos and project its OHLCV output into a close-only forecast."""
    # Local import keeps Kronos optional if disabled.
    from trader_stack.forecasters import _kronos_predictor  # type: ignore

    predictor = _kronos_predictor()
    df = snap.ohlcv
    # Kronos wants past + future timestamps.
    last_ts = df["timestamps"].iloc[-1]
    future_ts = pd.Series(pd.date_range(start=last_ts + pd.Timedelta(minutes=1),
                                        periods=horizon, freq="1min"))
    x_df = df[["open", "high", "low", "close", "volume", "amount"]].copy()

    pred = predictor.predict(
        df=x_df,
        x_timestamp=df["timestamps"],
        y_timestamp=future_ts,
        pred_len=horizon,
        T=1.0,
        top_p=0.9,
        sample_count=3,  # speed > variance reduction at the per-minute cadence
    )
    point = pred["close"].to_numpy(dtype=np.float32)
    last = float(df["close"].iloc[-1])
    final = float(point[-1])
    return CloseForecast(
        model="kronos",
        last_close=last,
        horizon_closes=point,
        final_close=final,
        pct_change=final / last - 1.0,
    )


def _timesfm_close_forecast(snap: MarketSnapshot, horizon: int) -> CloseForecast:
    from trader_stack.forecasters import _timesfm_model  # type: ignore

    model = _timesfm_model()
    closes = snap.ohlcv["close"].to_numpy(dtype=np.float32)
    point_forecast, quantile_forecast = model.forecast(horizon=horizon, inputs=[closes])
    point = np.asarray(point_forecast)[0][:horizon]
    quantiles = np.asarray(quantile_forecast)[0]
    last = float(closes[-1])
    final = float(point[-1])
    return CloseForecast(
        model="timesfm",
        last_close=last,
        horizon_closes=point,
        final_close=final,
        pct_change=final / last - 1.0,
        p10_final=float(quantiles[-1, 1]),
        p90_final=float(quantiles[-1, 9]),
    )


def _build_covariates_for_chronos2(snap: MarketSnapshot) -> pd.DataFrame | None:
    """Build past-only covariate series for Chronos-2.

    The OHLCV buffer already carries per-bar taker_buy_ratio (Binance kline,
    no API key needed). That's the strongest minute-scale predictor we have,
    so it goes in as a full historical series — not a broadcast constant.

    Order book and derivatives are single snapshots; we broadcast them as
    the current value. That's a regime hint, not history.
    """
    n = len(snap.ohlcv)
    cov = pd.DataFrame(index=range(n))

    # Per-bar taker buy ratio — true historical series.
    if "taker_buy_ratio" in snap.ohlcv.columns:
        cov["taker_buy_ratio"] = snap.ohlcv["taker_buy_ratio"].astype(np.float32).to_numpy()

    if snap.book is not None:
        cov["ob_imbalance"] = snap.book.imbalance
        cov["spread_bps"] = snap.book.spread_bps
    if snap.derivs is not None:
        cov["funding_rate"] = snap.derivs.funding_rate
        cov["oi_delta_5m"] = snap.derivs.open_interest_delta_5m
        cov["ls_ratio"] = snap.derivs.long_short_ratio
        cov["taker_ls_ratio_5m"] = snap.derivs.taker_buy_sell_ratio  # aggregated, distinct from per-bar

    if cov.empty:
        return None
    cov.index = snap.ohlcv["timestamps"].values
    return cov


def run_forecasts(snap: MarketSnapshot, cfg: OracleConfig) -> dict[str, CloseForecast]:
    """Run all enabled forecasters in parallel; return {model_name: CloseForecast}."""
    tasks: dict[str, callable] = {}

    if cfg.use_kronos:
        tasks["kronos"] = lambda: _kronos_close_forecast(snap, cfg.horizon_minutes)
    if cfg.use_timesfm:
        tasks["timesfm"] = lambda: _timesfm_close_forecast(snap, cfg.horizon_minutes)
    if cfg.use_tirex:
        closes = pd.Series(
            snap.ohlcv["close"].to_numpy(dtype=np.float32),
            index=snap.ohlcv["timestamps"].values,
        )
        tasks["tirex"] = lambda c=closes: tirex_forecast(c, cfg.horizon_minutes)
    if cfg.use_chronos2:
        closes = pd.Series(
            snap.ohlcv["close"].to_numpy(dtype=np.float32),
            index=snap.ohlcv["timestamps"].values,
        )
        cov = _build_covariates_for_chronos2(snap)
        tasks["chronos2"] = lambda c=closes, x=cov: chronos2_forecast(c, cfg.horizon_minutes, x)

    results: dict[str, CloseForecast] = {}
    if not tasks:
        return results

    # CPU-bound work — but each model releases the GIL in its native code.
    # ThreadPoolExecutor keeps memory shared and avoids forking overhead.
    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        futures = {pool.submit(fn): name for name, fn in tasks.items()}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                results[name] = fut.result()
            except Exception as e:
                log.warning("Forecaster %s failed: %s", name, e)

    return results


# ---------------------------------------------------------------------------
# Microstructure regime filter
# ---------------------------------------------------------------------------


def _microstructure_bias(book: Optional[OrderBookSnapshot],
                         derivs: Optional[DerivativesSnapshot],
                         cfg: OracleConfig,
                         taker_buy_ratio_per_bar: Optional[float] = None) -> tuple[float, list[str]]:
    """Compute a microstructure 'bias' in [-1, +1] independent of the forecasts.

    +1 → strong up bias (e.g. heavy bid book + negative extreme funding = shorts trapped)
    -1 → strong down bias (e.g. heavy ask book + positive extreme funding = longs trapped)
     0 → neutral

    The bias is then blended with the forecast-ensemble vote.

    Returns (bias, notes).
    """
    bias = 0.0
    notes: list[str] = []

    # Order book imbalance — direct microstructure signal.
    if book is not None:
        # imbalance is already in [-1, +1]
        bias += 0.35 * book.imbalance
        notes.append(f"OB imbalance: {book.imbalance:+.2f}")

    # Per-bar taker buy ratio (from Binance kline; in [0, 1], 0.5 = balanced).
    # This is the per-minute aggressor pressure — empirically one of the
    # strongest minute-scale predictors. Centered around 0 by subtracting 0.5.
    if taker_buy_ratio_per_bar is not None:
        per_bar = (taker_buy_ratio_per_bar - 0.5) * 2.0  # map to [-1, +1]
        bias += 0.25 * per_bar
        notes.append(f"Per-bar taker: {taker_buy_ratio_per_bar:.2f}")

    # Funding rate — contrarian at extremes.
    if derivs is not None:
        fr = derivs.funding_rate
        if abs(fr) >= cfg.funding_extreme_threshold:
            bias += -0.5 * np.sign(fr)
            notes.append(f"Funding {fr:+.4%} (extreme)")
        else:
            notes.append(f"Funding {fr:+.4%}")

        notes.append(f"OI Δ5m: {derivs.open_interest_delta_5m:+.2%}")

        # Long/short skew — when crowd is very long, mean reversion bias.
        if derivs.long_short_ratio > 2.0:
            bias += -0.2
            notes.append(f"L/S crowded long ({derivs.long_short_ratio:.2f})")
        elif derivs.long_short_ratio < 0.5:
            bias += +0.2
            notes.append(f"L/S crowded short ({derivs.long_short_ratio:.2f})")

        # 5m-aggregated taker ratio — distinct from the per-bar reading above.
        if derivs.taker_buy_sell_ratio > 1.1:
            bias += +0.1
            notes.append(f"5m aggressor buy ({derivs.taker_buy_sell_ratio:.2f})")
        elif derivs.taker_buy_sell_ratio < 0.9:
            bias += -0.1
            notes.append(f"5m aggressor sell ({derivs.taker_buy_sell_ratio:.2f})")

    return float(np.clip(bias, -1.0, +1.0)), notes


# ---------------------------------------------------------------------------
# Combine forecasts + microstructure → Signal
# ---------------------------------------------------------------------------


def _vote(pct: float, threshold: float) -> int:
    """Map predicted % change to a vote: +1 UP, -1 DOWN, 0 FLAT."""
    if pct >= threshold:
        return 1
    if pct <= -threshold:
        return -1
    return 0


def combine_signal(
    forecasts: dict[str, CloseForecast],
    snap: MarketSnapshot,
    cfg: OracleConfig,
    regime_state: Optional["RegimeState"] = None,   # type: ignore[name-defined]
    volatility: Optional["VolatilityForecast"] = None,  # type: ignore[name-defined]
) -> Signal:
    """Combine all forecast outputs + microstructure into a Signal.

    Phase 2: also takes optional regime classification and volatility forecast,
    and runs them through the trade-filter gate sequence to produce the final
    direction.
    """
    notes: list[str] = []

    # --- Per-model summary -------------------------------------------------
    per_model: dict[str, dict] = {}
    weighted_sum = 0.0
    total_weight = 0.0
    votes: list[tuple[int, float]] = []  # (vote, weight)

    for name, fc in forecasts.items():
        w = cfg.weights.get(name, 1.0)
        v = _vote(fc.pct_change, cfg.flat_threshold)
        per_model[name] = {
            "pct_change": fc.pct_change,
            "final_close": fc.final_close,
            "vote": {1: "UP", 0: "FLAT", -1: "DOWN"}[v],
            "weight": w,
            "p10": fc.p10_final,
            "p90": fc.p90_final,
        }
        weighted_sum += w * fc.pct_change
        total_weight += w
        votes.append((v, w))

    if total_weight == 0:
        # No forecasters succeeded.
        return Signal(
            timestamp=datetime.now(timezone.utc).isoformat(),
            last_close=snap.last_close,
            horizon_minutes=cfg.horizon_minutes,
            direction="FLAT",
            confidence=0.0,
            consensus_pct_change=0.0,
            per_model=per_model,
            microstructure={},
            raw_direction="FLAT",
            raw_confidence=0.0,
            notes=["No forecasters available"],
        )

    consensus_pct = weighted_sum / total_weight
    notes.append(f"Consensus forecast: {consensus_pct:+.3%}")

    # --- Microstructure bias ----------------------------------------------
    # Pick up the most recent per-bar taker buy ratio from the OHLCV buffer
    # (the Binance kline pipeline puts it there).
    per_bar_taker = None
    if "taker_buy_ratio" in snap.ohlcv.columns:
        try:
            per_bar_taker = float(snap.ohlcv["taker_buy_ratio"].iloc[-1])
        except Exception:
            per_bar_taker = None

    micro_bias, micro_notes = _microstructure_bias(
        snap.book, snap.derivs, cfg, taker_buy_ratio_per_bar=per_bar_taker
    )
    notes.extend(micro_notes)

    # Blend forecast + microstructure
    micro_nudge = micro_bias * cfg.flat_threshold * 0.5
    blended_pct = (1 - cfg.microstructure_weight) * consensus_pct + \
                   cfg.microstructure_weight * (micro_nudge * 5)

    # --- Raw direction + raw confidence (pre-filter) ----------------------
    blended_vote = _vote(blended_pct, cfg.flat_threshold)
    raw_direction = {1: "UP", 0: "FLAT", -1: "DOWN"}[blended_vote]

    if blended_vote == 0:
        agreement = 1.0 - (sum(w for v, w in votes if v == 0) / total_weight)
        magnitude = abs(blended_pct) / cfg.flat_threshold
        raw_confidence = 0.3 * (1 - agreement) + 0.1 * min(magnitude, 1.0)
    else:
        agree_weight = sum(w for v, w in votes if v == blended_vote)
        agreement = agree_weight / total_weight
        magnitude = min(abs(blended_pct) / (cfg.flat_threshold * 3), 1.0)
        micro_aligned = (np.sign(micro_bias) == blended_vote) or micro_bias == 0
        raw_confidence = 0.5 * agreement + 0.3 * magnitude + 0.2 * (1.0 if micro_aligned else 0.0)

    raw_confidence = float(np.clip(raw_confidence, 0.0, 1.0))

    # --- Microstructure record --------------------------------------------
    micro_record: dict = {}
    if snap.book is not None:
        micro_record["ob_imbalance"] = snap.book.imbalance
        micro_record["spread_bps"] = snap.book.spread_bps
    if snap.derivs is not None:
        micro_record["funding_rate"] = snap.derivs.funding_rate
        micro_record["oi_delta_5m"] = snap.derivs.open_interest_delta_5m
        micro_record["long_short_ratio"] = snap.derivs.long_short_ratio
        micro_record["taker_buy_sell_ratio_5m"] = snap.derivs.taker_buy_sell_ratio
    if per_bar_taker is not None:
        micro_record["taker_buy_ratio_per_bar"] = per_bar_taker
    micro_record["bias"] = micro_bias

    # --- Phase 2: regime + volatility ------------------------------------
    regime_dict: Optional[dict] = None
    if regime_state is not None:
        regime_dict = {
            "label": regime_state.label,
            "confidence": regime_state.confidence,
            "posteriors": regime_state.posteriors,
            "features": regime_state.feature_snapshot,
        }
        notes.append(f"Regime: {regime_state.label} @ {regime_state.confidence:.2f}")

    vol_dict: Optional[dict] = None
    expected_noise = 0.0
    if volatility is not None:
        vol_dict = {
            "realized_vol_1m": volatility.realized_vol_1m,
            "ewma_vol_1m": volatility.ewma_vol_1m,
            "atr_14": volatility.atr_14,
            "atr_50": volatility.atr_50,
            "atr_expansion": volatility.atr_expansion,
            "expected_noise_band_pct": volatility.expected_noise_band_pct,
        }
        expected_noise = volatility.expected_noise_band_pct
        notes.append(
            f"Noise band ±{expected_noise:.3%} (ATR exp {volatility.atr_expansion:.2f})"
        )

    # --- Phase 2: trade filter -------------------------------------------
    final_direction = raw_direction
    final_confidence = raw_confidence
    fired: list[str] = []

    if cfg.use_trade_filter:
        # Local import keeps the dependency on the filters module narrow.
        from .filters import apply_filters

        result = apply_filters(
            raw_direction=raw_direction,
            raw_confidence=raw_confidence,
            blended_pct=blended_pct,
            expected_noise_pct=expected_noise,
            regime_label=regime_state.label if regime_state is not None else None,
            regime_confidence=regime_state.confidence if regime_state is not None else 0.0,
            spread_bps=(snap.book.spread_bps if snap.book is not None else None),
            min_confidence=cfg.min_confidence,
            spread_bps_max=cfg.spread_bps_max,
            noise_floor_multiplier=cfg.noise_floor_multiplier,
            chop_blocks_signals=cfg.chop_blocks_signals,
        )
        final_direction = result.direction
        final_confidence = result.confidence
        fired = result.fired_gates
        notes.extend(result.notes)

    return Signal(
        timestamp=datetime.now(timezone.utc).isoformat(),
        last_close=snap.last_close,
        horizon_minutes=cfg.horizon_minutes,
        direction=final_direction,
        confidence=final_confidence,
        consensus_pct_change=consensus_pct,
        per_model=per_model,
        microstructure=micro_record,
        raw_direction=raw_direction,
        raw_confidence=raw_confidence,
        regime=regime_dict,
        volatility=vol_dict,
        filter_fired=fired,
        notes=notes,
    )
