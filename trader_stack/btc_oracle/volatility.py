"""Volatility forecaster — EWMA + ATR.

Used for two things:

1. **Noise floor estimation**: predict the expected 10-minute volatility band
   around the current price. If the forecast move is smaller than this band,
   the signal is noise — return FLAT regardless of model agreement.

2. **Regime hinting**: an ATR expansion (current ATR > 1.5 × 50-bar average ATR)
   is a strong breakout signal.

This is intentionally simple. A full GARCH(1,1) gives ~5% better point forecasts
but adds a model dependency and ~50ms per cycle. EWMA + ATR is good enough at
the 10-minute horizon.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class VolatilityForecast:
    realized_vol_1m: float       # most recent 1-bar log-return std (rolling 30)
    ewma_vol_1m: float           # EWMA-smoothed 1-bar vol
    atr_14: float                # 14-bar ATR (absolute price units)
    atr_50: float                # 50-bar ATR (baseline)
    atr_expansion: float         # atr_14 / atr_50  (>1 = expanding)
    horizon_minutes: int
    expected_noise_band_pct: float   # ±N% expected pure-noise move over the horizon


def _ewma(x: np.ndarray, halflife: int) -> float:
    """Last-value EWMA. halflife is in samples."""
    return float(pd.Series(x).ewm(halflife=halflife, adjust=False).mean().iloc[-1])


def forecast_volatility(ohlcv: pd.DataFrame, horizon_minutes: int) -> VolatilityForecast:
    """Compute the suite of vol statistics needed for the noise floor and ATR expansion."""
    close = ohlcv["close"].astype(float).to_numpy()
    high = ohlcv["high"].astype(float).to_numpy()
    low = ohlcv["low"].astype(float).to_numpy()

    log_ret = np.diff(np.log(close), prepend=np.log(close[0]))

    # Realized vol — last 30 bars
    s = pd.Series(log_ret)
    realized_vol_1m = float(s.tail(30).std()) if len(s) >= 5 else 0.0

    # EWMA-smoothed vol over the same window
    ewma_vol_1m = _ewma(np.abs(log_ret), halflife=15)

    # ATR ----------------------------------------------------------------
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ])
    tr_series = pd.Series(tr)
    atr_14 = float(tr_series.tail(14).mean()) if len(tr_series) >= 5 else 0.0
    atr_50 = float(tr_series.tail(50).mean()) if len(tr_series) >= 10 else atr_14
    atr_expansion = atr_14 / atr_50 if atr_50 > 0 else 1.0

    # Expected noise band over the horizon
    # Brownian-style scaling: T-bar std ≈ sqrt(T) × 1-bar std
    # The "band" is one std-dev → ~68% containment under normal-ish assumptions
    noise_band_pct = float(realized_vol_1m * np.sqrt(max(horizon_minutes, 1)))

    return VolatilityForecast(
        realized_vol_1m=realized_vol_1m,
        ewma_vol_1m=ewma_vol_1m,
        atr_14=atr_14,
        atr_50=atr_50,
        atr_expansion=atr_expansion,
        horizon_minutes=horizon_minutes,
        expected_noise_band_pct=noise_band_pct,
    )
