"""Combine Kronos and TimesFM outputs into a structured text brief.

The brief is what the LLM agents inside TradingAgents actually read. It's
plain text, lightly structured, with the numbers presented as text so the
LLM doesn't have to do float math (which it's bad at).
"""

from __future__ import annotations

import logging
from textwrap import dedent

import pandas as pd

from .config import CONFIG
from .data import fetch_ohlcv
from .forecasters import (
    KronosForecast,
    TimesFMForecast,
    kronos_forecast,
    timesfm_forecast,
)

log = logging.getLogger(__name__)


def _next_business_days(start: pd.Timestamp, n: int) -> pd.Series:
    """Generate n future business-day timestamps after `start`."""
    return pd.Series(pd.bdate_range(start=start + pd.Timedelta(days=1), periods=n))


def _pct(x: float) -> str:
    return f"{x * 100:+.2f}%"


def _agreement(kronos_pct: float, timesfm_pct: float) -> str:
    """One-line characterisation of whether the two models agree."""
    same_sign = (kronos_pct > 0) == (timesfm_pct > 0)
    gap = abs(kronos_pct - timesfm_pct)
    if same_sign and gap < 0.02:
        return "STRONG AGREEMENT — both models point the same direction with similar magnitude."
    if same_sign:
        return "DIRECTIONAL AGREEMENT — same direction, magnitudes differ."
    if abs(kronos_pct) < 0.005 and abs(timesfm_pct) < 0.005:
        return "BOTH FLAT — neither model sees meaningful movement."
    return "DISAGREEMENT — models point in opposite directions; treat the forecast as low-confidence."


def build_forecast_brief(ticker: str, end_date: str) -> dict:
    """Top-level entry point. Fetches data, runs both forecasters, returns the brief.

    Returns a dict with:
        - 'brief': str, the human-readable brief for the LLM
        - 'kronos': KronosForecast
        - 'timesfm': TimesFMForecast
        - 'meta': fetch + config metadata
    """
    log.info("Building forecast brief for %s as of %s", ticker, end_date)

    bars_needed = max(CONFIG.kronos_lookback_bars, CONFIG.timesfm_max_context)
    df = fetch_ohlcv(ticker=ticker, end_date=end_date, bars=bars_needed)

    # ---- Kronos ----------------------------------------------------------
    # Kronos wants OHLCV history + future timestamps, predicts OHLCV.
    kronos_lookback = min(CONFIG.kronos_lookback_bars, CONFIG.kronos_max_context)
    kronos_df = df.tail(kronos_lookback).reset_index(drop=True)
    future_ts = _next_business_days(
        kronos_df["timestamps"].iloc[-1], CONFIG.kronos_pred_bars
    )

    try:
        kr = kronos_forecast(kronos_df, CONFIG.kronos_pred_bars, future_ts)
    except Exception as e:
        log.exception("Kronos forecast failed")
        raise RuntimeError(f"Kronos forecast failed: {e}") from e

    # ---- TimesFM ---------------------------------------------------------
    # Univariate on close prices.
    tf_history = df["close"].tail(CONFIG.timesfm_max_context).reset_index(drop=True)
    try:
        tf = timesfm_forecast(tf_history, horizon=CONFIG.timesfm_horizon)
    except Exception as e:
        log.exception("TimesFM forecast failed")
        raise RuntimeError(f"TimesFM forecast failed: {e}") from e

    # ---- Brief -----------------------------------------------------------
    last_obs_date = df["timestamps"].iloc[-1].strftime("%Y-%m-%d")
    horizon_days = max(CONFIG.kronos_pred_bars, CONFIG.timesfm_horizon)

    brief = dedent(
        f"""
        QUANTITATIVE FORECAST BRIEF
        ===========================
        Ticker:           {ticker}
        As of:            {end_date}  (last observed bar: {last_obs_date})
        Horizon:          {horizon_days} business days
        Last close:       ${kr.last_close:,.2f}

        --- Kronos (OHLCV foundation model, multivariate) ---
        Final close:      ${kr.final_close:,.2f}
        Expected return:  {_pct(kr.pct_change)}
        Max run-up:       {_pct(kr.max_runup)}
        Max drawdown:     {_pct(kr.max_drawdown)}
        Sampling:         {CONFIG.kronos_sample_count} paths, T={CONFIG.kronos_temperature}, top_p={CONFIG.kronos_top_p}

        --- TimesFM 2.5 (univariate, quantile-aware) ---
        Final point:      ${tf.final_point:,.2f}
        Expected return:  {_pct(tf.pct_change_point)}
        p10 final:        ${tf.p10_final:,.2f}  ({_pct(tf.p10_final / tf.last_close - 1)})
        p90 final:        ${tf.p90_final:,.2f}  ({_pct(tf.p90_final / tf.last_close - 1)})

        --- Cross-model read ---
        {_agreement(kr.pct_change, tf.pct_change_point)}

        Notes:
        - Kronos is multivariate (uses OHLCV + dollar-volume proxy); TimesFM
          here is univariate on close. They are deliberately different views.
        - Forecasts assume continuation of the historical regime. Treat
          large p10/p90 spread as a regime-change signal, not as a prediction.
        - Past performance and model forecasts are not financial advice.
        """
    ).strip()

    return {
        "brief": brief,
        "kronos": kr,
        "timesfm": tf,
        "meta": {
            "ticker": ticker,
            "end_date": end_date,
            "last_observed": last_obs_date,
            "horizon_days": horizon_days,
        },
    }
