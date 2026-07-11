"""Kronos and TimesFM wrappers. Both load lazily, both cache as singletons.

Importing this module is cheap — model weights only download on the first
call to .predict(). After that, the predictor stays in memory for the life
of the process.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd

from .config import CONFIG

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Kronos
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _kronos_predictor():
    """Singleton KronosPredictor. Downloads weights from HF Hub on first call."""
    # The Kronos repo isn't a pip package; setup.sh added its directory to
    # sys.path via a .pth file. So `from model import ...` resolves to
    # third_party/Kronos/model/__init__.py.
    from model import Kronos, KronosPredictor, KronosTokenizer  # type: ignore

    log.info("Loading Kronos tokenizer: %s", CONFIG.kronos_tokenizer)
    tokenizer = KronosTokenizer.from_pretrained(CONFIG.kronos_tokenizer)

    log.info("Loading Kronos model: %s", CONFIG.kronos_model)
    model = Kronos.from_pretrained(CONFIG.kronos_model)

    predictor = KronosPredictor(
        model,
        tokenizer,
        max_context=CONFIG.kronos_max_context,
        device=CONFIG.device,
    )
    log.info("Kronos ready on device=%s", CONFIG.device)
    return predictor


@dataclass
class KronosForecast:
    pred_df: pd.DataFrame              # forecasted open/high/low/close/volume/amount
    last_close: float                  # last observed close (input)
    final_close: float                 # final forecasted close
    pct_change: float                  # final_close / last_close - 1
    max_drawdown: float                # within the forecast horizon, vs last_close
    max_runup: float                   # within the forecast horizon, vs last_close


def kronos_forecast(df: pd.DataFrame, pred_len: int, future_timestamps: pd.Series) -> KronosForecast:
    """Run a Kronos forecast on the supplied OHLCV history.

    Args:
        df: DataFrame with columns timestamps, open, high, low, close, volume, amount.
            The full window is treated as the lookback (KronosPredictor trims to max_context).
        pred_len: number of bars to forecast.
        future_timestamps: pd.Series of length pred_len with the future timestamps.

    Returns:
        KronosForecast with the raw pred_df and a few summary statistics.
    """
    predictor = _kronos_predictor()

    x_df = df[["open", "high", "low", "close", "volume", "amount"]].copy()
    x_timestamp = df["timestamps"]

    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=future_timestamps,
        pred_len=pred_len,
        T=CONFIG.kronos_temperature,
        top_p=CONFIG.kronos_top_p,
        sample_count=CONFIG.kronos_sample_count,
    )

    last_close = float(df["close"].iloc[-1])
    final_close = float(pred_df["close"].iloc[-1])
    pct = final_close / last_close - 1.0
    max_runup = float(pred_df["close"].max() / last_close - 1.0)
    max_drawdown = float(pred_df["close"].min() / last_close - 1.0)

    return KronosForecast(
        pred_df=pred_df,
        last_close=last_close,
        final_close=final_close,
        pct_change=pct,
        max_drawdown=max_drawdown,
        max_runup=max_runup,
    )


# ---------------------------------------------------------------------------
# TimesFM
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _timesfm_model():
    """Singleton TimesFM 2.5 model. Downloads weights on first call."""
    import timesfm
    import torch

    torch.set_float32_matmul_precision("high")

    log.info("Loading TimesFM: %s", CONFIG.timesfm_model)
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(CONFIG.timesfm_model)

    model.compile(
        timesfm.ForecastConfig(
            max_context=CONFIG.timesfm_max_context,
            max_horizon=CONFIG.timesfm_max_horizon,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
        )
    )
    log.info("TimesFM ready")
    return model


@dataclass
class TimesFMForecast:
    point: np.ndarray                  # shape (horizon,)
    quantiles: np.ndarray              # shape (horizon, 10): mean + p10..p90
    last_close: float
    final_point: float                 # point forecast at the final horizon step
    p10_final: float                   # 10th percentile at the final step
    p90_final: float                   # 90th percentile at the final step
    pct_change_point: float            # final_point / last_close - 1


def timesfm_forecast(close_series: pd.Series, horizon: int) -> TimesFMForecast:
    """Run a TimesFM 2.5 univariate forecast on the close prices.

    Args:
        close_series: pd.Series of historical close prices, oldest first.
        horizon: number of bars to forecast.

    Returns:
        TimesFMForecast with point and quantile arrays plus summary stats.
    """
    model = _timesfm_model()

    # TimesFM accepts a list of np arrays; we pass a single series.
    values = close_series.to_numpy(dtype=np.float32)
    point_forecast, quantile_forecast = model.forecast(
        horizon=horizon,
        inputs=[values],
    )
    # shapes: point (1, horizon), quantile (1, horizon, 10)
    point = np.asarray(point_forecast)[0]
    quantiles = np.asarray(quantile_forecast)[0]

    last_close = float(values[-1])
    final_point = float(point[-1])

    return TimesFMForecast(
        point=point,
        quantiles=quantiles,
        last_close=last_close,
        final_point=final_point,
        p10_final=float(quantiles[-1, 1]),
        p90_final=float(quantiles[-1, 9]),
        pct_change_point=final_point / last_close - 1.0,
    )


# ---------------------------------------------------------------------------
# Warm-up helper (optional; useful in long-lived servers)
# ---------------------------------------------------------------------------


def warm_up() -> None:
    """Force-load both models. Call once at startup to avoid first-request latency."""
    _kronos_predictor()
    _timesfm_model()
