"""Wrappers for the two additional forecasters in the BTC oracle ensemble:

- TiRex (NX-AI) — 35M xLSTM, top of Chronos-ZS benchmark, fast.
- Chronos-2 (Amazon) — 120M encoder-only T5 with covariate support.

Kronos and TimesFM are reused from trader_stack.forecasters (unchanged).
All four wrappers expose the same interface: forecast a close-price series
forward N steps and return a numpy array of predicted closes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared output type
# ---------------------------------------------------------------------------


@dataclass
class CloseForecast:
    """Standard output: forecast of close prices for the next `horizon` minutes."""

    model: str                       # "kronos" | "timesfm" | "tirex" | "chronos2"
    last_close: float                # observed final close
    horizon_closes: np.ndarray       # shape (horizon,) — predicted close at each step
    final_close: float               # horizon_closes[-1]
    pct_change: float                # final_close / last_close - 1
    # Optional quantile band at the horizon (p10/p90). Only some models provide it.
    p10_final: float | None = None
    p90_final: float | None = None
    # Diagnostic: how many covariates Chronos-2 actually consumed. None for other models.
    # Telemetry watches this — if it ever drops to 0, that's a silent degradation.
    _n_covariates_attached: int | None = None


# ---------------------------------------------------------------------------
# TiRex
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _tirex_model():
    """Singleton TiRex. Compiles a CUDA kernel on first load if CUDA present."""
    from tirex import load_model  # type: ignore

    log.info("Loading TiRex from HF Hub (NX-AI/TiRex)")
    model = load_model("NX-AI/TiRex")
    log.info("TiRex ready")
    return model


def tirex_forecast(close_series: pd.Series, horizon: int) -> CloseForecast:
    """Run TiRex on a univariate close-price series."""
    import torch

    model = _tirex_model()
    values = close_series.to_numpy(dtype=np.float32)
    # TiRex accepts a batch of series; we pass shape (1, L).
    ctx = torch.from_numpy(values).unsqueeze(0)
    result = model.forecast(context=ctx, prediction_length=horizon)

    # TiRex returns either (quantiles, mean) or a single object depending on version.
    # Handle both shapes defensively.
    if isinstance(result, tuple) and len(result) == 2:
        quantiles, mean = result
        point = np.asarray(mean).reshape(-1)[:horizon]
        qarr = np.asarray(quantiles)
        # qarr is typically shape (1, horizon, n_quantiles). Try to grab p10/p90.
        try:
            if qarr.ndim == 3:
                p10 = float(qarr[0, -1, 0])
                p90 = float(qarr[0, -1, -1])
            else:
                p10 = p90 = None
        except Exception:
            p10 = p90 = None
    else:
        point = np.asarray(result).reshape(-1)[:horizon]
        p10 = p90 = None

    last = float(values[-1])
    final = float(point[-1])
    return CloseForecast(
        model="tirex",
        last_close=last,
        horizon_closes=point,
        final_close=final,
        pct_change=final / last - 1.0,
        p10_final=p10,
        p90_final=p90,
    )


# ---------------------------------------------------------------------------
# Chronos-2 (with covariates)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _chronos2_pipeline():
    """Singleton Chronos-2 pipeline. Downloads ~480 MB on first call."""
    import torch
    from chronos import Chronos2Pipeline  # type: ignore

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    log.info("Loading Chronos-2 (amazon/chronos-2) on device=%s", device)
    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=device)
    log.info("Chronos-2 ready")
    return pipe


def _regularize_to_1min(close_series: pd.Series) -> pd.Series:
    """Reindex onto a complete, gap-free 1-minute grid, forward-filling any
    missing minutes.

    Chronos-2's predict_df() calls pd.infer_freq() on the timestamp column
    and raises "Could not infer frequency" if that returns None. A real M1
    series almost always has at least one dropped minute over a multi-hour
    window (feeds.py's _ticks_to_m1 drops zero-tick minutes entirely rather
    than synthesizing them — see its docstring), which breaks infer_freq's
    strict-regularity requirement. Reproduced empirically: 100% failure rate
    on real gold history windows during the 90-day walk-forward replay (see
    scripts/replay_backtest.py), and present in every observed live daemon
    cycle too — Chronos-2 was silently contributing zero forecasts, not
    gracefully degrading, until this fix. Forward-filling a flat bar for a
    zero-tick minute is the standard treatment (no trade happened, so
    "price didn't move" is the reasonable assumption) and only uses
    already-known past values — no lookahead.
    """
    ts = close_series.index
    full_index = pd.date_range(start=ts.min(), end=ts.max(), freq="1min", tz=ts.tz)
    return close_series.reindex(full_index).ffill()


def chronos2_forecast(
    close_series: pd.Series,
    horizon: int,
    covariates_past: pd.DataFrame | None = None,
) -> CloseForecast:
    """Run Chronos-2 with optional past covariates.

    Chronos-2 can ingest covariates as additional series in the same DataFrame.
    We pass it long-format with columns [id, timestamp, target, <covariate cols>].
    Covariates here are *past-only* — we don't know future order book imbalance
    when forecasting, only history. The model uses them to infer regime.

    Args:
        close_series: indexed by timestamp, the historical close prices.
        horizon: how many minutes ahead to forecast.
        covariates_past: optional DataFrame indexed by the same timestamps with
            extra feature columns (e.g. order_book_imbalance, funding_rate).
    """
    pipe = _chronos2_pipeline()

    # Regularize onto a gap-free 1-minute grid FIRST — see _regularize_to_1min's
    # docstring. `ts` below is the regularized index; covariates are aligned
    # to it (their own .reindex(ts).ffill().bfill() call already handles any
    # newly-introduced gap timestamps the same way).
    close_series = _regularize_to_1min(close_series)

    # Build the long-format context DataFrame expected by Chronos2Pipeline.predict_df.
    ts = close_series.index
    context_df = pd.DataFrame({
        "id": "btc",
        "timestamp": ts,
        "target": close_series.to_numpy(dtype=np.float32),
    })

    # Attach past covariates if provided. They must be aligned to the same index.
    n_covariates_attached = 0
    if covariates_past is not None and not covariates_past.empty:
        for col in covariates_past.columns:
            # Forward-fill any missing values; Chronos-2 is picky about NaNs.
            context_df[col] = covariates_past[col].reindex(ts).ffill().bfill().to_numpy(dtype=np.float32)
            n_covariates_attached += 1

    # Diagnostic: surface silent degradation. If we expected covariates but
    # none made it onto the context_df, that's almost always a NaN issue or
    # an index-alignment bug. Log loudly so this doesn't fail silently.
    if covariates_past is not None and n_covariates_attached == 0:
        log.warning(
            "Chronos-2: %d covariate columns requested but 0 attached — "
            "degrading to close-only forecast. Check covariate timestamp alignment.",
            len(covariates_past.columns) if covariates_past is not None else 0,
        )
    elif n_covariates_attached > 0:
        log.debug("Chronos-2: %d covariates attached (cols: %s)",
                  n_covariates_attached, list(covariates_past.columns))

    try:
        pred_df = pipe.predict_df(
            context_df,
            prediction_length=horizon,
            quantile_levels=[0.1, 0.5, 0.9],
            id_column="id",
            timestamp_column="timestamp",
            target="target",
        )
    except Exception as e:
        # Chronos-2 API surface has evolved; if covariates trip it, fall back to plain.
        log.warning("Chronos-2 covariate path failed (%s); retrying without covariates", e)
        plain_df = context_df[["id", "timestamp", "target"]]
        pred_df = pipe.predict_df(
            plain_df,
            prediction_length=horizon,
            quantile_levels=[0.1, 0.5, 0.9],
            id_column="id",
            timestamp_column="timestamp",
            target="target",
        )

    # pred_df typically has columns: id, timestamp, "0.1", "0.5", "0.9" (or "mean").
    # The point forecast is the median (0.5) when present, else "mean".
    if "0.5" in pred_df.columns:
        point = pred_df["0.5"].to_numpy(dtype=np.float32)
    elif "mean" in pred_df.columns:
        point = pred_df["mean"].to_numpy(dtype=np.float32)
    else:
        # Last resort: take the first non-id/timestamp numeric column
        numeric_cols = [c for c in pred_df.columns if c not in {"id", "timestamp"}]
        point = pred_df[numeric_cols[0]].to_numpy(dtype=np.float32)

    p10 = float(pred_df["0.1"].iloc[-1]) if "0.1" in pred_df.columns else None
    p90 = float(pred_df["0.9"].iloc[-1]) if "0.9" in pred_df.columns else None

    last = float(close_series.iloc[-1])
    final = float(point[-1])

    return CloseForecast(
        model="chronos2",
        last_close=last,
        horizon_closes=point,
        final_close=final,
        pct_change=final / last - 1.0,
        p10_final=p10,
        p90_final=p90,
        _n_covariates_attached=n_covariates_attached,
    )


# ---------------------------------------------------------------------------
# Warm-up helper
# ---------------------------------------------------------------------------


def warm_up(use_tirex: bool = True, use_chronos2: bool = True) -> None:
    """Force-load the extra forecasters. Useful at daemon startup."""
    if use_tirex:
        try:
            _tirex_model()
        except Exception as e:
            log.error("TiRex warm-up failed: %s", e)
    if use_chronos2:
        try:
            _chronos2_pipeline()
        except Exception as e:
            log.error("Chronos-2 warm-up failed: %s", e)
