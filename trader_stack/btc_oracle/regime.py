"""Regime classifier — Hidden Markov Model over market features.

Classifies the current market into one of 4 latent regimes:
  - CHOP      : low volatility, mean-reverting, low directional edge
  - TREND     : sustained directional moves, momentum models win
  - BREAKOUT  : volatility expansion, trust strong forecasts
  - CASCADE   : liquidation regime, microstructure dominates everything

The labels are assigned POST-HOC after fitting — HMM states are unordered.
We label by sorting on average abs-return and average volatility.

Features fed to the HMM (all standardized):
  1. log return of close (1m)
  2. realized volatility (rolling 30m std of returns)
  3. log ATR (rolling 14m)
  4. order book spread (bps)
  5. order book imbalance
  6. funding rate
  7. taker buy/sell ratio (centered around 1)
  8. session indicator: sin(2π·hour/24)
  9. session indicator: cos(2π·hour/24)

The model is fit once on a backfill of historical bars at daemon startup
(or loaded from disk if recent), and incrementally inferred each cycle.
A pre-trained model from scripts/pretrain_hmm.py is used as a cold-start
fallback so the first day of regime labels isn't garbage.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .oracle_config import OracleConfig

log = logging.getLogger(__name__)


REGIME_LABELS = ["CHOP", "TREND", "BREAKOUT", "CASCADE"]


@dataclass
class RegimeState:
    label: str                         # "CHOP" | "TREND" | "BREAKOUT" | "CASCADE"
    confidence: float                  # max posterior prob across states
    posteriors: dict[str, float]       # full distribution
    feature_snapshot: dict[str, float] # the features that produced this read


@dataclass
class _FittedHMM:
    """Container for a fitted HMM + the metadata needed to use it later."""
    model: object                      # GaussianHMM
    feature_means: np.ndarray          # per-feature mean from training set
    feature_stds: np.ndarray           # per-feature std from training set
    state_label_map: dict[int, str]    # HMM state idx → human label
    fit_time: datetime
    n_train_samples: int
    feature_names: list[str] = field(default_factory=lambda: [
        "log_ret", "realized_vol", "log_atr",
        "spread_bps", "imbalance", "funding_rate", "taker_centered",
        "hour_sin", "hour_cos",
    ])
    # Which of the 9 columns were actually fed to the HMM at fit time. Columns
    # with ~zero training-time variance are dropped before fitting (see
    # fit_hmm's docstring for why) — a GaussianHMM with covariance_type="diag"
    # fit on an exactly-constant dimension produces a singular covariance and
    # `model.fit()` silently returns NaN parameters (reproduced empirically,
    # not theoretical). Defaults to "nothing masked" so any _FittedHMM built
    # without going through fit_hmm's masking logic still behaves like before.
    active_mask: np.ndarray = field(default_factory=lambda: np.ones(9, dtype=bool))
    # Which instrument this was trained on (cfg.dukascopy_symbol, e.g.
    # "GBPJPY" or "XAUUSD"). get_or_fit() checks this against the CURRENT
    # instrument before reusing a cached/pretrained fit — reproduced in
    # practice, not theoretical: switching instruments within the 24h
    # freshness window silently reused a stale cross-instrument model
    # (GBPJPY-fitted means/stds applied to gold's completely different price
    # scale and return distribution) because the old freshness check was
    # purely time-based. Empty string for pickles predating this field —
    # treated as "unknown instrument", always a mismatch, safe fallback.
    symbol: str = ""
    # Sanity bounds — clamp standardized values to ±5σ. Crypto bars produce
    # genuine outliers (5σ+ moves happen daily) that would otherwise dominate
    # the Gaussian emissions and pull state means around. Clamping at fit and
    # inference time keeps the model from over-reacting to single bars.
    standardize_clip: float = 5.0

    def standardize(self, X: np.ndarray) -> np.ndarray:
        """Apply the fitted (mean, std) to new data, with outlier clamping.

        Returns all 9 standardized columns — callers that feed this into the
        HMM must apply `self.active_mask` themselves (fit_hmm and classify()
        both do). Kept separate from masking so `feature_snapshot` /
        debug output can still show all 9 raw standardized values.
        """
        safe = np.where(self.feature_stds < 1e-9, 1.0, self.feature_stds)
        z = (X - self.feature_means) / safe
        return np.clip(z, -self.standardize_clip, self.standardize_clip)


# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------


def _build_features(
    ohlcv: pd.DataFrame,
    book_imbalance: Optional[float] = None,
    book_spread_bps: Optional[float] = None,
    funding_rate: Optional[float] = None,
    taker_ratio: Optional[float] = None,
) -> np.ndarray:
    """Build the feature matrix of shape (T, 9) from an OHLCV buffer.

    Features 1–7 are price/microstructure. 8–9 are session encoding (sin/cos of
    UTC hour). BTC has well-documented session-of-day structure — Asia chop vs
    London/NY volatility — and giving the HMM this signal lets it cleanly
    separate "regime change because of market mechanics" from "regime change
    because Asia just woke up."

    Order book + funding fields are scalars (current snapshot); we broadcast
    them as the most-recent value. For historical fitting we pass them as None
    and zero-fill.
    """
    close = ohlcv["close"].astype(float).to_numpy()
    high = ohlcv["high"].astype(float).to_numpy()
    low = ohlcv["low"].astype(float).to_numpy()

    log_close = np.log(close)
    log_ret = np.diff(log_close, prepend=log_close[0])

    # Rolling realized vol over 30 bars
    s = pd.Series(log_ret)
    realized_vol = s.rolling(window=30, min_periods=5).std().bfill().to_numpy()

    # ATR (14)
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ])
    atr = pd.Series(tr).rolling(window=14, min_periods=5).mean().bfill().to_numpy()
    log_atr = np.log(np.maximum(atr, 1e-9))

    T = len(close)
    spread_arr = np.full(T, book_spread_bps if book_spread_bps is not None else 1.0, dtype=float)
    imb_arr = np.full(T, book_imbalance if book_imbalance is not None else 0.0, dtype=float)
    fund_arr = np.full(T, funding_rate if funding_rate is not None else 0.0, dtype=float)
    taker_arr = np.full(T, (taker_ratio - 1.0) if taker_ratio is not None else 0.0, dtype=float)

    # Session encoding (hour-of-day on the unit circle). Robust to all
    # timestamp formats we use (ms epoch, datetime, ISO string).
    ts = ohlcv["timestamps"]
    if not pd.api.types.is_datetime64_any_dtype(ts):
        ts = pd.to_datetime(ts, utc=True, errors="coerce")
    # Convert any timezone-naive timestamps to UTC; preserve UTC if already set
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    hours = ts.dt.hour.to_numpy().astype(float) + ts.dt.minute.to_numpy().astype(float) / 60.0
    hour_sin = np.sin(2 * np.pi * hours / 24.0)
    hour_cos = np.cos(2 * np.pi * hours / 24.0)

    return np.column_stack([
        log_ret,
        realized_vol,
        log_atr,
        spread_arr,
        imb_arr,
        fund_arr,
        taker_arr,
        hour_sin,
        hour_cos,
    ])


# ---------------------------------------------------------------------------
# Fit and label
# ---------------------------------------------------------------------------


def _label_states(model, X_std_active: np.ndarray, X_std_full: np.ndarray) -> dict[int, str]:
    """Sort states by characteristic activity → assign human labels.

    Strategy: compute (mean abs log_ret, mean realized_vol) per state from
    training data; sort low→high; assign CHOP, TREND, BREAKOUT, CASCADE.

    `X_std_active` (whatever columns the model was actually fit on) is what
    `model.predict()` needs. `X_std_full` (always all 9 columns, in the
    original log_ret/realized_vol/... order) is what the activity metric
    reads from — log_ret and realized_vol are columns 0 and 1 there
    regardless of what got masked out elsewhere.
    """
    states = model.predict(X_std_active)
    n_states = model.n_components

    metrics = []
    for s in range(n_states):
        mask = states == s
        if not mask.any():
            metrics.append((s, 0.0, 0.0))
            continue
        abs_ret = float(np.abs(X_std_full[mask, 0]).mean())
        vol = float(X_std_full[mask, 1].mean())
        # Combined activity score
        activity = abs_ret + vol
        metrics.append((s, activity, abs_ret))

    metrics.sort(key=lambda row: row[1])  # ascending
    # If we have exactly 4 states, label them in order; otherwise truncate.
    labels = REGIME_LABELS[: len(metrics)] + ["UNKNOWN"] * max(0, len(metrics) - len(REGIME_LABELS))
    return {row[0]: labels[i] for i, row in enumerate(metrics)}


def fit_hmm(
    ohlcv: pd.DataFrame,
    n_states: int = 4,
    seed: int = 42,
    symbol: str = "",
) -> _FittedHMM:
    """Fit a Gaussian HMM on the historical OHLCV buffer.

    The training statistics (mean, std) are frozen at fit time and applied
    identically at inference. This prevents the live regime classifier from
    drifting silently as the live buffer's distribution moves away from the
    training distribution. Outlier clamp at ±5σ on standardized features
    prevents single 5σ+ bars (crypto and FX both produce these — FX gets
    them at weekend-reopen gaps) from dominating the Gaussian emissions.

    Order-book/funding/taker columns are constant (zero-fill) during
    historical fitting — see _build_features's docstring — and a
    covariance_type="diag" GaussianHMM fit on an exactly-constant dimension
    produces a singular covariance: model.fit() silently returns NaN
    parameters (reproduced empirically fitting on real FX history, not a
    theoretical concern). Columns with ~zero training variance are therefore
    dropped before fitting; classify() applies the same mask at inference.

    Raises ImportError if hmmlearn is not installed.
    Raises ValueError if log_ret or realized_vol (the two columns the model
    cannot function without) end up with ~zero variance — that means the
    input data itself is degenerate (e.g. a frozen/duplicated price feed),
    not something to silently paper over.
    """
    from hmmlearn.hmm import GaussianHMM

    X = _build_features(ohlcv)
    if len(X) < 100:
        raise ValueError(f"Need at least 100 bars to fit HMM, got {len(X)}")

    means = X.mean(axis=0)
    stds = X.std(axis=0)

    # Shell first so we can use its .standardize() consistently between fit and inference.
    fitted = _FittedHMM(
        model=None,                # filled below
        feature_means=means,
        feature_stds=stds,
        state_label_map={},
        fit_time=datetime.now(timezone.utc),
        n_train_samples=len(X),
        symbol=symbol,
    )
    X_std = fitted.standardize(X)

    active_mask = stds > 1e-9
    if not active_mask[0] or not active_mask[1]:
        raise ValueError(
            "log_ret and/or realized_vol have ~zero variance in the training "
            "data — the input OHLCV looks degenerate (frozen prices?), not "
            "just missing the optional microstructure columns."
        )
    dropped = [name for name, keep in zip(fitted.feature_names, active_mask) if not keep]
    if dropped:
        log.info(
            "HMM fit: dropping constant-at-training-time columns %s (no historical "
            "data for these — they're only populated at live inference time).",
            dropped,
        )
    fitted.active_mask = active_mask
    X_std_active = X_std[:, active_mask]

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=80,
        random_state=seed,
        tol=1e-3,
    )
    log.info("Fitting HMM (%d states, %d samples, %d/%d active features)",
             n_states, len(X_std_active), int(active_mask.sum()), len(active_mask))
    model.fit(X_std_active)
    fitted.model = model

    label_map = _label_states(model, X_std_active, X_std)
    fitted.state_label_map = label_map
    log.info("HMM state labels: %s", label_map)
    log.debug("Feature means: %s", dict(zip(fitted.feature_names, means.round(4))))
    log.debug("Feature stds:  %s", dict(zip(fitted.feature_names, stds.round(4))))

    return fitted


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _model_path(cfg: OracleConfig) -> Path:
    return cfg.state_dir / "regime_hmm.pkl"


def save_fitted(fitted: _FittedHMM, cfg: OracleConfig) -> None:
    cfg.ensure_state_dir()
    path = _model_path(cfg)
    with open(path, "wb") as f:
        pickle.dump(fitted, f)
    log.info("HMM model saved to %s", path)


def load_fitted(cfg: OracleConfig) -> Optional[_FittedHMM]:
    path = _model_path(cfg)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        log.warning("Could not load cached HMM (%s); will refit.", e)
        return None


def is_fit_fresh(fitted: _FittedHMM, max_age_hours: float = 24.0) -> bool:
    """Whether the fitted model is fresh enough to skip refitting."""
    if fitted is None:
        return False
    age = datetime.now(timezone.utc) - fitted.fit_time
    return age.total_seconds() < max_age_hours * 3600


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def classify(
    fitted: _FittedHMM,
    ohlcv: pd.DataFrame,
    book_imbalance: Optional[float] = None,
    book_spread_bps: Optional[float] = None,
    funding_rate: Optional[float] = None,
    taker_ratio: Optional[float] = None,
) -> RegimeState:
    """Classify the regime of the most recent bar.

    Returns a RegimeState with the chosen label, max-posterior confidence,
    and the full posterior distribution. The 'features' field shows the
    standardized values that drove the call (useful for debugging).
    """
    X = _build_features(
        ohlcv,
        book_imbalance=book_imbalance,
        book_spread_bps=book_spread_bps,
        funding_rate=funding_rate,
        taker_ratio=taker_ratio,
    )
    X_std = fitted.standardize(X)
    X_std_active = X_std[:, fitted.active_mask]

    # Posteriors at the last time-step (forward-pass smoothed up to T).
    # GaussianHMM.predict_proba returns (T, n_states).
    posteriors_t = fitted.model.predict_proba(X_std_active)[-1]

    posteriors = {
        fitted.state_label_map.get(s, f"STATE_{s}"): float(posteriors_t[s])
        for s in range(len(posteriors_t))
    }
    best_state = int(np.argmax(posteriors_t))
    label = fitted.state_label_map.get(best_state, f"STATE_{best_state}")
    conf = float(posteriors_t[best_state])

    feature_snapshot = {
        "log_ret":         float(X[-1, 0]),
        "realized_vol":    float(X[-1, 1]),
        "log_atr":         float(X[-1, 2]),
        "spread_bps":      float(X[-1, 3]),
        "imbalance":       float(X[-1, 4]),
        "funding_rate":    float(X[-1, 5]),
        "taker_centered":  float(X[-1, 6]),
        "hour_sin":        float(X[-1, 7]),
        "hour_cos":        float(X[-1, 8]),
    }

    return RegimeState(
        label=label,
        confidence=conf,
        posteriors=posteriors,
        feature_snapshot=feature_snapshot,
    )


# ---------------------------------------------------------------------------
# Convenience: get-or-fit
# ---------------------------------------------------------------------------


def _pretrained_path(cfg: OracleConfig) -> Path:
    return cfg.state_dir / "regime_hmm_pretrained.pkl"


def load_pretrained(cfg: OracleConfig) -> Optional[_FittedHMM]:
    """Load a pre-trained HMM written by scripts/pretrain_hmm.py, if present.

    Unlike load_fitted(), the pretrained model never expires by clock — it's
    a long-history fit that's expected to outlive single-day refits.
    """
    path = _pretrained_path(cfg)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        log.warning("Could not load pretrained HMM (%s); ignoring.", e)
        return None


def get_or_fit(
    cfg: OracleConfig,
    ohlcv_for_training: pd.DataFrame,
    max_age_hours: float = 24.0,
) -> _FittedHMM:
    """Load a cached HMM if fresh AND for the current instrument, else a
    matching pretrained one, else fit fresh.

    Resolution order:
        1. ~/.trader-stack/btc-oracle/regime_hmm.pkl       (daily refit cache, if fresh + same instrument)
        2. ~/.trader-stack/btc-oracle/regime_hmm_pretrained.pkl  (offline pretrain, if same instrument)
        3. Fit from the live buffer (slowest path; first cold start, or after an instrument switch)

    The instrument check matters: freshness alone isn't enough. Switching
    instruments (e.g. GBPJPY -> XAUUSD) within the freshness window used to
    silently reuse the OLD instrument's fitted means/stds — reproduced in
    practice, not theoretical (see _FittedHMM.symbol's docstring). A cached
    fit from a different instrument is treated as not-usable regardless of
    how recently it was fit.

    Use this at daemon startup.
    """
    target = cfg.dukascopy_symbol

    cached = load_fitted(cfg)
    if cached is not None:
        if getattr(cached, "symbol", "") != target:
            log.info("Cached HMM was fit on %r, current instrument is %r — ignoring.",
                     getattr(cached, "symbol", "<unknown>"), target)
        elif is_fit_fresh(cached, max_age_hours):
            log.info("Using cached HMM (%d samples, fit %s, symbol=%s)",
                     cached.n_train_samples, cached.fit_time.isoformat(), target)
            return cached

    pretrained = load_pretrained(cfg)
    if pretrained is not None:
        if getattr(pretrained, "symbol", "") != target:
            log.info("Pretrained HMM was fit on %r, current instrument is %r — ignoring.",
                     getattr(pretrained, "symbol", "<unknown>"), target)
        else:
            log.info(
                "Using pretrained HMM (%d samples, fit %s, symbol=%s) — "
                "will refit from live data after %.1fh.",
                pretrained.n_train_samples,
                pretrained.fit_time.isoformat(),
                target,
                max_age_hours,
            )
            # Persist as the active model too so subsequent freshness checks pass.
            save_fitted(pretrained, cfg)
            return pretrained

    log.info("No matching cached or pretrained HMM for %s; fitting from live buffer.", target)
    fitted = fit_hmm(ohlcv_for_training, symbol=target)
    save_fitted(fitted, cfg)
    return fitted
