"""Confidence calibration via isotonic regression.

The problem this solves:
    Raw ensemble confidence is a number the math produces — it has no
    guaranteed relationship to actual hit rate. "0.72 confidence" might
    correspond to 51% accuracy, 68% accuracy, or 80% accuracy. You can't
    trust it for decision-making until you've measured.

The solution:
    Take every (signal, realized_outcome) pair from history. Look at the
    raw confidence the system assigned vs whether the directional call
    was correct. Fit a monotonically-increasing function (isotonic
    regression) that maps raw confidence → realized hit rate.

    Going forward, every new signal's raw confidence is passed through
    this function. The result IS what people expect "72% confidence" to mean.

Why isotonic (not Platt scaling, not a neural net):
    Isotonic regression assumes only that confidence is monotonically
    related to accuracy (higher confidence → higher accuracy on average).
    No parametric assumption. Robust to small samples. Standard practice
    in ML for binary-classifier calibration.

When the calibrator starts working:
    With < 200 resolved outcomes: identity function (cal = raw).
    With 200–500: a simple calibration fits but accuracy gains are modest.
    With 500+: meaningful corrections, useful for trade filtering.

Persistence:
    Fitted models live in Postgres `calibration` table. Daemon restart
    doesn't lose anything. History is kept so you can see how calibration
    drifted over time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import store

log = logging.getLogger(__name__)


# Minimum resolved outcomes before we even attempt to fit a calibrator.
# Below this, identity (cal=raw) is safer than a model fit on tiny data.
MIN_SAMPLES_TO_FIT = 200

# How often to refit (in cycles = minutes). Default 6h.
DEFAULT_REFIT_INTERVAL_CYCLES = 360


# ---------------------------------------------------------------------------
# Fitted model (serializable to JSON for Postgres storage)
# ---------------------------------------------------------------------------


@dataclass
class CalibrationModel:
    """A fitted isotonic regression, represented by its breakpoints.

    Stored in Postgres as JSONB. Tiny — ~1 KB even for hundreds of breakpoints.
    """
    x_breaks: list[float]    # raw-confidence breakpoints (sorted ascending)
    y_breaks: list[float]    # calibrated value at each breakpoint
    n_train_samples: int     # how many (signal, outcome) pairs we fit on
    is_identity: bool        # True if we fell back to identity (insufficient data)

    def apply(self, raw_confidence: float) -> float:
        """Map a raw confidence to its calibrated value via piecewise-linear interpolation."""
        if self.is_identity or not self.x_breaks:
            return float(raw_confidence)
        c = float(np.clip(raw_confidence, 0.0, 1.0))
        cal = float(np.interp(c, self.x_breaks, self.y_breaks))
        return float(np.clip(cal, 0.0, 1.0))

    def to_dict(self) -> dict:
        return {
            "x_breaks": list(self.x_breaks),
            "y_breaks": list(self.y_breaks),
            "n_train_samples": int(self.n_train_samples),
            "is_identity": bool(self.is_identity),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CalibrationModel":
        return cls(
            x_breaks=list(d.get("x_breaks", [])),
            y_breaks=list(d.get("y_breaks", [])),
            n_train_samples=int(d.get("n_train_samples", 0)),
            is_identity=bool(d.get("is_identity", True)),
        )


def identity_model() -> CalibrationModel:
    """The no-op calibrator used when there's not enough data to fit."""
    return CalibrationModel(x_breaks=[], y_breaks=[], n_train_samples=0, is_identity=True)


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------


def _brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Brier score: mean squared error between predicted prob and binary outcome.
    Lower is better. Perfect = 0. Always-coin-flip = 0.25."""
    return float(np.mean((probs - outcomes) ** 2))


def _expected_calibration_error(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> float:
    """ECE: weighted-mean gap between predicted confidence and realized accuracy
    within each confidence bucket. Lower is better. Perfect = 0."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if i == n_bins - 1:
            mask = (probs >= bins[i]) & (probs <= bins[i + 1])  # include the right edge
        if not mask.any():
            continue
        bucket_conf = probs[mask].mean()
        bucket_acc = outcomes[mask].mean()
        ece += (mask.sum() / n) * abs(bucket_conf - bucket_acc)
    return float(ece)


def fit_from_history() -> tuple[CalibrationModel, dict]:
    """Read all resolved (raw_confidence, hit) pairs from Postgres and fit.

    Returns (model, diagnostics). If insufficient data, returns identity_model().

    Diagnostics include n_samples, brier_score, ece, and a notes string.
    """
    diagnostics = {"n_samples": 0, "brier_score": None, "ece": None, "notes": ""}

    # Pull ALL signals that have outcomes. For now we just take the most
    # recent 5000 — enough for a stable fit, fast to fetch.
    rows = store.query_recent_with_outcomes(n=5000)

    # We calibrate using raw_confidence (pre-filter) — that's the number the
    # ensemble produced. Post-filter confidence is what we DISPLAY, but we
    # calibrate the underlying signal.
    # For directional signals only: a "hit" is when raw_direction == actual.
    # FLAT signals don't have a meaningful hit rate to calibrate against here.
    pairs = []
    for r in rows:
        raw_dir = r.get("raw_direction") or r.get("direction")
        raw_conf = r.get("raw_confidence")
        actual = r.get("actual_direction")
        if raw_dir == "FLAT" or raw_conf is None or actual is None:
            continue
        hit = 1.0 if raw_dir == actual else 0.0
        pairs.append((float(raw_conf), hit))

    diagnostics["n_samples"] = len(pairs)

    if len(pairs) < MIN_SAMPLES_TO_FIT:
        diagnostics["notes"] = f"Insufficient samples ({len(pairs)} < {MIN_SAMPLES_TO_FIT}); using identity"
        log.info("Calibration: %s", diagnostics["notes"])
        return identity_model(), diagnostics

    try:
        from sklearn.isotonic import IsotonicRegression
    except ImportError:
        diagnostics["notes"] = "scikit-learn not installed; using identity"
        log.warning("Calibration: %s", diagnostics["notes"])
        return identity_model(), diagnostics

    x = np.array([p[0] for p in pairs])
    y = np.array([p[1] for p in pairs])

    # Isotonic regression — out_of_bounds='clip' means values outside the
    # training range get mapped to the nearest endpoint, not extrapolated.
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(x, y)

    # Extract breakpoints — the unique x values at which the isotonic
    # function changes, plus their y values. This is the compact serializable
    # representation. (sklearn's IsotonicRegression stores these internally
    # as X_thresholds_ and y_thresholds_.)
    x_breaks = iso.X_thresholds_.tolist()
    y_breaks = iso.y_thresholds_.tolist()

    model = CalibrationModel(
        x_breaks=x_breaks,
        y_breaks=y_breaks,
        n_train_samples=len(pairs),
        is_identity=False,
    )

    # Diagnostics: how well does the fit do on the training data?
    y_pred = np.array([model.apply(xi) for xi in x])
    diagnostics["brier_score"] = _brier_score(y_pred, y)
    diagnostics["ece"] = _expected_calibration_error(y_pred, y)
    diagnostics["notes"] = f"Fitted on {len(pairs)} samples"

    log.info("Calibration fitted: n=%d  brier=%.4f  ece=%.4f",
             len(pairs), diagnostics["brier_score"], diagnostics["ece"])

    return model, diagnostics


def fit_and_persist() -> Optional[CalibrationModel]:
    """Fit a fresh calibrator and persist it to Postgres. Returns the model
    on success, None on failure (which falls back to whatever was already loaded)."""
    try:
        model, diag = fit_from_history()
        store.write_calibration(
            model_json=model.to_dict(),
            n_training_samples=diag["n_samples"],
            brier_score=diag["brier_score"],
            ece=diag["ece"],
            notes=diag["notes"],
        )
        return model
    except Exception as e:
        log.warning("Calibration fit_and_persist failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Load (used by daemon at startup + on each periodic refit)
# ---------------------------------------------------------------------------


def load_latest() -> CalibrationModel:
    """Load the latest fitted calibrator from Postgres, or return identity."""
    row = store.read_latest_calibration()
    if row is None or not row.get("model_json"):
        return identity_model()
    try:
        return CalibrationModel.from_dict(row["model_json"])
    except Exception as e:
        log.warning("Could not parse stored calibration model: %s — using identity", e)
        return identity_model()
