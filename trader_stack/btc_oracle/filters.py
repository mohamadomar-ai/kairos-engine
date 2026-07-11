"""Trade-filter layer — the discipline that turns a noisy 53% signal into a 57% one.

A signal that would have been "UP @ 51% confidence" usually shouldn't be acted on.
This module applies a sequence of gates; if any gate fires, the final signal is
forced to FLAT and the reason recorded in `notes`.

Gates (applied in order, stop-on-first-FLAT):

1. **Confidence floor**       — raw ensemble confidence < min_confidence → FLAT
2. **Volatility noise floor** — predicted move below the noise band → FLAT
3. **Regime gate**            — regime is CHOP (low-edge) → FLAT
                                (configurable per-direction overrides)
4. **Spread / liquidity gate**— spread blown out → FLAT (untradeable regardless)

Each gate is intentionally simple and inspectable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class FilterResult:
    direction: str                   # final direction after filtering: "UP" | "DOWN" | "FLAT"
    confidence: float                # final confidence (may be downscaled)
    fired_gates: list[str]           # ordered list of gates that triggered
    notes: list[str]                 # human-readable reasons

    @property
    def passed(self) -> bool:
        return not self.fired_gates and self.direction != "FLAT"


def apply_filters(
    raw_direction: str,
    raw_confidence: float,
    blended_pct: float,
    expected_noise_pct: float,
    regime_label: Optional[str],
    regime_confidence: float,
    spread_bps: Optional[float],
    *,
    min_confidence: float,
    spread_bps_max: float,
    noise_floor_multiplier: float,
    chop_blocks_signals: bool,
) -> FilterResult:
    """Run the gate sequence and return the final direction + confidence.

    Args:
        raw_direction: ensemble-blended direction before filtering.
        raw_confidence: ensemble confidence before filtering.
        blended_pct: predicted % change magnitude (already signed).
        expected_noise_pct: from VolatilityForecast.expected_noise_band_pct.
        regime_label: from RegimeState.label, or None if regime classifier off.
        regime_confidence: from RegimeState.confidence (0–1).
        spread_bps: current order book spread in bps, or None if no book data.
        min_confidence: minimum ensemble confidence to fire a signal.
        spread_bps_max: any wider than this → market untradeable.
        noise_floor_multiplier: blended_pct must be ≥ this × noise_band to fire.
        chop_blocks_signals: if True, regime=CHOP forces FLAT.
    """
    fired: list[str] = []
    notes: list[str] = []

    if raw_direction == "FLAT":
        # Even FLAT signals pass through but we still annotate.
        notes.append("Raw signal already FLAT")
        return FilterResult("FLAT", raw_confidence, fired, notes)

    # Gate 1 — Confidence floor
    if raw_confidence < min_confidence:
        fired.append("confidence_floor")
        notes.append(
            f"Confidence {raw_confidence:.2f} < floor {min_confidence:.2f}"
        )
        return FilterResult("FLAT", raw_confidence, fired, notes)

    # Gate 2 — Volatility noise floor
    if expected_noise_pct > 0:
        threshold = expected_noise_pct * noise_floor_multiplier
        if abs(blended_pct) < threshold:
            fired.append("noise_floor")
            notes.append(
                f"|forecast| {abs(blended_pct):.4f} < noise×{noise_floor_multiplier} "
                f"({threshold:.4f}); below noise band"
            )
            return FilterResult("FLAT", raw_confidence, fired, notes)

    # Gate 3 — Regime
    if regime_label is not None and chop_blocks_signals:
        if regime_label == "CHOP":
            fired.append("regime_chop")
            notes.append(f"Regime CHOP @ {regime_confidence:.2f} — low edge, skip")
            return FilterResult("FLAT", raw_confidence, fired, notes)
        # CASCADE regime is tradeable but high-risk; we don't block, just annotate
        if regime_label == "CASCADE":
            notes.append(
                f"Regime CASCADE @ {regime_confidence:.2f} — high-vol; "
                f"consider tighter stops"
            )

    # Gate 4 — Spread / liquidity
    if spread_bps is not None and spread_bps > spread_bps_max:
        fired.append("spread_blowout")
        notes.append(
            f"Spread {spread_bps:.1f} bps > max {spread_bps_max:.1f} bps; market thin"
        )
        return FilterResult("FLAT", raw_confidence, fired, notes)

    # All gates passed — keep the signal. Optionally lift confidence slightly when
    # regime aligns favorably (TREND/BREAKOUT for directional signals).
    final_conf = raw_confidence
    if regime_label in {"TREND", "BREAKOUT"}:
        final_conf = min(1.0, raw_confidence + 0.05)
        notes.append(f"Regime {regime_label} aligns; +0.05 confidence")

    return FilterResult(raw_direction, final_conf, fired, notes)
