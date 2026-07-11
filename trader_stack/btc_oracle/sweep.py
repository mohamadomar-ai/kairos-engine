"""Confidence-threshold sweep over the existing signals.jsonl.

The point of this module: the live daemon writes every signal with its
*raw* direction and *raw* confidence (pre-filter), plus the *final* direction
and confidence (post-filter). We can therefore replay history and ask:

    "What would non-FLAT accuracy and net P&L have been if I had
     used min_confidence = 0.55, 0.60, 0.65, 0.70, 0.75, 0.80?"

This answers the question the institutional advice doc raised:

    > Sweep min_confidence from 0.50 to 0.80 in steps of 0.05.

without having to re-run the four forecasters — we just re-apply the gate
threshold to the raw direction/confidence already recorded, and cost the
result with the same CostModel backtest.py uses (see backtest.py's module
docstring for what's CFI-confirmed vs. placeholder in that model).

What this DOESN'T do (deliberately):
  - It doesn't re-run the noise-floor or regime gates. Those gates' outputs
    aren't recorded in the signal log. So sweep results reflect "what if I
    had ONLY changed the confidence floor". To sweep noise-floor or regime
    gates, you need to capture more raw fields in the signal log first.

Use this after the daemon has logged >=200 signals; below that, the
per-bucket counts are too small for the numbers to mean anything.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

from .backtest import CostModel, gross_trade_pips, net_trade_pips, pair_signals_with_outcomes
from .oracle_config import ORACLE_CONFIG, OracleConfig
from .state import read_recent_signals

log = logging.getLogger(__name__)


@dataclass
class SweepRow:
    threshold: float
    n_total: int
    n_directional: int
    coverage: float
    accuracy_directional: Optional[float]
    pnl_pips_no_costs: float
    pnl_pips_with_costs: float
    sharpe_estimate: Optional[float]      # naive: mean trade pnl (pips) / std, not annualized


@dataclass
class SweepResult:
    n_signals_evaluated: int
    cost_model: dict
    rows: list[SweepRow]

    def to_dict(self) -> dict:
        return asdict(self)


def run_sweep(
    cfg: OracleConfig = ORACLE_CONFIG,
    last_n: int = 2000,
    thresholds: tuple[float, ...] = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80),
    cost_model: Optional[CostModel] = None,
) -> SweepResult:
    """Run the threshold sweep over the last `last_n` signals."""
    cost_model = cost_model or CostModel()
    signals = read_recent_signals(n=last_n, cfg=cfg)
    if not signals:
        return SweepResult(0, cost_model.to_dict(), [])

    paired, _ = pair_signals_with_outcomes(cfg, signals)
    if not paired:
        return SweepResult(0, cost_model.to_dict(), [])

    rows: list[SweepRow] = []
    for thr in thresholds:
        eligible = [p for p in paired if p.raw_direction != "FLAT" and p.raw_confidence >= thr]

        n_dir = len(eligible)
        hits = sum(1 for p in eligible if p.raw_direction == p.actual_direction)
        gross_pips = [gross_trade_pips(cfg, p, direction_override=p.raw_direction) for p in eligible]
        net_pips = [net_trade_pips(cfg, cost_model, p, direction_override=p.raw_direction) for p in eligible]

        n_total = len(paired)
        coverage = n_dir / n_total if n_total else 0.0
        acc = hits / n_dir if n_dir > 0 else None

        sharpe = None
        if len(net_pips) >= 10:
            arr = np.array(net_pips)
            sd = float(arr.std())
            if sd > 1e-9:
                sharpe = float(arr.mean() / sd)

        rows.append(SweepRow(
            threshold=thr,
            n_total=n_total,
            n_directional=n_dir,
            coverage=coverage,
            accuracy_directional=acc,
            pnl_pips_no_costs=float(sum(gross_pips)),
            pnl_pips_with_costs=float(sum(net_pips)),
            sharpe_estimate=sharpe,
        ))

    return SweepResult(
        n_signals_evaluated=len(paired),
        cost_model=cost_model.to_dict(),
        rows=rows,
    )
