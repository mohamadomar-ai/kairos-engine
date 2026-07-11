#!/usr/bin/env python3
"""Cost-scenario comparison for the winning M5/0.25xATR manual BB variant,
over 365 days of cached XAUUSD M5 (resampled from Dukascopy M1 cache-first
history) — see scripts/backtest_manual_bb_strategy.py for the strategy
itself (touch/confirm/exit on M5, EMA8 filter on M15, 0.25xATR(M5) stop).

Simulates trades ONCE (entry/exit/gross P&L don't depend on the cost
model) and re-costs the SAME trade list under three CostModel scenarios:

  Standard          — CFI Standard account, unchanged (150 pts London/NY
                       base, 3x Asian multiplier, $0 commission). ASSUMED,
                       not directly confirmed (see backtest.py CostModel).
  Raw confirmed      — CFI Dynamic Trader account. CONFIRMED (broker agent,
                       2026-07-09): spread 15 pts London/NY + commission
                       $9/lot round-turn == 9 pts = 24 pts round-trip.
                       Asian-session 3x multiplier still applied to the
                       15pt spread component (that multiplier itself was
                       NOT separately reconfirmed for this account type).
  Raw conservative   — same confirmed $9 commission, but spread padded to
                       26 pts (widening + slippage buffer, NOT itself
                       CFI-confirmed) for a 35 pt round-trip total.

Both raw scenarios set apply_measured_spread_floor=False on the CostModel
(see backtest.py). The default True flooring — "use the wider of our
assumed spread and Dukascopy's own measured historical spread" — was
calibrated against the Standard account's ASSUMED 150pt figure as a
safety margin. Left on for a CONFIRMED 15pt account, Dukascopy's
wholesale/interbank tick spread (routinely wider than 15pts) would floor
almost every trade back up near the OLD assumption, silently erasing the
whole point of confirming a cheaper account. Caught via a smoke test
where "Raw confirmed 24pts" net expectancy implied an average realized
cost of ~69pts/trade instead of the labeled 24.

Requires >=200 trades (per this run's instruction) — 90 days of M5 gave 59
trades, so 365 days should comfortably clear 200.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from trader_stack.btc_oracle.backtest import CostModel  # noqa: E402
from trader_stack.btc_oracle.oracle_config import ORACLE_CONFIG  # noqa: E402

import replay_backtest as rb  # noqa: E402
from backtest_manual_bb_strategy import (  # noqa: E402
    add_indicators, build_report, compute_regime_labels, print_report,
    resample_ohlcv, rescore_trades, simulate_trades,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s INFO %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("cost_scenario_report")

DAYS = 365
STOP_ATR_MULT = 0.25
BAR_TIMEFRAME = "5min"
FILTER_RULE = "15min"
MIN_TRADES_REQUIRED = 200

SCENARIOS = {
    "Standard (CFI Standard, ASSUMED)": CostModel(),
    "Raw confirmed 24pts (CFI Dynamic Trader, CONFIRMED broker agent 2026-07-09)": CostModel(
        spread_pips=15.0, commission_pips_round_trip=9.0, slippage_pips_per_side=0.0,
        apply_measured_spread_floor=False,
        CONFIRMED_FIELDS=("commission_per_lot_round_trip", "spread_pips", "commission_pips_round_trip"),
        ASSUMED_FIELDS=("asian_session_spread_multiplier", "asian_session_start_hour_utc",
                         "asian_session_end_hour_utc"),
    ),
    "Raw conservative 35pts (padded, NOT CFI-confirmed)": CostModel(
        spread_pips=26.0, commission_pips_round_trip=9.0, slippage_pips_per_side=0.0,
        apply_measured_spread_floor=False,
        CONFIRMED_FIELDS=("commission_per_lot_round_trip", "commission_pips_round_trip"),
        ASSUMED_FIELDS=("spread_pips", "asian_session_spread_multiplier",
                         "asian_session_start_hour_utc", "asian_session_end_hour_utc"),
    ),
}


def main():
    cfg = replace(ORACLE_CONFIG, symbol="XAUUSD")
    sim_cost_model = CostModel()  # cost model used only to drive simulate_trades(); re-scored below per scenario

    t0 = time.monotonic()
    full_df = rb._load_full_history(cfg, DAYS)
    log.info("Resampling to %s bars...", BAR_TIMEFRAME)
    full_df = resample_ohlcv(full_df, BAR_TIMEFRAME)

    log.info("Computing indicators (filter=%s)...", FILTER_RULE)
    full_df = add_indicators(full_df, sim_cost_model, filter_rule=FILTER_RULE)

    log.info("Computing walk-forward regime labels (reporting only)...")
    regime_labels = compute_regime_labels(full_df, cfg)

    log.info("Simulating trades ONCE (stop_atr_mult=%.2f)...", STOP_ATR_MULT)
    trades, censored = simulate_trades(full_df, sim_cost_model, cfg, stop_atr_mult=STOP_ATR_MULT,
                                        require_confirmation=True)
    log.info("Done in %.1fs. %d trades generated.", time.monotonic() - t0, len(trades))

    if len(trades) < MIN_TRADES_REQUIRED:
        log.warning("Only %d trades (< %d required) — reporting anyway, but flag this.",
                     len(trades), MIN_TRADES_REQUIRED)

    for label, cost_model in SCENARIOS.items():
        rescored = rescore_trades(trades, cost_model)
        report = build_report(rescored, regime_labels, cost_model)
        print(f"\n{'='*90}\n{label}\n{'='*90}")
        print_report(report, censored)


if __name__ == "__main__":
    main()
