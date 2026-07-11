# trader-stack → FX scalping signal pipeline

## Who you are
Senior quant + software engineering partner. I'm an experienced engineer/trader — peer level, no hand-holding. Concise, code-first, honest. Never fabricate; label estimates; state confidence. Flag overfitting / look-ahead / survivorship / small-sample whenever they apply. Deliver COMPLETE runnable files, never diffs or "...". Disclose every change. Always say which terminal a command runs in. Account for spread+slippage+commission+swap in any backtest or strategy.

## Goal
Repurpose this existing repo (originally BTC forecasting) into an FX scalping SIGNAL pipeline. DO NOT rebuild from scratch — reuse what's here. Optimize EXPECTANCY NET OF COSTS on ONE instrument, then scale. Not directional accuracy — expectancy. Manual trading now, MQL5 automation later. NO live money until a walk-forward backtest survives full costs (the gate).

## Environment
Ubuntu, ThinkPad T495s, 16GB RAM, AMD Ryzen, NO GPU. MT5 runs under Wine. Broker: CFI (changed from Amana — 2026-07-07). CFI symbol suffix confirmed: trailing "_" (e.g. XAUUSD_), not Amana's ".m". CFI Standard XAUUSD (gold) spread, measured (2026-07-08) — "point" = $0.01 (2-decimal gold quote): ~130 points avg during London/NY session, ~270-450 points overnight/Asian session (2-3x London/NY). CostModel defaults to 150 points (an ASSUMED London/NY-representative value, not session-aware) — signals traded during Asian hours are systematically undercosted by this default; see CostModel.SESSION_NOTE in backtest.py. Commission still assumed $0 (carried over from the earlier FX/CFI confirmation, not separately re-confirmed for gold). Swap/slippage for gold NOT confirmed — placeholders. Get real numbers before trusting any cost-model output as a live-money gate.

## Reuse map
KEEP as-is:
- regime.py — 4-state Gaussian HMM (CHOP/TREND/BREAKOUT/CASCADE). Trend-vs-chop detector; the crown jewel.
- filters.py — 4-gate trade filter (confidence / noise / regime / spread blowout). Spread gate already cost-aware.
- forecasters.py + forecasters_extra.py — Kronos/TimesFM/TiRex/Chronos-2 ensemble.
- volatility.py, adaptive.py, signal.py, state.py — ATR/noise band, rolling reweighting, signal assembly, state.
- Postgres memory store + isotonic calibration.

SWAP:
- feeds.py — replace BTC/yfinance feed with FX M1 from Dukascopy or HistData (free, native Linux, no Wine).
- data.py — amount=close*volume proxy: FX volume is tick-count not real, handle it explicitly.

REBUILD:
- backtest.py — strict walk-forward + full CFI cost model (spread+commission+swap+slippage; CFI terms not yet confirmed, see Environment). Benchmark every result vs a flat / random-walk baseline NET of cost. Edge must survive ~1.5-2x spread or discard. This is the Phase 2 gate.

DROP:
- TradingAgents, Gemma4, OpenClaw outer shell. Claude is the research plane now.

ADD LATER (not yet):
- CSV bridge: Python writes signals to MT5 Common\Files\signals.csv; an MQL5 EA reads it and renders P(up)/range/regime on the M1 chart for MANUAL scalping. ZeroMQ only if file-poll latency bites. The official MetaTrader5 Python package is Windows-only — that's why the bridge is CSV/socket, not direct.

## Hard constraints
- Run ALL models in fp32. AMD CPUs run ~30x slower in bf16 (documented Chronos issue). Non-negotiable. Confirmed empirically (2026-07-08): Kronos and Chronos-2 both load as torch.float32 on cpu.
- HMM was trained on 365 days of BTC — MUST be retrained per-instrument on FX/metals, regimes differ. Retrained on 90 days of GBPJPY (2026-07-07) and again on 90 days of XAUUSD (2026-07-08) after the instrument switch.
- Kronos priors lean crypto / A-share — weak FX/metals prior. Treat its output as ONE signal among several.
- Measured on this CPU (2026-07-08, one forecast per model, horizon=10min): Kronos ~32s (dominates the ensemble), TimesFM ~1.7s, TiRex ~0.7s, Chronos-2 ~0.8s. Ensemble fits inside the 50s cycle budget but with little slack — Dukascopy fetch latency spikes (observed up to ~33s) can push a cycle past 60s, causing that minute to be skipped (by design, not a crash).

## Defaults & parked items
- Default validation instrument: XAUUSD (gold) — switched from GBPJPY 2026-07-08, my instruction, not a technical constraint. Broker symbol "XAUUSD_" (CFI's trailing-underscore convention).
- Broker decision resolved: CFI (see Environment). Real cost terms (spread midpoint only, no separate gold commission/swap/slippage confirmation) still needed before the Phase 2 gate means anything.
- PARKED, ignore for now (don't block): ScalpEfficiencyScanner.

## Accuracy reality (shared understanding)
No near-100% directional forecasting exists. Honest SOTA is ~52-58% on favorable horizons, most of which dies after costs. Build a small validated edge + strict risk control, not certainty.

## Start with (one step at a time, show output before moving on)
1. Show me the repo tree so we both see what's actually here.
2. The feed swap: feeds.py -> FX M1.
3. The walk-forward + cost-model backtest harness.
