#!/usr/bin/env python3
"""trader-stack BTC oracle — top-level CLI.

Subcommands (all support --json for OpenClaw skills):

    oracle.py daemon start          # spawn background daemon
    oracle.py daemon stop           # stop background daemon
    oracle.py daemon status         # is it running? how fresh is the state?
    oracle.py daemon run            # foreground daemon (used internally by `start`)

    oracle.py forecast              # one-shot forecast, no daemon
    oracle.py status                # alias for `daemon status`
    oracle.py config show           # print effective config

    oracle.py backtest              # accuracy + P&L over recent signal history
    oracle.py backtest --n 500      # over the last 500 signals

The skills under skills/btc-oracle-* shell out to this script.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _setup_logging(verbose: bool, json_mode: bool) -> None:
    logging.basicConfig(
        # In JSON mode, all logs go to stderr so stdout stays parseable.
        stream=sys.stderr,
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def _cmd_daemon(args) -> int:
    from trader_stack.btc_oracle import daemon
    from trader_stack.btc_oracle.oracle_config import ORACLE_CONFIG

    if args.daemon_action == "run":
        daemon.run_forever(ORACLE_CONFIG)
        return 0

    if args.daemon_action == "start":
        pid = daemon.start_background(ORACLE_CONFIG)
        result = {"action": "start", "pid": pid, "log": str(ORACLE_CONFIG.daemon_log)}
        print(json.dumps(result, indent=2) if args.as_json else
              f"Started oracle daemon (PID={pid}). Logs: {ORACLE_CONFIG.daemon_log}")
        return 0

    if args.daemon_action == "stop":
        ok = daemon.stop_background(ORACLE_CONFIG)
        result = {"action": "stop", "stopped": ok}
        print(json.dumps(result, indent=2) if args.as_json else
              ("Stopped." if ok else "Stop failed; check daemon log."))
        return 0 if ok else 1

    if args.daemon_action == "status":
        st = daemon.status_background(ORACLE_CONFIG)
        if args.as_json:
            print(json.dumps(st, indent=2, default=str))
        else:
            print(f"alive: {st['alive']}  pid: {st['pid']}")
            print(f"state freshness: {st['state_freshness_seconds']:.1f}s ago"
                  if st["state_freshness_seconds"] is not None else "no state yet")
            if st["latest_signal"]:
                ls = st["latest_signal"]
                print(f"signal: {ls['direction']} @ confidence={ls['confidence']:.2f}")
                print(f"consensus: {ls['consensus_pct_change']:+.3%}")
        return 0

    print(f"Unknown daemon action: {args.daemon_action}", file=sys.stderr)
    return 2


def _cmd_forecast(args) -> int:
    from trader_stack.btc_oracle.daemon import run_once
    from trader_stack.btc_oracle.oracle_config import ORACLE_CONFIG

    result = run_once(ORACLE_CONFIG)
    if args.as_json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"Direction: {result['direction']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Consensus forecast: {result['consensus_pct_change']:+.3%}")
        print("Per-model:")
        for name, info in result["per_model"].items():
            print(f"  {name:10s} {info['vote']:5s} ({info['pct_change']:+.3%})")
        print("Notes:")
        for n in result["notes"]:
            print(f"  - {n}")
    return 0


def _cmd_status(args) -> int:
    from trader_stack.btc_oracle import daemon
    from trader_stack.btc_oracle.oracle_config import ORACLE_CONFIG

    st = daemon.status_background(ORACLE_CONFIG)
    if args.as_json:
        print(json.dumps(st, indent=2, default=str))
        return 0

    print(f"alive: {st['alive']}  pid: {st['pid']}")
    fresh = st["state_freshness_seconds"]
    print(f"state freshness: {fresh:.1f}s" if fresh is not None else "no state yet")
    sig = st["latest_signal"]
    if sig:
        print(f"\nLatest signal ({sig['timestamp']}):")
        print(f"  last close:  ${sig['last_close']:,.2f}")
        print(f"  direction:   {sig['direction']}  (raw: {sig.get('raw_direction', '?')})")
        print(f"  confidence:  {sig['confidence']:.2f}  (raw: {sig.get('raw_confidence', 0):.2f})")
        print(f"  consensus:   {sig['consensus_pct_change']:+.3%}")
        print(f"  horizon:     {sig['horizon_minutes']} minutes")
        if sig.get("regime"):
            r = sig["regime"]
            print(f"  regime:      {r['label']} @ {r['confidence']:.2f}")
        if sig.get("volatility"):
            v = sig["volatility"]
            print(f"  vol band:    ±{v['expected_noise_band_pct']:.3%} (ATR exp {v['atr_expansion']:.2f})")
        if sig.get("filter_fired"):
            print(f"  filters:     {', '.join(sig['filter_fired'])}")
        print("  per-model:")
        for n, info in sig["per_model"].items():
            print(f"    {n:10s} {info['vote']:5s} ({info['pct_change']:+.3%}  w={info['weight']:.2f})")
        if sig["microstructure"]:
            m = sig["microstructure"]
            print("  microstructure:")
            for k, v in m.items():
                if isinstance(v, float):
                    print(f"    {k}: {v:.4f}")
                else:
                    print(f"    {k}: {v}")
    return 0


def _cmd_config(args) -> int:
    from trader_stack.btc_oracle.oracle_config import ORACLE_CONFIG

    payload = {
        "symbol": ORACLE_CONFIG.symbol,
        "dukascopy_symbol": ORACLE_CONFIG.dukascopy_symbol,
        "horizon_minutes": ORACLE_CONFIG.horizon_minutes,
        "flat_threshold": ORACLE_CONFIG.flat_threshold,
        "buffer_bars": ORACLE_CONFIG.buffer_bars,
        "models": {
            "kronos": ORACLE_CONFIG.use_kronos,
            "timesfm": ORACLE_CONFIG.use_timesfm,
            "tirex": ORACLE_CONFIG.use_tirex,
            "chronos2": ORACLE_CONFIG.use_chronos2,
        },
        "weights": ORACLE_CONFIG.weights,
        "microstructure_weight": ORACLE_CONFIG.microstructure_weight,
        "funding_extreme_threshold": ORACLE_CONFIG.funding_extreme_threshold,
        "phase2": {
            "use_regime": ORACLE_CONFIG.use_regime,
            "regime_refit_interval_cycles": ORACLE_CONFIG.regime_refit_interval,
            "use_trade_filter": ORACLE_CONFIG.use_trade_filter,
            "min_confidence": ORACLE_CONFIG.min_confidence,
            "noise_floor_multiplier": ORACLE_CONFIG.noise_floor_multiplier,
            "spread_bps_max": ORACLE_CONFIG.spread_bps_max,
            "chop_blocks_signals": ORACLE_CONFIG.chop_blocks_signals,
            "adaptive_weighting": ORACLE_CONFIG.adaptive_weighting,
            "adaptive_refit_interval_cycles": ORACLE_CONFIG.adaptive_refit_interval,
        },
        "state_dir": str(ORACLE_CONFIG.state_dir),
    }
    if args.as_json:
        print(json.dumps(payload, indent=2))
    else:
        for k, v in payload.items():
            print(f"{k}: {v}")
    return 0


def _cmd_telemetry(args) -> int:
    """Scan recent telemetry records, summarize health, surface anomalies."""
    from datetime import datetime, timezone
    from pathlib import Path
    from trader_stack.btc_oracle.oracle_config import ORACLE_CONFIG

    state_dir = ORACLE_CONFIG.state_dir
    # Look across today and yesterday; daemon rotates daily.
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    candidates = sorted(Path(state_dir).glob("telemetry_*.jsonl"))
    if not candidates:
        msg = "No telemetry files found. Has the daemon been running with this build?"
        print(json.dumps({"error": msg}) if args.as_json else msg)
        return 1

    # Read the tail of the most recent file(s)
    lines: list[dict] = []
    for path in reversed(candidates):
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        lines.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except Exception:
            continue
        if len(lines) >= args.n:
            break
    lines = lines[-args.n:]

    if not lines:
        msg = "Telemetry file exists but is empty."
        print(json.dumps({"error": msg}) if args.as_json else msg)
        return 1

    # ---- Compute health summary -----------------------------------------
    n = len(lines)
    cycle_ms = [int(r.get("timings_ms", {}).get("total", 0)) for r in lines]
    over_budget = sum(1 for ms in cycle_ms if ms > ORACLE_CONFIG.cycle_budget_seconds * 1000)
    p50_ms = sorted(cycle_ms)[len(cycle_ms) // 2] if cycle_ms else 0
    p95_ms = sorted(cycle_ms)[int(len(cycle_ms) * 0.95)] if cycle_ms else 0

    forecasters_failed: dict[str, int] = {}
    fetch_failed: dict[str, int] = {}
    chronos2_no_cov = 0
    chronos2_seen = 0
    regime_distribution: dict[str, int] = {}
    suppressed = 0
    by_filter: dict[str, int] = {}
    directions: dict[str, int] = {"UP": 0, "DOWN": 0, "FLAT": 0}

    for r in lines:
        for m in r.get("forecasts", {}).get("failed", []):
            forecasters_failed[m] = forecasters_failed.get(m, 0) + 1
        for src in r.get("fetch_failures", {}):
            fetch_failed[src] = fetch_failed.get(src, 0) + 1
        cov = r.get("chronos2_covariates_attached")
        if cov is not None:
            chronos2_seen += 1
            if cov == 0:
                chronos2_no_cov += 1
        rl = (r.get("regime") or {}).get("label")
        if rl:
            regime_distribution[rl] = regime_distribution.get(rl, 0) + 1
        if r.get("signal", {}).get("suppressed"):
            suppressed += 1
        for gate in r.get("signal", {}).get("filter_fired", []) or []:
            by_filter[gate] = by_filter.get(gate, 0) + 1
        d = r.get("signal", {}).get("final_direction", "FLAT")
        directions[d] = directions.get(d, 0) + 1

    # ---- Detect anomalies -----------------------------------------------
    anomalies: list[str] = []
    if over_budget > n * 0.10:
        anomalies.append(f"⚠ {over_budget}/{n} cycles ({over_budget/n:.0%}) exceeded the {ORACLE_CONFIG.cycle_budget_seconds}s budget")
    if chronos2_seen and chronos2_no_cov / chronos2_seen > 0.50:
        anomalies.append(f"⚠ Chronos-2 ran with 0 covariates {chronos2_no_cov}/{chronos2_seen} cycles — silent degradation")
    for m, count in forecasters_failed.items():
        if count > n * 0.20:
            anomalies.append(f"⚠ {m} failed {count}/{n} cycles ({count/n:.0%}) — investigate")
    for src, count in fetch_failed.items():
        if count > n * 0.30:
            anomalies.append(f"⚠ {src} data fetch failed {count}/{n} cycles — endpoint may be rate-limited")
    if len(regime_distribution) == 1:
        anomalies.append(f"⚠ HMM stuck in single regime ({list(regime_distribution.keys())[0]}) for all {n} cycles — possible HMM collapse")
    if not anomalies:
        anomalies.append("✓ No anomalies detected")

    summary = {
        "cycles_scanned": n,
        "cycle_latency_ms": {"p50": p50_ms, "p95": p95_ms, "over_budget": over_budget},
        "regime_distribution": regime_distribution,
        "signal_distribution": directions,
        "suppressed_count": suppressed,
        "suppression_by_gate": by_filter,
        "forecaster_failures": forecasters_failed,
        "fetch_failures": fetch_failed,
        "chronos2_covariate_health": {
            "cycles_seen": chronos2_seen,
            "cycles_with_zero_covariates": chronos2_no_cov,
        },
        "anomalies": anomalies,
    }

    if args.as_json:
        print(json.dumps(summary, indent=2))
        return 0

    print(f"\nTelemetry health over last {n} cycles\n" + "=" * 50)
    print(f"Cycle latency:  p50={p50_ms}ms  p95={p95_ms}ms  over-budget={over_budget}")
    print(f"Regime distribution: {regime_distribution}")
    print(f"Signal distribution: {directions}  (suppressed={suppressed})")
    if by_filter:
        print(f"Suppression gates: {by_filter}")
    if forecasters_failed:
        print(f"Forecaster failures: {forecasters_failed}")
    if fetch_failed:
        print(f"Fetch failures: {fetch_failed}")
    print(f"Chronos-2 covariate health: {chronos2_seen - chronos2_no_cov}/{chronos2_seen} cycles ok")
    print("\nAnomalies:")
    for a in anomalies:
        print(f"  {a}")
    return 0


def _cost_model_from_args(args):
    from trader_stack.btc_oracle.backtest import CostModel
    return CostModel(
        spread_pips=args.spread_pips,
        commission_per_lot_round_trip=args.commission,
        slippage_pips_per_side=args.slippage_pips,
        swap_long_pips_per_day=args.swap_long_pips,
        swap_short_pips_per_day=args.swap_short_pips,
        asian_session_spread_multiplier=args.asian_session_multiplier,
        asian_session_start_hour_utc=args.asian_session_start_hour_utc,
        asian_session_end_hour_utc=args.asian_session_end_hour_utc,
    )


def _cmd_backtest(args) -> int:
    """Walk-forward backtest / confidence sweep over logged signals, net of
    the CFI cost model. See trader_stack/btc_oracle/backtest.py's module
    docstring for what's CFI-confirmed vs. placeholder in that cost model.
    """
    from trader_stack.btc_oracle.oracle_config import ORACLE_CONFIG

    cost_model = _cost_model_from_args(args)

    # Sweep mode — re-evaluate the existing signals.jsonl at multiple thresholds.
    if args.sweep:
        from trader_stack.btc_oracle.sweep import run_sweep
        sweep_res = run_sweep(ORACLE_CONFIG, last_n=args.n, cost_model=cost_model)
        if args.as_json:
            print(json.dumps(sweep_res.to_dict(), indent=2, default=str))
            return 0
        print(f"\nConfidence sweep over {sweep_res.n_signals_evaluated} paired signals "
              f"(spread={cost_model.spread_pips} pips London/NY, "
              f"x{cost_model.asian_session_spread_multiplier} Asian session, "
              f"commission={cost_model.commission_per_lot_round_trip}, "
              f"slippage={cost_model.slippage_pips_per_side}/side [placeholder])\n")
        print(f"{'thr':>6} | {'total':>7} | {'directional':>12} | {'coverage':>9} | "
              f"{'acc':>8} | {'pnl_raw(pips)':>13} | {'pnl_net(pips)':>13} | {'sharpe':>7}")
        print("-" * 100)
        for r in sweep_res.rows:
            acc_s = f"{r.accuracy_directional:.3f}" if r.accuracy_directional is not None else "  n/a"
            sh_s  = f"{r.sharpe_estimate:.3f}"      if r.sharpe_estimate is not None      else "  n/a"
            print(f"{r.threshold:>6.2f} | {r.n_total:>7d} | {r.n_directional:>12d} | "
                  f"{r.coverage:>9.3%} | {acc_s:>8} | {r.pnl_pips_no_costs:>+13.1f} | "
                  f"{r.pnl_pips_with_costs:>+13.1f} | {sh_s:>7}")
        return 0

    # Standard backtest mode — uses the live filter config as-is.
    from trader_stack.btc_oracle.backtest import run_backtest
    result = run_backtest(ORACLE_CONFIG, last_n=args.n, cost_model=cost_model, fold=args.fold)
    payload = asdict(result)
    if args.as_json:
        print(json.dumps(payload, indent=2, default=str))
        return 0

    print(result.scope_note + "\n")
    if result.small_sample_warning:
        print(f"⚠ {result.small_sample_warning}\n")

    print(f"n_signals: {result.n_signals}")
    print(f"n_directional (non-FLAT): {result.n_directional}")
    print(f"coverage (non-FLAT / total): {result.coverage:.3%}")
    if result.accuracy_directional is not None:
        print(f"accuracy (directional only): {result.accuracy_directional:.3f}")
    if result.accuracy_all is not None:
        print(f"accuracy (all signals):      {result.accuracy_all:.3f}")
    cm = result.cost_model
    print(f"\nCost model: spread={cm['spread_pips']} pips London/NY (ASSUMED, not a single confirmed number), "
          f"x{cm['asian_session_spread_multiplier']} for Asian-session entries "
          f"({cm['asian_session_start_hour_utc']:02d}:00-{cm['asian_session_end_hour_utc']:02d}:00 UTC, ASSUMED), "
          f"commission={cm['commission_per_lot_round_trip']} (CFI-confirmed), "
          f"slippage={cm['slippage_pips_per_side']}/side (PLACEHOLDER), "
          f"swap L/S={cm['swap_long_pips_per_day']}/{cm['swap_short_pips_per_day']} per day (PLACEHOLDER)")
    print(f"P&L (no costs):   {result.pnl_pips_no_costs:+.1f} pips")
    print(f"P&L (net of full cost model): {result.pnl_pips_with_costs:+.1f} pips")

    g = result.gate
    print("\nGate (CLAUDE.md: edge must survive ~1.5-2x spread or discard):")
    if g["gross_expectancy_pips_per_trade"] is not None:
        print(f"  gross expectancy/trade: {g['gross_expectancy_pips_per_trade']:+.2f} pips "
              f"({g['gross_edge_spread_multiple']:.2f}x spread)" if g['gross_edge_spread_multiple'] is not None
              else f"  gross expectancy/trade: {g['gross_expectancy_pips_per_trade']:+.2f} pips")
        print(f"  net expectancy/trade (full cost model): {g['net_expectancy_pips_per_trade']:+.2f} pips")
        print(f"  passes >=1.5x spread: {g['passes_1_5x_spread']}   passes >=2x spread: {g['passes_2x_spread']}   "
              f"net edge positive: {g['net_edge_positive']}")
    else:
        print("  n/a — no directional trades")

    b = result.baseline_random_direction
    if b.get("n_trials"):
        print(f"\nRandom-direction baseline ({b['n_trials']} trials over {b['n_directional']} trades):")
        print(f"  model net P&L: {b['model_pnl_pips']:+.1f} pips   "
              f"random mean: {b['random_mean_pnl_pips']:+.1f} pips   "
              f"random p95: {b['random_p95_pnl_pips']:+.1f} pips")
        print(f"  fraction of random trials beating/matching the model: "
              f"{b['pct_random_trials_beating_or_matching_model']:.3f} "
              "(lower is better — this should be small if there's real edge)")

    if result.folds:
        print(f"\nBy fold ({result.fold_granularity}):")
        for f in result.folds:
            acc_s = f"{f['accuracy_directional']:.3f}" if f["accuracy_directional"] is not None else " n/a "
            exp_s = f"{f['expectancy_pips_per_trade_net']:+.2f}" if f["expectancy_pips_per_trade_net"] is not None else "  n/a "
            print(f"  {f['fold']:10s} n={f['n_signals']:4d}  non_flat={f['n_directional']:4d}  "
                  f"acc={acc_s}  pnl_net={f['pnl_pips_with_costs']:+8.1f} pips  exp/trade={exp_s} pips")

    print("\nConfusion (pred → actual):")
    for pred, row in result.confusion.items():
        for actual, n in row.items():
            if n > 0:
                print(f"  {pred:5s} → {actual:5s}: {n}")
    if result.by_confidence:
        print("\nBy confidence bucket:")
        for row in result.by_confidence:
            print(f"  {row['bucket']:8s}  n={row['n']:4d}  acc={row['accuracy']:.3f}")
    if result.by_regime:
        print("\nBy regime:")
        for r, st in result.by_regime.items():
            acc_s = f"{st['accuracy_directional']:.3f}" if st["accuracy_directional"] is not None else " n/a "
            print(f"  {r:10s} n={st['n']:4d}  non_flat={st['n_directional']:4d}  "
                  f"cov={st['coverage']:.2%}  acc={acc_s}")
    if result.per_model_accuracy:
        print("\nPer-model accuracy:")
        for m, a in result.per_model_accuracy.items():
            if a is not None:
                print(f"  {m:10s}: {a:.3f}")
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(prog="oracle", description="BTC minute oracle")
    p.add_argument("--json", dest="as_json", action="store_true",
                   help="Emit a JSON object on stdout (logs to stderr).")
    p.add_argument("-v", "--verbose", action="store_true")

    sub = p.add_subparsers(dest="cmd", required=True)

    # daemon
    pd_ = sub.add_parser("daemon", help="Daemon lifecycle commands")
    pd_.add_argument("daemon_action",
                     choices=["start", "stop", "status", "run"],
                     help="What to do with the daemon. 'run' is the foreground loop.")

    # forecast
    sub.add_parser("forecast", help="Run one cycle inline (no daemon).")

    # status alias
    sub.add_parser("status", help="Show daemon status + latest signal.")

    # config
    pc = sub.add_parser("config", help="Configuration tools")
    pc.add_argument("config_action", choices=["show"])

    # backtest
    pb = sub.add_parser("backtest", help="Walk-forward backtest of signals.jsonl, net of the CFI cost model")
    pb.add_argument("--n", type=int, default=500,
                    help="How many recent signals to evaluate (default 500).")
    pb.add_argument("--fold", choices=["day", "week", "all"], default="week",
                    help="Walk-forward fold granularity for the per-period breakdown (default week).")
    pb.add_argument("--sweep", action="store_true",
                    help="Sweep min_confidence from 0.50 to 0.80 over the same signals. "
                         "Tells you where the real edge sits before tuning the daemon.")
    # Cost model defaults are pulled from CostModel itself (not re-typed as
    # separate literals here) so this can't drift out of sync with backtest.py
    # — see backtest.py's CostModel docstring for what's confirmed/assumed/
    # placeholder for the currently-configured instrument.
    from trader_stack.btc_oracle.backtest import CostModel as _CM
    _default_cost = _CM()
    pb.add_argument("--spread-pips", type=float, default=_default_cost.spread_pips,
                    help=f"Round-trip spread in pips/points (default {_default_cost.spread_pips} — "
                         "see CostModel docstring for confirmed vs. assumed).")
    pb.add_argument("--commission", type=float, default=_default_cost.commission_per_lot_round_trip,
                    help=f"Round-trip commission per lot (default {_default_cost.commission_per_lot_round_trip} — CFI-confirmed, no commission).")
    pb.add_argument("--slippage-pips", type=float, default=_default_cost.slippage_pips_per_side,
                    help=f"Slippage in pips/points PER SIDE (default {_default_cost.slippage_pips_per_side} — PLACEHOLDER, not CFI-confirmed).")
    pb.add_argument("--swap-long-pips", type=float, default=_default_cost.swap_long_pips_per_day,
                    help=f"Long swap/rollover in pips/points per day crossed (default {_default_cost.swap_long_pips_per_day} — PLACEHOLDER, not CFI-confirmed).")
    pb.add_argument("--swap-short-pips", type=float, default=_default_cost.swap_short_pips_per_day,
                    help=f"Short swap/rollover in pips/points per day crossed (default {_default_cost.swap_short_pips_per_day} — PLACEHOLDER, not CFI-confirmed).")
    pb.add_argument("--asian-session-multiplier", type=float, default=_default_cost.asian_session_spread_multiplier,
                    help=f"Multiplier applied to --spread-pips for trades entered during the Asian session "
                         f"(default {_default_cost.asian_session_spread_multiplier} — ASSUMED, CFI measured "
                         "~270-450pt Asian vs ~130pt London/NY spread, a 2-3x range; see CostModel.SESSION_NOTE).")
    pb.add_argument("--asian-session-start-hour-utc", type=int, default=_default_cost.asian_session_start_hour_utc,
                    help=f"Asian-session window start hour, UTC (default {_default_cost.asian_session_start_hour_utc} — approximate, not CFI-confirmed).")
    pb.add_argument("--asian-session-end-hour-utc", type=int, default=_default_cost.asian_session_end_hour_utc,
                    help=f"Asian-session window end hour, UTC (default {_default_cost.asian_session_end_hour_utc} — approximate, not CFI-confirmed).")

    # telemetry — silent-failure diagnostic
    pt = sub.add_parser("telemetry", help="Inspect recent cycle telemetry — surfaces silent degradations.")
    pt.add_argument("--n", type=int, default=60, help="How many recent cycle records to scan (default 60 ≈ last hour).")

    args = p.parse_args()
    _setup_logging(args.verbose, args.as_json)

    try:
        if args.cmd == "daemon":
            return _cmd_daemon(args)
        if args.cmd == "forecast":
            return _cmd_forecast(args)
        if args.cmd == "status":
            return _cmd_status(args)
        if args.cmd == "config":
            return _cmd_config(args)
        if args.cmd == "backtest":
            return _cmd_backtest(args)
        if args.cmd == "telemetry":
            return _cmd_telemetry(args)
    except KeyboardInterrupt:
        return 130

    print(f"Unknown command: {args.cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
