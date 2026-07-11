#!/usr/bin/env python3
"""Historical walk-forward replay over cached XAUUSD M1 data.

WHAT THIS DOES
    Walks forward bar-by-bar over the full cached Dukascopy M1 history (see
    scripts/pretrain_hmm.py for how that cache was populated), and at every
    `--stride`-th bar (default 3rd) generates a real ensemble forecast using
    ONLY data available up to and including that bar — the same
    run_forecasts()/combine_signal() production pipeline the live daemon
    uses (trader_stack/btc_oracle/signal.py), just fed a historical snapshot
    instead of a live one. Each signal is scored against the realized
    outcome `horizon` minutes later and run through backtest.py's
    session-aware CostModel + gate, exactly like a live-logged signal would
    be — see run_backtest_on_signals() in backtest.py (factored out of
    run_backtest() specifically so this script could reuse the real scoring
    logic instead of reimplementing it).

WHY THIS EXISTS
    signals.jsonl only has whatever the live daemon has accumulated since it
    started — currently a handful of signals, nowhere near enough for a
    meaningful accuracy/expectancy read (see backtest.py's
    MIN_DIRECTIONAL_FOR_CONFIDENCE gate). This regenerates a much larger,
    still-genuinely-walk-forward sample from the 90 days of cached history
    without needing 90 days of real wall-clock time.

NO-LOOKAHEAD, BY CONSTRUCTION
    - Forecast context at bar t is bars[t-context_bars+1 : t+1] — NEVER bar
      t+1 or later. The model predicts forward from the close of bar t.
    - The regime HMM is NOT the single 90-day pretrained fit (that fit's
      Gaussian means/stds were computed over the ENTIRE 90-day window,
      which would leak future distributional information into early-window
      classifications — see regime.py's _FittedHMM.symbol docstring for the
      general shape of this class of bug). Instead this script reproduces
      daemon.py's OWN periodic-refit policy exactly: an initial fit once
      MIN_HMM_BARS (1500) bars of history exist, then a refit every
      REGIME_REFIT_BARS (1440, i.e. daily) bars — each fit using ONLY the
      trailing MIN_HMM_BARS bars available AT THAT POINT in the walk,
      matching daemon.py's `fetch_ohlcv(cfg, limit=1500)` + periodic-refit
      cadence (daemon.py ~line 110 and ~line 297-302). Between refits,
      classify() runs on a rolling context window that also never reaches
      past bar t.
    - Outcome scoring (backtest.py's pair_signals_with_outcomes) only ever
      looks at the realized price at t+horizon, using the SAME logic the
      live backtest already uses.

WHAT'S DELIBERATELY DIFFERENT FROM A LIVE DAEMON RUN
    - Kronos is excluded (per instruction — too slow for a 90-day replay;
      CLAUDE.md measured it at ~32s/forecast, dominating the ensemble).
      Ensemble here is TimesFM + TiRex + Chronos-2 only.
    - No Postgres / calibration store — this replay is self-contained
      (JSONL + a JSON checkpoint only) so it doesn't depend on or pollute
      the live memory store, and calibrated_confidence isn't needed for the
      gate (which scores on raw/post-filter confidence, not calibrated).
    - adaptive_weighting is inert here — this script calls combine_signal()
      directly, not daemon.py's _cycle_once(), so the periodic
      recompute_weights() reweighting never fires. Ensemble weights are
      whatever cfg.weights is at script start, held FIXED for the whole
      replay. A full daemon run would let regime-dependent reweighting
      drift the model mix over 90 days; this replay tests one fixed
      configuration throughout — a deliberate simplification, not an
      oversight.
    - Order book is synthetic: `imbalance` is always 0.0 (neutral) — no
      tick-level order-flow direction survives into an M1 bar. `spread_bps`
      IS real (the bar's own tick-derived mean spread, computed in
      feeds.py's _ticks_to_m1), so the filter layer's spread gate is
      genuinely cost-aware.

RESUMABILITY (power loss happened before — see CLAUDE.md)
    Every processed bar appends one line to --out and atomically rewrites
    --checkpoint with the last completed bar index. On restart (without
    --fresh), the script fast-forwards through the SAME regime-refit
    schedule (cheap — no forecasting, just re-running the ~90 HMM fits that
    would have occurred) up to the checkpointed bar, reconstructing the
    exact fitted-HMM state that would have been active there, then resumes
    real forecasting from the next bar. --fresh wipes prior progress.

USAGE
    # 50-forecast timing test (throwaway output, does not touch the real
    # checkpoint — always use separate --out/--checkpoint paths for this):
    python scripts/replay_backtest.py --max-steps 50 \\
        --out /tmp/replay_timing_test.jsonl \\
        --checkpoint /tmp/replay_timing_test.json --fresh

    # Full run (intended to be launched via nohup — see RUNBOOK / the shell
    # command that launched it):
    python scripts/replay_backtest.py

    # Check progress / score whatever has accumulated so far without
    # advancing the replay (safe to run concurrently with a running batch —
    # read-only against --out):
    python scripts/replay_backtest.py --report-only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal as os_signal
import sys
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trader_stack.btc_oracle import regime as regime_mod  # noqa: E402
from trader_stack.btc_oracle.backtest import CostModel, run_backtest_on_signals  # noqa: E402
from trader_stack.btc_oracle.feeds import MarketSnapshot, OrderBookSnapshot, fetch_ohlcv_range  # noqa: E402
from trader_stack.btc_oracle.oracle_config import ORACLE_CONFIG, OracleConfig  # noqa: E402
from trader_stack.btc_oracle.signal import combine_signal, run_forecasts  # noqa: E402
from trader_stack.btc_oracle.volatility import forecast_volatility  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("replay_backtest")

# Matches daemon.py's regime warm-up / periodic-refit policy exactly (see
# daemon.py's _warm_up_models ~line 110 and _cycle_once's periodic refit
# ~line 297-302) — intentionally identical constants so this replay's
# regime classification matches what a live daemon would have produced
# walking through the same history.
MIN_HMM_BARS = 1500
REGIME_REFIT_BARS = 60 * 24  # daily


# ---------------------------------------------------------------------------
# History loading
# ---------------------------------------------------------------------------


def _load_full_history(cfg: OracleConfig, days: int) -> pd.DataFrame:
    """Load `days` of M1 bars from the Dukascopy cache, one day at a time —
    mirrors scripts/pretrain_hmm.py's fetch_fx_history (same reasoning:
    progress visibility, and one bad/uncached day doesn't sink the whole
    load). Should be near-100% cache hits after pretrain_hmm.py's run; any
    miss falls back to network, same as any other feeds.py caller.
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    log.info("Loading %d days of %s M1 bars (%s -> %s)...", days, cfg.dukascopy_symbol,
              start.date(), end.date())
    frames: list[pd.DataFrame] = []
    day_start = start.replace(hour=0, minute=0, second=0, microsecond=0)
    n_days = 0
    t0 = time.monotonic()
    while day_start < end:
        day_end = min(day_start + timedelta(days=1), end)
        try:
            df = fetch_ohlcv_range(cfg, day_start, day_end)
        except Exception as e:
            log.warning("  %s: fetch failed (%s) — skipping this day", day_start.date(), e)
            df = pd.DataFrame()
        if not df.empty:
            frames.append(df)
        n_days += 1
        if n_days % 14 == 0:
            log.info("  ...%d/%d days loaded (%d bars so far)", n_days, days,
                      sum(len(f) for f in frames))
        day_start = day_end

    if not frames:
        raise RuntimeError(f"No cached/fetchable bars for {cfg.dukascopy_symbol} over the last {days} days.")

    out = pd.concat(frames, ignore_index=True).sort_values("timestamps").reset_index(drop=True)
    log.info("Loaded %d bars in %.1fs.", len(out), time.monotonic() - t0)
    return out


# ---------------------------------------------------------------------------
# Regime: walk-forward-safe periodic refit (mirrors daemon.py's policy)
# ---------------------------------------------------------------------------


def _maybe_refit_hmm(full_df, bar_index, cfg, fitted_hmm, last_refit_block):
    """Apply daemon.py's exact regime-fit policy at this bar_index: an
    initial fit once MIN_HMM_BARS bars exist, then a refit every time a new
    REGIME_REFIT_BARS-sized block is crossed. Each fit uses ONLY the
    trailing MIN_HMM_BARS bars ending at bar_index — never anything past
    it. Called for every visited bar_index, including the fast-forward
    region on resume, so it's also how a resumed run reconstructs the
    correct fitted-HMM state without re-running any forecasting.

    Returns (fitted_hmm, last_refit_block), unchanged if no fit is due.
    """
    n_seen = bar_index + 1

    if fitted_hmm is None:
        if n_seen < MIN_HMM_BARS:
            return fitted_hmm, last_refit_block
        window = full_df.iloc[max(0, n_seen - MIN_HMM_BARS):n_seen]
        fitted_hmm = regime_mod.fit_hmm(window, symbol=cfg.dukascopy_symbol)
        last_refit_block = n_seen // REGIME_REFIT_BARS
        log.info("Regime HMM initial fit @ bar %d (%d samples, states=%s)",
                  bar_index, len(window), fitted_hmm.state_label_map)
        return fitted_hmm, last_refit_block

    block = n_seen // REGIME_REFIT_BARS
    if block > last_refit_block:
        window = full_df.iloc[max(0, n_seen - MIN_HMM_BARS):n_seen]
        fitted_hmm = regime_mod.fit_hmm(window, symbol=cfg.dukascopy_symbol)
        last_refit_block = block
        log.info("Regime HMM periodic refit @ bar %d (block %d, %d samples)",
                  bar_index, block, len(window))
    return fitted_hmm, last_refit_block


# ---------------------------------------------------------------------------
# One forecast step
# ---------------------------------------------------------------------------


def _build_snapshot(full_df: pd.DataFrame, bar_index: int, context_bars: int) -> MarketSnapshot:
    start = max(0, bar_index + 1 - context_bars)
    context_df = full_df.iloc[start:bar_index + 1].reset_index(drop=True)
    last = context_df.iloc[-1]
    spread = float(last["spread_bps"]) if pd.notna(last["spread_bps"]) else 0.0
    book = OrderBookSnapshot(
        timestamp=last["timestamps"],
        mid_price=float(last["close"]),
        bid_depth=0.0,
        ask_depth=0.0,
        # No tick-level order-flow direction survives into an M1 bar — see
        # module docstring's "WHAT'S DELIBERATELY DIFFERENT" section.
        imbalance=0.0,
        spread_bps=spread,
    )
    return MarketSnapshot(
        timestamp=last["timestamps"],
        ohlcv=context_df,
        last_close=float(last["close"]),
        book=book,
        derivs=None,
    )


def _process_bar(full_df, bar_index, cfg, fitted_hmm, context_bars):
    snap = _build_snapshot(full_df, bar_index, context_bars)

    regime_state = None
    if fitted_hmm is not None:
        try:
            regime_state = regime_mod.classify(
                fitted_hmm, snap.ohlcv,
                book_imbalance=(snap.book.imbalance if snap.book else None),
                book_spread_bps=(snap.book.spread_bps if snap.book else None),
                funding_rate=None,
                taker_ratio=None,
            )
        except Exception as e:
            log.warning("Regime classify failed @ bar %d (%s)", bar_index, e)

    volatility = None
    try:
        volatility = forecast_volatility(snap.ohlcv, horizon_minutes=cfg.horizon_minutes)
    except Exception as e:
        log.warning("Volatility forecast failed @ bar %d (%s)", bar_index, e)

    forecasts = run_forecasts(snap, cfg)
    sig = combine_signal(forecasts, snap, cfg, regime_state=regime_state, volatility=volatility)
    # combine_signal() stamps sig.timestamp with wall-clock now() — correct
    # for a live daemon, wrong here. Overwrite with the HISTORICAL bar
    # timestamp so pair_signals_with_outcomes() looks up the realized price
    # at bar_ts + horizon, not "replay execution time" + horizon.
    sig.timestamp = pd.Timestamp(snap.timestamp).isoformat()
    return sig


# ---------------------------------------------------------------------------
# Checkpoint / output I/O
# ---------------------------------------------------------------------------


def _load_checkpoint(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        log.warning("Checkpoint %s unreadable (%s); treating as no checkpoint.", path, e)
        return None


def _write_checkpoint(path: Path, state: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str))
    os.replace(tmp, path)


def _load_signals_jsonl(path: Path) -> list[dict]:
    """Tolerant line-by-line JSONL read — mirrors state.read_recent_signals's
    handling of a possibly-partial last line (e.g. a prior run killed
    mid-write)."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _fmt_duration(seconds: float) -> str:
    if seconds == float("inf") or seconds != seconds:  # inf or NaN
        return "?"
    seconds = int(seconds)
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    parts = []
    if d:
        parts.append(f"{d}d")
    if h or d:
        parts.append(f"{h}h")
    if m or h or d:
        parts.append(f"{m}m")
    parts.append(f"{s}s")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _print_report(result, n_signals_loaded: int) -> None:
    print(f"\n--- Backtest report over {n_signals_loaded} accumulated replay signals ---")
    print(result.scope_note)
    if result.small_sample_warning:
        print(f"WARNING: {result.small_sample_warning}")
    print(f"n_signals={result.n_signals}  n_directional={result.n_directional}  "
          f"coverage={result.coverage:.2%}")
    if result.accuracy_directional is not None:
        print(f"accuracy (directional): {result.accuracy_directional:.3f}")
    print(f"P&L no costs: {result.pnl_pips_no_costs:+.1f} pips   "
          f"P&L net of full cost model: {result.pnl_pips_with_costs:+.1f} pips")
    g = result.gate
    if g["gross_expectancy_pips_per_trade"] is not None:
        mult = g["gross_edge_spread_multiple"]
        mult_s = f"{mult:.2f}x" if mult is not None else "n/a"
        print(f"gross expectancy/trade: {g['gross_expectancy_pips_per_trade']:+.2f} pips "
              f"({mult_s} mean session-aware spread)")
        print(f"net expectancy/trade:   {g['net_expectancy_pips_per_trade']:+.2f} pips")
        print(f"passes >=1.5x spread: {g['passes_1_5x_spread']}   "
              f"passes >=2x spread: {g['passes_2x_spread']}   "
              f"net edge positive: {g['net_edge_positive']}")
    else:
        print("gate: n/a — no directional trades yet")
    b = result.baseline_random_direction
    if b.get("n_directional", 0):
        print(f"vs random-direction baseline: model_pnl={b['model_pnl_pips']:+.1f} pips, "
              f"{b['pct_random_trials_beating_or_matching_model']:.1%} of random trials matched/beat it")
    by_regime = result.by_regime
    if by_regime:
        print("\nBy regime:")
        for r, st in sorted(by_regime.items()):
            acc = f"{st['accuracy_directional']:.3f}" if st["accuracy_directional"] is not None else "n/a"
            print(f"  {r:10s} n={st['n']:6d}  non_flat={st['n_directional']:6d}  "
                  f"cov={st['coverage']:.2%}  acc={acc}")


# ---------------------------------------------------------------------------
# Report-only mode
# ---------------------------------------------------------------------------


def _report_only(cfg: OracleConfig, args, out_path: Path) -> int:
    signals = _load_signals_jsonl(out_path)
    if not signals:
        print(f"No signals yet at {out_path}")
        return 0
    log.info("Loading price history to score %d accumulated signals...", len(signals))
    full_df = _load_full_history(cfg, args.days)
    cost_model = CostModel()
    result = run_backtest_on_signals(cfg, signals, price_cache=full_df, cost_model=cost_model, fold="week")
    _print_report(result, len(signals))
    return 0


# ---------------------------------------------------------------------------
# Main replay loop
# ---------------------------------------------------------------------------


def _run_replay(cfg: OracleConfig, args, full_df: pd.DataFrame, out_path: Path, ckpt_path: Path):
    n_total_bars = len(full_df)
    first_eligible = args.context_bars - 1
    if first_eligible >= n_total_bars:
        raise RuntimeError(
            f"Only {n_total_bars} bars loaded, need at least {args.context_bars} "
            "for one full context window."
        )
    eligible = list(range(first_eligible, n_total_bars, args.stride))
    n_total_eligible = len(eligible)

    meta = {
        "symbol": cfg.dukascopy_symbol,
        "days": args.days,
        "stride": args.stride,
        "context_bars": args.context_bars,
        "horizon_minutes": args.horizon,
    }

    checkpoint = None if args.fresh else _load_checkpoint(ckpt_path)
    resume_start_idx = 0
    n_already_done = 0

    if args.fresh:
        for f in (out_path, ckpt_path):
            f.unlink(missing_ok=True)
        log.info("--fresh: wiped prior output/checkpoint at %s / %s", out_path, ckpt_path)
    elif checkpoint is not None:
        for k, v in meta.items():
            if checkpoint.get(k) != v:
                raise RuntimeError(
                    f"Checkpoint {ckpt_path} was created with {k}={checkpoint.get(k)!r}, "
                    f"this run has {k}={v!r}. Refusing to resume with mismatched params "
                    "(would silently corrupt the walk-forward). Pass --fresh to start over, "
                    "or match the original invocation's args."
                )
        last_bar_index = checkpoint["last_bar_index"]
        try:
            resume_start_idx = eligible.index(last_bar_index) + 1
        except ValueError:
            raise RuntimeError(
                f"Checkpoint's last_bar_index={last_bar_index} isn't in this run's eligible "
                "bar sequence — the underlying cached data may have changed since the "
                "checkpoint was written. Pass --fresh to start over."
            )
        n_already_done = resume_start_idx
        log.info("Resuming from checkpoint: %d/%d steps already done (last bar_ts=%s, index=%d).",
                  n_already_done, n_total_eligible, checkpoint.get("last_bar_ts"), last_bar_index)
    elif out_path.exists() or ckpt_path.exists():
        raise RuntimeError(
            f"{out_path} or {ckpt_path} exists but the other is missing/unreadable — "
            "inconsistent state, possibly from an interrupted first write. Inspect "
            "manually, or pass --fresh to wipe both and start over."
        )

    out_f = open(out_path, "a")
    fitted_hmm = None
    last_refit_block = -1
    n_new_processed = 0
    n_failed = 0
    t_start = time.monotonic()
    stop = {"flag": False}

    def _handle_signal(signum, frame):  # noqa: ARG001
        log.info("Received signal %s; will stop after the current bar.", signum)
        stop["flag"] = True

    os_signal.signal(os_signal.SIGINT, _handle_signal)
    os_signal.signal(os_signal.SIGTERM, _handle_signal)

    for i, bar_index in enumerate(eligible):
        # Regime bookkeeping runs for EVERY visited bar_index, including the
        # fast-forward (already-done) region on resume — cheap, and it's how
        # the fitted-HMM state at the resume point gets reconstructed
        # without re-running any forecasting. See _maybe_refit_hmm docstring.
        fitted_hmm, last_refit_block = _maybe_refit_hmm(full_df, bar_index, cfg, fitted_hmm, last_refit_block)

        if i < resume_start_idx:
            continue
        if stop["flag"]:
            log.info("Stopping at bar_index=%d (i=%d/%d) due to signal.", bar_index, i, n_total_eligible)
            break
        if args.max_steps is not None and n_new_processed >= args.max_steps:
            log.info("Reached --max-steps=%d; stopping.", args.max_steps)
            break

        t0 = time.monotonic()
        try:
            sig = _process_bar(full_df, bar_index, cfg, fitted_hmm, args.context_bars)
            rec = sig.to_dict()
            rec["_replay_bar_index"] = bar_index
            out_f.write(json.dumps(rec, default=str) + "\n")
            out_f.flush()
        except Exception as e:
            n_failed += 1
            log.warning("Step failed @ bar %d (ts=%s): %s", bar_index, full_df["timestamps"].iloc[bar_index], e)
        step_ms = (time.monotonic() - t0) * 1000

        n_new_processed += 1
        bar_ts = full_df["timestamps"].iloc[bar_index]
        ckpt_state = {
            **meta,
            "last_bar_index": bar_index,
            "last_bar_ts": str(bar_ts),
            "n_total_eligible": n_total_eligible,
            "n_processed_this_run": n_new_processed,
            "n_failed_this_run": n_failed,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_checkpoint(ckpt_path, ckpt_state)

        if n_new_processed % args.log_every == 0 or i == len(eligible) - 1:
            elapsed = time.monotonic() - t_start
            rate = n_new_processed / elapsed if elapsed > 0 else 0.0
            done_total = n_already_done + n_new_processed
            remaining = n_total_eligible - done_total
            eta_s = remaining / rate if rate > 0 else float("inf")
            log.info(
                "[%d/%d, %.1f%%] bar_ts=%s  step=%dms  rate=%.2f/s  elapsed=%s  eta=%s  failed=%d",
                done_total, n_total_eligible, 100.0 * done_total / n_total_eligible,
                bar_ts, int(step_ms), rate, _fmt_duration(elapsed), _fmt_duration(eta_s), n_failed,
            )

    out_f.close()
    elapsed = time.monotonic() - t_start
    log.info("Stopped after %d new steps (%d failed) in %s. Checkpoint: %s",
              n_new_processed, n_failed, _fmt_duration(elapsed), ckpt_path)
    return n_new_processed, n_failed, elapsed, n_total_eligible, n_already_done


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(
        description="Walk-forward replay over cached XAUUSD M1 history.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--symbol", default=ORACLE_CONFIG.symbol,
                    help=f"Broker/instrument symbol (default {ORACLE_CONFIG.symbol!r}).")
    p.add_argument("--days", type=int, default=90, help="Days of cached history to replay (default 90).")
    p.add_argument("--stride", type=int, default=3, help="Forecast every Nth bar (default 3).")
    p.add_argument("--horizon", type=int, default=ORACLE_CONFIG.horizon_minutes,
                    help=f"Forecast horizon in minutes (default {ORACLE_CONFIG.horizon_minutes}).")
    p.add_argument("--context-bars", type=int, default=ORACLE_CONFIG.buffer_bars,
                    help=f"Rolling context window fed to each forecast, in bars "
                         f"(default {ORACLE_CONFIG.buffer_bars} — matches the live daemon's buffer_bars).")
    p.add_argument("--out", type=Path, default=None, help="Signals JSONL output path.")
    p.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint JSON path.")
    p.add_argument("--fresh", action="store_true",
                    help="Ignore/wipe any existing checkpoint and output, start over.")
    p.add_argument("--max-steps", type=int, default=None,
                    help="Stop after this many NEW forecast steps this invocation "
                         "(for timing tests / chunked manual runs).")
    p.add_argument("--log-every", type=int, default=50, help="Progress log cadence, in steps (default 50).")
    p.add_argument("--report-only", action="store_true",
                    help="Score the existing --out file against the cost model/gate and exit; "
                         "does not advance the replay. Safe to run concurrently with a live batch.")
    args = p.parse_args()

    cfg = replace(ORACLE_CONFIG, symbol=args.symbol, horizon_minutes=args.horizon, use_kronos=False)
    cfg.weights = dict(ORACLE_CONFIG.weights)  # independent copy, not the live singleton's dict

    base_dir = cfg.state_dir / "replay"
    base_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{cfg.dukascopy_symbol}_{args.days}d_stride{args.stride}"
    out_path = args.out or (base_dir / f"replay_signals_{tag}.jsonl")
    ckpt_path = args.checkpoint or (base_dir / f"replay_checkpoint_{tag}.json")

    log.info("use_kronos=%s use_timesfm=%s use_tirex=%s use_chronos2=%s",
              cfg.use_kronos, cfg.use_timesfm, cfg.use_tirex, cfg.use_chronos2)
    log.info("out=%s checkpoint=%s", out_path, ckpt_path)

    if args.report_only:
        return _report_only(cfg, args, out_path)

    full_df = _load_full_history(cfg, args.days)
    n_new, n_failed, elapsed, n_total_eligible, n_already_done = _run_replay(
        cfg, args, full_df, out_path, ckpt_path,
    )

    if n_new > 0:
        per_step = elapsed / n_new
        done_total = n_already_done + n_new
        remaining = n_total_eligible - done_total
        print("\n" + "=" * 70)
        print(f"Steps this run: {n_new} (failed: {n_failed})   elapsed: {_fmt_duration(elapsed)}")
        print(f"Mean latency/step: {per_step:.2f}s")
        print(f"Progress: {done_total}/{n_total_eligible} ({100.0 * done_total / n_total_eligible:.1f}%)")
        if remaining > 0:
            projected_remaining = per_step * remaining
            projected_total = per_step * n_total_eligible
            print(f"\nPROJECTION (based on {n_new} real step(s) just measured):")
            print(f"  Total eligible forecast steps for the full {args.days}-day replay: {n_total_eligible}")
            print(f"  Projected total wall time (all {n_total_eligible} steps): {_fmt_duration(projected_total)}")
            print(f"  Remaining steps: {remaining}   Projected remaining wall time: {_fmt_duration(projected_remaining)}")
        print("=" * 70)

    # Always score whatever has accumulated so far (partial or complete) —
    # partial results are still informative, and this is what "feed
    # everything through the session-aware cost model and gate" means here.
    signals = _load_signals_jsonl(out_path)
    if signals:
        cost_model = CostModel()
        result = run_backtest_on_signals(cfg, signals, price_cache=full_df, cost_model=cost_model, fold="week")
        _print_report(result, len(signals))

    return 0


if __name__ == "__main__":
    sys.exit(main())
