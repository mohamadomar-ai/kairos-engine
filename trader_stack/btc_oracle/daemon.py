"""The BTC oracle daemon. One cycle per minute, runs forever until stopped.

Lifecycle:
  start  → fork off as background process, write pid file, begin loop
  stop   → read pid file, send SIGTERM, wait for cleanup
  status → check pid liveness and state freshness

The actual loop is `_run_loop()`, runnable in the foreground (handy for debugging).
"""

from __future__ import annotations

import logging
import os
import signal as os_signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .feeds import fetch_ohlcv, fetch_snapshot
from .oracle_config import ORACLE_CONFIG, OracleConfig
from .signal import combine_signal, run_forecasts
from .state import (
    clear_pid,
    is_daemon_alive,
    read_pid,
    write_pid,
)
from .state import write_state as write_state_file
from .telemetry import TelemetryWriter, build_cycle_record
from . import calibration, outcomes, store

log = logging.getLogger(__name__)


# How often (in cycles) to refit the calibration model from accumulated history.
CALIBRATION_REFIT_INTERVAL = 360   # 6 hours at 1 cycle/min
# How often to run the outcome resolver. Every cycle is fine — cheap when nothing's pending.
OUTCOME_RESOLVE_INTERVAL = 1


# Module-level container for daemon state that persists across cycles.
class _RuntimeState:
    """Daemon-level mutable state. Initialized in _warm_up_models, used per-cycle."""
    fitted_hmm = None              # _FittedHMM | None
    cycle_count: int = 0           # how many cycles completed
    telemetry: TelemetryWriter | None = None
    calibrator: calibration.CalibrationModel | None = None


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _sleep_until_next_minute(offset_seconds: int) -> None:
    """Sleep until <offset_seconds> past the next minute boundary."""
    now = time.time()
    next_minute = (int(now) // 60 + 1) * 60
    target = next_minute + offset_seconds
    delay = target - now
    if delay > 0:
        time.sleep(delay)


# ---------------------------------------------------------------------------
# Warm-up
# ---------------------------------------------------------------------------


def _warm_up_models(cfg: OracleConfig) -> None:
    """Load all enabled models into memory before the first cycle."""
    log.info("Warming up models — this is a one-time cost.")

    # ----- Postgres connectivity + calibrator load ------------------------
    try:
        store.ensure_schema()
        _RuntimeState.calibrator = calibration.load_latest()
        if _RuntimeState.calibrator.is_identity:
            log.info("Calibration: starting with identity (will fit after %d outcomes accumulate)",
                     calibration.MIN_SAMPLES_TO_FIT)
        else:
            log.info("Calibration: loaded model fit on %d samples",
                     _RuntimeState.calibrator.n_train_samples)
    except Exception as e:
        log.warning("Calibration warm-up failed (%s); falling back to identity", e)
        _RuntimeState.calibrator = calibration.identity_model()

    if cfg.use_kronos or cfg.use_timesfm:
        # Reuse the existing trader_stack warm-up for Kronos and TimesFM.
        try:
            from trader_stack import forecasters
            if cfg.use_kronos:
                forecasters._kronos_predictor()  # noqa: SLF001
            if cfg.use_timesfm:
                forecasters._timesfm_model()     # noqa: SLF001
        except Exception as e:
            log.error("Kronos/TimesFM warm-up issue: %s", e)
    if cfg.use_tirex or cfg.use_chronos2:
        from .forecasters_extra import warm_up
        warm_up(use_tirex=cfg.use_tirex, use_chronos2=cfg.use_chronos2)

    # ----- Phase 2: HMM regime classifier ---------------------------------
    if cfg.use_regime:
        try:
            from . import regime
            log.info("Fitting/loading regime HMM...")
            # Pull enough history to fit a stable HMM (1500 minute-bars ≈ 25h).
            train_df = fetch_ohlcv(cfg, limit=1500)
            _RuntimeState.fitted_hmm = regime.get_or_fit(
                cfg,
                ohlcv_for_training=train_df,
                max_age_hours=cfg.regime_max_age_hours,
            )
            log.info("Regime classifier ready.")
        except Exception as e:
            log.warning("Regime classifier setup failed (%s); proceeding without it.", e)
            _RuntimeState.fitted_hmm = None


# ---------------------------------------------------------------------------
# Cycle
# ---------------------------------------------------------------------------


def _cycle_once(cfg: OracleConfig) -> None:
    """One full cycle: fetch → forecast → regime → volatility → combine → write.

    Per-stage timings and outcomes flow into the telemetry stream so silent
    degradations surface immediately.
    """
    t0 = time.monotonic()
    timings_ms: dict[str, int] = {}
    fetch_failures: dict[str, str] = {}
    forecasts_ok: dict[str, bool] = {}
    forecasts_pct: dict[str, float] = {}

    # --- Fetch ------------------------------------------------------------
    t_stage = time.monotonic()
    snap = fetch_snapshot(cfg)
    timings_ms["fetch"] = int((time.monotonic() - t_stage) * 1000)
    if snap.book is None:
        fetch_failures["book"] = "missing"
    if snap.derivs is None:
        fetch_failures["derivs"] = "missing"
    log.info("Snapshot @ %s — close=%.2f (fetch %dms)",
             snap.timestamp.strftime("%H:%M:%S"), snap.last_close, timings_ms["fetch"])

    # --- Regime ----------------------------------------------------------
    regime_state = None
    t_stage = time.monotonic()
    if cfg.use_regime and _RuntimeState.fitted_hmm is not None:
        try:
            from . import regime as regime_mod
            book = snap.book
            derivs = snap.derivs
            regime_state = regime_mod.classify(
                _RuntimeState.fitted_hmm,
                snap.ohlcv,
                book_imbalance=(book.imbalance if book else None),
                book_spread_bps=(book.spread_bps if book else None),
                funding_rate=(derivs.funding_rate if derivs else None),
                taker_ratio=(derivs.taker_buy_sell_ratio if derivs else None),
            )
        except Exception as e:
            log.warning("Regime classification failed this cycle: %s", e)
            fetch_failures["regime"] = str(e)[:120]
    timings_ms["regime"] = int((time.monotonic() - t_stage) * 1000)

    # --- Volatility ------------------------------------------------------
    volatility = None
    t_stage = time.monotonic()
    try:
        from .volatility import forecast_volatility
        volatility = forecast_volatility(snap.ohlcv, horizon_minutes=cfg.horizon_minutes)
    except Exception as e:
        log.warning("Volatility forecast failed: %s", e)
        fetch_failures["volatility"] = str(e)[:120]
    timings_ms["volatility"] = int((time.monotonic() - t_stage) * 1000)

    # --- Forecasts -------------------------------------------------------
    t_stage = time.monotonic()
    forecasts = run_forecasts(snap, cfg)
    timings_ms["forecasts"] = int((time.monotonic() - t_stage) * 1000)
    enabled = {"kronos": cfg.use_kronos, "timesfm": cfg.use_timesfm,
               "tirex": cfg.use_tirex, "chronos2": cfg.use_chronos2}
    for name, on in enabled.items():
        if not on:
            continue
        if name in forecasts:
            forecasts_ok[name] = True
            forecasts_pct[name] = float(forecasts[name].pct_change)
        else:
            forecasts_ok[name] = False
    log.info("Forecasts: %s (%dms)",
             ", ".join(f"{n}={fc.pct_change:+.3%}" for n, fc in forecasts.items()),
             timings_ms["forecasts"])

    # --- Combine + write -------------------------------------------------
    t_stage = time.monotonic()
    sig = combine_signal(forecasts, snap, cfg, regime_state=regime_state, volatility=volatility)

    # Calibrate the raw confidence (NOT post-filter confidence). The widget
    # will eventually surface this as the "real" probability.
    cal = _RuntimeState.calibrator or calibration.identity_model()
    calibrated_conf = cal.apply(sig.raw_confidence)

    # JSONL write (legacy / fallback path — keep working during dual-write phase)
    write_state_file(sig, cfg)

    # Postgres write (new memory layer — non-fatal on failure)
    try:
        sig_dict = sig.to_dict()
        sig_dict["calibrated_confidence"] = calibrated_conf  # for downstream readers
        store.write_signal(sig_dict, calibrated_confidence=calibrated_conf)
    except Exception as e:
        log.warning("Postgres signal write failed (JSONL still wrote): %s", e)

    # Outcome resolver — fills in realized prices for past signals whose
    # horizon has closed. Cheap when nothing's pending.
    if _RuntimeState.cycle_count % OUTCOME_RESOLVE_INTERVAL == 0:
        try:
            outcomes.resolve_pending(
                cfg=cfg,
                flat_threshold=cfg.flat_threshold,
            )
        except Exception as e:
            log.warning("Outcome resolver failed: %s", e)

    timings_ms["combine_write"] = int((time.monotonic() - t_stage) * 1000)

    total_ms = int((time.monotonic() - t0) * 1000)
    timings_ms["total"] = total_ms

    log.info(
        "→ %s @ conf=%.2f (cal=%.2f)  raw=%s@%.2f  consensus=%s  regime=%s  cycle=%dms",
        sig.direction, sig.confidence, calibrated_conf,
        sig.raw_direction, sig.raw_confidence,
        f"{sig.consensus_pct_change:+.3%}",
        (regime_state.label if regime_state else "n/a"),
        total_ms,
    )

    # --- Telemetry emit --------------------------------------------------
    if _RuntimeState.telemetry is not None:
        try:
            # Pull the covariate-attached count from the Chronos-2 forecast if
            # it surfaced — we expose it through a side channel because the
            # CloseForecast type doesn't carry it directly.
            chronos2_cov = getattr(forecasts.get("chronos2"), "_n_covariates_attached", None)
            record = build_cycle_record(
                cycle_num=_RuntimeState.cycle_count + 1,
                snap_close=snap.last_close,
                timings_ms=timings_ms,
                forecasts_ok=forecasts_ok,
                forecasts_pct=forecasts_pct,
                regime_label=(regime_state.label if regime_state else None),
                regime_conf=(regime_state.confidence if regime_state else None),
                raw_direction=sig.raw_direction,
                raw_confidence=sig.raw_confidence,
                final_direction=sig.direction,
                final_confidence=sig.confidence,
                filter_fired=sig.filter_fired,
                weights=dict(cfg.weights),
                fetch_failures=fetch_failures,
                chronos2_covariates_attached=chronos2_cov,
            )
            _RuntimeState.telemetry.emit(record)
        except Exception as e:
            log.debug("Telemetry record build failed: %s", e)

    _RuntimeState.cycle_count += 1

    # ----- Periodic background refits -------------------------------------
    # Reweighting cadence depends on the CURRENT regime — react faster in
    # volatile regimes per the institutional advice.
    if cfg.adaptive_weighting:
        regime_label = regime_state.label if regime_state is not None else None
        interval = cfg.reweight_interval_for_regime(regime_label)
        if _RuntimeState.cycle_count % interval == 0:
            try:
                from .adaptive import recompute_weights
                enabled_models = [m for m, on in {
                    "kronos": cfg.use_kronos, "timesfm": cfg.use_timesfm,
                    "tirex": cfg.use_tirex, "chronos2": cfg.use_chronos2,
                }.items() if on]
                new_w = recompute_weights(cfg, enabled_models)
                if new_w:
                    cfg.weights.update(new_w)
                    log.info("Reweighted under regime=%s (interval %d cycles)",
                             regime_label, interval)
            except Exception as e:
                log.warning("Adaptive reweighting failed: %s", e)

    if cfg.use_regime and _RuntimeState.cycle_count % cfg.regime_refit_interval == 0:
        try:
            from . import regime as regime_mod
            log.info("Periodic HMM refit (cycle %d)", _RuntimeState.cycle_count)
            train_df = fetch_ohlcv(cfg, limit=1500)
            _RuntimeState.fitted_hmm = regime_mod.fit_hmm(train_df, symbol=cfg.dukascopy_symbol)
            regime_mod.save_fitted(_RuntimeState.fitted_hmm, cfg)
        except Exception as e:
            log.warning("Periodic HMM refit failed: %s", e)

    # Calibration: refit every CALIBRATION_REFIT_INTERVAL cycles (~6h).
    # The fitter is a no-op if we don't have MIN_SAMPLES_TO_FIT outcomes yet.
    if _RuntimeState.cycle_count % CALIBRATION_REFIT_INTERVAL == 0:
        try:
            log.info("Periodic calibration refit (cycle %d)", _RuntimeState.cycle_count)
            new_cal = calibration.fit_and_persist()
            if new_cal is not None:
                _RuntimeState.calibrator = new_cal
                log.info("Calibration updated: n_train=%d, identity=%s",
                         new_cal.n_train_samples, new_cal.is_identity)
        except Exception as e:
            log.warning("Periodic calibration refit failed: %s", e)


# ---------------------------------------------------------------------------
# Foreground loop
# ---------------------------------------------------------------------------


_should_stop = False


def _install_signal_handlers() -> None:
    def _handler(signum, frame):  # noqa: ARG001
        global _should_stop
        log.info("Received signal %s; shutting down after this cycle.", signum)
        _should_stop = True

    os_signal.signal(os_signal.SIGINT, _handler)
    os_signal.signal(os_signal.SIGTERM, _handler)


def run_forever(cfg: OracleConfig = ORACLE_CONFIG) -> None:
    """Foreground daemon loop. Block until SIGTERM or SIGINT."""
    cfg.ensure_state_dir()
    _install_signal_handlers()
    write_pid(cfg)
    _RuntimeState.telemetry = TelemetryWriter(cfg.state_dir)
    try:
        _warm_up_models(cfg)
        log.info("Daemon started; entering minute loop (PID=%d)", os.getpid())

        while not _should_stop:
            _sleep_until_next_minute(cfg.loop_offset_seconds)
            if _should_stop:
                break
            try:
                _cycle_once(cfg)
            except KeyboardInterrupt:
                break
            except Exception:
                log.exception("Cycle failed; continuing to next minute")
    finally:
        clear_pid(cfg)
        log.info("Daemon stopped.")


def run_once(cfg: OracleConfig = ORACLE_CONFIG) -> dict:
    """Run a single cycle (no warm-up loop, no background daemon). Returns the signal dict.

    Useful from the CLI for one-shot inspection: `oracle.py forecast`.
    Builds regime + volatility just like the daemon does.
    """
    cfg.ensure_state_dir()
    snap = fetch_snapshot(cfg)

    # Regime — fit on-the-fly from the snapshot's buffer if no cached model
    regime_state = None
    if cfg.use_regime:
        try:
            from . import regime as regime_mod
            fitted = regime_mod.load_fitted(cfg)
            if fitted is None or not regime_mod.is_fit_fresh(fitted, cfg.regime_max_age_hours):
                # If we don't have enough buffer for a fresh fit, fetch more.
                if len(snap.ohlcv) < 500:
                    train_df = fetch_ohlcv(cfg, limit=1500)
                else:
                    train_df = snap.ohlcv
                fitted = regime_mod.fit_hmm(train_df, symbol=cfg.dukascopy_symbol)
                regime_mod.save_fitted(fitted, cfg)
            book = snap.book
            derivs = snap.derivs
            regime_state = regime_mod.classify(
                fitted, snap.ohlcv,
                book_imbalance=(book.imbalance if book else None),
                book_spread_bps=(book.spread_bps if book else None),
                funding_rate=(derivs.funding_rate if derivs else None),
                taker_ratio=(derivs.taker_buy_sell_ratio if derivs else None),
            )
        except Exception as e:
            log.warning("One-shot regime classification failed: %s", e)

    volatility = None
    try:
        from .volatility import forecast_volatility
        volatility = forecast_volatility(snap.ohlcv, horizon_minutes=cfg.horizon_minutes)
    except Exception as e:
        log.warning("Volatility forecast failed: %s", e)

    forecasts = run_forecasts(snap, cfg)
    sig = combine_signal(forecasts, snap, cfg, regime_state=regime_state, volatility=volatility)
    write_state_file(sig, cfg)
    return sig.to_dict()


# ---------------------------------------------------------------------------
# Background lifecycle (start / stop / status)
# ---------------------------------------------------------------------------


def start_background(cfg: OracleConfig = ORACLE_CONFIG, python_executable: str | None = None) -> int:
    """Spawn the daemon as a detached background process.

    Returns the PID of the spawned process.

    We exec `oracle.py daemon run` rather than re-importing here so the daemon
    runs with a clean process group and inherits no parent-state surprises.
    """
    if is_daemon_alive(cfg):
        pid = read_pid(cfg)
        log.info("Daemon already running (PID=%d)", pid)
        return pid  # type: ignore[return-value]

    cfg.ensure_state_dir()
    python = python_executable or sys.executable
    # Repo root is one level up from trader_stack/btc_oracle/.
    repo_root = Path(__file__).resolve().parent.parent.parent
    oracle_script = repo_root / "oracle.py"
    log_file = cfg.daemon_log

    # Use nohup-like detachment: setsid, redirect stdio, close fds.
    with open(log_file, "ab") as lf:
        proc = subprocess.Popen(
            [python, str(oracle_script), "daemon", "run"],
            stdout=lf,
            stderr=lf,
            stdin=subprocess.DEVNULL,
            start_new_session=True,  # equivalent to setsid()
            cwd=str(repo_root),
        )
    # Give it a moment to write its pid file.
    time.sleep(1.0)
    log.info("Spawned daemon (PID=%d), log=%s", proc.pid, log_file)
    return proc.pid


def stop_background(cfg: OracleConfig = ORACLE_CONFIG, timeout: float = 90.0) -> bool:
    """Send SIGTERM to the background daemon. Returns True if it exited cleanly.

    Default timeout is 90s, not 10s — a single cycle routinely takes 35-70s+
    (Kronos's forecast alone is ~32s on this CPU; Dukascopy fetch latency can
    spike further, observed up to ~33s in practice). SIGTERM only takes effect
    between cycles (the loop checks _should_stop after _cycle_once() returns,
    not mid-cycle), so a too-short timeout forces an unnecessary SIGKILL on
    every stop whenever a cycle happens to be in flight — reproduced in
    practice, not theoretical.
    """
    pid = read_pid(cfg)
    if pid is None:
        log.info("No daemon recorded; nothing to stop.")
        return True

    if not is_daemon_alive(cfg):
        log.info("Recorded PID %d is dead; clearing pid file.", pid)
        clear_pid(cfg)
        return True

    log.info("Sending SIGTERM to PID %d", pid)
    try:
        os.kill(pid, os_signal.SIGTERM)
    except OSError as e:
        log.warning("kill(%d, SIGTERM) failed: %s", pid, e)
        return False

    # Wait for clean shutdown.
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not is_daemon_alive(cfg):
            return True
        time.sleep(0.5)

    log.warning("Daemon did not exit within %ss; sending SIGKILL", timeout)
    try:
        os.kill(pid, os_signal.SIGKILL)
    except OSError:
        pass
    clear_pid(cfg)
    return not is_daemon_alive(cfg)


def status_background(cfg: OracleConfig = ORACLE_CONFIG) -> dict:
    """Lightweight status check, suitable for the OpenClaw control skill."""
    from .state import freshness_seconds, read_state

    pid = read_pid(cfg)
    alive = is_daemon_alive(cfg)
    fresh = freshness_seconds(cfg)
    state = read_state(cfg)

    return {
        "pid": pid,
        "alive": alive,
        "state_freshness_seconds": fresh,
        "latest_signal": state,
        "config": {
            "symbol": cfg.symbol,
            "horizon_minutes": cfg.horizon_minutes,
            "flat_threshold": cfg.flat_threshold,
            "models": {
                "kronos": cfg.use_kronos,
                "timesfm": cfg.use_timesfm,
                "tirex": cfg.use_tirex,
                "chronos2": cfg.use_chronos2,
            },
            "phase2": {
                "use_regime": cfg.use_regime,
                "use_trade_filter": cfg.use_trade_filter,
                "min_confidence": cfg.min_confidence,
                "noise_floor_multiplier": cfg.noise_floor_multiplier,
                "spread_bps_max": cfg.spread_bps_max,
                "chop_blocks_signals": cfg.chop_blocks_signals,
                "adaptive_weighting": cfg.adaptive_weighting,
            },
        },
    }
