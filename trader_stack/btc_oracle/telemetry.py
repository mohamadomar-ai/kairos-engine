"""Structured telemetry — one JSON line per daemon cycle.

The signals.jsonl file records the *output* of each cycle (the signal that
got produced). This file records the *cycle itself*: stage timings, which
forecasters succeeded vs failed, model weights at that moment, filter gates
that fired, fetch latency for each data source.

Why this matters in production:

  Silent failures are the killer. When Chronos-2 silently degrades to
  close-only forecasting because covariates didn't attach, the daemon
  keeps running, the signals keep flowing, the backtest still computes
  accuracy — but a chunk of your alpha just disappeared. The only way
  you'd notice is when accuracy slowly drops over weeks. By then you've
  burned days of capital.

  This file lets you grep for "chronos2_covariates_attached=0" or
  "fetch_ohlcv_failed=1" or "cycle_ms > 50000" and find degradations
  the moment they start, not weeks later.

Format: JSON-lines (one self-contained JSON object per line), append-only.
Rotates daily so the file never grows unbounded.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class TelemetryWriter:
    """Append-only JSONL telemetry. Cheap, lossy on failure (we never let
    telemetry I/O kill the daemon loop)."""

    def __init__(self, state_dir: Path):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def _path_for_today(self) -> Path:
        # Rotate by UTC day so logs never grow unbounded.
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        return self.state_dir / f"telemetry_{day}.jsonl"

    def emit(self, record: dict[str, Any]) -> None:
        """Write one telemetry record. Errors are logged but never raised."""
        try:
            record.setdefault("ts", datetime.now(timezone.utc).isoformat())
            with open(self._path_for_today(), "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as e:
            # The daemon must never die from telemetry I/O.
            log.debug("Telemetry write failed (ignoring): %s", e)


def build_cycle_record(
    cycle_num: int,
    snap_close: float,
    timings_ms: dict[str, int],
    forecasts_ok: dict[str, bool],
    forecasts_pct: dict[str, float],
    regime_label: str | None,
    regime_conf: float | None,
    raw_direction: str,
    raw_confidence: float,
    final_direction: str,
    final_confidence: float,
    filter_fired: list[str],
    weights: dict[str, float],
    fetch_failures: dict[str, str],
    chronos2_covariates_attached: int | None,
) -> dict[str, Any]:
    """Build a full structured telemetry record for one cycle.

    Keeping this as a separate builder function (rather than building inline
    in the daemon) makes it easy to extend without touching the hot path.
    """
    return {
        "cycle": cycle_num,
        "close": snap_close,
        "timings_ms": timings_ms,
        "forecasts": {
            "succeeded": {k: forecasts_pct.get(k) for k, ok in forecasts_ok.items() if ok},
            "failed": [k for k, ok in forecasts_ok.items() if not ok],
        },
        "regime": {"label": regime_label, "confidence": regime_conf},
        "signal": {
            "raw_direction": raw_direction,
            "raw_confidence": raw_confidence,
            "final_direction": final_direction,
            "final_confidence": final_confidence,
            "filter_fired": list(filter_fired),
            "suppressed": (raw_direction != "FLAT" and final_direction == "FLAT"),
        },
        "weights": dict(weights),
        "fetch_failures": dict(fetch_failures),
        "chronos2_covariates_attached": chronos2_covariates_attached,
    }
