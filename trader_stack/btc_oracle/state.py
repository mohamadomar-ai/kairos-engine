"""Read/write the oracle's persisted state.

Two files:
  - state.json     — the latest signal (overwritten every minute)
  - signals.jsonl  — every signal, one per line, append-only (for backtest)

State writes use filelock to keep concurrent reads consistent (the daemon
writes; the OpenClaw skills read).
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

from filelock import FileLock, Timeout

from .oracle_config import ORACLE_CONFIG, OracleConfig
from .signal import Signal

log = logging.getLogger(__name__)


def _lock_path(state_file: Path) -> Path:
    return state_file.with_suffix(state_file.suffix + ".lock")


def write_state(sig: Signal, cfg: OracleConfig = ORACLE_CONFIG) -> None:
    """Atomically overwrite state.json with the latest signal, then append to log."""
    cfg.ensure_state_dir()
    state_file = cfg.state_file
    log_file = cfg.signals_log

    payload = sig.to_dict()
    tmp = state_file.with_suffix(".tmp")
    with FileLock(str(_lock_path(state_file)), timeout=5):
        tmp.write_text(json.dumps(payload, indent=2))
        # atomic rename
        os.replace(tmp, state_file)
        # append to log (best effort)
        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(payload) + "\n")
        except Exception as e:
            log.warning("Failed to append to signals log: %s", e)


def read_state(cfg: OracleConfig = ORACLE_CONFIG) -> Optional[dict]:
    """Read the latest signal. Returns None if no state file exists yet."""
    state_file = cfg.state_file
    if not state_file.exists():
        return None
    try:
        with FileLock(str(_lock_path(state_file)), timeout=2):
            return json.loads(state_file.read_text())
    except Timeout:
        # If the daemon is mid-write, just retry once without lock — readers
        # are best-effort.
        try:
            return json.loads(state_file.read_text())
        except Exception:
            return None
    except Exception as e:
        log.warning("Could not read state: %s", e)
        return None


def read_recent_signals(n: int = 100, cfg: OracleConfig = ORACLE_CONFIG) -> list[dict]:
    """Read the last `n` signals from the JSONL log.

    Used by the backtest skill and by adaptive weighting.
    """
    log_file = cfg.signals_log
    if not log_file.exists():
        return []
    try:
        # Read the whole file (typically small) and take the tail.
        lines = log_file.read_text().strip().split("\n")
        out = []
        for line in lines[-n:]:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out
    except Exception as e:
        log.warning("Could not read signals log: %s", e)
        return []


# ---------------------------------------------------------------------------
# PID file helpers (used by the daemon)
# ---------------------------------------------------------------------------


def write_pid(cfg: OracleConfig = ORACLE_CONFIG) -> None:
    cfg.ensure_state_dir()
    cfg.pid_file.write_text(str(os.getpid()))


def read_pid(cfg: OracleConfig = ORACLE_CONFIG) -> Optional[int]:
    if not cfg.pid_file.exists():
        return None
    try:
        return int(cfg.pid_file.read_text().strip())
    except Exception:
        return None


def clear_pid(cfg: OracleConfig = ORACLE_CONFIG) -> None:
    try:
        cfg.pid_file.unlink(missing_ok=True)  # type: ignore[arg-type]
    except Exception:
        pass


def is_daemon_alive(cfg: OracleConfig = ORACLE_CONFIG) -> bool:
    """Check if the daemon process recorded in pid_file is still running."""
    pid = read_pid(cfg)
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        # Process not found or no permission
        return False
    except Exception:
        return False


def freshness_seconds(cfg: OracleConfig = ORACLE_CONFIG) -> Optional[float]:
    """Seconds since the last state file write. None if no state."""
    if not cfg.state_file.exists():
        return None
    try:
        return time.time() - cfg.state_file.stat().st_mtime
    except Exception:
        return None
