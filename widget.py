#!/usr/bin/env python3
"""BTC oracle widget — a small movable always-on-top window.

Shows ONE of three states, color-coded, large and unambiguous:

    BET UP    (green)    — system says go long
    BET DOWN  (red)      — system says go short
    WAIT      (grey)     — no edge right now, do nothing

State is read from ~/.trader-stack/btc-oracle/state.json every minute.

How "BET vs WAIT" is decided
----------------------------
The signal layer already produces direction = UP / DOWN / FLAT plus a
calibrated confidence in [0, 1]. We map them to bet/wait like this:

    direction == FLAT                        →  WAIT
    direction == UP/DOWN AND cal_conf >= MIN →  BET (in that direction)
    otherwise                                →  WAIT

`MIN` defaults to 0.60. Override via the BET_THRESHOLD env var.

Why a separate threshold here on top of the daemon's filter:
    The daemon's filter already suppresses low-confidence signals. This
    extra check is your personal "I won't act unless cal_conf is at least
    this high" — independent of how aggressive the daemon is. Lets you
    tune just the widget without touching the daemon config.

Movable + always-on-top
-----------------------
The window has no title bar (cleaner look) and you drag it by clicking
anywhere on its surface and pulling. Right-click closes it.

Runs with the standard library only (tkinter ships with Python). Zero
extra dependencies.
"""

from __future__ import annotations

import json
import os
import sys
import time
import tkinter as tk
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

STATE_FILE = Path(
    os.getenv("BTC_ORACLE_STATE_DIR",
              str(Path.home() / ".trader-stack" / "btc-oracle"))
) / "state.json"

# How often to re-read state.json (ms).
REFRESH_MS = 30_000  # 30 seconds

# Minimum calibrated confidence to recommend BETTING. Below this → WAIT.
BET_THRESHOLD = float(os.getenv("BET_THRESHOLD", "0.60"))

# How stale state.json can be before we treat it as "no data".
MAX_FRESHNESS_SECONDS = 180  # 3 minutes

# Visual config
COLORS = {
    "UP":    {"bg": "#0f7a3d", "fg": "white"},   # green
    "DOWN":  {"bg": "#a31c1c", "fg": "white"},   # red
    "WAIT":  {"bg": "#3a3a3a", "fg": "#d0d0d0"}, # grey
    "STALE": {"bg": "#1a1a1a", "fg": "#888888"}, # darker grey for "no data"
}

WIDGET_WIDTH = 220
WIDGET_HEIGHT = 110


# ---------------------------------------------------------------------------
# State reader
# ---------------------------------------------------------------------------


def _read_state() -> dict | None:
    """Read state.json. Returns dict or None on failure."""
    try:
        if not STATE_FILE.exists():
            return None
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return None


def _file_age_seconds() -> float | None:
    try:
        return time.time() - STATE_FILE.stat().st_mtime
    except Exception:
        return None


def decide_action(state: dict | None, age_s: float | None) -> tuple[str, str, str]:
    """Translate the signal state into (action, subtext, color_key).

    Returns:
        action     : the big label — "BET UP", "BET DOWN", "WAIT", or "NO DATA"
        subtext    : the small line under it — confidence + freshness
        color_key  : key into COLORS
    """
    if state is None:
        return ("NO DATA", "Start the daemon", "STALE")

    if age_s is not None and age_s > MAX_FRESHNESS_SECONDS:
        mins = int(age_s // 60)
        return ("NO DATA", f"Last signal {mins}m ago", "STALE")

    direction = state.get("direction", "FLAT")
    # Prefer the calibrated confidence; fall back to plain confidence if calibration hasn't kicked in.
    cal_conf = state.get("calibrated_confidence")
    conf = state.get("confidence", 0.0)
    decisive_conf = cal_conf if cal_conf is not None else conf

    age_label = f"{int(age_s)}s ago" if age_s is not None and age_s < 60 \
                else (f"{int(age_s // 60)}m ago" if age_s is not None else "")

    if direction == "FLAT" or decisive_conf < BET_THRESHOLD:
        sub = f"conf {decisive_conf:.0%} · {age_label}".strip(" ·")
        return ("WAIT", sub, "WAIT")

    if direction == "UP":
        sub = f"conf {decisive_conf:.0%} · {age_label}".strip(" ·")
        return ("BET UP", sub, "UP")

    if direction == "DOWN":
        sub = f"conf {decisive_conf:.0%} · {age_label}".strip(" ·")
        return ("BET DOWN", sub, "DOWN")

    return ("WAIT", "Unknown signal state", "WAIT")


# ---------------------------------------------------------------------------
# Movable, always-on-top tkinter window
# ---------------------------------------------------------------------------


class BTCWidget:
    def __init__(self):
        self.root = tk.Tk()
        # No window decorations — clean, just the colored rectangle.
        self.root.overrideredirect(True)
        # Always on top.
        self.root.attributes("-topmost", True)
        # Initial size + position (top-right corner; user can drag from there).
        screen_w = self.root.winfo_screenwidth()
        x = screen_w - WIDGET_WIDTH - 20
        y = 40
        self.root.geometry(f"{WIDGET_WIDTH}x{WIDGET_HEIGHT}+{x}+{y}")

        # --- Frame + labels ---
        self.frame = tk.Frame(self.root, bg=COLORS["WAIT"]["bg"])
        self.frame.pack(fill="both", expand=True)

        self.action_label = tk.Label(
            self.frame,
            text="LOADING",
            font=("Helvetica", 28, "bold"),
            bg=COLORS["WAIT"]["bg"],
            fg=COLORS["WAIT"]["fg"],
        )
        self.action_label.pack(pady=(14, 2))

        self.sub_label = tk.Label(
            self.frame,
            text="reading state…",
            font=("Helvetica", 10),
            bg=COLORS["WAIT"]["bg"],
            fg=COLORS["WAIT"]["fg"],
        )
        self.sub_label.pack(pady=(0, 8))

        self.hint_label = tk.Label(
            self.frame,
            text="drag to move · right-click to close",
            font=("Helvetica", 7),
            bg=COLORS["WAIT"]["bg"],
            fg="#888888",
        )
        self.hint_label.pack(side="bottom", pady=(0, 4))

        # --- Drag-to-move plumbing ---
        # Bind on every child so clicking anywhere starts the drag.
        for w in (self.frame, self.action_label, self.sub_label, self.hint_label):
            w.bind("<Button-1>", self._start_drag)
            w.bind("<B1-Motion>", self._do_drag)
            w.bind("<Button-3>", self._close)   # right-click to close

        self._drag_offset = (0, 0)

        # --- Kick off the refresh loop ---
        self._tick()

    def _start_drag(self, event):
        self._drag_offset = (event.x_root - self.root.winfo_x(),
                             event.y_root - self.root.winfo_y())

    def _do_drag(self, event):
        x = event.x_root - self._drag_offset[0]
        y = event.y_root - self._drag_offset[1]
        self.root.geometry(f"+{x}+{y}")

    def _close(self, _event=None):
        self.root.destroy()

    def _apply_colors(self, key: str):
        c = COLORS[key]
        self.frame.config(bg=c["bg"])
        self.action_label.config(bg=c["bg"], fg=c["fg"])
        self.sub_label.config(bg=c["bg"], fg=c["fg"])
        self.hint_label.config(bg=c["bg"])

    def _tick(self):
        state = _read_state()
        age = _file_age_seconds()
        action, sub, key = decide_action(state, age)
        self.action_label.config(text=action)
        self.sub_label.config(text=sub)
        self._apply_colors(key)
        # Schedule next refresh.
        self.root.after(REFRESH_MS, self._tick)

    def run(self):
        self.root.mainloop()


def main():
    # Quick sanity: tkinter is available?
    try:
        tk.Tk().destroy()
    except Exception as e:
        print(f"ERROR: tkinter not available — install with: sudo apt install python3-tk\n({e})",
              file=sys.stderr)
        sys.exit(1)

    print(f"Reading state from: {STATE_FILE}")
    print(f"BET_THRESHOLD: {BET_THRESHOLD}")
    print("Widget starting. Drag with left-click, close with right-click.")
    BTCWidget().run()


if __name__ == "__main__":
    main()
