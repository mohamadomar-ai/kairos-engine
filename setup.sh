#!/usr/bin/env bash
# trader-stack/setup.sh
# Installer for the BTC minute oracle stack.
#
# What this installs:
#   - Kronos      (clone repo, no PyPI package)
#   - TimesFM 2.5 (clone repo, pip install -e .[torch])
#   - TiRex       (PyPI: tirex)
#   - Chronos-2   (PyPI: chronos-forecasting, package name `chronos`)
#   - CCXT, requests, filelock, pandas, etc.
#   - Symlinks ./skills into ~/.openclaw/skills/trader-stack/ so the 3 BTC
#     oracle skills become visible to your lobster.
#
# What this DOES NOT install (deliberately):
#   - TradingAgents — the multi-agent system. It targets day-or-longer horizons
#     and isn't relevant to per-minute BTC forecasting. The Python code in
#     trader_stack/ that integrates it remains, in case you want to enable
#     it later. To install it manually: pip install -e third_party/TradingAgents
#     (and clone it first).
#
# Requirements: git, python3.11 (or 3.12), curl. GPU optional but speeds things up.

set -euo pipefail

# ---------------------------------------------------------------------------
# 0. Sanity
# ---------------------------------------------------------------------------

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "==> trader-stack installer (BTC minute oracle)"
echo "==> Working in: $ROOT"

PY="${PYTHON:-python3.11}"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "ERROR: $PY not found. Install Python 3.11 (or set PYTHON=python3.12) and rerun." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# 1. Clone Kronos and TimesFM (BTC oracle dependencies)
# ---------------------------------------------------------------------------

mkdir -p third_party
cd third_party

clone_if_missing () {
  local url="$1" dir="$2"
  if [ -d "$dir/.git" ]; then
    echo "==> $dir already cloned, skipping"
  else
    echo "==> Cloning $dir"
    git clone --depth 1 "$url" "$dir"
  fi
}

clone_if_missing https://github.com/shiyu-coder/Kronos.git      Kronos
clone_if_missing https://github.com/google-research/timesfm.git timesfm

# TradingAgents is OPTIONAL — only install if --with-multi-agent flag is passed.
WITH_MULTI_AGENT=false
for arg in "$@"; do
  case "$arg" in
    --with-multi-agent) WITH_MULTI_AGENT=true ;;
  esac
done

if [ "$WITH_MULTI_AGENT" = "true" ]; then
  clone_if_missing https://github.com/TauricResearch/TradingAgents.git TradingAgents
fi

# Optional: pretrain the HMM during install. Without this the daemon's first
# day of regime labels is noisy. Pretrain takes ~5-10 min over the public
# Binance kline API and uses no API key.
PRETRAIN_HMM=false
for arg in "$@"; do
  case "$arg" in
    --pretrain-hmm) PRETRAIN_HMM=true ;;
  esac
done

cd "$ROOT"

# ---------------------------------------------------------------------------
# 2. Create the shared venv
# ---------------------------------------------------------------------------

if [ ! -d .venv ]; then
  echo "==> Creating venv with $PY"
  "$PY" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip wheel setuptools

# ---------------------------------------------------------------------------
# 3. PyTorch (CUDA on Linux+nvidia, CPU/MPS elsewhere)
# ---------------------------------------------------------------------------

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "==> CUDA GPU detected, installing torch with CUDA 12.4"
  pip install torch --index-url https://download.pytorch.org/whl/cu124
else
  echo "==> No CUDA detected, installing CPU torch (MPS on Apple Silicon works with this wheel)"
  pip install torch
fi

# ---------------------------------------------------------------------------
# 4. Forecasters
# ---------------------------------------------------------------------------

echo "==> Installing TimesFM (editable, torch backend)"
pip install -e "third_party/timesfm[torch]"

echo "==> Installing Kronos requirements (Kronos itself isn't a PyPI package)"
pip install -r third_party/Kronos/requirements.txt

# Put Kronos on the venv's path via a .pth file (scoped to this venv only).
SITE_PACKAGES="$(python -c 'import site; print(site.getsitepackages()[0])')"
echo "$ROOT/third_party/Kronos" > "$SITE_PACKAGES/kronos.pth"
echo "==> Wrote $SITE_PACKAGES/kronos.pth"

if [ "$WITH_MULTI_AGENT" = "true" ]; then
  echo "==> Installing TradingAgents (editable, multi-agent path enabled)"
  pip install -e third_party/TradingAgents
fi

# ---------------------------------------------------------------------------
# 5. trader-stack own deps + new forecasters (TiRex, Chronos-2) + CCXT
# ---------------------------------------------------------------------------

pip install -r requirements.txt

# ---------------------------------------------------------------------------
# 6. OpenClaw skill registration
# ---------------------------------------------------------------------------

if [[ "${SKIP_OPENCLAW:-false}" == "true" ]]; then
  echo "==> SKIP_OPENCLAW=true; skipping OpenClaw skill registration."
else
  SKILLS_ROOT="${OPENCLAW_SKILLS_DIR:-$HOME/.openclaw/skills}"
  if [ ! -d "$HOME/.openclaw" ]; then
    echo "==> OpenClaw not detected at ~/.openclaw — install it from https://openclaw.ai then"
    echo "    rerun this script (or manually): ln -s \"$ROOT/skills\" \"$SKILLS_ROOT/trader-stack\""
  else
    mkdir -p "$SKILLS_ROOT"
    TARGET="$SKILLS_ROOT/trader-stack"
    if [ -L "$TARGET" ] || [ -d "$TARGET" ]; then
      echo "==> $TARGET already exists; leaving as-is. Remove it manually to re-link."
    else
      ln -s "$ROOT/skills" "$TARGET"
      echo "==> Symlinked $ROOT/skills → $TARGET"
      echo "    OpenClaw will discover btc-oracle-status, btc-oracle-control, and"
      echo "    btc-oracle-backtest on its next session."
    fi
  fi
fi

# ---------------------------------------------------------------------------
# 7. Smoke test
# ---------------------------------------------------------------------------

echo "==> Running smoke test"
python - <<'PY'
import sys
ok = True

def check(name, mod):
    global ok
    try:
        __import__(mod)
        print(f"  [ok] {name}")
    except Exception as e:
        ok = False
        print(f"  [FAIL] {name}: {e}")

check("Kronos (module 'model')", "model")
check("TimesFM",                 "timesfm")
check("TiRex",                   "tirex")
check("Chronos (incl. Chronos-2)", "chronos")
check("filelock",                "filelock")
check("hmmlearn",                "hmmlearn")
check("trader_stack.btc_oracle", "trader_stack.btc_oracle")

sys.exit(0 if ok else 1)
PY

# ---------------------------------------------------------------------------
# 8. Optional: pretrain the HMM regime classifier
# ---------------------------------------------------------------------------

if [ "$PRETRAIN_HMM" = "true" ]; then
  echo
  echo "==> Pre-training the regime HMM on 90 days of BTC minute data."
  echo "    This downloads ~130k bars from Binance (no API key) and takes ~5-10 min."
  python scripts/pretrain_hmm.py --days 90
else
  echo
  echo "==> Skipping HMM pretrain. Run it later with:"
  echo "       source .venv/bin/activate"
  echo "       python scripts/pretrain_hmm.py --days 90"
  echo "    Without pretrain, regime labels are noisy for ~1 week."
fi

echo
echo "==> Done. Three ways to use it:"
echo
echo "  1) Standalone CLI:"
echo "       source .venv/bin/activate"
echo "       python oracle.py daemon start          # start the per-minute loop"
echo "       python oracle.py status                # see the latest signal"
echo "       python oracle.py forecast              # one-shot, no daemon"
echo "       python oracle.py backtest --n 500      # measure recent accuracy"
echo
echo "  2) Through OpenClaw (Gemma 4 talks to the lobster on WhatsApp/Telegram):"
echo "       Once your OpenClaw is configured with Gemma 4 (see openclaw.config.example.jsonc),"
echo "       message your lobster: 'start the BTC oracle' / 'what's the signal?' / 'backtest it'"
echo
echo "  3) Programmatically:"
echo "       from trader_stack.btc_oracle import daemon"
echo "       daemon.run_forever()                   # foreground"
