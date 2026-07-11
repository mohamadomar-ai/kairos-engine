#!/usr/bin/env bash
# trader-stack/scripts/init_postgres.sh
#
# One-command Postgres bring-up. Run this once before starting the daemon
# for the first time. Idempotent — safe to run multiple times.
#
#   $ ./scripts/init_postgres.sh
#
# What it does, in order:
#   1. Checks Docker is installed and the daemon is running
#   2. Creates ~/.trader-stack/pgdata/ if it doesn't exist
#   3. Generates a strong password and writes ~/.trader-stack/.pgpass (chmod 600)
#      (skipped if a password file already exists — we never overwrite)
#   4. docker compose up -d  (boots the container in the background)
#   5. Waits up to 60 s for Postgres to be healthy
#   6. Verifies the schema by running test_connection() from store.py
#
# Output is human-readable. Pass --json for machine-readable status.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

JSON_OUTPUT=false
for arg in "$@"; do
  case "$arg" in
    --json) JSON_OUTPUT=true ;;
  esac
done

log() {
  if [ "$JSON_OUTPUT" = "false" ]; then
    echo "$@"
  fi
}

fail() {
  if [ "$JSON_OUTPUT" = "true" ]; then
    echo "{\"ok\": false, \"error\": \"$1\"}"
  else
    echo "ERROR: $1" >&2
  fi
  exit 1
}

# ---------------------------------------------------------------------------
# 1. Sanity: Docker present and running?
# ---------------------------------------------------------------------------

log "==> Checking Docker"
if ! command -v docker >/dev/null 2>&1; then
  fail "Docker not installed. Install: curl -fsSL https://get.docker.com | sh"
fi
if ! docker info >/dev/null 2>&1; then
  fail "Docker daemon not running. Start it: sudo systemctl start docker (Linux) or open Docker Desktop"
fi

# `docker compose` (plugin) vs `docker-compose` (legacy). Prefer the plugin.
if docker compose version >/dev/null 2>&1; then
  DC="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
  DC="docker-compose"
else
  fail "Neither 'docker compose' (plugin) nor 'docker-compose' (legacy) is installed."
fi
log "    using: $DC"

# ---------------------------------------------------------------------------
# 2. Data directory
# ---------------------------------------------------------------------------

PGDATA="$HOME/.trader-stack/pgdata"
PGPASS_FILE="$HOME/.trader-stack/.pgpass"
mkdir -p "$PGDATA"
log "==> Data directory: $PGDATA"

# ---------------------------------------------------------------------------
# 3. Password — generate once, never overwrite
# ---------------------------------------------------------------------------

if [ -f "$PGPASS_FILE" ]; then
  log "==> Using existing password file at $PGPASS_FILE"
  POSTGRES_PASSWORD="$(cat "$PGPASS_FILE")"
else
  log "==> Generating new Postgres password"
  POSTGRES_PASSWORD="$(openssl rand -hex 24 2>/dev/null || head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n')"
  echo "$POSTGRES_PASSWORD" > "$PGPASS_FILE"
  chmod 600 "$PGPASS_FILE"
  log "    saved: $PGPASS_FILE  (mode 600)"
fi
export POSTGRES_PASSWORD

# ---------------------------------------------------------------------------
# 4. Boot the container
# ---------------------------------------------------------------------------

log "==> Starting Postgres container"
( cd docker && $DC up -d ) >/dev/null

# ---------------------------------------------------------------------------
# 5. Wait for healthy
# ---------------------------------------------------------------------------

log "==> Waiting for Postgres to be ready..."
for i in $(seq 1 30); do
  STATUS="$(docker inspect --format='{{.State.Health.Status}}' trader-stack-pg 2>/dev/null || echo unknown)"
  if [ "$STATUS" = "healthy" ]; then
    log "    ✓ Postgres is healthy"
    break
  fi
  if [ "$i" = "30" ]; then
    fail "Postgres did not become healthy within 60s. Check: docker logs trader-stack-pg"
  fi
  sleep 2
done

# ---------------------------------------------------------------------------
# 6. Verify schema via the Python store module
# ---------------------------------------------------------------------------

log "==> Verifying schema via store.test_connection()"

# Activate the venv if present (the caller might not have done it)
if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# Export the password so store.py picks it up
export PG_PASSWORD="$POSTGRES_PASSWORD"

RESULT="$(python - <<'PY'
import json, sys
try:
    from trader_stack.btc_oracle.store import test_connection
    print(json.dumps(test_connection()))
except Exception as e:
    print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}))
    sys.exit(1)
PY
)"

if [ "$JSON_OUTPUT" = "true" ]; then
  echo "$RESULT"
else
  echo "    $RESULT"
  echo
  if echo "$RESULT" | grep -q '"ok": true'; then
    echo "✓ Postgres is up and the schema is ready."
    echo
    echo "Password file: $PGPASS_FILE"
    echo "Connection:    psql -h 127.0.0.1 -p 5433 -U trader -d trader_stack"
    echo "Stop:          cd docker && $DC down"
    echo "Stop + wipe:   cd docker && $DC down -v   # also clears pgdata"
  else
    echo "⚠ Schema check failed. Output above should show why."
    exit 1
  fi
fi
