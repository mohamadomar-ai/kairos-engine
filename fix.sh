#!/usr/bin/env bash
# Run this once. Fixes the three issues.
set -e
cd ~/trader-stack

# 1. Kill the TradingAgents warning by replacing the parent __init__.py
cat > trader_stack/__init__.py << 'PY'
"""trader_stack package root. Intentionally empty."""
PY

# 2. Make password persistent by appending to your shell config
PASS=$(cat ~/.trader-stack/.pgpass)
if ! grep -q "PG_PASSWORD" ~/.bashrc 2>/dev/null; then
    echo "" >> ~/.bashrc
    echo "# trader-stack Postgres password" >> ~/.bashrc
    echo "export PG_PASSWORD=\"$PASS\"" >> ~/.bashrc
fi
export PG_PASSWORD="$PASS"

# 3. Make sure psycopg2 + the minimum deps are installed in the venv
source .venv/bin/activate
pip install -q psycopg2-binary python-dotenv pandas requests

# 4. Verify
python -c "from trader_stack.btc_oracle.store import test_connection; print(test_connection())"
