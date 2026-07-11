"""btc_oracle — minute-scale BTC direction oracle.

A daemon that, every minute, fetches:
  - 1m OHLCV from Binance via CCXT
  - Order book snapshot (depth imbalance)
  - Futures derivatives (funding rate, open interest, long/short ratio, taker volume)

Runs four foundation-model forecasters in parallel:
  - Kronos      (multivariate OHLCV)
  - TimesFM 2.5 (univariate close)
  - TiRex       (univariate close, xLSTM)
  - Chronos-2   (univariate close + microstructure covariates)

Combines forecasts and microstructure into a single UP / DOWN / FLAT
signal with a confidence score, written to disk for the OpenClaw skills
(and any other consumer) to read.

Entry points:
    from trader_stack.btc_oracle import daemon, state, signal
    daemon.run_forever(config)
    state.read_state()
"""

from .oracle_config import ORACLE_CONFIG, OracleConfig

__all__ = ["ORACLE_CONFIG", "OracleConfig"]
