"""Market data fetcher. yfinance is free and key-less; that's what we use.

Returns a pandas DataFrame with columns: open, high, low, close, volume, amount.
'amount' = close * volume (a notional dollar-volume proxy; Kronos was trained
with a real `amount` column from Chinese exchanges, but for US data this proxy
is the standard substitute).
"""

from __future__ import annotations

import pandas as pd


def fetch_ohlcv(
    ticker: str,
    end_date: str,
    bars: int,
    interval: str = "1d",
) -> pd.DataFrame:
    """Fetch the most recent `bars` rows of OHLCV ending on or before `end_date`.

    Args:
        ticker: e.g. "NVDA", "AAPL", "BTC-USD".
        end_date: ISO date string, inclusive.
        bars: how many rows to return.
        interval: yfinance interval. "1d" for daily, "5m"/"15m"/"1h" for intraday.

    Returns:
        DataFrame indexed 0..bars-1 with columns
        ['timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount'].
    """
    import yfinance as yf

    end = pd.Timestamp(end_date) + pd.Timedelta(days=1)  # yfinance end is exclusive
    # Pull plenty of history then trim. Daily lookback ~ 3x bars covers weekends/holidays.
    if interval.endswith("d"):
        start = end - pd.Timedelta(days=int(bars * 1.8) + 30)
    elif interval.endswith("h"):
        start = end - pd.Timedelta(days=int(bars / 6) + 5)
    else:  # minutes
        start = end - pd.Timedelta(days=int(bars / 78) + 3)

    raw = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval=interval,
        auto_adjust=False,
        progress=False,
    )

    if raw.empty:
        raise ValueError(
            f"yfinance returned no data for {ticker} ending {end_date} "
            f"(interval={interval}). Check the ticker and date."
        )

    # yfinance returns MultiIndex columns when ticker is a list; flatten just in case.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )[["open", "high", "low", "close", "volume"]].copy()
    df["amount"] = df["close"] * df["volume"]
    df = df.dropna().tail(bars).reset_index()
    df = df.rename(columns={"Date": "timestamps", "Datetime": "timestamps"})
    df["timestamps"] = pd.to_datetime(df["timestamps"])

    if len(df) < bars * 0.5:
        raise ValueError(
            f"Got only {len(df)} bars for {ticker} ending {end_date} "
            f"(asked for {bars}). Not enough history."
        )

    return df
