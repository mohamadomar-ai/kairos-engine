"""LangChain tool that exposes the Kronos+TimesFM forecast brief to TradingAgents.

The Technical Analyst agent (and any other agent that picks it up) can call
this tool during its LangGraph turn. The tool returns the text brief, which
the LLM then incorporates into its written analysis.
"""

from __future__ import annotations

import logging

from langchain_core.tools import tool

from .forecast_brief import build_forecast_brief

log = logging.getLogger(__name__)

# We cache per (ticker, date) for the life of the process — within a single
# TradingAgents run the same forecast may be requested by multiple agents.
_brief_cache: dict[tuple[str, str], str] = {}


@tool
def get_quantitative_forecast(ticker: str, date: str) -> str:
    """Get a quantitative ML forecast brief for the given ticker as of the given date.

    Runs two foundation models — Kronos (OHLCV multivariate) and TimesFM 2.5
    (univariate quantile) — and returns a synthesized text brief with:
      - point forecasts from both models
      - p10/p90 confidence band from TimesFM
      - max run-up and max drawdown from Kronos
      - a cross-model agreement read

    Use this whenever you want a model-based view of the price path over
    the next ~30 business days. Treat its output as a quantitative signal
    to weigh against fundamentals, news, and sentiment — not as a verdict.

    Args:
        ticker: Stock or asset ticker, e.g. "NVDA", "AAPL", "BTC-USD".
        date: ISO date string "YYYY-MM-DD" — the "as of" date for the forecast.

    Returns:
        A multi-line text brief with the forecast results.
    """
    key = (ticker.upper(), date)
    if key in _brief_cache:
        return _brief_cache[key]

    try:
        result = build_forecast_brief(ticker.upper(), date)
        brief = result["brief"]
    except Exception as e:  # surface failures as tool output, not exceptions
        log.exception("get_quantitative_forecast failed for %s @ %s", ticker, date)
        brief = (
            f"QUANTITATIVE FORECAST UNAVAILABLE for {ticker} as of {date}.\n"
            f"Reason: {type(e).__name__}: {e}\n"
            f"Proceed with the other analyst inputs; do not invent a forecast."
        )

    _brief_cache[key] = brief
    return brief


def clear_cache() -> None:
    """Reset the in-process forecast cache. Useful between unrelated runs."""
    _brief_cache.clear()
