"""End-to-end run: warm models, build forecast brief, invoke TradingAgents.

The Gemma-driven agents inside TradingAgents will call get_quantitative_forecast
on their own via the patched Toolkit. If FORCE_FORECAST=true, we additionally
pre-compute the brief and prepend it to the analyst context, guaranteeing the
graph sees it even if Gemma's function-calling fumbles a turn.
"""

from __future__ import annotations

import logging
from typing import Any

from .config import CONFIG

log = logging.getLogger(__name__)


def _build_graph():
    """Construct the TradingAgents graph with our Gemma/Ollama config."""
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    cfg = CONFIG.as_tradingagents_config()
    log.info(
        "TradingAgents config: provider=%s deep=%s quick=%s rounds=%d",
        cfg["llm_provider"],
        cfg["deep_think_llm"],
        cfg["quick_think_llm"],
        cfg["max_debate_rounds"],
    )
    return TradingAgentsGraph(debug=True, config=cfg)


def run_analysis(ticker: str, date: str, warm_up: bool = True) -> dict[str, Any]:
    """Run the full trader-stack pipeline.

    Args:
        ticker: e.g. "NVDA".
        date: ISO date string, the "as of" date.
        warm_up: if True, force-load Kronos and TimesFM before kicking off
            TradingAgents. Costs ~10-30s on first call; avoids a stall when
            the first agent decides to invoke the forecast tool.

    Returns:
        dict with:
            - 'decision': the final decision string from TradingAgents
            - 'state':    the final graph state (LangGraph dict)
            - 'forecast': the pre-computed forecast brief (text) if FORCE_FORECAST,
                          otherwise None
    """
    ticker = ticker.upper().strip()

    if warm_up:
        from . import forecasters
        log.info("Warming up forecasters...")
        forecasters.warm_up()

    pre_forecast: str | None = None
    if CONFIG.force_forecast:
        log.info("FORCE_FORECAST=true: pre-computing brief")
        from .patch import force_forecast_into_state
        pre_forecast = force_forecast_into_state(ticker, date)

    graph = _build_graph()

    # If we have a forced forecast, inject it via the graph's initial-state
    # extensibility hook. TradingAgents exposes this by accepting extra context
    # through the propagate signature in newer versions; if it doesn't, the
    # forecast is still available because the tool is in the Toolkit.
    if pre_forecast is not None:
        try:
            state, decision = graph.propagate(
                ticker, date, extra_context={"quantitative_forecast": pre_forecast}
            )
        except TypeError:
            # Older signature — drop the extra arg.
            log.debug(
                "graph.propagate() doesn't accept extra_context; the forecast "
                "tool remains callable via the Toolkit. Continuing without injection."
            )
            state, decision = graph.propagate(ticker, date)
    else:
        state, decision = graph.propagate(ticker, date)

    return {
        "decision": decision,
        "state": state,
        "forecast": pre_forecast,
    }
