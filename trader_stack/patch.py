"""Patches TradingAgents' Toolkit at import time so the new forecast tool is
visible to the analyst agents without forking the upstream repo.

How TradingAgents discovers tools (as of v0.2.5):
    The `Toolkit` class in tradingagents.agents.utils.agent_utils collects
    @tool-decorated methods. When the LangGraph nodes for each analyst are
    built, they pull the relevant subset of toolkit methods and bind them
    to the LLM via `llm.bind_tools(...)`. The Technical Analyst gets the
    technical-indicator tools; we attach get_quantitative_forecast there.

This patch is idempotent and safe to call multiple times.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)
_PATCHED = False


def apply_patches() -> None:
    """Attach get_quantitative_forecast to the TradingAgents Toolkit class."""
    global _PATCHED
    if _PATCHED:
        return

    try:
        from tradingagents.agents.utils import agent_utils
    except ImportError:
        log.warning(
            "TradingAgents not importable yet; patch will be retried on next "
            "apply_patches() call. (This is fine if you're just running unit tests.)"
        )
        return

    from .tools import get_quantitative_forecast

    Toolkit = getattr(agent_utils, "Toolkit", None)
    if Toolkit is None:
        log.error(
            "Could not find Toolkit class in tradingagents.agents.utils.agent_utils. "
            "TradingAgents internal layout may have changed; tool will not be registered."
        )
        return

    # Attach as a class attribute. LangChain @tool objects are callable and
    # discoverable; TradingAgents iterates over toolkit attributes when
    # building the analyst tool lists.
    if not hasattr(Toolkit, "get_quantitative_forecast"):
        Toolkit.get_quantitative_forecast = staticmethod(get_quantitative_forecast)
        log.info("Attached get_quantitative_forecast to TradingAgents.Toolkit")
    else:
        log.debug("Toolkit already has get_quantitative_forecast; skipping")

    # ------------------------------------------------------------------
    # Optional: also patch the Technical Analyst prompt so it knows about
    # the new tool. If the analyst module's layout changes upstream, we
    # silently skip — the tool will still be callable, just less discoverable.
    # ------------------------------------------------------------------
    _augment_technical_analyst_prompt()

    _PATCHED = True


def _augment_technical_analyst_prompt() -> None:
    """Best-effort: nudge the Technical Analyst's system prompt to mention the
    new tool. Wrapped in a broad try/except because upstream prompt layout
    is not part of any public contract."""
    try:
        from tradingagents.agents.analysts import market_analyst
    except ImportError:
        log.debug("market_analyst module not found; skipping prompt augmentation")
        return

    nudge = (
        "\n\nADDITIONAL TOOL AVAILABLE:\n"
        "You have access to `get_quantitative_forecast(ticker, date)`, which returns "
        "a forecast brief from two ML foundation models (Kronos for OHLCV and "
        "TimesFM 2.5 for quantile forecasts). When asked to analyze price action "
        "or future direction, call this tool and weave its output into your report "
        "as a 'Quantitative Forecast' section alongside your indicator-based analysis. "
        "Always cite the brief's numbers explicitly; do not paraphrase the percentages.\n"
    )

    # Find any module-level string that looks like the system prompt and append.
    # Defensive: walk the module's globals rather than guessing the name.
    patched_any = False
    for name, value in list(vars(market_analyst).items()):
        if (
            isinstance(value, str)
            and len(value) > 200
            and "technical" in value.lower()
            and "analyst" in value.lower()
            and "QUANTITATIVE FORECAST" not in value
        ):
            setattr(market_analyst, name, value + nudge)
            log.info("Augmented market_analyst.%s with forecast-tool hint", name)
            patched_any = True

    if not patched_any:
        log.debug(
            "No matching prompt string in market_analyst; the tool is still callable "
            "if the LLM picks it up from the toolkit. Set FORCE_FORECAST=true to bypass."
        )


def force_forecast_into_state(ticker: str, date: str) -> str:
    """When FORCE_FORECAST=true, the runner calls this BEFORE the graph runs and
    injects the brief into the initial state, sidestepping the LLM's tool-call
    decision. Useful when Gemma's function-calling is flaky."""
    from .tools import get_quantitative_forecast

    return get_quantitative_forecast.invoke({"ticker": ticker, "date": date})
