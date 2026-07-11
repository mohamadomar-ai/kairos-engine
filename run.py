#!/usr/bin/env python3
"""trader-stack CLI entry point.

Two modes:
  * Human mode (default): prints a readable report.
  * JSON mode (--json):   prints a single JSON object to stdout, all logs to stderr.
                          Use this when invoking from OpenClaw / any LLM that parses output.

Usage:
    python run.py --ticker NVDA --date 2026-01-15
    python run.py --ticker NVDA --date 2026-01-15 --json
    python run.py --forecast-only --ticker BTC-USD --date 2026-01-15 --json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys


def _parse_decision(text: str) -> tuple[str, str]:
    """Extract BUY/HOLD/SELL + confidence from the portfolio manager's free-text reply.

    TradingAgents returns the decision as natural language; we pattern-match it.
    Returns ('BUY'|'HOLD'|'SELL'|'UNKNOWN', 'high'|'medium'|'low').
    """
    upper = text.upper()
    if re.search(r"\bSTRONG\s+BUY\b|\bBUY\b", upper) and not re.search(r"\bDO\s+NOT\s+BUY\b|\bNOT\s+BUY\b", upper):
        decision = "BUY"
    elif re.search(r"\bSELL\b", upper) and not re.search(r"\bDO\s+NOT\s+SELL\b|\bNOT\s+SELL\b", upper):
        decision = "SELL"
    elif re.search(r"\bHOLD\b", upper):
        decision = "HOLD"
    else:
        decision = "UNKNOWN"

    if re.search(r"HIGH\s+CONFIDENCE|STRONGLY", upper):
        conf = "high"
    elif re.search(r"LOW\s+CONFIDENCE|UNCERTAIN|TENTATIVE", upper):
        conf = "low"
    else:
        conf = "medium"

    return decision, conf


def _agreement_label(kronos_pct: float, timesfm_pct: float) -> str:
    same_sign = (kronos_pct > 0) == (timesfm_pct > 0)
    gap = abs(kronos_pct - timesfm_pct)
    if same_sign and gap < 0.02:
        return "STRONG AGREEMENT"
    if same_sign:
        return "DIRECTIONAL AGREEMENT"
    if abs(kronos_pct) < 0.005 and abs(timesfm_pct) < 0.005:
        return "BOTH FLAT"
    return "DISAGREEMENT"


def main() -> int:
    parser = argparse.ArgumentParser(description="trader-stack: Gemma-brained multi-agent trader")
    parser.add_argument("--ticker", required=True, help="Ticker symbol, e.g. NVDA")
    parser.add_argument("--date", required=True, help='"As of" date, YYYY-MM-DD')
    parser.add_argument(
        "--no-warm-up",
        action="store_true",
        help="Skip pre-loading the forecasters",
    )
    parser.add_argument(
        "--forecast-only",
        action="store_true",
        help="Skip the multi-agent debate, just run Kronos + TimesFM",
    )
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Emit a structured JSON object on stdout (logs go to stderr). Use this from OpenClaw.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    # When emitting JSON, send all logs to stderr so stdout stays parseable.
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    ticker = args.ticker.upper()

    # ----- forecast-only path -------------------------------------------------
    if args.forecast_only:
        from trader_stack.forecast_brief import build_forecast_brief
        result = build_forecast_brief(ticker, args.date)

        if args.as_json:
            kr = result["kronos"]
            tf = result["timesfm"]
            out = {
                "ticker": ticker,
                "date": args.date,
                "mode": "forecast_only",
                "kronos": {
                    "pct_change": kr.pct_change,
                    "max_drawdown": kr.max_drawdown,
                    "max_runup": kr.max_runup,
                    "last_close": kr.last_close,
                    "final_close": kr.final_close,
                },
                "timesfm": {
                    "pct_change_point": tf.pct_change_point,
                    "p10_pct": tf.p10_final / tf.last_close - 1,
                    "p90_pct": tf.p90_final / tf.last_close - 1,
                    "final_point": tf.final_point,
                },
                "agreement": _agreement_label(kr.pct_change, tf.pct_change_point),
                "brief_text": result["brief"],
            }
            print(json.dumps(out, indent=2))
        else:
            print(result["brief"])
        return 0

    # ----- full pipeline path -------------------------------------------------
    from trader_stack import run_analysis

    result = run_analysis(
        ticker=ticker,
        date=args.date,
        warm_up=not args.no_warm_up,
    )

    decision_text = str(result["decision"])
    decision_label, confidence = _parse_decision(decision_text)

    if args.as_json:
        # Reach into the final state for the forecast numbers if they're there.
        # Forecast may or may not have been computed depending on whether the
        # LLM invoked the tool; fall back to recomputing if we need to.
        kr = tf = None
        try:
            from trader_stack.forecast_brief import build_forecast_brief
            brief = build_forecast_brief(ticker, args.date)
            kr = brief["kronos"]
            tf = brief["timesfm"]
        except Exception as e:
            logging.warning("Could not compute forecast numbers for JSON output: %s", e)

        out: dict = {
            "ticker": ticker,
            "date": args.date,
            "mode": "full",
            "decision": decision_label,
            "confidence": confidence,
            "rationale": decision_text,
        }
        if kr and tf:
            out["kronos"] = {
                "pct_change": kr.pct_change,
                "max_drawdown": kr.max_drawdown,
                "max_runup": kr.max_runup,
            }
            out["timesfm"] = {
                "pct_change_point": tf.pct_change_point,
                "p10_pct": tf.p10_final / tf.last_close - 1,
                "p90_pct": tf.p90_final / tf.last_close - 1,
            }
            out["agreement"] = _agreement_label(kr.pct_change, tf.pct_change_point)
        print(json.dumps(out, indent=2))
    else:
        print("=" * 70)
        print(f"DECISION for {ticker} as of {args.date}: {decision_label} ({confidence})")
        print("=" * 70)
        print(decision_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
