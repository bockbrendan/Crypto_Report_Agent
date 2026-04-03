"""
Macro Agent
===========
Fetches regulatory news, macro conditions, BTC dominance, and protocol
developments using 5 free data sources in parallel.

Called by graph.py's parallel_research_node as:
    macro_data = await run_macro_agent(coin, plan)

Returns a dict matching AGENTS.md's macro_data shape.
"""
from __future__ import annotations

import asyncio
import json
import os
from textwrap import dedent

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ..tools.macro_tools import (
    fetch_btc_vs_spy_correlation,
    fetch_fear_greed,
    fetch_fed_funds_rate,
    fetch_global_market,
    fetch_regulatory_news,
)


# ── Pydantic output model ─────────────────────────────────────────────────────

class KeyEvent(BaseModel):
    event:  str
    impact: str   # "bullish" | "bearish" | "neutral"
    source: str


class MacroAssessment(BaseModel):
    macro_assessment:      str              # "bullish" | "neutral" | "bearish"
    regulatory_risk:       str              # "low" | "medium" | "high"
    regulatory_summary:    str
    macro_environment:     str              # "risk-on" | "risk-off" | "mixed"
    key_events:            list[KeyEvent]
    protocol_developments: list[str]
    interpretation:        str


# ── LLM factory ──────────────────────────────────────────────────────────────

def _llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        temperature=0.1,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )


# ── Main entrypoint (called by graph.py) ─────────────────────────────────────

async def run_macro_agent(coin: str, plan: dict) -> dict:
    """
    Fetches macro data from 5 sources in parallel, then calls Gemini for a
    structured assessment. Returns a dict matching AGENTS.md macro_data shape.
    Degrades gracefully — never raises.
    """
    cryptopanic_key = os.getenv("CRYPTOPANIC_API_KEY", "")
    fred_key        = os.getenv("FRED_API_KEY", "")

    # Fan out all fetches concurrently
    results = await asyncio.gather(
        fetch_global_market(),
        fetch_fear_greed(),
        fetch_regulatory_news(cryptopanic_key),
        fetch_fed_funds_rate(fred_key),
        fetch_btc_vs_spy_correlation(),
        return_exceptions=True,
    )

    global_mkt, fear_greed, reg_news, fed_rate, btc_spy = results

    def _safe(val: object, fallback: object) -> object:
        return fallback if isinstance(val, Exception) else val

    global_mkt = _safe(global_mkt, {})
    fear_greed  = _safe(fear_greed,  {})
    reg_news    = _safe(reg_news,    [])
    fed_rate    = _safe(fed_rate,    {"current_rate_pct": None, "direction": None})
    btc_spy     = _safe(btc_spy,     {"btc_30d_return_pct": None, "spy_30d_return_pct": None,
                                       "btc_outperforms": None})

    btc_dominance = global_mkt.get("btc_dominance_pct")

    raw_json = json.dumps({
        "coin":            coin,
        "focus_areas":     plan.get("focus_areas", []),
        "known_risks":     plan.get("known_risks", []),
        "global_market":   global_mkt,
        "fear_greed":      fear_greed,
        "fed_rate":        fed_rate,
        "btc_vs_spy":      btc_spy,
        "regulatory_news": reg_news[:8],
    }, indent=2, default=str)

    system_prompt = dedent("""
        You are a macro economist specialising in cryptocurrency markets.
        Analyse the provided real-time data and classify the macro environment.

        Signal interpretation:
        - BTC dominance > 55%      → risk-off rotation (bearish for alts)
        - Fear & Greed < 25        → extreme fear (contrarian bullish)
        - Fear & Greed > 75        → extreme greed (caution: overbought)
        - Fed rate rising           → tightening (bearish for risk assets)
        - Fed rate falling          → easing (bullish for risk assets)
        - BTC outperforming SPY 30d → crypto relative strength (bullish)
        - Negative regulatory news → elevated regulatory_risk

        Be specific. Cite numbers. If a data source is missing, note it and
        weight the remaining signals proportionally.
    """).strip()

    user_msg = dedent(f"""
        Macro data for {coin}:

        {raw_json}

        Produce a structured macro assessment. For key_events, extract the 3–5 most
        significant events from the regulatory news and macro data provided.
        For protocol_developments, list any {coin}-specific protocol news found.
        If none found, return an empty list.
    """).strip()

    try:
        result: MacroAssessment = _llm().with_structured_output(MacroAssessment).invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_msg),
        ])
        return {
            "macro_assessment":      result.macro_assessment,
            "regulatory_risk":       result.regulatory_risk,
            "regulatory_summary":    result.regulatory_summary,
            "macro_environment":     result.macro_environment,
            "btc_dominance":         btc_dominance,
            "key_events":            [e.model_dump() for e in result.key_events],
            "protocol_developments": result.protocol_developments,
            "interpretation":        result.interpretation,
        }
    except Exception as exc:
        return {
            "macro_assessment":      "neutral",
            "regulatory_risk":       "medium",
            "regulatory_summary":    f"Data fetch succeeded but LLM call failed: {exc}",
            "macro_environment":     "mixed",
            "btc_dominance":         btc_dominance,
            "key_events":            [],
            "protocol_developments": [],
            "interpretation":        "Macro assessment unavailable due to LLM error.",
            "error":                 str(exc),
        }
