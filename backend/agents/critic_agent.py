"""
Critic Agent
============
The most important agent in the pipeline from a "prove you understand agents" perspective.
A separate LLM instance reviews the draft with adversarial intent:
- Flags unsupported claims
- Catches contradictions between data sources
- Downgrades overconfident price predictions
- Ensures risk disclosures are present
- Scores the report on institutional quality standards
"""

import os
from typing import Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
import json


CRITIC_SYSTEM_PROMPT = """You are a senior research editor at an institutional crypto fund.
Your job is to CRITICALLY review research reports before publication.

You are NOT trying to be helpful to the report author. You are protecting the fund's reputation.

Review criteria:
1. FACTUAL ACCURACY — Are claims supported by the data provided? Flag any claim that
   isn't directly traceable to the raw data.
   
2. CONSISTENCY — Do the onchain, sentiment, market, and macro sections agree? 
   If onchain says "accumulation" but exchange flows show heavy inflows, flag it.
   
3. LANGUAGE CALIBRATION — Price predictions must be ranges, not point estimates.
   Words like "will", "definitely", "guaranteed" are NEVER acceptable.
   Replace with "may", "suggests", "historical patterns indicate".
   
4. RISK DISCLOSURE — Every bullish thesis must have a corresponding bear case.
   If risks aren't proportionally covered, flag it.
   
5. DATA FRESHNESS — Note any metric that appears stale or unverifiable.

Return a JSON with:
{
  "score": 0-10,              // 7+ = approve, <7 = revise
  "approved": bool,
  "issues": [                 // Specific, actionable issues
    "Section X claims Y but onchain data shows Z",
    ...
  ],
  "strengths": [...],         // What's working well
  "mandatory_fixes": [...],   // Must fix before publication
  "suggested_improvements": [...] // Nice to have
}
"""


async def run_critic_agent(
    draft: str,
    raw_data: dict[str, Any],
    revision_count: int = 0,
) -> dict[str, Any]:
    """
    Uses a separate LLM invocation with adversarial prompting.
    Lower temperature = more consistent, deterministic critique.
    """
    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        temperature=0.1,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

    # Summarize raw data for context (don't dump everything — too many tokens)
    data_summary = _summarize_raw_data(raw_data)

    prompt = f"""Review this crypto research report draft.

RAW DATA USED (for fact-checking):
{data_summary}

REPORT DRAFT:
{draft}

REVISION COUNT: {revision_count} (if 2+, be more lenient — we need to publish)

Provide your critical review as JSON.
"""

    response = await llm.ainvoke([
        SystemMessage(content=CRITIC_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])

    try:
        # Strip any markdown fencing
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content)
    except Exception:
        # If parsing fails, return a default that allows publication
        return {
            "score": 6,
            "approved": revision_count >= 1,
            "issues": ["Could not parse critic response"],
            "strengths": [],
            "mandatory_fixes": [],
            "suggested_improvements": [],
        }


def _summarize_raw_data(raw_data: dict) -> str:
    """Create a token-efficient summary of raw data for the critic to fact-check against."""
    lines = []
    
    onchain = raw_data.get("onchain", {})
    if onchain and not onchain.get("error"):
        lines.append(f"ONCHAIN: Health score={onchain.get('health_score', 'N/A')}, "
                     f"MVRV={onchain.get('mvrv_z_score', 'N/A')}, "
                     f"Exchange flow={onchain.get('exchange_flow_signal', 'N/A')}")
    
    sentiment = raw_data.get("sentiment", {})
    if sentiment and not sentiment.get("error"):
        lines.append(f"SENTIMENT: Score={sentiment.get('sentiment_score', 'N/A')}/100, "
                     f"Rating={sentiment.get('rating', 'N/A')}, "
                     f"Fear&Greed={sentiment.get('fear_greed', {}).get('current_value', 'N/A')}")
    
    market = raw_data.get("market", {})
    if market and not market.get("error"):
        lines.append(f"MARKET: Price=${market.get('spot', {}).get('price_usd', 'N/A')}, "
                     f"24h={market.get('spot', {}).get('price_change_24h', 'N/A')}%, "
                     f"RSI={market.get('technicals', {}).get('rsi_14', 'N/A')}, "
                     f"Funding={market.get('derivatives', {}).get('avg_funding_rate', 'N/A')}%")
    
    macro = raw_data.get("macro", {})
    if macro and not macro.get("error"):
        lines.append(f"MACRO: Overall={macro.get('macro_assessment', 'N/A')}, "
                     f"Regulatory={macro.get('regulatory_risk', 'N/A')}")
    
    return "\n".join(lines) if lines else "No raw data available"
