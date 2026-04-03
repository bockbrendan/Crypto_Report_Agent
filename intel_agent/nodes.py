"""
LangGraph node functions for the Crypto Intel Agent.

Each node:
  - Receives the full CryptoIntelState
  - Calls data tools or an LLM
  - Returns a dict of ONLY the keys it modifies

Node order: resolve_asset → onchain_node → sentiment_node → docs_node → report_node
"""
from __future__ import annotations

import json
import math
import os
import re
import statistics
from textwrap import dedent

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from .state import CryptoIntelState
from .tools import (
    fetch_project_docs,
    get_coin_detail,
    get_crypto_news,
    get_defi_tvl,
    get_price_history,
    get_twitter_sentiment,
    search_coin,
)

# ── LLM factory ──────────────────────────────────────────────────────────────

def _llm(temperature: float = 0.1) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=temperature,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )


def _invoke(system: str, user: str) -> str:
    """Simple single-turn LLM call. Returns the response text."""
    resp = _llm().invoke([
        SystemMessage(content=system),
        HumanMessage(content=user),
    ])
    return resp.content


# ─────────────────────────────────────────────────────────────────────────────
# NODE 1 — resolve_asset
# ─────────────────────────────────────────────────────────────────────────────

def _compute_quant_metrics(onchain_raw: dict) -> dict:
    """
    Compute defensible quantitative risk metrics from raw on-chain data.

    Returns a dict with:
      - volatility_30d_pct      : annualised std dev of daily returns (%)
      - liquidity_ratio_pct     : 24h volume / market cap (%)
      - supply_overhang_pct     : circulating / max supply (%) — None if uncapped
      - tvl_to_mcap_pct         : TVL / market cap (%) — None if no TVL
    """
    mkt     = onchain_raw.get("market", {})
    history = onchain_raw.get("price_history_30d", [])
    tvl     = onchain_raw.get("defi_tvl") or {}

    # ── Volatility: annualised std dev of daily returns ───────────────────────
    prices = [p[1] for p in history if len(p) == 2 and p[1]]
    if len(prices) >= 2:
        daily_returns = [
            (prices[i] - prices[i - 1]) / prices[i - 1]
            for i in range(1, len(prices))
        ]
        vol_30d = statistics.stdev(daily_returns) * math.sqrt(365) * 100
    else:
        vol_30d = None

    # ── Liquidity ratio: 24h volume / market cap ──────────────────────────────
    volume = mkt.get("total_volume")
    mcap   = mkt.get("market_cap")
    if volume and mcap and mcap > 0:
        liquidity_pct = (volume / mcap) * 100
    else:
        liquidity_pct = None

    # ── Supply overhang: circulating / max supply ─────────────────────────────
    circulating = mkt.get("circulating_supply")
    max_supply  = mkt.get("max_supply")
    if circulating and max_supply and max_supply > 0:
        supply_pct = (circulating / max_supply) * 100
    else:
        supply_pct = None  # uncapped or unknown

    # ── TVL / market cap ──────────────────────────────────────────────────────
    tvl_usd = tvl.get("tvl_usd")
    if tvl_usd and mcap and mcap > 0:
        tvl_to_mcap_pct = (tvl_usd / mcap) * 100
    else:
        tvl_to_mcap_pct = None

    return {
        "volatility_30d_ann_pct":  round(vol_30d, 1)        if vol_30d        is not None else None,
        "liquidity_ratio_pct":     round(liquidity_pct, 2)  if liquidity_pct  is not None else None,
        "supply_overhang_pct":     round(supply_pct, 1)     if supply_pct     is not None else None,
        "tvl_to_mcap_pct":         round(tvl_to_mcap_pct, 2) if tvl_to_mcap_pct is not None else None,
    }


def _extract_asset_term(query: str) -> str:
    """
    Strip natural-language framing from the query and return just the asset name/ticker.
    E.g. "analyze solana" → "solana", "risk profile of BTC" → "BTC"
    Falls back to the raw query if extraction fails.
    """
    result = _invoke(
        system=(
            "Extract ONLY the cryptocurrency asset name or ticker symbol from the user query. "
            "Return just the name or ticker — no other words, no punctuation, no explanation. "
            "Examples: 'analyze solana' → 'solana', 'risk of BTC' → 'BTC', "
            "'is chainlink safe?' → 'chainlink'"
        ),
        user=query,
    )
    extracted = result.strip().split()[0] if result.strip() else query
    return extracted


def resolve_asset(state: CryptoIntelState) -> dict:
    """
    Map the raw user query to a CoinGecko coin_id / symbol / name.
    Extracts the asset term via LLM, then calls CoinGecko /search to disambiguate.
    """
    query       = state["query"]
    search_term = _extract_asset_term(query)

    try:
        candidates = search_coin(search_term)
        # Fallback: try the raw query if extraction returned something unexpected
        if not candidates:
            candidates = search_coin(query)
    except Exception as exc:
        return {
            "coin_id": None,
            "symbol":  None,
            "name":    None,
            "messages": [AIMessage(content=f"[resolve_asset] Search failed: {exc}")],
        }

    if not candidates:
        return {
            "coin_id": None,
            "symbol":  None,
            "name":    None,
            "messages": [AIMessage(content=f"[resolve_asset] No results for query: {query}")],
        }

    # Use LLM to pick the best match
    candidates_json = json.dumps(candidates, indent=2)
    answer = _invoke(
        system=(
            "You are an asset resolver. Pick the single best-matching crypto asset "
            "from the candidates. Return ONLY valid JSON: "
            '{"coin_id": "...", "symbol": "...", "name": "..."}'
        ),
        user=f'User query: "{query}"\n\nCandidates:\n{candidates_json}',
    )

    # Parse JSON from LLM response (handle markdown code fences)
    m = re.search(r'\{[^}]+\}', answer, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group())
            coin_id = obj.get("coin_id", candidates[0]["id"])
            symbol  = obj.get("symbol",  candidates[0]["symbol"])
            name    = obj.get("name",    candidates[0]["name"])
        except json.JSONDecodeError:
            coin_id, symbol, name = candidates[0]["id"], candidates[0]["symbol"], candidates[0]["name"]
    else:
        coin_id, symbol, name = candidates[0]["id"], candidates[0]["symbol"], candidates[0]["name"]

    return {
        "coin_id": coin_id,
        "symbol":  symbol,
        "name":    name,
        "messages": [
            AIMessage(content=f"[resolve_asset] Resolved '{query}' → {name} ({symbol}) [id: {coin_id}]")
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 2 — onchain_node
# ─────────────────────────────────────────────────────────────────────────────

def onchain_node(state: CryptoIntelState) -> dict:
    """
    Gather on-chain / market metrics from CoinGecko + DeFiLlama,
    then use LLM to write a metrics analysis section.
    """
    coin_id = state["coin_id"]
    symbol  = state["symbol"]
    name    = state["name"]

    # Fetch data
    try:
        coin_data     = get_coin_detail(coin_id)
        price_history = get_price_history(coin_id, days=30)
        tvl_data      = get_defi_tvl(symbol)
    except Exception as exc:
        return {
            "onchain_raw":       {},
            "metrics_analysis":  f"[onchain_node] Failed to fetch data: {exc}",
            "messages":          [AIMessage(content=f"[onchain_node] Error: {exc}")],
        }

    raw = {**coin_data, "price_history_30d": price_history, "defi_tvl": tvl_data}

    # Build analysis prompt
    mkt = coin_data.get("market", {})
    dev = coin_data.get("developer", {})

    prompt_data = json.dumps({
        "asset":          f"{name} ({symbol})",
        "market":         mkt,
        "developer":      dev,
        "tvl":            tvl_data,
        "price_30d_data": price_history[-7:],   # last 7 points for brevity
    }, default=str, indent=2)

    analysis = _invoke(
        system=dedent("""
            You are a senior crypto market analyst. Analyze the on-chain and market
            metrics for the given asset. Your analysis should cover:

            1. **Price Action & Momentum** – current price, 24h/7d/30d performance,
               distance from ATH (is it recovering or capitulating?).
            2. **Market Structure** – market cap rank, FDV vs market cap ratio
               (is supply overhang a concern?), circulating vs max supply.
            3. **Liquidity & Volume** – 24h volume relative to market cap
               (healthy = >5%; concerning = <1%).
            4. **DeFi TVL** (if available) – size of TVL, chain diversification,
               recent TVL trend.
            5. **Developer Health** – GitHub commits, open issues, PR activity.
               Is the team actively building?

            Be specific with numbers. Use markdown headers for each section.
            Keep total length under 500 words. End with a one-sentence verdict.
        """),
        user=f"On-chain data for analysis:\n\n{prompt_data}",
    )

    return {
        "onchain_raw":      raw,
        "metrics_analysis": analysis,
        "messages":         [AIMessage(content=f"[onchain_node] Metrics analysis complete for {name}.")],
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 3 — sentiment_node
# ─────────────────────────────────────────────────────────────────────────────

def sentiment_node(state: CryptoIntelState) -> dict:
    """
    Gather news + social data from CryptoCompare (and optionally Twitter),
    then use LLM to write a sentiment analysis section.
    """
    symbol = state["symbol"]
    name   = state["name"]

    # Fetch news
    news    = get_crypto_news(symbol, limit=15)
    tweets  = get_twitter_sentiment(symbol, name, max_results=50)

    # Community data from the onchain_raw we already have
    onchain_raw = state.get("onchain_raw") or {}
    community   = onchain_raw.get("community", {})

    prompt_data = json.dumps({
        "asset":       f"{name} ({symbol})",
        "community":   community,
        "news_count":  len(news),
        "news":        news[:12],   # top 12 articles
        "tweets":      tweets[:20] if tweets else "Twitter data not available",
        "twitter_available": bool(tweets),
    }, default=str, indent=2)

    analysis = _invoke(
        system=dedent("""
            You are a crypto sentiment analyst. Analyze the social and news sentiment
            for the given asset. Your analysis should cover:

            1. **Community Size & Activity** – Twitter followers, Reddit subscribers,
               engagement rates. Is the community growing or shrinking?
            2. **News Sentiment** – Summarize the dominant narrative from recent headlines.
               Is coverage bullish (partnerships, adoption) or bearish (hacks, regulation)?
            3. **Key Themes** – List 3–5 recurring themes in the news/tweets. What is the
               market narrative right now?
            4. **Sentiment Score** – Rate overall sentiment: Strongly Bullish / Bullish /
               Neutral / Bearish / Strongly Bearish, with a brief justification.
            5. **Red Flags** – Any concerning narratives (FUD, controversy, leadership issues)?

            If Twitter data is unavailable, note this and rely on news + community metrics.
            Use markdown headers. Keep under 400 words.
        """),
        user=f"Sentiment data for analysis:\n\n{prompt_data}",
    )

    return {
        "news_raw":           news,
        "twitter_raw":        tweets,
        "sentiment_analysis": analysis,
        "messages": [
            AIMessage(content=(
                f"[sentiment_node] Sentiment analysis complete. "
                f"{len(news)} news articles, {len(tweets)} tweets analyzed."
            ))
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 4 — docs_node
# ─────────────────────────────────────────────────────────────────────────────

def docs_node(state: CryptoIntelState) -> dict:
    """
    Fetch project homepage and whitepaper text, plus CoinGecko description
    and project links. Use LLM to write a fundamentals analysis section.
    """
    coin_id     = state["coin_id"]
    symbol      = state["symbol"]
    name        = state["name"]
    onchain_raw = state.get("onchain_raw") or {}

    # Fetch docs text from homepage/whitepaper
    try:
        docs = fetch_project_docs(onchain_raw)
    except Exception as exc:
        docs = {"error": str(exc)}

    description = onchain_raw.get("description", "")
    categories  = onchain_raw.get("categories",  [])
    links       = onchain_raw.get("links",        {})
    genesis     = onchain_raw.get("genesis_date", "unknown")

    prompt_data = json.dumps({
        "asset":             f"{name} ({symbol})",
        "categories":        categories,
        "genesis_date":      genesis,
        "description":       description,
        "homepage_text":     (docs.get("homepage_text",   "") or "")[:2000],
        "whitepaper_text":   (docs.get("whitepaper_text", "") or "")[:2000],
        "github_repos":      links.get("github",    []),
        "has_whitepaper":    bool(links.get("whitepaper")),
        "developer_stats":   onchain_raw.get("developer", {}),
    }, default=str, indent=2)

    analysis = _invoke(
        system=dedent("""
            You are a crypto fundamental analyst performing due diligence.
            Analyze the project documentation and fundamentals. Cover:

            1. **What Is It?** – In 2–3 sentences, what problem does this project solve?
               What is the use case and target market?
            2. **Technology & Architecture** – Based on the description and whitepaper,
               assess the technical approach. Is it novel or derivative? Layer 1/2/DeFi/NFT?
            3. **Team & Governance** – Is there evidence of an active team? Open-source?
               DAO-governed? Centralization concerns? Is it founder led?
            4. **Maturity** – When was it launched? How long has it survived? Project age
               is a proxy for survival probability.
            5. **Documentation Quality** – Does a whitepaper exist? Is the homepage
               professional and informative? Poor docs = red flag.
            6. **Competitive Positioning** – What category is it in? Who are the main
               competitors? Does it have a defensible moat?

            Use markdown headers. Keep under 500 words. Be direct — no hype.
        """),
        user=f"Documentation and fundamentals data:\n\n{prompt_data}",
    )

    return {
        "docs_raw":      docs,
        "docs_analysis": analysis,
        "messages": [
            AIMessage(content=f"[docs_node] Fundamentals analysis complete for {name}.")
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 5 — report_node
# ─────────────────────────────────────────────────────────────────────────────

class _ReportOutput(BaseModel):
    risk_score: int   = Field(ge=1, le=10, description="1 = safest, 10 = highest risk")
    report:     str   = Field(description="Full markdown due diligence report")


def report_node(state: CryptoIntelState) -> dict:
    """
    Synthesize all prior analyses into a final risk report with a 1–10 risk score.
    Computes hard quantitative metrics first, feeds them to the LLM as grounding facts,
    then asks it to interpret and justify the score using those numbers.
    """
    name        = state["name"]
    symbol      = state["symbol"]
    onchain_raw = state.get("onchain_raw") or {}

    quant = _compute_quant_metrics(onchain_raw)

    def _fmt(val: float | None, suffix: str = "%") -> str:
        return f"{val}{suffix}" if val is not None else "N/A"

    quant_block = dedent(f"""
        ## Quantitative Risk Factors (calculated, not estimated)

        | Metric | Value | Benchmark |
        |---|---|---|
        | 30-day Annualised Volatility | {_fmt(quant["volatility_30d_ann_pct"])} | <50% low · 50–120% medium · >120% high |
        | Liquidity Ratio (vol/mcap)   | {_fmt(quant["liquidity_ratio_pct"])} | >5% healthy · 1–5% thin · <1% illiquid |
        | Supply Overhang (circ/max)   | {_fmt(quant["supply_overhang_pct"])} | >80% low dilution · <50% high dilution risk |
        | TVL / Market Cap             | {_fmt(quant["tvl_to_mcap_pct"])} | >50% strong · 10–50% moderate · <10% low |

        Use these exact numbers to anchor the Risk Score. The score must be
        mathematically consistent with the table above — explain which metrics
        drove it up or down.
    """)

    synthesis_prompt = dedent(f"""
        You are a senior crypto investment analyst writing a full due diligence report
        for **{name} ({symbol})**.

        Below are three pre-written analysis sections. Synthesize them into ONE cohesive,
        investor-grade report. Do not simply concatenate — weave the insights together,
        highlight contradictions, and provide a unified verdict.

        ---
        ## ON-CHAIN METRICS ANALYSIS
        {state.get("metrics_analysis", "Not available.")}

        ---
        ## SENTIMENT ANALYSIS
        {state.get("sentiment_analysis", "Not available.")}

        ---
        ## FUNDAMENTALS ANALYSIS
        {state.get("docs_analysis", "Not available.")}

        ---
        {quant_block}
        ---

        Your report MUST follow this exact markdown structure:

        # {name} ({symbol}) — Crypto Due Diligence Report

        ## Executive Summary
        [2-3 sentence high-level verdict]

        ## Asset Overview
        [What it is, use case, category]

        ## On-Chain Metrics
        [Price, market cap, volume, supply dynamics, TVL if applicable, developer activity]

        ## Market Sentiment & Narratives
        [News themes, social sentiment, community health, key narratives]

        ## Fundamentals & Technology
        [Use case, tech stack, whitepaper quality, team activity, competitive moat]

        ## Risk Analysis
        ### Market Risk
        [Cite exact volatility % and liquidity ratio % from the Quantitative Risk Factors table. Compare to benchmarks.]
        ### Technology Risk
        [Code quality, audit status, centralization]
        ### Regulatory & Macro Risk
        [Exposure to regulation, sector headwinds]
        ### Team & Governance Risk
        [Centralization, anonymous team, DAO maturity]

        ## Bull Case
        [3 bullet points — best-case narrative]

        ## Bear Case
        [3 bullet points — worst-case narrative]

        ## Risk Score: X/10
        **Score: [1–10]** — [One sentence explaining the score, citing specific quant metrics that drove it]
        (1 = blue chip, minimal risk | 10 = extremely speculative/dangerous)

        ## Verdict
        [Final recommendation: STRONG BUY / BUY / HOLD / AVOID / STRONG AVOID]
        [2-3 sentence justification]

        ---
        *Report generated by CryptoIntelAgent. Not financial advice.*
    """)

    structured_llm = _llm().with_structured_output(_ReportOutput)

    try:
        result: _ReportOutput = structured_llm.invoke([
            SystemMessage(content=(
                "You are a professional crypto analyst. Generate a complete, detailed "
                "due diligence report following the exact markdown structure provided. "
                "Set risk_score as an integer 1-10 matching the ## Risk Score section."
            )),
            HumanMessage(content=synthesis_prompt),
        ])
        risk_score = result.risk_score
        report     = result.report
    except Exception:
        # Fallback: unstructured call, extract risk score manually
        resp      = _invoke(system="You are a professional crypto analyst.", user=synthesis_prompt)
        report    = resp
        m         = re.search(r'Risk Score.*?(\d+)\s*/\s*10', resp, re.IGNORECASE)
        risk_score = int(m.group(1)) if m else 5

    return {
        "risk_score": risk_score,
        "report":     report,
        "messages":   [
            AIMessage(content=f"[report_node] Report complete. Risk score: {risk_score}/10")
        ],
    }
