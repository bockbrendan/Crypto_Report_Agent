"""
Market Agent
============
Fetches live market data including:
- Spot price + OHLCV via CoinGecko
- Derivatives: funding rates, open interest, liquidations (via Coinglass)
- Options: put/call ratio, IV skew (via Deribit)
- Technical indicators computed locally
"""

import os
import httpx
import asyncio
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
async def get_spot_data(coin: str) -> dict:
    """
    Fetch current price, volume, market cap, and OHLCV from CoinGecko.
    Also returns 7-day price change and 30-day performance.
    """
    coin_id = _coingecko_id(coin)
    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"https://api.coingecko.com/api/v3/coins/{coin_id}",
            params={
                "localization": "false",
                "tickers": "false",
                "market_data": "true",
                "community_data": "false",
                "developer_data": "false",
            }
        )
        r.raise_for_status()
        d = r.json()["market_data"]
        return {
            "price_usd": d["current_price"]["usd"],
            "market_cap": d["market_cap"]["usd"],
            "market_cap_rank": r.json().get("market_cap_rank"),
            "volume_24h": d["total_volume"]["usd"],
            "price_change_24h": d["price_change_percentage_24h"],
            "price_change_7d": d["price_change_percentage_7d"],
            "price_change_30d": d["price_change_percentage_30d"],
            "ath": d["ath"]["usd"],
            "ath_change_percentage": d["ath_change_percentage"]["usd"],
            "circulating_supply": d["circulating_supply"],
            "total_supply": d["total_supply"],
        }


_CG_BASE = "https://open-api-v4.coinglass.com"


def _cg_headers() -> dict:
    return {"CG-API-KEY": os.environ.get("COINGLASS_API_KEY", "")}


def _cg_pair(coin: str) -> str:
    """Convert coin ticker to futures pair symbol (BTC → BTCUSDT)."""
    return f"{coin.upper()}USDT"


@tool
async def get_derivatives_data(coin: str) -> dict:
    """
    Fetch perpetual futures funding rates and aggregated open interest from Coinglass v4.
    Uses OI-weighted funding rate (more accurate than simple average).
    """
    api_key = os.environ.get("COINGLASS_API_KEY", "")
    if not api_key:
        return {"error": "COINGLASS_API_KEY not set — derivatives data unavailable"}

    headers = _cg_headers()
    pair = _cg_pair(coin)

    async with httpx.AsyncClient(timeout=15) as client:
        fr_r, oi_r = await asyncio.gather(
            client.get(
                f"{_CG_BASE}/api/futures/funding-rate/oi-weight-history",
                params={"symbol": coin.upper(), "interval": "8h", "limit": 3},
                headers=headers,
            ),
            client.get(
                f"{_CG_BASE}/api/futures/open-interest/aggregated-history",
                params={"symbol": coin.upper(), "interval": "4h", "limit": 2},
                headers=headers,
            ),
            return_exceptions=True,
        )

    result: dict = {}

    if not isinstance(fr_r, Exception) and fr_r.status_code == 200:
        fr_json = fr_r.json()
        fr_data = fr_json.get("data", [])
        if fr_json.get("code") in ("401", "403"):
            result["funding_error"] = fr_json.get("msg")
        elif fr_data:
            latest = fr_data[-1]
            rate = float(latest.get("close") or latest.get("c") or 0)
            result.update({
                "funding_rate_pct":  round(rate * 100, 4),
                "funding_sentiment": (
                    "overheated_longs"  if rate >  0.01 else
                    "overheated_shorts" if rate < -0.01 else "neutral"
                ),
            })
    else:
        result["funding_error"] = str(fr_r) if isinstance(fr_r, Exception) else f"HTTP {fr_r.status_code}: {fr_r.text[:200]}"

    if not isinstance(oi_r, Exception) and oi_r.status_code == 200:
        oi_json = oi_r.json()
        oi_data = oi_json.get("data", [])
        if oi_json.get("code") in ("401", "403"):
            result["oi_error"] = oi_json.get("msg")
        elif oi_data:
            latest = oi_data[-1]
            prev   = oi_data[0] if len(oi_data) > 1 else latest
            oi_now  = float(latest.get("close") or latest.get("c") or 0)
            oi_prev = float(prev.get("close")   or prev.get("c")   or 1)
            result.update({
                "open_interest_usd":   oi_now,
                "oi_change_4h_pct":    round((oi_now - oi_prev) / oi_prev * 100, 3) if oi_prev else None,
            })
    else:
        result["oi_error"] = str(oi_r) if isinstance(oi_r, Exception) else f"HTTP {oi_r.status_code}: {oi_r.text[:200]}"

    return result if result else {"error": "No derivatives data returned"}


@tool
async def get_liquidation_data(coin: str) -> dict:
    """
    Get 24h liquidation totals and global long/short account ratio from Coinglass v4.
    Large long liquidations = forced selling cascade risk.
    Large short liquidations = short squeeze potential.
    """
    api_key = os.environ.get("COINGLASS_API_KEY", "")
    if not api_key:
        return {"error": "COINGLASS_API_KEY not set"}

    async with httpx.AsyncClient(timeout=15) as client:
        liq_r = await client.get(
            f"{_CG_BASE}/api/futures/liquidation/coin-list",
            headers=_cg_headers(),
        )

    if liq_r.status_code != 200:
        return {"error": f"HTTP {liq_r.status_code}: {liq_r.text[:200]}"}

    cg = liq_r.json()
    if cg.get("code") in ("401", "403"):
        return {"error": f"Plan restriction: {cg.get('msg')}"}

    coin_list = cg.get("data", [])
    entry = next((c for c in coin_list if c.get("symbol", "").upper() == coin.upper()), None)
    if not entry:
        return {"error": f"{coin} not found in liquidation coin-list"}

    long_liq  = entry.get("long_liquidation_usd_24h")
    short_liq = entry.get("short_liquidation_usd_24h")
    return {
        "long_liquidations_24h_usd":   long_liq,
        "short_liquidations_24h_usd":  short_liq,
        "total_liquidations_24h_usd":  entry.get("liquidation_usd_24h"),
        "long_liquidations_12h_usd":   entry.get("long_liquidation_usd_12h"),
        "short_liquidations_12h_usd":  entry.get("short_liquidation_usd_12h"),
        "dominant_side": (
            "longs"   if (long_liq  or 0) > (short_liq or 0) else
            "shorts"  if (short_liq or 0) > (long_liq  or 0) else "balanced"
        ),
    }


@tool
async def get_taker_volume(coin: str) -> dict:
    """
    Fetch aggregated taker buy vs sell volume from Coinglass v4 over last 24h.
    High taker buy volume = aggressive buying pressure (bullish).
    High taker sell volume = aggressive selling / distribution (bearish).
    """
    api_key = os.environ.get("COINGLASS_API_KEY", "")
    if not api_key:
        return {"error": "COINGLASS_API_KEY not set"}

    symbol = coin.upper()  # API takes coin ticker (BTC), not pair (BTCUSDT)
    # The endpoint returns the same cross-exchange aggregate regardless of which exchange
    # is passed — one request is sufficient.
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(
            f"{_CG_BASE}/api/futures/taker-buy-sell-volume/exchange-list",
            params={"exchange": "Binance", "symbol": symbol, "range": "24h", "limit": 1},
            headers=_cg_headers(),
        )

    if r.status_code != 200:
        return {"error": f"HTTP {r.status_code}: {r.text[:200]}"}

    cg = r.json()
    if cg.get("code") not in ("0", 0):
        return {"error": f"CoinGlass error {cg.get('code')}: {cg.get('msg','')}"}

    rec = cg.get("data") or {}
    if not rec:
        return {"error": "No taker volume data returned"}

    buy_vol  = float(rec.get("buy_vol_usd")  or 0)
    sell_vol = float(rec.get("sell_vol_usd") or 0)
    total    = buy_vol + sell_vol
    if total == 0:
        return {"error": "Taker volume data is zero"}

    # Per-exchange breakdown from exchange_list inside the aggregate response
    per_exchange = [
        {
            "exchange":     e.get("exchange"),
            "buy_vol_usd":  round(float(e.get("buy_vol_usd")  or 0), 2),
            "sell_vol_usd": round(float(e.get("sell_vol_usd") or 0), 2),
            "buy_ratio":    e.get("buy_ratio"),
            "sell_ratio":   e.get("sell_ratio"),
        }
        for e in (rec.get("exchange_list") or [])[:5]
    ]

    return {
        "taker_buy_volume_usd":  round(buy_vol,  2),
        "taker_sell_volume_usd": round(sell_vol, 2),
        "buy_ratio_pct":         round(float(rec.get("buy_ratio")  or buy_vol  / total * 100), 2),
        "sell_ratio_pct":        round(float(rec.get("sell_ratio") or sell_vol / total * 100), 2),
        "bias": (
            "strong_buying"  if buy_vol / total > 0.6 else
            "strong_selling" if buy_vol / total < 0.4 else "neutral"
        ),
        "per_exchange": per_exchange,
    }


@tool
async def compute_technical_indicators(coin: str) -> dict:
    """
    Compute RSI, MACD, Bollinger Bands, and support/resistance levels
    from OHLCV data fetched from Binance.
    """
    # Fetch 200 daily closes from CoinGecko market_chart (no API key, never geo-blocked)
    coin_id = _coingecko_id(coin)
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(
            f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart",
            params={"vs_currency": "usd", "days": 200, "interval": "daily"},
        )
        if r.status_code != 200:
            return {"error": f"CoinGecko market_chart returned {r.status_code} for {coin}"}

        prices = r.json().get("prices", [])
        if len(prices) < 21:
            return {"error": f"Insufficient price history for {coin} ({len(prices)} days)"}

        closes = [float(p[1]) for p in prices]
        
        # RSI (14-period)
        rsi = _compute_rsi(closes, 14)
        
        # Simple Bollinger Bands (20-period)
        bb = _compute_bollinger(closes, 20)
        
        # 50/200 MA crossover
        ma50 = sum(closes[-50:]) / 50
        ma200 = sum(closes[-200:]) / 200
        
        return {
            "rsi_14": round(rsi, 2),
            "rsi_signal": "overbought" if rsi > 70 else "oversold" if rsi < 30 else "neutral",
            "bb_upper": round(bb["upper"], 2),
            "bb_lower": round(bb["lower"], 2),
            "bb_width": round((bb["upper"] - bb["lower"]) / bb["mid"] * 100, 2),
            "ma50": round(ma50, 2),
            "ma200": round(ma200, 2),
            "golden_cross": ma50 > ma200,  # True = bullish signal
            "current_price_vs_ma200": round((closes[-1] - ma200) / ma200 * 100, 2),
        }


# ---------------------------------------------------------------------------
# Response parser (Gemini returns content as list[dict] content blocks)
# ---------------------------------------------------------------------------

def _parse_agent_response(content: Any, agent_name: str) -> dict:
    import json, re
    # Gemini via LangChain returns content as a list of typed blocks:
    # [{'type': 'text', 'text': '...', ...}]
    if isinstance(content, list):
        text = "".join(
            block.get("text", "") for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    else:
        text = str(content)

    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        result = json.loads(text)
        print(f"DEBUG: {agent_name} parsed OK, keys={list(result.keys())}")
        return result
    except json.JSONDecodeError as e:
        # LLM sometimes emits invalid escape sequences (e.g. \s, \: in tweet URLs).
        # Fix by escaping any backslash not followed by a valid JSON escape character.
        try:
            cleaned = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)
            result = json.loads(cleaned)
            print(f"DEBUG: {agent_name} parsed OK after escape fix, keys={list(result.keys())}")
            return result
        except Exception:
            pass
        print(f"DEBUG: {agent_name} parse FAILED — {e}")
        return {"raw": text[:500], "parse_error": True}


# ---------------------------------------------------------------------------
# Agent runner
# ---------------------------------------------------------------------------

MARKET_TOOLS = [get_spot_data, get_derivatives_data, get_liquidation_data, get_taker_volume, compute_technical_indicators]

SYSTEM_PROMPT = """You are an expert crypto market structure analyst. Your job is to fetch and
interpret market data to assess the current price trend, derivatives positioning, and technicals.

For any asset, always fetch:
1. Spot price data (current price, 24h/7d/30d changes, volume, ATH distance)
2. Derivatives data (funding rate, open interest aggregated across exchanges)
3. Liquidation data (24h long/short liquidations + long/short account ratio)
4. Taker buy/sell volume (directional conviction — who is the aggressor)
5. Technical indicators (RSI, Bollinger Bands, MA50/MA200 crossover)

CRITICAL: Copy every raw number from Coinglass tool results directly into your JSON output.
Do NOT paraphrase, round, or omit values — include the exact figures returned by the tools:
- funding_rate_pct (exact decimal), funding_sentiment label
- open_interest_usd (exact dollar amount), oi_change_4h_pct
- long_liquidations_24h_usd, short_liquidations_24h_usd, total_liquidations_24h_usd, dominant_side
- taker_buy_volume_usd, taker_sell_volume_usd, buy_ratio_pct, sell_ratio_pct, bias
- top_exchanges list with per-exchange buy/sell ratios

Interpret each metric and assess overall market structure as bullish, bearish, or neutral.
Return structured JSON with a 'coinglass' key containing ALL raw Coinglass values exactly as
returned by the tools, a 'technicals' key for indicator values, and a 'spot' key for price data.
"""


async def run_market_agent(coin: str, plan: dict) -> dict[str, Any]:
    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    ).bind_tools(MARKET_TOOLS)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=f"Fetch comprehensive market data for {coin}. "
                    f"Include spot, derivatives, liquidations, and technicals. "
                    f"You MUST call the available tools to fetch real data before returning. "
                    f"Provide a MARKET STRUCTURE assessment (bullish/bearish/neutral) with reasoning. "
                    f"Return structured JSON."
        )
    ]

    raw_tool_results: dict = {}
    while True:
        response = await llm.ainvoke(messages)
        messages.append(response)
        tool_calls = response.tool_calls if hasattr(response, "tool_calls") else []
        if not tool_calls:
            break

        from langchain_core.messages import ToolMessage
        for tc in tool_calls:
            print(f"INFO: market_agent calling tool={tc['name']} args={tc['args']}")
            tool_fn = {t.name: t for t in MARKET_TOOLS}.get(tc["name"])
            if tool_fn:
                try:
                    result = await tool_fn.ainvoke(tc["args"])
                except Exception as e:
                    result = {"error": str(e)}
                print(f"INFO: market_agent tool={tc['name']} result={str(result)[:300]}")
                raw_tool_results[tc["name"]] = result
                messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    parsed = _parse_agent_response(response.content, "market_agent")
    parsed["_raw_tool_results"] = raw_tool_results
    return parsed


# ---------------------------------------------------------------------------
# Technical indicator helpers
# ---------------------------------------------------------------------------

def _compute_rsi(closes: list[float], period: int = 14) -> float:
    """Wilder's RSI. Requires at least period+1 data points."""
    if len(closes) < period + 1:
        return 50.0

    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains  = [max(d, 0.0) for d in deltas]
    losses = [max(-d, 0.0) for d in deltas]

    # Seed with simple average over first period
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    # Wilder smoothing for remaining bars
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _compute_bollinger(closes: list[float], period: int = 20, std_dev: int = 2) -> dict:
    recent = closes[-period:]
    mid = sum(recent) / period
    variance = sum((x - mid) ** 2 for x in recent) / period
    std = variance ** 0.5
    return {"upper": mid + std_dev * std, "mid": mid, "lower": mid - std_dev * std}


def _coingecko_id(coin: str) -> str:
    ids = {
        "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana",
        "BNB": "binancecoin", "XRP": "ripple", "ADA": "cardano",
        "AVAX": "avalanche-2", "DOT": "polkadot", "LINK": "chainlink",
        "MATIC": "matic-network", "DOGE": "dogecoin",
    }
    return ids.get(coin.upper(), coin.lower())
