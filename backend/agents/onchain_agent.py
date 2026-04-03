"""
Onchain Agent
=============
Fetches live on-chain metrics using Glassnode, Etherscan, and Dune Analytics.
Uses Claude's native tool_use to decide which metrics to pull based on the coin.
"""

import asyncio
import os
import httpx
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool


# ---------------------------------------------------------------------------
# Tools — real API calls
# ---------------------------------------------------------------------------

@tool
async def get_glassnode_metric(asset: str, metric: str) -> dict:
    """
    Fetch a metric from Glassnode API.
    metric examples: addresses/active_count, market/mvrv_z_score, 
    indicators/nvt, supply/current, transactions/transfers_volume_sum
    """
    api_key = os.getenv("GLASSNODE_API_KEY", "")
    if not api_key:
        return {"error": "Glassnode API key not configured — metric unavailable", "metric": metric}
    url = f"https://api.glassnode.com/v1/metrics/{metric}"
    async with httpx.AsyncClient() as client:
        r = await client.get(url, params={"a": asset, "api_key": api_key, "i": "24h"})
        r.raise_for_status()
        data = r.json()
        # Return the most recent value + 30-day trend
        recent = data[-1] if data else {}
        month_ago = data[-30] if len(data) >= 30 else data[0] if data else {}
        return {
            "metric": metric,
            "current": recent.get("v"),
            "timestamp": recent.get("t"),
            "30d_ago": month_ago.get("v"),
            "trend": "up" if recent.get("v", 0) > month_ago.get("v", 0) else "down",
        }


@tool
async def get_exchange_flows(asset: str) -> dict:
    """Get net exchange inflows/outflows — key signal for selling/buying pressure."""
    api_key = os.getenv("GLASSNODE_API_KEY", "")
    if not api_key:
        return {"error": "Glassnode API key not configured — exchange flows unavailable"}
    async with httpx.AsyncClient() as client:
        inflow = await client.get(
            "https://api.glassnode.com/v1/metrics/transactions/transfers_to_exchanges_sum",
            params={"a": asset, "api_key": api_key, "i": "24h"}
        )
        outflow = await client.get(
            "https://api.glassnode.com/v1/metrics/transactions/transfers_from_exchanges_sum",
            params={"a": asset, "api_key": api_key, "i": "24h"}
        )
        in_data = inflow.json()
        out_data = outflow.json()
        latest_in = in_data[-1]["v"] if in_data else 0
        latest_out = out_data[-1]["v"] if out_data else 0
        return {
            "exchange_inflow_24h": latest_in,
            "exchange_outflow_24h": latest_out,
            "net_flow": latest_out - latest_in,  # Positive = net outflow = bullish
            "signal": "bullish" if latest_out > latest_in else "bearish",
        }


@tool
async def get_whale_activity(asset: str) -> dict:
    """
    Track large wallet movements via Etherscan (ETH/ERC20) or 
    blockchain-specific APIs. Returns top 10 whale tx in last 24h.
    """
    if asset.upper() not in ("ETH", "USDT", "USDC", "LINK", "UNI"):
        return {"error": f"Whale tracking via Etherscan only supports EVM assets, not {asset}"}

    api_key = os.getenv("ETHERSCAN_API_KEY", "")
    if not api_key:
        return {"error": "Etherscan API key not configured — whale tracking unavailable"}
    # This uses Etherscan's token transfer endpoint filtered by large amounts
    async with httpx.AsyncClient() as client:
        r = await client.get(
            "https://api.etherscan.io/api",
            params={
                "module": "account",
                "action": "tokentx",
                "contractaddress": _token_contract(asset),
                "page": 1,
                "offset": 100,
                "sort": "desc",
                "apikey": api_key,
            }
        )
        data = r.json()
        txs = data.get("result", [])
        if not isinstance(txs, list):
            return {"error": f"Etherscan API error: {txs}"}
        # Filter to whale-tier (>$1M equivalent — rough filter by value)
        large_txs = [tx for tx in txs if int(tx.get("value", 0)) > 1e21]
        return {
            "large_transactions_24h": len(large_txs),
            "total_large_volume": sum(int(tx["value"]) for tx in large_txs[:10]) / 1e18,
            "sample_txs": large_txs[:3],
        }


@tool
async def get_dune_query(query_id: int) -> dict:
    """Run a pre-built Dune Analytics query for advanced onchain metrics."""
    api_key = os.getenv("DUNE_API_KEY", "")
    if not api_key:
        return {"error": "Dune API key not configured — advanced onchain metrics unavailable"}
    async with httpx.AsyncClient(timeout=30) as client:
        # Execute query
        exec_r = await client.post(
            f"https://api.dune.com/api/v1/query/{query_id}/execute",
            headers={"x-dune-api-key": api_key},
        )
        exec_id = exec_r.json().get("execution_id")
        
        # Poll for results (simplified — in prod use webhook)
        import asyncio
        for _ in range(10):
            await asyncio.sleep(2)
            result_r = await client.get(
                f"https://api.dune.com/api/v1/execution/{exec_id}/results",
                headers={"x-dune-api-key": api_key},
            )
            result = result_r.json()
            if result.get("state") == "QUERY_STATE_COMPLETED":
                return result.get("result", {})
        
        return {"error": "Dune query timed out"}


# ---------------------------------------------------------------------------
# CoinGlass v4 onchain tools
# ---------------------------------------------------------------------------

_CG_BASE = "https://open-api-v4.coinglass.com"


def _cg_headers() -> dict:
    return {"CG-API-KEY": os.getenv("COINGLASS_API_KEY", "")}


@tool
async def get_coinglass_exchange_flows(coin: str) -> dict:
    """
    Fetch exchange balance inflows/outflows from Coinglass v4.
    Net outflow = coins leaving exchanges = accumulation (bullish).
    Net inflow  = coins entering exchanges = selling pressure (bearish).
    Works for BTC, ETH, and most major assets.
    """
    api_key = os.getenv("COINGLASS_API_KEY", "")
    if not api_key:
        return {"error": "COINGLASS_API_KEY not set"}

    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(
            f"{_CG_BASE}/api/exchange/balance/list",
            params={"symbol": coin.upper()},
            headers=_cg_headers(),
        )
        print(f"DEBUG: exchange/balance/list status={r.status_code} data={r.text[:400]}")
        if r.status_code != 200:
            return {"error": f"Coinglass returned {r.status_code}: {r.text[:200]}"}

        raw = r.json()
        if raw.get("code") in ("401", "403"):
            return {"error": f"Plan restriction: {raw.get('msg')}"}

        # Response is per-exchange rows — aggregate across all exchanges
        rows = raw.get("data", [])
        if not rows:
            return {"error": f"No exchange balance data returned for {coin}"}

        total_balance   = sum(float(row.get("total_balance",         0) or 0) for row in rows)
        change_1d       = sum(float(row.get("balance_change_1d",     0) or 0) for row in rows)
        change_7d       = sum(float(row.get("balance_change_7d",     0) or 0) for row in rows)
        change_30d      = sum(float(row.get("balance_change_30d",    0) or 0) for row in rows)
        top_exchanges   = [{"exchange": r.get("exchange_name"), "balance": r.get("total_balance")} for r in rows[:5]]

        return {
            "total_exchange_balance":  round(total_balance, 2),
            "balance_change_1d":       round(change_1d,  2),
            "balance_change_7d":       round(change_7d,  2),
            "balance_change_30d":      round(change_30d, 2),
            "top_exchanges":           top_exchanges,
            "flow_signal": (
                "accumulation" if change_1d < 0 else
                "distribution" if change_1d > 0 else "neutral"
            ),
        }


@tool
async def get_coinglass_btc_metrics(coin: str) -> dict:
    """
    Fetch Bitcoin-specific on-chain metrics from Coinglass v4:
    - NUPL (Net Unrealized Profit/Loss) — proxy for MVRV sentiment
    - Active addresses — network participation trend
    For non-BTC assets, returns available exchange flow data instead.
    """
    api_key = os.getenv("COINGLASS_API_KEY", "")
    if not api_key:
        return {"error": "COINGLASS_API_KEY not set"}

    if coin.upper() != "BTC":
        return {"error": f"MVRV/active address metrics only available for BTC, not {coin}"}

    headers = _cg_headers()
    async with httpx.AsyncClient(timeout=15) as client:
        nupl_r, addr_r = await asyncio.gather(
            client.get(f"{_CG_BASE}/api/index/bitcoin-net-unrealized-profit-loss",
                       params={"interval": "1d", "limit": 7}, headers=headers),
            client.get(f"{_CG_BASE}/api/index/bitcoin-active-addresses",
                       params={"interval": "1d", "limit": 7}, headers=headers),
            return_exceptions=True,
        )

    result: dict = {}

    if not isinstance(nupl_r, Exception) and nupl_r.status_code == 200:
        nupl_raw = nupl_r.json()
        nupl_data = nupl_raw.get("data", [])
        print(f"DEBUG: bitcoin-nupl status={nupl_r.status_code} code={nupl_raw.get('code')} data={str(nupl_data)[:200]}")
        if nupl_raw.get("code") in ("401", "403"):
            result["nupl_error"] = f"Plan restriction: {nupl_raw.get('msg')}"
        elif nupl_data:
            latest = nupl_data[-1]
            week_ago = nupl_data[0]
            nupl_val = float(latest.get("c") or latest.get("value") or 0)
            result.update({
                "nupl_value":  nupl_val,
                "nupl_zone": (
                    "euphoria"     if nupl_val > 0.75 else
                    "belief"       if nupl_val > 0.50 else
                    "optimism"     if nupl_val > 0.25 else
                    "hope"         if nupl_val > 0    else
                    "fear"         if nupl_val > -0.25 else
                    "capitulation"
                ),
                "nupl_trend": "improving" if nupl_val > float(week_ago.get("c") or week_ago.get("value") or 0) else "deteriorating",
            })
    else:
        result["nupl_error"] = str(nupl_r) if isinstance(nupl_r, Exception) else f"HTTP {nupl_r.status_code}: {nupl_r.text[:200]}"

    if not isinstance(addr_r, Exception) and addr_r.status_code == 200:
        addr_raw  = addr_r.json()
        addr_data = addr_raw.get("data", [])
        print(f"DEBUG: bitcoin-active-addresses status={addr_r.status_code} code={addr_raw.get('code')} data={str(addr_data)[:200]}")
        if addr_raw.get("code") in ("401", "403"):
            result["addr_error"] = f"Plan restriction: {addr_raw.get('msg')}"
        if addr_data:
            latest   = addr_data[-1]
            week_ago = addr_data[0]
            addr_now  = float(latest.get("c")   or latest.get("value") or 0)
            addr_prev = float(week_ago.get("c") or week_ago.get("value") or 1)
            result.update({
                "active_addresses":         int(addr_now),
                "active_addr_7d_change_pct": round((addr_now - addr_prev) / addr_prev * 100, 2) if addr_prev else None,
                "addr_trend": "growing" if addr_now > addr_prev else "shrinking",
            })
    else:
        result["addr_error"] = str(addr_r) if isinstance(addr_r, Exception) else f"HTTP {addr_r.status_code}: {addr_r.text[:200]}"

    return result if result else {"error": "No BTC on-chain metrics returned"}


@tool
async def get_etf_flows(coin: str) -> dict:
    """
    Fetch institutional ETF inflow/outflow data from Coinglass v4.
    Available for BTC and ETH. ETF flows reflect institutional demand.
    Net inflow = institutional buying (bullish). Net outflow = selling (bearish).
    """
    api_key = os.getenv("COINGLASS_API_KEY", "")
    if not api_key:
        return {"error": "COINGLASS_API_KEY not set"}

    etf_endpoints = {
        "BTC": "/api/etf/bitcoin/flow-history",
        "ETH": "/api/etf/ethereum/flow-history",
    }
    endpoint = etf_endpoints.get(coin.upper())
    if not endpoint:
        return {"error": f"ETF flow data not available for {coin} — only BTC and ETH supported"}

    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(f"{_CG_BASE}{endpoint}", headers=_cg_headers())
        print(f"DEBUG: etf endpoint={endpoint} status={r.status_code} data={r.text[:400]}")
        if r.status_code != 200:
            return {"error": f"Coinglass ETF endpoint returned {r.status_code}: {r.text[:200]}"}

        raw = r.json()
        if raw.get("code") in ("401", "403"):
            return {"error": f"Plan restriction: {raw.get('msg')}"}

        records = raw.get("data", [])
        if not records:
            return {"error": "No ETF flow data returned"}

        # Most recent record is last in the list
        latest = records[-1]
        daily_flow = float(latest.get("flow_usd") or 0)

        # 7-day net flow from last 7 records
        weekly_flow = sum(float(r.get("flow_usd") or 0) for r in records[-7:])

        # Per-ETF breakdown from latest day
        etf_breakdown = [
            {"ticker": e.get("etf_ticker"), "flow_usd": e.get("flow_usd")}
            for e in (latest.get("etf_flows") or [])[:6]
        ]

        return {
            "daily_flow_usd":   daily_flow,
            "weekly_flow_usd":  weekly_flow,
            "etf_breakdown":    etf_breakdown,
            "flow_signal": (
                "institutional_buying"  if daily_flow > 0 else
                "institutional_selling" if daily_flow < 0 else "neutral"
            ),
        }


# ---------------------------------------------------------------------------
# Response parser (Gemini returns content as list[dict] content blocks)
# ---------------------------------------------------------------------------

def _parse_agent_response(content: Any, agent_name: str) -> dict:
    import json, re
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

ONCHAIN_TOOLS = [
    get_glassnode_metric,
    get_exchange_flows,
    get_whale_activity,
    get_dune_query,
    get_coinglass_exchange_flows,
    get_coinglass_btc_metrics,
    get_etf_flows,
]

SYSTEM_PROMPT = """You are an expert on-chain analyst. Your job is to fetch and interpret
on-chain metrics for crypto assets to determine network health and holder behavior.

For any asset, always fetch:
1. Exchange flows via get_coinglass_exchange_flows — net inflow/outflow is the most important signal
2. ETF flows via get_etf_flows — institutional demand (BTC and ETH only)
3. For BTC: get_coinglass_btc_metrics — NUPL zone and active address trend
4. Exchange flows via get_exchange_flows (Glassnode) if API key is available
5. Whale activity via get_whale_activity for EVM assets (ETH, LINK, etc.)

CRITICAL: Copy every raw number from Coinglass tool results directly into your JSON output.
Do NOT paraphrase, round, or omit values — include the exact figures returned by the tools:
- total_exchange_balance (exact coin amount), balance_change_1d, balance_change_7d, balance_change_30d
- flow_signal label, top_exchanges list with per-exchange balances
- ETF: total_net_assets_usd, daily_inflow_usd, weekly_inflow_usd, flow_signal
- BTC only: nupl_value (exact decimal), nupl_zone, nupl_trend, active_addresses, active_addr_7d_change_pct

Interpret each metric and provide a composite ONCHAIN HEALTH SCORE from 0-100.
Return structured JSON with all metrics and your interpretation.
"""


async def run_onchain_agent(coin: str, plan: dict) -> dict[str, Any]:
    """
    Runs the onchain agent with tool_use. Claude decides which tools to call
    based on the coin and the orchestrator's research plan.
    """
    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    ).bind_tools(ONCHAIN_TOOLS)

    focus = plan.get("focus_areas", [])
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=f"Fetch comprehensive on-chain data for {coin}. "
                    f"Pay special attention to: {', '.join(focus)}. "
                    f"You MUST call the available tools to fetch real data before returning. "
                    f"Return all metrics as structured JSON with an overall health score."
        )
    ]

    # Agentic loop — keep calling tools until Claude stops
    raw_tool_results: dict = {}
    while True:
        response = await llm.ainvoke(messages)
        messages.append(response)

        tool_calls = response.tool_calls if hasattr(response, "tool_calls") else []
        if not tool_calls:
            break  # Claude is done calling tools

        # Execute all tool calls
        from langchain_core.messages import ToolMessage
        for tc in tool_calls:
            print(f"INFO: onchain_agent calling tool={tc['name']} args={tc['args']}")
            tool_fn = {t.name: t for t in ONCHAIN_TOOLS}.get(tc["name"])
            if tool_fn:
                try:
                    result = await tool_fn.ainvoke(tc["args"])
                except Exception as e:
                    result = {"error": str(e)}
                print(f"INFO: onchain_agent tool={tc['name']} result={str(result)[:300]}")
                raw_tool_results[tc["name"]] = result
                messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    # Parse final structured output; attach raw results for synthesis formatter
    parsed = _parse_agent_response(response.content, "onchain_agent")
    parsed["_raw_tool_results"] = raw_tool_results
    return parsed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _token_contract(asset: str) -> str:
    contracts = {
        "USDT": "0xdac17f958d2ee523a2206206994597c13d831ec7",
        "USDC": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
        "LINK": "0x514910771af9ca656af840dff83e8264ecf986ca",
        "UNI": "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984",
    }
    return contracts.get(asset.upper(), "")
