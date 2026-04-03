"""
Async HTTP data-gathering functions for the macro agent.

All HTTP calls live here — zero LangChain or LangGraph imports.
Each function is async (httpx) so macro_node can await them all in parallel
via asyncio.gather(). API keys are passed as parameters, never read from env here.

Free data sources used:
  CoinGecko public API   – global market stats + BTC price history  (no auth)
  Alternative.me         – Fear & Greed index                       (no auth)
  Stooq.com              – SPY daily CSV for BTC/equity correlation  (no auth)
  CryptoPanic API        – regulatory & protocol news               (free key)
  FRED API               – Federal Funds Rate (FEDFUNDS series)      (free key)
"""
from __future__ import annotations

import asyncio
import csv
import io
from typing import Optional

import httpx

# ── Constants ─────────────────────────────────────────────────────────────────

_CG_BASE  = "https://api.coingecko.com/api/v3"
_AM_BASE  = "https://api.alternative.me"
_CP_BASE  = "https://cryptopanic.com/api/v1"
_FRED_BASE = "https://api.stlouisfed.org/fred"
_STOOQ_SPY = "https://stooq.com/q/d/l/?s=spy.us&i=d"

_HEADERS = {"User-Agent": "CryptoReportAgent/1.0 (educational research)"}
_TIMEOUT = 15.0


# ── Private async HTTP helper ─────────────────────────────────────────────────

async def _aget(
    url: str,
    params: dict | None = None,
    extra_headers: dict | None = None,
) -> dict | list:
    """
    Async GET with up to 3 retries on HTTP 429 (rate limit).
    Mirrors intel_agent/tools.py _get() but uses httpx.AsyncClient.
    """
    hdrs = {**_HEADERS, **(extra_headers or {})}
    async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
        for attempt in range(3):
            resp = await client.get(url, params=params, headers=hdrs)
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 10))
                await asyncio.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
    raise RuntimeError(f"Rate-limited after retries: {url}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. CoinGecko — global market snapshot
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_global_market() -> dict:
    """
    GET /global — returns BTC dominance, total market cap, 24h change,
    active cryptocurrencies, and total 24h volume.

    Returns:
        {
            btc_dominance_pct:        float,
            total_market_cap_usd:     float,
            market_cap_change_24h_pct: float,
            active_cryptocurrencies:  int,
            total_volume_24h_usd:     float,
        }
    """
    data = await _aget(f"{_CG_BASE}/global")
    gd = data.get("data", {})
    return {
        "btc_dominance_pct":         round(gd.get("market_cap_percentage", {}).get("btc", 0), 2),
        "total_market_cap_usd":      gd.get("total_market_cap", {}).get("usd"),
        "market_cap_change_24h_pct": round(gd.get("market_cap_change_percentage_24h_usd", 0), 2),
        "active_cryptocurrencies":   gd.get("active_cryptocurrencies"),
        "total_volume_24h_usd":      gd.get("total_volume", {}).get("usd"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Alternative.me — Fear & Greed index
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_fear_greed() -> dict:
    """
    GET /fng/?limit=1 — returns today's Fear & Greed value and classification.

    Returns:
        {
            value:               int,   # 0 (extreme fear) – 100 (extreme greed)
            value_classification: str,  # "Extreme Fear" | "Fear" | "Neutral" | ...
            timestamp:           str,
        }
    """
    data = await _aget(f"{_AM_BASE}/fng/", params={"limit": 1})
    entry = (data.get("data") or [{}])[0]
    return {
        "value":                int(entry.get("value", 0)),
        "value_classification": entry.get("value_classification", "Unknown"),
        "timestamp":            entry.get("timestamp", ""),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. CryptoPanic — regulatory & protocol news
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_regulatory_news(api_key: str, limit: int = 10) -> list[dict]:
    """
    GET /posts/ filtered to "important" news.
    Returns [] gracefully if api_key is empty or request fails.

    Args:
        api_key: CRYPTOPANIC_API_KEY (pass empty string to skip)
        limit:   max articles to return

    Returns list of:
        {title, published_at, url, source, votes_positive, votes_negative}
    """
    if not api_key:
        return []

    try:
        data = await _aget(
            f"{_CP_BASE}/posts/",
            params={
                "auth_token": api_key,
                "filter":     "important",
                "kind":       "news",
                "public":     "true",
            },
        )
        results = data.get("results", [])[:limit]
        return [
            {
                "title":           r.get("title", ""),
                "published_at":    r.get("published_at", ""),
                "url":             r.get("url", ""),
                "source":          (r.get("source") or {}).get("title", ""),
                "votes_positive":  (r.get("votes") or {}).get("positive", 0),
                "votes_negative":  (r.get("votes") or {}).get("negative", 0),
            }
            for r in results
        ]
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# 4. FRED — Federal Funds Rate
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_fed_funds_rate(api_key: str) -> dict:
    """
    GET /fred/series/observations for FEDFUNDS series (monthly, descending).
    Returns None fields gracefully if api_key is empty or FRED is unreachable.

    Args:
        api_key: FRED_API_KEY (pass empty string to skip)

    Returns:
        {
            current_rate_pct:  float | None,
            previous_rate_pct: float | None,
            date:              str | None,
            direction:         "rising" | "falling" | "unchanged" | None,
        }
    """
    if not api_key:
        return {
            "current_rate_pct":  None,
            "previous_rate_pct": None,
            "date":              None,
            "direction":         None,
        }

    try:
        data = await _aget(
            f"{_FRED_BASE}/series/observations",
            params={
                "series_id":  "FEDFUNDS",
                "api_key":    api_key,
                "file_type":  "json",
                "sort_order": "desc",
                "limit":      2,
            },
        )
        obs = data.get("observations", [])

        def _parse_rate(o: dict) -> Optional[float]:
            v = o.get("value", ".")
            return float(v) if v != "." else None

        current  = _parse_rate(obs[0]) if len(obs) > 0 else None
        previous = _parse_rate(obs[1]) if len(obs) > 1 else None
        date     = obs[0].get("date") if obs else None

        if current is not None and previous is not None:
            if current > previous:
                direction = "rising"
            elif current < previous:
                direction = "falling"
            else:
                direction = "unchanged"
        else:
            direction = None

        return {
            "current_rate_pct":  current,
            "previous_rate_pct": previous,
            "date":              date,
            "direction":         direction,
        }
    except Exception:
        return {
            "current_rate_pct":  None,
            "previous_rate_pct": None,
            "date":              None,
            "direction":         None,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 5. BTC vs SPY 30-day correlation
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_btc_vs_spy_correlation() -> dict:
    """
    Computes 30-day return for BTC (via CoinGecko) and SPY (via stooq.com CSV).
    Both requests fire concurrently.

    Returns:
        {
            btc_30d_return_pct: float | None,
            spy_30d_return_pct: float | None,
            btc_outperforms:    bool | None,
        }
    """
    btc_data, spy_text = await asyncio.gather(
        _aget(
            f"{_CG_BASE}/coins/bitcoin/market_chart",
            params={"vs_currency": "usd", "days": 30, "interval": "daily"},
        ),
        _fetch_stooq_csv(),
        return_exceptions=True,
    )

    # BTC 30d return
    btc_return: Optional[float] = None
    if not isinstance(btc_data, Exception):
        prices = [p[1] for p in btc_data.get("prices", []) if len(p) == 2 and p[1]]
        if len(prices) >= 2:
            btc_return = round((prices[-1] - prices[0]) / prices[0] * 100, 2)

    # SPY 30d return from CSV
    spy_return: Optional[float] = None
    if not isinstance(spy_text, Exception) and spy_text:
        try:
            reader = csv.DictReader(io.StringIO(spy_text))
            rows = [r for r in reader if r.get("Close")]
            if len(rows) >= 2:
                closes = [float(r["Close"]) for r in rows]
                first, last = closes[0], closes[-1]
                # stooq returns newest-first; take last 30 rows
                last_30 = closes[-30:] if len(closes) >= 30 else closes
                spy_return = round((last_30[-1] - last_30[0]) / last_30[0] * 100, 2)
        except Exception:
            spy_return = None

    btc_outperforms: Optional[bool] = None
    if btc_return is not None and spy_return is not None:
        btc_outperforms = btc_return > spy_return

    return {
        "btc_30d_return_pct": btc_return,
        "spy_30d_return_pct": spy_return,
        "btc_outperforms":    btc_outperforms,
    }


async def _fetch_stooq_csv() -> str:
    """Fetch SPY daily price CSV from stooq.com. Returns raw CSV text."""
    async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
        resp = await client.get(_STOOQ_SPY, headers=_HEADERS)
        resp.raise_for_status()
        return resp.text
