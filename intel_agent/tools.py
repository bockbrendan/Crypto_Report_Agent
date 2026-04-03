"""
Data-gathering functions for the Crypto Intel Agent.

All HTTP calls live here — zero LangChain or LangGraph imports.
Each function raises on hard failures; callers handle gracefully.

Free APIs used:
  CoinGecko public API    – market, community, developer stats (no auth)
  CoinDesk Data API       – recent news headlines              (no auth)
  DeFiLlama               – TVL for DeFi protocols             (no auth)
  Twitter API v2          – recent tweets                      (bearer token, optional)
  Generic HTTP fetch      – whitepaper / homepage content
"""
from __future__ import annotations

import os
import re
import time
from typing import Optional

import requests
from bs4 import BeautifulSoup

# ── Constants ─────────────────────────────────────────────────────────────────

_CG_BASE   = "https://api.coingecko.com/api/v3"
_CD_BASE   = "https://data-api.coindesk.com/news/v1"   # CoinDesk (formerly CryptoCompare)
_DL_BASE   = "https://api.llama.fi"
_TW_BASE = "https://api.x.com/2"

_HEADERS   = {"User-Agent": "CryptoIntelAgent/1.0 (educational research)"}
_TIMEOUT   = 15
_MAX_TEXT  = 5_000  # chars to keep from fetched pages


def _coindesk_headers() -> dict:
    """Return auth header for CoinDesk API if key is present in env."""
    key = os.getenv("COINDESK_API_KEY", "")
    return {"authorization": f"Apikey {key}"} if key else {}


def _get(url: str, params: dict | None = None, extra_headers: dict | None = None) -> dict | list:
    """GET with retry on 429 (rate limit)."""
    hdrs = {**_HEADERS, **(extra_headers or {})}
    for attempt in range(3):
        resp = requests.get(url, params=params, headers=hdrs, timeout=_TIMEOUT)
        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", 10))
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError(f"Rate-limited after retries: {url}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Asset discovery
# ─────────────────────────────────────────────────────────────────────────────

def search_coin(query: str) -> list[dict]:
    """
    Search CoinGecko for a coin by name/ticker.
    Returns top-5 candidates: [{id, name, symbol}, ...]
    """
    data  = _get(f"{_CG_BASE}/search", params={"query": query})
    coins = data.get("coins", [])[:5]
    return [
        {"id": c["id"], "name": c["name"], "symbol": c["symbol"].upper()}
        for c in coins
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 2. On-chain / market metrics
# ─────────────────────────────────────────────────────────────────────────────

def get_coin_detail(coin_id: str) -> dict:
    """
    Full CoinGecko coin detail: market data, community stats, developer activity,
    project links and description.
    """
    data = _get(
        f"{_CG_BASE}/coins/{coin_id}",
        params={
            "localization":   "false",
            "tickers":        "false",
            "community_data": "true",
            "developer_data": "true",
        },
    )
    mkt = data.get("market_data",    {})
    com = data.get("community_data", {})
    dev = data.get("developer_data", {})
    lnk = data.get("links",         {})

    return {
        # identity
        "name":               data.get("name"),
        "symbol":             (data.get("symbol") or "").upper(),
        "description":        (data.get("description", {}).get("en") or "")[:600],
        "categories":         data.get("categories", []),
        "genesis_date":       data.get("genesis_date"),
        "block_time_minutes": data.get("block_time_in_minutes"),
        "hashing_algorithm":  data.get("hashing_algorithm"),

        # links
        "links": {
            "homepage":        (lnk.get("homepage") or [])[:2],
            "whitepaper":       lnk.get("whitepaper", ""),
            "github":          (lnk.get("repos_url", {}).get("github") or [])[:2],
            "subreddit":        lnk.get("subreddit_url", ""),
            "twitter_handle":   lnk.get("twitter_screen_name", ""),
        },

        # market
        "market": {
            "price_usd":               mkt.get("current_price",          {}).get("usd"),
            "market_cap_usd":          mkt.get("market_cap",             {}).get("usd"),
            "fully_diluted_valuation": mkt.get("fully_diluted_valuation",{}).get("usd"),
            "total_volume_24h":        mkt.get("total_volume",           {}).get("usd"),
            "price_change_24h_pct":    mkt.get("price_change_percentage_24h"),
            "price_change_7d_pct":     mkt.get("price_change_percentage_7d"),
            "price_change_30d_pct":    mkt.get("price_change_percentage_30d"),
            "ath_usd":                 mkt.get("ath",                    {}).get("usd"),
            "ath_change_pct":          mkt.get("ath_change_percentage",  {}).get("usd"),
            "circulating_supply":      mkt.get("circulating_supply"),
            "total_supply":            mkt.get("total_supply"),
            "max_supply":              mkt.get("max_supply"),
            "market_cap_rank":         data.get("market_cap_rank"),
        },

        # community
        "community": {
            "twitter_followers":        com.get("twitter_followers"),
            "reddit_subscribers":       com.get("reddit_subscribers"),
            "reddit_avg_posts_48h":     com.get("reddit_average_posts_48h"),
            "reddit_avg_comments_48h":  com.get("reddit_average_comments_48h"),
            "sentiment_up_pct":         data.get("sentiment_votes_up_percentage"),
            "sentiment_down_pct":       data.get("sentiment_votes_down_percentage"),
        },

        # developer
        "developer": {
            "forks":                dev.get("forks"),
            "stars":                dev.get("stars"),
            "subscribers":          dev.get("subscribers"),
            "total_issues":         dev.get("total_issues"),
            "closed_issues":        dev.get("closed_issues"),
            "pull_requests_merged": dev.get("pull_requests_merged"),
            "commit_count_4_weeks": dev.get("commit_count_4_weeks"),
        },
    }


def get_price_history(coin_id: str, days: int = 30) -> list[tuple[int, float]]:
    """
    Daily closing prices for the last `days` days.
    Returns list of [timestamp_ms, price_usd].
    """
    data   = _get(
        f"{_CG_BASE}/coins/{coin_id}/market_chart",
        params={"vs_currency": "usd", "days": days, "interval": "daily"},
    )
    prices = data.get("prices", [])
    return prices[-30:]     # cap at 30 data points


def get_defi_tvl(symbol: str) -> Optional[dict]:
    """
    Look up TVL from DeFiLlama by matching the token symbol.
    Returns the highest-TVL protocol match, or None if not found.
    """
    try:
        protocols = _get(f"{_DL_BASE}/protocols")
        symbol_up = symbol.upper()
        matches = [
            p for p in protocols
            if (p.get("symbol") or "").upper() == symbol_up
        ]
        if not matches:
            return None
        best = max(matches, key=lambda p: p.get("tvl") or 0)
        return {
            "name":          best.get("name"),
            "tvl_usd":       best.get("tvl"),
            "category":      best.get("category"),
            "chains":        best.get("chains", [])[:5],
            "change_1d_pct": best.get("change_1d"),
            "change_7d_pct": best.get("change_7d"),
        }
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 3. News & sentiment
# ─────────────────────────────────────────────────────────────────────────────

def get_crypto_news(symbol: str, limit: int = 15) -> list[dict]:
    """
    Fetch recent crypto news from CoinDesk Data API (formerly CryptoCompare).
    Tries symbol-filtered news first, falls back to general crypto news.
    """
    def _parse(data: dict) -> list[dict]:
        articles = data.get("Data", [])
        if not isinstance(articles, list):
            raise ValueError(f"Unexpected response: {data.get('Err')}")
        return [
            {
                "title":     a.get("TITLE", ""),
                "source":    a.get("SOURCE_DATA", {}).get("NAME", ""),
                "body":      (a.get("BODY") or "")[:300],
                "tags":      a.get("KEYWORDS", ""),
                "sentiment": a.get("SENTIMENT", ""),
                "url":       a.get("URL", ""),
            }
            for a in articles[:limit]
        ]

    # Attempt 1: filtered by symbol category
    try:
        data = _get(f"{_CD_BASE}/article/list", params={"lang": "EN", "limit": limit, "categories": symbol}, extra_headers=_coindesk_headers())
        articles = _parse(data)
        if articles:
            return articles
    except Exception as e:
        print(f"[get_crypto_news] Filtered request failed ({symbol}): {e}")

    # Attempt 2: general crypto news (unfiltered)
    try:
        data = _get(f"{_CD_BASE}/article/list", params={"lang": "EN", "limit": limit}, extra_headers=_coindesk_headers())
        return _parse(data)
    except Exception as e:
        print(f"[get_crypto_news] Unfiltered request failed: {e}")
        return []


def get_twitter_sentiment(symbol: str, name: str, max_results: int = 50) -> list[dict]:
    """
    Fetch recent tweets mentioning $SYMBOL from Twitter API v2.
    Requires TWITTER_BEARER_TOKEN in environment.
    Returns [] gracefully if token is absent or quota is exceeded.
    """
    token = os.getenv("TWITTER_BEARER_TOKEN", "")
    if not token:
        return []

    query = f"(${symbol} OR #{symbol} OR \"{name}\") lang:en -is:retweet"
    try:
        data = _get(
            f"{_TW_BASE}/tweets/search/recent",
            params={
                "query":       query,
                "max_results": min(max_results, 100),
                "tweet.fields": "created_at,public_metrics,author_id",
            },
            extra_headers={"Authorization": f"Bearer {token}"},
        )
        tweets = data.get("data", [])
        return [
            {
                "text":         t.get("text", ""),
                "created_at":   t.get("created_at", ""),
                "likes":        t.get("public_metrics", {}).get("like_count", 0),
                "retweets":     t.get("public_metrics", {}).get("retweet_count", 0),
            }
            for t in tweets
        ]
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# 4. Documentation / whitepaper reader
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_page_text(url: str) -> str:
    """Fetch a URL, strip HTML, return plain text (capped at _MAX_TEXT chars)."""
    if not url or not url.startswith("http"):
        return ""
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT, allow_redirects=True)
        resp.raise_for_status()
        ct = resp.headers.get("content-type", "")
        if "pdf" in ct.lower():
            return "[PDF — text extraction not supported; review manually]"
        if "html" in ct.lower():
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()
            text = soup.get_text(separator=" ", strip=True)
            text = re.sub(r"\s{2,}", " ", text)
        else:
            text = resp.text
        return text[:_MAX_TEXT]
    except Exception as exc:
        return f"[fetch error: {exc}]"


def fetch_project_docs(coin_data: dict) -> dict:
    """
    Fetch homepage and whitepaper text using URLs from coin_data["links"].
    Returns dict with homepage_text and whitepaper_text.
    """
    links        = coin_data.get("links", {})
    homepage_url = next(iter(links.get("homepage") or []), "")
    wp_url       = links.get("whitepaper", "")

    return {
        "homepage_url":    homepage_url,
        "whitepaper_url":  wp_url,
        "homepage_text":   _fetch_page_text(homepage_url),
        "whitepaper_text": _fetch_page_text(wp_url),
    }
