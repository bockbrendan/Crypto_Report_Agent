"""
Shared state schema for the Crypto Intel Agent graph.

Every node reads from and writes to this TypedDict.
Nodes must return ONLY the keys they modify — never the full state.
"""
from __future__ import annotations

from typing import Annotated, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class CryptoIntelState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────
    query: str                      # raw user query, e.g. "analyze solana"

    # ── Resolved identity (set by resolve_asset) ───────────────────────────
    coin_id: Optional[str]          # CoinGecko slug,   e.g. "solana"
    symbol:  Optional[str]          # Ticker uppercase, e.g. "SOL"
    name:    Optional[str]          # Display name,     e.g. "Solana"

    # ── Raw data payloads (set by research nodes) ──────────────────────────
    onchain_raw:   Optional[dict]   # CoinGecko market + community + dev data
    news_raw:      Optional[list]   # CryptoCompare news articles
    twitter_raw:   Optional[list]   # Twitter v2 recent tweets (may be empty)
    docs_raw:      Optional[dict]   # fetched homepage / whitepaper text

    # ── Per-section LLM analysis (set by research nodes) ──────────────────
    metrics_analysis:   Optional[str]   # onchain_node output
    sentiment_analysis: Optional[str]   # sentiment_node output
    docs_analysis:      Optional[str]   # docs_node output

    # ── Final report (set by report_node) ──────────────────────────────────
    risk_score: Optional[int]       # 1 = lowest risk, 10 = highest risk
    report:     Optional[str]       # full markdown narrative

    # ── Audit trail ────────────────────────────────────────────────────────
    messages: Annotated[list[BaseMessage], add_messages]
