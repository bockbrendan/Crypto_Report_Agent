"""
LangGraph StateGraph assembly for the Crypto Intel Agent.

Flow:
  START → resolve_asset → [onchain | END on failure]
        → onchain_node → sentiment_node → docs_node → report_node → END
"""
from __future__ import annotations

from langgraph.graph import StateGraph, END

from .state import CryptoIntelState
from .nodes import (
    docs_node,
    onchain_node,
    report_node,
    resolve_asset,
    sentiment_node,
)


def _after_resolve(state: CryptoIntelState) -> str:
    """Route to onchain_node if resolution succeeded, else END."""
    return "onchain" if state.get("coin_id") else END


def build_graph():
    """Build and compile the Crypto Intel Agent graph."""
    workflow = StateGraph(CryptoIntelState)

    # ── Register nodes ────────────────────────────────────────────────────
    workflow.add_node("resolve",   resolve_asset)
    workflow.add_node("onchain",   onchain_node)
    workflow.add_node("sentiment", sentiment_node)
    workflow.add_node("docs",      docs_node)
    workflow.add_node("report",    report_node)

    # ── Edges ─────────────────────────────────────────────────────────────
    workflow.set_entry_point("resolve")

    # Conditional: if asset not found → bail early
    workflow.add_conditional_edges(
        "resolve",
        _after_resolve,
        {"onchain": "onchain", END: END},
    )

    workflow.add_edge("onchain",   "sentiment")
    workflow.add_edge("sentiment", "docs")
    workflow.add_edge("docs",      "report")
    workflow.add_edge("report",    END)

    return workflow.compile()


# Compiled graph — import this in run.py and any tests
graph = build_graph()
