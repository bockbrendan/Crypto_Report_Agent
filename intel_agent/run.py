#!/usr/bin/env python3
"""
CLI entry point for the Crypto Intel Agent.

Usage:
    python -m intel_agent.run "analyze solana"
    python -m intel_agent.run "ethereum risk report"
    python -m intel_agent.run "what is the risk profile of chainlink"
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the parent directory (crypto_agent/)
load_dotenv(Path(__file__).parent.parent / ".env")

from .graph import graph
from .state import CryptoIntelState
from langchain_core.messages import HumanMessage


_DIVIDER = "─" * 70


def run(query: str, stream: bool = True) -> CryptoIntelState:
    """
    Run the Crypto Intel Agent for a given query.

    Args:
        query:  Natural-language query, e.g. "analyze bitcoin"
        stream: If True, print each node's progress to stdout as it runs.

    Returns:
        Final CryptoIntelState with report and risk_score populated.
    """
    initial_state: CryptoIntelState = {
        "query":    query,
        "messages": [HumanMessage(content=query)],
        # all other fields default to None via LangGraph
    }

    print(f"\n{_DIVIDER}")
    print(f"  CRYPTO INTEL AGENT")
    print(f"  Query: {query}")
    print(f"{_DIVIDER}\n")

    final_state = None

    if stream:
        for step in graph.stream(initial_state, stream_mode="values"):
            final_state = step
            # Print the latest audit message
            msgs = step.get("messages", [])
            if msgs:
                last = msgs[-1]
                if hasattr(last, "content") and last.content.startswith("["):
                    print(f"  ⟶  {last.content}")
    else:
        final_state = graph.invoke(initial_state)

    return final_state


def print_report(state: CryptoIntelState) -> None:
    """Pretty-print the final report and risk score."""
    print(f"\n{_DIVIDER}")
    print(f"  FINAL REPORT")
    print(f"{_DIVIDER}\n")

    report = state.get("report", "")
    if not report:
        print("  [No report generated — check logs above for errors]")
        return

    print(report)

    score = state.get("risk_score")
    if score is not None:
        bar   = "█" * score + "░" * (10 - score)
        label = {
            1: "Minimal",  2: "Very Low", 3: "Low",    4: "Low-Mod",
            5: "Moderate", 6: "Mod-High", 7: "High",   8: "Very High",
            9: "Extreme",  10: "Critical"
        }.get(score, "Unknown")
        print(f"\n{_DIVIDER}")
        print(f"  RISK SCORE: {score}/10  [{bar}]  {label}")
        print(f"{_DIVIDER}\n")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m intel_agent.run \"<query>\"")
        print('Example: python -m intel_agent.run "analyze ethereum"')
        sys.exit(1)

    query        = " ".join(sys.argv[1:])
    final_state  = run(query, stream=True)
    print_report(final_state)


if __name__ == "__main__":
    main()
