"""
Shared state schema for the multi-agent crypto report system.

This TypedDict is the single source of truth for the LangGraph state machine
defined in backend/agents/graph.py. Every node reads from and writes to this.

Rules (from AGENTS.md):
  - `progress` is append-only. Always copy then append — never mutate in place.
  - `revision_count` starts at 0, incremented only by the revise node.
  - `draft_report` is overwritten on each revision (previous draft not stored).
  - Nodes return ONLY the keys they modify.
"""
from __future__ import annotations

from typing import Any
from typing_extensions import TypedDict


class ReportState(TypedDict):
    # ── Input ─────────────────────────────────────────────────────────────────
    coin:         str    # ticker, e.g. "ETH", "BTC", "SOL"
    depth:        str    # "quick" | "standard" | "deep"
    requested_at: str    # ISO timestamp

    # ── Orchestrator output ────────────────────────────────────────────────────
    orchestration_plan: dict[str, Any]   # focus_areas, known_risks, comparables, priority

    # ── Parallel research agent outputs ───────────────────────────────────────
    onchain_data:   dict[str, Any]
    sentiment_data: dict[str, Any]
    market_data:    dict[str, Any]
    macro_data:     dict[str, Any]

    # ── Synthesis + critic loop ────────────────────────────────────────────────
    draft_report:  str              # JSON string of the synthesis report
    critique:      dict[str, Any]   # {score, approved, issues, strengths, ...}
    revision_count: int             # 0-based; force-approve at >= 2

    # ── Final output ──────────────────────────────────────────────────────────
    final_report: dict[str, Any]
    pdf_path:     str | None

    # ── Observability (drives SSE stream) ────────────────────────────────────
    status:   str         # "running" | "complete" | "failed"
    progress: list[str]   # append-only log
