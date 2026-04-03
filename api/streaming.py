"""
Thread-queue bridge: runs the synchronous LangGraph pipeline in a worker
thread and feeds SSE events to an async generator for FastAPI.

Why threads instead of asyncio.to_thread(graph.invoke)?
  graph.invoke() returns only the final state — no per-node progress.
  graph.stream() yields state snapshots after each node, but it is a
  blocking synchronous iterator. The bridge below runs it in a daemon
  thread and pushes each snapshot as a structured event onto a Queue.
  The async generator drains the Queue without blocking the event loop.
"""
from __future__ import annotations

import asyncio
import json
import re
import threading
from queue import Queue as SyncQueue
from typing import AsyncGenerator

from langchain_core.messages import HumanMessage

from intel_agent.graph import graph
from intel_agent.state import CryptoIntelState

# ── Node metadata ─────────────────────────────────────────────────────────────

# Maps the bracket prefix used in each node's AIMessage to a (label, step) pair.
# Keep in sync with the node names in intel_agent/graph.py.
NODE_LABELS: dict[str, tuple[str, int]] = {
    "resolve_asset":  ("Resolving asset identity",      1),
    "onchain_node":   ("Fetching on-chain metrics",     2),
    "sentiment_node": ("Analyzing news & sentiment",    3),
    "docs_node":      ("Reading project documentation", 4),
    "report_node":    ("Generating final risk report",  5),
}
TOTAL_STEPS = len(NODE_LABELS)

# Sentinel object that signals end-of-stream from the worker thread
_DONE = object()

# ── Worker thread ─────────────────────────────────────────────────────────────

def _run_graph(query: str, q: SyncQueue) -> None:
    """
    Runs graph.stream() to completion, pushing structured event dicts onto q.
    Always pushes _DONE last (even after an exception) so the async side
    never hangs waiting for a sentinel that never arrives.
    """
    initial_state: CryptoIntelState = {
        "query":    query,
        "messages": [HumanMessage(content=query)],
    }
    try:
        for snapshot in graph.stream(initial_state, stream_mode="values"):
            msgs = snapshot.get("messages", [])
            if not msgs:
                continue

            content = getattr(msgs[-1], "content", "")
            if not content.startswith("["):
                continue

            # Extract node key from bracket prefix: "[resolve_asset] ..." → "resolve_asset"
            m = re.match(r"\[(\w+)\]", content)
            node_key = m.group(1) if m else "unknown"
            label, step_num = NODE_LABELS.get(node_key, (node_key, 0))

            # Progress event (one per node)
            q.put({
                "type": "progress",
                "data": {
                    "node":    node_key,
                    "label":   label,
                    "message": content,
                    "step":    step_num,
                    "total":   TOTAL_STEPS,
                },
            })

            # After report_node: also emit the completed report
            if node_key == "report_node":
                q.put({
                    "type": "report",
                    "data": {
                        "report":     snapshot.get("report", ""),
                        "risk_score": snapshot.get("risk_score"),
                        "coin_name":  snapshot.get("name", ""),
                        "symbol":     snapshot.get("symbol", ""),
                    },
                })

    except Exception as exc:
        q.put({
            "type": "error",
            "data": {"detail": str(exc), "node": None},
        })
    finally:
        q.put(_DONE)


# ── Async generator ───────────────────────────────────────────────────────────

async def graph_event_generator(query: str) -> AsyncGenerator[dict, None]:
    """
    Async generator that yields SSE-compatible dicts:
        {"event": "<type>", "data": "<json_string>"}

    EventSourceResponse from sse-starlette accepts this format directly.
    """
    q: SyncQueue = SyncQueue()

    thread = threading.Thread(
        target=_run_graph,
        args=(query, q),
        daemon=True,    # killed automatically if the process exits
    )
    thread.start()

    while True:
        # Await the next item without blocking the event loop
        item = await asyncio.to_thread(q.get)

        if item is _DONE:
            yield {"event": "done", "data": json.dumps({"status": "complete"})}
            break

        yield {"event": item["type"], "data": json.dumps(item["data"])}
