"""
Crypto Report Agent - LangGraph Multi-Agent Graph
================================================
Orchestrates 5 specialized agents to produce an institutional-grade
crypto research report. Each agent runs in parallel where possible,
then a synthesis + critic loop produces the final report.
"""

from __future__ import annotations

import asyncio
from typing import Any
from datetime import datetime

import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from backend.agents.onchain_agent import run_onchain_agent
from backend.agents.sentiment_agent import run_sentiment_agent
from backend.agents.market_agent import run_market_agent
from backend.agents.macro_agent import run_macro_agent
from backend.agents.synthesis_agent import run_synthesis_agent
from backend.agents.critic_agent import run_critic_agent
from backend.agents.renderer import render_report
from backend.core.state import ReportState  # single source of truth


# ---------------------------------------------------------------------------
# Graph definition
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    graph = StateGraph(ReportState)

    # Nodes
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("parallel_research", parallel_research_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("critic", critic_node)
    graph.add_node("revise", revision_node)
    graph.add_node("renderer", renderer_node)

    # Edges
    graph.add_edge(START, "orchestrator")
    graph.add_edge("orchestrator", "parallel_research")
    graph.add_edge("parallel_research", "synthesizer")
    graph.add_edge("synthesizer", "critic")
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "revise": "revise",
            "render": "renderer",
        },
    )
    graph.add_edge("revise", "critic")  # Loop: revise -> critic -> render (max 2 iterations)
    graph.add_edge("renderer", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

async def orchestrator_node(state: ReportState) -> dict:
    """
    Plans the research strategy based on the coin and depth requested.
    Returns enriched context that sub-agents will use.
    """
    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

    prompt = f"""You are orchestrating a crypto research report for {state['coin']}.
    
Research depth: {state['depth']}

Your job:
1. Identify the top 3 most important things to investigate for this specific coin
2. Flag any known risks or controversies to watch for
3. Identify 2-3 comparable coins for benchmarking

Respond in JSON with keys: focus_areas (list), known_risks (list), comparables (list), priority (str)
"""
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    
    import json
    try:
        plan = json.loads(response.content)
    except Exception:
        plan = {"focus_areas": [], "known_risks": [], "comparables": [], "priority": "balanced"}

    return {
        "status": "running",
        "progress": [f"[orchestrator] Research plan created for {state['coin']}"],
        "orchestration_plan": plan,
        "revision_count": 0,
    }


async def parallel_research_node(state: ReportState) -> dict:
    """
    Runs all 4 research agents concurrently using asyncio.gather.
    This is the most important performance optimization — 4 API calls
    in ~5s instead of ~20s sequential.
    """
    coin = state["coin"]
    plan = state.get("orchestration_plan", {})

    progress = list(state.get("progress", []))
    progress.append(f"[parallel_research] Spawning 4 agents for {coin}...")

    # Fan out — all run concurrently
    onchain, sentiment, market, macro = await asyncio.gather(
        run_onchain_agent(coin, plan),
        run_sentiment_agent(coin, plan),
        run_market_agent(coin, plan),
        run_macro_agent(coin, plan),
        return_exceptions=True,  # Don't let one failure kill all agents
    )

    # Graceful degradation: if an agent failed, return error dict
    def safe(result, name: str) -> dict:
        if isinstance(result, Exception):
            progress.append(f"[{name}] WARNING: failed — {result}")
            return {"error": str(result), "agent": name}
        progress.append(f"[{name}] Complete ✓")
        return result

    return {
        "onchain_data": safe(onchain, "onchain"),
        "sentiment_data": safe(sentiment, "sentiment"),
        "market_data": safe(market, "market"),
        "macro_data": safe(macro, "macro"),
        "progress": progress,
    }


async def synthesizer_node(state: ReportState) -> dict:
    """
    The main intelligence layer. Takes all 4 agent outputs + RAG context
    and writes the full institutional report draft.
    """
    progress = list(state.get("progress", []))
    progress.append("[synthesizer] Generating report draft...")

    draft = await run_synthesis_agent(
        coin=state["coin"],
        onchain=state["onchain_data"],
        sentiment=state["sentiment_data"],
        market=state["market_data"],
        macro=state["macro_data"],
        plan=state.get("orchestration_plan", {}),
    )

    progress.append("[synthesizer] Draft complete ✓")
    return {"draft_report": draft, "progress": progress}


async def critic_node(state: ReportState) -> dict:
    """
    Independent critic agent that reviews the draft for:
    - Unsupported claims
    - Contradictions between data sources
    - Overconfident language (price predictions stated as fact)
    - Missing risk disclosures
    - Data staleness issues

    If score >= 7 or revision_count >= 2, approves for rendering.
    """
    progress = list(state.get("progress", []))
    progress.append("[critic] Reviewing draft...")

    critique = await run_critic_agent(
        draft=state["draft_report"],
        raw_data={
            "onchain": state["onchain_data"],
            "sentiment": state["sentiment_data"],
            "market": state["market_data"],
            "macro": state["macro_data"],
        },
        revision_count=state.get("revision_count", 0),
    )

    # Force approval after 2 revisions to avoid infinite loops
    if state.get("revision_count", 0) >= 2:
        critique["approved"] = True

    # Force approval when confidence is low — missing data means revisions
    # degrade quality rather than improve it (revision uses unstructured output)
    import json as _json
    try:
        draft = _json.loads(state.get("draft_report", "{}"))
        if draft.get("confidence") == "low":
            critique["approved"] = True
    except Exception:
        pass

    approved = critique.get("approved", False)
    progress.append(
        f"[critic] Score: {critique.get('score', '?')}/10 — {'Approved ✓' if approved else 'Needs revision'}"
    )

    return {"critique": critique, "progress": progress}


async def revision_node(state: ReportState) -> dict:
    """
    Takes the critic's feedback and revises the report draft.
    Only runs when critic score < 7 and revision_count < 2.
    """
    progress = list(state.get("progress", []))
    progress.append(f"[revise] Applying critic feedback (pass {state.get('revision_count', 0) + 1})...")

    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        temperature=0.2,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

    issues = state["critique"].get("issues", [])
    issues_text = "\n".join(f"- {i}" for i in issues)

    prompt = f"""You are revising a crypto research report based on critic feedback.

ORIGINAL DRAFT:
{state['draft_report']}

CRITIC ISSUES TO FIX:
{issues_text}

Produce a revised report that addresses all issues. Maintain the same JSON structure and field names.
Keep all well-supported claims. Remove or soften unsupported ones.

IMPORTANT: Return ONLY the raw JSON object. No markdown, no backticks, no code fences.
"""
    response = await llm.ainvoke([HumanMessage(content=prompt)])

    # Merge revised output with original — preserve any structured fields the LLM dropped or nulled
    import json as _json
    original: dict = {}
    try:
        original = _json.loads(state["draft_report"])
    except Exception:
        pass

    revised: dict = {}
    try:
        rev_text = response.content.strip()
        if rev_text.startswith("```"):
            rev_text = rev_text.split("```", 2)[1]
            if rev_text.startswith("json"):
                rev_text = rev_text[4:]
            rev_text = rev_text.strip()
        revised = _json.loads(rev_text)
    except Exception:
        revised = original  # revision completely failed — keep original

    # Structured fields come from the synthesis (with_structured_output) — never let the
    # unstructured revision node override them. Revision only improves narrative text.
    # onchain_analysis, sentiment_analysis, market_analysis are built programmatically
    # in synthesis from _raw_tool_results — revision must not overwrite them with
    # Gemini-generated text that ignores the actual API data.
    _LOCKED_FIELDS = {
        "composite_scores", "price_target_range", "rating", "confidence",
        "onchain_analysis", "sentiment_analysis", "market_analysis",
    }

    merged = dict(original)
    for k, v in revised.items():
        if k in _LOCKED_FIELDS:
            continue  # always keep synthesis values
        if v is not None and v != {} and v != []:
            merged[k] = v

    progress.append("[revise] Revision complete ✓")
    return {
        "draft_report": _json.dumps(merged),
        "revision_count": state.get("revision_count", 0) + 1,
        "progress": progress,
    }


async def renderer_node(state: ReportState) -> dict:
    """
    Converts the approved draft into:
    1. Structured JSON (for the web dashboard)
    2. PDF (for download)
    """
    progress = list(state.get("progress", []))
    progress.append("[renderer] Rendering final report...")

    final_report, pdf_path = await render_report(
        coin=state["coin"],
        draft=state["draft_report"],
        onchain=state["onchain_data"],
        sentiment=state["sentiment_data"],
        market=state["market_data"],
        macro=state["macro_data"],
    )

    progress.append(f"[renderer] Report complete — PDF saved to {pdf_path} ✓")
    return {
        "final_report": final_report,
        "pdf_path": pdf_path,
        "status": "complete",
        "progress": progress,
    }


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def route_after_critic(state: ReportState) -> str:
    """Conditional edge: send to revise or directly to render."""
    critique = state.get("critique", {})
    if critique.get("approved", False):
        return "render"
    return "revise"


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

_graph = build_graph()


async def generate_report(coin: str, depth: str = "standard") -> ReportState:
    """Main entrypoint called by the FastAPI endpoint."""
    initial_state: ReportState = {
        "coin": coin.upper(),
        "depth": depth,
        "requested_at": datetime.utcnow().isoformat(),
        "onchain_data": {},
        "sentiment_data": {},
        "market_data": {},
        "macro_data": {},
        "draft_report": "",
        "critique": {},
        "revision_count": 0,
        "final_report": {},
        "pdf_path": None,
        "status": "running",
        "progress": [],
    }
    return await _graph.ainvoke(initial_state)
