"""
FastAPI application for the Crypto Intel Agent.

Endpoints:
  POST /analyze  — streams analysis via Server-Sent Events
  GET  /health   — liveness check

Run locally:
  uvicorn api.main:app --reload --port 8000

Or via Docker:
  docker-compose up --build
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from .models import AnalyzeRequest, HealthResponse
from .streaming import graph_event_generator

# Load secrets from .env (parent of the api/ directory → crypto_agent/.env)
load_dotenv(Path(__file__).parent.parent / ".env")

_VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # graph is compiled at import time in intel_agent/graph.py — nothing to warm up
    yield


app = FastAPI(
    title="Crypto Intel Agent",
    description="Full due-diligence reports on any crypto asset, streamed in real time.",
    version=_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten to your frontend domain in production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post(
    "/analyze",
    summary="Stream a crypto due-diligence report",
    response_description="Server-Sent Events stream",
)
async def analyze(request: AnalyzeRequest):
    """
    Accepts a natural-language query and streams analysis progress via SSE.

    **Event types emitted:**
    - `progress` — one per agent node (5 total), includes step number and label
    - `report`   — full markdown report + risk score (1-10), after all nodes complete
    - `error`    — if the agent fails at any point
    - `done`     — final frame, always sent to close the stream cleanly

    **Example client (JavaScript):**
    ```js
    import { fetchEventSource } from "@microsoft/fetch-event-source";

    fetchEventSource("/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: "analyze solana" }),
      onmessage(ev) {
        if (ev.event === "progress") console.log(JSON.parse(ev.data));
        if (ev.event === "report")   renderReport(JSON.parse(ev.data));
      },
    });
    ```
    """
    return EventSourceResponse(graph_event_generator(request.query))


@app.get("/health", response_model=HealthResponse, summary="Liveness check")
async def health():
    return HealthResponse(
        status="ok",
        version=_VERSION,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# Serve frontend — must be mounted last so /analyze and /health take priority
_frontend = Path(__file__).parent.parent / "frontend"
if _frontend.exists():
    app.mount("/", StaticFiles(directory=str(_frontend), html=True), name="frontend")
