"""
FastAPI Backend
===============
Exposes the LangGraph agent as an API with:
- POST /report — triggers async report generation
- GET /report/{id}/stream — SSE stream of agent progress
- GET /report/{id} — fetch completed report JSON
- GET /report/{id}/pdf — download PDF
- MCP server endpoint
"""

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from dotenv import load_dotenv
load_dotenv()

import os

import redis.asyncio as aioredis
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from backend.agents.graph import generate_report


# ---------------------------------------------------------------------------
# In-memory fallback store (used when Redis is not available)
# ---------------------------------------------------------------------------

class _MemoryStore:
    """Drop-in async Redis stub for local development without a Redis server."""
    def __init__(self):
        self._data: dict = {}

    async def set(self, key: str, value: str, ex: int = None) -> None:
        self._data[key] = value

    async def get(self, key: str) -> str | None:
        return self._data.get(key)

    async def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    try:
        client = await aioredis.from_url(redis_url)
        await client.ping()          # test connection immediately
        app.state.redis = client
        print("INFO:     Redis connected ✓")
    except Exception:
        app.state.redis = _MemoryStore()
        print("INFO:     Redis unavailable — using in-memory store (dev mode)")
    yield
    await app.state.redis.close()


app = FastAPI(title="Crypto Report Agent", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-app.railway.app"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ReportRequest(BaseModel):
    coin: str
    depth: str = "standard"  # quick | standard | deep


class ReportStatus(BaseModel):
    report_id: str
    status: str
    coin: str
    progress: list[str]
    final_report: dict | None = None
    pdf_available: bool = False


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    """Reports which API keys and services are configured."""
    import httpx
    from backend.agents.synthesis_agent import _DB_AVAILABLE

    keys = {
        "GOOGLE_API_KEY":      bool(os.getenv("GOOGLE_API_KEY")),
        "GLASSNODE_API_KEY":   bool(os.getenv("GLASSNODE_API_KEY")),
        "ETHERSCAN_API_KEY":   bool(os.getenv("ETHERSCAN_API_KEY")),
        "DUNE_API_KEY":        bool(os.getenv("DUNE_API_KEY")),
        "LUNARCRUSH_API_KEY":  bool(os.getenv("LUNARCRUSH_API_KEY")),
        "TWITTER_BEARER_TOKEN": bool(os.getenv("TWITTER_BEARER_TOKEN")),
        "CRYPTOPANIC_API_KEY": bool(os.getenv("CRYPTOPANIC_API_KEY")),
        "FRED_API_KEY":        bool(os.getenv("FRED_API_KEY")),
        "COINGLASS_API_KEY":   bool(os.getenv("COINGLASS_API_KEY")),
        "COINGECKO_API_KEY":   bool(os.getenv("COINGECKO_API_KEY")),
    }

    # Test free APIs that require no key
    free_apis = {}
    async with httpx.AsyncClient(timeout=5) as client:
        for name, url in [
            ("coingecko_global", "https://api.coingecko.com/api/v3/global"),
            ("fear_greed",       "https://api.alternative.me/fng/?limit=1"),
        ]:
            try:
                r = await client.get(url)
                free_apis[name] = r.status_code == 200
            except Exception:
                free_apis[name] = False

    return {
        "api_keys":    keys,
        "free_apis":   free_apis,
        "database":    _DB_AVAILABLE,
        "redis":       type(app.state.redis).__name__ != "_MemoryStore",
        "gemini_model": os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
    }


@app.post("/report", response_model=ReportStatus)
async def create_report(request: ReportRequest, background_tasks: BackgroundTasks):
    """
    Triggers async report generation. Returns a report_id immediately.
    Frontend polls /report/{id} or subscribes to /report/{id}/stream.
    """
    report_id = str(uuid.uuid4())
    redis = app.state.redis

    # Store initial state
    initial_state = {
        "report_id": report_id,
        "coin": request.coin.upper(),
        "status": "running",
        "progress": ["[orchestrator] Starting report generation..."],
        "final_report": None,
        "pdf_available": False,
    }
    await redis.set(f"report:{report_id}", json.dumps(initial_state), ex=3600)

    # Use FastAPI BackgroundTasks — keeps event loop alive for the full pipeline
    background_tasks.add_task(_run_report_task, report_id, request.coin, request.depth, redis)

    return ReportStatus(**initial_state)


@app.get("/report/{report_id}/stream")
async def stream_report_progress(report_id: str):
    """
    Server-Sent Events stream. Frontend subscribes and gets live progress updates
    as each agent completes. This is what makes the UI feel alive.
    """
    redis = app.state.redis

    async def event_generator() -> AsyncGenerator[str, None]:
        last_progress_count = 0
        while True:
            raw = await redis.get(f"report:{report_id}")
            if not raw:
                yield f"data: {json.dumps({'error': 'Report not found'})}\n\n"
                break

            state = json.loads(raw)
            progress = state.get("progress", [])

            # Send any new progress messages
            new_messages = progress[last_progress_count:]
            for msg in new_messages:
                yield f"data: {json.dumps({'type': 'progress', 'message': msg})}\n\n"
            last_progress_count = len(progress)

            if state["status"] == "complete":
                yield f"data: {json.dumps({'type': 'complete', 'report_id': report_id})}\n\n"
                break
            elif state["status"] == "failed":
                yield f"data: {json.dumps({'type': 'error', 'message': 'Report generation failed'})}\n\n"
                break

            await asyncio.sleep(0.5)  # Poll Redis every 500ms

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@app.get("/report/{report_id}", response_model=ReportStatus)
async def get_report(report_id: str):
    """Fetch the current state of a report."""
    redis = app.state.redis
    raw = await redis.get(f"report:{report_id}")
    if not raw:
        raise HTTPException(status_code=404, detail="Report not found")
    return ReportStatus(**json.loads(raw))


@app.get("/report/{report_id}/pdf")
async def download_pdf(report_id: str):
    """Download the generated PDF report."""
    redis = app.state.redis
    raw = await redis.get(f"report:{report_id}")
    if not raw:
        raise HTTPException(status_code=404, detail="Report not found")
    
    state = json.loads(raw)
    if state["status"] != "complete":
        raise HTTPException(status_code=202, detail="Report still generating")
    
    pdf_path = state.get("pdf_path")
    if not pdf_path:
        raise HTTPException(status_code=404, detail="PDF not found")
    
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"{state['coin']}_report_{report_id[:8]}.pdf",
    )


# ---------------------------------------------------------------------------
# MCP Server endpoint
# ---------------------------------------------------------------------------

@app.post("/mcp/generate_report")
async def mcp_generate_report(request: ReportRequest):
    """
    MCP tool endpoint — exposes the report agent as a tool that any
    Claude instance, Cursor session, or Claude Code can call directly.
    
    Register this in your MCP config:
    {
      "crypto-report": {
        "command": "npx",
        "args": ["mcp-remote", "https://your-app.railway.app/mcp"]
      }
    }
    """
    # For MCP, run synchronously and return the full report
    result = await generate_report(request.coin, request.depth)
    return {
        "success": result["status"] == "complete",
        "coin": request.coin,
        "report": result.get("final_report", {}),
        "pdf_path": result.get("pdf_path"),
    }


# ---------------------------------------------------------------------------
# Background task
# ---------------------------------------------------------------------------

from fastapi.staticfiles import StaticFiles

# Mount frontend last — catch-all must come after all API routes
_frontend_dir = (
    os.getenv("FRONTEND_DIR")
    or os.path.join(os.path.dirname(__file__), "..", "..", "frontend")
)
if os.path.isdir(_frontend_dir):
    app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="frontend")


async def _run_report_task(report_id: str, coin: str, depth: str, redis):
    """Runs the full LangGraph pipeline and persists state to Redis."""
    try:
        # Run the agent graph
        final_state = await generate_report(coin, depth)

        # Persist to Redis
        state = json.loads(await redis.get(f"report:{report_id}") or "{}")
        state.update({
            "status": final_state["status"],
            "progress": final_state["progress"],
            "final_report": final_state.get("final_report"),
            "pdf_path": final_state.get("pdf_path"),
            "pdf_available": bool(final_state.get("pdf_path")),
        })
        await redis.set(f"report:{report_id}", json.dumps(state), ex=3600)

    except Exception as e:
        raw = await redis.get(f"report:{report_id}")
        state = json.loads(raw) if raw else {}
        state["status"] = "failed"
        state.setdefault("progress", []).append(f"[error] {e}")
        await redis.set(f"report:{report_id}", json.dumps(state), ex=3600)
