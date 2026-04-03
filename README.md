# Crypto Report Agent

An institutional-grade crypto research report generator powered by a multi-agent AI pipeline. Submit a ticker symbol and receive a structured investment report with on-chain analysis, sentiment scoring, market structure, macro assessment, and a critic-reviewed rating.

**Live demo:** `https://bockbrendan--crypto-report-agent-web.modal.run`

---

## Architecture

The system is a [LangGraph](https://github.com/langchain-ai/langgraph) `StateGraph` with 6 nodes running on FastAPI, deployed serverlessly on Modal.

```
POST /report
     │
     ▼
orchestrator ──► parallel_research ──► synthesizer ──► critic ──► renderer
                 ┌──────────────┐                         │
                 │ onchain      │                    score < 7 &
                 │ sentiment    │                    revisions < 2
                 │ market       │                         │
                 │ macro        │                         ▼
                 └──────────────┘                      revise
                 (asyncio.gather)                         │
                                                          └──► critic (loop)
```

### Agent roles

| Agent | Type | Data sources |
|---|---|---|
| **Orchestrator** | Planner | LLM only — outputs `focus_areas`, `known_risks`, `comparables` |
| **Onchain** | Tool-use loop | CoinGlass (exchange balances, ETF flows, BTC metrics); Glassnode, Etherscan, Dune (optional) |
| **Sentiment** | Tool-use loop | X (Twitter) via API v2, Fear & Greed Index |
| **Market** | Tool-use loop | CoinGecko (spot, ATH, market cap), CoinGlass (derivatives, liquidations, taker volumes, technicals) |
| **Macro** | Tool-use loop | CoinGecko global, FRED (fed rate), alternative.me, stooq (SPY) |
| **Synthesis** | RAG + structured output | All 4 agent outputs + pgvector similarity search |
| **Critic** | Adversarial review | Reviews draft against raw data — scores 0–10 |
| **Revision** | Rewriter | Applies critic feedback, max 2 passes |
| **Renderer** | Output | Parses JSON, generates WeasyPrint PDF |

### Key design decisions

- **Parallel research** — 4 agents run concurrently via `asyncio.gather`, cutting data collection from ~20s to ~5s
- **`_raw_tool_results` pattern** — each agent captures raw tool outputs keyed by function name and passes them through state; synthesis reads these with deterministic programmatic builders (`_build_onchain_text`, `_build_market_text`, `_build_sentiment_text`) that produce the final analysis text directly — bypassing Gemini's tendency to hallucinate or omit API values in structured-output mode; these fields are also locked in the revision node so critic/revision cycles cannot overwrite them
- **Graceful degradation** — every agent returns an error dict instead of crashing when API keys are missing; synthesis proceeds with available data
- **Critic loop** — a separate LLM instance with adversarial prompting reviews each draft; forces 2 revision passes before approving
- **pgvector RAG** — completed reports are embedded (384-dim, `all-MiniLM-L6-v2`) and stored in PostgreSQL; future reports on similar coins retrieve past context
- **SSE streaming** — live progress updates pushed to the browser as each agent completes

---

## Project structure

```
crypto_agent/
├── backend/
│   ├── agents/
│   │   ├── main.py          # FastAPI server + SSE endpoints
│   │   ├── graph.py         # LangGraph StateGraph definition
│   │   ├── onchain_agent.py # Glassnode / Etherscan tool-use agent
│   │   ├── sentiment_agent.py
│   │   ├── market_agent.py
│   │   ├── macro_agent.py
│   │   ├── synthesis_agent.py  # RAG + report writer
│   │   ├── critic_agent.py
│   │   └── renderer.py      # JSON → PDF via WeasyPrint
│   ├── core/
│   │   └── state.py         # ReportState TypedDict
│   └── tools/
│       └── macro_tools.py   # Async HTTP fetch functions
├── frontend/
│   └── index.html           # Vanilla JS SPA
├── modal_app.py             # Modal serverless deployment
├── Dockerfile               # Docker deployment
├── requirements.txt
└── .env.example
```

---

## API endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/report` | Start report generation. Returns `report_id` immediately. |
| `GET` | `/report/{id}/stream` | SSE stream of live progress messages |
| `GET` | `/report/{id}` | Fetch completed report JSON |
| `GET` | `/report/{id}/pdf` | Download PDF |
| `POST` | `/mcp/generate_report` | MCP tool endpoint for Claude/Cursor integration |
| `GET` | `/` | Web UI |

### Report request
```json
POST /report
{
  "coin": "BTC",
  "depth": "standard"
}
```
`depth` options: `quick` | `standard` | `deep`

### Report response shape
```json
{
  "executive_summary": "...",
  "rating": "BUY | HOLD | SELL | AVOID",
  "price_target_range": { "low": 80000, "high": 105000, "timeframe": "90d" },
  "confidence": "low | medium | high",
  "composite_scores": {
    "onchain_health": 72,
    "sentiment": 58,
    "market_structure": 65,
    "macro": 70,
    "overall": 68
  },
  "bull_case": "...",
  "bear_case": "...",
  "key_risks": ["...", "..."],
  "onchain_analysis": "...",
  "sentiment_analysis": "...",
  "market_analysis": "...",
  "macro_analysis": "...",
  "comparable_analysis": "...",
  "conclusion": "..."
}
```

Rating rubric: `BUY` > 70 | `HOLD` 40–70 | `SELL` 20–40 | `AVOID` < 20  
Score weights: Onchain 35% | Market 30% | Macro 20% | Sentiment 15%

---

## Setup

### Prerequisites
- Python 3.11+
- `GOOGLE_API_KEY` (Gemini) — required

### Local development

```bash
# Clone and install
git clone https://github.com/YOUR_USERNAME/crypto-agent.git
cd crypto-agent
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your API keys

# Run
uvicorn backend.agents.main:app --reload
# Open http://localhost:8000
```

### Environment variables

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_API_KEY` | Yes | Gemini API key |
| `GEMINI_MODEL` | No | Model name (default: `gemini-2.0-flash`) |
| `COINGLASS_API_KEY` | No | CoinGlass v4 — exchange flows, ETF flows, liquidations, taker volumes, technicals |
| `TWITTER_BEARER_TOKEN` | No | X (Twitter) API v2 — social sentiment and top posts |
| `DATABASE_URL` | No | PostgreSQL connection string for pgvector RAG |
| `GLASSNODE_API_KEY` | No | Glassnode — MVRV, NVT, active addresses (optional, degrades gracefully) |
| `ETHERSCAN_API_KEY` | No | Etherscan — whale transaction tracking for EVM assets |
| `DUNE_API_KEY` | No | Dune Analytics — advanced on-chain queries |
| `FRED_API_KEY` | No | FRED — Federal funds rate data |
| `REDIS_URL` | No | Redis for report caching (falls back to in-memory) |

All optional keys degrade gracefully — the pipeline runs with whatever data is available. CoinGlass and X (Twitter) provide the highest-value optional data.

### pgvector RAG (optional)

Enables memory of past reports for comparable analysis.

1. Create a free [Supabase](https://supabase.com) project
2. Run in the SQL editor:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS reports (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    coin          VARCHAR(10)  NOT NULL,
    rating        VARCHAR(10),
    overall_score INT,
    report_json   JSONB        NOT NULL,
    embedding     vector(384)  NOT NULL,
    created_at    TIMESTAMP    NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS reports_embedding_idx
    ON reports USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS reports_coin_idx ON reports (coin);
```
3. Add `DATABASE_URL` to your `.env`

### PDF generation (optional)

Requires system libraries. On macOS with conda:
```bash
conda install -c conda-forge pango cairo gdk-pixbuf
```

---

## Deployment

### Modal (recommended)

```bash
pip install modal
modal token new

# Create secrets
modal secret create crypto-intel-secrets \
  GOOGLE_API_KEY=your_key \
  GEMINI_MODEL=gemini-2.0-flash \
  'DATABASE_URL=postgresql://...'

# Deploy
modal deploy modal_app.py
# → https://YOUR_USERNAME--crypto-report-agent-web.modal.run
```

### Docker

```bash
docker build -t crypto-report-agent .
docker run -p 8000:8000 --env-file .env crypto-report-agent
```

### Railway

1. Push to GitHub
2. New project → Deploy from GitHub repo
3. Add environment variables in the Railway dashboard
4. Public URL auto-assigned

---

## Tech stack

- **LLM** — Google Gemini 2.0 Flash via `langchain-google-genai`
- **Orchestration** — LangGraph `StateGraph`
- **Embeddings** — `sentence-transformers/all-MiniLM-L6-v2` (local, 384-dim)
- **Vector store** — PostgreSQL + pgvector
- **API** — FastAPI + SSE streaming
- **PDF** — WeasyPrint
- **Deployment** — Modal serverless

---

## Disclaimer

Reports are generated by AI and do not constitute financial advice. Past performance is not indicative of future results.
