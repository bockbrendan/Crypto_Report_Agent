# crypto_agent — Claude Code Rules

## SENSITIVE FILE ACCESS — DENY

Claude must NEVER read, print, log, or expose the contents of:
- `.env` and any `.env.*` files
- `*.pem`, `*.key`, `*.secret`, `*.secrets`
- `id_rsa`, `id_ed25519`, or any SSH private key files
- `credentials.json`, `credentials.*`
- `wallet_data.txt` and any wallet seed/mnemonic files
- Any file containing the words `private_key`, `secret`, `mnemonic`, or `seed`

If a task requires inspecting these files, stop and ask the user to provide
only the specific non-sensitive value needed.

## Editing Rules:
Before using any file editing or writing tools, you must first provide a brief explanation of the edit you are about to make. Detail exactly why you are making the change and which specific files and functions are being targeted.

## CODE CONVENTIONS

- Python 3.11+. Type hints required on all new functions.
- All API secrets are loaded from env variables — never hardcoded.
- All HTTP calls in `backend/agents/` use `httpx.AsyncClient` directly inside each agent file.
- LangGraph nodes return ONLY the state keys they modify (never the full state dict).
- Secrets must never appear in log output, `print()` calls, or LLM prompts.

## PROJECT STRUCTURE

```
crypto_agent/
  backend/
    agents/
      main.py             ← FastAPI server + SSE endpoints + static mount
      graph.py            ← LangGraph StateGraph definition
      onchain_agent.py    ← CoinGlass exchange flows, ETF flows, BTC metrics; Glassnode (optional)
      sentiment_agent.py  ← X (Twitter) v2, Fear & Greed Index
      market_agent.py     ← CoinGecko spot/ATH, CoinGlass derivatives/liquidations/technicals
      macro_agent.py      ← CoinGecko global, FRED, stooq (SPY)
      synthesis_agent.py  ← RAG + report writer; reads _raw_tool_results for deterministic data
      critic_agent.py     ← Adversarial reviewer (scores 0–10)
      renderer.py         ← JSON → WeasyPrint PDF
    core/
      state.py            ← ReportState TypedDict
    tools/
      macro_tools.py      ← Async HTTP helpers for macro agent
  frontend/
    index.html            ← Vanilla JS SPA (landing, progress, report views)
  modal_app.py            ← Modal serverless deployment (run from project root)
  Dockerfile
  requirements.txt
  .env                    ← NEVER READ THIS FILE
  wallet/                 ← NEVER READ FILES IN THIS DIRECTORY
```

## ENV VARIABLES (reference only — do not read the actual .env)

| Variable | Purpose |
|---|---|
| `GOOGLE_API_KEY` | Gemini LLM |
| `GEMINI_MODEL` | Model name (default: `gemini-2.0-flash`) |
| `COINGLASS_API_KEY` | CoinGlass v4 (`open-api-v4.coinglass.com`, header `CG-API-KEY`) |
| `TWITTER_BEARER_TOKEN` | X (Twitter) API v2 bearer token |
| `DATABASE_URL` | PostgreSQL + pgvector for RAG |
| `GLASSNODE_API_KEY` | Glassnode on-chain metrics (optional) |
| `ETHERSCAN_API_KEY` | Etherscan EVM whale tracking (optional) |
| `FRED_API_KEY` | FRED federal funds rate (optional) |
| `REDIS_URL` | Redis report cache (falls back to in-memory) |

## RUNNING THE AGENT

```bash
cd "/Volumes/drive_4tb/Hopkins MS in AI/crypto_agent"
# Local dev
uvicorn backend.agents.main:app --reload
# Open http://localhost:8000

# Modal deployment (always stop first to flush container)
modal app stop crypto-report-agent
modal deploy modal_app.py
```

## SYNTHESIS CONVENTIONS

- `_raw_tool_results` — each agent attaches `{"tool_fn_name": result}` to its return dict.
  Synthesis reads these with `raw.get("get_coinglass_exchange_flows")` etc. — never trust
  LLM-renamed keys from the parsed JSON output.
- **Programmatic analysis builders** — `_build_onchain_text`, `_build_sentiment_text`,
  `_build_market_text` are closures inside `run_synthesis_agent`. They read `_raw_tool_results`
  and produce deterministic prose strings with exact API numbers. After `result.model_dump()`,
  synthesis overrides the three corresponding fields with their output. This guarantees data
  citation regardless of how Gemini behaves in `with_structured_output` mode.
- **Locked fields in revision** — `graph.py`'s `revision_node._LOCKED_FIELDS` includes
  `onchain_analysis`, `sentiment_analysis`, and `market_analysis`. The revision LLM cannot
  overwrite these fields — they are always preserved from the synthesis programmatic build.
- `_fmt_market`, `_fmt_onchain`, `_fmt_sentiment` still exist and inject data into the synthesis
  prompt as context for Gemini's judgment (scores, rating, narrative) — but they no longer control
  the final text of the analysis fields.
- Composite scores must not penalize missing optional API sources (Glassnode, Etherscan, Dune).
  Missing = neutral weight (excluded from scoring), never negative evidence.
- Debug prints are intentionally left in all agents. Do not remove them.

## DEPENDENCIES (install if missing)

```bash
pip install -r requirements.txt
# PDF generation (macOS):
conda install -c conda-forge pango cairo gdk-pixbuf
```
