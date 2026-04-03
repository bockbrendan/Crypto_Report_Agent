"""
Synthesis Agent
===============
The main intelligence layer. Takes all 4 research agent outputs + RAG context
from pgvector and writes the full institutional report via Gemini 2.5 Flash.

Called by graph.py's synthesizer_node as:
    draft_json_str = await run_synthesis_agent(coin, onchain, sentiment, market, macro, plan)

Returns a JSON string (stored as draft_report: str in ReportState).

pgvector table schema:
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE TABLE IF NOT EXISTS reports (
        id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        coin        VARCHAR(10)  NOT NULL,
        rating      VARCHAR(10),
        overall_score INT,
        report_json JSONB        NOT NULL,
        embedding   vector(384)  NOT NULL,   -- all-MiniLM-L6-v2 (local, free)
        created_at  TIMESTAMP    NOT NULL DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS reports_embedding_idx
        ON reports USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
    CREATE INDEX IF NOT EXISTS reports_coin_idx ON reports (coin);

Note: AGENTS.md specifies vector(1536) (OpenAI text-embedding-3-small).
We use 384-dim all-MiniLM-L6-v2 instead — local, free, no extra API key.

Graceful degradation (3 layers):
  1. Module init: _DB_AVAILABLE=False if DATABASE_URL absent / DB unreachable.
  2. query_similar_reports() returns [] on any error.
  3. store_report() is fire-and-forget — synthesis succeeds regardless.
"""
from __future__ import annotations

import asyncio
import json
import os
from textwrap import dedent
from typing import Literal, Optional

import psycopg2
import psycopg2.extras
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from pgvector.psycopg2 import register_vector
from pydantic import BaseModel, Field

# ── Lazy embedding singleton ──────────────────────────────────────────────────

_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        _embedder = HuggingFaceEmbeddings(model_name=_EMBED_MODEL)
    return _embedder


# ── pgvector helpers ──────────────────────────────────────────────────────────

def _get_conn():
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL not set")
    conn = psycopg2.connect(url)
    register_vector(conn)
    return conn


def init_vector_store() -> bool:
    try:
        conn = _get_conn()
        with conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS reports (
                        id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        coin          VARCHAR(10)  NOT NULL,
                        rating        VARCHAR(10),
                        overall_score INT,
                        report_json   JSONB        NOT NULL,
                        embedding     vector(384)  NOT NULL,
                        created_at    TIMESTAMP    NOT NULL DEFAULT NOW()
                    );
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS reports_embedding_idx
                        ON reports USING ivfflat (embedding vector_cosine_ops)
                        WITH (lists = 100);
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS reports_coin_idx ON reports (coin);
                """)
        conn.close()
        return True
    except Exception:
        return False


def query_similar_reports(coin: str, embedding: list[float], limit: int = 3) -> list[dict]:
    """Cosine similarity search. Returns [] on any error (Layer 2 degradation)."""
    try:
        conn = _get_conn()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT report_json, rating, overall_score, created_at
                FROM   reports
                WHERE  coin = %s
                ORDER  BY embedding <=> %s
                LIMIT  %s
                """,
                (coin.upper(), embedding, limit),
            )
            rows = cur.fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def store_report(coin: str, report: dict, embedding: list[float]) -> None:
    """Insert completed report. Errors are swallowed (Layer 3 degradation)."""
    try:
        conn = _get_conn()
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO reports (coin, rating, overall_score, report_json, embedding)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        coin.upper(),
                        report.get("rating"),
                        (report.get("composite_scores") or {}).get("overall"),
                        json.dumps(report),
                        embedding,
                    ),
                )
                row_id = cur.fetchone()[0]
        conn.close()
        print(f"INFO: store_report — inserted {coin} report id={row_id}")
    except Exception as exc:
        print(f"WARNING: store_report failed — {exc}")


# Layer 1: module-level init
_DB_AVAILABLE: bool = False
try:
    _DB_AVAILABLE = init_vector_store()
    if _DB_AVAILABLE:
        print(f"INFO: pgvector DB connected — RAG enabled ✓")
    else:
        print(f"WARNING: pgvector DB init returned False — RAG disabled")
except Exception as _db_exc:
    print(f"WARNING: pgvector DB unavailable — {_db_exc}")


# ── Pydantic output model ─────────────────────────────────────────────────────

class PriceTargetRange(BaseModel):
    low:       float
    high:      float
    timeframe: str = "90d"


class CompositeScores(BaseModel):
    onchain_health:   int = Field(ge=0, le=100)
    sentiment:        int = Field(ge=0, le=100)
    market_structure: int = Field(ge=0, le=100)
    macro:            int = Field(ge=0, le=100)
    overall:          int = Field(ge=0, le=100)


class SynthesisReport(BaseModel):
    executive_summary:   str
    rating:              Literal["BUY", "HOLD", "SELL", "AVOID"]
    price_target_range:  PriceTargetRange
    confidence:          Literal["low", "medium", "high"]
    composite_scores:    CompositeScores
    bull_case:           str
    bear_case:           str
    key_risks:           list[str]
    onchain_analysis:    str
    sentiment_analysis:  str
    market_analysis:     str
    macro_analysis:      str
    comparable_analysis: str
    conclusion:          str


# ── LLM factory ──────────────────────────────────────────────────────────────

def _llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        temperature=0.2,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )


# ── Main entrypoint (called by graph.py) ─────────────────────────────────────

async def run_synthesis_agent(
    coin: str,
    onchain: dict,
    sentiment: dict,
    market: dict,
    macro: dict,
    plan: dict,
) -> str:
    """
    Embeds query context, fetches similar past reports via pgvector,
    calls Gemini with all 4 agent outputs + RAG context, stores the result.
    Returns a JSON string (stored as draft_report in ReportState).

    Rating rubric (from AGENTS.md):
      BUY  = overall > 70
      HOLD = 40–70
      SELL = 20–40
      AVOID = < 20

    Score weights:
      Onchain: 35% | Market: 30% | Macro: 20% | Sentiment: 15%
    """
    # Step 0: fetch current price directly from CoinGecko as price target anchor
    current_price: Optional[float] = None
    try:
        import httpx
        _id_map = {"BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana",
                   "BNB": "binancecoin", "XRP": "ripple", "ADA": "cardano",
                   "AVAX": "avalanche-2", "DOT": "polkadot", "LINK": "chainlink",
                   "MATIC": "matic-network", "UNI": "uniswap", "ATOM": "cosmos"}
        cg_id = _id_map.get(coin.upper(), coin.lower())
        _cg_key = os.getenv("COINGECKO_API_KEY", "")
        _headers = {"x-cg-demo-api-key": _cg_key} if _cg_key else {}
        async with httpx.AsyncClient(timeout=5) as _client:
            _r = await _client.get(
                "https://api.coingecko.com/api/v3/simple/price",
                params={"ids": cg_id, "vs_currencies": "usd"},
                headers=_headers,
            )
            current_price = _r.json().get(cg_id, {}).get("usd")
            print(f"INFO: CoinGecko price for {coin} ({cg_id}): ${current_price}")
    except Exception as _cg_exc:
        print(f"WARNING: CoinGecko price fetch failed for {coin} — {_cg_exc}")

    # Step 1: embed for RAG (HuggingFace is sync — run in thread)
    embed_text = f"{coin} crypto investment report {' '.join(plan.get('focus_areas', []))}"
    embedding: list[float] = await asyncio.to_thread(
        _get_embedder().embed_query, embed_text
    )

    # Step 2: RAG lookup (Layer 2 — returns [] on failure)
    past_reports = query_similar_reports(coin, embedding, limit=3)

    # Step 3: build RAG section
    rag_section = ""
    if past_reports:
        excerpts = []
        for r in past_reports:
            rj = r.get("report_json") or {}
            excerpt = json.dumps({
                "coin": r.get("coin"),
                "rating": r.get("rating"),
                "overall_score": r.get("overall_score"),
                "executive_summary": (rj.get("executive_summary") or "")[:400],
                "conclusion": (rj.get("conclusion") or "")[:300],
            }, default=str)
            excerpts.append(excerpt)
        rag_section = (
            "\n### PAST REPORT CONTEXT (pgvector similarity search — use for comparable_analysis)\n"
            + "\n---\n".join(excerpts) + "\n"
        )

    # Step 4: build prompt — extract key values as flat lines so Gemini cannot misread nested JSON

    def _v(val: any, fmt: str = "") -> str:
        """Format a value, returning 'N/A' only if truly missing."""
        if val is None or val == "" or val == {}:
            return "N/A"
        try:
            if fmt == "$":
                return f"${float(val):,.0f}"
            if fmt == "%":
                return f"{float(val):+.2f}%"
            if fmt == "bn":
                return f"${float(val)/1e9:.2f}B"
            if fmt == "m":
                return f"${float(val)/1e6:.1f}M"
        except (TypeError, ValueError):
            pass
        return str(val)

    def _fmt_market(d: Optional[dict]) -> str:
        if not d or d.get("error"):
            return "## MARKET STRUCTURE DATA\n*Unavailable*\n"
        raw   = d.get("_raw_tool_results") or {}
        spot  = raw.get("get_spot_data") or d.get("spot") or {}
        deriv = raw.get("get_derivatives_data") or d.get("coinglass") or {}
        liq   = raw.get("get_liquidation_data") or {}
        taker = raw.get("get_taker_volume") or {}
        tech  = raw.get("compute_technical_indicators") or d.get("technicals") or {}
        cg    = {**deriv, **liq, **taker}
        print(f"DEBUG _fmt_market: spot_price={spot.get('price_usd')} ath={spot.get('ath')} rsi={tech.get('rsi_14')} funding={cg.get('funding_rate_pct')} long_liq={cg.get('long_liquidations_24h_usd')}")

        facts = [
            f"The spot price is {_v(spot.get('price_usd'), '$')}, with 24h change of {_v(spot.get('price_change_24h'), '%')}, 7d change of {_v(spot.get('price_change_7d'), '%')}, and 30d change of {_v(spot.get('price_change_30d'), '%')}.",
            f"Market cap is {_v(spot.get('market_cap'), 'bn')} (rank #{_v(spot.get('market_cap_rank'))}); 24h volume is {_v(spot.get('volume_24h'), 'bn')}.",
            f"The all-time high is {_v(spot.get('ath'), '$')}, and the current price is {_v(spot.get('ath_change_percentage'), '%')} from that ATH.",
            f"Funding rate: {_v(cg.get('funding_rate_pct'))}% ({_v(cg.get('funding_sentiment'))}); open interest: {_v(cg.get('open_interest_usd'), 'bn')} ({_v(cg.get('oi_change_4h_pct'), '%')} in 4h).",
            f"24h liquidations: longs {_v(cg.get('long_liquidations_24h_usd'), 'm')} vs shorts {_v(cg.get('short_liquidations_24h_usd'), 'm')}; dominant side: {_v(cg.get('dominant_side'))}.",
            f"Taker buy volume: {_v(cg.get('taker_buy_volume_usd'), 'm')} ({_v(cg.get('buy_ratio_pct'))}%); taker sell volume: {_v(cg.get('taker_sell_volume_usd'), 'm')} ({_v(cg.get('sell_ratio_pct'))}%); bias: {_v(cg.get('bias'))}.",
            f"RSI(14): {_v(tech.get('rsi_14'))} ({_v(tech.get('rsi_signal'))}); Bollinger Bands: {_v(tech.get('bb_lower'), '$')}–{_v(tech.get('bb_upper'), '$')} (width {_v(tech.get('bb_width'))}%).",
            f"MA50: {_v(tech.get('ma50'), '$')}; MA200: {_v(tech.get('ma200'), '$')}; golden cross: {_v(tech.get('golden_cross'))}; price vs MA200: {_v(tech.get('current_price_vs_ma200'), '%')}.",
        ]
        assess = d.get("market_structure_assessment") or {}
        if isinstance(assess, str) and assess:
            facts.append(f"Overall market structure assessment: {assess}.")
        elif isinstance(assess, dict) and assess.get("overall_assessment"):
            facts.append(f"Overall market structure assessment: {assess['overall_assessment']}.")

        lines = ["## CONFIRMED MARKET FACTS — include ALL of the following sentences verbatim in market_analysis:"]
        lines += [f"  • {s}" for s in facts]
        return "\n".join(lines) + "\n"

    def _fmt_onchain(d: Optional[dict]) -> str:
        if not d or d.get("error"):
            return "## ONCHAIN DATA\n*Unavailable*\n"
        raw   = d.get("_raw_tool_results") or {}
        flows = raw.get("get_coinglass_exchange_flows") or {}
        etf   = raw.get("get_etf_flows") or {}
        btc   = raw.get("get_coinglass_btc_metrics") or {}
        print(f"DEBUG _fmt_onchain: flows={str(flows)[:200]} etf={str(etf)[:200]}")

        top_ex = flows.get("top_exchanges", [])[:3]
        top_ex_str = ", ".join(f"{e['exchange']} {_v(e.get('balance'))} {coin}" for e in top_ex) if top_ex else "N/A"
        etf_bd = etf.get("etf_breakdown", [])[:4]
        etf_bd_str = ", ".join(f"{e.get('ticker')} {_v(e.get('flow_usd'), 'm')}" for e in etf_bd) if etf_bd else "N/A"

        facts = [
            f"Total {coin} held on exchanges: {_v(flows.get('total_exchange_balance'))} {coin} (24h change: {_v(flows.get('balance_change_1d'))} {coin}; 7d: {_v(flows.get('balance_change_7d'))} {coin}; 30d: {_v(flows.get('balance_change_30d'))} {coin}).",
            f"Exchange flow signal: {_v(flows.get('flow_signal'))}.",
            f"Top exchanges by balance: {top_ex_str}.",
            f"{coin} spot ETF daily flow: {_v(etf.get('daily_flow_usd'), 'm')}; weekly flow: {_v(etf.get('weekly_flow_usd'), 'm')} (signal: {_v(etf.get('flow_signal'))}).",
            f"ETF breakdown: {etf_bd_str}.",
        ]
        if btc.get("nupl_value") is not None:
            facts.append(f"NUPL: {_v(btc.get('nupl_value'))} (zone: {_v(btc.get('nupl_zone'))}, trend: {_v(btc.get('nupl_trend'))}).")
        if btc.get("active_addresses"):
            facts.append(f"Active addresses: {_v(btc.get('active_addresses'))} (7d change: {_v(btc.get('active_addr_7d_change_pct'), '%')}).")

        lines = ["## CONFIRMED ONCHAIN FACTS — include ALL of the following sentences verbatim in onchain_analysis:"]
        lines += [f"  • {s}" for s in facts]
        return "\n".join(lines) + "\n"

    def _fmt_sentiment(d: Optional[dict]) -> str:
        if not d or d.get("error"):
            return "## SENTIMENT DATA\n*Unavailable*\n"
        raw    = d.get("_raw_tool_results") or {}
        x      = raw.get("get_x_sentiment") or d.get("sources", {}).get("x_sentiment") or {}
        fg     = raw.get("get_fear_and_greed") or d.get("sources", {}).get("fear_and_greed") or {}
        tweets = x.get("top_tweets") or d.get("top_tweets") or []
        print(f"DEBUG _fmt_sentiment: score={d.get('sentiment_score')} tweets={x.get('tweet_count_24h')} fg={fg.get('current_value')} fg_label={fg.get('current_label')}")

        facts = [
            f"Sentiment score: {_v(d.get('sentiment_score'))}/100 ({_v(d.get('sentiment_rating') or d.get('rating'))}).",
            f"X/Twitter: {_v(x.get('tweet_count_24h'))} tweets in 24h; total engagement: {_v(x.get('total_engagement'))} ({_v(x.get('total_likes'))} likes, {_v(x.get('total_retweets'))} retweets, {_v(x.get('total_replies'))} replies).",
            f"Fear & Greed Index: {_v(fg.get('current_value'))} ({_v(fg.get('current_label'))}), down from {_v(fg.get('week_ago_value'))} ({_v(fg.get('week_ago_label'))}) a week ago — trend is {_v(fg.get('trend'))}.",
        ]
        themes = d.get("narrative_themes", [])
        if themes:
            facts.append(f"Dominant narrative themes: {'; '.join(str(t) for t in themes[:3])}.")
        for i, t in enumerate(tweets[:2], 1):
            txt = str(t.get("text", ""))[:180].replace("\n", " ")
            facts.append(f"Top tweet #{i} ({_v(t.get('likes'))} likes, {_v(t.get('retweets'))} RT): \"{txt}\"")

        lines = ["## CONFIRMED SENTIMENT FACTS — include ALL of the following sentences verbatim in sentiment_analysis:"]
        lines += [f"  • {s}" for s in facts]
        return "\n".join(lines) + "\n"

    # ── Programmatic analysis builders ──────────────────────────────────────────
    # with_structured_output puts Gemini in function-calling mode where it fills
    # schema fields from general comprehension rather than reading "include verbatim"
    # instructions carefully. These functions build the three data-heavy analysis
    # fields directly from _raw_tool_results, guaranteeing every API number appears.
    # Gemini still owns: rating, scores, price_target, executive_summary,
    # bull_case, bear_case, conclusion, key_risks, comparable_analysis.

    def _build_onchain_text(d: Optional[dict]) -> str:
        raw   = (d or {}).get("_raw_tool_results") or {}
        flows = raw.get("get_coinglass_exchange_flows") or {}
        etf   = raw.get("get_etf_flows") or {}
        sentences: list[str] = []
        if flows.get("total_exchange_balance") is not None:
            total = flows["total_exchange_balance"]
            d1    = flows.get("balance_change_1d") or 0
            d7    = flows.get("balance_change_7d") or 0
            d30   = flows.get("balance_change_30d") or 0
            top   = [
                f"{e['exchange']} {e.get('balance', 0):,.0f} {coin}"
                for e in flows.get("top_exchanges", [])[:3]
            ]
            top_str   = ", ".join(top) if top else "N/A"
            flow_desc = "declining — bullish accumulation" if d30 < 0 else "rising — distribution pressure"
            sentences += [
                f"Total {coin} held on exchanges: {total:,.2f} {coin} "
                f"(24h: {d1:+,.2f} {coin}; 7d: {d7:+,.2f} {coin}; 30d: {d30:+,.2f} {coin}).",
                f"Exchange balance is {flow_desc}.",
                f"Top exchanges by balance: {top_str}.",
            ]
        if etf.get("daily_flow_usd") is not None:
            daily  = etf["daily_flow_usd"] / 1e6
            weekly = (etf.get("weekly_flow_usd") or 0) / 1e6
            sig    = etf.get("flow_signal") or "N/A"
            bd     = [e for e in etf.get("etf_breakdown", []) if (e.get("flow_usd") or 0) != 0][:5]
            bd_str = ", ".join(f"{e.get('ticker')} ${e['flow_usd']/1e6:+.1f}M" for e in bd) or "minimal flows"
            sentences += [
                f"{coin} spot ETF daily flow: ${daily:+.1f}M; weekly flow: ${weekly:+.1f}M (signal: {sig}).",
                f"ETF breakdown: {bd_str}.",
            ]
        return " ".join(sentences) if sentences else "On-chain exchange and ETF data unavailable from configured sources."

    def _build_sentiment_text(d: Optional[dict]) -> str:
        raw      = (d or {}).get("_raw_tool_results") or {}
        x        = raw.get("get_x_sentiment") or (d or {}).get("sources", {}).get("x_sentiment") or {}
        fg       = raw.get("get_fear_and_greed") or (d or {}).get("sources", {}).get("fear_and_greed") or {}
        score    = (d or {}).get("sentiment_score")
        rating_s = (d or {}).get("sentiment_rating") or (d or {}).get("rating") or "N/A"
        themes   = (d or {}).get("narrative_themes") or []
        tweets   = x.get("top_tweets") or (d or {}).get("top_tweets") or []
        sentences: list[str] = []
        if score is not None:
            sentences.append(f"Sentiment score: {score}/100 ({rating_s}).")
        if x.get("tweet_count_24h") is not None:
            sentences.append(
                f"X/Twitter: {x['tweet_count_24h']} tweets in 24h; "
                f"{x.get('total_engagement', 0)} total engagements "
                f"({x.get('total_likes', 0)} likes, "
                f"{x.get('total_retweets', 0)} retweets, "
                f"{x.get('total_replies', 0)} replies)."
            )
        if fg.get("current_value") is not None:
            fv    = fg["current_value"]
            fl    = fg.get("current_label", "N/A")
            wv    = fg.get("week_ago_value") or fv
            wl    = fg.get("week_ago_label", "N/A")
            trend = fg.get("trend", "N/A")
            note  = " Extreme fear has historically signaled potential market bottoms." if fv <= 15 else ""
            sentences.append(
                f"Fear & Greed Index: {fv}/100 ('{fl}'), from {wv} ('{wl}') a week ago — trend {trend}.{note}"
            )
        if themes:
            sentences.append(f"Key narrative themes: {'; '.join(str(t) for t in themes[:3])}.")
        if tweets:
            t0  = tweets[0]
            txt = str(t0.get("text", ""))[:180].replace("\n", " ")
            sentences.append(f"Top post ({t0.get('likes', 0)} likes, {t0.get('retweets', 0)} RT): \"{txt}\"")
        return " ".join(sentences) if sentences else "Sentiment data unavailable."

    def _build_market_text(d: Optional[dict]) -> str:
        raw   = (d or {}).get("_raw_tool_results") or {}
        spot  = raw.get("get_spot_data")            or (d or {}).get("spot")      or {}
        deriv = raw.get("get_derivatives_data")     or (d or {}).get("coinglass") or {}
        liq   = raw.get("get_liquidation_data")     or {}
        taker = raw.get("get_taker_volume")         or {}
        tech  = raw.get("compute_technical_indicators") or (d or {}).get("technicals") or {}
        sentences: list[str] = []
        price = spot.get("price_usd")
        if price is not None:
            p24  = spot.get("price_change_24h") or 0
            p7   = spot.get("price_change_7d")  or 0
            p30  = spot.get("price_change_30d") or 0
            mcap = spot.get("market_cap")        or 0
            rank = spot.get("market_cap_rank")   or "N/A"
            vol  = spot.get("volume_24h")        or 0
            sentences.append(
                f"Spot price: ${price:,.2f} ({p24:+.2f}% 24h, {p7:+.2f}% 7d, {p30:+.2f}% 30d). "
                f"Market cap: ${mcap/1e9:.2f}B (rank #{rank}); 24h volume: ${vol/1e9:.2f}B."
            )
            ath     = spot.get("ath")
            ath_pct = spot.get("ath_change_percentage")
            if ath is not None:
                sentences.append(f"All-time high: ${ath:,.2f} — current price is {ath_pct or 0:+.2f}% from ATH.")
        rsi = tech.get("rsi_14")
        if rsi is not None:
            bb_upper = tech.get("bb_upper") or 0
            bb_lower = tech.get("bb_lower") or 0
            bb_width = tech.get("bb_width") or 0
            ma50     = tech.get("ma50")     or 0
            ma200    = tech.get("ma200")    or 0
            gc       = tech.get("golden_cross", False)
            vs_ma200 = tech.get("current_price_vs_ma200") or 0
            sentences.append(
                f"RSI(14): {rsi:.1f} ({tech.get('rsi_signal', 'N/A')}); "
                f"Bollinger Bands: ${bb_lower:,.2f}–${bb_upper:,.2f} ({bb_width:.2f}% width). "
                f"MA50: ${ma50:,.2f}; MA200: ${ma200:,.2f}; "
                f"golden cross: {'Yes' if gc else 'No'}; price vs MA200: {vs_ma200:+.2f}%."
            )
        fr = deriv.get("funding_rate_pct")
        if fr is not None:
            oi     = deriv.get("open_interest_usd") or 0
            oi_chg = deriv.get("oi_change_4h_pct")  or 0
            sentences.append(
                f"Derivatives: funding rate {fr:.4f}% ({deriv.get('funding_sentiment', 'neutral')}); "
                f"open interest ${oi/1e9:.2f}B ({oi_chg:+.3f}% in 4h)."
            )
        ll = liq.get("long_liquidations_24h_usd")
        sl = liq.get("short_liquidations_24h_usd")
        if ll is not None and sl is not None:
            dom = liq.get("dominant_side", "N/A")
            sentences.append(f"24h liquidations: ${ll/1e6:.1f}M longs vs ${sl/1e6:.1f}M shorts ({dom} dominant).")
        bv = taker.get("taker_buy_volume_usd")
        sv = taker.get("taker_sell_volume_usd")
        if bv is not None and sv is not None:
            bp   = taker.get("buy_ratio_pct")  or 50
            sp_  = taker.get("sell_ratio_pct") or 50
            bias = taker.get("bias", "neutral")
            sentences.append(
                f"Taker volumes: ${bv/1e9:.2f}B buy ({bp:.2f}%) vs ${sv/1e9:.2f}B sell ({sp_:.2f}%) — {bias} bias."
            )
        assess = (d or {}).get("market_structure_assessment") or ""
        if isinstance(assess, str) and assess:
            sentences.append(f"Market structure assessment: {assess}.")
        elif isinstance(assess, dict):
            oa = assess.get("overall_assessment") or ""
            if oa:
                sentences.append(f"Market structure assessment: {oa}.")
        return " ".join(sentences) if sentences else "Market structure data unavailable."

    data_available = sum(1 for d in [onchain, sentiment, market, macro] if d and not d.get("error"))
    confidence = "high" if data_available == 4 else "medium" if data_available >= 2 else "low"

    def _block(label: str, data: Optional[dict]) -> str:
        if not data or data.get("error"):
            return f"## {label}\n*Data unavailable*\n"
        return f"## {label}\n{json.dumps(data, indent=2, default=str)}\n"

    user_msg = dedent(f"""
        # Synthesis Request: {coin}

        Research depth: {plan.get('priority', 'balanced')}
        Focus areas: {', '.join(plan.get('focus_areas', []))}
        Comparables: {', '.join(plan.get('comparables', []))}

        {_fmt_onchain(onchain)}
        {_fmt_sentiment(sentiment)}
        {_fmt_market(market)}
        {_block("MACRO & REGULATORY DATA", macro)}
        {rag_section}

        Generate the full investment report for {coin}.

        Score weights: Onchain 35% | Market 30% | Macro 20% | Sentiment 15%
        Rating rubric: BUY >70 | HOLD 40-70 | SELL 20-40 | AVOID <20
        Confidence: set to "{confidence}" ({data_available}/4 data sources available)
        Language: use "may", "suggests", "historical patterns indicate" — never "will" or "guaranteed"
        price_target_range: current spot price is ${current_price if current_price else "unknown — use general knowledge"}.
          Estimate a reasonable 90-day range anchored to this price. Never return 0 for low or high.
        composite_scores: score ONLY on signals from data that was actually provided.
          STRICT RULE — a missing or unavailable data source is NEUTRAL (contributes 0 weight),
          not negative. Do NOT lower a score because Glassnode, Etherscan, Dune, or any other
          optional API returned an error. Score the available evidence on its own merits.
          Use 50 as a starting baseline; move above/below 50 only when the available data
          gives explicit bullish or bearish signals. Never score below 30 unless live data
          shows a clearly negative signal (e.g. large net exchange inflows, very high funding rate,
          RSI >80 or <25, extreme negative ETF flows).
        IMPORTANT: Every specific number in the ONCHAIN, SENTIMENT, and MARKET sections above is
          CONFIRMED REAL DATA from live APIs. You MUST cite these exact values in the report.
          Do NOT write "unavailable" for any field that has a real value above.
          Do NOT substitute values from your training knowledge — the live data above supersedes
          everything you know. In particular: ATH, price changes, and ETF flows are live and may
          differ significantly from your training cutoff data. Use only the [LIVE] values provided.
    """).strip()

    system_prompt = dedent("""
        You are a senior crypto analyst at a Tier-1 institutional fund.
        Write a rigorous, data-driven investment report. Requirements:
        - Every claim must cite specific numbers from the data above — NO vague language like
          "significantly below its ATH" without stating the actual ATH price and % distance
        - Bull case AND bear case must be proportionally covered
        - Composite scores must reflect ONLY the signals present in the data provided.
          Missing API sources (Glassnode, Etherscan, Dune, etc.) are EXCLUDED from scoring —
          they do not count as negative evidence. Score only what you can see.
        - comparable_analysis: reference past report context if available,
          otherwise compare to the top 2-3 assets in the same category
        - Never state price predictions as fact — use calibrated ranges

        MANDATORY number citations per section:
        - onchain_analysis: cite total_exchange_balance (exact BTC), balance_change_1d/7d/30d,
          ETF daily_flow_usd and weekly_flow_usd (exact $ amounts), top 3 exchange names/balances
        - sentiment_analysis: cite sentiment_score, tweet_count_24h, total_engagement, F&G current
          value AND label, week_ago value, trend direction, and quote at least one top tweet text
        - market_analysis: cite spot price, ATH (exact $ from LIVE data), % below ATH, MA50, MA200,
          RSI value, BB upper/lower, funding_rate_pct, OI, long_liquidations_24h_usd,
          short_liquidations_24h_usd, taker buy/sell volumes and bias
        - macro_analysis: cite F&G value, BTC dominance %, Fed rate, 30d price change
    """).strip()

    # Step 5: call Gemini with structured output (sync → thread)
    try:
        result: SynthesisReport = await asyncio.to_thread(
            _llm().with_structured_output(SynthesisReport).invoke,
            [SystemMessage(content=system_prompt), HumanMessage(content=user_msg)],
        )
        report_dict = result.model_dump()
    except Exception as exc:
        # Fallback: unstructured call, return partial JSON
        report_dict = {
            "executive_summary": f"Synthesis failed: {exc}",
            "rating": "HOLD",
            "confidence": "low",
            "error": str(exc),
        }

    # Override the three data-heavy analysis fields with programmatically-built text.
    # Gemini owns everything else: rating, scores, price_target, executive_summary,
    # bull_case, bear_case, conclusion, key_risks, comparable_analysis.
    report_dict["onchain_analysis"]   = _build_onchain_text(onchain)
    report_dict["sentiment_analysis"] = _build_sentiment_text(sentiment)
    report_dict["market_analysis"]    = _build_market_text(market)
    print(f"DEBUG synthesis override: onchain={report_dict['onchain_analysis'][:80]!r}")
    print(f"DEBUG synthesis override: market={report_dict['market_analysis'][:80]!r}")

    # Log if price target is missing despite having current price
    pt = report_dict.get("price_target_range", {})
    if not pt.get("low") or not pt.get("high"):
        print(f"WARNING: LLM returned empty price_target_range={pt} despite current_price={current_price}")

    # Step 6: store in pgvector (Layer 3 — fire and forget)
    try:
        store_report(coin, report_dict, embedding)
    except Exception:
        pass

    return json.dumps(report_dict)
