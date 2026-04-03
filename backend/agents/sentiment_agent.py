"""
Sentiment Agent
===============
Aggregates sentiment from multiple sources:
- X (Twitter) social volume and top posts
- Fear & Greed Index
"""

import os
import httpx
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
async def get_x_sentiment(coin: str) -> dict:
    """
    Fetch recent X (Twitter) posts about the coin via API v2.
    Returns tweet volume, engagement metrics, and top posts by likes.
    """
    bearer_token = os.getenv("TWITTER_BEARER_TOKEN", "")
    if not bearer_token:
        return {"error": "Twitter bearer token not configured — X sentiment unavailable"}

    query = f"(#{coin} OR ${coin}) crypto -is:retweet lang:en"
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(
            "https://api.twitter.com/2/tweets/search/recent",
            headers={"Authorization": f"Bearer {bearer_token}"},
            params={
                "query": query,
                "max_results": 100,
                "tweet.fields": "public_metrics,created_at,text",
            },
        )
        if r.status_code != 200:
            print(f"WARNING: X API returned {r.status_code} — {r.text[:200]}")
            r.raise_for_status()
        tweets = r.json().get("data", [])

    if not tweets:
        return {"error": "No recent tweets found for this asset"}

    total_likes     = sum(t.get("public_metrics", {}).get("like_count",    0) for t in tweets)
    total_retweets  = sum(t.get("public_metrics", {}).get("retweet_count", 0) for t in tweets)
    total_replies   = sum(t.get("public_metrics", {}).get("reply_count",   0) for t in tweets)

    top = sorted(tweets, key=lambda t: t.get("public_metrics", {}).get("like_count", 0), reverse=True)

    return {
        "tweet_count_24h":   len(tweets),
        "total_likes":       total_likes,
        "total_retweets":    total_retweets,
        "total_replies":     total_replies,
        "total_engagement":  total_likes + total_retweets + total_replies,
        "top_tweets": [
            {
                "text":     t["text"][:280],
                "likes":    t.get("public_metrics", {}).get("like_count",    0),
                "retweets": t.get("public_metrics", {}).get("retweet_count", 0),
            }
            for t in top[:5]
        ],
    }


@tool
async def get_fear_and_greed() -> dict:
    """Fetch the Crypto Fear & Greed Index — 0 = extreme fear, 100 = extreme greed."""
    async with httpx.AsyncClient() as client:
        r = await client.get("https://api.alternative.me/fng/?limit=7")
        data = r.json().get("data", [])
        current = data[0] if data else {}
        week_ago = data[-1] if len(data) >= 7 else data[0] if data else {}
        return {
            "current_value": int(current.get("value", 50)),
            "current_label": current.get("value_classification"),
            "week_ago_value": int(week_ago.get("value", 50)),
            "week_ago_label": week_ago.get("value_classification"),
            "trend": "improving" if int(current.get("value", 50)) > int(week_ago.get("value", 50)) else "deteriorating",
        }


# ---------------------------------------------------------------------------
# Response parser (Gemini returns content as list[dict] content blocks)
# ---------------------------------------------------------------------------

def _parse_agent_response(content: Any, agent_name: str) -> dict:
    import json, re
    if isinstance(content, list):
        text = "".join(
            block.get("text", "") for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    else:
        text = str(content)

    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        result = json.loads(text)
        print(f"DEBUG: {agent_name} parsed OK, keys={list(result.keys())}")
        return result
    except json.JSONDecodeError as e:
        # LLM sometimes emits invalid escape sequences (e.g. \s, \: in tweet URLs).
        # Fix by escaping any backslash not followed by a valid JSON escape character.
        try:
            cleaned = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)
            result = json.loads(cleaned)
            print(f"DEBUG: {agent_name} parsed OK after escape fix, keys={list(result.keys())}")
            return result
        except Exception:
            pass
        print(f"DEBUG: {agent_name} parse FAILED — {e}")
        return {"raw": text[:500], "parse_error": True}


# ---------------------------------------------------------------------------
# Agent runner
# ---------------------------------------------------------------------------

SENTIMENT_TOOLS = [get_x_sentiment, get_fear_and_greed]

SYSTEM_PROMPT = """You are a crypto sentiment analyst. You aggregate and interpret signals
from social media and market psychology indicators.

Always fetch:
1. X (Twitter) social volume and top posts via get_x_sentiment
2. Fear & Greed index and trend via get_fear_and_greed

Produce a composite SENTIMENT SCORE from 0-100 and a clear bullish/neutral/bearish rating.
Identify the top 3 narrative themes (what people are talking about).
Display data from every metric fetched, including raw values (counts, scores, timestamps).
For each of the top 3 tweets, include the text, like count, retweet count, and a one-word sentiment label (bullish/bearish/neutral).
Return structured JSON with a 'sources' key containing the raw data per tool.
"""


async def run_sentiment_agent(coin: str, plan: dict) -> dict[str, Any]:
    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    ).bind_tools(SENTIMENT_TOOLS)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=f"Gather comprehensive sentiment data for {coin}. "
                    f"Known risks to watch: {plan.get('known_risks', [])}. "
                    f"You MUST call the available tools to fetch real data before returning. "
                    f"Return a structured JSON with sentiment score, key themes, and source breakdown."
        )
    ]

    raw_tool_results: dict = {}
    while True:
        response = await llm.ainvoke(messages)
        messages.append(response)
        tool_calls = response.tool_calls if hasattr(response, "tool_calls") else []
        if not tool_calls:
            break

        from langchain_core.messages import ToolMessage
        for tc in tool_calls:
            print(f"INFO: sentiment_agent calling tool={tc['name']} args={tc['args']}")
            tool_fn = {t.name: t for t in SENTIMENT_TOOLS}.get(tc["name"])
            if tool_fn:
                try:
                    result = await tool_fn.ainvoke(tc["args"])
                except Exception as e:
                    result = {"error": str(e)}
                print(f"INFO: sentiment_agent tool={tc['name']} result={str(result)[:200]}")
                raw_tool_results[tc["name"]] = result
                messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    parsed = _parse_agent_response(response.content, "sentiment_agent")
    parsed["_raw_tool_results"] = raw_tool_results
    return parsed
