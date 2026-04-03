"""
Renderer
========
Converts the approved draft_report JSON string into:
  1. final_report dict  — structured, ready for the web frontend
  2. PDF file           — saved to /tmp/reports/{coin}_{uuid}.pdf

Called by graph.py's renderer_node as:
    final_report, pdf_path = await render_report(coin, draft, onchain, sentiment, market, macro)

PDF generation uses WeasyPrint. If WeasyPrint is not installed or fails,
pdf_path is returned as None — the report dict is always returned.

Rating badge colors (AGENTS.md spec):
  BUY   → #10b981 (green)
  HOLD  → #3b82f6 (blue)
  SELL  → #f59e0b (amber)
  AVOID → #ef4444 (red)
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ── Rating config ─────────────────────────────────────────────────────────────

_RATING_COLOR = {
    "BUY":   "#10b981",
    "HOLD":  "#3b82f6",
    "SELL":  "#f59e0b",
    "AVOID": "#ef4444",
}


# ── Main entrypoint ───────────────────────────────────────────────────────────

async def render_report(
    coin: str,
    draft: str,
    onchain: dict[str, Any],
    sentiment: dict[str, Any],
    market: dict[str, Any],
    macro: dict[str, Any],
) -> tuple[dict[str, Any], str | None]:
    """
    Parse draft JSON → enrich with metadata → generate PDF.
    Returns (final_report_dict, pdf_path_or_None).
    """
    # Parse draft — strip markdown fences if present (revision_node returns raw text)
    try:
        text = (draft or "").strip()
        if text.startswith("```"):
            # Strip ```json\n...\n``` wrapper
            text = text.split("```", 2)[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        report = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        report = {"raw_draft": str(draft)}

    # Enrich with metadata
    report["coin"]         = coin.upper()
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    report["data_sources"] = {
        "onchain_available":   bool(onchain and not onchain.get("error")),
        "sentiment_available": bool(sentiment and not sentiment.get("error")),
        "market_available":    bool(market and not market.get("error")),
        "macro_available":     bool(macro and not macro.get("error")),
    }

    # Attempt PDF generation
    pdf_path = await _generate_pdf(coin, report)

    return report, pdf_path


# ── PDF generation ────────────────────────────────────────────────────────────

async def _generate_pdf(coin: str, report: dict[str, Any]) -> str | None:
    """
    Render HTML template → PDF via WeasyPrint.
    Returns file path on success, None if WeasyPrint unavailable or generation fails.
    """
    try:
        from weasyprint import HTML  # optional dependency
    except Exception:
        return None

    try:
        output_dir = Path("/tmp/reports")
        output_dir.mkdir(parents=True, exist_ok=True)

        report_id = str(uuid.uuid4())[:8]
        pdf_path  = output_dir / f"{coin.upper()}_{report_id}.pdf"

        html_str = _build_html(coin, report)
        HTML(string=html_str).write_pdf(str(pdf_path))
        return str(pdf_path)
    except Exception:
        return None


def _fmt_price(v: Any) -> str:
    try:
        return f"${float(v):,.0f}"
    except (TypeError, ValueError):
        return "N/A"


def _build_html(coin: str, report: dict[str, Any]) -> str:
    """Build the full HTML document that WeasyPrint converts to PDF."""
    rating       = report.get("rating", "HOLD")
    badge_color  = _RATING_COLOR.get(rating, "#6b7280")
    scores       = report.get("composite_scores") or {}
    price_target = report.get("price_target_range") or {}
    generated_at = report.get("generated_at", "")[:10]

    def score_bar(label: str, value: int | None) -> str:
        v = value or 0
        color = "#10b981" if v >= 70 else "#f59e0b" if v >= 40 else "#ef4444"
        return f"""
        <div class="score-row">
            <span class="score-label">{label}</span>
            <div class="score-track">
                <div class="score-fill" style="width:{v}%; background:{color};"></div>
            </div>
            <span class="score-value">{v}</span>
        </div>"""

    def section(title: str, content: str) -> str:
        if not content:
            return ""
        return f"""
        <div class="section">
            <h2>{title}</h2>
            <p>{content}</p>
        </div>"""

    risks_html = "".join(
        f"<li>{r}</li>" for r in (report.get("key_risks") or [])
    )

    pt_low  = price_target.get("low",  "N/A")
    pt_high = price_target.get("high", "N/A")
    pt_tf   = price_target.get("timeframe", "90d")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Georgia', serif;
    background: #0a0f1e;
    color: #e2e8f0;
    padding: 40px;
    font-size: 13px;
    line-height: 1.7;
  }}
  h1 {{ font-size: 28px; color: #f1f5f9; letter-spacing: -0.5px; }}
  h2 {{ font-size: 15px; color: #93c5fd; text-transform: uppercase;
       letter-spacing: 1px; margin: 24px 0 8px; border-bottom: 1px solid #1e3a5f;
       padding-bottom: 4px; }}
  p  {{ color: #cbd5e1; margin-bottom: 12px; }}

  .cover {{
    border-bottom: 2px solid #1e40af;
    padding-bottom: 24px;
    margin-bottom: 32px;
  }}
  .badge {{
    display: inline-block;
    background: {badge_color};
    color: #fff;
    font-family: monospace;
    font-size: 20px;
    font-weight: bold;
    padding: 6px 20px;
    border-radius: 4px;
    margin: 12px 0;
  }}
  .meta {{ color: #64748b; font-size: 11px; margin-top: 8px; }}

  .exec-box {{
    background: #0f172a;
    border-left: 4px solid #3b82f6;
    padding: 16px 20px;
    margin: 20px 0;
    border-radius: 0 4px 4px 0;
  }}
  .exec-box p {{ color: #e2e8f0; font-size: 14px; }}

  .price-target {{
    display: inline-block;
    background: #1e293b;
    border: 1px solid #334155;
    padding: 8px 16px;
    border-radius: 4px;
    font-family: monospace;
    color: #a5b4fc;
    margin: 8px 0;
  }}

  .scores {{ margin: 16px 0; }}
  .score-row {{
    display: flex;
    align-items: center;
    margin: 6px 0;
    gap: 10px;
  }}
  .score-label {{
    width: 140px;
    font-family: monospace;
    font-size: 11px;
    color: #94a3b8;
  }}
  .score-track {{
    flex: 1;
    height: 10px;
    background: #1e293b;
    border-radius: 5px;
    overflow: hidden;
  }}
  .score-fill {{
    height: 100%;
    border-radius: 5px;
    transition: width 0.3s;
  }}
  .score-value {{
    width: 30px;
    text-align: right;
    font-family: monospace;
    font-size: 11px;
    color: #e2e8f0;
  }}

  .section {{ margin: 20px 0; page-break-inside: avoid; }}
  .two-col {{ display: flex; gap: 24px; }}
  .two-col > div {{ flex: 1; }}

  ul.risks {{ padding-left: 20px; color: #fca5a5; }}
  ul.risks li {{ margin: 4px 0; }}

  .confidence {{
    display: inline-block;
    font-family: monospace;
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 3px;
    background: #1e293b;
    color: #94a3b8;
    margin-left: 8px;
  }}

  .footer {{
    margin-top: 40px;
    padding-top: 12px;
    border-top: 1px solid #1e293b;
    color: #475569;
    font-size: 10px;
    font-style: italic;
  }}
  @page {{
    size: A4;
    margin: 20mm;
    @bottom-center {{
      content: "This report is generated by AI and does not constitute financial advice.";
      font-size: 9px;
      color: #475569;
    }}
  }}
</style>
</head>
<body>

<!-- Cover -->
<div class="cover">
  <h1>{coin.upper()} Research Report</h1>
  <div class="badge">{rating}</div>
  <span class="confidence">Confidence: {report.get('confidence', 'N/A')}</span>
  <div class="price-target">
    Price Target ({pt_tf}): {_fmt_price(pt_low)} – {_fmt_price(pt_high)}
  </div>
  <div class="meta">
    Generated: {generated_at} &nbsp;|&nbsp; CONFIDENTIAL — FOR INSTITUTIONAL USE
  </div>
</div>

<!-- Executive Summary -->
<div class="exec-box">
  <h2>Executive Summary</h2>
  <p>{report.get('executive_summary', '')}</p>
</div>

<!-- Composite Scores -->
<div class="section">
  <h2>Composite Scores</h2>
  <div class="scores">
    {score_bar("Onchain Health",   scores.get('onchain_health'))}
    {score_bar("Sentiment",        scores.get('sentiment'))}
    {score_bar("Market Structure", scores.get('market_structure'))}
    {score_bar("Macro",            scores.get('macro'))}
    {score_bar("Overall",          scores.get('overall'))}
  </div>
</div>

<!-- Analysis sections -->
{section("Onchain Analysis",    report.get('onchain_analysis',    ''))}
{section("Sentiment Analysis",  report.get('sentiment_analysis',  ''))}
{section("Market Analysis",     report.get('market_analysis',     ''))}
{section("Macro Analysis",      report.get('macro_analysis',      ''))}
{section("Comparable Analysis", report.get('comparable_analysis', ''))}

<!-- Bull / Bear -->
<div class="two-col">
  <div>{section("Bull Case", report.get('bull_case', ''))}</div>
  <div>{section("Bear Case", report.get('bear_case', ''))}</div>
</div>

<!-- Key Risks -->
<div class="section">
  <h2>Key Risks</h2>
  <ul class="risks">{risks_html}</ul>
</div>

{section("Conclusion", report.get('conclusion', ''))}

<div class="footer">
  This report is generated by AI and does not constitute financial advice.
  Past performance is not indicative of future results.
  {coin.upper()} Report · {generated_at}
</div>

</body>
</html>"""
