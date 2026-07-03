"""Generate an HTML report from TradingAgents JSON output."""

import json
import sys
import os

import markdown
from markdown.extensions.tables import TableExtension

def md_to_html(text: str) -> str:
    """Convert Markdown text to HTML."""
    return markdown.markdown(
        text,
        extensions=[TableExtension(), "fenced_code", "nl2br"],
    )


def generate_html(data: dict) -> str:
    ticker = data["company_of_interest"]
    date = data["trade_date"]
    ids = data["investment_debate_state"]
    rds = data["risk_debate_state"]

    sections = [
        ("final", "Final Decision", data["final_trade_decision"]),
        ("market", "Market & Technical Analysis", data["market_report"]),
        ("sentiment", "Sentiment Analysis", data["sentiment_report"]),
        ("news", "News & Macro Analysis", data["news_report"]),
        ("fundamentals", "Fundamentals Analysis", data["fundamentals_report"]),
        ("debate", "Investment Debate", None),
        ("trader", "Trader Decision", data["trader_investment_decision"]),
        ("risk", "Risk Debate", None),
        ("plan", "Investment Plan", data["investment_plan"]),
    ]

    nav_items = "".join(
        f'<a href="#{sid}" class="nav-item">{title}</a>' for sid, title, _ in sections
    )

    section_html_parts = []
    for sid, title, content in sections:
        if sid == "debate":
            bull_html = md_to_html(ids.get("bull_history", ""))
            bear_html = md_to_html(ids.get("bear_history", ""))
            judge_html = md_to_html(ids.get("judge_decision", ""))
            section_html_parts.append(f"""
            <section id="{sid}" class="report-section">
                <h2>{title}</h2>
                <div class="debate-container">
                    <div class="debate-panel bull">
                        <div class="debate-label bull-label">BULL</div>
                        {bull_html}
                    </div>
                    <div class="debate-panel bear">
                        <div class="debate-label bear-label">BEAR</div>
                        {bear_html}
                    </div>
                </div>
                <div class="judge-panel">
                    <div class="debate-label judge-label">JUDGE</div>
                    {judge_html}
                </div>
            </section>""")
        elif sid == "risk":
            agg_html = md_to_html(rds.get("aggressive_history", ""))
            con_html = md_to_html(rds.get("conservative_history", ""))
            neu_html = md_to_html(rds.get("neutral_history", ""))
            rjudge_html = md_to_html(rds.get("judge_decision", ""))
            section_html_parts.append(f"""
            <section id="{sid}" class="report-section">
                <h2>{title}</h2>
                <div class="risk-container">
                    <div class="debate-panel aggressive">
                        <div class="debate-label agg-label">AGGRESSIVE</div>
                        {agg_html}
                    </div>
                    <div class="debate-panel conservative">
                        <div class="debate-label con-label">CONSERVATIVE</div>
                        {con_html}
                    </div>
                    <div class="debate-panel neutral">
                        <div class="debate-label neu-label">NEUTRAL</div>
                        {neu_html}
                    </div>
                </div>
                <div class="judge-panel">
                    <div class="debate-label judge-label">PORTFOLIO MANAGER</div>
                    {rjudge_html}
                </div>
            </section>""")
        elif sid == "final":
            body = md_to_html(content)
            section_html_parts.append(f"""
            <section id="{sid}" class="report-section final-section">
                <h2>{title}</h2>
                {body}
            </section>""")
        else:
            body = md_to_html(content)
            section_html_parts.append(f"""
            <section id="{sid}" class="report-section">
                <h2>{title}</h2>
                {body}
            </section>""")

    all_sections = "\n".join(section_html_parts)

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{ticker} Analysis Report - {date}</title>
<style>
:root {{
  --bg: #0f1117;
  --surface: #1a1d28;
  --surface2: #232736;
  --border: #2d3148;
  --text: #e1e4ed;
  --text-muted: #8b90a5;
  --accent: #6c8aff;
  --green: #22c55e;
  --red: #ef4444;
  --orange: #f59e0b;
  --blue: #3b82f6;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC',
               'Microsoft YaHei', sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.7;
  font-size: 15px;
}}
.header {{
  background: linear-gradient(135deg, #1e2235 0%, #141728 100%);
  border-bottom: 1px solid var(--border);
  padding: 32px 0;
  text-align: center;
}}
.header h1 {{
  font-size: 2.2em;
  font-weight: 700;
  letter-spacing: 1px;
}}
.header .ticker {{ color: var(--accent); }}
.header .date {{ color: var(--text-muted); font-size: 0.9em; margin-top: 6px; }}
.header .badge {{
  display: inline-block;
  margin-top: 12px;
  padding: 6px 20px;
  border-radius: 6px;
  font-weight: 700;
  font-size: 1.1em;
  text-transform: uppercase;
}}
.badge-sell {{ background: rgba(239,68,68,.15); color: var(--red); border: 1px solid var(--red); }}
.badge-buy  {{ background: rgba(34,197,94,.15); color: var(--green); border: 1px solid var(--green); }}
.badge-hold {{ background: rgba(245,158,11,.15); color: var(--orange); border: 1px solid var(--orange); }}

nav {{
  position: sticky; top: 0; z-index: 100;
  background: rgba(15,17,23,.92);
  backdrop-filter: blur(12px);
  border-bottom: 1px solid var(--border);
  display: flex; gap: 0; overflow-x: auto;
  padding: 0 24px;
}}
.nav-item {{
  padding: 12px 18px;
  color: var(--text-muted);
  text-decoration: none;
  font-size: 13px;
  font-weight: 500;
  white-space: nowrap;
  border-bottom: 2px solid transparent;
  transition: all .2s;
}}
.nav-item:hover {{ color: var(--text); border-color: var(--accent); }}

.container {{ max-width: 1100px; margin: 0 auto; padding: 24px 20px 80px; }}

.report-section {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 32px 36px;
  margin-bottom: 24px;
}}
.report-section h2 {{
  font-size: 1.4em;
  margin-bottom: 20px;
  padding-bottom: 12px;
  border-bottom: 2px solid var(--accent);
  color: var(--accent);
}}
.report-section h1 {{ font-size: 1.3em; margin: 20px 0 12px; color: var(--text); }}
.report-section h3 {{ font-size: 1.1em; margin: 16px 0 8px; color: #c0c5db; }}
.report-section h4 {{ font-size: 1em; margin: 12px 0 6px; color: #a8adc4; }}

.report-section p {{ margin: 8px 0; }}
.report-section ul, .report-section ol {{ margin: 8px 0 8px 24px; }}
.report-section li {{ margin: 4px 0; }}
.report-section strong {{ color: #fff; }}
.report-section hr {{ border: none; border-top: 1px solid var(--border); margin: 20px 0; }}

table {{
  width: 100%;
  border-collapse: collapse;
  margin: 12px 0;
  font-size: 14px;
}}
th {{
  background: var(--surface2);
  padding: 10px 14px;
  text-align: left;
  font-weight: 600;
  color: var(--accent);
  border-bottom: 2px solid var(--border);
}}
td {{
  padding: 8px 14px;
  border-bottom: 1px solid var(--border);
}}
tr:hover td {{ background: rgba(108,138,255,.04); }}

.final-section {{
  border-color: var(--red);
  background: linear-gradient(135deg, rgba(239,68,68,.06), var(--surface));
}}

.debate-container {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  margin-bottom: 16px;
}}
.risk-container {{
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 16px;
  margin-bottom: 16px;
}}
@media (max-width: 900px) {{
  .debate-container, .risk-container {{ grid-template-columns: 1fr; }}
}}
.debate-panel {{
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 20px 24px;
  max-height: 600px;
  overflow-y: auto;
  font-size: 14px;
}}
.debate-panel h1 {{ font-size: 1.1em; }}
.debate-panel h2 {{ font-size: 1em; border: none; padding: 0; margin: 14px 0 8px; }}
.debate-panel h3 {{ font-size: 0.95em; }}

.debate-label {{
  display: inline-block;
  padding: 3px 12px;
  border-radius: 4px;
  font-weight: 700;
  font-size: 12px;
  letter-spacing: 1px;
  margin-bottom: 12px;
}}
.bull-label {{ background: rgba(34,197,94,.15); color: var(--green); }}
.bear-label {{ background: rgba(239,68,68,.15); color: var(--red); }}
.judge-label {{ background: rgba(108,138,255,.15); color: var(--accent); }}
.agg-label {{ background: rgba(239,68,68,.15); color: var(--red); }}
.con-label {{ background: rgba(59,130,246,.15); color: var(--blue); }}
.neu-label {{ background: rgba(245,158,11,.15); color: var(--orange); }}

.judge-panel {{
  background: var(--surface2);
  border: 1px solid var(--accent);
  border-radius: 8px;
  padding: 20px 24px;
}}
.judge-panel h1 {{ font-size: 1.1em; }}
.judge-panel h2 {{ font-size: 1em; border: none; padding: 0; margin: 14px 0 8px; }}

.debate-panel::-webkit-scrollbar {{ width: 6px; }}
.debate-panel::-webkit-scrollbar-track {{ background: transparent; }}
.debate-panel::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 3px; }}

.footer {{
  text-align: center;
  padding: 24px;
  color: var(--text-muted);
  font-size: 12px;
  border-top: 1px solid var(--border);
}}
</style>
</head>
<body>

<div class="header">
  <h1><span class="ticker">{ticker}</span> Analysis Report</h1>
  <div class="date">Analysis Date: {date} | Generated by TradingAgents</div>
  <div class="badge badge-sell">SELL / Underweight</div>
</div>

<nav>{nav_items}</nav>

<div class="container">
{all_sections}
</div>

<div class="footer">
  Generated by TradingAgents Multi-Agent Framework | Claude Opus 4.6 |
  Data: Sina Finance + Alpha Vantage + FRED
</div>

</body>
</html>"""


if __name__ == "__main__":
    import os
    from tradingagents.default_config import DEFAULT_CONFIG
    logs_dir = DEFAULT_CONFIG["results_dir"]
    src = os.path.join(logs_dir, "BABA", "TradingAgentsStrategy_logs", "full_states_log_2026-06-20.json")
    dst = os.path.join(logs_dir, "BABA", "TradingAgentsStrategy_logs", "BABA_report_2026-06-20.html")

    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)

    html = generate_html(data)

    with open(dst, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Report generated: {dst}")
