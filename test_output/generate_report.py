"""Generate an HTML report from the analysis results and data bundle."""
import json
import sys
from pathlib import Path
from datetime import datetime

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def escape_html(text):
    if not text:
        return ""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def md_to_html_basic(text):
    """Very basic markdown to HTML for tables and headers."""
    if not text:
        return "<p><em>No data</em></p>"

    lines = text.split("\n")
    result = []
    in_table = False

    for line in lines:
        stripped = line.strip()

        # Table rows
        if "|" in stripped and stripped.startswith("|"):
            cells = [c.strip() for c in stripped.split("|")[1:-1]]
            if all(set(c) <= set("-: ") for c in cells):
                continue  # skip separator row
            if not in_table:
                result.append("<table class='data-table'>")
                tag = "th"
                in_table = True
            else:
                tag = "td"
            row = "".join(f"<{tag}>{escape_html(c)}</{tag}>" for c in cells)
            result.append(f"<tr>{row}</tr>")
            continue
        else:
            if in_table:
                result.append("</table>")
                in_table = False

        if stripped.startswith("### "):
            result.append(f"<h4>{escape_html(stripped[4:])}</h4>")
        elif stripped.startswith("## "):
            result.append(f"<h3>{escape_html(stripped[3:])}</h3>")
        elif stripped.startswith("# "):
            result.append(f"<h2>{escape_html(stripped[2:])}</h2>")
        elif stripped.startswith("- "):
            result.append(f"<li>{escape_html(stripped[2:])}</li>")
        elif stripped.startswith("**") and stripped.endswith("**"):
            result.append(f"<p><strong>{escape_html(stripped[2:-2])}</strong></p>")
        elif stripped:
            # Handle inline bold
            import re
            html_line = escape_html(stripped)
            html_line = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html_line)
            result.append(f"<p>{html_line}</p>")

    if in_table:
        result.append("</table>")

    return "\n".join(result)

def build_data_source_section(bundle):
    """Build the data source verification section."""
    sections = []

    # Market Data
    if bundle.get("market"):
        m = bundle["market"]
        sections.append("<h3>1. 市场行情数据 (Market Data)</h3>")
        sections.append("<h4>OHLCV 价格数据</h4>")
        sections.append(f"<pre class='data-raw'>{escape_html(m.get('stock_data', ''))}</pre>")

        sections.append("<h4>技术指标最新值</h4>")
        sections.append("<table class='data-table'><tr><th>指标</th><th>最新值</th></tr>")
        if m.get("indicators"):
            for name, data in m["indicators"].items():
                # Extract latest value from the indicator text
                lines = data.strip().split("\n")
                latest = "N/A"
                for line in lines[2:]:  # skip header lines
                    if line.strip() and ":" in line and "N/A" not in line:
                        parts = line.strip().split(":")
                        if len(parts) >= 2:
                            latest = parts[-1].strip()
                            break
                sections.append(f"<tr><td>{escape_html(name)}</td><td>{escape_html(latest)}</td></tr>")
        sections.append("</table>")

        sections.append("<h4>验证快照 (Verified Snapshot)</h4>")
        sections.append(md_to_html_basic(m.get("verified_snapshot", "")))

    # Sentiment
    if bundle.get("sentiment"):
        s = bundle["sentiment"]
        sections.append("<h3>2. 情绪数据 (Sentiment Data)</h3>")
        sections.append("<table class='data-table'><tr><th>数据源</th><th>状态</th><th>内容预览</th></tr>")
        for field, label in [("ticker_news", "新闻情绪"), ("stocktwits", "StockTwits"), ("reddit", "Reddit")]:
            val = s.get(field, "")
            status = "❌ 不可用" if val.startswith("<") else f"✅ {len(val)} chars"
            preview = val[:200] if val else ""
            sections.append(f"<tr><td>{label}</td><td>{status}</td><td><code>{escape_html(preview)}</code></td></tr>")
        sections.append("</table>")

    # News
    if bundle.get("news"):
        n = bundle["news"]
        sections.append("<h3>3. 新闻数据 (News Data)</h3>")
        sections.append("<table class='data-table'><tr><th>数据源</th><th>状态</th><th>数据量</th></tr>")
        for field, label in [("ticker_news", "个股新闻"), ("global_news", "全球新闻"), ("insider_transactions", "内部交易")]:
            val = n.get(field, "")
            status = "✅ 有数据" if len(val) > 50 else "⚠️ 数据较少"
            sections.append(f"<tr><td>{label}</td><td>{status}</td><td>{len(val)} chars</td></tr>")
        sections.append("</table>")

        # Macro indicators
        sections.append("<h4>宏观经济指标 (FRED)</h4>")
        sections.append("<table class='data-table'><tr><th>指标</th><th>状态</th><th>最新值预览</th></tr>")
        if n.get("macro_indicators"):
            for name, data in n["macro_indicators"].items():
                status = "❌" if data.startswith("<unavailable") else "✅"
                # Extract latest value
                latest = ""
                for line in data.split("\n"):
                    if "**Latest:**" in line or "Latest:" in line:
                        latest = line.split("Latest:")[-1].strip().rstrip("*")
                        break
                sections.append(f"<tr><td>{escape_html(name)}</td><td>{status}</td><td>{escape_html(latest[:80])}</td></tr>")
        sections.append("</table>")

        # Prediction markets
        sections.append("<h4>预测市场 (Polymarket)</h4>")
        sections.append("<table class='data-table'><tr><th>查询主题</th><th>状态</th></tr>")
        if n.get("prediction_markets"):
            for topic, data in n["prediction_markets"].items():
                status = "❌ 不可用 (网络超时)" if "unavailable" in data.lower() else "✅ 有数据"
                sections.append(f"<tr><td>{escape_html(topic)}</td><td>{status}</td></tr>")
        sections.append("</table>")

    # Fundamentals
    if bundle.get("fundamentals"):
        f = bundle["fundamentals"]
        sections.append("<h3>4. 基本面数据 (Fundamentals)</h3>")

        # Overview
        try:
            overview = json.loads(f.get("overview", "{}"))
            sections.append("<h4>公司概况</h4>")
            sections.append("<table class='data-table'><tr><th>字段</th><th>值</th></tr>")
            key_fields = ["Symbol", "Name", "Exchange", "Currency", "Country", "Sector", "Industry",
                          "MarketCapitalization", "PERatio", "DividendYield", "EPS", "52WeekHigh", "52WeekLow"]
            for k in key_fields:
                if k in overview:
                    sections.append(f"<tr><td>{escape_html(k)}</td><td>{escape_html(str(overview[k]))}</td></tr>")
            sections.append("</table>")
        except json.JSONDecodeError:
            sections.append(f"<pre>{escape_html(f.get('overview', '')[:500])}</pre>")

        # Financial statements summary
        sections.append("<h4>财务报表数据量</h4>")
        sections.append("<table class='data-table'><tr><th>报表</th><th>数据量 (chars)</th><th>状态</th></tr>")
        for field, label in [
            ("balance_sheet_quarterly", "资产负债表(季度)"), ("balance_sheet_annual", "资产负债表(年度)"),
            ("cashflow_quarterly", "现金流量表(季度)"), ("cashflow_annual", "现金流量表(年度)"),
            ("income_quarterly", "利润表(季度)"), ("income_annual", "利润表(年度)"),
        ]:
            val = f.get(field, "")
            status = "❌" if val.startswith("<unavailable") else "✅"
            sections.append(f"<tr><td>{label}</td><td>{len(val):,}</td><td>{status}</td></tr>")
        sections.append("</table>")

    return "\n".join(sections)

def generate_html(state, bundle, output_path):
    ticker = state.get("company_of_interest", "BABA")
    trade_date = state.get("trade_date", "2025-06-20")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    data_source_html = build_data_source_section(bundle)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TradingAgents 分析报告 - {escape_html(ticker)} ({escape_html(trade_date)})</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background: #f5f7fa; color: #1a1a2e; line-height: 1.6; }}
.container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
.header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
           color: white; padding: 40px; border-radius: 12px; margin-bottom: 24px;
           text-align: center; }}
.header h1 {{ font-size: 28px; margin-bottom: 8px; }}
.header .meta {{ font-size: 14px; opacity: 0.9; }}
.section {{ background: white; border-radius: 12px; padding: 24px; margin-bottom: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
.section h2 {{ font-size: 20px; color: #333; margin-bottom: 16px; padding-bottom: 8px;
              border-bottom: 2px solid #667eea; }}
.section h3 {{ font-size: 17px; color: #444; margin: 16px 0 8px 0; }}
.section h4 {{ font-size: 15px; color: #555; margin: 12px 0 6px 0; }}
.section p {{ margin: 6px 0; }}
.section li {{ margin-left: 24px; }}
.report-content {{ white-space: pre-wrap; word-wrap: break-word; }}
.data-table {{ width: 100%; border-collapse: collapse; margin: 10px 0; font-size: 13px; }}
.data-table th {{ background: #f0f2f8; padding: 8px 12px; text-align: left;
                  border: 1px solid #ddd; font-weight: 600; }}
.data-table td {{ padding: 6px 12px; border: 1px solid #eee; }}
.data-table tr:nth-child(even) {{ background: #fafbfc; }}
.data-table tr:hover {{ background: #f0f4ff; }}
pre.data-raw {{ background: #f8f9fa; padding: 12px; border-radius: 6px; font-size: 12px;
                overflow-x: auto; max-height: 400px; overflow-y: auto; border: 1px solid #e0e0e0; }}
code {{ background: #f0f0f0; padding: 2px 4px; border-radius: 3px; font-size: 12px; }}
.signal {{ display: inline-block; padding: 8px 20px; border-radius: 20px; font-weight: bold;
           font-size: 18px; margin: 10px 0; }}
.signal-buy {{ background: #d4edda; color: #155724; }}
.signal-sell {{ background: #f8d7da; color: #721c24; }}
.signal-hold {{ background: #fff3cd; color: #856404; }}
.tabs {{ display: flex; gap: 4px; margin-bottom: 16px; flex-wrap: wrap; }}
.tab {{ padding: 8px 16px; background: #e9ecef; border-radius: 8px 8px 0 0; cursor: pointer;
        font-size: 14px; border: none; }}
.tab.active {{ background: #667eea; color: white; }}
.tab-content {{ display: none; }}
.tab-content.active {{ display: block; }}
.status-ok {{ color: #28a745; }}
.status-fail {{ color: #dc3545; }}
.status-warn {{ color: #ffc107; }}
.toc {{ background: #f8f9fa; padding: 16px; border-radius: 8px; margin-bottom: 20px; }}
.toc a {{ color: #667eea; text-decoration: none; display: block; padding: 4px 0; }}
.toc a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
<div class="container">

<div class="header">
<h1>TradingAgents 分析报告</h1>
<div class="meta">
    {escape_html(ticker)} | 分析日期: {escape_html(trade_date)} | 报告生成时间: {now}
</div>
</div>

<!-- Table of Contents -->
<div class="section toc">
<h2>目录</h2>
<a href="#signal">最终交易信号</a>
<a href="#market">I. 市场分析报告</a>
<a href="#sentiment">II. 情绪分析报告</a>
<a href="#news">III. 新闻分析报告</a>
<a href="#fundamentals">IV. 基本面分析报告</a>
<a href="#debate">V. 多空辩论</a>
<a href="#trader">VI. 交易策略</a>
<a href="#risk">VII. 风险管理</a>
<a href="#decision">VIII. 最终交易决策</a>
<a href="#datasource">IX. 数据源核对</a>
</div>

<!-- Final Signal -->
<div class="section" id="signal">
<h2>最终交易信号</h2>
<div class="report-content">{md_to_html_basic(state.get('final_trade_decision', ''))}</div>
</div>

<!-- Market Report -->
<div class="section" id="market">
<h2>I. 市场分析报告 (Market Analyst)</h2>
<div class="report-content">{md_to_html_basic(state.get('market_report', ''))}</div>
</div>

<!-- Sentiment Report -->
<div class="section" id="sentiment">
<h2>II. 情绪分析报告 (Sentiment Analyst)</h2>
<div class="report-content">{md_to_html_basic(state.get('sentiment_report', ''))}</div>
</div>

<!-- News Report -->
<div class="section" id="news">
<h2>III. 新闻分析报告 (News Analyst)</h2>
<div class="report-content">{md_to_html_basic(state.get('news_report', ''))}</div>
</div>

<!-- Fundamentals Report -->
<div class="section" id="fundamentals">
<h2>IV. 基本面分析报告 (Fundamentals Analyst)</h2>
<div class="report-content">{md_to_html_basic(state.get('fundamentals_report', ''))}</div>
</div>

<!-- Investment Debate -->
<div class="section" id="debate">
<h2>V. 多空辩论 (Bull vs Bear)</h2>
"""

    debate = state.get("investment_debate_state", {})
    if debate.get("bull_history"):
        html += f"<h3>多方观点 (Bull Researcher)</h3>\n<div class='report-content'>{md_to_html_basic(debate['bull_history'])}</div>\n"
    if debate.get("bear_history"):
        html += f"<h3>空方观点 (Bear Researcher)</h3>\n<div class='report-content'>{md_to_html_basic(debate['bear_history'])}</div>\n"
    if debate.get("judge_decision"):
        html += f"<h3>研究经理裁定 (Research Manager)</h3>\n<div class='report-content'>{md_to_html_basic(debate['judge_decision'])}</div>\n"

    html += """</div>

<!-- Trader -->
<div class="section" id="trader">
<h2>VI. 交易策略 (Trader)</h2>
"""
    html += f"<div class='report-content'>{md_to_html_basic(state.get('trader_investment_plan', ''))}</div>\n"

    html += """</div>

<!-- Risk Management -->
<div class="section" id="risk">
<h2>VII. 风险管理辩论 (Risk Management)</h2>
"""
    risk = state.get("risk_debate_state", {})
    if risk.get("aggressive_history"):
        html += f"<h3>激进派 (Aggressive Analyst)</h3>\n<div class='report-content'>{md_to_html_basic(risk['aggressive_history'])}</div>\n"
    if risk.get("conservative_history"):
        html += f"<h3>保守派 (Conservative Analyst)</h3>\n<div class='report-content'>{md_to_html_basic(risk['conservative_history'])}</div>\n"
    if risk.get("neutral_history"):
        html += f"<h3>中立派 (Neutral Analyst)</h3>\n<div class='report-content'>{md_to_html_basic(risk['neutral_history'])}</div>\n"
    if risk.get("judge_decision"):
        html += f"<h3>投资组合经理裁定 (Portfolio Manager)</h3>\n<div class='report-content'>{md_to_html_basic(risk['judge_decision'])}</div>\n"

    html += """</div>

<!-- Final Decision -->
<div class="section" id="decision">
<h2>VIII. 最终交易决策</h2>
"""
    html += f"<div class='report-content'>{md_to_html_basic(state.get('final_trade_decision', ''))}</div>\n"

    html += f"""</div>

<!-- Data Source Verification -->
<div class="section" id="datasource">
<h2>IX. 数据源核对</h2>
<p><em>以下是本次分析使用的全部原始数据指标值，供核对数据源准确性。</em></p>
{data_source_html}
</div>

</div>
</body>
</html>"""

    Path(output_path).write_text(html, encoding="utf-8")
    print(f"HTML report saved to: {output_path}")

if __name__ == "__main__":
    state = load_json("test_output/BABA_final_state.json")
    bundle = load_json("test_output/BABA_2025-06-20_data.json")
    generate_report_path = "test_output/BABA_report.html"
    generate_html(state, bundle, generate_report_path)
