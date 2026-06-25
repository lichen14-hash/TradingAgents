"""End-to-end BABA analysis: collect data → run analysis → generate HTML report.

Usage:
    python test_output/run_baba_analysis.py [--collect-only] [--bundle PATH]
"""

import json
import logging
import sys
import os
from datetime import datetime
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tradingagents.datacollector import DataBundle, DataCollector
from tradingagents.default_config import DEFAULT_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent
TICKER = "BABA"
TRADE_DATE = datetime.now().strftime("%Y-%m-%d")


def collect_data(trade_date: str = TRADE_DATE) -> tuple[DataBundle, Path]:
    config = DEFAULT_CONFIG.copy()
    collector = DataCollector(config)

    logger.info("Collecting data for %s on %s ...", TICKER, trade_date)
    bundle, filepath = collector.collect_and_save(
        TICKER, trade_date, save_dir=OUTPUT_DIR,
    )
    logger.info("Data bundle saved to %s", filepath)

    meta = bundle.metadata
    logger.info("  trade_date (corrected): %s", meta.trade_date)
    if meta.original_trade_date:
        logger.info("  original_trade_date:    %s", meta.original_trade_date)
    else:
        logger.info("  (no correction needed)")

    return bundle, filepath


def run_analysis(bundle: DataBundle):
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    config = DEFAULT_CONFIG.copy()
    config["max_debate_rounds"] = 1
    config["max_risk_discuss_rounds"] = 1

    graph = TradingAgentsGraph(
        selected_analysts=("market", "social", "news", "fundamentals"),
        config=config,
        debug=False,
    )

    logger.info("Running analysis for %s on %s ...", TICKER, bundle.metadata.trade_date)
    final_state, signal = graph.propagate(
        TICKER, bundle.metadata.trade_date,
        data_bundle=bundle,
    )
    logger.info("Analysis complete. Signal: %s", signal)

    state_path = OUTPUT_DIR / f"{TICKER}_final_state.json"
    serializable = {}
    for k, v in final_state.items():
        if k == "messages":
            continue
        try:
            json.dumps(v)
            serializable[k] = v
        except (TypeError, ValueError):
            serializable[k] = str(v)
    state_path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Final state saved to %s", state_path)
    return final_state


def escape_html(text: str) -> str:
    if not text:
        return ""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def generate_html_report(final_state: dict, bundle: DataBundle):
    meta = bundle.metadata
    ticker = meta.ticker
    trade_date = meta.trade_date
    original_date = meta.original_trade_date
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    date_note = ""
    if original_date:
        reason = getattr(meta, 'date_correction_reason', '')
        if reason == "data_not_ready":
            msg = f'用户输入日期 <strong>{escape_html(original_date)}</strong> 的行情数据尚未更新，已使用最近交易日 <strong>{escape_html(trade_date)}</strong> 的数据'
        else:
            msg = f'用户输入日期 <strong>{escape_html(original_date)}</strong> 为非交易日，已自动校正为 <strong>{escape_html(trade_date)}</strong>'
        date_note = f'<p class="date-correction">⚠️ {msg}</p>'

    data_tables_html = build_data_tables(bundle)
    decision_html = build_decision_section(final_state)
    analysis_html = build_analysis_sections(final_state)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="stock-name" content="{escape_html(ticker)}">
<meta name="stock-ticker" content="{escape_html(ticker)}">
<title>TradingAgents 分析报告 - {escape_html(ticker)} ({escape_html(trade_date)})</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background: #f5f7fa; color: #1a1a2e; line-height: 1.6; }}
.container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
.header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
           color: white; padding: 40px; border-radius: 12px; margin-bottom: 24px;
           text-align: center; }}
.header h1 {{ font-size: 28px; margin-bottom: 8px; }}
.header .meta {{ font-size: 14px; opacity: 0.9; }}
.date-correction {{ background: #fff3cd; color: #856404; padding: 12px 16px;
                     border-radius: 8px; margin-bottom: 20px; border-left: 4px solid #ffc107; }}
.section {{ background: white; border-radius: 12px; padding: 24px; margin-bottom: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
.section h2 {{ font-size: 20px; color: #333; margin-bottom: 16px; padding-bottom: 8px;
              border-bottom: 2px solid #667eea; }}
.section h3 {{ font-size: 17px; color: #444; margin: 16px 0 8px 0; }}
.section h4 {{ font-size: 15px; color: #555; margin: 12px 0 6px 0; }}
.section p {{ margin: 6px 0; }}
.data-table {{ width: 100%; border-collapse: collapse; margin: 10px 0; font-size: 13px; }}
.data-table th {{ background: #f0f2f8; padding: 8px 12px; text-align: left;
                  border: 1px solid #ddd; font-weight: 600; position: sticky; top: 0; }}
.data-table td {{ padding: 6px 12px; border: 1px solid #eee; }}
.data-table tr:nth-child(even) {{ background: #fafbfc; }}
.data-table tr:hover {{ background: #f0f4ff; }}
.data-table .empty {{ color: #999; font-style: italic; }}
.data-table .unavailable {{ color: #dc3545; font-style: italic; }}
pre.data-raw {{ background: #f8f9fa; padding: 12px; border-radius: 6px; font-size: 12px;
                overflow-x: auto; max-height: 400px; overflow-y: auto; border: 1px solid #e0e0e0; }}
.report-content {{ white-space: pre-wrap; word-wrap: break-word; }}
.toc {{ background: #f8f9fa; padding: 16px; border-radius: 8px; margin-bottom: 20px; }}
.toc a {{ color: #667eea; text-decoration: none; display: block; padding: 4px 0; }}
.toc a:hover {{ text-decoration: underline; }}
.status-ok {{ color: #28a745; font-weight: bold; }}
.status-empty {{ color: #dc3545; font-weight: bold; }}
.status-partial {{ color: #ffc107; font-weight: bold; }}
.decision-section {{ background: linear-gradient(135deg, #f8fff8 0%, #f0faf0 100%);
                     border: 2px solid #28a745; }}
.decision-section h2 {{ color: #155724; border-bottom-color: #28a745; }}
.metadata-table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
.metadata-table td {{ padding: 8px 12px; border-bottom: 1px solid #eee; }}
.metadata-table td:first-child {{ font-weight: 600; color: #555; width: 200px; }}
</style>
</head>
<body>
<div class="container">

<div class="header">
<h1>TradingAgents 分析报告</h1>
<div class="meta">
    {escape_html(ticker)} | 交易日: {escape_html(trade_date)} | 报告生成时间: {now}
</div>
</div>

{date_note}

<!-- TOC -->
<div class="section toc">
<h2>目录</h2>
<a href="#decision">一、最终交易决策建议</a>
<a href="#analysis">二、论证过程</a>
<a href="#source-data">三、详细源数据</a>
</div>

{decision_html}

{analysis_html}

<!-- Source Data -->
<div class="section" id="source-data">
<h2>三、详细源数据</h2>
</div>

<div class="section" id="metadata">
<h3>3.1 数据采集元数据</h3>
<table class="metadata-table">
<tr><td>股票代码</td><td>{escape_html(ticker)}</td></tr>
<tr><td>交易日（校正后）</td><td>{escape_html(trade_date)}</td></tr>
<tr><td>用户输入日期</td><td>{escape_html(original_date or trade_date)}</td></tr>
<tr><td>是否校正</td><td>{"是（数据尚未更新）" if getattr(meta, 'date_correction_reason', '') == "data_not_ready" else "是（非交易日）" if original_date else "否（输入即为交易日）"}</td></tr>
<tr><td>采集时间</td><td>{escape_html(meta.collection_timestamp)}</td></tr>
<tr><td>数据版本</td><td>{escape_html(meta.bundle_version)}</td></tr>
<tr><td>选中分析师</td><td>{escape_html(', '.join(meta.selected_analysts))}</td></tr>
</table>
</div>

{data_tables_html}

</div>
</body>
</html>"""

    output_path = OUTPUT_DIR / f"{ticker}_report.html"
    output_path.write_text(html, encoding="utf-8")
    logger.info("HTML report saved to %s", output_path)
    return output_path


def build_data_tables(bundle: DataBundle) -> str:
    sections = []
    trade_date = bundle.metadata.trade_date

    # ── Market Data: OHLCV ──
    sections.append('<div class="section" id="market-data">')
    sections.append("<h3>3.2 市场行情原始数据 (OHLCV)</h3>")
    if bundle.market and bundle.market.stock_data:
        val = bundle.market.stock_data
        if val.startswith("<unavailable"):
            sections.append(f'<p class="unavailable">{escape_html(val)}</p>')
        else:
            sections.append(_csv_text_to_table(val, "OHLCV 价格数据"))
    else:
        sections.append('<p class="empty">无数据</p>')
    sections.append("</div>")

    # ── Technical Indicators ──
    sections.append('<div class="section" id="indicators">')
    sections.append("<h3>3.3 技术指标原始数据</h3>")
    sections.append(f"<p>交易日: <strong>{escape_html(trade_date)}</strong></p>")
    sections.append('<table class="data-table">')
    sections.append("<tr><th>指标名称</th><th>交易日</th><th>最新值</th><th>数据状态</th><th>原始数据预览</th></tr>")
    if bundle.market and bundle.market.indicators:
        for name, data in bundle.market.indicators.items():
            if data.startswith("<unavailable"):
                sections.append(
                    f'<tr><td>{escape_html(name)}</td>'
                    f'<td>{escape_html(trade_date)}</td>'
                    f'<td class="empty">—</td>'
                    f'<td class="unavailable">不可用</td>'
                    f'<td class="unavailable">{escape_html(data[:120])}</td></tr>'
                )
            else:
                latest_val = _extract_latest_value(data, trade_date)
                status = '<span class="status-ok">✓ 有数据</span>'
                preview = data.replace("\n", " ")[:120]
                sections.append(
                    f"<tr><td>{escape_html(name)}</td>"
                    f"<td>{escape_html(trade_date)}</td>"
                    f"<td><strong>{escape_html(latest_val)}</strong></td>"
                    f"<td>{status}</td>"
                    f"<td><code>{escape_html(preview)}</code></td></tr>"
                )
    else:
        sections.append('<tr><td colspan="5" class="empty">无技术指标数据</td></tr>')
    sections.append("</table>")
    sections.append("</div>")

    # ── Verified Snapshot ──
    sections.append('<div class="section" id="snapshot">')
    sections.append("<h3>3.4 验证快照 (Verified Snapshot)</h3>")
    if bundle.market and bundle.market.verified_snapshot:
        val = bundle.market.verified_snapshot
        if val.startswith("<unavailable"):
            sections.append(f'<p class="unavailable">{escape_html(val)}</p>')
        else:
            sections.append(_md_to_html(val))
    else:
        sections.append('<p class="empty">无验证快照</p>')
    sections.append("</div>")

    # ── Sentiment Data ──
    sections.append('<div class="section" id="sentiment-data">')
    sections.append("<h3>3.5 情绪数据原始数据</h3>")
    sections.append('<table class="data-table">')
    sections.append("<tr><th>数据源</th><th>交易日</th><th>状态</th><th>数据量</th><th>内容预览</th></tr>")
    if bundle.sentiment:
        for field, label in [("ticker_news", "新闻情绪"), ("stocktwits", "StockTwits"), ("reddit", "Reddit")]:
            val = getattr(bundle.sentiment, field, "")
            if not val:
                sections.append(
                    f'<tr><td>{label}</td><td>{escape_html(trade_date)}</td>'
                    f'<td class="empty">—</td><td>0</td><td class="empty">无数据</td></tr>'
                )
            elif val.startswith("<unavailable"):
                sections.append(
                    f'<tr><td>{label}</td><td>{escape_html(trade_date)}</td>'
                    f'<td class="unavailable">不可用</td><td>—</td>'
                    f'<td class="unavailable">{escape_html(val[:120])}</td></tr>'
                )
            else:
                sections.append(
                    f'<tr><td>{label}</td><td>{escape_html(trade_date)}</td>'
                    f'<td class="status-ok">✓</td><td>{len(val):,} chars</td>'
                    f'<td><code>{escape_html(val[:150])}</code></td></tr>'
                )
    else:
        sections.append(f'<tr><td colspan="5" class="empty">未采集情绪数据</td></tr>')
    sections.append("</table>")
    sections.append("</div>")

    # ── News Data ──
    sections.append('<div class="section" id="news-data">')
    sections.append("<h3>3.6 新闻数据原始数据</h3>")

    if bundle.news:
        # Basic news fields
        sections.append("<h3>新闻概览</h3>")
        sections.append('<table class="data-table">')
        sections.append("<tr><th>数据源</th><th>交易日</th><th>状态</th><th>数据量</th></tr>")
        for field, label in [("ticker_news", "个股新闻"), ("global_news", "全球新闻"), ("insider_transactions", "内部交易")]:
            val = getattr(bundle.news, field, "")
            if not val:
                sections.append(f'<tr><td>{label}</td><td>{escape_html(trade_date)}</td><td class="empty">—</td><td>0</td></tr>')
            elif val.startswith("<unavailable"):
                sections.append(f'<tr><td>{label}</td><td>{escape_html(trade_date)}</td><td class="unavailable">不可用</td><td>—</td></tr>')
            else:
                sections.append(f'<tr><td>{label}</td><td>{escape_html(trade_date)}</td><td class="status-ok">✓</td><td>{len(val):,} chars</td></tr>')
        sections.append("</table>")

        # Macro indicators
        sections.append("<h3>宏观经济指标 (FRED)</h3>")
        sections.append('<table class="data-table">')
        sections.append("<tr><th>指标名称</th><th>交易日</th><th>状态</th><th>最新值</th></tr>")
        if bundle.news.macro_indicators:
            for name, data in bundle.news.macro_indicators.items():
                if data.startswith("<unavailable"):
                    sections.append(f'<tr><td>{escape_html(name)}</td><td>{escape_html(trade_date)}</td><td class="unavailable">不可用</td><td>—</td></tr>')
                else:
                    latest = _extract_fred_latest(data)
                    sections.append(f'<tr><td>{escape_html(name)}</td><td>{escape_html(trade_date)}</td><td class="status-ok">✓</td><td>{escape_html(latest)}</td></tr>')
        else:
            sections.append(f'<tr><td colspan="4" class="empty">无宏观指标</td></tr>')
        sections.append("</table>")

        # Prediction markets
        sections.append("<h3>预测市场 (Polymarket)</h3>")
        sections.append('<table class="data-table">')
        sections.append("<tr><th>查询主题</th><th>交易日</th><th>状态</th></tr>")
        if bundle.news.prediction_markets:
            for topic, data in bundle.news.prediction_markets.items():
                if "unavailable" in data.lower():
                    sections.append(f'<tr><td>{escape_html(topic)}</td><td>{escape_html(trade_date)}</td><td class="unavailable">不可用</td></tr>')
                else:
                    sections.append(f'<tr><td>{escape_html(topic)}</td><td>{escape_html(trade_date)}</td><td class="status-ok">✓ 有数据</td></tr>')
        else:
            sections.append(f'<tr><td colspan="3" class="empty">无预测市场数据</td></tr>')
        sections.append("</table>")
    else:
        sections.append('<p class="empty">未采集新闻数据</p>')
    sections.append("</div>")

    # ── Fundamentals ──
    sections.append('<div class="section" id="fundamentals-data">')
    sections.append("<h3>3.7 基本面原始数据</h3>")
    if bundle.fundamentals:
        # Company overview
        sections.append("<h3>公司概况</h3>")
        try:
            overview = json.loads(bundle.fundamentals.overview)
            sections.append('<table class="data-table">')
            sections.append("<tr><th>字段</th><th>值</th></tr>")
            key_fields = [
                "Symbol", "Name", "Exchange", "Currency", "Country", "Sector", "Industry",
                "MarketCapitalization", "PERatio", "PEGRatio", "DividendYield", "EPS",
                "52WeekHigh", "52WeekLow", "50DayMovingAverage", "200DayMovingAverage",
                "BookValue", "PriceToBookRatio", "RevenuePerShareTTM", "ProfitMargin",
            ]
            for k in key_fields:
                if k in overview:
                    sections.append(f"<tr><td>{escape_html(k)}</td><td>{escape_html(str(overview[k]))}</td></tr>")
            sections.append("</table>")
        except (json.JSONDecodeError, TypeError):
            if bundle.fundamentals.overview.startswith("<unavailable"):
                sections.append(f'<p class="unavailable">{escape_html(bundle.fundamentals.overview[:200])}</p>')
            else:
                sections.append(f'<pre class="data-raw">{escape_html(bundle.fundamentals.overview[:500])}</pre>')

        # Financial statements
        sections.append("<h3>财务报表数据</h3>")
        sections.append('<table class="data-table">')
        sections.append("<tr><th>报表类型</th><th>交易日</th><th>状态</th><th>数据量</th></tr>")
        for field, label in [
            ("balance_sheet_quarterly", "资产负债表(季度)"), ("balance_sheet_annual", "资产负债表(年度)"),
            ("cashflow_quarterly", "现金流量表(季度)"), ("cashflow_annual", "现金流量表(年度)"),
            ("income_quarterly", "利润表(季度)"), ("income_annual", "利润表(年度)"),
        ]:
            val = getattr(bundle.fundamentals, field, "")
            if not val:
                sections.append(f'<tr><td>{label}</td><td>{escape_html(trade_date)}</td><td class="empty">—</td><td>0</td></tr>')
            elif val.startswith("<unavailable"):
                sections.append(f'<tr><td>{label}</td><td>{escape_html(trade_date)}</td><td class="unavailable">不可用</td><td>—</td></tr>')
            else:
                sections.append(f'<tr><td>{label}</td><td>{escape_html(trade_date)}</td><td class="status-ok">✓</td><td>{len(val):,} chars</td></tr>')
        sections.append("</table>")
    else:
        sections.append('<p class="empty">未采集基本面数据</p>')
    sections.append("</div>")

    return "\n".join(sections)


def build_decision_section(final_state: dict) -> str:
    sections = []
    sections.append('<div class="section decision-section" id="decision">')
    sections.append("<h2>一、最终交易决策建议</h2>")
    if final_state.get("final_trade_decision"):
        sections.append(f'<div class="report-content">{_md_to_html(final_state["final_trade_decision"])}</div>')
    else:
        sections.append('<p class="empty">无交易决策</p>')
    sections.append("</div>")
    return "\n".join(sections)


def build_analysis_sections(final_state: dict) -> str:
    sections = []

    sections.append('<div class="section" id="analysis">')
    sections.append("<h2>二、论证过程</h2>")

    # 2.1 Analyst reports
    sections.append("<h3>2.1 分析师报告</h3>")
    for key, title in [
        ("market_report", "市场分析"),
        ("sentiment_report", "情绪分析"),
        ("news_report", "新闻分析"),
        ("fundamentals_report", "基本面分析"),
    ]:
        val = final_state.get(key, "")
        if val:
            sections.append(f"<h4>{title}</h4>")
            sections.append(f'<div class="report-content">{_md_to_html(val)}</div>')

    # 2.2 Investment debate
    sections.append("<h3>2.2 投资研究辩论</h3>")
    debate = final_state.get("investment_debate_state", {})
    if debate.get("bull_history"):
        sections.append("<h4>多方观点 (Bull)</h4>")
        sections.append(f'<div class="report-content">{_md_to_html(debate["bull_history"])}</div>')
    if debate.get("bear_history"):
        sections.append("<h4>空方观点 (Bear)</h4>")
        sections.append(f'<div class="report-content">{_md_to_html(debate["bear_history"])}</div>')
    if debate.get("judge_decision"):
        sections.append("<h4>研究经理裁定</h4>")
        sections.append(f'<div class="report-content">{_md_to_html(debate["judge_decision"])}</div>')

    # 2.3 Trader
    sections.append("<h3>2.3 交易策略</h3>")
    if final_state.get("trader_investment_plan"):
        sections.append(f'<div class="report-content">{_md_to_html(final_state["trader_investment_plan"])}</div>')
    else:
        sections.append('<p class="empty">无交易策略</p>')

    # 2.4 Risk debate
    sections.append("<h3>2.4 风险管理辩论</h3>")
    risk = final_state.get("risk_debate_state", {})
    for key, title in [("aggressive_history", "激进派"), ("conservative_history", "保守派"), ("neutral_history", "中立派")]:
        if risk.get(key):
            sections.append(f"<h4>{title}</h4>")
            sections.append(f'<div class="report-content">{_md_to_html(risk[key])}</div>')
    if risk.get("judge_decision"):
        sections.append("<h4>投资组合经理最终裁定</h4>")
        sections.append(f'<div class="report-content">{_md_to_html(risk["judge_decision"])}</div>')

    sections.append("</div>")
    return "\n".join(sections)


# ── Helpers ──

def _extract_latest_value(indicator_text: str, trade_date: str) -> str:
    """Extract the value for trade_date from indicator text."""
    for line in indicator_text.strip().split("\n"):
        if trade_date in line:
            parts = line.split()
            if len(parts) >= 2:
                return parts[-1]
    lines = indicator_text.strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        if not line or line.startswith("Date") or set(line) <= set("-| "):
            continue
        parts = line.split()
        if len(parts) >= 2:
            return parts[-1]
    return "—"


def _extract_fred_latest(data: str) -> str:
    for line in data.split("\n"):
        if "Latest:" in line:
            return line.split("Latest:")[-1].strip().rstrip("*")
    lines = [l.strip() for l in data.strip().split("\n") if l.strip()]
    return lines[-1][:60] if lines else "—"


def _csv_text_to_table(csv_text: str, title: str = "") -> str:
    lines = [l.strip() for l in csv_text.strip().split("\n") if l.strip()]
    if not lines:
        return '<p class="empty">无数据</p>'

    result = []
    if title:
        result.append(f"<h4>{escape_html(title)}</h4>")
    result.append('<div style="max-height:400px;overflow:auto;">')
    result.append('<table class="data-table">')

    for i, line in enumerate(lines):
        sep = "," if "," in line else "|"
        cells = [c.strip() for c in line.split(sep)]
        tag = "th" if i == 0 else "td"
        row = "".join(f"<{tag}>{escape_html(c)}</{tag}>" for c in cells if c)
        if row:
            result.append(f"<tr>{row}</tr>")

    result.append("</table>")
    result.append("</div>")
    return "\n".join(result)


def _md_to_html(text: str) -> str:
    import re
    if not text:
        return '<p class="empty">无数据</p>'

    lines = text.split("\n")
    result = []
    in_table = False

    for line in lines:
        stripped = line.strip()

        if "|" in stripped and stripped.startswith("|"):
            cells = [c.strip() for c in stripped.split("|")[1:-1]]
            if all(set(c) <= set("-: ") for c in cells):
                continue
            if not in_table:
                result.append('<table class="data-table">')
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
        elif stripped:
            html_line = escape_html(stripped)
            html_line = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html_line)
            result.append(f"<p>{html_line}</p>")

    if in_table:
        result.append("</table>")

    return "\n".join(result)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--collect-only", action="store_true", help="Only collect data, skip analysis")
    parser.add_argument("--bundle", type=str, help="Path to existing data bundle JSON")
    parser.add_argument("--date", type=str, default=TRADE_DATE, help="Trade date (YYYY-MM-DD)")
    args = parser.parse_args()

    if args.bundle:
        logger.info("Loading existing bundle from %s", args.bundle)
        bundle = DataCollector.load(args.bundle)
    else:
        bundle, _ = collect_data(args.date)

    if args.collect_only:
        bundle_path = OUTPUT_DIR / f"{TICKER}_{bundle.metadata.trade_date}_data.json"
        DataCollector.save(bundle, bundle_path)
        logger.info("Data collection complete. Bundle: %s", bundle_path)
        sys.exit(0)

    final_state = run_analysis(bundle)
    report_path = generate_html_report(final_state, bundle)
    print(f"\n{'='*60}")
    print(f"Report generated: {report_path}")
    print(f"{'='*60}")
