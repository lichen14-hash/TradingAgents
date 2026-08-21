"""HTML report section renderers shared by every report producer.

These renderers used to live in ``test_output/run_baba_analysis.py``, and the
production batch runner imported them from that test script
(``run_batch_analysis.generate_html_report``). Moving them here removes the
production-depends-on-test-directory edge; ``test_output/run_baba_analysis.py``
now re-exports them so the existing script keeps working.

Report-integrity annotations (per-stage ratings, missing-output notices) are
rendered here too, so the CLI, the web server, and the batch runner all show
them — see :mod:`tradingagents.reporting.integrity_report`.
"""

from __future__ import annotations

import json
import re

from tradingagents.agents.utils.integrity import is_blank_output
from tradingagents.datacollector import DataBundle

from .allocation_report import build_position_advice
from .integrity_report import build_stage_rating_table, stage_rating_note


def escape_html(text: str) -> str:
    if not text:
        return ""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# One label per ``BundleMetadata.date_correction_reason`` value. Centralised
# because three report producers used to inline the same
# ``== "data_not_ready" ... else "非交易日"`` ternary, so a new reason value
# silently rendered as "非交易日" on a date that was in fact a trading day.
_DATE_CORRECTION_LABELS = {
    "data_not_ready": "数据尚未更新",
    "intraday_daily_bar_incomplete": "当日日线未收盘",
    "non_trading_day": "非交易日",
}

_DATE_CORRECTION_MESSAGES = {
    "data_not_ready": (
        "用户输入日期 <strong>{original}</strong> 的行情数据尚未更新，"
        "已使用最近交易日 <strong>{corrected}</strong> 的数据"
    ),
    "intraday_daily_bar_incomplete": (
        "用户输入日期 <strong>{original}</strong> 当日日线尚未收盘，"
        "已改用最近一个<strong>完整</strong>交易日 <strong>{corrected}</strong> 的数据"
        "（当日盘中涨跌不在本报告的行情数据内）"
    ),
    "non_trading_day": (
        "用户输入日期 <strong>{original}</strong> 为非交易日，"
        "已自动校正为 <strong>{corrected}</strong>"
    ),
}


def date_correction_label(reason: str, original_date: str | None) -> str:
    """Short label for the "是否校正" metadata row."""
    if not original_date:
        return "否（输入即为交易日）"
    return f"是（{_DATE_CORRECTION_LABELS.get(reason or '', '非交易日')}）"


def date_correction_note(reason: str, original_date: str | None, corrected_date: str) -> str:
    """Banner shown above the report when the trade date was rolled back."""
    if not original_date:
        return ""
    template = _DATE_CORRECTION_MESSAGES.get(
        reason or "", _DATE_CORRECTION_MESSAGES["non_trading_day"],
    )
    msg = template.format(
        original=escape_html(original_date), corrected=escape_html(corrected_date),
    )
    return f'<p class="date-correction">⚠️ {msg}</p>'


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


def build_decision_section(
    final_state: dict,
    allocation_item=None,
    allocation=None,
) -> str:
    """The decision section, optionally carrying the portfolio layer's target.

    The target weight is rendered inside this section rather than as a section of
    its own so that it sits directly under the decision it belongs to, and so the
    existing section numbering (一、二) does not shift on old reports.
    ``allocation_item=None`` renders no target at all — see
    :mod:`tradingagents.reporting.allocation_report`.
    """
    sections = []
    sections.append('<div class="section decision-section" id="decision">')
    sections.append("<h2>一、最终交易决策建议</h2>")
    if final_state.get("final_trade_decision"):
        sections.append(f'<div class="report-content">{_md_to_html(final_state["final_trade_decision"])}</div>')
    else:
        sections.append('<p class="empty">无交易决策</p>')
    advice = build_position_advice(allocation_item, allocation)
    if advice:
        sections.append(advice)
    # Put the three stages side by side right under the final decision: an
    # override is otherwise only visible to a reader who compares three
    # sections of section 二 in full.
    sections.append(build_stage_rating_table(final_state))
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
    for key, title in [("bull_history", "多方观点 (Bull)"), ("bear_history", "空方观点 (Bear)")]:
        sections.append(f"<h4>{title}</h4>")
        sections.append(_debate_side_html(debate.get(key, "")))
    if debate.get("judge_decision"):
        note = stage_rating_note(final_state, "research")
        sections.append(f"<h4>研究经理裁定 {note}</h4>")
        sections.append(f'<div class="report-content">{_md_to_html(debate["judge_decision"])}</div>')

    # 2.3 Trader
    sections.append(f'<h3>2.3 交易策略 {stage_rating_note(final_state, "trader")}</h3>')
    if final_state.get("trader_investment_plan"):
        sections.append(f'<div class="report-content">{_md_to_html(final_state["trader_investment_plan"])}</div>')
    else:
        sections.append('<p class="empty">无交易策略</p>')

    # 2.4 Risk debate
    sections.append("<h3>2.4 风险管理辩论</h3>")
    risk = final_state.get("risk_debate_state", {})
    for key, title in [("aggressive_history", "激进派"), ("conservative_history", "保守派"), ("neutral_history", "中立派")]:
        sections.append(f"<h4>{title}</h4>")
        sections.append(_debate_side_html(risk.get(key, "")))
    if risk.get("judge_decision"):
        note = stage_rating_note(final_state, "portfolio")
        sections.append(f"<h4>投资组合经理最终裁定 {note}</h4>")
        sections.append(f'<div class="report-content">{_md_to_html(risk["judge_decision"])}</div>')

    sections.append("</div>")
    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# "Bear Analyst:" / "Conservative Risk Analyst:" style prefixes that each debate
# node prepends to its own turn.
_ROLE_PREFIX_RE = re.compile(
    r"^\s*(?:Bull|Bear|Aggressive|Conservative|Neutral|Risky|Safe)"
    r"(?:\s+(?:Analyst|Researcher|Risk\s+Analyst))?\s*:",
    re.MULTILINE,
)


def _debate_side_html(history: str) -> str:
    """Render one debate side, making a silent no-show impossible to miss.

    The heading is now always emitted, and a side that said nothing gets an
    explicit notice. Previously the renderer used ``if history:``, which passed
    for the 2026-08-11 300760.SZ run whose ``bear_history`` was nothing but
    three role prefixes — the report showed an ordinary-looking empty section
    and the one-sided debate was invisible.
    """
    body = _ROLE_PREFIX_RE.sub("", history or "")
    if is_blank_output(body):
        rounds = len(_ROLE_PREFIX_RE.findall(history or ""))
        detail = f"（共 {rounds} 轮发言，正文均为空）" if rounds else "（无任何发言记录）"
        return (
            '<p class="status-empty">⚠️ 本方未产生有效输出'
            f"{detail}，请勿将其视为“无异议”或“已被说服”。</p>"
        )
    return f'<div class="report-content">{_md_to_html(history)}</div>'


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
