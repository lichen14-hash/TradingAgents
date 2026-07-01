"""Batch analysis: analyze multiple A-share tickers and generate HTML reports.

Usage:
    python run_batch_analysis.py [--collect-only] [--date YYYY-MM-DD]
"""

import json
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tradingagents.datacollector import DataBundle, DataCollector
from tradingagents.dataflows.market_utils import is_etf
from tradingagents.default_config import DEFAULT_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "test_output"
OUTPUT_DIR.mkdir(exist_ok=True)

TRADE_DATE = datetime.now().strftime("%Y-%m-%d")

TICKERS = [
    ("688599.SS", "天合光能"),
    ("589130.SS", "科创芯片ETF易方达"),
    ("515880.SS", "通信ETF国泰"),
]


def get_analysts_for_ticker(ticker: str) -> tuple[str, ...]:
    if is_etf(ticker):
        return ("market", "social", "news")
    return ("market", "social", "news", "fundamentals")


def _make_config() -> dict:
    config = DEFAULT_CONFIG.copy()
    config["max_debate_rounds"] = 1
    config["max_risk_discuss_rounds"] = 1
    config["output_language"] = "Chinese"
    return config


def collect_data(ticker: str, trade_date: str = TRADE_DATE) -> tuple[DataBundle, Path]:
    config = _make_config()
    collector = DataCollector(config)
    analysts = get_analysts_for_ticker(ticker)

    logger.info("Collecting data for %s on %s (analysts: %s) ...", ticker, trade_date, analysts)
    bundle, filepath = collector.collect_and_save(
        ticker, trade_date,
        selected_analysts=analysts,
        save_dir=OUTPUT_DIR,
    )
    logger.info("Data bundle saved to %s", filepath)
    return bundle, filepath


def run_analysis(ticker: str, bundle: DataBundle):
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    config = _make_config()
    analysts = get_analysts_for_ticker(ticker)

    graph = TradingAgentsGraph(
        selected_analysts=analysts,
        config=config,
        debug=False,
    )

    logger.info("Running analysis for %s on %s ...", ticker, bundle.metadata.trade_date)
    final_state, signal = graph.propagate(
        ticker, bundle.metadata.trade_date,
        data_bundle=bundle,
    )
    logger.info("Analysis complete for %s. Signal: %s", ticker, signal)

    safe_name = ticker.replace(".", "_")
    state_path = OUTPUT_DIR / f"{safe_name}_final_state.json"
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


def _check_data_completeness(bundle: DataBundle) -> list[dict]:
    """Scan the bundle for missing or unavailable data fields.

    Returns a list of dicts with keys: category, field, status.
    status is one of: "ok", "missing", "unavailable".
    """
    issues: list[dict] = []

    def _check_field(category: str, field: str, value: str):
        if not value or not value.strip():
            issues.append({"category": category, "field": field, "status": "missing"})
        elif "<unavailable" in value.lower():
            issues.append({"category": category, "field": field, "status": "unavailable"})

    def _check_dict_fields(category: str, d: dict[str, str]):
        unavail = 0
        total = len(d)
        for k, v in d.items():
            if not v or "<unavailable" in v.lower() or "data unavailable" in v.lower():
                unavail += 1
        if total > 0 and unavail == total:
            issues.append({"category": category, "field": f"全部 {total} 项", "status": "unavailable"})
        elif unavail > 0:
            issues.append({"category": category, "field": f"{unavail}/{total} 项", "status": "unavailable"})

    if bundle.market:
        _check_field("行情数据", "股价/成交量", bundle.market.stock_data)
        _check_field("行情数据", "验证快照", bundle.market.verified_snapshot)
        if not bundle.market.indicators:
            issues.append({"category": "行情数据", "field": "技术指标", "status": "missing"})
    else:
        issues.append({"category": "行情数据", "field": "全部", "status": "missing"})

    if bundle.sentiment:
        _check_field("情绪数据", "个股新闻", bundle.sentiment.ticker_news)
        _check_field("情绪数据", "StockTwits/股吧", bundle.sentiment.stocktwits)
        _check_field("情绪数据", "Reddit/新浪", bundle.sentiment.reddit)
    else:
        issues.append({"category": "情绪数据", "field": "全部", "status": "missing"})

    if bundle.news:
        _check_field("新闻数据", "个股新闻", bundle.news.ticker_news)
        _check_field("新闻数据", "全球/宏观新闻", bundle.news.global_news)
        _check_field("新闻数据", "内部交易", bundle.news.insider_transactions)
        if bundle.news.macro_indicators:
            _check_dict_fields("宏观指标", bundle.news.macro_indicators)
        if bundle.news.prediction_markets:
            _check_dict_fields("市场信号", bundle.news.prediction_markets)
    else:
        issues.append({"category": "新闻数据", "field": "全部", "status": "missing"})

    if bundle.fundamentals:
        _check_field("财务数据", "概览", bundle.fundamentals.overview)
        _check_field("财务数据", "资产负债表(季度)", bundle.fundamentals.balance_sheet_quarterly)
        _check_field("财务数据", "资产负债表(年度)", bundle.fundamentals.balance_sheet_annual)
        _check_field("财务数据", "现金流(季度)", bundle.fundamentals.cashflow_quarterly)
        _check_field("财务数据", "现金流(年度)", bundle.fundamentals.cashflow_annual)
        _check_field("财务数据", "利润表(季度)", bundle.fundamentals.income_quarterly)
        _check_field("财务数据", "利润表(年度)", bundle.fundamentals.income_annual)
    elif "fundamentals" in (bundle.metadata.selected_analysts or []):
        issues.append({"category": "财务数据", "field": "全部", "status": "missing"})

    return issues


def _build_completeness_banner(issues: list[dict]) -> str:
    """Build an HTML banner summarizing data completeness issues."""
    if not issues:
        return ""

    missing = [i for i in issues if i["status"] == "missing"]
    unavailable = [i for i in issues if i["status"] == "unavailable"]

    rows = []
    for issue in issues:
        icon = "❌" if issue["status"] == "missing" else "⚠️"
        label = "缺失" if issue["status"] == "missing" else "不可用"
        css = "status-empty" if issue["status"] == "missing" else "status-partial"
        rows.append(
            f'<tr><td>{escape_html(issue["category"])}</td>'
            f'<td>{escape_html(issue["field"])}</td>'
            f'<td class="{css}">{icon} {label}</td></tr>'
        )

    summary_parts = []
    if missing:
        summary_parts.append(f"{len(missing)} 项数据缺失")
    if unavailable:
        summary_parts.append(f"{len(unavailable)} 项数据不可用")
    summary = "、".join(summary_parts)

    return f"""
<div class="section" style="background: #fff8f0; border: 2px solid #ff9800; border-left-width: 6px;">
<h2 style="color: #e65100; border-bottom-color: #ff9800;">⚠️ 数据完整性警告</h2>
<p style="margin-bottom: 12px; color: #bf360c; font-weight: 600;">
本次分析存在 {summary}，可能影响分析结论的准确性。请结合实际情况审慎参考。
</p>
<table class="data-table" style="font-size: 13px;">
<tr><th style="width:120px;">数据类别</th><th>字段</th><th style="width:100px;">状态</th></tr>
{''.join(rows)}
</table>
</div>
"""


def generate_html_report(ticker: str, name: str, final_state: dict, bundle: DataBundle, output_dir: Path | None = None) -> Path:
    from test_output.run_baba_analysis import (
        build_analysis_sections,
        build_data_tables,
        build_decision_section,
    )

    meta = bundle.metadata
    trade_date = meta.trade_date
    original_date = meta.original_trade_date
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    asset_label = "ETF" if is_etf(ticker) else "股票"

    date_note = ""
    if original_date:
        reason = getattr(meta, 'date_correction_reason', '')
        if reason == "data_not_ready":
            msg = f'用户输入日期 <strong>{escape_html(original_date)}</strong> 的行情数据尚未更新，已使用最近交易日 <strong>{escape_html(trade_date)}</strong> 的数据'
        else:
            msg = f'用户输入日期 <strong>{escape_html(original_date)}</strong> 为非交易日，已自动校正为 <strong>{escape_html(trade_date)}</strong>'
        date_note = f'<p class="date-correction">⚠️ {msg}</p>'

    completeness_issues = _check_data_completeness(bundle)
    completeness_banner = _build_completeness_banner(completeness_issues)

    data_tables_html = build_data_tables(bundle)
    decision_html = build_decision_section(final_state)
    analysis_html = build_analysis_sections(final_state)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="stock-name" content="{escape_html(name)}">
<meta name="stock-ticker" content="{escape_html(ticker)}">
<title>TradingAgents 分析报告 - {escape_html(name)}({escape_html(ticker)}) ({escape_html(trade_date)})</title>
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
.header .asset-type {{ display: inline-block; background: rgba(255,255,255,0.2);
                       padding: 4px 12px; border-radius: 4px; margin-top: 8px; font-size: 13px; }}
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
    {escape_html(name)} ({escape_html(ticker)}) | 交易日: {escape_html(trade_date)} | 报告生成时间: {now}
</div>
<div class="asset-type">{asset_label}</div>
</div>

{date_note}

{completeness_banner}

<div class="section toc">
<h2>目录</h2>
<a href="#decision">一、最终交易决策建议</a>
<a href="#analysis">二、论证过程</a>
<a href="#source-data">三、详细源数据</a>
</div>

{decision_html}

{analysis_html}

<div class="section" id="source-data">
<h2>三、详细源数据</h2>
</div>

<div class="section" id="metadata">
<h3>3.1 数据采集元数据</h3>
<table class="metadata-table">
<tr><td>名称</td><td>{escape_html(name)}</td></tr>
<tr><td>代码</td><td>{escape_html(ticker)}</td></tr>
<tr><td>类型</td><td>{escape_html(asset_label)}</td></tr>
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

    safe_name = ticker.replace(".", "_")
    _out_dir = output_dir if output_dir is not None else OUTPUT_DIR
    _out_dir.mkdir(exist_ok=True)
    output_path = _out_dir / f"{safe_name}_report.html"
    output_path.write_text(html, encoding="utf-8")
    logger.info("HTML report saved to %s", output_path)
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--collect-only", action="store_true", help="Only collect data, skip analysis")
    parser.add_argument("--date", type=str, default=TRADE_DATE, help="Trade date (YYYY-MM-DD)")
    args = parser.parse_args()

    results = []
    for ticker, name in TICKERS:
        print(f"\n{'='*60}")
        print(f"  Processing: {name} ({ticker})")
        print(f"{'='*60}")

        try:
            bundle, _ = collect_data(ticker, args.date)

            if args.collect_only:
                safe = ticker.replace(".", "_")
                bundle_path = OUTPUT_DIR / f"{safe}_{bundle.metadata.trade_date}_data.json"
                DataCollector.save(bundle, bundle_path)
                logger.info("Data collection complete for %s. Bundle: %s", ticker, bundle_path)
                results.append((ticker, name, "collected", str(bundle_path)))
                continue

            final_state = run_analysis(ticker, bundle)
            report_path = generate_html_report(ticker, name, final_state, bundle)
            results.append((ticker, name, "success", str(report_path)))

        except Exception as e:
            logger.error("Failed to process %s (%s): %s", name, ticker, e)
            traceback.print_exc()
            results.append((ticker, name, "failed", str(e)))

    print(f"\n\n{'='*60}")
    print("  BATCH ANALYSIS RESULTS")
    print(f"{'='*60}")
    for ticker, name, status, detail in results:
        icon = "OK" if status == "success" else ("COLLECTED" if status == "collected" else "FAIL")
        print(f"  [{icon}] {name} ({ticker}): {detail}")
    print(f"{'='*60}\n")
