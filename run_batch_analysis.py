"""Batch analysis: analyze multiple A-share tickers and generate HTML reports.

Usage:
    python run_batch_analysis.py [--collect-only] [--date YYYY-MM-DD]
"""

import json
import logging
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tradingagents.agents.utils.market_status_guard import (
    MarketStatusConflictError,
    classify_market_status_conflicts,
)
from tradingagents.datacollector import DataBundle, DataCollector
from tradingagents.dataflows.market_utils import is_etf
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.reporting import (
    build_integrity_banner,
    collect_integrity_findings,
    split_data_completeness_findings,
)
from tradingagents.reporting.data_completeness import check_data_completeness
from tradingagents.reporting.html_sections import (
    build_analysis_sections,
    build_data_tables,
    build_decision_section,
    date_correction_label,
    date_correction_note,
)

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
    """Shim kept for existing callers; see tradingagents.reporting.data_completeness."""
    return check_data_completeness(bundle)


# status -> (icon, label, css class). The banner used to collapse everything
# that was not "missing" into "不可用", which hid the two most actionable kinds:
# a feed that returned prose admitting a gap ("行业资金流向数据暂不可用") and one
# that returned real content from years ago (northbound_flow labelled 2024-08-16
# in a 2026-08-18 run). Both had to be spotted by hand.
_COMPLETENESS_STATUS_STYLES = {
    "missing": ("❌", "缺失", "status-empty"),
    "unavailable": ("⚠️", "不可用", "status-partial"),
    "partial": ("⚠️", "部分缺失", "status-partial"),
    "stale": ("🕒", "数据陈旧", "status-partial"),
}
_COMPLETENESS_SUMMARY_LABELS = {
    "missing": "项数据缺失",
    "unavailable": "项数据不可用",
    "partial": "项数据部分缺失",
    "stale": "项数据陈旧",
}


def _build_completeness_banner(issues: list[dict]) -> str:
    """Build an HTML banner summarizing data completeness issues."""
    if not issues:
        return ""

    # Blocking first, then by kind, so the reader hits the load-bearing gaps
    # before the long tail of empty optional fields.
    order = list(_COMPLETENESS_STATUS_STYLES)
    issues = sorted(
        issues,
        key=lambda i: (
            i.get("severity", "block") != "block",
            order.index(i["status"]) if i["status"] in order else len(order),
            i.get("category", ""),
        ),
    )

    rows = []
    for issue in issues:
        icon, label, css = _COMPLETENESS_STATUS_STYLES.get(
            issue["status"], ("⚠️", issue["status"], "status-partial"),
        )
        if issue.get("severity", "block") == "block":
            label = f"{label}（关键）"
        rows.append(
            f'<tr><td>{escape_html(issue["category"])}</td>'
            f'<td>{escape_html(issue["field"])}</td>'
            f'<td class="{css}">{icon} {label}</td>'
            f'<td>{escape_html(issue.get("reason", ""))}</td></tr>'
        )

    counts: dict[str, int] = {}
    for issue in issues:
        counts[issue["status"]] = counts.get(issue["status"], 0) + 1
    summary = "、".join(
        f"{counts[status]} {_COMPLETENESS_SUMMARY_LABELS[status]}"
        for status in _COMPLETENESS_SUMMARY_LABELS
        if status in counts
    )
    blocking = sum(1 for i in issues if i.get("severity", "block") == "block")
    if blocking:
        summary += f"（其中 {blocking} 项为关键字段）"

    return f"""
<div class="section" style="background: #fff8f0; border: 2px solid #ff9800; border-left-width: 6px;">
<h2 style="color: #e65100; border-bottom-color: #ff9800;">⚠️ 数据完整性警告</h2>
<p style="margin-bottom: 12px; color: #bf360c; font-weight: 600;">
本次分析存在 {summary}，可能影响分析结论的准确性。请结合实际情况审慎参考。
</p>
<table class="data-table" style="font-size: 13px;">
<tr><th style="width:110px;">数据类别</th><th style="width:150px;">字段</th>
<th style="width:120px;">状态</th><th>说明</th></tr>
{''.join(rows)}
</table>
</div>
"""


def check_rating_action_warning(final_state: dict) -> dict | None:
    """Warn-level check: PM rating label vs its own position plan.

    Kept as a thin shim for callers outside this module. The check itself now
    runs inside the graph's ``Decision Reconciliation`` node, so a current run
    already carries it on ``integrity_findings``; this only reaches the
    computation for legacy final states — see
    :func:`tradingagents.reporting.collect_integrity_findings`.
    """
    if "integrity_findings" in final_state:
        return None
    findings = collect_integrity_findings(final_state)
    return findings[0] if findings else None


def _build_market_status_warning_banner(warnings: list[dict]) -> str:
    """Render warn-level market status findings without discarding the report."""
    if not warnings:
        return ""
    rows = "".join(
        f'<tr><td>{escape_html(w.get("section", ""))}</td>'
        f'<td>{escape_html(w.get("reason", ""))}</td>'
        f'<td>{escape_html(w.get("snippet", ""))}</td></tr>'
        for w in warnings
    )
    return f"""
<div class="section" style="background: #fff8f0; border: 2px solid #ff9800; border-left-width: 6px;">
<h2 style="color: #e65100; border-bottom-color: #ff9800;">⚠️ 市场状态提示</h2>
<p style="margin-bottom: 12px; color: #bf360c; font-weight: 600;">
以下表述与已验证市场状态存在不一致，已保留完整分析，请结合事实卡审慎参考。
</p>
<table class="data-table" style="font-size: 13px;">
<tr><th style="width:180px;">报告章节</th><th style="width:220px;">提示</th><th>相关文本</th></tr>
{rows}
</table>
</div>
"""


def _build_market_status_card(bundle: DataBundle) -> str:
    status = bundle.metadata.market_status
    conflicts = "；".join(status.conflicts) if status.conflicts else "无"
    sources = ", ".join(status.sources) if status.sources else "N/A"
    return f"""
<div class="section" id="market-status">
<h2>市场状态事实卡</h2>
<table class="metadata-table">
<tr><td>证券简称</td><td>{escape_html(status.security_name or 'N/A')}</td></tr>
<tr><td>风险警示/ST状态</td><td>{escape_html(status.risk_warning_status)}</td></tr>
<tr><td>状态生效日期</td><td>{escape_html(status.effective_date or 'N/A')}</td></tr>
<tr><td>涨跌幅限制</td><td>{escape_html(str(status.price_limit_ratio) + '%' if status.price_limit_ratio else 'N/A')}</td></tr>
<tr><td>交易状态</td><td>{escape_html(status.trading_status)}</td></tr>
<tr><td>验证时间</td><td>{escape_html(status.verified_at or 'N/A')}</td></tr>
<tr><td>验证来源</td><td>{escape_html(sources)}</td></tr>
<tr><td>置信度</td><td>{escape_html(str(status.confidence))}</td></tr>
<tr><td>来源冲突</td><td>{escape_html(conflicts)}</td></tr>
</table>
</div>
"""


def _build_model_footer(config: dict | None) -> str:
    """Record which models produced the report.

    Price targets for the same ticker have drifted hard between runs
    (182.85 → 140.00 → 180.00). Diagnosing that requires knowing which model
    and which sampling settings each report came from, and until now no report
    recorded it.
    """
    cfg = config or DEFAULT_CONFIG
    rows = [
        ("LLM 提供方", str(cfg.get("llm_provider", "—"))),
        ("深度思考模型", str(cfg.get("deep_think_llm", "—"))),
        ("快速思考模型", str(cfg.get("quick_think_llm", "—"))),
        ("辩论温度 / 决策温度", f"{cfg.get('debate_temperature', '—')} / {cfg.get('temperature', '—')}"),
        ("max_tokens", str(cfg.get("max_tokens") or "provider 默认")),
        ("辩论轮数 / 风险讨论轮数", f"{cfg.get('max_debate_rounds', '—')} / {cfg.get('max_risk_discuss_rounds', '—')}"),
        ("历史反馈注入", "启用" if cfg.get("backtest_feedback_enabled") else "关闭"),
    ]
    body = "".join(
        f"<tr><td>{escape_html(label)}</td><td>{escape_html(value)}</td></tr>"
        for label, value in rows
    )
    return f"""
<div class="section" id="run-environment">
<h3>3.3 运行环境（模型与参数）</h3>
<table class="metadata-table">
{body}
</table>
</div>
"""


def _report_timestamp_suffix(value: str | None = None) -> str:
    """Return a filesystem-safe timestamp suffix for versioned reports."""
    if value is None:
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    digits = re.sub(r"\D", "", value)
    if len(digits) >= 14:
        return f"{digits[:8]}_{digits[8:14]}"
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def generate_html_report(
    ticker: str,
    name: str,
    final_state: dict,
    bundle: DataBundle,
    output_dir: Path | None = None,
    report_timestamp: str | None = None,
    config: dict | None = None,
    allocation_item=None,
    allocation=None,
) -> Path:
    """Render one ticker's HTML report.

    ``allocation_item`` is that ticker's row from the portfolio layer
    (:func:`tradingagents.portfolio.allocator.allocate`), which only exists once
    every name in the batch has finished — so this function is called *after* the
    barrier. Omit it and the report shows the risk ceiling with no target, which
    is what a run with no position context honestly has to say. ``allocation``
    adds the portfolio totals alongside it.
    """
    meta = bundle.metadata
    trade_date = meta.trade_date
    original_date = meta.original_trade_date
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    asset_label = "ETF" if is_etf(ticker) else "股票"

    correction_reason = getattr(meta, 'date_correction_reason', '')
    date_note = date_correction_note(correction_reason, original_date, trade_date)

    # Computed from the bundle rather than read off the final state so that a
    # legacy state (saved before the graph emitted completeness findings) still
    # gets the full per-field table.
    completeness_issues = _check_data_completeness(bundle)
    completeness_banner = _build_completeness_banner(completeness_issues)
    blocking_conflicts, status_warnings = classify_market_status_conflicts(final_state, bundle)
    if blocking_conflicts:
        raise MarketStatusConflictError(blocking_conflicts)
    status_warning_banner = _build_market_status_warning_banner(status_warnings)
    # Run-integrity findings (empty LLM output, structured-output degradation,
    # cross-stage override, high conviction on an incomplete bar) get their own
    # banner rather than being folded into the market-status one: they are about
    # how the analysis was produced, not about what it claims.
    # Data-completeness findings are dropped here because the banner above
    # already renders them per field, with the reason text.
    _, run_findings = split_data_completeness_findings(
        collect_integrity_findings(final_state)
    )
    integrity_banner = build_integrity_banner(run_findings)
    market_status_card = _build_market_status_card(bundle)

    data_tables_html = build_data_tables(bundle)
    decision_html = build_decision_section(final_state, allocation_item, allocation)
    analysis_html = build_analysis_sections(final_state)
    model_footer = _build_model_footer(config)

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

{status_warning_banner}

{integrity_banner}

{market_status_card}

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
<tr><td>是否校正</td><td>{date_correction_label(correction_reason, original_date)}</td></tr>
<tr><td>采集时间</td><td>{escape_html(meta.collection_timestamp)}</td></tr>
<tr><td>数据版本</td><td>{escape_html(meta.bundle_version)}</td></tr>
<tr><td>选中分析师</td><td>{escape_html(', '.join(meta.selected_analysts))}</td></tr>
<tr><td>风险警示/ST状态</td><td>{escape_html(meta.market_status.risk_warning_status)}</td></tr>
<tr><td>证券简称（状态源）</td><td>{escape_html(meta.market_status.security_name or 'N/A')}</td></tr>
<tr><td>状态验证来源</td><td>{escape_html(', '.join(meta.market_status.sources) if meta.market_status.sources else 'N/A')}</td></tr>
</table>
</div>

{model_footer}

{data_tables_html}

</div>
</body>
</html>"""

    safe_name = ticker.replace(".", "_")
    timestamp = _report_timestamp_suffix(report_timestamp)
    _out_dir = output_dir if output_dir is not None else OUTPUT_DIR
    _out_dir.mkdir(exist_ok=True)
    output_path = _out_dir / f"{safe_name}_{timestamp}_report.html"
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
            report_path = generate_html_report(
                ticker, name, final_state, bundle, config=_make_config(),
            )
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
