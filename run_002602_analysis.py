"""End-to-end 002602.SZ (世纪华通) analysis: collect data -> run analysis -> generate HTML report.

Usage:
    python run_002602_analysis.py [--collect-only] [--bundle PATH] [--date YYYY-MM-DD]
"""

import json
import logging
import sys
import os
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tradingagents.datacollector import DataBundle, DataCollector
from tradingagents.default_config import DEFAULT_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "test_output"
OUTPUT_DIR.mkdir(exist_ok=True)

TICKER = "002602.SZ"
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

    return bundle, filepath


def run_analysis(bundle: DataBundle):
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    config = DEFAULT_CONFIG.copy()
    config["max_debate_rounds"] = 1
    config["max_risk_discuss_rounds"] = 1
    config["output_language"] = "Chinese"

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

    state_path = OUTPUT_DIR / f"{TICKER.replace('.', '_')}_final_state.json"
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
    from test_output.run_baba_analysis import (
        build_data_tables,
        build_decision_section,
        build_analysis_sections,
    )

    meta = bundle.metadata
    ticker = meta.ticker
    trade_date = meta.trade_date
    original_date = meta.original_trade_date
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    date_note = ""
    if original_date:
        date_note = f'<p class="date-correction">⚠️ 用户输入日期 <strong>{escape_html(original_date)}</strong> 为非交易日，已自动校正为 <strong>{escape_html(trade_date)}</strong></p>'

    data_tables_html = build_data_tables(bundle)
    decision_html = build_decision_section(final_state)
    analysis_html = build_analysis_sections(final_state)

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
<tr><td>股票代码</td><td>{escape_html(ticker)}</td></tr>
<tr><td>交易日（校正后）</td><td>{escape_html(trade_date)}</td></tr>
<tr><td>用户输入日期</td><td>{escape_html(original_date or trade_date)}</td></tr>
<tr><td>是否校正</td><td>{"是" if original_date else "否（输入即为交易日）"}</td></tr>
<tr><td>采集时间</td><td>{escape_html(meta.collection_timestamp)}</td></tr>
<tr><td>数据版本</td><td>{escape_html(meta.bundle_version)}</td></tr>
<tr><td>选中分析师</td><td>{escape_html(', '.join(meta.selected_analysts))}</td></tr>
</table>
</div>

{data_tables_html}

</div>
</body>
</html>"""

    output_path = OUTPUT_DIR / f"{ticker.replace('.', '_')}_report.html"
    output_path.write_text(html, encoding="utf-8")
    logger.info("HTML report saved to %s", output_path)
    return output_path


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
        bundle_path = OUTPUT_DIR / f"{TICKER.replace('.', '_')}_{bundle.metadata.trade_date}_data.json"
        DataCollector.save(bundle, bundle_path)
        logger.info("Data collection complete. Bundle: %s", bundle_path)
        sys.exit(0)

    final_state = run_analysis(bundle)
    report_path = generate_html_report(final_state, bundle)
    print(f"\n{'='*60}")
    print(f"Report generated: {report_path}")
    print(f"{'='*60}")
