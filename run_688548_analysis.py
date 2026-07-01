"""Run full analysis for 688548.SS (广钢气体) with cost_price=35.6"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tradingagents.datacollector import DataBundle, DataCollector
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "test_output"
OUTPUT_DIR.mkdir(exist_ok=True)

TICKER = "688548.SS"
TRADE_DATE = "2026-06-30"
COST_PRICE = 35.6


def _make_config() -> dict:
    config = DEFAULT_CONFIG.copy()
    config["output_language"] = "Chinese"
    # Use env-based debate rounds (3 rounds from .env)
    return config


def _format_position_context(ticker: str, cost_price: float) -> str:
    parts = [f"用户当前持有 {ticker} 的仓位信息："]
    parts.append(f"- 持仓成本价: {cost_price}")
    parts.append("请结合用户的实际成本和仓位，给出针对性的操作建议（如浮盈/浮亏幅度、是否止盈止损、是否加仓减仓等）。")
    return "\n".join(parts)


def main():
    config = _make_config()
    analysts = ("market", "social", "news", "fundamentals")

    # Load pre-collected data bundle
    bundle_path = Path("skill_output/688548.SS_2026-06-30_20260630T123116.json")
    if not bundle_path.exists():
        logger.error("Data bundle not found at %s, run collect first", bundle_path)
        sys.exit(1)

    logger.info("Loading data bundle from %s", bundle_path)
    bundle = DataCollector.load(str(bundle_path))

    # Build graph
    graph = TradingAgentsGraph(
        selected_analysts=analysts,
        config=config,
        debug=True,
    )

    # Build portfolio context
    portfolio_ctx = _format_position_context(TICKER, COST_PRICE)
    instrument_context = graph.resolve_instrument_context(TICKER)

    # Create initial state with user_portfolio_context
    init_state = graph.propagator.create_initial_state(
        TICKER,
        bundle.metadata.trade_date,
        instrument_context=instrument_context,
        data_bundle=bundle.model_dump(),
        user_portfolio_context=portfolio_ctx,
    )
    args = graph.propagator.get_graph_args()

    # Stream analysis
    logger.info("Starting analysis for %s on %s (cost=%.2f) ...", TICKER, TRADE_DATE, COST_PRICE)
    trace = []
    for chunk in graph.graph.stream(init_state, **args):
        trace.append(chunk)
        # Log progress
        if chunk.get("market_report"):
            logger.info("Market analyst complete")
        if chunk.get("sentiment_report"):
            logger.info("Sentiment analyst complete")
        if chunk.get("news_report"):
            logger.info("News analyst complete")
        if chunk.get("fundamentals_report"):
            logger.info("Fundamentals analyst complete")
        inv_state = chunk.get("investment_debate_state")
        if inv_state and inv_state.get("judge_decision"):
            logger.info("Investment debate concluded")
        if chunk.get("trader_investment_plan"):
            logger.info("Trader plan ready")
        risk_state = chunk.get("risk_debate_state")
        if risk_state and risk_state.get("judge_decision"):
            logger.info("Risk debate concluded")
        if chunk.get("final_trade_decision"):
            logger.info("Final decision ready")

    # Get final state
    final_state = trace[-1] if trace else {}
    signal = final_state.get("final_trade_decision", "N/A")
    logger.info("Analysis complete. Signal preview: %s", signal[:200] if isinstance(signal, str) else signal)

    # Save final state
    safe_name = TICKER.replace(".", "_")
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

    # Generate HTML report
    try:
        from run_batch_analysis import generate_html_report
        html_path = generate_html_report(TICKER, "广钢气体", final_state, bundle)
        logger.info("HTML report: %s", html_path)
    except Exception as e:
        logger.warning("HTML report generation failed: %s", e)

    # Generate PDF
    try:
        from generate_report import generate_pdf
        pdf_path = OUTPUT_DIR / f"{safe_name}_report.pdf"
        html_report = OUTPUT_DIR / f"{safe_name}_report.html"
        if html_report.exists():
            generate_pdf(str(html_report), str(pdf_path))
            logger.info("PDF report: %s", pdf_path)
    except Exception as e:
        logger.warning("PDF generation failed: %s", e)

    print("\n" + "="*60)
    print(f"分析完成: {TICKER} ({TRADE_DATE})")
    print(f"成本价: {COST_PRICE}")
    print(f"最终决策:\n{signal[:500] if isinstance(signal, str) else signal}")
    print("="*60)


if __name__ == "__main__":
    main()
