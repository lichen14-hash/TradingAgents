"""Data collection script for Qoder Skill — no LLM dependency.

Usage:
    python scripts/collect_data_for_skill.py <ticker> [--date YYYY-MM-DD] [--output-dir DIR]

Example:
    python scripts/collect_data_for_skill.py 002602.SZ --date 2026-06-29
    python scripts/collect_data_for_skill.py 09988.HK

Outputs a JSON DataBundle file and prints the file path to stdout.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tradingagents.datacollector import DataCollector, validate_bundle_completeness
from tradingagents.dataflows.market_utils import is_etf
from tradingagents.default_config import DEFAULT_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "skill_output"


def get_analysts_for_ticker(ticker: str) -> tuple[str, ...]:
    """Determine which analysts to use based on ticker type."""
    if is_etf(ticker):
        return ("market", "social", "news")
    return ("market", "social", "news", "fundamentals")


def main():
    parser = argparse.ArgumentParser(description="Collect market data for Qoder Skill analysis")
    parser.add_argument("ticker", help="Stock/ETF ticker symbol (e.g. 002602.SZ, 09988.HK)")
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"),
                        help="Trade date (default: today)")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                        help="Output directory for data bundle")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = DEFAULT_CONFIG.copy()
    config["max_debate_rounds"] = 1
    config["max_risk_discuss_rounds"] = 1
    config["output_language"] = "Chinese"

    collector = DataCollector(config)
    analysts = get_analysts_for_ticker(args.ticker)

    logger.info("Collecting data for %s on %s (analysts: %s)", args.ticker, args.date, analysts)

    bundle, filepath = collector.collect_and_save(
        args.ticker,
        args.date,
        selected_analysts=analysts,
        save_dir=output_dir,
    )

    # Validate data completeness — abort if any field is unavailable
    issues = validate_bundle_completeness(bundle)
    if issues:
        detail = [{"category": i["category"], "field": i["field"], "reason": i["reason"][:120]} for i in issues]
        result = {
            "status": "incomplete",
            "ticker": args.ticker,
            "trade_date": bundle.metadata.trade_date,
            "bundle_path": str(filepath),
            "issues_count": len(issues),
            "issues": detail,
        }
        print(json.dumps(result, ensure_ascii=False))
        sys.exit(1)

    # Output result as JSON for Skill to parse
    result = {
        "status": "success",
        "ticker": args.ticker,
        "trade_date": bundle.metadata.trade_date,
        "original_trade_date": bundle.metadata.original_trade_date,
        "date_correction_reason": bundle.metadata.date_correction_reason,
        "asset_type": bundle.metadata.asset_type,
        "selected_analysts": bundle.metadata.selected_analysts,
        "bundle_path": str(filepath),
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
