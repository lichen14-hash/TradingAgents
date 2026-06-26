"""Daily runner for backtest data accumulation.

Usage:
    py -m tradingagents.backtest.daily_runner
    py -m tradingagents.backtest.daily_runner --resolve-only
    py -m tradingagents.backtest.daily_runner --analyze-only
    py -m tradingagents.backtest.daily_runner --date 2026-06-26
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def _fetch_returns(
    ticker: str,
    trade_date: str,
    holding_days: int = 5,
    benchmark: str = "SPY",
) -> tuple[float | None, float | None, int | None]:
    """Compute actual returns for a ticker over a holding period.

    Replicates the logic from TradingAgentsGraph._fetch_returns without
    requiring a full graph instance (no LLM clients needed).
    """
    from tradingagents.dataflows.market_utils import is_a_share

    try:
        start = datetime.strptime(trade_date, "%Y-%m-%d")
        end = start + timedelta(days=holding_days + 7)
        end_str = end.strftime("%Y-%m-%d")

        if is_a_share(ticker):
            from tradingagents.dataflows.stockstats_utils import load_ohlcv

            stock_df = load_ohlcv(ticker, end_str)
            bench_df = load_ohlcv(benchmark, end_str)
            stock_df = stock_df[stock_df["Date"] >= start].head(holding_days + 1)
            bench_df = bench_df[bench_df["Date"] >= start].head(holding_days + 1)

            if len(stock_df) < 2 or len(bench_df) < 2:
                return None, None, None

            actual_days = min(holding_days, len(stock_df) - 1, len(bench_df) - 1)
            raw = float(
                (stock_df["Close"].iloc[actual_days] - stock_df["Close"].iloc[0])
                / stock_df["Close"].iloc[0]
            )
            bench_ret = float(
                (bench_df["Close"].iloc[actual_days] - bench_df["Close"].iloc[0])
                / bench_df["Close"].iloc[0]
            )
        else:
            import yfinance as yf
            from tradingagents.dataflows.symbol_utils import normalize_symbol

            stock = yf.Ticker(normalize_symbol(ticker)).history(start=trade_date, end=end_str)
            bench = yf.Ticker(benchmark).history(start=trade_date, end=end_str)

            if len(stock) < 2 or len(bench) < 2:
                return None, None, None

            actual_days = min(holding_days, len(stock) - 1, len(bench) - 1)
            raw = float(
                (stock["Close"].iloc[actual_days] - stock["Close"].iloc[0])
                / stock["Close"].iloc[0]
            )
            bench_ret = float(
                (bench["Close"].iloc[actual_days] - bench["Close"].iloc[0])
                / bench["Close"].iloc[0]
            )

        return raw, raw - bench_ret, actual_days
    except Exception as e:
        logger.warning("Could not resolve outcome for %s on %s: %s", ticker, trade_date, e)
        return None, None, None


def _resolve_benchmark(ticker: str, config: dict) -> str:
    explicit = config.get("benchmark_ticker")
    if explicit:
        return explicit
    benchmark_map = config.get("benchmark_map", {})
    ticker_upper = ticker.upper()
    for suffix, benchmark in benchmark_map.items():
        if suffix and ticker_upper.endswith(suffix.upper()):
            return benchmark
    return benchmark_map.get("", "SPY")


class DailyRunner:
    def __init__(self, config: dict | None = None):
        from tradingagents.default_config import DEFAULT_CONFIG

        self.config = config or DEFAULT_CONFIG
        db_path = self.config.get("backtest_db_path")
        if not db_path:
            raise RuntimeError("backtest_db_path not set in config")

        from .db import BacktestDB
        from .store import BacktestStore

        self.db = BacktestDB(db_path)
        self.db.migrate()
        self.store = BacktestStore(self.db)

    def resolve_pending_outcomes(self) -> int:
        """Resolve all predictions whose holding period has elapsed."""
        holding_days = self.config.get("backtest_holding_days", 5)
        threshold = self.config.get("backtest_direction_threshold", 0.02)
        pending = self.store.get_pending_predictions()

        today = datetime.now()
        resolved = 0

        for pred in pending:
            trade_dt = datetime.strptime(pred["trade_date"], "%Y-%m-%d")
            if (today - trade_dt).days < holding_days + 2:
                continue

            benchmark = _resolve_benchmark(pred["ticker"], self.config)
            raw, alpha, days = _fetch_returns(
                pred["ticker"], pred["trade_date"],
                holding_days=holding_days, benchmark=benchmark,
            )
            if raw is None:
                logger.info("Cannot resolve %s on %s yet, will retry", pred["ticker"], pred["trade_date"])
                continue

            reflection = ""
            try:
                from tradingagents.graph.reflection import Reflector
                from tradingagents.llm_clients import create_llm_client

                client = create_llm_client(
                    provider=self.config["llm_provider"],
                    model=self.config["quick_think_llm"],
                    base_url=self.config.get("backend_url"),
                )
                reflector = Reflector(client.get_llm())
                conn = self.db.get_connection()
                row = conn.execute(
                    "SELECT executive_summary FROM predictions WHERE id = ?",
                    (pred["id"],),
                ).fetchone()
                decision_text = row["executive_summary"] or "" if row else ""
                reflection = reflector.reflect_on_final_decision(
                    final_decision=decision_text,
                    raw_return=raw,
                    alpha_return=alpha,
                    benchmark_name=benchmark,
                )
            except Exception:
                logger.warning("Reflection generation failed for %s", pred["ticker"], exc_info=True)

            self.store.resolve_outcome(
                prediction_id=pred["id"],
                raw_return=raw,
                alpha_return=alpha,
                benchmark=benchmark,
                actual_days=days,
                reflection=reflection,
                threshold=threshold,
            )
            resolved += 1
            logger.info(
                "Resolved %s on %s: return=%.2f%% alpha=%.2f%%",
                pred["ticker"], pred["trade_date"], raw * 100, alpha * 100,
            )

        return resolved

    def run_watchlist_analysis(self, trade_date: str | None = None) -> tuple[int, int]:
        """Run analysis for each active watchlist ticker. Returns (succeeded, failed)."""
        from tradingagents.datacollector import DataCollector
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        watchlist = self.store.get_active_watchlist()
        if not watchlist:
            logger.info("Watchlist is empty, nothing to analyze")
            return 0, 0

        trade_date = trade_date or datetime.now().strftime("%Y-%m-%d")
        output_dir = ROOT_DIR / "test_output"
        output_dir.mkdir(exist_ok=True)

        run_id = self.store.start_daily_run(trade_date, len(watchlist))
        succeeded, failed = 0, 0
        errors = []

        for item in watchlist:
            ticker = item["ticker"]
            logger.info("Analyzing %s (%s) for %s", ticker, item.get("name", ""), trade_date)
            try:
                collector = DataCollector(self.config)
                bundle, _ = collector.collect_and_save(
                    ticker, trade_date,
                    selected_analysts=("market", "social", "news", "fundamentals"),
                    save_dir=output_dir,
                )

                graph = TradingAgentsGraph(
                    selected_analysts=("market", "social", "news", "fundamentals"),
                    config=self.config,
                )
                final_state, signal = graph.propagate(
                    ticker, trade_date, data_bundle=bundle,
                )

                try:
                    sys.path.insert(0, str(ROOT_DIR))
                    from run_batch_analysis import generate_html_report
                    generate_html_report(ticker, item.get("name", ticker), final_state, bundle)
                except Exception:
                    logger.warning("HTML report generation failed for %s", ticker, exc_info=True)

                succeeded += 1
                logger.info("Completed %s: signal=%s", ticker, signal)
            except Exception as e:
                failed += 1
                errors.append(f"{ticker}: {e}")
                logger.error("Failed %s: %s", ticker, e, exc_info=True)

        self.store.finish_daily_run(run_id, succeeded, failed, "\n".join(errors))
        return succeeded, failed


def main():
    from dotenv import load_dotenv
    load_dotenv()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="TradingAgents daily backtest runner")
    parser.add_argument("--resolve-only", action="store_true", help="Only resolve pending outcomes")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze watchlist")
    parser.add_argument("--date", type=str, default=None, help="Trade date (YYYY-MM-DD)")
    args = parser.parse_args()

    from tradingagents.default_config import DEFAULT_CONFIG

    runner = DailyRunner(DEFAULT_CONFIG)

    if not args.analyze_only:
        resolved = runner.resolve_pending_outcomes()
        logger.info("Resolved %d pending predictions", resolved)

    if not args.resolve_only:
        ok, fail = runner.run_watchlist_analysis(trade_date=args.date)
        logger.info("Analysis complete: %d succeeded, %d failed", ok, fail)


if __name__ == "__main__":
    main()
