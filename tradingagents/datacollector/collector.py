"""Standalone data collector — pre-fetches all data needed for analysis."""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.interface import route_to_vendor
from tradingagents.dataflows.market_data_validator import build_verified_market_snapshot
from tradingagents.dataflows.market_utils import is_a_share, is_hk_stock
from tradingagents.dataflows.reddit import fetch_reddit_posts
from tradingagents.dataflows.stockstats_utils import load_ohlcv
from tradingagents.dataflows.stocktwits import fetch_stocktwits_messages
from tradingagents.dataflows.utils import safe_ticker_component
from tradingagents.default_config import DEFAULT_CONFIG

from .constants import (
    ALL_INDICATORS,
    BUNDLE_VERSION,
    CN_MACRO_INDICATORS,
    CN_PREDICTION_QUERIES,
    DEFAULT_MACRO_INDICATORS,
    DEFAULT_PREDICTION_QUERIES,
    HK_MACRO_INDICATORS,
    HK_PREDICTION_QUERIES,
)
from .schema import (
    BundleMetadata,
    DataBundle,
    FundamentalsData,
    MarketData,
    NewsData,
    SentimentData,
)

logger = logging.getLogger(__name__)

_UNAVAILABLE_PREFIX = "<unavailable: "


def _unavailable(reason: str) -> str:
    return f"{_UNAVAILABLE_PREFIX}{reason}>"


def _safe_call(label: str, func, *args, **kwargs) -> str:
    try:
        result = func(*args, **kwargs)
        return result if isinstance(result, str) else str(result)
    except Exception as e:
        logger.warning("Data collection failed for %s: %s", label, e)
        return _unavailable(f"{type(e).__name__}: {e}")


def _resolve_trading_date(ticker: str, trade_date: str) -> tuple[str, pd.DataFrame, str]:
    """Roll *trade_date* back to the most recent actual trading day.

    Returns ``(corrected_date, ohlcv_df, reason)`` where *reason* is one of:
    - ``""``              — no correction needed
    - ``"non_trading_day"`` — weekend or holiday
    - ``"data_not_ready"``  — trading day but data source hasn't updated yet
    """
    df = load_ohlcv(ticker, trade_date)
    if df is None or df.empty:
        raise ValueError(f"No OHLCV data for {ticker}, cannot resolve trading date")
    latest_date = df["Date"].max()
    corrected = pd.to_datetime(latest_date).strftime("%Y-%m-%d")

    if corrected == trade_date:
        return corrected, df, ""

    input_dt = pd.to_datetime(trade_date)
    if input_dt.weekday() >= 5:
        return corrected, df, "non_trading_day"

    today = pd.Timestamp.today().normalize()
    if input_dt == today:
        return corrected, df, "data_not_ready"

    return corrected, df, "non_trading_day"


def _validate_market_date(value: str, corrected_date: str, label: str) -> str:
    """Return *value* only if it contains *corrected_date*; otherwise unavailable."""
    if value.startswith(_UNAVAILABLE_PREFIX):
        return value
    if corrected_date not in value:
        return _unavailable(f"{label}: no data matching trade date {corrected_date}")
    return value


class DataCollector:
    """Pre-fetch all data needed for a TradingAgents analysis run."""

    def __init__(self, config: dict | None = None):
        self.config = config or DEFAULT_CONFIG.copy()
        set_config(self.config)

    def collect(
        self,
        ticker: str,
        trade_date: str,
        asset_type: str = "stock",
        selected_analysts: tuple[str, ...] | list[str] = (
            "market", "social", "news", "fundamentals",
        ),
    ) -> DataBundle:
        selected = set(selected_analysts)
        logger.info("Collecting data for %s on %s (analysts: %s)", ticker, trade_date, selected)

        original_trade_date: str | None = None
        date_correction_reason: str = ""
        ohlcv_df: pd.DataFrame | None = None
        try:
            corrected, ohlcv_df, reason = _resolve_trading_date(ticker, trade_date)
            if corrected != trade_date:
                logger.info("Trading date corrected: %s → %s (%s)", trade_date, corrected, reason)
                original_trade_date = trade_date
                date_correction_reason = reason
                trade_date = corrected
        except Exception as e:
            logger.warning("Trading date resolution failed: %s, using original", e)

        metadata = BundleMetadata(
            ticker=ticker,
            trade_date=trade_date,
            original_trade_date=original_trade_date,
            date_correction_reason=date_correction_reason,
            asset_type=asset_type,
            collection_timestamp=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            selected_analysts=sorted(selected),
            vendor_config={
                "data_vendors": self.config.get("data_vendors", {}),
                "tool_vendors": self.config.get("tool_vendors", {}),
            },
            bundle_version=BUNDLE_VERSION,
        )

        market = self._collect_market_data(ticker, trade_date, ohlcv_df=ohlcv_df) if "market" in selected else None
        sentiment = self._collect_sentiment_data(ticker, trade_date) if "social" in selected else None
        news = self._collect_news_data(ticker, trade_date) if "news" in selected else None
        fundamentals = self._collect_fundamentals_data(ticker, trade_date) if "fundamentals" in selected else None

        bundle = DataBundle(
            metadata=metadata,
            market=market,
            sentiment=sentiment,
            news=news,
            fundamentals=fundamentals,
        )
        logger.info("Data collection complete for %s on %s", ticker, trade_date)
        return bundle

    def collect_and_save(
        self,
        ticker: str,
        trade_date: str,
        save_dir: str | Path | None = None,
        **kwargs,
    ) -> tuple[DataBundle, Path]:
        bundle = self.collect(ticker, trade_date, **kwargs)
        corrected_date = bundle.metadata.trade_date
        if save_dir:
            filepath = Path(save_dir) / self._filename(ticker, corrected_date)
        else:
            filepath = self._default_save_path(ticker, corrected_date)
        saved = self.save(bundle, filepath)
        return bundle, saved

    @staticmethod
    def save(bundle: DataBundle, filepath: str | Path) -> Path:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(bundle.model_dump(), f, indent=2, ensure_ascii=False)

        latest = filepath.parent / _latest_name(bundle.metadata.ticker, bundle.metadata.trade_date)
        _update_latest(filepath, latest)

        logger.info("Data bundle saved to %s", filepath)
        return filepath

    @staticmethod
    def load(filepath: str | Path) -> DataBundle:
        filepath = Path(filepath)
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        return DataBundle.model_validate(data)

    # ------------------------------------------------------------------
    # Internal collection methods
    # ------------------------------------------------------------------

    def _collect_market_data(
        self,
        ticker: str,
        trade_date: str,
        ohlcv_df: pd.DataFrame | None = None,
    ) -> MarketData:
        start_date = _lookback_date(trade_date, 30)

        stock_data = _safe_call(
            "stock_data",
            route_to_vendor, "get_stock_data", ticker, start_date, trade_date,
        )
        stock_data = _validate_market_date(stock_data, trade_date, "stock_data")

        indicators: dict[str, str] = {}
        for ind in ALL_INDICATORS:
            val = _safe_call(
                f"indicator:{ind}",
                route_to_vendor, "get_indicators", ticker, ind, trade_date, 30,
            )
            indicators[ind] = _validate_market_date(val, trade_date, f"indicator:{ind}")

        verified_snapshot = _safe_call(
            "verified_snapshot",
            build_verified_market_snapshot, ticker, trade_date,
            preloaded_ohlcv=ohlcv_df,
        )

        return MarketData(
            stock_data=stock_data,
            indicators=indicators,
            verified_snapshot=verified_snapshot,
        )

    def _collect_sentiment_data(self, ticker: str, trade_date: str) -> SentimentData:
        start_date = _lookback_date(trade_date, 7)

        ticker_news = _safe_call(
            "sentiment:news",
            route_to_vendor, "get_news", ticker, start_date, trade_date,
        )

        if is_a_share(ticker) or is_hk_stock(ticker):
            from tradingagents.dataflows.eastmoney import (
                fetch_eastmoney_guba,
                fetch_sina_finance_comments,
            )
            stocktwits = _safe_call(
                "sentiment:eastmoney_guba",
                fetch_eastmoney_guba, ticker, 30,
            )
            reddit = _safe_call(
                "sentiment:sina_comments",
                fetch_sina_finance_comments, ticker,
            )
        else:
            stocktwits = _safe_call(
                "sentiment:stocktwits",
                fetch_stocktwits_messages, ticker, 30,
            )
            reddit = _safe_call(
                "sentiment:reddit",
                fetch_reddit_posts, ticker,
            )

        return SentimentData(
            ticker_news=ticker_news,
            stocktwits=stocktwits,
            reddit=reddit,
        )

    def _collect_news_data(self, ticker: str, trade_date: str) -> NewsData:
        start_date = _lookback_date(trade_date, 7)
        lookback = self.config.get("global_news_lookback_days", 7)
        limit = self.config.get("global_news_article_limit", 10)

        ticker_news = _safe_call(
            "news:ticker",
            route_to_vendor, "get_news", ticker, start_date, trade_date,
        )

        if is_a_share(ticker):
            from tradingagents.dataflows.akshare_provider import get_global_news as _ak_global_news
            global_news = _safe_call(
                "news:global(akshare)",
                _ak_global_news, trade_date, lookback, limit,
            )
        elif is_hk_stock(ticker):
            from tradingagents.dataflows.hk_akshare_provider import get_global_news as _hk_global_news
            global_news = _safe_call(
                "news:global(hk_akshare)",
                _hk_global_news, trade_date, lookback, limit,
            )
        else:
            global_news = _safe_call(
                "news:global",
                route_to_vendor, "get_global_news", trade_date, lookback, limit,
            )

        insider = _safe_call(
            "news:insider",
            route_to_vendor, "get_insider_transactions", ticker,
        )

        if is_a_share(ticker):
            macro_list = self.config.get("cn_macro_indicators", list(CN_MACRO_INDICATORS))
        elif is_hk_stock(ticker):
            macro_list = self.config.get("hk_macro_indicators", list(HK_MACRO_INDICATORS))
        else:
            macro_list = self.config.get("standard_macro_indicators", list(DEFAULT_MACRO_INDICATORS))
        macro: dict[str, str] = {}
        if is_a_share(ticker):
            from tradingagents.dataflows.china_macro import get_cn_macro_data
            for ind in macro_list:
                macro[ind] = _safe_call(
                    f"macro:{ind}", get_cn_macro_data, ind, trade_date, None,
                )
        elif is_hk_stock(ticker):
            from tradingagents.dataflows.hk_macro import get_hk_macro_data
            for ind in macro_list:
                macro[ind] = _safe_call(
                    f"macro:{ind}", get_hk_macro_data, ind, trade_date, None,
                )
        else:
            for ind in macro_list:
                macro[ind] = _safe_call(
                    f"macro:{ind}",
                    route_to_vendor, "get_macro_indicators", ind, trade_date, None,
                )

        if is_a_share(ticker):
            pred_queries = self.config.get("cn_prediction_queries", list(CN_PREDICTION_QUERIES))
        elif is_hk_stock(ticker):
            pred_queries = self.config.get("hk_prediction_queries", list(HK_PREDICTION_QUERIES))
        else:
            pred_queries = self.config.get("standard_prediction_queries", list(DEFAULT_PREDICTION_QUERIES))
        predictions: dict[str, str] = {}
        if is_a_share(ticker):
            from tradingagents.dataflows.cn_market_signals import get_cn_market_signals
            for query in pred_queries:
                predictions[query] = _safe_call(
                    f"cn_signal:{query}", get_cn_market_signals, query, None,
                )
        elif is_hk_stock(ticker):
            from tradingagents.dataflows.hk_market_signals import get_hk_market_signals
            for query in pred_queries:
                predictions[query] = _safe_call(
                    f"hk_signal:{query}", get_hk_market_signals, query, None,
                )
        else:
            for query in pred_queries:
                predictions[query] = _safe_call(
                    f"prediction:{query}",
                    route_to_vendor, "get_prediction_markets", query, None,
                )

        return NewsData(
            ticker_news=ticker_news,
            global_news=global_news,
            insider_transactions=insider,
            macro_indicators=macro,
            prediction_markets=predictions,
        )

    def _collect_fundamentals_data(self, ticker: str, trade_date: str) -> FundamentalsData:
        overview = _safe_call(
            "fundamentals:overview",
            route_to_vendor, "get_fundamentals", ticker, trade_date,
        )
        bs_q = _safe_call(
            "fundamentals:balance_sheet_q",
            route_to_vendor, "get_balance_sheet", ticker, "quarterly", trade_date,
        )
        bs_a = _safe_call(
            "fundamentals:balance_sheet_a",
            route_to_vendor, "get_balance_sheet", ticker, "annual", trade_date,
        )
        cf_q = _safe_call(
            "fundamentals:cashflow_q",
            route_to_vendor, "get_cashflow", ticker, "quarterly", trade_date,
        )
        cf_a = _safe_call(
            "fundamentals:cashflow_a",
            route_to_vendor, "get_cashflow", ticker, "annual", trade_date,
        )
        inc_q = _safe_call(
            "fundamentals:income_q",
            route_to_vendor, "get_income_statement", ticker, "quarterly", trade_date,
        )
        inc_a = _safe_call(
            "fundamentals:income_a",
            route_to_vendor, "get_income_statement", ticker, "annual", trade_date,
        )

        return FundamentalsData(
            overview=overview,
            balance_sheet_quarterly=bs_q,
            balance_sheet_annual=bs_a,
            cashflow_quarterly=cf_q,
            cashflow_annual=cf_a,
            income_quarterly=inc_q,
            income_annual=inc_a,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _default_save_path(self, ticker: str, trade_date: str) -> Path:
        data_dir = Path(self.config.get("data_dir", os.path.expanduser("~/.tradingagents/data")))
        safe_ticker = safe_ticker_component(ticker)
        return data_dir / safe_ticker / self._filename(ticker, trade_date)

    @staticmethod
    def _filename(ticker: str, trade_date: str) -> str:
        safe_ticker = safe_ticker_component(ticker)
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        return f"{safe_ticker}_{trade_date}_{ts}.json"


def _lookback_date(trade_date: str, days: int) -> str:
    dt = datetime.strptime(trade_date, "%Y-%m-%d")
    return (dt - timedelta(days=days)).strftime("%Y-%m-%d")


def _latest_name(ticker: str, trade_date: str) -> str:
    safe_ticker = safe_ticker_component(ticker)
    return f"{safe_ticker}_{trade_date}_latest.json"


def _update_latest(source: Path, latest: Path) -> None:
    try:
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        if platform.system() == "Windows":
            shutil.copy2(str(source), str(latest))
        else:
            latest.symlink_to(source.name)
    except OSError as e:
        logger.warning("Could not update latest link %s: %s", latest, e)
