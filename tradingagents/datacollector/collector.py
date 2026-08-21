"""Standalone data collector — pre-fetches all data needed for analysis."""

from __future__ import annotations

import json
import logging
import platform
import re
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from tradingagents.dataflows.competitive_intel import get_competitive_intelligence
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.interface import route_to_vendor
from tradingagents.dataflows.market_data_validator import build_verified_market_snapshot
from tradingagents.dataflows.market_utils import is_a_share, is_hk_stock
from tradingagents.dataflows.reddit import fetch_reddit_posts
from tradingagents.dataflows.st_status_provider import (
    annotate_historical_st_mentions,
    resolve_market_status,
)
from tradingagents.dataflows.stockstats_utils import load_ohlcv
from tradingagents.dataflows.stocktwits import fetch_stocktwits_messages
from tradingagents.dataflows.utils import safe_ticker_component
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.utils.time_utils import now, now_iso, pd_today

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

    today = pd_today()
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


def _market_close_minutes(ticker: str) -> int:
    if is_hk_stock(ticker):
        return 16 * 60 + 15
    if is_a_share(ticker):
        return 15 * 60
    return 5 * 60


def _force_last_complete_daily_bar(
    ticker: str,
    requested_trade_date: str,
    resolved_trade_date: str,
    ohlcv_df: pd.DataFrame | None,
) -> tuple[str, pd.DataFrame | None, str]:
    """During live sessions, avoid treating today's partial daily bar as final.

    Returns ``"intraday_daily_bar_incomplete"`` whenever the run is asking about
    *today* while the session is still open, regardless of whether the vendor
    has published a (partial) bar for today:

    - vendor published today's bar → roll back to the previous session below;
    - vendor hasn't published it yet → ``_resolve_trading_date`` already rolled
      back and labelled it ``"data_not_ready"``, which is true but describes the
      vendor rather than the decision. Relabelling matters because the
      conviction guard keyed on this reason
      (:func:`tradingagents.agents.utils.integrity.check_conviction_vs_data_quality`)
      used to see ``data_not_ready`` and stay silent: in the 2026-08-19 intraday
      batch all 8 bundles were labelled ``data_not_ready`` and 6 of 7 tickers
      still came out with a Sell (>60% position change) on a mid-session view.
    """
    try:
        requested_dt = pd.to_datetime(requested_trade_date)
    except Exception:
        return resolved_trade_date, ohlcv_df, ""
    today = pd_today()
    if requested_dt != today:
        return resolved_trade_date, ohlcv_df, ""

    current = now()
    current_minutes = current.hour * 60 + current.minute
    if current_minutes >= _market_close_minutes(ticker):
        return resolved_trade_date, ohlcv_df, ""

    if resolved_trade_date != today.strftime("%Y-%m-%d"):
        # Already on the last complete session; nothing to roll back, but the
        # run is still looking at today mid-session.
        return resolved_trade_date, ohlcv_df, "intraday_daily_bar_incomplete"

    if ohlcv_df is None or ohlcv_df.empty or "Date" not in ohlcv_df.columns:
        return resolved_trade_date, ohlcv_df, ""

    df = ohlcv_df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    prev = df[df["Date"] < today]
    if prev.empty:
        return resolved_trade_date, ohlcv_df, ""
    corrected = prev["Date"].max().strftime("%Y-%m-%d")
    return corrected, prev, "intraday_daily_bar_incomplete"


class DataCollector:
    """Pre-fetch all data needed for a TradingAgents analysis run."""

    def __init__(self, config: dict | None = None):
        self.config = config or DEFAULT_CONFIG.copy()
        set_config(self.config)

    def collect_shared_data(
        self,
        trade_date: str,
        market_type: str = "a_share",
    ) -> dict:
        """Pre-fetch data common to all stocks of the same market type on same date.

        Args:
            trade_date: The trading date (yyyy-mm-dd).
            market_type: One of "a_share", "hk", "us".

        Returns:
            Dict with keys: macro_indicators, global_news, predictions.
        """
        logger.info("Pre-fetching shared data for market_type=%s, date=%s", market_type, trade_date)
        shared: dict = {}

        # --- Macro indicators ---
        macro: dict[str, str] = {}
        if market_type == "a_share":
            macro_list = self.config.get("cn_macro_indicators", list(CN_MACRO_INDICATORS))
            from tradingagents.dataflows.china_macro import get_cn_macro_data
            for ind in macro_list:
                macro[ind] = _safe_call(f"shared_macro:{ind}", get_cn_macro_data, ind, trade_date, None)
        elif market_type == "hk":
            macro_list = self.config.get("hk_macro_indicators", list(HK_MACRO_INDICATORS))
            from tradingagents.dataflows.hk_macro import get_hk_macro_data
            for ind in macro_list:
                macro[ind] = _safe_call(f"shared_macro:{ind}", get_hk_macro_data, ind, trade_date, None)
        else:
            macro_list = self.config.get("standard_macro_indicators", list(DEFAULT_MACRO_INDICATORS))
            for ind in macro_list:
                macro[ind] = _safe_call(
                    f"shared_macro:{ind}",
                    route_to_vendor, "get_macro_indicators", ind, trade_date, None,
                )
        shared["macro_indicators"] = macro

        # --- Global news ---
        lookback = self.config.get("global_news_lookback_days", 7)
        limit = self.config.get("global_news_article_limit", 10)
        if market_type == "a_share":
            from tradingagents.dataflows.akshare_provider import get_global_news as _ak_global_news

            shared["global_news"] = _safe_call(
                "shared_news:global(akshare)", _ak_global_news, trade_date, lookback, limit,
            )
        elif market_type == "hk":
            from tradingagents.dataflows.hk_akshare_provider import (
                get_global_news as _hk_global_news,
            )

            shared["global_news"] = _safe_call(
                "shared_news:global(hk_akshare)", _hk_global_news, trade_date, lookback, limit,
            )
        else:
            shared["global_news"] = _safe_call(
                "shared_news:global", route_to_vendor, "get_global_news", trade_date, lookback, limit,
            )

        # --- Prediction / market signals ---
        predictions: dict[str, str] = {}
        if market_type == "a_share":
            pred_queries = self.config.get("cn_prediction_queries", list(CN_PREDICTION_QUERIES))
            from tradingagents.dataflows.cn_market_signals import get_cn_market_signals
            for query in pred_queries:
                predictions[query] = _safe_call(f"shared_signal:{query}", get_cn_market_signals, query, None)
        elif market_type == "hk":
            pred_queries = self.config.get("hk_prediction_queries", list(HK_PREDICTION_QUERIES))
            from tradingagents.dataflows.hk_market_signals import get_hk_market_signals
            for query in pred_queries:
                predictions[query] = _safe_call(f"shared_signal:{query}", get_hk_market_signals, query, None)
        else:
            pred_queries = self.config.get("standard_prediction_queries", list(DEFAULT_PREDICTION_QUERIES))
            for query in pred_queries:
                predictions[query] = _safe_call(
                    f"shared_signal:{query}", route_to_vendor, "get_prediction_markets", query, None,
                )
        shared["predictions"] = predictions

        logger.info("Shared data pre-fetch complete: %d macro, %d predictions",
                    len(macro), len(predictions))
        return shared

    def collect(
        self,
        ticker: str,
        trade_date: str,
        asset_type: str = "stock",
        selected_analysts: tuple[str, ...] | list[str] = (
            "market", "social", "news", "fundamentals",
        ),
        shared_data: dict | None = None,
        intraday: bool = False,
    ) -> DataBundle:
        selected = set(selected_analysts)
        analysis_mode = "intraday" if intraday else "daily"
        logger.info("Collecting data for %s on %s (mode=%s, analysts: %s)", ticker, trade_date, analysis_mode, selected)

        original_trade_date: str | None = None
        requested_trade_date = trade_date
        date_correction_reason: str = ""
        ohlcv_df: pd.DataFrame | None = None
        try:
            corrected, ohlcv_df, reason = _resolve_trading_date(ticker, trade_date)
            if intraday:
                forced, forced_df, forced_reason = _force_last_complete_daily_bar(
                    ticker, requested_trade_date, corrected, ohlcv_df,
                )
                if forced_reason:
                    corrected = forced
                    ohlcv_df = forced_df
                    reason = forced_reason
            if corrected != trade_date:
                logger.info("Trading date corrected: %s → %s (%s)", trade_date, corrected, reason)
                original_trade_date = trade_date
                date_correction_reason = reason
                trade_date = corrected
        except Exception as e:
            logger.warning("Trading date resolution failed: %s, using original", e)

        market_status = resolve_market_status(ticker, trade_date)

        metadata = BundleMetadata(
            ticker=ticker,
            trade_date=trade_date,
            original_trade_date=original_trade_date,
            date_correction_reason=date_correction_reason,
            asset_type=asset_type,
            collection_timestamp=now_iso(),
            analysis_mode=analysis_mode,
            intraday_asof=now_iso() if intraday else "",
            selected_analysts=sorted(selected),
            vendor_config={
                "data_vendors": self.config.get("data_vendors", {}),
                "tool_vendors": self.config.get("tool_vendors", {}),
            },
            bundle_version=BUNDLE_VERSION,
            market_status=market_status,
        )

        market = self._collect_market_data(
            ticker, trade_date, ohlcv_df=ohlcv_df, intraday=intraday,
        ) if "market" in selected else None

        # Fetch ticker news once and share between sentiment and news analysts
        shared_ticker_news: str | None = None
        if "social" in selected or "news" in selected:
            start_date = _lookback_date(trade_date, 7)
            shared_ticker_news = _safe_call(
                "shared:ticker_news",
                route_to_vendor, "get_news", ticker, start_date, trade_date,
            )

        sentiment = self._collect_sentiment_data(ticker, trade_date, shared_ticker_news) if "social" in selected else None
        news = self._collect_news_data(ticker, trade_date, shared_ticker_news, shared_data=shared_data) if "news" in selected else None
        fundamentals = self._collect_fundamentals_data(
            ticker, trade_date,
            # Already resolved and cross-source verified above; the competitive
            # intelligence search needs a real name, and parsing it back out of
            # the overview text is the fragile path that silently failed.
            security_name=getattr(market_status, "security_name", "") or "",
        ) if "fundamentals" in selected else None

        # Source-level sanitization: tag historical ST mentions in per-ticker
        # text before it reaches any analyst, so verified-normal instruments
        # are never polluted by stale ST-era articles. Global/industry feeds
        # are excluded because they may legitimately mention other ST stocks.
        if sentiment:
            sentiment.ticker_news = annotate_historical_st_mentions(sentiment.ticker_news, ticker, market_status)
            sentiment.stocktwits = annotate_historical_st_mentions(sentiment.stocktwits, ticker, market_status)
            sentiment.reddit = annotate_historical_st_mentions(sentiment.reddit, ticker, market_status)
        if news:
            news.ticker_news = annotate_historical_st_mentions(news.ticker_news, ticker, market_status)

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
        shared_data: dict | None = None,
        **kwargs,
    ) -> tuple[DataBundle, Path]:
        bundle = self.collect(ticker, trade_date, shared_data=shared_data, **kwargs)
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
        intraday: bool = False,
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

        intraday_snapshot = ""
        if intraday:
            from tradingagents.dataflows.intraday_provider import get_intraday_snapshot
            intraday_snapshot = _safe_call(
                "intraday_snapshot", get_intraday_snapshot, ticker,
            )

        return MarketData(
            stock_data=stock_data,
            indicators=indicators,
            verified_snapshot=verified_snapshot,
            intraday_snapshot=intraday_snapshot,
        )

    def _collect_sentiment_data(self, ticker: str, trade_date: str, ticker_news: str | None = None) -> SentimentData:
        if ticker_news is None:
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

    def _collect_news_data(
        self, ticker: str, trade_date: str,
        ticker_news: str | None = None,
        shared_data: dict | None = None,
    ) -> NewsData:
        if ticker_news is None:
            start_date = _lookback_date(trade_date, 7)
            ticker_news = _safe_call(
                "news:ticker",
                route_to_vendor, "get_news", ticker, start_date, trade_date,
            )

        # --- Global news: use shared or fetch per-stock ---
        if shared_data and "global_news" in shared_data:
            global_news = shared_data["global_news"]
        else:
            lookback = self.config.get("global_news_lookback_days", 7)
            limit = self.config.get("global_news_article_limit", 10)
            if is_a_share(ticker):
                from tradingagents.dataflows.akshare_provider import (
                    get_global_news as _ak_global_news,
                )
                global_news = _safe_call(
                    "news:global(akshare)",
                    _ak_global_news, trade_date, lookback, limit,
                )
            elif is_hk_stock(ticker):
                from tradingagents.dataflows.hk_akshare_provider import (
                    get_global_news as _hk_global_news,
                )
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

        # --- Macro indicators: use shared or fetch per-stock ---
        if shared_data and "macro_indicators" in shared_data:
            macro = shared_data["macro_indicators"]
        else:
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

        # --- Predictions / market signals: use shared or fetch per-stock ---
        if shared_data and "predictions" in shared_data:
            predictions = shared_data["predictions"]
        else:
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

        # Industry rotation + stock money flow (per-stock, not shared)
        industry_data = ""
        stock_moneyflow = ""
        if is_a_share(ticker):
            from tradingagents.dataflows.akshare_provider import get_industry_data as _ak_industry
            from tradingagents.dataflows.tushare_provider import get_moneyflow as _ts_moneyflow

            stock_moneyflow = _safe_call(
                "news:moneyflow", _ts_moneyflow, ticker, trade_date, 5,
            )
            industry_data = _safe_call(
                "news:industry", _ak_industry, ticker, trade_date,
            )
        elif is_hk_stock(ticker):
            from tradingagents.dataflows.hk_akshare_provider import get_hk_industry_info

            industry_data = _safe_call(
                "news:hk_industry", get_hk_industry_info, ticker,
            )

        return NewsData(
            ticker_news=ticker_news,
            global_news=global_news,
            insider_transactions=insider,
            macro_indicators=macro,
            prediction_markets=predictions,
            industry_data=industry_data,
            stock_moneyflow=stock_moneyflow,
        )

    def _collect_fundamentals_data(
        self, ticker: str, trade_date: str, security_name: str = "",
    ) -> FundamentalsData:
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

        # Competitive intelligence via web search. The verified 证券简称 wins:
        # it is already cross-checked against three sources, while the overview
        # text shape varies by vendor and market.
        company_name = security_name.strip() or self._extract_company_name(overview)
        comp_intel = _safe_call(
            "fundamentals:competitive_intelligence",
            get_competitive_intelligence, ticker, company_name, trade_date,
        )

        return FundamentalsData(
            overview=overview,
            balance_sheet_quarterly=bs_q,
            balance_sheet_annual=bs_a,
            cashflow_quarterly=cf_q,
            cashflow_annual=cf_a,
            income_quarterly=inc_q,
            income_annual=inc_a,
            competitive_intelligence=comp_intel,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_company_name(overview: str) -> str:
        """Try to extract company name from the overview JSON or text.

        This returned ``""`` for every ticker in the 2026-08-19 batch, so every
        competitive-intelligence search ran on a bare ticker symbol. Two shapes
        defeated it: the A-share overview prefixes the JSON with markdown
        headers (``# Fundamentals for 300760.SZ`` … ``{ … }``), so
        ``json.loads`` on the whole string always raised; and the HK overview is
        not JSON at all but a markdown bullet list whose first entry is
        ``- **阿里巴巴集团控股有限公司**: Alibaba Group Holding Limited``.
        """
        if not overview or overview.startswith(_UNAVAILABLE_PREFIX):
            return ""

        import json as _json
        import re as _re

        # Embedded JSON object, with or without surrounding markdown.
        start, end = overview.find("{"), overview.rfind("}")
        if start != -1 and end > start:
            try:
                data = _json.loads(overview[start:end + 1])
            except (ValueError, TypeError):
                data = None
            if isinstance(data, dict):
                # AKShare stock_individual_info_em returns keys like
                # "股票简称", "公司名称".
                for key in ("股票简称", "公司名称", "name", "Name", "shortName"):
                    value = data.get(key)
                    if value:
                        return str(value).strip()

        # HK/markdown shape: the first bold bullet label is the company name.
        for line in overview.splitlines():
            match = _re.match(r"\s*[-*]\s*\*\*(.+?)\*\*\s*[:：]", line)
            if match:
                candidate = match.group(1).strip()
                # Skip metric rows ("基本每股收益(元)"); a company name has no
                # unit parenthesis and is not a pure number.
                if candidate and "(" not in candidate and "（" not in candidate:
                    return candidate
                break
        return ""

    def _default_save_path(self, ticker: str, trade_date: str) -> Path:
        from tradingagents.default_config import DEFAULT_CONFIG
        data_dir = Path(self.config.get("data_dir", DEFAULT_CONFIG["data_dir"]))
        safe_ticker = safe_ticker_component(ticker)
        return data_dir / safe_ticker / self._filename(ticker, trade_date)

    @staticmethod
    def _filename(ticker: str, trade_date: str) -> str:
        safe_ticker = safe_ticker_component(ticker)
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        return f"{safe_ticker}_{trade_date}_{ts}.json"


class DataIncompleteError(Exception):
    """Raised when a *blocking* field is unusable.

    Only blocking issues belong here. All-or-nothing admission was the wrong
    trade: a missing annual cash-flow statement or an unreachable industry
    ranking endpoint cost the entire run, which pushed callers toward not
    checking at all. The gate now keys on
    :func:`classify_bundle_issues`, and warn-level gaps ride through on
    ``integrity_findings`` so the decision is made *with* the gap on record
    rather than not made.
    """

    def __init__(self, issues: list[dict]):
        self.issues = issues
        summary = "; ".join(f"{i['category']}/{i['field']}" for i in issues[:5])
        if len(issues) > 5:
            summary += f" ... 等共 {len(issues)} 项"
        super().__init__(f"关键数据不完备，共 {len(issues)} 项不可用: {summary}")


SEVERITY_BLOCK = "block"
SEVERITY_WARN = "warn"

# What kind of gap an issue describes, so the report layer can label it without
# re-parsing the reason text.
KIND_UNAVAILABLE = "unavailable"  # collector recorded an <unavailable: …> marker
KIND_MISSING = "missing"          # field empty / whole section absent
KIND_PARTIAL = "partial"          # provider returned prose admitting a gap
KIND_STALE = "stale"              # content present but older than its cadence allows

# Soft failures: the provider caught its own error and returned prose instead of
# raising, so no ``<unavailable: …>`` marker was ever recorded and the string is
# non-empty — it passes every present/absent check while carrying no data. Real
# examples from the 2026-08-19 batch: ``industry_data`` = "行业涨跌排名数据暂不可用 /
# 行业资金流向数据暂不可用", and ``intraday_snapshot`` = "盘中数据暂不可用：…".
# Keep this list to phrases the providers actually emit (grep dataflows/) so a
# legitimate document is never flagged for containing the word "unavailable".
_SOFT_FAIL_PATTERNS = (
    "暂不可用",
    "数据不可用",
    "接口无返回",
    "data unavailable",
    "no data available",
    "no data returned",
)

# Fields whose absence makes everything downstream fiction rather than merely
# less informed. Everything else degrades to a warning so one missing annual
# cash-flow statement no longer costs the whole run (the web gate used to raise
# on any single issue, producing no report and no DB row at all).
_BLOCKING_FIELDS = frozenset({
    ("行情数据", "stock_data"),
    ("行情数据", "verified_snapshot"),
    ("市场状态", "risk_warning_status"),
})

# Freshness budget in calendar days, measured from the bundle's trade date to
# the newest date the field's own text reports. Keyed by ``(category, field)``,
# where ``field=None`` covers every field in that category.
#
# This exists because "present" and "current" are different questions and only
# the first was ever asked. In the 2026-08-19 batch,
# ``prediction_markets/northbound_flow`` was a fully-formed table labelled
# "最新数据 (2024-08-16)" — two years stale — sitting next to a correctly dated
# ``margin_trading``, and it passed every check.
#
# Deliberately restricted to feeds that update every trading day, so a lag is
# unambiguously a broken feed. Series with irregular cadence would produce
# false positives that train the reader to ignore the banner: the newest insider
# filing can legitimately be months old, ``rrr`` reports the date of the last
# reserve-ratio *change* (sometimes years back), and quarterly statements lag by
# design. Widen this only with a per-field budget that matches real cadence.
_FRESHNESS_BUDGET_DAYS = {
    ("市场信号", None): 7,
    ("新闻数据", "stock_moneyflow"): 7,
}

# ``YYYY-MM-DD`` / ``YYYY/M/D`` / ``YYYY年M月D日`` / bare ``YYYYMMDD``.
_DATE_RE = re.compile(
    r"\b(20\d{2})[-/年](\d{1,2})[-/月](\d{1,2})\b"
    r"|\b(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\b"
)


def _soft_fail_notice(value: str) -> str:
    """Return the offending line when *value* carries a soft-failure notice."""
    lowered = value.lower()
    if not any(p in lowered for p in _SOFT_FAIL_PATTERNS):
        return ""
    for line in value.splitlines():
        low = line.lower()
        if any(p in low for p in _SOFT_FAIL_PATTERNS):
            return line.strip()[:160]
    return ""


def _latest_reported_date(value: str, not_after: datetime) -> datetime | None:
    """Newest plausible date mentioned in *value*, ignoring future noise.

    Dates past *not_after* are discarded rather than trusted: a forward-looking
    date in the text (a scheduled earnings release, a maturity) would otherwise
    mask a stale feed.
    """
    latest: datetime | None = None
    for match in _DATE_RE.finditer(value):
        y, m, d = match.group(1, 2, 3)
        if y is None:
            y, m, d = match.group(4, 5, 6)
        try:
            parsed = datetime(int(y), int(m), int(d))
        except ValueError:
            continue
        if parsed > not_after:
            continue
        if latest is None or parsed > latest:
            latest = parsed
    return latest


def validate_bundle_completeness(bundle: DataBundle) -> list[dict]:
    """Check every bundle field for missing, soft-failed, or stale content.

    Returns a list of issues, each a dict with keys ``category``, ``field``,
    ``reason``, ``severity`` (``"block"`` / ``"warn"``) and ``kind``
    (``"unavailable"`` / ``"missing"`` / ``"partial"`` / ``"stale"``). Use
    :func:`classify_bundle_issues` to split by severity; callers that only look
    at ``category``/``field``/``reason`` keep working unchanged.
    """
    issues: list[dict] = []
    trade_date = bundle.metadata.trade_date
    try:
        # +1 day of slack: an intraday bundle can legitimately carry a snapshot
        # timestamped in a market timezone that is already on the next date.
        freshness_anchor = datetime.strptime(trade_date, "%Y-%m-%d")
        not_after = freshness_anchor + timedelta(days=1)
    except (TypeError, ValueError):
        freshness_anchor = None
        not_after = None

    def _add(
        category: str, field: str, reason: str,
        severity: str | None = None, kind: str = KIND_UNAVAILABLE,
    ):
        if severity is None:
            severity = (
                SEVERITY_BLOCK if (category, field) in _BLOCKING_FIELDS else SEVERITY_WARN
            )
        issues.append({
            "category": category, "field": field, "reason": reason,
            "severity": severity, "kind": kind,
        })

    def _check_staleness(category: str, field: str, value: str):
        budget = _FRESHNESS_BUDGET_DAYS.get(
            (category, field), _FRESHNESS_BUDGET_DAYS.get((category, None)),
        )
        if budget is None or freshness_anchor is None:
            return
        latest = _latest_reported_date(value, not_after)
        if latest is None:
            return
        lag = (freshness_anchor - latest).days
        if lag > budget:
            _add(
                category, field,
                f"数据陈旧: 自报最新日期 {latest:%Y-%m-%d}，落后交易日 {trade_date} 共 {lag} 天"
                f"（该类数据的容忍上限为 {budget} 天）",
                SEVERITY_WARN, KIND_STALE,
            )

    def _check(category: str, field: str, value: str):
        if not value:
            return
        if value.startswith(_UNAVAILABLE_PREFIX):
            reason = value[len(_UNAVAILABLE_PREFIX):-1] if value.endswith(">") else value
            _add(category, field, reason)
            return
        notice = _soft_fail_notice(value)
        if notice:
            # Partial by nature: the section rendered its header and whatever it
            # did retrieve, then reported the gap in prose.
            _add(category, field, f"部分不可用: {notice}", SEVERITY_WARN, KIND_PARTIAL)
            # No staleness check on top: once a field says it is unusable, any
            # date left in its text is explanatory prose, not a data date. The
            # northbound notice names the 2024-08-19 disclosure cut-off, which
            # would otherwise be re-reported as "the data is 730 days old".
            return
        _check_staleness(category, field, value)

    def _check_dict(category: str, d: dict[str, str]):
        for k, v in d.items():
            _check(category, k, v)

    market_status = getattr(bundle.metadata, "market_status", None)
    if market_status:
        if market_status.risk_warning_status == "unknown":
            reason = "; ".join(market_status.conflicts) or "无法验证风险警示/ST状态"
            _add("市场状态", "risk_warning_status", reason)
        if market_status.conflicts:
            _add("市场状态", "source_conflicts", "; ".join(market_status.conflicts))
        if is_a_share(bundle.metadata.ticker) and not market_status.effective_date:
            _add("市场状态", "effective_date", "缺少风险警示状态生效日期")

    if bundle.market:
        _check("行情数据", "stock_data", bundle.market.stock_data)
        _check("行情数据", "verified_snapshot", bundle.market.verified_snapshot)
        _check_dict("行情数据/技术指标", bundle.market.indicators)
        # Only meaningful when the run asked for it — a daily-mode bundle leaves
        # it empty by design. When intraday mode *did* ask, this is the only
        # incremental data the mode provides, so a silent failure means the run
        # was effectively a daily run wearing an intraday label. It failed 8/8
        # in the 2026-08-19 batch and appeared in no check, banner, or DB row.
        if bundle.metadata.analysis_mode == "intraday":
            if not (bundle.market.intraday_snapshot or "").strip():
                _add(
                    "行情数据", "intraday_snapshot", "盘中模式下未取到盘中快照",
                    kind=KIND_MISSING,
                )
            else:
                _check("行情数据", "intraday_snapshot", bundle.market.intraday_snapshot)

    if bundle.sentiment:
        _check("情绪数据", "ticker_news", bundle.sentiment.ticker_news)
        _check("情绪数据", "stocktwits", bundle.sentiment.stocktwits)
        _check("情绪数据", "reddit", bundle.sentiment.reddit)

    if bundle.news:
        _check("新闻数据", "ticker_news", bundle.news.ticker_news)
        _check("新闻数据", "global_news", bundle.news.global_news)
        _check("新闻数据", "insider_transactions", bundle.news.insider_transactions)
        _check("新闻数据", "industry_data", bundle.news.industry_data)
        _check("新闻数据", "stock_moneyflow", bundle.news.stock_moneyflow)
        _check_dict("宏观指标", bundle.news.macro_indicators)
        _check_dict("市场信号", bundle.news.prediction_markets)

    if bundle.fundamentals:
        _check("财务数据", "overview", bundle.fundamentals.overview)
        _check("财务数据", "balance_sheet_quarterly", bundle.fundamentals.balance_sheet_quarterly)
        _check("财务数据", "balance_sheet_annual", bundle.fundamentals.balance_sheet_annual)
        _check("财务数据", "cashflow_quarterly", bundle.fundamentals.cashflow_quarterly)
        _check("财务数据", "cashflow_annual", bundle.fundamentals.cashflow_annual)
        _check("财务数据", "income_quarterly", bundle.fundamentals.income_quarterly)
        _check("财务数据", "income_annual", bundle.fundamentals.income_annual)
        # Feeds the Moat section of the fundamentals report, which instructs the
        # model to use it as supporting evidence — so an error string here is
        # worse than an empty one. It was unavailable for 6/6 A-shares in the
        # 2026-08-19 batch (the HK ticker succeeded) and checked nowhere.
        _check("财务数据", "competitive_intelligence", bundle.fundamentals.competitive_intelligence)

        # Aggregate rule: individually a missing statement is a warning, but a
        # fundamentals analyst with no statements at all writes fiction.
        statements = (
            bundle.fundamentals.balance_sheet_quarterly,
            bundle.fundamentals.balance_sheet_annual,
            bundle.fundamentals.cashflow_quarterly,
            bundle.fundamentals.cashflow_annual,
            bundle.fundamentals.income_quarterly,
            bundle.fundamentals.income_annual,
        )
        if all(
            not (s or "").strip() or s.startswith(_UNAVAILABLE_PREFIX) for s in statements
        ):
            _add(
                "财务数据", "全部财报", "六张财务报表全部缺失或不可用",
                SEVERITY_BLOCK, KIND_MISSING,
            )

    return issues


def classify_bundle_issues(issues: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split completeness issues into ``(blocking, warnings)``.

    Issues without a ``severity`` key (older callers, hand-built fixtures) count
    as blocking so the split can never quietly downgrade an unknown issue.
    """
    blocking = [i for i in issues if i.get("severity", SEVERITY_BLOCK) == SEVERITY_BLOCK]
    warnings = [i for i in issues if i.get("severity", SEVERITY_BLOCK) != SEVERITY_BLOCK]
    return blocking, warnings


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
