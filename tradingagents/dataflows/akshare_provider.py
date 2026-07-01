"""AKShare data provider for Chinese A-share market data.

AKShare is a free, open-source library wrapping Chinese financial data sources
(Sina, EastMoney, SSE, SZSE). No API key required.

Install: ``pip install akshare`` or ``pip install "tradingagents[china]"``
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime
from typing import Annotated

import pandas as pd
from stockstats import wrap

from .config import get_config
from .errors import NoMarketDataError
from .market_utils import a_share_to_akshare_symbol, detect_exchange, is_etf
from .retry import call_with_retry
from .stockstats_utils import MAX_OHLCV_STALE_DAYS_CN, _assert_ohlcv_not_stale, _clean_dataframe
from .utils import is_cache_fresh, safe_ticker_component

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 30

# Lock to protect pd.options.mode.string_storage in concurrent contexts
_string_storage_lock = threading.Lock()

_COL_MAP = {
    "日期": "Date",
    "开盘": "Open",
    "最高": "High",
    "最低": "Low",
    "收盘": "Close",
    "成交量": "Volume",
    # fund_etf_hist_sina returns lowercase English columns
    "date": "Date",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume",
}


def _get_ak():
    """Lazy-import akshare to allow graceful degradation when not installed."""
    try:
        import akshare as ak
        return ak
    except ImportError as exc:
        raise ImportError(
            "akshare is not installed. Install with: pip install akshare "
            "or pip install 'tradingagents[china]'"
        ) from exc


# ---------------------------------------------------------------------------
# OHLCV
# ---------------------------------------------------------------------------

def _load_ohlcv_akshare(symbol: str, curr_date: str) -> pd.DataFrame:
    """Fetch and cache AKShare A-share daily OHLCV, filtered for look-ahead bias."""
    ak = _get_ak()
    code = a_share_to_akshare_symbol(symbol)
    safe_symbol = safe_ticker_component(symbol)
    config = get_config()

    os.makedirs(config["data_cache_dir"], exist_ok=True)
    today_str = datetime.now().strftime("%Y-%m-%d")
    cache_file = os.path.join(
        config["data_cache_dir"], f"{safe_symbol}-AKShare-daily-{today_str}.csv"
    )

    data = None
    if is_cache_fresh(cache_file, symbol):
        cached = pd.read_csv(cache_file, on_bad_lines="skip", encoding="utf-8")
        if not cached.empty and "Close" in cached.columns:
            data = cached

    if data is None:
        if is_etf(symbol):
            df = None
            try:
                df = call_with_retry(
                    ak.fund_etf_hist_em,
                    symbol=code,
                    period="daily",
                    start_date="20200101",
                    end_date=datetime.now().strftime("%Y%m%d"),
                    adjust="qfq",
                )
            except Exception as e:
                logger.warning("fund_etf_hist_em failed for %s: %s", code, e)
            if df is None or df.empty:
                suffix = detect_exchange(symbol)
                sina_prefix = "sh" if suffix == ".SS" else "sz"
                try:
                    df = call_with_retry(
                        ak.fund_etf_hist_sina, symbol=f"{sina_prefix}{code}",
                    )
                except Exception as e:
                    logger.warning("fund_etf_hist_sina fallback also failed for %s: %s", code, e)
                    df = None
        else:
            df = call_with_retry(
                ak.stock_zh_a_hist,
                symbol=code,
                period="daily",
                start_date="20200101",
                end_date=datetime.now().strftime("%Y%m%d"),
                adjust="qfq",
                timeout=REQUEST_TIMEOUT,
            )
        if df is None or df.empty:
            raise NoMarketDataError(symbol, symbol, "AKShare returned no data")

        df = df.rename(columns=_COL_MAP)
        keep = [c for c in ("Date", "Open", "High", "Low", "Close", "Volume") if c in df.columns]
        data = df[keep].copy()
        data.to_csv(cache_file, index=False, encoding="utf-8")

    data = _clean_dataframe(data)
    curr_date_dt = pd.to_datetime(curr_date)
    data = data[data["Date"] <= curr_date_dt]
    _assert_ohlcv_not_stale(data, curr_date, symbol, symbol, max_stale_days=MAX_OHLCV_STALE_DAYS_CN)
    return data


def get_stock_data(
    symbol: Annotated[str, "A-share ticker symbol (600519.SS, 300750.SZ)"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Get OHLCV stock data from AKShare."""
    data = _load_ohlcv_akshare(symbol, end_date)

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    df = data[(data["Date"] >= start_dt) & (data["Date"] <= end_dt)].copy()

    if df.empty:
        raise NoMarketDataError(
            symbol, symbol, f"no rows between {start_date} and {end_date}"
        )

    for col in ("Open", "High", "Low", "Close"):
        if col in df.columns:
            df[col] = df[col].round(2)

    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    df = df.set_index("Date")
    csv_string = df.to_csv()

    header = f"# Stock data for {symbol.upper()} from {start_date} to {end_date}\n"
    header += f"# Total records: {len(df)}\n"
    header += "# Data source: AKShare (EastMoney)\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    return header + csv_string


# ---------------------------------------------------------------------------
# Technical Indicators
# ---------------------------------------------------------------------------

def get_indicators(
    symbol: Annotated[str, "A-share ticker symbol"],
    indicator: Annotated[str, "technical indicator to compute"],
    curr_date: Annotated[str, "current trading date, YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:
    """Compute technical indicators from AKShare OHLCV data."""
    from dateutil.relativedelta import relativedelta

    best_ind_params = {
        "close_50_sma": "50 SMA: medium-term trend indicator.",
        "close_200_sma": "200 SMA: long-term trend benchmark.",
        "close_10_ema": "10 EMA: responsive short-term average.",
        "macd": "MACD: momentum via EMA differences.",
        "macds": "MACD Signal: EMA smoothing of MACD.",
        "macdh": "MACD Histogram: gap between MACD and signal.",
        "rsi": "RSI: overbought/oversold momentum indicator.",
        "boll": "Bollinger Middle: 20 SMA basis.",
        "boll_ub": "Bollinger Upper Band: overbought/breakout zone.",
        "boll_lb": "Bollinger Lower Band: oversold zone.",
        "atr": "ATR: average true range volatility.",
        "vwma": "VWMA: volume-weighted moving average.",
        "mfi": "MFI: volume+price money flow index.",
    }

    if indicator not in best_ind_params:
        raise ValueError(
            f"Indicator {indicator} is not supported. "
            f"Choose from: {list(best_ind_params.keys())}"
        )

    curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    before = curr_date_dt - relativedelta(days=look_back_days)

    data = _load_ohlcv_akshare(symbol, curr_date)
    df = wrap(data)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    df[indicator]

    ind_string = ""
    current_dt = curr_date_dt
    while current_dt >= before:
        date_str = current_dt.strftime("%Y-%m-%d")
        matching = df[df["Date"] == date_str]
        if not matching.empty:
            val = matching[indicator].values[0]
            ind_string += f"{date_str}: {'N/A' if pd.isna(val) else val}\n"
        else:
            ind_string += f"{date_str}: N/A: Not a trading day\n"
        current_dt = current_dt - relativedelta(days=1)

    return (
        f"## {indicator} values from {before.strftime('%Y-%m-%d')} to {curr_date}:\n\n"
        + ind_string
        + "\n\n"
        + best_ind_params.get(indicator, "")
    )


# ---------------------------------------------------------------------------
# Fundamentals
# ---------------------------------------------------------------------------

def get_fundamentals(
    ticker: Annotated[str, "A-share ticker symbol"],
    curr_date: Annotated[str, "current date"] = None,
) -> str:
    """Get company fundamentals from AKShare (EastMoney)."""
    ak = _get_ak()
    code = a_share_to_akshare_symbol(ticker)

    result = {}
    try:
        info = call_with_retry(ak.stock_individual_info_em, symbol=code)
        if info is not None and not info.empty:
            for _, row in info.iterrows():
                key = str(row.iloc[0]).strip()
                val = str(row.iloc[1]).strip()
                result[key] = val
    except Exception as e:
        logger.warning("AKShare stock_individual_info_em failed for %s: %s", ticker, e)

    try:
        abstract = call_with_retry(ak.stock_financial_abstract_ths, symbol=code, indicator="按报告期")
        if abstract is not None and not abstract.empty:
            latest = abstract.iloc[-1]
            for col in abstract.columns:
                if col not in result:
                    val = latest[col]
                    if pd.notna(val):
                        result[col] = str(val)
    except Exception as e:
        logger.warning("AKShare stock_financial_abstract_ths failed for %s: %s", ticker, e)

    if not result:
        raise NoMarketDataError(ticker, ticker, "AKShare returned no fundamental data")

    return json.dumps(result, ensure_ascii=False, indent=2)


def get_balance_sheet(
    ticker: Annotated[str, "A-share ticker symbol"],
    freq: Annotated[str, "frequency: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    """Get A-share balance sheet data from AKShare, with EastMoney HTTP fallback."""
    df = _try_akshare_financial(ticker, "stock_balance_sheet_by_report_em", "balance sheet")
    if df is None:
        df = _fetch_eastmoney_financial(ticker, "RPT_DMSK_FN_BALANCE", "balance sheet")
    return _format_financial_statement(df, ticker, "Balance Sheet", freq, curr_date)


def get_cashflow(
    ticker: Annotated[str, "A-share ticker symbol"],
    freq: Annotated[str, "frequency: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    """Get A-share cash flow data from AKShare, with EastMoney HTTP fallback."""
    df = _try_akshare_financial(ticker, "stock_cash_flow_sheet_by_report_em", "cash flow")
    if df is None:
        df = _fetch_eastmoney_financial(ticker, "RPT_DMSK_FN_CASHFLOW", "cash flow")
    return _format_financial_statement(df, ticker, "Cash Flow", freq, curr_date)


def get_income_statement(
    ticker: Annotated[str, "A-share ticker symbol"],
    freq: Annotated[str, "frequency: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    """Get A-share income statement data from AKShare, with EastMoney HTTP fallback."""
    df = _try_akshare_financial(ticker, "stock_profit_sheet_by_report_em", "income")
    if df is None:
        df = _fetch_eastmoney_financial(ticker, "RPT_DMSK_FN_INCOME", "income")
    return _format_financial_statement(df, ticker, "Income Statement", freq, curr_date)


def _try_akshare_financial(ticker: str, ak_func_name: str, label: str) -> pd.DataFrame | None:
    """Try fetching financial data via AKShare; return None on failure."""
    try:
        ak = _get_ak()
        code = a_share_to_akshare_symbol(ticker)
        func = getattr(ak, ak_func_name)
        df = call_with_retry(func, symbol=code)
        if df is not None and not df.empty:
            return df
    except (TypeError, AttributeError, KeyError) as exc:
        logger.warning("AKShare %s failed for %s (curl_cffi blocked?): %s", label, ticker, exc)
    except Exception as exc:  # noqa: BLE001
        logger.warning("AKShare %s failed for %s: %s", label, ticker, exc)
    return None


def _fetch_eastmoney_financial(
    ticker: str, report_name: str, label: str, page_size: int = 8,
) -> pd.DataFrame:
    """Fetch financial statement data from EastMoney datacenter API (stdlib only).

    Uses urllib which respects NO_PROXY=*, bypassing the curl_cffi proxy issue.
    """
    import re
    from urllib.request import Request, urlopen

    code = re.sub(r"\.(SZ|SS)$", "", ticker, flags=re.IGNORECASE)
    url = (
        "https://datacenter.eastmoney.com/securities/api/data/v1/get"
        f"?reportName={report_name}"
        "&columns=ALL"
        f"&filter=(SECURITY_CODE%3D%22{code}%22)"
        f"&pageSize={page_size}"
        "&sortColumns=REPORT_DATE&sortTypes=-1"
    )
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        raw = json.loads(urlopen(req, timeout=15).read().decode("utf-8"))
    except Exception as exc:
        raise NoMarketDataError(
            ticker, ticker, f"EastMoney {label} API failed: {exc}",
        ) from exc

    rows = (raw.get("result") or {}).get("data") or []
    if not rows:
        raise NoMarketDataError(ticker, ticker, f"EastMoney returned no {label} data")

    return pd.DataFrame(rows)


def _format_financial_statement(
    df: pd.DataFrame, ticker: str, title: str, freq: str, curr_date: str | None,
) -> str:
    """Format a financial statement DataFrame into a readable string."""
    date_col = None
    for c in ("REPORT_DATE_NAME", "报告日期", "REPORT_DATE", "REPORTDATE"):
        if c in df.columns:
            date_col = c
            break

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        if curr_date:
            df = df[df[date_col] <= pd.to_datetime(curr_date)]
        if freq == "annual":
            df = df[df[date_col].dt.month == 12]

    df = df.head(4)

    header = f"# {title} for {ticker.upper()}\n"
    header += f"# Frequency: {freq}\n"
    header += "# Data source: AKShare (EastMoney)\n"
    header += f"# Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    return header + df.to_string(max_rows=20, max_cols=15)


# ---------------------------------------------------------------------------
# News
# ---------------------------------------------------------------------------

def _fetch_stock_announcements(code: str, ticker: str, limit: int = 10) -> list[str]:
    """个股公告 — 上市公司一手披露（业绩预告/定增/回购/股权变动等）。"""
    ak = _get_ak()
    lines: list[str] = []
    try:
        df = call_with_retry(ak.stock_individual_notice_report, security=code, symbol="全部")
        if df is None or df.empty:
            return lines
        title_col = _col(df, "公告标题", "title") or df.columns[2]
        type_col = _col(df, "公告类型", "type")
        date_col = _col(df, "公告日期", "date")
        for _, row in df.head(limit).iterrows():
            title = str(row[title_col]).strip()
            # Remove the "贵州茅台:" prefix that repeats the company name
            if ":" in title:
                title = title.split(":", 1)[1].strip()
            elif "：" in title:
                title = title.split("：", 1)[1].strip()
            ann_type = f"[{row[type_col]}] " if type_col else ""
            date_str = f" ({row[date_col]})" if date_col else ""
            lines.append(f"- {ann_type}{title}{date_str}")
    except Exception as e:
        logger.warning("stock_individual_notice_report failed for %s: %s", ticker, e)
    return lines


def _fetch_research_reports(code: str, ticker: str, limit: int = 10) -> list[str]:
    """券商研报 — 机构评级 + EPS 预测。"""
    ak = _get_ak()
    lines: list[str] = []
    try:
        df = call_with_retry(ak.stock_research_report_em, symbol=code)
        if df is None or df.empty:
            return lines
        title_col = _col(df, "报告名称", "report_name") or df.columns[3]
        rating_col = _col(df, "东财评级", "rating")
        broker_col = _col(df, "机构", "broker")
        date_col = _col(df, "日期", "date")
        # EPS forecast columns (dynamic year-based names)
        eps_cols = [c for c in df.columns if "盈利预测-收益" in c]
        pe_cols = [c for c in df.columns if "盈利预测-市盈率" in c]

        for _, row in df.head(limit).iterrows():
            title = str(row[title_col]).strip()
            rating = f" [{row[rating_col]}]" if rating_col and pd.notna(row.get(rating_col)) else ""
            broker = f" — {row[broker_col]}" if broker_col and pd.notna(row.get(broker_col)) else ""
            date_str = f" ({row[date_col]})" if date_col and pd.notna(row.get(date_col)) else ""

            line = f"- {title}{rating}{broker}{date_str}"
            # Add EPS forecasts if available
            forecasts = []
            for ec, pc in zip(eps_cols[:2], pe_cols[:2]):
                year = ec.split("-")[0] if "-" in ec else ""
                eps_val = row.get(ec)
                pe_val = row.get(pc)
                if pd.notna(eps_val):
                    f_str = f"{year}E EPS:{eps_val}"
                    if pd.notna(pe_val):
                        f_str += f"/PE:{pe_val}"
                    forecasts.append(f_str)
            if forecasts:
                line += f"  ({', '.join(forecasts)})"
            lines.append(line)
    except Exception as e:
        logger.warning("stock_research_report_em failed for %s: %s", ticker, e)
    return lines


def get_news(
    ticker: Annotated[str, "A-share ticker symbol"],
    start_date: Annotated[str, "Start date"],
    end_date: Annotated[str, "End date"],
) -> str:
    """Get A-share news from multiple sources.

    Sources:
    1. EastMoney stock news (东方财富个股新闻)
    2. Company announcements (上市公司公告)
    3. Analyst research reports (券商研报)
    """
    ak = _get_ak()
    code = a_share_to_akshare_symbol(ticker)

    lines = [f"# News for {ticker.upper()} ({start_date} to {end_date})\n"]

    # Source 1: EastMoney stock news
    # NOTE: Keep string_storage="python" for ALL DataFrame operations to avoid
    # ArrowInvalid errors when news content contains \u escape sequences.
    # Use a lock because pd.options.mode.string_storage is a global setting
    # and concurrent threads can interfere with each other.
    with _string_storage_lock:
        _prev = pd.options.mode.string_storage
        pd.options.mode.string_storage = "python"
        try:
            df = call_with_retry(ak.stock_news_em, symbol=code)

            news_count = 0
            if df is not None and not df.empty:
                title_col = _col(df, "新闻标题", "title", "标题")
                time_col = _col(df, "发布时间", "publish_time", "时间")
                source_col = _col(df, "文章来源", "source", "来源")
                content_col = _col(df, "新闻内容", "content", "内容")

                if time_col:
                    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
                    start_dt = pd.to_datetime(start_date)
                    end_dt = pd.to_datetime(end_date)
                    df = df[(df[time_col] >= start_dt) & (df[time_col] <= end_dt + pd.Timedelta(days=1))]

                lines.append("## 东方财富 · 个股新闻\n")
                for _, row in df.head(15).iterrows():
                    title = str(row[title_col]) if title_col else "N/A"
                    time_str = f" [{row[time_col]}]" if time_col else ""
                    source = f" ({row[source_col]})" if source_col else ""
                    lines.append(f"- {title}{time_str}{source}")
                    if content_col:
                        snippet = str(row[content_col])[:200]
                        if snippet:
                            lines.append(f"  {snippet}")
                    news_count += 1
                lines.append("")
        finally:
            pd.options.mode.string_storage = _prev

    # Source 2: Company announcements
    ann_lines = _fetch_stock_announcements(code, ticker, limit=10)
    if ann_lines:
        lines.append("## 上市公司公告\n")
        lines.extend(ann_lines)
        lines.append("")

    # Source 3: Analyst research reports
    report_lines = _fetch_research_reports(code, ticker, limit=8)
    if report_lines:
        lines.append("## 券商研报\n")
        lines.extend(report_lines)
        lines.append("")

    if news_count == 0 and not ann_lines and not report_lines:
        raise NoMarketDataError(ticker, ticker, "AKShare returned no news data")

    return "\n".join(lines)


def _col(df: pd.DataFrame, *candidates: str) -> str | None:
    """Return the first column name that exists in *df*."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _fetch_eastmoney_global(limit: int) -> list[str]:
    """东方财富全球财经快讯 — 实时 200 条。"""
    ak = _get_ak()
    lines: list[str] = []
    try:
        df = call_with_retry(ak.stock_info_global_em)
        if df is None or df.empty:
            return lines
        title_col = _col(df, "标题", "title") or df.columns[0]
        summary_col = _col(df, "摘要", "summary")
        time_col = _col(df, "发布时间", "publish_time")
        for _, row in df.head(limit).iterrows():
            title = str(row[title_col]).strip()
            time_str = f" [{row[time_col]}]" if time_col else ""
            lines.append(f"- {title}{time_str}")
            if summary_col:
                snippet = str(row[summary_col]).strip()[:200]
                if snippet and snippet != title:
                    lines.append(f"  {snippet}")
    except Exception as e:
        logger.warning("stock_info_global_em failed: %s", e)
    return lines


def _fetch_caixin_news(limit: int) -> list[str]:
    """财新网头条 — 深度财经报道 100 条。"""
    ak = _get_ak()
    lines: list[str] = []
    try:
        df = call_with_retry(ak.stock_news_main_cx)
        if df is None or df.empty:
            return lines
        tag_col = _col(df, "tag", "标签") or "tag"
        summary_col = _col(df, "summary", "摘要") or "summary"
        for _, row in df.head(limit).iterrows():
            tag = str(row[tag_col]).strip() if tag_col in df.columns else ""
            summary = str(row[summary_col]).strip() if summary_col in df.columns else str(row.iloc[0]).strip()
            prefix = f"[{tag}] " if tag else ""
            lines.append(f"- {prefix}{summary}")
    except Exception as e:
        logger.warning("stock_news_main_cx failed: %s", e)
    return lines


def _fetch_economic_calendar(limit: int) -> list[str]:
    """百度经济日历 — 宏观数据发布（实际值 vs 预期值）。"""
    ak = _get_ak()
    lines: list[str] = []
    try:
        df = call_with_retry(ak.news_economic_baidu)
        if df is None or df.empty:
            return lines
        # Columns: 日期, 时间, 地区, 事件, 公布, 预期, 前值, 重要性
        event_col = _col(df, "事件", "event") or df.columns[3] if len(df.columns) > 3 else df.columns[0]
        region_col = _col(df, "地区", "region")
        actual_col = _col(df, "公布", "actual")
        expect_col = _col(df, "预期", "forecast")
        prev_col = _col(df, "前值", "previous")
        importance_col = _col(df, "重要性", "importance")

        # Only show high-importance events (重要性 >= 2) or China-related
        for _, row in df.iterrows():
            if len(lines) >= limit:
                break
            importance = 0
            if importance_col:
                try:
                    importance = int(row[importance_col])
                except (ValueError, TypeError):
                    pass
            region = str(row[region_col]).strip() if region_col else ""
            is_china = "中国" in region or "中" in region
            if importance < 2 and not is_china:
                continue
            event = str(row[event_col]).strip()
            parts = [f"- {region} {event}" if region else f"- {event}"]
            vals = []
            if actual_col and pd.notna(row.get(actual_col)):
                vals.append(f"公布: {row[actual_col]}")
            if expect_col and pd.notna(row.get(expect_col)):
                vals.append(f"预期: {row[expect_col]}")
            if prev_col and pd.notna(row.get(prev_col)):
                vals.append(f"前值: {row[prev_col]}")
            if vals:
                parts[0] += f" ({', '.join(vals)})"
            lines.append(parts[0])
    except Exception as e:
        logger.warning("news_economic_baidu failed: %s", e)
    return lines


def _fetch_cctv_news(curr_date: str, limit: int) -> list[str]:
    """CCTV 新闻联播 — 备用源。"""
    ak = _get_ak()
    lines: list[str] = []
    date_str = curr_date.replace("-", "")
    try:
        df = call_with_retry(ak.news_cctv, date=date_str)
        if df is None or df.empty:
            return lines
        title_col = _col(df, "title", "标题") or df.columns[0]
        content_col = _col(df, "content", "内容")
        for _, row in df.head(limit).iterrows():
            title = str(row[title_col]).strip()
            lines.append(f"- {title}")
            if content_col:
                snippet = str(row[content_col]).strip()[:150]
                if snippet:
                    lines.append(f"  {snippet}")
    except Exception as e:
        logger.warning("news_cctv failed: %s", e)
    return lines


def get_global_news(
    curr_date: str,
    look_back_days: int | None = None,
    limit: int | None = None,
) -> str:
    """Get Chinese macro/market news from multiple sources.

    Sources (in order):
    1. EastMoney global financial news (实时快讯, ~200 articles)
    2. Caixin headlines (财新深度报道, ~100 articles)
    3. Baidu economic calendar (经济日历, macro data releases)
    4. CCTV news (fallback only)
    """
    max_per_source = (limit or 30) // 3 or 10
    sections: list[str] = [f"# China macro & financial news around {curr_date}\n"]

    # Source 1: EastMoney real-time financial news
    em_lines = _fetch_eastmoney_global(max_per_source)
    if em_lines:
        sections.append("## 东方财富 · 财经快讯\n")
        sections.extend(em_lines)
        sections.append("")

    # Source 2: Caixin headlines
    cx_lines = _fetch_caixin_news(max_per_source)
    if cx_lines:
        sections.append("## 财新网 · 深度报道\n")
        sections.extend(cx_lines)
        sections.append("")

    # Source 3: Economic calendar (high-importance events)
    ec_lines = _fetch_economic_calendar(max_per_source)
    if ec_lines:
        sections.append("## 经济日历 · 重要数据\n")
        sections.extend(ec_lines)
        sections.append("")

    # Fallback: if all three sources returned nothing, try CCTV
    if not em_lines and not cx_lines and not ec_lines:
        cctv_lines = _fetch_cctv_news(curr_date, limit or 15)
        if cctv_lines:
            sections.append("## CCTV 新闻联播（备用）\n")
            sections.extend(cctv_lines)
            sections.append("")
        else:
            sections.append("(No news data available)\n")

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Insider Transactions
# ---------------------------------------------------------------------------

def get_insider_transactions(
    ticker: Annotated[str, "A-share ticker symbol"],
) -> str:
    """Get A-share major shareholder transactions (大股东增减持)."""
    ak = _get_ak()
    code = a_share_to_akshare_symbol(ticker)

    lines = [f"# Major shareholder transactions for {ticker.upper()}\n"]
    lines.append("# Source: AKShare\n\n")

    try:
        df = call_with_retry(ak.stock_inner_trade_xq, symbol=code)
        if df is not None and not df.empty:
            lines.append(df.head(20).to_string())
        else:
            lines.append("No insider transaction data available.")
    except Exception as e:
        logger.warning("AKShare insider transactions failed for %s: %s", ticker, e)
        lines.append(f"Data unavailable: {e}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Industry / sector rotation data
# ---------------------------------------------------------------------------

def _resolve_industry_name(ticker: str) -> str | None:
    """Look up the industry name for *ticker*.

    Tries Tushare ``get_stock_industry`` first (reliable Shenwan classification),
    falls back to AKShare ``stock_individual_info_em`` if Tushare is unavailable.
    """
    try:
        from .tushare_provider import get_stock_industry
        name, _code = get_stock_industry(ticker)
        return name
    except Exception:
        pass

    try:
        ak = _get_ak()
        code = a_share_to_akshare_symbol(ticker)
        df = call_with_retry(ak.stock_individual_info_em, symbol=code)
        if df is not None and not df.empty:
            for _, row in df.iterrows():
                key = str(row.iloc[0])
                if "行业" in key:
                    return str(row.iloc[1]).strip()
    except Exception as e:
        logger.warning("AKShare stock_individual_info_em failed for %s: %s", ticker, e)

    return None


def get_industry_data(
    ticker: Annotated[str, "A-share ticker symbol"],
    trade_date: Annotated[str, "Trade date YYYY-MM-DD"],
) -> str:
    """Get industry rotation context: sector ranking + fund flow ranking.

    Combines Tushare (industry classification) with AKShare (industry ranking
    and fund flow data that requires 5000+ Tushare points).
    """
    ak = _get_ak()
    industry_name = _resolve_industry_name(ticker)

    sections: list[str] = [f"## 行业/板块轮动分析 — {ticker}\n"]

    if industry_name:
        sections.append(f"**所属行业（申万一级）**: {industry_name}\n")
    else:
        sections.append("**所属行业**: 未能识别\n")

    # --- 1. Industry ranking by change % ---
    try:
        df = call_with_retry(ak.stock_board_industry_name_em)
        if df is not None and not df.empty:
            rank_col = _col(df, "排名", "序号")
            name_col = _col(df, "板块名称", "板块")
            chg_col = _col(df, "涨跌幅")
            lead_col = _col(df, "领涨股票")
            lead_chg_col = _col(df, "领涨股票-涨跌幅")

            total = len(df)
            sections.append(f"### 行业涨跌排名 (共{total}个行业)\n")

            target_row = None
            if industry_name and name_col:
                match = df[df[name_col].astype(str).str.contains(industry_name, na=False)]
                if not match.empty:
                    target_row = match.iloc[0]

            def _fmt_row(row):
                rank = row[rank_col] if rank_col else "?"
                name = row[name_col] if name_col else "?"
                chg = row[chg_col] if chg_col else "?"
                lead = row[lead_col] if lead_col else ""
                lead_c = row[lead_chg_col] if lead_chg_col else ""
                return f"| {rank} | {name} | {chg}% | {lead} ({lead_c}%) |"

            sections.append("| 排名 | 行业 | 涨跌幅 | 领涨股 |")
            sections.append("|---|---|---|---|")

            for _, row in df.head(5).iterrows():
                sections.append(_fmt_row(row))

            if target_row is not None:
                t_rank = target_row[rank_col] if rank_col else 0
                if isinstance(t_rank, (int, float)) and t_rank > 5:
                    sections.append("| ... | ... | ... | ... |")
                    sections.append(f"| **{_fmt_row(target_row)[2:]}  ← 目标行业")

            sections.append("| ... | ... | ... | ... |")
            for _, row in df.tail(3).iterrows():
                sections.append(_fmt_row(row))

            sections.append("")
    except Exception as e:
        logger.warning("stock_board_industry_name_em failed: %s", e)
        sections.append("行业涨跌排名数据暂不可用\n")

    # --- 2. Industry fund flow ranking ---
    try:
        df = call_with_retry(
            ak.stock_sector_fund_flow_rank,
            indicator="今日",
            sector_type="行业资金流",
        )
        if df is not None and not df.empty:
            rank_col = _col(df, "序号")
            name_col = _col(df, "名称", "行业")
            chg_col = _col(df, "今日涨跌幅")
            net_col = _col(df, "今日主力净流入-净额")
            pct_col = _col(df, "今日主力净流入-净占比")

            sections.append("### 行业资金流向排名\n")
            sections.append("| 排名 | 行业 | 涨跌幅 | 主力净流入(亿) | 净占比 |")
            sections.append("|---|---|---|---|---|")

            def _fmt_flow(row):
                rank = row[rank_col] if rank_col else "?"
                name = row[name_col] if name_col else "?"
                chg = row[chg_col] if chg_col else "?"
                net = float(row[net_col]) / 1e8 if net_col and pd.notna(row.get(net_col)) else 0
                pct = row[pct_col] if pct_col else "?"
                return f"| {rank} | {name} | {chg}% | {net:.1f} | {pct}% |"

            for _, row in df.head(5).iterrows():
                sections.append(_fmt_flow(row))

            target_flow = None
            if industry_name and name_col:
                match = df[df[name_col].astype(str).str.contains(industry_name, na=False)]
                if not match.empty:
                    target_flow = match.iloc[0]

            if target_flow is not None:
                t_rank = target_flow[rank_col] if rank_col else 0
                if isinstance(t_rank, (int, float)) and t_rank > 5:
                    sections.append("| ... | ... | ... | ... | ... |")
                    sections.append(f"| **{_fmt_flow(target_flow)[2:]}  ← 目标行业")

            sections.append("| ... | ... | ... | ... | ... |")
            for _, row in df.tail(3).iterrows():
                sections.append(_fmt_flow(row))

            sections.append("")
    except Exception as e:
        logger.warning("stock_sector_fund_flow_rank failed: %s", e)
        sections.append("行业资金流向数据暂不可用\n")

    return "\n".join(sections)
