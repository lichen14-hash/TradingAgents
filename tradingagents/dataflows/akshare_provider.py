"""AKShare data provider for Chinese A-share market data.

AKShare is a free, open-source library wrapping Chinese financial data sources
(Sina, EastMoney, SSE, SZSE). No API key required.

Install: ``pip install akshare`` or ``pip install "tradingagents[china]"``
"""

from __future__ import annotations

import json
import logging
import os
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

def get_news(
    ticker: Annotated[str, "A-share ticker symbol"],
    start_date: Annotated[str, "Start date"],
    end_date: Annotated[str, "End date"],
) -> str:
    """Get A-share specific news from EastMoney via AKShare."""
    ak = _get_ak()
    code = a_share_to_akshare_symbol(ticker)

    # AKShare's stock_news_em uses r"　" as a regex pattern internally.
    # pandas 3.0+ defaults to pyarrow's RE2 engine which doesn't support \u
    # escapes. Temporarily switch to Python's re module for this call.
    _prev = pd.options.mode.string_storage
    pd.options.mode.string_storage = "python"
    try:
        df = call_with_retry(ak.stock_news_em, symbol=code)
    finally:
        pd.options.mode.string_storage = _prev
    if df is None or df.empty:
        raise NoMarketDataError(ticker, ticker, "AKShare returned no news")

    title_col = None
    for c in ("新闻标题", "title", "标题"):
        if c in df.columns:
            title_col = c
            break

    time_col = None
    for c in ("发布时间", "publish_time", "时间"):
        if c in df.columns:
            time_col = c
            break

    source_col = None
    for c in ("文章来源", "source", "来源"):
        if c in df.columns:
            source_col = c
            break

    content_col = None
    for c in ("新闻内容", "content", "内容"):
        if c in df.columns:
            content_col = c
            break

    lines = [f"# News for {ticker.upper()} ({start_date} to {end_date})\n"]
    lines.append("# Source: EastMoney via AKShare\n\n")

    if time_col:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        df = df[(df[time_col] >= start_dt) & (df[time_col] <= end_dt + pd.Timedelta(days=1))]

    count = 0
    for _, row in df.head(30).iterrows():
        title = str(row[title_col]) if title_col else "N/A"
        time_str = str(row[time_col]) if time_col else ""
        source = str(row[source_col]) if source_col else ""
        content = str(row[content_col])[:200] if content_col else ""

        lines.append(f"**{title}**")
        if time_str:
            lines.append(f"  Time: {time_str}")
        if source:
            lines.append(f"  Source: {source}")
        if content:
            lines.append(f"  {content}")
        lines.append("")
        count += 1

    lines.insert(2, f"# Total articles: {count}\n")
    return "\n".join(lines)


def get_global_news(
    curr_date: str,
    look_back_days: int | None = None,
    limit: int | None = None,
) -> str:
    """Get Chinese macro/market news via AKShare (CCTV news)."""
    ak = _get_ak()
    date_str = curr_date.replace("-", "")

    lines = [f"# China macro news around {curr_date}\n"]
    lines.append("# Source: CCTV Finance via AKShare\n\n")

    try:
        df = call_with_retry(ak.news_cctv, date=date_str)
        if df is not None and not df.empty:
            title_col = None
            for c in ("title", "标题"):
                if c in df.columns:
                    title_col = c
                    break
            content_col = None
            for c in ("content", "内容"):
                if c in df.columns:
                    content_col = c
                    break

            max_items = limit or 15
            for _, row in df.head(max_items).iterrows():
                title = str(row[title_col]) if title_col else str(row.iloc[0])
                lines.append(f"- {title}")
                if content_col:
                    snippet = str(row[content_col])[:150]
                    lines.append(f"  {snippet}")
                lines.append("")
    except Exception as e:
        logger.warning("AKShare news_cctv failed: %s", e)
        lines.append("(CCTV news data unavailable)\n")

    return "\n".join(lines)


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
