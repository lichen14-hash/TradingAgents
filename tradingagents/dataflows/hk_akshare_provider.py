"""AKShare data provider for Hong Kong stock market data.

Wraps AKShare's ``stock_hk_*`` functions (EastMoney / Sina sources) for
OHLCV, fundamentals, and news. No API key required.

OHLCV uses dual-source fallback: ``stock_hk_hist`` (EastMoney) first,
then ``stock_hk_daily`` (Sina) if the primary fails.

Install: ``pip install akshare`` or ``pip install "tradingagents[china]"``
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Annotated

import pandas as pd
from stockstats import wrap

from .config import get_config
from .errors import NoMarketDataError
from .market_utils import hk_to_akshare_symbol
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
}


def _get_ak():
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

def _load_ohlcv_hk(symbol: str, curr_date: str) -> pd.DataFrame:
    """Fetch and cache HK daily OHLCV, filtered for look-ahead bias.

    Dual-source: tries ``stock_hk_hist`` (EastMoney) first, then
    ``stock_hk_daily`` (Sina) as fallback.
    """
    ak = _get_ak()
    code = hk_to_akshare_symbol(symbol)
    safe_symbol = safe_ticker_component(symbol)
    config = get_config()

    os.makedirs(config["data_cache_dir"], exist_ok=True)
    today_str = datetime.now().strftime("%Y-%m-%d")
    cache_file = os.path.join(
        config["data_cache_dir"], f"{safe_symbol}-AKShare-HK-daily-{today_str}.csv"
    )

    data = None
    if is_cache_fresh(cache_file, symbol):
        cached = pd.read_csv(cache_file, on_bad_lines="skip", encoding="utf-8")
        if not cached.empty and "Close" in cached.columns:
            data = cached

    if data is None:
        df = None
        # Primary: EastMoney
        try:
            df = call_with_retry(
                ak.stock_hk_hist,
                symbol=code,
                period="daily",
                start_date="20200101",
                end_date=datetime.now().strftime("%Y%m%d"),
                adjust="qfq",
            )
        except Exception as e:
            logger.warning("stock_hk_hist failed for %s: %s", code, e)

        # Fallback: Sina
        if df is None or df.empty:
            try:
                df = call_with_retry(
                    ak.stock_hk_daily,
                    symbol=code,
                    adjust="qfq",
                )
            except Exception as e:
                logger.warning("stock_hk_daily fallback also failed for %s: %s", code, e)

        if df is None or df.empty:
            raise NoMarketDataError(symbol, symbol, "No HK OHLCV from AKShare")

        df = df.rename(columns=_COL_MAP)
        # stock_hk_daily uses lowercase English column names already
        col_remap = {"date": "Date", "open": "Open", "high": "High",
                     "low": "Low", "close": "Close", "volume": "Volume"}
        df = df.rename(columns=col_remap)

        keep = [c for c in ("Date", "Open", "High", "Low", "Close", "Volume") if c in df.columns]
        data = df[keep].copy()
        data.to_csv(cache_file, index=False, encoding="utf-8")

    data = _clean_dataframe(data)
    curr_date_dt = pd.to_datetime(curr_date)
    data = data[data["Date"] <= curr_date_dt]
    _assert_ohlcv_not_stale(data, curr_date, symbol, symbol, max_stale_days=MAX_OHLCV_STALE_DAYS_CN)
    return data


def get_stock_data(
    symbol: Annotated[str, "HK ticker symbol (00700.HK)"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Get OHLCV stock data from AKShare for HK stocks."""
    data = _load_ohlcv_hk(symbol, end_date)

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
    header += "# Data source: AKShare HK (EastMoney/Sina)\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    return header + csv_string


# ---------------------------------------------------------------------------
# Technical Indicators
# ---------------------------------------------------------------------------

def get_indicators(
    symbol: Annotated[str, "HK ticker symbol"],
    indicator: Annotated[str, "technical indicator to compute"],
    curr_date: Annotated[str, "current trading date, YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:
    """Compute technical indicators from AKShare HK OHLCV data."""
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

    data = _load_ohlcv_hk(symbol, curr_date)
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
    ticker: Annotated[str, "HK ticker symbol (00700.HK)"],
    curr_date: Annotated[str, "current date"] = None,
) -> str:
    """Get company fundamentals from AKShare (EastMoney) for HK stocks."""
    ak = _get_ak()
    code = hk_to_akshare_symbol(ticker)

    result = {}

    # Company profile
    try:
        df = call_with_retry(ak.stock_hk_company_profile_em, symbol=code)
        if df is not None and not df.empty:
            for _, row in df.iterrows():
                key = str(row.iloc[0]).strip() if len(row) > 0 else ""
                val = str(row.iloc[1]).strip() if len(row) > 1 else ""
                if key:
                    result[key] = val
    except Exception as e:
        logger.warning("stock_hk_company_profile_em failed for %s: %s", code, e)

    # Financial indicators
    try:
        df = call_with_retry(ak.stock_hk_financial_indicator_em, symbol=code)
        if df is not None and not df.empty:
            latest = df.iloc[0]
            for col in df.columns:
                val = latest.get(col)
                if pd.notna(val):
                    result[col] = str(val)
    except Exception as e:
        logger.warning("stock_hk_financial_indicator_em failed for %s: %s", code, e)

    if not result:
        return f"No fundamentals data available for {ticker} from AKShare."

    lines = [f"## Company Fundamentals for {ticker} (Hong Kong)\n"]
    lines.append(f"Data source: AKShare (EastMoney) | As of: {curr_date or 'latest'}\n")
    for k, v in result.items():
        lines.append(f"- **{k}**: {v}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# News
# ---------------------------------------------------------------------------

def get_news(
    ticker: Annotated[str, "HK ticker symbol"],
    start_date: Annotated[str, "start date"],
    end_date: Annotated[str, "end date"],
) -> str:
    """Get stock news from EastMoney for HK stocks."""
    ak = _get_ak()
    code = hk_to_akshare_symbol(ticker)

    try:
        old_setting = pd.options.mode.string_storage
        pd.options.mode.string_storage = "python"
        try:
            df = call_with_retry(ak.stock_news_em, symbol=code)
        finally:
            pd.options.mode.string_storage = old_setting
    except Exception as e:
        logger.warning("stock_news_em failed for HK %s: %s", code, e)
        return f"No news available for {ticker} from AKShare."

    if df is None or df.empty:
        return f"No news available for {ticker}."

    date_col = None
    for c in ("发布时间", "日期", "time", "date"):
        if c in df.columns:
            date_col = c
            break

    title_col = None
    for c in ("新闻标题", "标题", "title"):
        if c in df.columns:
            title_col = c
            break

    content_col = None
    for c in ("新闻内容", "内容", "content"):
        if c in df.columns:
            content_col = c
            break

    lines = [f"## Recent news for {ticker} (Hong Kong)\n"]
    limit = 15
    for _, row in df.head(limit).iterrows():
        date_str = str(row[date_col]) if date_col else ""
        title = str(row[title_col]) if title_col else ""
        snippet = ""
        if content_col and pd.notna(row.get(content_col)):
            snippet = str(row[content_col])[:120] + "..."
        lines.append(f"- [{date_str}] **{title}**")
        if snippet:
            lines.append(f"  {snippet}")

    return "\n".join(lines)


def get_global_news(
    curr_date: str,
    look_back_days: int = 7,
    limit: int = 10,
) -> str:
    """Get global financial news relevant to HK market (reuses CCTV news)."""
    from .akshare_provider import get_global_news as _ak_global_news
    return _ak_global_news(curr_date, look_back_days, limit)
