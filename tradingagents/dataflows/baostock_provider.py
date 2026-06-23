"""BaoStock data provider for Chinese A-share market data.

BaoStock (证券宝) uses its own TCP-based servers, completely independent from
EastMoney / Sina, making it an ideal fallback when AKShare is rate-limited.
Free, no API key required.

Install: ``pip install baostock`` or ``pip install "tradingagents[china]"``
"""

from __future__ import annotations

import atexit
import contextlib
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
from .market_utils import a_share_to_baostock_symbol
from .retry import call_with_retry
from .stockstats_utils import (
    MAX_OHLCV_STALE_DAYS_CN,
    _assert_ohlcv_not_stale,
    _clean_dataframe,
)
from .utils import safe_ticker_component

logger = logging.getLogger(__name__)

_bs = None
_logged_in = False
_lock = threading.Lock()
_atexit_registered = False


def _get_bs():
    """Lazy import + auto login/logout via atexit. Thread-safe."""
    global _bs, _logged_in, _atexit_registered
    if _bs is not None and _logged_in:
        return _bs
    with _lock:
        if _bs is not None and _logged_in:
            return _bs
        try:
            import baostock as bs
        except ImportError as exc:
            raise ImportError(
                "baostock is not installed. Install with: pip install baostock "
                "or pip install 'tradingagents[china]'"
            ) from exc
        _bs = bs
        lg = bs.login()
        if lg.error_code != "0":
            raise ConnectionError(f"BaoStock login failed: {lg.error_msg}")
        _logged_in = True
        if not _atexit_registered:
            atexit.register(_logout)
            _atexit_registered = True
        return bs


def _logout():
    global _logged_in
    with _lock:
        if _bs is not None and _logged_in:
            with contextlib.suppress(Exception):
                _bs.logout()
            _logged_in = False


def _query_to_df(rs) -> pd.DataFrame:
    """Convert a BaoStock ResultData to a DataFrame."""
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=rs.fields)


def _recent_quarters(curr_date: str, n: int = 4) -> list[tuple[int, int]]:
    """Return the most recent *n* (year, quarter) tuples before *curr_date*."""
    dt = datetime.strptime(curr_date, "%Y-%m-%d")
    month = dt.month
    year = dt.year
    q = (month - 1) // 3
    if q == 0:
        year -= 1
        q = 4
    result = []
    for _ in range(n):
        result.append((year, q))
        q -= 1
        if q == 0:
            q = 4
            year -= 1
    return result


# ---------------------------------------------------------------------------
# OHLCV
# ---------------------------------------------------------------------------

def _load_ohlcv_baostock(symbol: str, curr_date: str) -> pd.DataFrame:
    """Fetch and cache BaoStock A-share daily OHLCV, filtered for look-ahead bias."""
    bs = _get_bs()
    bs_code = a_share_to_baostock_symbol(symbol)
    safe_symbol = safe_ticker_component(symbol)
    config = get_config()

    os.makedirs(config["data_cache_dir"], exist_ok=True)
    today_str = datetime.now().strftime("%Y-%m-%d")
    cache_file = os.path.join(
        config["data_cache_dir"], f"{safe_symbol}-BaoStock-daily-{today_str}.csv"
    )

    data = None
    if os.path.exists(cache_file):
        cached = pd.read_csv(cache_file, on_bad_lines="skip", encoding="utf-8")
        if not cached.empty and "Close" in cached.columns:
            data = cached

    if data is None:
        rs = call_with_retry(
            bs.query_history_k_data_plus,
            bs_code,
            "date,open,high,low,close,volume",
            start_date="2020-01-01",
            end_date=datetime.now().strftime("%Y-%m-%d"),
            frequency="d",
            adjustflag="2",
        )
        raw = _query_to_df(rs)
        if raw.empty:
            raise NoMarketDataError(symbol, symbol, "BaoStock returned no data")

        raw = raw.rename(columns={
            "date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        })
        for col in ("Open", "High", "Low", "Close"):
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
        raw["Volume"] = pd.to_numeric(raw["Volume"], errors="coerce").fillna(0).astype(int)
        raw = raw.dropna(subset=["Close"])
        data = raw[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
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
    """Get OHLCV stock data from BaoStock."""
    data = _load_ohlcv_baostock(symbol, end_date)

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
    header += "# Data source: BaoStock\n"
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
    """Compute technical indicators from BaoStock OHLCV data."""
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

    data = _load_ohlcv_baostock(symbol, curr_date)
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

def _safe_float(val) -> float | None:
    """Convert BaoStock string value to float, returning None for empty/invalid."""
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def get_fundamentals(
    ticker: Annotated[str, "A-share ticker symbol"],
    curr_date: Annotated[str, "current date"] = None,
) -> str:
    """Get company fundamentals from BaoStock (profitability + growth)."""
    bs = _get_bs()
    bs_code = a_share_to_baostock_symbol(ticker)
    if curr_date is None:
        curr_date = datetime.now().strftime("%Y-%m-%d")

    quarters = _recent_quarters(curr_date)
    result: dict = {}

    for year, quarter in quarters:
        try:
            rs = call_with_retry(
                bs.query_profit_data, code=bs_code, year=year, quarter=quarter,
            )
            df = _query_to_df(rs)
            if not df.empty:
                row = df.iloc[0]
                period = f"{year}Q{quarter}"
                entry: dict = {}
                for col in df.columns:
                    v = _safe_float(row[col])
                    if v is not None:
                        entry[col] = v
                if entry:
                    result[period] = entry
                break
        except Exception as e:
            logger.warning("BaoStock profit query failed for %s %dQ%d: %s", ticker, year, quarter, e)

    for year, quarter in quarters:
        try:
            rs = call_with_retry(
                bs.query_growth_data, code=bs_code, year=year, quarter=quarter,
            )
            df = _query_to_df(rs)
            if not df.empty:
                row = df.iloc[0]
                period = f"{year}Q{quarter}"
                growth: dict = {}
                for col in df.columns:
                    v = _safe_float(row[col])
                    if v is not None:
                        growth[col] = v
                if growth:
                    result.setdefault(period, {}).update(growth)
                break
        except Exception as e:
            logger.warning("BaoStock growth query failed for %s %dQ%d: %s", ticker, year, quarter, e)

    if not result:
        raise NoMarketDataError(ticker, ticker, "BaoStock returned no fundamental data")

    return json.dumps(result, ensure_ascii=False, indent=2)


def get_balance_sheet(
    ticker: Annotated[str, "A-share ticker symbol"],
    freq: Annotated[str, "frequency: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    """Get A-share balance sheet data from BaoStock."""
    bs = _get_bs()
    bs_code = a_share_to_baostock_symbol(ticker)
    if curr_date is None:
        curr_date = datetime.now().strftime("%Y-%m-%d")

    quarters = _recent_quarters(curr_date, n=8 if freq == "annual" else 4)
    if freq == "annual":
        quarters = [(y, q) for y, q in quarters if q == 4][:4]

    rows = []
    for year, quarter in quarters:
        try:
            rs = call_with_retry(
                bs.query_balance_data, code=bs_code, year=year, quarter=quarter,
            )
            df = _query_to_df(rs)
            if not df.empty:
                rows.append(df.iloc[0])
        except Exception as e:
            logger.warning("BaoStock balance query failed for %s %dQ%d: %s", ticker, year, quarter, e)

    if not rows:
        raise NoMarketDataError(ticker, ticker, "BaoStock returned no balance sheet data")

    result_df = pd.DataFrame(rows)
    header = f"# Balance Sheet for {ticker.upper()}\n"
    header += f"# Frequency: {freq}\n"
    header += "# Data source: BaoStock\n"
    header += f"# Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    return header + result_df.to_string(max_rows=20, max_cols=15)


def get_cashflow(
    ticker: Annotated[str, "A-share ticker symbol"],
    freq: Annotated[str, "frequency: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    """Get A-share cash flow data from BaoStock."""
    bs = _get_bs()
    bs_code = a_share_to_baostock_symbol(ticker)
    if curr_date is None:
        curr_date = datetime.now().strftime("%Y-%m-%d")

    quarters = _recent_quarters(curr_date, n=8 if freq == "annual" else 4)
    if freq == "annual":
        quarters = [(y, q) for y, q in quarters if q == 4][:4]

    rows = []
    for year, quarter in quarters:
        try:
            rs = call_with_retry(
                bs.query_cash_flow_data, code=bs_code, year=year, quarter=quarter,
            )
            df = _query_to_df(rs)
            if not df.empty:
                rows.append(df.iloc[0])
        except Exception as e:
            logger.warning("BaoStock cashflow query failed for %s %dQ%d: %s", ticker, year, quarter, e)

    if not rows:
        raise NoMarketDataError(ticker, ticker, "BaoStock returned no cash flow data")

    result_df = pd.DataFrame(rows)
    header = f"# Cash Flow for {ticker.upper()}\n"
    header += f"# Frequency: {freq}\n"
    header += "# Data source: BaoStock\n"
    header += f"# Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    return header + result_df.to_string(max_rows=20, max_cols=15)


def get_income_statement(
    ticker: Annotated[str, "A-share ticker symbol"],
    freq: Annotated[str, "frequency: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    """Get A-share income statement (profit) data from BaoStock."""
    bs = _get_bs()
    bs_code = a_share_to_baostock_symbol(ticker)
    if curr_date is None:
        curr_date = datetime.now().strftime("%Y-%m-%d")

    quarters = _recent_quarters(curr_date, n=8 if freq == "annual" else 4)
    if freq == "annual":
        quarters = [(y, q) for y, q in quarters if q == 4][:4]

    rows = []
    for year, quarter in quarters:
        try:
            rs = call_with_retry(
                bs.query_profit_data, code=bs_code, year=year, quarter=quarter,
            )
            df = _query_to_df(rs)
            if not df.empty:
                rows.append(df.iloc[0])
        except Exception as e:
            logger.warning("BaoStock profit query failed for %s %dQ%d: %s", ticker, year, quarter, e)

    if not rows:
        raise NoMarketDataError(ticker, ticker, "BaoStock returned no income data")

    result_df = pd.DataFrame(rows)
    header = f"# Income Statement for {ticker.upper()}\n"
    header += f"# Frequency: {freq}\n"
    header += "# Data source: BaoStock\n"
    header += f"# Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    return header + result_df.to_string(max_rows=20, max_cols=15)
