"""TuShare data provider — optional backup for Chinese A-share market data.

TuShare requires a user token set in ``TUSHARE_TOKEN``. When the token is
missing, every function raises ``VendorNotConfiguredError`` so the routing
layer silently skips to the next vendor.

Install: ``pip install tushare`` or ``pip install "tradingagents[china-tushare]"``
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Annotated

import pandas as pd

from .config import get_config
from .errors import NoMarketDataError, VendorNotConfiguredError
from .market_utils import a_share_to_akshare_symbol
from .retry import call_with_retry
from .stockstats_utils import MAX_OHLCV_STALE_DAYS_CN, _assert_ohlcv_not_stale, _clean_dataframe
from .utils import safe_ticker_component

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 30


class TuShareNotConfiguredError(VendorNotConfiguredError):
    """Raised when TuShare is selected but TUSHARE_TOKEN is not set."""


def _get_pro():
    """Return a tushare pro_api instance, or raise if not configured."""
    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        raise TuShareNotConfiguredError(
            "TUSHARE_TOKEN environment variable is not set. "
            "Get a token at https://tushare.pro/register"
        )
    try:
        import tushare as ts
        ts.set_token(token)
        return ts.pro_api()
    except ImportError as exc:
        raise ImportError(
            "tushare is not installed. Install with: pip install tushare "
            "or pip install 'tradingagents[china-tushare]'"
        ) from exc


def _to_ts_code(ticker: str) -> str:
    """Convert canonical ticker to TuShare format (600519.SH / 000001.SZ)."""
    code = a_share_to_akshare_symbol(ticker)
    normalized = ticker.upper()
    if normalized.endswith(".SS"):
        return f"{code}.SH"
    elif normalized.endswith(".SZ"):
        return f"{code}.SZ"
    from .market_utils import detect_exchange
    exchange = detect_exchange(ticker)
    if exchange == ".SS":
        return f"{code}.SH"
    return f"{code}.SZ"


def _load_ohlcv_tushare(symbol: str, curr_date: str) -> pd.DataFrame:
    """Fetch and cache TuShare A-share daily OHLCV data."""
    pro = _get_pro()
    ts_code = _to_ts_code(symbol)
    safe_symbol = safe_ticker_component(symbol)
    config = get_config()

    os.makedirs(config["data_cache_dir"], exist_ok=True)
    today_str = datetime.now().strftime("%Y-%m-%d")
    cache_file = os.path.join(
        config["data_cache_dir"], f"{safe_symbol}-TuShare-daily-{today_str}.csv"
    )

    data = None
    if os.path.exists(cache_file):
        cached = pd.read_csv(cache_file, on_bad_lines="skip", encoding="utf-8")
        if not cached.empty and "Close" in cached.columns:
            data = cached

    if data is None:
        df = call_with_retry(
            pro.daily,
            ts_code=ts_code,
            start_date="20200101",
            end_date=datetime.now().strftime("%Y%m%d"),
        )
        if df is None or df.empty:
            raise NoMarketDataError(symbol, symbol, "TuShare returned no data")

        df = df.rename(columns={
            "trade_date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "vol": "Volume",
        })

        keep = [c for c in ("Date", "Open", "High", "Low", "Close", "Volume") if c in df.columns]
        data = df[keep].copy()
        data = data.sort_values("Date").reset_index(drop=True)
        data.to_csv(cache_file, index=False, encoding="utf-8")

    data = _clean_dataframe(data)
    curr_date_dt = pd.to_datetime(curr_date)
    data = data[data["Date"] <= curr_date_dt]
    _assert_ohlcv_not_stale(data, curr_date, symbol, symbol, max_stale_days=MAX_OHLCV_STALE_DAYS_CN)
    return data


def get_stock_data(
    symbol: Annotated[str, "A-share ticker symbol"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Get OHLCV stock data from TuShare."""
    data = _load_ohlcv_tushare(symbol, end_date)

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
    header += "# Data source: TuShare\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    return header + csv_string


def get_indicators(
    symbol: Annotated[str, "A-share ticker symbol"],
    indicator: Annotated[str, "technical indicator to compute"],
    curr_date: Annotated[str, "current trading date, YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:
    """Compute technical indicators from TuShare OHLCV data."""
    from dateutil.relativedelta import relativedelta
    from stockstats import wrap

    best_ind_params = {
        "close_50_sma": "50 SMA", "close_200_sma": "200 SMA",
        "close_10_ema": "10 EMA", "macd": "MACD", "macds": "MACD Signal",
        "macdh": "MACD Histogram", "rsi": "RSI", "boll": "Bollinger Middle",
        "boll_ub": "Bollinger Upper", "boll_lb": "Bollinger Lower",
        "atr": "ATR", "vwma": "VWMA", "mfi": "MFI",
    }

    if indicator not in best_ind_params:
        raise ValueError(f"Indicator {indicator} not supported. Choose from: {list(best_ind_params.keys())}")

    curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    before = curr_date_dt - relativedelta(days=look_back_days)

    data = _load_ohlcv_tushare(symbol, curr_date)
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

    return f"## {indicator} values from {before.strftime('%Y-%m-%d')} to {curr_date}:\n\n{ind_string}"
