"""efinance data provider for Chinese A-share market data.

efinance wraps EastMoney's API with a simpler, more stable interface than
AKShare. While both use the same upstream data, efinance's API surface rarely
changes, making it a useful last-resort OHLCV fallback when AKShare's
interface breaks after an upgrade.

Install: ``pip install efinance`` or ``pip install "tradingagents[china]"``
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
from .market_utils import a_share_to_akshare_symbol
from .retry import call_with_retry
from .stockstats_utils import (
    MAX_OHLCV_STALE_DAYS_CN,
    _assert_ohlcv_not_stale,
    _clean_dataframe,
)
from .utils import is_cache_fresh, safe_ticker_component

logger = logging.getLogger(__name__)

_COL_MAP = {
    "日期": "Date",
    "开盘": "Open",
    "最高": "High",
    "最低": "Low",
    "收盘": "Close",
    "成交量": "Volume",
}


def _get_ef():
    """Lazy import efinance to allow graceful degradation when not installed."""
    try:
        import efinance as ef
        return ef
    except ImportError as exc:
        raise ImportError(
            "efinance is not installed. Install with: pip install efinance "
            "or pip install 'tradingagents[china]'"
        ) from exc


# ---------------------------------------------------------------------------
# OHLCV
# ---------------------------------------------------------------------------

def _load_ohlcv_efinance(symbol: str, curr_date: str) -> pd.DataFrame:
    """Fetch and cache efinance A-share daily OHLCV, filtered for look-ahead bias."""
    ef = _get_ef()
    code = a_share_to_akshare_symbol(symbol)
    safe_symbol = safe_ticker_component(symbol)
    config = get_config()

    os.makedirs(config["data_cache_dir"], exist_ok=True)
    today_str = datetime.now().strftime("%Y-%m-%d")
    cache_file = os.path.join(
        config["data_cache_dir"], f"{safe_symbol}-efinance-daily-{today_str}.csv"
    )

    data = None
    if is_cache_fresh(cache_file, symbol):
        cached = pd.read_csv(cache_file, on_bad_lines="skip", encoding="utf-8")
        if not cached.empty and "Close" in cached.columns:
            data = cached

    if data is None:
        df = call_with_retry(
            ef.stock.get_quote_history,
            code,
            beg="20200101",
            end=datetime.now().strftime("%Y%m%d"),
        )
        if df is None or df.empty:
            raise NoMarketDataError(symbol, symbol, "efinance returned no data")

        df = df.rename(columns=_COL_MAP)
        keep = [c for c in ("Date", "Open", "High", "Low", "Close", "Volume") if c in df.columns]
        if not keep or "Close" not in df.columns:
            raise NoMarketDataError(symbol, symbol, "efinance returned unexpected columns")

        data = df[keep].copy()
        for col in ("Open", "High", "Low", "Close"):
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors="coerce")
        data["Volume"] = pd.to_numeric(data["Volume"], errors="coerce").fillna(0).astype(int)
        data.to_csv(cache_file, index=False, encoding="utf-8")

    data = _clean_dataframe(data)
    curr_date_dt = pd.to_datetime(curr_date)
    data = data[data["Date"] <= curr_date_dt]
    _assert_ohlcv_not_stale(
        data, curr_date, symbol, symbol, max_stale_days=MAX_OHLCV_STALE_DAYS_CN,
    )
    return data


def get_stock_data(
    symbol: Annotated[str, "A-share ticker symbol (600519.SS, 300750.SZ)"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Get OHLCV stock data from efinance."""
    data = _load_ohlcv_efinance(symbol, end_date)

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
    header += "# Data source: efinance (EastMoney)\n"
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
    """Compute technical indicators from efinance OHLCV data."""
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

    data = _load_ohlcv_efinance(symbol, curr_date)
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
