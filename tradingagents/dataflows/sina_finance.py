"""Sina Finance data provider for Chinese-accessible stock data.

Sina Finance APIs are accessible from mainland China where Yahoo Finance
is blocked.  This module provides OHLCV price data and technical indicators
for US-listed stocks via Sina's public US stock API.
"""

import json
import logging
import os
from datetime import datetime
from typing import Annotated

import pandas as pd
import requests
from stockstats import wrap

from .config import get_config
from .errors import NoMarketDataError
from .market_utils import a_share_to_sina_symbol, is_a_share
from .retry import call_with_retry
from .stockstats_utils import (
    MAX_OHLCV_STALE_DAYS,
    MAX_OHLCV_STALE_DAYS_CN,
    _assert_ohlcv_not_stale,
    _clean_dataframe,
)
from .utils import is_cache_fresh, safe_ticker_component

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 30

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Referer": "https://finance.sina.com.cn",
}


def _fetch_a_share_daily_klines(symbol: str) -> list[dict]:
    """Fetch A-share daily OHLCV data from Sina Finance CN stock API."""
    sina_sym = a_share_to_sina_symbol(symbol)
    url = (
        "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php"
        "/CN_MarketData.getKLineData"
    )
    params = {"symbol": sina_sym, "scale": "240", "ma": "no", "datalen": "5000"}

    r = call_with_retry(requests.get, url, params=params, headers=_HEADERS, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()

    data = json.loads(r.text)
    if not data:
        return []

    rows = []
    for k in data:
        rows.append({
            "d": k.get("day", k.get("d", "")),
            "o": k.get("open", k.get("o", "0")),
            "h": k.get("high", k.get("h", "0")),
            "l": k.get("low", k.get("l", "0")),
            "c": k.get("close", k.get("c", "0")),
            "v": k.get("volume", k.get("v", "0")),
        })
    return rows


def _fetch_us_daily_klines(symbol: str) -> list[dict]:
    """Fetch US stock daily OHLCV data from Sina Finance US stock API."""
    sym = symbol.upper()
    url = (
        f"https://stock.finance.sina.com.cn/usstock/api/jsonp_v2.php"
        f"/var%20_{sym}/US_MinKService.getDailyK"
    )
    params = {"symbol": sym, "_": "1"}

    r = call_with_retry(requests.get, url, params=params, headers=_HEADERS, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()

    text = r.text
    start = text.index("(")
    end = text.rindex(")")
    return json.loads(text[start + 1 : end])


def _fetch_daily_klines(symbol: str) -> list[dict]:
    """Fetch daily OHLCV data from Sina Finance, routing by market."""
    if is_a_share(symbol):
        return _fetch_a_share_daily_klines(symbol)
    return _fetch_us_daily_klines(symbol)


def _load_ohlcv_sina(symbol: str, curr_date: str) -> pd.DataFrame:
    """Fetch and cache Sina OHLCV data, filtered to prevent look-ahead bias."""
    safe_symbol = safe_ticker_component(symbol.upper())
    config = get_config()

    os.makedirs(config["data_cache_dir"], exist_ok=True)
    today_str = datetime.now().strftime("%Y-%m-%d")
    cache_file = os.path.join(
        config["data_cache_dir"], f"{safe_symbol}-Sina-daily-{today_str}.csv"
    )

    data = None
    if is_cache_fresh(cache_file, symbol):
        cached = pd.read_csv(cache_file, on_bad_lines="skip", encoding="utf-8")
        if not cached.empty and "Close" in cached.columns:
            data = cached

    if data is None:
        klines = _fetch_daily_klines(symbol)
        if not klines:
            raise NoMarketDataError(symbol, symbol, "Sina returned no data")

        rows = []
        for k in klines:
            rows.append(
                {
                    "Date": k["d"],
                    "Open": float(k["o"]),
                    "High": float(k["h"]),
                    "Low": float(k["l"]),
                    "Close": float(k["c"]),
                    "Volume": int(k["v"]),
                }
            )
        data = pd.DataFrame(rows)
        data.to_csv(cache_file, index=False, encoding="utf-8")

    data = _clean_dataframe(data)
    curr_date_dt = pd.to_datetime(curr_date)
    data = data[data["Date"] <= curr_date_dt]
    stale_limit = MAX_OHLCV_STALE_DAYS_CN if is_a_share(symbol) else MAX_OHLCV_STALE_DAYS
    _assert_ohlcv_not_stale(data, curr_date, symbol, symbol, max_stale_days=stale_limit)
    return data


def get_stock_data(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Get OHLCV stock data from Sina Finance."""
    data = _load_ohlcv_sina(symbol, end_date)

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
    header += "# Data source: Sina Finance\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    return header + csv_string


def get_indicators(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to compute"],
    curr_date: Annotated[str, "current trading date, YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:
    """Compute technical indicators from Sina Finance OHLCV data."""
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

    data = _load_ohlcv_sina(symbol, curr_date)
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
