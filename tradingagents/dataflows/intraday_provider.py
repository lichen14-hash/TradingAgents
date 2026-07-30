"""Intraday snapshot provider for optional real-time analysis overlays.

This module intentionally keeps intraday data separate from daily OHLCV data.
Daily indicators should continue to use the last complete daily bar; this
snapshot is only an execution-time overlay for current price/volume context.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import pandas as pd

from tradingagents.utils.time_utils import CN_TZ, now_iso, now_str

from .market_utils import a_share_to_akshare_symbol, is_a_share, is_etf, is_hk_stock, hk_to_akshare_symbol
from .retry import call_with_retry

logger = logging.getLogger(__name__)


def _get_ak():
    try:
        import akshare as ak
        return ak
    except ImportError as exc:
        raise ImportError("akshare is not installed; intraday snapshot unavailable") from exc


def _normalize_code(value: Any) -> str:
    text = str(value or "").strip().upper()
    return text.replace("SH", "").replace("SZ", "").replace("HK", "").replace(".", "")


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    columns = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in columns:
            return columns[key]
    for col in df.columns:
        text = str(col).strip().lower()
        if any(candidate.strip().lower() in text for candidate in candidates):
            return col
    return None


def _pick_row(df: pd.DataFrame, ticker: str) -> pd.Series | None:
    code = _normalize_code(hk_to_akshare_symbol(ticker) if is_hk_stock(ticker) else a_share_to_akshare_symbol(ticker))
    for candidate in ("代码", "证券代码", "code", "symbol"):
        col = _find_col(df, [candidate])
        if col is None:
            continue
        matched = df[df[col].map(_normalize_code) == code]
        if not matched.empty:
            return matched.iloc[0]
    if len(df) == 1:
        return df.iloc[0]
    return None


def _get_value(row: pd.Series, names: list[str]) -> Any:
    for name in names:
        for key in row.index:
            if str(key).strip().lower() == name.strip().lower():
                return row[key]
    for key in row.index:
        text = str(key).strip().lower()
        if any(name.strip().lower() in text for name in names):
            return row[key]
    return None


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        if pd.isna(value):
            return "N/A"
    except Exception:
        pass
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _market_session_note(ticker: str) -> str:
    current = datetime.now(CN_TZ)
    hm = current.hour * 60 + current.minute
    if is_a_share(ticker):
        if 9 * 60 + 30 <= hm <= 11 * 60 + 30 or 13 * 60 <= hm <= 15 * 60:
            return "A股交易时段内"
        if 11 * 60 + 30 < hm < 13 * 60:
            return "A股午间休市，快照反映上午盘/最近可得数据"
        return "非A股连续交易时段，快照可能为最近可得行情"
    if is_hk_stock(ticker):
        if 9 * 60 + 30 <= hm <= 12 * 60 or 13 * 60 <= hm <= 16 * 60:
            return "港股交易时段内"
        if 12 * 60 < hm < 13 * 60:
            return "港股午间休市，快照反映上午盘/最近可得数据"
        return "非港股连续交易时段，快照可能为最近可得行情"
    return "非本地市场时段判断，快照可能为延迟/最近可得行情"


def _fetch_spot_frame(ticker: str) -> tuple[str, pd.DataFrame]:
    ak = _get_ak()
    if is_hk_stock(ticker):
        return "AKShare stock_hk_spot_em", call_with_retry(ak.stock_hk_spot_em)
    if is_etf(ticker):
        return "AKShare fund_etf_spot_em", call_with_retry(ak.fund_etf_spot_em)
    if is_a_share(ticker):
        return "AKShare stock_zh_a_spot_em", call_with_retry(ak.stock_zh_a_spot_em)
    raise ValueError(f"Intraday snapshot only supports A-share/ETF/HK tickers for now: {ticker}")


def get_intraday_snapshot(ticker: str) -> str:
    """Return a markdown intraday snapshot for current execution context."""
    source, df = _fetch_spot_frame(ticker)
    if df is None or df.empty:
        return f"## 盘中行情快照 — {ticker}\n\n盘中数据暂不可用：{source} returned no rows."

    row = _pick_row(df, ticker)
    if row is None:
        return f"## 盘中行情快照 — {ticker}\n\n盘中数据暂不可用：{source} 未找到匹配代码。"

    fields = [
        ("名称", ["名称", "name"]),
        ("当前价", ["最新价", "现价", "最新", "price", "last"]),
        ("涨跌幅", ["涨跌幅", "涨幅", "changepercent", "pct_chg"]),
        ("涨跌额", ["涨跌额", "change"]),
        ("今开", ["今开", "开盘", "open"]),
        ("最高", ["最高", "high"]),
        ("最低", ["最低", "low"]),
        ("昨收", ["昨收", "prev_close", "昨收价"]),
        ("成交量", ["成交量", "volume"]),
        ("成交额", ["成交额", "amount", "turnover"]),
        ("换手率", ["换手率", "turnover_rate"]),
        ("量比", ["量比"]),
    ]

    lines = [
        f"## 盘中行情快照 — {ticker}",
        "",
        f"> **盘中分析提示:** 该快照为实时/盘中 overlay，日线行情与技术指标仍基于最近完整交易日。盘中数据未收盘，不能当作最终日K；交易动作应区分盘中执行与收盘确认。",
        "",
        f"- 数据源: {source}",
        f"- 获取时间: {now_str()} ({now_iso()})",
        f"- 时段状态: {_market_session_note(ticker)}",
    ]
    for label, names in fields:
        value = _get_value(row, names)
        if _fmt(value) != "N/A":
            lines.append(f"- {label}: {_fmt(value)}")

    return "\n".join(lines)
