"""A-share market signal data sources via AKShare.

Replaces Polymarket (unreachable from China) with locally-available
forward-looking sentiment signals: northbound capital flows, margin
trading data, and institutional activity (龙虎榜).

Each signal is market-wide (not ticker-specific), matching the prediction
market vendor signature: ``(topic: str, limit: int | None = None) -> str``.

AKShare wraps free APIs (SSE, SZSE, EastMoney) — no API key required.
Install: ``pip install akshare`` or ``pip install "tradingagents[china]"``
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime, timedelta

import pandas as pd

from .retry import call_with_retry

logger = logging.getLogger(__name__)

DEFAULT_ROWS = 10


def _get_ak():
    try:
        import akshare as ak
        return ak
    except ImportError as exc:
        raise ImportError(
            "akshare is not installed. Install with: pip install akshare "
            "or pip install 'tradingagents[china]'"
        ) from exc


def _safe_fetch(func, *args, **kwargs) -> pd.DataFrame | None:
    try:
        df = call_with_retry(func, *args, **kwargs)
        if df is None or df.empty:
            return None
        return df
    except Exception as e:
        logger.warning("AKShare signal fetch failed (%s): %s", func.__name__, e)
        return None


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _unavailable_section(title: str, reason: str) -> str:
    return (
        f"## A股市场信号：{title}\n"
        f"数据不可用：{reason}\n"
        f"请在缺少该信号的情况下继续分析。"
    )


def _fmt_num(val, unit: str = "亿元") -> str:
    if pd.isna(val):
        return "N/A"
    try:
        v = float(val)
        sign = "+" if v > 0 else ""
        return f"{sign}{v:,.2f} {unit}"
    except (ValueError, TypeError):
        return str(val)


# ---------------------------------------------------------------------------
# Signal fetchers
# ---------------------------------------------------------------------------

def _fetch_northbound_flow(limit: int | None = None) -> str:
    """北向资金 — Stock Connect northbound net flow."""
    ak = _get_ak()
    rows = limit or DEFAULT_ROWS

    df = _safe_fetch(ak.stock_hsgt_hist_em, symbol="北向资金")
    if df is None:
        return _unavailable_section("北向资金 (Northbound Flow)", "AKShare 未返回数据")

    date_col = _find_col(df, ["日期", "date", "trade_date", "DateTime"])
    value_col = _find_col(df, [
        "当日成交净买额", "当日净流入", "北向净流入", "净流入",
        "北向资金", "value",
    ])

    if date_col is None or value_col is None:
        all_cols = ", ".join(df.columns.tolist()[:15])
        logger.warning("Northbound flow columns not matched. Available: %s", all_cols)
        date_col = date_col or df.columns[0]
        value_col = value_col or df.columns[1]

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[date_col, value_col]).sort_values(date_col, ascending=False)

    recent = df.head(rows).copy()

    lines = [
        "## A股市场信号：北向资金 (Northbound Capital Flow)",
        "",
        "沪深港通北向资金净流入是A股最受关注的情绪指标。",
        "持续流入代表外资看好，持续流出代表外资谨慎。",
        "",
    ]

    if not recent.empty:
        latest = recent.iloc[0]
        latest_val = latest[value_col]
        latest_date = latest[date_col].strftime("%Y-%m-%d") if pd.notna(latest[date_col]) else "N/A"
        lines.append(f"- **最新数据** ({latest_date}): 净流入 {_fmt_num(latest_val)}")

        recent_vals = recent[value_col].dropna()
        if len(recent_vals) >= 5:
            sum_5d = recent_vals.head(5).sum()
            inflow_days = (recent_vals.head(5) > 0).sum()
            lines.append(f"- **近5日累计**: {_fmt_num(sum_5d)}（{inflow_days}日净流入，{5 - inflow_days}日净流出）")
        if len(recent_vals) >= 2:
            sum_all = recent_vals.sum()
            lines.append(f"- **近{len(recent_vals)}日累计**: {_fmt_num(sum_all)}")

        lines.append("")
        lines.append(f"| 日期 | 净流入({value_col}) |")
        lines.append("| --- | --- |")
        for _, row in recent.iterrows():
            d = row[date_col].strftime("%Y-%m-%d") if pd.notna(row[date_col]) else "?"
            v = _fmt_num(row[value_col])
            lines.append(f"| {d} | {v} |")

    return "\n".join(lines)


def _fetch_margin_trading(limit: int | None = None) -> str:
    """融资融券 — Margin trading balance (SSE aggregate)."""
    ak = _get_ak()
    rows = limit or DEFAULT_ROWS

    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")

    df = _safe_fetch(ak.stock_margin_sse, start_date=start_date, end_date=end_date)
    if df is None:
        return _unavailable_section("融资融券 (Margin Trading)", "AKShare 未返回数据")

    date_col = _find_col(df, ["信用交易日期", "日期", "date", "trade_date"])
    rzye_col = _find_col(df, ["融资余额", "融资余额(元)", "rzye"])
    rqye_col = _find_col(df, ["融券余量金额", "融券余额", "融券余额(元)", "rqye"])
    rzrqye_col = _find_col(df, ["融资融券余额", "融资融券余额(元)", "rzrqye"])

    if date_col is None:
        all_cols = ", ".join(df.columns.tolist()[:15])
        logger.warning("Margin trading columns not matched. Available: %s", all_cols)
        date_col = df.columns[0]

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col, ascending=False)
    recent = df.head(rows).copy()

    lines = [
        "## A股市场信号：融资融券 (Margin Trading Balance)",
        "",
        "融资余额增加反映杠杆多头力量增强（看涨信号），",
        "融券余额增加反映做空力量增强（看跌信号）。",
        "",
    ]

    if not recent.empty:
        latest = recent.iloc[0]
        d = latest[date_col].strftime("%Y-%m-%d") if pd.notna(latest[date_col]) else "N/A"
        lines.append(f"- **最新数据日期**: {d}")

        if rzye_col and pd.notna(latest.get(rzye_col)):
            val = float(latest[rzye_col]) / 1e8
            lines.append(f"- **融资余额**: {val:,.2f} 亿元")
        if rqye_col and pd.notna(latest.get(rqye_col)):
            val = float(latest[rqye_col]) / 1e8
            lines.append(f"- **融券余额**: {val:,.2f} 亿元")
        if rzrqye_col and pd.notna(latest.get(rzrqye_col)):
            val = float(latest[rzrqye_col]) / 1e8
            lines.append(f"- **融资融券余额合计**: {val:,.2f} 亿元")

        if rzye_col and len(recent) >= 5:
            recent[rzye_col] = pd.to_numeric(recent[rzye_col], errors="coerce")
            vals = recent[rzye_col].dropna()
            if len(vals) >= 5:
                change = (vals.iloc[0] - vals.iloc[4]) / 1e8
                direction = "增加" if change > 0 else "减少"
                lines.append(f"- **近5日融资余额变动**: {direction} {abs(change):,.2f} 亿元")

        lines.append("")
        header_cols = ["日期"]
        if rzye_col:
            header_cols.append("融资余额(亿元)")
        if rqye_col:
            header_cols.append("融券余额(亿元)")
        lines.append("| " + " | ".join(header_cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(header_cols)) + " |")
        for _, row in recent.iterrows():
            d = row[date_col].strftime("%Y-%m-%d") if pd.notna(row[date_col]) else "?"
            cells = [d]
            if rzye_col:
                v = float(row[rzye_col]) / 1e8 if pd.notna(row.get(rzye_col)) else 0
                cells.append(f"{v:,.2f}")
            if rqye_col:
                v = float(row[rqye_col]) / 1e8 if pd.notna(row.get(rqye_col)) else 0
                cells.append(f"{v:,.2f}")
            lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def _fetch_top_institutional(limit: int | None = None) -> str:
    """龙虎榜 — Top institutional trading activity."""
    ak = _get_ak()
    rows = limit or 15

    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d")

    df = _safe_fetch(
        ak.stock_lhb_detail_em,
        start_date=start_date,
        end_date=end_date,
    )
    if df is None:
        return _unavailable_section("龙虎榜 (Top Institutional Activity)", "AKShare 未返回数据")

    date_col = _find_col(df, ["上榜日", "上榜日期", "日期", "date"])
    name_col = _find_col(df, ["名称", "股票名称", "name"])
    code_col = _find_col(df, ["代码", "股票代码", "code"])
    reason_col = _find_col(df, ["上榜原因", "原因", "reason"])
    buy_col = _find_col(df, ["龙虎榜买入额", "买入额", "buy"])
    net_col = _find_col(df, ["龙虎榜净买额", "净买额", "净额", "net"])

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).sort_values(date_col, ascending=False)
    recent = df.head(rows)

    lines = [
        "## A股市场信号：龙虎榜 (Top Institutional Activity)",
        "",
        "龙虎榜记录了涨跌幅异动、换手率异常等股票的机构/游资席位交易明细。",
        "机构净买入集中的个股/板块反映主力资金方向。",
        "",
    ]

    if not recent.empty:
        header = ["日期"]
        if code_col:
            header.append("代码")
        if name_col:
            header.append("名称")
        if reason_col:
            header.append("上榜原因")
        if net_col:
            header.append("净买额(万元)")
        elif buy_col:
            header.append("买入额(万元)")

        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")

        for _, row in recent.iterrows():
            cells = []
            if date_col and pd.notna(row.get(date_col)):
                cells.append(row[date_col].strftime("%Y-%m-%d"))
            else:
                cells.append("?")
            if code_col:
                cells.append(str(row.get(code_col, "")))
            if name_col:
                cells.append(str(row.get(name_col, "")))
            if reason_col:
                cells.append(str(row.get(reason_col, ""))[:20])
            if net_col:
                v = row.get(net_col)
                cells.append(f"{float(v) / 1e4:,.0f}" if pd.notna(v) else "N/A")
            elif buy_col:
                v = row.get(buy_col)
                cells.append(f"{float(v) / 1e4:,.0f}" if pd.notna(v) else "N/A")
            lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_SIGNAL_FETCHERS: dict[str, Callable[..., str]] = {
    "northbound_flow": _fetch_northbound_flow,
    "margin_trading": _fetch_margin_trading,
    "top_institutional": _fetch_top_institutional,
}

_KEYWORD_MAP: dict[str, str] = {
    "northbound": "northbound_flow",
    "north": "northbound_flow",
    "北向": "northbound_flow",
    "margin": "margin_trading",
    "融资": "margin_trading",
    "lhb": "top_institutional",
    "龙虎": "top_institutional",
    "institutional": "top_institutional",
}


def get_cn_market_signals(topic: str, limit: int | None = None) -> str:
    """Fetch A-share market signal data for a given topic.

    Drop-in replacement for Polymarket's ``get_prediction_markets``.
    """
    key = topic.strip().lower().replace(" ", "_").replace("-", "_")
    fetcher = _SIGNAL_FETCHERS.get(key)
    if fetcher is None:
        for kw, canonical in _KEYWORD_MAP.items():
            if kw in key:
                fetcher = _SIGNAL_FETCHERS.get(canonical)
                break
    if fetcher is None:
        return _unavailable_section(
            topic,
            f"未知信号主题 '{topic}'，可选: {', '.join(_SIGNAL_FETCHERS.keys())}",
        )
    return fetcher(limit)
