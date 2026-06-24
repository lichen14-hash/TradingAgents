"""Hong Kong market signal data sources via AKShare.

Provides forward-looking sentiment signals for HK analysis:
- Southbound capital flow (南向资金) via Stock Connect
- HK Connect fund flow summary (沪深港通资金流向汇总)
- AH premium data for dual-listed stocks

Each signal is market-wide (not ticker-specific), matching the prediction
market vendor signature: ``(topic: str, limit: int | None = None) -> str``.

AKShare wraps free APIs (EastMoney) — no API key required.
Install: ``pip install akshare`` or ``pip install "tradingagents[china]"``
"""

from __future__ import annotations

import logging
from collections.abc import Callable

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
        logger.warning("AKShare HK signal fetch failed (%s): %s", func.__name__, e)
        return None


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _unavailable_section(title: str, reason: str) -> str:
    return (
        f"## 港股市场信号：{title}\n"
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

def _fetch_southbound_flow(limit: int | None) -> str:
    """Southbound capital flow via Stock Connect (南向资金).

    The HK analog of northbound flow — tracks mainland capital flowing INTO
    Hong Kong, the most watched sentiment indicator for HK stocks.
    """
    ak = _get_ak()
    df = _safe_fetch(ak.stock_hsgt_hist_em, symbol="南向资金")
    if df is None:
        return _unavailable_section("南向资金", "AKShare 接口无返回")

    date_col = _find_col(df, ["日期", "date"])
    value_col = _find_col(df, ["当日成交净买额", "净买额", "当日净买额"])

    if date_col is None or value_col is None:
        return _unavailable_section("南向资金", f"列名不匹配: {list(df.columns)}")

    df = df.dropna(subset=[date_col, value_col])
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[date_col, value_col])
    df = df.sort_values(date_col, ascending=False)

    n = min(limit or DEFAULT_ROWS, len(df))
    recent = df.head(n)
    latest = recent.iloc[0]

    cumulative_5d = df.head(5)[value_col].sum()

    lines = [
        "## 港股市场信号：南向资金 (Southbound Capital Flow via Stock Connect)",
        "",
        "南向资金是港股最受关注的情绪指标——代表内地资金流入香港市场的规模。",
        "持续大额净流入通常被解读为利好港股的情绪信号。",
        "",
        f"**最新数据 ({latest[date_col].strftime('%Y-%m-%d')})**:",
        f"- 当日净买入: {_fmt_num(latest[value_col])}",
        f"- 近5日累计: {_fmt_num(cumulative_5d)}",
        "",
        f"**近 {n} 个交易日明细:**",
        "",
        "| 日期 | 净买入(亿元) |",
        "|------|------------|",
    ]

    for _, row in recent.iterrows():
        d = row[date_col].strftime("%Y-%m-%d") if pd.notna(row[date_col]) else "N/A"
        v = _fmt_num(row[value_col])
        lines.append(f"| {d} | {v} |")

    return "\n".join(lines)


def _fetch_hk_connect_summary(limit: int | None) -> str:
    """HK Stock Connect fund flow summary (沪深港通资金流向汇总).

    Shows aggregate fund flow status across all four channels:
    Shanghai→HK, Shenzhen→HK, HK→Shanghai, HK→Shenzhen.
    """
    ak = _get_ak()
    df = _safe_fetch(ak.stock_hsgt_fund_flow_summary_em)
    if df is None:
        return _unavailable_section("港股通资金流向", "AKShare 接口无返回")

    lines = [
        "## 港股市场信号：沪深港通资金流向汇总 (Stock Connect Flow Summary)",
        "",
        "沪深港通四条通道的实时资金流向状态，反映跨境资金动向。",
        "",
        "| 通道 | 类型 | 资金流向 | 成交净买额(亿) |",
        "|------|------|---------|--------------|",
    ]

    for _, row in df.iterrows():
        channel = str(row.get("类型", row.get("名称", "")))
        direction = str(row.get("通道", ""))
        status = str(row.get("资金流向", ""))
        net_buy_col = _find_col(
            pd.DataFrame([row]),
            ["成交净买额", "净买额", "资金净流入"],
        )
        net_val = _fmt_num(row.get(net_buy_col, "N/A")) if net_buy_col else "N/A"
        lines.append(f"| {channel} | {direction} | {status} | {net_val} |")

    return "\n".join(lines)


def _fetch_ah_premium(_limit: int | None) -> str:
    """AH premium data for dual-listed stocks.

    Shows the premium/discount between A-share and H-share prices for
    companies listed on both exchanges. A high premium means A-shares
    are trading at a significant markup over their HK-listed equivalents.
    """
    ak = _get_ak()
    df = _safe_fetch(ak.stock_zh_ah_spot_em)
    if df is None:
        return _unavailable_section("AH溢价", "AKShare 接口无返回")

    name_col = _find_col(df, ["名称", "股票名称"])
    a_code_col = _find_col(df, ["A股代码", "A代码"])
    h_code_col = _find_col(df, ["H股代码", "H代码"])
    premium_col = _find_col(df, ["比价(A/H)", "溢价率", "AH比价"])

    if premium_col is None:
        for c in df.columns:
            if "比价" in c or "溢价" in c or "A/H" in c:
                premium_col = c
                break
    if premium_col is None:
        return _unavailable_section("AH溢价", f"列名不匹配: {list(df.columns)}")

    df[premium_col] = pd.to_numeric(df[premium_col], errors="coerce")
    df = df.dropna(subset=[premium_col])
    df = df.sort_values(premium_col, ascending=False)

    avg_premium = df[premium_col].mean()
    median_premium = df[premium_col].median()

    lines = [
        "## 港股市场信号：AH股溢价 (A/H Premium)",
        "",
        "AH溢价反映同一公司在A股和港股的价格差异。",
        "溢价率 > 1 表示A股溢价（A股比港股贵），< 1 表示A股折价。",
        "",
        f"**整体统计** (共 {len(df)} 只AH股):",
        f"- 平均AH比价: {avg_premium:.2f}",
        f"- 中位AH比价: {median_premium:.2f}",
        "",
        "**溢价最高 Top 10:**",
        "",
        "| 名称 | A股代码 | H股代码 | AH比价 |",
        "|------|--------|--------|-------|",
    ]

    for _, row in df.head(10).iterrows():
        name = str(row.get(name_col, "")) if name_col else ""
        a_code = str(row.get(a_code_col, "")) if a_code_col else ""
        h_code = str(row.get(h_code_col, "")) if h_code_col else ""
        prem = f"{row[premium_col]:.2f}" if pd.notna(row[premium_col]) else "N/A"
        lines.append(f"| {name} | {a_code} | {h_code} | {prem} |")

    lines.extend([
        "",
        "**折价最大 Bottom 5:**",
        "",
        "| 名称 | A股代码 | H股代码 | AH比价 |",
        "|------|--------|--------|-------|",
    ])

    for _, row in df.tail(5).iterrows():
        name = str(row.get(name_col, "")) if name_col else ""
        a_code = str(row.get(a_code_col, "")) if a_code_col else ""
        h_code = str(row.get(h_code_col, "")) if h_code_col else ""
        prem = f"{row[premium_col]:.2f}" if pd.notna(row[premium_col]) else "N/A"
        lines.append(f"| {name} | {a_code} | {h_code} | {prem} |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_SIGNAL_FETCHERS: dict[str, Callable[..., str]] = {
    "southbound_flow": _fetch_southbound_flow,
    "hk_connect_summary": _fetch_hk_connect_summary,
    "ah_premium": _fetch_ah_premium,
}

_KEYWORD_MAP: dict[str, str] = {
    "southbound": "southbound_flow",
    "south": "southbound_flow",
    "南向": "southbound_flow",
    "connect": "hk_connect_summary",
    "港股通": "hk_connect_summary",
    "资金流向": "hk_connect_summary",
    "ah": "ah_premium",
    "溢价": "ah_premium",
    "premium": "ah_premium",
}


def get_hk_market_signals(topic: str, limit: int | None = None) -> str:
    """Fetch HK market signal data for a given topic.

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
