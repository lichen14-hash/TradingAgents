"""Intraday snapshot provider for optional real-time analysis overlays.

This module intentionally keeps intraday data separate from daily OHLCV data.
Daily indicators should continue to use the last complete daily bar; this
snapshot is only an execution-time overlay for current price/volume context.

Acquisition lives in :mod:`tradingagents.dataflows.spot_quote` (multi-vendor
chain, one request per instrument); this module only renders. The previous
implementation pulled AKShare's full-market spot table -- ~5899 rows over ~59
paginated requests -- to read a single row, and it read that table from
EastMoney's realtime ``push2`` cluster, which is what made the 2026-08-19
intraday batch fail on every A-share.
"""

from __future__ import annotations

import logging
from datetime import datetime

from tradingagents.utils.time_utils import CN_TZ, now_iso, now_str

from .market_utils import is_a_share, is_etf, is_hk_stock
from .spot_quote import SpotQuote, get_spot_quote

logger = logging.getLogger(__name__)


def _fmt_price(value: float | None) -> str | None:
    """Format a price without discarding significant digits.

    The previous renderer used ``f"{value:.4g}"`` for every numeric field, which
    silently degraded the output: a 150.63 close rendered as ``150.6`` and a
    volume of 44,000,000 rendered as ``4.4e+07`` -- two significant digits, from
    which no volume comparison can be made. ETF prices need three decimals
    (0.652) where a three-digit stock price needs two.
    """
    if value is None:
        return None
    return f"{value:.3f}" if abs(value) < 10 else f"{value:.2f}"


def _fmt_pct(value: float | None, *, signed: bool = False) -> str | None:
    if value is None:
        return None
    return f"{value:+.2f}%" if signed else f"{value:.2f}%"


def _fmt_change(value: float | None) -> str | None:
    if value is None:
        return None
    return f"{value:+.3f}" if abs(value) < 10 else f"{value:+.2f}"


def _fmt_volume(value: float | None, *, lots: bool, unit: str = "股") -> str | None:
    """Render volume in shares/units, adding 手 for markets quoted that way.

    Both units are shown for A-shares/ETFs because the vendors disagree about
    which one they report and the analysts' prompts historically saw 手 -- being
    explicit costs one parenthesis and removes a factor-of-100 ambiguity.
    """
    if value is None:
        return None
    text = f"{value:,.0f} {unit}"
    if lots:
        text += f"（{value / 100:,.0f} 手）"
    return text


def _fmt_amount(value: float | None) -> str | None:
    if value is None:
        return None
    if abs(value) >= 1e8:
        return f"{value / 1e8:,.2f} 亿元"
    if abs(value) >= 1e4:
        return f"{value / 1e4:,.2f} 万元"
    return f"{value:,.0f} 元"


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


def render_intraday_snapshot(ticker: str, quote: SpotQuote) -> str:
    """Render a fetched quote as the markdown block the analysts receive."""
    lots = not is_hk_stock(ticker)
    # ETF quantities are 份, not 股. The vendors make no such distinction, so the
    # label is ours to get right.
    unit = "份" if is_etf(ticker) else "股"
    header = (
        "> **盘中分析提示:** 该快照为实时/盘中 overlay，日线行情与技术指标仍基于最近完整交易日。"
        "盘中数据未收盘，不能当作最终日K；交易动作应区分盘中执行与收盘确认。"
    )
    lines = [f"## 盘中行情快照 — {ticker}", "", header, ""]
    if quote.delayed:
        # The realtime vendors are down whenever this fires, so the model would
        # otherwise read a 15-minute-old print as the current price.
        lines.extend([
            "> **延迟数据警告:** 实时行情源不可用，以下报价来自延迟约 15 分钟的备用源。"
            "请勿将其当作当前价执行；仅可用于判断大致方向与量能，涉及具体价位的动作需另行确认。",
            "",
        ])

    lines.extend([
        f"- 数据源: {quote.source}",
        f"- 获取时间: {now_str()} ({now_iso()})",
    ])
    if quote.quote_time:
        lines.append(f"- 行情时间戳: {quote.quote_time}")
    lines.append(f"- 时段状态: {_market_session_note(ticker)}")

    fields = [
        ("名称", quote.name or None),
        ("当前价", _fmt_price(quote.last)),
        ("涨跌幅", _fmt_pct(quote.change_pct, signed=True)),
        ("涨跌额", _fmt_change(quote.change)),
        ("今开", _fmt_price(quote.open)),
        ("最高", _fmt_price(quote.high)),
        ("最低", _fmt_price(quote.low)),
        ("昨收", _fmt_price(quote.prev_close)),
        ("成交量", _fmt_volume(quote.volume_shares, lots=lots, unit=unit)),
        ("成交额", _fmt_amount(quote.amount_yuan)),
        ("换手率", _fmt_pct(quote.turnover_rate)),
        ("量比", None if quote.volume_ratio is None else f"{quote.volume_ratio:.2f}"),
    ]
    for label, value in fields:
        if value:
            lines.append(f"- {label}: {value}")

    return "\n".join(lines)


def get_intraday_snapshot(ticker: str) -> str:
    """Return a markdown intraday snapshot for current execution context.

    Raises on failure rather than returning prose: the collector wraps this in
    ``_safe_call``, which turns the exception into an ``<unavailable: ...>``
    placeholder *and* lets ``validate_bundle_completeness`` see the gap. A
    provider that swallows its own error and returns an explanatory sentence
    instead passes every presence check -- that soft-failure shape is exactly
    what kept the northbound-flow and industry-ranking gaps invisible.
    """
    return render_intraday_snapshot(ticker, get_spot_quote(ticker))
