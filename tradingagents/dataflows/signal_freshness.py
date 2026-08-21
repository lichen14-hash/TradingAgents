"""日频市场信号的新鲜度守卫（A股与港股共用）。

"有内容"和"是当期内容"是两个问题，取数代码通常只问了第一个：从数据源拿到 DataFrame、
`dropna` 之后取最新一行、把它的日期原样写成"最新数据 (YYYY-MM-DD)"。一旦上游的某一列
从某天起全变成 NaN，"最新一行"就永久停在那一天，而输出仍然是一张格式完好的表格——
任何存在性检查都是干净的，模型会照着它推理。

2026-08-19 批量里 6 只 A股/ETF 的提示词都写着"北向资金 最新数据 (2024-08-16)"，
落后 732 天，紧挨着日期正确的融资融券。根因不是取数失败：沪深交易所自 2024-08-19 起
停止披露北向资金每日净买额，AKShare 仍然返回到今天的日期行，但金额列全是 NaN。

detection 放在这里而不是各模块内联，是因为同一个检查在 `cn_market_signals` 和
`hk_market_signals` 里各有一份的话必然漂移——本项目已经在数据完整性检查上出过一次。
"""

from __future__ import annotations

import logging

import pandas as pd

from tradingagents.utils.time_utils import now

logger = logging.getLogger(__name__)

# 落后多少个自然日就不该再被称作"最新数据"。留出长假（最长 9 天）加 T+1 披露的余量。
MAX_SIGNAL_LAG_DAYS = 12

# 沪深交易所自 2024-08-19 起停止披露沪深港通北向资金每日净买额，最后一期 2024-08-16。
NORTHBOUND_DISCLOSURE_END = pd.Timestamp("2024-08-16")

NORTHBOUND_DISCONTINUED_NOTE = (
    "沪深交易所自 2024-08-19 起停止披露北向资金每日净买额，该指标已无当期数据，"
    "请勿据此判断外资动向；可参考南向资金或个股沪深港通持股"
)


def lag_days(latest) -> int | None:
    """自报最新日期距今的自然日数；日期不可解析时返回 None。"""
    ts = pd.to_datetime(latest, errors="coerce")
    if pd.isna(ts):
        return None
    return (pd.Timestamp(now().date()) - ts.normalize()).days


def stale_reason(label: str, latest, *, extra: str = "") -> str | None:
    """落后超过预算时返回不可用理由文案，否则返回 ``None``。

    调用方负责把理由包进自己的不可用段落格式里——两个市场的段落标题不同，
    但"什么算陈旧"必须只有一个定义。
    """
    lag = lag_days(latest)
    if lag is None or lag <= MAX_SIGNAL_LAG_DAYS:
        return None
    last_date = pd.to_datetime(latest).strftime("%Y-%m-%d")
    reason = f"数据源最新一期为 {last_date}，距今 {lag} 天，不足以反映当期情况"
    if extra:
        reason = f"{reason}；{extra}"
    logger.warning("%s feed is stale (latest %s, %s days behind)", label, last_date, lag)
    return reason


def northbound_extra_note(latest) -> str:
    """北向资金停止披露是已知事实，值得在理由里说清，而不是只说"陈旧"。"""
    ts = pd.to_datetime(latest, errors="coerce")
    if pd.isna(ts) or ts.normalize() > NORTHBOUND_DISCLOSURE_END:
        return ""
    return NORTHBOUND_DISCONTINUED_NOTE
