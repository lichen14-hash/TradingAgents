"""Data-completeness issues for the report banner.

Detection semantics live in one place —
:func:`tradingagents.datacollector.collector.validate_bundle_completeness` — and
this module only adds what the report needs on top: *empty* fields (the
collector's check looks for markers, prose gaps and staleness, not blanks) and
human-readable Chinese labels.

``run_batch_analysis._check_data_completeness`` used to reimplement the
unavailable-marker scan, and the two copies had already drifted: the report copy
never checked ``industry_data``, ``stock_moneyflow``, the per-indicator dict, or
the A-share ``effective_date``, so those gaps were invisible in reports even
though the collector knew about them.
"""

from __future__ import annotations

from tradingagents.datacollector import DataBundle
from tradingagents.datacollector.collector import (
    KIND_MISSING,
    KIND_PARTIAL,
    KIND_STALE,
    KIND_UNAVAILABLE,
    SEVERITY_BLOCK,
    SEVERITY_WARN,
    validate_bundle_completeness,
)

STATUS_MISSING = KIND_MISSING
STATUS_UNAVAILABLE = KIND_UNAVAILABLE
STATUS_PARTIAL = KIND_PARTIAL
STATUS_STALE = KIND_STALE

# Raw field name -> label shown in the report banner. Fields absent from this
# map fall back to their raw name, so a newly added collector field still shows
# up (unlabelled) rather than being silently dropped.
_FIELD_LABELS = {
    "risk_warning_status": "风险警示/ST状态",
    "source_conflicts": "数据源冲突",
    "effective_date": "状态生效日期",
    "stock_data": "股价/成交量",
    "verified_snapshot": "验证快照",
    "intraday_snapshot": "盘中快照",
    "ticker_news": "个股新闻",
    "stocktwits": "StockTwits/股吧",
    "reddit": "Reddit/新浪",
    "global_news": "全球/宏观新闻",
    "insider_transactions": "内部交易",
    "industry_data": "行业数据",
    "stock_moneyflow": "资金流",
    "overview": "概览",
    "balance_sheet_quarterly": "资产负债表(季度)",
    "balance_sheet_annual": "资产负债表(年度)",
    "cashflow_quarterly": "现金流(季度)",
    "cashflow_annual": "现金流(年度)",
    "income_quarterly": "利润表(季度)",
    "income_annual": "利润表(年度)",
    "competitive_intelligence": "竞争格局情报",
    "northbound_flow": "北向资金",
    "southbound_flow": "南向资金",
    "margin_trading": "融资融券",
    "top_institutional": "龙虎榜",
    "ah_premium": "AH溢价",
    "hk_connect_summary": "港股通汇总",
}

# Sections scanned for *empty* values, as (category, attribute, field labels).
_EMPTY_SCAN: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("行情数据", "market", ("stock_data", "verified_snapshot")),
    ("情绪数据", "sentiment", ("ticker_news", "stocktwits", "reddit")),
    ("新闻数据", "news", ("ticker_news", "global_news", "insider_transactions")),
    (
        "财务数据",
        "fundamentals",
        (
            "overview",
            "balance_sheet_quarterly",
            "balance_sheet_annual",
            "cashflow_quarterly",
            "cashflow_annual",
            "income_quarterly",
            "income_annual",
            "competitive_intelligence",
        ),
    ),
)


def _label(field: str) -> str:
    return _FIELD_LABELS.get(field, field)


def check_data_completeness(bundle: DataBundle) -> list[dict]:
    """Return report-facing issues as ``{category, field, status, ...}`` dicts.

    ``status`` is one of ``"missing"`` (field empty / section absent),
    ``"unavailable"`` (collector recorded an ``<unavailable: …>`` marker),
    ``"partial"`` (provider returned prose admitting a gap) or ``"stale"``
    (content present but older than its update cadence allows). ``severity``
    and ``reason`` are carried through so the banner can show *why* — a stale
    feed is only actionable if the reader sees how stale.
    """
    issues: list[dict] = []

    for issue in validate_bundle_completeness(bundle):
        issues.append({
            "category": issue["category"],
            "field": _label(issue["field"]),
            "status": issue.get("kind", STATUS_UNAVAILABLE),
            "severity": issue.get("severity", SEVERITY_BLOCK),
            "reason": issue.get("reason", ""),
        })

    selected = bundle.metadata.selected_analysts or []
    for category, attr, fields in _EMPTY_SCAN:
        section = getattr(bundle, attr, None)
        if section is None:
            # An unselected analyst has no section by design; only report the
            # gap when the analyst was supposed to run.
            if attr == "fundamentals" and "fundamentals" not in selected:
                continue
            issues.append({
                "category": category, "field": "全部", "status": STATUS_MISSING,
                "severity": SEVERITY_BLOCK, "reason": "整个数据段缺失",
            })
            continue
        for field in fields:
            value = getattr(section, field, "") or ""
            if not value.strip():
                issues.append({
                    "category": category,
                    "field": _label(field),
                    "status": STATUS_MISSING,
                    "severity": SEVERITY_WARN,
                    "reason": "字段为空",
                })

    if bundle.market is not None and not bundle.market.indicators:
        issues.append({
            "category": "行情数据", "field": "技术指标", "status": STATUS_MISSING,
            "severity": SEVERITY_BLOCK, "reason": "技术指标全部缺失",
        })

    return issues
