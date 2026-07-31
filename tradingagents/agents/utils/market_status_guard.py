"""Guards that prevent verified market status from being contradicted by LLM output."""

from __future__ import annotations

import re
from collections.abc import Mapping

from tradingagents.datacollector.schema import DataBundle, MarketStatus

_NORMAL_STATUS_FORBIDDEN_PATTERNS = [
    (r"当前简称[:：]?\s*(?:\*?ST|ＳＴ)", "把当前简称写成ST"),
    (r"当前[^。；\n]{0,20}(?:带有|属于|是|为)[^。；\n]{0,10}(?:\*?ST|ＳＴ)", "把当前状态写成ST"),
    (r"(?:ST|\*ST|ＳＴ)身份", "引用当前ST身份"),
    (r"(?:ST|\*ST|ＳＴ)股", "引用ST股交易属性"),
    (r"戴帽期间", "引用当前戴帽期间"),
    (r"5%\s*涨跌幅", "套用ST专属5%涨跌幅"),
    (r"一字跌停", "套用ST一字跌停风险"),
    (r"制度性折价", "套用ST制度性折价"),
    (r"摘帽(?:进程|预期|催化|不确定|尚未|等待)", "把摘帽当作当前待兑现事项"),
    (r"不确定的摘帽", "把摘帽当作当前待兑现事项"),
]


def find_market_status_conflicts(text: str, market_status: MarketStatus | Mapping | None) -> list[dict[str, str]]:
    """Return contradictions between generated text and verified status."""

    if not text or not market_status:
        return []
    status = _get(market_status, "risk_warning_status", "unknown")
    if status != "normal":
        return []

    conflicts: list[dict[str, str]] = []
    for pattern, reason in _NORMAL_STATUS_FORBIDDEN_PATTERNS:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            conflicts.append({
                "reason": reason,
                "snippet": _snippet(text, match.start(), match.end()),
                "verified_status": status,
            })
    return conflicts


def validate_final_state_market_status(final_state: Mapping, bundle: DataBundle) -> list[dict[str, str]]:
    """Validate all report-bound final state sections against MarketStatus."""

    status = bundle.metadata.market_status
    checks = {
        "final_trade_decision": final_state.get("final_trade_decision", ""),
        "market_report": final_state.get("market_report", ""),
        "sentiment_report": final_state.get("sentiment_report", ""),
        "news_report": final_state.get("news_report", ""),
        "fundamentals_report": final_state.get("fundamentals_report", ""),
        "investment_plan": final_state.get("investment_plan", ""),
        "trader_investment_plan": final_state.get("trader_investment_plan", ""),
    }
    risk_state = final_state.get("risk_debate_state", {})
    if isinstance(risk_state, Mapping):
        checks["risk_debate_state.history"] = risk_state.get("history", "")
    invest_state = final_state.get("investment_debate_state", {})
    if isinstance(invest_state, Mapping):
        checks["investment_debate_state.history"] = invest_state.get("history", "")

    issues: list[dict[str, str]] = []
    for section, text in checks.items():
        for issue in find_market_status_conflicts(str(text or ""), status):
            issues.append({"section": section, **issue})
    return issues


class MarketStatusConflictError(RuntimeError):
    """Raised when LLM output contradicts verified market status."""

    def __init__(self, issues: list[dict[str, str]]):
        self.issues = issues
        first = issues[0] if issues else {}
        super().__init__(
            "市场状态事实冲突："
            f"{first.get('section', 'unknown')} {first.get('reason', '')} "
            f"({first.get('snippet', '')})"
        )


def raise_if_market_status_conflicts(final_state: Mapping, bundle: DataBundle) -> None:
    issues = validate_final_state_market_status(final_state, bundle)
    if issues:
        raise MarketStatusConflictError(issues)


def _get(obj: MarketStatus | Mapping, key: str, default: str = ""):
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _snippet(text: str, start: int, end: int) -> str:
    left = max(0, start - 35)
    right = min(len(text), end + 35)
    return text[left:right].replace("\n", " ")
