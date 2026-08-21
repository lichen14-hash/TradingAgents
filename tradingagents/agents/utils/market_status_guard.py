"""Guards that prevent verified market status from being contradicted by LLM output.

Conflicts are classified by severity:
- ``block``: unambiguous claims that the instrument is *currently* ST/*ST.
  These invalidate the decision logic and block report generation.
- ``warn``: lower-precision findings (e.g. a price-limit figure that differs
  from the verified ratio). These are surfaced as report warnings instead of
  discarding an entire completed analysis, because pattern matching on natural
  language cannot reach blocking-grade precision for them.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from tradingagents.datacollector.schema import DataBundle, MarketStatus
from tradingagents.dataflows.market_utils import is_a_share, is_etf

_CURRENT_ST_CLAIM_PATTERNS = [
    (r"当前简称[:：]?\s*(?:\*?ST|ＳＴ)", "把当前简称写成ST"),
    (
        r"当前[^，,、。；\n]{0,20}(?:带有|属于|是|为)[^，,、。；\n]{0,10}(?:\*?ST|ＳＴ)",
        "把当前状态写成ST",
    ),
    (r"(?:当前|现为|属于|作为)[^，,、。；\n]{0,12}(?:\*?ST|ＳＴ)身份", "引用当前ST身份"),
    (r"(?:当前|现为|属于|作为)[^，,、。；\n]{0,12}(?:\*?ST|ＳＴ)股", "引用当前ST股属性"),
    (r"(?:当前|目前|现处于)[^，,、。；\n]{0,12}戴帽", "引用当前戴帽状态"),
    (r"(?:当前|目前)[^，,、。；\n]{0,12}摘帽(?:进程|预期|催化|尚未|等待)", "把摘帽当作当前待兑现事项"),
]
_PRICE_LIMIT_CLAIM = re.compile(
    r"(?P<ratio>\d+(?:\.\d+)?)\s*%\s*(?:的)?(?:涨跌幅|涨跌停|价格涨跌幅|限价)(?:限制|上限|规则|制度)?",
    flags=re.IGNORECASE,
)


_NEGATED_TARGET = re.compile(
    r"(?:"
    r"(?:非|不是|并非|而非|不同于|区别于|不属于|不具备|没有|并无|未(?:使用|采用|套用)?|不(?:使用|采用|套用|适用|按))"
    r"[^，,。；\n]{0,24}(?:\*?ST|ＳＴ|\d+(?:\.\d+)?\s*%\s*(?:涨跌幅|涨跌停|价格涨跌幅|限价)|戴帽|摘帽)"
    r"|无\s*(?:风险警示[/／]?)?(?:\*?ST|ＳＴ)"
    r")",
    flags=re.IGNORECASE,
)
_HISTORY_MARKERS = ("历史", "曾经", "此前", "过去", "原为", "曾为")


def find_market_status_conflicts(text: str, market_status: MarketStatus | Mapping | None) -> list[dict[str, str]]:
    """Return contradictions between generated text and verified status."""

    if not text or not market_status:
        return []
    status = _get(market_status, "risk_warning_status", "unknown")
    if status != "normal":
        return []

    conflicts: list[dict[str, str]] = []
    for pattern, reason in _CURRENT_ST_CLAIM_PATTERNS:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            if _is_negated_or_historical(text, match.start(), match.end()):
                continue
            conflicts.append({
                "reason": reason,
                "snippet": _snippet(text, match.start(), match.end()),
                "verified_status": status,
                "severity": "block",
            })

    expected_limit = _get_float(market_status, "price_limit_ratio")
    if expected_limit > 0:
        for match in _PRICE_LIMIT_CLAIM.finditer(text):
            claimed_limit = float(match.group("ratio"))
            if abs(claimed_limit - expected_limit) < 0.01:
                continue
            if _is_negated_or_historical(text, match.start(), match.end()):
                continue
            conflicts.append({
                "reason": f"涨跌幅限制与已验证值{expected_limit:g}%不一致",
                "snippet": _snippet(text, match.start(), match.end()),
                "verified_status": status,
                "severity": "warn",
            })
    return conflicts


def validate_final_state_market_status(final_state: Mapping, bundle: DataBundle) -> list[dict[str, str]]:
    """Validate all report-bound final state sections against MarketStatus."""

    status = bundle.metadata.market_status
    ticker = bundle.metadata.ticker
    if not is_a_share(ticker) or is_etf(ticker):
        return []
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


def classify_market_status_conflicts(
    final_state: Mapping, bundle: DataBundle,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Split conflicts into (blocking, warnings) by severity."""

    issues = validate_final_state_market_status(final_state, bundle)
    blocking = [i for i in issues if i.get("severity") == "block"]
    warnings = [i for i in issues if i.get("severity") != "block"]
    return blocking, warnings


def raise_if_market_status_conflicts(final_state: Mapping, bundle: DataBundle) -> list[dict[str, str]]:
    """Raise on blocking conflicts; return warning-level conflicts."""

    blocking, warnings = classify_market_status_conflicts(final_state, bundle)
    if blocking:
        raise MarketStatusConflictError(blocking)
    return warnings


def _get(obj: MarketStatus | Mapping, key: str, default: str = ""):
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _get_float(obj: MarketStatus | Mapping, key: str) -> float:
    try:
        return float(_get(obj, key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _is_negated_or_historical(text: str, start: int, end: int) -> bool:
    """Ignore explicit denials and clearly historical references around a match."""

    left = max(0, start - 30)
    context = text[left:end]
    sentence_start = max(
        context.rfind("。"),
        context.rfind("；"),
        context.rfind("\n"),
        context.rfind("!"),
        context.rfind("！"),
        context.rfind("?"),
        context.rfind("？"),
    )
    clause = context[sentence_start + 1:]
    if _NEGATED_TARGET.search(clause):
        return True
    return any(marker in clause for marker in _HISTORY_MARKERS) and "当前" not in clause


def _snippet(text: str, start: int, end: int) -> str:
    left = max(0, start - 35)
    right = min(len(text), end + 35)
    return text[left:right].replace("\n", " ")
