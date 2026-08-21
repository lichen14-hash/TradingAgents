"""Run-integrity guards: make missing output, degraded capability, and
cross-stage disagreement impossible to lose silently.

Incident that motivated this module (300760.SZ, 2026-08-11): two runs on the
same trading day produced opposite signals (Hold vs Underweight). The audit
found three silent failures, none of which appeared anywhere in the report:

1. The Bear Researcher returned empty content in all three debate rounds --
   ``bear_history`` was literally ``"Bear Analyst:\\nBear Analyst:\\nBear
   Analyst:"``. The debate ``count`` still advanced, so the pipeline treated a
   one-sided debate as a complete one.
2. The Research Manager's structured output silently degraded to free text, so
   the run had no machine-readable research rating at all.
3. The Portfolio Manager rated Underweight while the Research Manager and the
   Trader both said Hold, with nothing recording that the override happened.

Design constraints:

- Findings share the exact shape used by :mod:`market_status_guard`
  (``{section, reason, snippet, severity}``) so one report banner renderer and
  one classify signature serve both.
- **Integrity findings never abort a run.** Unlike market-status conflicts,
  ``severity="block"`` here means "render prominently", not "raise". The
  pipeline must still finish and still write a decision, because the point is
  to record what happened -- not to replace the model's judgement. This mirrors
  the contract already documented on
  :func:`tradingagents.agents.utils.rating.check_rating_action_consistency`.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from tradingagents.agents.utils.rating import RATINGS_5_TIER

logger = logging.getLogger(__name__)

SEVERITY_BLOCK = "block"
SEVERITY_WARN = "warn"


def make_finding(
    section: str,
    reason: str,
    snippet: str = "",
    severity: str = SEVERITY_WARN,
) -> dict[str, str]:
    """Build a finding in the shared market-status/rating finding shape."""

    return {
        "section": section,
        "reason": reason,
        "snippet": snippet,
        "severity": severity,
    }


def classify_integrity_findings(
    findings: list[dict[str, str]] | None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Split findings into (high-severity, warnings).

    Signature mirrors :func:`market_status_guard.classify_market_status_conflicts`,
    but neither list is ever raised -- see the module docstring.
    """
    items = list(findings or [])
    high = [f for f in items if f.get("severity") == SEVERITY_BLOCK]
    warns = [f for f in items if f.get("severity") != SEVERITY_BLOCK]
    return high, warns


# ---------------------------------------------------------------------------
# Empty-output guard
# ---------------------------------------------------------------------------


def response_text(response: Any) -> str:
    """Extract plain text from an LLM response, tolerating block-list content.

    The provider clients already run
    :func:`tradingagents.llm_clients.base_client.normalize_content`, but this
    stays independent of them so the guard also works with raw LangChain
    messages and with the MagicMock responses used in tests.
    """
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(p for p in parts if p)
    return "" if content is None else str(content)


def is_blank_output(text: str) -> bool:
    """True when the text carries no substantive content.

    Whitespace and bare markdown punctuation both count as blank: a response
    of ``"**"`` or ``"---"`` is as useless to the next agent as ``""``.
    """
    return not text.strip(" \t\r\n*-_#`>|.")


def invoke_text_guarded(
    llm: Any,
    prompt: Any,
    *,
    section: str,
    role: str,
    retries: int = 1,
) -> tuple[str, list[dict[str, str]]]:
    """Invoke ``llm`` for free-text output, retrying and recording blank results.

    Returns ``(text, findings)``. On a blank response the call is retried
    ``retries`` times; if every attempt comes back blank, the returned text is
    an explicit placeholder (so downstream agents and the report show the gap
    instead of an invisible empty string) plus one high-severity finding.

    The caller must still advance any debate counter as usual -- the guard
    deliberately does not interfere with loop termination
    (see :mod:`tradingagents.graph.conditional_logic`).
    """
    attempts = max(1, retries + 1)
    for attempt in range(attempts):
        text = response_text(llm.invoke(prompt))
        if not is_blank_output(text):
            if attempt > 0:
                return text, [
                    make_finding(
                        section,
                        f"{role} 首次调用返回空正文，重试第 {attempt} 次后才产出内容",
                        snippet=f"role={role} blank_attempts={attempt}",
                        severity=SEVERITY_WARN,
                    )
                ]
            return text, []
        logger.warning(
            "%s produced empty content (attempt %d/%d) for section %s",
            role, attempt + 1, attempts, section,
        )

    placeholder = (
        f"[{role}: 本轮未产生有效输出——LLM 连续 {attempts} 次返回空正文，"
        "该轮观点缺席，请勿将其视为“无异议”。]"
    )
    return placeholder, [
        make_finding(
            section,
            f"{role} 连续 {attempts} 次返回空正文，本轮观点缺席（常见原因：思考预算耗尽或输出被截断）",
            snippet=f"role={role} blank_attempts={attempts}",
            severity=SEVERITY_BLOCK,
        )
    ]


def structured_degradation_finding(
    section: str, agent_name: str, exc: Exception | str,
) -> dict[str, str]:
    """Finding for a structured-output call that fell back to free text.

    Matters because the fallback loses the typed rating: without it there is no
    machine-readable stage rating to reconcile against, which is exactly why
    the 2026-08-11 override went unnoticed.
    """
    detail = f"{type(exc).__name__}: {exc}" if isinstance(exc, Exception) else str(exc)
    return make_finding(
        section,
        f"{agent_name} 结构化输出降级为自由文本，本阶段评级不可机器读取（{detail}）",
        snippet=f"agent={agent_name} error={detail}"[:300],
        severity=SEVERITY_WARN,
    )


# ---------------------------------------------------------------------------
# Cross-stage divergence
# ---------------------------------------------------------------------------

# All three stages now express a *view* (how bullish is the evidence), so the
# comparison here is finally like-for-like — callers must pass the Portfolio
# Manager's ``portfolio_view``, not its ``portfolio_rating``. The latter is a
# position-delta label derived from the risk budget by
# :mod:`tradingagents.agents.utils.position_sizing`, and comparing it against a
# view manufactures false conflicts: a neutral view on an oversized position
# legitimately produces a bearish label. Reduce to a direction either way, since
# a one-tier gap is not a disagreement.
_DIRECTION_BY_RATING: dict[str, int] = {
    "Buy": 1,
    "Overweight": 1,
    "Hold": 0,
    "Underweight": -1,
    "Sell": -1,
}

_TIER_INDEX = {name: i for i, name in enumerate(RATINGS_5_TIER)}

_DIRECTION_LABEL = {1: "看多/加仓", 0: "中性/维持", -1: "看空/减仓"}


def rating_direction(rating: str | None) -> int | None:
    """Reduce a 5-tier or 3-tier label to -1 / 0 / +1, or None if unparseable."""

    if not rating:
        return None
    return _DIRECTION_BY_RATING.get(str(rating).strip().capitalize())


def _tier_distance(a: str, b: str) -> int:
    ia, ib = _TIER_INDEX.get(a.capitalize()), _TIER_INDEX.get(b.capitalize())
    if ia is None or ib is None:
        return 0
    return abs(ia - ib)


def _divergence_finding(
    left_name: str, left: str, right_name: str, right: str,
) -> dict[str, str] | None:
    ld, rd = rating_direction(left), rating_direction(right)
    if ld is None or rd is None:
        return None
    if ld * rd >= 0:
        # Same direction, or one side is neutral. A neutral view paired with a
        # position change is legitimate (position sizing, not disagreement).
        return None
    distance = _tier_distance(left, right)
    return make_finding(
        "跨阶段一致性",
        f"{left_name}（{left}，{_DIRECTION_LABEL[ld]}）与 {right_name}"
        f"（{right}，{_DIRECTION_LABEL[rd]}）方向相反",
        snippet=f"{left_name}={left} {right_name}={right} tier_distance={distance}",
        severity=SEVERITY_BLOCK if distance >= 3 else SEVERITY_WARN,
    )


def check_cross_stage_divergence(
    research_rating: str | None,
    trader_direction: str | None,
    pm_rating: str | None,
) -> list[dict[str, str]]:
    """Flag direction conflicts between the three decision stages.

    Missing ratings are themselves findings: if a stage produced no readable
    rating, the override it may have performed cannot be checked at all.
    """
    findings: list[dict[str, str]] = []

    if not research_rating or rating_direction(research_rating) is None:
        findings.append(make_finding(
            "跨阶段一致性",
            "研究经理评级缺失或不可解析，无法核对组合经理是否越过研究结论",
            snippet=f"research_rating={research_rating!r}",
            severity=SEVERITY_WARN,
        ))
    if not pm_rating or rating_direction(pm_rating) is None:
        findings.append(make_finding(
            "跨阶段一致性",
            "组合经理评级缺失或不可解析",
            snippet=f"pm_rating={pm_rating!r}",
            severity=SEVERITY_WARN,
        ))

    for left_name, left, right_name, right in (
        ("研究经理", research_rating, "交易员", trader_direction),
        ("交易员", trader_direction, "组合经理", pm_rating),
        ("研究经理", research_rating, "组合经理", pm_rating),
    ):
        if not left or not right:
            continue
        finding = _divergence_finding(left_name, left, right_name, right)
        if finding:
            findings.append(finding)
    return findings


# ---------------------------------------------------------------------------
# Data-quality vs conviction
# ---------------------------------------------------------------------------

# Set by collector._force_last_complete_daily_bar when the requested date's
# daily bar is still forming, so the analysis runs on the previous complete bar.
INCOMPLETE_BAR_REASON = "intraday_daily_bar_incomplete"

# ``data_not_ready`` means the vendor had not published the requested date's bar
# yet, so the run reads the previous session. For the decision layer that is the
# same situation as INCOMPLETE_BAR_REASON — the latest session is missing from
# the evidence — so both trigger the caveat and the conviction check. Matching
# only the intraday label is how the 2026-08-19 batch slipped through: the
# collector emitted ``data_not_ready`` for all 8 bundles (fixed in
# ``_force_last_complete_daily_bar``, but non-intraday mid-session runs can
# still legitimately produce it) and neither guard fired.
_STALE_VIEW_REASONS = (INCOMPLETE_BAR_REASON, "data_not_ready")

# The two decisive ends of the view scale. Checked against the *view*, not the
# position-delta label: since position_sizing took over the numbers, a Sell
# label can be produced by the risk budget alone on a neutral view, and flagging
# that as over-confidence would be wrong.
_HIGH_CONVICTION_RATINGS = ("Buy", "Sell")


def date_correction_reason(state: Mapping) -> str:
    """Read ``data_bundle.metadata.date_correction_reason`` out of graph state."""

    bundle = state.get("data_bundle")
    if not isinstance(bundle, Mapping):
        return ""
    metadata = bundle.get("metadata", {})
    if not isinstance(metadata, Mapping):
        return ""
    return str(metadata.get("date_correction_reason") or "")


def data_quality_caveat(state: Mapping) -> str:
    """Prompt fragment cautioning against high conviction on incomplete data.

    Returns an empty string when the run's data is clean, so the prompt is
    unchanged in the normal case.
    """
    if date_correction_reason(state) not in _STALE_VIEW_REASONS:
        return ""
    return (
        "\n**Data-quality caveat:** the requested date's daily bar was still "
        "forming, so this analysis runs on the last *complete* daily bar. "
        "Today's intraday move is therefore not in the data you are reading. "
        "Do not justify a decisive view (a Buy or Sell label) on single-session "
        "price action under these conditions — prefer the graduated Overweight "
        "/ Underweight labels and state the data limitation in your thesis.\n"
    )


def check_conviction_vs_data_quality(
    pm_rating: str | None, correction_reason: str,
) -> list[dict[str, str]]:
    """Flag a high-conviction rating taken on a knowingly incomplete bar."""

    if correction_reason not in _STALE_VIEW_REASONS or not pm_rating:
        return []
    if str(pm_rating).strip().capitalize() not in _HIGH_CONVICTION_RATINGS:
        return []
    return [make_finding(
        "数据质量与置信度",
        f"当日日线未收盘（{correction_reason}），却给出 {pm_rating} 级别（>60% 仓位变动）的高置信裁定",
        snippet=f"pm_rating={pm_rating} date_correction_reason={correction_reason}",
        severity=SEVERITY_WARN,
    )]


# ---------------------------------------------------------------------------
# Data completeness
# ---------------------------------------------------------------------------

# Only the collector knows *what* was requested and *what* came back, so the
# detection lives there. This side turns its issues into findings so that a data
# gap travels the same channel as an empty debate round: one banner renderer,
# one DB column, one place to look. Until now completeness was computed only in
# ``run_batch_analysis.generate_html_report``, which meant the CLI and the web
# server never saw it and nothing was ever persisted -- the 2026-08-19 batch
# recorded 7 signals with no trace that the intraday snapshot had failed on 8/8
# tickers and northbound flow was 732 days stale.
_KIND_LABELS = {
    "unavailable": "不可用",
    "missing": "缺失",
    "partial": "部分缺失",
    "stale": "陈旧",
}

# Findings from this producer are tagged so a renderer that already has a
# dedicated data-completeness banner can avoid printing them twice.
DATA_COMPLETENESS_SECTION_PREFIX = "数据完整性/"


def _bundle_issues(bundle: Any) -> list[dict]:
    """Completeness issues for *bundle*, tolerating every shape it arrives in.

    Accepts either a :class:`~tradingagents.datacollector.schema.DataBundle` or
    the ``model_dump()`` dict carried on graph state, because the graph holds
    the serialised form while the collector node holds the object.
    """
    # Imported lazily: this module is pulled in by every agent, and the
    # collector drags in pandas and the whole dataflows package.
    from tradingagents.datacollector.schema import DataBundle
    from tradingagents.datacollector.collector import validate_bundle_completeness

    if bundle is None:
        return []
    if isinstance(bundle, Mapping):
        if not bundle:
            return []
        try:
            bundle = DataBundle.model_validate(bundle)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Cannot validate data bundle for completeness: %s", exc)
            return []

    try:
        return validate_bundle_completeness(bundle)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Data completeness check failed: %s", exc)
        return []


# Which completeness categories each analyst is responsible for reading. An
# analyst told about every gap in the bundle would learn to skim the block; it
# is told about the gaps in *its own* inputs.
#
# The category strings must match the ones ``validate_bundle_completeness``
# emits; a typo here silently drops a whole category from every prompt, which is
# indistinguishable from a clean run. ``市场状态`` is deliberately absent: it has
# its own dedicated channel (:mod:`market_status_guard`, which can also abort the
# run) and is already rendered into every analyst's instrument context.
ANALYST_GAP_CATEGORIES = {
    "market": ("行情数据", "行情数据/技术指标"),
    "social": ("情绪数据",),
    "news": ("新闻数据", "宏观指标", "市场信号"),
    "fundamentals": ("财务数据",),
}

_GAP_REASON_CHARS = 140


def data_gap_notice(bundle: Any, analyst: str = "") -> str:
    """Prompt fragment naming the gaps in this analyst's own inputs.

    Returns ``""`` when there is nothing to report, so a clean run's prompt is
    byte-identical to before.

    The gaps were detectable all along but were only ever rendered *after* the
    fact, in the report. The analysts — the only stage that can actually
    compensate, by qualifying a conclusion or declining to draw one — were told
    nothing, and the substitute value ``<unavailable: ConnectionError: …>`` sat
    inline in the data section where a model routinely reads past it. In the
    2026-08-19 batch the intraday snapshot failed on 8/8 tickers and not one
    market report mentioned it.
    """
    categories = ANALYST_GAP_CATEGORIES.get(analyst)
    issues = [
        issue for issue in _bundle_issues(bundle)
        if categories is None or issue.get("category") in categories
    ]
    if not issues:
        return ""

    lines = [
        "",
        "**Data gaps in your own inputs** — the collector could not retrieve "
        "the following, so the corresponding sections are absent, truncated, or "
        "carry an `<unavailable: …>` placeholder instead of data:",
    ]
    for issue in issues:
        kind = _KIND_LABELS.get(issue.get("kind", ""), issue.get("kind", ""))
        reason = str(issue.get("reason", "")).replace("\n", " ")[:_GAP_REASON_CHARS]
        lines.append(
            f"- {issue.get('category', '')}/{issue.get('field', '')}"
            f"（{kind}）: {reason}"
        )
    lines.extend([
        "",
        "Work with what you do have. Do not infer, estimate, or fill in a value "
        "for anything listed above, and do not describe a gap as a neutral or "
        "unremarkable reading. State in your report which specific judgement you "
        "could not make because of these gaps, and lower your confidence "
        "accordingly.",
        "",
    ])
    return "\n".join(lines)


def data_completeness_findings(bundle: Any) -> list[dict[str, str]]:
    """Convert bundle-completeness issues into integrity findings."""
    findings = []
    issues = _bundle_issues(bundle)
    for issue in issues:
        kind = issue.get("kind", "unavailable")
        label = _KIND_LABELS.get(kind, kind)
        findings.append(make_finding(
            f"{DATA_COMPLETENESS_SECTION_PREFIX}{issue['category']}",
            f"{issue['field']} {label}：{issue.get('reason', '')}".strip(),
            snippet=f"category={issue['category']} field={issue['field']} kind={kind}",
            severity=issue.get("severity", SEVERITY_BLOCK),
        ))
    return findings


def stage_ratings(final_state: Mapping) -> dict[str, str]:
    """Collect the per-stage ratings for report annotation and persistence.

    ``portfolio`` is the position-change label (what the DB stores and what the
    signal consumers read); ``portfolio_view`` is the view that is actually
    comparable with the two stages before it.
    """

    return {
        "research": str(final_state.get("research_recommendation") or ""),
        "trader": str(final_state.get("trader_direction") or ""),
        "portfolio": str(final_state.get("portfolio_rating") or ""),
        "portfolio_view": str(final_state.get("portfolio_view") or ""),
    }
