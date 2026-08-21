"""Pydantic schemas used by agents that produce structured output.

The framework's primary artifact is still prose: each agent's natural-language
reasoning is what users read in the saved markdown reports and what the
downstream agents read as context.  Structured output is layered onto the
three decision-making agents (Research Manager, Trader, Portfolio Manager)
so that:

- Their outputs follow consistent section headers across runs and providers
- Each provider's native structured-output mode is used (json_schema for
  OpenAI/xAI, response_schema for Gemini, tool-use for Anthropic)
- Schema field descriptions become the model's output instructions, freeing
  the prompt body to focus on context and the rating-scale guidance
- A render helper turns the parsed Pydantic instance back into the same
  markdown shape the rest of the system already consumes, so display,
  memory log, and saved reports keep working unchanged
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from tradingagents.agents.utils.position_sizing import describe_ceiling

# ---------------------------------------------------------------------------
# Shared rating types
# ---------------------------------------------------------------------------


class PortfolioRating(str, Enum):
    """5-tier rating used by the Research Manager and Portfolio Manager."""

    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"


class TraderAction(str, Enum):
    """3-tier transaction direction used by the Trader.

    The Trader's job is to translate the Research Manager's investment plan
    into a concrete transaction proposal: should the desk execute a Buy, a
    Sell, or sit on Hold this round.  The nuanced Overweight / Underweight
    call happens later at the Portfolio Manager, and the *size* is not any
    agent's to choose — it is computed from volatility by
    :mod:`tradingagents.agents.utils.position_sizing`.
    """

    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"


# Shared by ResearchPlan and PortfolioDecision. Kept as one string so the two
# decision stages grade data quality on the same scale — a plan rated `high`
# feeding a decision rated `low` should mean something, not just reflect two
# differently-worded prompts.
#
# The 2026-08-19 batch is why this field exists: the intraday snapshot failed on
# 8/8 tickers, northbound flow was two years stale, and 5/5 A-share competitive
# intelligence sections were empty, yet every decision read as if the evidence
# base were complete. The analysts now receive an explicit gap list (see
# :func:`tradingagents.agents.utils.integrity.data_gap_notice`); this is where
# that reaches the decision layer and the database.
DATA_CONFIDENCE_DESCRIPTION = (
    "How complete the evidence behind this conclusion actually is — a "
    "statement about the DATA, not about how strongly you hold the view. "
    "Use 'low' when a data gap listed in the prompt, or an "
    "'<unavailable: …>' placeholder in the reports you were given, blocked a "
    "judgement you would otherwise have made; 'medium' when the inputs are "
    "present but thin, partly stale, or one analyst flagged low confidence; "
    "'high' only when every input this conclusion rests on was present and "
    "current. Do not raise this because the case feels compelling: a "
    "confident view built on incomplete data is exactly the combination this "
    "field is meant to expose."
)


# ---------------------------------------------------------------------------
# Research Manager
# ---------------------------------------------------------------------------


class ResearchPlan(BaseModel):
    """Structured investment plan produced by the Research Manager.

    Hand-off to the Trader: the recommendation pins the directional view,
    the rationale captures which side of the bull/bear debate carried the
    argument, and the strategic actions translate that into concrete
    conditional triggers the trader can execute against.

    The recommendation is deliberately a **view** rating, not a position
    instruction. The Research Manager never receives the user's holdings, so
    any position language here would be a guess about an unseen portfolio —
    and, worse, would look like a contradiction whenever the Portfolio Manager
    (which does see the holdings, and whose identical labels mean position
    *changes*) sized the position differently. The two scales share the label
    vocabulary and nothing else; see
    :func:`tradingagents.agents.utils.integrity.check_cross_stage_divergence`
    for the mapping that reconciles them.
    """

    recommendation: PortfolioRating = Field(
        description=(
            "How the evidence leans after the debate — a VIEW rating, not a "
            "position instruction. Exactly one of Buy (decisively bullish) / "
            "Overweight (bullish on balance) / Hold (genuinely balanced, or "
            "both cases weak) / Underweight (bearish on balance) / Sell "
            "(decisively bearish). Reserve Hold for situations where the "
            "evidence on both sides is genuinely balanced; otherwise commit to "
            "the side with the stronger arguments."
        ),
    )
    rationale: str = Field(
        description=(
            "Conversational summary of the key points from both sides of the "
            "debate, ending with which arguments led to the recommendation. "
            "Speak naturally, as if to a teammate. If either side failed to "
            "produce an argument, state that plainly and note that it lowers "
            "confidence in this recommendation."
        ),
    )
    strategic_actions: str = Field(
        description=(
            "Concrete conditional triggers for the trader: the price, volume, "
            "or fundamental conditions that would confirm or invalidate this "
            "view (e.g. 'thesis confirmed on a close above X with rising "
            "volume; invalidated below Y'). Do NOT specify a target position "
            "size or percentage — you cannot see the user's holdings, so "
            "sizing belongs to the Portfolio Manager."
        ),
    )
    data_confidence: Literal["low", "medium", "high"] = Field(
        default="medium",
        description=DATA_CONFIDENCE_DESCRIPTION,
    )


def render_research_plan(plan: ResearchPlan) -> str:
    """Render a ResearchPlan to markdown for storage and the trader's prompt context.

    The ``(view rating)`` qualifier is part of the output on purpose: the
    Trader and the Portfolio Manager read this markdown as prompt context, and
    both need to know that this label describes the evidence, not a position
    change. The ``**Recommendation**:`` prefix is preserved so existing
    parsers keep matching.
    """
    return "\n".join([
        f"**Recommendation**: {plan.recommendation.value}  (view rating — how the "
        f"evidence leans; not a position instruction)",
        "",
        f"**Data Confidence**: {plan.data_confidence.capitalize()}",
        "",
        f"**Rationale**: {plan.rationale}",
        "",
        f"**Strategic Actions**: {plan.strategic_actions}",
    ])


# ---------------------------------------------------------------------------
# Trader
# ---------------------------------------------------------------------------


class TraderProposal(BaseModel):
    """Structured transaction proposal produced by the Trader.

    The trader reads the Research Manager's investment plan and the analyst
    reports, then turns them into a concrete transaction: what action to
    take, the reasoning that justifies it, and the practical levels for
    entry and stop-loss.

    Deliberately **no sizing field**: the weight is computed from volatility
    (see :mod:`tradingagents.agents.utils.position_sizing`). What the trader
    contributes instead is *staging* — how to work into or out of the
    computed target.
    """

    action: TraderAction = Field(
        description="The transaction direction. Exactly one of Buy / Hold / Sell.",
    )
    reasoning: str = Field(
        description=(
            "The case for this action, anchored in the analysts' reports and "
            "the research plan. Two to four sentences."
        ),
    )
    entry_price: float | None = Field(
        default=None,
        description="Optional entry price target in the instrument's quote currency.",
    )
    stop_loss: float | None = Field(
        default=None,
        description="Optional stop-loss price in the instrument's quote currency.",
    )
    execution_plan: str | None = Field(
        default=None,
        description=(
            "Optional execution staging: how to work into or out of the "
            "position — how many tranches, at which levels, and what "
            "invalidates the remaining tranches (e.g. 'two tranches, first at "
            "market, second only on a close above X; abandon the rest below "
            "Y'). Do NOT state a position size, percentage of portfolio, or "
            "lot count: the size is computed from the stock's volatility "
            "against a fixed risk budget, so a number here would either "
            "duplicate it or contradict it."
        ),
    )


def render_trader_proposal(proposal: TraderProposal) -> str:
    """Render a TraderProposal to markdown.

    The trailing ``FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**`` line is
    preserved for backward compatibility with the analyst stop-signal text
    and any external code that greps for it.
    """
    parts = [
        f"**Action**: {proposal.action.value}",
        "",
        f"**Reasoning**: {proposal.reasoning}",
    ]
    if proposal.entry_price is not None:
        parts.extend(["", f"**Entry Price**: {proposal.entry_price}"])
    if proposal.stop_loss is not None:
        parts.extend(["", f"**Stop Loss**: {proposal.stop_loss}"])
    if proposal.execution_plan:
        parts.extend(["", f"**Execution Plan**: {proposal.execution_plan}"])
    parts.extend([
        "",
        f"FINAL TRANSACTION PROPOSAL: **{proposal.action.value.upper()}**",
    ])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Portfolio Manager
# ---------------------------------------------------------------------------


class PortfolioDecision(BaseModel):
    """Structured output produced by the Portfolio Manager.

    The model fills every field as part of its primary LLM call; no separate
    extraction pass is required. Field descriptions double as the model's
    output instructions, so the prompt body only needs to convey context and
    the rating-scale guidance.

    **This schema no longer carries a position size.** It used to have
    ``current_position_pct`` / ``target_position_pct``, filled freely by the
    model, and the audit over 327 predictions showed what the model was
    actually anchoring them on: the user's unrealised loss (deeply-underwater
    positions drew 0% add / 71% trim advice, profitable ones 23.8% / 19.0%,
    p=0.00012). The size is now computed from volatility against a fixed risk
    budget in :mod:`tradingagents.agents.utils.position_sizing`, and the fields
    are gone so the model cannot anchor on — or contradict — it.

    What is left here is the part a model is actually good at: the direction of
    the evidence (``portfolio_view``), the thesis, and the risk judgement. The
    position-change label that downstream consumers read as ``**Rating**`` is
    derived by the sizer, not chosen here — see :func:`render_pm_decision`.

    ``divergence_from_research`` exists because of the 2026-08-11 300760.SZ
    incident: the Portfolio Manager rated Underweight while the Research
    Manager and the Trader both said Hold, and nothing in the run recorded
    that the override had happened or why. Now that all three stages express a
    *view*, the comparison is finally like-for-like.
    """

    portfolio_view: PortfolioRating = Field(
        description=(
            "How the evidence leans once the risk debate is taken into account "
            "— a VIEW rating, on the same scale the Research Manager uses. "
            "Exactly one of Buy (decisively bullish) / Overweight (bullish on "
            "balance) / Hold (genuinely balanced) / Underweight (bearish on "
            "balance) / Sell (decisively bearish). This is NOT a position "
            "instruction and NOT a position change: the size is computed "
            "separately from the stock's volatility. Judge the evidence, not "
            "the user's portfolio."
        ),
    )
    executive_summary: str = Field(
        description=(
            "A concise action plan covering entry/exit strategy, key risk "
            "levels, and time horizon. Two to four sentences. Do NOT state a "
            "position size, a percentage of portfolio, or a lot count — those "
            "are computed from volatility and appended to your decision "
            "automatically; a number here would contradict them."
        ),
    )
    investment_thesis: str = Field(
        description=(
            "Detailed reasoning anchored in specific evidence from the analysts' "
            "debate. If prior lessons are referenced in the prompt context, "
            "incorporate them; otherwise rely solely on the current analysis."
        ),
    )
    divergence_from_research: str = Field(
        default="None",
        description=(
            "Audit trail for overriding the earlier stages. If your view "
            "differs in direction from the Research Manager's view rating or "
            "the Trader's proposed action — i.e. you lean bullish where they "
            "leaned bearish, or the reverse — you MUST name here (a) whose "
            "conclusion you are overriding, and (b) the specific piece of "
            "evidence from the risk debate or the data that overturned it. "
            "Write exactly 'None' when your direction is consistent with "
            "theirs. Note that a *position* reduction driven by the risk "
            "budget is not a divergence and does not belong here — that is "
            "sizing, and it is recorded separately."
        ),
    )
    data_confidence: Literal["low", "medium", "high"] = Field(
        default="medium",
        description=DATA_CONFIDENCE_DESCRIPTION,
    )
    price_target: float | None = Field(
        default=None,
        description="Optional target price in the instrument's quote currency.",
    )
    time_horizon: str | None = Field(
        default=None,
        description="Optional recommended holding period, e.g. '3-6 months'.",
    )


def render_pm_decision(
    decision: PortfolioDecision,
    ceiling=None,
    current_pct: float | None = None,
    config=None,
) -> str:
    """Render a PortfolioDecision back to the markdown shape the rest of the system expects.

    Memory log, CLI display, and saved report files all read this markdown,
    so the rendered output preserves the exact section headers (``**Rating**``,
    ``**Current Position**``, ``**Executive Summary**``, ``**Investment
    Thesis**``) that downstream parsers and the report writers already handle —
    only their *source* changed.

    ``**Rating**`` is the **view**, always. It briefly was the sizer's
    position-change label, which collided with the 327 historical rows where it
    means the view (and with ``analytics.accuracy_by_rating()``, which groups by
    it): a neutral view on an oversized position legitimately produces a bearish
    *label*, and the scorecard was reading that as a bearish *opinion*.

    ``ceiling`` is the :class:`~tradingagents.agents.utils.position_sizing.Ceiling`
    for this name. It renders as ``**Risk Ceiling**`` — "at most this much of the
    portfolio, given this stock's volatility" — plus a ``**Sizing**`` line showing
    the arithmetic. There is deliberately **no target weight here**: a target
    needs to know what the other N-1 holdings are doing, so it is produced after
    the batch by :func:`tradingagents.portfolio.allocator.allocate` and rendered
    by the report layer. That way exactly one position number exists per name.

    ``**Data Confidence**`` sits directly under the rating rather than at the
    end: it qualifies the rating, and a reader who stops after the first two
    lines is exactly the reader who needs to know the evidence was thin.
    """
    parts = [
        f"**Rating**: {decision.portfolio_view.value}",
        "",
        f"**View**: {decision.portfolio_view.value}  (how the evidence leans; "
        f"the position change is decided at the portfolio level)",
        "",
        f"**Data Confidence**: {decision.data_confidence.capitalize()}",
    ]
    if current_pct is not None:
        parts.extend(["", f"**Current Position**: {current_pct}%"])
    if ceiling is not None:
        parts.extend(["", f"**Risk Ceiling**: {ceiling.allowed_pct}%"])
        parts.extend(["", f"**Sizing**: {describe_ceiling(ceiling, config)}"])
    divergence = (decision.divergence_from_research or "").strip()
    if divergence and divergence.lower() not in ("none", "n/a", "无"):
        parts.extend(["", f"**Divergence**: {divergence}"])
    parts.extend([
        "",
        f"**Executive Summary**: {decision.executive_summary}",
        "",
        f"**Investment Thesis**: {decision.investment_thesis}",
    ])
    if decision.price_target is not None:
        parts.extend(["", f"**Price Target**: {decision.price_target}"])
    if decision.time_horizon:
        parts.extend(["", f"**Time Horizon**: {decision.time_horizon}"])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Sentiment Analyst
# ---------------------------------------------------------------------------


class SentimentBand(str, Enum):
    """Discrete sentiment direction produced by the Sentiment Analyst.

    Six tiers keep the signal granular enough to be actionable while remaining
    small enough for every provider to map reliably from its JSON output.
    """

    BULLISH = "Bullish"
    MILDLY_BULLISH = "Mildly Bullish"
    NEUTRAL = "Neutral"
    MIXED = "Mixed"
    MILDLY_BEARISH = "Mildly Bearish"
    BEARISH = "Bearish"


class SentimentReport(BaseModel):
    """Structured sentiment report produced by the Sentiment Analyst.

    Replaces the previous free-form prose output so downstream consumers
    (dashboards, audit logs, PDF renderers, other agents) can read
    ``overall_band`` and ``overall_score`` without maintaining fragile regex
    fallbacks that drift with every model release. ``narrative`` preserves the
    rich source-by-source analysis; ``render_sentiment_report`` prepends a
    deterministic header so the saved report stays human-readable.
    """

    overall_band: SentimentBand = Field(
        description=(
            "Overall sentiment direction. Exactly one of: "
            "Bullish / Mildly Bullish / Neutral / Mixed / Mildly Bearish / Bearish. "
            "Use Mixed when sources point in clearly different directions. "
            "Use Neutral only when all sources are genuinely silent or non-committal."
        ),
    )
    overall_score: float = Field(
        ge=0.0,
        le=10.0,
        description=(
            "Numeric sentiment intensity on a 0–10 scale. "
            "0 = maximally bearish, 5 = neutral, 10 = maximally bullish. "
            "Guideline for consistency with overall_band: "
            "Bullish ~6.5–10, Mildly Bullish ~5.5–6.4, Neutral/Mixed ~4.5–5.5, "
            "Mildly Bearish ~3.5–4.4, Bearish ~0–3.4. "
            "Only the 0–10 bounds are enforced."
        ),
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description=(
            "Confidence in the assessment based on data quality and sample size. "
            "Use 'low' when one or more sources returned a placeholder or fewer "
            "than 5 data points; 'medium' when data is present but sparse; "
            "'high' when all three sources returned substantive data."
        ),
    )
    narrative: str = Field(
        description=(
            "Full sentiment report covering, in order: "
            "(1) source-by-source breakdown with specific evidence (cite message "
            "counts, ratios, notable posts); "
            "(2) cross-source divergences and alignments; "
            "(3) dominant narrative themes; "
            "(4) catalysts and risks surfaced by the data; "
            "(5) a markdown table summarising key sentiment signals, their "
            "direction, source, and supporting evidence. "
            "Keep it informative and substantive: develop each section thoroughly "
            "with concrete evidence so every point adds new signal for the trader."
        ),
    )


def render_sentiment_report(report: SentimentReport) -> str:
    """Render a SentimentReport to the markdown shape the rest of the system expects.

    The structured header (band + score + confidence) is prepended to the
    narrative so the saved report is both human-readable and machine-parseable
    without regex.
    """
    return "\n".join([
        f"**Overall Sentiment:** **{report.overall_band.value}** "
        f"(Score: {report.overall_score:.1f}/10)",
        f"**Confidence:** {report.confidence.capitalize()}",
        "",
        report.narrative,
    ])
