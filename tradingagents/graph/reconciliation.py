"""Decision Reconciliation: the deterministic audit step after the Portfolio Manager.

Before this node existed the graph went straight from ``Portfolio Manager`` to
``END``, so nothing ever compared the three decision stages against each other.
That is how the 2026-08-11 300760.SZ run could end with the Research Manager at
Hold, the Trader at Hold, and the Portfolio Manager at Underweight, with no
record anywhere that an override had happened.

The node makes no LLM call and never rewrites a decision. It only reads what
the pipeline already produced and appends findings to ``integrity_findings``,
which the report banner and the backtest store then persist. Keeping it as a
graph node rather than a report-layer check matters because the CLI, the web
server, and the batch runner all consume the graph's final state but render it
through different code paths — a check that lives in one renderer protects only
that renderer.
"""

from __future__ import annotations

import logging

from tradingagents.agents.utils.integrity import (
    check_conviction_vs_data_quality,
    check_cross_stage_divergence,
    date_correction_reason,
)
from tradingagents.agents.utils.rating import parse_rating

logger = logging.getLogger(__name__)


def create_decision_reconciliation():
    """Build the reconciliation node."""

    def reconciliation_node(state) -> dict:
        decision = state.get("final_trade_decision") or ""

        # Prefer the typed rating captured by the PM node; fall back to parsing
        # the rendered markdown so a free-text (degraded) run still reconciles.
        pm_rating = state.get("portfolio_rating") or parse_rating(decision, default="")
        research_rating = state.get("research_recommendation") or ""
        trader_direction = state.get("trader_direction") or ""

        # Compare **view against view**. Both are views now, but keep reading
        # ``portfolio_view`` explicitly: it is the field that is guaranteed to be
        # the view even if the rendered **Rating** line ever changes again.
        pm_view = state.get("portfolio_view") or pm_rating
        findings = check_cross_stage_divergence(
            research_rating, trader_direction, pm_view,
        )

        # There used to be a rating/target round-trip assertion here. It cannot
        # run in the graph any more: the per-ticker decision carries a risk
        # *ceiling*, not a target, because the target needs the other N-1
        # holdings. The equivalent check now lives in
        # :func:`tradingagents.portfolio.allocator.allocate`, which derives the
        # position-change label from the target it just computed.

        # Conviction, so the view — a Sell *label* forced by the risk budget is
        # not a high-conviction call and must not be flagged as one.
        findings.extend(
            check_conviction_vs_data_quality(pm_view, date_correction_reason(state))
        )

        if findings:
            logger.warning(
                "Decision reconciliation findings for %s: %s",
                state.get("company_of_interest", "unknown"),
                "; ".join(f"{f.get('section')}:{f.get('reason')}" for f in findings),
            )

        return {
            "portfolio_rating": pm_rating,
            "portfolio_view": pm_view,
            "integrity_findings": findings,
        }

    return reconciliation_node
