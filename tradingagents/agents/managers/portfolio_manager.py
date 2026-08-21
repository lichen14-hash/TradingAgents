"""Portfolio Manager: synthesises the risk-analyst debate into the final decision.

Uses LangChain's ``with_structured_output`` so the LLM produces a typed
``PortfolioDecision`` directly, in a single call.  The result is rendered
back to markdown for storage in ``final_trade_decision`` so memory log,
CLI display, and saved reports continue to consume the same shape they do
today.  When a provider does not expose structured output, the agent falls
back gracefully to free-text generation.

**The size is not the model's to choose.** The model supplies a view tier only.
The reason is measured, not theoretical: while the model chose the number, it
anchored on the user's unrealised loss (0% add / 71% trim on deeply-underwater
positions vs 23.8% / 19.0% on profitable ones, p=0.00012 over 327 predictions),
and forward returns often contradicted the resulting trims. The prompt therefore
no longer receives a cost price at all — see
``tests/test_position_context_isolation.py``.

This node computes only the **risk ceiling** ("at most this much of the
portfolio, given this stock's volatility" —
:func:`tradingagents.agents.utils.position_sizing.size_ceiling`), because that
needs nothing but this ticker's ATR and close. The **target weight** is not
computed here: it depends on what the other N-1 names are doing, so it is
produced after the whole batch finishes by
:func:`tradingagents.portfolio.allocator.allocate`. One consequence matters for
anyone reading the state: ``portfolio_rating`` here is the *view*, not a
position-change label.
"""

from __future__ import annotations

from tradingagents.agents.schemas import PortfolioDecision, render_pm_decision
from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.agents.utils.integrity import data_quality_caveat
from tradingagents.agents.utils.position_sizing import describe_ceiling, size_ceiling
from tradingagents.agents.utils.rating import parse_rating
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_guarded,
)
from tradingagents.utils.bundle_inputs import market_sizing_inputs


def _current_position_pct(state) -> float | None:
    """The user's present weight in this name, or None when unknown.

    Read from the structured ``user_position`` state key rather than by parsing
    the prompt text: the weight is the *only* holding datum the decision layer
    is allowed to see, and keeping it structured is what lets the cost price be
    dropped from the prompt without losing the sizing input.
    """
    position = state.get("user_position")
    if not isinstance(position, dict):
        return None
    value = position.get("position_pct")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def create_portfolio_manager(llm, config: dict | None = None):
    structured_llm = bind_structured(llm, PortfolioDecision, "Portfolio Manager")

    def portfolio_manager_node(state) -> dict:
        instrument_context = get_instrument_context_from_state(state)

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        research_plan = state["investment_plan"]
        trader_plan = state["trader_investment_plan"]

        past_context = state.get("past_context", "")
        lessons_line = (
            f"- Lessons from prior decisions and outcomes:\n{past_context}\n"
            if past_context
            else ""
        )

        current_pct = _current_position_pct(state)
        holding_line = (
            f"- The user currently holds **{current_pct}%** of their portfolio in "
            "this instrument. This is given so you know whether the decision is "
            "an entry, an exit, or an adjustment — nothing more. You are not "
            "told their cost price, and you must not ask for it, guess it, or "
            "reason about unrealised profit or loss: where a position sits "
            "relative to its entry price says nothing about what it will do "
            "next, and reasoning from it is a documented bias in this system.\n"
            if current_pct is not None
            else "- No position context was provided, so treat this as a "
                 "fresh-entry judgement.\n"
        )
        data_caveat = data_quality_caveat(state)

        prompt = f"""As the Portfolio Manager, synthesize the risk analysts' debate and deliver the final trading decision.

{instrument_context}

---

**Your job is the view, not the size.** State how the evidence leans in `portfolio_view`; the position size is computed after you, deterministically, from this stock's volatility against a fixed risk budget. Do NOT state a target position, a percentage of portfolio, a lot count, or a share count anywhere in your output — such a number would either duplicate the computed size or contradict it, and it will not be used.

**View Scale** (use exactly one) — this describes the *evidence*, not a position change:
- **Buy**: decisively bullish; the bull case is well supported and the bear case is weak
- **Overweight**: bullish on balance
- **Hold**: genuinely balanced, or both cases weak. Use this when you would neither add nor reduce on the merits.
- **Underweight**: bearish on balance
- **Sell**: decisively bearish; the thesis is broken or the risk is unacceptable

Reserve **Hold** for genuine balance rather than as a way to avoid committing. Do not use **Sell** to mean "do not enter" — with no position, staying out is **Hold**.

Express anything actionable as *conditional triggers* (price, volume, or fundamental conditions that would confirm or invalidate the view), not as sizes.

**Context — the two stages before you (read their conclusions before forming yours):**
- Research Manager's investment plan: **{research_plan}**
  (Their rating is a **view** rating on this same scale, so it is directly comparable to yours. They cannot see the user's holdings.)
- Trader's transaction proposal: **{trader_plan}**
{lessons_line}{holding_line}{data_caveat}
**Risk Analysts Debate History:**
{history}

---

**Override discipline:** you may reach a different conclusion than the Research Manager or the Trader — you see the risk debate and they do not. But if your view differs in *direction* from theirs (you lean bullish where they leaned bearish, or the reverse), you must record it in `divergence_from_research`: name whose conclusion you are overriding and the specific evidence that overturned it.

If a debate participant's argument is missing or marked "本轮未产生有效输出", treat that viewpoint as absent rather than as agreement, and say so in your thesis.

Be decisive and ground every conclusion in specific evidence from the analysts.{get_language_instruction()}"""

        atr, close = market_sizing_inputs(
            state.get("data_bundle"),
            (state.get("data_bundle") or {}).get("metadata", {}).get("trade_date", "")
            if isinstance(state.get("data_bundle"), dict) else "",
        )
        # The ceiling depends only on this stock's volatility, not on the view and
        # not on the other holdings, so it can be computed before the call and
        # used by both the structured and the free-text path. The *target* is not
        # computed here at all — see the module docstring.
        ceiling = size_ceiling(atr, close, config)

        def _render(decision: PortfolioDecision) -> str:
            return render_pm_decision(decision, ceiling, current_pct, config)

        final_trade_decision, parsed, findings = invoke_structured_guarded(
            structured_llm,
            llm,
            prompt,
            _render,
            "Portfolio Manager",
            section="组合经理最终裁定",
        )

        if parsed is not None:
            pm_view = parsed.portfolio_view.value
        else:
            pm_view = parse_rating(final_trade_decision, default="")
            # Free-text fallback: ``_render`` never ran, so append the ceiling
            # lines here. The model was told not to produce numbers, so without
            # this a degraded run would carry no risk ceiling at all.
            if current_pct is not None:
                final_trade_decision += f"\n\n**Current Position**: {current_pct}%"
            final_trade_decision += (
                f"\n\n**Risk Ceiling**: {ceiling.allowed_pct}%"
                f"\n\n**Sizing**: {describe_ceiling(ceiling, config)}"
            )

        findings = list(findings) + list(ceiling.findings)

        new_risk_debate_state = {
            "judge_decision": final_trade_decision,
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state["aggressive_history"],
            "conservative_history": risk_debate_state["conservative_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_debate_state["current_aggressive_response"],
            "current_conservative_response": risk_debate_state["current_conservative_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": final_trade_decision,
            # Both are the view. ``portfolio_rating`` briefly carried the sizer's
            # position-change label, which broke comparability with the 327
            # historical rows where ``predictions.rating`` means the view. The
            # position-change label now lives in the ``allocations`` table, where
            # it describes the *portfolio-level* target it is actually derived
            # from. The two keys stay separate because ``portfolio_rating`` is
            # what the signal processor parses out of the markdown.
            "portfolio_rating": pm_view,
            "portfolio_view": pm_view,
            "position_ceiling": ceiling.as_dict(),
            "integrity_findings": findings,
        }

    return portfolio_manager_node
