"""Trader: turns the Research Manager's investment plan into a concrete transaction proposal.

The trader used to receive ``user_portfolio_context`` — including the user's
cost price — which made it the second of two agents reasoning from unrealised
P&L (the Portfolio Manager was the other). That is the measured source of the
"cut whatever is down the most" bias, so the holding context here is reduced to
the one fact a trader legitimately needs: whether there is a position to work
with at all. The size itself is computed downstream from volatility
(:mod:`tradingagents.agents.utils.position_sizing`), so the trader contributes
execution *staging* rather than a weight.
"""

from __future__ import annotations

import functools

from langchain_core.messages import AIMessage

from tradingagents.agents.schemas import TraderProposal, render_trader_proposal
from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.agents.utils.rating import parse_rating
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_guarded,
)


def create_trader(llm):
    structured_llm = bind_structured(llm, TraderProposal, "Trader")

    def trader_node(state, name):
        company_name = state["company_of_interest"]
        instrument_context = get_instrument_context_from_state(state)
        investment_plan = state["investment_plan"]
        position = state.get("user_position")
        held_pct = position.get("position_pct") if isinstance(position, dict) else None
        if held_pct is None:
            portfolio_section = (
                "\n\nThe user holds no position in this instrument, so any Buy "
                "is an entry. Do not state a position size or percentage of "
                "portfolio — that is computed from volatility downstream. "
                "Describe execution staging instead."
            )
        else:
            portfolio_section = (
                f"\n\nThe user currently holds {held_pct}% of their portfolio in "
                "this instrument, so a Sell is a reduction of an existing "
                "position rather than a short. You are deliberately not told "
                "their cost price: where the position sits relative to its "
                "entry price is irrelevant to what the stock does next. Do not "
                "state a target size or percentage of portfolio — that is "
                "computed from volatility downstream. Describe execution "
                "staging instead."
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a trading agent analyzing market data to make investment decisions. "
                    "Based on your analysis, provide a specific recommendation to buy, sell, or hold. "
                    "Anchor your reasoning in the analysts' reports and the research plan."
                    + get_language_instruction()
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Based on a comprehensive analysis by a team of analysts, here is an investment "
                    f"plan tailored for {company_name}. {instrument_context} This plan incorporates "
                    f"insights from current technical market trends, macroeconomic indicators, and "
                    f"social media sentiment. Use this plan as a foundation for evaluating your next "
                    f"trading decision.\n\nProposed Investment Plan: {investment_plan}\n\n"
                    f"Leverage these insights to make an informed and strategic decision."
                    f"{portfolio_section}"
                ),
            },
        ]

        trader_plan, parsed, findings = invoke_structured_guarded(
            structured_llm,
            llm,
            messages,
            render_trader_proposal,
            "Trader",
            section="交易员提案",
        )
        direction = (
            parsed.action.value if parsed is not None
            else parse_rating(trader_plan, default="")
        )

        return {
            "messages": [AIMessage(content=trader_plan)],
            "trader_investment_plan": trader_plan,
            "trader_direction": direction,
            "sender": name,
            "integrity_findings": findings,
        }

    return functools.partial(trader_node, name="Trader")
