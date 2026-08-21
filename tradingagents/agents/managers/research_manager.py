"""Research Manager: turns the bull/bear debate into a structured investment plan for the trader."""

from __future__ import annotations

from tradingagents.agents.schemas import ResearchPlan, render_research_plan
from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.agents.utils.rating import parse_rating
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_guarded,
)


def create_research_manager(llm):
    structured_llm = bind_structured(llm, ResearchPlan, "Research Manager")

    def research_manager_node(state) -> dict:
        instrument_context = get_instrument_context_from_state(state)
        history = state["investment_debate_state"].get("history", "")

        investment_debate_state = state["investment_debate_state"]

        prompt = f"""As the Research Manager and debate facilitator, your role is to critically evaluate this round of debate and deliver a clear, actionable investment plan for the trader.

{instrument_context}

---

**Rating Scale — this is a VIEW rating, not a position instruction** (use exactly one):
- **Buy**: Decisively bullish; the bull case survived the bear's strongest attacks
- **Overweight**: Bullish on balance; the bull case is stronger but the bear raised real risks
- **Hold**: Genuinely balanced, or both cases are weak / the evidence is insufficient
- **Underweight**: Bearish on balance; the bear case is stronger but the bull case is not dead
- **Sell**: Decisively bearish; the bear case survived the bull's strongest defence

You do NOT know the user's current holdings, and you must not state a target position, a
percentage, or a lot size. Your rating expresses only how the evidence leans. Whether that
view implies buying, trimming, or doing nothing depends on the position the Portfolio
Manager can see and you cannot — inventing a position action here would be a guess dressed
up as instruction. Express your actionable content as *conditional triggers* instead
(e.g. "the bull case requires a close above X on rising volume; below Y the thesis fails").

Commit to a clear stance whenever the debate's strongest arguments warrant one; reserve Hold
for situations where the evidence on both sides is genuinely balanced.

**If either side's argument is missing, empty, or marked as "本轮未产生有效输出"**: say so
explicitly in your rationale and lower your conviction accordingly. A side that did not speak
has NOT conceded — do not read silence as agreement, and do not substitute your own version of
their argument for the argument they failed to make.

---

**Debate History:**
{history}""" + get_language_instruction()

        investment_plan, parsed, findings = invoke_structured_guarded(
            structured_llm,
            llm,
            prompt,
            render_research_plan,
            "Research Manager",
            section="研究经理裁定",
        )
        recommendation = (
            parsed.recommendation.value if parsed is not None
            else parse_rating(investment_plan, default="")
        )

        new_investment_debate_state = {
            "judge_decision": investment_plan,
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": investment_plan,
            "count": investment_debate_state["count"],
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": investment_plan,
            "research_recommendation": recommendation,
            "integrity_findings": findings,
        }

    return research_manager_node
