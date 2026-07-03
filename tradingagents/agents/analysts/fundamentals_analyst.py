from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.datacollector.schema import DataBundle


def create_fundamentals_analyst(llm):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = get_instrument_context_from_state(state)

        bundle = DataBundle.model_validate(state["data_bundle"])
        f = bundle.fundamentals
        overview_block = f.overview if f else "<unavailable>"
        bs_q_block = f.balance_sheet_quarterly if f else "<unavailable>"
        bs_a_block = f.balance_sheet_annual if f else "<unavailable>"
        cf_q_block = f.cashflow_quarterly if f else "<unavailable>"
        cf_a_block = f.cashflow_annual if f else "<unavailable>"
        inc_q_block = f.income_quarterly if f else "<unavailable>"
        inc_a_block = f.income_annual if f else "<unavailable>"
        comp_intel_block = f.competitive_intelligence if f else "<unavailable>"

        system_message = (
            "You are a researcher tasked with analyzing fundamental information about a company."
            " The following pre-fetched financial data has been collected for you."
            " Please write a comprehensive report of the company's fundamental information"
            " including financial documents, company profile, basic company financials,"
            " and company financial history to gain a full view of the company's"
            " fundamental information to inform traders. Make sure to include as much"
            " detail as possible. Provide specific, actionable insights with supporting"
            " evidence to help traders make informed decisions.\n\n"
            "IMPORTANT - Competitive Moat & Barriers Analysis:\n"
            "Include a dedicated section (approximately 20-30% of your report) analyzing the company's competitive moat.\n"
            "The majority of your report (70-80%) should still focus on financial fundamentals analysis\n"
            "(revenue trends, profitability, cash flow health, balance sheet strength, valuation ratios).\n"
            "Specifically address:\n"
            "1. Core Technologies: What specific technologies, patents, or proprietary processes"
            "   does the company possess? What makes them technically difficult to replicate?\n"
            "2. Switching Costs: Why would existing customers find it costly or risky to switch"
            "   to competitors? (e.g., integration depth, certification requirements, ecosystem lock-in)\n"
            "3. Scale & Cost Advantages: Does the company benefit from economies of scale,"
            "   manufacturing learning curves, or cost structures competitors cannot match?\n"
            "4. Market Position: What is the company's market share in key segments?"
            "   Who are the top 3-5 competitors and why are they weaker?\n"
            "5. Intangible Assets: Brand reputation, regulatory licenses, customer relationships,"
            "   industry certifications that create barriers to entry.\n"
            "If the data is insufficient to fully answer these, state what is known and what gaps exist.\n\n"
            "<company_overview>\n" + overview_block + "\n</company_overview>\n\n"
            "<balance_sheet_quarterly>\n" + bs_q_block + "\n</balance_sheet_quarterly>\n\n"
            "<balance_sheet_annual>\n" + bs_a_block + "\n</balance_sheet_annual>\n\n"
            "<cashflow_quarterly>\n" + cf_q_block + "\n</cashflow_quarterly>\n\n"
            "<cashflow_annual>\n" + cf_a_block + "\n</cashflow_annual>\n\n"
            "<income_statement_quarterly>\n" + inc_q_block + "\n</income_statement_quarterly>\n\n"
            "<income_statement_annual>\n" + inc_a_block + "\n</income_statement_annual>\n\n"
            "<competitive_intelligence>\n" + comp_intel_block + "\n</competitive_intelligence>\n\n"
            "The <competitive_intelligence> section above contains web search results about the company's"
            " competitive moat, patents, market share, key customers, and competitive landscape."
            " Use this as supporting evidence in your Moat section, but do NOT let it overshadow"
            " your financial fundamentals analysis. Treat it as supplementary context, not the core.\n\n"
            "Make sure to append a Markdown table at the end of the report to organize"
            " key points in the report, organized and easy to read."
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    "\n{system_message}"
                    "For your reference, the current date is {current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        formatted_messages = prompt.format_messages(messages=state["messages"])
        result = llm.invoke(formatted_messages)
        report = result.content

        return {
            "messages": [AIMessage(content=report)],
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node
