from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.datacollector.schema import DataBundle


def create_news_analyst(llm):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        asset_type = state.get("asset_type", "stock")
        asset_label = "company" if asset_type == "stock" else "asset"
        instrument_context = get_instrument_context_from_state(state)

        bundle = DataBundle.model_validate(state["data_bundle"])
        n = bundle.news

        ticker_news_block = n.ticker_news if n else "<unavailable>"
        global_news_block = n.global_news if n else "<unavailable>"
        insider_block = n.insider_transactions if n else "<unavailable>"

        macro_block = "<unavailable>"
        if n and n.macro_indicators:
            macro_block = "\n\n".join(
                f"### {name}\n{data}" for name, data in n.macro_indicators.items()
            )

        prediction_block = "<unavailable>"
        if n and n.prediction_markets:
            prediction_block = "\n\n".join(
                f"### {query}\n{data}" for query, data in n.prediction_markets.items()
            )

        system_message = (
            f"You are a news researcher tasked with analyzing recent news and trends over the past week."
            f" Please write a comprehensive report of the current state of the world that is relevant"
            f" for trading and macroeconomics for this {asset_label}."
            f" The following pre-fetched data has been collected for you.\n\n"
            f"<ticker_news>\n{ticker_news_block}\n</ticker_news>\n\n"
            f"<global_news>\n{global_news_block}\n</global_news>\n\n"
            f"<insider_transactions>\n{insider_block}\n</insider_transactions>\n\n"
            f"<macro_indicators>\n{macro_block}\n</macro_indicators>\n\n"
            f"<prediction_markets>\n{prediction_block}\n</prediction_markets>\n\n"
            "Provide specific, actionable insights with supporting evidence to help traders"
            " make informed decisions. Make sure to append a Markdown table at the end of"
            " the report to organize key points in the report, organized and easy to read."
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
            "news_report": report,
        }

    return news_analyst_node
