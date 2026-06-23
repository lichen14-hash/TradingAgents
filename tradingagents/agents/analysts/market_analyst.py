from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.datacollector.schema import DataBundle


_INDICATOR_GUIDE = """Indicator reference (use for interpreting the data below):

Moving Averages:
- close_50_sma: 50 SMA — medium-term trend indicator. Identifies trend direction and dynamic support/resistance.
- close_200_sma: 200 SMA — long-term trend benchmark. Confirms overall market trend, golden/death cross setups.
- close_10_ema: 10 EMA — responsive short-term average. Captures quick momentum shifts and potential entry points.

MACD Related:
- macd: MACD line — momentum via differences of EMAs. Watch for crossovers and divergence.
- macds: MACD Signal — EMA smoothing of the MACD line. Crossovers with MACD trigger trade signals.
- macdh: MACD Histogram — gap between MACD and its signal. Visualizes momentum strength.

Momentum:
- rsi: RSI — measures momentum to flag overbought (>70) / oversold (<30) conditions.
- mfi: MFI — Money Flow Index, volume-weighted RSI. Combines price and volume for flow analysis.

Volatility:
- boll: Bollinger Middle — 20 SMA basis for Bollinger Bands.
- boll_ub: Bollinger Upper Band — 2 std dev above middle, signals potential overbought / breakout zones.
- boll_lb: Bollinger Lower Band — 2 std dev below middle, signals potential oversold conditions.
- atr: ATR — Average True Range, measures volatility for stop-loss and position sizing.

Volume-Based:
- vwma: VWMA — volume-weighted moving average. Confirms trends by integrating price with volume."""


def create_market_analyst(llm):

    def market_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = get_instrument_context_from_state(state)

        bundle = DataBundle.model_validate(state["data_bundle"])
        m = bundle.market

        stock_block = m.stock_data if m else "<unavailable>"
        snapshot_block = m.verified_snapshot if m else "<unavailable>"

        indicators_block = "<unavailable>"
        if m and m.indicators:
            indicators_block = "\n\n".join(
                f"### {name}\n{data}" for name, data in m.indicators.items()
            )

        system_message = (
            "You are a trading assistant tasked with analyzing financial markets."
            " The following pre-fetched market data has been collected for you,"
            " including OHLCV price data, all available technical indicators,"
            " and a verified market snapshot.\n\n"
            + _INDICATOR_GUIDE + "\n\n"
            "Focus your analysis on the indicators most relevant to the current"
            " market regime. Not all 13 indicators will be equally informative —"
            " identify which ones are sending the strongest signals and which"
            " provide complementary confirmation.\n\n"
            "<stock_data>\n" + stock_block + "\n</stock_data>\n\n"
            "<technical_indicators>\n" + indicators_block + "\n</technical_indicators>\n\n"
            "<verified_snapshot>\n" + snapshot_block + "\n</verified_snapshot>\n\n"
            "The verified snapshot is the source of truth for any exact OHLCV,"
            " price-level, or indicator-value claim. If other data conflicts with"
            " the verified snapshot, flag the discrepancy rather than inventing a"
            " reconciled number. Do not claim historical validation, support/resistance"
            " bounces, or exact percentage moves unless directly supported by the"
            " data above with concrete dates and prices.\n\n"
            "Write a very detailed and nuanced report of the trends you observe."
            " Provide specific, actionable insights with supporting evidence to"
            " help traders make informed decisions."
            " Make sure to append a Markdown table at the end of the report to"
            " organize key points in the report, organized and easy to read."
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
            "market_report": report,
        }

    return market_analyst_node
