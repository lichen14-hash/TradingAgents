# TradingAgents/graph/propagation.py

from typing import Any

from tradingagents.agents.utils.agent_states import (
    InvestDebateState,
    RiskDebateState,
)


class Propagator:
    """Handles state initialization and propagation through the graph."""

    def __init__(self, max_recur_limit=100):
        """Initialize with configuration parameters."""
        self.max_recur_limit = max_recur_limit

    def create_initial_state(
        self,
        company_name: str,
        trade_date: str,
        asset_type: str = "stock",
        past_context: str = "",
        instrument_context: str = "",
        data_bundle: dict | None = None,
        user_portfolio_context: str = "",
        user_position: dict | None = None,
    ) -> dict[str, Any]:
        """Create the initial state for the agent graph.

        ``instrument_context`` is the deterministic ticker-identity string
        resolved once at run start (see
        ``TradingAgentsGraph.resolve_instrument_context``). When empty, agents
        fall back to ticker-only context via
        ``get_instrument_context_from_state``.

        The two holding parameters are deliberately split.
        ``user_portfolio_context`` is prose for the **report** (it may mention
        the cost price, which users want to see) and must never reach a prompt.
        ``user_position`` carries the structured facts the decision layer is
        allowed to use — currently just ``position_pct``, which is a sizing
        input. See :mod:`tradingagents.agents.utils.position_sizing` for the
        measured bias that motivated the split.
        """
        market_status = {}
        if data_bundle:
            metadata = data_bundle.get("metadata", {})
            if isinstance(metadata, dict):
                market_status = metadata.get("market_status", {}) or {}

        return {
            "messages": [("human", company_name)],
            "company_of_interest": company_name,
            "asset_type": asset_type,
            "instrument_context": instrument_context,
            "trade_date": str(trade_date),
            "past_context": past_context,
            "user_portfolio_context": user_portfolio_context,
            "user_position": user_position or {},
            "data_bundle": data_bundle or {},
            "market_status": market_status,
            "investment_debate_state": InvestDebateState(
                {
                    "bull_history": "",
                    "bear_history": "",
                    "history": "",
                    "current_response": "",
                    "judge_decision": "",
                    "count": 0,
                }
            ),
            "risk_debate_state": RiskDebateState(
                {
                    "aggressive_history": "",
                    "conservative_history": "",
                    "neutral_history": "",
                    "history": "",
                    "latest_speaker": "",
                    "current_aggressive_response": "",
                    "current_conservative_response": "",
                    "current_neutral_response": "",
                    "judge_decision": "",
                    "count": 0,
                }
            ),
            "market_report": "",
            "fundamentals_report": "",
            "sentiment_report": "",
            "news_report": "",
            # Machine-readable per-stage ratings, filled by each decision node.
            # Seeded here so a stage that never ran leaves an empty string
            # rather than a missing key for the reconciliation node to read.
            "research_recommendation": "",
            "trader_direction": "",
            "portfolio_rating": "",
            "portfolio_view": "",
            "position_ceiling": {},
            "integrity_findings": [],
        }

    def get_graph_args(self, callbacks: list | None = None) -> dict[str, Any]:
        """Get arguments for the graph invocation.

        Args:
            callbacks: Optional list of callback handlers for tool execution tracking.
                       Note: LLM callbacks are handled separately via LLM constructor.
        """
        config = {"recursion_limit": self.max_recur_limit}
        if callbacks:
            config["callbacks"] = callbacks
        return {
            "stream_mode": "values",
            "config": config,
        }
