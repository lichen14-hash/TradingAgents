"""Tests for the configurable sampling temperature (#178/#168).

Temperature is a cross-provider knob: when set it must reach the underlying
chat client; when unset the provider keeps its own default.
"""

import importlib

import pytest

from tradingagents.llm_clients.factory import create_llm_client


@pytest.mark.unit
class TestTemperatureForwarding:
    @pytest.mark.parametrize(
        "provider,model",
        [
            ("openai", "gpt-4.1"),
            ("anthropic", "claude-sonnet-4-6"),
            ("google", "gemini-2.5-flash"),
            ("deepseek", "deepseek-chat"),
        ],
    )
    def test_temperature_reaches_client_when_set(self, provider, model):
        llm = create_llm_client(
            provider=provider, model=model, temperature=0.0, api_key="placeholder"
        ).get_llm()
        assert llm.temperature == 0.0

    def test_temperature_omitted_leaves_provider_default(self):
        # Not passing temperature must not force it to a value.
        llm = create_llm_client(
            provider="openai", model="gpt-4.1", api_key="placeholder"
        ).get_llm()
        # langchain's default is unset/None, not 0.0
        assert llm.temperature is None


@pytest.mark.unit
class TestTemperatureEnvOverlay:
    def test_env_sets_temperature(self, monkeypatch):
        import tradingagents.default_config as dc
        monkeypatch.setenv("TRADINGAGENTS_TEMPERATURE", "0.2")
        importlib.reload(dc)
        # Stored on config (string from env is fine; consumed via float()).
        assert dc.DEFAULT_CONFIG["temperature"] in ("0.2", 0.2)
        assert float(dc.DEFAULT_CONFIG["temperature"]) == 0.2
        monkeypatch.delenv("TRADINGAGENTS_TEMPERATURE", raising=False)
        importlib.reload(dc)

    def test_default_temperature_is_none(self, monkeypatch):
        import tradingagents.default_config as dc
        monkeypatch.delenv("TRADINGAGENTS_TEMPERATURE", raising=False)
        importlib.reload(dc)
        assert dc.DEFAULT_CONFIG["temperature"] is None


@pytest.mark.unit
class TestProviderKwargsTemperature:
    """_get_provider_kwargs float-coerces and forwards temperature, or omits it."""

    def _kwargs_for(self, temperature, tier_key=None, **extra):
        from tradingagents.graph.trading_graph import TradingAgentsGraph
        # Call the method without constructing the full graph.
        graph = TradingAgentsGraph.__new__(TradingAgentsGraph)
        graph.config = {"llm_provider": "openai", "temperature": temperature, **extra}
        return TradingAgentsGraph._get_provider_kwargs(graph, tier_key)

    def test_float_string_coerced(self):
        assert self._kwargs_for("0.3")["temperature"] == 0.3

    def test_float_passthrough(self):
        assert self._kwargs_for(0.0)["temperature"] == 0.0

    def test_none_omitted(self):
        assert "temperature" not in self._kwargs_for(None)

    def test_empty_string_omitted(self):
        assert "temperature" not in self._kwargs_for("")


@pytest.mark.unit
class TestTieredTemperature:
    """Per-tier temperatures: decision/debate tiers with global override."""

    def _kwargs_for(self, temperature, tier_key=None, **extra):
        from tradingagents.graph.trading_graph import TradingAgentsGraph
        graph = TradingAgentsGraph.__new__(TradingAgentsGraph)
        graph.config = {"llm_provider": "openai", "temperature": temperature, **extra}
        return TradingAgentsGraph._get_provider_kwargs(graph, tier_key)

    def test_tier_used_when_global_unset(self):
        kwargs = self._kwargs_for(None, "decision_temperature", decision_temperature=0.2)
        assert kwargs["temperature"] == 0.2

    def test_tier_string_coerced(self):
        kwargs = self._kwargs_for(None, "debate_temperature", debate_temperature="0.8")
        assert kwargs["temperature"] == 0.8

    def test_global_overrides_tier(self):
        kwargs = self._kwargs_for(0.5, "decision_temperature", decision_temperature=0.2)
        assert kwargs["temperature"] == 0.5

    def test_both_unset_omitted(self):
        kwargs = self._kwargs_for(None, "decision_temperature", decision_temperature=None)
        assert "temperature" not in kwargs

    def test_defaults_present_in_config(self):
        from tradingagents.default_config import DEFAULT_CONFIG
        # Default is opt-in (None): thinking-enabled Claude models reject
        # any temperature other than 1, so tiers must not be forced on.
        assert DEFAULT_CONFIG["decision_temperature"] is None
        assert DEFAULT_CONFIG["debate_temperature"] is None

    def test_env_overrides_tier_temperatures(self, monkeypatch):
        import tradingagents.default_config as dc
        monkeypatch.setenv("TRADINGAGENTS_DECISION_TEMPERATURE", "0.1")
        monkeypatch.setenv("TRADINGAGENTS_DEBATE_TEMPERATURE", "0.9")
        importlib.reload(dc)
        assert float(dc.DEFAULT_CONFIG["decision_temperature"]) == 0.1
        assert float(dc.DEFAULT_CONFIG["debate_temperature"]) == 0.9
        monkeypatch.delenv("TRADINGAGENTS_DECISION_TEMPERATURE", raising=False)
        monkeypatch.delenv("TRADINGAGENTS_DEBATE_TEMPERATURE", raising=False)
        importlib.reload(dc)


@pytest.mark.unit
class TestGraphSetupDebateLLM:
    """GraphSetup wires the debate LLM to debate roles, falling back to quick."""

    def test_fallback_to_quick_when_absent(self):
        from tradingagents.graph.setup import GraphSetup
        quick, deep = object(), object()
        gs = GraphSetup(quick, deep, conditional_logic=None, config={})
        assert gs.debate_llm is quick

    def test_dedicated_debate_llm_kept(self):
        from tradingagents.graph.setup import GraphSetup
        quick, deep, debate = object(), object(), object()
        gs = GraphSetup(quick, deep, conditional_logic=None, config={}, debate_llm=debate)
        assert gs.debate_llm is debate
