"""长回答必须走流式 + 显式超时（整只标的丢失的根因）。

事故：2026-08-18/19 两次批量里，各有标的以 `Connection error.` 整只失败
（09988.HK），日志里还出现过网关的 Tengine/AServer-Ingress 504。机制是：
非流式请求下，从发出到收到第一个字节之间**没有任何流量**，一次深度思考的回答要花
几分钟，网关在自己的空闲超时上把连接掐掉；客户端侧 `default_request_timeout` 是
None，等于永不超时，于是没有可控的失败点，只能等对端断开、整轮分析报废。

流式改变的是这一点：token 一产生就发，连接上持续有流量，网关的空闲计时器不断复位。
配合显式 timeout，超时变成"块之间的间隔"而不是"整个回答的总时长"。

两处实现细节值得钉住：

1. 默认值放在 provider client 里，而不是调用方的 config：CLI、web、回测、各个
   smoke 脚本都各自 `create_llm_client(...)`，放在 config 里就要每处都记得写。
2. `langchain_anthropic` 自己从不读 `self.streaming` 来决定走不走流式；是
   langchain_core 的 `_should_stream` 在看 `"streaming" in self.model_fields_set
   and self.streaming is True`。所以必须**显式传参**构造，靠类属性默认值无效。
"""

from unittest.mock import MagicMock

import pytest

from tradingagents.llm_clients.anthropic_client import (
    _DEFAULT_REQUEST_TIMEOUT,
    _DEFAULT_STREAMING,
    AnthropicClient,
)
from tradingagents.llm_clients.openai_client import OpenAIClient


def _anthropic(**kwargs):
    return AnthropicClient("claude-opus-5", api_key="test-key", **kwargs).get_llm()


@pytest.mark.unit
class TestAnthropicDefaults:
    def test_streaming_is_on_by_default(self):
        assert _DEFAULT_STREAMING is True
        assert _anthropic().streaming is True

    def test_streaming_actually_routes_through_the_streaming_path(self):
        """回归锚点：只设 `streaming=True` 不够，得让 `_should_stream` 看见它。

        langchain_core 判断的是字段有没有被**显式设置**（`model_fields_set`），
        类属性默认值不算。
        """
        llm = _anthropic()
        assert "streaming" in llm.model_fields_set
        assert llm._should_stream(async_api=False) is True

    def test_timeout_is_explicit_by_default(self):
        assert _DEFAULT_REQUEST_TIMEOUT == 300.0
        assert _anthropic().default_request_timeout == _DEFAULT_REQUEST_TIMEOUT

    def test_explicit_values_win(self):
        llm = _anthropic(streaming=False, timeout=45.0)
        assert llm.streaming is False
        assert llm.default_request_timeout == 45.0

    def test_disabling_streaming_warns_about_the_timeout_semantics(self, caplog):
        """关掉流式后 timeout 从"块间隔"变成"总时长"，这个语义变化必须说出来。"""
        with caplog.at_level("WARNING"):
            _anthropic(streaming=False)
        assert any("Streaming disabled" in r.message for r in caplog.records)

    def test_streaming_preserves_the_truncation_signal(self):
        """`stop_reason` 到达于 message_delta 且能穿过 chunk 聚合。

        否则 `_retry_if_truncated`（max_tokens 截断后加倍重试）会在流式下失效——
        那是另一个已修的事故，不能被这次修复反向打破。
        """
        from tradingagents.llm_clients.anthropic_client import _stop_reason

        aggregated = MagicMock()
        aggregated.response_metadata = {"stop_reason": "max_tokens"}
        assert _stop_reason(aggregated) == "max_tokens"


@pytest.mark.unit
class TestOpenAICompatibleProviders:
    """OpenAI 兼容端点上默认不开流式：部分端点对 `stream=true` + 结构化输出直接报错。"""

    def test_streaming_is_passthrough_but_not_defaulted(self):
        llm = OpenAIClient(
            "gpt-4o", api_key="test-key", base_url="https://example.invalid/v1",
        ).get_llm()
        assert getattr(llm, "streaming", False) is False

    def test_streaming_can_still_be_opted_into(self):
        llm = OpenAIClient(
            "gpt-4o", api_key="test-key", base_url="https://example.invalid/v1",
            streaming=True,
        ).get_llm()
        assert llm.streaming is True


@pytest.mark.unit
class TestConfigWiring:
    """`TRADINGAGENTS_LLM_STREAMING` / `_TIMEOUT` 要能一路走到 provider kwargs。"""

    def _kwargs(self, **overrides) -> dict:
        from tradingagents.default_config import DEFAULT_CONFIG
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        config = DEFAULT_CONFIG.copy()
        config.update(overrides)
        graph = TradingAgentsGraph.__new__(TradingAgentsGraph)
        graph.config = config
        return graph._get_provider_kwargs()

    def test_defaults_leave_the_decision_to_the_provider_client(self):
        """默认 None：Anthropic 开流式、OpenAI 兼容端点不开，各自决定。"""
        kwargs = self._kwargs()
        assert "streaming" not in kwargs
        assert "timeout" not in kwargs

    @pytest.mark.parametrize("value,expected", [
        ("true", True), ("1", True), ("yes", True), ("on", True),
        ("false", False), ("0", False), ("", None),
        (True, True), (False, False),
    ])
    def test_streaming_env_string_is_coerced(self, value, expected):
        """环境变量到达时是字符串；`"false"` 按 bool 求值会是 True。"""
        kwargs = self._kwargs(llm_streaming=value)
        if expected is None:
            assert "streaming" not in kwargs
        else:
            assert kwargs["streaming"] is expected

    @pytest.mark.parametrize("value,expected", [("600", 600.0), (120, 120.0)])
    def test_timeout_is_coerced_to_float(self, value, expected):
        assert self._kwargs(llm_request_timeout=value)["timeout"] == expected

    def test_env_override_keys_are_registered(self):
        from tradingagents.default_config import _ENV_OVERRIDES

        assert _ENV_OVERRIDES["TRADINGAGENTS_LLM_STREAMING"] == "llm_streaming"
        assert _ENV_OVERRIDES["TRADINGAGENTS_LLM_TIMEOUT"] == "llm_request_timeout"
