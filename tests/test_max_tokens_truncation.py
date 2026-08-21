"""输出预算：不能让库替新模型猜 max_tokens（截断的根因）。

事故背景（300760.SZ，2026-08-11）：基本面分析师连续两次返回空正文、情绪分析师
的结构化输出被截断成半个 JSON（Pydantic ValidationError）。根因不在 agent 层，
而在 `max_tokens` 的解析链上：配置里 `max_tokens=None`，于是
`langchain_anthropic.ChatAnthropic` 自己去它内置的 model-profile 表里查模型的输出
上限，查不到就回落到 `_FALLBACK_MAX_OUTPUT_TOKENS = 4096`。`claude-opus-5` 比装机
的包新，所以每一次调用都跑在 4096 的输出上限下——同一个包给 `claude-opus-4-5` 的是
64000，整整 16 倍的静默降级。叠加服务端开启的 extended thinking（推理与正文共享同
一个预算），推理吃光预算后正文就是空的。

这是结构性的：每次升级模型都会重演一次。所以本测试钉死两层防御：

1. 库不认识的模型必须拿到显式的、宽裕的上限；库认识的模型必须**原样放过**
   （老模型的真实上限比我们的下限更低——claude-3-opus 就是 4096，硬塞 32000 是
   400 报错而不是修复）；
2. 响应带 `stop_reason="max_tokens"` 时自动加倍预算重试一次——留痕守卫只能记录
   截断，无法把被砍掉的正文找回来。
"""

from unittest.mock import MagicMock, patch

import pytest

from tradingagents.llm_clients import anthropic_client as ac
from tradingagents.llm_clients.anthropic_client import (
    _TRUNCATION_RETRY_CEILING,
    _UNKNOWN_MODEL_MAX_OUTPUT_TOKENS,
    AnthropicClient,
    NormalizedChatAnthropic,
    _resolve_default_max_tokens,
    _stop_reason,
)

# 事故当时在用的模型，以及库确实认识的对照组。
UNKNOWN_MODELS = ("claude-opus-5", "claude-sonnet-5", "claude-mythos-preview")
KNOWN_MODELS = (
    "claude-opus-4-5-20251101",   # 库里 64000
    "claude-haiku-4-5-20251001",  # 库里 64000
    "claude-3-opus-20240229",     # 库里 4096——真实上限就是 4096，不能被抬高
)


def _reply(text: str = "ok", stop_reason: str = "end_turn"):
    msg = MagicMock()
    msg.content = text
    msg.response_metadata = {"stop_reason": stop_reason}
    return msg


@pytest.mark.unit
class TestResolveDefaultMaxTokens:
    @pytest.mark.parametrize("model", UNKNOWN_MODELS)
    def test_unknown_models_get_an_explicit_floor(self, model):
        """库不认识 → 必须给显式上限，不能继承 4096 的回落值。"""
        assert _resolve_default_max_tokens(model) == _UNKNOWN_MODEL_MAX_OUTPUT_TOKENS

    @pytest.mark.parametrize("model", KNOWN_MODELS)
    def test_known_models_are_left_to_the_library(self, model):
        """库认识 → 返回 None，由库用它自己表里的真实上限。"""
        assert _resolve_default_max_tokens(model) is None

    def test_missing_library_helper_leaves_the_cap_alone(self):
        """私有 helper 是版本相关的：读不到时宁可不动，也不猜一个会被拒的值。"""
        with patch.dict(
            "sys.modules",
            {"langchain_anthropic.chat_models": MagicMock(spec=[])},
        ):
            assert _resolve_default_max_tokens("claude-opus-5") is None

    def test_floor_stays_within_the_retry_ceiling(self):
        assert _UNKNOWN_MODEL_MAX_OUTPUT_TOKENS <= _TRUNCATION_RETRY_CEILING


@pytest.mark.unit
class TestClientAppliesTheFloor:
    def _llm(self, model: str, **kwargs):
        return AnthropicClient(model, api_key="test-key", **kwargs).get_llm()

    @pytest.mark.parametrize("model", UNKNOWN_MODELS)
    def test_unknown_model_is_constructed_with_the_floor(self, model):
        assert self._llm(model).max_tokens == _UNKNOWN_MODEL_MAX_OUTPUT_TOKENS

    def test_incident_model_is_no_longer_capped_at_4096(self):
        """回归锚点：事故当时这里就是 4096。"""
        assert self._llm("claude-opus-5").max_tokens > 4096

    @pytest.mark.parametrize("model,expected", [
        ("claude-opus-4-5-20251101", 64000),
        ("claude-3-opus-20240229", 4096),  # 真实上限更低，抬高就是 400
    ])
    def test_known_model_keeps_the_library_value(self, model, expected):
        assert self._llm(model).max_tokens == expected

    def test_explicit_config_value_wins(self):
        assert self._llm("claude-opus-5", max_tokens=8000).max_tokens == 8000

    def test_explicit_none_is_treated_as_absent(self):
        """`config["max_tokens"]` 默认是 None，不能当成"用户要求 None"传给 pydantic。"""
        assert self._llm("claude-opus-5", max_tokens=None).max_tokens == (
            _UNKNOWN_MODEL_MAX_OUTPUT_TOKENS
        )


@pytest.mark.unit
class TestStopReason:
    @pytest.mark.parametrize("meta,expected", [
        ({"stop_reason": "max_tokens"}, "max_tokens"),
        ({"stop_reason": "end_turn"}, "end_turn"),
        ({"stop_reason": None}, ""),
        ({}, ""),
        (None, ""),
    ])
    def test_reads_stop_reason_defensively(self, meta, expected):
        response = MagicMock()
        response.response_metadata = meta
        assert _stop_reason(response) == expected

    def test_response_without_metadata_attribute(self):
        assert _stop_reason(object()) == ""


@pytest.mark.unit
class TestTruncationRetry:
    """`_retry_if_truncated` 的行为。

    这里直接构造 `NormalizedChatAnthropic` 并打桩它父类的 `invoke`，因为要验证的
    正是"父类返回被截断的响应之后，我们做了什么"。
    """

    def _llm(self, max_tokens: int = 4096) -> NormalizedChatAnthropic:
        return NormalizedChatAnthropic(
            model="claude-opus-5", api_key="test-key", max_tokens=max_tokens,
        )

    def _invoke(self, llm, responses):
        """打桩 ChatAnthropic.invoke，返回 (正文, 每次调用收到的 kwargs)。"""
        calls = []

        def fake_invoke(self, input, config=None, **kwargs):  # noqa: A002
            calls.append(kwargs)
            return responses[len(calls) - 1]

        with patch.object(ac.ChatAnthropic, "invoke", fake_invoke):
            result = llm.invoke("prompt")
        return result.content, calls

    def test_truncated_response_triggers_one_retry_with_a_doubled_budget(self):
        llm = self._llm(max_tokens=4096)
        result, calls = self._invoke(llm, [
            _reply("", stop_reason="max_tokens"),   # 事故形态：正文为空
            _reply("完整正文", stop_reason="end_turn"),
        ])
        assert len(calls) == 2, "截断必须重试，且只重试一次"
        assert calls[0].get("max_tokens") is None, "首次调用沿用实例上的预算"
        assert calls[1]["max_tokens"] == 8192, "重试预算 = 2 x 4096"
        assert result == "完整正文", "返回的必须是重试后的完整正文"

    def test_normal_response_is_not_retried(self):
        llm = self._llm()
        result, calls = self._invoke(llm, [_reply("正文", stop_reason="end_turn")])
        assert len(calls) == 1
        assert result == "正文"

    @pytest.mark.parametrize("stop_reason", ["end_turn", "stop_sequence", "tool_use", ""])
    def test_only_max_tokens_stop_reason_retries(self, stop_reason):
        _result, calls = self._invoke(
            self._llm(), [_reply("正文", stop_reason=stop_reason)]
        )
        assert len(calls) == 1

    def test_retry_budget_has_a_floor_of_8000(self):
        """极小的上限加倍后仍然太小，直接跳到一个够用的值。"""
        _result, calls = self._invoke(self._llm(max_tokens=1024), [
            _reply("", stop_reason="max_tokens"),
            _reply("完整正文"),
        ])
        assert calls[1]["max_tokens"] == 8_000

    def test_retry_budget_is_capped_at_the_ceiling(self):
        _result, calls = self._invoke(self._llm(max_tokens=40_000), [
            _reply("", stop_reason="max_tokens"),
            _reply("完整正文"),
        ])
        assert calls[1]["max_tokens"] == _TRUNCATION_RETRY_CEILING

    def test_no_retry_when_already_at_the_ceiling(self):
        """已经顶到天花板还被截断，就没有可加的预算了——不做无意义的重试。"""
        result, calls = self._invoke(
            self._llm(max_tokens=_TRUNCATION_RETRY_CEILING),
            [_reply("被截断的正文", stop_reason="max_tokens")],
        )
        assert len(calls) == 1
        assert result == "被截断的正文", "重试无望时返回已有内容，不能丢弃"

    def test_retry_happens_at_most_once(self):
        """重试后仍然截断也不再加倍——避免一个长提示词把预算推到天上。"""
        result, calls = self._invoke(self._llm(max_tokens=4096), [
            _reply("", stop_reason="max_tokens"),
            _reply("仍被截断", stop_reason="max_tokens"),
        ])
        assert len(calls) == 2
        assert result == "仍被截断"

    def test_per_call_max_tokens_override_is_the_retry_baseline(self):
        """调用点显式传了预算时，加倍要基于它，而不是实例上的值。"""
        llm = self._llm(max_tokens=4096)
        calls = []

        def fake_invoke(self, input, config=None, **kwargs):  # noqa: A002
            calls.append(kwargs)
            return _reply("ok") if calls[-1].get("max_tokens") != 2000 else _reply(
                "", stop_reason="max_tokens"
            )

        with patch.object(ac.ChatAnthropic, "invoke", fake_invoke):
            llm.invoke("prompt", max_tokens=2000)
        assert calls[1]["max_tokens"] == 8_000  # max(2 x 2000, 8000)
