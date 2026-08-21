"""空输出守卫测试：LLM 返回空正文时必须留痕，且流程照常收敛。

事故背景（300760.SZ，2026-08-11）：报告 B 的 Bear Researcher 三轮全部返回
空内容，`bear_history` 全文只有 "Bear Analyst:\\nBear Analyst:\\nBear Analyst:"
（41 字符）。九个 `llm.invoke` 调用点当时都直接取 `.content`，空串被静默
吞掉；辩论 `count` 照常 +1，于是"单边辩论"被当成一场完整辩论，研究经理替
缺席的空方"代言"，Underweight 信号照样写进数据库，报告里没有任何提示。

本测试锁定三条契约：
1. 首轮空、重试有内容 → 返回内容，并留一条 warn；
2. 连续两次都空 → 返回显式占位文本 + 一条 severity="block"；
3. 空轮次仍然让 `count` 递增（守卫不得改变辩论收敛行为，否则会死循环）。
"""

from unittest.mock import MagicMock

import pytest

from tradingagents.agents.analysts.fundamentals_analyst import create_fundamentals_analyst
from tradingagents.agents.analysts.market_analyst import create_market_analyst
from tradingagents.agents.analysts.news_analyst import create_news_analyst
from tradingagents.agents.researchers.bear_researcher import create_bear_researcher
from tradingagents.agents.researchers.bull_researcher import create_bull_researcher
from tradingagents.agents.risk_mgmt.aggressive_debator import create_aggressive_debator
from tradingagents.agents.risk_mgmt.conservative_debator import create_conservative_debator
from tradingagents.agents.risk_mgmt.neutral_debator import create_neutral_debator
from tradingagents.agents.utils.integrity import (
    SEVERITY_BLOCK,
    SEVERITY_WARN,
    invoke_text_guarded,
    is_blank_output,
)

PLACEHOLDER_MARK = "本轮未产生有效输出"


def _llm(*contents):
    """LLM mock whose successive invoke() calls return the given contents."""
    llm = MagicMock()
    llm.invoke.side_effect = [MagicMock(content=c) for c in contents]
    return llm


def _base_state(**overrides) -> dict:
    state = {
        "company_of_interest": "300760.SZ",
        "trade_date": "2026-08-11",
        "asset_type": "stock",
        "market_report": "MA 均线下穿",
        "sentiment_report": "情绪中性",
        "news_report": "无重大新闻",
        "fundamentals_report": "毛利率稳定",
        "messages": [],
        "data_bundle": {"metadata": {"ticker": "300760.SZ", "trade_date": "2026-08-11"}},
    }
    state.update(overrides)
    return state


def _debate_state(count: int = 2) -> dict:
    return {
        "history": "Bull Analyst: 上涨逻辑",
        "bull_history": "Bull Analyst: 上涨逻辑",
        "bear_history": "",
        "current_response": "Bull Analyst: 上涨逻辑",
        "count": count,
    }


def _risk_state(count: int = 1) -> dict:
    return {
        "history": "",
        "aggressive_history": "",
        "conservative_history": "",
        "neutral_history": "",
        "latest_speaker": "",
        "current_aggressive_response": "",
        "current_conservative_response": "",
        "current_neutral_response": "",
        "count": count,
    }


@pytest.mark.unit
class TestBlankDetection:
    @pytest.mark.parametrize("text", ["", "   ", "\n\n", "**", "---", "  #  "])
    def test_blank_variants(self, text):
        assert is_blank_output(text)

    @pytest.mark.parametrize("text", ["空方认为估值过高", "0", "- 风险一"])
    def test_substantive_text_is_not_blank(self, text):
        assert not is_blank_output(text)

    def test_thinking_only_block_list_is_blank(self):
        """Claude 思考预算吃满时只回 thinking 块，归一化后为空串。"""
        response = MagicMock(content=[{"type": "thinking", "thinking": "推理很长……"}])
        llm = MagicMock()
        llm.invoke.return_value = response
        text, findings = invoke_text_guarded(
            llm, "p", section="多空辩论·空方", role="Bear Analyst",
        )
        assert PLACEHOLDER_MARK in text
        assert [f["severity"] for f in findings] == [SEVERITY_BLOCK]


@pytest.mark.unit
class TestInvokeTextGuarded:
    def test_retry_recovers_and_records_warning(self):
        llm = _llm("", "空方观点：产能过剩")
        text, findings = invoke_text_guarded(
            llm, "p", section="多空辩论·空方", role="Bear Analyst",
        )
        assert text == "空方观点：产能过剩"
        assert llm.invoke.call_count == 2
        assert len(findings) == 1
        assert findings[0]["severity"] == SEVERITY_WARN

    def test_all_attempts_blank_yields_placeholder_and_block(self):
        llm = _llm("", "")
        text, findings = invoke_text_guarded(
            llm, "p", section="多空辩论·空方", role="Bear Analyst",
        )
        assert PLACEHOLDER_MARK in text
        assert "Bear Analyst" in text
        assert len(findings) == 1
        assert findings[0]["severity"] == SEVERITY_BLOCK
        assert findings[0]["section"] == "多空辩论·空方"

    def test_clean_output_produces_no_findings(self):
        text, findings = invoke_text_guarded(
            _llm("空方观点"), "p", section="多空辩论·空方", role="Bear Analyst",
        )
        assert text == "空方观点"
        assert findings == []


@pytest.mark.unit
class TestDebateNodesRecordEmptyTurns:
    def test_bear_blank_turn_is_visible_and_count_advances(self):
        """事故点：空方三轮全空，报告与状态里都看不出来。"""
        node = create_bear_researcher(_llm("", ""))
        out = node(_base_state(investment_debate_state=_debate_state(count=2)))

        debate = out["investment_debate_state"]
        assert PLACEHOLDER_MARK in debate["bear_history"]
        # count 必须照常递增，否则 should_continue_debate 无法收敛
        assert debate["count"] == 3
        assert [f["severity"] for f in out["integrity_findings"]] == [SEVERITY_BLOCK]

    def test_bull_blank_turn_is_visible(self):
        node = create_bull_researcher(_llm("", ""))
        out = node(_base_state(investment_debate_state=_debate_state(count=1)))
        assert PLACEHOLDER_MARK in out["investment_debate_state"]["bull_history"]
        assert out["investment_debate_state"]["count"] == 2
        assert out["integrity_findings"][0]["severity"] == SEVERITY_BLOCK

    @pytest.mark.parametrize("factory,history_key", [
        (create_aggressive_debator, "aggressive_history"),
        (create_conservative_debator, "conservative_history"),
        (create_neutral_debator, "neutral_history"),
    ])
    def test_risk_debators_blank_turn_is_visible(self, factory, history_key):
        node = factory(_llm("", ""))
        out = node(_base_state(risk_debate_state=_risk_state(count=1),
                               investment_plan="持有",
                               trader_investment_plan="持有"))
        risk = out["risk_debate_state"]
        assert PLACEHOLDER_MARK in risk[history_key]
        assert risk["count"] == 2
        assert out["integrity_findings"][0]["severity"] == SEVERITY_BLOCK


@pytest.mark.unit
class TestAnalystNodesRecordEmptyReports:
    @pytest.mark.parametrize("factory,report_key", [
        (create_market_analyst, "market_report"),
        (create_news_analyst, "news_report"),
        (create_fundamentals_analyst, "fundamentals_report"),
    ])
    def test_blank_report_is_flagged(self, factory, report_key):
        node = factory(_llm("", ""))
        out = node(_base_state())
        assert PLACEHOLDER_MARK in out[report_key]
        assert out["integrity_findings"][0]["severity"] == SEVERITY_BLOCK
