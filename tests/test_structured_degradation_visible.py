"""结构化输出降级必须可见（不能只写一行日志）。

事故背景（300760.SZ，2026-08-11）：两次运行的研究经理都退化成自由文本
——两份 HTML 里 `**Recommendation**` 出现 0 次。`invoke_structured_or_freetext`
当时只打一条 `logger.warning` 就静默 fallback，于是报告 B 根本没有机器可读的
研究经理评级，系统连"组合经理越过了研究经理"这件事都无法识别（`portfolio_rating`
有值、`research_recommendation` 为空，跨阶段核对无从下手）。

本测试锁定三条契约：
1. provider 不支持 `with_structured_output`（绑定期失败）→ findings 里有 warn；
2. 结构化调用本身抛异常（弱模型吐坏 JSON）→ findings 里有 warn，且 reason
   带上异常类型，便于区分"能力缺失"与"偶发失败"；
3. `parsed is None` 时调用方必须退回 `parse_rating` 解析正文，评级不能丢；
   若正文也解析不出评级，则该缺失本身会在对账节点被记录（见
   `tests/test_cross_stage_divergence.py`）。
"""

from unittest.mock import MagicMock

import pytest

from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.agents.managers.research_manager import create_research_manager
from tradingagents.agents.schemas import (
    PortfolioRating,
    ResearchPlan,
    render_research_plan,
)
from tradingagents.agents.trader.trader import create_trader
from tradingagents.agents.utils.integrity import SEVERITY_BLOCK, SEVERITY_WARN
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_guarded,
)

DEGRADED_MARK = "结构化输出降级为自由文本"
PLACEHOLDER_MARK = "本轮未产生有效输出"


def _unsupported_llm(plain_text: str):
    """Provider without structured-output support (older Ollama models)."""
    llm = MagicMock()
    llm.with_structured_output.side_effect = NotImplementedError("provider unsupported")
    llm.invoke.return_value = MagicMock(content=plain_text)
    return llm


def _failing_structured_llm(plain_text: str, exc: Exception | None = None):
    """Structured binding succeeds, but the structured call itself blows up."""
    structured = MagicMock()
    structured.invoke.side_effect = exc or ValueError("bad JSON from model")
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    llm.invoke.return_value = MagicMock(content=plain_text)
    return llm


def _make_rm_state() -> dict:
    return {
        "company_of_interest": "300760.SZ",
        "investment_debate_state": {
            "history": "Bull Analyst: 集采出清\nBear Analyst: 增速下滑",
            "bull_history": "Bull Analyst: 集采出清",
            "bear_history": "Bear Analyst: 增速下滑",
            "current_response": "",
            "judge_decision": "",
            "count": 2,
        },
    }


def _make_trader_state() -> dict:
    return {
        "company_of_interest": "300760.SZ",
        "investment_plan": "**Recommendation**: Hold\n**Rationale**: ...",
    }


def _make_pm_state() -> dict:
    return {
        "company_of_interest": "300760.SZ",
        "investment_plan": "**Recommendation**: Hold\n**Rationale**: ...",
        "trader_investment_plan": "**Action**: Hold\n**Reasoning**: ...",
        "risk_debate_state": {
            "history": "Risky Analyst: 反弹在即\nSafe Analyst: 趋势未破",
            "aggressive_history": "Risky Analyst: 反弹在即",
            "conservative_history": "Safe Analyst: 趋势未破",
            "neutral_history": "",
            "latest_speaker": "Safe Analyst",
            "current_aggressive_response": "",
            "current_conservative_response": "",
            "current_neutral_response": "",
            "count": 3,
        },
    }


def _warns(findings: list[dict]) -> list[dict]:
    return [f for f in findings if f.get("severity") == SEVERITY_WARN]


def _degradation_warns(findings: list[dict]) -> list[dict]:
    """只取"结构化降级"这一类留痕。

    同一个节点合法地产出别的 warn：组合经理的 state 里没有行情 bundle，于是风险上限
    走兜底止损并报一条「止损距离按兜底值 12% 计算」。那条与本文件锁的契约无关，
    按标记筛掉——但不能改成"至少有一条"，否则"三级各留一条、不许互相顶替"这条
    契约（事故里丢的正是前两条）就测不出来了。
    """
    return [f for f in _warns(findings) if DEGRADED_MARK in f.get("reason", "")]


@pytest.mark.unit
class TestInvokeStructuredGuarded:
    def test_binding_failure_yields_none_binding_and_a_warn_finding(self):
        llm = _unsupported_llm("**Recommendation**: Sell\n**Rationale**: 空方胜出")
        structured = bind_structured(llm, ResearchPlan, "Research Manager")
        assert structured is None, "不支持时 bind_structured 必须返回 None 而不是抛出"

        text, parsed, findings = invoke_structured_guarded(
            structured, llm, "p", render_research_plan,
            "Research Manager", section="研究经理裁定",
        )
        assert text == "**Recommendation**: Sell\n**Rationale**: 空方胜出"
        assert parsed is None
        assert len(findings) == 1
        assert findings[0]["severity"] == SEVERITY_WARN
        assert findings[0]["section"] == "研究经理裁定"
        assert DEGRADED_MARK in findings[0]["reason"]
        assert "with_structured_output" in findings[0]["reason"]

    def test_structured_call_failure_records_the_exception_type(self):
        """区分"provider 不支持"与"这次调用坏了"——排查手段完全不同。"""
        llm = _failing_structured_llm("**Recommendation**: Hold\n**Rationale**: ...")
        structured = bind_structured(llm, ResearchPlan, "Research Manager")
        _text, parsed, findings = invoke_structured_guarded(
            structured, llm, "p", render_research_plan,
            "Research Manager", section="研究经理裁定",
        )
        assert parsed is None
        assert len(findings) == 1
        assert findings[0]["severity"] == SEVERITY_WARN
        assert "ValueError" in findings[0]["reason"]
        assert "bad JSON from model" in findings[0]["reason"]

    def test_successful_structured_call_leaves_no_findings(self):
        plan = ResearchPlan(
            recommendation=PortfolioRating.OVERWEIGHT,
            rationale="多方论据更强。",
            strategic_actions="收盘站上 157.5 上方再确认。",
        )
        structured = MagicMock()
        structured.invoke.return_value = plan
        llm = MagicMock()
        llm.with_structured_output.return_value = structured

        text, parsed, findings = invoke_structured_guarded(
            bind_structured(llm, ResearchPlan, "Research Manager"),
            llm, "p", render_research_plan,
            "Research Manager", section="研究经理裁定",
        )
        assert findings == []
        assert parsed is plan
        assert "**Recommendation**: Overweight" in text

    def test_degraded_and_blank_stacks_both_findings(self):
        """降级 + 自由文本也返回空：两条问题都要留痕，不能互相掩盖。"""
        llm = MagicMock()
        llm.with_structured_output.side_effect = NotImplementedError("provider unsupported")
        llm.invoke.side_effect = [MagicMock(content=""), MagicMock(content="")]

        text, parsed, findings = invoke_structured_guarded(
            None, llm, "p", render_research_plan,
            "Research Manager", section="研究经理裁定",
        )
        assert parsed is None
        assert PLACEHOLDER_MARK in text
        assert [f["severity"] for f in findings] == [SEVERITY_WARN, SEVERITY_BLOCK]


@pytest.mark.unit
class TestDecisionAgentsSurfaceDegradation:
    def test_research_manager_degradation_is_on_the_state(self):
        """事故点本身：研究经理降级后，评级从正文兜底解析，降级留痕在 state 上。"""
        llm = _unsupported_llm("**Recommendation**: Underweight\n**Rationale**: 空方更强")
        out = create_research_manager(llm)(_make_rm_state())

        assert out["research_recommendation"] == "Underweight", "降级也不能丢评级"
        warns = _warns(out["integrity_findings"])
        assert len(warns) == 1
        assert DEGRADED_MARK in warns[0]["reason"]
        assert "Research Manager" in warns[0]["reason"]

    def test_research_manager_unreadable_rating_leaves_it_empty_for_reconciliation(self):
        """正文里没有评级行时评级为空——由对账节点把"缺失"本身记为 finding。"""
        llm = _unsupported_llm("空方的论据更充分，但没有给出明确评级标签。")
        out = create_research_manager(llm)(_make_rm_state())
        assert out["research_recommendation"] == ""
        assert _warns(out["integrity_findings"]), "评级不可读时至少要有降级留痕"

    def test_trader_degradation_is_on_the_state(self):
        llm = _failing_structured_llm(
            "**Action**: Hold\n\n持有观望。\n\nFINAL TRANSACTION PROPOSAL: **HOLD**"
        )
        out = create_trader(llm)(_make_trader_state())
        assert out["trader_direction"] == "Hold"
        warns = _warns(out["integrity_findings"])
        assert len(warns) == 1
        assert warns[0]["section"] == "交易员提案"

    def test_portfolio_manager_degradation_is_on_the_state(self):
        llm = _failing_structured_llm(
            "**Rating**: Underweight\n**Current Position**: 13.16%\n**Target Position**: 7.5%"
        )
        out = create_portfolio_manager(llm)(_make_pm_state())
        assert out["portfolio_rating"] == "Underweight"
        warns = _degradation_warns(out["integrity_findings"])
        assert len(warns) == 1
        assert warns[0]["section"] == "组合经理最终裁定"

    def test_all_three_stages_degrading_accumulates_three_findings(self):
        """完整事故形态：三级全部降级时，findings 应各留一条，不能只剩最后一条。"""
        rm = create_research_manager(
            _unsupported_llm("**Recommendation**: Hold\n**Rationale**: ...")
        )(_make_rm_state())
        trader = create_trader(
            _unsupported_llm("**Action**: Hold\n\nFINAL TRANSACTION PROPOSAL: **HOLD**")
        )(_make_trader_state())
        pm = create_portfolio_manager(
            _unsupported_llm("**Rating**: Underweight\n**Current Position**: 13.16%")
        )(_make_pm_state())

        # operator.add reducer 在图里做的事，这里手工拼一次
        merged = (
            rm["integrity_findings"] + trader["integrity_findings"] + pm["integrity_findings"]
        )
        sections = [f["section"] for f in _degradation_warns(merged)]
        assert sections == ["研究经理裁定", "交易员提案", "组合经理最终裁定"]
