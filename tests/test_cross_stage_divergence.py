"""跨阶段一致性测试：组合经理越过研究经理/交易员时必须留痕。

事故背景（300760.SZ，2026-08-11）：研究经理 Hold、交易员 Hold、组合经理
Underweight，系统里没有任何一处记录"越过"这件事——图在
`Portfolio Manager → END` 直接结束，没有对账环节；数据库只存组合经理的评级。

关键语义（本测试的核心约束）：两级评级用同一套标签词，含义却不同——
研究经理是**观点制**（证据偏向），组合经理是**仓位变动制**（目标仓位相对
当前仓位的变化）。所以"中性观点 + 减仓 40%"是合法的（属仓位管理），
只有**方向相反**才算分歧。若按标签字面比较，会把大量正常组合误判为分歧。
"""

import pytest

from tradingagents.agents.utils.integrity import (
    SEVERITY_BLOCK,
    SEVERITY_WARN,
    check_conviction_vs_data_quality,
    check_cross_stage_divergence,
    data_quality_caveat,
    rating_direction,
)
from tradingagents.graph.reconciliation import create_decision_reconciliation

INCOMPLETE = "intraday_daily_bar_incomplete"


def _reasons(findings):
    return " | ".join(f["reason"] for f in findings)


@pytest.mark.unit
class TestRatingDirection:
    @pytest.mark.parametrize("rating,expected", [
        ("Buy", 1), ("Overweight", 1), ("Hold", 0),
        ("Underweight", -1), ("Sell", -1), ("underweight", -1),
    ])
    def test_direction_mapping(self, rating, expected):
        assert rating_direction(rating) == expected

    @pytest.mark.parametrize("rating", ["", None, "Maybe", "强烈推荐"])
    def test_unparseable_is_none(self, rating):
        assert rating_direction(rating) is None


@pytest.mark.unit
class TestCrossStageDivergence:
    def test_neutral_view_plus_position_trim_is_not_divergence(self):
        """事故场景的合法一半：观点中性 + 减仓是仓位管理，不是分歧。"""
        findings = check_cross_stage_divergence("Hold", "Hold", "Underweight")
        assert findings == []

    def test_neutral_view_plus_position_add_is_not_divergence(self):
        assert check_cross_stage_divergence("Hold", "Hold", "Overweight") == []

    def test_all_three_aligned_is_clean(self):
        assert check_cross_stage_divergence("Overweight", "Buy", "Buy") == []

    def test_bullish_view_versus_sell_is_flagged(self):
        findings = check_cross_stage_divergence("Overweight", "Buy", "Sell")
        assert findings, "看多观点 + 清仓必须留痕"
        assert all(f["section"] == "跨阶段一致性" for f in findings)
        assert "研究经理" in _reasons(findings)

    def test_extreme_distance_is_block_severity(self):
        findings = check_cross_stage_divergence("Buy", "Buy", "Sell")
        assert any(f["severity"] == SEVERITY_BLOCK for f in findings)

    def test_missing_research_rating_is_itself_a_finding(self):
        """结构化降级导致研究经理评级不可读时，越过与否无法核对——本身即为问题。"""
        findings = check_cross_stage_divergence("", "Hold", "Underweight")
        assert len(findings) == 1
        assert findings[0]["severity"] == SEVERITY_WARN
        assert "研究经理评级缺失" in findings[0]["reason"]

    def test_missing_pm_rating_is_flagged(self):
        findings = check_cross_stage_divergence("Hold", "Hold", "")
        assert any("组合经理评级缺失" in f["reason"] for f in findings)


@pytest.mark.unit
class TestConvictionVersusDataQuality:
    def test_high_conviction_on_incomplete_bar_is_flagged(self):
        findings = check_conviction_vs_data_quality("Sell", INCOMPLETE)
        assert len(findings) == 1
        assert findings[0]["severity"] == SEVERITY_WARN

    @pytest.mark.parametrize("rating", ["Hold", "Underweight", "Overweight"])
    def test_graduated_ratings_are_allowed(self, rating):
        assert check_conviction_vs_data_quality(rating, INCOMPLETE) == []

    def test_clean_data_never_flags(self):
        assert check_conviction_vs_data_quality("Sell", "") == []

    def test_caveat_only_injected_when_bar_incomplete(self):
        clean = {"data_bundle": {"metadata": {"date_correction_reason": ""}}}
        dirty = {"data_bundle": {"metadata": {"date_correction_reason": INCOMPLETE}}}
        assert data_quality_caveat(clean) == ""
        assert "last *complete* daily bar" in data_quality_caveat(dirty)


@pytest.mark.unit
class TestReconciliationNode:
    def _state(self, **overrides) -> dict:
        state = {
            "company_of_interest": "300760.SZ",
            "final_trade_decision": (
                "**Rating**: Underweight\n"
                "**Current Position**: 13.16%\n**Target Position**: 7.5%\n"
            ),
            "research_recommendation": "Hold",
            "trader_direction": "Hold",
            "portfolio_rating": "Underweight",
            "data_bundle": {"metadata": {"date_correction_reason": ""}},
        }
        state.update(overrides)
        return state

    def test_incident_shape_is_coherent_and_stays_clean(self):
        """Hold/Hold/Underweight + 一致的仓位计划：不该造假警报。"""
        out = create_decision_reconciliation()(self._state())
        assert out["portfolio_rating"] == "Underweight"
        assert out["integrity_findings"] == []

    def test_label_vs_position_plan_is_no_longer_checked_in_the_graph(self):
        """评级 ↔ 仓位动作的一致性检查**已经搬走**，这里不再报。

        原因不是放弃了这项检查，而是图内已经没有目标仓位可比：单票裁定只带风险
        上限，目标仓位需要知道另外 N−1 只在干什么，由
        :func:`tradingagents.portfolio.allocator.allocate` 产出，那一层自己用
        同一个函数推标签、自校验（见 ``tests/test_allocator.py``）。

        这条用例保留下来，是为了让「图内不再报」是一个被写下来的决定，而不是
        某次改动顺手漏掉的检查。
        """
        out = create_decision_reconciliation()(self._state(
            final_trade_decision=(
                "**Rating**: Hold\n"
                "**Current Position**: 13.16%\n**Target Position**: 7.5%\n"
            ),
            portfolio_rating="Hold",
        ))
        assert not any(
            f["section"] == "组合经理最终裁定" for f in out["integrity_findings"]
        )

    def test_opposite_directions_are_flagged(self):
        out = create_decision_reconciliation()(self._state(
            research_recommendation="Overweight",
            trader_direction="Buy",
            portfolio_rating="Sell",
            final_trade_decision="**Rating**: Sell\n",
        ))
        assert any(f["section"] == "跨阶段一致性" for f in out["integrity_findings"])

    def test_pm_rating_falls_back_to_parsing_the_markdown(self):
        """结构化降级时 portfolio_rating 为空，仍须从渲染文本里解析出评级。"""
        out = create_decision_reconciliation()(self._state(portfolio_rating=""))
        assert out["portfolio_rating"] == "Underweight"

    def test_high_conviction_on_incomplete_bar_is_flagged(self):
        out = create_decision_reconciliation()(self._state(
            final_trade_decision="**Rating**: Sell\n",
            portfolio_rating="Sell",
            research_recommendation="Underweight",
            trader_direction="Sell",
            data_bundle={"metadata": {"date_correction_reason": INCOMPLETE}},
        ))
        assert any(f["section"] == "数据质量与置信度" for f in out["integrity_findings"])

    def test_node_never_rewrites_the_decision(self):
        state = self._state(research_recommendation="Buy", portfolio_rating="Sell",
                            final_trade_decision="**Rating**: Sell\n")
        out = create_decision_reconciliation()(state)
        assert "final_trade_decision" not in out
        assert out["portfolio_rating"] == "Sell"
