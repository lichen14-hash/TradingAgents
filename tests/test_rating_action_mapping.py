"""评级-动作挂钩测试：评级标签由仓位变动幅度确定性映射。

背景：589130.SS 同日两次分析对同一个"减仓超配持仓"计划分别贴了
Hold / Underweight 标签（实质动作一致），根因是评级标签是模型自由
选择、与仓位动作无量化挂钩。本测试锁定映射规则与一致性校验行为。
规则：目标仓位相对当前仓位变动 ±20% 内 → Hold；减 20-60% →
Underweight；减 >60% → Sell；增仓对称（Overweight / Buy）；
空仓建仓 ≥3% → Buy，<3% → Overweight。
"""

import pytest

from tradingagents.agents.schemas import (
    PortfolioDecision,
    PortfolioRating,
    render_pm_decision,
)
from tradingagents.agents.utils.position_sizing import size_ceiling
from tradingagents.agents.utils.rating import (
    check_rating_action_consistency,
    expected_rating_for_position_change,
    parse_position_percents,
    parse_rating,
)


@pytest.mark.unit
class TestExpectedRatingMapping:
    @pytest.mark.parametrize("current,target,expected", [
        # 实际事故案例：13.16% 减到 7-8% (~43% 减幅) 必须是 Underweight
        (13.16, 7.5, "Underweight"),
        # 同事故 A 报告的计划：13.16% → 10% (~24% 减幅) 也应为 Underweight
        (13.16, 10.0, "Underweight"),
        # ±20% 内 → Hold
        (13.16, 12.0, "Hold"),
        (10.0, 10.0, "Hold"),
        (10.0, 11.9, "Hold"),
        # 减 >60% → Sell
        (13.0, 2.5, "Sell"),
        (10.0, 0.0, "Sell"),
        # 增仓
        (10.0, 13.0, "Overweight"),
        (10.0, 17.0, "Buy"),
        # 空仓建仓
        (0.0, 5.0, "Buy"),
        (0.0, 2.0, "Overweight"),
        (0.0, 0.0, "Hold"),
    ])
    def test_bands(self, current, target, expected):
        assert expected_rating_for_position_change(current, target) == expected

    def test_boundary_20_percent_reduction_is_underweight(self):
        # 恰好 -20% 落入 Underweight（闭区间下沿）
        assert expected_rating_for_position_change(10.0, 8.0) == "Underweight"

    def test_boundary_60_percent_reduction_is_sell(self):
        assert expected_rating_for_position_change(10.0, 4.0) == "Sell"

    def test_missing_context_returns_none(self):
        assert expected_rating_for_position_change(None, 5.0) is None
        assert expected_rating_for_position_change(5.0, None) is None
        assert expected_rating_for_position_change(None, None) is None

    def test_negative_values_unverifiable(self):
        assert expected_rating_for_position_change(-1.0, 5.0) is None


@pytest.mark.unit
class TestBandSymmetry:
    """加仓方向与减仓方向必须严格镜像。

    301 条历史 predictions 里 Buy=0、Sell=21，带持仓上下文时加仓:减仓 = 20:87
    （1:4.35），不带时 10:9（对称）。为了把"提示词单向锚定"与"映射本身偏向减仓"
    这两个嫌疑源分开，这里把映射的对称性钉死：同一比例的增减必须落在互为镜像的
    档位上，否则偏置就出在规则层而不是提示词层。
    """

    MIRROR = {"Sell": "Buy", "Underweight": "Overweight", "Hold": "Hold"}

    @pytest.mark.parametrize("current", [10.0, 13.16, 4.0])
    @pytest.mark.parametrize("fraction", [0.0, 0.05, 0.19, 0.20, 0.25, 0.43, 0.59, 0.60, 0.75, 1.0])
    def test_equal_magnitude_changes_map_to_mirror_tiers(self, current, fraction):
        down = expected_rating_for_position_change(current, current * (1 - fraction))
        up = expected_rating_for_position_change(current, current * (1 + fraction))
        assert self.MIRROR[down] == up, (
            f"{current}% 变动 ∓{fraction:.0%}：减仓={down} 加仓={up}，映射不对称"
        )

    @pytest.mark.parametrize("current,down_target,up_target,down,up", [
        # 恰好 ±20%（Hold 带边界）
        (10.0, 8.0, 12.0, "Underweight", "Overweight"),
        # 恰好 ±60%（重仓变动边界）
        (10.0, 4.0, 16.0, "Sell", "Buy"),
        # 事故案例的镜像：13.16% ∓43%
        (13.16, 7.5, 18.82, "Underweight", "Overweight"),
    ])
    def test_named_boundary_pairs(self, current, down_target, up_target, down, up):
        assert expected_rating_for_position_change(current, down_target) == down
        assert expected_rating_for_position_change(current, up_target) == up

    def test_entry_and_exit_are_the_asymmetric_pair_by_design(self):
        """建仓/清仓不是镜像：0→0 是"维持空仓"，必须是 Hold 而不是 Sell。

        Sell 档过去同时承担"清仓"与"不建仓"两个含义，是 Sell=21 而 Buy=0 的
        最可能来源之一——"不建仓"的仓位变动为零，语义上属于 Hold。
        """
        assert expected_rating_for_position_change(0.0, 0.0) == "Hold"
        assert expected_rating_for_position_change(5.0, 0.0) == "Sell"
        # 建仓侧按占组合比例分档，与减仓侧的相对比例分档不同源，故不参与镜像断言
        assert expected_rating_for_position_change(0.0, 3.0) == "Buy"
        assert expected_rating_for_position_change(0.0, 2.9) == "Overweight"


@pytest.mark.unit
class TestConsistencyCheck:
    def test_consistent_returns_none(self):
        assert check_rating_action_consistency("Underweight", 13.16, 7.5) is None

    def test_mismatch_returns_warning(self):
        # 事故场景：减仓计划贴 Hold 标签
        w = check_rating_action_consistency("Hold", 13.16, 7.5)
        assert w is not None
        assert w["section"] == "组合经理最终裁定"
        assert "Underweight" in w["reason"]

    def test_no_position_context_skips(self):
        assert check_rating_action_consistency("Hold", None, None) is None

    def test_case_insensitive_rating(self):
        assert check_rating_action_consistency("underweight", 13.16, 7.5) is None


@pytest.mark.unit
class TestRenderParseRoundTrip:
    """单票裁定里渲染的是**风险上限**，不是目标仓位。

    目标仓位需要知道另外 N−1 只在干什么，所以它由
    :func:`tradingagents.portfolio.allocator.allocate` 在整批跑完之后产出，
    再由报告层渲染。这样全系统每只票只存在**一个**仓位数字。
    `**Target Position**` 因此在新报告里不再出现——`parse_position_percents`
    保留下来只为读旧报告。
    """

    def test_ceiling_fields_render(self):
        # 300760.SZ 实测值：ATR 4.0723 / 收盘 150.63 → 上限 12.33%。
        ceiling = size_ceiling(4.0723, 150.63)
        assert ceiling.allowed_pct == 12.33
        decision = PortfolioDecision(
            portfolio_view=PortfolioRating.UNDERWEIGHT,
            executive_summary="跌破 145 止损；不加仓。",
            investment_thesis="集中度过高。",
        )
        text = render_pm_decision(decision, ceiling, current_pct=13.16)
        assert "**Current Position**: 13.16%" in text
        assert "**Risk Ceiling**: 12.33%" in text
        assert "3.0×ATR 2.7%" in text          # 推导过程可读
        assert "Target Position" not in text   # 目标仓位不在这一层

    def test_rating_line_is_the_view_not_a_position_delta(self):
        """`**Rating**` 必须是观点档。

        它曾短暂地变成仓位变动标签，与库里 327 行历史数据（`rating` = 观点）
        以及按 `rating` 分组的 `analytics.accuracy_by_rating()` 直接冲突：
        中性观点在超配仓位上正当地产生看空**标签**，计分卡却读成看空**观点**。
        """
        decision = PortfolioDecision(
            portfolio_view=PortfolioRating.HOLD,
            executive_summary="持有。",
            investment_thesis="估值合理。",
        )
        # 44% 远超 12.33% 上限——旧渲染会把这一行写成 Sell。
        text = render_pm_decision(decision, size_ceiling(4.0723, 150.63), current_pct=44.0)
        assert "**Rating**: Hold" in text
        assert parse_rating(text) == "Hold"

    def test_null_position_fields_omitted(self):
        """无持仓上下文时没有可移动的仓位，那一行不渲染——不是渲染成 0%。"""
        decision = PortfolioDecision(
            portfolio_view=PortfolioRating.HOLD,
            executive_summary="观望。",
            investment_thesis="无持仓上下文。",
        )
        for text in (
            render_pm_decision(decision),
            render_pm_decision(decision, size_ceiling(4.0723, 150.63)),
        ):
            assert "Current Position" not in text
            assert parse_position_percents(text) == (None, None)
