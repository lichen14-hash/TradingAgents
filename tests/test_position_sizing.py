"""仓位测算器的单测——把"仓位由波动率算，不由浮亏算"这件事钉住。

背景见 :mod:`tradingagents.agents.utils.position_sizing` 的模块说明：改版前
目标仓位是模型自由填的数字，实测锚在用户浮亏上（深亏档 0.0% 加仓 / 71.0% 减仓，
浮盈档 23.8% / 19.0%，p=0.00012）。

这里的用例分三类：
1. **公式性质**——高波动必然给出更小的仓位；中性观点绝不加仓；空仓+中性维持空仓。
2. **可归因**——每个反直觉的结果都要能指到一个具名的 binding_constraint，
   且风险预算强制减仓时必须发 finding（否则报告里会读成"系统在看空"）。
3. **回归基准**——300760.SZ 的六行实测值，写死数字。这些数字来自真实 bundle
   （ATR 4.0723 / 收盘 150.63），改公式会让它们变，那时必须是有意识地改。
"""

from __future__ import annotations

import pytest

from tradingagents.agents.utils.position_sizing import (
    CONVICTION,
    DEFAULT_SIZING,
    METHOD_ATR,
    METHOD_FALLBACK,
    METHOD_NO_CONTEXT,
    RISK_BUDGET,
    SINGLE_NAME_CAP,
    STATUS_QUO,
    describe_sizing,
    size_position,
)

# 300760.SZ（迈瑞医疗）2026-08-11 的真实 bundle 取值。
# atr_pct = 4.0723 / 150.63 = 2.7035%；止损距离 = 3 × 2.7035 = 8.1105%；
# allowed = 1.0 / 8.1105 × 100 = 12.33%。
ATR = 4.0723
CLOSE = 150.63
ALLOWED = 12.33

# 688507.SS：52 份 bundle 里波动最高的一只（ATR% 13.05）——同一套参数下
# allowed 只有 2.55%，是"高波动 → 小仓位"这条性质的实测端点。
HIGH_VOL_ATR = 8.6
HIGH_VOL_CLOSE = 65.9


@pytest.mark.unit
class TestFormulaProperties:
    def test_higher_volatility_yields_a_smaller_ceiling(self):
        """同一观点、同一当前仓位，波动越大目标仓位越小。这是整个改版的目的。"""
        calm = size_position("Buy", 0.0, atr=1.0, close=100.0)
        wild = size_position("Buy", 0.0, atr=8.0, close=100.0)
        assert calm.allowed_pct > wild.allowed_pct
        assert calm.target_pct > wild.target_pct

    def test_ceiling_is_inversely_proportional_to_stop_distance(self):
        """止损距离翻倍，上限减半——公式没有被别的项污染。"""
        one = size_position("Hold", 50.0, atr=2.0, close=200.0)   # atr_pct 1.0
        two = size_position("Hold", 50.0, atr=4.0, close=200.0)   # atr_pct 2.0
        # 1.0% ATR → 止损 3% → 33.3%，被 15% 单票上限截断，所以用更高的两档比。
        assert one.allowed_pct == pytest.approx(2 * two.allowed_pct, rel=1e-6) or (
            one.binding_constraint == SINGLE_NAME_CAP
        )

    def test_neutral_view_never_adds(self):
        """中性观点只做风险校正（min），任何情况下不会比当前仓位更高。"""
        for current in (0.0, 1.0, 5.0, 12.0, 40.0):
            plan = size_position("Hold", current, atr=ATR, close=CLOSE)
            assert plan.target_pct <= current

    def test_flat_position_stays_flat_on_a_neutral_view(self):
        """空仓 + 中性 = 空仓。单步 conviction×allowed 会在这里凭空开仓。"""
        plan = size_position("Hold", 0.0, atr=ATR, close=CLOSE)
        assert plan.target_pct == 0.0
        assert plan.rating == "Hold"

    def test_bearish_view_reduces_below_the_risk_corrected_weight(self):
        """看空档在风险校正之后再缩，而不是取代它。"""
        under = size_position("Underweight", 5.0, atr=ATR, close=CLOSE)
        sell = size_position("Sell", 5.0, atr=ATR, close=CLOSE)
        assert under.target_pct == pytest.approx(5.0 * DEFAULT_SIZING["trim"]["Underweight"])
        assert sell.target_pct == 0.0

    def test_bullish_view_never_reduces_a_within_budget_position(self):
        """看多档取 max：已在预算内的仓位不会因为"看多"反而被削。"""
        plan = size_position("Overweight", ALLOWED - 0.5, atr=ATR, close=CLOSE)
        assert plan.target_pct >= ALLOWED - 0.5

    def test_single_name_cap_binds_on_a_very_calm_stock(self):
        """波动极低时风险预算允许的仓位超过单票上限，此时上限接管并具名。"""
        plan = size_position("Buy", 0.0, atr=2.0, close=100.0)  # atr_pct 2% → 止损 6%
        assert plan.allowed_pct == DEFAULT_SIZING["max_single_name_pct"]
        plan_over = size_position("Hold", 20.0, atr=2.0, close=100.0)
        assert plan_over.binding_constraint == SINGLE_NAME_CAP


@pytest.mark.unit
class TestAttribution:
    def test_forced_trim_emits_a_finding_naming_the_risk_budget(self):
        """中性观点 + 超预算仓位 → 减仓。报告必须说清这不是看空。"""
        plan = size_position("Hold", 44.0, atr=ATR, close=CLOSE)
        assert plan.binding_constraint == RISK_BUDGET
        assert len(plan.findings) == 1
        reason = plan.findings[0]["reason"]
        assert "风险预算" in reason
        assert "观点本身" in reason
        assert plan.findings[0]["severity"] == "warn"

    def test_within_budget_plan_emits_no_findings(self):
        plan = size_position("Buy", 5.0, atr=ATR, close=CLOSE)
        assert plan.findings == []
        # 5% → 12.33%：加到了上限本身，所以定住这个数字的是风险预算，不是观点档。
        assert plan.binding_constraint == RISK_BUDGET

    def test_missing_atr_falls_back_and_says_so(self):
        plan = size_position("Buy", 0.0, atr=None, close=CLOSE)
        assert plan.method == METHOD_FALLBACK
        assert plan.atr_pct is None
        # 1.0 / 12.0 * 100
        assert plan.allowed_pct == 8.33
        assert len(plan.findings) == 1
        assert "ATR" in plan.findings[0]["reason"]

    def test_zero_close_is_treated_as_missing_not_as_a_division_by_zero(self):
        plan = size_position("Buy", 0.0, atr=ATR, close=0.0)
        assert plan.method == METHOD_FALLBACK

    def test_no_position_context_yields_the_view_tier_and_no_number(self):
        """未提供持仓时没有可移动的仓位，只回观点档。"""
        plan = size_position("Buy", None, atr=ATR, close=CLOSE)
        assert plan.method == METHOD_NO_CONTEXT
        assert plan.rating == "Buy"
        assert plan.target_pct is None
        assert "未提供持仓上下文" in describe_sizing(plan)

    def test_unparseable_view_is_treated_as_neutral_with_a_finding(self):
        plan = size_position("赶紧买", 10.0, atr=ATR, close=CLOSE)
        assert plan.view == "Hold"
        assert any("观点档不可解析" in f["reason"] for f in plan.findings)
        assert plan.target_pct <= 10.0

    def test_description_shows_the_arithmetic(self):
        """44% → 12.33% 在中性观点下必须读起来像算术，而不像系统自相矛盾。"""
        text = describe_sizing(size_position("Hold", 44.0, atr=ATR, close=CLOSE))
        assert "风险预算 1.0%" in text
        assert "ATR 2.7%" in text
        assert f"{ALLOWED}%" in text
        assert "风险预算" in text

    def test_config_overrides_are_honoured(self):
        # 上限一起提高，否则 24.66% 会被 15% 的单票上限截断，测不到风险预算这一项。
        doubled = size_position(
            "Buy", 0.0, atr=ATR, close=CLOSE,
            config={"position_sizing": {
                "risk_budget_pct": 2.0, "max_single_name_pct": 40.0,
            }},
        )
        assert doubled.allowed_pct == pytest.approx(2 * ALLOWED, abs=0.01)
        assert doubled.binding_constraint == RISK_BUDGET

    def test_partial_config_block_keeps_the_other_defaults(self):
        plan = size_position(
            "Underweight", 5.0, atr=ATR, close=CLOSE,
            config={"position_sizing": {"trim": {"Underweight": 0.5}}},
        )
        assert plan.target_pct == pytest.approx(2.5)
        assert plan.allowed_pct == ALLOWED  # risk_budget_pct 未被覆盖


@pytest.mark.unit
class TestRegression300760:
    """六行写死的实测基准。第一行 7.4% 与模型当初自己写的数字几乎一致——
    公式不是在推翻模型的合理判断，只是把浮亏驱动的那部分拿掉了。"""

    # ``binding_constraint`` answers "which rule set this number":
    # target == current → status_quo; target == the ceiling → whichever of
    # risk_budget / single_name_cap produced that ceiling; anything strictly in
    # between → the view's own multiplier (conviction). The separate finding is
    # what discloses a ceiling-driven reduction of the *base*, so row 1 reads
    # "conviction set 7.40 = 0.6 × 12.33, and the 12.33 came from the budget".
    CASES = [
        # view,          current, target, rating,        binding,      findings
        ("Underweight",   13.16,   7.40,  "Underweight", CONVICTION,   1),
        ("Hold",          44.00,  12.33,  "Sell",        RISK_BUDGET,  1),
        # Buy takes the full ceiling, so relaxing the risk budget is the only way
        # to get a bigger number — the conviction multiplier is already 1.0.
        ("Buy",            0.00,  12.33,  "Buy",         RISK_BUDGET,  0),
        # Flat + neutral: no constraint was active at all. Reporting this as
        # "观点档" claimed the view chose to stay out; it did not choose anything.
        ("Hold",           0.00,   0.00,  "Hold",        STATUS_QUO,   0),
        ("Overweight",    13.16,  12.33,  "Hold",        RISK_BUDGET,  1),
        # Sell takes it to zero on its own. The budget contributed nothing, so
        # unlike the rows above this one must NOT emit the risk-budget finding.
        ("Sell",          13.16,   0.00,  "Sell",        CONVICTION,   0),
    ]

    @pytest.mark.parametrize("view,current,target,rating,binding,n_findings", CASES)
    def test_case(self, view, current, target, rating, binding, n_findings):
        plan = size_position(view, current, atr=ATR, close=CLOSE)
        assert plan.method == METHOD_ATR
        assert plan.allowed_pct == ALLOWED
        assert plan.target_pct == target
        assert plan.rating == rating
        assert plan.binding_constraint == binding
        assert len(plan.findings) == n_findings

    def test_neutral_view_can_legitimately_produce_a_bearish_label(self):
        """语义碰撞的关键案例：存储的 rating 是仓位变动，不是观点。

        跨阶段一致性检查因此必须比 ``portfolio_view``，不能比 ``rating``
        （见 :func:`tradingagents.agents.utils.integrity.check_cross_stage_divergence`）。
        """
        plan = size_position("Hold", 44.0, atr=ATR, close=CLOSE)
        assert plan.view == "Hold"
        assert plan.rating == "Sell"

    def test_plan_is_json_serialisable_for_the_db(self):
        import json

        payload = size_position("Hold", 44.0, atr=ATR, close=CLOSE).as_dict()
        assert json.loads(json.dumps(payload, ensure_ascii=False))["method"] == METHOD_ATR
        assert payload["binding_constraint"] == RISK_BUDGET


@pytest.mark.unit
class TestHighVolatility:
    def test_full_entry_on_a_high_vol_name_is_still_labelled_buy(self):
        """高波动标的 allowed 仅 2.6%，固定 3.0% 的入场阈值会把满额建仓错标成 Overweight。

        这就是 ``expected_rating_for_position_change`` 需要 ``allowed_pct`` 的实测理由。
        """
        plan = size_position("Buy", 0.0, atr=HIGH_VOL_ATR, close=HIGH_VOL_CLOSE)
        assert plan.allowed_pct < 3.0
        assert plan.rating == "Buy"

    def test_cost_price_is_not_a_parameter(self):
        """构造性证明的一半：这个函数的签名里没有成本价可传。

        另一半在 tests/test_position_context_isolation.py（提示词里也没有）。
        两者合起来，浮亏梯度按构造不可能再出现。
        """
        import inspect

        params = set(inspect.signature(size_position).parameters)
        assert params == {"view", "current_pct", "atr", "close", "config"}
