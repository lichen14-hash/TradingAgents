"""组合分配层的单测——把「搬到图外没有偷偷改变任何数字」这件事钉住。

三类用例：

1. **回归基准（最重要）**——2026-08-20 那批 6 只的真实
   `(view, current_pct, ceiling)` 全部写死在 :data:`BATCH_0820` 里。
   `allocate` 逐票跑出来的 `target_pct` 必须等于当时库里 `position_sizing` 存下的
   数字（7.37 / 9.00 / 5.45 / 5.22 / 3.96），组合合计必须是 Σcurrent 79.92% →
   Σtarget 31.00%。搬家如果动了数字，这里立刻红。
2. **预算与配给**——减仓无条件执行；加仓只能花 `cash_pct + Σ减仓释放`
   这一笔算术额度，不够就按观点档序 + 同档按比例配给，零碎归零。
   额度**没有**总权益硬顶、**没有**现金底线，所以 Σtarget 允许到 100%——
   这是明确决定，:class:`TestRationing` 里有用例钉住，防止日后被「顺手」加回一个 `min()`。
3. **不抛异常**——分析失败、缺 ceiling、无持仓上下文、脏输入，全部要降级成
   「维持现状 + finding」，因为这一层跑在 N 只票之后，抛一次就毁掉整批报告。
"""

from __future__ import annotations

import pytest

from tradingagents.agents.utils.position_sizing import (
    CONVICTION,
    METHOD_ATR,
    RISK_BUDGET,
    STATUS_QUO,
    Ceiling,
    constraint_label,
    size_ceiling,
    size_position,
)
from tradingagents.portfolio.allocator import (
    PORTFOLIO_BUDGET,
    STATUS_FAILED,
    STATUS_OK,
    Allocation,
    Holding,
    allocate,
    describe_allocation,
    holding_from_ceiling_dict,
)


def ceiling(allowed: float, atr_pct: float | None = 3.0,
            binding: str = RISK_BUDGET) -> Ceiling:
    """A Ceiling with the shape the graph produces (stop distance = 3×ATR%)."""
    return Ceiling(
        allowed_pct=allowed,
        stop_distance_pct=round(3.0 * atr_pct, 4) if atr_pct else 12.0,
        method=METHOD_ATR,
        binding_constraint=binding,
        atr_pct=atr_pct,
    )


# 2026-08-20 的真实批次。ticker, current_pct, view, allowed_pct, atr_pct,
# 期望 target_pct, 期望 binding_constraint。
# allowed/atr 取自当时存进 predictions.position_sizing 的值，target 也是。
BATCH_0820 = [
    ("002602.SZ", 49.00, "Hold",        7.37,  4.5198,  7.37, RISK_BUDGET),
    ("09988.HK",  None,  "Hold",        None,  None,    None, CONVICTION),
    ("300760.SZ",  9.00, "Hold",       11.90,  2.8016,  9.00, STATUS_QUO),
    ("515880.SS",  6.32, "Hold",        5.45,  6.1165,  5.45, RISK_BUDGET),
    ("589130.SS",  9.00, "Hold",        5.22,  6.3875,  5.22, RISK_BUDGET),
    ("600320.SS",  6.60, "Underweight", 13.86, 2.4045,  3.96, CONVICTION),
]

SIGMA_CURRENT_0820 = 79.92
SIGMA_TARGET_0820 = 31.00


def batch_0820_holdings() -> list[Holding]:
    return [
        Holding(
            ticker=tk,
            current_pct=cur,
            view=view,
            # 09988.HK 那行当时走的是 no_position_context：没有仓位可动。
            # 但它的 ceiling 与仓位无关，真实运行里照样算得出来，所以这里给它一个。
            ceiling=ceiling(allowed, atr) if allowed else ceiling(8.0, 4.0),
            name=tk,
        )
        for tk, cur, view, allowed, atr, _target, _binding in BATCH_0820
    ]


@pytest.mark.unit
class TestRegressionBatch0820:
    """真实批次逐票 + 合计的写死基准。"""

    @pytest.mark.parametrize(
        "ticker,current,view,allowed,atr,target,binding",
        BATCH_0820,
        ids=[row[0] for row in BATCH_0820],
    )
    def test_target_matches_the_stored_number(
        self, ticker, current, view, allowed, atr, target, binding,
    ):
        item = allocate(batch_0820_holdings()).by_ticker(ticker)
        assert item is not None
        assert item.target_pct == target
        assert item.binding_constraint == binding

    def test_portfolio_totals(self):
        result = allocate(batch_0820_holdings())
        assert result.sigma_current == SIGMA_CURRENT_0820
        assert result.sigma_target == SIGMA_TARGET_0820
        assert result.cash_pct == pytest.approx(100 - SIGMA_CURRENT_0820, abs=0.01)
        assert result.frozen_pct == 0.0

    def test_this_batch_proposes_no_adds_at_all(self):
        """6 只全是 Hold/Underweight——出现任何加仓就是算错了。"""
        result = allocate(batch_0820_holdings())
        for item in result.items:
            if item.current_pct is None or item.target_pct is None:
                continue
            assert item.target_pct <= item.current_pct
        assert not any("加仓" in f["reason"] for f in result.findings)

    def test_concentration_is_reported(self):
        """这一层存在的理由：49% 压在一只票上，其风险上限只有 7.37%。

        改版前这句话在系统的任何一份报告里都找不到——没有代码同时看得见 6 只。
        """
        result = allocate(batch_0820_holdings())
        text = " ".join(f["reason"] for f in result.findings)
        assert "002602.SZ" in text
        assert "49.0%" in text
        assert "7.37%" in text

    def test_total_equity_warning_does_not_fire_on_this_batch(self):
        """79.92% 在 90% 的总权益线以下——这批的问题是集中度，不是满仓。

        写下来是为了区分两件容易混起来的事：Σcurrent 高（这批不算高）和
        单只占比高（这批 49%，很高）。阈值被跨过时的行为见
        :class:`TestConfigOverrides`。
        """
        result = allocate(batch_0820_holdings())
        assert SIGMA_CURRENT_0820 < 90.0
        assert not any("总权益上限" in f["reason"] for f in result.findings)


@pytest.mark.unit
class TestDegenerateToSizePosition:
    """N=1 必须与图内的 `size_position` 逐字段一致。

    两者共用 `size_from_ceiling`，所以这是构造性的；用例仍然写出来，因为第二阶段
    加入配给时，这里是唯一会告诉我们「单票口径被组合层改动了」的地方。
    """

    @pytest.mark.parametrize("view", ["Buy", "Overweight", "Hold", "Underweight", "Sell"])
    @pytest.mark.parametrize("current", [0.0, 5.0, 13.16, 44.0])
    def test_one_name_matches_the_single_name_sizer(self, view, current):
        atr, close = 4.0723, 150.63   # 300760.SZ 的真实取值
        plan = size_position(view, current, atr=atr, close=close)
        item = allocate([Holding(
            ticker="300760.SZ", current_pct=current, view=view,
            ceiling=size_ceiling(atr, close),
        )]).items[0]
        assert item.target_pct == plan.target_pct
        assert item.ceiling_pct == plan.allowed_pct
        assert item.binding_constraint == plan.binding_constraint
        assert item.delta_label == plan.rating
        assert item.view == plan.view

    def test_single_name_findings_are_carried_through(self):
        """「减仓中有一部分来自风险预算」这条提示不能在搬家途中丢掉。"""
        item = allocate([Holding(
            ticker="X", current_pct=44.0, view="Hold", ceiling=ceiling(12.33, 2.7035),
        )]).items[0]
        assert any("风险预算" in f["reason"] for f in item.findings)


@pytest.mark.unit
class TestBudgetBoundaries:
    """预算层的边界：什么需要额度、什么不需要、额度怎么来。"""

    def test_the_budget_is_always_a_number(self):
        """`investable_pct` 不再是 `None`：额度为 0 与「没有额度概念」是两件事。

        这批的额度 69.00% = 未提交 20.08% + 减仓释放 48.92%，而请求是 0——
        额度算得出来但没人花，这正是这一层在多数批次上的样子。
        """
        result = allocate(batch_0820_holdings())
        assert result.cash_pct == pytest.approx(20.08, abs=0.01)
        assert result.released_pct == pytest.approx(48.92, abs=0.01)
        assert result.investable_pct == pytest.approx(69.00, abs=0.01)
        assert result.requested_pct == 0.0
        assert result.granted_pct == 0.0

    def test_rationing_is_a_no_op_on_the_real_batch(self):
        """08-20 那批零加仓，所以配给必须什么都不做——这里一动就是改坏了第一段。"""
        result = allocate(batch_0820_holdings())
        for item in result.items:
            assert item.target_pct == item.single_name_target_pct
        assert not any(i.binding_constraint == PORTFOLIO_BUDGET for i in result.items)

    def test_trims_are_unconditional(self):
        """减仓不需要预算：即使组合已满仓，减仓照旧足额执行。"""
        result = allocate([
            Holding("A", 40.0, "Sell", ceiling(5.0)),
            Holding("B", 55.0, "Hold", ceiling(6.0)),
        ])
        assert result.by_ticker("A").target_pct == 0.0
        assert result.by_ticker("B").target_pct == 6.0
        assert result.cash_pct == 5.0                       # 100 − 95
        assert result.released_pct == 89.0                  # 40 + 49
        assert result.investable_pct == 94.0                # 减仓自己就把钱腾出来了

    def test_affordable_adds_are_granted_in_full(self):
        result = allocate([
            Holding("A", 0.0, "Buy", ceiling(10.0)),
            Holding("B", 1.0, "Buy", ceiling(8.0)),
        ])
        assert result.by_ticker("A").target_pct == 10.0
        assert result.by_ticker("B").target_pct == 8.0
        assert result.sigma_target == 18.0
        assert result.requested_pct == 17.0
        assert result.granted_pct == 17.0
        assert not any(i.binding_constraint == PORTFOLIO_BUDGET for i in result.items)

    def test_an_add_always_discloses_the_cash_assumption(self):
        """加仓额度的基准是「未提交部分全是现金」这个假设，必须说出来。"""
        result = allocate([Holding("A", 0.0, "Buy", ceiling(10.0))])
        disclosure = [f for f in result.findings if "全部为现金" in f["reason"]]
        assert len(disclosure) == 1
        assert "100.0%" in disclosure[0]["reason"]
        assert disclosure[0]["severity"] == "warn"

    def test_no_disclosure_when_nothing_is_added(self):
        result = allocate([Holding("A", 40.0, "Hold", ceiling(5.0))])
        assert not any("全部为现金" in f["reason"] for f in result.findings)
        assert not any("来自同批减仓释放" in f["reason"] for f in result.findings)

    def test_adds_funded_by_trims_must_say_the_trims_come_first(self):
        """「只买不卖」是真实的失败模式，这条 finding 是它唯一的留痕。"""
        result = allocate([
            Holding("A", 20.0, "Sell", ceiling(5.0)),    # 释放 20pp
            Holding("B", 5.0, "Buy", ceiling(15.0)),     # 请求 10pp
        ])
        assert result.released_pct == 20.0
        assert result.granted_pct == 10.0
        note = [f for f in result.findings if "来自同批减仓释放" in f["reason"]]
        assert len(note) == 1
        assert "10.0%" in note[0]["reason"]

    def test_no_target_ever_exceeds_its_ceiling(self):
        result = allocate([
            Holding("A", 90.0, "Buy", ceiling(4.0)),
            Holding("B", 0.0, "Buy", ceiling(15.0)),
        ])
        for item in result.items:
            assert item.target_pct <= item.ceiling_pct


def bullish_batch(n: int, current: float, allowed: float,
                  view: str = "Buy") -> list[Holding]:
    """N 只同档、同权重、同上限的看多票——用来把额度压到不够。"""
    return [
        Holding(f"T{i:02d}", current, view, ceiling(allowed), f"T{i:02d}")
        for i in range(n)
    ]


@pytest.mark.unit
class TestRationing:
    """额度不够时怎么分。这一层在小批次上基本不触发（波动率上限先把请求压小了），
    真正会触发的是「接近满仓 + 减仓不多 + 几个看多信号」。"""

    def test_budget_is_cash_plus_releases(self):
        result = allocate([
            Holding("A", 30.0, "Sell", ceiling(10.0)),   # 释放 30pp
            Holding("B", 10.0, "Hold", ceiling(20.0)),
        ])
        assert result.cash_pct == 60.0
        assert result.released_pct == 30.0
        assert result.investable_pct == 90.0

    def test_only_releases_are_spendable_when_the_cash_assumption_is_off(self):
        """`assume_uncommitted_is_cash=False` 退化成自筹式再平衡：没减仓就没钱加仓。"""
        holdings = [
            Holding("AAA", 6.0, "Buy", ceiling(15.0)),
            Holding("BBB", 6.0, "Overweight", ceiling(15.0)),
        ]
        assert allocate(holdings).granted_pct == 15.0      # 默认口径：现金 88，够用
        strict = allocate(holdings, config={"portfolio": {
            "assume_uncommitted_is_cash": False,
        }})
        assert strict.investable_pct == 0.0
        assert strict.granted_pct == 0.0
        assert all(i.target_pct == 6.0 for i in strict.items)
        # 假设关掉了，所以那条现金披露不该出现。
        assert not any("全部为现金" in f["reason"] for f in strict.findings)

    def test_bullish_tier_is_served_before_the_next_one(self):
        """Buy 先拿满，Overweight 拿剩下的——档序来自 RATINGS_5_TIER。"""
        holdings = [
            Holding("AAA", 6.0, "Buy", ceiling(15.0)),           # 请求 9
            Holding("BBB", 6.0, "Buy", ceiling(15.0)),           # 请求 9
            Holding("CCC", 6.0, "Overweight", ceiling(15.0)),    # 请求 6（0.8×15=12）
            Holding("DDD", 6.0, "Overweight", ceiling(15.0)),    # 请求 6
            Holding("ZZZ", 20.0, "Sell", ceiling(2.0)),          # 释放 20
        ]
        # 只花减仓释放的 20：请求合计 30 > 20，必须配给。
        result = allocate(holdings, config={"portfolio": {
            "assume_uncommitted_is_cash": False,
        }})
        assert result.investable_pct == 20.0
        assert result.requested_pct == 30.0
        assert result.by_ticker("AAA").target_pct == 15.0    # Buy 档拿满 9+9=18
        assert result.by_ticker("BBB").target_pct == 15.0
        assert result.by_ticker("CCC").target_pct == 7.0     # 剩 2 → 同档各 1.0
        assert result.by_ticker("DDD").target_pct == 7.0
        assert result.granted_pct == 20.0
        # 拿满的那两只不算被配给压过：约束仍是它们自己的单票上限。
        assert result.by_ticker("AAA").binding_constraint != PORTFOLIO_BUDGET
        assert result.by_ticker("CCC").binding_constraint == PORTFOLIO_BUDGET

    def test_pro_rata_within_a_tier_never_exceeds_the_budget(self):
        """同档按请求大小比例缩减，且逐只发放，四舍五入的零头只会剩下不发。"""
        # CCC 用 Hold 而不是 Sell：Sell 会一路减到 0，释放整整 30pp；
        # 「Hold + 上限低于当前」才能精确释放 current − ceiling 这么多。
        holdings = [
            Holding("AAA", 1.0, "Buy", ceiling(10.0)),   # 请求 9
            Holding("BBB", 2.0, "Buy", ceiling(5.0)),    # 请求 3
            Holding("CCC", 30.0, "Hold", ceiling(4.0)),  # 释放 26
        ]
        result = allocate(holdings, config={"portfolio": {
            "assume_uncommitted_is_cash": False,
        }})
        assert result.investable_pct == 26.0     # > 请求 12：这批够用
        assert result.granted_pct == 12.0

        tight = allocate(
            [Holding("AAA", 1.0, "Buy", ceiling(10.0)),
             Holding("BBB", 2.0, "Buy", ceiling(5.0)),
             Holding("CCC", 30.0, "Hold", ceiling(26.0))],   # 只释放 4
            config={"portfolio": {"assume_uncommitted_is_cash": False}},
        )
        assert tight.investable_pct == 4.0
        assert tight.granted_pct <= tight.investable_pct + 1e-9
        assert tight.by_ticker("AAA").target_pct == 4.0   # 1 + 9×(4/12)=3.0
        assert tight.by_ticker("BBB").target_pct == 3.0   # 2 + 3×(4/12)=1.0

    def test_rationed_names_are_attributed_and_annotated(self):
        result = allocate(
            [Holding("AAA", 1.0, "Buy", ceiling(10.0)),
             Holding("CCC", 30.0, "Hold", ceiling(26.0))],
            config={"portfolio": {"assume_uncommitted_is_cash": False}},
        )
        item = result.by_ticker("AAA")
        assert item.single_name_target_pct == 10.0
        assert item.target_pct == 5.0                       # 1 + 4
        assert item.binding_constraint == PORTFOLIO_BUDGET
        assert constraint_label(PORTFOLIO_BUDGET) == "组合预算"
        assert any("单票期望仓位 10.0%" in f["reason"] for f in item.findings)
        assert any("已按观点档序" in f["reason"] for f in result.findings)

    def test_the_change_label_follows_the_rationed_number(self):
        """标签必须描述最终数字：被砍到不动的加仓是 Hold，不是 Buy。"""
        result = allocate([
            Holding("A", 50.0, "Hold", ceiling(60.0)),
            Holding("B", 50.0, "Buy", ceiling(60.0)),   # 想到 60，但一分钱都没有
        ])
        assert result.sigma_current == 100.0
        assert result.investable_pct == 0.0
        b = result.by_ticker("B")
        assert b.single_name_target_pct == 60.0
        assert b.target_pct == 50.0
        assert b.delta_label == "Hold"
        assert b.view == "Buy"                          # 观点没有被改写
        assert b.binding_constraint == PORTFOLIO_BUDGET

    def test_crumbs_below_min_add_pct_are_dropped(self):
        """+0.3pp 换不回交易成本，标签本来也还是 Hold——发这条指令等于噪音。"""
        holdings = [
            Holding("AAA", 1.0, "Buy", ceiling(10.0)),
            Holding("BBB", 1.0, "Buy", ceiling(10.0)),
            Holding("CCC", 30.0, "Hold", ceiling(29.4)),   # 只释放 0.6
        ]
        cfg = {"portfolio": {"assume_uncommitted_is_cash": False}}
        result = allocate(holdings, config=cfg)
        assert result.investable_pct == 0.6
        assert result.granted_pct == 0.0                  # 各 0.3pp，全部归零
        for tk in ("AAA", "BBB"):
            item = result.by_ticker(tk)
            assert item.target_pct == 1.0
            assert item.binding_constraint == PORTFOLIO_BUDGET
            assert any("不足最小加仓额度" in f["reason"] for f in item.findings)

        # 阈值调低后同一批就该发出去，证明归零来自 min_add_pct 而非算错。
        loose = allocate(holdings, config={"portfolio": {
            "assume_uncommitted_is_cash": False, "min_add_pct": 0.1,
        }})
        assert loose.granted_pct == 0.6
        assert loose.by_ticker("AAA").target_pct == 1.3

    def test_sigma_target_may_reach_one_hundred_percent(self):
        """不设总权益硬顶、不设现金底线是明确决定：这里钉住，防止有人加回一个 min()。"""
        result = allocate(bullish_batch(8, current=6.0, allowed=15.0))
        assert result.requested_pct == 72.0
        assert result.investable_pct == 52.0
        assert result.granted_pct == 52.0
        assert result.sigma_target == 100.0
        # 而且这不该触发任何阈值告警：`max_total_equity_pct` 只看 Σcurrent（48%），
        # 本段刻意没有给 Σtarget 加告警——满仓与否由 `describe_allocation`
        # 那一行如实印出来，读者自己判断。
        assert not any("总权益上限" in f["reason"] for f in result.findings)
        assert "目标合计 100.00%" in describe_allocation(result)

    def test_rationing_is_reproducible(self):
        holdings = bullish_batch(8, current=6.0, allowed=15.0)
        assert allocate(holdings).as_dict() == allocate(holdings).as_dict()

    def test_granted_never_exceeds_the_budget(self):
        for n in range(1, 13):
            for current in (0.0, 3.0, 6.0, 9.0):
                result = allocate(bullish_batch(n, current=current, allowed=15.0))
                assert result.granted_pct <= (result.investable_pct or 0.0) + 1e-9
                for item in result.items:
                    assert item.target_pct <= item.single_name_target_pct + 1e-9


@pytest.mark.unit
class TestNeverRaises:
    """这一层跑在 N 只票全部完成之后，抛一次异常就毁掉整批报告。"""

    def test_failed_analysis_is_frozen_but_still_counted(self):
        """失败票没有观点可执行，但那笔钱是真实存在的，必须计入 Σcurrent。"""
        result = allocate([
            Holding("OK", 10.0, "Hold", ceiling(20.0)),
            Holding("BAD", 30.0, None, None, status=STATUS_FAILED),
        ])
        bad = result.by_ticker("BAD")
        assert bad.target_pct == 30.0
        assert bad.binding_constraint == STATUS_QUO
        assert any("分析未完成" in f["reason"] for f in bad.findings)
        assert result.sigma_current == 40.0
        assert result.frozen_pct == 30.0

    def test_missing_ceiling_is_frozen_with_a_finding(self):
        item = allocate([Holding("A", 12.0, "Sell", None)]).items[0]
        assert item.target_pct == 12.0        # 不敢在没有上限的情况下动仓位
        assert item.ceiling_pct is None
        assert any("缺少风险上限" in f["reason"] for f in item.findings)

    def test_no_position_context_is_excluded_from_the_totals(self):
        result = allocate([
            Holding("A", None, "Buy", ceiling(9.0)),
            Holding("B", 10.0, "Hold", ceiling(20.0)),
        ])
        assert result.by_ticker("A").target_pct is None
        assert result.sigma_current == 10.0
        assert result.sigma_target == 10.0
        assert any("未提供持仓上下文" in f["reason"] for f in result.findings)

    def test_empty_portfolio(self):
        result = allocate([])
        assert result.items == []
        assert result.sigma_current == 0.0
        assert result.cash_pct == 100.0
        assert result.findings == []

    def test_garbage_inputs_degrade_instead_of_raising(self):
        result = allocate([
            Holding("A", "not a number", "赶紧买", ceiling(9.0)),
            Holding("B", -5.0, None, ceiling(9.0)),
        ])
        assert isinstance(result, Allocation)
        assert result.by_ticker("A").target_pct == 0.0   # 脏仓位按 0 处理
        assert result.by_ticker("B").target_pct == 0.0   # 负仓位按 0 处理

    def test_malformed_persisted_ceiling_yields_none(self):
        holding = holding_from_ceiling_dict("A", 5.0, "Hold", {"allowed_pct": "x"})
        assert holding.ceiling is None
        assert allocate([holding]).items[0].target_pct == 5.0

    def test_persisted_ceiling_round_trips(self):
        original = size_ceiling(4.0723, 150.63)
        holding = holding_from_ceiling_dict("A", 44.0, "Hold", original.as_dict())
        assert holding.ceiling.allowed_pct == original.allowed_pct
        assert holding.ceiling.binding_constraint == original.binding_constraint
        assert allocate([holding]).items[0].target_pct == 12.33


@pytest.mark.unit
class TestReproducibility:
    def test_same_input_allocates_identically(self):
        first = allocate(batch_0820_holdings()).as_dict()
        second = allocate(batch_0820_holdings()).as_dict()
        assert first == second

    def test_item_order_follows_input_order(self):
        holdings = batch_0820_holdings()
        result = allocate(holdings)
        assert [i.ticker for i in result.items] == [h.ticker for h in holdings]

    def test_allocation_is_json_serialisable_for_the_db(self):
        import json

        payload = allocate(batch_0820_holdings()).as_dict()
        assert json.loads(json.dumps(payload, ensure_ascii=False))["sigma_target"] == (
            SIGMA_TARGET_0820
        )

    def test_summary_line_states_both_totals(self):
        text = describe_allocation(allocate(batch_0820_holdings()))
        # 两位小数：`round()` 的原始值会把 31.00 印成 "31.0"，与报告层里同一个
        # 权重的写法不一致。
        assert f"{SIGMA_CURRENT_0820:.2f}%" in text
        assert f"{SIGMA_TARGET_0820:.2f}%" in text


@pytest.mark.unit
class TestConfigOverrides:
    def test_thresholds_are_configurable(self):
        holdings = [
            Holding("A", 20.0, "Hold", ceiling(30.0)),
            Holding("B", 20.0, "Hold", ceiling(30.0)),
        ]
        quiet = allocate(holdings, config={"portfolio": {
            "concentration_warn_pct": 90.0, "max_total_equity_pct": 90.0,
        }})
        assert not any("集中度" in f["reason"] for f in quiet.findings)

        loud = allocate(holdings, config={"portfolio": {
            "concentration_warn_pct": 40.0, "max_total_equity_pct": 30.0,
        }})
        assert any("集中度" in f["reason"] for f in loud.findings)
        assert any("总权益上限" in f["reason"] for f in loud.findings)

    def test_sizing_overrides_reach_the_single_name_step(self):
        item = allocate(
            [Holding("A", 10.0, "Underweight", ceiling(20.0))],
            config={"position_sizing": {"trim": {"Underweight": 0.5}}},
        ).items[0]
        assert item.target_pct == pytest.approx(5.0)


@pytest.mark.unit
def test_status_ok_is_the_default():
    assert Holding("A").status == STATUS_OK
