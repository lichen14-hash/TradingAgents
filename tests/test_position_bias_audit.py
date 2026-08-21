"""持仓上下文偏置审计必须能自己发现"这条结论其实是一只票"。

事故（2026-08-19 手工复盘，327 条 predictions）：注入持仓上下文后减仓档占比从 12.8%
升到 43.4%（p=9.4e-07），同标的、同交易日两种控制下都成立。但同一份数据里还有一条
**看起来更强**的结论——"超配 ≥20% → 34/37 减仓、零加仓"——它是假的：那 37 条里
33 条来自 002602.SZ 一只票，剔除后只剩 4 条。

两条结论的卡方都极显著，差别只在标的集中度。所以这里测的不是"能不能算出 p 值"，
而是**判定门槛能不能拦住共线**：任何按持仓字段分组的结论都天然与"哪几只票被持有"
共线，而显著性检验对共线毫无抵抗力。:func:`dominance` 门槛是这个模块的主要价值，
所以它有最多的用例。

另外三个易错点也在这里钉住：
1. 前向收益的信号日必须取**不晚于** trade_date 的最后一根K线——取当天会丢掉盘后/
   非交易日跑的那些轮次，取下一根会引入未来信息；
2. 浮亏必须用信号日收盘价现算，不能读 `price_at_signal`（历史 301 条全为 NULL）；
3. "没测出来"（INSUFFICIENT_DATA）与"测出来没有"（NOT_ESTABLISHED）必须是两种判定，
   合并成一个布尔值等于把样本不足读成"没有偏置"。
"""

import math

import pytest

from tradingagents.agents.utils.position_sizing import METHOD_ATR
from tradingagents.backtest.position_bias import (
    ADD_TIERS,
    ERA_DETERMINISTIC,
    ERA_LEGACY,
    HORIZONS,
    MAX_TICKER_SHARE,
    MIN_GROUP_N,
    RATING_DIRECTION,
    TRIM_TIERS,
    Row,
    Verdict,
    analyse_date_controlled,
    analyse_headline,
    analyse_hit_rate,
    analyse_pnl_gradient,
    analyse_position_size,
    analyse_ticker_controlled,
    build_rows,
    chi2_2x2,
    chi2_p,
    dominance,
    era_label,
    forward_returns,
    leave_one_out,
    per_ticker_cases,
    run_audit,
    run_stratified_audit,
    sizing_method_of,
    split_by_era,
)

# 30 个连续交易日，收盘价从 100 起每天 +1（涨势），方便手算前向收益。
BARS = {f"2026-07-{d:02d}": 100.0 + d for d in range(1, 31)}


def _row(
    ticker="300760.SZ",
    rating="Hold",
    position_pct=10.0,
    pnl_pct=0.0,
    fwd=None,
    trade_date="2026-07-01",
    sizing_method="",
) -> Row:
    # 空 sizing_method = 改版前（仓位由模型自选）。默认留空，因为 327 条历史数据
    # 全是那一层，绝大多数用例测的也是那一层的判定逻辑。
    return Row(
        ticker=ticker,
        name=ticker,
        trade_date=trade_date,
        rating=rating,
        position_pct=position_pct,
        cost_price=100.0,
        signal_close=100.0,
        pnl_pct=pnl_pct,
        sizing_method=sizing_method,
        forward=dict(fwd or {h: None for h in HORIZONS}),
    )


def _group(n, *, ticker_prefix="T", rating="Hold", position_pct=10.0, **kw) -> list[Row]:
    """n 条行，标的名各不相同——默认就通过集中度门槛，除非用例故意破坏它。"""
    return [
        _row(ticker=f"{ticker_prefix}{i:03d}.SZ", rating=rating,
             position_pct=position_pct, **kw)
        for i in range(n)
    ]


@pytest.mark.unit
class TestStatistics:
    def test_chi2_matches_the_hand_checked_incident_figure(self):
        """带持仓 108/249 减仓 vs 无持仓 10/78——复盘正文里引用的就是这个数。"""
        result = chi2_2x2(108, 141, 10, 68)
        assert result.chi2 == pytest.approx(24.04, abs=0.05)
        assert result.p == pytest.approx(9.4e-07, rel=0.1)
        assert result.pct_a == pytest.approx(43.4, abs=0.1)
        assert result.pct_b == pytest.approx(12.8, abs=0.1)

    def test_p_value_agrees_with_the_chi2_cdf_at_known_points(self):
        """没有 scipy，所以 1 自由度的上尾概率是自己算的，必须对得上教科书值。"""
        assert chi2_p(3.841) == pytest.approx(0.05, abs=0.001)
        assert chi2_p(6.635) == pytest.approx(0.01, abs=0.001)
        assert chi2_p(10.828) == pytest.approx(0.001, abs=0.0001)
        assert chi2_p(0.0) == 1.0

    def test_higher_dof_is_refused_rather_than_approximated(self):
        with pytest.raises(ValueError):
            chi2_p(5.0, dof=2)

    def test_empty_margin_yields_no_effect_instead_of_dividing_by_zero(self):
        """全 0 的一列（例如某档一条加仓都没有）不能让整轮审计崩掉。"""
        result = chi2_2x2(0, 30, 0, 30)
        assert result.chi2 == 0.0
        assert result.p == 1.0

    def test_identical_shares_are_not_significant(self):
        assert chi2_2x2(10, 40, 10, 40).p == 1.0


@pytest.mark.unit
class TestDominanceGuard:
    """本模块存在的主要理由。超配档 33/37 是一只票，卡方却极显著。"""

    def test_single_ticker_group_is_flagged(self):
        rows = [_row(ticker="002602.SZ") for _ in range(37)]
        dom = dominance(rows)
        assert dom.top_ticker == "002602.SZ"
        assert dom.top_share == 1.0
        assert dom.distinct == 1
        assert dom.dominated

    def test_the_actual_incident_ratio_is_flagged(self):
        """33/37 = 89%，按仓位聚合的表里完全看不出来。"""
        rows = [_row(ticker="002602.SZ") for _ in range(33)]
        rows += [_row(ticker=f"X{i}.SZ") for i in range(4)]
        dom = dominance(rows)
        assert dom.top_share == pytest.approx(33 / 37)
        assert dom.dominated

    def test_a_diverse_group_is_not_flagged(self):
        assert not dominance(_group(20)).dominated

    def test_threshold_is_exclusive_so_an_even_split_survives(self):
        """恰好 50% 不算主导：两只票对半分仍然是两只票的共同行为。"""
        rows = [_row(ticker="A.SZ") for _ in range(10)]
        rows += [_row(ticker="B.SZ") for _ in range(10)]
        dom = dominance(rows)
        assert dom.top_share == MAX_TICKER_SHARE
        assert not dom.dominated

    def test_empty_group_is_safe(self):
        dom = dominance([])
        assert dom.n == 0 and dom.top_ticker is None and not dom.dominated


@pytest.mark.unit
class TestVerdictLadder:
    """"没测出来"和"测出来没有"必须分开——合并等于把样本不足读成"没有偏置"。"""

    def test_dominated_group_cannot_be_established_however_significant(self):
        """核心用例：p 极小但一边全是同一只票 → 必须降为 NOT_ESTABLISHED。"""
        heavy = [_row(ticker="002602.SZ", rating="Sell", position_pct=44.0)
                 for _ in range(33)]
        heavy += [_row(ticker=f"H{i}.SZ", rating="Sell", position_pct=25.0)
                  for i in range(4)]
        light = _group(40, ticker_prefix="L", rating="Hold", position_pct=2.0)
        finding, _ = analyse_position_size(heavy + light)
        assert finding.test.p < 1e-9, "构造的效应本身必须是极显著的，否则测的不是门槛"
        assert finding.verdict is Verdict.NOT_ESTABLISHED
        assert "002602.SZ" in " ".join(finding.notes)

    def test_small_sample_is_insufficient_not_negative(self):
        rows = _group(5, ticker_prefix="A", rating="Sell", position_pct=30.0)
        rows += _group(5, ticker_prefix="B", rating="Hold", position_pct=2.0)
        finding, _ = analyse_position_size(rows)
        assert finding.verdict is Verdict.INSUFFICIENT_DATA
        assert str(MIN_GROUP_N) in " ".join(finding.notes)

    def test_a_diverse_significant_effect_is_established(self):
        with_pos = _group(60, ticker_prefix="P", rating="Sell")
        with_pos += _group(40, ticker_prefix="Q", rating="Hold")
        without = [_row(ticker=f"P{i:03d}.SZ", rating="Hold", position_pct=None)
                   for i in range(60)]
        finding = analyse_headline(with_pos + without)
        assert finding.verdict is Verdict.ESTABLISHED

    def test_a_marginal_effect_is_only_suggestive(self):
        """+5d 命中率那条就落在这一档（p=0.038），不能与 p=1e-6 的结论同级呈现。"""
        with_pos = _group(30, ticker_prefix="P", rating="Sell")
        with_pos += _group(40, ticker_prefix="Q", rating="Hold")
        without = _group(15, ticker_prefix="R", rating="Sell", position_pct=None)
        without += _group(45, ticker_prefix="S", rating="Hold", position_pct=None)
        finding = analyse_headline(with_pos + without)
        assert 0.01 <= finding.test.p < 0.05
        assert finding.verdict is Verdict.SUGGESTIVE

    def test_an_effect_that_dies_without_one_ticker_is_downgraded(self):
        """50% 门槛拦不住"一只票占 25% 却独自撑起整个效应"，留一检查才能。

        构造：60 条 X.SZ 全 Sell + 60 条杂票全 Hold（带持仓）对 60 条杂票全 Hold
        （无持仓）。X.SZ 只占合计 33%，过得了集中度门槛；但把它剔掉后带持仓组与
        无持仓组一模一样，效应整体消失。
        """
        with_pos = [_row(ticker="X.SZ", rating="Sell") for _ in range(60)]
        with_pos += _group(60, ticker_prefix="Q", rating="Hold")
        without = _group(60, ticker_prefix="R", rating="Hold", position_pct=None)
        finding = analyse_headline(with_pos + without)
        assert not dominance(with_pos).dominated, "构造必须过得了集中度门槛"
        assert finding.test.p < 1e-6, "剔除前效应必须是显著的"
        assert finding.verdict is Verdict.NOT_ESTABLISHED
        assert "留一检查" in " ".join(finding.notes)
        assert "X.SZ" in " ".join(finding.notes)

    def test_an_effect_that_survives_leave_one_out_stays_established(self):
        with_pos = _group(60, ticker_prefix="P", rating="Sell")
        with_pos += _group(40, ticker_prefix="Q", rating="Hold")
        without = _group(60, ticker_prefix="R", rating="Hold", position_pct=None)
        finding = analyse_headline(with_pos + without)
        assert finding.verdict is Verdict.ESTABLISHED
        assert "仍成立" in " ".join(finding.notes)

    def test_leave_one_out_reports_nothing_for_an_empty_sample(self):
        assert leave_one_out([], [], lambda r: True) == (None, None, None)

    def test_no_effect_is_not_established(self):
        with_pos = _group(30, ticker_prefix="P", rating="Sell")
        with_pos += _group(30, ticker_prefix="Q", rating="Hold")
        without = _group(30, ticker_prefix="R", rating="Sell", position_pct=None)
        without += _group(30, ticker_prefix="S", rating="Hold", position_pct=None)
        finding = analyse_headline(with_pos + without)
        assert finding.verdict is Verdict.NOT_ESTABLISHED
        assert "不显著" in " ".join(finding.notes)


@pytest.mark.unit
class TestStratifiedControls:
    def test_ticker_control_drops_tickers_that_have_only_one_context(self):
        """整体差异可以完全由"无持仓那批恰好是另一批标的"造成。"""
        # A 系列两种上下文都有且确有差异；B 系列只有带持仓且全是 Sell。
        rows = []
        for i in range(20):
            rows.append(_row(ticker=f"A{i:02d}.SZ", rating="Sell"))
            rows.append(_row(ticker=f"A{i:02d}.SZ", rating="Hold", position_pct=None))
        rows += _group(50, ticker_prefix="B", rating="Sell")
        finding = analyse_ticker_controlled(rows)
        # 只有 A 系列进入检验：20 条带持仓（全 Sell）vs 20 条无持仓（全 Hold）。
        assert finding.test.a + finding.test.b == 20
        assert "20 只标的内" in finding.detail

    def test_ticker_control_reports_no_effect_when_the_gap_was_only_selection(self):
        """同一标的内两种上下文分布一致 → 整体差异纯粹来自选股。"""
        rows = []
        for i in range(20):
            rows.append(_row(ticker=f"A{i:02d}.SZ", rating="Hold"))
            rows.append(_row(ticker=f"A{i:02d}.SZ", rating="Hold", position_pct=None))
        rows += _group(50, ticker_prefix="B", rating="Sell")
        assert analyse_ticker_controlled(rows).verdict is not Verdict.ESTABLISHED

    def test_date_control_uses_days_with_both_contexts(self):
        rows = []
        for day in range(1, 21):
            date = f"2026-07-{day:02d}"
            rows.append(_row(ticker=f"A{day:02d}.SZ", rating="Sell", trade_date=date))
            rows.append(_row(ticker=f"B{day:02d}.SZ", rating="Hold",
                             position_pct=None, trade_date=date))
        rows += _group(30, ticker_prefix="C", rating="Sell", trade_date="2026-08-01")
        finding = analyse_date_controlled(rows)
        assert "20 个交易日内" in finding.detail
        assert finding.test.a + finding.test.b == 20


@pytest.mark.unit
class TestPnlGradient:
    def test_leave_one_out_table_is_always_produced(self):
        """深亏档天然由被套重仓票构成，"剔了还剩多少"是读这张表的前提。"""
        rows = [_row(ticker="002602.SZ", rating="Sell", pnl_pct=-28.0)
                for _ in range(60)]
        rows += _group(20, ticker_prefix="L", rating="Hold", pnl_pct=-10.0)
        rows += _group(20, ticker_prefix="G", rating="Overweight", pnl_pct=12.0)
        _, full_table, ex_table, excluded = analyse_pnl_gradient(rows)
        assert excluded == "002602.SZ"
        assert next(r for r in full_table if r["bucket"] == "深亏 <-20%")["n"] == 60
        assert next(r for r in ex_table if r["bucket"] == "深亏 <-20%")["n"] == 0

    def test_monotone_gradient_is_detected(self):
        rows = _group(30, ticker_prefix="L", rating="Hold", pnl_pct=-25.0)
        rows += _group(30, ticker_prefix="M", rating="Hold", pnl_pct=-10.0)
        rows += _group(30, ticker_prefix="F", rating="Hold", pnl_pct=0.0)
        rows += _group(30, ticker_prefix="G", rating="Overweight", pnl_pct=12.0)
        finding, table, _, _ = analyse_pnl_gradient(rows)
        add_pcts = [r["add_pct"] for r in table]
        assert add_pcts == [0.0, 0.0, 0.0, 100.0], "分档顺序必须是深亏→浮盈"
        assert finding.verdict is Verdict.ESTABLISHED

    def test_rows_without_cost_price_are_skipped_not_bucketed_as_flat(self):
        """把"没有成本价"当成"持平"会凭空造出一大堆浮亏为 0 的样本。"""
        rows = _group(20, ticker_prefix="L", rating="Sell", pnl_pct=-25.0)
        rows += _group(20, ticker_prefix="G", rating="Overweight", pnl_pct=12.0)
        rows += _group(50, ticker_prefix="N", rating="Sell", pnl_pct=None)
        _, table, _, _ = analyse_pnl_gradient(rows)
        assert sum(r["n"] for r in table) == 40

    def test_positionless_rows_never_enter_the_pnl_analysis(self):
        rows = _group(20, ticker_prefix="L", rating="Sell", pnl_pct=-25.0)
        rows += [_row(ticker=f"Z{i}.SZ", rating="Overweight",
                      position_pct=None, pnl_pct=12.0) for i in range(20)]
        _, table, _, _ = analyse_pnl_gradient(rows)
        assert next(r for r in table if r["bucket"] == "浮盈 >+5%")["n"] == 0


@pytest.mark.unit
class TestForwardReturns:
    def test_signal_day_is_the_last_bar_at_or_before_the_trade_date(self):
        """盘后/非交易日跑的轮次也要能算，但不能借用之后的K线。"""
        base, fwd = forward_returns(BARS, "2026-07-05")
        assert base == 105.0
        assert fwd[5] == pytest.approx((110.0 - 105.0) / 105.0 * 100)

    def test_a_non_trading_trade_date_falls_back_to_the_previous_bar(self):
        bars = {k: v for k, v in BARS.items() if k != "2026-07-05"}
        base, _ = forward_returns(bars, "2026-07-05")
        assert base == 104.0

    def test_horizons_are_counted_in_bars_not_calendar_days(self):
        sparse = {"2026-07-01": 100.0, "2026-07-20": 110.0, "2026-07-31": 120.0}
        _, fwd = forward_returns(sparse, "2026-07-01", horizons=(1, 2))
        assert fwd[1] == pytest.approx(10.0)
        assert fwd[2] == pytest.approx(20.0)

    def test_missing_future_bars_yield_none_not_zero(self):
        """最近几轮的 +20d 还没走完，记成 0% 会把它们算作"预测失败"。"""
        _, fwd = forward_returns(BARS, "2026-07-29")
        assert fwd[5] is None and fwd[10] is None and fwd[20] is None

    def test_trade_date_before_every_bar_yields_nothing(self):
        base, fwd = forward_returns(BARS, "2026-06-01")
        assert base is None
        assert all(v is None for v in fwd.values())

    def test_empty_bars_are_safe(self):
        base, fwd = forward_returns({}, "2026-07-05")
        assert base is None and all(v is None for v in fwd.values())


@pytest.mark.unit
class TestBuildRows:
    def _raw(self, **kw) -> dict:
        raw = {
            "ticker": "300760.SZ", "name": "迈瑞医疗", "trade_date": "2026-07-05",
            "rating": "Underweight", "cost_price": 120.0, "shares": 100.0,
            "position_pct": 9.0, "price_at_signal": None,
        }
        raw.update(kw)
        return raw

    def test_pnl_is_recomputed_from_bars_not_read_from_the_db(self):
        """`price_at_signal` 在 2026-08-20 之前的 301 条里全为 NULL。"""
        row = build_rows([self._raw()], {"300760.SZ": BARS})[0]
        assert row.signal_close == 105.0
        assert row.pnl_pct == pytest.approx((105.0 - 120.0) / 120.0 * 100)

    def test_price_at_signal_is_only_a_fallback_when_bars_are_absent(self):
        row = build_rows([self._raw(price_at_signal=99.0)], {})[0]
        assert row.signal_close == 99.0
        assert row.pnl_pct == pytest.approx((99.0 - 120.0) / 120.0 * 100)

    def test_null_position_means_no_position_context_was_injected(self):
        rows = build_rows(
            [self._raw(position_pct=None, cost_price=None, shares=None)],
            {"300760.SZ": BARS},
        )
        assert rows[0].has_position is False
        assert rows[0].pnl_pct is None

    def test_zero_position_pct_still_counts_as_context_injected(self):
        """零仓位但上下文已注入 ≠ 没注入；用真值判断会把它错分到对照组。"""
        assert build_rows([self._raw(position_pct=0.0)], {})[0].has_position is True

    def test_missing_rating_defaults_to_hold_rather_than_crashing(self):
        assert build_rows([self._raw(rating=None)], {})[0].rating == "Hold"

    def test_unknown_ticker_yields_no_forward_returns(self):
        row = build_rows([self._raw(ticker="XXXX.SZ")], {"300760.SZ": BARS})[0]
        assert all(v is None for v in row.forward.values())


@pytest.mark.unit
class TestHitRate:
    def test_trim_calls_are_correct_when_the_price_falls(self):
        rows = _group(20, ticker_prefix="P", rating="Sell",
                      fwd={5: -3.0, 10: 2.0, 20: -1.0})
        rows += _group(20, ticker_prefix="Q", rating="Sell", position_pct=None,
                       fwd={5: -3.0, 10: 2.0, 20: -1.0})
        by_h = {f.key: f for f in analyse_hit_rate(rows)}
        assert by_h["hit_rate_gap_5d"].test.pct_a == 100.0
        assert by_h["hit_rate_gap_10d"].test.pct_a == 0.0

    def test_add_calls_are_correct_when_the_price_rises(self):
        rows = _group(20, ticker_prefix="P", rating="Overweight", fwd={5: 4.0})
        rows += _group(20, ticker_prefix="Q", rating="Overweight",
                       position_pct=None, fwd={5: 4.0})
        assert analyse_hit_rate(rows)[0].test.pct_a == 100.0

    def test_hold_is_excluded_from_the_denominator(self):
        """Hold 没有可判对错的方向，算进分母会把命中率往 0 拖。"""
        rows = _group(20, ticker_prefix="P", rating="Hold", fwd={5: 4.0})
        rows += _group(20, ticker_prefix="Q", rating="Sell", fwd={5: -4.0})
        rows += _group(20, ticker_prefix="R", rating="Sell",
                       position_pct=None, fwd={5: -4.0})
        finding = analyse_hit_rate(rows)[0]
        assert finding.test.a + finding.test.b == 20

    def test_each_horizon_is_reported_separately(self):
        """+5d 显著、+10d/+20d 不显著——合成一个数字会夸大结论。"""
        keys = {f.key for f in analyse_hit_rate(_group(40, rating="Sell"))}
        assert keys == {f"hit_rate_gap_{h}d" for h in HORIZONS}


@pytest.mark.unit
class TestPerTickerCases:
    def test_lopsided_trim_into_a_rally_sorts_first(self):
        """"劝减 9 次然后涨了 12%"这种实例才是定位提示词问题的入口。"""
        rows = [_row(ticker="09988.HK", rating="Underweight", pnl_pct=-0.8,
                     fwd={5: 4.0, 10: 8.0, 20: 12.15}) for _ in range(9)]
        rows += [_row(ticker="300760.SZ", rating="Overweight", pnl_pct=9.4,
                      fwd={5: 1.0, 10: 2.0, 20: 3.45}) for _ in range(6)]
        cases = per_ticker_cases(rows)
        assert cases[0]["ticker"] == "09988.HK"
        assert cases[0]["trim"] == 9 and cases[0]["add"] == 0
        assert cases[0]["fwd"][20] == pytest.approx(12.15)

    def test_thin_tickers_are_dropped(self):
        rows = [_row(ticker="A.SZ") for _ in range(2)]
        rows += [_row(ticker="B.SZ") for _ in range(5)]
        assert {c["ticker"] for c in per_ticker_cases(rows)} == {"B.SZ"}

    def test_positionless_rows_are_not_listed(self):
        rows = [_row(ticker="Z.SZ", position_pct=None) for _ in range(5)]
        assert per_ticker_cases(rows) == []


@pytest.mark.unit
class TestReportContract:
    """结论要落成 JSON 供下一轮做基线对比，形状不能漂。"""

    def _rows(self) -> list[Row]:
        rows = _group(60, ticker_prefix="P", rating="Sell", pnl_pct=-20.0,
                      fwd={5: -2.0, 10: -3.0, 20: -4.0})
        rows += _group(40, ticker_prefix="Q", rating="Overweight", pnl_pct=10.0,
                       fwd={5: 2.0, 10: 3.0, 20: 4.0})
        rows += [_row(ticker=f"R{i:03d}.SZ", rating="Hold", position_pct=None,
                      fwd={5: 1.0, 10: 1.0, 20: 1.0}) for i in range(30)]
        return rows

    def test_every_expected_finding_is_present(self):
        report = run_audit(self._rows())
        expected = {
            "headline_trim_share", "ticker_controlled_trim_share",
            "date_controlled_trim_share", "pnl_gradient_add_share",
            "position_size_trim_share",
            *(f"hit_rate_gap_{h}d" for h in HORIZONS),
        }
        assert {f.key for f in report.findings} == expected

    def test_json_payload_is_serialisable_and_carries_the_verdicts(self):
        payload = run_audit(self._rows()).as_dict()
        import json

        text = json.dumps(payload, ensure_ascii=False)
        assert "ESTABLISHED" in text or "NOT_ESTABLISHED" in text
        assert json.loads(text)["coverage"]["rows"] == 130
        for f in payload["findings"]:
            assert f["verdict"] in {v.value for v in Verdict}
            assert f["question"] and f["detail"]

    def test_coverage_separates_context_groups_and_forward_availability(self):
        cov = run_audit(self._rows()).coverage
        assert cov["with_position"] == 100
        assert cov["without_position"] == 30
        assert cov["forward_available"]["20"] == 130

    def test_market_baseline_is_reported_so_a_falling_tape_is_visible(self):
        """全样本在跌时看空本该好做，命中率必须对着这个基准读。"""
        baseline = run_audit(self._rows()).baseline
        assert baseline[20] is not None
        assert math.isfinite(baseline[20])

    def test_established_shortlist_drives_the_exit_code(self):
        report = run_audit(self._rows())
        assert all(f.verdict is Verdict.ESTABLISHED for f in report.established)

    def test_an_empty_sample_does_not_crash(self):
        report = run_audit([])
        assert report.coverage["rows"] == 0
        assert all(
            f.verdict is Verdict.INSUFFICIENT_DATA for f in report.findings
        )


@pytest.mark.unit
class TestEraStratification:
    """两个era绝不能合并统计。

    改版前的行（仓位由模型自选）必然带着浮亏梯度——那是已经发生的事实。改版后的行
    由风险预算算仓位，不含成本价这条通路。把两者放进同一个卡方里，新样本会把旧梯度
    **稀释**成"看起来修好了"，而这恰恰是这次改动最需要避免的假阳性。
    """

    def test_a_row_without_a_sizing_method_is_pre_change(self):
        assert _row().era == ERA_LEGACY

    def test_a_row_with_a_sizing_method_is_post_change(self):
        assert _row(sizing_method=METHOD_ATR).era == ERA_DETERMINISTIC

    @pytest.mark.parametrize("stored,expected", [
        ('{"method": "atr_risk_budget"}', "atr_risk_budget"),   # 库里的真实形态：JSON 文本
        ({"method": "fallback_stop"}, "fallback_stop"),         # 调用方给 dict 也接
        (None, ""),                                            # 迁移后的历史行
        ("", ""),
        ("not json at all", ""),        # 坏行落进改版前那层，而不是把整轮审计搞崩
        ('{"method": null}', ""),
        ('["method"]', ""),             # JSON 合法但不是对象
        ('{"binding_constraint": "risk_budget"}', ""),
    ])
    def test_sizing_method_is_read_defensively(self, stored, expected):
        assert sizing_method_of({"position_sizing": stored}) == expected

    def test_a_row_missing_the_column_entirely_is_pre_change(self):
        """`--offline` 打到 v5 库时 SELECT 里根本没有这一列。"""
        assert sizing_method_of({"ticker": "300760.SZ"}) == ""

    def test_split_keeps_the_two_eras_apart(self):
        rows = _group(4, ticker_prefix="OLD") + _group(
            6, ticker_prefix="NEW", sizing_method=METHOD_ATR,
        )
        split = split_by_era(rows)
        assert set(split) == {ERA_LEGACY, ERA_DETERMINISTIC}
        assert len(split[ERA_LEGACY]) == 4
        assert len(split[ERA_DETERMINISTIC]) == 6

    def test_empty_strata_are_omitted_not_reported_as_zero(self):
        """只有改版前样本时，不要凭空造一层"改版后 0 条"的空报告。"""
        split = split_by_era(_group(3))
        assert list(split) == [ERA_LEGACY]

    def test_ordering_is_chronological(self):
        split = split_by_era(
            _group(2, ticker_prefix="NEW", sizing_method=METHOD_ATR) + _group(2),
        )
        assert list(split) == [ERA_LEGACY, ERA_DETERMINISTIC]

    def test_each_stratum_reports_its_own_era_and_counts(self):
        audit = run_stratified_audit(
            _group(3) + _group(5, ticker_prefix="NEW", sizing_method=METHOD_ATR),
        )
        assert audit.counts == {ERA_LEGACY: 3, ERA_DETERMINISTIC: 5}
        assert audit.report(ERA_LEGACY).era == ERA_LEGACY
        assert audit.report(ERA_LEGACY).coverage["rows"] == 3
        assert audit.report(ERA_DETERMINISTIC).coverage["rows"] == 5

    def test_coverage_names_the_sizing_algorithms_in_play(self):
        """混合期（部分标的先上线）要能看出每层里各算法各多少条。"""
        audit = run_stratified_audit(
            _group(2) + _group(3, ticker_prefix="A", sizing_method=METHOD_ATR)
            + _group(1, ticker_prefix="F", sizing_method="fallback_stop"),
        )
        assert audit.report(ERA_LEGACY).coverage["sizing_methods"] == {"(none)": 2}
        assert audit.report(ERA_DETERMINISTIC).coverage["sizing_methods"] == {
            METHOD_ATR: 3, "fallback_stop": 1,
        }

    def test_regressions_ignore_the_pre_change_stratum(self):
        """退出码只看改版后那层——否则这个检查从上线第一天起就恒为红灯。"""
        # 与 TestPnlGradient::test_monotone_gradient_is_detected 同一套构造：
        # 深亏档零加仓、浮盈档全加仓，标的各不相同，足以判到 ESTABLISHED。
        biased_legacy = (
            _group(30, ticker_prefix="L", rating="Hold", pnl_pct=-25.0)
            + _group(30, ticker_prefix="M", rating="Hold", pnl_pct=-10.0)
            + _group(30, ticker_prefix="F", rating="Hold", pnl_pct=0.0)
            + _group(30, ticker_prefix="G", rating="Overweight", pnl_pct=12.0)
        )
        audit = run_stratified_audit(biased_legacy)
        assert audit.established(ERA_LEGACY), "构造的偏置样本应当在改版前那层被检出"
        assert audit.regressions == []

    def test_no_rows_yields_no_strata_rather_than_an_empty_verdict(self):
        audit = run_stratified_audit([])
        assert audit.strata == {}
        assert audit.counts == {}
        assert audit.regressions == []

    def test_as_dict_is_json_serialisable_and_keyed_by_era(self):
        import json

        audit = run_stratified_audit(
            _group(2) + _group(2, ticker_prefix="N", sizing_method=METHOD_ATR),
        )
        payload = json.loads(json.dumps(audit.as_dict(), ensure_ascii=False))
        assert set(payload["strata"]) == {ERA_LEGACY, ERA_DETERMINISTIC}
        assert payload["strata"][ERA_DETERMINISTIC]["era"] == ERA_DETERMINISTIC
        # 基线对比按 key 逐条匹配，所以每层都必须带着自己的 findings 列表。
        assert payload["strata"][ERA_LEGACY]["findings"]

    def test_era_labels_are_human_readable_for_the_report_header(self):
        assert "改版前" in era_label(ERA_LEGACY)
        assert "改版后" in era_label(ERA_DETERMINISTIC)
        assert era_label("something_else") == "something_else"


@pytest.mark.unit
class TestVocabularyStaysInSync:
    def test_tiers_partition_the_five_tier_scale(self):
        assert ADD_TIERS == {"Buy", "Overweight"}
        assert TRIM_TIERS == {"Underweight", "Sell"}
        assert ADD_TIERS & TRIM_TIERS == set()
        assert set(RATING_DIRECTION) - ADD_TIERS - TRIM_TIERS == {"Hold"}

    def test_direction_signs_match_the_tiers(self):
        assert all(RATING_DIRECTION[t] == 1 for t in ADD_TIERS)
        assert all(RATING_DIRECTION[t] == -1 for t in TRIM_TIERS)
        assert RATING_DIRECTION["Hold"] == 0
