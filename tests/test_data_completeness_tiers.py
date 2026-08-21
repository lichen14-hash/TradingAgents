"""数据完整性检查的三个盲区 + 全有全无准入。

事故（2026-08-19 批量，8 只标的）：7 只出结论、6 只 Sell，而当天的数据里有三类缺口
一个都没被检出：

1. `intraday_snapshot` 8/8 全部 ConnectionError——盘中模式唯一的增量数据整体失败，
   却因为该字段从不在校验清单里，报告、横幅、数据库都没有留痕。
2. `industry_data` 返回的是"行业涨跌排名数据暂不可用"这种**散文式软失败**：
   provider 自己吞掉异常后返回正常字符串，没有 `<unavailable: …>` 标记、也非空，
   所有"存在性"检查全部通过，模型却读到一段没有数据的文本。
3. `northbound_flow` 是一张格式完好的表，自报"最新数据 (2024-08-16)"——落后 732 天，
   紧挨着日期正确的 `margin_trading`。"有内容"和"是当期内容"是两个问题，只问了第一个。

同时准入是全有全无：web 入口只要有**任何**一项 issue 就 `raise DataIncompleteError`，
一份缺失的年度现金流表就让整轮分析没有报告、没有数据库行。因此检出能力一旦补强，
准入必须同步分级，否则每一轮都会被拒。
"""

import pytest

from tradingagents.agents.utils.integrity import (
    DATA_COMPLETENESS_SECTION_PREFIX,
    data_completeness_findings,
)
from tradingagents.datacollector.collector import (
    KIND_MISSING,
    KIND_PARTIAL,
    KIND_STALE,
    KIND_UNAVAILABLE,
    SEVERITY_BLOCK,
    SEVERITY_WARN,
    classify_bundle_issues,
    validate_bundle_completeness,
)
from tradingagents.datacollector.schema import (
    BundleMetadata,
    DataBundle,
    FundamentalsData,
    MarketData,
    MarketStatus,
    NewsData,
)
from tradingagents.reporting import split_data_completeness_findings

TRADE_DATE = "2026-08-18"

# A verified, non-conflicting status so market-status issues never pollute a
# test about a different field.
_CLEAN_STATUS = MarketStatus(
    security_name="测试股份",
    risk_warning_status="normal",
    effective_date="2025-01-01",
    sources=["test"],
)


def _metadata(**kwargs) -> BundleMetadata:
    params = {
        "ticker": "600000.SS",
        "trade_date": TRADE_DATE,
        "market_status": _CLEAN_STATUS,
    }
    params.update(kwargs)
    return BundleMetadata(**params)


def _bundle(**sections) -> DataBundle:
    meta = sections.pop("metadata", None) or _metadata()
    return DataBundle(metadata=meta, **sections)


def _find(issues, category, field):
    for issue in issues:
        if issue["category"] == category and issue["field"] == field:
            return issue
    return None


@pytest.mark.unit
class TestIntradaySnapshot:
    """盘中模式下快照缺失必须留痕；日线模式下同一个空值不是缺陷。"""

    def test_intraday_mode_missing_snapshot_is_reported(self):
        bundle = _bundle(
            metadata=_metadata(analysis_mode="intraday"),
            market=MarketData(stock_data="Date,Close\n2026-08-18,10.0", intraday_snapshot=""),
        )
        issue = _find(validate_bundle_completeness(bundle), "行情数据", "intraday_snapshot")
        assert issue is not None, "盘中模式取不到盘中快照，必须报"
        assert issue["kind"] == KIND_MISSING

    def test_intraday_mode_unavailable_marker_is_reported(self):
        bundle = _bundle(
            metadata=_metadata(analysis_mode="intraday"),
            market=MarketData(
                stock_data="Date,Close\n2026-08-18,10.0",
                intraday_snapshot="<unavailable: ConnectionError: Connection aborted.>",
            ),
        )
        issue = _find(validate_bundle_completeness(bundle), "行情数据", "intraday_snapshot")
        assert issue is not None
        assert issue["kind"] == KIND_UNAVAILABLE
        assert "ConnectionError" in issue["reason"]

    def test_daily_mode_empty_snapshot_is_not_an_issue(self):
        """日线模式本就不取盘中快照，报它只会训练读者忽略横幅。"""
        bundle = _bundle(
            metadata=_metadata(analysis_mode="daily"),
            market=MarketData(stock_data="Date,Close\n2026-08-18,10.0", intraday_snapshot=""),
        )
        assert _find(
            validate_bundle_completeness(bundle), "行情数据", "intraday_snapshot",
        ) is None


@pytest.mark.unit
class TestSoftFailProse:
    """provider 自己吞掉异常、返回一段"暂不可用"的正常字符串。"""

    @pytest.mark.parametrize("text", [
        "行业涨跌排名数据暂不可用",
        "行业资金流向数据暂不可用",
        "数据不可用：AKShare 接口无返回",
        "No data available for this symbol",
    ])
    def test_soft_fail_phrases_are_detected_as_partial(self, text):
        bundle = _bundle(news=NewsData(industry_data=f"## 行业数据\n{text}\n"))
        issue = _find(validate_bundle_completeness(bundle), "新闻数据", "industry_data")
        assert issue is not None, f"软失败文本未被识别: {text!r}"
        assert issue["kind"] == KIND_PARTIAL
        assert issue["severity"] == SEVERITY_WARN
        assert text.split("：")[0][:6] in issue["reason"]

    def test_reason_carries_the_offending_line_only(self):
        bundle = _bundle(news=NewsData(
            industry_data="## 行业数据\n行业涨跌排名: 电子 +1.2%\n行业资金流向数据暂不可用\n",
        ))
        issue = _find(validate_bundle_completeness(bundle), "新闻数据", "industry_data")
        assert "行业资金流向数据暂不可用" in issue["reason"]
        assert "电子 +1.2%" not in issue["reason"], "只引用出问题的那一行"

    def test_ordinary_report_text_is_not_flagged(self):
        """真实文档里出现"unavailable"一词不能触发误报。"""
        bundle = _bundle(news=NewsData(
            industry_data="Industry rotation was strong; no constituent was unavailable for trading.",
        ))
        assert _find(
            validate_bundle_completeness(bundle), "新闻数据", "industry_data",
        ) is None


@pytest.mark.unit
class TestStaleness:
    """内容齐全但是旧的——northbound_flow 落后 732 天。"""

    def test_two_year_old_daily_feed_is_flagged(self):
        bundle = _bundle(news=NewsData(prediction_markets={
            "northbound_flow": "北向资金\n最新数据 (2024-08-16): 净流入 12.3 亿元",
        }))
        issue = _find(validate_bundle_completeness(bundle), "市场信号", "northbound_flow")
        assert issue is not None
        assert issue["kind"] == KIND_STALE
        assert issue["severity"] == SEVERITY_WARN
        assert "732" in issue["reason"], "落后天数要写进 reason，否则读者无法判断严重性"

    def test_current_daily_feed_is_clean(self):
        bundle = _bundle(news=NewsData(prediction_markets={
            "margin_trading": "融资融券\n最新数据 (2026-08-18): 余额 1.8 万亿",
        }))
        assert validate_bundle_completeness(bundle) == []

    def test_future_dates_do_not_mask_a_stale_feed(self):
        """文本里的前瞻日期（财报日、到期日）不能被当成数据新鲜度。"""
        bundle = _bundle(news=NewsData(prediction_markets={
            "northbound_flow": "北向资金\n最新数据 (2024-08-16)\n下次披露: 2026-12-31",
        }))
        issue = _find(validate_bundle_completeness(bundle), "市场信号", "northbound_flow")
        assert issue is not None and issue["kind"] == KIND_STALE

    def test_irregular_cadence_fields_have_no_budget(self):
        """内部交易、季报天然滞后，给它们设预算只会产生噪音。"""
        bundle = _bundle(news=NewsData(
            insider_transactions="Latest Form 4 filing: 2026-02-03, CFO sold 10,000 shares",
        ))
        assert validate_bundle_completeness(bundle) == []


@pytest.mark.unit
class TestTieredAdmission:
    """分级准入：只有关键字段缺失才值得放弃整轮分析。"""

    def test_price_series_is_blocking(self):
        bundle = _bundle(market=MarketData(
            stock_data="<unavailable: HTTPError: 502>",
            verified_snapshot="收盘价 10.00",
        ))
        blocking, _ = classify_bundle_issues(validate_bundle_completeness(bundle))
        assert [i["field"] for i in blocking] == ["stock_data"]

    def test_single_missing_statement_is_only_a_warning(self):
        """一份缺失的年度现金流表不该让整轮分析没有报告、没有数据库行。"""
        bundle = _bundle(fundamentals=FundamentalsData(
            overview="市值 1800 亿",
            balance_sheet_quarterly="资产合计 ...",
            income_quarterly="营业收入 ...",
            cashflow_annual="<unavailable: KeyError: 'cashflow'>",
        ))
        blocking, warnings = classify_bundle_issues(validate_bundle_completeness(bundle))
        assert blocking == []
        assert [i["field"] for i in warnings] == ["cashflow_annual"]

    def test_all_six_statements_missing_is_blocking(self):
        """六张表全没有的基本面分析师只能编造。"""
        bundle = _bundle(fundamentals=FundamentalsData(overview="市值 1800 亿"))
        blocking, _ = classify_bundle_issues(validate_bundle_completeness(bundle))
        assert [i["field"] for i in blocking] == ["全部财报"]

    def test_issue_without_severity_counts_as_blocking(self):
        """未知形状的 issue 不能被静默降级。"""
        blocking, warnings = classify_bundle_issues(
            [{"category": "x", "field": "y", "reason": "z"}],
        )
        assert len(blocking) == 1 and warnings == []

    def test_2026_08_19_gap_profile_would_not_have_blocked(self):
        """当天的四类缺口都是 warn：分析应当继续，但必须全部留痕。"""
        bundle = _bundle(
            metadata=_metadata(analysis_mode="intraday"),
            market=MarketData(
                stock_data="Date,Close\n2026-08-18,10.0",
                verified_snapshot="收盘价 10.00",
                intraday_snapshot="<unavailable: ConnectionError: Connection aborted.>",
            ),
            news=NewsData(
                industry_data="行业涨跌排名数据暂不可用",
                prediction_markets={"northbound_flow": "最新数据 (2024-08-16): 净流入 12.3 亿"},
            ),
            fundamentals=FundamentalsData(
                overview="市值 1800 亿",
                balance_sheet_quarterly="资产合计 ...",
                balance_sheet_annual="资产合计 ...",
                cashflow_quarterly="经营现金流 ...",
                cashflow_annual="经营现金流 ...",
                income_quarterly="营业收入 ...",
                income_annual="营业收入 ...",
                competitive_intelligence="<unavailable: competitive intelligence search returned no results>",
            ),
        )
        issues = validate_bundle_completeness(bundle)
        blocking, warnings = classify_bundle_issues(issues)
        assert blocking == []
        assert {i["field"] for i in warnings} == {
            "intraday_snapshot", "industry_data", "northbound_flow",
            "competitive_intelligence",
        }


@pytest.mark.unit
class TestFindingsChannel:
    """缺口必须走 integrity_findings，才能进横幅和数据库。"""

    def _bundle_with_gap(self):
        return _bundle(
            metadata=_metadata(analysis_mode="intraday"),
            market=MarketData(
                stock_data="Date,Close\n2026-08-18,10.0",
                intraday_snapshot="<unavailable: ConnectionError: Connection aborted.>",
            ),
        )

    def test_findings_are_emitted_with_section_prefix(self):
        findings = data_completeness_findings(self._bundle_with_gap())
        assert len(findings) == 1
        finding = findings[0]
        assert finding["section"].startswith(DATA_COMPLETENESS_SECTION_PREFIX)
        assert set(finding) == {"section", "reason", "snippet", "severity"}
        assert "intraday_snapshot" in finding["snippet"]
        assert f"kind={KIND_UNAVAILABLE}" in finding["snippet"]

    def test_serialised_bundle_yields_the_same_findings(self):
        """图状态里带的是 model_dump() 后的 dict，两条路径必须一致。"""
        bundle = self._bundle_with_gap()
        assert data_completeness_findings(bundle.model_dump()) == \
            data_completeness_findings(bundle)

    @pytest.mark.parametrize("value", [None, {}])
    def test_absent_bundle_is_not_a_finding(self, value):
        assert data_completeness_findings(value) == []

    def test_clean_bundle_emits_nothing(self):
        """干净的运行不能产生 findings，否则 integrity_flags 永远非空。"""
        bundle = _bundle(market=MarketData(
            stock_data="Date,Close\n2026-08-18,10.0",
            verified_snapshot="收盘价 10.00",
        ))
        assert data_completeness_findings(bundle) == []

    def test_split_keeps_completeness_out_of_the_run_banner(self):
        """报告已有逐字段的完整性表格，同一个缺口不该再出现在流程横幅里。"""
        completeness = data_completeness_findings(self._bundle_with_gap())
        other = {"section": "跨阶段一致性", "reason": "方向相反", "snippet": "",
                 "severity": SEVERITY_WARN}
        comp, rest = split_data_completeness_findings(completeness + [other])
        assert comp == completeness
        assert rest == [other]

    def test_blocking_severity_survives_into_the_finding(self):
        bundle = _bundle(market=MarketData(
            stock_data="<unavailable: HTTPError: 502>",
            verified_snapshot="收盘价 10.00",
        ))
        findings = data_completeness_findings(bundle)
        assert [f["severity"] for f in findings] == [SEVERITY_BLOCK]
