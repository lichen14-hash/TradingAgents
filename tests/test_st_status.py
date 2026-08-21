import pytest

from tradingagents.agents.utils.agent_utils import build_instrument_context
from tradingagents.agents.utils.market_status_guard import (
    MarketStatusConflictError,
    find_market_status_conflicts,
    raise_if_market_status_conflicts,
)
from tradingagents.datacollector.collector import validate_bundle_completeness
from tradingagents.datacollector.schema import BundleMetadata, DataBundle, MarketStatus
from tradingagents.dataflows import st_status_provider
from tradingagents.dataflows.st_status_provider import annotate_historical_st_mentions


def _disable_live_sources(monkeypatch):
    for name in (
        "_fetch_akshare_name",
        "_fetch_eastmoney_name",
        "_fetch_sina_name",
        "_fetch_tencent_name",
    ):
        monkeypatch.setattr(st_status_provider, name, lambda ticker: None)
    st_status_provider.clear_market_status_cache()


@pytest.mark.unit
def test_century_huatong_is_normal_after_risk_warning_removed(monkeypatch):
    _disable_live_sources(monkeypatch)

    status = st_status_provider.resolve_market_status("002602.SZ", "2026-07-30")

    assert status.risk_warning_status == "normal"
    assert status.security_name == "世纪华通"
    assert status.effective_date == "2025-11-12"
    assert status.price_limit_ratio == 10.0
    assert status.conflicts == []


@pytest.mark.unit
def test_century_huatong_historical_status_uses_trade_date(monkeypatch):
    _disable_live_sources(monkeypatch)

    before = st_status_provider.resolve_market_status("002602.SZ", "2025-11-11")
    after = st_status_provider.resolve_market_status("002602.SZ", "2025-11-12")

    assert before.risk_warning_status == "ST"
    assert before.security_name == "ST华通"
    assert before.price_limit_ratio == 5.0
    assert after.risk_warning_status == "normal"
    assert after.security_name == "世纪华通"
    assert after.price_limit_ratio == 10.0


@pytest.mark.unit
def test_true_st_name_gets_five_percent_limit(monkeypatch):
    st_status_provider.clear_market_status_cache()
    monkeypatch.setattr(
        st_status_provider,
        "_fetch_akshare_name",
        lambda ticker: {
            "source": "akshare:test",
            "security_name": "ST测试",
            "risk_warning_status": "ST",
            "effective_date": "2026-01-01",
        },
    )
    monkeypatch.setattr(
        st_status_provider,
        "_fetch_eastmoney_name",
        lambda ticker: {
            "source": "eastmoney:test",
            "security_name": "ST测试",
            "risk_warning_status": "ST",
            "effective_date": "2026-01-01",
        },
    )
    monkeypatch.setattr(st_status_provider, "_fetch_sina_name", lambda ticker: None)
    monkeypatch.setattr(st_status_provider, "_fetch_tencent_name", lambda ticker: None)

    status = st_status_provider.resolve_market_status("000001.SZ", "2026-07-30")

    assert status.risk_warning_status == "ST"
    assert status.price_limit_ratio == 5.0
    assert status.confidence == 1.0


@pytest.mark.unit
def test_source_conflict_returns_unknown_and_fails_completeness(monkeypatch):
    st_status_provider.clear_market_status_cache()
    monkeypatch.setattr(
        st_status_provider,
        "_fetch_akshare_name",
        lambda ticker: {
            "source": "akshare:test",
            "security_name": "ST冲突",
            "risk_warning_status": "ST",
            "effective_date": "2026-01-01",
        },
    )
    monkeypatch.setattr(
        st_status_provider,
        "_fetch_eastmoney_name",
        lambda ticker: {
            "source": "eastmoney:test",
            "security_name": "冲突股份",
            "risk_warning_status": "normal",
            "effective_date": "2026-01-01",
        },
    )
    monkeypatch.setattr(st_status_provider, "_fetch_sina_name", lambda ticker: None)
    monkeypatch.setattr(st_status_provider, "_fetch_tencent_name", lambda ticker: None)

    status = st_status_provider.resolve_market_status("000001.SZ", "2026-07-30")
    bundle = DataBundle(
        metadata=BundleMetadata(ticker="000001.SZ", trade_date="2026-07-30", market_status=status)
    )

    assert status.risk_warning_status == "unknown"
    assert status.conflicts
    issues = validate_bundle_completeness(bundle)
    assert any(i["category"] == "市场状态" for i in issues)


@pytest.mark.unit
def test_unverified_a_share_status_fails_completeness(monkeypatch):
    _disable_live_sources(monkeypatch)

    status = st_status_provider.resolve_market_status("000001.SZ", "2026-07-30")
    bundle = DataBundle(
        metadata=BundleMetadata(ticker="000001.SZ", trade_date="2026-07-30", market_status=status)
    )

    assert status.risk_warning_status == "unknown"
    issues = validate_bundle_completeness(bundle)
    assert any(i["field"] == "risk_warning_status" for i in issues)


@pytest.mark.unit
def test_context_for_normal_status_forbids_current_st_assumptions():
    context = build_instrument_context(
        "002602.SZ",
        "stock",
        {"company_name": "世纪华通", "exchange": "深交所主板"},
        market_status=MarketStatus(
            risk_warning_status="normal",
            security_name="世纪华通",
            effective_date="2025-11-12",
            price_limit_ratio=10.0,
            trading_status="normal",
            verified_at="2026-07-31T12:00:00+08:00",
            sources=["known_announcement:test"],
            confidence=1.0,
        ).model_dump(),
    )

    assert "Risk-warning/ST status: normal" in context
    assert "do not describe the instrument as currently ST" in context
    assert "do not apply ST-only 5% price-limit" in context


@pytest.mark.unit
def test_final_state_conflict_guard_blocks_report_when_normal_status_is_contradicted():
    bundle = DataBundle(
        metadata=BundleMetadata(
            ticker="002602.SZ",
            trade_date="2026-07-30",
            market_status=MarketStatus(
                risk_warning_status="normal",
                security_name="世纪华通",
                effective_date="2025-11-12",
                price_limit_ratio=10.0,
                trading_status="normal",
                verified_at="2026-07-31T12:00:00+08:00",
                sources=["known_announcement:test"],
                confidence=1.0,
            ),
        )
    )
    final_state = {
        "final_trade_decision": "当前属于ST股，适用5%涨跌幅限制，建议Sell。",
    }

    conflicts = find_market_status_conflicts(final_state["final_trade_decision"], bundle.metadata.market_status)
    assert conflicts
    with pytest.raises(MarketStatusConflictError):
        raise_if_market_status_conflicts(final_state, bundle)


@pytest.mark.unit
def test_final_state_conflict_guard_allows_explicit_denials_and_corrections():
    status = MarketStatus(risk_warning_status="normal")
    corrections = [
        "当前为正常状态（非ST，涨跌幅10%），历史信息不影响当前规则。",
        "这只ETF正常交易、涨跌幅限制10%，不是ST的5%，更没有一字跌停风险。",
        "本标的非ST，未使用任何ST专属5%限价或一字跌停假设。",
        "迈瑞医疗当前为正常交易状态、非ST、涨跌幅限制20%。",
        "该证券当前正常交易，无ST风险警示。",
        "跳空/一字跌停可能使止损失效，需要控制仓位。",
        "考虑跌停风险下的仓位管理，不改变已验证的正常市场状态。",
        "不同于ST股的5%限制，该股有20%涨跌幅限制。",
        "不同于ST股的5%涨跌幅限制，该股有20%涨跌幅限制。",
        "当前风险警示状态为“正常”，历史ST信息仅作历史参考，不得据此套用5%涨跌幅或一字跌停假设。",
    ]

    for text in corrections:
        assert find_market_status_conflicts(text, status) == []


@pytest.mark.unit
def test_final_state_conflict_guard_still_finds_later_positive_claim():
    status = MarketStatus(risk_warning_status="normal", price_limit_ratio=20.0)
    text = "该公司并非ST。当前属于ST股，存在5%涨跌幅限制。"

    conflicts = find_market_status_conflicts(text, status)

    assert conflicts
    assert any(issue["reason"] == "把当前状态写成ST" for issue in conflicts)


@pytest.mark.unit
def test_normal_a_share_allows_generic_limit_down_tail_risk():
    bundle = DataBundle(
        metadata=BundleMetadata(
            ticker="300760.SZ",
            trade_date="2026-08-04",
            market_status=MarketStatus(
                risk_warning_status="normal",
                price_limit_ratio=20.0,
            ),
        )
    )
    final_state = {
        "final_trade_decision": "跳空/一字跌停可能使止损失效，因此需要控制仓位。",
    }

    raise_if_market_status_conflicts(final_state, bundle)


@pytest.mark.unit
def test_normal_a_share_wrong_price_limit_is_warning_not_block():
    bundle = DataBundle(
        metadata=BundleMetadata(
            ticker="300760.SZ",
            trade_date="2026-08-04",
            market_status=MarketStatus(
                risk_warning_status="normal",
                price_limit_ratio=20.0,
            ),
        )
    )
    final_state = {"market_report": "该股当前适用5%涨跌幅限制。"}

    conflicts = find_market_status_conflicts(final_state["market_report"], bundle.metadata.market_status)
    assert any("20%" in issue["reason"] for issue in conflicts)
    assert all(issue["severity"] == "warn" for issue in conflicts)

    # Warn-level findings must NOT discard the completed analysis.
    warnings = raise_if_market_status_conflicts(final_state, bundle)
    assert warnings and warnings[0]["section"] == "market_report"


@pytest.mark.unit
def test_current_st_claim_is_blocking():
    bundle = DataBundle(
        metadata=BundleMetadata(
            ticker="300760.SZ",
            trade_date="2026-08-04",
            market_status=MarketStatus(
                risk_warning_status="normal",
                price_limit_ratio=20.0,
            ),
        )
    )
    final_state = {"market_report": "该股当前属于ST股。"}

    with pytest.raises(MarketStatusConflictError):
        raise_if_market_status_conflicts(final_state, bundle)


@pytest.mark.unit
def test_etf_market_status_skips_st_sources_and_guard(monkeypatch):
    _disable_live_sources(monkeypatch)
    status = st_status_provider.resolve_market_status("515880.SS", "2026-07-30")
    bundle = DataBundle(
        metadata=BundleMetadata(
            ticker="515880.SS",
            trade_date="2026-07-30",
            market_status=status,
        )
    )

    assert status.risk_warning_status == "normal"
    assert status.price_limit_ratio == 10.0
    assert status.sources == ["market_rules:etf_no_st_regime"]
    raise_if_market_status_conflicts(
        {"final_trade_decision": "ETF不是ST，不适用ST专属5%或一字跌停假设。"},
        bundle,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("ticker", "text"),
    [
        ("09988.HK", "当前为正常交易状态，无 ST/风险警示，无特殊涨跌停限制。"),
        ("09992.HK", "当前状态正常，无风险警示/ST，无特殊涨跌幅限制。"),
    ],
)
def test_non_a_share_market_status_skips_a_share_st_guard(ticker, text):
    bundle = DataBundle(
        metadata=BundleMetadata(
            ticker=ticker,
            trade_date="2026-08-04",
            market_status=MarketStatus(risk_warning_status="normal"),
        )
    )

    raise_if_market_status_conflicts({"market_report": text}, bundle)


_NORMAL_STATUS = MarketStatus(
    risk_warning_status="normal",
    security_name="世纪华通",
    effective_date="2025-11-12",
    price_limit_ratio=10.0,
)


@pytest.mark.unit
def test_annotate_historical_st_tags_normal_a_share_news():
    text = "ST华通发布公告，公司经营改善。"

    result = annotate_historical_st_mentions(text, "002602.SZ", _NORMAL_STATUS)

    assert result.startswith("> [数据消毒]")
    assert "当前风险警示/ST状态为“正常”" in result
    assert "ST华通（历史简称，风险警示已解除）" in result
    # Idempotent: annotating twice must not double-tag.
    assert annotate_historical_st_mentions(result, "002602.SZ", _NORMAL_STATUS) == result


@pytest.mark.unit
def test_annotate_historical_st_skips_non_applicable_cases():
    st_status = MarketStatus(risk_warning_status="ST", security_name="ST测试")
    plain = "公司发布年报，业绩稳健。"
    st_text = "ST板块今日普涨。"

    # Currently-ST stock: text must stay untouched.
    assert annotate_historical_st_mentions(st_text, "000001.SZ", st_status) == st_text
    # No ST mention at all: untouched.
    assert annotate_historical_st_mentions(plain, "002602.SZ", _NORMAL_STATUS) == plain
    # HK stock and ETF: ST regime does not apply, untouched.
    assert annotate_historical_st_mentions(st_text, "09988.HK", _NORMAL_STATUS) == st_text
    assert annotate_historical_st_mentions(st_text, "515880.SS", _NORMAL_STATUS) == st_text
    # Empty text passthrough.
    assert annotate_historical_st_mentions("", "002602.SZ", _NORMAL_STATUS) == ""
