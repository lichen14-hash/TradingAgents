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
        "final_trade_decision": "因为ST股5%涨跌幅和一字跌停风险，建议Sell。",
    }

    conflicts = find_market_status_conflicts(final_state["final_trade_decision"], bundle.metadata.market_status)
    assert conflicts
    with pytest.raises(MarketStatusConflictError):
        raise_if_market_status_conflicts(final_state, bundle)
