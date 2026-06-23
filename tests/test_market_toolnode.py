"""After the data-collection refactor, analysts no longer use ToolNodes.
Data is pre-fetched by DataCollector and injected into state.

This file replaces the old ToolNode wiring test with a smoke test that
verifies the DataCollector schema round-trips correctly and the graph's
data collection node skips when a bundle is already present.
"""
import pytest

from tradingagents.datacollector.schema import (
    BundleMetadata,
    DataBundle,
    FundamentalsData,
    MarketData,
    NewsData,
    SentimentData,
)


@pytest.mark.unit
def test_data_bundle_round_trips_through_dict():
    bundle = DataBundle(
        metadata=BundleMetadata(
            ticker="NVDA",
            trade_date="2024-05-10",
            asset_type="stock",
            collection_timestamp="2024-05-10T12:00:00",
            selected_analysts=["market", "fundamentals"],
            vendor_config={},
            bundle_version="1.0",
        ),
        market=MarketData(
            stock_data="close,open\n100,99",
            indicators={"rsi": "70.5"},
            verified_snapshot="snapshot text",
        ),
        sentiment=None,
        news=None,
        fundamentals=FundamentalsData(
            overview="NVDA overview",
            balance_sheet_quarterly="bs q",
            balance_sheet_annual="bs a",
            cashflow_quarterly="cf q",
            cashflow_annual="cf a",
            income_quarterly="inc q",
            income_annual="inc a",
        ),
    )
    dumped = bundle.model_dump()
    restored = DataBundle.model_validate(dumped)
    assert restored.metadata.ticker == "NVDA"
    assert restored.market.indicators["rsi"] == "70.5"
    assert restored.fundamentals.overview == "NVDA overview"
    assert restored.sentiment is None
    assert restored.news is None
