"""Pydantic models for the pre-fetched data bundle."""

from __future__ import annotations

from pydantic import BaseModel, Field


class MarketStatus(BaseModel):
    """Verified market/risk-warning status for the analysis date."""

    risk_warning_status: str = "unknown"
    security_name: str = ""
    effective_date: str = ""
    price_limit_ratio: float = 0.0
    trading_status: str = "unknown"
    verified_at: str = ""
    sources: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    conflicts: list[str] = Field(default_factory=list)


class BundleMetadata(BaseModel):
    ticker: str
    trade_date: str
    original_trade_date: str | None = None
    date_correction_reason: str = ""
    asset_type: str = "stock"
    collection_timestamp: str = ""
    analysis_mode: str = "daily"
    intraday_asof: str = ""
    selected_analysts: list[str] = Field(default_factory=list)
    vendor_config: dict = Field(default_factory=dict)
    bundle_version: str = "1.0"
    market_status: MarketStatus = Field(default_factory=MarketStatus)


class MarketData(BaseModel):
    stock_data: str = ""
    indicators: dict[str, str] = Field(default_factory=dict)
    verified_snapshot: str = ""
    intraday_snapshot: str = ""


class SentimentData(BaseModel):
    ticker_news: str = ""
    stocktwits: str = ""
    reddit: str = ""


class NewsData(BaseModel):
    ticker_news: str = ""
    global_news: str = ""
    insider_transactions: str = ""
    macro_indicators: dict[str, str] = Field(default_factory=dict)
    prediction_markets: dict[str, str] = Field(default_factory=dict)
    industry_data: str = ""
    stock_moneyflow: str = ""


class FundamentalsData(BaseModel):
    overview: str = ""
    balance_sheet_quarterly: str = ""
    balance_sheet_annual: str = ""
    cashflow_quarterly: str = ""
    cashflow_annual: str = ""
    income_quarterly: str = ""
    income_annual: str = ""
    competitive_intelligence: str = ""


class DataBundle(BaseModel):
    metadata: BundleMetadata
    market: MarketData | None = None
    sentiment: SentimentData | None = None
    news: NewsData | None = None
    fundamentals: FundamentalsData | None = None
