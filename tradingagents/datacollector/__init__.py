"""Data collection module — pre-fetches all data for TradingAgents analysis."""

from .collector import (
    DataCollector,
    DataIncompleteError,
    classify_bundle_issues,
    validate_bundle_completeness,
)
from .constants import (
    ALL_INDICATORS,
    BUNDLE_VERSION,
    DEFAULT_MACRO_INDICATORS,
    DEFAULT_PREDICTION_QUERIES,
)
from .schema import (
    BundleMetadata,
    DataBundle,
    FundamentalsData,
    MarketData,
    MarketStatus,
    NewsData,
    SentimentData,
)

__all__ = [
    "DataCollector",
    "DataIncompleteError",
    "classify_bundle_issues",
    "validate_bundle_completeness",
    "DataBundle",
    "BundleMetadata",
    "MarketData",
    "MarketStatus",
    "SentimentData",
    "NewsData",
    "FundamentalsData",
    "ALL_INDICATORS",
    "DEFAULT_MACRO_INDICATORS",
    "DEFAULT_PREDICTION_QUERIES",
    "BUNDLE_VERSION",
]
