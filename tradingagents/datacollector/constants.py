"""Data collection constants — indicator sets, macro series, and prediction queries."""

ALL_INDICATORS: tuple[str, ...] = (
    "close_50_sma",
    "close_200_sma",
    "close_10_ema",
    "macd",
    "macds",
    "macdh",
    "rsi",
    "boll",
    "boll_ub",
    "boll_lb",
    "atr",
    "vwma",
    "mfi",
)

DEFAULT_MACRO_INDICATORS: tuple[str, ...] = (
    "fed_funds_rate",
    "2y_treasury",
    "10y_treasury",
    "30y_treasury",
    "yield_curve",
    "cpi",
    "core_cpi",
    "pce",
    "core_pce",
    "inflation_expectations",
    "real_gdp",
    "gdp",
    "industrial_production",
    "unemployment",
    "nonfarm_payrolls",
    "initial_claims",
    "m2",
    "vix",
    "dollar_index",
    "consumer_sentiment",
    "housing_starts",
    "retail_sales",
)

DEFAULT_PREDICTION_QUERIES: tuple[str, ...] = (
    "Fed rate cut",
    "recession",
    "inflation",
    "stock market crash",
    "geopolitical conflict",
)

CN_MACRO_INDICATORS: tuple[str, ...] = (
    "lpr_1y",
    "lpr_5y",
    "mlf_rate",
    "shibor_overnight",
    "rrr",
    "cn_cpi",
    "cn_ppi",
    "cn_pmi_mfg",
    "cn_pmi_non_mfg",
    "cn_m2",
    "cn_m1",
    "social_financing",
    "new_yuan_loans",
    "cn_gdp",
    "cn_industrial_production",
    "cn_fixed_asset_investment",
    "cn_retail_sales",
    "cn_forex_reserves",
    "cn_trade_balance",
    "cn_housing_price",
    "cn_10y_treasury",
    "cn_1y_treasury",
    "cn_unemployment",
)

CN_PREDICTION_QUERIES: tuple[str, ...] = (
    "northbound_flow",
    "margin_trading",
    "top_institutional",
)

CN_GLOBAL_NEWS_QUERIES: list[str] = [
    "央行货币政策 利率 中国",
    "中国 GDP 经济增长 展望",
    "A股 沪深 市场",
    "中美 贸易 关税",
    "中国 房地产 市场",
]

HK_MACRO_INDICATORS: tuple[str, ...] = (
    "hk_cpi",
    "hk_ppi",
    "hk_unemployment",
    "hk_gdp",
    "hk_gdp_rate",
    "hk_trade_balance",
    "cn_pmi_mfg",
    "cn_cpi",
    "cn_gdp",
    "cn_m2",
    "cn_trade_balance",
)

HK_PREDICTION_QUERIES: tuple[str, ...] = (
    "southbound_flow",
    "hk_connect_summary",
    "ah_premium",
)

HK_GLOBAL_NEWS_QUERIES: list[str] = [
    "香港 金管局 利率",
    "港股 恒生指数 市场",
    "中国 GDP 经济增长 展望",
    "中美 贸易 关税",
    "香港 房地产 楼市",
]

BUNDLE_VERSION = "1.0"
