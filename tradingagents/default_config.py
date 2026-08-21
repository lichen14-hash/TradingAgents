import os

# Data root: project-local so everything migrates with the repo.
# Override via TRADINGAGENTS_HOME env var if needed.
_TRADINGAGENTS_HOME = os.getenv(
    "TRADINGAGENTS_HOME",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "local_data"),
)

# Single source of truth for env-var → config-key overrides. To expose
# a new config key for environment-based override, add a row here — no
# entry-point script changes required. Coercion is driven by the type
# of the existing default, so users can keep writing plain strings in
# their .env file.
_ENV_OVERRIDES = {
    "TRADINGAGENTS_LLM_PROVIDER":         "llm_provider",
    "TRADINGAGENTS_DEEP_THINK_LLM":       "deep_think_llm",
    "TRADINGAGENTS_QUICK_THINK_LLM":      "quick_think_llm",
    "TRADINGAGENTS_LLM_BACKEND_URL":      "backend_url",
    "TRADINGAGENTS_OUTPUT_LANGUAGE":      "output_language",
    "TRADINGAGENTS_MAX_DEBATE_ROUNDS":    "max_debate_rounds",
    "TRADINGAGENTS_MAX_RISK_ROUNDS":      "max_risk_discuss_rounds",
    "TRADINGAGENTS_CHECKPOINT_ENABLED":   "checkpoint_enabled",
    "TRADINGAGENTS_BENCHMARK_TICKER":     "benchmark_ticker",
    "TRADINGAGENTS_TEMPERATURE":          "temperature",
    "TRADINGAGENTS_DECISION_TEMPERATURE": "decision_temperature",
    "TRADINGAGENTS_DEBATE_TEMPERATURE":   "debate_temperature",
    "TRADINGAGENTS_MAX_TOKENS":           "max_tokens",
    "TRADINGAGENTS_LLM_STREAMING":        "llm_streaming",
    "TRADINGAGENTS_LLM_TIMEOUT":          "llm_request_timeout",
    "TRADINGAGENTS_DATA_DIR":             "data_dir",
}

# Position-sizing knobs live one level down (config["position_sizing"][key]), so
# they need their own env table rather than a row in _ENV_OVERRIDES.
_POSITION_SIZING_ENV = {
    "TRADINGAGENTS_RISK_BUDGET_PCT":     "risk_budget_pct",
    "TRADINGAGENTS_STOP_ATR_MULTIPLE":   "stop_atr_multiple",
    "TRADINGAGENTS_MAX_SINGLE_NAME_PCT": "max_single_name_pct",
    "TRADINGAGENTS_FALLBACK_STOP_PCT":   "fallback_stop_pct",
}

# Same one-level-down treatment for the portfolio block (config["portfolio"]).
_PORTFOLIO_ENV = {
    "TRADINGAGENTS_MAX_TOTAL_EQUITY_PCT":   "max_total_equity_pct",
    "TRADINGAGENTS_CONCENTRATION_WARN_PCT": "concentration_warn_pct",
    "TRADINGAGENTS_ASSUME_UNCOMMITTED_IS_CASH": "assume_uncommitted_is_cash",
    "TRADINGAGENTS_MIN_ADD_PCT":            "min_add_pct",
}


def _coerce(value: str, reference):
    """Coerce env-var string to the type of the existing default value."""
    if isinstance(reference, bool):
        return value.strip().lower() in ("true", "1", "yes", "on")
    if isinstance(reference, int) and not isinstance(reference, bool):
        return int(value)
    if isinstance(reference, float):
        return float(value)
    return value


_DATA_VENDOR_ENV = {
    "TRADINGAGENTS_STOCK_VENDOR":       "core_stock_apis",
    "TRADINGAGENTS_INDICATOR_VENDOR":   "technical_indicators",
    "TRADINGAGENTS_FUNDAMENTAL_VENDOR": "fundamental_data",
    "TRADINGAGENTS_NEWS_VENDOR":        "news_data",
}


def _apply_env_overrides(config: dict) -> dict:
    """Apply TRADINGAGENTS_* env vars to the config dict in-place."""
    for env_var, key in _ENV_OVERRIDES.items():
        raw = os.environ.get(env_var)
        if raw is None or raw == "":
            continue
        config[key] = _coerce(raw, config.get(key))
    for env_var, category in _DATA_VENDOR_ENV.items():
        raw = os.environ.get(env_var)
        if raw is not None and raw != "":
            config.setdefault("data_vendors", {})[category] = raw
    for env_var, key in _POSITION_SIZING_ENV.items():
        raw = os.environ.get(env_var)
        if raw is None or raw == "":
            continue
        block = config.setdefault("position_sizing", {})
        block[key] = _coerce(raw, block.get(key))
    for env_var, key in _PORTFOLIO_ENV.items():
        raw = os.environ.get(env_var)
        if raw is None or raw == "":
            continue
        block = config.setdefault("portfolio", {})
        block[key] = _coerce(raw, block.get(key))
    return config


DEFAULT_CONFIG = _apply_env_overrides({
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", os.path.join(_TRADINGAGENTS_HOME, "logs")),
    "data_cache_dir": os.getenv("TRADINGAGENTS_CACHE_DIR", os.path.join(_TRADINGAGENTS_HOME, "cache")),
    "memory_log_path": os.getenv("TRADINGAGENTS_MEMORY_LOG_PATH", os.path.join(_TRADINGAGENTS_HOME, "memory", "trading_memory.md")),
    # Optional cap on the number of resolved memory log entries. When set,
    # the oldest resolved entries are pruned once this limit is exceeded.
    # Pending entries are never pruned. None disables rotation entirely.
    "memory_log_max_entries": None,
    # LLM settings
    "llm_provider": "openai",
    "deep_think_llm": "gpt-5.5",
    "quick_think_llm": "gpt-5.4-mini",
    # When None, each provider's client falls back to its own default endpoint
    # (api.openai.com for OpenAI, generativelanguage.googleapis.com for Gemini, ...).
    # The CLI overrides this per provider when the user picks one. Keeping a
    # provider-specific URL here would leak (e.g. OpenAI's /v1 was previously
    # being forwarded to Gemini, producing malformed request URLs).
    "backend_url": None,
    # Provider-specific thinking configuration
    "google_thinking_level": None,      # "high", "minimal", etc.
    "openai_reasoning_effort": None,    # "medium", "high", "low"
    "anthropic_effort": None,           # "high", "medium", "low"
    # Sampling temperature, forwarded to every provider when set. None leaves
    # each provider at its own default. Lower values reduce run-to-run
    # variation on models that honor it; reasoning models largely ignore it
    # and no setting makes LLM output bit-identical across runs (see README).
    # When set, this global value overrides both per-tier temperatures below.
    "temperature": None,
    # Per-tier sampling temperatures. Decision/factual roles (analysts,
    # research manager, trader, portfolio manager, signal processor) run
    # cold so the same evidence maps to the same rating across runs;
    # debate roles (bull/bear researchers, risk debators) stay warm to
    # preserve adversarial diversity. None (default) leaves the provider
    # default for that tier. CAUTION: thinking-enabled Claude models
    # (e.g. claude-opus-4-x via the idealab proxy) reject any temperature
    # other than 1 with a 400/MPE-001 error, so only opt in on models
    # that accept custom temperatures.
    "decision_temperature": None,
    "debate_temperature": None,
    # Maximum output tokens per LLM call, forwarded to every provider when set.
    # Leave at None: an explicit value here would be applied to every model,
    # including ones whose real cap is lower (claude-3-opus is 4096), turning a
    # truncation fix into a 400 error.
    #
    # None does NOT mean "unlimited" — each provider resolves its own default,
    # and langchain-anthropic resolves it by looking the model up in the profile
    # table bundled with the installed package, falling back to 4096 on a miss.
    # Any model newer than that package (claude-opus-5, claude-sonnet-5) hits the
    # fallback: a 16x silent downgrade from the 64000 the same package gives
    # claude-opus-4-5. With thinking enabled, reasoning and the visible answer
    # share that budget, so reasoning consumes it and the agent returns empty
    # text or a tool_use block truncated mid-JSON — the root cause of the
    # 2026-08-11 300760.SZ incident.
    #
    # That trap is now closed at the client instead of here:
    # anthropic_client._resolve_default_max_tokens supplies a generous floor for
    # models the library does not recognize (and stays out of the way for the
    # ones it does), and NormalizedChatAnthropic retries once with a doubled
    # budget when a response comes back with stop_reason="max_tokens". Set this
    # knob only to override that per deployment.
    "max_tokens": None,
    # Transport settings for long agent turns. None means "let the provider
    # client decide": the Anthropic client streams by default and applies a
    # 300s gap deadline, the OpenAI-compatible client leaves both alone
    # (several proxies reject stream=true alongside structured output).
    #
    # Why streaming matters here: a non-streaming request that generates tens
    # of thousands of tokens holds an idle socket for minutes, and every hop in
    # front of the model kills it at its own idle timeout — that is what cost
    # 3 of 8 tickers on 2026-08-18 (504 Gateway Time-out / MPE-001) and
    # 09988.HK on 2026-08-19 ("Connection error." after ~2h). With streaming,
    # ``llm_request_timeout`` bounds the gap between chunks instead of the whole
    # response; if you turn streaming off, raise it well above the slowest turn.
    "llm_streaming": None,
    "llm_request_timeout": None,
    # Checkpoint/resume: when True, LangGraph saves state after each node
    # so a crashed run can resume from the last successful step.
    "checkpoint_enabled": False,
    # Output language for analyst reports and final decision
    # Internal agent debate stays in English for reasoning quality
    "output_language": "English",
    # Debate and discussion settings
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,
    "analyst_concurrency_limit": 1,
    # News / data fetching parameters
    # Increase for longer lookback strategies or to broaden macro coverage;
    # decrease to reduce token usage in agent prompts.
    "news_article_limit": 20,             # max articles per ticker (ticker-news)
    "global_news_article_limit": 10,      # max articles for global/macro news
    "global_news_lookback_days": 7,       # macro news lookback window
    # Search queries used by get_global_news for macro headlines. Extend or
    # replace to broaden geographic / sector coverage.
    "global_news_queries": [
        "Federal Reserve interest rates inflation",
        "S&P 500 earnings GDP economic outlook",
        "geopolitical risk trade war sanctions",
        "ECB Bank of England BOJ central bank policy",
        "oil commodities supply chain energy",
    ],
    # Data collection settings
    "data_dir": os.getenv("TRADINGAGENTS_DATA_DIR", os.path.join(_TRADINGAGENTS_HOME, "data")),
    "save_data_bundle": True,
    "standard_macro_indicators": [
        "fed_funds_rate", "2y_treasury", "10y_treasury", "30y_treasury",
        "yield_curve", "cpi", "core_cpi", "pce", "core_pce",
        "inflation_expectations", "real_gdp", "gdp", "industrial_production",
        "unemployment", "nonfarm_payrolls", "initial_claims", "m2",
        "vix", "dollar_index", "consumer_sentiment", "housing_starts",
        "retail_sales",
    ],
    "standard_prediction_queries": [
        "Fed rate cut", "recession", "inflation",
        "stock market crash", "geopolitical conflict",
    ],
    "cn_macro_indicators": [
        "lpr_1y", "lpr_5y", "shibor_overnight", "rrr",
        "cn_cpi", "cn_ppi", "cn_pmi_mfg", "cn_pmi_non_mfg",
        "cn_m2", "cn_m1", "social_financing", "new_yuan_loans",
        "cn_gdp", "cn_industrial_production", "cn_fixed_asset_investment",
        "cn_retail_sales", "cn_forex_reserves", "cn_trade_balance",
        "cn_housing_price", "cn_10y_treasury", "cn_1y_treasury", "cn_unemployment",
    ],
    "cn_prediction_queries": [
        "northbound_flow", "margin_trading", "top_institutional",
    ],
    "cn_global_news_queries": [
        "央行货币政策 利率 中国",
        "中国 GDP 经济增长 展望",
        "A股 沪深 市场",
        "中美 贸易 关税",
        "中国 房地产 市场",
    ],
    "hk_macro_indicators": [
        "hk_hibor", "hk_exchange_rate", "hk_monetary_base",
        "us_treasury", "hk_rmb_hibor",
        "cn_pmi_mfg", "cn_cpi", "cn_gdp", "cn_m2", "cn_trade_balance",
    ],
    "hk_prediction_queries": [
        "southbound_flow", "hk_connect_summary", "ah_premium",
    ],
    "hk_global_news_queries": [
        "香港 金管局 利率",
        "港股 恒生指数 市场",
        "中国 GDP 经济增长 展望",
        "中美 贸易 关税",
        "香港 房地产 楼市",
    ],
    # Data vendor configuration
    # Category-level configuration (default for all tools in category).
    # The configured value is the exact vendor chain — requests are NOT silently
    # routed to vendors you didn't choose. For ordered fallback, list several,
    # e.g. "yfinance,alpha_vantage". "default" uses all available vendors.
    "data_vendors": {
        "core_stock_apis": "yfinance",       # Options: alpha_vantage, yfinance
        "technical_indicators": "yfinance",  # Options: alpha_vantage, yfinance
        "fundamental_data": "yfinance",      # Options: alpha_vantage, yfinance
        "news_data": "yfinance",             # Options: alpha_vantage, yfinance
        "macro_data": "fred",                # Options: fred (needs FRED_API_KEY)
        "prediction_markets": "polymarket",  # Options: polymarket (keyless)
    },
    # Tool-level configuration (takes precedence over category-level)
    "tool_vendors": {
        # Example: "get_stock_data": "alpha_vantage",  # Override category default
    },
    # Benchmark for alpha calculation in the reflection layer.
    # ``benchmark_ticker`` (when set) overrides the suffix map for all
    # tickers; leave it None to use ``benchmark_map`` for auto-detection
    # based on the ticker's exchange suffix. SPY remains the US default
    # so the reflection label keeps reading "Alpha vs SPY" for US tickers
    # while non-US tickers get their regional index automatically.
    # Backtest / daily accumulation settings
    "backtest_db_path": os.getenv(
        "TRADINGAGENTS_BACKTEST_DB",
        os.path.join(_TRADINGAGENTS_HOME, "backtest.db"),
    ),
    "backtest_holding_days": 5,
    "backtest_direction_threshold": 0.02,
    "backtest_feedback_enabled": False,

    # Position sizing. The target weight is computed from these, deterministically,
    # instead of being chosen by the model — see
    # tradingagents/agents/utils/position_sizing.py for the measured bias that
    # motivated the change (the model was anchoring the size on the user's
    # unrealised loss, p=0.00012 over 327 predictions).
    #
    # Calibrated over all 52 bundles in test_output/: these values put the
    # per-name ceiling between 2.55% and 15.00% with a 5.76% median, and only
    # 1 of 52 names reaches max_single_name_pct — i.e. the risk budget is what
    # normally binds and the cap is a safety valve. Raising risk_budget_pct
    # scales every position linearly; raising stop_atr_multiple widens stops and
    # therefore shrinks positions.
    "position_sizing": {
        "risk_budget_pct": 1.0,       # portfolio drawdown tolerated per name, %
        "stop_atr_multiple": 3.0,     # stop distance in ATRs
        "max_single_name_pct": 15.0,  # single-name ceiling, %
        "fallback_stop_pct": 12.0,    # stop distance assumed when ATR is missing, %
        # Share of the per-name ceiling each bullish tier takes, and the share of
        # the risk-corrected weight each bearish tier keeps.
        "conviction": {"Buy": 1.00, "Overweight": 0.80},
        "trim": {"Underweight": 0.60, "Sell": 0.00},
    },

    # Portfolio level (all N names at once) — see
    # tradingagents/portfolio/allocator.py.
    #
    # Measured motivation: the 2026-08-20 batch held 79.92% across six names
    # with one at 49% against a 7.37% risk ceiling, and no report said so.
    #
    # The first two are *reporting thresholds only* — the allocator says something
    # when they are crossed, it does not clamp anything to them. In particular
    # max_total_equity_pct is deliberately NOT a budget cap: the add budget is
    # plain arithmetic ("how much cash is there"), never a policy about how much
    # cash to keep. Σtarget is allowed to reach 100%.
    "portfolio": {
        "max_total_equity_pct": 90.0,     # Σ current above this → warning
        "concentration_warn_pct": 25.0,   # one name ≥ this share of the invested book → warning
        # Whether the share of the book not covered by the submitted names may be
        # treated as cash. True is an assumption, not a fact — the user may hold
        # other positions that were never submitted — so every add carries a
        # finding saying so. Set False to fund adds only from the trims in the
        # same batch (self-financed rebalancing), which needs no assumption.
        "assume_uncommitted_is_cash": True,
        # Rationed adds below this many percentage points are dropped to zero: a
        # +0.2pp add does not repay its transaction cost, and
        # expected_rating_for_position_change would label it "Hold" anyway.
        "min_add_pct": 0.5,
    },

    "benchmark_ticker": None,
    "benchmark_map": {
        ".NS":  "^NSEI",       # NSE India (Nifty 50)
        ".BO":  "^BSESN",      # BSE India (Sensex)
        ".T":   "^N225",       # Tokyo (Nikkei 225)
        ".HK":  "^HSI",        # Hong Kong (Hang Seng)
        ".L":   "^FTSE",       # London (FTSE 100)
        ".TO":  "^GSPTSE",     # Toronto (TSX Composite)
        ".AX":  "^AXJO",       # Australia (ASX 200)
        ".SS":  "000001.SS",   # Shanghai (SSE Composite)
        ".SZ":  "399001.SZ",   # Shenzhen (SZSE Component)
        "":     "SPY",         # default for US-listed tickers (no suffix)
    },
})
