"""Tests for tradingagents.dataflows.market_utils."""

import pytest

from tradingagents.dataflows.market_utils import (
    a_share_to_akshare_symbol,
    a_share_to_sina_symbol,
    detect_exchange,
    get_board_name,
    is_a_share,
    normalize_a_share_symbol,
)


class TestIsAShare:
    """is_a_share detection for various formats."""

    @pytest.mark.parametrize("ticker", [
        "600519.SS", "600519.ss",
        "000001.SZ", "000001.sz",
        "300750.SZ",
        "688111.SS",
        "600519",
        "000001",
        "sh600519", "SH600519",
        "sz300750", "SZ000001",
    ])
    def test_positive(self, ticker):
        assert is_a_share(ticker), f"{ticker} should be detected as A-share"

    @pytest.mark.parametrize("ticker", [
        "AAPL", "BABA", "SPY",
        "BTC-USD", "ETH-USDT",
        "0700.HK", "7203.T",
        "^GSPC", "^HSI",
        "EURUSD=X",
        "123",
        "1234567",
        "",
    ])
    def test_negative(self, ticker):
        assert not is_a_share(ticker), f"{ticker} should NOT be detected as A-share"


class TestDetectExchange:
    @pytest.mark.parametrize("ticker,expected", [
        ("600519.SS", ".SS"),
        ("600519", ".SS"),
        ("601318", ".SS"),
        ("603259", ".SS"),
        ("605499", ".SS"),
        ("688111", ".SS"),
        ("000001.SZ", ".SZ"),
        ("000001", ".SZ"),
        ("002594", ".SZ"),
        ("300750", ".SZ"),
        ("301269", ".SZ"),
        ("sh600519", ".SS"),
        ("sz300750", ".SZ"),
    ])
    def test_exchange_detection(self, ticker, expected):
        assert detect_exchange(ticker) == expected

    @pytest.mark.parametrize("ticker", ["AAPL", "BABA", "0700.HK", ""])
    def test_none_for_non_a_share(self, ticker):
        assert detect_exchange(ticker) is None


class TestNormalizeAShareSymbol:
    @pytest.mark.parametrize("raw,expected", [
        ("600519.SS", "600519.SS"),
        ("600519.ss", "600519.SS"),
        ("600519", "600519.SS"),
        ("sh600519", "600519.SS"),
        ("SH600519", "600519.SS"),
        ("000001.SZ", "000001.SZ"),
        ("000001", "000001.SZ"),
        ("sz000001", "000001.SZ"),
        ("300750", "300750.SZ"),
        ("688111", "688111.SS"),
    ])
    def test_normalization(self, raw, expected):
        assert normalize_a_share_symbol(raw) == expected

    def test_non_a_share_raises(self):
        with pytest.raises(ValueError):
            normalize_a_share_symbol("AAPL")

    def test_invalid_code_raises(self):
        with pytest.raises(ValueError):
            normalize_a_share_symbol("999999")


class TestAShareToAkshare:
    @pytest.mark.parametrize("ticker,expected", [
        ("600519.SS", "600519"),
        ("000001.SZ", "000001"),
        ("sh600519", "600519"),
        ("300750", "300750"),
    ])
    def test_conversion(self, ticker, expected):
        assert a_share_to_akshare_symbol(ticker) == expected


class TestAShareToSina:
    @pytest.mark.parametrize("ticker,expected", [
        ("600519.SS", "sh600519"),
        ("000001.SZ", "sz000001"),
        ("688111.SS", "sh688111"),
        ("300750.SZ", "sz300750"),
    ])
    def test_conversion(self, ticker, expected):
        assert a_share_to_sina_symbol(ticker) == expected


class TestGetBoardName:
    @pytest.mark.parametrize("ticker,expected", [
        ("600519.SS", "上交所主板"),
        ("601318", "上交所主板"),
        ("688111.SS", "科创板"),
        ("688981", "科创板"),
        ("000001.SZ", "深交所主板"),
        ("002594", "深交所主板"),
        ("300750.SZ", "创业板"),
        ("301269", "创业板"),
        ("AAPL", "Unknown"),
    ])
    def test_board_names(self, ticker, expected):
        assert get_board_name(ticker) == expected
