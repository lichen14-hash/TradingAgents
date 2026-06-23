"""Market detection utilities for Chinese A-share tickers.

Pure string operations — no network calls, no external dependencies.
This module must not import from any other dataflows submodule to avoid
circular imports.

Board classification:
- Shanghai main board (上交所主板): 600xxx, 601xxx, 603xxx, 605xxx
- STAR Market (科创板): 688xxx → .SS
- Shenzhen main board (深交所主板): 000xxx, 001xxx, 002xxx
- ChiNext (创业板): 300xxx, 301xxx → .SZ
"""

from __future__ import annotations

import re

_SHANGHAI_PREFIXES = re.compile(r"^(600|601|603|605|688)\d{3}$")
_SHENZHEN_PREFIXES = re.compile(r"^(000|001|002|003|300|301)\d{3}$")

_EXCHANGE_SUFFIX = re.compile(r"^(\d{6})\.(SS|SZ)$", re.IGNORECASE)
_SH_SZ_PREFIX = re.compile(r"^(sh|sz)(\d{6})$", re.IGNORECASE)
_BARE_CODE = re.compile(r"^\d{6}$")


def _extract_code(ticker: str) -> str | None:
    """Extract the 6-digit code from various input formats.

    Returns the bare code or None if the ticker is not A-share shaped.
    """
    s = ticker.strip()

    m = _EXCHANGE_SUFFIX.match(s)
    if m:
        return m.group(1)

    m = _SH_SZ_PREFIX.match(s)
    if m:
        return m.group(2)

    if _BARE_CODE.match(s):
        return s

    return None


def detect_exchange(ticker: str) -> str | None:
    """Return ``'.SS'`` or ``'.SZ'`` for a Chinese A-share ticker, else ``None``.

    When the input already carries an explicit suffix (``.SS`` / ``.SZ``) or
    prefix (``sh`` / ``sz``), that declaration is trusted.  Code-range
    inference is only used for bare 6-digit codes.  This avoids misrouting
    index tickers whose codes overlap with stock codes on the other exchange
    (e.g. ``000001.SS`` is the SSE Composite Index, not Ping An Bank).
    """
    s = ticker.strip()

    m = _EXCHANGE_SUFFIX.match(s)
    if m:
        return f".{m.group(2).upper()}"

    m = _SH_SZ_PREFIX.match(s)
    if m:
        return ".SS" if m.group(1).lower() == "sh" else ".SZ"

    code = _extract_code(s)
    if code is None:
        return None
    if _SHANGHAI_PREFIXES.match(code):
        return ".SS"
    if _SHENZHEN_PREFIXES.match(code):
        return ".SZ"
    return None


def is_a_share(ticker: str) -> bool:
    """Return ``True`` if *ticker* is a Chinese A-share."""
    return detect_exchange(ticker) is not None


def normalize_a_share_symbol(ticker: str) -> str:
    """Normalize various A-share input formats to ``NNNNNN.SS`` / ``NNNNNN.SZ``.

    Accepted inputs::

        600519.SS  →  600519.SS   (pass-through)
        000001.SS  →  000001.SS   (explicit suffix preserved — index code)
        600519     →  600519.SS   (infer from code range)
        sh600519   →  600519.SS
        SZ300750   →  300750.SZ

    Raises ``ValueError`` if the ticker cannot be recognized as A-share.
    """
    code = _extract_code(ticker)
    if code is None:
        raise ValueError(f"Cannot normalize '{ticker}' as an A-share symbol")

    exchange = detect_exchange(ticker)
    if exchange is None:
        raise ValueError(f"Code '{code}' does not match any A-share board range")

    return f"{code}{exchange}"


def a_share_to_akshare_symbol(ticker: str) -> str:
    """Convert canonical A-share ticker to AKShare's bare-code format.

    ``600519.SS`` → ``600519``
    """
    code = _extract_code(ticker)
    if code is None:
        raise ValueError(f"Cannot convert '{ticker}' to AKShare format")
    return code


def a_share_to_baostock_symbol(ticker: str) -> str:
    """Convert canonical A-share ticker to BaoStock's ``sh.NNNNNN`` / ``sz.NNNNNN`` format.

    ``600519.SS`` → ``sh.600519``
    ``300750.SZ`` → ``sz.300750``
    """
    code = _extract_code(ticker)
    exchange = detect_exchange(ticker)
    if code is None or exchange is None:
        raise ValueError(f"Cannot convert '{ticker}' to BaoStock format")
    prefix = "sh" if exchange == ".SS" else "sz"
    return f"{prefix}.{code}"


def a_share_to_sina_symbol(ticker: str) -> str:
    """Convert canonical A-share ticker to Sina's ``shNNNNNN`` / ``szNNNNNN`` format.

    ``600519.SS`` → ``sh600519``
    ``300750.SZ`` → ``sz300750``
    """
    code = _extract_code(ticker)
    exchange = detect_exchange(ticker)
    if code is None or exchange is None:
        raise ValueError(f"Cannot convert '{ticker}' to Sina format")
    prefix = "sh" if exchange == ".SS" else "sz"
    return f"{prefix}{code}"


_BOARD_NAMES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^688\d{3}$"), "科创板"),
    (re.compile(r"^(600|601|603|605)\d{3}$"), "上交所主板"),
    (re.compile(r"^(300|301)\d{3}$"), "创业板"),
    (re.compile(r"^(000|001|002|003)\d{3}$"), "深交所主板"),
]


def get_board_name(ticker: str) -> str:
    """Return the board name (板块) for display purposes."""
    code = _extract_code(ticker)
    if code is None:
        return "Unknown"
    for pattern, name in _BOARD_NAMES:
        if pattern.match(code):
            return name
    return "Unknown"
