"""Single-instrument spot quote with a multi-vendor fallback chain.

Incident that motivated this module (2026-08-19, 12:55 intraday batch): the
intraday snapshot failed on every A-share while both ETFs succeeded. The split
was exactly along the vendor host, not the ticker:

- ``ak.stock_zh_a_spot_em`` → ``82.push2.eastmoney.com``  (realtime)   ✗
- ``ak.stock_hk_spot_em``   → ``72.push2.eastmoney.com``  (realtime)   ✗
- ``ak.fund_etf_spot_em``   → ``push2delay.eastmoney.com`` (delayed)   ✓

EastMoney's *realtime* ``push2`` cluster closes the connection at the TCP/HTTP
layer from this network — no HTTP response at all, on every path
(``clist/get``, ``stock/get``, ``ulist.np/get``), every edge IP behind the
``push2ipv6.trafficmanager.cn`` traffic manager, both schemes, with and without
a browser User-Agent. Its *delayed* host answers all of them with 200. Retrying
therefore never helped: all three attempts hit the same wall, which is why one
failed snapshot cost ~35s.

Two design consequences:

1. **The chain is the retry.** Each vendor gets exactly one attempt. Retrying a
   single vendor is what turned a hard vendor outage into a 35-second stall;
   moving to the next vendor is both faster and the only thing that can
   actually succeed.
2. **One request per quote.** The previous path pulled the entire market table
   (~5899 rows, ``pz=100`` → ~59 HTTP requests) to read a single row. Every
   vendor here is queried per instrument.

Units are normalised on the way in, because the vendors disagree and the
disagreement is silent: Sina reports volume in 股 for every market, Tencent and
EastMoney report 手 for A-shares/ETFs but 股 for HK, and Tencent reports
turnover in 万元 for A-shares but 元 for HK. :class:`SpotQuote` always holds
股 and 元; the renderer decides what to display.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from urllib.request import Request, urlopen

from tradingagents.utils.time_utils import CN_TZ

from .market_utils import (
    a_share_to_sina_symbol,
    detect_exchange,
    hk_to_akshare_symbol,
    is_a_share,
    is_hk_stock,
)

logger = logging.getLogger(__name__)

_TIMEOUT = 6
_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"


class SpotQuoteError(RuntimeError):
    """Raised when every vendor in the chain failed to produce a usable quote."""


@dataclass(frozen=True)
class SpotQuote:
    """A normalised spot quote for one instrument.

    ``volume_shares`` is always 股 and ``amount_yuan`` always 元, regardless of
    which vendor served it -- see the module docstring on why that matters.

    ``delayed`` is carried explicitly rather than folded into ``source``: an
    intraday analysis reading a 15-minute-old price as "current" is a wrong
    answer, not a missing one, so the renderer must be able to say so.
    """

    source: str
    delayed: bool
    name: str = ""
    last: float | None = None
    prev_close: float | None = None
    open: float | None = None
    high: float | None = None
    low: float | None = None
    volume_shares: float | None = None
    amount_yuan: float | None = None
    turnover_rate: float | None = None
    volume_ratio: float | None = None
    quote_time: str = ""

    @property
    def change(self) -> float | None:
        if self.last is None or self.prev_close is None:
            return None
        return self.last - self.prev_close

    @property
    def change_pct(self) -> float | None:
        if self.last is None or not self.prev_close:
            return None
        return (self.last - self.prev_close) / self.prev_close * 100.0


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------


def _get(url: str, *, referer: str = "", encoding: str = "gbk") -> str:
    headers = {"User-Agent": _UA}
    if referer:
        headers["Referer"] = referer
    raw = urlopen(Request(url, headers=headers), timeout=_TIMEOUT).read()
    return raw.decode(encoding, errors="replace")


def _num(value: object) -> float | None:
    """Parse a vendor field to float, mapping blanks and ``0`` sentinels to None.

    Vendors pad absent fields with ``0`` rather than omitting them (HK quotes
    from Tencent carry a dozen zeroed bid/ask slots), so a zero here means "not
    reported" for every field this module reads. A real price is never 0, and a
    real turnover rate of exactly 0.00 is indistinguishable from an absent one.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in ("-", "--"):
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    return None if parsed == 0 else parsed


def _tencent_time(value: str) -> str:
    """Normalise Tencent's timestamp, which differs by market.

    A-shares arrive as ``20260819161445`` and HK as ``2026/08/19 16:08:50``. The
    compact form is not a date to a reader, and this string goes straight into an
    analyst's prompt.
    """
    text = str(value).strip()
    if re.fullmatch(r"\d{14}", text):
        return f"{text[0:4]}-{text[4:6]}-{text[6:8]} {text[8:10]}:{text[10:12]}:{text[12:14]}"
    return text


def _quoted(raw: str) -> str | None:
    """Extract the payload inside the vendor's ``"..."`` wrapper.

    Both text feeds answer an unknown code with an empty payload
    (``var hq_str_sz999999="";``) rather than an HTTP error, so an empty match is
    a vendor failure, not a quote.
    """
    match = re.search(r'"([^"]*)"', raw)
    if not match or not match.group(1).strip():
        return None
    return match.group(1)


# ---------------------------------------------------------------------------
# Vendor: Sina (realtime)
# ---------------------------------------------------------------------------

_SINA_URL = "https://hq.sinajs.cn/list={symbol}"
_SINA_REFERER = "https://finance.sina.com.cn"


def quote_from_sina(ticker: str) -> SpotQuote:
    """Realtime quote from ``hq.sinajs.cn``.

    Field order is positional and differs between markets; both layouts below
    were verified against EastMoney's delayed quote for the same instrument on
    2026-08-19 (A-share 300760/600406, ETF 515880, HK 09988).
    """
    if is_hk_stock(ticker):
        symbol = f"rt_hk{hk_to_akshare_symbol(ticker)}"
    else:
        symbol = a_share_to_sina_symbol(ticker)

    payload = _quoted(_get(_SINA_URL.format(symbol=symbol), referer=_SINA_REFERER))
    if payload is None:
        raise SpotQuoteError(f"Sina returned an empty quote for {symbol}")
    parts = payload.split(",")

    if is_hk_stock(ticker):
        # 英文名,中文名,今开,昨收,最高,最低,现价,涨跌额,涨跌幅,买一,卖一,成交额(元),成交量(股),...,日期,时间
        if len(parts) < 19:
            raise SpotQuoteError(f"Sina HK quote for {symbol} has {len(parts)} fields")
        return SpotQuote(
            source="Sina hq.sinajs.cn (HK, 实时)",
            delayed=False,
            name=parts[1].strip(),
            open=_num(parts[2]),
            prev_close=_num(parts[3]),
            high=_num(parts[4]),
            low=_num(parts[5]),
            last=_num(parts[6]),
            amount_yuan=_num(parts[11]),
            volume_shares=_num(parts[12]),
            quote_time=f"{parts[17].strip()} {parts[18].strip()}".strip(),
        )

    # 名称,今开,昨收,现价,最高,最低,买一,卖一,成交量(股),成交额(元),...,日期,时间
    if len(parts) < 32:
        raise SpotQuoteError(f"Sina A-share quote for {symbol} has {len(parts)} fields")
    return SpotQuote(
        source="Sina hq.sinajs.cn (实时)",
        delayed=False,
        name=parts[0].strip(),
        open=_num(parts[1]),
        prev_close=_num(parts[2]),
        last=_num(parts[3]),
        high=_num(parts[4]),
        low=_num(parts[5]),
        volume_shares=_num(parts[8]),
        amount_yuan=_num(parts[9]),
        quote_time=f"{parts[30].strip()} {parts[31].strip()}".strip(),
    )


# ---------------------------------------------------------------------------
# Vendor: Tencent (realtime)
# ---------------------------------------------------------------------------

_TENCENT_URL = "https://qt.gtimg.cn/q={symbol}"


def quote_from_tencent(ticker: str) -> SpotQuote:
    """Realtime quote from ``qt.gtimg.cn``.

    Indices 1/3/4/5/30/31/32/33/34 hold the same meaning in both markets, but
    the volume and turnover units do not: A-shares report 手 and 万元, HK
    reports 股 and 元. Index 49 is 量比 for A-shares and the 52-week low for
    HK, so it is only read for A-shares.
    """
    hk = is_hk_stock(ticker)
    if hk:
        symbol = f"hk{hk_to_akshare_symbol(ticker)}"
    else:
        exchange = detect_exchange(ticker)
        if exchange is None:
            raise SpotQuoteError(f"Cannot map {ticker} to a Tencent symbol")
        symbol = a_share_to_sina_symbol(ticker)  # same shNNNNNN / szNNNNNN shape

    payload = _quoted(_get(_TENCENT_URL.format(symbol=symbol)))
    if payload is None:
        raise SpotQuoteError(f"Tencent returned an empty quote for {symbol}")
    parts = payload.split("~")
    if len(parts) < 39:
        raise SpotQuoteError(f"Tencent quote for {symbol} has {len(parts)} fields")

    volume = _num(parts[6])
    amount = _num(parts[37])
    return SpotQuote(
        source="Tencent qt.gtimg.cn (实时)",
        delayed=False,
        name=parts[1].strip(),
        last=_num(parts[3]),
        prev_close=_num(parts[4]),
        open=_num(parts[5]),
        high=_num(parts[33]),
        low=_num(parts[34]),
        volume_shares=volume if hk else (volume * 100 if volume else None),
        amount_yuan=amount if hk else (amount * 10000 if amount else None),
        turnover_rate=None if hk else _num(parts[38]),
        volume_ratio=None if hk or len(parts) < 50 else _num(parts[49]),
        quote_time=_tencent_time(parts[30]),
    )


# ---------------------------------------------------------------------------
# Vendor: EastMoney delayed (fallback)
# ---------------------------------------------------------------------------

_EM_URL = (
    "https://push2delay.eastmoney.com/api/qt/stock/get"
    "?secid={secid}&fltt=2&invt=2"
    "&fields=f43,f44,f45,f46,f47,f48,f50,f57,f58,f60,f86,f168,f169,f170"
)


def _em_secid(ticker: str) -> str:
    if is_hk_stock(ticker):
        return f"116.{hk_to_akshare_symbol(ticker)}"
    exchange = detect_exchange(ticker)
    if exchange == ".SS":
        return f"1.{ticker.split('.')[0]}"
    if exchange == ".SZ":
        return f"0.{ticker.split('.')[0]}"
    raise SpotQuoteError(f"Cannot map {ticker} to an EastMoney secid")


def quote_from_eastmoney_delayed(ticker: str) -> SpotQuote:
    """15-minute delayed quote from ``push2delay.eastmoney.com``.

    Last in the chain and flagged ``delayed=True``: this is the host that stayed
    up through the 2026-08-19 outage, so it is the floor under the two realtime
    vendors -- but a delayed price presented as the current one is worse than no
    price at all for an intraday decision.

    JSON here is UTF-8, unlike the GBK text feeds.
    """
    hk = is_hk_stock(ticker)
    raw = _get(_EM_URL.format(secid=_em_secid(ticker)), encoding="utf-8")
    data = (json.loads(raw) or {}).get("data")
    if not data:
        raise SpotQuoteError(f"EastMoney delayed quote for {ticker} carried no data")

    volume = _num(data.get("f47"))
    stamp = _num(data.get("f86"))
    return SpotQuote(
        source="EastMoney push2delay (延迟约15分钟)",
        delayed=True,
        name=str(data.get("f58") or "").strip(),
        last=_num(data.get("f43")),
        high=_num(data.get("f44")),
        low=_num(data.get("f45")),
        open=_num(data.get("f46")),
        prev_close=_num(data.get("f60")),
        volume_shares=volume if hk else (volume * 100 if volume else None),
        amount_yuan=_num(data.get("f48")),
        turnover_rate=_num(data.get("f168")),
        volume_ratio=_num(data.get("f50")),
        quote_time=(
            datetime.fromtimestamp(stamp, CN_TZ).strftime("%Y-%m-%d %H:%M:%S")
            if stamp else ""
        ),
    )


# ---------------------------------------------------------------------------
# Chain
# ---------------------------------------------------------------------------

# Realtime first, delayed as the floor. One attempt each -- see the module
# docstring: retrying a dead vendor is what made the original failure slow.
#
# Tencent leads because it is the only realtime feed that also carries 换手率 and
# 量比, which the replaced EastMoney table did carry. 量比 in particular is an
# intraday volume-versus-average measure, so dropping it from an *intraday*
# snapshot would be a quiet content regression rather than a formatting change.
VENDOR_CHAIN = (
    ("tencent", quote_from_tencent),
    ("sina", quote_from_sina),
    ("eastmoney_delayed", quote_from_eastmoney_delayed),
)


def get_spot_quote(ticker: str) -> SpotQuote:
    """Return a spot quote for *ticker*, trying each vendor in turn.

    A quote without a usable ``last`` price is treated as a vendor failure and
    the chain moves on: a halted or pre-open feed reporting ``0`` would
    otherwise render as ``当前价: 0``, which reads like a real crash.

    Raises :class:`SpotQuoteError` naming every vendor that failed, so the
    ``<unavailable: ...>`` placeholder the collector substitutes says which
    vendors were tried rather than just naming the last exception.
    """
    if not (is_a_share(ticker) or is_hk_stock(ticker)):
        raise SpotQuoteError(
            f"Spot quotes support A-share/ETF/HK tickers only, got {ticker}"
        )

    failures: list[str] = []
    for name, fetch in VENDOR_CHAIN:
        try:
            quote = fetch(ticker)
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            logger.warning("Spot quote vendor %s failed for %s: %s", name, ticker, exc)
            continue
        if quote.last is None:
            failures.append(f"{name}: no last price (halted or pre-open feed)")
            logger.warning("Spot quote vendor %s returned no price for %s", name, ticker)
            continue
        return quote

    raise SpotQuoteError(f"All spot-quote vendors failed for {ticker} — " + "; ".join(failures))
