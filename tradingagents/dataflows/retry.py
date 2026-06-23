"""Shared retry-with-backoff utility for transient network / rate-limit failures.

Every data vendor module should wrap outbound calls with :func:`call_with_retry`
so a single transient error (429, timeout, connection reset) does not kill an
entire analysis run.  The helper retries only on network-class exceptions and
HTTP status codes that indicate a server-side transient — programming errors
(TypeError, ValueError, KeyError) propagate immediately.

When all retries are exhausted on what looks like a rate limit, the helper wraps
the original exception in :class:`VendorRateLimitError` so the vendor router
can skip to the next vendor in the chain.
"""

from __future__ import annotations

import logging
import random
import time

logger = logging.getLogger(__name__)

_RATE_LIMIT_PHRASES = (
    "rate limit",
    "too many requests",
    "throttl",
    "频率限制",
    "请求过于频繁",
    "429",
)

_RETRIABLE_HTTP_CODES = {429, 500, 502, 503, 504}


def _is_retriable(exc: Exception) -> bool:
    """Decide whether *exc* is a transient failure worth retrying."""
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True

    try:
        from requests.exceptions import (
            ConnectionError as ReqConn,
            ReadTimeout,
            Timeout as ReqTimeout,
        )

        if isinstance(exc, (ReqConn, ReqTimeout, ReadTimeout)):
            return True
    except ImportError:
        pass

    try:
        from urllib.error import URLError

        if isinstance(exc, URLError):
            return True
    except ImportError:
        pass

    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status in _RETRIABLE_HTTP_CODES:
        return True

    msg = str(exc).lower()
    return any(phrase in msg for phrase in _RATE_LIMIT_PHRASES)


def call_with_retry(
    func,
    *args,
    max_retries: int = 3,
    base_delay: float = 1.5,
    max_delay: float = 30.0,
    **kwargs,
):
    """Call *func* with exponential backoff on transient failures.

    Retries on network errors (``ConnectionError``, ``TimeoutError``,
    ``requests`` / ``urllib`` transport errors) and HTTP 429/5xx responses.

    Does NOT retry on programming errors (``TypeError``, ``ValueError``,
    ``KeyError``, ``ImportError``) or data errors (``NoMarketDataError``,
    ``VendorNotConfiguredError``).

    After exhausting retries on a rate-limit-like error, raises
    :class:`VendorRateLimitError` so the vendor router can fall back.
    """
    last_exc: Exception | None = None
    fname = getattr(func, "__name__", str(func))

    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            if not _is_retriable(exc):
                raise

            last_exc = exc
            if attempt < max_retries:
                delay = min(base_delay * (2**attempt), max_delay)
                delay += random.uniform(0, delay * 0.15)
                logger.warning(
                    "%s failed (attempt %d/%d), retrying in %.1fs: %s",
                    fname,
                    attempt + 1,
                    max_retries,
                    delay,
                    exc,
                )
                time.sleep(delay)

    assert last_exc is not None  # noqa: S101
    msg = str(last_exc).lower()
    if any(phrase in msg for phrase in _RATE_LIMIT_PHRASES):
        from .errors import VendorRateLimitError

        raise VendorRateLimitError(
            f"{fname} rate limited after {max_retries} retries: {last_exc}"
        ) from last_exc
    raise last_exc
