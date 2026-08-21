"""Token lifecycle and model-name resolution for the Qoder model gateway.

Everything here encodes a measured fact from ``findings.md``; the section
references in the comments point at the experiment that pinned the behavior
down, because none of this is documented upstream and none of it is guessable.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

# findings.md T1: the gateway lives at this exact path. /v1/chat/completions and
# /api/v1/chat/completions both 404, so there is no alias to fall back on.
_DEFAULT_MODEL_HOST = "api2-v2.qoder.sh"
EXCHANGE_URL = "https://openapi.qoder.sh/api/v1/jobToken/exchange"
USERINFO_URL = "https://openapi.qoder.sh/api/v1/userinfo"

# findings.md T8: the gateway accepts these 11 names. The value is what it
# actually routes to, recorded so a caller can tell that e.g. `lite` is not the
# zero-cost "Lite" model from the SDK plane but plain qwen3-coder-plus.
MODEL_ROUTES = {
    "deepseek-v4-flash": "deepseek-v4-flash",
    "qmodel": "qwen3.7-plus",
    "mmodel": "MiniMax-M2.5",
    "gmodel": "glm-5",
    "kmodel": "kimi-k2.7-code",
    "dmodel": "deepseek-v4-pro",
    "efficient": "efficient",
    "ultimate": "ultimate",
    "auto": "qwen3-coder-plus",
    "lite": "qwen3-coder-plus",
    "performance": "qwen3-coder-plus",
}

# findings.md T8: the SDK's getAvailableModels() keys are a *different*
# namespace from the gateway's, with no rule connecting them -- `dfmodel` is
# rejected while the provider name `deepseek-v4-flash` works, and `dmodel`
# works while `deepseek-v4-pro` is rejected. Anyone who copied the T2 table
# would otherwise get an opaque upstream error, so translate the two known
# inversions instead.
MODEL_ALIASES = {
    "dfmodel": "deepseek-v4-flash",
    "deepseek-v4-pro": "dmodel",
}

# findings.md T10: 1139ms median, jaccard 0.600 at temperature 0, and it holds
# the response-length cap. The cheapest model that was actually measured.
DEFAULT_MODEL = "deepseek-v4-flash"

# findings.md T7: expires_in is 86400000 -- milliseconds, not seconds. Treating
# it as seconds yields ~1000 days and the refresh never fires, so the proxy
# would start 401-ing exactly 24h in. Values at or above this bound are read as
# milliseconds, smaller ones as seconds, so a future unit change degrades into
# a shorter TTL rather than a silent outage.
_MS_THRESHOLD = 1_000_000
_FALLBACK_TTL_S = 23 * 3600
_DEFAULT_REFRESH_MARGIN_S = 3600


def model_server_host() -> str:
    """Honor the same override the qodercli bundle reads (findings.md T1)."""
    override = (os.environ.get("QODER_MODEL_SERVER_HOST") or "").strip()
    return override or _DEFAULT_MODEL_HOST


def chat_completions_url() -> str:
    return f"https://{model_server_host()}/model/v1/chat/completions"


CHAT_COMPLETIONS_URL = chat_completions_url()


class TokenExchangeError(RuntimeError):
    """The PAT could not be exchanged for a usable ``jt-`` token."""


class UnknownModelError(ValueError):
    """Model name is not one the gateway accepts (findings.md T8)."""


def resolve_model(name: str | None) -> str:
    """Map a caller-supplied model name onto one the gateway accepts."""
    candidate = (name or DEFAULT_MODEL).strip()
    candidate = MODEL_ALIASES.get(candidate, candidate)
    if candidate not in MODEL_ROUTES:
        accepted = ", ".join(sorted(MODEL_ROUTES))
        raise UnknownModelError(
            f"{name!r} is not accepted by the Qoder gateway. Accepted: {accepted}"
        )
    return candidate


def _ttl_seconds(expires_in: object) -> float:
    try:
        raw = float(expires_in)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        logger.warning("exchange response had no usable expires_in; assuming 23h")
        return _FALLBACK_TTL_S
    if raw <= 0:
        logger.warning("exchange response had non-positive expires_in; assuming 23h")
        return _FALLBACK_TTL_S
    return raw / 1000.0 if raw >= _MS_THRESHOLD else raw


@dataclass(frozen=True)
class TokenState:
    """Non-secret view of the cached token, safe to log or expose on /healthz."""

    present: bool
    expires_at: float | None
    seconds_remaining: float | None
    exchanges: int


class QoderTokenManager:
    """Holds a PAT and keeps a fresh ``jt-`` token in front of it.

    findings.md never located a refresh endpoint -- only the presence and TTL of
    ``refresh_token`` were recorded. The PAT is the long-lived credential, so
    this re-exchanges instead of refreshing; that also collapses the documented
    "refresh failed -> fall back to exchange" branch into a single code path.
    """

    def __init__(
        self,
        pat: str,
        *,
        exchange_url: str = EXCHANGE_URL,
        user_agent: str = "qoder/1.1.21",
        refresh_margin_s: float = _DEFAULT_REFRESH_MARGIN_S,
        client: httpx.Client | None = None,
        clock=time.time,
    ) -> None:
        if not pat or not pat.strip():
            raise TokenExchangeError("no PAT supplied (set QODER_PAT)")
        self._pat = pat.strip()
        self._exchange_url = exchange_url
        self._user_agent = user_agent
        self._refresh_margin_s = max(0.0, refresh_margin_s)
        self._client = client or httpx.Client(timeout=30.0)
        self._clock = clock
        self._lock = threading.Lock()
        self._token: str | None = None
        self._expires_at: float = 0.0
        self._lifetime_s: float = 0.0
        self._exchanges = 0

    def token(self, *, force_refresh: bool = False) -> str:
        """Return a valid token, exchanging if the cache is stale or forced."""
        with self._lock:
            if not force_refresh and self._token and not self._is_stale():
                return self._token
            return self._exchange_locked()

    def state(self) -> TokenState:
        with self._lock:
            if not self._token:
                return TokenState(False, None, None, self._exchanges)
            return TokenState(
                True,
                self._expires_at,
                max(0.0, self._expires_at - self._clock()),
                self._exchanges,
            )

    def close(self) -> None:
        self._client.close()

    def _is_stale(self) -> bool:
        # Clamp the margin to half the token's *original* lifetime so a
        # short-lived token cannot look stale the moment it is issued (which
        # would re-exchange on every request). Clamping against the remaining
        # TTL instead would shrink the window as the token ages and the
        # proactive refresh would never fire at all.
        margin = min(self._refresh_margin_s, self._lifetime_s / 2)
        return (self._expires_at - self._clock()) <= margin

    def _exchange_locked(self) -> str:
        # findings.md T7: the PAT goes in the body, not the Authorization
        # header, and the exchange call itself sends no Authorization at all.
        try:
            response = self._client.post(
                self._exchange_url,
                json={"personal_token": self._pat},
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "User-Agent": self._user_agent,
                },
            )
        except httpx.HTTPError as exc:
            raise TokenExchangeError(f"exchange request failed: {exc}") from exc

        if response.status_code != 200:
            # Deliberately excludes the body: it can echo the PAT back.
            raise TokenExchangeError(
                f"exchange returned HTTP {response.status_code} (PAT invalid or revoked?)"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise TokenExchangeError("exchange returned a non-JSON body") from exc

        token = payload.get("token")
        if not isinstance(token, str) or not token:
            raise TokenExchangeError("exchange response contained no token")

        self._token = token
        self._lifetime_s = _ttl_seconds(payload.get("expires_in"))
        self._expires_at = self._clock() + self._lifetime_s
        self._exchanges += 1
        logger.info(
            "exchanged PAT for gateway token (len=%d, ttl=%.0fs)", len(token), self._lifetime_s
        )
        return token


class QoderGateway:
    """Forwards OpenAI-shaped chat requests to the gateway."""

    def __init__(
        self,
        token_manager: QoderTokenManager,
        *,
        chat_url: str | None = None,
        client: httpx.Client | None = None,
        user_agent: str = "qoder/1.1.21",
    ) -> None:
        self._tokens = token_manager
        self._chat_url = chat_url or chat_completions_url()
        self._client = client or httpx.Client(timeout=120.0)
        self._user_agent = user_agent

    def chat_completions(self, payload: dict) -> tuple[int, dict]:
        """Send one completion request; returns ``(status_code, json_body)``.

        ``model`` is rewritten to a gateway-accepted name. A 401 triggers one
        forced re-exchange and one retry, then the response is returned as-is.
        """
        body = dict(payload)
        body["model"] = resolve_model(body.get("model"))

        response = self._post(body, self._tokens.token())
        if response.status_code == 401:
            logger.info("gateway returned 401; re-exchanging and retrying once")
            response = self._post(body, self._tokens.token(force_refresh=True))

        try:
            return response.status_code, response.json()
        except ValueError:
            return response.status_code, {
                "error": {
                    "message": "gateway returned a non-JSON body",
                    "type": "upstream_error",
                    "status": response.status_code,
                }
            }

    def _post(self, body: dict, token: str) -> httpx.Response:
        return self._client.post(
            self._chat_url,
            json=body,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": self._user_agent,
            },
        )

    def close(self) -> None:
        self._client.close()
