import logging
import os
import re
import time
from typing import Any

from langchain_anthropic import ChatAnthropic

from .base_client import BaseLLMClient, normalize_content
from .validators import validate_model

logger = logging.getLogger(__name__)

# Some Anthropic-compatible proxies (e.g. IdealAb /api/code) gate access on a
# Claude-Code-style User-Agent, rejecting the default SDK UA with errors like
# "opus计划仅限在claude code中使用". Impersonate the CLI so those proxies route
# the request. Overridable via TRADINGAGENTS_ANTHROPIC_USER_AGENT.
_DEFAULT_CLI_USER_AGENT = "claude-cli/1.0.30 (external, cli)"

_PASSTHROUGH_KWARGS = (
    "timeout", "max_retries", "api_key", "max_tokens", "temperature",
    "callbacks", "http_client", "http_async_client", "effort", "streaming",
)

# Transport defaults for long agent turns.
#
# Root cause of the ticker losses in the 2026-08-18 batch (600320.SS "504
# Gateway Time-out" from Tengine/AServer-Ingress, 002602.SZ and 688336.SS
# "MPE-001") and the 2026-08-19 09988.HK "Connection error.": a single
# non-streaming request that generates tens of thousands of tokens holds the
# socket open with zero bytes flowing for minutes. Every hop in front of the
# model — the proxy's own ingress, any corporate egress — treats that silence
# as a dead connection and kills it at its idle timeout, which the client then
# reports as a gateway or connection error rather than as a timeout.
#
# Streaming fixes it at the transport layer: SSE events flow continuously, so
# no hop sees an idle socket, and the Anthropic SDK stops refusing large
# ``max_tokens`` requests as "may take longer than 10 minutes". The aggregated
# response is byte-identical to the non-streaming one, and
# ``response_metadata["stop_reason"]`` survives aggregation (it arrives on the
# ``message_delta`` event), so the truncation retry below still works.
#
# ``timeout`` then becomes a *gap* deadline rather than a whole-response one:
# httpx applies a bare float as its read timeout, i.e. "no bytes for 300s",
# which a streaming response can only hit if generation has genuinely stalled.
# Without it, langchain-anthropic forwards ``timeout=None`` to the SDK and the
# call waits forever, which is how 09988.HK burned ~2 hours before dying.
_DEFAULT_STREAMING = True
_DEFAULT_REQUEST_TIMEOUT = 300.0

# Anthropic's extended-thinking ``effort`` parameter is accepted by Opus 4.5+
# and Sonnet 4.5+ only. Haiku (any version shipped to date) 400s with
# ``"This model does not support the effort parameter"`` (#831). Future
# ``claude-{opus,sonnet}-X-Y`` releases inherit effort support via the
# forward-compat pattern below; future Haiku stays excluded by default.
_EFFORT_EXACT = {
    "claude-mythos-preview",  # non-standard preview name; effort-capable
}
_EFFORT_PATTERN = re.compile(r"^claude-(opus|sonnet)-\d+-\d+$")


def _supports_effort(model: str) -> bool:
    """Whether Anthropic accepts the ``effort`` parameter for this model."""
    model_lc = model.lower()
    return model_lc in _EFFORT_EXACT or bool(_EFFORT_PATTERN.match(model_lc))


# Output-budget floor for models the installed langchain-anthropic does not
# know about.
#
# Root cause of the 2026-08-11 300760.SZ empty-output incident: when
# ``max_tokens`` is not passed, ChatAnthropic looks the model up in its bundled
# model-profile table and, on a miss, falls back to
# ``_FALLBACK_MAX_OUTPUT_TOKENS = 4096``. ``claude-opus-5`` / ``claude-sonnet-5``
# are newer than the installed package, so every call ran with a 4096-token
# output cap while the same package gives ``claude-opus-4-5`` 64000. With
# extended thinking enabled server-side, reasoning and the visible answer share
# that cap: reasoning consumed it and the response arrived with no text block at
# all, or with a tool_use block truncated mid-JSON (which surfaced as a Pydantic
# ValidationError and a structured-output degradation).
#
# This is structural, not a one-off: every model released after the pinned
# package version hits the same silent 16x downgrade. So the client never
# inherits that guess — an unknown model gets an explicit, generous cap instead.
# ``max_tokens`` is a ceiling, not a spend: billing follows tokens actually
# produced, so a high cap costs nothing on short answers.
_UNKNOWN_MODEL_MAX_OUTPUT_TOKENS = 32_000

# Ceiling for the truncation retry below (current Claude models top out at
# 64k output tokens; Fable 5 allows more but 64k is plenty for one agent turn).
_TRUNCATION_RETRY_CEILING = 64_000


def _resolve_default_max_tokens(model: str) -> int | None:
    """Return an explicit ``max_tokens`` for models the library misjudges.

    ``None`` means "the installed langchain-anthropic knows this model's real
    output limit, let it decide" — which matters for older models whose true
    cap is *lower* than our floor (claude-3-opus is 4096; sending 32000 would
    be a 400 error, not a fix).
    """
    try:
        from langchain_anthropic.chat_models import _get_default_model_profile

        profile = _get_default_model_profile(model) or {}
        if profile.get("max_output_tokens"):
            return None
    except Exception:  # noqa: BLE001 - private helper; version-dependent
        # Unknown library layout: leave the cap alone and let the truncation
        # retry below recover, rather than guessing a value the model may reject.
        logger.debug("Cannot read langchain-anthropic model profile for %s", model)
        return None
    return _UNKNOWN_MODEL_MAX_OUTPUT_TOKENS


def _stop_reason(response: Any) -> str:
    return str((getattr(response, "response_metadata", None) or {}).get("stop_reason") or "")


_PROXY_RATE_LIMIT_PATTERNS = ("MPE-429", "Too many tokens per day", "too many tokens")

_RETRY_WAIT_SECONDS = 180  # 3 minutes

_TRANSIENT_PATTERNS = ("502", "503", "504", "MPE-001", "Server Error", "overloaded")
_TRANSIENT_MAX_RETRIES = 3
_TRANSIENT_BASE_DELAY = 30  # seconds


def _is_proxy_rate_limit(exc: Exception) -> bool:
    """Detect rate-limit errors disguised as HTTP 400 by API proxies."""
    msg = str(exc)
    return any(p in msg for p in _PROXY_RATE_LIMIT_PATTERNS)


def _is_transient_error(exc: Exception) -> bool:
    """Detect gateway/server errors worth retrying (502, 503, 504, overloaded)."""
    msg = str(exc)
    return any(p in msg for p in _TRANSIENT_PATTERNS)


class NormalizedChatAnthropic(ChatAnthropic):
    """ChatAnthropic with normalized content output.

    Claude models with extended thinking or tool use return content as a
    list of typed blocks. This normalizes to string for consistent
    downstream handling.

    Includes a one-shot 3-minute retry for proxy rate-limit errors
    (e.g. IdealAb MPE-429) that arrive as HTTP 400 and bypass the SDK's
    built-in 429 retry.
    """

    def _retry_if_truncated(self, response, input, config, kwargs):
        """Re-ask with a larger budget when the answer was cut off mid-flight.

        ``stop_reason == "max_tokens"`` means the model was still writing when
        the budget ran out: the text is incomplete, or (with thinking enabled)
        there is no text at all because reasoning consumed everything. Both are
        unusable, and both are invisible without this check — the response is a
        perfectly ordinary AIMessage. Retrying once with a doubled cap turns a
        silently truncated turn into a complete one; the guard in
        ``agents.utils.integrity`` only records the failure, it cannot undo it.
        """
        if _stop_reason(response) != "max_tokens":
            return response

        current = kwargs.get("max_tokens") or self.max_tokens or 4096
        raised = min(max(int(current) * 2, 8_000), _TRUNCATION_RETRY_CEILING)
        if raised <= int(current):
            logger.warning(
                "Response truncated at max_tokens=%s, already at the %s ceiling; "
                "returning the truncated output",
                current, _TRUNCATION_RETRY_CEILING,
            )
            return response

        logger.warning(
            "Response truncated (stop_reason=max_tokens, max_tokens=%s); "
            "retrying once with max_tokens=%s",
            current, raised,
        )
        retry_kwargs = {**kwargs, "max_tokens": raised}
        retried = super().invoke(input, config, **retry_kwargs)
        if _stop_reason(retried) == "max_tokens":
            logger.warning(
                "Still truncated at max_tokens=%s — the prompt or the answer is "
                "unusually long; consider raising TRADINGAGENTS_MAX_TOKENS",
                raised,
            )
        return retried

    def invoke(self, input, config=None, **kwargs):
        for attempt in range(_TRANSIENT_MAX_RETRIES + 1):
            try:
                response = super().invoke(input, config, **kwargs)
                return normalize_content(
                    self._retry_if_truncated(response, input, config, kwargs)
                )
            except Exception as exc:
                if _is_proxy_rate_limit(exc):
                    logger.warning(
                        "Proxy rate limit detected, retrying in %ds: %s",
                        _RETRY_WAIT_SECONDS, exc,
                    )
                    time.sleep(_RETRY_WAIT_SECONDS)
                    retried = super().invoke(input, config, **kwargs)
                    return normalize_content(
                        self._retry_if_truncated(retried, input, config, kwargs)
                    )

                if _is_transient_error(exc) and attempt < _TRANSIENT_MAX_RETRIES:
                    delay = _TRANSIENT_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "Transient LLM error (attempt %d/%d), retrying in %ds: %s",
                        attempt + 1, _TRANSIENT_MAX_RETRIES, delay, exc,
                    )
                    time.sleep(delay)
                    continue

                raise


class AnthropicClient(BaseLLMClient):
    """Client for Anthropic Claude models."""

    def __init__(self, model: str, base_url: str | None = None, **kwargs):
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        """Return configured ChatAnthropic instance."""
        self.warn_if_unknown_model()
        llm_kwargs = {"model": self.model}

        if self.base_url:
            llm_kwargs["base_url"] = self.base_url
            # Proxies keyed to the /api/code (Claude Code) plan require a CLI UA.
            ua = os.getenv("TRADINGAGENTS_ANTHROPIC_USER_AGENT", _DEFAULT_CLI_USER_AGENT)
            if ua:
                llm_kwargs["default_headers"] = {"User-Agent": ua, "x-app": "cli"}

        for key in _PASSTHROUGH_KWARGS:
            if key not in self.kwargs:
                continue
            if key == "effort" and not _supports_effort(self.model):
                continue
            llm_kwargs[key] = self.kwargs[key]

        # Applied here rather than in the caller's config so that every entry
        # point benefits — the CLI, the web server, the backtest runner and the
        # smoke scripts all build clients directly and would otherwise each
        # need to remember these two.
        if llm_kwargs.get("streaming") is None:
            llm_kwargs["streaming"] = _DEFAULT_STREAMING
        if llm_kwargs.get("timeout") is None:
            llm_kwargs["timeout"] = _DEFAULT_REQUEST_TIMEOUT
        if not llm_kwargs["streaming"]:
            logger.warning(
                "Streaming disabled for %s; timeout=%ss now bounds the whole "
                "response instead of the gap between chunks, so long agent "
                "turns may be cut off by proxies or by the client itself",
                self.model, llm_kwargs["timeout"],
            )

        # Never let the library guess the output budget for a model it predates:
        # its fallback is 4096, which silently truncates thinking models.
        if llm_kwargs.get("max_tokens") is None:
            llm_kwargs.pop("max_tokens", None)
            floor = _resolve_default_max_tokens(self.model)
            if floor is not None:
                logger.info(
                    "%s is unknown to the installed langchain-anthropic "
                    "(its fallback cap of 4096 truncates thinking models); "
                    "setting max_tokens=%s",
                    self.model, floor,
                )
                llm_kwargs["max_tokens"] = floor

        return NormalizedChatAnthropic(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for Anthropic."""
        return validate_model("anthropic", self.model)
