import logging
import re
import time
from typing import Any

from langchain_anthropic import ChatAnthropic

from .base_client import BaseLLMClient, normalize_content
from .validators import validate_model

logger = logging.getLogger(__name__)

_PASSTHROUGH_KWARGS = (
    "timeout", "max_retries", "api_key", "max_tokens", "temperature",
    "callbacks", "http_client", "http_async_client", "effort",
)

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

    def invoke(self, input, config=None, **kwargs):
        for attempt in range(_TRANSIENT_MAX_RETRIES + 1):
            try:
                return normalize_content(super().invoke(input, config, **kwargs))
            except Exception as exc:
                if _is_proxy_rate_limit(exc):
                    logger.warning(
                        "Proxy rate limit detected, retrying in %ds: %s",
                        _RETRY_WAIT_SECONDS, exc,
                    )
                    time.sleep(_RETRY_WAIT_SECONDS)
                    return normalize_content(super().invoke(input, config, **kwargs))

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

        for key in _PASSTHROUGH_KWARGS:
            if key not in self.kwargs:
                continue
            if key == "effort" and not _supports_effort(self.model):
                continue
            llm_kwargs[key] = self.kwargs[key]

        return NormalizedChatAnthropic(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for Anthropic."""
        return validate_model("anthropic", self.model)
