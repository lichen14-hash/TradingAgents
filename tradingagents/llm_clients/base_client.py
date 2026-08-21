import logging
import warnings
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


def normalize_content(response):
    """Normalize LLM response content to a plain string.

    Multiple providers (OpenAI Responses API, Google Gemini 3) return content
    as a list of typed blocks, e.g. [{'type': 'reasoning', ...}, {'type': 'text', 'text': '...'}].
    Downstream agents expect response.content to be a string. This extracts
    and joins the text blocks, discarding reasoning/metadata blocks.

    When every block is discarded the result is an empty string, which used to
    be indistinguishable from "the model had nothing to say". That is the exact
    failure mode behind the 2026-08-11 300760.SZ incident: a thinking-enabled
    Claude model spent its whole budget on reasoning blocks and returned no
    text block, so the Bear Researcher's argument silently became "". The
    warning below is the only place that fact is observable, so it must be
    logged even though the guard in
    :func:`tradingagents.agents.utils.integrity.invoke_text_guarded` handles
    the retry.
    """
    content = response.content
    if isinstance(content, list):
        texts = [
            item.get("text", "") if isinstance(item, dict) and item.get("type") == "text"
            else item if isinstance(item, str) else ""
            for item in content
        ]
        response.content = "\n".join(t for t in texts if t)
        if content and not response.content:
            dropped = sorted({
                str(item.get("type", "unknown")) if isinstance(item, dict) else type(item).__name__
                for item in content
            })
            logger.warning(
                "normalize_content dropped every block (%d blocks, types=%s) — "
                "no text block in the response; likely thinking budget exhausted "
                "or output truncated (finish_reason=%s)",
                len(content), ",".join(dropped),
                (getattr(response, "response_metadata", None) or {}).get("stop_reason", "unknown"),
            )
    return response


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(self, model: str, base_url: str | None = None, **kwargs):
        self.model = model
        self.base_url = base_url
        self.kwargs = kwargs

    def get_provider_name(self) -> str:
        """Return the provider name used in warning messages."""
        provider = getattr(self, "provider", None)
        if provider:
            return str(provider)
        return self.__class__.__name__.removesuffix("Client").lower()

    def warn_if_unknown_model(self) -> None:
        """Warn when the model is outside the known list for the provider."""
        if self.validate_model():
            return

        warnings.warn(
            (
                f"Model '{self.model}' is not in the known model list for "
                f"provider '{self.get_provider_name()}'. Continuing anyway."
            ),
            RuntimeWarning,
            stacklevel=2,
        )

    @abstractmethod
    def get_llm(self) -> Any:
        """Return the configured LLM instance."""
        pass

    @abstractmethod
    def validate_model(self) -> bool:
        """Validate that the model is supported by this client."""
        pass
