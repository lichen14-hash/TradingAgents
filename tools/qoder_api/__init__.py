"""Local bridge that turns a Qoder PAT into an OpenAI-compatible endpoint.

Self-contained on purpose (only fastapi/httpx): this package is meant to be
copied into whichever repo consumes it. See ``findings.md`` for the field
research this implements.
"""

from tools.qoder_api.client import (
    CHAT_COMPLETIONS_URL,
    EXCHANGE_URL,
    MODEL_ROUTES,
    QoderGateway,
    QoderTokenManager,
    TokenExchangeError,
    UnknownModelError,
    resolve_model,
)

__all__ = [
    "CHAT_COMPLETIONS_URL",
    "EXCHANGE_URL",
    "MODEL_ROUTES",
    "QoderGateway",
    "QoderTokenManager",
    "TokenExchangeError",
    "UnknownModelError",
    "resolve_model",
]
