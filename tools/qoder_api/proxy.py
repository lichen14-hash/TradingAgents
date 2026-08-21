"""Loopback OpenAI-compatible proxy in front of the Qoder model gateway.

Run it:

    $env:QODER_PAT = "<personal access token>"
    python -m tools.qoder_api.proxy

Then point any OpenAI-compatible client at ``http://127.0.0.1:8787/v1`` with an
arbitrary placeholder api key. The proxy owns the PAT and the token lifecycle so
the consumer keeps its single static-key assumption.

It binds loopback only and performs no authentication of its own -- a client on
this machine can spend the account's quota through it. Do not expose the port.
"""

from __future__ import annotations

import logging
import os
import time

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from tools.qoder_api.client import (
    DEFAULT_MODEL,
    MODEL_ROUTES,
    QoderGateway,
    QoderTokenManager,
    TokenExchangeError,
    UnknownModelError,
    chat_completions_url,
)

logger = logging.getLogger(__name__)

DEFAULT_PORT = 8787


def load_pat() -> str:
    """Read the PAT from the environment, falling back to a .env file."""
    pat = (os.environ.get("QODER_PAT") or "").strip()
    if pat:
        return pat
    try:
        from dotenv import load_dotenv
    except ImportError:
        pass
    else:
        load_dotenv()
        pat = (os.environ.get("QODER_PAT") or "").strip()
    if not pat:
        raise TokenExchangeError(
            "QODER_PAT is not set. Create a PAT in Qoder and export it as QODER_PAT "
            "(or put it in .env, which is gitignored)."
        )
    return pat


def create_app(gateway: QoderGateway, tokens: QoderTokenManager) -> FastAPI:
    app = FastAPI(title="qoder-api local proxy", docs_url=None, redoc_url=None)

    @app.post("/v1/chat/completions")
    def chat_completions(payload: dict) -> JSONResponse:
        if payload.get("stream"):
            # findings.md never tested SSE against the gateway. Failing loudly
            # beats handing back a body the caller will parse as a single chunk.
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": "stream=true is not supported by this proxy (untested upstream)",
                        "type": "invalid_request_error",
                    }
                },
            )
        try:
            status, body = gateway.chat_completions(payload)
        except UnknownModelError as exc:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": str(exc), "type": "invalid_request_error"}},
            )
        except TokenExchangeError as exc:
            return JSONResponse(
                status_code=502,
                content={"error": {"message": str(exc), "type": "auth_error"}},
            )
        return JSONResponse(status_code=status, content=body)

    @app.get("/v1/models")
    def list_models() -> dict:
        # The gateway has no listing endpoint (findings.md T8: /model/v1/models,
        # /model/v1/model/list and /v1/models all 404), so serve the probed set.
        created = int(time.time())
        return {
            "object": "list",
            "data": [
                {
                    "id": name,
                    "object": "model",
                    "created": created,
                    "owned_by": "qoder",
                    "routed_to": target,
                }
                for name, target in sorted(MODEL_ROUTES.items())
            ],
        }

    @app.get("/healthz")
    def healthz() -> dict:
        state = tokens.state()
        return {
            "ok": True,
            "upstream": chat_completions_url(),
            "default_model": DEFAULT_MODEL,
            "token_present": state.present,
            "token_seconds_remaining": (
                None if state.seconds_remaining is None else round(state.seconds_remaining)
            ),
            "exchanges": state.exchanges,
        }

    return app


def build_app() -> FastAPI:
    tokens = QoderTokenManager(load_pat())
    return create_app(QoderGateway(tokens), tokens)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    host = os.environ.get("QODER_PROXY_HOST", "127.0.0.1")
    port = int(os.environ.get("QODER_PROXY_PORT", DEFAULT_PORT))
    tokens = QoderTokenManager(load_pat())
    # Exchange eagerly so a bad PAT fails at startup instead of on first call.
    tokens.token()
    app = create_app(QoderGateway(tokens), tokens)
    logger.info("serving http://%s:%d/v1 -> %s", host, port, chat_completions_url())
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
