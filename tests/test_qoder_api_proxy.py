"""Tests for the local Qoder gateway proxy (tools/qoder_api).

The upstream endpoint is undocumented and credential-gated, so everything here
runs against httpx.MockTransport. The cases mirror the specific traps recorded
in findings.md rather than the happy path.
"""

import httpx
import pytest
from fastapi.testclient import TestClient

from tools.qoder_api.client import (
    MODEL_ROUTES,
    QoderGateway,
    QoderTokenManager,
    TokenExchangeError,
    UnknownModelError,
    resolve_model,
)
from tools.qoder_api.proxy import create_app

pytestmark = pytest.mark.unit


class FakeClock:
    def __init__(self, now: float = 1_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_tokens(
    *,
    responses=None,
    clock=None,
    refresh_margin_s=3600.0,
    record=None,
):
    """Build a token manager whose exchange endpoint is mocked."""
    queue = list(responses or [{"token": "jt-abc", "expires_in": 86_400_000}])

    def handler(request: httpx.Request) -> httpx.Response:
        if record is not None:
            record.append(request)
        payload = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(payload, int):
            return httpx.Response(payload, json={"error": "unauthorized"})
        return httpx.Response(200, json=payload)

    return QoderTokenManager(
        "pat-secret",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        clock=clock or FakeClock(),
        refresh_margin_s=refresh_margin_s,
    )


# --- token lifecycle -------------------------------------------------------


def test_expires_in_is_read_as_milliseconds():
    """findings.md T7: 86400000 means 24h. Read as seconds it is ~1000 days."""
    clock = FakeClock()
    tokens = make_tokens(clock=clock)
    tokens.token()

    remaining = tokens.state().seconds_remaining
    assert remaining == pytest.approx(86_400, abs=1)


def test_second_valued_expires_in_still_works():
    clock = FakeClock()
    tokens = make_tokens(responses=[{"token": "jt-abc", "expires_in": 7200}], clock=clock)
    tokens.token()

    assert tokens.state().seconds_remaining == pytest.approx(7200, abs=1)


def test_missing_expires_in_falls_back_to_23h():
    tokens = make_tokens(responses=[{"token": "jt-abc"}])
    tokens.token()

    assert tokens.state().seconds_remaining == pytest.approx(23 * 3600, abs=1)


def test_token_is_cached_between_calls():
    record = []
    tokens = make_tokens(record=record)

    assert tokens.token() == "jt-abc"
    assert tokens.token() == "jt-abc"
    assert len(record) == 1


def test_token_refreshes_before_expiry():
    clock = FakeClock()
    record = []
    tokens = make_tokens(clock=clock, refresh_margin_s=3600, record=record)
    tokens.token()

    clock.advance(86_400 - 1800)  # inside the 1h margin
    tokens.token()

    assert len(record) == 2
    assert tokens.state().exchanges == 2


def test_short_ttl_does_not_re_exchange_every_call():
    """Margin is clamped to half the TTL, otherwise a short token hot-loops."""
    clock = FakeClock()
    record = []
    tokens = make_tokens(
        responses=[{"token": "jt-abc", "expires_in": 600}],
        clock=clock,
        refresh_margin_s=3600,
        record=record,
    )
    tokens.token()
    tokens.token()

    assert len(record) == 1


def test_short_ttl_still_refreshes_once_inside_its_own_margin():
    """The clamp must not disable refresh entirely -- 600s token, 300s margin."""
    clock = FakeClock()
    record = []
    tokens = make_tokens(
        responses=[{"token": "jt-abc", "expires_in": 600}],
        clock=clock,
        refresh_margin_s=3600,
        record=record,
    )
    tokens.token()

    clock.advance(350)
    tokens.token()

    assert len(record) == 2


def test_pat_travels_in_body_not_header():
    """findings.md T7: PAT in the body; the exchange call sends no Authorization."""
    record = []
    tokens = make_tokens(record=record)
    tokens.token()

    request = record[0]
    assert "authorization" not in {k.lower() for k in request.headers}
    assert b"pat-secret" in request.content
    assert request.headers["User-Agent"].startswith("qoder/")


def test_failed_exchange_error_does_not_echo_response_body():
    tokens = make_tokens(responses=[401])

    with pytest.raises(TokenExchangeError) as excinfo:
        tokens.token()
    assert "401" in str(excinfo.value)
    assert "unauthorized" not in str(excinfo.value)


def test_empty_pat_is_rejected_up_front():
    with pytest.raises(TokenExchangeError):
        QoderTokenManager("   ")


# --- model resolution ------------------------------------------------------


def test_rejected_sdk_key_is_translated():
    """findings.md T8: dfmodel is rejected upstream, deepseek-v4-flash is not."""
    assert resolve_model("dfmodel") == "deepseek-v4-flash"


def test_rejected_provider_name_is_translated():
    """...and the inverse: deepseek-v4-pro is rejected, dmodel works."""
    assert resolve_model("deepseek-v4-pro") == "dmodel"


def test_missing_model_uses_measured_default():
    assert resolve_model(None) == "deepseek-v4-flash"


def test_unknown_model_lists_the_accepted_names():
    with pytest.raises(UnknownModelError) as excinfo:
        resolve_model("gpt-4o")
    assert "qmodel" in str(excinfo.value)


def test_every_route_resolves_to_itself():
    for name in MODEL_ROUTES:
        assert resolve_model(name) == name


# --- gateway forwarding ----------------------------------------------------


def make_gateway(statuses, *, record=None):
    queue = list(statuses)

    def handler(request: httpx.Request) -> httpx.Response:
        if record is not None:
            record.append(request)
        status = queue.pop(0) if len(queue) > 1 else queue[0]
        if status == 200:
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})
        return httpx.Response(status, json={"error": "unauthorized"})

    tokens = make_tokens()
    gateway = QoderGateway(
        tokens,
        chat_url="https://api2-v2.qoder.sh/model/v1/chat/completions",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    return gateway, tokens


def test_forwards_with_bearer_and_rewritten_model():
    record = []
    gateway, _ = make_gateway([200], record=record)

    status, body = gateway.chat_completions({"model": "dfmodel", "messages": []})

    assert status == 200
    assert body["choices"][0]["message"]["content"] == "ok"
    assert record[0].headers["Authorization"] == "Bearer jt-abc"
    assert b'"deepseek-v4-flash"' in record[0].content


def test_401_triggers_one_re_exchange_and_retry():
    record = []
    gateway, tokens = make_gateway([401, 200], record=record)

    status, _ = gateway.chat_completions({"messages": []})

    assert status == 200
    assert len(record) == 2
    assert tokens.state().exchanges == 2


def test_persistent_401_is_not_retried_forever():
    record = []
    gateway, _ = make_gateway([401], record=record)

    status, _ = gateway.chat_completions({"messages": []})

    assert status == 401
    assert len(record) == 2


def test_non_json_upstream_body_is_wrapped():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, text="<html>bad gateway</html>")

    gateway = QoderGateway(
        make_tokens(),
        chat_url="https://api2-v2.qoder.sh/model/v1/chat/completions",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    status, body = gateway.chat_completions({"messages": []})

    assert status == 502
    assert body["error"]["type"] == "upstream_error"


# --- proxy surface ---------------------------------------------------------


@pytest.fixture()
def client():
    gateway, tokens = make_gateway([200])
    with TestClient(create_app(gateway, tokens)) as test_client:
        yield test_client


def test_proxy_passes_a_completion_through(client):
    response = client.post("/v1/chat/completions", json={"messages": [], "temperature": 0})

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "ok"


def test_proxy_rejects_streaming(client):
    response = client.post("/v1/chat/completions", json={"messages": [], "stream": True})

    assert response.status_code == 400
    assert "stream" in response.json()["error"]["message"]


def test_proxy_rejects_unknown_model_locally(client):
    response = client.post("/v1/chat/completions", json={"model": "gpt-4o", "messages": []})

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_models_listing_exposes_routing_targets(client):
    data = client.get("/v1/models").json()["data"]
    by_id = {entry["id"]: entry["routed_to"] for entry in data}

    assert by_id["lite"] == "qwen3-coder-plus"
    assert by_id["mmodel"] == "MiniMax-M2.5"
    assert len(by_id) == len(MODEL_ROUTES)


def test_healthz_reports_ttl_without_leaking_the_token(client):
    client.post("/v1/chat/completions", json={"messages": []})
    body = client.get("/healthz").json()

    assert body["token_present"] is True
    assert body["token_seconds_remaining"] == pytest.approx(86_400, abs=2)
    assert "jt-abc" not in client.get("/healthz").text
