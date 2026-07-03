"""Test which models are available on idealab endpoint."""
import requests

API_KEY = "dd17919f6f91b01c95351ddc3271d74a"
BASE = "https://idealab.alibaba-inc.com/api/code/v1/messages"
HEADERS = {
    "x-api-key": API_KEY,
    "content-type": "application/json",
    "anthropic-version": "2023-06-01",
}

models = [
    "claude-sonnet-4-6",
    "claude-sonnet-4-20250514",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    "claude-3-haiku-20240307",
]

for m in models:
    try:
        r = requests.post(BASE, headers=HEADERS, json={
            "model": m, "max_tokens": 20,
            "messages": [{"role": "user", "content": "Say OK"}],
        }, timeout=30)
        status = r.status_code
        body = r.text[:300]
        print(f"{m}: [{status}] {body}")
    except Exception as e:
        print(f"{m}: ERROR - {e}")
    print()
