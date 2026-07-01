"""Test HKMA API availability for HK macro data."""
import requests

# HKMA Open API: https://apidocs.hkma.gov.hk/
# No API key required, free access

BASE = "https://api.hkma.gov.hk/public"

print("=== HKMA API Test ===\n")

# Try multiple endpoint path patterns to discover the correct one
endpoints = {
    "HIBOR end-of-period": "/market-data-and-statistics/monthly-statistical-bulletin/er-ir/hk-interbank-interest-rate-end-of-period",
    "HIBOR daily": "/market-data-and-statistics/monthly-statistical-bulletin/er-ir/hk-interbank-interest-rates-daily",
    "Exchange rate end-of-period": "/market-data-and-statistics/monthly-statistical-bulletin/er-ir/exchange-rates-end-of-period",
    "Exchange rate daily": "/market-data-and-statistics/monthly-statistical-bulletin/er-ir/exchange-rates-daily",
    "Interbank liquidity": "/market-data-and-statistics/daily-monetary-statistics/daily-figures-interbank-liquidity",
    "Monetary base": "/market-data-and-statistics/daily-monetary-statistics/daily-figures-monetary-base",
    "Money supply": "/market-data-and-statistics/monthly-statistical-bulletin/money/money-supply",
    "Composite interest": "/market-data-and-statistics/monthly-statistical-bulletin/er-ir/composite-interest-rate",
}

for name, path in endpoints.items():
    url = BASE + path
    try:
        r = requests.get(url, params={"pagesize": 2}, timeout=10)
        d = r.json()
        ok = d.get("header", {}).get("success", False)
        status = "OK" if ok else "FAIL"
        print(f"{status}: {name}")
        if ok:
            records = d.get("result", {}).get("records", [])
            if records:
                keys = list(records[0].keys())
                print(f"  Fields: {keys[:10]}")
                print(f"  Sample: {records[0]}")
        else:
            err = d.get("header", {}).get("err_msg", "unknown")
            print(f"  Error: {err}")
    except Exception as e:
        print(f"ERR: {name} - {e}")
    print()

print("=== Test Complete ===")
