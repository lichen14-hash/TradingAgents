"""Test FRED HK macro series availability."""
from dotenv import load_dotenv
load_dotenv()

import os
import requests

key = os.getenv("FRED_API_KEY")
assert key, "FRED_API_KEY not set"

# Candidate FRED series for HK macro
series_map = {
    "HK CPI": "HKCPIALLMINMEI",
    "HK Unemployment (OECD)": "LRHUTTTTHHQ156S",
    "HK GDP (World Bank)": "MKTGDPHKA646NWDB",
    "HK Interest Rate (OECD)": "IRSTCI01HKM156N",
    "HK Trade Balance (OECD)": "XTEXVA01HKM659S",
    "HK PPI": "PITGCG01HKM661N",
}

print("=== FRED HK Macro Series Test ===\n")
for name, sid in series_map.items():
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": sid,
        "api_key": key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": 3,
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        data = r.json()
        if "observations" in data and data["observations"]:
            obs = data["observations"]
            latest = obs[0]
            print(f"OK   {name} ({sid})")
            print(f"     latest: {latest['date']} = {latest['value']}")
        else:
            err = data.get("error_message", "no observations")
            print(f"FAIL {name} ({sid})")
            print(f"     {err}")
    except Exception as e:
        print(f"ERR  {name} ({sid})")
        print(f"     {e}")
    print()
