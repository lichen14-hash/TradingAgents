"""Quick test: HKMA fetchers in hk_macro.py."""
from tradingagents.dataflows.hk_macro import get_hk_macro_data

for ind in ["hk_hibor", "hk_exchange_rate", "hk_monetary_base"]:
    result = get_hk_macro_data(ind, "2026-06-30")
    first_line = result.split("\n")[0]
    has_data = "Data unavailable" not in result
    rows = result.count("| 2026")
    status = "OK" if has_data else "FAIL"
    print(f"{status}: {ind} ({rows} data rows) - {first_line}")
