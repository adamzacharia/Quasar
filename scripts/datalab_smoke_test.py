"""Live NOIRLab Astro Data Lab smoke test (the P0 empirical gate).

Performs real network calls against the Data Lab query-manager REST API via the
requests-based DatalabClient (no astro-datalab dependency; runs on Python 3.13).
It proves the two things P0 depends on:
  1. anonymous native SQL with `WITH ... MATERIALIZED` works (the P9 path), and
  2. a q3c cone COUNT works.
Run manually in an environment with outbound network access:
    python scripts/datalab_smoke_test.py
Set DATALAB_TOKEN (read-only) first if anonymous access is rejected/rate-limited.
"""

from __future__ import annotations

import os
import sys

# Allow running as a bare script: put the repo root on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integrations.datalab_client import DatalabClient, DatalabClientError


def main() -> int:
    client = DatalabClient()  # anonymous token by default; honors DATALAB_TOKEN / DATALAB_QUERY_URL
    gate_sql = (
        "WITH g AS MATERIALIZED ("
        "SELECT ra,dec FROM gaia_dr3.gaia_source "
        "WHERE q3c_radial_query(ra,dec,229.022,-0.112,0.1) LIMIT 5"
        ") SELECT * FROM g"
    )
    count_sql = (
        "SELECT COUNT(*) AS row_count FROM gaia_dr3.gaia_source "
        "WHERE q3c_radial_query(ra,dec,229.022,-0.112,0.1)"
    )
    try:
        gate = client.query(sql=gate_sql)
        count = client.query(sql=count_sql)
        n = len(gate.dataframe)
        cnt = count.dataframe.iloc[0].to_dict() if not count.dataframe.empty else "empty"
        print(f"service: {client.service_url}  token: {'env/custom' if os.getenv('DATALAB_TOKEN') else 'anonymous'}")
        print(f"PASS: native SQL + WITH MATERIALIZED gate returned {n} row(s).")
        print(f"PASS: q3c cone COUNT(*) = {cnt}.")
        print("=> Anonymous native-SQL/CTE path WORKS. P0 transport validated.")
        return 0
    except DatalabClientError as exc:
        print(f"FAIL: {exc}")
        print("If anonymous native SQL is rejected/rate-limited, set DATALAB_TOKEN (read-only) and retry.")
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: unexpected error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
