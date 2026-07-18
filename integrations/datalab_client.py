"""NOIRLab Astro Data Lab query client over the query-manager REST API.

This talks to the Data Lab query service directly with ``requests`` (the same
HTTP contract the official ``dl.queryClient`` wraps), so it is independent of the
``astro-datalab`` package and runs on Python 3.13 where that package will not
install. The public interface (query/submit/status/results/abort, DatalabResult,
round_numeric) is unchanged, so builders, the SQL governor, the registry, the
result store, the tools, and the tests are all transport-agnostic.

REST contract (verified against dl/queryClient.py):
  base:    https://datalab.noirlab.edu/query   (override via DATALAB_QUERY_URL)
  auth:    header X-DL-AuthToken: <token>       (anonymous token by default)
  sync:    GET {base}/query?sql=<quote_plus>&ofmt=csv&out=&async=False&drop=True
           (ADQL uses adql= instead of sql=); body = CSV result
  async:   same with async=True; body = job id
  status:  GET {base}/status?jobid=<id>          -> QUEUED|EXECUTING|COMPLETED|ERROR
  results: GET {base}/results?jobid=<id>&delete=False  -> CSV
  abort:   GET {base}/abort?jobid=<id>
"""

from __future__ import annotations

import io
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

import pandas as pd
import requests


# Well-known Data Lab anonymous auth token (read-only public access).
ANON_TOKEN = "anonymous.0.0.anon_access"
DEFAULT_SERVICE_URL = "https://datalab.noirlab.edu/query"


class DatalabClientError(RuntimeError):
    """Raised for Data Lab query-service transport/setup failures.

    ``jobid`` is set when a server-side job exists despite the failure (async
    fallback still running / errored), so callers can register it with the
    local job service and later polls resolve instead of "Unknown job".
    """

    def __init__(self, message: str, *, jobid: Optional[str] = None):
        super().__init__(message)
        self.jobid = jobid


@dataclass
class DatalabResult:
    """Normalized Data Lab query result."""

    dataframe: pd.DataFrame
    columns: List[str] = field(default_factory=list)
    dtypes: Dict[str, str] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dataframe(cls, dataframe: pd.DataFrame, provenance: Dict[str, Any]) -> "DatalabResult":
        frame = dataframe.copy()
        return cls(
            dataframe=frame,
            columns=[str(col) for col in frame.columns],
            dtypes={str(col): str(dtype) for col, dtype in frame.dtypes.items()},
            provenance=dict(provenance),
        )


def round_numeric(expr: str, n: int) -> str:
    """Emit a PostgreSQL/Data Lab-safe numeric rounding expression."""

    decimals = int(n)
    if decimals < 0 or decimals > 12:
        raise ValueError("round_numeric decimal places must be between 0 and 12")
    text = str(expr or "").strip()
    if not text:
        raise ValueError("round_numeric requires an expression")
    return f"ROUND(({text})::numeric, {decimals})"


class DatalabClient:
    """Thin requests-based wrapper over the Data Lab query-manager REST API."""

    def __init__(
        self,
        *,
        token: Optional[str] = None,
        timeout: Optional[float] = None,
        service_url: Optional[str] = None,
    ):
        # Fall back to the anonymous token so public queries work out of the box.
        self.token = token or os.getenv("DATALAB_TOKEN") or ANON_TOKEN
        self.timeout = (
            float(timeout)
            if timeout is not None
            else float(os.getenv("DATALAB_TIMEOUT_SECONDS", "60"))
        )
        self.service_url = (service_url or os.getenv("DATALAB_QUERY_URL") or DEFAULT_SERVICE_URL).rstrip("/")

    _AS_MATERIALIZED_RE = re.compile(r"\bAS\s+(?:NOT\s+)?MATERIALIZED\b", re.IGNORECASE)

    @classmethod
    def strip_materialized_for_async(cls, query_text: str) -> str:
        """Drop PostgreSQL `AS [NOT] MATERIALIZED` CTE fences for async jobs.

        The async query manager parses SQL with JSQLParser, which rejects the
        keyword ("JSQLParserException: Encountered MATERIALIZED" — live P9,
        2026-07-13); the sync endpoint accepts it. Only async submissions are
        rewritten: on the sync path the fence is what keeps crossmatch CTEs
        fast, so it must survive there. Replacement is quote-aware so string
        literals are never touched.
        """

        text = str(query_text or "")
        if "materialized" not in text.lower():
            return text
        parts = text.split("'")
        for i in range(0, len(parts), 2):  # even segments are outside literals
            parts[i] = cls._AS_MATERIALIZED_RE.sub("AS", parts[i])
        return "'".join(parts)

    @staticmethod
    def sanitize_query(query: str) -> str:
        """Strip one trailing semicolon and reject COPY-hostile final comments."""

        text = str(query or "").strip()
        if not text:
            raise ValueError("Data Lab query is empty")
        if text.endswith(";"):
            text = text[:-1].rstrip()
        lines = [line.rstrip() for line in text.splitlines()]
        last_non_empty = next((line.strip() for line in reversed(lines) if line.strip()), "")
        if last_non_empty.startswith("--"):
            raise ValueError(
                "Data Lab queries must not end with a '--' comment line; move the "
                "comment above the SELECT or remove it because COPY wrapping can fail."
            )
        return text

    def query(
        self,
        *,
        sql: Optional[str] = None,
        adql: Optional[str] = None,
        fmt: str = "pandas",
        timeout: Optional[float] = None,
        async_: bool = False,
        async_fallback: bool = True,
    ) -> "DatalabResult | str":
        """Run a native SQL or ADQL query. Returns DatalabResult, or a job id if async_.

        When a synchronous query hits the HTTP read timeout (heavy aggregates
        routinely exceed the sync window), the same query is transparently
        resubmitted as an async job and polled, instead of failing outright.
        Disable with async_fallback=False or DATALAB_ASYNC_FALLBACK=0.
        """

        query_text, mode = self._one_query(sql=sql, adql=adql)
        if async_:
            return self.submit(
                sql=query_text if mode == "sql" else None,
                adql=query_text if mode == "adql" else None,
                timeout=timeout,
            )
        try:
            body = self._get(
                "query",
                {mode: query_text, "ofmt": "csv", "out": "", "async": "False", "drop": "True"},
                timeout=timeout,
            )
        except DatalabClientError as exc:
            fallback_enabled = async_fallback and os.getenv("DATALAB_ASYNC_FALLBACK", "1") != "0"
            if fallback_enabled and "timed out" in str(exc).lower():
                return self._query_via_async_job(query_text, mode)
            raise
        frame = self._csv_to_dataframe(body, self._string_dtypes(query_text))
        catalog, table = self._first_table(query_text)
        provenance = {
            "catalog": catalog,
            "table": table,
            "query": query_text,
            "rowcount": int(len(frame)),
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "mode": mode,
        }
        return DatalabResult.from_dataframe(frame, provenance)

    def submit(
        self,
        *,
        sql: Optional[str] = None,
        adql: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> str:
        """Submit an asynchronous Data Lab query and return the job id."""

        query_text, mode = self._one_query(sql=sql, adql=adql)
        query_text = self.strip_materialized_for_async(query_text)
        body = self._get(
            "query",
            {mode: query_text, "ofmt": "csv", "out": "", "async": "True", "drop": "True"},
            timeout=timeout,
        )
        return body.strip().strip('"')

    def status(self, jobid: str) -> str:
        try:
            return self._get("status", {"jobid": str(jobid)}).strip()
        except DatalabClientError as exc:
            # /status returns the literal state string, and "ERROR" is a valid
            # job STATE — _get's ERROR-prefix transport check misread it as a
            # failure (live P15/P14: "Data Lab /status error: ERROR" with no
            # reason, and the caller's ERROR-state branch never ran).
            if str(exc).rstrip().endswith("error: ERROR"):
                return "ERROR"
            raise

    def error(self, jobid: str) -> str:
        """Server-side error text for a failed job (query-manager /error)."""
        try:
            return self._get("error", {"jobid": str(jobid)}).strip()
        except DatalabClientError as exc:
            # The /error body itself starts with "ERROR ..." for real failures —
            # that IS the reason text, not a transport error.
            text = str(exc)
            marker = f"/error error: "
            if marker in text:
                return text.split(marker, 1)[1]
            raise

    def results(self, jobid: str, *, fmt: str = "pandas", query_text: Optional[str] = None) -> DatalabResult:
        body = self._get("results", {"jobid": str(jobid), "delete": "False"})
        frame = self._csv_to_dataframe(body, self._string_dtypes(query_text))
        provenance = {
            "catalog": None,
            "table": None,
            "query": None,
            "rowcount": int(len(frame)),
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "jobid": str(jobid),
            "mode": "async_results",
        }
        return DatalabResult.from_dataframe(frame, provenance)

    def abort(self, jobid: str) -> str:
        return self._get("abort", {"jobid": str(jobid)}).strip()

    # ----- internals -------------------------------------------------------

    def _query_via_async_job(self, query_text: str, mode: str) -> DatalabResult:
        """Sync-timeout fallback: rerun the query as an async job and poll it.

        Bounded by DATALAB_ASYNC_MAX_WAIT_SECONDS (default 150s) so a single tool
        call cannot eat the whole agent turn; on poll expiry the job id is left
        running server-side and surfaced in the error for datalab_job_results.
        """
        import time as _time

        # Live-verified 2026-07-05: the Data Lab /status and /results endpoints
        # reject the anonymous token with HTTP 401 ("provided security token is
        # invalid"), so async jobs are only usable with a real login token.
        if self.token == ANON_TOKEN:
            raise DatalabClientError(
                "Data Lab sync query timed out, and the async-job fallback requires a "
                "real Data Lab login token (anonymous tokens get HTTP 401 from /status). "
                "Narrow the query instead: smaller radius, a coarser HEALPix column, a "
                "server-side aggregate, or selective value cuts."
            )
        max_wait = float(os.getenv("DATALAB_ASYNC_MAX_WAIT_SECONDS", "150"))
        jobid = self.query(
            sql=query_text if mode == "sql" else None,
            adql=query_text if mode == "adql" else None,
            async_=True,
        )
        deadline = _time.monotonic() + max_wait
        poll_s = 4.0
        while _time.monotonic() < deadline:
            state = self.status(str(jobid)).upper()
            if state == "COMPLETED":
                try:
                    result = self.results(str(jobid), query_text=query_text)
                except DatalabClientError as exc:
                    # A results download that blows the wall clock must still
                    # carry the jobid: the job COMPLETED server-side and the
                    # caller registers exc.jobid so datalab_job_results can
                    # fetch it later — otherwise the finished result is
                    # unreachable (dl-results-download-timeout-loses-jobid).
                    if getattr(exc, "jobid", None):
                        raise
                    raise DatalabClientError(
                        f"{exc} — async fallback job {jobid} COMPLETED server-side; "
                        f"fetch the rows with datalab_job_results using jobid={jobid}.",
                        jobid=str(jobid),
                    ) from exc
                catalog, table = self._first_table(query_text)
                # results() pre-fills catalog/table with None, so overwrite
                # explicitly — setdefault would keep the Nones.
                result.provenance.update(
                    {
                        "query": query_text,
                        "mode": f"{mode}_async_fallback",
                        "sync_timeout_fallback": True,
                        "jobid": str(jobid),
                        "catalog": catalog,
                        "table": table,
                    }
                )
                return result
            if state == "ERROR":
                # Surface the server's actual reason (live P15/P14: bare
                # "status ERROR" gave the model nothing to adapt to).
                reason = ""
                try:
                    reason = self.error(str(jobid))
                except Exception:  # noqa: BLE001 - reason fetch is best-effort
                    reason = ""
                raise DatalabClientError(
                    f"Data Lab async fallback job {jobid} failed with status ERROR"
                    + (f": {reason[:400]}" if reason else ""),
                    jobid=str(jobid),
                )
            _time.sleep(poll_s)
        raise DatalabClientError(
            f"Data Lab sync query timed out and the async fallback job is still running "
            f"after {max_wait:.0f}s. The job continues server-side — retrieve it with "
            f"datalab_job_status/datalab_job_results using jobid={jobid}.",
            jobid=str(jobid),
        )

    def _one_query(self, *, sql: Optional[str], adql: Optional[str]) -> tuple[str, str]:
        has_sql = sql is not None and str(sql).strip() != ""
        has_adql = adql is not None and str(adql).strip() != ""
        if has_sql == has_adql:
            raise ValueError("Provide exactly one of sql or adql")
        if has_sql:
            return self.sanitize_query(str(sql)), "sql"
        return self.sanitize_query(str(adql)), "adql"

    def _get(self, endpoint: str, params: Dict[str, Any], *, timeout: Optional[float] = None) -> str:
        seconds = float(timeout if timeout is not None else self.timeout)
        if seconds <= 0:
            raise ValueError("Data Lab timeout must be positive")
        # A scalar requests timeout is connect + PER-SOCKET-READ, not a total: a
        # server that trickles bytes every <60s held a "60s sync window" open for
        # 8 minutes (live P8). Stream the body under a hard wall-clock deadline;
        # the "timed out" wording is load-bearing (the async fallback and the
        # density-aggregate tiling both trigger on that substring).
        wall_seconds = float(os.getenv("DATALAB_SYNC_WALL_SECONDS", "0") or 0) or seconds * 1.5
        # Mirror the official client: query string is GET with the SQL url-encoded.
        encoded = "&".join(f"{key}={quote_plus(str(value))}" for key, value in params.items())
        url = f"{self.service_url}/{endpoint}?{encoded}"
        headers = {"Content-Type": "application/x-www-form-urlencoded", "X-DL-AuthToken": self.token}
        deadline = time.monotonic() + wall_seconds
        try:
            resp = requests.get(url, headers=headers, timeout=(min(seconds, 15.0), seconds), stream=True)
            chunks = []
            for chunk in resp.iter_content(chunk_size=65536):
                if time.monotonic() > deadline:
                    resp.close()
                    raise DatalabClientError(
                        f"Data Lab request to /{endpoint} failed: wall-clock deadline of "
                        f"{wall_seconds:.0f}s exceeded while the response was still streaming "
                        "(request timed out)"
                    )
                if chunk:
                    chunks.append(chunk)
            body = b"".join(chunks)
            encoding = resp.encoding or "utf-8"
            text = body.decode(encoding, errors="replace")
        except requests.RequestException as exc:
            raise DatalabClientError(f"Data Lab request to /{endpoint} failed: {exc}") from exc
        if resp.status_code != 200:
            raise DatalabClientError(
                f"Data Lab /{endpoint} returned HTTP {resp.status_code}: {text.strip()[:500]}"
            )
        if text.lstrip().startswith("ERROR"):
            raise DatalabClientError(f"Data Lab /{endpoint} error: {text.strip()[:500]}")
        return text

    @staticmethod
    def _string_dtypes(query_text: Optional[str]) -> Dict[str, Any]:
        """dtype overrides that keep id-like columns as strings, never floats.

        SMASH ids like '169.429960' carry a significant trailing zero; pandas
        float inference corrupts them and downstream exact-id lookups then match
        a different star (live P14). The registry declares which columns are
        string-typed per table; imported lazily so the transport module stays
        importable on its own.
        """
        try:
            from services import datalab_registry as registry
        except Exception:  # noqa: BLE001 - registry unavailable: no overrides
            return {}
        catalog = table = None
        if query_text:
            catalog, table = DatalabClient._first_table(query_text)
        return {col: str for col in registry.string_id_columns(catalog, table)}

    @staticmethod
    def _csv_to_dataframe(text: str, dtype_overrides: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        payload = (text or "").strip()
        if not payload:
            return pd.DataFrame()
        try:
            dtype = None
            if dtype_overrides:
                header = pd.read_csv(io.StringIO(text), nrows=0).columns
                dtype = {col: kind for col, kind in dtype_overrides.items() if col in set(header)} or None
            return pd.read_csv(io.StringIO(text), dtype=dtype)
        except Exception as exc:  # noqa: BLE001 - surface a clear transport error
            raise DatalabClientError(
                f"Could not parse Data Lab CSV response: {exc}; body starts: {payload[:200]}"
            ) from exc

    @staticmethod
    def _first_table(query: str) -> tuple[Optional[str], Optional[str]]:
        match = re.search(
            r"\bFROM\s+([A-Za-z_][\w]*)\.([A-Za-z_][\w]*)\b",
            query,
            flags=re.IGNORECASE,
        )
        if not match:
            return None, None
        return match.group(1), match.group(2)


__all__ = ["ANON_TOKEN", "DatalabClient", "DatalabClientError", "DatalabResult", "round_numeric"]
