"""NOIRLab Astro Data Lab SIA client.

The module imports PyVO only when a SIA search is executed so the application
can start on deployments that do not have optional VO/image dependencies.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional

from services import datalab_registry


DEFAULT_SIA_ENDPOINT = "https://datalab.noirlab.edu/sia/coadd_all"


class DatalabSiaClientError(RuntimeError):
    """Raised for SIA transport/setup failures."""


@dataclass
class SiaSearchResult:
    rows: List[Dict[str, Any]]
    coverage_gap: bool
    used_endpoint: str
    endpoints_tried: List[Dict[str, Any]] = field(default_factory=list)
    query_size: Dict[str, float] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": True,
            "rows": self.rows,
            "coverage_gap": self.coverage_gap,
            "used_endpoint": self.used_endpoint,
            "endpoints_tried": self.endpoints_tried,
            "query_size": self.query_size,
            "provenance": self.provenance,
        }


class _TimeoutSession:
    """Small requests.Session wrapper that gives PyVO calls a default timeout.

    get/post MUST be overridden here, not just request(): pyvo calls
    ``session.get(...)``, and __getattr__ hands back the REAL session's bound
    method — whose internal ``self.request`` is the real session's too — so a
    request()-only override never ran and the timeout was dead config
    (dl-sia-timeout-never-applied).
    """

    def __init__(self, timeout: float):
        import requests

        self._session = requests.Session()
        self.timeout = float(timeout)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._session, name)

    def request(self, method: str, url: str, **kwargs: Any):
        kwargs.setdefault("timeout", self.timeout)
        return self._session.request(method, url, **kwargs)

    def get(self, url: str, **kwargs: Any):
        kwargs.setdefault("timeout", self.timeout)
        return self._session.get(url, **kwargs)

    def post(self, url: str, **kwargs: Any):
        kwargs.setdefault("timeout", self.timeout)
        return self._session.post(url, **kwargs)


class DatalabSiaClient:
    """Thin wrapper over ``pyvo.dal.sia.SIAService`` for Data Lab images."""

    def __init__(self, *, catalog: Optional[str] = None, timeout: Optional[float] = None):
        self.catalog = catalog
        self.timeout = (
            float(timeout)
            if timeout is not None
            else float(os.getenv("DATALAB_SIA_TIMEOUT_SECONDS", "60"))
        )

    def search(
        self,
        ra: float,
        dec: float,
        fov_deg: float,
        *,
        endpoint: Optional[str] = None,
        catalog: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Search Data Lab SIA endpoints around an ICRS position.

        The return value is structured rather than a bare row list so callers can
        distinguish ordinary coverage gaps from transport errors.
        """

        ra_f = self._finite(ra, "ra")
        dec_f = self._finite(dec, "dec")
        fov_f = self._finite(fov_deg, "fov_deg")
        if not 0.0 <= ra_f < 360.0:
            raise ValueError("ra must be in [0, 360) degrees")
        if not -90.0 <= dec_f <= 90.0:
            raise ValueError("dec must be in [-90, 90] degrees")
        if fov_f <= 0:
            raise ValueError("fov_deg must be positive")

        size = self._sia_size(fov_f, dec_f)
        endpoints = self._candidate_endpoints(endpoint=endpoint, catalog=catalog or self.catalog)
        endpoints_tried: List[Dict[str, Any]] = []
        last_error: Optional[Exception] = None
        last_zero: Optional[SiaSearchResult] = None

        for ep in endpoints:
            try:
                rows = self._search_endpoint(ep, ra_f, dec_f, size)
            except Exception as exc:  # noqa: BLE001 - keep endpoint fallback available
                endpoints_tried.append({"endpoint": ep, "error": str(exc)})
                last_error = exc
                continue

            endpoints_tried.append({"endpoint": ep, "rowcount": len(rows)})
            result = SiaSearchResult(
                rows=rows,
                coverage_gap=len(rows) == 0,
                used_endpoint=ep,
                endpoints_tried=list(endpoints_tried),
                query_size={"ra_deg": size[0], "dec_deg": size[1]},
                provenance={
                    "service": "NOIRLab Astro Data Lab SIA",
                    "endpoint": ep,
                    "catalog": catalog or self.catalog,
                    "ra": ra_f,
                    "dec": dec_f,
                    "fov_deg": fov_f,
                    # Exact wire parameters of the SIA query, so the provenance
                    # surface can render a reproducible request line (CX-06):
                    # the descriptive scalars above are not what went over the
                    # wire (SIZE is cos(dec)-stretched).
                    "params": {
                        "POS": f"{ra_f},{dec_f}",
                        "SIZE": f"{size[0]},{size[1]}",
                    },
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            if rows:
                return result.to_dict()
            last_zero = result

        if last_zero is not None:
            return last_zero.to_dict()
        if last_error is not None:
            raise DatalabSiaClientError(f"All Data Lab SIA endpoints failed: {last_error}") from last_error
        raise DatalabSiaClientError("No Data Lab SIA endpoints configured")

    def _search_endpoint(self, endpoint: str, ra: float, dec: float, size: tuple[float, float]) -> List[Dict[str, Any]]:
        try:
            from pyvo.dal.sia import SIAService
        except ImportError as exc:
            raise ImportError("pyvo is required for Data Lab SIA image searches") from exc

        service = SIAService(endpoint, session=_TimeoutSession(self.timeout))
        result = service.search(pos=(ra, dec), size=size)
        table = result.to_table() if hasattr(result, "to_table") else result
        return self._table_to_rows(table)

    def _candidate_endpoints(self, *, endpoint: Optional[str], catalog: Optional[str]) -> List[str]:
        endpoints: List[str] = []
        if endpoint:
            endpoints.append(str(endpoint).strip())
        if catalog:
            key = str(catalog or "").strip().lower()
            entry = datalab_registry.DATALAB_CATALOGS.get(key) or {}
            endpoints.extend(str(ep).strip() for ep in entry.get("sia_endpoints") or [])
        endpoints.append(DEFAULT_SIA_ENDPOINT)
        result: List[str] = []
        for ep in endpoints:
            if ep and ep not in result:
                result.append(ep)
        return result

    @staticmethod
    def _sia_size(fov_deg: float, dec: float) -> tuple[float, float]:
        min_cos = float(os.getenv("DATALAB_SIA_MIN_COS_DEC", "0.05"))
        cos_dec = max(min_cos, abs(math.cos(math.radians(dec))))
        return min(360.0, fov_deg / cos_dec), fov_deg

    @staticmethod
    def _finite(value: float, name: str) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{name} must be finite")
        return number

    @staticmethod
    def _table_to_rows(table: Any) -> List[Dict[str, Any]]:
        if table is None:
            return []
        if hasattr(table, "to_pandas"):
            frame = table.to_pandas()
            return [_clean_mapping(row) for row in frame.to_dict(orient="records")]
        colnames = list(getattr(table, "colnames", []) or getattr(table, "columns", []) or [])
        rows: List[Dict[str, Any]] = []
        for row in table:
            if isinstance(row, Mapping):
                rows.append(_clean_mapping(row))
            elif colnames:
                rows.append(_clean_mapping({name: row[name] for name in colnames}))
            else:
                rows.append(_clean_mapping(dict(row)))
        return rows


def _clean_mapping(row: Mapping[str, Any]) -> Dict[str, Any]:
    clean: Dict[str, Any] = {}
    for key, value in row.items():
        clean[str(key)] = _clean_value(value)
    return clean


def _clean_value(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "filled"):
        try:
            value = value.filled(None)
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


__all__ = ["DEFAULT_SIA_ENDPOINT", "DatalabSiaClient", "DatalabSiaClientError", "SiaSearchResult"]
