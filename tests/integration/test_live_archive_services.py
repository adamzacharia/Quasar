import os
from functools import lru_cache

import pytest


pytestmark = [
    pytest.mark.integration,
    pytest.mark.live,
    pytest.mark.skipif(
        os.getenv("RUN_LIVE_ARCHIVE_TESTS") != "1",
        reason="Set RUN_LIVE_ARCHIVE_TESTS=1 to run live archive tests.",
    ),
]


ALMA_TAP_URL = "https://almascience.nrao.edu/tap"


def _alma_tap_service():
    import pyvo

    return pyvo.dal.TAPService(ALMA_TAP_URL)


def _discover_public_mous_uids(limit=10):
    query = """
    SELECT TOP 20 member_ous_uid, target_name, proposal_id
    FROM ivoa.obscore
    WHERE member_ous_uid IS NOT NULL
      AND data_rights = 'Public'
      AND access_url IS NOT NULL
    ORDER BY proposal_id DESC
    """
    result = _alma_tap_service().search(query).to_table().to_pandas()
    if result.empty:
        pytest.skip("ALMA TAP returned no public MOUS rows.")
    return list(dict.fromkeys(str(value) for value in result["member_ous_uid"].dropna().tolist()))[:limit]


@lru_cache(maxsize=1)
def _working_datalink_listing():
    from integrations.datalink import DataLinkClient

    candidates = [os.getenv("QUASAR_LIVE_ALMA_MOUS_UID")] if os.getenv("QUASAR_LIVE_ALMA_MOUS_UID") else []
    candidates.extend(_discover_public_mous_uids())
    client = DataLinkClient()
    failures = []
    for mous_uid in candidates:
        listing = client.list_files(mous_uid=mous_uid)
        if listing.get("success") and listing.get("files"):
            return listing
        failures.append(f"{mous_uid}: {listing.get('error', 'no files')}")
    pytest.skip("ALMA DataLink returned no products for sampled public MOUS IDs: " + "; ".join(failures[:5]))


def _discover_datalink_fits_url():
    from services.data_product_triage import is_fits_product, product_rank

    listing = _working_datalink_listing()
    for file_info in sorted(listing["files"], key=product_rank):
        if is_fits_product(file_info) and file_info.get("access_url"):
            return file_info["access_url"]
    pytest.skip(f"ALMA DataLink returned no FITS products for {listing['mous_uid']}.")


def test_live_alma_tap_obscore_query_returns_rows():
    query = """
    SELECT TOP 1 target_name, proposal_id, member_ous_uid, s_ra, s_dec
    FROM ivoa.obscore
    WHERE proposal_id IS NOT NULL
    """
    df = _alma_tap_service().search(query).to_table().to_pandas()

    assert not df.empty
    assert {"target_name", "proposal_id", "s_ra", "s_dec"} <= set(df.columns)


def test_live_mast_query_returns_rows():
    from integrations.mast_client import MASTClient

    df = MASTClient().search_by_target("M87", mission="HST", radius="30s", max_results=5)

    assert not df.empty
    assert any(col in df.columns for col in ("target_name", "obs_id", "telescope"))


def test_live_alma_datalink_lists_products():
    listing = _working_datalink_listing()

    assert listing["success"] is True
    assert listing["mous_uid"].startswith("uid://")
    assert listing["total_files"] > 0
    assert all("filename" in item for item in listing["files"])


def test_live_remote_fits_header_access():
    from services.fits_processing import FITSProcessingService

    url = os.getenv("QUASAR_LIVE_FITS_URL") or _discover_datalink_fits_url()
    metadata = FITSProcessingService.extract_metadata_from_url(url)

    assert metadata["success"] is True
    assert metadata["filename"]
    assert "all_headers" in metadata
