"""ALMA Science Archive: cone search around coordinates with astroquery (tested recipe).

library: astroquery >= 0.4.7 (tested with 0.4.11)
last_tested: 2026-09-22
tested_against: https://almascience.nrao.edu/tap (live) and recorded responses in tests/unit/test_code_recipes.py
notes: Alma.query_region takes a SkyCoord and a radius (Quantity). Rows are ObsCore coverage records, not observations:
notes: count distinct member_ous_uid for datasets. Use payload keywords (band_list, spatial_resolution, ...) for server-side filters.
"""
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.alma import Alma

alma = Alma()
alma.archive_url = "https://almascience.nrao.edu"

center = SkyCoord(204.2538, -29.8658, unit="deg", frame="icrs")   # M83
table = alma.query_region(center, radius=1.0 * u.arcmin, public=True)

print(len(table), "coverage rows;", len(set(table["member_ous_uid"])), "datasets (MOUS)")

# Server-side filters through the archive's query payload, e.g. Band 6 with resolution better than 1 arcsec
filtered = alma.query_region(center, radius=1.0 * u.arcmin, public=True,
                             payload={"band_list": ["6"], "spatial_resolution": "<1"})
print("Band 6, <1 arcsec:", len(filtered), "rows")
