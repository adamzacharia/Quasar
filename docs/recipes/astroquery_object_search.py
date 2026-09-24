"""ALMA Science Archive: query by object name with astroquery (tested recipe).

library: astroquery >= 0.4.7 (tested with 0.4.11)
last_tested: 2026-09-22
tested_against: https://almascience.nrao.edu/tap (live) and recorded responses in tests/unit/test_code_recipes.py
notes: Alma.query_object resolves the name with SESAME and runs a cone search (default 10 arcmin) over ivoa.obscore. The
notes: result is an astropy Table with one row per (MOUS, target, spectral window) coverage record; 'member_ous_uid' is the
notes: dataset identifier. There is no Alma.query_sql or Alma.get_data_links -- use query_tap and get_data_info (see the other recipes).
notes: public is three-state: True = public only, False = PROPRIETARY only, None = both (ALminer uses the same convention).
"""
from astroquery.alma import Alma

alma = Alma()
alma.archive_url = "https://almascience.nrao.edu"   # or almascience.eso.org / almascience.nao.ac.jp

# One row per ObsCore coverage record for M83. public=True: public data only
# (public=False returns PROPRIETARY data only; public=None returns both).
table = alma.query_object("M83", public=True)

print(len(table), "rows")
print(table["proposal_id", "member_ous_uid", "target_name", "band_list", "spatial_resolution", "obs_release_date"][:10])

# Distinct datasets (MOUS) and projects
import numpy as np
print("MOUS:", len(np.unique(table["member_ous_uid"])), "projects:", len(np.unique(table["proposal_id"])))

# Keep Band 6 only: band_list is a space-separated token list ("6", "5 10"), so match the token
band6 = table[[("6" in str(b).split()) for b in table["band_list"]]]
print("Band 6 rows:", len(band6))
