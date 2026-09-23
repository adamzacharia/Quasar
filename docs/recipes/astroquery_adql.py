"""ALMA Science Archive: ADQL through the TAP service with astroquery (tested recipe).

library: astroquery >= 0.4.7 (tested with 0.4.11)
last_tested: 2026-09-22
tested_against: https://almascience.nrao.edu/tap (live) and recorded responses in tests/unit/test_code_recipes.py
notes: Alma.query_tap(adql) runs any ADQL against ivoa.obscore and returns a pyvo TAPResults (.to_table()). Use TOP to cap
notes: rows; band_list is a space-separated token list; science_observation='T' selects science scans (calibrators are 'F').
notes: Aggregates (GROUP BY, COUNT(DISTINCT ...)) are supported by the ALMA TAP service.
"""
from astroquery.alma import Alma

alma = Alma()
alma.archive_url = "https://almascience.nrao.edu"

adql = """
SELECT TOP 500 proposal_id, member_ous_uid, target_name, band_list, spatial_resolution,
       frequency, bandwidth, obs_release_date, data_rights, science_observation
FROM ivoa.obscore
WHERE CONTAINS(POINT('ICRS', s_ra, s_dec), CIRCLE('ICRS', 204.2538, -29.8658, 0.0167)) = 1
  AND (band_list = '6' OR band_list LIKE '6 %' OR band_list LIKE '% 6' OR band_list LIKE '% 6 %')
  AND science_observation = 'T'
"""
result = alma.query_tap(adql).to_table()
print(len(result), "rows")

# Per-project counts computed at the server (no row cap needed)
agg = alma.query_tap("""
SELECT proposal_id, COUNT(DISTINCT member_ous_uid) AS n_mous, COUNT(*) AS n_rows
FROM ivoa.obscore
WHERE CONTAINS(POINT('ICRS', s_ra, s_dec), CIRCLE('ICRS', 204.2538, -29.8658, 0.0167)) = 1
GROUP BY proposal_id
""").to_table()
print(agg)
