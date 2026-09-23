"""ALMA Science Archive: ADQL with pyvo's TAPService (tested recipe).

library: pyvo >= 1.4 (tested with 1.8.1)
last_tested: 2026-09-22
tested_against: https://almascience.nrao.edu/tap (live) and recorded responses in tests/unit/test_code_recipes.py
notes: Three mirrors serve the same ivoa.obscore table: almascience.nrao.edu, almascience.eso.org, almascience.nao.ac.jp.
notes: Use maxrec to cap rows; result.to_table() gives an astropy Table, .to_table().to_pandas() a DataFrame.
"""
import pyvo

service = pyvo.dal.TAPService("https://almascience.nrao.edu/tap")

adql = """
SELECT TOP 200 proposal_id, member_ous_uid, target_name, band_list, spatial_resolution, obs_release_date
FROM ivoa.obscore
WHERE CONTAINS(POINT('ICRS', s_ra, s_dec), CIRCLE('ICRS', 204.2538, -29.8658, 0.0167)) = 1
  AND (band_list = '6' OR band_list LIKE '6 %' OR band_list LIKE '% 6' OR band_list LIKE '% 6 %')
"""
result = service.search(adql, maxrec=200)
table = result.to_table()
print(len(table), "rows")
print(table[:5])

# Table description (columns of ivoa.obscore)
obscore = service.tables["ivoa.obscore"]
print([c.name for c in obscore.columns][:20])
