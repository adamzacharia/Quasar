"""ALMA Science Archive: footprint-aware cone search in ADQL with pyvo (tested recipe).

library: pyvo >= 1.4 (tested with 1.8.1)
last_tested: 2026-09-22
tested_against: https://almascience.nrao.edu/tap (live) and recorded responses in tests/unit/test_code_recipes.py
notes: INTERSECTS(CIRCLE, s_region) catches mosaics whose footprint overlaps the cone without a pointing centre inside it;
notes: OR the point test so rows with a NULL s_region are not lost. Large multi-source footprint queries time out -- run one
notes: cone per source instead.
"""
import pyvo

service = pyvo.dal.TAPService("https://almascience.nrao.edu/tap")

ra, dec, radius_deg = 83.8221, -5.3911, 0.05     # Orion KL, 3 arcmin
adql = f"""
SELECT TOP 500 proposal_id, member_ous_uid, target_name, band_list, s_ra, s_dec, is_mosaic
FROM ivoa.obscore
WHERE (INTERSECTS(CIRCLE('ICRS', {ra}, {dec}, {radius_deg}), s_region) = 1
       OR CONTAINS(POINT('ICRS', s_ra, s_dec), CIRCLE('ICRS', {ra}, {dec}, {radius_deg})) = 1)
  AND science_observation = 'T'
"""
table = service.search(adql, maxrec=500).to_table()
print(len(table), "rows;", len(set(table["member_ous_uid"])), "datasets")
