"""ALMA Science Archive: list every dataset (MOUS) of a source in given bands released before a date, with pyvo (tested recipe).

library: pyvo >= 1.4 (tested with 1.8.1); astropy SkyCoord.from_name (CDS Sesame name resolver)
last_tested: 2026-10-16
tested_against: https://almascience.eso.org/tap (live: NGC 253, Band 3, released before 2026-01-01)
notes: Resolve the name to a position, then search a cone that covers the WHOLE source. For an extended source use at
notes: least half its catalogued major axis (SIMBAD/NED galaxy size) as the radius, not a fixed 1 arcmin: pointings on
notes: the outer disc are otherwise missed. Footprints that overlap the cone count (INTERSECTS on s_region), as well as
notes: pointing centres inside it. ObsCore rows are one per spectral window and execution, so keep DISTINCT
notes: member_ous_uid (the dataset id). The release-date cut and the band filter go in the ADQL (band_list is a
notes: space-separated token list). science_observation = 'T' drops calibrator rows. The large-cone INTERSECTS query
notes: takes 30-60 s; pass a generous timeout.
"""
import pyvo
from astropy.coordinates import SkyCoord

TARGET = "NGC 253"            # any SIMBAD-resolvable name
RADIUS_ARCMIN = 14.0          # about half the catalogued major axis, so the whole disc is covered
BANDS = [3]                   # ALMA receiver bands to keep
RELEASED_BEFORE = "2026-01-01"

c = SkyCoord.from_name(TARGET)
ra, dec, r = c.ra.deg, c.dec.deg, RADIUS_ARCMIN / 60.0
band_sql = " OR ".join(
    f"band_list = '{b}' OR band_list LIKE '{b} %' OR band_list LIKE '% {b}' OR band_list LIKE '% {b} %'" for b in BANDS
)
adql = f"""
SELECT DISTINCT member_ous_uid, proposal_id
FROM ivoa.obscore
WHERE (INTERSECTS(CIRCLE('ICRS', {ra:.6f}, {dec:.6f}, {r:.6f}), s_region) = 1
       OR CONTAINS(POINT('ICRS', s_ra, s_dec), CIRCLE('ICRS', {ra:.6f}, {dec:.6f}, {r:.6f})) = 1)
  AND ({band_sql})
  AND obs_release_date < '{RELEASED_BEFORE}'
  AND science_observation = 'T'
"""
service = pyvo.dal.TAPService("https://almascience.eso.org/tap")   # or almascience.nrao.edu / almascience.nao.ac.jp
result = service.search(adql, maxrec=20000)
# A list of EVERY dataset must not be silently capped: the service marks a cut answer OVERFLOW.
if getattr(result, "query_status", "OK") == "OVERFLOW":
    raise SystemExit("row cap reached: this list would be incomplete; narrow the query or raise the row limit")
rows = result.to_table()

mous = sorted({str(uid) for uid in rows["member_ous_uid"]})
print(f"{len(mous)} MOUS in {len(set(rows['proposal_id']))} projects ({TARGET}, bands {BANDS}, released before {RELEASED_BEFORE})")
for uid in mous:
    print(uid)
