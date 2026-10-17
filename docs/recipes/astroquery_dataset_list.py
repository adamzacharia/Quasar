"""ALMA Science Archive: list every dataset (MOUS) of a source in given bands released before a date, with astroquery (tested recipe).

library: astroquery >= 0.4.7 (tested with 0.4.11); astropy SkyCoord.from_name (CDS Sesame name resolver)
last_tested: 2026-10-16
tested_against: https://almascience.eso.org/tap (live: NGC 253, Band 3, released before 2026-01-01)
notes: Alma.query_object uses a fixed 10 arcmin cone and has no release-date filter, so a complete, dated list is an ADQL
notes: query through Alma.query_tap: resolve the name, choose a radius of at least half the source's catalogued major
notes: axis (not a fixed 1 arcmin), keep footprints that overlap the cone (INTERSECTS on s_region) or pointing centres
notes: inside it, filter band and obs_release_date in the WHERE clause, and keep DISTINCT member_ous_uid (ObsCore rows
notes: are one per spectral window and execution). Files for the listed MOUS: Alma.get_data_info(uids) then
notes: Alma.download_files(urls). The large-cone INTERSECTS query takes 30-60 s.
"""
from astropy.coordinates import SkyCoord
from astroquery.alma import Alma

TARGET = "NGC 253"            # any SIMBAD-resolvable name
RADIUS_ARCMIN = 14.0          # about half the catalogued major axis, so the whole disc is covered
BAND = 3
RELEASED_BEFORE = "2026-01-01"

alma = Alma()
alma.archive_url = "https://almascience.eso.org"   # or almascience.nrao.edu / almascience.nao.ac.jp

c = SkyCoord.from_name(TARGET)
ra, dec, r = c.ra.deg, c.dec.deg, RADIUS_ARCMIN / 60.0
adql = f"""
SELECT DISTINCT member_ous_uid, proposal_id
FROM ivoa.obscore
WHERE (INTERSECTS(CIRCLE('ICRS', {ra:.6f}, {dec:.6f}, {r:.6f}), s_region) = 1
       OR CONTAINS(POINT('ICRS', s_ra, s_dec), CIRCLE('ICRS', {ra:.6f}, {dec:.6f}, {r:.6f})) = 1)
  AND (band_list = '{BAND}' OR band_list LIKE '{BAND} %' OR band_list LIKE '% {BAND}' OR band_list LIKE '% {BAND} %')
  AND obs_release_date < '{RELEASED_BEFORE}'
  AND science_observation = 'T'
"""
result = alma.query_tap(adql)
# A list of EVERY dataset must not be silently capped: the service marks a cut answer OVERFLOW.
if getattr(result, "query_status", "OK") == "OVERFLOW":
    raise SystemExit("row cap reached: this list would be incomplete; narrow the query or raise the row limit")
table = result.to_table()

mous = sorted({str(uid) for uid in table["member_ous_uid"]})
print(f"{len(mous)} MOUS in {len(set(table['proposal_id']))} projects ({TARGET}, Band {BAND}, released before {RELEASED_BEFORE})")
for uid in mous:
    print(uid)
