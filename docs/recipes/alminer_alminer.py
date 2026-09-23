"""ALMA Science Archive with ALMiner: target search, cone search, custom ADQL, download (tested recipe).

library: alminer >= 0.1.2 (tested with 0.1.2)
last_tested: 2026-09-22
tested_against: https://almascience.nrao.edu/tap (live) and recorded responses in tests/unit/test_code_recipes.py
notes: alminer functions return pandas DataFrames with one row per observation record; columns include project_code,
notes: member_ous_uid, ALMA_source_name, band_list, ang_res_arcsec, min_freq_GHz, max_freq_GHz, obs_release_date.
notes: alminer.target / conesearch / keysearch / run_query; alminer.download_data(df, fitsonly=True, dryrun=True) lists files.
"""
import alminer

# 1. By target name (SESAME-resolved; search_radius in arcmin)
obs = alminer.target(["M83"], search_radius=1.0, public=True, print_targets=False)
print(obs[["project_code", "member_ous_uid", "ALMA_source_name", "band_list", "ang_res_arcsec"]].head())

# 2. By coordinates (degrees; search_radius in arcmin)
cone = alminer.conesearch(ra=204.2538, dec=-29.8658, search_radius=1.0, public=True, print_targets=False)
print(len(cone), "rows;", cone["member_ous_uid"].nunique(), "datasets")

# 3. Custom ADQL through the same TAP service
adql = ("SELECT TOP 100 * FROM ivoa.obscore WHERE target_name LIKE '%M83%' "
        "AND (band_list = '6' OR band_list LIKE '6 %' OR band_list LIKE '% 6' OR band_list LIKE '% 6 %')")
custom = alminer.run_query(adql, print_targets=False)
print(len(custom), "rows from ADQL")

# 4. Summaries and plots
alminer.summary(cone, print_targets=False)
# alminer.plot_bands(cone)  ;  alminer.plot_sky(cone)

# 5. Files: dry run first, then download FITS products only
alminer.download_data(cone.head(1), fitsonly=True, dryrun=True, location="./alma_data", print_urls=True)
# alminer.download_data(cone.head(1), fitsonly=True, dryrun=False, location="./alma_data")
