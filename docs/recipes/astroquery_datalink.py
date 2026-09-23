"""ALMA Science Archive: list and download the files of a dataset (DataLink) with astroquery (tested recipe).

library: astroquery >= 0.4.7 (tested with 0.4.11)
last_tested: 2026-09-22
tested_against: https://almascience.nrao.edu/datalink (live) and recorded responses in tests/unit/test_code_recipes.py
notes: Alma.get_data_info(uids) returns the DataLink table (one row per file/tarball: access_url, content_length, semantics).
notes: Alma.download_files(urls, cache=True) downloads them; Alma.download_and_extract_files also unpacks tarballs.
notes: Proprietary data need Alma.login(username) first. There is no Alma.get_data_links / get_data_products / query_sql.
"""
from astroquery.alma import Alma

alma = Alma()
alma.archive_url = "https://almascience.nrao.edu"

mous = "uid://A001/X1465/X2f2c"          # a member_ous_uid from a query (e.g. table['member_ous_uid'][0])
links = alma.get_data_info([mous], expand_tarfiles=False)   # DataLink table for the dataset
print(links["access_url", "content_length", "semantics"][:10])

# Only the FITS products (skip raw ASDM tarballs)
fits_links = [row["access_url"] for row in links if str(row["access_url"]).endswith(".fits") or "product" in str(row["semantics"])]
print(len(fits_links), "FITS product links")

# Download (to the astroquery cache directory unless savedir is given)
# paths = alma.download_files(fits_links[:2], savedir=".", cache=True)
# print(paths)
