import unittest
from unittest.mock import patch

from astropy.table import Table

from integrations.alminer_client import _resolve_simbad_cached


import pytest

pytestmark = pytest.mark.slow


class SimbadResolverCompatTests(unittest.TestCase):
    def tearDown(self):
        _resolve_simbad_cached.cache_clear()

    def test_resolver_accepts_lowercase_degree_columns(self):
        result_table = Table(rows=[(187.7059, 12.3911)], names=("ra", "dec"))

        with patch("astroquery.simbad.Simbad.query_object", return_value=result_table):
            ra_deg, dec_deg = _resolve_simbad_cached("M87")

        self.assertAlmostEqual(ra_deg, 187.7059)
        self.assertAlmostEqual(dec_deg, 12.3911)

    def test_failed_resolution_is_not_cached(self):
        # C16 regression: a transient SIMBAD outage (query_object → None) used
        # to be memoized by the LRU, permanently poisoning the target with
        # (None, None). A later successful lookup must succeed.
        with patch("astroquery.simbad.Simbad.query_object", return_value=None):
            self.assertEqual(_resolve_simbad_cached("M87"), (None, None))

        result_table = Table(rows=[(187.7059, 12.3911)], names=("ra", "dec"))
        with patch("astroquery.simbad.Simbad.query_object", return_value=result_table):
            ra_deg, dec_deg = _resolve_simbad_cached("M87")

        self.assertAlmostEqual(ra_deg, 187.7059)
        self.assertAlmostEqual(dec_deg, 12.3911)

    def test_successful_resolution_is_cached(self):
        result_table = Table(rows=[(187.7059, 12.3911)], names=("ra", "dec"))
        with patch("astroquery.simbad.Simbad.query_object", return_value=result_table) as q:
            _resolve_simbad_cached("M87")
            _resolve_simbad_cached("M87")
        self.assertEqual(q.call_count, 1)


if __name__ == "__main__":
    unittest.main()
