import unittest
from unittest.mock import patch

from astropy.table import Table

from integrations.alminer_client import _resolve_simbad_cached


class SimbadResolverCompatTests(unittest.TestCase):
    def tearDown(self):
        _resolve_simbad_cached.cache_clear()

    def test_resolver_accepts_lowercase_degree_columns(self):
        result_table = Table(rows=[(187.7059, 12.3911)], names=("ra", "dec"))

        with patch("astroquery.simbad.Simbad.query_object", return_value=result_table):
            ra_deg, dec_deg = _resolve_simbad_cached("M87")

        self.assertAlmostEqual(ra_deg, 187.7059)
        self.assertAlmostEqual(dec_deg, 12.3911)


if __name__ == "__main__":
    unittest.main()
