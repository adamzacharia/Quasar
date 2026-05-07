import unittest

import pandas as pd

from integrations.eso_tap_client import ESOTAPClient
from integrations.irsa_client import IRSAClient
from integrations.mast_client import MASTClient
from services.multi_archive import MultiArchiveMatcher


class _StubSearchClient:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def search_by_target(self, **kwargs):
        return self.df.copy()


class MultiArchiveMatcherTests(unittest.TestCase):
    def test_client_classes_import(self):
        self.assertIsNotNone(MASTClient)
        self.assertIsNotNone(ESOTAPClient)
        self.assertIsNotNone(IRSAClient)

    def test_deep_search_mast_returns_dataframe_and_summary(self):
        df = pd.DataFrame(
            [
                {
                    "target_name": "M87",
                    "telescope": "JWST",
                    "instrument_name": "NIRCAM",
                    "filters": "F200W",
                    "project_code": "1345",
                }
            ]
        )
        matcher = MultiArchiveMatcher(mast_client=_StubSearchClient(df))

        result = matcher.deep_search_mast("M87", mission="JWST")

        self.assertTrue(result["found"])
        self.assertEqual(result["n_observations"], 1)
        self.assertEqual(result["missions"], ["JWST"])
        pd.testing.assert_frame_equal(result["data"], df)

    def test_deep_search_eso_returns_dataframe_and_summary(self):
        df = pd.DataFrame(
            [
                {
                    "target_name": "NGC 1068",
                    "instrument_name": "MUSE",
                    "obs_collection": "ESO",
                    "dataproduct_type": "cube",
                }
            ]
        )
        matcher = MultiArchiveMatcher(eso_client=_StubSearchClient(df))

        result = matcher.deep_search_eso("NGC 1068", instrument="MUSE")

        self.assertTrue(result["found"])
        self.assertEqual(result["n_observations"], 1)
        self.assertEqual(result["instruments"], ["MUSE"])
        pd.testing.assert_frame_equal(result["data"], df)

    def test_deep_search_irsa_returns_dataframe_and_summary(self):
        df = pd.DataFrame(
            [
                {
                    "target_name": "M31",
                    "telescope": "IRSA",
                    "instrument_name": "ALLWISE_P3AS_PSD",
                }
            ]
        )
        matcher = MultiArchiveMatcher(irsa_client=_StubSearchClient(df))

        result = matcher.deep_search_irsa("M31", catalog="allwise")

        self.assertTrue(result["found"])
        self.assertEqual(result["n_sources"], 1)
        self.assertEqual(result["catalogs"], ["ALLWISE_P3AS_PSD"])
        pd.testing.assert_frame_equal(result["data"], df)

    def test_cross_match_can_request_optional_eso_and_irsa_archives(self):
        matcher = MultiArchiveMatcher()
        matcher._query_eso = lambda target: {"archive": "ESO", "found": True, "n_observations": 2}
        matcher._query_irsa = lambda target: {"archive": "IRSA", "found": True, "n_sources": 3}

        result = matcher.cross_match_source("M87", archives=["eso", "irsa"])

        self.assertEqual(result["archives_queried"], ["eso", "irsa"])
        self.assertEqual(set(result["found_in"]), {"eso", "irsa"})


if __name__ == "__main__":
    unittest.main()
