import unittest

import pandas as pd

from utils.archive_links import build_archive_link, infer_archive_kind


class ArchiveLinkTests(unittest.TestCase):
    def test_infer_mast_from_source_hint(self):
        df = pd.DataFrame([{"target_name": "Carina Nebula"}])
        self.assertEqual(infer_archive_kind(df, source_hint="MAST"), "mast")

    def test_build_mast_link_uses_target_query(self):
        df = pd.DataFrame([{"target_name": "Carina Nebula", "telescope": "JWST"}])
        self.assertEqual(
            build_archive_link(df, source_hint="MAST"),
            "https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html?searchQuery=Carina+Nebula",
        )

    def test_build_eso_link_uses_science_portal(self):
        df = pd.DataFrame([{"target_name": "NGC 1068", "instrument_name": "MUSE"}])
        self.assertEqual(
            build_archive_link(df, source_hint="ESO"),
            "https://archive.eso.org/scienceportal/home",
        )

    def test_build_irsa_link_uses_frontpage(self):
        df = pd.DataFrame([{"target_name": "M31", "telescope": "IRSA"}])
        self.assertEqual(
            build_archive_link(df, source_hint="IRSA"),
            "https://irsa.ipac.caltech.edu/frontpage/",
        )

    def test_build_cadc_link_uses_target_search(self):
        df = pd.DataFrame([{"target_name": "M87", "obs_publisher_did": "ivo://cadc.nrc.ca/test"}])
        self.assertEqual(
            build_archive_link(df, source_hint="CADC"),
            "https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/en/search/?Observation.target.name=M87",
        )

    def test_build_alma_link_prefers_member_ous_uid(self):
        df = pd.DataFrame([{"target_name": "M87", "member_ous_uid": "uid://A001/X1"}])
        self.assertEqual(
            build_archive_link(df, source_hint="ALMA"),
            "https://almascience.nrao.edu/aq/?member_ous_id=uid%3A%2F%2FA001%2FX1",
        )

    def test_unknown_archive_returns_none(self):
        df = pd.DataFrame([{"foo": "bar"}])
        self.assertIsNone(build_archive_link(df))


if __name__ == "__main__":
    unittest.main()
