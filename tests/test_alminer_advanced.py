import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestALminerAdvanced(unittest.TestCase):

    def setUp(self):
        # Mock dependencies BEFORE importing modules that might use them
        self.alminer_mock = MagicMock()
        self.plt_mock = MagicMock()
        self.pyvo_mock = MagicMock()
        
        # Patch sys.modules to provide mocks for missing libraries
        self.modules_patch = patch.dict(sys.modules, {
            'alminer': self.alminer_mock,
            'matplotlib': MagicMock(),
            'matplotlib.pyplot': self.plt_mock,
            'pyvo': self.pyvo_mock,
            'astropy': MagicMock(),
            'astropy.coordinates': MagicMock(),
            'astropy.table': MagicMock(),
            'astroquery': MagicMock(),
            'astroquery.alma': MagicMock(),
            'astroquery.simbad': MagicMock()
        })
        self.modules_patch.start()
        
        # Always reload to ensure mocks are picked up if module was already loaded
        import integrations.alminer_client
        import services.search
        import importlib
        importlib.reload(integrations.alminer_client)
        importlib.reload(services.search)
        
        from integrations.alminer_client import ALminerClient
        from services.search import SearchService
            
        self.ALminerClient = ALminerClient
        self.SearchService = SearchService

        self.client = self.ALminerClient()
        self.service = self.SearchService()
        
        # Determine behavior of mocks
        # Create a sample dataframe
        self.sample_df = pd.DataFrame({
            'target_name': ['M87', 'Centaurus A'],
            'ra': [187.7, 201.3],
            'dec': [12.3, -43.0],
            'project_code': ['2017.1.000', '2018.1.000'],
            'member_ous_uid': ['uid://A001/X1/X1', 'uid://A001/X2/X2']
        })
        
    def tearDown(self):
        self.modules_patch.stop()

    def test_search_by_keywords(self):
        # Setup mock return
        self.alminer_mock.key_search.return_value = self.sample_df
        
        keywords = {'pi_name': 'Smith'}
        result = self.client.search_by_keywords(keywords)
        
        self.alminer_mock.key_search.assert_called_with(public=True, print_targets=False, pi_name='Smith')
        self.assertFalse(result.empty)
        self.assertTrue('s_ra' in result.columns) # Check standardization
        
        # Test service layer
        self.service.search_alma_with_keywords(keywords)
        
    def test_search_by_sql(self):
        self.alminer_mock.run_tap_query.return_value = self.sample_df
        
        query = "SELECT * FROM ivoa.obscore"
        result = self.client.search_by_sql(query)
        
        self.alminer_mock.run_tap_query.assert_called_with(query, print_targets=False)
        self.assertFalse(result.empty)
        
    def test_plot_sky(self):
        # We need to ensure logic flow allows plotting
        import integrations.alminer_client
        path = self.client.plot_sky_distribution(self.sample_df, "test.png")
        
        self.alminer_mock.plot_sky.assert_called()
        self.assertTrue(path.endswith("test.png"))
            
    def test_download_dry_run(self):
        result = self.client.download_data(self.sample_df, dry_run=True)
        self.alminer_mock.download_data.assert_called()
        self.assertIn("Dry run", result)

if __name__ == '__main__':
    unittest.main()
