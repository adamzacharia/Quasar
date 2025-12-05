# config/__init__.py
"""
Quasar Configuration Module
Application settings and configuration
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration defaults
class Config:
    """Application configuration"""

    # API Keys
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    NASA_ADS_API_KEY = os.getenv('NASA_ADS_API_KEY', '')

    # NRAO Endpoints
    NRAO_TAP_URL = os.getenv('NRAO_TAP_URL', 'https://data.nrao.edu/tap')
    NRAO_ARCHIVE_URL = os.getenv('NRAO_ARCHIVE_URL', 'https://data.nrao.edu/portal/')

    # Paths
    DATA_DIR = Path(os.getenv('DATA_DIR', './data'))
    CACHE_DIR = Path(os.getenv('CACHE_DIR', './cache'))
    LOGS_DIR = Path(os.getenv('LOGS_DIR', './logs'))

    # Create directories if they don't exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Application settings
    MAX_SEARCH_RESULTS = int(os.getenv('MAX_SEARCH_RESULTS', 1000))
    DEFAULT_SEARCH_RESULTS = int(os.getenv('DEFAULT_SEARCH_RESULTS', 100))
    REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', 60))

    # LLM Settings
    DEFAULT_LLM_MODEL = os.getenv('DEFAULT_LLM_MODEL', 'gpt-4-turbo-preview')
    LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', 0.7))
    MAX_TOKENS = int(os.getenv('MAX_TOKENS', 2000))
    MAX_MEMORY_TURNS = int(os.getenv('MAX_MEMORY_TURNS', 10))

    # Feature flags
    ENABLE_CASA_PIPELINE = os.getenv('ENABLE_CASA_PIPELINE', 'true').lower() == 'true'
    ENABLE_CARTA_INTEGRATION = os.getenv('ENABLE_CARTA_INTEGRATION', 'true').lower() == 'true'
    ENABLE_ADVANCED_SEARCH = os.getenv('ENABLE_ADVANCED_SEARCH', 'true').lower() == 'true'

    @classmethod
    def get_config(cls):
        """Get configuration dictionary"""
        return {
            key: getattr(cls, key)
            for key in dir(cls)
            if not key.startswith('_') and not callable(getattr(cls, key))
        }