# config/__init__.py
"""
Quasar Configuration

Provides unified configuration through the Settings class.
Environment variables are loaded from .env file.
"""

import os
from dotenv import load_dotenv

# Load .env from project root
load_dotenv()

# Re-export the canonical Settings class
from config.settings import Settings

# Convenience: singleton instance
settings = Settings()


class Config:
    """
    Lightweight config accessor for backward compatibility.
    Delegates to the canonical Settings singleton.
    
    .. deprecated::
        Use ``from config import settings`` or ``from config.settings import Settings`` instead.
    """

    @property
    def openai_api_key(self):
        return os.getenv("OPENAI_API_KEY", "")

    @property
    def tacc_api_key(self):
        return os.getenv("TACC_API_KEY", "")

    @property
    def tacc_base_url(self):
        return os.getenv("TACC_BASE_URL", "https://ai.tejas.tacc.utexas.edu/v1")

    @property
    def nasa_ads_api_key(self):
        return os.getenv("NASA_ADS_API_KEY", "")

    @property
    def default_model(self):
        return os.getenv("DEFAULT_LLM_MODEL", "gpt-oss-120b")

    @property
    def ads_query_model(self):
        return os.getenv("ADS_QUERY_MODEL", os.getenv("DEFAULT_LLM_MODEL", "gpt-oss-120b"))

    @property
    def verbose(self):
        return os.getenv("QUASAR_VERBOSE", "false").lower() == "true"
