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
    def nasa_ads_api_key(self):
        return os.getenv("NASA_ADS_API_KEY", "")

    @property
    def default_model(self):
        return os.getenv("DEFAULT_LLM_MODEL", "gpt-5.4-mini")

    @property
    def ads_query_model(self):
        return os.getenv("ADS_QUERY_MODEL", "gpt-5.4-mini")

    @property
    def verbose(self):
        return os.getenv("QUASAR_VERBOSE", "false").lower() == "true"