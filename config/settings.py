"""
Quasar Configuration Settings
Central configuration management for the application
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import json
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Environment(Enum):
    """Application environment"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"

class LogLevel(Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class APIConfig:
    """API configuration"""
    openai_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = field(default_factory=lambda: os.getenv("DEFAULT_LLM_MODEL", "gpt-5.4-mini"))
    openai_temperature: float = field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.7")))
    openai_max_tokens: int = field(default_factory=lambda: int(os.getenv("MAX_TOKENS", "2000")))

    anthropic_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))

    nasa_ads_key: str = field(default_factory=lambda: os.getenv("NASA_ADS_API_KEY", ""))

    nrao_tap_url: str = field(default_factory=lambda: os.getenv("NRAO_TAP_URL", "https://data.nrao.edu/tap"))
    nrao_archive_url: str = field(default_factory=lambda: os.getenv("NRAO_ARCHIVE_URL", "https://data.nrao.edu/portal/"))
    nrao_datalink_url: str = field(default_factory=lambda: os.getenv("NRAO_DATALINK_URL", "https://data.nrao.edu/datalink"))

@dataclass
class PathConfig:
    """Path configuration"""
    base_dir: Path = field(default_factory=lambda: Path.cwd())
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", "./data")))
    cache_dir: Path = field(default_factory=lambda: Path(os.getenv("CACHE_DIR", "./cache")))
    logs_dir: Path = field(default_factory=lambda: Path(os.getenv("LOGS_DIR", "./logs")))
    temp_dir: Path = field(default_factory=lambda: Path("./tmp"))

    def __post_init__(self):
        """Create directories if they don't exist"""
        for path in [self.data_dir, self.cache_dir, self.logs_dir, self.temp_dir]:
            path.mkdir(parents=True, exist_ok=True)

    def get_download_path(self, filename: str) -> Path:
        """Get path for downloaded file"""
        return self.data_dir / filename

    def get_cache_path(self, key: str) -> Path:
        """Get path for cache file"""
        return self.cache_dir / f"{key}.cache"

@dataclass
class SearchConfig:
    """Search configuration"""
    max_results: int = field(default_factory=lambda: int(os.getenv("MAX_SEARCH_RESULTS", "1000")))
    default_results: int = field(default_factory=lambda: int(os.getenv("DEFAULT_SEARCH_RESULTS", "100")))
    timeout_seconds: int = field(default_factory=lambda: int(os.getenv("REQUEST_TIMEOUT", "60")))
    batch_size: int = field(default_factory=lambda: int(os.getenv("BATCH_SIZE", "100")))

    # Default search parameters
    default_radius_deg: float = 0.5
    default_freq_range_ghz: tuple = (1.0, 2.0)
    default_facilities: List[str] = field(default_factory=lambda: ["VLA", "VLBA", "ALMA", "GBT"])

    # Query templates
    query_templates: Dict[str, str] = field(default_factory=lambda: {
        "recent_vla": "SELECT TOP 100 * FROM ivoa.obscore WHERE facility_name='VLA' ORDER BY t_min DESC",
        "bright_sources": "SELECT * FROM ivoa.obscore WHERE flux_density > 1.0",
        "large_projects": "SELECT * FROM ivoa.obscore WHERE t_exptime > 3600"
    })

@dataclass
class ProcessingConfig:
    """Data processing configuration"""
    # CASA settings
    casa_enabled: bool = field(default_factory=lambda: os.getenv("ENABLE_CASA_PIPELINE", "false").lower() == "true")
    casa_path: Optional[Path] = field(default_factory=lambda: Path(os.getenv("CASA_PATH", "/usr/local/casa")) if os.getenv("CASA_PATH") else None)
    casa_data_path: Optional[Path] = field(default_factory=lambda: Path(os.getenv("CASA_DATA_PATH", "/usr/local/casa-data")) if os.getenv("CASA_DATA_PATH") else None)

    # CARTA settings
    carta_enabled: bool = field(default_factory=lambda: os.getenv("ENABLE_CARTA_INTEGRATION", "false").lower() == "true")
    carta_backend_port: int = field(default_factory=lambda: int(os.getenv("CARTA_BACKEND_PORT", "3002")))
    carta_frontend_port: int = field(default_factory=lambda: int(os.getenv("CARTA_FRONTEND_PORT", "3000")))
    carta_base_dir: Path = field(default_factory=lambda: Path(os.getenv("CARTA_BASE_DIR", "./carta_sessions")))

    # Processing defaults
    default_weighting: str = "natural"
    default_robust: float = 0.5
    default_cell_size: str = "1arcsec"
    default_image_size: int = 1024
    default_niter: int = 1000
    default_threshold: str = "0.1mJy"

    # Flagging defaults
    flag_autocorr: bool = True
    flag_edge_channels: bool = True
    edge_channel_percent: float = 5.0
    flag_zeros: bool = True
    flag_shadowed: bool = True

@dataclass
class UIConfig:
    """UI configuration"""
    streamlit_port: int = field(default_factory=lambda: int(os.getenv("STREAMLIT_SERVER_PORT", "8501")))
    streamlit_address: str = field(default_factory=lambda: os.getenv("STREAMLIT_SERVER_ADDRESS", "localhost"))
    streamlit_theme: str = field(default_factory=lambda: os.getenv("STREAMLIT_THEME", "dark"))

    # Display settings
    max_display_rows: int = 1000
    default_plot_height: int = 500
    default_plot_width: int = 800

    # Session settings
    session_timeout_minutes: int = field(default_factory=lambda: int(os.getenv("SESSION_TIMEOUT_MINUTES", "60")))
    max_memory_turns: int = field(default_factory=lambda: int(os.getenv("MAX_MEMORY_TURNS", "10")))

@dataclass
class PerformanceConfig:
    """Performance configuration"""
    num_workers: int = field(default_factory=lambda: int(os.getenv("NUM_WORKERS", "4")))
    cache_ttl_seconds: int = field(default_factory=lambda: int(os.getenv("CACHE_TTL", "3600")))
    max_download_size_gb: float = field(default_factory=lambda: float(os.getenv("MAX_DOWNLOAD_SIZE_GB", "50")))
    chunk_size: int = 8192  # For file downloads

    # Memory limits
    max_memory_usage_gb: float = 8.0
    max_array_size_gb: float = 2.0

@dataclass
class FeatureFlags:
    """Feature flags for enabling/disabling functionality"""
    enable_advanced_search: bool = field(default_factory=lambda: os.getenv("ENABLE_ADVANCED_SEARCH", "true").lower() == "true")
    enable_batch_downloads: bool = field(default_factory=lambda: os.getenv("ENABLE_BATCH_DOWNLOADS", "false").lower() == "true")
    enable_ml_features: bool = field(default_factory=lambda: os.getenv("ENABLE_ML_FEATURES", "false").lower() == "true")
    enable_export_scripts: bool = True
    enable_visualization: bool = True
    enable_literature_search: bool = field(default_factory=lambda: bool(os.getenv("NASA_ADS_API_KEY")))

class Settings:
    """Main settings class"""

    def __init__(self, config_file: Optional[str] = None):
        """Initialize settings from environment and optional config file"""
        self.environment = Environment(os.getenv("QUASAR_ENV", "development"))
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        self.log_level = LogLevel(os.getenv("LOG_LEVEL", "INFO"))

        # Initialize component configs
        self.api = APIConfig()
        self.paths = PathConfig()
        self.search = SearchConfig()
        self.processing = ProcessingConfig()
        self.ui = UIConfig()
        self.performance = PerformanceConfig()
        self.features = FeatureFlags()

        # Load from config file if provided
        if config_file and Path(config_file).exists():
            self.load_from_file(config_file)

    def load_from_file(self, filepath: str):
        """Load settings from JSON or YAML file"""
        path = Path(filepath)

        if path.suffix == '.json':
            with open(path) as f:
                config = json.load(f)
        elif path.suffix in ['.yml', '.yaml']:
            with open(path) as f:
                config = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")

        # Update settings from config
        self._update_from_dict(config)

    def _update_from_dict(self, config: Dict[str, Any]):
        """Update settings from dictionary"""
        for key, value in config.items():
            if hasattr(self, key):
                attr = getattr(self, key)
                if isinstance(attr, (APIConfig, PathConfig, SearchConfig, ProcessingConfig, UIConfig, PerformanceConfig, FeatureFlags)):
                    # Update dataclass fields
                    for field_name, field_value in value.items():
                        if hasattr(attr, field_name):
                            setattr(attr, field_name, field_value)
                else:
                    setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary"""
        return {
            "environment": self.environment.value,
            "debug": self.debug,
            "log_level": self.log_level.value,
            "api": {k: v for k, v in self.api.__dict__.items() if not k.startswith('_')},
            "paths": {k: str(v) if isinstance(v, Path) else v for k, v in self.paths.__dict__.items()},
            "search": {k: v for k, v in self.search.__dict__.items()},
            "processing": {k: str(v) if isinstance(v, Path) else v for k, v in self.processing.__dict__.items()},
            "ui": {k: v for k, v in self.ui.__dict__.items()},
            "performance": {k: v for k, v in self.performance.__dict__.items()},
            "features": {k: v for k, v in self.features.__dict__.items()}
        }

    def save_to_file(self, filepath: str):
        """Save settings to file"""
        path = Path(filepath)
        config = self.to_dict()

        if path.suffix == '.json':
            with open(path, 'w') as f:
                json.dump(config, f, indent=2)
        elif path.suffix in ['.yml', '.yaml']:
            with open(path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

    def validate(self) -> List[str]:
        """Validate settings and return list of issues"""
        issues = []

        # Check required API keys
        if not self.api.openai_key and self.features.enable_ml_features:
            issues.append("OpenAI API key is required for AI features")

        # Check paths exist
        if self.processing.casa_enabled and self.processing.casa_path:
            if not self.processing.casa_path.exists():
                issues.append(f"CASA path does not exist: {self.processing.casa_path}")

        # Check port availability (basic check)
        if self.ui.streamlit_port < 1024 or self.ui.streamlit_port > 65535:
            issues.append(f"Invalid Streamlit port: {self.ui.streamlit_port}")

        if self.processing.carta_backend_port < 1024 or self.processing.carta_backend_port > 65535:
            issues.append(f"Invalid CARTA backend port: {self.processing.carta_backend_port}")

        return issues

    def get_tap_config(self) -> Dict[str, Any]:
        """Get TAP service configuration"""
        return {
            "service_url": self.api.nrao_tap_url,
            "timeout": self.search.timeout_seconds,
            "max_results": self.search.max_results
        }

    def get_processing_params(self, pipeline_type: str = "standard") -> Dict[str, Any]:
        """Get processing parameters for pipeline"""
        params = {
            "weighting": self.processing.default_weighting,
            "robust": self.processing.default_robust,
            "cell": self.processing.default_cell_size,
            "imsize": self.processing.default_image_size,
            "niter": self.processing.default_niter,
            "threshold": self.processing.default_threshold
        }

        if pipeline_type == "quick":
            params["niter"] = 100
            params["imsize"] = 512
        elif pipeline_type == "deep":
            params["niter"] = 5000
            params["imsize"] = 2048

        return params

# Singleton instance
_settings = None

def get_settings(config_file: Optional[str] = None) -> Settings:
    """Get or create settings singleton"""
    global _settings
    if _settings is None:
        _settings = Settings(config_file)
    return _settings

def reset_settings():
    """Reset settings singleton"""
    global _settings
    _settings = None

# Convenience exports
settings = get_settings()