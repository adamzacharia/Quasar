"""
Quasar integrations package.

Imports are kept optional so archive-specific clients remain importable even if
another archive's dependency is missing from the current environment.
"""

from __future__ import annotations

__all__ = []


try:
    from .tap import NRAOTapClient

    __all__.append("NRAOTapClient")
except Exception:
    NRAOTapClient = None

try:
    from .datalink import DataLinkClient

    __all__.append("DataLinkClient")
except Exception:
    DataLinkClient = None

try:
    from .mast_client import MASTClient

    __all__.append("MASTClient")
except Exception:
    MASTClient = None

try:
    from .eso_tap_client import ESOTAPClient

    __all__.append("ESOTAPClient")
except Exception:
    ESOTAPClient = None

try:
    from .irsa_client import IRSAClient

    __all__.append("IRSAClient")
except Exception:
    IRSAClient = None
