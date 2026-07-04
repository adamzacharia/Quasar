# services/__init__.py
"""
Quasar Services Module
Business logic and data processing
"""

# NOTE: SearchService and RadioAnalysisService are exposed lazily (PEP 562).
# Importing them eagerly here pulls in heavy dependency chains (~7.5 s) that
# every `import services.<anything>` would otherwise pay, even for fully-lazy
# submodules. Access them as attributes (`services.SearchService`) or via
# `from services import SearchService` and the import happens on first use.

__all__ = [
    'SearchService',
    'RadioAnalysisService'
]

_LAZY_EXPORTS = {
    'SearchService': '.search',
    'RadioAnalysisService': '.analysis',
}


def __getattr__(name):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module = import_module(module_name, __name__)
    value = getattr(module, name)
    # Cache on the package so subsequent lookups skip __getattr__ entirely.
    globals()[name] = value
    return value


def __dir__():
    return sorted(list(globals().keys()) + __all__)

