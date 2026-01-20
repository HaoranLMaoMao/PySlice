"""Convenience imports for the SEA-eco package."""

import importlib


__all__ = [
    'plot',
    'signal_processing',
    'calculators']

def __dir__():
    return sorted(__all__)

# mapping following the pattern: from value import key
_import_mapping = {}

def __getattr__(name):
    if name in __all__:
        if name in _import_mapping.keys():
            import_path = 'pySEA.sea_eco' + _import_mapping.get(name) #? Have not tested if pSEA.sea_eco works
            return getattr(importlib.import_module(import_path), name)
        else:
            return importlib.import_module("." + name, 'pySEA.sea_eco') #? Have not tested if pSEA.sea_eco works
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
