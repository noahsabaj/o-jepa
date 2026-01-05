"""
Video format adapters for O-JEPA.

Provides pluggable adapters for different video dataset formats.
New formats can be added by implementing FormatAdapter and registering here.
"""

from .base import FormatAdapter, ClipInfo
from .generic import GenericVideoAdapter
from .ego_exo4d import EgoExo4DAdapter


# Format registry: name -> adapter class
# Order matters for auto-detection (specific formats before generic)
FORMAT_REGISTRY = {
    "ego_exo4d": EgoExo4DAdapter,
    "generic": GenericVideoAdapter,
}


def register_format(name: str, adapter_class: type):
    """
    Register a new format adapter.

    Args:
        name: Format name (used in VideoAudioDataset format parameter)
        adapter_class: FormatAdapter subclass
    """
    FORMAT_REGISTRY[name] = adapter_class


def get_available_formats() -> list:
    """Get list of available format names."""
    return list(FORMAT_REGISTRY.keys())


__all__ = [
    "FormatAdapter",
    "ClipInfo",
    "GenericVideoAdapter",
    "EgoExo4DAdapter",
    "FORMAT_REGISTRY",
    "register_format",
    "get_available_formats",
]
