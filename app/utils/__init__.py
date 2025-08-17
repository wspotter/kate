"""
Utility modules for Kate LLM Client.

This package contains various utility functions and classes used throughout
the application for logging, platform detection, system information, etc.
"""

from .logging import setup_logging
from .platform import setup_platform, get_platform_info

__all__ = [
    "setup_logging",
    "setup_platform",
    "get_platform_info",
]