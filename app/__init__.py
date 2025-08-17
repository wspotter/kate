"""
Kate LLM Client - A modern desktop application for multiple LLM providers.

This package contains the complete Kate application including:
- Core application framework
- UI components built with PySide6
- Multiple LLM provider integrations
- Advanced features like themes, plugins, and voice processing
"""

__version__ = "1.0.0"
__author__ = "Kate Team"
__email__ = "team@kate-llm.com"
__license__ = "MIT"

# Re-export commonly used components for convenience
from .core.application import KateApplication
from .core.config import AppSettings, get_settings
from .core.events import EventBus

__all__ = [
    "KateApplication",
    "AppSettings", 
    "get_settings",
    "EventBus",
    "__version__",
    "__author__",
    "__email__",
    "__license__",
]