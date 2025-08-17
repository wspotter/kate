"""
Core Kate application modules.

This package contains the fundamental components of the Kate application:
- Application framework and lifecycle management
- Configuration system
- Event system for inter-component communication
- Logging and error handling
"""

from .application import KateApplication
from .config import AppSettings, get_settings
from .events import EventBus

__all__ = [
    "KateApplication",
    "AppSettings",
    "get_settings", 
    "EventBus",
]