"""
Theme system for Kate LLM Client.

This package provides a comprehensive theming system including:
- Theme management and switching
- Qt stylesheet generation
- Dark/light mode support
- Custom theme creation
"""

from .manager import ThemeManager, initialize_theme_manager, cleanup_theme_manager
from .base import ThemeData

__all__ = [
    "ThemeManager",
    "initialize_theme_manager", 
    "cleanup_theme_manager",
    "ThemeData",
]