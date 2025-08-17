"""
Theme manager for Kate LLM Client.
"""

from typing import Dict, Optional, List
from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QObject
from loguru import logger

from .base import ThemeData, get_kate_dark_theme, get_kate_light_theme
from ..core.events import EventBus, ThemeChangedEvent


class ThemeManager(QObject):
    """
    Manages themes and applies Qt stylesheets.
    """
    
    def __init__(self, event_bus: EventBus, themes_dir: Path):
        super().__init__()
        self.event_bus = event_bus
        self.themes_dir = themes_dir
        self.logger = logger.bind(component="ThemeManager")
        
        # Built-in themes
        self._themes: Dict[str, ThemeData] = {
            "Kate Dark": get_kate_dark_theme(),
            "Kate Light": get_kate_light_theme(),
        }
        
        self.current_theme: Optional[ThemeData] = None
        
        # Ensure themes directory exists
        self.themes_dir.mkdir(parents=True, exist_ok=True)
        
    def get_available_themes(self) -> List[str]:
        """Get list of available theme names."""
        return list(self._themes.keys())
        
    def get_theme(self, theme_name: str) -> Optional[ThemeData]:
        """Get theme data by name."""
        return self._themes.get(theme_name)
        
    def apply_theme(self, theme_name: str) -> bool:
        """Apply a theme by name."""
        theme = self.get_theme(theme_name)
        if not theme:
            self.logger.error(f"Theme not found: {theme_name}")
            return False
            
        try:
            # Generate Qt stylesheet
            stylesheet = self._generate_stylesheet(theme)
            
            # Apply to application
            app = QApplication.instance()
            if app:
                app.setStyleSheet(stylesheet)
                
            self.current_theme = theme
            
            # Emit theme changed event
            self.event_bus.emit(ThemeChangedEvent(theme_name=theme_name))
            
            self.logger.info(f"Applied theme: {theme_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to apply theme {theme_name}: {e}")
            return False
            
    def set_default_theme(self) -> bool:
        """Set the default Kate Dark theme."""
        return self.apply_theme("Kate Dark")
        
    def _generate_stylesheet(self, theme: ThemeData) -> str:
        """Generate Qt stylesheet from theme data."""
        return f"""
        /* Main Application */
        QMainWindow {{
            background-color: {theme.background};
            color: {theme.text_primary};
            font-family: {theme.font_family};
            font-size: {theme.font_size}px;
        }}
        
        /* Widgets */
        QWidget {{
            background-color: {theme.background};
            color: {theme.text_primary};
            border: none;
        }}
        
        /* Splitter */
        QSplitter {{
            background-color: {theme.background};
        }}
        
        QSplitter::handle {{
            background-color: {theme.border};
            width: 1px;
            height: 1px;
        }}
        
        /* Buttons */
        QPushButton {{
            background-color: {theme.button_background};
            color: {theme.text_primary};
            border: 1px solid {theme.border};
            border-radius: {theme.border_radius}px;
            padding: {theme.padding}px;
            font-weight: {theme.font_weight};
        }}
        
        QPushButton:hover {{
            background-color: {theme.hover};
        }}
        
        QPushButton:pressed {{
            background-color: {theme.pressed};
        }}
        
        QPushButton:disabled {{
            background-color: {theme.disabled};
            color: {theme.text_disabled};
        }}
        
        /* Text Input */
        QLineEdit, QTextEdit, QPlainTextEdit {{
            background-color: {theme.input_background};
            color: {theme.text_primary};
            border: 1px solid {theme.border};
            border-radius: {theme.border_radius}px;
            padding: {theme.padding}px;
        }}
        
        QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
            border-color: {theme.focus};
        }}
        
        /* Menu Bar */
        QMenuBar {{
            background-color: {theme.surface};
            color: {theme.text_primary};
            border-bottom: 1px solid {theme.border};
        }}
        
        QMenuBar::item {{
            background-color: transparent;
            padding: 4px 8px;
        }}
        
        QMenuBar::item:selected {{
            background-color: {theme.hover};
        }}
        
        /* Menu */
        QMenu {{
            background-color: {theme.surface};
            color: {theme.text_primary};
            border: 1px solid {theme.border};
        }}
        
        QMenu::item {{
            padding: 8px 16px;
        }}
        
        QMenu::item:selected {{
            background-color: {theme.hover};
        }}
        
        /* Toolbar */
        QToolBar {{
            background-color: {theme.surface};
            border-bottom: 1px solid {theme.border};
            spacing: 2px;
        }}
        
        QToolButton {{
            background-color: transparent;
            border: none;
            padding: 4px;
        }}
        
        QToolButton:hover {{
            background-color: {theme.hover};
        }}
        
        /* Status Bar */
        QStatusBar {{
            background-color: {theme.surface};
            color: {theme.text_secondary};
            border-top: 1px solid {theme.border};
        }}
        
        /* Scroll Bar */
        QScrollBar:vertical {{
            background-color: {theme.surface};
            width: 12px;
            border: none;
        }}
        
        QScrollBar::handle:vertical {{
            background-color: {theme.secondary};
            border-radius: 6px;
            min-height: 20px;
        }}
        
        QScrollBar::handle:vertical:hover {{
            background-color: {theme.hover};
        }}
        
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            border: none;
            background: none;
        }}
        
        /* List Widget */
        QListWidget {{
            background-color: {theme.surface};
            color: {theme.text_primary};
            border: none;
            outline: none;
        }}
        
        QListWidget::item {{
            padding: 8px;
            border-bottom: 1px solid {theme.border};
        }}
        
        QListWidget::item:selected {{
            background-color: {theme.selected};
        }}
        
        QListWidget::item:hover {{
            background-color: {theme.hover};
        }}
        
        /* Frame */
        QFrame {{
            background-color: {theme.surface};
            border: 1px solid {theme.border};
        }}
        
        /* Label */
        QLabel {{
            color: {theme.text_primary};
        }}
        """


# Global theme manager instance
_theme_manager: Optional[ThemeManager] = None


def initialize_theme_manager(event_bus: EventBus, themes_dir: Path) -> ThemeManager:
    """Initialize the global theme manager."""
    global _theme_manager
    _theme_manager = ThemeManager(event_bus, themes_dir)
    return _theme_manager


def get_theme_manager() -> Optional[ThemeManager]:
    """Get the global theme manager instance."""
    return _theme_manager


def cleanup_theme_manager() -> None:
    """Cleanup the global theme manager."""
    global _theme_manager
    _theme_manager = None