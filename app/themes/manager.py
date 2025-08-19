"""
Theme manager for Kate LLM Client.
"""

from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger
from PySide6.QtCore import QObject
from PySide6.QtWidgets import QApplication

from ..core.events import EventBus, ThemeChangedEvent
from .base import ThemeData, get_kate_dark_theme, get_kate_light_theme


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

        # Prevent noisy repeat attempts if a theme already failed in this session
        if not hasattr(self, "_failed_themes"):
            self._failed_themes = set()  # type: ignore[attr-defined]
        if theme_name in self._failed_themes:  # type: ignore[operator]
            self.logger.warning(f"Skipping re-application of previously failed theme '{theme_name}'")
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
            # Remember failure to avoid repeated log spam
            try:
                self._failed_themes.add(theme_name)  # type: ignore[attr-defined]
            except Exception:
                pass
            return False
            
    def set_default_theme(self) -> bool:
        """Set the default Kate Dark theme."""
        return self.apply_theme("Kate Dark")
        
    def _generate_stylesheet(self, theme: ThemeData) -> str:
        """Generate Qt stylesheet from theme data."""
        # Build stylesheet safely without f-string interpolation conflicts by using format_map
        values = {
            "background": theme.background,
            "text_primary": theme.text_primary,
            "font_family": theme.font_family,
            "font_size": theme.font_size,
            "button_background": theme.button_background,
            "border": theme.border,
            "border_radius": theme.border_radius,
            "padding": theme.padding,
            "font_weight": theme.font_weight,
            "text_disabled": theme.text_disabled,
            "input_background": theme.input_background,
            "focus": theme.focus,
            "surface": theme.surface,
            "hover": theme.hover,
            "pressed": theme.pressed,
            "disabled": theme.disabled,
            "text_secondary": theme.text_secondary,
            "secondary": theme.secondary,
            "selected": theme.selected,
        }
        template = """
        /* Main Application */
        QMainWindow {{
            background-color: {background};
            color: {text_primary};
            font-family: {font_family};
            font-size: {font_size}px;
        }}

        QWidget {{
            background-color: {background};
            color: {text_primary};
            border: none;
        }}

        QSplitter {{ background-color: {background}; }}
        QSplitter::handle {{ background-color: {border}; width: 1px; height: 1px; }}

        QPushButton {{
            background-color: {button_background};
            color: {text_primary};
            border: 1px solid {border};
            border-radius: {border_radius}px;
            padding: {padding}px;
            font-weight: {font_weight};
        }}
        QPushButton:hover {{ background-color: {hover}; }}
        QPushButton:pressed {{ background-color: {pressed}; }}
        QPushButton:disabled {{ background-color: {disabled}; color: {text_disabled}; }}

        QLineEdit, QTextEdit, QPlainTextEdit {{
            background-color: {input_background};
            color: {text_primary};
            border: 1px solid {border};
            border-radius: {border_radius}px;
            padding: {padding}px;
        }}
        QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{ border-color: {focus}; }}

        QMenuBar {{ background-color: {surface}; color: {text_primary}; border-bottom: 1px solid {border}; }}
        QMenuBar::item {{ background-color: {surface}; padding: 4px 8px; }}
        QMenuBar::item:selected {{ background-color: {hover}; }}

        QMenu {{ background-color: {surface}; color: {text_primary}; border: 1px solid {border}; }}
        QMenu::item {{ padding: 8px 16px; }}
        QMenu::item:selected {{ background-color: {hover}; }}

        QToolBar {{ background-color: {surface}; border-bottom: 1px solid {border}; spacing: 2px; }}
        QToolButton {{ background-color: {surface}; border: none; padding: 4px; }}
        QToolButton:hover {{ background-color: {hover}; }}

        QStatusBar {{ background-color: {surface}; color: {text_secondary}; border-top: 1px solid {border}; }}

        QScrollBar:vertical {{ background-color: {surface}; width: 12px; border: none; }}
        QScrollBar::handle:vertical {{ background-color: {secondary}; border-radius: 6px; min-height: 20px; }}
        QScrollBar::handle:vertical:hover {{ background-color: {hover}; }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ border: none; background: none; }}

        QListWidget {{ background-color: {surface}; color: {text_primary}; border: none; outline: none; }}
        QListWidget::item {{ padding: 8px; border-bottom: 1px solid {border}; }}
        QListWidget::item:selected {{ background-color: {selected}; }}
        QListWidget::item:hover {{ background-color: {hover}; }}

        QFrame {{ background-color: {surface}; border: 1px solid {border}; }}
        QLabel {{ color: {text_primary}; }}
        """
        try:
            return template.format_map(values)
        except KeyError as e:
            self.logger.error(f"Missing theme value for stylesheet: {e}")
            return ""


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